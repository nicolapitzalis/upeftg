#!/usr/bin/env python3
"""
Train a GMM on train adapters (after SVD), then score a separate inference set.

Workflow:
1. Load train adapters listed in a TXT/JSON manifest.
2. Stream train adapters to build raw Gram + feature means, then fit dual-space SVD.
3. Tune k + GMM config by train-set BIC across multiple seeds.
4. Stream train+inference adapters to project inference with the same SVD basis (no Vt materialization).
5. Emit calibration/separation report and score tables.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from cluster_z_space import sanitize_gmm_components
from prepare_data import (
    center_sample_gram,
    get_tensor_shape_safe,
    inspect_adapter_schema,
    iter_tensor_flat_chunks,
    json_ready,
    parse_label,
    sanitize_component_grid,
)

SCRIPT_VERSION = "1.2.0"
DEFAULT_DATASET_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("gmm_train_inference_results")


@dataclass(frozen=True)
class ManifestItem:
    raw_entry: str
    model_dir: Path
    adapter_path: Path
    model_name: str
    label: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SVD + GMM training on a train set, then mixed-set inference scoring"
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=None,
        help="TXT manifest with training adapters (structured path+indices or one entry per line)",
    )
    parser.add_argument(
        "--infer-list",
        type=Path,
        default=None,
        help="TXT manifest with inference adapters (structured path+indices or one entry per line)",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help=(
            "Single JSON manifest containing both train and infer source sets. "
            "Expected shape: {\"train\":[{\"path\":\"...\",\"indices\":[a,b]}],"
            "\"infer\":[{\"path\":\"...\",\"indices\":[a,b]}]}"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root used to resolve relative manifest paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for report and score tables",
    )
    parser.add_argument(
        "--svd-components-grid",
        nargs="+",
        type=int,
        default=[20, 25, 30],
        help="SVD component-count grid for tuning",
    )
    parser.add_argument(
        "--gmm-components",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="GMM component-count grid",
    )
    parser.add_argument(
        "--gmm-covariance-types",
        nargs="+",
        default=["diag", "full", "tied", "spherical"],
        help="GMM covariance type grid (order used as tie-breaker)",
    )
    parser.add_argument(
        "--stability-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Seeds used during GMM tuning",
    )
    parser.add_argument(
        "--score-percentiles",
        nargs="+",
        type=float,
        default=[90, 95, 97, 99],
        help="Train-score percentiles used for thresholding",
    )
    parser.add_argument(
        "--stream-block-size",
        type=int,
        default=131072,
        help="Features processed per streaming block",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Working dtype for streamed parameter matrix",
    )
    parser.add_argument(
        "--reg-covar",
        type=float,
        default=1e-5,
        help="GMM reg_covar",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="GMM n_init",
    )
    parser.add_argument(
        "--keep-vt-cache",
        action="store_true",
        help="Deprecated no-op kept for CLI backward compatibility",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Deprecated no-op kept for CLI backward compatibility",
    )
    return parser.parse_args()


def _strip_manifest_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    return line


def _parse_key_value_line(line: str) -> tuple[str, str] | None:
    if "=" in line:
        key, value = line.split("=", 1)
        return key.strip().lower(), value.strip()
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip().lower(), value.strip()
    return None


def _parse_indices_spec(spec: str, *, manifest_path: Path, key: str) -> list[int]:
    text = spec.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(
            f"Invalid index spec for '{key}' in {manifest_path}. Expected bracket form like [0,10], got '{spec}'"
        )
    body = text[1:-1].strip()
    if not body:
        raise ValueError(f"Empty index spec for '{key}' in {manifest_path}")

    parts = [p.strip() for p in body.split(",") if p.strip()]
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"Non-integer index value in '{key}' in {manifest_path}: '{spec}'"
        ) from exc

    if any(v < 0 for v in values):
        raise ValueError(f"Negative indices are not allowed for '{key}' in {manifest_path}: '{spec}'")

    # Convention requested by user: [a,b] means inclusive range a..b.
    if len(values) == 2 and values[1] >= values[0]:
        return list(range(values[0], values[1] + 1))

    # Otherwise treat as an explicit index set while preserving order.
    dedup: list[int] = []
    seen: set[int] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def _expand_structured_paths(path_pattern: str, indices: list[int]) -> list[str]:
    entries: list[str] = []
    for i in indices:
        if "{i}" in path_pattern:
            entries.append(path_pattern.format(i=i))
        else:
            entries.append(f"{path_pattern}{i}")
    return entries


def _parse_indices_value(
    value: Any,
    *,
    manifest_path: Path,
    key: str,
) -> list[int]:
    if isinstance(value, str):
        return _parse_indices_spec(value, manifest_path=manifest_path, key=key)

    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty index list for '{key}' in {manifest_path}")
        try:
            ints = [int(v) for v in value]
        except Exception as exc:
            raise ValueError(f"Non-integer index value in '{key}' in {manifest_path}: {value}") from exc
        if any(v < 0 for v in ints):
            raise ValueError(f"Negative indices are not allowed for '{key}' in {manifest_path}: {value}")
        if len(ints) == 2 and ints[1] >= ints[0]:
            return list(range(ints[0], ints[1] + 1))
        dedup: list[int] = []
        seen: set[int] = set()
        for v in ints:
            if v in seen:
                continue
            seen.add(v)
            dedup.append(v)
        return dedup

    raise ValueError(f"Unsupported index format for '{key}' in {manifest_path}: {value}")


def _parse_source_set_value(
    value: str,
    *,
    manifest_path: Path,
    key: str,
) -> list[tuple[str, list[int]]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON for '{key}' in {manifest_path}. "
            "Expected something like [{\"path\":\"...\",\"indices\":[0,10]}]"
        ) from exc

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"'{key}' in {manifest_path} must be a non-empty JSON list")

    sources: list[tuple[str, list[int]]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{idx}] in {manifest_path} must be an object with path/indices")
        path = item.get("path")
        indices_raw = item.get("indices")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{key}[{idx}] in {manifest_path} is missing a non-empty 'path'")
        if indices_raw is None:
            raise ValueError(f"{key}[{idx}] in {manifest_path} is missing 'indices'")
        indices = _parse_indices_value(indices_raw, manifest_path=manifest_path, key=f"{key}[{idx}].indices")
        sources.append((path.strip(), indices))
    return sources


def _parse_structured_manifest_entries(lines: list[str], manifest_path: Path) -> list[str] | None:
    parsed_lines: list[tuple[str, str]] = []
    for line in lines:
        kv = _parse_key_value_line(line)
        if kv is None:
            return None
        parsed_lines.append(kv)

    clean_path_keys = {"clean_path"}
    clean_indices_keys = {"clean_indices"}
    clean_sources_keys = {"clean_sources"}
    back_path_keys = {"backdoored_path", "backdoor_path"}
    back_indices_keys = {"backdoored_indices", "backdoor_indices"}
    back_sources_keys = {"backdoored_sources", "backdoor_sources"}
    all_structured_keys = (
        clean_path_keys
        | clean_indices_keys
        | clean_sources_keys
        | back_path_keys
        | back_indices_keys
        | back_sources_keys
    )

    has_any_structured_key = any(key in all_structured_keys for key, _ in parsed_lines)
    if not has_any_structured_key:
        return None

    for key, _ in parsed_lines:
        if key not in all_structured_keys:
            raise ValueError(
                f"Unknown structured manifest key '{key}' in {manifest_path}. "
                f"Allowed keys: {sorted(all_structured_keys)}"
            )

    clean_sources: list[tuple[str, list[int]]] = []
    back_sources: list[tuple[str, list[int]]] = []
    pending_clean_path: str | None = None
    pending_back_path: str | None = None

    for key, value in parsed_lines:
        if key in clean_sources_keys:
            clean_sources.extend(
                _parse_source_set_value(value, manifest_path=manifest_path, key=key)
            )
            continue
        if key in back_sources_keys:
            back_sources.extend(
                _parse_source_set_value(value, manifest_path=manifest_path, key=key)
            )
            continue

        if key in clean_path_keys:
            if pending_clean_path is not None:
                raise ValueError(
                    f"Found '{key}' before indices for previous clean_path in {manifest_path}"
                )
            pending_clean_path = value
            continue
        if key in clean_indices_keys:
            if pending_clean_path is None:
                raise ValueError(f"Found '{key}' without a preceding clean_path in {manifest_path}")
            clean_indices = _parse_indices_value(value, manifest_path=manifest_path, key=key)
            clean_sources.append((pending_clean_path, clean_indices))
            pending_clean_path = None
            continue

        if key in back_path_keys:
            if pending_back_path is not None:
                raise ValueError(
                    f"Found '{key}' before indices for previous backdoored_path in {manifest_path}"
                )
            pending_back_path = value
            continue
        if key in back_indices_keys:
            if pending_back_path is None:
                raise ValueError(f"Found '{key}' without a preceding backdoored_path in {manifest_path}")
            back_indices = _parse_indices_value(value, manifest_path=manifest_path, key=key)
            back_sources.append((pending_back_path, back_indices))
            pending_back_path = None
            continue

    if pending_clean_path is not None:
        raise ValueError(f"clean_path in {manifest_path} is missing clean_indices")
    if pending_back_path is not None:
        raise ValueError(f"backdoored_path in {manifest_path} is missing backdoored_indices")

    if not clean_sources and not back_sources:
        raise ValueError(
            f"Structured manifest {manifest_path} must include at least one source set"
        )

    entries: list[str] = []
    for path, indices in clean_sources:
        entries.extend(_expand_structured_paths(path, indices))
    for path, indices in back_sources:
        entries.extend(_expand_structured_paths(path, indices))

    return entries


def _resolve_manifest_entry(entry: str, dataset_root: Path) -> tuple[Path, Path]:
    raw = Path(entry)
    if raw.is_absolute():
        resolved = raw
    else:
        resolved = dataset_root / raw
    resolved = resolved.expanduser().resolve()

    if resolved.is_dir():
        model_dir = resolved
        adapter_path = model_dir / "adapter_model.safetensors"
    else:
        model_dir = resolved.parent
        adapter_path = resolved

    if adapter_path.name != "adapter_model.safetensors":
        raise ValueError(
            f"Manifest entry must resolve to a model directory or adapter_model.safetensors: '{entry}'"
        )
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter file not found for manifest entry '{entry}': {adapter_path}")

    return model_dir, adapter_path


def parse_manifest(manifest_path: Path, dataset_root: Path) -> list[ManifestItem]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    raw_lines: list[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = _strip_manifest_line(line)
            if stripped:
                raw_lines.append(stripped)

    structured_entries = _parse_structured_manifest_entries(raw_lines, manifest_path)
    if structured_entries is not None:
        entries = structured_entries
    else:
        entries = raw_lines

    return _parse_manifest_entries(
        entries=entries,
        dataset_root=dataset_root,
        manifest_name=str(manifest_path),
    )


def _parse_manifest_entries(
    entries: list[str],
    dataset_root: Path,
    manifest_name: str,
) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    seen: set[Path] = set()
    for line_no, entry in enumerate(entries, start=1):
        model_dir, adapter_path = _resolve_manifest_entry(entry, dataset_root=dataset_root)
        key = adapter_path.resolve()
        if key in seen:
            raise ValueError(f"Duplicate adapter in manifest {manifest_name}:{line_no} -> {adapter_path}")
        seen.add(key)
        items.append(
            ManifestItem(
                raw_entry=entry,
                model_dir=model_dir,
                adapter_path=adapter_path,
                model_name=model_dir.name,
                label=parse_label(model_dir.name),
            )
        )
    return items


def _parse_json_section_sources(
    section: Any,
    *,
    section_name: str,
    manifest_path: Path,
) -> list[tuple[str, list[int]]]:
    # Preferred form: {"train":[{"path":"...","indices":[0,10]}], "infer":[...]}
    if isinstance(section, list):
        source_list = section
    elif isinstance(section, dict):
        if isinstance(section.get("sources"), list):
            source_list = section["sources"]
        else:
            raise ValueError(
                f"Section '{section_name}' in {manifest_path} must be a list or an object with a 'sources' list"
            )
    else:
        raise ValueError(
            f"Section '{section_name}' in {manifest_path} must be a list or object"
        )

    if not source_list:
        raise ValueError(f"Section '{section_name}' in {manifest_path} is empty")

    sources: list[tuple[str, list[int]]] = []
    for i, item in enumerate(source_list):
        if not isinstance(item, dict):
            raise ValueError(
                f"{section_name}[{i}] in {manifest_path} must be an object with path/indices"
            )
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{section_name}[{i}] in {manifest_path} is missing non-empty 'path'")
        if "indices" not in item:
            raise ValueError(f"{section_name}[{i}] in {manifest_path} is missing 'indices'")
        indices = _parse_indices_value(
            item["indices"],
            manifest_path=manifest_path,
            key=f"{section_name}[{i}].indices",
        )
        sources.append((path.strip(), indices))
    return sources


def parse_joint_manifest_json(
    manifest_path: Path,
    dataset_root: Path,
) -> tuple[list[ManifestItem], list[ManifestItem]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in manifest file {manifest_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON {manifest_path} must be an object")
    if "train" not in payload or "infer" not in payload:
        raise ValueError(f"Manifest JSON {manifest_path} must include 'train' and 'infer' keys")

    train_sources = _parse_json_section_sources(
        payload["train"],
        section_name="train",
        manifest_path=manifest_path,
    )
    infer_sources = _parse_json_section_sources(
        payload["infer"],
        section_name="infer",
        manifest_path=manifest_path,
    )

    train_entries: list[str] = []
    infer_entries: list[str] = []
    for path, indices in train_sources:
        train_entries.extend(_expand_structured_paths(path, indices))
    for path, indices in infer_sources:
        infer_entries.extend(_expand_structured_paths(path, indices))

    train_items = _parse_manifest_entries(
        entries=train_entries,
        dataset_root=dataset_root,
        manifest_name=f"{manifest_path}::train",
    )
    infer_items = _parse_manifest_entries(
        entries=infer_entries,
        dataset_root=dataset_root,
        manifest_name=f"{manifest_path}::infer",
    )
    return train_items, infer_items


def validate_manifests(train_items: list[ManifestItem], infer_items: list[ManifestItem]) -> list[str]:
    warnings: list[str] = []

    if not train_items:
        raise ValueError("Training manifest is empty after filtering comments/blank lines")
    if not infer_items:
        raise ValueError("Inference manifest is empty after filtering comments/blank lines")

    train_paths = {item.adapter_path.resolve() for item in train_items}
    infer_paths = {item.adapter_path.resolve() for item in infer_items}
    overlap = sorted(train_paths & infer_paths)
    if overlap:
        preview = ", ".join(str(p) for p in overlap[:5])
        raise ValueError(f"Training and inference manifests must be disjoint. Overlap examples: {preview}")

    n_unknown = sum(item.label is None for item in infer_items)
    if n_unknown > 0:
        warnings.append(
            f"{n_unknown} inference entries do not match label naming convention; "
            "label-based metrics will be omitted"
        )

    return warnings


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def precision_at_k(labels_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(labels_true[idx] == 1))


def compute_offline_metrics(labels_true: np.ndarray | None, scores: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "auroc": None,
        "auprc": None,
        "precision_at_num_positives": None,
        "precision_at_5": None,
        "precision_at_10": None,
    }
    if labels_true is None:
        return metrics

    positives = int(np.sum(labels_true == 1))
    if positives <= 0 or positives >= len(labels_true):
        return metrics

    try:
        metrics["auroc"] = float(roc_auc_score(labels_true, scores))
    except Exception:
        metrics["auroc"] = None

    try:
        metrics["auprc"] = float(average_precision_score(labels_true, scores))
    except Exception:
        metrics["auprc"] = None

    metrics["precision_at_num_positives"] = precision_at_k(labels_true, scores, positives)
    metrics["precision_at_5"] = precision_at_k(labels_true, scores, 5)
    metrics["precision_at_10"] = precision_at_k(labels_true, scores, 10)
    return metrics


def compute_infer_threshold_rows(
    train_scores: np.ndarray,
    infer_scores: np.ndarray,
    percentiles: list[float],
    infer_labels: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_infer = int(infer_scores.size)

    for pct in percentiles:
        threshold = float(np.percentile(train_scores, pct))
        flagged = infer_scores >= threshold
        n_flagged = int(np.sum(flagged))

        row: dict[str, Any] = {
            "percentile_from_train": float(pct),
            "percentile_from_train_clean": float(pct),
            "threshold": threshold,
            "n_flagged_in_inference": n_flagged,
            "fraction_flagged_in_inference": float(n_flagged / max(1, n_infer)),
        }

        if infer_labels is not None:
            positives = int(np.sum(infer_labels == 1))
            negatives = int(np.sum(infer_labels == 0))
            tp = int(np.sum((infer_labels == 1) & flagged))
            fp = int(np.sum((infer_labels == 0) & flagged))
            if n_flagged > 0:
                row["precision"] = float(tp / n_flagged)
            if positives > 0:
                row["recall"] = float(tp / positives)
            if negatives > 0:
                row["false_positive_rate"] = float(fp / negatives)

        rows.append(row)

    return rows


def save_score_csv(
    output_path: Path,
    model_names: list[str],
    labels: list[int | None],
    scores: np.ndarray,
) -> None:
    ranks = np.argsort(np.argsort(scores))
    pct = ranks / max(1, len(scores) - 1)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "model_name",
                "label",
                "score",
                "score_percentile_rank",
            ],
        )
        writer.writeheader()
        for i, (name, label, score, rank_pct) in enumerate(zip(model_names, labels, scores, pct)):
            writer.writerow(
                {
                    "index": i,
                    "model_name": name,
                    "label": label,
                    "score": float(score),
                    "score_percentile_rank": float(rank_pct),
                }
            )


def _iter_reader_flat_chunks(
    reader: Any,
    *,
    adapter_path: Path,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    block_size: int,
    dtype: np.dtype,
) -> Any:
    keys = tuple(sorted(reader.keys()))
    if keys != expected_keys:
        raise ValueError(f"Key mismatch for {adapter_path}. Expected exactly the same tensor key set")

    for i, key in enumerate(keys):
        expected_shape = expected_shapes[i]
        actual_shape = get_tensor_shape_safe(reader, key)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {adapter_path} at key {key}: "
                f"expected {expected_shape}, found {actual_shape}"
            )

        if hasattr(reader, "get_slice"):
            tensor_reader = reader.get_slice(key)
            chunks = iter_tensor_flat_chunks(
                tensor_slice=tensor_reader,
                shape=actual_shape,
                block_size=block_size,
                dtype=dtype,
            )
        else:
            full = np.asarray(reader.get_tensor(key), dtype=dtype).reshape(-1)
            chunks = (full[j : j + block_size] for j in range(0, full.size, block_size))

        for chunk in chunks:
            yield np.asarray(chunk, dtype=dtype).reshape(-1)


def _stream_matrix_blocks(
    adapter_paths: list[Path],
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    block_size: int,
    dtype: np.dtype,
    n_features: int,
) -> Any:
    if not adapter_paths:
        raise ValueError("adapter_paths must be non-empty")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    with ExitStack() as stack:
        readers = [stack.enter_context(safe_open(path, framework="numpy")) for path in adapter_paths]
        iters = [
            _iter_reader_flat_chunks(
                reader,
                adapter_path=path,
                expected_keys=expected_keys,
                expected_shapes=expected_shapes,
                block_size=block_size,
                dtype=dtype,
            )
            for reader, path in zip(readers, adapter_paths)
        ]

        write_offset = 0
        while True:
            chunks = [next(it, None) for it in iters]
            n_done = sum(chunk is None for chunk in chunks)
            if n_done == len(chunks):
                break
            if n_done != 0:
                raise RuntimeError("Adapter chunk streams are misaligned across inputs")

            first = chunks[0]
            if first is None:
                raise RuntimeError("Unexpected None chunk for first iterator")
            chunk_size = int(first.size)
            if chunk_size <= 0:
                continue

            block = np.empty((len(chunks), chunk_size), dtype=np.float64)
            for i, chunk in enumerate(chunks):
                if chunk is None:
                    raise RuntimeError("Unexpected None chunk for non-finished iterator")
                if int(chunk.size) != chunk_size:
                    raise RuntimeError("Chunk-size mismatch across adapters in streamed block")
                block[i, :] = np.asarray(chunk, dtype=np.float64)

            end = write_offset + chunk_size
            if end > n_features:
                raise RuntimeError(
                    "Feature write overflow in streamed blocks: "
                    f"attempted end={end}, n_features={n_features}"
                )
            yield write_offset, block
            write_offset = end

        if write_offset != n_features:
            raise RuntimeError(
                f"Feature write underflow in streamed blocks: wrote {write_offset}, expected {n_features}"
            )


def compute_train_gram_and_mean_streamed(
    train_items: list[ManifestItem],
    *,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    n_features: int,
    block_size: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    train_paths = [item.adapter_path for item in train_items]
    n_train = len(train_paths)
    gram_raw = np.zeros((n_train, n_train), dtype=np.float64)
    x_mean = np.empty(n_features, dtype=np.float32)

    for start, block in _stream_matrix_blocks(
        train_paths,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        block_size=block_size,
        dtype=dtype,
        n_features=n_features,
    ):
        end = start + block.shape[1]
        gram_raw += block @ block.T
        x_mean[start:end] = block.mean(axis=0).astype(np.float32, copy=False)

    return 0.5 * (gram_raw + gram_raw.T), x_mean


def project_items_with_dual_basis_streamed(
    train_items: list[ManifestItem],
    target_items: list[ManifestItem],
    *,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    n_features: int,
    block_size: int,
    dtype: np.dtype,
    x_mean: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    if x_mean.shape[0] != n_features:
        raise ValueError(f"x_mean length mismatch: expected {n_features}, got {x_mean.shape[0]}")
    if u.shape[0] != len(train_items):
        raise ValueError(f"u rows must match n_train={len(train_items)}, got {u.shape[0]}")
    if s.shape[0] != u.shape[1]:
        raise ValueError(f"s length must match u cols; got len(s)={s.shape[0]}, u={u.shape}")

    train_paths = [item.adapter_path for item in train_items]
    target_paths = [item.adapter_path for item in target_items]
    n_target = len(target_paths)
    n_train = len(train_paths)
    rank = int(s.shape[0])
    if rank <= 0:
        raise ValueError("No positive singular values available for projection")

    cross = np.zeros((n_target, n_train), dtype=np.float64)

    train_iter = _stream_matrix_blocks(
        train_paths,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        block_size=block_size,
        dtype=dtype,
        n_features=n_features,
    )
    target_iter = _stream_matrix_blocks(
        target_paths,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        block_size=block_size,
        dtype=dtype,
        n_features=n_features,
    )

    while True:
        train_chunk = next(train_iter, None)
        target_chunk = next(target_iter, None)
        if train_chunk is None and target_chunk is None:
            break
        if train_chunk is None or target_chunk is None:
            raise RuntimeError("Train/target stream lengths differ while projecting")

        train_start, train_block = train_chunk
        target_start, target_block = target_chunk
        if train_start != target_start:
            raise RuntimeError(
                f"Chunk offset mismatch during projection: train={train_start}, target={target_start}"
            )
        if train_block.shape[1] != target_block.shape[1]:
            raise RuntimeError(
                "Chunk width mismatch during projection: "
                f"train={train_block.shape[1]}, target={target_block.shape[1]}"
            )

        end = train_start + train_block.shape[1]
        mean_block = np.asarray(x_mean[train_start:end], dtype=np.float64)[None, :]
        train_block_centered = train_block - mean_block
        target_block_centered = target_block - mean_block
        cross += target_block_centered @ train_block_centered.T

    u_rank = u[:, :rank]
    inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0.0)
    z = (cross @ u_rank) * inv_s[None, :]
    return z


def _fit_dual_svd_once(
    gram_centered: np.ndarray,
    requested_max_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(gram_centered)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    positive = eigvals > 0.0
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]
    if eigvals.size == 0:
        raise RuntimeError("No positive singular values found in train Gram matrix")

    rank = min(requested_max_k, eigvals.size)
    s = np.sqrt(eigvals[:rank])
    u = eigvecs[:, :rank]
    z = u * s
    return z, s, u


def main() -> None:
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.stream_block_size <= 0:
        raise ValueError(f"--stream-block-size must be positive, got {args.stream_block_size}")
    if args.n_init <= 0:
        raise ValueError(f"--n-init must be positive, got {args.n_init}")
    if args.reg_covar < 0.0:
        raise ValueError(f"--reg-covar must be >= 0, got {args.reg_covar}")
    if args.manifest_json is not None and (args.train_list is not None or args.infer_list is not None):
        raise ValueError("Use either --manifest-json or (--train-list and --infer-list), not both")
    if args.manifest_json is None and (args.train_list is None or args.infer_list is None):
        raise ValueError("Provide --manifest-json, or provide both --train-list and --infer-list")

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SVD + GMM inference pipeline")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    if args.manifest_json is not None:
        print(f"Joint manifest JSON: {args.manifest_json}")
    else:
        print(f"Train manifest: {args.train_list}")
        print(f"Infer manifest: {args.infer_list}")
    print(f"Output dir: {args.output_dir}")

    if args.manifest_json is not None:
        train_items, infer_items = parse_joint_manifest_json(
            manifest_path=args.manifest_json,
            dataset_root=dataset_root,
        )
    else:
        if args.train_list is None or args.infer_list is None:
            raise RuntimeError("Internal error: missing train/infer manifest paths")
        train_items = parse_manifest(args.train_list, dataset_root=dataset_root)
        infer_items = parse_manifest(args.infer_list, dataset_root=dataset_root)
    warnings = validate_manifests(train_items, infer_items)

    train_labels_list = [item.label for item in train_items]
    n_train_clean = int(np.sum([label == 0 for label in train_labels_list]))
    n_train_backdoored = int(np.sum([label == 1 for label in train_labels_list]))

    print(
        f"Resolved train samples: {len(train_items)} "
        f"(clean={n_train_clean}, backdoored={n_train_backdoored})"
    )
    print(f"Resolved inference samples: {len(infer_items)}")

    # Build raw train Gram + feature means via block streaming (no full X_train materialization).
    first_adapter = train_items[0].adapter_path
    expected_keys, expected_shapes, layers, n_params = inspect_adapter_schema(
        adapter_path=first_adapter,
        expected_keys=None,
        expected_shapes=None,
    )

    n_train = len(train_items)

    component_warnings: list[str] = []
    gmm_warnings: list[str] = []

    if args.keep_vt_cache:
        warnings.append("--keep-vt-cache is deprecated and ignored (streamed projection path)")
    if args.scratch_dir is not None:
        warnings.append("--scratch-dir is deprecated and ignored (no memmap scratch files are used)")

    gram_raw, x_mean = compute_train_gram_and_mean_streamed(
        train_items=train_items,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        n_features=n_params,
        block_size=args.stream_block_size,
        dtype=dtype,
    )
    gram_centered = center_sample_gram(gram_raw)

    component_grid, component_warnings, max_rank = sanitize_component_grid(
        requested=args.svd_components_grid,
        n_samples=n_train,
        n_features=n_params,
    )
    gmm_components, gmm_warnings = sanitize_gmm_components(args.gmm_components, n_train)
    warnings.extend(component_warnings + gmm_warnings)

    requested_max_k = max(component_grid)
    z_train_max, s_max, u_max = _fit_dual_svd_once(
        gram_centered=gram_centered,
        requested_max_k=requested_max_k,
    )

    if s_max.size < requested_max_k:
        warnings.append(
            f"Requested max k={requested_max_k}, but only {s_max.size} positive singular values were available"
        )
        component_grid = [k for k in component_grid if k <= s_max.size]
        if not component_grid:
            raise RuntimeError("No valid SVD component counts after positive-singular-value filtering")

    z_infer_max = project_items_with_dual_basis_streamed(
        train_items=train_items,
        target_items=infer_items,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        n_features=n_params,
        block_size=args.stream_block_size,
        dtype=dtype,
        x_mean=x_mean,
        u=u_max[:, : max(component_grid)],
        s=s_max[: max(component_grid)],
    )

    # Tune k + GMM by mean BIC on train embeddings.
    covariance_order = {name: i for i, name in enumerate(args.gmm_covariance_types)}
    candidate_rows: list[dict[str, Any]] = []
    winner_tuple: tuple[float, float, int, int] | None = None
    winner_candidate: dict[str, Any] | None = None

    for k in component_grid:
        z_train_k = z_train_max[:, :k]
        for cov in args.gmm_covariance_types:
            if cov not in {"full", "tied", "diag", "spherical"}:
                warnings.append(f"Skipping unknown covariance type: {cov}")
                continue
            for n_components in gmm_components:
                run_rows: list[dict[str, Any]] = []
                bics: list[float] = []
                aics: list[float] = []
                for seed in args.stability_seeds:
                    try:
                        gmm = GaussianMixture(
                            n_components=n_components,
                            covariance_type=cov,
                            reg_covar=args.reg_covar,
                            n_init=args.n_init,
                            random_state=seed,
                        )
                        gmm.fit(z_train_k)
                        bic = float(gmm.bic(z_train_k))
                        aic = float(gmm.aic(z_train_k))
                        bics.append(bic)
                        aics.append(aic)
                        run_rows.append(
                            {
                                "seed": int(seed),
                                "bic": bic,
                                "aic": aic,
                                "converged": bool(getattr(gmm, "converged_", False)),
                                "n_iter": int(getattr(gmm, "n_iter_", 0)),
                            }
                        )
                    except Exception as exc:
                        run_rows.append(
                            {
                                "seed": int(seed),
                                "error": str(exc),
                            }
                        )

                if not bics:
                    candidate_rows.append(
                        {
                            "k": int(k),
                            "n_components": int(n_components),
                            "covariance_type": cov,
                            "successful_runs": 0,
                            "total_runs": len(args.stability_seeds),
                            "mean_bic": None,
                            "std_bic": None,
                            "mean_aic": None,
                            "run_details": run_rows,
                        }
                    )
                    continue

                mean_bic = float(np.mean(bics))
                std_bic = float(np.std(bics))
                mean_aic = float(np.mean(aics))
                candidate = {
                    "k": int(k),
                    "n_components": int(n_components),
                    "covariance_type": cov,
                    "successful_runs": len(bics),
                    "total_runs": len(args.stability_seeds),
                    "mean_bic": mean_bic,
                    "std_bic": std_bic,
                    "mean_aic": mean_aic,
                    "run_details": run_rows,
                }
                candidate_rows.append(candidate)

                tie_key = (
                    mean_bic,
                    std_bic,
                    int(n_components),
                    covariance_order.get(cov, int(1e9)),
                )
                if winner_tuple is None or tie_key < winner_tuple:
                    winner_tuple = tie_key
                    winner_candidate = candidate

    if winner_candidate is None:
        raise RuntimeError("No valid GMM candidate could be fit from the requested grid")

    # Refit winner config, choose best seed-run by minimum BIC.
    k_star = int(winner_candidate["k"])
    n_star = int(winner_candidate["n_components"])
    cov_star = str(winner_candidate["covariance_type"])
    z_train_star = z_train_max[:, :k_star]
    z_infer_star = z_infer_max[:, :k_star]

    best_model: GaussianMixture | None = None
    best_seed = None
    best_bic = None
    winner_seed_rows: list[dict[str, Any]] = []

    for seed in args.stability_seeds:
        try:
            gmm = GaussianMixture(
                n_components=n_star,
                covariance_type=cov_star,
                reg_covar=args.reg_covar,
                n_init=args.n_init,
                random_state=seed,
            )
            gmm.fit(z_train_star)
            bic = float(gmm.bic(z_train_star))
            winner_seed_rows.append(
                {
                    "seed": int(seed),
                    "bic": bic,
                    "aic": float(gmm.aic(z_train_star)),
                    "converged": bool(getattr(gmm, "converged_", False)),
                    "n_iter": int(getattr(gmm, "n_iter_", 0)),
                }
            )
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_model = gmm
                best_seed = int(seed)
        except Exception as exc:
            winner_seed_rows.append({"seed": int(seed), "error": str(exc)})

    if best_model is None:
        raise RuntimeError("Winner config selected but refit failed for all seeds")

    train_scores = -best_model.score_samples(z_train_star)
    infer_scores = -best_model.score_samples(z_infer_star)

    train_clean_scores = np.asarray(
        [score for score, label in zip(train_scores, train_labels_list) if label == 0],
        dtype=np.float64,
    )
    train_backdoor_scores = np.asarray(
        [score for score, label in zip(train_scores, train_labels_list) if label == 1],
        dtype=np.float64,
    )

    infer_labels_list = [item.label for item in infer_items]
    infer_known_mask = np.asarray([label is not None for label in infer_labels_list], dtype=bool)
    infer_labels_np: np.ndarray | None = None
    if np.all(infer_known_mask):
        infer_labels_np = np.asarray(infer_labels_list, dtype=np.int32)
    else:
        infer_labels_np = None

    threshold_rows = compute_infer_threshold_rows(
        train_scores=train_scores,
        infer_scores=infer_scores,
        percentiles=args.score_percentiles,
        infer_labels=infer_labels_np,
    )
    offline_metrics = compute_offline_metrics(infer_labels_np, infer_scores)

    infer_clean_scores = np.asarray(
        [score for score, label in zip(infer_scores, infer_labels_list) if label == 0],
        dtype=np.float64,
    )
    infer_backdoor_scores = np.asarray(
        [score for score, label in zip(infer_scores, infer_labels_list) if label == 1],
        dtype=np.float64,
    )

    candidate_rows.sort(
        key=lambda row: (
            float("inf") if row.get("mean_bic") is None else float(row["mean_bic"]),
            float("inf") if row.get("std_bic") is None else float(row["std_bic"]),
            int(row["n_components"]),
            covariance_order.get(str(row["covariance_type"]), int(1e9)),
        )
    )

    run_config = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "resolved_component_grid": component_grid,
        "resolved_gmm_components": gmm_components,
        "warnings": warnings,
    }

    report = {
        "data_info": {
            "dataset_root": str(dataset_root),
            "n_train": len(train_items),
            "n_train_clean": n_train_clean,
            "n_train_backdoored": n_train_backdoored,
            "n_train_unknown_label": int(np.sum([label is None for label in train_labels_list])),
            "n_inference": len(infer_items),
            "n_inference_clean": int(np.sum([label == 0 for label in infer_labels_list])),
            "n_inference_backdoored": int(np.sum([label == 1 for label in infer_labels_list])),
            "n_inference_unknown_label": int(np.sum([label is None for label in infer_labels_list])),
            "train_model_names": [item.model_name for item in train_items],
            "inference_model_names": [item.model_name for item in infer_items],
        },
        "representation": {
            "n_features_raw": int(n_params),
            "layers": layers,
            "svd_components_grid_requested": args.svd_components_grid,
            "svd_components_grid_resolved": component_grid,
            "chosen_k": k_star,
            "max_available_rank": int(max_rank),
        },
        "gmm_selection": {
            "criterion": "mean_bic_on_train",
            "tie_breaker": [
                "lower_bic_std",
                "lower_n_components",
                "covariance_order_as_passed",
            ],
            "candidates": candidate_rows,
            "winner": {
                "k": k_star,
                "n_components": n_star,
                "covariance_type": cov_star,
                "mean_bic": winner_candidate["mean_bic"],
                "std_bic": winner_candidate["std_bic"],
                "selected_seed": best_seed,
                "selected_seed_bic": best_bic,
                "seed_runs": winner_seed_rows,
            },
        },
        "fit_assessment": {
            "score_definition": "negative_log_likelihood",
            "train_score_summary": summarize_scores(train_scores),
            "train_clean_score_summary": (
                summarize_scores(train_clean_scores) if train_clean_scores.size > 0 else None
            ),
            "train_backdoor_score_summary": (
                summarize_scores(train_backdoor_scores) if train_backdoor_scores.size > 0 else None
            ),
            "inference_score_summary": summarize_scores(infer_scores),
            "inference_clean_score_summary": (
                summarize_scores(infer_clean_scores) if infer_clean_scores.size > 0 else None
            ),
            "inference_backdoor_score_summary": (
                summarize_scores(infer_backdoor_scores) if infer_backdoor_scores.size > 0 else None
            ),
            "threshold_evaluation": threshold_rows,
            "offline_metrics": offline_metrics,
        },
        "artifacts": {
            "train_scores_csv": str(args.output_dir / "train_scores.csv"),
            "train_clean_scores_csv": str(args.output_dir / "train_clean_scores.csv"),
            "inference_scores_csv": str(args.output_dir / "inference_scores.csv"),
            "run_config": str(args.output_dir / "run_config.json"),
            "report": str(args.output_dir / "gmm_train_inference_report.json"),
            "vt_cache": None,
        },
        "warnings": warnings,
    }

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(run_config), f, indent=2)
    with open(args.output_dir / "gmm_train_inference_report.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    save_score_csv(
        output_path=args.output_dir / "train_scores.csv",
        model_names=[item.model_name for item in train_items],
        labels=train_labels_list,
        scores=train_scores,
    )
    save_score_csv(
        output_path=args.output_dir / "train_clean_scores.csv",
        model_names=[item.model_name for item in train_items],
        labels=train_labels_list,
        scores=train_scores,
    )
    save_score_csv(
        output_path=args.output_dir / "inference_scores.csv",
        model_names=[item.model_name for item in infer_items],
        labels=infer_labels_list,
        scores=infer_scores,
    )

    print("\nSaved artifacts:")
    print(f"  - {args.output_dir / 'gmm_train_inference_report.json'}")
    print(f"  - {args.output_dir / 'inference_scores.csv'}")
    print(f"  - {args.output_dir / 'train_scores.csv'}")
    print(f"  - {args.output_dir / 'train_clean_scores.csv'}")
    print(f"  - {args.output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
