#!/usr/bin/env python3
"""
Phase 1: Data Preparation + SVD + Representation Audit

This script:
1. Loads LoRA adapters from a JSON manifest
2. Validates key/shape consistency across selected adapters
3. Flattens adapter tensors into parameter vectors
4. Fits memory-safe dual-space truncated SVD from a streamed centered Gram matrix
5. Computes representativeness diagnostics for each embedding space
6. Saves artifacts and run metadata for downstream unsupervised analysis
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from safetensors import safe_open
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score

SCRIPT_VERSION = "3.1.0"
DEFAULT_OUTPUT_DIR = Path("processed_data")
DEFAULT_DATASET_ROOT = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LoRA adapter vectors and SVD embeddings with representativeness metrics"
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        required=True,
        help=(
            "JSON manifest describing adapters via structured path+indices entries. "
            "Expected single-set shape: {\"path\":[{\"path\":\"...\",\"indices\":[a,b]}]}"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root used to resolve relative manifest paths",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output artifact directory")
    parser.add_argument(
        "--trunc-svds-components",
        nargs="+",
        type=int,
        default=[20, 25, 30],
        help="Requested truncated-SVD component counts",
    )
    parser.add_argument(
        "--svd-backend",
        choices=["auto", "dual"],
        default="auto",
        help="SVD backend. 'auto' resolves to memory-safe dual SVD",
    )
    parser.add_argument(
        "--save-vt",
        action="store_true",
        help="Save Vt_<k>.npy (can be very large for high k)",
    )
    parser.add_argument(
        "--stream-block-size",
        type=int,
        default=131072,
        help="Number of feature columns processed per streaming block",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Working dtype for feature matrix",
    )
    parser.add_argument(
        "--save-x-raw",
        action="store_true",
        help="Save full raw feature matrix and mean vector for downstream audits",
    )
    parser.add_argument(
        "--disable-offline-label-diagnostics",
        action="store_true",
        help="Skip offline label diagnostics (linear probe, class distance stats)",
    )
    parser.add_argument(
        "--acceptance-spearman-threshold",
        type=float,
        default=0.99,
        help="Minimum Spearman distance correlation to pass representativeness gate",
    )
    parser.add_argument(
        "--acceptance-variance-threshold",
        type=float,
        default=0.95,
        help="Minimum cumulative explained variance to pass representativeness gate",
    )
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(x) for x in obj]
    return obj


def parse_label(model_dir_name: str) -> int | None:
    if "label0" in model_dir_name:
        return 0
    if "label1" in model_dir_name:
        return 1
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

    if len(values) == 2 and values[1] >= values[0]:
        return list(range(values[0], values[1] + 1))

    dedup: list[int] = []
    seen: set[int] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def _parse_indices_value(value: Any, *, manifest_path: Path, key: str) -> list[int]:
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


def _expand_structured_paths(path_pattern: str, indices: list[int]) -> list[str]:
    entries: list[str] = []
    for i in indices:
        if "{i}" in path_pattern:
            entries.append(path_pattern.format(i=i))
        else:
            entries.append(f"{path_pattern}{i}")
    return entries


def _parse_json_manifest_sources(
    section: Any,
    *,
    section_name: str,
    manifest_path: Path,
) -> list[tuple[str, list[int]]]:
    if isinstance(section, list):
        source_list = section
    elif isinstance(section, dict):
        if "path" in section and "indices" in section:
            source_list = [section]
        elif isinstance(section.get("sources"), list):
            source_list = section["sources"]
        else:
            raise ValueError(
                f"Section '{section_name}' in {manifest_path} must be a list, a path/indices object, "
                "or an object with a 'sources' list"
            )
    else:
        raise ValueError(f"Section '{section_name}' in {manifest_path} must be a list or object")

    if not source_list:
        raise ValueError(f"Section '{section_name}' in {manifest_path} is empty")

    sources: list[tuple[str, list[int]]] = []
    for i, item in enumerate(source_list):
        if not isinstance(item, dict):
            raise ValueError(f"{section_name}[{i}] in {manifest_path} must be an object with path/indices")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{section_name}[{i}] in {manifest_path} is missing non-empty 'path'")
        if "indices" not in item:
            raise ValueError(f"{section_name}[{i}] in {manifest_path} is missing 'indices'")
        indices = _parse_indices_value(item["indices"], manifest_path=manifest_path, key=f"{section_name}[{i}].indices")
        sources.append((path.strip(), indices))
    return sources


def _resolve_manifest_entry(entry: str, dataset_root: Path) -> Path:
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

    return model_dir


def select_model_dirs_from_manifest(
    manifest_json: Path,
    dataset_root: Path,
) -> list[Path]:
    if not manifest_json.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_json}")

    with open(manifest_json, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in manifest file {manifest_json}") from exc

    sources: list[tuple[str, list[int]]]
    if isinstance(payload, list):
        sources = _parse_json_manifest_sources(
            payload,
            section_name="manifest",
            manifest_path=manifest_json,
        )
    elif isinstance(payload, dict):
        if "path" in payload and "indices" in payload:
            sources = _parse_json_manifest_sources(
                payload,
                section_name="manifest",
                manifest_path=manifest_json,
            )
        elif "path" in payload:
            sources = _parse_json_manifest_sources(
                payload["path"],
                section_name="path",
                manifest_path=manifest_json,
            )
        elif "sources" in payload:
            sources = _parse_json_manifest_sources(
                payload["sources"],
                section_name="sources",
                manifest_path=manifest_json,
            )
        else:
            raise ValueError(
                f"Manifest JSON {manifest_json} must include one of: "
                "'path' (single-set), 'sources', or top-level path/indices object"
            )
    else:
        raise ValueError(f"Manifest JSON {manifest_json} must be an object or list")

    entries: list[str] = []
    for path_pattern, indices in sources:
        entries.extend(_expand_structured_paths(path_pattern, indices))

    selected_dirs: list[Path] = []
    seen: set[Path] = set()
    for line_no, entry in enumerate(entries, start=1):
        model_dir = _resolve_manifest_entry(entry=entry, dataset_root=dataset_root)
        key = model_dir.resolve()
        if key in seen:
            raise ValueError(f"Duplicate adapter in manifest {manifest_json}:{line_no} -> {model_dir}")
        seen.add(key)
        selected_dirs.append(model_dir)

    return selected_dirs


def extract_layers_from_keys(keys: tuple[str, ...]) -> list[int]:
    layers: set[int] = set()
    for key in keys:
        if ".layers." not in key:
            continue
        parts = key.split(".")
        layer_idx = parts.index("layers") + 1
        layers.add(int(parts[layer_idx]))
    return sorted(layers)


def get_tensor_shape_safe(f: Any, key: str) -> tuple[int, ...]:
    if hasattr(f, "get_slice"):
        return tuple(int(x) for x in f.get_slice(key).get_shape())
    return tuple(int(x) for x in f.get_tensor(key).shape)


def inspect_adapter_schema(
    adapter_path: Path,
    expected_keys: tuple[str, ...] | None,
    expected_shapes: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], list[int], int]:
    with safe_open(adapter_path, framework="numpy") as f:
        keys = tuple(sorted(f.keys()))
        if expected_keys is not None and keys != expected_keys:
            raise ValueError(f"Key mismatch for {adapter_path}. Expected exactly the same tensor key set")

        shapes: list[tuple[int, ...]] = []
        n_params = 0
        for i, key in enumerate(keys):
            shape = get_tensor_shape_safe(f, key)
            shapes.append(shape)
            if expected_shapes is not None and shape != expected_shapes[i]:
                raise ValueError(
                    f"Shape mismatch for {adapter_path} at key {key}: "
                    f"expected {expected_shapes[i]}, found {shape}"
                )
            n_params += int(np.prod(shape, dtype=np.int64))

    return keys, tuple(shapes), extract_layers_from_keys(keys), n_params


def iter_tensor_flat_chunks(
    tensor_slice: Any,
    shape: tuple[int, ...],
    block_size: int,
    dtype: np.dtype,
) -> Any:
    if len(shape) == 0:
        chunk = np.asarray(tensor_slice[()], dtype=dtype).reshape(-1)
        if chunk.size:
            yield chunk
        return

    if len(shape) == 1:
        step = max(1, block_size)
        for start in range(0, shape[0], step):
            end = min(start + step, shape[0])
            chunk = np.asarray(tensor_slice[start:end], dtype=dtype).reshape(-1)
            if chunk.size:
                yield chunk
        return

    row_width = int(np.prod(shape[1:], dtype=np.int64))
    rows_per_chunk = max(1, block_size // max(1, row_width))
    for row_start in range(0, shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, shape[0])
        chunk = np.asarray(tensor_slice[row_start:row_end], dtype=dtype).reshape(-1)
        if chunk.size == 0:
            continue
        for offset in range(0, chunk.size, block_size):
            sub = chunk[offset : offset + block_size]
            if sub.size:
                yield sub


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
    *,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    block_size: int,
    dtype: np.dtype,
    n_features: int,
) -> Any:
    if not adapter_paths:
        raise ValueError("adapter_paths must be non-empty")
    if block_size <= 0:
        raise ValueError(f"stream_block_size must be positive, got {block_size}")

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


def compute_dataset_signature(model_names: list[str], keys: tuple[str, ...], shapes: tuple[tuple[int, ...], ...]) -> str:
    h = hashlib.sha256()
    for name in model_names:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
    for key, shape in zip(keys, shapes):
        h.update(key.encode("utf-8"))
        h.update(str(shape).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_gram_and_feature_means_streamed(
    adapter_paths: list[Path],
    *,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    n_features: int,
    block_size: int,
    dtype: np.dtype,
    compute_feature_means: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    n_samples = len(adapter_paths)
    gram_raw = np.zeros((n_samples, n_samples), dtype=np.float64)
    feature_means = np.empty(n_features, dtype=np.float32) if compute_feature_means else None

    for start, block in _stream_matrix_blocks(
        adapter_paths=adapter_paths,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        block_size=block_size,
        dtype=dtype,
        n_features=n_features,
    ):
        end = start + block.shape[1]
        gram_raw += block @ block.T
        if feature_means is not None:
            feature_means[start:end] = block.mean(axis=0).astype(np.float32, copy=False)

    return 0.5 * (gram_raw + gram_raw.T), feature_means


def center_sample_gram(gram_raw: np.ndarray) -> np.ndarray:
    """
    Center a sample-space Gram matrix without materializing centered features.

    If A = X X^T, this computes H A H where H = I - 11^T / n.
    """
    if gram_raw.ndim != 2 or gram_raw.shape[0] != gram_raw.shape[1]:
        raise ValueError(f"gram_raw must be square, got shape={gram_raw.shape}")

    row_mean = gram_raw.mean(axis=1, dtype=np.float64)
    grand_mean = float(row_mean.mean())
    centered = gram_raw - row_mean[:, None] - row_mean[None, :] + grand_mean
    return 0.5 * (centered + centered.T)


def pairwise_distances_from_gram(gram: np.ndarray) -> np.ndarray:
    diag = np.diag(gram).astype(np.float64, copy=False)
    dist2 = diag[:, None] + diag[None, :] - (2.0 * gram)
    np.maximum(dist2, 0.0, out=dist2)
    np.fill_diagonal(dist2, 0.0)
    return np.sqrt(dist2, out=dist2)


def compute_vt_from_adapters_streaming(
    adapter_paths: list[Path],
    *,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    feature_means: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    n_features: int,
    block_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    if block_size <= 0:
        raise ValueError(f"stream_block_size must be positive, got {block_size}")
    if feature_means.shape[0] != n_features:
        raise ValueError(f"feature_means length mismatch: expected {n_features}, got {feature_means.shape[0]}")

    rank = int(s.size)
    if rank <= 0:
        raise ValueError("No singular values provided for Vt computation")
    if u.shape != (len(adapter_paths), rank):
        raise ValueError(
            f"U shape mismatch for Vt computation: expected ({len(adapter_paths)}, {rank}), got {u.shape}"
        )

    vt = np.empty((rank, n_features), dtype=np.float64)
    inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0.0)

    for start, block in _stream_matrix_blocks(
        adapter_paths=adapter_paths,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        block_size=block_size,
        dtype=dtype,
        n_features=n_features,
    ):
        end = start + block.shape[1]
        mean_block = np.asarray(feature_means[start:end], dtype=np.float64)[None, :]
        vt[:, start:end] = (u.T @ (block - mean_block)) * inv_s[:, None]

    return vt


def save_x_raw_streamed(
    output_path: Path,
    *,
    adapter_paths: list[Path],
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    n_features: int,
    block_size: int,
    dtype: np.dtype,
) -> None:
    x_raw = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(adapter_paths), n_features),
    )
    try:
        for start, block in _stream_matrix_blocks(
            adapter_paths=adapter_paths,
            expected_keys=expected_keys,
            expected_shapes=expected_shapes,
            block_size=block_size,
            dtype=dtype,
            n_features=n_features,
        ):
            end = start + block.shape[1]
            x_raw[:, start:end] = block.astype(np.float32, copy=False)
    finally:
        x_raw.flush()
        del x_raw


def sanitize_component_grid(requested: list[int], n_samples: int, n_features: int) -> tuple[list[int], list[str], int]:
    max_rank = max(1, min(n_samples - 1, n_features))
    warnings: list[str] = []
    sanitized: list[int] = []

    for k in requested:
        if k <= 0:
            warnings.append(f"Ignoring non-positive component request: {k}")
            continue
        clipped = min(k, max_rank)
        if clipped != k:
            warnings.append(f"Clipped n_components from {k} to {clipped} (rank limit={max_rank})")
        sanitized.append(clipped)

    sanitized = sorted(set(sanitized))
    if not sanitized:
        sanitized = [max_rank]
        warnings.append(f"No valid n_components requested; using fallback [{max_rank}]")

    return sanitized, warnings, max_rank


def resolve_svd_backend(backend: str, n_samples: int, n_features: int) -> str:
    """Resolve backend choice. Auto maps to memory-safe dual SVD."""
    del n_samples, n_features
    if backend != "auto":
        return backend
    return "dual"


def truncated_svd_dual(
    gram_centered: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-safe truncated SVD via sample-space eigendecomposition.

    Given centered Gram matrix G = X_c X_c^T:
      G = U diag(s^2) U^T
    gives singular values s and left singular vectors U directly.
    """
    eigvals, eigvecs = np.linalg.eigh(gram_centered)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    positive = eigvals > 0.0
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]
    if eigvals.size == 0:
        raise RuntimeError("No positive singular values found in dual SVD")

    rank = min(n_components, eigvals.size)
    s = np.sqrt(eigvals[:rank])
    u = eigvecs[:, :rank]
    z = u * s

    return z, s, u


def compute_nn_indices(distance_matrix: np.ndarray) -> np.ndarray:
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    return np.argmin(d, axis=1)


def compute_representation_metrics(
    dx: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    run_offline_label_diagnostics: bool,
) -> dict[str, Any]:
    dz = pairwise_distances(z, metric="euclidean")
    tri = np.triu_indices_from(dx, k=1)

    x_flat = dx[tri]
    z_flat = dz[tri]

    metrics: dict[str, Any] = {
        "pairwise_distance_spearman": float(spearmanr(x_flat, z_flat).correlation),
        "pairwise_distance_pearson": float(pearsonr(x_flat, z_flat).statistic),
        "mean_relative_distance_error": float(np.mean(np.abs(x_flat - z_flat) / (x_flat + 1e-8))),
    }

    nn_x = compute_nn_indices(dx)
    nn_z = compute_nn_indices(dz)
    metrics["nn_overlap"] = float(np.mean(nn_x == nn_z))
    metrics["z_1nn_label_agreement"] = float(np.mean(labels[nn_z] == labels))

    if run_offline_label_diagnostics:
        unique_labels, counts = np.unique(labels, return_counts=True)
        if unique_labels.size >= 2 and counts.min() >= 2:
            n_splits = min(5, int(counts.min()))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=5000)
            scores = cross_val_score(clf, z, labels, cv=cv, scoring="accuracy")
            metrics["linear_probe_cv_accuracy_mean"] = float(scores.mean())
            metrics["linear_probe_cv_accuracy_std"] = float(scores.std())

            z0 = z[labels == 0]
            z1 = z[labels == 1]
            c0 = z0.mean(axis=0)
            c1 = z1.mean(axis=0)
            metrics["class_centroid_distance"] = float(np.linalg.norm(c0 - c1))
            metrics["within_class_mean_distance_label0"] = float(np.mean(np.linalg.norm(z0 - c0, axis=1)))
            metrics["within_class_mean_distance_label1"] = float(np.mean(np.linalg.norm(z1 - c1, axis=1)))

    return metrics


def fit_svd_embeddings(
    gram_centered: np.ndarray,
    labels: np.ndarray,
    component_grid: list[int],
    run_offline_label_diagnostics: bool,
    svd_backend: str,
    n_features: int,
    save_vt: bool,
    vt_builder: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    n_samples = int(gram_centered.shape[0])
    total_energy = float(np.trace(gram_centered))
    total_variance = float(total_energy / (n_samples - 1))
    dx = pairwise_distances_from_gram(gram_centered)

    if svd_backend != "dual":
        raise ValueError(f"Unsupported svd_backend={svd_backend}; expected 'dual'")

    for n_components in component_grid:
        print(f"\nComputing {svd_backend} SVD with n_components={n_components}...")
        svd_start = perf_counter()
        z, s, u = truncated_svd_dual(
            gram_centered=gram_centered,
            n_components=n_components,
        )
        vt: np.ndarray | None = None
        if save_vt:
            if vt_builder is None:
                raise RuntimeError("save_vt=True but vt_builder is not provided")
            vt = vt_builder(u, s)
        svd_time_seconds = float(perf_counter() - svd_start)

        explained_variance = (s ** 2) / (n_samples - 1)
        explained_variance_ratio = explained_variance / total_variance
        cumulative_variance = float(np.sum(explained_variance_ratio))

        captured_energy = float(np.sum(s ** 2))
        residual_energy = max(0.0, total_energy - captured_energy)
        reconstruction_mse = residual_energy / (n_samples * n_features)
        relative_reconstruction_error = residual_energy / (total_energy + 1e-12)

        representativeness = compute_representation_metrics(
            dx=dx,
            z=z,
            labels=labels,
            run_offline_label_diagnostics=run_offline_label_diagnostics,
        )

        results[n_components] = {
            "Z": z.astype(np.float32),
            "Vt": None if vt is None else vt.astype(np.float32),
            "singular_values": s.astype(np.float64),
            "explained_variance_ratio": explained_variance_ratio.astype(np.float64),
            "svd_time_seconds": svd_time_seconds,
            "cumulative_variance": cumulative_variance,
            "reconstruction_error": reconstruction_mse,
            "relative_error": relative_reconstruction_error,
            "representativeness": representativeness,
        }

        print(f"  Z shape: {results[n_components]['Z'].shape}")
        print(f"  SVD time: {svd_time_seconds:.3f} s")
        print(f"  Cumulative variance: {cumulative_variance:.6f}")
        print(f"  Pairwise distance Spearman: {representativeness['pairwise_distance_spearman']:.6f}")
        print(f"  Mean relative distance error: {representativeness['mean_relative_distance_error']:.6f}")

    return results


def pick_acceptance_winner(
    svd_results: dict[int, dict[str, Any]],
    spearman_threshold: float,
    variance_threshold: float,
) -> dict[str, Any]:
    candidates = []
    for k in sorted(svd_results.keys()):
        r = svd_results[k]
        spearman_val = r["representativeness"]["pairwise_distance_spearman"]
        var_val = r["cumulative_variance"]
        passed = (spearman_val >= spearman_threshold) and (var_val >= variance_threshold)
        candidates.append(
            {
                "k": int(k),
                "spearman": float(spearman_val),
                "cumulative_variance": float(var_val),
                "passes_gate": bool(passed),
            }
        )

    winner = None
    for entry in candidates:
        if entry["passes_gate"]:
            winner = entry
            break

    return {
        "spearman_threshold": spearman_threshold,
        "variance_threshold": variance_threshold,
        "winner": winner,
        "candidates": candidates,
    }


def save_outputs(
    output_dir: Path,
    x_raw: np.ndarray | None,
    x_mean: np.ndarray | None,
    labels: np.ndarray,
    model_names: list[str],
    metadata: dict[str, Any],
    svd_results: dict[int, dict[str, Any]],
    run_config: dict[str, Any],
    save_x_raw: bool,
    save_vt: bool,
    acceptance_gate: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "labels.npy", labels)
    with open(output_dir / "model_names.json", "w", encoding="utf-8") as f:
        json.dump(model_names, f, indent=2)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(metadata), f, indent=2)

    if save_x_raw:
        if x_raw is None or x_mean is None:
            raise RuntimeError("save_x_raw=True but x_raw/x_mean is not available")
        np.save(output_dir / "X_raw.npy", x_raw.astype(np.float32))
        np.save(output_dir / "X_mean.npy", x_mean.astype(np.float32))

    for n_components, result in svd_results.items():
        np.save(output_dir / f"Z_{n_components}.npy", result["Z"])
        if save_vt and result.get("Vt") is not None:
            np.save(output_dir / f"Vt_{n_components}.npy", result["Vt"])

        svd_info = {
            "n_components": int(n_components),
            "svd_time_seconds": float(result.get("svd_time_seconds", 0.0)),
            "cumulative_variance": float(result["cumulative_variance"]),
            "explained_variance_ratio": result["explained_variance_ratio"].tolist(),
            "singular_values": result["singular_values"].tolist(),
            "reconstruction_error": float(result["reconstruction_error"]),
            "relative_error": float(result["relative_error"]),
            "representativeness": json_ready(result["representativeness"]),
        }
        with open(output_dir / f"svd_info_{n_components}.json", "w", encoding="utf-8") as f:
            json.dump(svd_info, f, indent=2)

    with open(output_dir / "representativeness_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(acceptance_gate), f, indent=2)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(run_config), f, indent=2)

    print(f"\nSaved outputs to {output_dir}")
    for file in sorted(output_dir.iterdir()):
        size = file.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 ** 2:
            size_str = f"{size / 1024:.1f} KB"
        elif size < 1024 ** 3:
            size_str = f"{size / 1024 ** 2:.1f} MB"
        else:
            size_str = f"{size / 1024 ** 3:.2f} GB"
        print(f"  {file.name}: {size_str}")


def main() -> None:
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    print("=" * 80)
    print("PHASE 1: Data Preparation + Truncated SVD + Representativeness Audit")
    print("=" * 80)
    print(f"Output dir: {args.output_dir}")
    print(f"Manifest JSON: {args.manifest_json}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"dtype: {args.dtype}")
    print(f"requested SVD backend: {args.svd_backend}")
    print(f"stream block size: {args.stream_block_size:,}")

    selected_dirs = select_model_dirs_from_manifest(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
    )

    print(f"Found {len(selected_dirs)} selected models")
    if not selected_dirs:
        raise ValueError("No models selected from configured input source")

    labels: list[int] = []
    model_names: list[str] = []
    adapter_paths: list[Path] = []

    first_adapter_path = selected_dirs[0] / "adapter_model.safetensors"
    if not first_adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter file: {first_adapter_path}")
    expected_keys, expected_shapes, layers, n_features = inspect_adapter_schema(
        adapter_path=first_adapter_path,
        expected_keys=None,
        expected_shapes=None,
    )

    for model_dir in selected_dirs:
        adapter_path = model_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing adapter file: {adapter_path}")
        label = parse_label(model_dir.name)
        if label is None:
            raise ValueError(f"Cannot infer label from directory name: {model_dir.name}")
        labels.append(label)
        model_names.append(model_dir.name)
        adapter_paths.append(adapter_path)

    labels_np = np.asarray(labels, dtype=np.int32)
    n_samples = len(adapter_paths)
    per_model_bytes = n_features * np.dtype(dtype).itemsize
    centered_size_bytes = int(n_samples * n_features * np.dtype(dtype).itemsize)

    print(f"  Parameter vector size: {n_features:,}")
    print(f"  Per-model memory ({args.dtype}): {per_model_bytes / 1024 ** 2:.1f} MB")
    print(f"\nData matrix shape: ({n_samples}, {n_features})")
    print(f"Label 0 count: {int(np.sum(labels_np == 0))}")
    print(f"Label 1 count: {int(np.sum(labels_np == 1))}")
    print(f"Centered matrix size (logical): {centered_size_bytes / 1024 ** 3:.2f} GB")
    print("Raw matrix storage: streamed blocks (no /tmp memmap)")

    need_feature_means = args.save_x_raw or args.save_vt
    gram_stream_start = perf_counter()
    gram_raw, feature_means = compute_gram_and_feature_means_streamed(
        adapter_paths=adapter_paths,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        n_features=n_features,
        block_size=args.stream_block_size,
        dtype=dtype,
        compute_feature_means=need_feature_means,
    )
    gram_stream_time_seconds = float(perf_counter() - gram_stream_start)

    if need_feature_means:
        if feature_means is None:
            raise RuntimeError("Feature means were requested but not computed")
        mean_time_seconds = gram_stream_time_seconds
        print("Feature means computed during streamed Gram accumulation pass")
    else:
        mean_time_seconds = 0.0
        print("Feature means skipped (not required without --save-x-raw/--save-vt)")

    gram_center_start = perf_counter()
    gram_centered = center_sample_gram(gram_raw)
    gram_center_time_seconds = float(perf_counter() - gram_center_start)
    gram_time_seconds = gram_stream_time_seconds + gram_center_time_seconds
    print(f"Centered Gram matrix shape: {gram_centered.shape}")
    print(f"Centered Gram matrix size: {gram_centered.nbytes / 1024 ** 2:.2f} MB")
    print(f"Raw Gram streaming accumulation time: {gram_stream_time_seconds:.3f} s")
    print(f"Sample-space centering time: {gram_center_time_seconds:.3f} s")
    print(f"Total Gram preparation time: {gram_time_seconds:.3f} s")

    component_grid, warnings, max_rank = sanitize_component_grid(
        requested=args.trunc_svds_components,
        n_samples=n_samples,
        n_features=n_features,
    )
    resolved_backend = resolve_svd_backend(
        backend=args.svd_backend,
        n_samples=n_samples,
        n_features=n_features,
    )

    if warnings:
        print("\nComponent-grid warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"Using component grid: {component_grid} (max_rank={max_rank})")
    print(f"Resolved SVD backend: {resolved_backend}")

    vt_builder: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    if args.save_vt:
        if feature_means is None:
            raise RuntimeError("save_vt=True requires feature means")

        def _vt_builder(u: np.ndarray, s: np.ndarray) -> np.ndarray:
            return compute_vt_from_adapters_streaming(
                adapter_paths=adapter_paths,
                expected_keys=expected_keys,
                expected_shapes=expected_shapes,
                feature_means=feature_means,
                u=u,
                s=s,
                n_features=n_features,
                block_size=args.stream_block_size,
                dtype=dtype,
            )

        vt_builder = _vt_builder

    svd_results = fit_svd_embeddings(
        gram_centered=gram_centered,
        labels=labels_np,
        component_grid=component_grid,
        run_offline_label_diagnostics=not args.disable_offline_label_diagnostics,
        svd_backend=resolved_backend,
        n_features=n_features,
        save_vt=args.save_vt,
        vt_builder=vt_builder,
    )

    acceptance_gate = pick_acceptance_winner(
        svd_results=svd_results,
        spearman_threshold=args.acceptance_spearman_threshold,
        variance_threshold=args.acceptance_variance_threshold,
    )

    winner = acceptance_gate["winner"]
    if winner is not None:
        print(
            f"\nAcceptance gate winner: k={winner['k']} "
            f"(spearman={winner['spearman']:.4f}, variance={winner['cumulative_variance']:.4f})"
        )
    else:
        print("\nAcceptance gate: no k satisfied thresholds. Inspect representativeness_summary.json")

    save_x_raw_time_seconds = 0.0
    if args.save_x_raw:
        if feature_means is None:
            raise RuntimeError("save_x_raw=True requires feature means")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_x_raw_start = perf_counter()
        save_x_raw_streamed(
            output_path=args.output_dir / "X_raw.npy",
            adapter_paths=adapter_paths,
            expected_keys=expected_keys,
            expected_shapes=expected_shapes,
            n_features=n_features,
            block_size=args.stream_block_size,
            dtype=dtype,
        )
        np.save(args.output_dir / "X_mean.npy", feature_means.astype(np.float32))
        save_x_raw_time_seconds = float(perf_counter() - save_x_raw_start)
        print(f"Saved streamed X_raw.npy/X_mean.npy in {save_x_raw_time_seconds:.3f} s")

    metadata = {
        "layers": layers,
        "n_layers": len(layers),
        "n_params": int(n_features),
        "n_models": int(n_samples),
        "keys": list(expected_keys),
        "shapes": [list(s) for s in expected_shapes],
    }

    dataset_signature = compute_dataset_signature(model_names, expected_keys, expected_shapes)
    run_config = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "dtype": args.dtype,
        "component_grid": component_grid,
        "max_rank": max_rank,
        "resolved_svd_backend": resolved_backend,
        "save_vt": args.save_vt,
        "stream_block_size": args.stream_block_size,
        "feature_means_time_seconds": mean_time_seconds,
        "centered_gram_time_seconds": gram_time_seconds,
        "raw_gram_stream_time_seconds": gram_stream_time_seconds,
        "gram_centering_time_seconds": gram_center_time_seconds,
        "save_x_raw_time_seconds": save_x_raw_time_seconds,
        "dataset_signature": dataset_signature,
        "selected_models": model_names,
        "component_warnings": warnings,
    }

    save_outputs(
        output_dir=args.output_dir,
        x_raw=None,
        x_mean=None,
        labels=labels_np,
        model_names=model_names,
        metadata=metadata,
        svd_results=svd_results,
        run_config=run_config,
        save_x_raw=False,
        save_vt=args.save_vt,
        acceptance_gate=acceptance_gate,
    )

    print("\n" + "=" * 80)
    print("Phase 1 complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
