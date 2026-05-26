from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter
import re
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd().resolve() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from safetensors import safe_open
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ..features.delta import (
    check_consistency_reader,
    iter_block_factors,
    load_delta_block_schema,
    schema_has_adalora_scaling,
    shorten_block_name,
)
from ..features.spectral import build_qv_sum_specs
from ..utilities.artifacts.dataset_references import (
    DATASET_REFERENCE_REPORT_NAME,
    load_dataset_reference_report,
    resolve_dataset_reference_payload_for_artifact,
)
from ..utilities.artifacts.export_feature_subset import (
    _feature_group_for_feature_name,
    _normalize_requested_features,
    _resolve_model_owned_feature_names,
)
from ..utilities.artifacts.spectral_metadata import load_spectral_metadata
from ..utilities.core.run_context import create_run_context
from ..utilities.core.serialization import json_ready


SCRIPT_VERSION = "1.2.0"
DEFAULT_FEATURE_EXTRACT_ROOT = Path("runs") / "feature_extract"
DEFAULT_SPECTRAL_MOMENT_SOURCE = "sv"
SUPPORTED_GROUPINGS = ("rank",)
SUPPORTED_TSNE_VIEWS = ("full", "per_layer")
_RANK_RE = re.compile(r"(?:^|[_-])rank(?P<rank>\d+)(?:[^0-9]|$)")
_LABEL_RE = re.compile(r"_label(?P<label>-?\d+)_")
_LAYER_RE = re.compile(r"layer(?P<layer>\d+)")
_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MOMENT_FEATURE_SET = {"kurtosis", "l1_norm", "linf_norm", "mean_abs"}
_SV_MOMENT_BY_ENTRYWISE = {
    "kurtosis": "sv_kurtosis",
    "l1_norm": "sv_l1_norm",
    "linf_norm": "sv_linf_norm",
    "mean_abs": "sv_mean_abs",
}
_LABEL_STYLE = {
    0: ("clean", "#1f77b4"),
    1: ("backdoor", "#d62728"),
    -1: ("unknown", "#7f7f7f"),
}


@dataclass(frozen=True)
class LoadedFeatureBundle:
    feature_file: Path
    model_names_file: Path
    labels_file: Path | None
    metadata_file: Path | None
    dataset_reference_report: Path | None
    features: np.ndarray
    model_names: list[str]
    labels: np.ndarray | None
    metadata: dict[str, Any]
    feature_names: list[str]
    dataset_reference_payload: dict[str, Any] | None
    sample_dataset_names: list[str | None]
    sample_ranks: list[int | None]


@dataclass(frozen=True)
class RawDeltaBlockSpec:
    kind: str
    block_name: str
    raw_block_name: str
    pair_index: int | None = None
    q_index: int | None = None
    v_index: int | None = None


def _candidate_companion_paths(feature_path: Path, suffix: str) -> list[Path]:
    stem = feature_path.stem
    candidates: list[Path] = []
    if stem.endswith("_features"):
        prefix = stem[: -len("_features")]
        candidates.append(feature_path.with_name(f"{prefix}{suffix}"))
    candidates.append(feature_path.with_name(f"{stem}{suffix}"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _resolve_existing_companion_path(feature_path: Path, suffix: str, *, required: bool) -> Path | None:
    for candidate in _candidate_companion_paths(feature_path, suffix):
        if candidate.exists():
            return candidate.resolve()
    if required:
        joined = ", ".join(str(path) for path in _candidate_companion_paths(feature_path, suffix))
        raise FileNotFoundError(
            f"Could not find required companion file for {feature_path}. Tried: {joined}"
        )
    return None


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _looks_like_explicit_path(path_spec: str | Path) -> bool:
    resolved_path = _coerce_path(path_spec)
    return resolved_path.is_absolute() or len(resolved_path.parts) > 1 or resolved_path.suffix == ".npy"


def _resolve_input_feature_path(feature_spec: str | Path, *, feature_root: Path) -> Path:
    candidate = _coerce_path(feature_spec).expanduser()
    if _looks_like_explicit_path(candidate):
        resolved = candidate if candidate.is_absolute() else (Path.cwd().resolve() / candidate)
        return resolved.resolve()

    run_name = candidate.name
    search_paths = [
        feature_root / run_name / "merged" / "spectral_features.npy",
        feature_root / run_name / "features" / "spectral_features.npy",
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()

    joined = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(
        f"Could not resolve feature run name '{run_name}' under {feature_root}. Tried: {joined}"
    )


def _resolve_feature_root(feature_root: str | Path) -> Path:
    resolved = _coerce_path(feature_root).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd().resolve() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return resolved


def _resolve_path(path: str | Path) -> Path:
    candidate = _coerce_path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd().resolve() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _safe_slug(value: str) -> str:
    return _FILENAME_SANITIZE_RE.sub("_", str(value).strip() or "unknown")


def _extract_rank(*texts: str | None) -> int | None:
    for text in texts:
        if not text:
            continue
        match = _RANK_RE.search(str(text))
        if match is not None:
            return int(match.group("rank"))
    return None


def _extract_label_from_model_name(model_name: str) -> int | None:
    match = _LABEL_RE.search(str(model_name))
    if match is None:
        return None
    return int(match.group("label"))


def _extract_layer(feature_name: str) -> int | None:
    try:
        block_name = _feature_block_name(feature_name)
    except ValueError:
        block_name = str(feature_name)
    match = _LAYER_RE.search(block_name)
    if match is None:
        return None
    return int(match.group("layer"))


def _emitted_feature_name(feature_name: str) -> str:
    _, _, suffix = str(feature_name).rpartition(".")
    if not suffix:
        raise ValueError(f"Could not determine emitted feature name from '{feature_name}'")
    return suffix


def _feature_block_name(feature_name: str) -> str:
    block_name, sep, _ = str(feature_name).rpartition(".")
    if not sep or not block_name:
        raise ValueError(f"Invalid spectral feature name: {feature_name}")
    return block_name


def _expand_spectral_feature_names(
    *,
    selected_features: list[str],
    spectral_moment_source: str,
) -> list[str]:
    resolved_moment_source = str(spectral_moment_source or DEFAULT_SPECTRAL_MOMENT_SOURCE).strip().lower()
    emitted: list[str] = []
    for feature in selected_features:
        if feature not in _MOMENT_FEATURE_SET:
            emitted.append(feature)
            continue
        if resolved_moment_source in {"entrywise", "both"}:
            emitted.append(feature)
        if resolved_moment_source in {"sv", "both"}:
            emitted.append(_SV_MOMENT_BY_ENTRYWISE[feature])
    return emitted


def _build_spectral_feature_names(
    *,
    block_names: list[str],
    selected_features: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
) -> list[str]:
    emitted_features = _expand_spectral_feature_names(
        selected_features=selected_features,
        spectral_moment_source=spectral_moment_source,
    )
    names: list[str] = []
    for block_name in block_names:
        prefix = str(block_name)
        for emitted_feature in emitted_features:
            if emitted_feature == "sv_topk":
                for i in range(int(sv_top_k)):
                    names.append(f"{prefix}.sv_{i + 1}")
            else:
                names.append(f"{prefix}.{emitted_feature}")
    return names


def _resolved_feature_names(metadata: Mapping[str, Any], feature_dim: int) -> list[str]:
    raw_feature_names = metadata.get("feature_names")
    if isinstance(raw_feature_names, list):
        names = [str(x) for x in raw_feature_names]
        if len(names) != feature_dim:
            raise ValueError(
                f"feature_names length ({len(names)}) does not match feature dimension ({feature_dim})"
            )
        return names

    raw_block_names = metadata.get("block_names")
    if isinstance(raw_block_names, list):
        block_names = [str(x) for x in raw_block_names]
        raw_resolved_features = metadata.get("resolved_features")
        if isinstance(raw_resolved_features, list) and raw_resolved_features:
            resolved_features = [str(x) for x in raw_resolved_features]
        else:
            extractor_params = metadata.get("extractor_params")
            if isinstance(extractor_params, dict) and isinstance(extractor_params.get("spectral_features"), list):
                resolved_features = [str(x) for x in extractor_params["spectral_features"]]
            else:
                resolved_features = []

        sv_top_k = metadata.get("sv_top_k")
        if sv_top_k is None:
            extractor_params = metadata.get("extractor_params")
            if isinstance(extractor_params, dict):
                sv_top_k = extractor_params.get("spectral_sv_top_k")

        spectral_moment_source = metadata.get("spectral_moment_source")
        if spectral_moment_source is None:
            extractor_params = metadata.get("extractor_params")
            if isinstance(extractor_params, dict):
                spectral_moment_source = extractor_params.get("spectral_moment_source")
        if spectral_moment_source is None:
            spectral_moment_source = DEFAULT_SPECTRAL_MOMENT_SOURCE

        if resolved_features and sv_top_k is not None:
            inferred = _build_spectral_feature_names(
                block_names=block_names,
                selected_features=resolved_features,
                sv_top_k=int(sv_top_k),
                spectral_moment_source=str(spectral_moment_source),
            )
            if len(inferred) == feature_dim:
                return inferred

    return [f"feature_{i:05d}" for i in range(feature_dim)]


def _resolved_labels(
    *,
    labels_file: Path | None,
    dataset_reference_payload: Mapping[str, Any] | None,
    model_names: list[str],
) -> np.ndarray | None:
    if labels_file is not None and labels_file.exists():
        labels = np.asarray(np.load(labels_file), dtype=np.int32)
        if int(labels.shape[0]) != len(model_names):
            raise ValueError(
                f"Labels length ({labels.shape[0]}) does not match model names length ({len(model_names)})"
            )
        if np.any(labels >= 0):
            return labels
        return None

    labels_from_names = np.full(len(model_names), -1, dtype=np.int32)
    found_known = False

    model_index = dataset_reference_payload.get("model_index") if isinstance(dataset_reference_payload, Mapping) else None
    for i, model_name in enumerate(model_names):
        label_value: int | None = None
        if isinstance(model_index, Mapping):
            raw_entry = model_index.get(model_name)
            if isinstance(raw_entry, Mapping):
                raw_label = raw_entry.get("label")
                if raw_label is not None:
                    try:
                        label_value = int(raw_label)
                    except (TypeError, ValueError):
                        label_value = None
        if label_value is None:
            label_value = _extract_label_from_model_name(model_name)
        if label_value is None:
            continue
        labels_from_names[i] = int(label_value)
        found_known = True

    if found_known:
        return labels_from_names
    return None


def load_feature_bundle(
    *,
    feature_file: Path,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
) -> LoadedFeatureBundle:
    resolved_feature_root = _resolve_feature_root(feature_root)
    resolved_feature_file = _resolve_input_feature_path(feature_file, feature_root=resolved_feature_root)
    if not resolved_feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {resolved_feature_file}")

    resolved_model_names_file = (
        _resolve_path(model_names_file)
        if model_names_file is not None
        else _resolve_existing_companion_path(resolved_feature_file, "_model_names.json", required=True)
    )
    if resolved_model_names_file is None or not resolved_model_names_file.exists():
        raise FileNotFoundError(f"Model names file not found: {resolved_model_names_file}")

    resolved_labels_file = (
        _resolve_path(labels_file)
        if labels_file is not None
        else _resolve_existing_companion_path(resolved_feature_file, "_labels.npy", required=False)
    )
    if resolved_labels_file is not None and not resolved_labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {resolved_labels_file}")

    resolved_metadata_file = (
        _resolve_path(metadata_file)
        if metadata_file is not None
        else _resolve_existing_companion_path(resolved_feature_file, "_metadata.json", required=False)
    )
    if resolved_metadata_file is not None and not resolved_metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {resolved_metadata_file}")

    resolved_dataset_reference_report = (
        _resolve_path(dataset_reference_report) if dataset_reference_report is not None else None
    )
    if resolved_dataset_reference_report is not None and not resolved_dataset_reference_report.exists():
        raise FileNotFoundError(
            f"Dataset reference report not found: {resolved_dataset_reference_report}"
        )
    if resolved_dataset_reference_report is None:
        sibling_report = resolved_feature_file.parent / DATASET_REFERENCE_REPORT_NAME
        if sibling_report.exists():
            resolved_dataset_reference_report = sibling_report.resolve()

    features = np.asarray(np.load(resolved_feature_file), dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix at {resolved_feature_file}, got shape={features.shape}")

    with open(resolved_model_names_file, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(features.shape[0]):
        raise ValueError(
            f"Model names length ({len(model_names)}) does not match feature rows ({features.shape[0]})"
        )

    metadata: dict[str, Any] = {}
    if resolved_metadata_file is not None:
        loaded_metadata = load_spectral_metadata(resolved_metadata_file)
        if isinstance(loaded_metadata, dict):
            metadata = dict(loaded_metadata)

    dataset_reference_payload: dict[str, Any] | None = None
    if resolved_dataset_reference_report is not None:
        dataset_reference_payload = load_dataset_reference_report(resolved_dataset_reference_report)
    else:
        try:
            dataset_reference_payload = resolve_dataset_reference_payload_for_artifact(resolved_feature_file)
        except Exception:
            dataset_reference_payload = None

    labels = _resolved_labels(
        labels_file=resolved_labels_file,
        dataset_reference_payload=dataset_reference_payload,
        model_names=model_names,
    )

    feature_names = _resolved_feature_names(metadata, int(features.shape[1]))
    if len(feature_names) != int(features.shape[1]):
        raise ValueError(
            f"Feature names length ({len(feature_names)}) does not match feature dimension ({features.shape[1]})"
        )

    sample_dataset_names: list[str | None] = []
    sample_ranks: list[int | None] = []
    model_index = dataset_reference_payload.get("model_index") if isinstance(dataset_reference_payload, Mapping) else None
    for model_name in model_names:
        dataset_name: str | None = None
        subset_name: str | None = None
        if isinstance(model_index, Mapping):
            raw_entry = model_index.get(model_name)
            if isinstance(raw_entry, Mapping):
                raw_dataset_name = raw_entry.get("dataset_name")
                if raw_dataset_name:
                    dataset_name = str(raw_dataset_name)
                raw_subset_name = raw_entry.get("subset_name")
                if raw_subset_name:
                    subset_name = str(raw_subset_name)
        sample_dataset_names.append(dataset_name)
        sample_ranks.append(_extract_rank(dataset_name, subset_name, model_name))

    return LoadedFeatureBundle(
        feature_file=resolved_feature_file,
        model_names_file=resolved_model_names_file,
        labels_file=resolved_labels_file,
        metadata_file=resolved_metadata_file,
        dataset_reference_report=resolved_dataset_reference_report,
        features=features,
        model_names=model_names,
        labels=labels,
        metadata=metadata,
        feature_names=feature_names,
        dataset_reference_payload=dataset_reference_payload,
        sample_dataset_names=sample_dataset_names,
        sample_ranks=sample_ranks,
    )


def _group_row_indices(bundle: LoadedFeatureBundle, *, over: str) -> list[dict[str, Any]]:
    if over not in SUPPORTED_GROUPINGS:
        raise ValueError(f"Unsupported grouping '{over}'. Supported: {list(SUPPORTED_GROUPINGS)}")

    if over == "rank":
        groups: dict[int | None, list[int]] = {}
        for i, rank in enumerate(bundle.sample_ranks):
            groups.setdefault(rank, []).append(i)

        known_ranks = sorted(rank for rank in groups if rank is not None)
        if not known_ranks:
            raise ValueError(
                "Could not resolve any sample ranks from the feature bundle. "
                "Provide a dataset reference report or model names that encode rank."
            )
        ordered_keys: list[int | None] = known_ranks + ([None] if None in groups else [])

        out: list[dict[str, Any]] = []
        for key in ordered_keys:
            if key is None:
                label = "rank=unknown"
                slug = "rank_unknown"
            else:
                label = f"rank={int(key)}"
                slug = f"rank_{int(key)}"
            out.append(
                {
                    "value": key,
                    "label": label,
                    "slug": slug,
                    "row_indices": np.asarray(groups[key], dtype=np.int64),
                }
            )
        return out

    raise AssertionError(f"Unhandled grouping: {over}")


def _label_count_summary(labels: np.ndarray | None) -> dict[str, int]:
    if labels is None or labels.size == 0:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique.tolist(), counts.tolist())}


def _sanitize_matrix(values: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    replaced = int(array.size - int(np.sum(finite)))
    if replaced > 0:
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array, replaced


def _iter_label_styles(labels: np.ndarray | None) -> list[tuple[int, str, str]]:
    if labels is None:
        return []
    present = sorted({int(x) for x in labels.tolist()})
    rows: list[tuple[int, str, str]] = []
    for value in present:
        name, color = _LABEL_STYLE.get(value, (f"label={value}", "#9467bd"))
        rows.append((value, name, color))
    return rows


def _save_tsne_embedding_csv(
    *,
    output_path: Path,
    group_label: str,
    model_names: list[str],
    labels: np.ndarray | None,
    ranks: list[int | None],
    dataset_names: list[str | None],
    embedding: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "group",
                "model_name",
                "label",
                "rank",
                "dataset_name",
                "tsne_1",
                "tsne_2",
            ],
        )
        writer.writeheader()
        for i, model_name in enumerate(model_names):
            writer.writerow(
                {
                    "index": int(i),
                    "group": group_label,
                    "model_name": model_name,
                    "label": None if labels is None else int(labels[i]),
                    "rank": None if ranks[i] is None else int(ranks[i]),
                    "dataset_name": dataset_names[i],
                    "tsne_1": float(embedding[i, 0]),
                    "tsne_2": float(embedding[i, 1]),
                }
            )


def _plot_tsne_embedding(
    *,
    output_path: Path,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    point_size: float,
    alpha: float,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    label_styles = _iter_label_styles(labels)
    if not label_styles:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=point_size,
            alpha=alpha,
            color="#1f77b4",
        )
    else:
        for label_value, label_name, color in label_styles:
            mask = labels == label_value
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=point_size,
                alpha=alpha,
                color=color,
                label=label_name,
            )
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _normalize_tsne_view(view: str | None) -> str:
    resolved = "full" if view is None else str(view).strip().lower()
    if resolved not in SUPPORTED_TSNE_VIEWS:
        raise ValueError(
            f"Unsupported t-SNE view '{view}'. Supported: {list(SUPPORTED_TSNE_VIEWS)}"
        )
    return resolved


def _tsne_combo_slug(
    *,
    perplexity: float,
    learning_rate: str | float,
    max_iter: int,
    metric: str,
    init: str,
    random_state: int,
    standardize: bool,
) -> str:
    parts = [
        ("perplexity", perplexity),
        ("learning_rate", learning_rate),
        ("max_iter", max_iter),
        ("metric", metric),
        ("init", init),
        ("random_state", random_state),
        ("standardize", standardize),
    ]
    return "__".join(
        f"{key}_{_safe_slug(_format_param_value(value))}"
        for key, value in parts
    )


def _normalize_tsne_grid_values(
    values: list[Any],
    *,
    name: str,
) -> list[Any]:
    if not values:
        raise ValueError(f"{name} grid must include at least one value")
    deduped: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _layer_column_groups(feature_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for col_idx, feature_name in enumerate(feature_names):
        layer = _extract_layer(feature_name)
        if layer is None:
            continue
        if layer not in grouped:
            grouped[layer] = {
                "layer": int(layer),
                "column_indices": [],
                "feature_names": [],
            }
            order.append(int(layer))
        grouped[layer]["column_indices"].append(int(col_idx))
        grouped[layer]["feature_names"].append(str(feature_name))

    return [grouped[layer] for layer in sorted(order)]


def _fit_tsne_embedding(
    *,
    values: np.ndarray,
    perplexity: float,
    learning_rate: str | float,
    max_iter: int,
    metric: str,
    init: str,
    random_state: int,
    standardize: bool,
) -> tuple[np.ndarray, int, float, str]:
    sanitized_values, replaced = _sanitize_matrix(values)
    if standardize:
        sanitized_values = StandardScaler().fit_transform(sanitized_values)

    resolved_perplexity = min(float(perplexity), float(max(1, sanitized_values.shape[0] - 1)))
    resolved_init = init
    if resolved_init == "pca" and min(sanitized_values.shape[0], sanitized_values.shape[1]) < 2:
        resolved_init = "random"

    tsne = TSNE(
        n_components=2,
        perplexity=resolved_perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=resolved_init,
        random_state=random_state,
    )
    embedding = np.asarray(tsne.fit_transform(sanitized_values), dtype=np.float32)
    return embedding, replaced, float(resolved_perplexity), str(resolved_init)


def _scatter_tsne_points(
    *,
    ax: Any,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    point_size: float,
    alpha: float,
) -> None:
    label_styles = _iter_label_styles(labels)
    if not label_styles:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=point_size,
            alpha=alpha,
            color="#1f77b4",
        )
        return

    for label_value, _label_name, color in label_styles:
        mask = labels == label_value
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=alpha,
            color=color,
        )


def _subplot_grid(n_items: int) -> tuple[int, int]:
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    n_cols = int(min(6, max(1, np.ceil(np.sqrt(n_items)))))
    n_rows = int(np.ceil(n_items / max(1, n_cols)))
    return n_rows, n_cols


def _plot_per_layer_tsne_embeddings(
    *,
    output_path: Path,
    layer_reports: list[dict[str, Any]],
    labels: np.ndarray | None,
    title: str,
    point_size: float,
    alpha: float,
) -> None:
    if not layer_reports:
        raise ValueError("layer_reports must include at least one layer embedding")

    n_rows, n_cols = _subplot_grid(len(layer_reports))
    fig_width = max(18.0, n_cols * 4.2)
    fig_height = max(12.0, n_rows * 4.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for ax, layer_report in zip(axes_flat, layer_reports):
        embedding = np.asarray(layer_report["embedding"], dtype=np.float32)
        _scatter_tsne_points(
            ax=ax,
            embedding=embedding,
            labels=labels,
            point_size=point_size,
            alpha=alpha,
        )
        ax.set_title(f"layer {int(layer_report['layer'])}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.2)

    for ax in axes_flat[len(layer_reports) :]:
        ax.axis("off")

    label_styles = _iter_label_styles(labels)
    if label_styles:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor=color,
                label=label_name,
            )
            for _label_value, label_name, color in label_styles
        ]
        fig.legend(handles=handles, loc="upper right")

    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _normalize_block_filters(block_filters: list[str] | tuple[str, ...] | None) -> list[str]:
    if not block_filters:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_filter in block_filters:
        text = str(raw_filter).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _block_matches_filters(
    *,
    block_name: str,
    raw_block_name: str | None,
    block_filters: list[str],
) -> bool:
    if not block_filters:
        return True
    haystacks = [str(block_name).lower()]
    if raw_block_name:
        haystacks.append(str(raw_block_name).lower())
    return any(
        str(block_filter).lower() in haystack
        for block_filter in block_filters
        for haystack in haystacks
    )


def _normalize_dataset_folders(folders: Any) -> list[str]:
    def iter_values(value: Any) -> list[Any]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            flattened: list[Any] = []
            for item in value:
                flattened.extend(iter_values(item))
            return flattened
        return [value]

    raw_values = iter_values(folders)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        text = str(raw_value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    if not normalized:
        raise ValueError("--folder/--dataset-name must include at least one non-empty value")
    return normalized


def _entry_matching_dataset_folders(entry: Mapping[str, Any], folders: list[str]) -> list[str]:
    candidates = [
        entry.get("dataset_name"),
        entry.get("subset_name"),
    ]
    dataset_path = entry.get("dataset_path")
    if dataset_path:
        candidates.append(Path(str(dataset_path)).name)
    candidate_set = {str(candidate) for candidate in candidates if candidate}
    return [folder for folder in folders if folder in candidate_set]


def _folder_selection_slug(folders: list[str]) -> str:
    if len(folders) == 1:
        return _safe_slug(folders[0])
    digest = hashlib.sha256("\n".join(folders).encode("utf-8")).hexdigest()[:10]
    return f"multi_{len(folders)}_folders_{digest}"


def _folder_selection_label(folders: list[str]) -> str:
    if len(folders) == 1:
        return folders[0]
    if len(folders) <= 3:
        return ", ".join(folders)
    return ", ".join(folders[:3]) + f", +{len(folders) - 3} more"


def _select_dataset_row_indices(
    *,
    bundle: LoadedFeatureBundle,
    folders: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    payload = bundle.dataset_reference_payload
    model_index = payload.get("model_index") if isinstance(payload, Mapping) else None
    if not isinstance(model_index, Mapping) or not model_index:
        raise ValueError("layer raw-feature t-SNE requires dataset-reference state with model_index")

    selected: list[int] = []
    matched_counts = {folder: 0 for folder in folders}
    for row_idx, model_name in enumerate(bundle.model_names):
        raw_entry = model_index.get(model_name)
        if not isinstance(raw_entry, Mapping):
            continue
        matching_folders = _entry_matching_dataset_folders(raw_entry, folders)
        if matching_folders:
            selected.append(int(row_idx))
            for folder in matching_folders:
                matched_counts[folder] += 1

    missing_folders = [folder for folder, count in matched_counts.items() if count <= 0]
    if not selected:
        available = sorted(
            {
                str(entry.get("dataset_name"))
                for entry in model_index.values()
                if isinstance(entry, Mapping) and entry.get("dataset_name")
            }
        )
        preview = ", ".join(available[:8])
        raise ValueError(
            f"No rows matched --folder/--dataset-name values {folders}. "
            + (f"Available datasets include: {preview}" if preview else "No dataset names were available.")
        )
    if missing_folders:
        raise ValueError(
            "No rows matched requested folder(s): "
            + ", ".join(missing_folders)
        )
    if len(selected) < 2:
        raise ValueError(
            "layer raw-feature t-SNE requires at least two selected rows, "
            f"found {len(selected)} for folders={folders}"
        )
    return np.asarray(selected, dtype=np.int64), matched_counts


def _select_layer_feature_columns(
    *,
    bundle: LoadedFeatureBundle,
    layer: int,
    block_filters: list[str],
    features: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray, list[str], list[str], list[str] | None]:
    requested_features = _normalize_requested_features(features)
    requested_feature_set = set(requested_features or [])
    column_indices: list[int] = []
    feature_names: list[str] = []
    block_names: list[str] = []
    seen_blocks: set[str] = set()

    for column_idx, feature_name in enumerate(bundle.feature_names):
        try:
            block_name = _feature_block_name(feature_name)
        except ValueError:
            continue
        resolved_layer = _extract_layer(block_name)
        if resolved_layer != int(layer):
            continue
        if not _block_matches_filters(
            block_name=block_name,
            raw_block_name=None,
            block_filters=block_filters,
        ):
            continue
        if requested_features is not None:
            feature_group = _feature_group_for_feature_name(feature_name)
            if feature_group not in requested_feature_set:
                continue
        column_indices.append(int(column_idx))
        feature_names.append(str(feature_name))
        if block_name not in seen_blocks:
            seen_blocks.add(block_name)
            block_names.append(block_name)

    if not column_indices:
        feature_detail = (
            "all feature groups"
            if requested_features is None
            else ", ".join(str(x) for x in requested_features)
        )
        filter_detail = ", ".join(block_filters) if block_filters else "no block filter"
        raise ValueError(
            "No extracted feature columns matched "
            f"layer={layer}, {filter_detail}, features={feature_detail}"
        )
    return (
        np.asarray(column_indices, dtype=np.int64),
        block_names,
        feature_names,
        requested_features,
    )


def _adapter_path_candidates(model_dir: Path) -> tuple[Path, ...]:
    return (
        model_dir / "adapter_model.safetensors",
        model_dir / "best_model" / "adapter_model.safetensors",
    )


def _resolve_adapter_path_for_selected_model(
    *,
    model_name: str,
    entry: Mapping[str, Any],
    folders: list[str],
    dataset_root: Path | None,
) -> Path:
    candidate_model_dirs: list[Path] = []
    dataset_path = entry.get("dataset_path")
    if dataset_path:
        candidate_model_dirs.append(Path(str(dataset_path)).expanduser() / model_name)

    fallback_dataset_names = list(folders)
    for key in ("dataset_name", "subset_name"):
        raw_value = entry.get(key)
        if raw_value and str(raw_value) not in fallback_dataset_names:
            fallback_dataset_names.append(str(raw_value))
    if dataset_path:
        path_name = Path(str(dataset_path)).name
        if path_name and path_name not in fallback_dataset_names:
            fallback_dataset_names.append(path_name)

    if dataset_root is not None:
        resolved_dataset_root = _resolve_path(dataset_root)
        for dataset_name in fallback_dataset_names:
            candidate_model_dirs.append(resolved_dataset_root / dataset_name / model_name)

    tried: list[Path] = []
    seen: set[str] = set()
    for model_dir in candidate_model_dirs:
        resolved_model_dir = model_dir if model_dir.is_absolute() else (Path.cwd().resolve() / model_dir)
        for candidate in _adapter_path_candidates(resolved_model_dir):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            tried.append(candidate)
            if candidate.exists():
                return candidate.resolve()

    preview = ", ".join(str(path) for path in tried[:6])
    raise FileNotFoundError(
        f"Adapter file not found for selected model '{model_name}'. "
        + (f"Tried: {preview}" if preview else "No candidate paths could be built.")
    )


def _resolve_selected_adapter_paths(
    *,
    bundle: LoadedFeatureBundle,
    row_indices: np.ndarray,
    folders: list[str],
    dataset_root: Path | None,
) -> list[Path]:
    payload = bundle.dataset_reference_payload
    model_index = payload.get("model_index") if isinstance(payload, Mapping) else None
    if not isinstance(model_index, Mapping):
        raise ValueError("layer raw-feature t-SNE requires dataset-reference state with model_index")

    adapter_paths: list[Path] = []
    for row_idx in row_indices.tolist():
        model_name = bundle.model_names[int(row_idx)]
        entry = model_index.get(model_name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Dataset-reference entry missing for selected model '{model_name}'")
        adapter_paths.append(
            _resolve_adapter_path_for_selected_model(
                model_name=model_name,
                entry=entry,
                folders=folders,
                dataset_root=dataset_root,
            )
        )
    return adapter_paths


def _raw_delta_block_specs_for_feature_blocks(
    *,
    schema: Any,
    feature_block_names: list[str],
) -> tuple[list[RawDeltaBlockSpec], list[str]]:
    base_specs = {
        shorten_block_name(str(raw_block_name)): RawDeltaBlockSpec(
            kind="base",
            block_name=shorten_block_name(str(raw_block_name)),
            raw_block_name=str(raw_block_name),
            pair_index=int(pair_idx),
        )
        for pair_idx, raw_block_name in enumerate(schema.block_names)
    }

    needs_qv_sum = any(".qv_sum" in str(block_name) or str(block_name).endswith("qv_sum") for block_name in feature_block_names)
    qv_specs_by_block: dict[str, RawDeltaBlockSpec] = {}
    if needs_qv_sum:
        for qv_spec in build_qv_sum_specs(schema):
            block_name = shorten_block_name(qv_spec.qv_block_name_raw)
            qv_specs_by_block[block_name] = RawDeltaBlockSpec(
                kind="qv_sum",
                block_name=block_name,
                raw_block_name=str(qv_spec.qv_block_name_raw),
                q_index=int(qv_spec.q_index),
                v_index=int(qv_spec.v_index),
            )

    specs: list[RawDeltaBlockSpec] = []
    missing: list[str] = []
    for block_name in feature_block_names:
        if block_name in base_specs:
            specs.append(base_specs[block_name])
        elif block_name in qv_specs_by_block:
            specs.append(qv_specs_by_block[block_name])
        else:
            missing.append(block_name)

    if not specs:
        preview = ", ".join(feature_block_names[:8])
        raise ValueError(f"No raw B@A blocks matched selected feature blocks. Examples: {preview}")
    return specs, missing


def _projection_seed(*, seed: int, block_name: str, shape: tuple[int, ...], side: str) -> int:
    payload = f"{seed}|{block_name}|{side}|{'x'.join(str(x) for x in shape)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _signed_projection_matrix(
    *,
    shape: tuple[int, int],
    seed: int,
    scale_denominator: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    signs = rng.integers(0, 2, size=shape, dtype=np.int8)
    projection = np.where(signs == 0, -1.0, 1.0).astype(np.float32)
    projection /= float(np.sqrt(max(1, int(scale_denominator))))
    return projection


def _raw_delta_sketch_shape(raw_projection_dim: int) -> tuple[int, int]:
    if raw_projection_dim <= 0:
        raise ValueError(f"raw_projection_dim must be positive, got {raw_projection_dim}")
    out_dim = max(1, int(np.floor(np.sqrt(int(raw_projection_dim)))))
    in_dim = max(1, int(np.ceil(float(raw_projection_dim) / float(out_dim))))
    return out_dim, in_dim


def _raw_delta_projection_matrices(
    *,
    block_name: str,
    out_features: int,
    in_features: int,
    raw_projection_dim: int,
    projection_seed: int,
    projection_cache: dict[tuple[str, int, int, int, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    out_sketch_dim, in_sketch_dim = _raw_delta_sketch_shape(raw_projection_dim)
    key = (str(block_name), int(out_features), int(in_features), int(out_sketch_dim), int(in_sketch_dim))
    cached = projection_cache.get(key)
    if cached is not None:
        return cached

    output_projection = _signed_projection_matrix(
        shape=(out_sketch_dim, int(out_features)),
        seed=_projection_seed(
            seed=projection_seed,
            block_name=block_name,
            shape=(out_sketch_dim, int(out_features)),
            side="out",
        ),
        scale_denominator=out_sketch_dim,
    )
    input_projection = _signed_projection_matrix(
        shape=(int(in_features), in_sketch_dim),
        seed=_projection_seed(
            seed=projection_seed,
            block_name=block_name,
            shape=(int(in_features), in_sketch_dim),
            side="in",
        ),
        scale_denominator=in_sketch_dim,
    )
    projection_cache[key] = (output_projection, input_projection)
    return output_projection, input_projection


def _sketch_lora_delta(
    *,
    a: np.ndarray,
    b: np.ndarray,
    block_name: str,
    raw_projection_dim: int,
    projection_seed: int,
    projection_cache: dict[tuple[str, int, int, int, int], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    a_array = np.asarray(a, dtype=np.float32)
    b_array = np.asarray(b, dtype=np.float32)
    if a_array.ndim != 2 or b_array.ndim != 2:
        raise ValueError(f"Expected rank-2 LoRA factors for {block_name}, got A{a_array.shape}, B{b_array.shape}")
    if int(a_array.shape[0]) != int(b_array.shape[1]):
        raise ValueError(f"LoRA rank mismatch for {block_name}: A{a_array.shape}, B{b_array.shape}")

    output_projection, input_projection = _raw_delta_projection_matrices(
        block_name=block_name,
        out_features=int(b_array.shape[0]),
        in_features=int(a_array.shape[1]),
        raw_projection_dim=raw_projection_dim,
        projection_seed=projection_seed,
        projection_cache=projection_cache,
    )
    sketch = (output_projection @ b_array) @ (a_array @ input_projection)
    return np.asarray(sketch, dtype=np.float32).reshape(-1)[: int(raw_projection_dim)]


def _build_raw_delta_sketch_matrix(
    *,
    adapter_paths: list[Path],
    raw_block_specs: list[RawDeltaBlockSpec],
    raw_projection_dim: int,
    projection_seed: int,
    dtype: np.dtype,
) -> np.ndarray:
    if not adapter_paths:
        raise ValueError("adapter_paths must be non-empty")
    if not raw_block_specs:
        raise ValueError("raw_block_specs must be non-empty")

    schema = load_delta_block_schema(adapter_paths[0])
    expected_pairs = [tuple(x) for x in schema.pairs]
    expected_a_shapes = [tuple(x) for x in schema.a_shapes]
    expected_b_shapes = [tuple(x) for x in schema.b_shapes]
    expected_e_keys = [str(x) if x is not None else None for x in schema.e_keys]
    expected_e_shapes = [
        tuple(int(y) for y in x) if x is not None else None
        for x in schema.e_shapes
    ]
    allow_rank_variation = bool(schema_has_adalora_scaling(schema))
    needed_pair_indices = {
        int(index)
        for spec in raw_block_specs
        for index in (spec.pair_index, spec.q_index, spec.v_index)
        if index is not None
    }

    rows: list[np.ndarray] = []
    projection_cache: dict[tuple[str, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for adapter_path in adapter_paths:
        factor_by_pair_index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with safe_open(adapter_path, framework="numpy") as reader:
            check_consistency_reader(
                reader=reader,
                adapter_path=adapter_path,
                expected_pairs=expected_pairs,
                expected_a_shapes=expected_a_shapes,
                expected_b_shapes=expected_b_shapes,
                expected_e_keys=expected_e_keys,
                expected_e_shapes=expected_e_shapes,
                allow_rank_variation=allow_rank_variation,
            )
            for pair_idx, (_raw_block_name, a, b) in enumerate(
                iter_block_factors(reader=reader, schema=schema, dtype=dtype)
            ):
                if pair_idx in needed_pair_indices:
                    factor_by_pair_index[int(pair_idx)] = (a, b)

        pieces: list[np.ndarray] = []
        for spec in raw_block_specs:
            if spec.kind == "base":
                if spec.pair_index is None or spec.pair_index not in factor_by_pair_index:
                    raise RuntimeError(f"Missing raw factors for block {spec.block_name} in {adapter_path}")
                a, b = factor_by_pair_index[int(spec.pair_index)]
            elif spec.kind == "qv_sum":
                if spec.q_index is None or spec.v_index is None:
                    raise RuntimeError(f"Invalid qv_sum raw block spec for {spec.block_name}")
                if spec.q_index not in factor_by_pair_index or spec.v_index not in factor_by_pair_index:
                    raise RuntimeError(f"Missing q/v factors for block {spec.block_name} in {adapter_path}")
                a_q, b_q = factor_by_pair_index[int(spec.q_index)]
                a_v, b_v = factor_by_pair_index[int(spec.v_index)]
                a = np.concatenate([a_q, a_v], axis=0)
                b = np.concatenate([b_q, b_v], axis=1)
            else:
                raise RuntimeError(f"Unsupported raw block spec kind '{spec.kind}'")

            pieces.append(
                _sketch_lora_delta(
                    a=a,
                    b=b,
                    block_name=spec.block_name,
                    raw_projection_dim=raw_projection_dim,
                    projection_seed=projection_seed,
                    projection_cache=projection_cache,
                )
            )
        rows.append(np.concatenate(pieces).astype(np.float32, copy=False))

    return np.vstack(rows).astype(np.float32, copy=False)


def _plot_layer_raw_feature_tsne_comparison(
    *,
    output_path: Path,
    raw_embedding: np.ndarray,
    feature_embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    point_size: float,
    alpha: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        (axes[0], raw_embedding, "Raw B@A sketch"),
        (axes[1], feature_embedding, "Extracted features"),
    ]
    for ax, embedding, panel_title in panels:
        _scatter_tsne_points(
            ax=ax,
            embedding=embedding,
            labels=labels,
            point_size=point_size,
            alpha=alpha,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, linestyle="--", alpha=0.3)

    label_styles = _iter_label_styles(labels)
    if label_styles:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor=color,
                label=label_name,
            )
            for _label_value, label_name, color in label_styles
        ]
        fig.legend(handles=handles, loc="upper right")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_grouped_tsne_analysis_from_bundle(
    *,
    bundle: LoadedFeatureBundle,
    output_dir: Path,
    over: str,
    view: str,
    perplexity: float,
    learning_rate: str | float,
    max_iter: int,
    metric: str,
    init: str,
    random_state: int,
    standardize: bool,
    point_size: float,
    alpha: float,
    embedding_csv_dir: Path | None,
) -> dict[str, Any]:
    if perplexity <= 0:
        raise ValueError(f"perplexity must be positive, got {perplexity}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    resolved_view = _normalize_tsne_view(view)

    row_groups = _group_row_indices(bundle, over=over)
    layer_groups = _layer_column_groups(bundle.feature_names) if resolved_view == "per_layer" else []
    if resolved_view == "per_layer" and not layer_groups:
        raise ValueError(
            "Could not resolve layer-specific feature columns from the feature metadata. "
            "Per-layer t-SNE requires spectral feature names with layer identifiers."
        )

    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_embedding_csv_dir = (
        _resolve_path(embedding_csv_dir) if embedding_csv_dir is not None else None
    )
    if resolved_embedding_csv_dir is not None:
        resolved_embedding_csv_dir.mkdir(parents=True, exist_ok=True)

    combo_slug = _tsne_combo_slug(
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=init,
        random_state=random_state,
        standardize=standardize,
    )

    warnings: list[str] = []
    group_reports: list[dict[str, Any]] = []

    for group in row_groups:
        row_indices = np.asarray(group["row_indices"], dtype=np.int64)
        n_rows = int(row_indices.size)
        group_report: dict[str, Any] = {
            "group": group["label"],
            "group_value": group["value"],
            "n_samples": n_rows,
        }

        if n_rows < 2:
            warning = f"Skipping {group['label']} because it has fewer than two samples"
            warnings.append(warning)
            group_report["warning"] = warning
            group_reports.append(group_report)
            continue

        group_labels = None if bundle.labels is None else bundle.labels[row_indices]
        group_report["label_counts"] = _label_count_summary(group_labels)

        if resolved_view == "full":
            embedding, replaced, resolved_perplexity, resolved_init = _fit_tsne_embedding(
                values=bundle.features[row_indices],
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=max_iter,
                metric=metric,
                init=init,
                random_state=random_state,
                standardize=standardize,
            )
            if replaced > 0:
                warnings.append(
                    f"Replaced {replaced} non-finite values with zero before t-SNE for {group['label']}"
                )

            plot_path = resolved_output_dir / f"tsne_{group['slug']}.png"
            _plot_tsne_embedding(
                output_path=plot_path,
                embedding=embedding,
                labels=group_labels,
                title=f"t-SNE ({group['label']})",
                point_size=point_size,
                alpha=alpha,
            )

            group_report["plot"] = str(plot_path)
            group_report["perplexity"] = float(resolved_perplexity)
            group_report["resolved_init"] = resolved_init

            if resolved_embedding_csv_dir is not None:
                csv_path = resolved_embedding_csv_dir / f"tsne_{group['slug']}.csv"
                _save_tsne_embedding_csv(
                    output_path=csv_path,
                    group_label=str(group["label"]),
                    model_names=[bundle.model_names[int(i)] for i in row_indices.tolist()],
                    labels=group_labels,
                    ranks=[bundle.sample_ranks[int(i)] for i in row_indices.tolist()],
                    dataset_names=[bundle.sample_dataset_names[int(i)] for i in row_indices.tolist()],
                    embedding=embedding,
                )
                group_report["embedding_csv"] = str(csv_path)
        else:
            layer_reports: list[dict[str, Any]] = []
            layer_embedding_dir = (
                resolved_embedding_csv_dir / f"tsne_{group['slug']}_layers"
                if resolved_embedding_csv_dir is not None
                else None
            )
            for layer_group in layer_groups:
                layer = int(layer_group["layer"])
                layer_column_indices = np.asarray(layer_group["column_indices"], dtype=np.int64)
                layer_embedding, replaced, resolved_perplexity, resolved_init = _fit_tsne_embedding(
                    values=bundle.features[row_indices][:, layer_column_indices],
                    perplexity=perplexity,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    metric=metric,
                    init=init,
                    random_state=random_state,
                    standardize=standardize,
                )
                if replaced > 0:
                    warnings.append(
                        "Replaced "
                        f"{replaced} non-finite values with zero before t-SNE for {group['label']} "
                        f"layer={layer}"
                    )

                layer_report: dict[str, Any] = {
                    "layer": layer,
                    "n_columns": int(layer_column_indices.size),
                    "perplexity": float(resolved_perplexity),
                    "resolved_init": resolved_init,
                    "embedding": layer_embedding,
                }
                if layer_embedding_dir is not None:
                    csv_path = layer_embedding_dir / f"layer_{layer:03d}.csv"
                    _save_tsne_embedding_csv(
                        output_path=csv_path,
                        group_label=f"{group['label']} layer={layer}",
                        model_names=[bundle.model_names[int(i)] for i in row_indices.tolist()],
                        labels=group_labels,
                        ranks=[bundle.sample_ranks[int(i)] for i in row_indices.tolist()],
                        dataset_names=[bundle.sample_dataset_names[int(i)] for i in row_indices.tolist()],
                        embedding=layer_embedding,
                    )
                    layer_report["embedding_csv"] = str(csv_path)
                layer_reports.append(layer_report)

            plot_path = resolved_output_dir / f"tsne_{group['slug']}_per_layer.png"
            _plot_per_layer_tsne_embeddings(
                output_path=plot_path,
                layer_reports=layer_reports,
                labels=group_labels,
                title=f"Per-layer t-SNE ({group['label']})",
                point_size=point_size,
                alpha=alpha,
            )
            group_report["plot"] = str(plot_path)
            group_report["layer_count"] = int(len(layer_reports))
            group_report["layers"] = [
                {
                    key: value
                    for key, value in layer_report.items()
                    if key != "embedding"
                }
                for layer_report in layer_reports
            ]

        group_reports.append(group_report)

    return {
        "analysis": "tsne",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_file": str(bundle.feature_file),
        "model_names_file": str(bundle.model_names_file),
        "labels_file": str(bundle.labels_file) if bundle.labels_file is not None else None,
        "metadata_file": str(bundle.metadata_file) if bundle.metadata_file is not None else None,
        "dataset_reference_report": (
            str(bundle.dataset_reference_report) if bundle.dataset_reference_report is not None else None
        ),
        "feature_shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        "over": over,
        "view": resolved_view,
        "combo_slug": combo_slug,
        "plot_dir": str(resolved_output_dir),
        "embedding_csv_dir": (
            str(resolved_embedding_csv_dir) if resolved_embedding_csv_dir is not None else None
        ),
        "standardize": bool(standardize),
        "metric": metric,
        "init": init,
        "random_state": int(random_state),
        "max_iter": int(max_iter),
        "learning_rate": learning_rate,
        "groups": group_reports,
        "warnings": warnings,
    }


def run_grouped_tsne_analysis(
    *,
    feature_file: Path,
    output_dir: Path,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    over: str = "rank",
    view: str = "full",
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
    init: str = "pca",
    random_state: int = 42,
    standardize: bool = True,
    point_size: float = 28.0,
    alpha: float = 0.85,
    embedding_csv_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = load_feature_bundle(
        feature_file=feature_file,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
    )
    return _run_grouped_tsne_analysis_from_bundle(
        bundle=bundle,
        output_dir=output_dir,
        over=over,
        view=view,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=init,
        random_state=random_state,
        standardize=standardize,
        point_size=point_size,
        alpha=alpha,
        embedding_csv_dir=embedding_csv_dir,
    )


def run_grouped_tsne_sweep_analysis(
    *,
    feature_file: Path,
    output_dir: Path,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    over: str = "rank",
    view: str = "full",
    perplexities: list[float] | None = None,
    learning_rates: list[str | float] | None = None,
    max_iters: list[int] | None = None,
    metrics: list[str] | None = None,
    inits: list[str] | None = None,
    random_states: list[int] | None = None,
    standardize: bool = True,
    point_size: float = 28.0,
    alpha: float = 0.85,
    embedding_csv_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_view = _normalize_tsne_view(view)
    bundle = load_feature_bundle(
        feature_file=feature_file,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
    )

    resolved_perplexities = _normalize_tsne_grid_values(
        [30.0] if perplexities is None else [float(x) for x in perplexities],
        name="perplexities",
    )
    resolved_learning_rates = _normalize_tsne_grid_values(
        ["auto"] if learning_rates is None else list(learning_rates),
        name="learning_rates",
    )
    resolved_max_iters = _normalize_tsne_grid_values(
        [1000] if max_iters is None else [int(x) for x in max_iters],
        name="max_iters",
    )
    resolved_metrics = _normalize_tsne_grid_values(
        ["euclidean"] if metrics is None else [str(x) for x in metrics],
        name="metrics",
    )
    resolved_inits = _normalize_tsne_grid_values(
        ["pca"] if inits is None else [str(x) for x in inits],
        name="inits",
    )
    resolved_random_states = _normalize_tsne_grid_values(
        [42] if random_states is None else [int(x) for x in random_states],
        name="random_states",
    )

    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_embedding_csv_dir = (
        _resolve_path(embedding_csv_dir) if embedding_csv_dir is not None else None
    )
    if resolved_embedding_csv_dir is not None:
        resolved_embedding_csv_dir.mkdir(parents=True, exist_ok=True)

    combo_reports: list[dict[str, Any]] = []
    combo_summaries: list[dict[str, Any]] = []
    for perplexity, learning_rate, max_iter, metric, init, random_state in product(
        resolved_perplexities,
        resolved_learning_rates,
        resolved_max_iters,
        resolved_metrics,
        resolved_inits,
        resolved_random_states,
    ):
        combo_slug = _tsne_combo_slug(
            perplexity=float(perplexity),
            learning_rate=learning_rate,
            max_iter=int(max_iter),
            metric=str(metric),
            init=str(init),
            random_state=int(random_state),
            standardize=standardize,
        )
        combo_plot_dir = resolved_output_dir / combo_slug
        combo_embedding_dir = (
            resolved_embedding_csv_dir / combo_slug if resolved_embedding_csv_dir is not None else None
        )
        combo_report = _run_grouped_tsne_analysis_from_bundle(
            bundle=bundle,
            output_dir=combo_plot_dir,
            over=over,
            view=resolved_view,
            perplexity=float(perplexity),
            learning_rate=learning_rate,
            max_iter=int(max_iter),
            metric=str(metric),
            init=str(init),
            random_state=int(random_state),
            standardize=standardize,
            point_size=point_size,
            alpha=alpha,
            embedding_csv_dir=combo_embedding_dir,
        )
        combo_report_path: Path | None = None
        if combo_embedding_dir is not None:
            combo_embedding_dir.mkdir(parents=True, exist_ok=True)
            combo_report_path = combo_embedding_dir / f"tsne_analysis_report_{resolved_view}.json"
            with open(combo_report_path, "w", encoding="utf-8") as f:
                json.dump(json_ready(combo_report), f, indent=2)
            combo_report["report"] = str(combo_report_path)
        combo_reports.append(combo_report)
        combo_summaries.append(
            {
                "combo_slug": combo_slug,
                "parameters": {
                    "perplexity": float(perplexity),
                    "learning_rate": learning_rate,
                    "max_iter": int(max_iter),
                    "metric": str(metric),
                    "init": str(init),
                    "random_state": int(random_state),
                    "standardize": bool(standardize),
                    "view": resolved_view,
                },
                "plot_dir": str(combo_plot_dir),
                "embedding_csv_dir": str(combo_embedding_dir) if combo_embedding_dir is not None else None,
                "report": str(combo_report_path) if combo_report_path is not None else None,
                "group_count": int(len(combo_report["groups"])),
                "warning_count": int(len(combo_report["warnings"])),
            }
        )

    return {
        "analysis": "tsne_sweep",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_file": str(bundle.feature_file),
        "model_names_file": str(bundle.model_names_file),
        "labels_file": str(bundle.labels_file) if bundle.labels_file is not None else None,
        "metadata_file": str(bundle.metadata_file) if bundle.metadata_file is not None else None,
        "dataset_reference_report": (
            str(bundle.dataset_reference_report) if bundle.dataset_reference_report is not None else None
        ),
        "feature_shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        "over": over,
        "view": resolved_view,
        "standardize": bool(standardize),
        "point_size": float(point_size),
        "alpha": float(alpha),
        "grid": {
            "perplexities": [float(x) for x in resolved_perplexities],
            "learning_rates": list(resolved_learning_rates),
            "max_iters": [int(x) for x in resolved_max_iters],
            "metrics": list(resolved_metrics),
            "inits": list(resolved_inits),
            "random_states": [int(x) for x in resolved_random_states],
        },
        "combo_count": int(len(combo_reports)),
        "combos": combo_summaries,
        "combo_reports": combo_reports,
    }


def _default_cnn_tsne_output_root(supervised_run_dir: Path) -> Path:
    resolved = supervised_run_dir.expanduser().resolve()
    for parent in resolved.parents:
        if parent.name == "supervised":
            return parent.parent
    return resolved.parent


def _default_cnn_tsne_run_id(supervised_run_dir: Path) -> str:
    resolved = supervised_run_dir.expanduser().resolve()
    relative_parts: tuple[str, ...] = (resolved.name,)
    for parent in resolved.parents:
        if parent.name == "supervised":
            try:
                relative_parts = resolved.relative_to(parent).parts
            except ValueError:
                relative_parts = (resolved.name,)
            break
    return _safe_slug("__".join(relative_parts))


def _read_json_object(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _resolve_supervised_winner_model_name(supervised_run_dir: Path) -> str | None:
    run_config_path = supervised_run_dir / "run_config.json"
    if run_config_path.exists():
        run_config = _read_json_object(run_config_path)
        winner = run_config.get("winner")
        if isinstance(winner, Mapping) and winner.get("model_name"):
            return str(winner["model_name"])
        if run_config.get("model_name"):
            return str(run_config["model_name"])

    report_path = supervised_run_dir / "reports" / "supervised_report.json"
    if report_path.exists():
        report = _read_json_object(report_path)
        winner = report.get("tuning", {}).get("winner") if isinstance(report.get("tuning"), Mapping) else None
        if isinstance(winner, Mapping) and winner.get("model_name"):
            return str(winner["model_name"])
    return None


def _resolve_supervised_best_cnn_path(supervised_run_dir: Path) -> Path:
    artifact_index_path = supervised_run_dir / "artifact_index.json"
    if artifact_index_path.exists():
        artifact_index = _read_json_object(artifact_index_path)
        raw_best_model = artifact_index.get("best_model")
        if isinstance(raw_best_model, str) and raw_best_model:
            candidate = Path(raw_best_model).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd().resolve() / candidate).resolve()
            if candidate.exists():
                return candidate

    candidate = supervised_run_dir / "models" / "best_model.pt"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"CNN checkpoint not found for supervised run: {candidate}")


def _unique_probability_field_names(class_names: tuple[str, ...] | list[str]) -> list[str]:
    fields: list[str] = []
    seen: dict[str, int] = {}
    for class_name in class_names:
        base = f"prob_{_safe_slug(str(class_name))}"
        count = int(seen.get(base, 0))
        seen[base] = count + 1
        fields.append(base if count == 0 else f"{base}_{count + 1}")
    return fields


def _category_label(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    return text if text else "unknown"


def _plot_categorical_tsne_embedding(
    *,
    output_path: Path,
    embedding: np.ndarray,
    categories: list[Any],
    title: str,
    point_size: float,
    alpha: float,
) -> None:
    if int(embedding.shape[0]) != len(categories):
        raise ValueError("t-SNE category count does not match embedding rows")
    labels = [_category_label(value) for value in categories]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(idx % cmap.N) for idx in range(max(1, len(unique_labels)))]
    color_by_label = {label: colors[idx] for idx, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for label in unique_labels:
        mask = np.asarray([value == label for value in labels], dtype=bool)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=alpha,
            color=color_by_label[label],
            label=label,
        )
    if unique_labels:
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _row_float(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _cnn_partition_label_group(row: Mapping[str, Any]) -> str:
    partition = _category_label(row.get("partition"))
    if _is_cnn_backdoor_embedding_row(dict(row)):
        return f"{partition} backdoor"
    binary_label = row.get("binary_label")
    if binary_label is not None:
        try:
            if int(binary_label) == 0:
                return f"{partition} clean"
        except (TypeError, ValueError):
            pass
    if row.get("task_label_name") == "clean":
        return f"{partition} clean"
    return f"{partition} unknown"


def _cnn_group_order(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["train clean", "train backdoor", "inference clean", "inference backdoor"]
    observed = {_cnn_partition_label_group(row) for row in rows}
    ordered = [label for label in preferred if label in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def _values_by_group(rows: list[dict[str, Any]], field: str) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {label: [] for label in _cnn_group_order(rows)}
    for row in rows:
        value = _row_float(row, field)
        if value is None:
            continue
        values.setdefault(_cnn_partition_label_group(row), []).append(value)
    return {label: np.asarray(group_values, dtype=np.float64) for label, group_values in values.items()}


def _finite_metric_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [_row_float(row, field) for row in rows]
    finite = [value for value in values if value is not None]
    return np.asarray(finite, dtype=np.float64)


def _histogram_bins(values: np.ndarray) -> np.ndarray | int:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 10
    if float(np.min(finite)) == float(np.max(finite)):
        center = float(finite[0])
        width = max(abs(center) * 0.05, 0.5)
        return np.linspace(center - width, center + width, 11)
    return np.histogram_bin_edges(finite, bins="auto")


def _plot_cnn_metric_histogram(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    xlabel: str,
) -> None:
    values_by_group = _values_by_group(rows, field)
    all_values = _finite_metric_values(rows, field)
    bins = _histogram_bins(all_values)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    plotted = False
    for idx, (group, values) in enumerate(values_by_group.items()):
        if values.size == 0:
            continue
        plotted = True
        ax.hist(
            values,
            bins=bins,
            alpha=0.45,
            label=f"{group} (n={values.size})",
            color=cmap(idx % cmap.N),
            edgecolor="black",
            linewidth=0.4,
        )
    if not plotted:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cnn_metric_ecdf(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    xlabel: str,
) -> None:
    values_by_group = _values_by_group(rows, field)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    plotted = False
    for idx, (group, values) in enumerate(values_by_group.items()):
        if values.size == 0:
            continue
        plotted = True
        ordered = np.sort(values)
        y_values = np.arange(1, int(ordered.size) + 1, dtype=np.float64) / float(ordered.size)
        ax.step(ordered, y_values, where="post", label=f"{group} (n={values.size})", color=cmap(idx % cmap.N))
    if not plotted:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cnn_metric_by_partition_label(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    ylabel: str,
    point_size: float,
    alpha: float,
) -> None:
    groups = _cnn_group_order(rows)
    values_by_group = _values_by_group(rows, field)
    cmap = plt.get_cmap("tab10")
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    plotted = False
    for idx, group in enumerate(groups):
        values = values_by_group.get(group, np.asarray([], dtype=np.float64))
        if values.size == 0:
            continue
        plotted = True
        jitter = rng.uniform(-0.12, 0.12, size=int(values.size))
        ax.scatter(
            np.full(int(values.size), idx, dtype=np.float64) + jitter,
            values,
            s=point_size,
            alpha=alpha,
            color=cmap(idx % cmap.N),
            label=f"{group} (n={values.size})",
        )
        ax.hlines(
            float(np.median(values)),
            idx - 0.22,
            idx + 0.22,
            color="black",
            linewidth=1.5,
        )
    if not plotted:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(groups), dtype=np.int64))
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cnn_head_projection_scatter(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    point_size: float,
    alpha: float,
) -> None:
    labels = [_category_label(row.get("effective_attack_name")) for row in rows]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    color_by_label = {label: cmap(idx % cmap.N) for idx, label in enumerate(unique_labels)}
    marker_by_partition = {"train": "o", "inference": "^"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plotted = False
    for partition, marker in marker_by_partition.items():
        for label in unique_labels:
            indices = [
                idx
                for idx, row in enumerate(rows)
                if row.get("partition") == partition
                and labels[idx] == label
                and _row_float(row, "head_projection") is not None
                and _row_float(row, "prediction_entropy") is not None
            ]
            if not indices:
                continue
            plotted = True
            x_values = np.asarray([_row_float(rows[idx], "head_projection") for idx in indices], dtype=np.float64)
            y_values = np.asarray([_row_float(rows[idx], "prediction_entropy") for idx in indices], dtype=np.float64)
            ax.scatter(
                x_values,
                y_values,
                s=point_size,
                alpha=alpha,
                marker=marker,
                color=color_by_label[label],
                edgecolor="black" if partition == "inference" else "none",
                linewidth=0.3,
                label=f"{partition} {label}",
            )
    if not plotted:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("CNN head projection vs prediction entropy")
    ax.set_xlabel("Head projection (backdoor logit axis)")
    ax.set_ylabel("Prediction entropy")
    ax.grid(True, linestyle="--", alpha=0.3)
    if plotted:
        handles, handle_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(handle_labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_cnn_behavior_plots(
    *,
    rows: list[dict[str, Any]],
    plot_root: Path,
    point_size: float,
    alpha: float,
) -> dict[str, Any]:
    score_dir = plot_root / "scores"
    margin_dir = plot_root / "margins"
    projection_dir = plot_root / "head_projection"
    for directory in [score_dir, margin_dir, projection_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    score_histogram = score_dir / "backdoor_score_histogram.png"
    score_ecdf = score_dir / "backdoor_score_ecdf.png"
    score_by_group = score_dir / "backdoor_score_by_partition_and_label.png"
    margin_histogram = margin_dir / "head_margin_histogram.png"
    projection_scatter = projection_dir / "head_projection_scatter.png"
    projection_by_group = projection_dir / "head_projection_by_partition_and_label.png"

    _plot_cnn_metric_histogram(
        output_path=score_histogram,
        rows=rows,
        field="backdoor_score",
        title="CNN backdoor score distribution",
        xlabel="Backdoor score",
    )
    _plot_cnn_metric_ecdf(
        output_path=score_ecdf,
        rows=rows,
        field="backdoor_score",
        title="CNN backdoor score ECDF",
        xlabel="Backdoor score",
    )
    _plot_cnn_metric_by_partition_label(
        output_path=score_by_group,
        rows=rows,
        field="backdoor_score",
        title="CNN backdoor score by partition and label",
        ylabel="Backdoor score",
        point_size=point_size,
        alpha=alpha,
    )
    _plot_cnn_metric_histogram(
        output_path=margin_histogram,
        rows=rows,
        field="head_margin",
        title="CNN classifier-head margin distribution",
        xlabel="Best attack logit - clean logit",
    )
    _plot_cnn_head_projection_scatter(
        output_path=projection_scatter,
        rows=rows,
        point_size=point_size,
        alpha=alpha,
    )
    _plot_cnn_metric_by_partition_label(
        output_path=projection_by_group,
        rows=rows,
        field="head_projection",
        title="CNN head projection by partition and label",
        ylabel="Head projection",
        point_size=point_size,
        alpha=alpha,
    )

    return {
        "scores": {
            "plot_dir": str(score_dir),
            "backdoor_score_histogram": str(score_histogram),
            "backdoor_score_ecdf": str(score_ecdf),
            "backdoor_score_by_partition_and_label": str(score_by_group),
        },
        "margins": {
            "plot_dir": str(margin_dir),
            "head_margin_histogram": str(margin_histogram),
        },
        "head_projection": {
            "plot_dir": str(projection_dir),
            "head_projection_scatter": str(projection_scatter),
            "head_projection_by_partition_and_label": str(projection_by_group),
        },
    }


def _save_cnn_feature_embedding_rows_csv(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    probability_fields: list[str],
) -> None:
    fieldnames = [
        "embedding_index",
        "source_row_index",
        "partition",
        "model_name",
        "label",
        "binary_label",
        "task_label",
        "task_label_name",
        "predicted_label",
        "predicted_label_name",
        "backdoor_score",
        "clean_probability",
        "predicted_probability",
        "max_probability",
        "prediction_entropy",
        "head_projection",
        "head_margin",
        "clean_logit",
        "best_attack_logit",
        "best_attack_logit_name",
        "dataset_name",
        "subset_name",
        "model_family",
        "attack_name",
        "effective_attack_name",
        *probability_fields,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _save_cnn_tsne_coordinates_csv(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    embedding: np.ndarray,
) -> None:
    fieldnames = [
        "embedding_index",
        "source_row_index",
        "partition",
        "model_name",
        "label",
        "binary_label",
        "task_label_name",
        "predicted_label_name",
        "attack_name",
        "effective_attack_name",
        "backdoor_score",
        "head_projection",
        "head_margin",
        "prediction_entropy",
        "tsne_1",
        "tsne_2",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    **{field: row.get(field) for field in fieldnames if not field.startswith("tsne_")},
                    "tsne_1": float(embedding[idx, 0]),
                    "tsne_2": float(embedding[idx, 1]),
                }
            )


def _run_cnn_embedding_tsne_view(
    *,
    view_name: str,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    plot_dir: Path,
    embedding_dir: Path,
    perplexity: float,
    learning_rate: str | float,
    max_iter: int,
    metric: str,
    init: str,
    random_state: int,
    standardize: bool,
    point_size: float,
    alpha: float,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    view_slug = _safe_slug(view_name)
    view_report: dict[str, Any] = {
        "view": str(view_name),
        "n_samples": int(len(rows)),
    }
    if len(rows) < 2:
        warning = f"Skipping CNN feature t-SNE view {view_name!r} because it has fewer than two samples"
        warnings.append(warning)
        view_report["warning"] = warning
        return view_report, warnings

    embedding, replaced, resolved_perplexity, resolved_init = _fit_tsne_embedding(
        values=embeddings,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=init,
        random_state=random_state,
        standardize=standardize,
    )
    if replaced > 0:
        warnings.append(
            f"Replaced {replaced} non-finite CNN feature embedding values with zero for view {view_name!r}"
        )

    csv_path = embedding_dir / f"{view_slug}_tsne_coordinates.csv"
    _save_cnn_tsne_coordinates_csv(
        output_path=csv_path,
        rows=rows,
        embedding=embedding,
    )

    partition_plot = plot_dir / f"{view_slug}_by_partition.png"
    label_plot = plot_dir / f"{view_slug}_by_label.png"
    attack_plot = plot_dir / f"{view_slug}_by_attack.png"
    attack_and_clean_plot = plot_dir / f"{view_slug}_by_attack_and_clean.png"
    attack_and_clean_categories = [row.get("effective_attack_name") for row in rows]
    _plot_categorical_tsne_embedding(
        output_path=partition_plot,
        embedding=embedding,
        categories=[row.get("partition") for row in rows],
        title=f"CNN feature t-SNE ({view_name}) by partition",
        point_size=point_size,
        alpha=alpha,
    )
    _plot_categorical_tsne_embedding(
        output_path=label_plot,
        embedding=embedding,
        categories=[row.get("task_label_name") for row in rows],
        title=f"CNN feature t-SNE ({view_name}) by label",
        point_size=point_size,
        alpha=alpha,
    )
    _plot_categorical_tsne_embedding(
        output_path=attack_and_clean_plot,
        embedding=embedding,
        categories=attack_and_clean_categories,
        title=f"CNN feature t-SNE ({view_name}) by attack and clean",
        point_size=point_size,
        alpha=alpha,
    )
    attack_indices = [idx for idx, row in enumerate(rows) if _is_cnn_backdoor_embedding_row(row)]
    attack_rows = [rows[idx] for idx in attack_indices]
    attack_embedding = embedding[np.asarray(attack_indices, dtype=np.int64)] if attack_indices else embedding[:0]
    _plot_categorical_tsne_embedding(
        output_path=attack_plot,
        embedding=attack_embedding,
        categories=[row.get("attack_name") or row.get("effective_attack_name") for row in attack_rows],
        title=f"CNN feature t-SNE ({view_name}) by attack",
        point_size=point_size,
        alpha=alpha,
    )

    view_report.update(
        {
            "perplexity": float(resolved_perplexity),
            "resolved_init": resolved_init,
            "embedding_csv": str(csv_path),
            "plots": {
                "partition": str(partition_plot),
                "label": str(label_plot),
                "attack": str(attack_plot),
                "attack_and_clean": str(attack_and_clean_plot),
            },
            "attack_plot_field": "attack_name",
            "attack_plot_scope": "backdoor_rows_only",
            "attack_plot_n_samples": int(len(attack_rows)),
            "attack_and_clean_plot_field": "effective_attack_name",
            "attack_and_clean_plot_scope": "all_rows",
            "attack_and_clean_plot_n_samples": int(len(rows)),
            "attack_and_clean_plot_categories": sorted(
                {_category_label(value) for value in attack_and_clean_categories}
            ),
        }
    )
    return view_report, warnings


def _is_cnn_backdoor_embedding_row(row: dict[str, Any]) -> bool:
    binary_label = row.get("binary_label")
    if binary_label is not None:
        try:
            return int(binary_label) == 1
        except (TypeError, ValueError):
            pass
    return row.get("task_label_name") not in {None, "", "clean", "unknown"}


def _resolve_cnn_effective_attack_name(
    *,
    attack_name: str | None,
    task_label_name: str | None,
    binary_label: int | None,
) -> str:
    if binary_label is not None:
        if int(binary_label) == 0:
            return "clean"
        if attack_name:
            return str(attack_name)
    if task_label_name == "clean":
        return "clean"
    if attack_name:
        return str(attack_name)
    if task_label_name:
        return str(task_label_name)
    return "unknown"


def _logsumexp_axis1(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    max_values = np.max(arr, axis=1, keepdims=True)
    shifted = arr - max_values
    return (max_values.reshape(-1) + np.log(np.sum(np.exp(shifted), axis=1))).astype(np.float64, copy=False)


def _compute_cnn_head_behavior(
    *,
    raw_logits: np.ndarray | None,
    probabilities: np.ndarray | None,
    predicted_labels: np.ndarray,
    task_spec: Any,
) -> dict[str, np.ndarray | list[str]]:
    n_rows = int(predicted_labels.shape[0])
    nan_values = np.full(n_rows, np.nan, dtype=np.float64)
    metrics: dict[str, np.ndarray | list[str]] = {
        "clean_probability": nan_values.copy(),
        "predicted_probability": nan_values.copy(),
        "max_probability": nan_values.copy(),
        "prediction_entropy": nan_values.copy(),
        "head_projection": nan_values.copy(),
        "head_margin": nan_values.copy(),
        "clean_logit": nan_values.copy(),
        "best_attack_logit": nan_values.copy(),
        "best_attack_logit_name": ["unknown"] * n_rows,
    }

    if probabilities is not None:
        prob = np.asarray(probabilities, dtype=np.float64)
        if prob.ndim == 2 and prob.shape[0] == n_rows and prob.shape[1] >= int(task_spec.n_classes):
            clipped = np.clip(prob, 1e-12, 1.0)
            metrics["clean_probability"] = prob[:, int(task_spec.clean_class_index)].astype(np.float64, copy=False)
            metrics["max_probability"] = np.max(prob, axis=1).astype(np.float64, copy=False)
            predicted_prob = np.full(n_rows, np.nan, dtype=np.float64)
            for idx, predicted_label in enumerate(np.asarray(predicted_labels, dtype=np.int64).reshape(-1).tolist()):
                if 0 <= int(predicted_label) < int(prob.shape[1]):
                    predicted_prob[idx] = float(prob[idx, int(predicted_label)])
            metrics["predicted_probability"] = predicted_prob
            metrics["prediction_entropy"] = (-np.sum(clipped * np.log(clipped), axis=1)).astype(np.float64, copy=False)

    if raw_logits is None:
        return metrics

    logits = np.asarray(raw_logits, dtype=np.float64)
    if task_spec.is_binary:
        binary_logits = logits.reshape(n_rows, -1)[:, -1] if logits.ndim == 2 else logits.reshape(-1)
        metrics["head_projection"] = binary_logits.astype(np.float64, copy=False)
        metrics["head_margin"] = binary_logits.astype(np.float64, copy=False)
        metrics["clean_logit"] = np.zeros(n_rows, dtype=np.float64)
        metrics["best_attack_logit"] = binary_logits.astype(np.float64, copy=False)
        attack_name = str(task_spec.class_names[1]) if int(task_spec.n_classes) > 1 else "backdoor"
        metrics["best_attack_logit_name"] = [attack_name] * n_rows
        return metrics

    if logits.ndim == 1:
        logits = np.column_stack([-logits.reshape(-1), logits.reshape(-1)])
    if logits.ndim != 2 or logits.shape[0] != n_rows:
        return metrics

    clean_idx = int(task_spec.clean_class_index)
    if clean_idx < 0 or clean_idx >= int(logits.shape[1]):
        return metrics
    attack_indices = [idx for idx in range(min(int(logits.shape[1]), int(task_spec.n_classes))) if idx != clean_idx]
    if not attack_indices:
        return metrics

    clean_logits = logits[:, clean_idx].astype(np.float64, copy=False)
    attack_logits = logits[:, attack_indices].astype(np.float64, copy=False)
    best_local = np.argmax(attack_logits, axis=1).astype(np.int64, copy=False)
    best_indices = np.asarray([attack_indices[int(local_idx)] for local_idx in best_local], dtype=np.int64)
    best_attack_logits = attack_logits[np.arange(n_rows, dtype=np.int64), best_local]
    metrics["clean_logit"] = clean_logits
    metrics["best_attack_logit"] = best_attack_logits.astype(np.float64, copy=False)
    metrics["best_attack_logit_name"] = [
        str(task_spec.class_names[int(index)]) if 0 <= int(index) < int(task_spec.n_classes) else f"class_{int(index)}"
        for index in best_indices.tolist()
    ]
    metrics["head_margin"] = (best_attack_logits - clean_logits).astype(np.float64, copy=False)
    metrics["head_projection"] = (_logsumexp_axis1(attack_logits) - clean_logits).astype(np.float64, copy=False)
    return metrics


def _metric_value_for_row(metrics: Mapping[str, np.ndarray | list[str]], field: str, index: int) -> Any:
    values = metrics.get(field)
    if values is None:
        return None
    value = values[int(index)]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def run_supervised_cnn_feature_tsne_pipeline(
    *,
    run_dir: Path,
    output_root: Path | None = None,
    run_id: str | None = None,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
    init: str = "pca",
    random_state: int | None = None,
    standardize: bool = True,
    point_size: float = 28.0,
    alpha: float = 0.85,
) -> dict[str, Any]:
    from ..supervised.cnn import load_cnn_checkpoint
    from ..supervised.pipeline import (
        _load_features_for_tuning_manifest,
        _load_manifest_items_for_tuning_manifest,
        _predict_task_outputs,
        _slice_supervised_features,
        _task_spec_from_manifest,
    )
    from ..supervised.registry import model_backend
    from ..utilities.core.manifest import infer_attack_sample_identities

    start = perf_counter()
    supervised_run_dir = _resolve_path(run_dir)
    tuning_manifest_path = supervised_run_dir / "reports" / "tuning_manifest.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")
    manifest = _read_json_object(tuning_manifest_path)

    winner_model_name = _resolve_supervised_winner_model_name(supervised_run_dir)
    if winner_model_name is not None and model_backend(winner_model_name) != "cnn":
        raise ValueError(
            "supervised-cnn feature t-SNE requires a CNN winner, "
            f"got winner model {winner_model_name!r}"
        )

    checkpoint_path = _resolve_supervised_best_cnn_path(supervised_run_dir)
    model = load_cnn_checkpoint(checkpoint_path)
    task_spec = _task_spec_from_manifest(manifest)
    features = _load_features_for_tuning_manifest(manifest)

    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("Tuning manifest is missing data")
    train_indices = np.asarray(data.get("train_indices", []), dtype=np.int64)
    infer_indices = np.asarray(data.get("infer_indices", []), dtype=np.int64)
    if train_indices.size == 0:
        raise ValueError("CNN feature t-SNE requires at least one train row in data.train_indices")
    if infer_indices.size == 0:
        raise ValueError("CNN feature t-SNE requires at least one held-out inference row in data.infer_indices")

    selected_indices = np.concatenate([train_indices, infer_indices]).astype(np.int64, copy=False)
    if int(selected_indices.size) < 2:
        raise ValueError("CNN feature t-SNE requires at least two total train/inference rows")
    selected_features = _slice_supervised_features(features, selected_indices)
    embeddings = np.asarray(model.extract_features(selected_features), dtype=np.float32)
    if int(embeddings.shape[0]) != int(selected_indices.size):
        raise ValueError(
            "CNN feature extractor row count does not match selected train/inference rows: "
            f"{embeddings.shape[0]} != {selected_indices.size}"
        )

    labels_value = np.load(data["labels_value_path"]).astype(np.int32)
    labels_known = np.load(data["labels_known_path"]).astype(bool)
    with open(data["model_names_path"], "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(labels_value.shape[0]):
        raise ValueError("Stored supervised model names and labels are not aligned")

    selected_model_names = [model_names[int(i)] for i in selected_indices.tolist()]
    selected_task_labels = labels_value[selected_indices]
    selected_known = labels_known[selected_indices]
    embedding_labels = np.asarray(selected_task_labels, dtype=np.int32)
    embedding_labels = np.where(selected_known, embedding_labels, -1).astype(np.int32, copy=False)

    outputs = _predict_task_outputs(model, selected_features, task_spec=task_spec)
    probabilities = outputs.probabilities
    probability_fields = _unique_probability_field_names(list(task_spec.class_names))
    raw_logits: np.ndarray | None = None
    if hasattr(model, "decision_function"):
        raw_logits = np.asarray(model.decision_function(selected_features), dtype=np.float64)
    head_behavior = _compute_cnn_head_behavior(
        raw_logits=raw_logits,
        probabilities=probabilities,
        predicted_labels=np.asarray(outputs.predicted_labels, dtype=np.int32),
        task_spec=task_spec,
    )

    manifest_items = _load_manifest_items_for_tuning_manifest(manifest)
    identities = infer_attack_sample_identities(manifest_items)
    identity_by_name = {identity.model_name: identity for identity in identities}
    item_by_name = {item.model_name: item for item in manifest_items}

    rows: list[dict[str, Any]] = []
    for embedding_index, source_row_index in enumerate(selected_indices.tolist()):
        model_name = selected_model_names[int(embedding_index)]
        identity = identity_by_name.get(model_name)
        item = item_by_name.get(model_name)
        dataset_name = None
        if item is not None and item.model_dir.parent.name:
            dataset_name = str(item.model_dir.parent.name)
        task_label = int(embedding_labels[int(embedding_index)])
        binary_label = None if task_label < 0 else int(task_spec.project_label_to_binary(task_label))
        predicted_label = int(outputs.predicted_labels[int(embedding_index)])
        task_label_name = (
            "unknown"
            if task_label < 0 or task_label >= task_spec.n_classes
            else str(task_spec.class_names[int(task_label)])
        )
        attack_name = None if identity is None else str(identity.attack_name)
        row: dict[str, Any] = {
            "embedding_index": int(embedding_index),
            "source_row_index": int(source_row_index),
            "partition": "train" if embedding_index < int(train_indices.size) else "inference",
            "model_name": model_name,
            "label": None if binary_label is None else int(binary_label),
            "binary_label": None if binary_label is None else int(binary_label),
            "task_label": None if task_label < 0 else int(task_label),
            "task_label_name": task_label_name,
            "predicted_label": int(predicted_label),
            "predicted_label_name": (
                str(task_spec.class_names[predicted_label])
                if 0 <= predicted_label < task_spec.n_classes
                else f"class_{predicted_label}"
            ),
            "backdoor_score": float(outputs.backdoor_scores[int(embedding_index)]),
            "clean_probability": _metric_value_for_row(head_behavior, "clean_probability", int(embedding_index)),
            "predicted_probability": _metric_value_for_row(
                head_behavior, "predicted_probability", int(embedding_index)
            ),
            "max_probability": _metric_value_for_row(head_behavior, "max_probability", int(embedding_index)),
            "prediction_entropy": _metric_value_for_row(head_behavior, "prediction_entropy", int(embedding_index)),
            "head_projection": _metric_value_for_row(head_behavior, "head_projection", int(embedding_index)),
            "head_margin": _metric_value_for_row(head_behavior, "head_margin", int(embedding_index)),
            "clean_logit": _metric_value_for_row(head_behavior, "clean_logit", int(embedding_index)),
            "best_attack_logit": _metric_value_for_row(head_behavior, "best_attack_logit", int(embedding_index)),
            "best_attack_logit_name": _metric_value_for_row(
                head_behavior, "best_attack_logit_name", int(embedding_index)
            ),
            "dataset_name": dataset_name,
            "subset_name": None if identity is None else str(identity.subset_name),
            "model_family": None if identity is None else str(identity.model_family),
            "attack_name": attack_name,
            "effective_attack_name": _resolve_cnn_effective_attack_name(
                attack_name=attack_name,
                task_label_name=task_label_name,
                binary_label=None if binary_label is None else int(binary_label),
            ),
        }
        if probabilities is not None:
            for field_name, probability in zip(
                probability_fields,
                np.asarray(probabilities[int(embedding_index)], dtype=np.float64).reshape(-1).tolist(),
            ):
                row[field_name] = float(probability)
        rows.append(row)

    resolved_output_root = (
        _resolve_path(output_root)
        if output_root is not None
        else _default_cnn_tsne_output_root(supervised_run_dir)
    )
    ctx = create_run_context(
        pipeline="unsupervised_cnn_tsne",
        output_root=resolved_output_root,
        run_id=run_id or _default_cnn_tsne_run_id(supervised_run_dir),
    )

    embeddings_path = ctx.features_dir / "cnn_feature_embeddings.npy"
    labels_path = ctx.features_dir / "cnn_feature_embedding_labels.npy"
    model_names_path = ctx.features_dir / "cnn_feature_embedding_model_names.json"
    np.save(embeddings_path, embeddings)
    np.save(labels_path, embedding_labels)
    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump(selected_model_names, f, indent=2)

    row_metadata_path = ctx.reports_dir / "cnn_feature_embedding_rows.csv"
    _save_cnn_feature_embedding_rows_csv(
        output_path=row_metadata_path,
        rows=rows,
        probability_fields=probability_fields,
    )

    plot_root = ctx.plots_dir / "cnn_feature_extractor"
    plot_dir = plot_root / "tsne"
    embedding_root = ctx.reports_dir / "cnn_feature_extractor"
    embedding_dir = embedding_root / "tsne"
    plot_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    behavior_plots = _run_cnn_behavior_plots(
        rows=rows,
        plot_root=plot_root,
        point_size=float(point_size),
        alpha=float(alpha),
    )

    resolved_random_state = (
        int(random_state)
        if random_state is not None
        else int(manifest.get("tuning", {}).get("random_state", 42))
    )
    view_specs = [
        ("combined", np.arange(int(embeddings.shape[0]), dtype=np.int64)),
        ("train", np.arange(0, int(train_indices.size), dtype=np.int64)),
        ("inference", np.arange(int(train_indices.size), int(embeddings.shape[0]), dtype=np.int64)),
    ]
    view_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    for view_name, rel_indices in view_specs:
        view_rows = [rows[int(i)] for i in rel_indices.tolist()]
        view_embeddings = embeddings[rel_indices]
        view_report, view_warnings = _run_cnn_embedding_tsne_view(
            view_name=view_name,
            rows=view_rows,
            embeddings=view_embeddings,
            plot_dir=plot_dir,
            embedding_dir=embedding_dir,
            perplexity=float(perplexity),
            learning_rate=learning_rate,
            max_iter=int(max_iter),
            metric=str(metric),
            init=str(init),
            random_state=resolved_random_state,
            standardize=bool(standardize),
            point_size=float(point_size),
            alpha=float(alpha),
        )
        view_reports.append(view_report)
        warnings.extend(view_warnings)

    elapsed_seconds = float(perf_counter() - start)
    report = {
        "analysis": "supervised_cnn_feature_tsne",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "supervised_run_dir": str(supervised_run_dir),
        "tuning_manifest": str(tuning_manifest_path),
        "checkpoint": str(checkpoint_path),
        "task": task_spec.to_dict(),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "n_train": int(train_indices.size),
        "n_inference": int(infer_indices.size),
        "train_indices": [int(x) for x in train_indices.tolist()],
        "infer_indices": [int(x) for x in infer_indices.tolist()],
        "parameters": {
            "perplexity": float(perplexity),
            "learning_rate": learning_rate,
            "max_iter": int(max_iter),
            "metric": str(metric),
            "init": str(init),
            "random_state": int(resolved_random_state),
            "standardize": bool(standardize),
            "point_size": float(point_size),
            "alpha": float(alpha),
        },
        "artifacts": {
            "embeddings": str(embeddings_path),
            "labels": str(labels_path),
            "model_names": str(model_names_path),
            "row_metadata": str(row_metadata_path),
            "plot_dir": str(plot_root),
            "tsne_plot_dir": str(plot_dir),
            "embedding_dir": str(embedding_dir),
            "behavior_plots": behavior_plots,
        },
        "views": view_reports,
        "behavior_plots": behavior_plots,
        "warnings": warnings,
        "elapsed_seconds": elapsed_seconds,
    }
    report_path = ctx.reports_dir / "cnn_feature_tsne_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("cnn_feature_embeddings", embeddings_path)
    ctx.add_artifact("cnn_feature_embedding_labels", labels_path)
    ctx.add_artifact("cnn_feature_embedding_model_names", model_names_path)
    ctx.add_artifact("cnn_feature_embedding_rows_csv", row_metadata_path)
    ctx.add_artifact("cnn_feature_tsne_report", report_path)
    ctx.add_artifact("cnn_feature_plots", plot_root)
    ctx.add_artifact("cnn_feature_tsne_plots", plot_dir)
    ctx.add_artifact("cnn_feature_tsne_embeddings", embedding_dir)
    ctx.add_artifact("cnn_feature_score_plots", plot_root / "scores")
    ctx.add_artifact("cnn_feature_margin_plots", plot_root / "margins")
    ctx.add_artifact("cnn_feature_head_projection_plots", plot_root / "head_projection")
    ctx.add_timing("elapsed_seconds", elapsed_seconds)
    ctx.finalize(
        {
            "pipeline": "unsupervised_cnn_tsne",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "supervised_run_dir": str(supervised_run_dir),
            "tuning_manifest": str(tuning_manifest_path),
            "checkpoint": str(checkpoint_path),
            "task": task_spec.to_dict(),
            "parameters": report["parameters"],
            "n_train": int(train_indices.size),
            "n_inference": int(infer_indices.size),
            "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
            "warnings": warnings,
        }
    )

    return {
        "run_dir": ctx.run_dir,
        "report": report_path,
        "plot_dir": plot_root,
        "embedding_dir": embedding_dir,
        "embeddings": embeddings_path,
        "row_metadata": row_metadata_path,
    }


def _feature_column_groups(feature_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for col_idx, feature_name in enumerate(feature_names):
        emitted_feature = _emitted_feature_name(feature_name)
        layer = _extract_layer(feature_name)
        if layer is None:
            continue
        try:
            block_name = _feature_block_name(feature_name)
        except ValueError:
            block_name = str(feature_name)

        if emitted_feature not in grouped:
            grouped[emitted_feature] = {
                "feature_name": emitted_feature,
                "column_indices": [],
                "layers": [],
                "block_names": [],
            }
            order.append(emitted_feature)

        grouped[emitted_feature]["column_indices"].append(int(col_idx))
        grouped[emitted_feature]["layers"].append(int(layer))
        grouped[emitted_feature]["block_names"].append(str(block_name))

    return [grouped[name] for name in order]


def _plot_layer_value_scatter(
    *,
    output_path: Path,
    panel_specs: list[dict[str, Any]],
    layers: list[int],
    title: str,
    y_label: str,
    point_size: float,
    alpha: float,
) -> None:
    if not panel_specs:
        raise ValueError("panel_specs must include at least one panel for layer scatter plotting")

    expected_columns = int(len(layers))
    if expected_columns <= 0:
        raise ValueError("layers must include at least one layer for layer scatter plotting")

    layers_np = np.asarray(layers, dtype=np.int32)
    unique_layers = sorted(set(int(x) for x in layers))
    panel_width = max(6.0, min(12.0, len(unique_layers) * 0.35))
    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(panel_width * len(panel_specs), 6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    for ax, spec in zip(axes_flat, panel_specs):
        values = np.asarray(spec["values"], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(
                f"Expected 2D values for layer scatter plotting, got shape={values.shape}"
            )
        if int(values.shape[1]) != expected_columns:
            raise ValueError(
                "Layer scatter values column count does not match the provided layer list: "
                f"{values.shape[1]} != {expected_columns}"
            )

        n_samples = int(values.shape[0])
        if n_samples > 0:
            x = np.tile(layers_np, n_samples)
            y = values.reshape(-1)
            ax.scatter(x, y, s=point_size, alpha=alpha, color=str(spec["color"]))
        else:
            ax.text(
                0.5,
                0.5,
                "no samples",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#6b6b6b",
            )

        ax.set_title(f"{spec['title']} (n={n_samples})")
        ax.set_xticks(unique_layers)
        ax.set_xlabel("layer")
        ax.grid(True, linestyle="--", alpha=0.25)

    axes_flat[0].set_ylabel(y_label)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_value_boxplot_on_axis(
    *,
    ax: Any,
    panel_specs: list[dict[str, Any]],
    layers: list[int],
    y_label: str | None,
    title: str | None,
    title_fontsize: float | None,
    show_legend: bool,
    x_label: str | None,
    tick_layers: list[int] | None = None,
) -> bool:
    if not panel_specs:
        raise ValueError("panel_specs must include at least one panel for layer boxplot plotting")

    expected_columns = int(len(layers))
    if expected_columns <= 0:
        raise ValueError("layers must include at least one layer for layer boxplot plotting")

    layers_np = np.asarray(layers, dtype=np.int32)
    unique_layers = sorted(set(int(x) for x in layers))
    layer_masks = [layers_np == int(layer) for layer in unique_layers]

    box_offsets = np.linspace(-0.18, 0.18, num=len(panel_specs), dtype=np.float32)
    box_width = 0.32 / max(len(panel_specs), 1)
    drew_any_boxes = False

    for spec, offset in zip(panel_specs, box_offsets):
        values = np.asarray(spec["values"], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(
                f"Expected 2D values for layer boxplot plotting, got shape={values.shape}"
            )
        if int(values.shape[1]) != expected_columns:
            raise ValueError(
                "Layer boxplot values column count does not match the provided layer list: "
                f"{values.shape[1]} != {expected_columns}"
            )

        grouped_values: list[np.ndarray] = []
        positions: list[float] = []
        for layer, layer_mask in zip(unique_layers, layer_masks):
            layer_values = values[:, layer_mask].reshape(-1)
            if int(layer_values.size) <= 0:
                continue
            grouped_values.append(layer_values)
            positions.append(float(layer) + float(offset))

        if not grouped_values:
            continue

        boxplot = ax.boxplot(
            grouped_values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": str(spec["color"]), "linewidth": 1.0},
            capprops={"color": str(spec["color"]), "linewidth": 1.0},
            boxprops={"edgecolor": str(spec["color"]), "linewidth": 1.0},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(str(spec["color"]))
            patch.set_alpha(0.5)
        drew_any_boxes = True

    if drew_any_boxes:
        if show_legend:
            handles = [
                Patch(
                    facecolor=str(spec["color"]),
                    edgecolor=str(spec["color"]),
                    alpha=0.5,
                    label=str(spec["title"]),
                )
                for spec in panel_specs
            ]
            ax.legend(handles=handles, loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "no samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#6b6b6b",
        )

    visible_tick_layers = unique_layers if tick_layers is None else [int(x) for x in tick_layers]
    ax.set_xticks(visible_tick_layers)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    if unique_layers:
        ax.set_xlim(float(min(unique_layers)) - 0.75, float(max(unique_layers)) + 0.75)
    if title is not None:
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
    return drew_any_boxes


def _summary_tick_layers(layers: list[int], *, max_ticks: int = 8) -> list[int]:
    unique_layers = sorted(set(int(x) for x in layers))
    if len(unique_layers) <= max_ticks:
        return unique_layers

    step = int(np.ceil(len(unique_layers) / max_ticks))
    ticks = unique_layers[::step]
    if ticks[-1] != unique_layers[-1]:
        ticks.append(unique_layers[-1])
    return ticks


def _plot_layer_value_boxplot(
    *,
    output_path: Path,
    panel_specs: list[dict[str, Any]],
    layers: list[int],
    title: str,
    y_label: str,
) -> None:
    unique_layers = sorted(set(int(x) for x in layers))
    panel_width = max(8.0, min(16.0, len(unique_layers) * 0.45))
    fig, ax = plt.subplots(1, 1, figsize=(panel_width, 6))
    _plot_layer_value_boxplot_on_axis(
        ax=ax,
        panel_specs=panel_specs,
        layers=layers,
        y_label=y_label,
        title=None,
        title_fontsize=None,
        show_legend=True,
        x_label="layer",
    )

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_all_layer_value_boxplots(
    *,
    output_path: Path,
    feature_specs: list[dict[str, Any]],
    label_specs: list[dict[str, Any]],
    title: str,
) -> None:
    if not feature_specs:
        raise ValueError("feature_specs must include at least one feature boxplot")

    n_rows, n_cols = _subplot_grid(len(feature_specs))
    fig_width = max(18.0, n_cols * 4.2)
    fig_height = max(12.0, n_rows * 3.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for ax, feature_spec in zip(axes_flat, feature_specs):
        _plot_layer_value_boxplot_on_axis(
            ax=ax,
            panel_specs=list(feature_spec["panel_specs"]),
            layers=list(feature_spec["layers"]),
            y_label=None,
            title=str(feature_spec["feature_name"]),
            title_fontsize=10,
            show_legend=False,
            x_label="layer",
            tick_layers=_summary_tick_layers(list(feature_spec["layers"])),
        )
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes_flat[len(feature_specs) :]:
        ax.axis("off")

    handles = [
        Patch(
            facecolor=str(spec["color"]),
            edgecolor=str(spec["color"]),
            alpha=0.5,
            label=str(spec["name"]).replace("_", " ").title(),
        )
        for spec in label_specs
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_layer_value_scatter_analysis(
    *,
    feature_file: Path,
    output_dir: Path,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    point_size: float = 6.0,
    alpha: float = 0.18,
) -> dict[str, Any]:
    bundle = load_feature_bundle(
        feature_file=feature_file,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
    )
    feature_groups = _feature_column_groups(bundle.feature_names)
    if not feature_groups:
        raise ValueError(
            "Could not resolve any layer-aware feature columns from the feature metadata. "
            "Layer scatter plots require spectral feature names with layer identifiers."
        )
    if bundle.labels is None:
        raise ValueError(
            "Layer scatter plots split by clean/backdoor require sample labels. "
            "Provide a labels file or dataset reference provenance for the feature bundle."
        )

    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    sanitized_features, replaced = _sanitize_matrix(bundle.features)
    warnings: list[str] = []
    if replaced > 0:
        warnings.append(
            f"Replaced {replaced} non-finite values with zero before generating layer scatter plots"
        )

    labels = np.asarray(bundle.labels, dtype=np.int32)
    label_specs = [
        {
            "value": 0,
            "name": "clean",
            "color": _LABEL_STYLE[0][1],
        },
        {
            "value": 1,
            "name": "backdoor",
            "color": _LABEL_STYLE[1][1],
        },
    ]
    unknown_count = int(np.sum(labels < 0))
    if unknown_count > 0:
        warnings.append(
            f"Ignoring {unknown_count} samples with unknown labels when generating clean/backdoor layer scatter plots"
        )

    plot_rows: list[dict[str, Any]] = []
    combined_boxplot_specs: list[dict[str, Any]] = []
    scatter_plot_count = 0
    box_plot_count = 0
    for group in feature_groups:
        emitted_feature = str(group["feature_name"])
        column_indices = np.asarray(group["column_indices"], dtype=np.int64)
        layers = [int(x) for x in group["layers"]]
        if column_indices.size == 0:
            continue

        feature_row: dict[str, Any] = {
            "feature_name": emitted_feature,
            "n_columns": int(column_indices.size),
            "layers": sorted(set(layers)),
        }
        feature_slug = _safe_slug(emitted_feature)
        panel_specs: list[dict[str, Any]] = []
        n_known_label_samples = 0

        for spec in label_specs:
            label_mask = labels == int(spec["value"])
            n_label_samples = int(np.sum(label_mask))
            feature_row[f"n_{spec['name']}_samples"] = n_label_samples
            if n_label_samples <= 0:
                warnings.append(
                    f"Leaving the {spec['name']} panel empty for feature '{emitted_feature}' because no samples were found"
                )
            else:
                n_known_label_samples += n_label_samples

            panel_specs.append(
                {
                    "title": str(spec["name"]).replace("_", " ").title(),
                    "color": str(spec["color"]),
                    "values": sanitized_features[label_mask][:, column_indices],
                }
            )

        if n_known_label_samples <= 0:
            warnings.append(
                f"Skipping combined layer scatter for feature '{emitted_feature}' because no clean/backdoor samples were found"
            )
            plot_rows.append(feature_row)
            continue

        plot_path = resolved_output_dir / f"{feature_slug}.png"
        _plot_layer_value_scatter(
            output_path=plot_path,
            panel_specs=panel_specs,
            layers=layers,
            title=f"Layer vs value: {emitted_feature}",
            y_label=emitted_feature,
            point_size=point_size,
            alpha=alpha,
        )
        box_plot_path = resolved_output_dir / f"{feature_slug}_boxplot.png"
        _plot_layer_value_boxplot(
            output_path=box_plot_path,
            panel_specs=panel_specs,
            layers=layers,
            title=f"Layer distributions: {emitted_feature}",
            y_label=emitted_feature,
        )
        feature_row["plot"] = str(plot_path)
        feature_row["scatter_plot"] = str(plot_path)
        feature_row["box_plot"] = str(box_plot_path)
        combined_boxplot_specs.append(
            {
                "feature_name": emitted_feature,
                "layers": list(layers),
                "panel_specs": panel_specs,
            }
        )
        scatter_plot_count += 1
        box_plot_count += 1

        plot_rows.append(feature_row)

    combined_box_plot_path: str | None = None
    if combined_boxplot_specs:
        combined_box_plot = resolved_output_dir / "all_features_boxplots.png"
        _plot_all_layer_value_boxplots(
            output_path=combined_box_plot,
            feature_specs=combined_boxplot_specs,
            label_specs=label_specs,
            title="Layer Distributions Across All Features",
        )
        combined_box_plot_path = str(combined_box_plot)

    return {
        "analysis": "layer_value_scatter",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_file": str(bundle.feature_file),
        "model_names_file": str(bundle.model_names_file),
        "labels_file": str(bundle.labels_file) if bundle.labels_file is not None else None,
        "metadata_file": str(bundle.metadata_file) if bundle.metadata_file is not None else None,
        "dataset_reference_report": (
            str(bundle.dataset_reference_report) if bundle.dataset_reference_report is not None else None
        ),
        "feature_shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        "plot_mode": "side_by_side_labels",
        "box_plot_mode": "paired_label_boxplots_by_layer",
        "combined_box_plot": combined_box_plot_path,
        "label_panels": [str(spec["name"]) for spec in label_specs],
        "n_feature_groups": int(len(plot_rows)),
        "n_feature_plots": int(scatter_plot_count),
        "n_feature_scatter_plots": int(scatter_plot_count),
        "n_feature_box_plots": int(box_plot_count),
        "n_total_plot_files": int(
            scatter_plot_count + box_plot_count + (1 if combined_box_plot_path is not None else 0)
        ),
        "feature_plots": plot_rows,
        "warnings": warnings,
    }


def _selected_rank_feature_names(
    feature_names: list[str],
    *,
    requested_features: list[str] | None,
) -> tuple[list[str], list[str] | None]:
    normalized_requested_features = _normalize_requested_features(requested_features)
    requested_feature_set = (
        set(normalized_requested_features) if normalized_requested_features is not None else None
    )

    selected: list[str] = []
    seen: set[str] = set()
    available_feature_groups: set[str] = set()
    for feature_name in feature_names:
        emitted_feature = _emitted_feature_name(feature_name)
        feature_group = _feature_group_for_feature_name(emitted_feature)
        if feature_group is not None:
            available_feature_groups.add(feature_group)
        if requested_feature_set is not None and feature_group not in requested_feature_set:
            continue
        if emitted_feature in seen:
            continue
        seen.add(emitted_feature)
        selected.append(emitted_feature)

    if requested_feature_set is not None:
        missing_features = [
            feature_name
            for feature_name in normalized_requested_features or []
            if feature_name not in available_feature_groups
        ]
        if missing_features:
            preview = ", ".join(missing_features[:5])
            raise ValueError(
                "Requested --features are not available for this feature bundle. "
                f"Examples: {preview}"
            )

    if not selected:
        raise ValueError("Rank feature value analysis resolved to zero emitted features")
    return selected, normalized_requested_features


def _rank_feature_label_specs(labels: np.ndarray) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for label_value in (0, 1):
        if not np.any(labels == label_value):
            continue
        label_name, color = _LABEL_STYLE[label_value]
        specs.append(
            {
                "value": int(label_value),
                "name": str(label_name),
                "title": str(label_name).replace("_", " ").title(),
                "color": str(color),
            }
        )
    return specs


def _spread_positions(center: float, count: int, jitter: float) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    if count == 1 or jitter <= 0:
        return np.full((count,), float(center), dtype=np.float32)
    return np.linspace(
        float(center) - float(jitter),
        float(center) + float(jitter),
        num=int(count),
        dtype=np.float32,
    )


def _save_rank_feature_value_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "model_name",
                "label",
                "rank",
                "feature_name",
                "value",
                "source_column_count",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "index": int(index),
                    "model_name": str(row["model_name"]),
                    "label": int(row["label"]),
                    "rank": int(row["rank"]),
                    "feature_name": str(row["feature_name"]),
                    "value": float(row["value"]),
                    "source_column_count": int(row["source_column_count"]),
                }
            )


def _plot_rank_feature_value_scatter_on_axis(
    *,
    ax: Any,
    rows: list[dict[str, Any]],
    ordered_ranks: list[int],
    rank_positions: dict[int, float],
    label_specs: list[dict[str, Any]],
    point_size: float,
    alpha: float,
    jitter: float,
    title: str,
    show_legend: bool,
) -> bool:
    if not ordered_ranks:
        raise ValueError("ordered_ranks must include at least one rank")

    label_offsets = np.linspace(-0.16, 0.16, num=max(1, len(label_specs)), dtype=np.float32)
    drew_any_points = False

    for label_offset, spec in zip(label_offsets, label_specs):
        x_values: list[float] = []
        y_values: list[float] = []
        for rank in ordered_ranks:
            group_rows = sorted(
                (
                    row
                    for row in rows
                    if int(row["rank"]) == int(rank) and int(row["label"]) == int(spec["value"])
                ),
                key=lambda row: str(row["model_name"]),
            )
            if not group_rows:
                continue

            base_x = float(rank_positions[int(rank)]) + float(label_offset)
            x_values.extend(_spread_positions(base_x, len(group_rows), jitter).tolist())
            y_values.extend(float(row["value"]) for row in group_rows)

        if not x_values:
            continue

        ax.scatter(
            x_values,
            y_values,
            s=point_size,
            alpha=alpha,
            color=str(spec["color"]),
            label=str(spec["title"]) if show_legend else None,
        )
        drew_any_points = True

    if not drew_any_points:
        ax.text(
            0.5,
            0.5,
            "no samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#6b6b6b",
        )

    ax.set_title(title)
    ax.set_xticks([rank_positions[int(rank)] for rank in ordered_ranks])
    ax.set_xticklabels([str(int(rank)) for rank in ordered_ranks], rotation=45, ha="right")
    ax.set_xlabel("rank")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlim(-0.6, float(len(ordered_ranks) - 1) + 0.6)
    if show_legend and drew_any_points:
        ax.legend(loc="best")
    return drew_any_points


def _plot_rank_feature_value_boxplot_on_axis(
    *,
    ax: Any,
    rows: list[dict[str, Any]],
    ordered_ranks: list[int],
    rank_positions: dict[int, float],
    label_specs: list[dict[str, Any]],
    title: str,
    show_legend: bool,
) -> bool:
    if not ordered_ranks:
        raise ValueError("ordered_ranks must include at least one rank")

    label_offsets = np.linspace(-0.18, 0.18, num=max(1, len(label_specs)), dtype=np.float32)
    box_width = 0.3 / max(len(label_specs), 1)
    drew_any_boxes = False

    for label_offset, spec in zip(label_offsets, label_specs):
        grouped_values: list[np.ndarray] = []
        positions: list[float] = []
        for rank in ordered_ranks:
            rank_values = np.asarray(
                [
                    float(row["value"])
                    for row in rows
                    if int(row["rank"]) == int(rank) and int(row["label"]) == int(spec["value"])
                ],
                dtype=np.float32,
            )
            if int(rank_values.size) <= 0:
                continue
            grouped_values.append(rank_values)
            positions.append(float(rank_positions[int(rank)]) + float(label_offset))

        if not grouped_values:
            continue

        boxplot = ax.boxplot(
            grouped_values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": str(spec["color"]), "linewidth": 1.0},
            capprops={"color": str(spec["color"]), "linewidth": 1.0},
            boxprops={"edgecolor": str(spec["color"]), "linewidth": 1.0},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(str(spec["color"]))
            patch.set_alpha(0.5)
        drew_any_boxes = True

    if not drew_any_boxes:
        ax.text(
            0.5,
            0.5,
            "no samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#6b6b6b",
        )
    elif show_legend:
        handles = [
            Patch(
                facecolor=str(spec["color"]),
                edgecolor=str(spec["color"]),
                alpha=0.5,
                label=str(spec["title"]),
            )
            for spec in label_specs
        ]
        ax.legend(handles=handles, loc="best")

    ax.set_title(title)
    ax.set_xticks([rank_positions[int(rank)] for rank in ordered_ranks])
    ax.set_xticklabels([str(int(rank)) for rank in ordered_ranks], rotation=45, ha="right")
    ax.set_xlabel("rank")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_xlim(-0.6, float(len(ordered_ranks) - 1) + 0.6)
    return drew_any_boxes


def _plot_all_rank_feature_value_scatters(
    *,
    output_path: Path,
    feature_rows_by_name: dict[str, list[dict[str, Any]]],
    ordered_features: list[str],
    ordered_ranks: list[int],
    label_specs: list[dict[str, Any]],
    point_size: float,
    alpha: float,
    jitter: float,
    title: str,
) -> None:
    if not ordered_features:
        raise ValueError("ordered_features must include at least one feature")

    rank_positions = {int(rank): float(index) for index, rank in enumerate(ordered_ranks)}
    n_rows, n_cols = _subplot_grid(len(ordered_features))
    fig_width = max(18.0, n_cols * 4.4)
    fig_height = max(12.0, n_rows * 3.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for ax, feature_name in zip(axes_flat, ordered_features):
        _plot_rank_feature_value_scatter_on_axis(
            ax=ax,
            rows=list(feature_rows_by_name.get(feature_name, [])),
            ordered_ranks=ordered_ranks,
            rank_positions=rank_positions,
            label_specs=label_specs,
            point_size=point_size,
            alpha=alpha,
            jitter=jitter,
            title=str(feature_name),
            show_legend=False,
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes_flat[len(ordered_features) :]:
        ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=str(spec["color"]),
            markeredgecolor=str(spec["color"]),
            label=str(spec["title"]),
        )
        for spec in label_specs
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_all_rank_feature_value_boxplots(
    *,
    output_path: Path,
    feature_rows_by_name: dict[str, list[dict[str, Any]]],
    ordered_features: list[str],
    ordered_ranks: list[int],
    label_specs: list[dict[str, Any]],
    title: str,
) -> None:
    if not ordered_features:
        raise ValueError("ordered_features must include at least one feature")

    rank_positions = {int(rank): float(index) for index, rank in enumerate(ordered_ranks)}
    n_rows, n_cols = _subplot_grid(len(ordered_features))
    fig_width = max(18.0, n_cols * 4.4)
    fig_height = max(12.0, n_rows * 3.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for ax, feature_name in zip(axes_flat, ordered_features):
        _plot_rank_feature_value_boxplot_on_axis(
            ax=ax,
            rows=list(feature_rows_by_name.get(feature_name, [])),
            ordered_ranks=ordered_ranks,
            rank_positions=rank_positions,
            label_specs=label_specs,
            title=str(feature_name),
            show_legend=False,
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes_flat[len(ordered_features) :]:
        ax.axis("off")

    handles = [
        Patch(
            facecolor=str(spec["color"]),
            edgecolor=str(spec["color"]),
            alpha=0.5,
            label=str(spec["title"]),
        )
        for spec in label_specs
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_rank_feature_value_analysis(
    *,
    feature_file: Path,
    output_dir: Path,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    requested_features: list[str] | None = None,
    point_size: float = 10.0,
    alpha: float = 0.28,
    jitter: float = 0.08,
    sample_table_path: Path | None = None,
) -> dict[str, Any]:
    bundle = load_feature_bundle(
        feature_file=feature_file,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
    )
    representation_kind = str(bundle.metadata.get("representation_kind") or "").strip().lower()
    if representation_kind.startswith("architecture_independent"):
        raise ValueError(
            "Rank feature value analysis requires a raw or merged per-layer spectral bundle, "
            "not an architecture-independent aggregated bundle"
        )
    if not _feature_column_groups(bundle.feature_names):
        raise ValueError(
            "Could not resolve per-layer spectral feature columns from the feature metadata. "
            "Rank feature value analysis requires original per-layer spectral feature names."
        )
    if bundle.labels is None:
        raise ValueError(
            "Rank feature value plots split by clean/backdoor require sample labels. "
            "Provide a labels file or dataset reference provenance for the feature bundle."
        )
    if point_size <= 0:
        raise ValueError(f"point_size must be positive, got {point_size}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if jitter < 0:
        raise ValueError(f"jitter must be non-negative, got {jitter}")

    selected_feature_names, normalized_requested_features = _selected_rank_feature_names(
        bundle.feature_names,
        requested_features=requested_features,
    )

    try:
        owned_feature_names_by_model, selected_leaf_paths = _resolve_model_owned_feature_names(
            root_feature_path=bundle.feature_file,
            root_feature_names=bundle.feature_names,
            selected_model_names=bundle.model_names,
        )
    except Exception as exc:
        raise ValueError(
            "Rank feature value analysis requires dataset-reference provenance that can resolve "
            "each sample's owned raw feature columns"
        ) from exc

    root_feature_index = {str(feature_name): int(index) for index, feature_name in enumerate(bundle.feature_names)}
    labels = np.asarray(bundle.labels, dtype=np.int32)
    label_specs = _rank_feature_label_specs(labels)
    if not label_specs:
        raise ValueError(
            "Rank feature value plots split by clean/backdoor require labels 0 and/or 1 in the feature bundle"
        )

    warnings: list[str] = []
    unknown_rank_count = int(sum(rank is None for rank in bundle.sample_ranks))
    if unknown_rank_count > 0:
        warnings.append(
            f"Ignoring {unknown_rank_count} samples with unknown rank while generating rank feature value plots"
        )

    unknown_label_count = int(np.sum(~np.isin(labels, np.asarray([0, 1], dtype=np.int32))))
    if unknown_label_count > 0:
        warnings.append(
            "Ignoring "
            f"{unknown_label_count} samples with labels outside {{0, 1}} while generating rank feature value plots"
        )
    known_rank_count = int(
        sum(
            rank is not None and int(label) in {0, 1}
            for rank, label in zip(bundle.sample_ranks, labels.tolist())
        )
    )
    if known_rank_count <= 0:
        raise ValueError(
            "Could not resolve any sample ranks from the feature bundle. "
            "Provide a dataset reference report or model names that encode rank."
        )

    feature_name_set = set(selected_feature_names)
    sample_value_rows: list[dict[str, Any]] = []
    sample_counts_by_rank: dict[int, dict[str, int]] = {}
    n_samples_used = 0
    for row_idx, model_name in enumerate(bundle.model_names):
        rank = bundle.sample_ranks[row_idx]
        label = int(labels[row_idx])
        if rank is None or label not in {0, 1}:
            continue

        owned_feature_names = list(owned_feature_names_by_model.get(str(model_name), []))
        emitted_to_columns: dict[str, list[int]] = {}
        for feature_name in owned_feature_names:
            emitted_feature = _emitted_feature_name(feature_name)
            if emitted_feature not in feature_name_set:
                continue
            emitted_to_columns.setdefault(emitted_feature, []).append(int(root_feature_index[feature_name]))

        row_values = np.asarray(bundle.features[row_idx], dtype=np.float32)
        used_this_sample = False
        for emitted_feature in selected_feature_names:
            column_indices = emitted_to_columns.get(emitted_feature, [])
            if not column_indices:
                continue

            used_this_sample = True
            sample_value_rows.append(
                {
                    "model_name": str(model_name),
                    "label": int(label),
                    "rank": int(rank),
                    "feature_name": str(emitted_feature),
                    "value": float(
                        np.mean(
                            row_values[np.asarray(column_indices, dtype=np.int64)],
                            dtype=np.float64,
                        )
                    ),
                    "source_column_count": int(len(column_indices)),
                }
            )

        if used_this_sample:
            sample_counts_by_rank.setdefault(int(rank), {}).setdefault(str(label), 0)
            sample_counts_by_rank[int(rank)][str(label)] = int(sample_counts_by_rank[int(rank)][str(label)]) + 1
            n_samples_used += 1

    if not sample_value_rows:
        raise ValueError("No per-sample rank feature values were available after filtering labels and ranks")

    ordered_ranks = sorted(sample_counts_by_rank)
    if not ordered_ranks:
        raise ValueError(
            "Could not resolve any sample ranks from the feature bundle. "
            "Provide a dataset reference report or model names that encode rank."
        )

    feature_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in sample_value_rows:
        feature_rows_by_name.setdefault(str(row["feature_name"]), []).append(row)

    ordered_features = [
        feature_name
        for feature_name in selected_feature_names
        if feature_rows_by_name.get(feature_name)
    ]
    dropped_features = [
        feature_name
        for feature_name in selected_feature_names
        if not feature_rows_by_name.get(feature_name)
    ]
    if dropped_features:
        warnings.append(
            "Skipping emitted features with no usable rank/labeled samples: "
            + ", ".join(dropped_features[:8])
        )
    if not ordered_features:
        raise ValueError("No emitted features had usable rows after label/rank filtering")

    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_sample_table_path = (
        _resolve_path(sample_table_path)
        if sample_table_path is not None
        else resolved_output_dir / "rank_feature_values.csv"
    )
    scatter_plot_path = resolved_output_dir / "rank_feature_values_scatter.png"
    box_plot_path = resolved_output_dir / "rank_feature_values_boxplots.png"

    _save_rank_feature_value_csv(resolved_sample_table_path, sample_value_rows)
    _plot_all_rank_feature_value_scatters(
        output_path=scatter_plot_path,
        feature_rows_by_name=feature_rows_by_name,
        ordered_features=ordered_features,
        ordered_ranks=ordered_ranks,
        label_specs=label_specs,
        point_size=point_size,
        alpha=alpha,
        jitter=jitter,
        title="Per-Sample Feature Values by Rank",
    )
    _plot_all_rank_feature_value_boxplots(
        output_path=box_plot_path,
        feature_rows_by_name=feature_rows_by_name,
        ordered_features=ordered_features,
        ordered_ranks=ordered_ranks,
        label_specs=label_specs,
        title="Feature Value Distributions by Rank",
    )

    feature_panels: list[dict[str, Any]] = []
    for feature_name in ordered_features:
        rows = list(feature_rows_by_name.get(feature_name, []))
        label_counts_by_rank: dict[str, dict[str, int]] = {}
        for rank in ordered_ranks:
            rank_counts: dict[str, int] = {}
            for spec in label_specs:
                count = int(
                    sum(
                        int(row["label"]) == int(spec["value"]) and int(row["rank"]) == int(rank)
                        for row in rows
                    )
                )
                if count > 0:
                    rank_counts[str(spec["value"])] = int(count)
            label_counts_by_rank[str(rank)] = rank_counts

        feature_panels.append(
            {
                "feature_name": str(feature_name),
                "n_points": int(len(rows)),
                "ranks": [int(rank) for rank in sorted({int(row["rank"]) for row in rows})],
                "label_counts_by_rank": label_counts_by_rank,
            }
        )

    return {
        "analysis": "rank_feature_values",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_file": str(bundle.feature_file),
        "model_names_file": str(bundle.model_names_file),
        "labels_file": str(bundle.labels_file) if bundle.labels_file is not None else None,
        "metadata_file": str(bundle.metadata_file) if bundle.metadata_file is not None else None,
        "dataset_reference_report": (
            str(bundle.dataset_reference_report) if bundle.dataset_reference_report is not None else None
        ),
        "feature_shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        "requested_features": (
            list(normalized_requested_features) if normalized_requested_features is not None else "all"
        ),
        "plotted_features": list(ordered_features),
        "ranks": [int(rank) for rank in ordered_ranks],
        "label_panels": [str(spec["name"]) for spec in label_specs],
        "n_feature_panels": int(len(ordered_features)),
        "n_samples_total": int(len(bundle.model_names)),
        "n_samples_used": int(n_samples_used),
        "n_points": int(len(sample_value_rows)),
        "sample_table_csv": str(resolved_sample_table_path),
        "scatter_plot": str(scatter_plot_path),
        "box_plot": str(box_plot_path),
        "selected_source_feature_files": [str(path) for path in selected_leaf_paths],
        "label_counts_by_rank": {
            str(rank): {
                str(label): int(count)
                for label, count in sorted(sample_counts_by_rank[int(rank)].items())
            }
            for rank in ordered_ranks
        },
        "feature_panels": feature_panels,
        "warnings": warnings,
    }


def run_unsupervised_rank_feature_values_pipeline(
    *,
    feature_file: Path,
    output_root: Path,
    run_id: str | None,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    requested_features: list[str] | None = None,
    point_size: float = 10.0,
    alpha: float = 0.28,
    jitter: float = 0.08,
) -> dict[str, Any]:
    start = perf_counter()
    ctx = create_run_context(
        pipeline="unsupervised_rank_feature_values",
        output_root=output_root,
        run_id=run_id,
    )

    plot_dir = ctx.plots_dir / "rank_feature_values"
    sample_table_path = ctx.reports_dir / "rank_feature_values.csv"
    report = run_rank_feature_value_analysis(
        feature_file=feature_file,
        output_dir=plot_dir,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
        requested_features=requested_features,
        point_size=point_size,
        alpha=alpha,
        jitter=jitter,
        sample_table_path=sample_table_path,
    )

    elapsed_seconds = float(perf_counter() - start)
    report["elapsed_seconds"] = elapsed_seconds
    report_path = ctx.reports_dir / "rank_feature_values_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("report", report_path)
    ctx.add_artifact("plots", plot_dir)
    ctx.add_artifact("sample_table", sample_table_path)
    ctx.add_timing("elapsed_seconds", elapsed_seconds)
    ctx.finalize(
        {
            "pipeline": "unsupervised_rank_feature_values",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_file": str(feature_file),
            "feature_root": str(feature_root),
            "model_names_file": str(model_names_file) if model_names_file is not None else None,
            "labels_file": str(labels_file) if labels_file is not None else None,
            "metadata_file": str(metadata_file) if metadata_file is not None else None,
            "dataset_reference_report": (
                str(dataset_reference_report) if dataset_reference_report is not None else None
            ),
            "requested_features": list(requested_features) if requested_features is not None else None,
            "point_size": float(point_size),
            "alpha": float(alpha),
            "jitter": float(jitter),
        }
    )

    return {
        "run_dir": ctx.run_dir,
        "report": report_path,
        "plot_dir": plot_dir,
        "sample_table": sample_table_path,
    }


def run_layer_raw_feature_tsne_pipeline(
    *,
    feature_file: Path,
    output_root: Path,
    run_id: str | None,
    folder: str | list[str] | tuple[str, ...],
    layer: int,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    dataset_root: Path | None = None,
    block_filters: list[str] | tuple[str, ...] | None = None,
    features: list[str] | tuple[str, ...] | None = None,
    raw_projection_dim: int = 256,
    projection_seed: int = 42,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
    init: str = "pca",
    random_state: int = 42,
    standardize: bool = True,
    point_size: float = 28.0,
    alpha: float = 0.85,
) -> dict[str, Any]:
    start = perf_counter()
    folders = _normalize_dataset_folders(folder)
    if int(layer) < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if raw_projection_dim <= 0:
        raise ValueError(f"raw_projection_dim must be positive, got {raw_projection_dim}")

    bundle = load_feature_bundle(
        feature_file=feature_file,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
    )
    row_indices, matched_folder_counts = _select_dataset_row_indices(
        bundle=bundle,
        folders=folders,
    )
    if bundle.labels is None:
        raise ValueError("layer raw-feature t-SNE requires sample labels for clean/backdoor coloring")
    selected_labels = np.asarray(bundle.labels[row_indices], dtype=np.int32)
    if np.any(selected_labels < 0):
        raise ValueError("layer raw-feature t-SNE requires known sample labels for every selected row")

    resolved_block_filters = _normalize_block_filters(block_filters)
    feature_column_indices, feature_block_names, selected_feature_names, requested_features = (
        _select_layer_feature_columns(
            bundle=bundle,
            layer=int(layer),
            block_filters=resolved_block_filters,
            features=features,
        )
    )
    selected_model_names = [bundle.model_names[int(i)] for i in row_indices.tolist()]
    selected_ranks = [bundle.sample_ranks[int(i)] for i in row_indices.tolist()]
    selected_dataset_names = [bundle.sample_dataset_names[int(i)] for i in row_indices.tolist()]
    adapter_paths = _resolve_selected_adapter_paths(
        bundle=bundle,
        row_indices=row_indices,
        folders=folders,
        dataset_root=dataset_root,
    )

    schema = load_delta_block_schema(adapter_paths[0])
    raw_block_specs, missing_raw_blocks = _raw_delta_block_specs_for_feature_blocks(
        schema=schema,
        feature_block_names=feature_block_names,
    )
    raw_values = _build_raw_delta_sketch_matrix(
        adapter_paths=adapter_paths,
        raw_block_specs=raw_block_specs,
        raw_projection_dim=int(raw_projection_dim),
        projection_seed=int(projection_seed),
        dtype=np.float32,
    )
    feature_values = np.asarray(
        bundle.features[row_indices][:, feature_column_indices],
        dtype=np.float32,
    )

    raw_embedding, raw_replaced, raw_perplexity, raw_init = _fit_tsne_embedding(
        values=raw_values,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=init,
        random_state=random_state,
        standardize=standardize,
    )
    feature_embedding, feature_replaced, feature_perplexity, feature_init = _fit_tsne_embedding(
        values=feature_values,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        init=init,
        random_state=random_state,
        standardize=standardize,
    )

    ctx = create_run_context(
        pipeline="layer_raw_feature_tsne",
        output_root=output_root,
        run_id=run_id,
    )
    plot_dir = ctx.plots_dir / "layer_raw_feature_tsne"
    embedding_dir = ctx.reports_dir / "layer_raw_feature_tsne"
    folder_slug = _folder_selection_slug(folders)
    folder_label = _folder_selection_label(folders)
    slug = f"{folder_slug}__layer_{int(layer):03d}"
    comparison_plot = plot_dir / f"{slug}__comparison.png"
    raw_plot = plot_dir / f"{slug}__raw_ba_tsne.png"
    feature_plot = plot_dir / f"{slug}__feature_tsne.png"
    raw_csv = embedding_dir / f"{slug}__raw_ba_tsne.csv"
    feature_csv = embedding_dir / f"{slug}__feature_tsne.csv"

    title_suffix = f"{folder_label}, layer={int(layer)}"
    if resolved_block_filters:
        title_suffix += ", filters=" + ",".join(resolved_block_filters)
    _plot_layer_raw_feature_tsne_comparison(
        output_path=comparison_plot,
        raw_embedding=raw_embedding,
        feature_embedding=feature_embedding,
        labels=selected_labels,
        title=f"Raw B@A vs extracted-feature t-SNE ({title_suffix})",
        point_size=point_size,
        alpha=alpha,
    )
    _plot_tsne_embedding(
        output_path=raw_plot,
        embedding=raw_embedding,
        labels=selected_labels,
        title=f"Raw B@A sketch t-SNE ({title_suffix})",
        point_size=point_size,
        alpha=alpha,
    )
    _plot_tsne_embedding(
        output_path=feature_plot,
        embedding=feature_embedding,
        labels=selected_labels,
        title=f"Extracted-feature t-SNE ({title_suffix})",
        point_size=point_size,
        alpha=alpha,
    )
    _save_tsne_embedding_csv(
        output_path=raw_csv,
        group_label="raw_ba",
        model_names=selected_model_names,
        labels=selected_labels,
        ranks=selected_ranks,
        dataset_names=selected_dataset_names,
        embedding=raw_embedding,
    )
    _save_tsne_embedding_csv(
        output_path=feature_csv,
        group_label="features",
        model_names=selected_model_names,
        labels=selected_labels,
        ranks=selected_ranks,
        dataset_names=selected_dataset_names,
        embedding=feature_embedding,
    )

    warnings: list[str] = []
    if raw_replaced > 0:
        warnings.append(f"Replaced {raw_replaced} non-finite raw B@A sketch values before t-SNE")
    if feature_replaced > 0:
        warnings.append(f"Replaced {feature_replaced} non-finite extracted feature values before t-SNE")
    if missing_raw_blocks:
        warnings.append(
            "No raw B@A block was available for selected feature block(s): "
            + ", ".join(missing_raw_blocks[:8])
        )

    elapsed_seconds = float(perf_counter() - start)
    report = {
        "analysis": "layer_raw_feature_tsne",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "feature_file": str(bundle.feature_file),
        "model_names_file": str(bundle.model_names_file),
        "labels_file": str(bundle.labels_file) if bundle.labels_file is not None else None,
        "metadata_file": str(bundle.metadata_file) if bundle.metadata_file is not None else None,
        "dataset_reference_report": (
            str(bundle.dataset_reference_report) if bundle.dataset_reference_report is not None else None
        ),
        "folder": folders[0] if len(folders) == 1 else list(folders),
        "folders": list(folders),
        "matched_folder_counts": matched_folder_counts,
        "layer": int(layer),
        "block_filters": resolved_block_filters,
        "requested_features": "all" if requested_features is None else list(requested_features),
        "n_samples": int(row_indices.size),
        "label_counts": _label_count_summary(selected_labels),
        "selected_feature_columns": int(feature_column_indices.size),
        "selected_feature_blocks": feature_block_names,
        "selected_feature_names": selected_feature_names,
        "raw_blocks": [
            {
                "kind": spec.kind,
                "block_name": spec.block_name,
                "raw_block_name": spec.raw_block_name,
            }
            for spec in raw_block_specs
        ],
        "raw_projection_dim_per_block": int(raw_projection_dim),
        "raw_projection_seed": int(projection_seed),
        "raw_input_shape": [int(raw_values.shape[0]), int(raw_values.shape[1])],
        "feature_input_shape": [int(feature_values.shape[0]), int(feature_values.shape[1])],
        "perplexity": float(perplexity),
        "raw_resolved_perplexity": float(raw_perplexity),
        "feature_resolved_perplexity": float(feature_perplexity),
        "learning_rate": learning_rate,
        "max_iter": int(max_iter),
        "metric": metric,
        "init": init,
        "raw_resolved_init": raw_init,
        "feature_resolved_init": feature_init,
        "random_state": int(random_state),
        "standardize": bool(standardize),
        "artifacts": {
            "comparison_plot": str(comparison_plot),
            "raw_plot": str(raw_plot),
            "feature_plot": str(feature_plot),
            "raw_embedding_csv": str(raw_csv),
            "feature_embedding_csv": str(feature_csv),
            "plot_dir": str(plot_dir),
            "embedding_dir": str(embedding_dir),
        },
        "warnings": warnings,
    }

    report_path = ctx.reports_dir / "layer_raw_feature_tsne_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("report", report_path)
    ctx.add_artifact("comparison_plot", comparison_plot)
    ctx.add_artifact("raw_plot", raw_plot)
    ctx.add_artifact("feature_plot", feature_plot)
    ctx.add_artifact("raw_embedding_csv", raw_csv)
    ctx.add_artifact("feature_embedding_csv", feature_csv)
    ctx.add_artifact("plots", plot_dir)
    ctx.add_artifact("embeddings", embedding_dir)
    ctx.add_timing("elapsed_seconds", elapsed_seconds)
    ctx.finalize(
        {
            "pipeline": "layer_raw_feature_tsne",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_file": str(feature_file),
            "feature_root": str(feature_root),
            "model_names_file": str(model_names_file) if model_names_file is not None else None,
            "labels_file": str(labels_file) if labels_file is not None else None,
            "metadata_file": str(metadata_file) if metadata_file is not None else None,
            "dataset_reference_report": (
                str(dataset_reference_report) if dataset_reference_report is not None else None
            ),
            "dataset_root": str(dataset_root) if dataset_root is not None else None,
            "folder": folders[0] if len(folders) == 1 else list(folders),
            "folders": list(folders),
            "layer": int(layer),
            "block_filters": resolved_block_filters,
            "features": None if features is None else [str(x) for x in features],
            "raw_projection_dim": int(raw_projection_dim),
            "projection_seed": int(projection_seed),
            "perplexity": float(perplexity),
            "learning_rate": learning_rate,
            "max_iter": int(max_iter),
            "metric": metric,
            "init": init,
            "random_state": int(random_state),
            "standardize": bool(standardize),
            "point_size": float(point_size),
            "alpha": float(alpha),
        }
    )

    return {
        "run_dir": ctx.run_dir,
        "report": report_path,
        "plot_dir": plot_dir,
        "embedding_dir": embedding_dir,
    }


def run_unsupervised_tsne_pipeline(
    *,
    feature_file: Path,
    output_root: Path,
    run_id: str | None,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    over: str = "rank",
    view: str = "full",
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
    init: str = "pca",
    random_state: int = 42,
    perplexities: list[float] | None = None,
    learning_rates: list[str | float] | None = None,
    max_iters: list[int] | None = None,
    metrics: list[str] | None = None,
    inits: list[str] | None = None,
    random_states: list[int] | None = None,
    standardize: bool = True,
    point_size: float = 28.0,
    alpha: float = 0.85,
) -> dict[str, Any]:
    start = perf_counter()
    resolved_view = _normalize_tsne_view(view)
    ctx = create_run_context(
        pipeline="unsupervised_tsne",
        output_root=output_root,
        run_id=run_id,
    )

    plot_dir = ctx.plots_dir / f"tsne_over_{_safe_slug(over)}_{_safe_slug(resolved_view)}"
    embedding_dir = ctx.reports_dir / f"tsne_over_{_safe_slug(over)}_{_safe_slug(resolved_view)}"
    report = run_grouped_tsne_sweep_analysis(
        feature_file=feature_file,
        output_dir=plot_dir,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
        over=over,
        view=resolved_view,
        perplexities=[float(perplexity)] if perplexities is None else list(perplexities),
        learning_rates=[learning_rate] if learning_rates is None else list(learning_rates),
        max_iters=[int(max_iter)] if max_iters is None else list(max_iters),
        metrics=[metric] if metrics is None else list(metrics),
        inits=[init] if inits is None else list(inits),
        random_states=[int(random_state)] if random_states is None else list(random_states),
        standardize=standardize,
        point_size=point_size,
        alpha=alpha,
        embedding_csv_dir=embedding_dir,
    )

    elapsed_seconds = float(perf_counter() - start)
    report["elapsed_seconds"] = elapsed_seconds
    report_path = ctx.reports_dir / "tsne_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("report", report_path)
    ctx.add_artifact("plots", plot_dir)
    ctx.add_artifact("embeddings", embedding_dir)
    ctx.add_timing("elapsed_seconds", elapsed_seconds)
    ctx.finalize(
        {
            "pipeline": "unsupervised_tsne",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_file": str(feature_file),
            "feature_root": str(feature_root),
            "model_names_file": str(model_names_file) if model_names_file is not None else None,
            "labels_file": str(labels_file) if labels_file is not None else None,
            "metadata_file": str(metadata_file) if metadata_file is not None else None,
            "dataset_reference_report": (
                str(dataset_reference_report) if dataset_reference_report is not None else None
            ),
            "over": over,
            "view": resolved_view,
            "perplexity": float(perplexity),
            "learning_rate": learning_rate,
            "max_iter": int(max_iter),
            "metric": metric,
            "init": init,
            "random_state": int(random_state),
            "perplexities": None if perplexities is None else [float(x) for x in perplexities],
            "learning_rates": None if learning_rates is None else list(learning_rates),
            "max_iters": None if max_iters is None else [int(x) for x in max_iters],
            "metrics": None if metrics is None else [str(x) for x in metrics],
            "inits": None if inits is None else [str(x) for x in inits],
            "random_states": None if random_states is None else [int(x) for x in random_states],
            "standardize": bool(standardize),
            "point_size": float(point_size),
            "alpha": float(alpha),
        }
    )

    return {
        "run_dir": ctx.run_dir,
        "report": report_path,
        "plot_dir": plot_dir,
        "embedding_dir": embedding_dir,
    }


def run_unsupervised_layer_scatter_pipeline(
    *,
    feature_file: Path,
    output_root: Path,
    run_id: str | None,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    model_names_file: Path | None = None,
    labels_file: Path | None = None,
    metadata_file: Path | None = None,
    dataset_reference_report: Path | None = None,
    point_size: float = 6.0,
    alpha: float = 0.18,
) -> dict[str, Any]:
    start = perf_counter()
    ctx = create_run_context(
        pipeline="unsupervised_layer_scatter",
        output_root=output_root,
        run_id=run_id,
    )

    plot_dir = ctx.plots_dir / "layer_value_scatter"
    report = run_layer_value_scatter_analysis(
        feature_file=feature_file,
        output_dir=plot_dir,
        feature_root=feature_root,
        model_names_file=model_names_file,
        labels_file=labels_file,
        metadata_file=metadata_file,
        dataset_reference_report=dataset_reference_report,
        point_size=point_size,
        alpha=alpha,
    )

    elapsed_seconds = float(perf_counter() - start)
    report["elapsed_seconds"] = elapsed_seconds
    report_path = ctx.reports_dir / "layer_value_scatter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("report", report_path)
    ctx.add_artifact("plots", plot_dir)
    ctx.add_timing("elapsed_seconds", elapsed_seconds)
    ctx.finalize(
        {
            "pipeline": "unsupervised_layer_scatter",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_file": str(feature_file),
            "feature_root": str(feature_root),
            "model_names_file": str(model_names_file) if model_names_file is not None else None,
            "labels_file": str(labels_file) if labels_file is not None else None,
            "metadata_file": str(metadata_file) if metadata_file is not None else None,
            "dataset_reference_report": (
                str(dataset_reference_report) if dataset_reference_report is not None else None
            ),
            "point_size": float(point_size),
            "alpha": float(alpha),
        }
    )

    return {
        "run_dir": ctx.run_dir,
        "report": report_path,
        "plot_dir": plot_dir,
    }
