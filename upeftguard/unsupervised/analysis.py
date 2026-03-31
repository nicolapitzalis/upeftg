from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ..utilities.artifacts.dataset_references import (
    DATASET_REFERENCE_REPORT_NAME,
    load_dataset_reference_report,
    resolve_dataset_reference_payload_for_artifact,
)
from ..utilities.artifacts.spectral_metadata import load_spectral_metadata
from ..utilities.core.run_context import create_run_context
from ..utilities.core.serialization import json_ready


SCRIPT_VERSION = "1.1.0"
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
