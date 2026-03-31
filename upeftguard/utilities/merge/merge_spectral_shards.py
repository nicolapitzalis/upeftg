from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from ...features.spectral import (
    ordered_block_names_from_feature_names,
    spectral_block_lora_dims_by_block,
    sanitize_spectral_metadata,
)
from ..artifacts.dataset_references import (
    build_dataset_reference_payload_from_items,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from ..artifacts.spectral_metadata import (
    dataset_layouts_from_source,
    load_spectral_metadata,
    merge_dataset_layouts,
    write_spectral_metadata,
)
from ..core.manifest import parse_single_manifest_json, resolve_manifest_path
from ..core.paths import default_dataset_root, dataset_root_help
from ..core.serialization import json_ready


@dataclass(frozen=True)
class SpectralShardBundle:
    run_dir: Path
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    feature_names: list[str]
    feature_names_inferred: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SpectralFeatureTable:
    source: str
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    feature_names: list[str]
    feature_names_inferred: bool
    metadata: dict[str, Any]


class FeatureTableLike(Protocol):
    source: str
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    feature_names: list[str]
    feature_names_inferred: bool


@dataclass(frozen=True)
class ZeroFillMergeResult:
    model_names: list[str]
    feature_names: list[str]
    features: np.ndarray
    labels: np.ndarray | None
    feature_names_inferred: bool
    stats: dict[str, Any]


def merge_lora_dim_maps(*metadata_payloads: dict[str, Any]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for payload in metadata_payloads:
        for block_name, dims in spectral_block_lora_dims_by_block(payload).items():
            existing = merged.get(block_name)
            if existing is not None and existing != dims:
                raise ValueError(
                    f"Conflicting LoRA dimensions for block '{block_name}': {existing} vs {dims}"
                )
            merged[block_name] = dict(dims)
    return merged


def resolved_qv_sum_mode(block_names: list[str]) -> str:
    has_qv_sum = any(".qv_sum" in name for name in block_names)
    has_base = any(".qv_sum" not in name for name in block_names)
    if has_qv_sum and has_base:
        return "append"
    if has_qv_sum:
        return "only"
    return "none"


def merge_skipped_models(*metadata_payloads: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for payload in metadata_payloads:
        raw_entries = payload.get("skipped_models")
        if not isinstance(raw_entries, list):
            continue
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            model_name = str(raw_entry.get("model_name") or "").strip()
            if not model_name:
                continue
            merged[model_name] = {
                "model_name": model_name,
                "adapter_path": str(raw_entry.get("adapter_path") or ""),
                "label": raw_entry.get("label"),
                "exception_type": str(raw_entry.get("exception_type") or ""),
                "exception_message": str(raw_entry.get("exception_message") or ""),
            }
    return [merged[name] for name in sorted(merged)]


def _resolved_feature_name_views(
    *,
    base: FeatureTableLike,
    incoming: FeatureTableLike,
) -> tuple[list[str], list[str], bool]:
    used_positional_feature_names = bool(base.feature_names_inferred or incoming.feature_names_inferred)
    if not used_positional_feature_names:
        return list(base.feature_names), list(incoming.feature_names), False

    if len(base.feature_names) != len(incoming.feature_names):
        raise ValueError(
            "Cannot merge feature schemas with inferred positional names when dimensions differ. "
            "Provide metadata with explicit feature_names for both sources."
        )
    positional = [f"feature_{i:05d}" for i in range(len(base.feature_names))]
    return positional, positional, True


def _merge_feature_name_union(
    *,
    base_feature_names: list[str],
    incoming_feature_names: list[str],
) -> list[str]:
    merged_feature_names = list(base_feature_names)
    base_feature_name_set = set(base_feature_names)
    merged_feature_names.extend(name for name in incoming_feature_names if name not in base_feature_name_set)
    return merged_feature_names


def _place_table_values(
    *,
    merged_features: np.ndarray,
    coverage: np.ndarray,
    merged_row_index: dict[str, int],
    merged_col_index: dict[str, int],
    table: FeatureTableLike,
    table_feature_names: list[str],
) -> None:
    row_idx = np.asarray([merged_row_index[name] for name in table.model_names], dtype=np.int64)
    col_idx = np.asarray([merged_col_index[name] for name in table_feature_names], dtype=np.int64)
    table_block = np.asarray(table.features, dtype=np.float32)
    current = merged_features[np.ix_(row_idx, col_idx)]
    seen = coverage[np.ix_(row_idx, col_idx)]

    if np.any(seen):
        conflict = seen & (~np.isclose(current, table_block, rtol=1e-5, atol=1e-6, equal_nan=True))
        if np.any(conflict):
            conflict_idx = np.argwhere(conflict)[0]
            i = int(conflict_idx[0])
            j = int(conflict_idx[1])
            row_name = table.model_names[i]
            feature_name = table_feature_names[j]
            raise ValueError(
                "Conflicting feature values for overlapping row/feature cell: "
                f"model='{row_name}', feature='{feature_name}', "
                f"existing={float(current[i, j])}, incoming={float(table_block[i, j])}"
            )

    merged_features[np.ix_(row_idx, col_idx)] = np.where(seen, current, table_block)
    coverage[np.ix_(row_idx, col_idx)] = True


def _merge_labels_for_dense_rows(
    *,
    base: FeatureTableLike,
    incoming: FeatureTableLike,
    merged_row_index: dict[str, int],
) -> np.ndarray | None:
    labels_unknown = np.iinfo(np.int32).min
    merged_label_values = np.full(len(merged_row_index), labels_unknown, dtype=np.int32)
    merged_label_known = np.zeros(len(merged_row_index), dtype=bool)

    def _place_labels(table: FeatureTableLike) -> None:
        if table.labels is None:
            return
        row_idx = np.asarray([merged_row_index[name] for name in table.model_names], dtype=np.int64)
        vals = np.asarray(table.labels, dtype=np.int32)
        known = merged_label_known[row_idx]
        existing_vals = merged_label_values[row_idx]

        conflict = known & (existing_vals != vals)
        if np.any(conflict):
            i = int(np.argwhere(conflict)[0, 0])
            raise ValueError(
                "Conflicting labels for overlapping model: "
                f"model='{table.model_names[i]}', existing={int(existing_vals[i])}, incoming={int(vals[i])}"
            )

        write_vals = existing_vals.copy()
        write_vals[~known] = vals[~known]
        merged_label_values[row_idx] = write_vals
        merged_label_known[row_idx] = True

    _place_labels(base)
    _place_labels(incoming)
    return merged_label_values if bool(np.all(merged_label_known)) else None


def merge_disjoint_feature_tables_zero_fill(
    *,
    base: FeatureTableLike,
    incoming: FeatureTableLike,
    index_by_name,
    overlap_error_prefix: str,
) -> ZeroFillMergeResult:
    base_feature_names, incoming_feature_names, used_positional_feature_names = _resolved_feature_name_views(
        base=base,
        incoming=incoming,
    )

    base_row_index = index_by_name(base.model_names, context=base.source, entity="model names")
    incoming_row_index = index_by_name(
        incoming.model_names,
        context=incoming.source,
        entity="model names",
    )
    index_by_name(base_feature_names, context=f"{base.source} feature schema", entity="feature names")
    index_by_name(
        incoming_feature_names,
        context=f"{incoming.source} feature schema", entity="feature names"
    )

    overlap_rows = sorted(set(base_row_index) & set(incoming_row_index))
    if overlap_rows:
        preview = ", ".join(overlap_rows[:5])
        raise ValueError(f"{overlap_error_prefix}. Examples: {preview}")

    merged_model_names = list(base.model_names)
    merged_model_names.extend(incoming.model_names)
    merged_feature_names = _merge_feature_name_union(
        base_feature_names=base_feature_names,
        incoming_feature_names=incoming_feature_names,
    )

    merged_row_index = index_by_name(merged_model_names, context="merged output", entity="model names")
    merged_col_index = index_by_name(merged_feature_names, context="merged output", entity="feature names")

    merged_features = np.zeros((len(merged_model_names), len(merged_feature_names)), dtype=np.float32)
    coverage = np.zeros_like(merged_features, dtype=bool)

    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=base,
        table_feature_names=base_feature_names,
    )
    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=incoming,
        table_feature_names=incoming_feature_names,
    )

    merged_label_values = np.empty(len(merged_model_names), dtype=np.int32)
    labels_complete = base.labels is not None and incoming.labels is not None
    if labels_complete:
        merged_label_values[: len(base.model_names)] = np.asarray(base.labels, dtype=np.int32)
        merged_label_values[len(base.model_names) :] = np.asarray(incoming.labels, dtype=np.int32)
        merged_labels: np.ndarray | None = merged_label_values
    else:
        merged_labels = None

    overlap_features = sorted(set(base_feature_names) & set(incoming_feature_names))
    zero_filled_cells = int(merged_features.size - int(np.count_nonzero(coverage)))
    stats = {
        "merge_mode": "zero_fill_disjoint_rows",
        "base_rows": int(len(base.model_names)),
        "base_feature_dim": int(len(base_feature_names)),
        "incoming_rows": int(len(incoming.model_names)),
        "incoming_feature_dim": int(len(incoming_feature_names)),
        "merged_rows": int(len(merged_model_names)),
        "merged_feature_dim": int(len(merged_feature_names)),
        "rows_added": int(len(incoming.model_names)),
        "features_added": int(len(merged_feature_names) - len(base_feature_names)),
        "row_overlap": 0,
        "feature_overlap": int(len(overlap_features)),
        "labels_complete": bool(merged_labels is not None),
        "zero_filled_cells": zero_filled_cells,
    }
    return ZeroFillMergeResult(
        model_names=merged_model_names,
        feature_names=merged_feature_names,
        features=merged_features,
        labels=merged_labels,
        feature_names_inferred=used_positional_feature_names,
        stats=stats,
    )


def _unique_index_by_name(names: list[str], *, context: str, entity: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for i, name in enumerate(names):
        if name in index:
            duplicates.append(name)
            continue
        index[name] = int(i)
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"Duplicate {entity} in {context}; cannot align merged data safely. "
            f"Examples: {preview}"
        )
    return index


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_spectral_metadata(path)
    return dict(payload) if isinstance(payload, dict) else {}


def _parse_timestamp_utc(raw: Any) -> datetime | None:
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _valid_elapsed_seconds(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


def _summarize_shard_timings(shard_run_dirs: list[Path]) -> tuple[dict[str, Any], datetime | None]:
    elapsed_values: list[float] = []
    derived_start_times: list[float] = []
    completion_times: list[datetime] = []

    for run_dir in shard_run_dirs:
        report = _load_metadata(run_dir / "reports" / "feature_extraction_report.json")
        elapsed_seconds = _valid_elapsed_seconds(report.get("elapsed_seconds"))
        completed_at = _parse_timestamp_utc(report.get("timestamp_utc"))

        if elapsed_seconds is not None:
            elapsed_values.append(elapsed_seconds)
        if completed_at is not None:
            completion_times.append(completed_at)
        if elapsed_seconds is not None and completed_at is not None:
            derived_start_times.append(completed_at.timestamp() - elapsed_seconds)

    earliest_start = (
        datetime.fromtimestamp(min(derived_start_times), tz=timezone.utc) if derived_start_times else None
    )
    latest_end = max(completion_times) if completion_times else None

    summary: dict[str, Any] = {
        "reported_elapsed_count": int(len(elapsed_values)),
        "missing_elapsed_count": int(len(shard_run_dirs) - len(elapsed_values)),
        "earliest_shard_start_timestamp_utc": earliest_start.isoformat() if earliest_start is not None else None,
        "latest_shard_end_timestamp_utc": latest_end.isoformat() if latest_end is not None else None,
    }
    if elapsed_values:
        summary["elapsed_seconds_sum"] = float(sum(elapsed_values))
        summary["elapsed_seconds_mean"] = float(sum(elapsed_values) / len(elapsed_values))
        summary["elapsed_seconds_min"] = float(min(elapsed_values))
        summary["elapsed_seconds_max"] = float(max(elapsed_values))
    else:
        summary["elapsed_seconds_sum"] = None
        summary["elapsed_seconds_mean"] = None
        summary["elapsed_seconds_min"] = None
        summary["elapsed_seconds_max"] = None

    return summary, earliest_start


def _resolve_feature_names(
    *,
    metadata: dict[str, Any],
    feature_dim: int,
    context: str,
) -> tuple[list[str], bool]:
    raw = metadata.get("feature_names")
    if isinstance(raw, list):
        names = [str(x) for x in raw]
        if len(names) != int(feature_dim):
            raise ValueError(
                f"feature_names length mismatch in {context}: "
                f"metadata has {len(names)}, matrix has {feature_dim} columns"
            )
        _unique_index_by_name(names, context=context, entity="feature names")
        return names, False

    # Fallback for legacy metadata: positional names keep row-appends possible.
    inferred = [f"feature_{i:05d}" for i in range(int(feature_dim))]
    return inferred, True


def _rebuild_output_metadata(
    *,
    source_metadata: dict[str, Any],
    output_table: SpectralFeatureTable,
    shard_bundles: list[SpectralShardBundle],
    metadata_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = sanitize_spectral_metadata(source_metadata)
    metadata["n_models"] = int(len(output_table.model_names))
    metadata["feature_dim"] = int(len(output_table.feature_names))
    metadata["feature_names"] = list(output_table.feature_names)
    metadata["merge_source_shards"] = [str(bundle.run_dir) for bundle in shard_bundles]
    metadata["merged_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    skipped_models = merge_skipped_models(*metadata_sources)
    if skipped_models:
        metadata["skipped_model_count"] = int(len(skipped_models))
        metadata["skipped_models"] = skipped_models

    if not output_table.feature_names_inferred:
        try:
            block_names = ordered_block_names_from_feature_names(output_table.feature_names)
        except ValueError:
            block_names = []
        if block_names:
            metadata["block_names"] = list(block_names)
            metadata["n_blocks"] = int(len(block_names))
            metadata["base_block_names"] = [name for name in block_names if ".qv_sum" not in name]
            metadata["qv_sum_block_names"] = [name for name in block_names if ".qv_sum" in name]
            metadata["spectral_qv_sum_mode"] = resolved_qv_sum_mode(block_names)

            dim_map = merge_lora_dim_maps(*metadata_sources)
            if all(block_name in dim_map for block_name in block_names):
                metadata["lora_adapter_dims"] = [dim_map[block_name] for block_name in block_names]
            else:
                metadata.pop("lora_adapter_dims", None)

    return metadata


def _load_feature_table(
    *,
    source: str,
    feature_path: Path,
    model_names_path: Path,
    labels_path: Path,
    metadata_path: Path,
    context: str,
) -> SpectralFeatureTable:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing spectral feature file for {context}: {feature_path}")
    if not model_names_path.exists():
        raise FileNotFoundError(f"Missing spectral model names file for {context}: {model_names_path}")

    features = np.asarray(np.load(feature_path), dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Spectral features must be 2D in {context}: shape={features.shape}")

    with open(model_names_path, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(features.shape[0]):
        raise ValueError(
            f"Row mismatch in {context}: features rows={features.shape[0]} "
            f"but model names={len(model_names)}"
        )
    _unique_index_by_name(model_names, context=context, entity="model names")

    labels = None
    if labels_path.exists():
        labels = np.asarray(np.load(labels_path), dtype=np.int32)
        if int(labels.shape[0]) != int(features.shape[0]):
            raise ValueError(
                f"Label mismatch in {context}: labels rows={labels.shape[0]} "
                f"but features rows={features.shape[0]}"
            )

    metadata = _load_metadata(metadata_path)
    feature_names, feature_names_inferred = _resolve_feature_names(
        metadata=metadata,
        feature_dim=int(features.shape[1]),
        context=context,
    )
    return SpectralFeatureTable(
        source=source,
        features=features,
        labels=labels,
        model_names=model_names,
        feature_names=feature_names,
        feature_names_inferred=feature_names_inferred,
        metadata=metadata,
    )


def _load_shard_bundle(run_dir: Path) -> SpectralShardBundle:
    table = _load_feature_table(
        source=str(run_dir),
        feature_path=run_dir / "features" / "spectral_features.npy",
        model_names_path=run_dir / "features" / "spectral_model_names.json",
        labels_path=run_dir / "features" / "spectral_labels.npy",
        metadata_path=run_dir / "features" / "spectral_metadata.json",
        context=f"shard run {run_dir}",
    )
    return SpectralShardBundle(
        run_dir=run_dir,
        features=table.features,
        labels=table.labels,
        model_names=table.model_names,
        feature_names=table.feature_names,
        feature_names_inferred=bool(table.feature_names_inferred),
        metadata=table.metadata,
    )


def _compatibility_signature(bundle: SpectralShardBundle) -> dict[str, Any]:
    keys = [
        "extractor",
        "resolved_features",
        "spectral_moment_source",
        "spectral_qv_sum_mode",
        "sv_top_k",
        "feature_dim",
        "block_names",
        "delta_schema_version",
    ]
    sig = {key: bundle.metadata.get(key) for key in keys}
    sig["feature_dim"] = int(bundle.features.shape[1])
    sig["feature_names"] = list(bundle.feature_names)
    return sig


def _load_expected_manifest_items(
    *,
    manifest_json: Path,
    dataset_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=dataset_root,
        section_key="path",
    )
    expected_names = [item.model_name for item in items]
    _unique_index_by_name(
        expected_names,
        context=f"{manifest_json}::path",
        entity="model names",
    )
    expected_items_by_name = {item.model_name: item for item in items}
    return expected_names, expected_items_by_name


def _validate_shard_bundle_compatibility(bundles: list[SpectralShardBundle]) -> bool:
    labels_complete = all(bundle.labels is not None for bundle in bundles)
    if any(bundle.labels is not None for bundle in bundles) and not labels_complete:
        raise ValueError("Either all shards must include labels or none of them should include labels")

    base_signature = _compatibility_signature(bundles[0])
    for bundle in bundles[1:]:
        sig = _compatibility_signature(bundle)
        if sig != base_signature:
            raise ValueError(
                "Incompatible shard metadata; all shards must use the same spectral feature schema"
            )
    return labels_complete


def _stack_shard_bundle_rows(
    *,
    bundles: list[SpectralShardBundle],
    labels_complete: bool,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    merged_features = np.vstack([bundle.features for bundle in bundles])
    merged_names = [name for bundle in bundles for name in bundle.model_names]
    merged_labels = (
        np.concatenate([bundle.labels for bundle in bundles if bundle.labels is not None], axis=0)
        if labels_complete
        else None
    )
    return merged_features, merged_names, merged_labels


def _resolve_missing_shard_models(
    *,
    bundles: list[SpectralShardBundle],
    missing_names: list[str],
    expected_items_by_name: dict[str, Any],
) -> list[dict[str, Any]]:
    skipped_by_name = {
        str(entry["model_name"]): dict(entry)
        for entry in merge_skipped_models(*[dict(bundle.metadata) for bundle in bundles])
    }
    for model_name in missing_names:
        item = expected_items_by_name.get(model_name)
        if item is None or model_name in skipped_by_name:
            continue
        skipped_by_name[model_name] = {
            "model_name": str(model_name),
            "adapter_path": str(item.adapter_path),
            "label": int(item.label) if item.label is not None else None,
            "exception_type": "MissingShardRow",
            "exception_message": "Model was not present in merged shard outputs",
        }
    return [skipped_by_name[name] for name in sorted(skipped_by_name)]


def _build_incoming_table_from_shard_bundles(
    *,
    bundles: list[SpectralShardBundle],
    expected_names: list[str],
    expected_items_by_name: dict[str, Any],
    manifest_context: str,
) -> tuple[SpectralFeatureTable, list[dict[str, Any]]]:
    labels_complete = _validate_shard_bundle_compatibility(bundles)
    merged_features, merged_names, merged_labels = _stack_shard_bundle_rows(
        bundles=bundles,
        labels_complete=labels_complete,
    )

    expected_index = _unique_index_by_name(
        expected_names,
        context=manifest_context,
        entity="model names",
    )
    merged_index = _unique_index_by_name(
        merged_names,
        context="merged shard model names",
        entity="model names",
    )
    extra = sorted(name for name in merged_index if name not in expected_index)
    if extra:
        raise ValueError(
            "Merged shard model names do not match manifest model names: "
            + "; ".join([f"extra={extra[:5]}"])
        )

    resolved_names = [name for name in expected_names if name in merged_index]
    missing_names = [name for name in expected_names if name not in merged_index]
    reorder = np.asarray([merged_index[name] for name in resolved_names], dtype=np.int64)
    reordered_features = merged_features[reorder]
    reordered_labels = merged_labels[reorder] if merged_labels is not None else None
    skipped_models = _resolve_missing_shard_models(
        bundles=bundles,
        missing_names=missing_names,
        expected_items_by_name=expected_items_by_name,
    )

    incoming_metadata = {
        **dict(bundles[0].metadata),
        "input_n_models": int(len(expected_names)),
        "n_models": int(len(resolved_names)),
        "feature_dim": int(reordered_features.shape[1]),
        "feature_names": list(bundles[0].feature_names),
        "merge_source_shards": [str(bundle.run_dir) for bundle in bundles],
        "merged_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if skipped_models:
        incoming_metadata["skipped_model_count"] = int(len(skipped_models))
        incoming_metadata["skipped_models"] = skipped_models
    return (
        SpectralFeatureTable(
            source="incoming_shards",
            features=np.asarray(reordered_features, dtype=np.float32),
            labels=(np.asarray(reordered_labels, dtype=np.int32) if reordered_labels is not None else None),
            model_names=list(resolved_names),
            feature_names=list(bundles[0].feature_names),
            feature_names_inferred=bool(bundles[0].feature_names_inferred),
            metadata=incoming_metadata,
        ),
        skipped_models,
    )


def _write_output_table_artifacts(
    *,
    output_dir: Path,
    output_table: SpectralFeatureTable,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "spectral_features.npy"
    model_names_path = output_dir / "spectral_model_names.json"
    labels_path = output_dir / "spectral_labels.npy"
    metadata_path = output_dir / "spectral_metadata.json"
    merge_report_path = output_dir / "spectral_merge_report.json"
    dataset_reference_report_path = default_dataset_reference_report_path(output_dir)

    np.save(feature_path, output_table.features.astype(np.float32, copy=False))
    if output_table.labels is not None:
        np.save(labels_path, output_table.labels.astype(np.int32, copy=False))
    elif labels_path.exists():
        labels_path.unlink()

    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump(output_table.model_names, f, indent=2)

    return {
        "feature_path": feature_path,
        "model_names_path": model_names_path,
        "labels_path": labels_path,
        "metadata_path": metadata_path,
        "merge_report_path": merge_report_path,
        "dataset_reference_report_path": dataset_reference_report_path,
    }


def _pipeline_timing_fields(
    *,
    merge_completed_at: datetime,
    pipeline_start_epoch_seconds: float | None,
    derived_pipeline_start: datetime | None,
) -> dict[str, Any]:
    payload = {
        "pipeline_start_timestamp_utc": None,
        "pipeline_start_epoch_seconds": (
            float(pipeline_start_epoch_seconds) if pipeline_start_epoch_seconds is not None else None
        ),
        "pipeline_start_source": None,
        "pipeline_elapsed_seconds": None,
    }
    if pipeline_start_epoch_seconds is not None:
        pipeline_start = datetime.fromtimestamp(float(pipeline_start_epoch_seconds), tz=timezone.utc)
        payload["pipeline_start_timestamp_utc"] = pipeline_start.isoformat()
        payload["pipeline_start_source"] = "launcher"
        payload["pipeline_elapsed_seconds"] = float((merge_completed_at - pipeline_start).total_seconds())
    elif derived_pipeline_start is not None:
        payload["pipeline_start_timestamp_utc"] = derived_pipeline_start.isoformat()
        payload["pipeline_start_source"] = "derived_from_shard_reports"
        payload["pipeline_elapsed_seconds"] = float((merge_completed_at - derived_pipeline_start).total_seconds())
    return payload


def _merge_feature_tables(
    *,
    base: SpectralFeatureTable,
    incoming: SpectralFeatureTable,
) -> tuple[SpectralFeatureTable, dict[str, Any]]:
    base_feature_names, incoming_feature_names, used_positional_feature_names = _resolved_feature_name_views(
        base=base,
        incoming=incoming,
    )

    base_row_index = _unique_index_by_name(base.model_names, context=base.source, entity="model names")
    _unique_index_by_name(incoming.model_names, context=incoming.source, entity="model names")
    _unique_index_by_name(base_feature_names, context=f"{base.source} feature schema", entity="feature names")
    _unique_index_by_name(incoming_feature_names, context=f"{incoming.source} feature schema", entity="feature names")

    merged_model_names = list(base.model_names)
    merged_model_names.extend(name for name in incoming.model_names if name not in base_row_index)
    merged_feature_names = _merge_feature_name_union(
        base_feature_names=base_feature_names,
        incoming_feature_names=incoming_feature_names,
    )

    merged_row_index = _unique_index_by_name(merged_model_names, context="merged output", entity="model names")
    merged_col_index = _unique_index_by_name(merged_feature_names, context="merged output", entity="feature names")

    merged_features = np.zeros((len(merged_model_names), len(merged_feature_names)), dtype=np.float32)
    coverage = np.zeros_like(merged_features, dtype=bool)

    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=base,
        table_feature_names=base_feature_names,
    )
    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=incoming,
        table_feature_names=incoming_feature_names,
    )

    if not bool(np.all(coverage)):
        missing = np.argwhere(~coverage)
        preview = [
            f"{merged_model_names[int(r)]}:{merged_feature_names[int(c)]}"
            for r, c in missing[:5].tolist()
        ]
        raise ValueError(
            "Merged output would contain missing feature cells; this usually means the new run "
            "does not cover all required rows/features. "
            f"Missing example(s): {preview}"
        )

    merged_labels = _merge_labels_for_dense_rows(
        base=base,
        incoming=incoming,
        merged_row_index=merged_row_index,
    )

    overlap_rows = sorted(set(base.model_names) & set(incoming.model_names))
    overlap_features = sorted(set(base_feature_names) & set(incoming_feature_names))
    stats = {
        "base_rows": int(len(base.model_names)),
        "base_feature_dim": int(len(base_feature_names)),
        "incoming_rows": int(len(incoming.model_names)),
        "incoming_feature_dim": int(len(incoming_feature_names)),
        "merged_rows": int(len(merged_model_names)),
        "merged_feature_dim": int(len(merged_feature_names)),
        "rows_added": int(len(merged_model_names) - len(base.model_names)),
        "features_added": int(len(merged_feature_names) - len(base_feature_names)),
        "row_overlap": int(len(overlap_rows)),
        "feature_overlap": int(len(overlap_features)),
        "labels_complete": bool(merged_labels is not None),
        "used_positional_feature_names": used_positional_feature_names,
    }

    merged_table = SpectralFeatureTable(
        source="merged_output",
        features=merged_features,
        labels=merged_labels,
        model_names=merged_model_names,
        feature_names=merged_feature_names,
        feature_names_inferred=used_positional_feature_names,
        metadata={},
    )
    return merged_table, stats


def merge_spectral_shards(
    *,
    manifest_json: Path,
    dataset_root: Path,
    shard_run_dirs: list[Path],
    output_dir: Path,
    pipeline_start_epoch_seconds: float | None = None,
) -> dict[str, Path | None]:
    merge_started_perf = perf_counter()
    merge_started_at = datetime.now(timezone.utc)
    if not shard_run_dirs:
        raise ValueError("At least one shard run directory is required")
    manifest_json = resolve_manifest_path(manifest_json)

    expected_names, expected_items_by_name = _load_expected_manifest_items(
        manifest_json=manifest_json,
        dataset_root=dataset_root,
    )

    bundles = [_load_shard_bundle(Path(run_dir).expanduser().resolve()) for run_dir in shard_run_dirs]
    shard_timing_summary, derived_pipeline_start = _summarize_shard_timings(
        [bundle.run_dir for bundle in bundles]
    )
    incoming_table, skipped_models = _build_incoming_table_from_shard_bundles(
        bundles=bundles,
        expected_names=expected_names,
        expected_items_by_name=expected_items_by_name,
        manifest_context=f"{manifest_json}::path",
    )
    output_table = incoming_table
    output_paths = _write_output_table_artifacts(
        output_dir=output_dir,
        output_table=output_table,
    )

    incoming_dataset_reference_payload = build_dataset_reference_payload_from_items(
        items=[expected_items_by_name[name] for name in incoming_table.model_names],
        artifact_kind="merge_spectral_shards",
        manifest_json=manifest_json,
        dataset_root=dataset_root,
        artifact_model_count=int(incoming_table.features.shape[0]),
        source_artifacts=[str(manifest_json), *(str(bundle.run_dir) for bundle in bundles)],
    )
    dataset_reference_payload = incoming_dataset_reference_payload
    source_dataset_layouts = [
        dataset_layouts_from_source(
            metadata=incoming_table.metadata,
            dataset_reference_payload=incoming_dataset_reference_payload,
        )
    ]

    merged_metadata = _rebuild_output_metadata(
        source_metadata=incoming_table.metadata,
        output_table=output_table,
        shard_bundles=bundles,
        metadata_sources=[dict(incoming_table.metadata)],
    )
    write_spectral_metadata(
        output_paths["metadata_path"],
        internal_metadata=merged_metadata,
        dataset_layouts=merge_dataset_layouts(
            output_dataset_reference_payload=dataset_reference_payload,
            source_layouts=source_dataset_layouts,
        ),
    )
    write_dataset_reference_report(output_paths["dataset_reference_report_path"], dataset_reference_payload)

    merge_completed_at = datetime.now(timezone.utc)
    merge_report = {
        "timestamp_utc": merge_completed_at.isoformat(),
        "merge_started_timestamp_utc": merge_started_at.isoformat(),
        "merge_completed_timestamp_utc": merge_completed_at.isoformat(),
        "merge_elapsed_seconds": float(perf_counter() - merge_started_perf),
        **_pipeline_timing_fields(
            merge_completed_at=merge_completed_at,
            pipeline_start_epoch_seconds=pipeline_start_epoch_seconds,
            derived_pipeline_start=derived_pipeline_start,
        ),
        "manifest_json": str(manifest_json),
        "dataset_root": str(dataset_root),
        "n_shards": int(len(bundles)),
        "manifest_model_count": int(len(expected_names)),
        "skipped_model_count": int(len(skipped_models)),
        "incoming_rows": int(incoming_table.features.shape[0]),
        "incoming_feature_dim": int(incoming_table.features.shape[1]),
        "n_rows": int(output_table.features.shape[0]),
        "feature_dim": int(output_table.features.shape[1]),
        "merged_with_existing_output": False,
        "merge_stats": None,
        "shard_runtime": shard_timing_summary,
        "output": {
            "feature_path": str(output_paths["feature_path"]),
            "model_names_path": str(output_paths["model_names_path"]),
            "labels_path": (
                str(output_paths["labels_path"]) if output_table.labels is not None else None
            ),
            "metadata_path": str(output_paths["metadata_path"]),
            "dataset_reference_report_path": str(output_paths["dataset_reference_report_path"]),
        },
    }
    with open(output_paths["merge_report_path"], "w", encoding="utf-8") as f:
        json.dump(json_ready(merge_report), f, indent=2)

    return {
        "feature_path": output_paths["feature_path"],
        "model_names_path": output_paths["model_names_path"],
        "labels_path": output_paths["labels_path"] if output_table.labels is not None else None,
        "metadata_path": output_paths["metadata_path"],
        "merge_report_path": output_paths["merge_report_path"],
        "dataset_reference_report_path": output_paths["dataset_reference_report_path"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge sharded spectral extraction runs")
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help=dataset_root_help(),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pipeline-start-epoch-seconds",
        type=float,
        default=None,
        help=(
            "Optional UTC epoch seconds for when the end-to-end extraction pipeline began. "
            "If provided, the merge report includes total pipeline elapsed wall time."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--shard-run-dirs",
        type=Path,
        nargs="+",
        help="Paths to shard run dirs (each containing features/spectral_*.{npy,json})",
    )
    group.add_argument(
        "--shard-run-dir-glob",
        type=str,
        default=None,
        help="Glob pattern resolving to shard run dirs (e.g. '/path/to/feature_extract/shard_*')",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    shard_run_dirs: list[Path]
    if args.shard_run_dirs is not None:
        shard_run_dirs = list(args.shard_run_dirs)
    else:
        shard_run_dirs = [Path(p).expanduser().resolve() for p in sorted(glob.glob(args.shard_run_dir_glob))]
        if not shard_run_dirs:
            raise FileNotFoundError(f"No shard run directories matched glob: {args.shard_run_dir_glob}")

    outputs = merge_spectral_shards(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        shard_run_dirs=shard_run_dirs,
        output_dir=args.output_dir,
        pipeline_start_epoch_seconds=args.pipeline_start_epoch_seconds,
    )
    print("Merged spectral shards")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Merge report: {outputs['merge_report_path']}")
    print(f"Dataset references: {outputs['dataset_reference_report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
