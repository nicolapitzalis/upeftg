from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .schema import (
    normalize_requested_features as _normalize_requested_features,
    resolve_output_feature_names as _resolve_output_feature_names,
)

from ..features.spectral import (
    ordered_block_names_from_feature_names,
    sanitize_spectral_metadata,
    spectral_block_lora_dims_by_block,
)
from .provenance.datasets import (
    finalize_dataset_reference_payload,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from .paths import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    default_output_companion_path as _default_output_companion_path,
    resolve_existing_companion_path as _resolve_existing_companion_path,
    resolve_feature_extract_root as _resolve_feature_extract_root,
    resolve_input_feature_path as _resolve_input_feature_path,
    resolve_output_feature_path as _resolve_output_feature_path,
)
from .merge.shards import _unique_index_by_name, resolved_qv_sum_mode
from ..utilities.core.serialization import json_ready
from .metadata.spectral import dataset_layouts_from_source, write_spectral_metadata
from .provenance.subsets import (
    _load_source_payload,
    _load_table_from_feature_path,
    _resolve_provenance_feature_names,
)


@dataclass(frozen=True)
class RowFilters:
    dataset_names: frozenset[str] | None
    subset_names: frozenset[str] | None
    model_families: frozenset[str] | None
    attack_names: frozenset[str] | None
    model_names: frozenset[str] | None

    def as_metadata_dict(self) -> dict[str, list[str]]:
        payload: dict[str, list[str]] = {}
        for key, values in [
            ("dataset_names", self.dataset_names),
            ("subset_names", self.subset_names),
            ("model_families", self.model_families),
            ("attack_names", self.attack_names),
            ("model_names", self.model_names),
        ]:
            if values:
                payload[key] = sorted(values)
        return payload


def _normalize_text_filter(values: list[str] | tuple[str, ...] | None) -> frozenset[str] | None:
    if not values:
        return None
    cleaned = frozenset(str(value).strip() for value in values if str(value).strip())
    return cleaned or None


def _entry_matches_filters(entry: dict[str, Any], filters: RowFilters) -> bool:
    checks = [
        ("dataset_name", filters.dataset_names),
        ("subset_name", filters.subset_names),
        ("model_family", filters.model_families),
        ("attack_name", filters.attack_names),
    ]
    for key, allowed in checks:
        if allowed is None:
            continue
        value = str(entry.get(key) or "").strip()
        if value not in allowed:
            return False
    return True


def _select_output_model_names(
    *,
    ordered_model_names: list[str],
    source_payload: dict[str, Any],
    filters: RowFilters,
) -> list[str]:
    raw_model_index = source_payload.get("model_index")
    if not isinstance(raw_model_index, dict):
        raise ValueError("Dataset-reference payload is missing model_index")

    missing_reference_names = [name for name in ordered_model_names if name not in raw_model_index]
    if missing_reference_names:
        preview = ", ".join(missing_reference_names[:5])
        raise ValueError(
            "Feature subset export requires dataset-reference entries for every candidate row. "
            f"Missing {len(missing_reference_names)} model(s). Examples: {preview}"
        )

    selected_model_names: list[str] = []
    requested_model_names = filters.model_names
    for model_name in ordered_model_names:
        if requested_model_names is not None and model_name not in requested_model_names:
            continue
        entry = raw_model_index.get(model_name)
        if not isinstance(entry, dict):
            continue
        if _entry_matches_filters(entry, filters):
            selected_model_names.append(model_name)

    if not selected_model_names:
        filters_payload = filters.as_metadata_dict()
        if filters_payload:
            raise ValueError(
                "No rows matched the requested selectors: "
                + ", ".join(f"{key}={value}" for key, value in sorted(filters_payload.items()))
            )
        raise ValueError("No rows are available in the selected feature bundle")
    return selected_model_names


def _filter_dataset_reference_payload(
    *,
    source_feature_path: Path,
    output_model_names: list[str],
    selected_leaf_paths: list[Path],
) -> dict[str, Any]:
    source_payload = _load_source_payload(source_feature_path)
    raw_model_index = source_payload["model_index"]
    filtered_model_index = {
        model_name: dict(raw_model_index[model_name])
        for model_name in output_model_names
        if model_name in raw_model_index and isinstance(raw_model_index[model_name], dict)
    }
    missing_from_reference = [name for name in output_model_names if name not in filtered_model_index]

    provenance_gaps = [str(x) for x in source_payload.get("provenance_gaps", []) if str(x).strip()]
    if missing_from_reference:
        preview = ", ".join(missing_from_reference[:5])
        provenance_gaps.append(
            "Filtered feature subset is missing dataset-reference entries for "
            f"{len(missing_from_reference)} model(s). Examples: {preview}"
        )

    dataset_root_raw = source_payload.get("dataset_root")
    dataset_root = Path(str(dataset_root_raw)).expanduser() if dataset_root_raw else None
    return finalize_dataset_reference_payload(
        artifact_kind="export_feature_subset",
        model_index=filtered_model_index,
        artifact_model_count=int(len(output_model_names)),
        manifest_json=None,
        dataset_root=dataset_root,
        source_artifacts=[str(source_feature_path), *[str(path) for path in selected_leaf_paths]],
        provenance_gaps=provenance_gaps,
        is_complete=bool(source_payload.get("is_complete", True))
        and not missing_from_reference
        and len(filtered_model_index) == len(output_model_names),
    )


def _build_filtered_metadata(
    *,
    source_metadata: dict[str, Any],
    output_model_names: list[str],
    output_feature_names: list[str],
    source_feature_path: Path,
    selected_leaf_paths: list[Path],
    row_filters: RowFilters,
    requested_features: list[str] | None,
    provenance_feature_names: list[str],
) -> dict[str, Any]:
    metadata = sanitize_spectral_metadata(source_metadata)
    for key in [
        "feature_names",
        "feature_dim",
        "n_models",
        "block_names",
        "n_blocks",
        "base_block_names",
        "qv_sum_block_names",
        "lora_adapter_dims",
        "base_lora_adapter_dims",
        "qv_sum_lora_adapter_dims",
        "schema_layout_summary",
        "dataset_layouts",
    ]:
        metadata.pop(key, None)

    metadata["n_models"] = int(len(output_model_names))
    metadata["feature_dim"] = int(len(output_feature_names))
    metadata["feature_names"] = list(output_feature_names)
    metadata["feature_subset_created_by"] = "export_feature_subset"
    metadata["feature_subset_source_feature_file"] = str(source_feature_path)
    metadata["feature_subset_selected_source_feature_files"] = [str(path) for path in selected_leaf_paths]
    metadata["feature_subset_row_filters"] = row_filters.as_metadata_dict()
    metadata["feature_subset_requested_features"] = (
        list(requested_features) if requested_features is not None else "all"
    )
    metadata["feature_subset_provenance_feature_column_count"] = int(len(provenance_feature_names))

    if not output_feature_names:
        return metadata

    try:
        block_names = ordered_block_names_from_feature_names(output_feature_names)
    except ValueError:
        return metadata

    if not block_names:
        return metadata

    metadata["block_names"] = list(block_names)
    metadata["n_blocks"] = int(len(block_names))
    metadata["base_block_names"] = [name for name in block_names if ".qv_sum" not in name]
    metadata["qv_sum_block_names"] = [name for name in block_names if ".qv_sum" in name]
    metadata["spectral_qv_sum_mode"] = resolved_qv_sum_mode(block_names)

    source_dim_map = spectral_block_lora_dims_by_block(source_metadata)
    if all(block_name in source_dim_map for block_name in block_names):
        metadata["lora_adapter_dims"] = [dict(source_dim_map[block_name]) for block_name in block_names]

    base_block_names = list(metadata["base_block_names"])
    if base_block_names and all(block_name in source_dim_map for block_name in base_block_names):
        metadata["base_lora_adapter_dims"] = [dict(source_dim_map[block_name]) for block_name in base_block_names]

    qv_sum_block_names = list(metadata["qv_sum_block_names"])
    if qv_sum_block_names and all(block_name in source_dim_map for block_name in qv_sum_block_names):
        metadata["qv_sum_lora_adapter_dims"] = [dict(source_dim_map[block_name]) for block_name in qv_sum_block_names]

    return metadata


def export_feature_subset(
    *,
    feature_file: Path,
    output_filename: Path,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    dataset_names: list[str] | tuple[str, ...] | None = None,
    subset_names: list[str] | tuple[str, ...] | None = None,
    model_families: list[str] | tuple[str, ...] | None = None,
    attack_names: list[str] | tuple[str, ...] | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
    features: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path | None]:
    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()

    row_filters = RowFilters(
        dataset_names=_normalize_text_filter(dataset_names),
        subset_names=_normalize_text_filter(subset_names),
        model_families=_normalize_text_filter(model_families),
        attack_names=_normalize_text_filter(attack_names),
        model_names=_normalize_text_filter(model_names),
    )
    requested_features = _normalize_requested_features(features)

    resolved_feature_root = _resolve_feature_extract_root(feature_root)
    resolved_feature_path = _resolve_input_feature_path(
        Path(feature_file),
        feature_root=resolved_feature_root,
    )
    table = _load_table_from_feature_path(resolved_feature_path)
    if table.feature_names_inferred:
        raise ValueError(
            "Feature subset export requires explicit feature_names metadata; inferred positional names are not supported"
        )

    source_payload = _load_source_payload(resolved_feature_path)
    output_model_names = _select_output_model_names(
        ordered_model_names=list(table.model_names),
        source_payload=source_payload,
        filters=row_filters,
    )
    provenance_feature_names, selected_leaf_paths = _resolve_provenance_feature_names(
        root_feature_path=resolved_feature_path,
        root_feature_names=[str(name) for name in table.feature_names],
        selected_model_names=output_model_names,
    )
    output_feature_names = _resolve_output_feature_names(
        available_feature_names=provenance_feature_names,
        requested_features=requested_features,
    )

    row_index = _unique_index_by_name(
        list(table.model_names),
        context=str(resolved_feature_path),
        entity="feature model names",
    )
    feature_index = _unique_index_by_name(
        [str(name) for name in table.feature_names],
        context=str(resolved_feature_path),
        entity="feature names",
    )
    row_indices = np.asarray([row_index[name] for name in output_model_names], dtype=np.int64)
    col_indices = np.asarray([feature_index[name] for name in output_feature_names], dtype=np.int64)

    output_features = np.asarray(table.features[np.ix_(row_indices, col_indices)], dtype=np.float32)
    output_labels = None if table.labels is None else np.asarray(table.labels[row_indices], dtype=np.int32)

    output_feature_path = _resolve_output_feature_path(
        Path(output_filename),
        feature_root=resolved_feature_root,
    )
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_names_path = _default_output_companion_path(output_feature_path, "_model_names.json")
    output_labels_path = _default_output_companion_path(output_feature_path, "_labels.npy")
    output_metadata_path = _default_output_companion_path(output_feature_path, "_metadata.json")
    output_report_path = _default_output_companion_path(output_feature_path, "_feature_subset_report.json")

    np.save(output_feature_path, output_features.astype(np.float32, copy=False))
    with open(output_model_names_path, "w", encoding="utf-8") as f:
        json.dump(output_model_names, f, indent=2)

    if output_labels is not None:
        np.save(output_labels_path, output_labels.astype(np.int32, copy=False))
    elif output_labels_path.exists():
        output_labels_path.unlink()

    filtered_dataset_reference_payload = _filter_dataset_reference_payload(
        source_feature_path=resolved_feature_path,
        output_model_names=output_model_names,
        selected_leaf_paths=selected_leaf_paths,
    )
    output_metadata = _build_filtered_metadata(
        source_metadata=dict(table.metadata),
        output_model_names=output_model_names,
        output_feature_names=output_feature_names,
        source_feature_path=resolved_feature_path,
        selected_leaf_paths=selected_leaf_paths,
        row_filters=row_filters,
        requested_features=requested_features,
        provenance_feature_names=provenance_feature_names,
    )
    write_spectral_metadata(
        output_metadata_path,
        internal_metadata=output_metadata,
        dataset_layouts=dataset_layouts_from_source(
            metadata=output_metadata,
            dataset_reference_payload=filtered_dataset_reference_payload,
        ),
    )

    output_dataset_reference_report_path = default_dataset_reference_report_path(output_feature_path.parent)
    write_dataset_reference_report(output_dataset_reference_report_path, filtered_dataset_reference_payload)

    completed_at = datetime.now(timezone.utc)
    subset_report = {
        "timestamp_utc": completed_at.isoformat(),
        "subset_started_timestamp_utc": started_at.isoformat(),
        "subset_completed_timestamp_utc": completed_at.isoformat(),
        "subset_elapsed_seconds": float(perf_counter() - started_perf),
        "feature_extract_root": str(resolved_feature_root),
        "selection": {
            "row_filters": row_filters.as_metadata_dict(),
            "requested_features": list(requested_features) if requested_features is not None else "all",
            "selected_source_feature_files": [str(path) for path in selected_leaf_paths],
        },
        "stats": {
            "input_rows": int(table.features.shape[0]),
            "input_feature_dim": int(table.features.shape[1]),
            "output_rows": int(output_features.shape[0]),
            "output_feature_dim": int(output_features.shape[1]),
            "available_provenance_feature_column_count": int(len(provenance_feature_names)),
            "selected_source_feature_file_count": int(len(selected_leaf_paths)),
        },
        "input": {
            "feature_path": str(resolved_feature_path),
            "model_names_path": str(
                _resolve_existing_companion_path(
                    resolved_feature_path,
                    "_model_names.json",
                    required=True,
                )
            ),
        },
        "output": {
            "feature_path": str(output_feature_path),
            "model_names_path": str(output_model_names_path),
            "labels_path": str(output_labels_path) if output_labels is not None else None,
            "metadata_path": str(output_metadata_path),
            "dataset_reference_report_path": str(output_dataset_reference_report_path),
        },
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(subset_report), f, indent=2)

    return {
        "feature_path": output_feature_path,
        "model_names_path": output_model_names_path,
        "labels_path": output_labels_path if output_labels is not None else None,
        "metadata_path": output_metadata_path,
        "dataset_reference_report_path": output_dataset_reference_report_path,
        "subset_report_path": output_report_path,
    }
