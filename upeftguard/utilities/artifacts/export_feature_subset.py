from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ...features.spectral import (
    ordered_block_names_from_feature_names,
    resolve_spectral_features,
    sanitize_spectral_metadata,
    spectral_block_lora_dims_by_block,
)
from .dataset_references import (
    _finalize_payload,
    default_dataset_reference_report_path,
    resolve_dataset_reference_payload_for_artifact,
    write_dataset_reference_report,
)
from ..merge.merge_feature_files import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    _default_output_companion_path,
    _resolve_existing_companion_path,
    _resolve_feature_extract_root,
    _resolve_input_feature_path,
    _resolve_output_feature_path,
)
from ..merge.merge_spectral_shards import (
    _load_feature_table,
    _unique_index_by_name,
    resolved_qv_sum_mode,
)
from ..core.serialization import json_ready
from .spectral_metadata import dataset_layouts_from_source, write_spectral_metadata


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


@dataclass(frozen=True)
class LeafFeatureSource:
    feature_path: Path
    model_names: frozenset[str]
    feature_names: tuple[str, ...]


def _normalize_text_filter(values: list[str] | tuple[str, ...] | None) -> frozenset[str] | None:
    if not values:
        return None
    cleaned = frozenset(str(value).strip() for value in values if str(value).strip())
    return cleaned or None


def _normalize_requested_features(features: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if not features:
        return None
    cleaned = [str(value).strip().lower() for value in features if str(value).strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1 and cleaned[0].lower() == "all":
        return None
    if any(value.lower() == "all" for value in cleaned):
        raise ValueError(
            "--features/--columns must either be omitted, set to 'all', or list supported spectral feature groups"
        )
    return resolve_spectral_features(cleaned)


def _feature_group_for_feature_name(feature_name: str) -> str | None:
    suffix = str(feature_name).rpartition(".")[2]
    if not suffix:
        return None
    if suffix.startswith("sv_") and suffix[3:].isdigit():
        return "sv_topk"
    if suffix == "sv_kurtosis":
        return "kurtosis"
    if suffix == "sv_l1_norm":
        return "l1_norm"
    if suffix == "sv_linf_norm":
        return "linf_norm"
    if suffix == "sv_mean_abs":
        return "mean_abs"
    try:
        resolved = resolve_spectral_features([suffix])
    except ValueError:
        return None
    return resolved[0] if resolved else None


def _resolve_companion_paths(feature_path: Path) -> tuple[Path, Path, Path]:
    model_names_path = _resolve_existing_companion_path(
        feature_path,
        "_model_names.json",
        required=True,
    )
    labels_path = _resolve_existing_companion_path(
        feature_path,
        "_labels.npy",
        required=False,
    )
    metadata_path = _resolve_existing_companion_path(
        feature_path,
        "_metadata.json",
        required=False,
    )
    return model_names_path, labels_path, metadata_path


def _load_table_from_feature_path(feature_path: Path):
    resolved_feature_path = feature_path.expanduser()
    if not resolved_feature_path.is_absolute():
        resolved_feature_path = (Path.cwd().resolve() / resolved_feature_path).resolve()
    else:
        resolved_feature_path = resolved_feature_path.resolve()
    model_names_path, labels_path, metadata_path = _resolve_companion_paths(resolved_feature_path)
    return _load_feature_table(
        source=str(resolved_feature_path),
        feature_path=resolved_feature_path,
        model_names_path=model_names_path,
        labels_path=labels_path,
        metadata_path=metadata_path,
        context=f"feature subset input {resolved_feature_path}",
    )


def _resolve_child_feature_path(raw_path: Any, *, parent_feature_path: Path) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    parent_relative = (parent_feature_path.parent / candidate).resolve()
    if parent_relative.exists():
        return parent_relative
    return (Path.cwd().resolve() / candidate).resolve()


def _collect_leaf_feature_sources(
    feature_path: Path,
    *,
    memo: dict[Path, dict[Path, LeafFeatureSource]],
    active: set[Path],
) -> dict[Path, LeafFeatureSource]:
    resolved_feature_path = feature_path.resolve()
    if resolved_feature_path in memo:
        return memo[resolved_feature_path]
    if resolved_feature_path in active:
        raise ValueError(f"Detected recursive merge-source reference while resolving {resolved_feature_path}")

    active.add(resolved_feature_path)
    try:
        table = _load_table_from_feature_path(resolved_feature_path)
        raw_sources = table.metadata.get("merge_source_feature_files")
        source_paths = (
            [
                _resolve_child_feature_path(raw_path, parent_feature_path=resolved_feature_path)
                for raw_path in raw_sources
                if str(raw_path).strip()
            ]
            if isinstance(raw_sources, list)
            else []
        )

        if not source_paths:
            payload = resolve_dataset_reference_payload_for_artifact(resolved_feature_path)
            raw_model_index = payload.get("model_index")
            if isinstance(raw_model_index, dict):
                model_names = frozenset(str(name) for name in raw_model_index)
            else:
                model_names = frozenset(str(name) for name in table.model_names)
            result = {
                resolved_feature_path: LeafFeatureSource(
                    feature_path=resolved_feature_path,
                    model_names=model_names,
                    feature_names=tuple(str(name) for name in table.feature_names),
                )
            }
        else:
            result: dict[Path, LeafFeatureSource] = {}
            for source_path in source_paths:
                result.update(
                    _collect_leaf_feature_sources(
                        source_path,
                        memo=memo,
                        active=active,
                    )
                )

        memo[resolved_feature_path] = result
        return result
    finally:
        active.discard(resolved_feature_path)


def _load_source_payload(feature_path: Path) -> dict[str, Any]:
    payload = resolve_dataset_reference_payload_for_artifact(feature_path)
    raw_model_index = payload.get("model_index")
    if not isinstance(raw_model_index, dict) or not raw_model_index:
        raise ValueError(
            "Feature subset export requires dataset-reference state with a non-empty model_index for "
            f"{feature_path}"
        )
    return payload


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


def _resolve_provenance_feature_names(
    *,
    root_feature_path: Path,
    root_feature_names: list[str],
    selected_model_names: list[str],
) -> tuple[list[str], list[Path]]:
    _unique_index_by_name(root_feature_names, context=str(root_feature_path), entity="feature names")

    leaf_sources = _collect_leaf_feature_sources(
        root_feature_path,
        memo={},
        active=set(),
    )
    selected_model_name_set = set(selected_model_names)
    required_feature_names: set[str] = set()
    covered_model_names: set[str] = set()
    matched_leaf_paths: list[Path] = []

    for leaf_path in sorted(leaf_sources):
        leaf = leaf_sources[leaf_path]
        overlap = selected_model_name_set & set(leaf.model_names)
        if not overlap:
            continue
        matched_leaf_paths.append(leaf.feature_path)
        covered_model_names.update(overlap)
        required_feature_names.update(str(name) for name in leaf.feature_names)

    uncovered = sorted(selected_model_name_set - covered_model_names)
    if uncovered:
        preview = ", ".join(uncovered[:5])
        raise ValueError(
            "Could not resolve provenance-owning source feature files for "
            f"{len(uncovered)} selected model(s). Examples: {preview}"
        )

    if not required_feature_names:
        raise ValueError("Resolved provenance leaf sources but no feature names were discovered")

    root_feature_name_set = set(root_feature_names)
    missing_in_root = sorted(name for name in required_feature_names if name not in root_feature_name_set)
    if missing_in_root:
        preview = ", ".join(missing_in_root[:5])
        raise ValueError(
            "Provenance source feature names are not present in the requested feature bundle. "
            f"Examples: {preview}"
        )

    available_feature_names = [name for name in root_feature_names if name in required_feature_names]
    if not available_feature_names:
        raise ValueError("No provenance-backed columns were available for the selected rows")
    return available_feature_names, matched_leaf_paths


def _resolve_output_feature_names(
    *,
    available_feature_names: list[str],
    requested_features: list[str] | None,
) -> list[str]:
    if requested_features is None:
        return list(available_feature_names)

    requested_feature_set = set(requested_features)
    selected_feature_names = [
        name
        for name in available_feature_names
        if _feature_group_for_feature_name(name) in requested_feature_set
    ]
    available_feature_groups = {
        group
        for group in (_feature_group_for_feature_name(name) for name in available_feature_names)
        if group is not None
    }
    missing_features = [name for name in requested_features if name not in available_feature_groups]
    if missing_features:
        preview = ", ".join(missing_features[:5])
        raise ValueError(
            "Requested --features/--columns are not available for the selected provenance subset. "
            f"Examples: {preview}"
        )
    return selected_feature_names


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
    return _finalize_payload(
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
        metadata["qv_sum_lora_adapter_dims"] = [
            dict(source_dim_map[block_name]) for block_name in qv_sum_block_names
        ]

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
    output_labels = (
        None if table.labels is None else np.asarray(table.labels[row_indices], dtype=np.int32)
    )

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
            "model_names_path": str(_resolve_companion_paths(resolved_feature_path)[0]),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a provenance-backed subset of a spectral feature file by selecting rows from "
            "dataset-reference metadata, then keeping the exact columns owned by the leaf source "
            "artifacts that produced those rows."
        )
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        required=True,
        help="Input run name or explicit spectral feature .npy file",
    )
    parser.add_argument(
        "--output-filename",
        type=Path,
        required=True,
        help="Output run name or explicit output feature matrix path (.npy)",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_EXTRACT_ROOT,
        help="Base directory used to resolve bare feature run names (default: runs/feature_extract)",
    )
    parser.add_argument("--dataset-name", dest="dataset_names", nargs="+", default=None)
    parser.add_argument("--subset-name", dest="subset_names", nargs="+", default=None)
    parser.add_argument("--model-family", dest="model_families", nargs="+", default=None)
    parser.add_argument("--attack-name", dest="attack_names", nargs="+", default=None)
    parser.add_argument("--model-name", dest="model_names", nargs="+", default=None)
    parser.add_argument(
        "--features",
        "--columns",
        dest="features",
        nargs="+",
        default=None,
        help=(
            "Spectral feature groups to keep after provenance selection across all matched blocks/layers; "
            "omit or pass 'all' to keep every available feature family"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = export_feature_subset(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        feature_root=args.feature_root,
        dataset_names=args.dataset_names,
        subset_names=args.subset_names,
        model_families=args.model_families,
        attack_names=args.attack_names,
        model_names=args.model_names,
        features=args.features,
    )
    print("Feature subset export complete")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Dataset references: {outputs['dataset_reference_report_path']}")
    print(f"Subset report: {outputs['subset_report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
