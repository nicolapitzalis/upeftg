from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ...features.spectral import (
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    _layer_identifier_for_block_name,
    build_spectral_feature_names,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
    sanitize_spectral_metadata,
    spectral_extractor_params,
)
from .dataset_references import (
    _finalize_payload,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from .export_feature_subset import (
    _feature_group_for_feature_name,
    _load_source_payload,
    _load_table_from_feature_path,
    _normalize_requested_features,
    _resolve_model_owned_feature_names,
    _resolve_output_feature_names,
)
from .spectral_metadata import dataset_layouts_from_source, write_spectral_metadata
from ..merge.merge_feature_files import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    _default_output_companion_path,
    _resolve_feature_extract_root,
    _resolve_input_feature_path,
    _resolve_output_feature_path,
)
from ..merge.merge_spectral_shards import _unique_index_by_name, resolved_qv_sum_mode
from ..core.serialization import json_ready


ROLE_BUCKET_ORDER = ("q", "v", "qv_sum", "other")
SUPPORTED_AGGREGATION_OPERATORS = ("avg", "max", "min")

_Q_ROLE_ALIASES = {"q", "q_proj", "query"}
_V_ROLE_ALIASES = {"v", "v_proj", "value"}


def _normalize_aggregation_operator(operator: str) -> str:
    resolved = str(operator).strip().lower()
    if resolved not in SUPPORTED_AGGREGATION_OPERATORS:
        raise ValueError(
            f"Unsupported aggregation operator '{operator}'. "
            f"Supported: {list(SUPPORTED_AGGREGATION_OPERATORS)}"
        )
    return resolved


def _feature_block_name(feature_name: str) -> str:
    block_name, sep, _ = str(feature_name).rpartition(".")
    if not sep or not block_name:
        raise ValueError(f"Invalid spectral feature name: {feature_name}")
    return block_name


def _emitted_feature_name(feature_name: str) -> str:
    return str(feature_name).rpartition(".")[2]


def _role_bucket_for_block_name(block_name: str) -> str:
    text = str(block_name).strip()
    if text.endswith(".qv_sum"):
        return "qv_sum"
    module = text.rsplit(".", 1)[-1].strip().lower()
    if module in _Q_ROLE_ALIASES:
        return "q"
    if module in _V_ROLE_ALIASES:
        return "v"
    return "other"


def _role_bucket_for_feature_name(feature_name: str) -> str:
    return _role_bucket_for_block_name(_feature_block_name(feature_name))

def _indexed_part_span(parts: list[str], *, prefix: str) -> tuple[int, int] | None:
    prefix_text = str(prefix).strip().lower()
    for idx, raw_part in enumerate(parts):
        part = str(raw_part).strip().lower()
        if part.startswith(prefix_text) and part[len(prefix_text) :].isdigit():
            return idx, idx + 1
        if part == prefix_text and idx + 1 < len(parts) and str(parts[idx + 1]).strip().isdigit():
            return idx, idx + 2
    return None


def _structural_group_for_block_name(block_name: str) -> str:
    parts = [part for part in str(block_name).split(".") if part]
    if not parts:
        return str(block_name)

    block_span = _indexed_part_span(parts, prefix="block")
    if block_span is not None:
        _, end_idx = block_span
        return ".".join(parts[:end_idx]).strip() or str(block_name)

    layer_span = _indexed_part_span(parts, prefix="layer")
    if layer_span is not None:
        _, end_idx = layer_span
        return ".".join(parts[:end_idx]).strip() or str(block_name)

    return _layer_identifier_for_block_name(str(block_name))


def _structural_group_for_feature_name(feature_name: str) -> str:
    return _structural_group_for_block_name(_feature_block_name(feature_name))


def _filter_selected_input_feature_names(
    *,
    root_feature_names: list[str],
    requested_features: list[str] | None,
    spectral_qv_sum_mode: str,
) -> list[str]:
    resolved_qv_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    available_feature_names = list(root_feature_names)
    if resolved_qv_mode == "none":
        available_feature_names = [
            name
            for name in available_feature_names
            if _role_bucket_for_feature_name(name) != "qv_sum"
        ]
    elif resolved_qv_mode == "only":
        available_feature_names = [
            name
            for name in available_feature_names
            if _role_bucket_for_feature_name(name) == "qv_sum"
        ]

    if not available_feature_names:
        raise ValueError(
            "Input feature bundle does not contain any columns compatible with "
            f"--spectral-qv-sum-mode={resolved_qv_mode}"
        )

    selected_feature_names = _resolve_output_feature_names(
        available_feature_names=available_feature_names,
        requested_features=requested_features,
    )
    if not selected_feature_names:
        raise ValueError("Requested aggregation resolved to zero input columns")
    return selected_feature_names


def _ordered_feature_groups(feature_names: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        feature_group = _feature_group_for_feature_name(feature_name)
        if feature_group is None or feature_group in seen:
            continue
        seen.add(feature_group)
        ordered.append(feature_group)
    return ordered


def _resolved_output_roles(feature_names: list[str]) -> list[str]:
    present_roles = {_role_bucket_for_feature_name(feature_name) for feature_name in feature_names}
    ordered = [role for role in ROLE_BUCKET_ORDER if role in present_roles]
    if not ordered:
        raise ValueError("Requested aggregation resolved to zero role buckets")
    return ordered


def _sv_top_k_from_feature_names(feature_names: list[str], *, default: int) -> int:
    max_rank = 0
    for feature_name in feature_names:
        emitted = _emitted_feature_name(feature_name)
        if not emitted.startswith("sv_"):
            continue
        suffix = emitted[3:]
        if suffix.isdigit():
            max_rank = max(max_rank, int(suffix))
    return max_rank if max_rank > 0 else int(default)


def _aggregation_value(
    grouped_values: dict[str, np.ndarray],
    *,
    operator: str,
) -> float:
    if operator == "avg":
        # Give each higher-level structural group one vote so architectures with
        # more repeated substructure do not dominate the average by column count.
        group_means = [
            float(np.mean(np.asarray(values, dtype=np.float32), dtype=np.float64))
            for values in grouped_values.values()
        ]
        return float(np.mean(np.asarray(group_means, dtype=np.float64), dtype=np.float64))

    all_values = np.concatenate(
        [np.asarray(values, dtype=np.float32).reshape(-1) for values in grouped_values.values()],
        axis=0,
    )
    if operator == "min":
        return float(np.min(all_values))
    if operator == "max":
        return float(np.max(all_values))
    raise ValueError(f"Unsupported aggregation operator '{operator}'")


def _build_output_feature_names(
    *,
    role_buckets: list[str],
    selected_feature_groups: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
) -> list[str]:
    output_block_names = [f"role.{role}" for role in role_buckets]
    return build_spectral_feature_names(
        block_names=output_block_names,
        selected_features=selected_feature_groups,
        sv_top_k=int(sv_top_k),
        spectral_moment_source=spectral_moment_source,
        shorten_block_names=False,
    )


def _build_aggregated_metadata(
    *,
    source_metadata: dict[str, Any],
    output_model_names: list[str],
    output_feature_names: list[str],
    source_feature_path: Path,
    selected_leaf_paths: list[Path],
    aggregation_operator: str,
    requested_features: list[str] | None,
    requested_qv_sum_mode: str,
    role_buckets: list[str],
    selected_feature_groups: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
    empty_fill_counts: dict[str, int],
    selected_input_feature_count: int,
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

    output_block_names = [f"role.{role}" for role in role_buckets]
    resolved_output_qv_mode = resolved_qv_sum_mode(output_block_names)
    metadata["n_models"] = int(len(output_model_names))
    metadata["feature_dim"] = int(len(output_feature_names))
    metadata["feature_names"] = list(output_feature_names)
    metadata["block_names"] = list(output_block_names)
    metadata["n_blocks"] = int(len(output_block_names))
    metadata["base_block_names"] = [name for name in output_block_names if ".qv_sum" not in name]
    metadata["qv_sum_block_names"] = [name for name in output_block_names if ".qv_sum" in name]
    metadata["resolved_features"] = list(selected_feature_groups)
    metadata["sv_top_k"] = int(sv_top_k)
    metadata["spectral_moment_source"] = str(spectral_moment_source)
    metadata["spectral_qv_sum_mode"] = str(resolved_output_qv_mode)
    metadata["representation_kind"] = "architecture_independent_aggregation"
    metadata["aggregation_operator"] = str(aggregation_operator)
    metadata["aggregation_grouping"] = "role_feature"
    metadata["aggregation_avg_semantics"] = (
        "mean_of_structural_group_means" if str(aggregation_operator) == "avg" else "global_extreme"
    )
    metadata["aggregation_structural_grouping"] = "higher_level_block_or_layer"
    metadata["aggregation_role_buckets"] = list(role_buckets)
    metadata["aggregation_source_feature_file"] = str(source_feature_path)
    metadata["aggregation_selected_source_feature_files"] = [str(path) for path in selected_leaf_paths]
    metadata["aggregation_requested_features"] = (
        list(requested_features) if requested_features is not None else "all"
    )
    metadata["aggregation_requested_qv_sum_mode"] = str(requested_qv_sum_mode)
    metadata["aggregation_selected_input_feature_count"] = int(selected_input_feature_count)
    metadata["aggregation_empty_fill_total"] = int(sum(empty_fill_counts.values()))
    metadata["aggregation_empty_fill_counts"] = {
        key: int(value)
        for key, value in empty_fill_counts.items()
        if int(value) > 0
    }
    metadata["extractor_params"] = spectral_extractor_params(
        {
            "spectral_features": list(selected_feature_groups),
            "spectral_sv_top_k": int(sv_top_k),
            "spectral_moment_source": str(spectral_moment_source),
            "spectral_qv_sum_mode": str(resolved_output_qv_mode),
            "spectral_entrywise_delta_mode": metadata.get("spectral_entrywise_delta_mode"),
        }
    )
    return metadata


def _build_aggregated_dataset_reference_payload(
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
            "Aggregated feature bundle is missing dataset-reference entries for "
            f"{len(missing_from_reference)} model(s). Examples: {preview}"
        )

    dataset_root_raw = source_payload.get("dataset_root")
    dataset_root = Path(str(dataset_root_raw)).expanduser() if dataset_root_raw else None
    return _finalize_payload(
        artifact_kind="aggregate_features",
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


def aggregate_features(
    *,
    feature_file: Path,
    output_filename: Path,
    operator: str,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    features: list[str] | tuple[str, ...] | None = None,
    spectral_qv_sum_mode: str = "append",
) -> dict[str, Path | None]:
    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()

    resolved_operator = _normalize_aggregation_operator(operator)
    requested_features = _normalize_requested_features(features)
    resolved_requested_qv_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)

    resolved_feature_root = _resolve_feature_extract_root(feature_root)
    resolved_feature_path = _resolve_input_feature_path(
        Path(feature_file),
        feature_root=resolved_feature_root,
    )
    table = _load_table_from_feature_path(resolved_feature_path)
    if table.feature_names_inferred:
        raise ValueError(
            "Feature aggregation requires explicit feature_names metadata; inferred positional names are not supported"
        )

    root_feature_names = [str(name) for name in table.feature_names]
    selected_input_feature_names = _filter_selected_input_feature_names(
        root_feature_names=root_feature_names,
        requested_features=requested_features,
        spectral_qv_sum_mode=resolved_requested_qv_mode,
    )
    selected_input_feature_name_set = set(selected_input_feature_names)
    selected_feature_groups = _ordered_feature_groups(selected_input_feature_names)
    source_metadata = dict(table.metadata)
    raw_moment_source = source_metadata.get("spectral_moment_source")
    extractor_params = source_metadata.get("extractor_params")
    if raw_moment_source is None and isinstance(extractor_params, dict):
        raw_moment_source = extractor_params.get("spectral_moment_source")
    resolved_moment_source = resolve_spectral_moment_source(
        None if raw_moment_source is None else str(raw_moment_source)
    )
    raw_sv_top_k = source_metadata.get("sv_top_k")
    if raw_sv_top_k is None and isinstance(extractor_params, dict):
        raw_sv_top_k = extractor_params.get("spectral_sv_top_k")
    resolved_sv_top_k = _sv_top_k_from_feature_names(
        selected_input_feature_names,
        default=int(raw_sv_top_k) if raw_sv_top_k is not None else 8,
    )
    role_buckets = _resolved_output_roles(selected_input_feature_names)
    output_feature_names = _build_output_feature_names(
        role_buckets=role_buckets,
        selected_feature_groups=selected_feature_groups,
        sv_top_k=resolved_sv_top_k,
        spectral_moment_source=resolved_moment_source,
    )

    output_model_names = [str(name) for name in table.model_names]
    owned_feature_names_by_model, selected_leaf_paths = _resolve_model_owned_feature_names(
        root_feature_path=resolved_feature_path,
        root_feature_names=root_feature_names,
        selected_model_names=output_model_names,
    )
    root_feature_index = _unique_index_by_name(
        root_feature_names,
        context=str(resolved_feature_path),
        entity="feature names",
    )
    output_feature_index = _unique_index_by_name(
        output_feature_names,
        context="aggregated output feature names",
        entity="feature names",
    )

    output_features = np.zeros((len(output_model_names), len(output_feature_names)), dtype=np.float32)
    empty_fill_counts = {name: 0 for name in output_feature_names}
    for row_idx, model_name in enumerate(output_model_names):
        owned_selected_feature_names = [
            name
            for name in owned_feature_names_by_model[model_name]
            if name in selected_input_feature_name_set
        ]
        grouped_input_columns: dict[tuple[str, str], dict[str, list[int]]] = {}
        for feature_name in owned_selected_feature_names:
            pair = (
                _role_bucket_for_feature_name(feature_name),
                _emitted_feature_name(feature_name),
            )
            structural_group = _structural_group_for_feature_name(feature_name)
            grouped_input_columns.setdefault(pair, {}).setdefault(structural_group, []).append(
                int(root_feature_index[feature_name])
            )

        row_values = np.asarray(table.features[row_idx], dtype=np.float32)
        for output_feature_name in output_feature_names:
            role_bucket = _role_bucket_for_feature_name(output_feature_name)
            emitted_feature = _emitted_feature_name(output_feature_name)
            grouped_column_indices = grouped_input_columns.get((role_bucket, emitted_feature), {})
            output_col_idx = int(output_feature_index[output_feature_name])
            if not grouped_column_indices:
                empty_fill_counts[output_feature_name] = int(empty_fill_counts[output_feature_name]) + 1
                continue
            grouped_values = {
                structural_group: row_values[np.asarray(column_indices, dtype=np.int64)]
                for structural_group, column_indices in grouped_column_indices.items()
            }
            output_features[row_idx, output_col_idx] = _aggregation_value(
                grouped_values,
                operator=resolved_operator,
            )

    output_feature_path = _resolve_output_feature_path(
        Path(output_filename),
        feature_root=resolved_feature_root,
    )
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_names_path = _default_output_companion_path(output_feature_path, "_model_names.json")
    output_labels_path = _default_output_companion_path(output_feature_path, "_labels.npy")
    output_metadata_path = _default_output_companion_path(output_feature_path, "_metadata.json")
    output_report_path = _default_output_companion_path(output_feature_path, "_aggregation_report.json")

    np.save(output_feature_path, output_features.astype(np.float32, copy=False))
    with open(output_model_names_path, "w", encoding="utf-8") as f:
        json.dump(output_model_names, f, indent=2)

    output_labels = None if table.labels is None else np.asarray(table.labels, dtype=np.int32)
    if output_labels is not None:
        np.save(output_labels_path, output_labels.astype(np.int32, copy=False))
    elif output_labels_path.exists():
        output_labels_path.unlink()

    aggregated_dataset_reference_payload = _build_aggregated_dataset_reference_payload(
        source_feature_path=resolved_feature_path,
        output_model_names=output_model_names,
        selected_leaf_paths=selected_leaf_paths,
    )
    output_metadata = _build_aggregated_metadata(
        source_metadata=source_metadata,
        output_model_names=output_model_names,
        output_feature_names=output_feature_names,
        source_feature_path=resolved_feature_path,
        selected_leaf_paths=selected_leaf_paths,
        aggregation_operator=resolved_operator,
        requested_features=requested_features,
        requested_qv_sum_mode=resolved_requested_qv_mode,
        role_buckets=role_buckets,
        selected_feature_groups=selected_feature_groups,
        sv_top_k=resolved_sv_top_k,
        spectral_moment_source=resolved_moment_source,
        empty_fill_counts=empty_fill_counts,
        selected_input_feature_count=len(selected_input_feature_names),
    )
    write_spectral_metadata(
        output_metadata_path,
        internal_metadata=output_metadata,
        dataset_layouts=dataset_layouts_from_source(
            metadata=output_metadata,
            dataset_reference_payload=aggregated_dataset_reference_payload,
        ),
    )

    output_dataset_reference_report_path = default_dataset_reference_report_path(output_feature_path.parent)
    write_dataset_reference_report(
        output_dataset_reference_report_path,
        aggregated_dataset_reference_payload,
    )

    completed_at = datetime.now(timezone.utc)
    aggregation_report = {
        "timestamp_utc": completed_at.isoformat(),
        "aggregation_started_timestamp_utc": started_at.isoformat(),
        "aggregation_completed_timestamp_utc": completed_at.isoformat(),
        "aggregation_elapsed_seconds": float(perf_counter() - started_perf),
        "feature_extract_root": str(resolved_feature_root),
        "representation_kind": "architecture_independent_aggregation",
        "aggregation_operator": resolved_operator,
        "aggregation_grouping": "role_feature",
        "aggregation_avg_semantics": (
            "mean_of_structural_group_means" if resolved_operator == "avg" else "global_extreme"
        ),
        "aggregation_structural_grouping": "higher_level_block_or_layer",
        "selection": {
            "requested_features": list(requested_features) if requested_features is not None else "all",
            "requested_qv_sum_mode": resolved_requested_qv_mode,
            "role_buckets": list(role_buckets),
            "selected_source_feature_files": [str(path) for path in selected_leaf_paths],
        },
        "stats": {
            "input_rows": int(table.features.shape[0]),
            "input_feature_dim": int(table.features.shape[1]),
            "selected_input_feature_dim": int(len(selected_input_feature_names)),
            "output_rows": int(output_features.shape[0]),
            "output_feature_dim": int(output_features.shape[1]),
            "empty_fill_total": int(sum(empty_fill_counts.values())),
            "empty_fill_columns": int(sum(1 for count in empty_fill_counts.values() if int(count) > 0)),
        },
        "empty_fill_counts": {
            key: int(value)
            for key, value in empty_fill_counts.items()
            if int(value) > 0
        },
        "input": {
            "feature_path": str(resolved_feature_path),
            "model_names_path": str(_default_output_companion_path(resolved_feature_path, "_model_names.json")),
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
        json.dump(json_ready(aggregation_report), f, indent=2)

    return {
        "feature_path": output_feature_path,
        "model_names_path": output_model_names_path,
        "labels_path": output_labels_path if output_labels is not None else None,
        "metadata_path": output_metadata_path,
        "dataset_reference_report_path": output_dataset_reference_report_path,
        "aggregation_report_path": output_report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a spectral feature bundle into an architecture-independent representation by "
            "grouping provenance-owned columns by role bucket and feature family. "
            "For --operator=avg, columns are first averaged within higher-level structural groups "
            "(for example block or layer prefixes) and then averaged across those group means."
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
        "--operator",
        choices=list(SUPPORTED_AGGREGATION_OPERATORS),
        required=True,
        help=(
            "Aggregation operator applied over each role-feature group; avg uses equal-weight "
            "structural-group means while min/max use global extrema over owned columns"
        ),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_EXTRACT_ROOT,
        help="Base directory used to resolve bare feature run names (default: runs/feature_extract)",
    )
    parser.add_argument(
        "--features",
        "--columns",
        dest="features",
        nargs="+",
        default=None,
        help=(
            "Spectral feature groups to keep before aggregation; "
            "omit or pass 'all' to aggregate every available feature family"
        ),
    )
    parser.add_argument(
        "--spectral-qv-sum-mode",
        choices=["none", "append", "only"],
        default="append",
        help="Which q+v-sum columns to keep before aggregation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = aggregate_features(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        operator=args.operator,
        feature_root=args.feature_root,
        features=args.features,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
    )
    print("Feature aggregation complete")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Dataset references: {outputs['dataset_reference_report_path']}")
    print(f"Aggregation report: {outputs['aggregation_report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
