from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ...features.spectral import (
    resolve_spectral_attention_granularity,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
)
from ..provenance.datasets import (
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from ..schema import normalize_requested_features as _normalize_requested_features
from ..provenance.subsets import (
    _load_table_from_feature_path,
    _resolve_model_owned_feature_names,
)
from ..metadata.spectral import dataset_layouts_from_source, write_spectral_metadata
from ..paths import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    default_output_companion_path as _default_output_companion_path,
    resolve_feature_extract_root as _resolve_feature_extract_root,
    resolve_input_feature_path as _resolve_input_feature_path,
    resolve_output_feature_path as _resolve_output_feature_path,
)
from ..tables import unique_index_by_name as _unique_index_by_name
from .metadata import (
    _build_aggregated_dataset_reference_payload,
    _build_aggregated_metadata,
    _build_layer_sequence_metadata,
)
from .layout import (
    GROUP_MASK_SUFFIX,
    GROUP_NAMES_SUFFIX,
    VALUE_MASK_SUFFIX,
    _aggregation_value,
    _build_output_emitted_feature_names,
    _build_output_feature_names,
    _canonical_depth_label_for_feature_name,
    _canonical_slot_name_for_feature_name,
    _emitted_feature_name,
    _filter_selected_input_feature_names,
    _normalize_aggregation_layout,
    _normalize_aggregation_operator,
    _ordered_canonical_depth_labels,
    _ordered_canonical_slot_names,
    _ordered_feature_groups,
    _resolved_output_roles,
    _role_bucket_for_feature_name,
    _structural_group_for_feature_name,
    _sv_top_k_from_feature_names,
)
from ...utilities.core.serialization import json_ready


def aggregate_features(
    *,
    feature_file: Path,
    output_filename: Path,
    operator: str = "avg",
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
    features: list[str] | tuple[str, ...] | None = None,
    spectral_qv_sum_mode: str = "append",
    spectral_attention_granularity: str | None = None,
    layout: str = "flat",
) -> dict[str, Path | None]:
    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()

    resolved_operator = _normalize_aggregation_operator(operator)
    resolved_layout = _normalize_aggregation_layout(layout)
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
    extractor_params = source_metadata.get("extractor_params")
    raw_attention_granularity = spectral_attention_granularity
    if raw_attention_granularity is None:
        raw_attention_granularity = source_metadata.get("spectral_attention_granularity")
    if raw_attention_granularity is None and isinstance(extractor_params, dict):
        raw_attention_granularity = extractor_params.get("spectral_attention_granularity")
    resolved_attention_granularity = resolve_spectral_attention_granularity(
        None if raw_attention_granularity is None else str(raw_attention_granularity)
    )
    raw_moment_source = source_metadata.get("spectral_moment_source")
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
    emitted_feature_names = _build_output_emitted_feature_names(
        selected_feature_groups=selected_feature_groups,
        sv_top_k=resolved_sv_top_k,
        spectral_moment_source=resolved_moment_source,
    )
    if resolved_layout == "flat":
        role_buckets = _resolved_output_roles(selected_input_feature_names)
        slot_names: list[str] = []
        depth_labels: list[str] = []
        attention_kind_names: list[str] = []
        adapter_block_names: list[str] = []
        output_feature_names = _build_output_feature_names(
            output_block_names=[f"role.{role}" for role in role_buckets],
            selected_feature_groups=selected_feature_groups,
            sv_top_k=resolved_sv_top_k,
            spectral_moment_source=resolved_moment_source,
        )
    else:
        role_buckets = []
        depth_labels = _ordered_canonical_depth_labels(selected_input_feature_names)
        slot_names = _ordered_canonical_slot_names(selected_input_feature_names)
        attention_kind_names = []
        seen_attention_kinds: set[str] = set()
        adapter_block_names = []
        seen_adapter_blocks: set[str] = set()
        for slot_name in slot_names:
            attention_kind, _, adapter_block = str(slot_name).partition(".")
            if attention_kind not in seen_attention_kinds:
                seen_attention_kinds.add(attention_kind)
                attention_kind_names.append(attention_kind)
            if adapter_block and adapter_block not in seen_adapter_blocks:
                seen_adapter_blocks.add(adapter_block)
                adapter_block_names.append(adapter_block)
        output_feature_names = _build_output_feature_names(
            output_block_names=[f"slot.{slot_name}" for slot_name in slot_names],
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

    output_feature_path = _resolve_output_feature_path(
        Path(output_filename),
        feature_root=resolved_feature_root,
    )
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_names_path = _default_output_companion_path(output_feature_path, "_model_names.json")
    output_labels_path = _default_output_companion_path(output_feature_path, "_labels.npy")
    output_metadata_path = _default_output_companion_path(output_feature_path, "_metadata.json")
    output_report_path = _default_output_companion_path(output_feature_path, "_aggregation_report.json")
    output_group_mask_path = _default_output_companion_path(output_feature_path, GROUP_MASK_SUFFIX)
    output_value_mask_path = _default_output_companion_path(output_feature_path, VALUE_MASK_SUFFIX)
    output_group_names_path = _default_output_companion_path(output_feature_path, GROUP_NAMES_SUFFIX)

    output_group_mask: np.ndarray | None = None
    output_value_mask: np.ndarray | None = None
    output_group_names: list[list[str]] | None = None
    total_padding_groups = 0

    if resolved_layout == "flat":
        output_feature_index = _unique_index_by_name(
            output_feature_names,
            context="aggregated output feature names",
            entity="feature names",
        )
        output_features = np.zeros((len(output_model_names), len(output_feature_names)), dtype=np.float32)
        empty_fill_counts = {name: 0 for name in output_feature_names}
        for row_idx, model_name in enumerate(output_model_names):
            owned_selected_feature_names = [
                name for name in owned_feature_names_by_model[model_name] if name in selected_input_feature_name_set
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
    else:
        output_features = np.zeros(
            (
                len(output_model_names),
                len(depth_labels),
                len(slot_names),
                len(emitted_feature_names),
            ),
            dtype=np.float32,
        )
        output_group_mask = np.zeros((len(output_model_names), len(depth_labels)), dtype=bool)
        output_value_mask = np.zeros(
            (
                len(output_model_names),
                len(depth_labels),
                len(slot_names),
                len(emitted_feature_names),
            ),
            dtype=bool,
        )
        empty_fill_counts = {name: 0 for name in output_feature_names}
        depth_index = {depth_label: idx for idx, depth_label in enumerate(depth_labels)}
        slot_index = {slot_name: idx for idx, slot_name in enumerate(slot_names)}
        emitted_feature_index = {name: idx for idx, name in enumerate(emitted_feature_names)}
        total_valid_groups = 0
        output_group_names = []

        for row_idx, model_name in enumerate(output_model_names):
            owned_selected_feature_names = [
                name for name in owned_feature_names_by_model[model_name] if name in selected_input_feature_name_set
            ]
            depth_set = {
                _canonical_depth_label_for_feature_name(feature_name) for feature_name in owned_selected_feature_names
            }
            ordered_depths = [depth_label for depth_label in depth_labels if depth_label in depth_set]
            output_group_names.append(list(ordered_depths))
            total_padding_groups += int(len(depth_labels) - len(ordered_depths))
            total_valid_groups += int(len(ordered_depths))
            for depth_label in ordered_depths:
                output_group_mask[row_idx, int(depth_index[depth_label])] = True
            row_values = np.asarray(table.features[row_idx], dtype=np.float32)
            for feature_name in owned_selected_feature_names:
                depth_label = _canonical_depth_label_for_feature_name(feature_name)
                depth_idx = int(depth_index[depth_label])
                slot_name = _canonical_slot_name_for_feature_name(feature_name)
                slot_idx = int(slot_index[slot_name])
                emitted_feature = _emitted_feature_name(feature_name)
                feature_idx = int(emitted_feature_index[emitted_feature])
                if output_value_mask[row_idx, depth_idx, slot_idx, feature_idx]:
                    raise ValueError(
                        "Layer-sequence layout encountered duplicate canonical placement for "
                        f"model={model_name!r}, depth_label={depth_label!r}, "
                        f"slot_name={slot_name!r}, emitted_feature={emitted_feature!r}"
                    )
                output_features[row_idx, depth_idx, slot_idx, feature_idx] = float(
                    row_values[int(root_feature_index[feature_name])]
                )
                output_value_mask[row_idx, depth_idx, slot_idx, feature_idx] = True

        for slot_idx, slot_name in enumerate(slot_names):
            output_block_name = f"slot.{slot_name}"
            for feature_idx, emitted_feature in enumerate(emitted_feature_names):
                flat_feature_name = f"{output_block_name}.{emitted_feature}"
                populated = int(output_value_mask[:, :, slot_idx, feature_idx].sum())
                empty_fill_counts[flat_feature_name] = int(total_valid_groups - populated)

    np.save(output_feature_path, output_features.astype(np.float32, copy=False))
    with open(output_model_names_path, "w", encoding="utf-8") as f:
        json.dump(output_model_names, f, indent=2)
    if output_group_mask is not None:
        np.save(output_group_mask_path, output_group_mask.astype(np.bool_, copy=False))
    else:
        output_group_mask_path.unlink(missing_ok=True)
    if output_value_mask is not None:
        np.save(output_value_mask_path, output_value_mask.astype(np.bool_, copy=False))
    else:
        output_value_mask_path.unlink(missing_ok=True)
    if output_group_names is not None:
        with open(output_group_names_path, "w", encoding="utf-8") as f:
            json.dump(output_group_names, f, indent=2)
    else:
        output_group_names_path.unlink(missing_ok=True)

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
    if resolved_layout == "flat":
        output_metadata = _build_aggregated_metadata(
            source_metadata=source_metadata,
            output_model_names=output_model_names,
            output_feature_names=output_feature_names,
            source_feature_path=resolved_feature_path,
            selected_leaf_paths=selected_leaf_paths,
            aggregation_operator=resolved_operator,
            requested_features=requested_features,
            requested_qv_sum_mode=resolved_requested_qv_mode,
            spectral_attention_granularity=resolved_attention_granularity,
            role_buckets=role_buckets,
            selected_feature_groups=selected_feature_groups,
            sv_top_k=resolved_sv_top_k,
            spectral_moment_source=resolved_moment_source,
            empty_fill_counts=empty_fill_counts,
            selected_input_feature_count=len(selected_input_feature_names),
        )
    else:
        assert output_group_names is not None
        output_metadata = _build_layer_sequence_metadata(
            source_metadata=source_metadata,
            output_model_names=output_model_names,
            output_feature_names=output_feature_names,
            output_group_names=output_group_names,
            depth_labels=depth_labels,
            source_feature_path=resolved_feature_path,
            selected_leaf_paths=selected_leaf_paths,
            aggregation_operator=resolved_operator,
            requested_features=requested_features,
            requested_qv_sum_mode=resolved_requested_qv_mode,
            spectral_attention_granularity=resolved_attention_granularity,
            slot_names=slot_names,
            attention_kind_names=attention_kind_names,
            adapter_block_names=adapter_block_names,
            emitted_feature_names=emitted_feature_names,
            selected_feature_groups=selected_feature_groups,
            sv_top_k=resolved_sv_top_k,
            spectral_moment_source=resolved_moment_source,
            tensor_shape=tuple(int(x) for x in output_features.shape),
            empty_fill_counts=empty_fill_counts,
            selected_input_feature_count=len(selected_input_feature_names),
            total_padding_groups=total_padding_groups,
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
        "representation_kind": (
            "architecture_independent_aggregation"
            if resolved_layout == "flat"
            else "architecture_independent_layer_sequence_aggregation"
        ),
        "aggregation_operator": resolved_operator if resolved_layout == "flat" else None,
        "selection": {
            "requested_features": list(requested_features) if requested_features is not None else "all",
            "requested_qv_sum_mode": resolved_requested_qv_mode,
            "requested_attention_granularity": resolved_attention_granularity,
            "emitted_feature_names": list(emitted_feature_names),
            "selected_source_feature_files": [str(path) for path in selected_leaf_paths],
        },
        "stats": {
            "input_rows": int(table.features.shape[0]),
            "input_feature_dim": int(table.features.shape[1]),
            "selected_input_feature_dim": int(len(selected_input_feature_names)),
            "output_rows": int(output_features.shape[0]),
            "output_feature_dim": int(len(output_feature_names)),
            "empty_fill_total": int(sum(empty_fill_counts.values())),
            "empty_fill_columns": int(sum(1 for count in empty_fill_counts.values() if int(count) > 0)),
        },
        "empty_fill_counts": {key: int(value) for key, value in empty_fill_counts.items() if int(value) > 0},
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
    if resolved_layout == "flat":
        aggregation_report["selection"]["role_buckets"] = list(role_buckets)
        aggregation_report["aggregation_grouping"] = "role_feature"
        aggregation_report["aggregation_avg_semantics"] = (
            "mean_of_structural_group_means" if resolved_operator == "avg" else "global_extreme"
        )
        aggregation_report["aggregation_structural_grouping"] = "higher_level_block_or_layer"
        aggregation_report["stats"]["output_feature_dim"] = int(output_features.shape[1])
    else:
        aggregation_report["selection"]["slot_names"] = list(slot_names)
        aggregation_report["selection"]["depth_labels"] = list(depth_labels)
        aggregation_report["selection"]["attention_kind_names"] = list(attention_kind_names)
        aggregation_report["selection"]["adapter_block_names"] = list(adapter_block_names)
        aggregation_report["aggregation_layout"] = "architecture_block_layer_attention_adapter_feature"
        aggregation_report["aggregation_grouping"] = "architecture_block_layer_attention_adapter_feature"
        aggregation_report["aggregation_operator_ignored"] = True
        aggregation_report["depth_axis_kind"] = "architecture_block_layer"
        aggregation_report["slot_axis_kind"] = "attention_kind_adapter_block"
        aggregation_report["aggregation_padding_strategy"] = "canonical_axis_mask"
        aggregation_report["tensor_axes"] = ["model", "architecture_layer", "attention_adapter", "feature"]
        aggregation_report["tensor_shape"] = [int(x) for x in output_features.shape]
        aggregation_report["stats"]["max_architecture_layers"] = int(output_features.shape[1])
        aggregation_report["stats"]["max_structural_groups"] = int(output_features.shape[1])
        aggregation_report["stats"]["max_attention_adapter_slots"] = int(output_features.shape[2])
        aggregation_report["stats"]["max_structural_group_slots"] = int(output_features.shape[2])
        aggregation_report["stats"]["output_tensor_shape"] = [int(x) for x in output_features.shape]
        aggregation_report["stats"]["padding_group_total"] = int(total_padding_groups)
        aggregation_report["output"]["group_mask_path"] = str(output_group_mask_path)
        aggregation_report["output"]["value_mask_path"] = str(output_value_mask_path)
        aggregation_report["output"]["group_names_path"] = str(output_group_names_path)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(aggregation_report), f, indent=2)

    return {
        "feature_path": output_feature_path,
        "model_names_path": output_model_names_path,
        "labels_path": output_labels_path if output_labels is not None else None,
        "metadata_path": output_metadata_path,
        "dataset_reference_report_path": output_dataset_reference_report_path,
        "aggregation_report_path": output_report_path,
        "group_mask_path": output_group_mask_path if output_group_mask is not None else None,
        "value_mask_path": output_value_mask_path if output_value_mask is not None else None,
        "group_names_path": output_group_names_path if output_group_names is not None else None,
    }
