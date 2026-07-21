"""Metadata construction for aggregated feature artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...features.spectral import sanitize_spectral_metadata, spectral_extractor_params
from ..metadata.merge import resolved_qv_sum_mode
from ..provenance.subsets import _load_source_payload
from .layout import _architecture_block_sort_key
from ..provenance.datasets import finalize_dataset_reference_payload


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
    spectral_attention_granularity: str,
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
    metadata["spectral_attention_granularity"] = str(spectral_attention_granularity)
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
    metadata["aggregation_requested_features"] = list(requested_features) if requested_features is not None else "all"
    metadata["aggregation_requested_qv_sum_mode"] = str(requested_qv_sum_mode)
    metadata["aggregation_selected_input_feature_count"] = int(selected_input_feature_count)
    metadata["aggregation_empty_fill_total"] = int(sum(empty_fill_counts.values()))
    metadata["aggregation_empty_fill_counts"] = {
        key: int(value) for key, value in empty_fill_counts.items() if int(value) > 0
    }
    metadata["extractor_params"] = spectral_extractor_params(
        {
            "spectral_features": list(selected_feature_groups),
            "spectral_sv_top_k": int(sv_top_k),
            "spectral_moment_source": str(spectral_moment_source),
            "spectral_qv_sum_mode": str(resolved_output_qv_mode),
            "spectral_entrywise_delta_mode": metadata.get("spectral_entrywise_delta_mode"),
            "spectral_attention_granularity": str(spectral_attention_granularity),
        }
    )
    return metadata


def _build_layer_sequence_metadata(
    *,
    source_metadata: dict[str, Any],
    output_model_names: list[str],
    output_feature_names: list[str],
    output_group_names: list[list[str]],
    depth_labels: list[str],
    source_feature_path: Path,
    selected_leaf_paths: list[Path],
    aggregation_operator: str,
    requested_features: list[str] | None,
    requested_qv_sum_mode: str,
    spectral_attention_granularity: str,
    slot_names: list[str],
    attention_kind_names: list[str],
    adapter_block_names: list[str],
    emitted_feature_names: list[str],
    selected_feature_groups: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
    tensor_shape: tuple[int, int, int, int],
    empty_fill_counts: dict[str, int],
    selected_input_feature_count: int,
    total_padding_groups: int,
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

    output_block_names = [f"slot.{slot_name}" for slot_name in slot_names]
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
    metadata["spectral_attention_granularity"] = str(spectral_attention_granularity)
    metadata["representation_kind"] = "architecture_independent_layer_sequence_aggregation"
    metadata["aggregation_operator"] = None
    metadata["aggregation_operator_ignored"] = bool(str(aggregation_operator).strip())
    metadata["aggregation_layout"] = "architecture_block_layer_attention_adapter_feature"
    metadata["aggregation_grouping"] = "architecture_block_layer_attention_adapter_feature"
    metadata["aggregation_padding_strategy"] = "canonical_axis_mask"
    metadata["tensor_axes"] = ["model", "architecture_layer", "attention_adapter", "feature"]
    metadata["tensor_shape"] = [int(x) for x in tensor_shape]
    metadata["depth_axis_kind"] = "architecture_block_layer"
    metadata["depth_labels"] = list(depth_labels)
    metadata["architecture_block_names"] = sorted(
        {str(depth_label).split(".", 1)[0] for depth_label in depth_labels if str(depth_label).strip()},
        key=lambda name: _architecture_block_sort_key(name),
    )
    metadata["max_architecture_layers"] = int(tensor_shape[1])
    metadata["max_structural_groups"] = int(tensor_shape[1])
    metadata["slot_axis_kind"] = "attention_kind_adapter_block"
    metadata["attention_kind_names"] = list(attention_kind_names)
    metadata["adapter_block_names"] = list(adapter_block_names)
    metadata["slot_names"] = list(slot_names)
    metadata["max_attention_adapter_slots"] = int(tensor_shape[2])
    metadata["max_structural_group_slots"] = int(tensor_shape[2])
    metadata["emitted_feature_names"] = list(emitted_feature_names)
    metadata["aggregation_source_feature_file"] = str(source_feature_path)
    metadata["aggregation_selected_source_feature_files"] = [str(path) for path in selected_leaf_paths]
    metadata["aggregation_requested_features"] = list(requested_features) if requested_features is not None else "all"
    metadata["aggregation_requested_qv_sum_mode"] = str(requested_qv_sum_mode)
    metadata["aggregation_selected_input_feature_count"] = int(selected_input_feature_count)
    metadata["aggregation_empty_fill_total"] = int(sum(empty_fill_counts.values()))
    metadata["aggregation_empty_fill_counts"] = {
        key: int(value) for key, value in empty_fill_counts.items() if int(value) > 0
    }
    metadata["aggregation_total_padding_groups"] = int(total_padding_groups)
    metadata["structural_group_names"] = [list(names) for names in output_group_names]
    metadata["group_names"] = [list(names) for names in output_group_names]
    metadata["extractor_params"] = spectral_extractor_params(
        {
            "spectral_features": list(selected_feature_groups),
            "spectral_sv_top_k": int(sv_top_k),
            "spectral_moment_source": str(spectral_moment_source),
            "spectral_qv_sum_mode": str(resolved_output_qv_mode),
            "spectral_entrywise_delta_mode": metadata.get("spectral_entrywise_delta_mode"),
            "spectral_attention_granularity": str(spectral_attention_granularity),
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
    return finalize_dataset_reference_payload(
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
