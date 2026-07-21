from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..metadata.merge import (
    merge_lora_dim_maps,
    merge_skipped_models,
    resolved_qv_sum_mode,
)
from ..io import (
    load_feature_table as _load_feature_table,
)
from ..tables import (
    FeatureTable as SpectralFeatureTable,
    unique_index_by_name as _unique_index_by_name,
)

from ...features.spectral import (
    ordered_block_names_from_feature_names,
    sanitize_spectral_metadata,
)
from ..provenance.datasets import (
    build_dataset_reference_payload_from_items,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from ..metadata.spectral import (
    dataset_layouts_from_source,
    merge_dataset_layouts,
    write_spectral_metadata,
)
from ...utilities.core.manifest import parse_single_manifest_json, resolve_manifest_path
from ...utilities.core.serialization import json_ready


@dataclass(frozen=True)
class SpectralShardBundle:
    run_dir: Path
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    feature_names: list[str]
    feature_names_inferred: bool
    metadata: dict[str, Any]


def _summarize_shard_inputs(shard_run_dirs: list[Path]) -> dict[str, Any]:
    return {"shard_count": int(len(shard_run_dirs))}


def _rebuild_output_metadata(
    *,
    source_metadata: dict[str, Any],
    output_table: SpectralFeatureTable,
    metadata_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = sanitize_spectral_metadata(source_metadata)
    metadata["n_models"] = int(len(output_table.model_names))
    metadata["feature_dim"] = int(len(output_table.feature_names))
    metadata["feature_names"] = list(output_table.feature_names)
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


def _load_shard_bundle(run_dir: Path) -> SpectralShardBundle:
    table = _load_feature_table(
        source=str(run_dir),
        feature_path=run_dir / "spectral_features.npy",
        model_names_path=run_dir / "spectral_model_names.json",
        labels_path=run_dir / "spectral_labels.npy",
        metadata_path=run_dir / "spectral_metadata.json",
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
            raise ValueError("Incompatible shard metadata; all shards must use the same spectral feature schema")
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
            "Merged shard model names do not match manifest model names: " + "; ".join([f"extra={extra[:5]}"])
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
    return payload


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
    shard_summary = _summarize_shard_inputs([bundle.run_dir for bundle in bundles])
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
        source_artifacts=[str(manifest_json)],
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
        ),
        "manifest_json": str(manifest_json),
        "dataset_root": str(dataset_root),
        "n_shards": int(len(bundles)),
        "manifest_model_count": int(len(expected_names)),
        "skipped_model_count": int(len(skipped_models)),
        "skipped_models": skipped_models,
        "incoming_rows": int(incoming_table.features.shape[0]),
        "incoming_feature_dim": int(incoming_table.features.shape[1]),
        "n_rows": int(output_table.features.shape[0]),
        "feature_dim": int(output_table.features.shape[1]),
        "merged_with_existing_output": False,
        "merge_stats": None,
        "shards": shard_summary,
        "output": {
            "feature_path": str(output_paths["feature_path"]),
            "model_names_path": str(output_paths["model_names_path"]),
            "labels_path": (str(output_paths["labels_path"]) if output_table.labels is not None else None),
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
