from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..paths import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    default_output_companion_path as _default_output_companion_path,
    resolve_existing_companion_path as _resolve_existing_companion_path,
    resolve_feature_extract_root as _resolve_feature_extract_root,
    resolve_input_feature_path as _resolve_input_feature_path,
    resolve_output_feature_path as _resolve_output_feature_path,
)
from ..io import load_feature_table as _load_feature_table
from ..metadata.merge import (
    merge_lora_dim_maps,
    merge_skipped_models,
    resolved_qv_sum_mode,
)

from ..tables import (
    FeatureTable as SpectralFeatureTable,
    merge_feature_tables_dense,
    merge_disjoint_feature_tables_zero_fill as shared_merge_disjoint_feature_tables_zero_fill,
    unique_index_by_name as _unique_index_by_name,
)
from ...features.spectral import (
    ordered_block_names_from_feature_names,
    sanitize_spectral_metadata,
)
from ..provenance.datasets import (
    default_dataset_reference_report_path,
    merge_dataset_reference_payloads,
    resolve_dataset_reference_payload_for_artifact,
    write_dataset_reference_report,
)
from ..metadata.spectral import (
    dataset_layouts_from_source,
    merge_dataset_layouts,
    write_spectral_metadata,
)
from ...utilities.core.serialization import json_ready


def _build_merged_metadata(
    *,
    source_metadata: dict[str, Any],
    model_names: list[str],
    feature_names: list[str],
    feature_names_inferred: bool,
    metadata_sources: list[dict[str, Any]],
    merge_sources: list[Path],
) -> dict[str, Any]:
    metadata = sanitize_spectral_metadata(source_metadata)
    metadata["n_models"] = int(len(model_names))
    metadata["feature_dim"] = int(len(feature_names))
    metadata["feature_names"] = list(feature_names)
    metadata["merge_source_feature_files"] = [str(path) for path in merge_sources]
    metadata["merged_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["merged_by"] = "merge_feature_files"
    skipped_models = merge_skipped_models(*metadata_sources)
    if skipped_models:
        metadata["skipped_model_count"] = int(len(skipped_models))
        metadata["skipped_models"] = skipped_models

    if feature_names_inferred:
        return metadata

    try:
        block_names = ordered_block_names_from_feature_names(feature_names)
    except ValueError:
        block_names = []
    if not block_names:
        return metadata

    metadata["block_names"] = list(block_names)
    metadata["n_blocks"] = int(len(block_names))
    metadata["base_block_names"] = [name for name in block_names if ".qv_sum" not in name]
    metadata["qv_sum_block_names"] = [name for name in block_names if ".qv_sum" in name]
    metadata["spectral_qv_sum_mode"] = resolved_qv_sum_mode(block_names)

    try:
        dim_map = merge_lora_dim_maps(*metadata_sources)
    except ValueError:
        dim_map = {}
    if dim_map and all(block_name in dim_map for block_name in block_names):
        metadata["lora_adapter_dims"] = [dim_map[block_name] for block_name in block_names]
    else:
        metadata.pop("lora_adapter_dims", None)
        metadata.pop("base_lora_adapter_dims", None)
        metadata.pop("qv_sum_lora_adapter_dims", None)

    return metadata


def _merge_disjoint_feature_tables_zero_fill(
    *,
    base: SpectralFeatureTable,
    incoming: SpectralFeatureTable,
) -> tuple[SpectralFeatureTable, dict[str, Any]]:
    merged = shared_merge_disjoint_feature_tables_zero_fill(
        base=base,
        incoming=incoming,
        index_by_name=_unique_index_by_name,
        overlap_error_prefix=(
            "Zero-fill feature merge requires disjoint model rows across inputs; "
            f"found overlap between {base.source} and {incoming.source}"
        ),
    )
    merged_table = SpectralFeatureTable(
        source="merged_output",
        features=merged.features,
        labels=merged.labels,
        model_names=merged.model_names,
        feature_names=merged.feature_names,
        feature_names_inferred=merged.feature_names_inferred,
        metadata={},
    )
    return merged_table, merged.stats


def merge_feature_files(
    *,
    feature_paths: list[Path],
    output_filename: Path,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
) -> dict[str, Path | None]:
    if len(feature_paths) != 2:
        raise ValueError(f"Expected exactly two feature files, received {len(feature_paths)}")

    merge_started_at = datetime.now(timezone.utc)
    merge_started_perf = perf_counter()
    resolved_feature_root = _resolve_feature_extract_root(feature_root)
    resolved_feature_paths = [
        _resolve_input_feature_path(Path(path), feature_root=resolved_feature_root) for path in feature_paths
    ]

    input_artifacts: list[dict[str, Path | None]] = []
    tables = []
    for index, feature_path in enumerate(resolved_feature_paths, start=1):
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file {index}: {feature_path}")
        if feature_path.suffix != ".npy":
            raise ValueError(f"Feature file {index} must be a .npy file: {feature_path}")

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

        table = _load_feature_table(
            source=str(feature_path),
            feature_path=feature_path,
            model_names_path=model_names_path,
            labels_path=labels_path,
            metadata_path=metadata_path,
            context=f"feature file {index}",
        )
        tables.append(table)
        input_artifacts.append(
            {
                "feature_path": feature_path,
                "model_names_path": model_names_path,
                "labels_path": labels_path if labels_path.exists() else None,
                "metadata_path": metadata_path if metadata_path.exists() else None,
            }
        )

    try:
        merged_table, merge_stats = merge_feature_tables_dense(
            base=tables[0],
            incoming=tables[1],
        )
        merge_stats = {"merge_mode": "dense", **dict(merge_stats)}
    except ValueError as exc:
        missing_cells_error = "Merged output would contain missing feature cells"
        shared_models = sorted(set(tables[0].model_names) & set(tables[1].model_names))
        if missing_cells_error not in str(exc) or shared_models:
            raise
        merged_table, merge_stats = _merge_disjoint_feature_tables_zero_fill(
            base=tables[0],
            incoming=tables[1],
        )

    output_feature_path = _resolve_output_feature_path(
        Path(output_filename),
        feature_root=resolved_feature_root,
    )
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)

    output_model_names_path = _default_output_companion_path(output_feature_path, "_model_names.json")
    output_labels_path = _default_output_companion_path(output_feature_path, "_labels.npy")
    output_metadata_path = _default_output_companion_path(output_feature_path, "_metadata.json")
    output_report_path = _default_output_companion_path(output_feature_path, "_merge_report.json")
    output_dataset_reference_report_path = default_dataset_reference_report_path(output_feature_path.parent)

    np.save(output_feature_path, merged_table.features.astype(np.float32, copy=False))
    with open(output_model_names_path, "w", encoding="utf-8") as f:
        json.dump(merged_table.model_names, f, indent=2)

    if merged_table.labels is not None:
        np.save(output_labels_path, merged_table.labels.astype(np.int32, copy=False))
    elif output_labels_path.exists():
        output_labels_path.unlink()

    source_reference_payloads = []
    source_reference_gaps: list[str] = []
    for feature_path in resolved_feature_paths:
        try:
            source_reference_payloads.append(resolve_dataset_reference_payload_for_artifact(feature_path))
        except Exception as exc:
            source_reference_gaps.append(f"Could not resolve dataset provenance for {feature_path}: {exc}")
    dataset_reference_payload = merge_dataset_reference_payloads(
        payloads=source_reference_payloads,
        artifact_kind="merge_feature_files",
        artifact_model_count=int(merged_table.features.shape[0]),
        source_artifacts=[str(path) for path in resolved_feature_paths],
        provenance_gaps=source_reference_gaps,
    )
    merged_metadata = _build_merged_metadata(
        source_metadata=dict(tables[0].metadata),
        model_names=list(merged_table.model_names),
        feature_names=list(merged_table.feature_names),
        feature_names_inferred=bool(merged_table.feature_names_inferred),
        metadata_sources=[dict(table.metadata) for table in tables],
        merge_sources=resolved_feature_paths,
    )
    write_spectral_metadata(
        output_metadata_path,
        internal_metadata=merged_metadata,
        dataset_layouts=merge_dataset_layouts(
            output_dataset_reference_payload=dataset_reference_payload,
            source_layouts=[
                dataset_layouts_from_source(
                    metadata=table.metadata,
                    dataset_reference_payload=payload,
                )
                for table, payload in zip(tables, source_reference_payloads)
            ],
        ),
    )
    write_dataset_reference_report(output_dataset_reference_report_path, dataset_reference_payload)

    merge_completed_at = datetime.now(timezone.utc)
    merge_report = {
        "timestamp_utc": merge_completed_at.isoformat(),
        "merge_started_timestamp_utc": merge_started_at.isoformat(),
        "merge_completed_timestamp_utc": merge_completed_at.isoformat(),
        "merge_elapsed_seconds": float(perf_counter() - merge_started_perf),
        "feature_extract_root": str(resolved_feature_root),
        "inputs": [
            {
                "feature_path": str(entry["feature_path"]),
                "model_names_path": str(entry["model_names_path"]),
                "labels_path": str(entry["labels_path"]) if entry["labels_path"] is not None else None,
                "metadata_path": (str(entry["metadata_path"]) if entry["metadata_path"] is not None else None),
            }
            for entry in input_artifacts
        ],
        "merge_stats": merge_stats,
        "output": {
            "feature_path": str(output_feature_path),
            "model_names_path": str(output_model_names_path),
            "labels_path": str(output_labels_path) if merged_table.labels is not None else None,
            "metadata_path": str(output_metadata_path),
            "dataset_reference_report_path": str(output_dataset_reference_report_path),
            "merge_report_path": str(output_report_path),
        },
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(merge_report), f, indent=2)

    return {
        "feature_path": output_feature_path,
        "model_names_path": output_model_names_path,
        "labels_path": output_labels_path if merged_table.labels is not None else None,
        "metadata_path": output_metadata_path,
        "merge_report_path": output_report_path,
        "dataset_reference_report_path": output_dataset_reference_report_path,
    }


def _resolve_feature_artifacts(spec: Path) -> tuple[Path, Path, Path, Path]:
    path = spec.expanduser()
    if not path.is_absolute():
        path = (Path.cwd().resolve() / path).resolve()
    else:
        path = path.resolve()

    if path.is_dir():
        feature_path = path / "spectral_features.npy"
        model_names_path = path / "spectral_model_names.json"
        labels_path = path / "spectral_labels.npy"
        metadata_path = path / "spectral_metadata.json"
        return feature_path, model_names_path, labels_path, metadata_path

    feature_path = path
    model_names_path = _resolve_existing_companion_path(feature_path, "_model_names.json", required=True)
    labels_path = _resolve_existing_companion_path(feature_path, "_labels.npy", required=False)
    metadata_path = _resolve_existing_companion_path(feature_path, "_metadata.json", required=False)
    return feature_path, model_names_path, labels_path, metadata_path


def _load_feature_table_from_spec(spec: Path, *, context: str) -> tuple[SpectralFeatureTable, Path]:
    feature_path, model_names_path, labels_path, metadata_path = _resolve_feature_artifacts(spec)
    table = _load_feature_table(
        source=str(feature_path),
        feature_path=feature_path,
        model_names_path=model_names_path,
        labels_path=labels_path,
        metadata_path=metadata_path,
        context=context,
    )
    return table, feature_path


def finalize_schema_group_merge(
    *,
    schema_report_path: Path,
    output_dir: Path,
) -> dict[str, Path | None]:
    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()

    report_payload = json.loads(Path(schema_report_path).expanduser().resolve().read_text(encoding="utf-8"))
    groups = list(report_payload.get("groups", []))
    if not groups:
        raise ValueError(f"No groups found in schema partition report: {schema_report_path}")

    sources: list[tuple[str, SpectralFeatureTable, Path]] = []
    for group in groups:
        group_id = str(group["group_id"])
        merged_output_dir = Path(str(group["merged_output_dir"]))
        table, feature_path = _load_feature_table_from_spec(
            merged_output_dir,
            context=f"schema group {group_id}",
        )
        sources.append((group_id, table, feature_path))

    merged_table = sources[0][1]
    merge_stats: list[dict[str, Any]] = []
    for source_id, table, _feature_path in sources[1:]:
        merged_table, stats = _merge_disjoint_feature_tables_zero_fill(
            base=merged_table,
            incoming=table,
        )
        stats["incoming_source_id"] = source_id
        merge_stats.append(stats)

    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd().resolve() / output_path).resolve()
    else:
        output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    feature_path = output_path / "spectral_features.npy"
    model_names_path = output_path / "spectral_model_names.json"
    labels_path = output_path / "spectral_labels.npy"
    metadata_path = output_path / "spectral_metadata.json"
    merge_report_path = output_path / "schema_group_merge_report.json"
    dataset_reference_report_path = default_dataset_reference_report_path(output_path)

    np.save(feature_path, merged_table.features.astype(np.float32, copy=False))
    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump(merged_table.model_names, f, indent=2)
    if merged_table.labels is not None:
        np.save(labels_path, np.asarray(merged_table.labels, dtype=np.int32))
    elif labels_path.exists():
        labels_path.unlink()

    source_reference_payloads = []
    source_reference_gaps: list[str] = []
    source_layouts: list[list[dict[str, Any]]] = []
    for source_id, table, source_feature_path in sources:
        try:
            payload = resolve_dataset_reference_payload_for_artifact(source_feature_path)
            source_reference_payloads.append(payload)
            source_layouts.append(
                dataset_layouts_from_source(
                    metadata=table.metadata,
                    dataset_reference_payload=payload,
                )
            )
        except Exception as exc:
            source_reference_gaps.append(f"Could not resolve dataset provenance for {source_id}: {exc}")
    dataset_reference_payload = merge_dataset_reference_payloads(
        payloads=source_reference_payloads,
        artifact_kind="finalize_schema_group_merge",
        artifact_model_count=int(merged_table.features.shape[0]),
        source_artifacts=[str(source_feature_path) for _, _, source_feature_path in sources],
        provenance_gaps=source_reference_gaps,
    )
    metadata = _build_merged_metadata(
        source_metadata=dict(sources[0][1].metadata),
        model_names=list(merged_table.model_names),
        feature_names=list(merged_table.feature_names),
        feature_names_inferred=bool(merged_table.feature_names_inferred),
        metadata_sources=[dict(table.metadata) for _, table, _ in sources],
        merge_sources=[feature_path for _, _, feature_path in sources],
    )
    metadata["merged_by"] = "finalize_schema_group_merge"
    metadata["schema_partition_report"] = str(Path(schema_report_path).expanduser().resolve())
    metadata["schema_group_ids"] = [source_id for source_id, _, _ in sources]
    write_spectral_metadata(
        metadata_path,
        internal_metadata=metadata,
        dataset_layouts=merge_dataset_layouts(
            output_dataset_reference_payload=dataset_reference_payload,
            source_layouts=source_layouts,
        ),
    )
    write_dataset_reference_report(dataset_reference_report_path, dataset_reference_payload)

    completed_at = datetime.now(timezone.utc)
    merge_report = {
        "timestamp_utc": completed_at.isoformat(),
        "merge_started_timestamp_utc": started_at.isoformat(),
        "merge_completed_timestamp_utc": completed_at.isoformat(),
        "merge_elapsed_seconds": float(perf_counter() - started_perf),
        "schema_partition_report": str(Path(schema_report_path).expanduser().resolve()),
        "inputs": [
            {
                "source_id": source_id,
                "feature_path": str(feature_path),
            }
            for source_id, _, feature_path in sources
        ],
        "merge_steps": merge_stats,
        "output": {
            "feature_path": str(feature_path),
            "model_names_path": str(model_names_path),
            "labels_path": str(labels_path) if merged_table.labels is not None else None,
            "metadata_path": str(metadata_path),
            "dataset_reference_report_path": str(dataset_reference_report_path),
            "merge_report_path": str(merge_report_path),
        },
    }
    with open(merge_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(merge_report), f, indent=2)

    return {
        "feature_path": feature_path,
        "model_names_path": model_names_path,
        "labels_path": labels_path if merged_table.labels is not None else None,
        "metadata_path": metadata_path,
        "merge_report_path": merge_report_path,
        "dataset_reference_report_path": dataset_reference_report_path,
    }
