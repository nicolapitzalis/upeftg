from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ...features.spectral import (
    ordered_block_names_from_feature_names,
    sanitize_spectral_metadata,
)
from ..artifacts.dataset_references import (
    default_dataset_reference_report_path,
    merge_dataset_reference_payloads,
    resolve_dataset_reference_payload_for_artifact,
    write_dataset_reference_report,
)
from .merge_spectral_shards import (
    SpectralFeatureTable,
    _load_feature_table,
    _merge_feature_tables,
    _unique_index_by_name,
    merge_disjoint_feature_tables_zero_fill as shared_merge_disjoint_feature_tables_zero_fill,
    merge_lora_dim_maps,
    merge_skipped_models,
    resolved_qv_sum_mode,
)
from ..artifacts.spectral_metadata import (
    dataset_layouts_from_source,
    merge_dataset_layouts,
    write_spectral_metadata,
)
from ..core.serialization import json_ready

DEFAULT_FEATURE_EXTRACT_ROOT = Path("runs") / "feature_extract"


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


def _resolve_existing_companion_path(feature_path: Path, suffix: str, *, required: bool) -> Path:
    candidates = _candidate_companion_paths(feature_path, suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not find required companion file for {feature_path}. Tried: {joined}"
        )
    return candidates[0]


def _default_output_companion_path(feature_path: Path, suffix: str) -> Path:
    return _candidate_companion_paths(feature_path, suffix)[0]


def _resolve_feature_extract_root(feature_root: Path) -> Path:
    root = feature_root.expanduser()
    if not root.is_absolute():
        root = Path.cwd().resolve() / root
    return root.resolve()


def _looks_like_explicit_path(path_spec: Path) -> bool:
    return path_spec.is_absolute() or len(path_spec.parts) > 1 or path_spec.suffix == ".npy"


def _resolve_input_feature_path(feature_spec: Path, *, feature_root: Path) -> Path:
    candidate = feature_spec.expanduser()
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


def _resolve_output_feature_path(output_spec: Path, *, feature_root: Path) -> Path:
    candidate = output_spec.expanduser()
    if _looks_like_explicit_path(candidate):
        resolved = candidate if candidate.is_absolute() else (Path.cwd().resolve() / candidate)
        output_path = resolved.resolve()
    else:
        output_path = (feature_root / candidate.name / "merged" / "spectral_features.npy").resolve()

    if output_path.suffix == "":
        output_path = output_path.with_name(output_path.name + ".npy")
    elif output_path.suffix != ".npy":
        raise ValueError(f"--output-filename must end with .npy: {output_path}")
    return output_path


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
        _resolve_input_feature_path(Path(path), feature_root=resolved_feature_root)
        for path in feature_paths
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
        merged_table, merge_stats = _merge_feature_tables(
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
                "metadata_path": (
                    str(entry["metadata_path"]) if entry["metadata_path"] is not None else None
                ),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two spectral feature files. Companion model_names/metadata/labels files "
            "are resolved automatically. Bare names resolve under runs/feature_extract/<name>/merged."
        )
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs=2,
        metavar=("FILE1", "FILE2"),
        required=True,
        help="Two run names or explicit spectral feature .npy files to merge",
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
        help="Base directory used to resolve bare run names (default: runs/feature_extract)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = merge_feature_files(
        feature_paths=list(args.merge),
        output_filename=args.output_filename,
        feature_root=args.feature_root,
    )
    print("Merged feature files")
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
