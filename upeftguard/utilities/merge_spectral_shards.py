from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import parse_single_manifest_json
from .paths import default_dataset_root, dataset_root_help
from .serialization import json_ready


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
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload) if isinstance(payload, dict) else {}


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
        "extractor_version",
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


def _load_existing_output_table(output_dir: Path) -> SpectralFeatureTable | None:
    feature_path = output_dir / "spectral_features.npy"
    model_names_path = output_dir / "spectral_model_names.json"
    labels_path = output_dir / "spectral_labels.npy"
    metadata_path = output_dir / "spectral_metadata.json"

    present = [path for path in [feature_path, model_names_path, labels_path, metadata_path] if path.exists()]
    if not present:
        return None
    if not feature_path.exists() or not model_names_path.exists():
        raise FileNotFoundError(
            "Existing output dir has partial spectral artifacts; expected both "
            f"{feature_path} and {model_names_path}"
        )

    return _load_feature_table(
        source=str(output_dir),
        feature_path=feature_path,
        model_names_path=model_names_path,
        labels_path=labels_path,
        metadata_path=metadata_path,
        context=f"existing merged output {output_dir}",
    )


def _merge_feature_tables(
    *,
    base: SpectralFeatureTable,
    incoming: SpectralFeatureTable,
) -> tuple[SpectralFeatureTable, dict[str, Any]]:
    if base.feature_names_inferred or incoming.feature_names_inferred:
        if len(base.feature_names) != len(incoming.feature_names):
            raise ValueError(
                "Cannot merge feature schemas with inferred positional names when dimensions differ. "
                "Provide metadata with explicit feature_names for both sources."
            )
        positional = [f"feature_{i:05d}" for i in range(len(base.feature_names))]
        base_feature_names = positional
        incoming_feature_names = positional
    else:
        base_feature_names = list(base.feature_names)
        incoming_feature_names = list(incoming.feature_names)

    base_row_index = _unique_index_by_name(base.model_names, context=base.source, entity="model names")
    _unique_index_by_name(incoming.model_names, context=incoming.source, entity="model names")
    _unique_index_by_name(base_feature_names, context=f"{base.source} feature schema", entity="feature names")
    _unique_index_by_name(incoming_feature_names, context=f"{incoming.source} feature schema", entity="feature names")

    merged_model_names = list(base.model_names)
    merged_model_names.extend(name for name in incoming.model_names if name not in base_row_index)
    base_feature_name_set = set(base_feature_names)
    merged_feature_names = list(base_feature_names)
    merged_feature_names.extend(name for name in incoming_feature_names if name not in base_feature_name_set)

    merged_row_index = _unique_index_by_name(merged_model_names, context="merged output", entity="model names")
    merged_col_index = _unique_index_by_name(merged_feature_names, context="merged output", entity="feature names")

    merged_features = np.zeros((len(merged_model_names), len(merged_feature_names)), dtype=np.float32)
    coverage = np.zeros_like(merged_features, dtype=bool)

    def _place_values(table: SpectralFeatureTable, table_feature_names: list[str]) -> None:
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
                feat_name = table_feature_names[j]
                raise ValueError(
                    "Conflicting feature values for overlapping row/feature cell: "
                    f"model='{row_name}', feature='{feat_name}', "
                    f"existing={float(current[i, j])}, incoming={float(table_block[i, j])}"
                )

        merged_features[np.ix_(row_idx, col_idx)] = np.where(seen, current, table_block)
        coverage[np.ix_(row_idx, col_idx)] = True

    _place_values(base, base_feature_names)
    _place_values(incoming, incoming_feature_names)

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

    labels_unknown = np.iinfo(np.int32).min
    merged_label_values = np.full(len(merged_model_names), labels_unknown, dtype=np.int32)
    merged_label_known = np.zeros(len(merged_model_names), dtype=bool)

    def _place_labels(table: SpectralFeatureTable) -> None:
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
    merged_labels = merged_label_values if bool(np.all(merged_label_known)) else None

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
        "used_positional_feature_names": bool(base.feature_names_inferred or incoming.feature_names_inferred),
    }

    merged_table = SpectralFeatureTable(
        source="merged_output",
        features=merged_features,
        labels=merged_labels,
        model_names=merged_model_names,
        feature_names=merged_feature_names,
        feature_names_inferred=bool(base.feature_names_inferred or incoming.feature_names_inferred),
        metadata={},
    )
    return merged_table, stats


def merge_spectral_shards(
    *,
    manifest_json: Path,
    dataset_root: Path,
    shard_run_dirs: list[Path],
    output_dir: Path,
    merge_with_existing_output: bool = False,
) -> dict[str, Path | None]:
    if not shard_run_dirs:
        raise ValueError("At least one shard run directory is required")

    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=dataset_root,
        section_key="path",
    )
    expected_names = [item.model_name for item in items]
    expected_index = _unique_index_by_name(
        expected_names,
        context=f"{manifest_json}::path",
        entity="model names",
    )

    bundles = [_load_shard_bundle(Path(run_dir).expanduser().resolve()) for run_dir in shard_run_dirs]

    label_presence = [bundle.labels is not None for bundle in bundles]
    if any(label_presence) and not all(label_presence):
        raise ValueError("Either all shards must include labels or none of them should include labels")

    base_signature = _compatibility_signature(bundles[0])
    for bundle in bundles[1:]:
        sig = _compatibility_signature(bundle)
        if sig != base_signature:
            raise ValueError(
                "Incompatible shard metadata; all shards must use the same spectral feature schema"
            )

    merged_features = np.vstack([bundle.features for bundle in bundles])
    merged_names = [name for bundle in bundles for name in bundle.model_names]
    merged_labels = (
        np.concatenate([bundle.labels for bundle in bundles if bundle.labels is not None], axis=0)
        if all(label_presence)
        else None
    )

    if int(merged_features.shape[0]) != len(expected_names):
        raise ValueError(
            f"Merged row count ({merged_features.shape[0]}) does not match manifest size ({len(expected_names)})"
        )

    merged_index = _unique_index_by_name(
        merged_names,
        context="merged shard model names",
        entity="model names",
    )
    missing = sorted(name for name in expected_index if name not in merged_index)
    extra = sorted(name for name in merged_index if name not in expected_index)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        raise ValueError(
            "Merged shard model names do not match manifest model names: " + "; ".join(details)
        )

    reorder = np.asarray([merged_index[name] for name in expected_names], dtype=np.int64)
    reordered_features = merged_features[reorder]
    reordered_labels = merged_labels[reorder] if merged_labels is not None else None

    incoming_metadata = {
        **dict(bundles[0].metadata),
        "n_models": int(len(expected_names)),
        "feature_dim": int(reordered_features.shape[1]),
        "feature_names": list(bundles[0].feature_names),
        "merge_source_shards": [str(bundle.run_dir) for bundle in bundles],
        "merged_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    incoming_table = SpectralFeatureTable(
        source="incoming_shards",
        features=np.asarray(reordered_features, dtype=np.float32),
        labels=(np.asarray(reordered_labels, dtype=np.int32) if reordered_labels is not None else None),
        model_names=list(expected_names),
        feature_names=list(bundles[0].feature_names),
        feature_names_inferred=bool(bundles[0].feature_names_inferred),
        metadata=incoming_metadata,
    )

    merged_with_existing = False
    merge_stats: dict[str, Any] | None = None
    output_table = incoming_table
    existing_table = None
    if merge_with_existing_output:
        existing_table = _load_existing_output_table(output_dir)
        if existing_table is not None:
            output_table, merge_stats = _merge_feature_tables(
                base=existing_table,
                incoming=incoming_table,
            )
            merged_with_existing = True

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "spectral_features.npy"
    model_names_path = output_dir / "spectral_model_names.json"
    labels_path = output_dir / "spectral_labels.npy"
    metadata_path = output_dir / "spectral_metadata.json"
    merge_report_path = output_dir / "spectral_merge_report.json"

    np.save(feature_path, output_table.features.astype(np.float32, copy=False))
    if output_table.labels is not None:
        np.save(labels_path, output_table.labels.astype(np.int32, copy=False))
    elif labels_path.exists():
        labels_path.unlink()

    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump(output_table.model_names, f, indent=2)

    merged_metadata: dict[str, Any]
    if merged_with_existing and existing_table is not None:
        merged_metadata = {
            **dict(existing_table.metadata),
            "n_models": int(len(output_table.model_names)),
            "feature_dim": int(len(output_table.feature_names)),
            "feature_names": list(output_table.feature_names),
            "merged_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "merge_source_shards": [str(bundle.run_dir) for bundle in bundles],
            "merged_with_existing_output": True,
            "merge_existing_output_dir": str(output_dir),
            "merge_stats": merge_stats or {},
            "incoming_metadata": incoming_table.metadata,
        }
    else:
        merged_metadata = {
            **dict(incoming_table.metadata),
            "n_models": int(len(output_table.model_names)),
            "feature_dim": int(len(output_table.feature_names)),
            "feature_names": list(output_table.feature_names),
            "merged_with_existing_output": False,
        }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(merged_metadata), f, indent=2)

    merge_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": str(manifest_json),
        "dataset_root": str(dataset_root),
        "n_shards": int(len(bundles)),
        "incoming_rows": int(incoming_table.features.shape[0]),
        "incoming_feature_dim": int(incoming_table.features.shape[1]),
        "n_rows": int(output_table.features.shape[0]),
        "feature_dim": int(output_table.features.shape[1]),
        "merged_with_existing_output": bool(merged_with_existing),
        "merge_stats": merge_stats,
        "output": {
            "feature_path": str(feature_path),
            "model_names_path": str(model_names_path),
            "labels_path": str(labels_path) if output_table.labels is not None else None,
            "metadata_path": str(metadata_path),
        },
    }
    with open(merge_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(merge_report), f, indent=2)

    return {
        "feature_path": feature_path,
        "model_names_path": model_names_path,
        "labels_path": labels_path if output_table.labels is not None else None,
        "metadata_path": metadata_path,
        "merge_report_path": merge_report_path,
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
        "--merge-with-existing-output",
        action="store_true",
        help=(
            "If --output-dir already contains spectral_features/model_names, merge incoming rows/features "
            "into that table instead of overwriting it."
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
        merge_with_existing_output=bool(args.merge_with_existing_output),
    )
    print("Merged spectral shards")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Merge report: {outputs['merge_report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
