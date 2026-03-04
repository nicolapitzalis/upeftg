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
from .serialization import json_ready


@dataclass(frozen=True)
class SpectralShardBundle:
    run_dir: Path
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    metadata: dict[str, Any]


def _unique_index_by_name(names: list[str], *, context: str) -> dict[str, int]:
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
            f"Duplicate model names in {context}; cannot align merged rows safely. "
            f"Examples: {preview}"
        )
    return index


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload) if isinstance(payload, dict) else {}


def _load_shard_bundle(run_dir: Path) -> SpectralShardBundle:
    feature_path = run_dir / "features" / "spectral_features.npy"
    model_names_path = run_dir / "features" / "spectral_model_names.json"
    labels_path = run_dir / "features" / "spectral_labels.npy"
    metadata_path = run_dir / "features" / "spectral_metadata.json"

    if not feature_path.exists():
        raise FileNotFoundError(f"Missing shard feature file: {feature_path}")
    if not model_names_path.exists():
        raise FileNotFoundError(f"Missing shard model names file: {model_names_path}")

    features = np.asarray(np.load(feature_path), dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Shard features must be 2D: {feature_path} shape={features.shape}")

    with open(model_names_path, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(features.shape[0]):
        raise ValueError(
            f"Shard row mismatch for {run_dir}: features rows={features.shape[0]} "
            f"but model names={len(model_names)}"
        )

    labels = None
    if labels_path.exists():
        labels = np.asarray(np.load(labels_path), dtype=np.int32)
        if int(labels.shape[0]) != int(features.shape[0]):
            raise ValueError(
                f"Shard label mismatch for {run_dir}: labels rows={labels.shape[0]} "
                f"but features rows={features.shape[0]}"
            )

    metadata = _load_metadata(metadata_path)
    return SpectralShardBundle(
        run_dir=run_dir,
        features=features,
        labels=labels,
        model_names=model_names,
        metadata=metadata,
    )


def _compatibility_signature(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "extractor",
        "extractor_version",
        "resolved_features",
        "sv_top_k",
        "feature_dim",
        "feature_names",
        "block_names",
        "delta_schema_version",
    ]
    return {key: metadata.get(key) for key in keys}


def merge_spectral_shards(
    *,
    manifest_json: Path,
    dataset_root: Path,
    shard_run_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Path | None]:
    if not shard_run_dirs:
        raise ValueError("At least one shard run directory is required")

    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=dataset_root,
        section_key="path",
    )
    expected_names = [item.model_name for item in items]
    expected_index = _unique_index_by_name(expected_names, context=f"{manifest_json}::path")

    bundles = [_load_shard_bundle(Path(run_dir).expanduser().resolve()) for run_dir in shard_run_dirs]

    label_presence = [bundle.labels is not None for bundle in bundles]
    if any(label_presence) and not all(label_presence):
        raise ValueError("Either all shards must include labels or none of them should include labels")

    base_signature = _compatibility_signature(bundles[0].metadata)
    for bundle in bundles[1:]:
        sig = _compatibility_signature(bundle.metadata)
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

    merged_index = _unique_index_by_name(merged_names, context="merged shard model names")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "spectral_features.npy"
    model_names_path = output_dir / "spectral_model_names.json"
    labels_path = output_dir / "spectral_labels.npy"
    metadata_path = output_dir / "spectral_metadata.json"
    merge_report_path = output_dir / "spectral_merge_report.json"

    np.save(feature_path, reordered_features.astype(np.float32, copy=False))
    if reordered_labels is not None:
        np.save(labels_path, reordered_labels.astype(np.int32, copy=False))
    elif labels_path.exists():
        labels_path.unlink()

    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump(expected_names, f, indent=2)

    merged_metadata = {
        **dict(bundles[0].metadata),
        "n_models": int(len(expected_names)),
        "merge_source_shards": [str(bundle.run_dir) for bundle in bundles],
        "merged_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(merged_metadata), f, indent=2)

    merge_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": str(manifest_json),
        "dataset_root": str(dataset_root),
        "n_shards": int(len(bundles)),
        "n_rows": int(reordered_features.shape[0]),
        "feature_dim": int(reordered_features.shape[1]),
        "output": {
            "feature_path": str(feature_path),
            "model_names_path": str(model_names_path),
            "labels_path": str(labels_path) if reordered_labels is not None else None,
            "metadata_path": str(metadata_path),
        },
    }
    with open(merge_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(merge_report), f, indent=2)

    return {
        "feature_path": feature_path,
        "model_names_path": model_names_path,
        "labels_path": labels_path if reordered_labels is not None else None,
        "metadata_path": metadata_path,
        "merge_report_path": merge_report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge sharded spectral extraction runs")
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
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
