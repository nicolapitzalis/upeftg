from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from ..features.spectral import (
    build_spectral_feature_names,
    ordered_block_names_from_feature_names,
    rank_normalized_feature_group,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
    sanitize_spectral_metadata,
)
from ..supervised.pipeline import run_supervised_pipeline
from ..unsupervised.analysis import LoadedFeatureBundle, load_feature_bundle
from ..utilities.artifacts.aggregate_features import aggregate_features
from ..utilities.artifacts.dataset_references import (
    _build_incomplete_payload_from_model_names,
    _finalize_payload,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from ..utilities.artifacts.spectral_metadata import load_spectral_metadata, write_spectral_metadata
from ..utilities.core.manifest import (
    ManifestItem,
    parse_joint_manifest_json_by_model_name,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
)
from ..utilities.core.paths import default_dataset_root
from ..utilities.core.run_context import create_run_context
from ..utilities.core.serialization import json_ready


SCRIPT_VERSION = "1.0.0"
PIPELINE_NAME = "rank_normalization_study"
DEFAULT_FEATURE_ROOT = Path("runs") / "feature_extract"
DEFAULT_OUTPUT_ROOT = Path("runs")
DEFAULT_SOURCE_FEATURE_FILE = Path("list2_features-merged")
DEFAULT_BASELINE_FEATURE_FILE = Path("list2_features-merged-cnn")
DEFAULT_BASELINE_REFERENCE_RUN = "list2_features_cnn"
DEFAULT_BASELINE_TRANSFER_RESULTS_ROOT = Path("runs") / "supervised" / "zero_shot_cnn"
DEFAULT_TUNING_MANIFEST = Path("manifests") / "others" / "list2.json"
DEFAULT_ZERO_SHOT_MANIFEST_ROOT = Path("manifests") / "zero_shots" / "rank_wise"
DEFAULT_ZERO_SHOT_MANIFEST_FILTER = "llama2_7b_tbh_zero_shot_r256_to_rank"
DEFAULT_RANK_NORM_FEATURE_OUTPUT_RUN = "list2_features-merged-rank_norm"
DEFAULT_RANK_NORM_CNN_OUTPUT_RUN = "list2_features-merged-rank_norm-cnn"
DEFAULT_RANK_NORM_REFERENCE_RUN = "list2_features_rank_norm_cnn"
DEFAULT_RANK_NORM_ZERO_SHOT_PREFIX = "zero_shot_cnn_rank_norm"
DEFAULT_RAW_RANK_FEATURE_OUTPUT_RUN = "list2_features-merged-raw_rank"
DEFAULT_RAW_RANK_CNN_OUTPUT_RUN = "list2_features-merged-raw_rank-cnn"
DEFAULT_RAW_RANK_REFERENCE_RUN = "list2_features_raw_rank_cnn"
DEFAULT_RAW_RANK_ZERO_SHOT_PREFIX = "zero_shot_cnn_raw_rank"
DEFAULT_STUDY_ARMS = ("rank_norm", "raw_rank")
DEFAULT_REFERENCE_TUNING_EXECUTOR = "slurm_array"
DEFAULT_REFERENCE_RANDOM_STATE = 42
DEFAULT_REFERENCE_SCORE_PERCENTILES = (90.0, 95.0, 99.0)
DEFAULT_STUDY_STAGE = "prepare_reference"
DEFAULT_SLURM_PARTITION = "extra"
DEFAULT_SLURM_LOG_DIR = Path("logs")
DEFAULT_CONDA_SH = Path("/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh")
DEFAULT_CONDA_ENV = "upeftg"

_FIXED_RANK_RE = re.compile(r"(?:^|_)rank(\d+(?:\.\d+)?)(?:_|$)", re.IGNORECASE)
_TARGET_RANK_RE = re.compile(r"_to_rank(\d+(?:\.\d+)?)$", re.IGNORECASE)

_DERIVED_EMITTED_FEATURE_SOURCES: dict[str, str | None] = {
    "block_rank": None,
    "energy_per_rank": "energy",
    "l1_norm_per_rank": "l1_norm",
    "sv_l1_norm_per_rank": "sv_l1_norm",
    "l2_norm_per_sqrt_rank": "l2_norm",
    "mean_abs_per_rank": "mean_abs",
    "sv_mean_abs_per_rank": "sv_mean_abs",
    "stable_rank_frac": "stable_rank",
    "normalized_spectral_entropy": "spectral_entropy",
    "effective_rank_frac": "effective_rank",
}
_RANK_NORM_STUDY_FEATURE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "mean_abs": (),
}


@dataclass(frozen=True)
class StudyArmSpec:
    name: str
    selected_features: tuple[str, ...]
    feature_output_run: str
    cnn_output_run: str
    reference_run_id: str
    zero_shot_run_prefix: str
    baseline_read_only: bool = False


@dataclass(frozen=True)
class ZeroShotExpectation:
    manifest_path: Path
    manifest_name: str
    target_rank: int | None


@dataclass(frozen=True)
class ReferenceRunSettings:
    reference_run_dir: Path
    model_name: str
    train_split_percent: int
    split_by_folder: bool
    calibration_split_percent: int | None
    accepted_fprs: list[float] | None
    cv_folds: int
    cv_random_states: list[int]
    spectral_sv_top_k: int
    spectral_moment_source: str
    spectral_qv_sum_mode: str
    spectral_entrywise_delta_mode: str
    baseline_selected_features: list[str]
    score_percentiles: list[float]
    random_state: int


@dataclass(frozen=True)
class DerivedBundleArtifacts:
    arm_name: str
    output_dir: Path
    feature_path: Path
    model_names_path: Path
    labels_path: Path | None
    metadata_path: Path
    dataset_reference_report_path: Path
    derivation_report_path: Path
    feature_names: list[str]
    model_names: list[str]
    row_indices: np.ndarray
    requested_model_count: int
    missing_model_names: list[str]


@dataclass(frozen=True)
class StudyConfig:
    stage: str
    source_feature_file: Path
    baseline_feature_file: Path
    baseline_reference_run: str
    baseline_transfer_results_root: Path
    tuning_manifest: Path
    zero_shot_manifest_root: Path
    zero_shot_manifest_filter: str
    feature_root: Path
    output_root: Path
    dataset_root: Path
    run_id: str | None
    rank_norm_feature_output_run: str
    rank_norm_cnn_output_run: str
    rank_norm_reference_run: str
    rank_norm_zero_shot_prefix: str
    raw_rank_feature_output_run: str
    raw_rank_cnn_output_run: str
    raw_rank_reference_run: str
    raw_rank_zero_shot_prefix: str
    arms: tuple[str, ...]
    reference_cnn_hyperparams: Path | None
    reference_tuning_executor: str
    reference_n_jobs: int
    reference_dry_run: bool
    zero_shot_dry_run: bool
    slurm_partition: str
    slurm_log_dir: Path
    conda_sh: Path
    conda_env: str
    skip_derive: bool
    skip_aggregate: bool
    skip_reference: bool
    skip_zero_shot_launch: bool
    report_only: bool


def _resolve_path(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd().resolve() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return resolved


def _progress(message: str) -> None:
    print(f"[rank-normalization-study] {message}", flush=True)


def _resolve_feature_bundle_path(feature_spec: Path, *, feature_root: Path) -> Path:
    candidate = feature_spec.expanduser()
    local_candidate = candidate if candidate.is_absolute() else (Path.cwd().resolve() / candidate)
    if local_candidate.exists():
        resolved = local_candidate.resolve()
        return (resolved / "spectral_features.npy") if resolved.is_dir() else resolved

    resolved_feature_root = _resolve_path(feature_root)
    if candidate.suffix == ".npy":
        return (resolved_feature_root / candidate).resolve()
    return (resolved_feature_root / candidate / "merged" / "spectral_features.npy").resolve()


def _resolve_supervised_run_dir(run_spec: str, *, output_root: Path) -> Path:
    raw = str(run_spec).strip()
    if not raw:
        raise ValueError("Reference run id/path cannot be empty")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or "/" in raw:
        return _resolve_path(candidate)
    return (_resolve_path(output_root) / "supervised" / raw).resolve()


def _normalized_zero_shot_run_name(run_name: str) -> str:
    return str(run_name).replace("_zero_shot_cnn_", "_zero_shot_")


def parse_fixed_rank_from_model_name(model_name: str) -> int:
    match = _FIXED_RANK_RE.search(str(model_name))
    if match is None:
        raise ValueError(f"Could not parse a fixed-rank token from model name: {model_name}")
    return int(float(match.group(1)))


def zero_shot_target_rank_from_name(name: str) -> int | None:
    match = _TARGET_RANK_RE.search(str(name))
    if match is None:
        return None
    return int(float(match.group(1)))


def block_rank_scale_for_sample(*, model_name: str, block_name: str) -> int:
    base_rank = parse_fixed_rank_from_model_name(model_name)
    if ".qv_sum" in str(block_name):
        return int(base_rank * 2)
    return int(base_rank)


def _ordered_manifest_paths(manifest_root: Path, manifest_filter: str) -> list[Path]:
    resolved_root = _resolve_path(manifest_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Zero-shot manifest root not found: {resolved_root}")
    manifests = sorted(path.resolve() for path in resolved_root.rglob("*.json"))
    selected = [path for path in manifests if manifest_filter in path.name]
    if not selected:
        raise ValueError(
            f"No zero-shot manifests matched filter '{manifest_filter}' under {resolved_root}"
        )
    return selected


def collect_zero_shot_expectations(*, manifest_root: Path, manifest_filter: str) -> list[ZeroShotExpectation]:
    expectations: list[ZeroShotExpectation] = []
    for manifest_path in _ordered_manifest_paths(manifest_root, manifest_filter):
        manifest_name = manifest_path.stem
        expectations.append(
            ZeroShotExpectation(
                manifest_path=manifest_path,
                manifest_name=manifest_name,
                target_rank=zero_shot_target_rank_from_name(manifest_name),
            )
        )
    return expectations


def _manifest_items_by_model_name(manifest_path: Path) -> list[ManifestItem]:
    resolved_manifest = resolve_manifest_path(manifest_path)
    with open(resolved_manifest, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON must be an object: {resolved_manifest}")
    if "train" in payload and "infer" in payload:
        train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=resolved_manifest)
        return [*train_items, *infer_items]
    return parse_single_manifest_json_by_model_name(manifest_path=resolved_manifest)


def collect_manifest_model_names(
    *,
    tuning_manifest: Path,
    zero_shot_expectations: Sequence[ZeroShotExpectation],
) -> tuple[list[str], list[Path]]:
    ordered_names: list[str] = []
    seen: set[str] = set()
    manifests = [resolve_manifest_path(tuning_manifest), *[item.manifest_path for item in zero_shot_expectations]]
    for manifest_path in manifests:
        for item in _manifest_items_by_model_name(manifest_path):
            if item.model_name in seen:
                continue
            seen.add(item.model_name)
            ordered_names.append(str(item.model_name))
    if not ordered_names:
        raise ValueError("Manifest-driven selection resolved to zero model names")
    return ordered_names, manifests


def _subset_dataset_reference_payload(
    *,
    bundle: LoadedFeatureBundle,
    output_model_names: Sequence[str],
    artifact_kind: str,
    source_artifacts: list[str],
) -> dict[str, Any]:
    payload = bundle.dataset_reference_payload
    labels_by_name = {
        str(model_name): (int(bundle.labels[idx]) if bundle.labels is not None else None)
        for idx, model_name in enumerate(bundle.model_names)
    }
    if not isinstance(payload, dict):
        return _build_incomplete_payload_from_model_names(
            model_names=[str(x) for x in output_model_names],
            labels_by_name=labels_by_name,
            artifact_kind=artifact_kind,
            artifact_model_count=len(output_model_names),
            provenance_gaps=["Dataset-reference payload was unavailable; built an incomplete fallback payload"],
            source_artifacts=source_artifacts,
        )

    model_index = payload.get("model_index")
    if not isinstance(model_index, dict) or not model_index:
        return _build_incomplete_payload_from_model_names(
            model_names=[str(x) for x in output_model_names],
            labels_by_name=labels_by_name,
            artifact_kind=artifact_kind,
            artifact_model_count=len(output_model_names),
            provenance_gaps=[
                "Dataset-reference payload did not include model_index; built an incomplete fallback payload"
            ],
            source_artifacts=source_artifacts,
        )

    filtered_model_index = {
        str(model_name): dict(model_index[str(model_name)])
        for model_name in output_model_names
        if str(model_name) in model_index and isinstance(model_index[str(model_name)], dict)
    }
    missing = [str(model_name) for model_name in output_model_names if str(model_name) not in filtered_model_index]
    gaps = [str(x) for x in payload.get("provenance_gaps", []) if str(x).strip()]
    if missing:
        preview = ", ".join(missing[:5])
        gaps.append(
            "Source dataset-reference payload was missing "
            f"{len(missing)} model(s) for the rank-normalization study bundle. Examples: {preview}"
        )

    dataset_root_raw = payload.get("dataset_root")
    dataset_root = Path(str(dataset_root_raw)).expanduser() if dataset_root_raw else None
    manifest_json_raw = payload.get("manifest_json")
    manifest_json = Path(str(manifest_json_raw)).expanduser() if manifest_json_raw else None
    return _finalize_payload(
        artifact_kind=artifact_kind,
        model_index=filtered_model_index,
        artifact_model_count=len(output_model_names),
        manifest_json=manifest_json,
        dataset_root=dataset_root,
        source_artifacts=source_artifacts,
        provenance_gaps=gaps,
        is_complete=bool(payload.get("is_complete", True))
        and not missing
        and len(filtered_model_index) == len(output_model_names),
    )


def _subset_dataset_layouts(
    *,
    source_layouts: Any,
    dataset_reference_payload: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not isinstance(source_layouts, list):
        return None

    groups = dataset_reference_payload.get("dataset_groups")
    if not isinstance(groups, list):
        return [dict(entry) for entry in source_layouts if isinstance(entry, dict)]

    source_by_name = {
        str(entry.get("dataset_name") or "unknown"): dict(entry)
        for entry in source_layouts
        if isinstance(entry, dict)
    }
    subset: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        dataset_name = str(group.get("dataset_name") or "unknown")
        merged = dict(source_by_name.get(dataset_name, {}))
        merged["dataset_name"] = dataset_name
        merged["sample_count"] = int(group.get("sample_count", 0))
        subset.append(merged)
    return subset


def _select_available_rows(
    *,
    bundle: LoadedFeatureBundle,
    selected_model_names: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    name_to_index: dict[str, int] = {}
    for idx, model_name in enumerate(bundle.model_names):
        if model_name in name_to_index:
            raise ValueError(f"Duplicate model name in source bundle: {model_name}")
        name_to_index[model_name] = int(idx)

    wanted = {str(name) for name in selected_model_names}
    missing = [str(name) for name in selected_model_names if str(name) not in name_to_index]
    row_indices = np.asarray(
        [idx for idx, model_name in enumerate(bundle.model_names) if model_name in wanted],
        dtype=np.int64,
    )
    if row_indices.size == 0:
        preview = ", ".join(str(name) for name in list(selected_model_names)[:5])
        raise ValueError(
            "Manifest-driven selection has no overlap with the source bundle model-name set. "
            f"Examples: {preview}"
        )
    output_model_names = [str(bundle.model_names[int(idx)]) for idx in row_indices.tolist()]
    return row_indices, output_model_names, missing


def build_arm_feature_groups(*, baseline_features: Sequence[str]) -> tuple[list[str], list[str]]:
    baseline = [str(feature).strip() for feature in baseline_features]
    rank_norm: list[str] = []
    seen_rank_norm: set[str] = set()
    for feature in baseline:
        overrides = _RANK_NORM_STUDY_FEATURE_OVERRIDES.get(feature)
        normalized_features = (
            overrides
            if overrides is not None
            else (rank_normalized_feature_group(feature),)
        )
        for normalized_feature in normalized_features:
            if normalized_feature in seen_rank_norm:
                continue
            seen_rank_norm.add(normalized_feature)
            rank_norm.append(normalized_feature)
    raw_rank = [*baseline, "block_rank"]
    return rank_norm, raw_rank


def _selected_block_names_from_bundle(
    *,
    bundle: LoadedFeatureBundle,
    spectral_qv_sum_mode: str,
) -> list[str]:
    raw_block_names = bundle.metadata.get("block_names")
    if isinstance(raw_block_names, list) and raw_block_names:
        all_block_names = [str(x) for x in raw_block_names]
    else:
        all_block_names = ordered_block_names_from_feature_names([str(x) for x in bundle.feature_names])

    resolved_qv_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    if resolved_qv_mode == "none":
        return [name for name in all_block_names if ".qv_sum" not in name]
    if resolved_qv_mode == "only":
        return [name for name in all_block_names if ".qv_sum" in name]
    return list(all_block_names)


def _derived_source_suffix(output_suffix: str) -> str | None:
    return _DERIVED_EMITTED_FEATURE_SOURCES.get(str(output_suffix))


def _derive_column_from_source(
    *,
    output_feature_name: str,
    selected_model_names: Sequence[str],
    source_features: np.ndarray,
    source_feature_index: dict[str, int],
) -> np.ndarray:
    if output_feature_name in source_feature_index:
        return np.asarray(
            source_features[:, int(source_feature_index[output_feature_name])],
            dtype=np.float32,
        )

    block_name, _, suffix = str(output_feature_name).rpartition(".")
    if not block_name or not suffix:
        raise ValueError(f"Invalid spectral feature name: {output_feature_name}")

    source_suffix = _derived_source_suffix(suffix)
    rank_scales = np.asarray(
        [
            block_rank_scale_for_sample(model_name=str(model_name), block_name=block_name)
            for model_name in selected_model_names
        ],
        dtype=np.float32,
    )
    if suffix == "block_rank":
        return rank_scales.astype(np.float32, copy=False)

    if source_suffix is None:
        raise ValueError(
            f"Could not derive requested feature '{output_feature_name}' from the source bundle"
        )

    source_feature_name = f"{block_name}.{source_suffix}"
    if source_feature_name not in source_feature_index:
        raise ValueError(
            "The source spectral bundle is missing a column needed to derive the requested feature: "
            f"{source_feature_name}"
        )

    source_values = np.asarray(
        source_features[:, int(source_feature_index[source_feature_name])],
        dtype=np.float32,
    )
    if suffix in {"energy_per_rank", "l1_norm_per_rank", "sv_l1_norm_per_rank", "mean_abs_per_rank", "sv_mean_abs_per_rank", "stable_rank_frac", "effective_rank_frac"}:
        return np.asarray(source_values / rank_scales, dtype=np.float32)
    if suffix == "l2_norm_per_sqrt_rank":
        return np.asarray(source_values / np.sqrt(rank_scales), dtype=np.float32)
    if suffix == "normalized_spectral_entropy":
        output = np.zeros_like(source_values, dtype=np.float32)
        valid = rank_scales > 1.0
        output[valid] = source_values[valid] / np.log(rank_scales[valid])
        return output

    raise ValueError(f"Unsupported derived feature suffix: {suffix}")


def _build_derived_bundle_metadata(
    *,
    source_metadata: dict[str, Any],
    output_model_names: Sequence[str],
    output_feature_names: Sequence[str],
    selected_block_names: Sequence[str],
    selected_features: Sequence[str],
    sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    study_arm: str,
    source_feature_file: Path,
    manifest_paths: Sequence[Path],
    source_row_count: int,
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

    extractor_params = dict(metadata.get("extractor_params", {}))
    extractor_params["spectral_features"] = [str(x) for x in selected_features]
    extractor_params["spectral_sv_top_k"] = int(sv_top_k)
    extractor_params["spectral_moment_source"] = str(spectral_moment_source)
    extractor_params["spectral_qv_sum_mode"] = str(spectral_qv_sum_mode)
    metadata["extractor_params"] = extractor_params
    metadata["n_models"] = int(len(output_model_names))
    metadata["feature_dim"] = int(len(output_feature_names))
    metadata["feature_names"] = [str(x) for x in output_feature_names]
    metadata["block_names"] = [str(x) for x in selected_block_names]
    metadata["n_blocks"] = int(len(selected_block_names))
    metadata["base_block_names"] = [name for name in selected_block_names if ".qv_sum" not in name]
    metadata["qv_sum_block_names"] = [name for name in selected_block_names if ".qv_sum" in name]
    metadata["resolved_features"] = [str(x) for x in selected_features]
    metadata["sv_top_k"] = int(sv_top_k)
    metadata["spectral_moment_source"] = str(spectral_moment_source)
    metadata["spectral_qv_sum_mode"] = str(spectral_qv_sum_mode)
    metadata["merge_source_feature_files"] = [str(source_feature_file.resolve())]
    metadata["rank_normalization_study_arm"] = str(study_arm)
    metadata["rank_normalization_study_source_feature_file"] = str(source_feature_file.resolve())
    metadata["rank_normalization_study_manifest_paths"] = [str(path.resolve()) for path in manifest_paths]
    metadata["rank_normalization_study_source_row_count"] = int(source_row_count)
    metadata["rank_normalization_study_note"] = (
        "This bundle was derived offline from an existing merged spectral bundle for the rank-normalization study."
    )
    return metadata


def derive_arm_feature_bundle(
    *,
    bundle: LoadedFeatureBundle,
    arm: StudyArmSpec,
    feature_root: Path,
    selected_model_names: Sequence[str],
    manifest_paths: Sequence[Path],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
) -> DerivedBundleArtifacts:
    row_indices, output_model_names, missing_model_names = _select_available_rows(
        bundle=bundle,
        selected_model_names=selected_model_names,
    )
    selected_rows = np.asarray(bundle.features[row_indices], dtype=np.float32)

    source_feature_names = [str(x) for x in bundle.feature_names]
    source_feature_index = {name: idx for idx, name in enumerate(source_feature_names)}
    selected_block_names = _selected_block_names_from_bundle(
        bundle=bundle,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
    )
    output_feature_names = build_spectral_feature_names(
        block_names=list(selected_block_names),
        selected_features=list(arm.selected_features),
        sv_top_k=int(spectral_sv_top_k),
        spectral_moment_source=str(spectral_moment_source),
        shorten_block_names=False,
    )
    output_columns = [
        _derive_column_from_source(
            output_feature_name=feature_name,
            selected_model_names=output_model_names,
            source_features=selected_rows,
            source_feature_index=source_feature_index,
        )
        for feature_name in output_feature_names
    ]
    output_features = np.stack(output_columns, axis=1).astype(np.float32, copy=False)

    output_dir = _resolve_path(feature_root) / arm.feature_output_run / "merged"
    if output_dir.resolve() == bundle.feature_file.parent.resolve():
        raise ValueError("Derived study bundle must not overwrite the source bundle")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "spectral_features.npy"
    model_names_path = output_dir / "spectral_model_names.json"
    labels_path = output_dir / "spectral_labels.npy"
    metadata_path = output_dir / "spectral_metadata.json"
    derivation_report_path = output_dir / "rank_normalization_bundle_report.json"

    np.save(feature_path, output_features)
    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump(output_model_names, f, indent=2)

    resolved_labels_path: Path | None
    if bundle.labels is not None:
        np.save(labels_path, np.asarray(bundle.labels[row_indices], dtype=np.int32))
        resolved_labels_path = labels_path
    else:
        resolved_labels_path = None
        labels_path.unlink(missing_ok=True)

    dataset_reference_payload = _subset_dataset_reference_payload(
        bundle=bundle,
        output_model_names=output_model_names,
        artifact_kind=f"rank_normalization_study_{arm.name}",
        source_artifacts=[str(bundle.feature_file.resolve()), *[str(path.resolve()) for path in manifest_paths]],
    )
    metadata = _build_derived_bundle_metadata(
        source_metadata=dict(bundle.metadata),
        output_model_names=output_model_names,
        output_feature_names=output_feature_names,
        selected_block_names=selected_block_names,
        selected_features=arm.selected_features,
        sv_top_k=int(spectral_sv_top_k),
        spectral_moment_source=str(spectral_moment_source),
        spectral_qv_sum_mode=str(spectral_qv_sum_mode),
        study_arm=arm.name,
        source_feature_file=bundle.feature_file,
        manifest_paths=manifest_paths,
        source_row_count=int(bundle.features.shape[0]),
    )
    dataset_layouts = _subset_dataset_layouts(
        source_layouts=bundle.metadata.get("dataset_layouts"),
        dataset_reference_payload=dataset_reference_payload,
    )
    write_spectral_metadata(
        metadata_path,
        internal_metadata=metadata,
        dataset_layouts=dataset_layouts,
    )
    dataset_reference_report_path = default_dataset_reference_report_path(output_dir)
    write_dataset_reference_report(dataset_reference_report_path, dataset_reference_payload)

    derivation_report = {
        "script_version": SCRIPT_VERSION,
        "study_arm": arm.name,
        "source_feature_file": str(bundle.feature_file.resolve()),
        "source_row_count": int(bundle.features.shape[0]),
        "selected_row_count": int(len(output_model_names)),
        "requested_model_count": int(len(selected_model_names)),
        "missing_model_count": int(len(missing_model_names)),
        "missing_model_examples": [str(x) for x in missing_model_names[:10]],
        "selected_feature_groups": list(arm.selected_features),
        "output_feature_dim": int(output_features.shape[1]),
        "manifest_paths": [str(path.resolve()) for path in manifest_paths],
        "output": {
            "feature_path": str(feature_path),
            "model_names_path": str(model_names_path),
            "labels_path": str(resolved_labels_path) if resolved_labels_path is not None else None,
            "metadata_path": str(metadata_path),
            "dataset_reference_report_path": str(dataset_reference_report_path),
        },
    }
    with open(derivation_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(derivation_report), f, indent=2)

    return DerivedBundleArtifacts(
        arm_name=arm.name,
        output_dir=output_dir,
        feature_path=feature_path,
        model_names_path=model_names_path,
        labels_path=resolved_labels_path,
        metadata_path=metadata_path,
        dataset_reference_report_path=dataset_reference_report_path,
        derivation_report_path=derivation_report_path,
        feature_names=list(output_feature_names),
        model_names=output_model_names,
        row_indices=row_indices,
        requested_model_count=int(len(selected_model_names)),
        missing_model_names=[str(x) for x in missing_model_names],
    )


def aggregate_arm_feature_bundle(
    *,
    arm: StudyArmSpec,
    feature_root: Path,
    feature_path: Path,
    selected_features: Sequence[str],
    spectral_qv_sum_mode: str,
) -> dict[str, Path | None]:
    output_feature_path = _resolve_path(feature_root) / arm.cnn_output_run / "merged" / "spectral_features.npy"
    return aggregate_features(
        feature_file=feature_path,
        output_filename=output_feature_path,
        feature_root=feature_root,
        features=list(selected_features),
        spectral_qv_sum_mode=str(spectral_qv_sum_mode),
        layout="layer_sequence",
    )


def load_reference_run_settings(
    *,
    baseline_reference_run: str,
    output_root: Path,
) -> ReferenceRunSettings:
    reference_run_dir = _resolve_supervised_run_dir(baseline_reference_run, output_root=output_root)
    run_config_path = reference_run_dir / "run_config.json"
    tuning_manifest_path = reference_run_dir / "reports" / "tuning_manifest.json"
    baseline_report_path = reference_run_dir / "reports" / "supervised_report.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Baseline reference run_config.json not found: {run_config_path}")
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Baseline tuning_manifest.json not found: {tuning_manifest_path}")
    if not baseline_report_path.exists():
        raise FileNotFoundError(f"Baseline supervised_report.json not found: {baseline_report_path}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    tuning_manifest = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    extractor = tuning_manifest.get("extractor", {})
    extractor_params = extractor.get("params", {})
    threshold_selection = tuning_manifest.get("threshold_selection", {})
    tuning = tuning_manifest.get("tuning", {})

    baseline_selected_features = extractor_params.get("spectral_features")
    if not isinstance(baseline_selected_features, list) or not baseline_selected_features:
        raise ValueError(
            f"Baseline tuning manifest does not expose extractor.params.spectral_features: {tuning_manifest_path}"
        )

    return ReferenceRunSettings(
        reference_run_dir=reference_run_dir,
        model_name=str(run_config.get("model_name") or "cnn_1d"),
        train_split_percent=int(run_config.get("train_split_percent") or 100),
        split_by_folder=bool(run_config.get("split_by_folder", False)),
        calibration_split_percent=(
            int(threshold_selection["calibration_split_percent"])
            if threshold_selection.get("calibration_split_percent") is not None
            else None
        ),
        accepted_fprs=(
            [float(x) for x in threshold_selection.get("accepted_fprs", [])]
            if isinstance(threshold_selection.get("accepted_fprs"), list)
            else None
        ),
        cv_folds=int(tuning.get("cv_folds_requested", 5)),
        cv_random_states=[
            int(x) for x in (
                tuning.get("cv_random_states")
                or run_config.get("cv_random_states")
                or [DEFAULT_REFERENCE_RANDOM_STATE]
            )
        ],
        spectral_sv_top_k=int(extractor_params.get("spectral_sv_top_k", 8)),
        spectral_moment_source=str(
            resolve_spectral_moment_source(extractor_params.get("spectral_moment_source"))
        ),
        spectral_qv_sum_mode=str(
            resolve_spectral_qv_sum_mode(extractor_params.get("spectral_qv_sum_mode"))
        ),
        spectral_entrywise_delta_mode=str(extractor_params.get("spectral_entrywise_delta_mode", "dense")),
        baseline_selected_features=[str(x) for x in baseline_selected_features],
        score_percentiles=[
            float(x)
            for x in (run_config.get("score_percentiles") or list(DEFAULT_REFERENCE_SCORE_PERCENTILES))
        ],
        random_state=int(run_config.get("random_state") or DEFAULT_REFERENCE_RANDOM_STATE),
    )


def study_arm_specs(config: StudyConfig, *, baseline_features: Sequence[str]) -> list[StudyArmSpec]:
    rank_norm_features, raw_rank_features = build_arm_feature_groups(
        baseline_features=baseline_features,
    )
    requested_arms = tuple(str(arm) for arm in config.arms)
    unknown_arms = sorted(set(requested_arms).difference(DEFAULT_STUDY_ARMS))
    if unknown_arms:
        raise ValueError(
            f"Unsupported study arm(s): {unknown_arms}. Expected one of {DEFAULT_STUDY_ARMS}"
        )
    if not requested_arms:
        raise ValueError("At least one study arm must be selected")

    baseline_arm = StudyArmSpec(
        name="baseline",
        selected_features=tuple(str(x) for x in baseline_features),
        feature_output_run="",
        cnn_output_run="",
        reference_run_id=str(config.baseline_reference_run),
        zero_shot_run_prefix="",
        baseline_read_only=True,
    )
    candidate_arms = [
        StudyArmSpec(
            name="rank_norm",
            selected_features=tuple(rank_norm_features),
            feature_output_run=str(config.rank_norm_feature_output_run),
            cnn_output_run=str(config.rank_norm_cnn_output_run),
            reference_run_id=str(config.rank_norm_reference_run),
            zero_shot_run_prefix=str(config.rank_norm_zero_shot_prefix),
        ),
        StudyArmSpec(
            name="raw_rank",
            selected_features=tuple(raw_rank_features),
            feature_output_run=str(config.raw_rank_feature_output_run),
            cnn_output_run=str(config.raw_rank_cnn_output_run),
            reference_run_id=str(config.raw_rank_reference_run),
            zero_shot_run_prefix=str(config.raw_rank_zero_shot_prefix),
        ),
    ]
    requested_set = set(requested_arms)
    return [baseline_arm, *[arm for arm in candidate_arms if arm.name in requested_set]]


def run_reference_tuning(
    *,
    arm: StudyArmSpec,
    config: StudyConfig,
    reference_settings: ReferenceRunSettings,
    cnn_feature_path: Path,
) -> dict[str, Any]:
    resolved_executor = str(config.reference_tuning_executor)
    stage = "prepare" if resolved_executor == "slurm_array" else "all"
    return run_supervised_pipeline(
        manifest_json=resolve_manifest_path(config.tuning_manifest),
        dataset_root=_resolve_path(config.dataset_root),
        output_root=_resolve_path(config.output_root),
        run_id=arm.reference_run_id,
        model_name=reference_settings.model_name,
        spectral_features=list(arm.selected_features),
        spectral_sv_top_k=reference_settings.spectral_sv_top_k,
        spectral_moment_source=reference_settings.spectral_moment_source,
        spectral_qv_sum_mode=reference_settings.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=reference_settings.spectral_entrywise_delta_mode,
        stream_block_size=131072,
        dtype_name="float32",
        cv_folds=reference_settings.cv_folds,
        random_state=reference_settings.random_state,
        train_split_percent=reference_settings.train_split_percent,
        calibration_split_percent=reference_settings.calibration_split_percent,
        accepted_fpr=reference_settings.accepted_fprs,
        split_by_folder=reference_settings.split_by_folder,
        cv_random_states=reference_settings.cv_random_states,
        n_jobs=int(config.reference_n_jobs),
        score_percentiles=reference_settings.score_percentiles,
        feature_file=cnn_feature_path,
        tuning_executor=resolved_executor,
        slurm_partition=str(config.slurm_partition),
        slurm_max_concurrent="auto",
        slurm_cpus_per_task="auto",
        finalize_export_shards=1,
        stage=stage,
        run_dir=None,
        task_index=None,
        cnn_hyperparams=config.reference_cnn_hyperparams,
        skip_feature_importance=False,
    )


def submit_reference_tuning_slurm(
    *,
    arm: StudyArmSpec,
    config: StudyConfig,
    prepared_run: dict[str, Any],
    reference_settings: ReferenceRunSettings,
) -> dict[str, Any]:
    run_dir = Path(str(prepared_run["run_dir"])).resolve()
    n_tasks = int(prepared_run["n_tasks"])
    if n_tasks <= 0:
        raise ValueError(f"Prepared reference run has no tasks: {run_dir}")

    tuning_manifest_path = Path(str(prepared_run["tuning_manifest"])).resolve()
    tuning_manifest = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    runtime = tuning_manifest.get("runtime", {})
    slurm_max_concurrent = int(runtime.get("slurm_max_concurrent", 1))
    slurm_cpus_per_task = int(runtime.get("slurm_cpus_per_task", 1))
    score_percentiles = [
        float(x) for x in (
            runtime.get("score_percentiles")
            or reference_settings.score_percentiles
            or list(DEFAULT_REFERENCE_SCORE_PERCENTILES)
        )
    ]

    repo_root = Path(__file__).resolve().parents[2]
    slurm_log_dir = _resolve_path(config.slurm_log_dir)
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = str(arm.reference_run_id).replace("/", "__")

    worker_wrap = (
        f"source {shlex.quote(str(_resolve_path(config.conda_sh)))}"
        f" && conda activate {shlex.quote(str(config.conda_env))}"
        f" && cd {shlex.quote(str(repo_root))}"
        f" && python -m upeftguard.cli run supervised --stage worker"
        f" --run-dir {shlex.quote(str(run_dir))}"
        f" --task-index ${'{'}SLURM_ARRAY_TASK_ID{'}'}"
        f" --n-jobs {int(slurm_cpus_per_task)}"
    )
    finalize_wrap = (
        f"source {shlex.quote(str(_resolve_path(config.conda_sh)))}"
        f" && conda activate {shlex.quote(str(config.conda_env))}"
        f" && cd {shlex.quote(str(repo_root))}"
        f" && python -m upeftguard.cli run supervised --stage finalize"
        f" --run-dir {shlex.quote(str(run_dir))}"
        f" --n-jobs 4"
    )
    if score_percentiles:
        finalize_wrap += " --score-percentiles " + " ".join(str(float(x)) for x in score_percentiles)

    worker_cmd = [
        "sbatch",
        "--parsable",
        "--partition",
        str(config.slurm_partition),
        "--array",
        f"0-{n_tasks - 1}%{slurm_max_concurrent}",
        "--cpus-per-task",
        str(slurm_cpus_per_task),
        "--job-name",
        f"upeftguard_supervised_ref_worker_{safe_run_id}",
        "--output",
        str(slurm_log_dir / f"supervised_ref_worker_{safe_run_id}_%A_%a.out"),
        "--error",
        str(slurm_log_dir / f"supervised_ref_worker_{safe_run_id}_%A_%a.err"),
        "--wrap",
        worker_wrap,
    ]
    finalize_cmd_base = [
        "sbatch",
        "--parsable",
        "--partition",
        str(config.slurm_partition),
        "--cpus-per-task",
        "4",
        "--job-name",
        f"upeftguard_supervised_ref_finalize_{safe_run_id}",
        "--output",
        str(slurm_log_dir / f"supervised_ref_finalize_{safe_run_id}_%j.out"),
        "--error",
        str(slurm_log_dir / f"supervised_ref_finalize_{safe_run_id}_%j.err"),
    ]

    if config.reference_dry_run:
        return {
            "mode": "dry_run",
            "run_dir": str(run_dir),
            "tuning_manifest": str(tuning_manifest_path),
            "n_tasks": int(n_tasks),
            "slurm_max_concurrent": int(slurm_max_concurrent),
            "slurm_cpus_per_task": int(slurm_cpus_per_task),
            "worker_command": worker_cmd,
            "finalize_command": finalize_cmd_base + ["--dependency", "afterok:<worker_job_id>", "--wrap", finalize_wrap],
            "next_steps": [str(step) for step in prepared_run.get("next_steps", [])],
        }

    worker_result = subprocess.run(
        worker_cmd,
        check=True,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    worker_job_id = worker_result.stdout.strip()
    finalize_cmd = [
        *finalize_cmd_base,
        "--dependency",
        f"afterok:{worker_job_id}",
        "--wrap",
        finalize_wrap,
    ]
    finalize_result = subprocess.run(
        finalize_cmd,
        check=True,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    finalize_job_id = finalize_result.stdout.strip()
    return {
        "mode": "submitted",
        "run_dir": str(run_dir),
        "tuning_manifest": str(tuning_manifest_path),
        "n_tasks": int(n_tasks),
        "slurm_max_concurrent": int(slurm_max_concurrent),
        "slurm_cpus_per_task": int(slurm_cpus_per_task),
        "worker_job_id": worker_job_id,
        "finalize_job_id": finalize_job_id,
    }


def launch_zero_shot_suite(
    *,
    arm: StudyArmSpec,
    config: StudyConfig,
    reference_run_dir: Path,
    cnn_feature_path: Path,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["ZERO_SHOT_MANIFEST_ROOT"] = str(_resolve_path(config.zero_shot_manifest_root))
    env["RUN_ID_PREFIX"] = str(arm.zero_shot_run_prefix)
    env["FEATURE_FILE"] = str(cnn_feature_path.resolve())
    env["OUTPUT_ROOT"] = str(_resolve_path(config.output_root))

    cmd = [
        "python",
        "-m",
        "upeftguard.cli",
        "experiment",
        "supervised-cnn-suite",
        "--suite",
        "zero-shot",
        "--hyperparam-config",
        str(reference_run_dir.resolve()),
        "--manifest-filter",
        str(config.zero_shot_manifest_filter),
    ]
    if config.zero_shot_dry_run:
        cmd.append("--dry-run")
    return subprocess.run(
        cmd,
        check=True,
        cwd=str(repo_root),
        env=env,
        text=True,
    )


def _assert_reference_run_completed(reference_run_dir: Path) -> None:
    report_path = reference_run_dir / "reports" / "supervised_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            "Reference run is not finalized yet. Expected supervised_report.json at "
            f"{report_path}"
        )


def _read_inference_rows(inference_scores_csv: Path) -> list[dict[str, Any]]:
    with open(inference_scores_csv, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _roc_auc_from_inference_scores(inference_scores_csv: Path) -> float | None:
    rows = _read_inference_rows(inference_scores_csv)
    if not rows:
        return None
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _expected_zero_shot_run_dir(
    *,
    arm: StudyArmSpec,
    config: StudyConfig,
    expectation: ZeroShotExpectation,
) -> Path:
    if arm.baseline_read_only:
        root = _resolve_path(config.baseline_transfer_results_root)
        candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
        matches = [
            path
            for path in candidates
            if _normalized_zero_shot_run_name(path.name) == expectation.manifest_name
        ]
        if len(matches) == 1:
            return matches[0].resolve()
        return (root / expectation.manifest_name).resolve()
    return (_resolve_path(config.output_root) / "supervised" / arm.zero_shot_run_prefix / expectation.manifest_name).resolve()


def collect_zero_shot_metrics(
    *,
    arm: StudyArmSpec,
    config: StudyConfig,
    expectations: Sequence[ZeroShotExpectation],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for expectation in expectations:
        run_dir = _expected_zero_shot_run_dir(arm=arm, config=config, expectation=expectation)
        inference_scores_csv = run_dir / "reports" / "inference_scores.csv"
        auroc: float | None = None
        status = "missing_run_dir"
        if run_dir.exists():
            status = "missing_inference_scores"
            if inference_scores_csv.exists():
                auroc = _roc_auc_from_inference_scores(inference_scores_csv)
                status = "ok" if auroc is not None else "invalid_auc"
        row = {
            "arm": arm.name,
            "manifest_name": expectation.manifest_name,
            "target_rank": expectation.target_rank,
            "run_dir": str(run_dir),
            "inference_scores_csv": str(inference_scores_csv),
            "status": status,
            "roc_auc": auroc,
        }
        rows.append(row)
        if auroc is not None:
            scores.append(float(auroc))
    return {
        "arm": arm.name,
        "rows": rows,
        "mean_roc_auc": (float(np.mean(scores)) if scores else None),
        "worst_rank_roc_auc": (float(np.min(scores)) if scores else None),
    }


def resolve_existing_arm_artifacts(
    *,
    arm: StudyArmSpec,
    config: StudyConfig,
) -> dict[str, Any]:
    derived_feature_path = _resolve_feature_bundle_path(
        Path(arm.feature_output_run),
        feature_root=config.feature_root,
    )
    derived_metadata_path = derived_feature_path.with_name("spectral_metadata.json")
    derived_dataset_reference_report_path = default_dataset_reference_report_path(derived_feature_path.parent)
    derivation_report_path = derived_feature_path.parent / "rank_normalization_bundle_report.json"
    cnn_feature_path = _resolve_feature_bundle_path(
        Path(arm.cnn_output_run),
        feature_root=config.feature_root,
    )
    cnn_metadata_path = cnn_feature_path.with_name("spectral_metadata.json")
    cnn_dataset_reference_report_path = default_dataset_reference_report_path(cnn_feature_path.parent)
    aggregation_report_path = cnn_feature_path.with_name("spectral_aggregation_report.json")
    reference_run_dir = _resolve_supervised_run_dir(
        arm.reference_run_id,
        output_root=config.output_root,
    )

    payload: dict[str, Any] = {
        "derived_feature_path": str(derived_feature_path),
        "derived_feature_metadata": str(derived_metadata_path),
        "derived_dataset_reference_report": str(derived_dataset_reference_report_path),
        "cnn_feature_path": str(cnn_feature_path),
        "cnn_feature_metadata": str(cnn_metadata_path),
        "cnn_dataset_reference_report": str(cnn_dataset_reference_report_path),
        "aggregation_report_path": str(aggregation_report_path),
        "reference_run_dir": str(reference_run_dir),
    }
    if derivation_report_path.exists():
        report_payload = json.loads(derivation_report_path.read_text(encoding="utf-8"))
        payload["requested_model_count"] = int(report_payload.get("requested_model_count", 0))
        payload["selected_model_count"] = int(report_payload.get("selected_row_count", 0))
        payload["missing_model_count"] = int(report_payload.get("missing_model_count", 0))
        payload["missing_model_examples"] = [str(x) for x in report_payload.get("missing_model_examples", [])]
    return payload


def write_comparison_report(
    *,
    reports_dir: Path,
    metrics_by_arm: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "zero_shot_rank_comparison.json"
    csv_path = reports_dir / "zero_shot_rank_comparison.csv"
    markdown_path = reports_dir / "zero_shot_rank_comparison.md"

    comparison_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for arm_name, payload in metrics_by_arm.items():
        summary_rows.append(
            {
                "arm": arm_name,
                "mean_roc_auc": payload.get("mean_roc_auc"),
                "worst_rank_roc_auc": payload.get("worst_rank_roc_auc"),
            }
        )
        comparison_rows.extend(payload.get("rows", []))

    ordered_rows = sorted(
        comparison_rows,
        key=lambda row: (
            row.get("target_rank") is None,
            float(row["target_rank"]) if row.get("target_rank") is not None else float("inf"),
            str(row["arm"]),
        ),
    )
    summary_order = sorted(
        summary_rows,
        key=lambda row: (
            -(float(row["mean_roc_auc"]) if row.get("mean_roc_auc") is not None else float("-inf")),
            -(float(row["worst_rank_roc_auc"]) if row.get("worst_rank_roc_auc") is not None else float("-inf")),
            0 if row["arm"] == "rank_norm" else 1 if row["arm"] == "raw_rank" else 2,
        ),
    )

    payload = {
        "script_version": SCRIPT_VERSION,
        "summary": summary_order,
        "rows": ordered_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["arm", "target_rank", "roc_auc", "status", "manifest_name", "run_dir"],
        )
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "arm": row["arm"],
                    "target_rank": row["target_rank"],
                    "roc_auc": row["roc_auc"],
                    "status": row["status"],
                    "manifest_name": row["manifest_name"],
                    "run_dir": row["run_dir"],
                }
            )

    lines = [
        "| Arm | Mean AUROC | Worst-Rank AUROC |",
        "| --- | --- | --- |",
    ]
    for row in summary_order:
        mean_value = (
            f"{float(row['mean_roc_auc']):.6f}" if row.get("mean_roc_auc") is not None else "n/a"
        )
        worst_value = (
            f"{float(row['worst_rank_roc_auc']):.6f}" if row.get("worst_rank_roc_auc") is not None else "n/a"
        )
        lines.append(f"| {row['arm']} | {mean_value} | {worst_value} |")
    lines.append("")
    lines.append("| Arm | Target Rank | AUROC | Status |")
    lines.append("| --- | --- | --- | --- |")
    for row in ordered_rows:
        auroc = f"{float(row['roc_auc']):.6f}" if row.get("roc_auc") is not None else "n/a"
        rank_text = str(row["target_rank"]) if row.get("target_rank") is not None else "n/a"
        lines.append(f"| {row['arm']} | {rank_text} | {auroc} | {row['status']} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }


def run_rank_normalization_study(config: StudyConfig) -> dict[str, Any]:
    if str(config.stage) == "all" and str(config.reference_tuning_executor) == "slurm_array":
        raise ValueError(
            "stage=all is only supported with --reference-tuning-executor local. "
            "For Slurm-array reference tuning, run the study in phases: "
            "prepare_reference -> launch_zero_shot -> report."
        )

    started_at = perf_counter()
    reference_settings = load_reference_run_settings(
        baseline_reference_run=config.baseline_reference_run,
        output_root=config.output_root,
    )
    zero_shot_expectations = collect_zero_shot_expectations(
        manifest_root=config.zero_shot_manifest_root,
        manifest_filter=config.zero_shot_manifest_filter,
    )
    selected_model_names, manifest_paths = collect_manifest_model_names(
        tuning_manifest=config.tuning_manifest,
        zero_shot_expectations=zero_shot_expectations,
    )
    baseline_feature_path = _resolve_feature_bundle_path(
        config.baseline_feature_file,
        feature_root=config.feature_root,
    )
    source_bundle: LoadedFeatureBundle | None = None

    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=config.output_root,
        run_id=config.run_id,
    )
    arms = study_arm_specs(config, baseline_features=reference_settings.baseline_selected_features)
    arm_artifacts: dict[str, dict[str, Any]] = {}
    metrics_by_arm: dict[str, dict[str, Any]] | None = None
    comparison_paths: dict[str, Path] | None = None

    run_prepare_reference = str(config.stage) in {"prepare_reference", "all"}
    run_launch_zero_shot = str(config.stage) in {"launch_zero_shot", "all"}
    run_report = str(config.stage) in {"report", "all"}

    if run_prepare_reference:
        _progress(f"Stage {config.stage}: loading source bundle")
        source_bundle = load_feature_bundle(
            feature_file=config.source_feature_file,
            feature_root=config.feature_root,
        )
        for arm in arms:
            if arm.baseline_read_only:
                continue

            _progress(f"{arm.name}: preparing derived feature bundle")
            derived_bundle: DerivedBundleArtifacts | None = None
            if not config.skip_derive:
                assert source_bundle is not None
                derived_bundle = derive_arm_feature_bundle(
                    bundle=source_bundle,
                    arm=arm,
                    feature_root=config.feature_root,
                    selected_model_names=selected_model_names,
                    manifest_paths=manifest_paths,
                    spectral_sv_top_k=reference_settings.spectral_sv_top_k,
                    spectral_moment_source=reference_settings.spectral_moment_source,
                    spectral_qv_sum_mode=reference_settings.spectral_qv_sum_mode,
                )
            else:
                derived_feature_path = _resolve_feature_bundle_path(
                    Path(arm.feature_output_run),
                    feature_root=config.feature_root,
                )
                derived_model_names_path = derived_feature_path.with_name("spectral_model_names.json")
                derived_metadata_path = derived_feature_path.with_name("spectral_metadata.json")
                derived_report_path = derived_feature_path.parent / "rank_normalization_bundle_report.json"
                with open(derived_model_names_path, "r", encoding="utf-8") as f:
                    model_names = [str(x) for x in json.load(f)]
                metadata = load_spectral_metadata(derived_metadata_path)
                derived_bundle = DerivedBundleArtifacts(
                    arm_name=arm.name,
                    output_dir=derived_feature_path.parent,
                    feature_path=derived_feature_path,
                    model_names_path=derived_model_names_path,
                    labels_path=(
                        derived_feature_path.with_name("spectral_labels.npy")
                        if derived_feature_path.with_name("spectral_labels.npy").exists()
                        else None
                    ),
                    metadata_path=derived_metadata_path,
                    dataset_reference_report_path=default_dataset_reference_report_path(derived_feature_path.parent),
                    derivation_report_path=derived_report_path,
                    feature_names=[str(x) for x in metadata.get("feature_names", [])],
                    model_names=model_names,
                    row_indices=np.asarray([], dtype=np.int64),
                    requested_model_count=int(len(model_names)),
                    missing_model_names=[],
                )

            assert derived_bundle is not None
            arm_artifacts[arm.name] = {
                "derived_feature_path": str(derived_bundle.feature_path),
                "derived_feature_metadata": str(derived_bundle.metadata_path),
                "derived_dataset_reference_report": str(derived_bundle.dataset_reference_report_path),
                "requested_model_count": int(derived_bundle.requested_model_count),
                "selected_model_count": int(len(derived_bundle.model_names)),
                "missing_model_count": int(len(derived_bundle.missing_model_names)),
                "missing_model_examples": [str(x) for x in derived_bundle.missing_model_names[:10]],
            }

            if not config.skip_aggregate:
                _progress(f"{arm.name}: aggregating to layer_sequence bundle")
                aggregate_outputs = aggregate_arm_feature_bundle(
                    arm=arm,
                    feature_root=config.feature_root,
                    feature_path=derived_bundle.feature_path,
                    selected_features=arm.selected_features,
                    spectral_qv_sum_mode=reference_settings.spectral_qv_sum_mode,
                )
            else:
                aggregate_feature_path = _resolve_feature_bundle_path(
                    Path(arm.cnn_output_run),
                    feature_root=config.feature_root,
                )
                aggregate_outputs = {
                    "feature_path": aggregate_feature_path,
                    "model_names_path": aggregate_feature_path.with_name("spectral_model_names.json"),
                    "labels_path": (
                        aggregate_feature_path.with_name("spectral_labels.npy")
                        if aggregate_feature_path.with_name("spectral_labels.npy").exists()
                        else None
                    ),
                    "metadata_path": aggregate_feature_path.with_name("spectral_metadata.json"),
                    "dataset_reference_report_path": default_dataset_reference_report_path(aggregate_feature_path.parent),
                    "aggregation_report_path": aggregate_feature_path.with_name("spectral_aggregation_report.json"),
                    "group_mask_path": aggregate_feature_path.with_name("spectral_group_mask.npy"),
                    "value_mask_path": aggregate_feature_path.with_name("spectral_value_mask.npy"),
                    "group_names_path": aggregate_feature_path.with_name("spectral_group_names.json"),
                }

            arm_artifacts[arm.name].update(
                {
                    "cnn_feature_path": str(aggregate_outputs["feature_path"]),
                    "cnn_feature_metadata": str(aggregate_outputs["metadata_path"]),
                    "cnn_dataset_reference_report": str(aggregate_outputs["dataset_reference_report_path"]),
                }
            )

            reference_run_result: dict[str, Any] | None = None
            reference_run_dir = _resolve_supervised_run_dir(
                arm.reference_run_id,
                output_root=config.output_root,
            )
            if not config.skip_reference:
                _progress(
                    f"{arm.name}: preparing reference tuning with executor={config.reference_tuning_executor}"
                )
                reference_run_result = run_reference_tuning(
                    arm=arm,
                    config=config,
                    reference_settings=reference_settings,
                    cnn_feature_path=Path(str(aggregate_outputs["feature_path"])),
                )
                reference_run_dir = Path(str(reference_run_result["run_dir"])).resolve()
                if str(config.reference_tuning_executor) == "slurm_array":
                    submission = submit_reference_tuning_slurm(
                        arm=arm,
                        config=config,
                        prepared_run=reference_run_result,
                        reference_settings=reference_settings,
                    )
                    arm_artifacts[arm.name]["reference_submission"] = submission
                    if submission.get("mode") == "submitted":
                        _progress(
                            f"{arm.name}: submitted reference tuning "
                            f"(worker_job_id={submission.get('worker_job_id')}, "
                            f"finalize_job_id={submission.get('finalize_job_id')})"
                        )
                    else:
                        _progress(f"{arm.name}: prepared reference tuning dry-run submission")
                else:
                    _progress(f"{arm.name}: completed local reference tuning")
            arm_artifacts[arm.name]["reference_run_dir"] = str(reference_run_dir)
    elif run_launch_zero_shot or run_report:
        for arm in arms:
            if arm.baseline_read_only:
                continue
            arm_artifacts[arm.name] = resolve_existing_arm_artifacts(
                arm=arm,
                config=config,
            )

    if run_launch_zero_shot and not config.skip_zero_shot_launch:
        for arm in arms:
            if arm.baseline_read_only:
                continue
            reference_run_dir = _resolve_supervised_run_dir(
                arm.reference_run_id,
                output_root=config.output_root,
            )
            _assert_reference_run_completed(reference_run_dir)
            cnn_feature_path = _resolve_feature_bundle_path(
                Path(arm.cnn_output_run),
                feature_root=config.feature_root,
            )
            _progress(f"{arm.name}: launching zero-shot suite")
            launch_zero_shot_suite(
                arm=arm,
                config=config,
                reference_run_dir=reference_run_dir,
                cnn_feature_path=cnn_feature_path,
            )

    if run_report:
        _progress(f"Stage {config.stage}: collecting zero-shot metrics")
        metrics_by_arm = {
            arm.name: collect_zero_shot_metrics(
                arm=arm,
                config=config,
                expectations=zero_shot_expectations,
            )
            for arm in arms
        }
        comparison_paths = write_comparison_report(
            reports_dir=ctx.reports_dir,
            metrics_by_arm=metrics_by_arm,
        )

    study_report = {
        "script_version": SCRIPT_VERSION,
        "stage": str(config.stage),
        "source_feature_file": (
            str(source_bundle.feature_file.resolve())
            if source_bundle is not None
            else str(_resolve_feature_bundle_path(config.source_feature_file, feature_root=config.feature_root))
        ),
        "baseline_feature_file": str(baseline_feature_path),
        "baseline_reference_run": str(reference_settings.reference_run_dir),
        "baseline_transfer_results_root": str(_resolve_path(config.baseline_transfer_results_root)),
        "tuning_manifest": str(resolve_manifest_path(config.tuning_manifest)),
        "zero_shot_manifest_root": str(_resolve_path(config.zero_shot_manifest_root)),
        "zero_shot_manifest_filter": str(config.zero_shot_manifest_filter),
        "active_arms": [str(arm) for arm in config.arms],
        "reference_cnn_hyperparams": (
            str(_resolve_path(config.reference_cnn_hyperparams))
            if config.reference_cnn_hyperparams is not None
            else None
        ),
        "selected_model_count": int(len(selected_model_names)),
        "selected_manifest_paths": [str(path.resolve()) for path in manifest_paths],
        "reference_settings": {
            "model_name": reference_settings.model_name,
            "train_split_percent": reference_settings.train_split_percent,
            "split_by_folder": reference_settings.split_by_folder,
            "calibration_split_percent": reference_settings.calibration_split_percent,
            "accepted_fprs": reference_settings.accepted_fprs,
            "cv_folds": reference_settings.cv_folds,
            "cv_random_states": reference_settings.cv_random_states,
            "spectral_sv_top_k": reference_settings.spectral_sv_top_k,
            "spectral_moment_source": reference_settings.spectral_moment_source,
            "spectral_qv_sum_mode": reference_settings.spectral_qv_sum_mode,
            "spectral_entrywise_delta_mode": reference_settings.spectral_entrywise_delta_mode,
            "baseline_selected_features": reference_settings.baseline_selected_features,
        },
        "arm_artifacts": arm_artifacts,
        "metrics_by_arm": metrics_by_arm,
        "comparison_report_paths": (
            {key: str(path) for key, path in comparison_paths.items()}
            if comparison_paths is not None
            else None
        ),
        "elapsed_seconds": float(perf_counter() - started_at),
    }

    study_report_path = ctx.reports_dir / "rank_normalization_study_report.json"
    with open(study_report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(study_report), f, indent=2)

    ctx.add_artifact("study_report", study_report_path)
    if comparison_paths is not None:
        ctx.add_artifact("comparison_json", comparison_paths["json"])
        ctx.add_artifact("comparison_csv", comparison_paths["csv"])
        ctx.add_artifact("comparison_markdown", comparison_paths["markdown"])
    ctx.add_timing("rank_normalization_study_elapsed_seconds", float(perf_counter() - started_at))
    ctx.finalize(
        {
            "pipeline": PIPELINE_NAME,
            "script_version": SCRIPT_VERSION,
            "stage": str(config.stage),
            "source_feature_file": str(config.source_feature_file),
            "baseline_feature_file": str(config.baseline_feature_file),
            "baseline_reference_run": str(config.baseline_reference_run),
            "tuning_manifest": str(config.tuning_manifest),
            "zero_shot_manifest_root": str(config.zero_shot_manifest_root),
            "zero_shot_manifest_filter": str(config.zero_shot_manifest_filter),
            "active_arms": [str(arm) for arm in config.arms],
            "reference_cnn_hyperparams": (
                str(config.reference_cnn_hyperparams)
                if config.reference_cnn_hyperparams is not None
                else None
            ),
            "reference_tuning_executor": str(config.reference_tuning_executor),
            "reference_dry_run": bool(config.reference_dry_run),
            "slurm_partition": str(config.slurm_partition),
            "slurm_log_dir": str(config.slurm_log_dir),
            "conda_sh": str(config.conda_sh),
            "conda_env": str(config.conda_env),
            "report_only": bool(config.report_only),
            "skip_derive": bool(config.skip_derive),
            "skip_aggregate": bool(config.skip_aggregate),
            "skip_reference": bool(config.skip_reference),
            "skip_zero_shot_launch": bool(config.skip_zero_shot_launch),
            "zero_shot_dry_run": bool(config.zero_shot_dry_run),
        }
    )

    return {
        "run_dir": ctx.run_dir,
        "study_report": study_report_path,
        "comparison_json": comparison_paths["json"] if comparison_paths is not None else None,
        "comparison_csv": comparison_paths["csv"] if comparison_paths is not None else None,
        "comparison_markdown": comparison_paths["markdown"] if comparison_paths is not None else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive rank-aware CNN feature bundles from an existing spectral bundle, "
            "retune rank-aware reference CNN runs, and compare zero-shot rank transfer."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["prepare_reference", "launch_zero_shot", "report", "all"],
        default=DEFAULT_STUDY_STAGE,
        help=(
            "Study phase to run. Use prepare_reference for bundle derivation + reference tuning prep, "
            "launch_zero_shot after reference runs finish, report to collect metrics only, "
            "or all for the original local end-to-end flow."
        ),
    )
    parser.add_argument("--source-feature-file", type=Path, default=DEFAULT_SOURCE_FEATURE_FILE)
    parser.add_argument("--baseline-feature-file", type=Path, default=DEFAULT_BASELINE_FEATURE_FILE)
    parser.add_argument("--baseline-reference-run", type=str, default=DEFAULT_BASELINE_REFERENCE_RUN)
    parser.add_argument(
        "--baseline-transfer-results-root",
        type=Path,
        default=DEFAULT_BASELINE_TRANSFER_RESULTS_ROOT,
    )
    parser.add_argument("--tuning-manifest", type=Path, default=DEFAULT_TUNING_MANIFEST)
    parser.add_argument("--zero-shot-manifest-root", type=Path, default=DEFAULT_ZERO_SHOT_MANIFEST_ROOT)
    parser.add_argument("--zero-shot-manifest-filter", type=str, default=DEFAULT_ZERO_SHOT_MANIFEST_FILTER)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--run-id", type=str, default=None)

    parser.add_argument("--rank-norm-feature-output-run", type=str, default=DEFAULT_RANK_NORM_FEATURE_OUTPUT_RUN)
    parser.add_argument("--rank-norm-cnn-output-run", type=str, default=DEFAULT_RANK_NORM_CNN_OUTPUT_RUN)
    parser.add_argument("--rank-norm-reference-run", type=str, default=DEFAULT_RANK_NORM_REFERENCE_RUN)
    parser.add_argument("--rank-norm-zero-shot-prefix", type=str, default=DEFAULT_RANK_NORM_ZERO_SHOT_PREFIX)

    parser.add_argument("--raw-rank-feature-output-run", type=str, default=DEFAULT_RAW_RANK_FEATURE_OUTPUT_RUN)
    parser.add_argument("--raw-rank-cnn-output-run", type=str, default=DEFAULT_RAW_RANK_CNN_OUTPUT_RUN)
    parser.add_argument("--raw-rank-reference-run", type=str, default=DEFAULT_RAW_RANK_REFERENCE_RUN)
    parser.add_argument("--raw-rank-zero-shot-prefix", type=str, default=DEFAULT_RAW_RANK_ZERO_SHOT_PREFIX)

    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(DEFAULT_STUDY_ARMS),
        default=list(DEFAULT_STUDY_ARMS),
        help=(
            "Study arms to prepare, train, and launch. Use '--arms rank_norm' "
            "to run only the rank-normalized feature workflow."
        ),
    )
    parser.add_argument(
        "--reference-cnn-hyperparams",
        type=Path,
        default=None,
        help=(
            "Optional CNN hyperparameter grid JSON for rank-aware reference tuning. "
            "When omitted, the supervised cnn_1d default grid is used."
        ),
    )
    parser.add_argument(
        "--reference-tuning-executor",
        choices=["local", "slurm_array"],
        default=DEFAULT_REFERENCE_TUNING_EXECUTOR,
    )
    parser.add_argument("--reference-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--reference-dry-run",
        action="store_true",
        help="Prepare the Slurm-array reference tuning commands without submitting them.",
    )
    parser.add_argument("--zero-shot-dry-run", action="store_true")
    parser.add_argument("--slurm-partition", type=str, default=DEFAULT_SLURM_PARTITION)
    parser.add_argument("--slurm-log-dir", type=Path, default=DEFAULT_SLURM_LOG_DIR)
    parser.add_argument("--conda-sh", type=Path, default=DEFAULT_CONDA_SH)
    parser.add_argument("--conda-env", type=str, default=DEFAULT_CONDA_ENV)
    parser.add_argument("--skip-derive", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-zero-shot-launch", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def args_to_config(args: argparse.Namespace) -> StudyConfig:
    report_only = bool(args.report_only)
    stage = "report" if report_only else str(args.stage)
    return StudyConfig(
        stage=stage,
        source_feature_file=Path(args.source_feature_file),
        baseline_feature_file=Path(args.baseline_feature_file),
        baseline_reference_run=str(args.baseline_reference_run),
        baseline_transfer_results_root=Path(args.baseline_transfer_results_root),
        tuning_manifest=Path(args.tuning_manifest),
        zero_shot_manifest_root=Path(args.zero_shot_manifest_root),
        zero_shot_manifest_filter=str(args.zero_shot_manifest_filter),
        feature_root=Path(args.feature_root),
        output_root=Path(args.output_root),
        dataset_root=Path(args.dataset_root),
        run_id=args.run_id,
        rank_norm_feature_output_run=str(args.rank_norm_feature_output_run),
        rank_norm_cnn_output_run=str(args.rank_norm_cnn_output_run),
        rank_norm_reference_run=str(args.rank_norm_reference_run),
        rank_norm_zero_shot_prefix=str(args.rank_norm_zero_shot_prefix),
        raw_rank_feature_output_run=str(args.raw_rank_feature_output_run),
        raw_rank_cnn_output_run=str(args.raw_rank_cnn_output_run),
        raw_rank_reference_run=str(args.raw_rank_reference_run),
        raw_rank_zero_shot_prefix=str(args.raw_rank_zero_shot_prefix),
        arms=tuple(str(arm) for arm in args.arms),
        reference_cnn_hyperparams=(
            Path(args.reference_cnn_hyperparams)
            if args.reference_cnn_hyperparams is not None
            else None
        ),
        reference_tuning_executor=str(args.reference_tuning_executor),
        reference_n_jobs=int(args.reference_n_jobs),
        reference_dry_run=bool(args.reference_dry_run),
        zero_shot_dry_run=bool(args.zero_shot_dry_run),
        slurm_partition=str(args.slurm_partition),
        slurm_log_dir=Path(args.slurm_log_dir),
        conda_sh=Path(args.conda_sh),
        conda_env=str(args.conda_env),
        skip_derive=bool(args.skip_derive or report_only),
        skip_aggregate=bool(args.skip_aggregate or report_only),
        skip_reference=bool(args.skip_reference or report_only),
        skip_zero_shot_launch=bool(args.skip_zero_shot_launch or report_only),
        report_only=report_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_rank_normalization_study(args_to_config(args))
    print("Rank-normalization study complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Study report: {result['study_report']}")
    if result["comparison_json"] is not None:
        print(f"Comparison JSON: {result['comparison_json']}")
        print(f"Comparison CSV: {result['comparison_csv']}")
        print(f"Comparison Markdown: {result['comparison_markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
