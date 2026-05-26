from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..unsupervised.analysis import LoadedFeatureBundle, load_feature_bundle
from ..utilities.artifacts.dataset_references import (
    _build_incomplete_payload_from_model_names,
    _finalize_payload,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from ..utilities.artifacts.spectral_metadata import write_spectral_metadata
from ..utilities.core.manifest import (
    ManifestItem,
    parse_joint_manifest_json_by_model_name,
    resolve_manifest_path,
)
from ..utilities.core.run_context import create_run_context
from ..utilities.core.serialization import json_ready


SCRIPT_VERSION = "1.0.0"
PIPELINE_NAME = "paper_qv_reference"
DEFAULT_FEATURE_FILE = Path("list2_features-merged")
DEFAULT_FEATURE_ROOT = Path("runs") / "feature_extract"
DEFAULT_OUTPUT_ROOT = Path("runs")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SPLIT_PERCENT = 20
DEFAULT_CALIBRATION_SPLIT_PERCENT = 20
DETECTOR_FILENAME = "paper_qv_reference_detector.joblib"
SELECTED_THRESHOLD_FILENAME = "selected_threshold.json"
METRICS_FILENAME = "metrics.json"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
REFERENCE_BANK_SUMMARY_FILENAME = "reference_bank_summary.json"
CALIBRATION_SCORES_FILENAME = "calibration_scores.csv"
TEST_SCORES_FILENAME = "test_scores.csv"
PAPER_QV_SUMMARY_FILENAME_TEMPLATE = "paper_qv_reference_{suite_slug}_summary.csv"
PAPER_QV_COMBINED_SUMMARY_FILENAME = "paper_qv_reference_leave_one_out_summary.csv"
RANK_LEAVE_ONE_OUT_GLOB = "holdout_llama2_7b_toxic_backdoors_hard_rank*_qv.json"
_RANK_LEAVE_ONE_OUT_RE = re.compile(r"^holdout_llama2_7b_toxic_backdoors_hard_(rank\d+)_qv$")
TBH_TBA_RANK_ZERO_SHOT_GLOB = "llama2_7b_tbh+tba_zero_shot_r256_to_rank*.json"
_TBH_TBA_RANK_ZERO_SHOT_RE = re.compile(r"^llama2_7b_tbh\+tba_zero_shot_r256_to_(rank\d+)$")
PAPER_QV_SUITE_CHOICES = (
    "rank-leave-one-out",
    "adapter-leave-one-out",
    "architecture-leave-one-out",
    "tbh-tba-rank-zero-shot",
    "all-leave-one-out",
)
SCORING_MODE_SUPERVISED_LOGREG = "supervised-logreg"
SCORING_MODE_CLEAN_UNIFORM = "clean-uniform"
PAPER_QV_SCORING_MODE_CHOICES = (
    SCORING_MODE_SUPERVISED_LOGREG,
    SCORING_MODE_CLEAN_UNIFORM,
)
DEFAULT_CLEAN_THRESHOLD_PERCENTILE = 95.0
SCORE_AGGREGATION_SUPERVISED_LOGREG = "supervised_logistic_positive_coefficients"
SCORE_AGGREGATION_METRIC_EQUAL_UNIFORM = "metric_equal_uniform"
PAPER_QV_SELECTED_FEATURES = (
    "kurtosis",
    "l2_norm",
    "concentration_of_energy",
    "sv_topk",
    "spectral_entropy",
)
PAPER_QV_SELECTED_SUFFIXES = (
    "kurtosis",
    "l2_norm",
    "concentration_of_energy",
    "sv_1",
    "spectral_entropy",
)
PAPER_QV_FEATURE_MAPPING = {
    "leading_singular_value": "sv_1",
    "frobenius_norm": "l2_norm",
    "energy_concentration": "concentration_of_energy",
    "spectral_entropy": "spectral_entropy",
    "kurtosis": "kurtosis",
}


@dataclass(frozen=True)
class DerivedFeatureBundle:
    features: np.ndarray
    feature_names: list[str]
    block_names: list[str]
    feature_indices: np.ndarray
    output_dir: Path
    feature_path: Path
    model_names_path: Path
    labels_path: Path | None
    metadata_path: Path
    dataset_reference_report_path: Path


@dataclass(frozen=True)
class SplitAssignments:
    split_strategy: str
    train_indices: np.ndarray
    fit_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    test_stratify_kind: str
    calibration_stratify_kind: str
    warnings: list[str]
    manifest_json: Path | None = None


@dataclass(frozen=True)
class PaperQvSuiteManifestSpec:
    suite: str
    heldout_group: str
    run_id: str
    manifest_path: Path


@dataclass(frozen=True)
class PaperQvSuiteRunOutputs:
    suite: str
    run_outputs: dict[str, dict[str, Path]]
    summary_path: Path
    combined_summary_path: Path | None = None


def _feature_block_name(feature_name: str) -> str:
    block_name, sep, _ = str(feature_name).rpartition(".")
    if not sep or not block_name:
        raise ValueError(f"Invalid spectral feature name: {feature_name}")
    return block_name


def ordered_block_names_from_feature_names(feature_names: Sequence[str]) -> list[str]:
    block_names: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        block_name = _feature_block_name(str(feature_name))
        if block_name in seen:
            continue
        seen.add(block_name)
        block_names.append(block_name)
    return block_names


def _normalize_lora_dims_entry(raw_dims: Any) -> dict[str, int] | None:
    if not isinstance(raw_dims, dict):
        return None
    try:
        m = int(raw_dims["m"])
        n = int(raw_dims["n"])
        r = int(raw_dims["r"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"m": m, "n": n, "r": r}


def _update_lora_dims_from_explicit_metadata(
    dim_map: dict[str, dict[str, int]],
    *,
    block_names: Any,
    dims: Any,
) -> None:
    if not isinstance(block_names, list) or not isinstance(dims, list) or len(block_names) != len(dims):
        return
    for block_name, raw_dims in zip(block_names, dims):
        normalized = _normalize_lora_dims_entry(raw_dims)
        if normalized is None:
            continue
        dim_map[str(block_name)] = normalized


def _derive_qv_sum_lora_dims(
    *,
    dim_map: dict[str, dict[str, int]],
    qv_sum_block_names: Any,
) -> None:
    if not isinstance(qv_sum_block_names, list):
        return
    for raw_qv_name in qv_sum_block_names:
        qv_name = str(raw_qv_name)
        if qv_name in dim_map or not qv_name.endswith(".qv_sum"):
            continue
        prefix = qv_name[: -len("qv_sum")]
        q_name = f"{prefix}q_proj"
        v_name = f"{prefix}v_proj"
        q_dims = dim_map.get(q_name)
        v_dims = dim_map.get(v_name)
        if q_dims is None or v_dims is None:
            continue
        if q_dims.get("m") != v_dims.get("m") or q_dims.get("n") != v_dims.get("n"):
            continue
        dim_map[qv_name] = {
            "m": int(q_dims["m"]),
            "n": int(q_dims["n"]),
            "r": int(q_dims["r"] + v_dims["r"]),
        }


def spectral_block_lora_dims_by_block(metadata: dict[str, Any]) -> dict[str, dict[str, int]]:
    dim_map: dict[str, dict[str, int]] = {}
    _update_lora_dims_from_explicit_metadata(
        dim_map,
        block_names=metadata.get("block_names"),
        dims=metadata.get("lora_adapter_dims"),
    )
    _update_lora_dims_from_explicit_metadata(
        dim_map,
        block_names=metadata.get("base_block_names"),
        dims=metadata.get("base_lora_adapter_dims"),
    )
    _update_lora_dims_from_explicit_metadata(
        dim_map,
        block_names=metadata.get("qv_sum_block_names"),
        dims=metadata.get("qv_sum_lora_adapter_dims"),
    )
    _derive_qv_sum_lora_dims(
        dim_map=dim_map,
        qv_sum_block_names=metadata.get("qv_sum_block_names"),
    )
    return dim_map


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text).strip())
    return cleaned or "paper_qv_reference"


def _candidate_source_name(bundle: LoadedFeatureBundle) -> str:
    parent = bundle.feature_file.parent
    if parent.name in {"merged", "features"}:
        return parent.parent.name
    return bundle.feature_file.stem


def default_feature_output_run_name(bundle: LoadedFeatureBundle) -> str:
    return _safe_slug(f"{_candidate_source_name(bundle)}_paper_qv_reference")


def resolve_feature_output_dir(*, feature_root: Path, feature_output_run: str | None, bundle: LoadedFeatureBundle) -> Path:
    run_name = str(feature_output_run).strip() if feature_output_run is not None else default_feature_output_run_name(bundle)
    if not run_name:
        run_name = default_feature_output_run_name(bundle)
    output_dir = feature_root / run_name / "merged"
    if output_dir.resolve() == bundle.feature_file.parent.resolve():
        raise ValueError("Derived paper-style feature bundle must not overwrite the source bundle")
    return output_dir


def select_paper_qv_feature_indices(feature_names: Sequence[str]) -> np.ndarray:
    selected: list[int] = []
    suffix_set = set(PAPER_QV_SELECTED_SUFFIXES)
    for idx, raw_name in enumerate(feature_names):
        name = str(raw_name)
        if ".qv_sum." not in name:
            continue
        suffix = name.rpartition(".")[2]
        if suffix not in suffix_set:
            continue
        selected.append(int(idx))
    if not selected:
        raise ValueError("No paper-style q+v spectral columns were found in the input bundle")
    return np.asarray(selected, dtype=np.int64)


def select_paper_qv_feature_names(feature_names: Sequence[str]) -> list[str]:
    indices = select_paper_qv_feature_indices(feature_names)
    return [str(feature_names[int(idx)]) for idx in indices.tolist()]


def _subset_dataset_reference_payload(
    *,
    bundle: LoadedFeatureBundle,
    artifact_kind: str,
    source_artifacts: list[str],
) -> dict[str, Any]:
    payload = bundle.dataset_reference_payload
    if not isinstance(payload, dict):
        return _build_incomplete_payload_from_model_names(
            model_names=list(bundle.model_names),
            labels_by_name={
                model_name: (int(bundle.labels[idx]) if bundle.labels is not None else None)
                for idx, model_name in enumerate(bundle.model_names)
            },
            artifact_kind=artifact_kind,
            artifact_model_count=len(bundle.model_names),
            provenance_gaps=["Dataset-reference payload was unavailable; built an incomplete fallback payload"],
            source_artifacts=source_artifacts,
        )

    model_index = payload.get("model_index")
    if not isinstance(model_index, dict) or not model_index:
        return _build_incomplete_payload_from_model_names(
            model_names=list(bundle.model_names),
            labels_by_name={
                model_name: (int(bundle.labels[idx]) if bundle.labels is not None else None)
                for idx, model_name in enumerate(bundle.model_names)
            },
            artifact_kind=artifact_kind,
            artifact_model_count=len(bundle.model_names),
            provenance_gaps=["Dataset-reference payload did not include model_index; built an incomplete fallback payload"],
            source_artifacts=source_artifacts,
        )

    filtered_model_index = {
        str(model_name): dict(model_index[str(model_name)])
        for model_name in bundle.model_names
        if str(model_name) in model_index
    }
    missing = [str(model_name) for model_name in bundle.model_names if str(model_name) not in model_index]
    gaps = [str(x) for x in payload.get("provenance_gaps", []) if str(x).strip()]
    if missing:
        preview = ", ".join(missing[:5])
        gaps.append(
            f"Source dataset-reference payload was missing {len(missing)} model(s) while deriving the paper-style bundle. "
            f"Examples: {preview}"
        )
    return _finalize_payload(
        artifact_kind=artifact_kind,
        model_index=filtered_model_index,
        artifact_model_count=len(bundle.model_names),
        manifest_json=Path(str(payload["manifest_json"])) if payload.get("manifest_json") else None,
        dataset_root=Path(str(payload["dataset_root"])) if payload.get("dataset_root") else None,
        source_artifacts=source_artifacts,
        provenance_gaps=gaps,
        is_complete=bool(payload.get("is_complete", False)) and not missing,
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


def build_paper_qv_metadata(
    *,
    source_metadata: dict[str, Any],
    selected_feature_names: list[str],
    source_feature_file: Path,
) -> dict[str, Any]:
    block_names = ordered_block_names_from_feature_names(selected_feature_names)
    dim_map = spectral_block_lora_dims_by_block(source_metadata)
    qv_dims = [dict(dim_map[name]) for name in block_names if name in dim_map]

    extractor_params = dict(source_metadata.get("extractor_params", {}))
    extractor_params["spectral_features"] = list(PAPER_QV_SELECTED_FEATURES)
    extractor_params["spectral_sv_top_k"] = 1
    extractor_params["spectral_qv_sum_mode"] = "only"

    metadata: dict[str, Any] = {
        "extractor": str(source_metadata.get("extractor", "spectral")),
        "extractor_name": str(source_metadata.get("extractor_name", "spectral")),
        "extractor_version": source_metadata.get("extractor_version"),
        "delta_schema_version": source_metadata.get("delta_schema_version"),
        "feature_dim": int(len(selected_feature_names)),
        "feature_names": list(selected_feature_names),
        "block_names": list(block_names),
        "base_block_names": [],
        "qv_sum_block_names": list(block_names),
        "n_blocks": int(len(block_names)),
        "resolved_features": list(PAPER_QV_SELECTED_FEATURES),
        "sv_top_k": 1,
        "spectral_moment_source": "entrywise",
        "spectral_qv_sum_mode": "only",
        "spectral_entrywise_delta_mode": source_metadata.get("spectral_entrywise_delta_mode"),
        "extractor_params": extractor_params,
        "merge_source_feature_files": [str(source_feature_file.resolve())],
        "paper_reference_feature_mapping": dict(PAPER_QV_FEATURE_MAPPING),
        "paper_reference_selected_suffixes": list(PAPER_QV_SELECTED_SUFFIXES),
        "paper_reference_note": (
            "This bundle is a paper-style q+v-only subset derived from an existing merged spectral bundle. "
            "It preserves zero-filled mixed-architecture columns from the source bundle."
        ),
    }
    if qv_dims and len(qv_dims) == len(block_names):
        metadata["lora_adapter_dims"] = [dict(entry) for entry in qv_dims]
        metadata["qv_sum_lora_adapter_dims"] = [dict(entry) for entry in qv_dims]
    return metadata


def export_paper_qv_feature_bundle(
    *,
    bundle: LoadedFeatureBundle,
    feature_root: Path,
    feature_output_run: str | None,
) -> DerivedFeatureBundle:
    output_dir = resolve_feature_output_dir(
        feature_root=feature_root,
        feature_output_run=feature_output_run,
        bundle=bundle,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_indices = select_paper_qv_feature_indices(bundle.feature_names)
    selected_feature_names = [str(bundle.feature_names[int(idx)]) for idx in feature_indices.tolist()]
    output_features = np.asarray(bundle.features[:, feature_indices], dtype=np.float32)
    metadata = build_paper_qv_metadata(
        source_metadata=dict(bundle.metadata),
        selected_feature_names=selected_feature_names,
        source_feature_file=bundle.feature_file,
    )

    feature_path = output_dir / "spectral_features.npy"
    model_names_path = output_dir / "spectral_model_names.json"
    labels_path = output_dir / "spectral_labels.npy"
    metadata_path = output_dir / "spectral_metadata.json"
    np.save(feature_path, output_features.astype(np.float32, copy=False))
    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump([str(name) for name in bundle.model_names], f, indent=2)
    if bundle.labels is not None:
        np.save(labels_path, np.asarray(bundle.labels, dtype=np.int32))
    else:
        labels_path = None

    dataset_reference_payload = _subset_dataset_reference_payload(
        bundle=bundle,
        artifact_kind="paper_qv_reference_feature_subset",
        source_artifacts=[str(bundle.feature_file.resolve())],
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
    block_names = [str(x) for x in metadata["block_names"]]
    return DerivedFeatureBundle(
        features=output_features,
        feature_names=selected_feature_names,
        block_names=block_names,
        feature_indices=feature_indices,
        output_dir=output_dir,
        feature_path=feature_path,
        model_names_path=model_names_path,
        labels_path=labels_path,
        metadata_path=metadata_path,
        dataset_reference_report_path=dataset_reference_report_path,
    )


def _label_counts(values: Iterable[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _dataset_reference_value(bundle: LoadedFeatureBundle, model_name: str, key: str) -> str | None:
    payload = bundle.dataset_reference_payload
    if not isinstance(payload, dict):
        return None
    model_index = payload.get("model_index")
    if not isinstance(model_index, dict):
        return None
    raw_entry = model_index.get(str(model_name))
    if not isinstance(raw_entry, dict):
        return None
    raw_value = raw_entry.get(key)
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def build_candidate_stratify_vectors(bundle: LoadedFeatureBundle) -> list[tuple[str, np.ndarray]]:
    labels = np.asarray(bundle.labels, dtype=np.int32)
    candidates: list[tuple[str, np.ndarray]] = []

    dataset_name_values = np.asarray(
        [
            _dataset_reference_value(bundle, model_name, "dataset_name")
            or _dataset_reference_value(bundle, model_name, "subset_name")
            or "unknown"
            for model_name in bundle.model_names
        ],
        dtype=object,
    )
    if len(set(dataset_name_values.tolist())) > 1:
        candidates.append(
            (
                "dataset_name_x_label",
                np.asarray(
                    [f"{dataset_name_values[idx]}|{int(labels[idx])}" for idx in range(int(labels.shape[0]))],
                    dtype=object,
                ),
            )
        )

    model_family_values = np.asarray(
        [
            _dataset_reference_value(bundle, model_name, "model_family") or "unknown"
            for model_name in bundle.model_names
        ],
        dtype=object,
    )
    if len(set(model_family_values.tolist())) > 1:
        candidates.append(
            (
                "model_family_x_label",
                np.asarray(
                    [f"{model_family_values[idx]}|{int(labels[idx])}" for idx in range(int(labels.shape[0]))],
                    dtype=object,
                ),
            )
        )

    candidates.append(("label", np.asarray([str(int(x)) for x in labels.tolist()], dtype=object)))
    return candidates


def _stratify_is_feasible(values: np.ndarray, test_size_fraction: float) -> bool:
    if values.size == 0:
        return False
    unique, counts = np.unique(values, return_counts=True)
    if unique.size <= 1 or counts.min() < 2:
        return False
    n_total = int(values.shape[0])
    n_test = int(ceil(n_total * float(test_size_fraction)))
    n_train = int(n_total - n_test)
    if n_test <= 0 or n_train <= 0:
        return False
    return int(unique.size) <= n_test and int(unique.size) <= n_train


def choose_stratify_vector(
    *,
    candidate_vectors: Sequence[tuple[str, np.ndarray]],
    indices: np.ndarray,
    split_fraction: float,
) -> tuple[str, np.ndarray | None]:
    subset_indices = np.asarray(indices, dtype=np.int64)
    for kind, values in candidate_vectors:
        subset_values = np.asarray(values[subset_indices], dtype=object)
        if _stratify_is_feasible(subset_values, split_fraction):
            return str(kind), subset_values
    return "none", None


def build_split_assignments(
    *,
    bundle: LoadedFeatureBundle,
    random_state: int,
    test_split_percent: int,
    calibration_split_percent: int,
) -> SplitAssignments:
    if bundle.labels is None:
        raise ValueError("The input bundle must provide binary labels for paper-style detector fitting")
    labels = np.asarray(bundle.labels, dtype=np.int32)
    if not np.all(np.isin(labels, np.asarray([0, 1], dtype=np.int32))):
        raise ValueError("Paper-style detector fitting requires labels in {0, 1}")

    all_indices = np.arange(int(labels.shape[0]), dtype=np.int64)
    candidate_vectors = build_candidate_stratify_vectors(bundle)
    warnings: list[str] = []

    test_fraction = float(test_split_percent) / 100.0
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_split_percent must be between 1 and 99, got {test_split_percent}")
    calibration_fraction = float(calibration_split_percent) / 100.0
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_split_percent must be between 1 and 99, got {calibration_split_percent}"
        )

    test_kind, test_stratify = choose_stratify_vector(
        candidate_vectors=candidate_vectors,
        indices=all_indices,
        split_fraction=test_fraction,
    )
    if test_stratify is None:
        warnings.append(
            "Falling back to an unstratified fit/test split because no candidate stratification was feasible"
        )
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=int(random_state),
        shuffle=True,
        stratify=test_stratify,
    )

    train_val_indices = np.asarray(train_val_indices, dtype=np.int64)
    calibration_kind, calibration_stratify = choose_stratify_vector(
        candidate_vectors=candidate_vectors,
        indices=train_val_indices,
        split_fraction=calibration_fraction,
    )
    if calibration_stratify is None:
        warnings.append(
            "Falling back to an unstratified fit/calibration split because no candidate stratification was feasible"
        )

    fit_indices, calibration_indices = train_test_split(
        train_val_indices,
        test_size=calibration_fraction,
        random_state=int(random_state) + 1,
        shuffle=True,
        stratify=calibration_stratify,
    )
    return SplitAssignments(
        split_strategy="random_stratified_holdout",
        train_indices=np.asarray(sorted(int(x) for x in train_val_indices), dtype=np.int64),
        fit_indices=np.asarray(sorted(int(x) for x in fit_indices), dtype=np.int64),
        calibration_indices=np.asarray(sorted(int(x) for x in calibration_indices), dtype=np.int64),
        test_indices=np.asarray(sorted(int(x) for x in test_indices), dtype=np.int64),
        test_stratify_kind=str(test_kind),
        calibration_stratify_kind=str(calibration_kind),
        warnings=warnings,
    )


def _feature_bundle_row_index_by_model_name(bundle: LoadedFeatureBundle) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for row_index, raw_name in enumerate(bundle.model_names):
        model_name = str(raw_name)
        if model_name in index:
            duplicates.append(model_name)
            continue
        index[model_name] = int(row_index)
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            "Feature bundle contains duplicate model names; cannot align a manifest safely. "
            f"Examples: {preview}"
        )
    return index


def _manifest_partition_indices(
    *,
    bundle: LoadedFeatureBundle,
    items: Sequence[ManifestItem],
    partition_name: str,
) -> np.ndarray:
    name_to_index = _feature_bundle_row_index_by_model_name(bundle)
    missing = [str(item.model_name) for item in items if str(item.model_name) not in name_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Manifest {partition_name} partition references {len(missing)} model(s) not present "
            f"in the feature bundle. Examples: {preview}"
        )

    indices = np.asarray([name_to_index[str(item.model_name)] for item in items], dtype=np.int64)
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Manifest {partition_name} partition maps to duplicate feature rows")

    if bundle.labels is not None:
        labels = np.asarray(bundle.labels, dtype=np.int32)
        mismatches: list[str] = []
        for item, row_index in zip(items, indices.tolist()):
            if item.label is None:
                continue
            actual = int(labels[int(row_index)])
            expected = int(item.label)
            if actual != expected:
                mismatches.append(f"{item.model_name}: manifest={expected}, bundle={actual}")
        if mismatches:
            preview = "; ".join(mismatches[:5])
            raise ValueError(
                f"Manifest {partition_name} labels disagree with feature bundle labels. Examples: {preview}"
            )
    return np.asarray(sorted(int(x) for x in indices.tolist()), dtype=np.int64)


def build_manifest_split_assignments(
    *,
    bundle: LoadedFeatureBundle,
    manifest_json: Path,
    random_state: int,
    calibration_split_percent: int,
) -> SplitAssignments:
    if bundle.labels is None:
        raise ValueError("The input bundle must provide binary labels for paper-style detector fitting")
    labels = np.asarray(bundle.labels, dtype=np.int32)
    if not np.all(np.isin(labels, np.asarray([0, 1], dtype=np.int32))):
        raise ValueError("Paper-style detector fitting requires labels in {0, 1}")

    resolved_manifest = resolve_manifest_path(manifest_json)
    train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=resolved_manifest)
    train_indices = _manifest_partition_indices(
        bundle=bundle,
        items=train_items,
        partition_name="train",
    )
    test_indices = _manifest_partition_indices(
        bundle=bundle,
        items=infer_items,
        partition_name="infer",
    )
    overlap = sorted(set(train_indices.tolist()) & set(test_indices.tolist()))
    if overlap:
        preview = ", ".join(str(x) for x in overlap[:5])
        raise ValueError(f"Manifest train and infer partitions overlap after feature alignment: {preview}")

    calibration_fraction = float(calibration_split_percent) / 100.0
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_split_percent must be between 1 and 99, got {calibration_split_percent}"
        )

    candidate_vectors = build_candidate_stratify_vectors(bundle)
    warnings: list[str] = []
    calibration_kind, calibration_stratify = choose_stratify_vector(
        candidate_vectors=candidate_vectors,
        indices=train_indices,
        split_fraction=calibration_fraction,
    )
    if calibration_stratify is None:
        warnings.append(
            "Falling back to an unstratified manifest train/calibration split because no candidate "
            "stratification was feasible"
        )

    fit_indices, calibration_indices = train_test_split(
        train_indices,
        test_size=calibration_fraction,
        random_state=int(random_state) + 1,
        shuffle=True,
        stratify=calibration_stratify,
    )
    return SplitAssignments(
        split_strategy="manifest_defined",
        train_indices=np.asarray(sorted(int(x) for x in train_indices.tolist()), dtype=np.int64),
        fit_indices=np.asarray(sorted(int(x) for x in fit_indices), dtype=np.int64),
        calibration_indices=np.asarray(sorted(int(x) for x in calibration_indices), dtype=np.int64),
        test_indices=np.asarray(sorted(int(x) for x in test_indices.tolist()), dtype=np.int64),
        test_stratify_kind="manifest_defined",
        calibration_stratify_kind=str(calibration_kind),
        warnings=warnings,
        manifest_json=resolved_manifest,
    )


def build_experiment_split_assignments(
    *,
    bundle: LoadedFeatureBundle,
    manifest_json: Path | None,
    random_state: int,
    test_split_percent: int,
    calibration_split_percent: int,
) -> SplitAssignments:
    if manifest_json is not None:
        return build_manifest_split_assignments(
            bundle=bundle,
            manifest_json=manifest_json,
            random_state=random_state,
            calibration_split_percent=calibration_split_percent,
        )
    return build_split_assignments(
        bundle=bundle,
        random_state=random_state,
        test_split_percent=test_split_percent,
        calibration_split_percent=calibration_split_percent,
    )


def clean_reference_indices(labels: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    candidate_np = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    return np.asarray(
        [int(idx) for idx in candidate_np.tolist() if int(labels_np[int(idx)]) == 0],
        dtype=np.int64,
    )


def compute_reference_bank_stats(
    raw_features: np.ndarray,
    labels: np.ndarray,
    *,
    reference_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if reference_indices is None:
        clean_indices = np.asarray(np.flatnonzero(labels == 0), dtype=np.int64)
    else:
        clean_indices = clean_reference_indices(labels, np.asarray(reference_indices, dtype=np.int64))
    if clean_indices.size == 0:
        raise ValueError("Reference-bank fitting requires at least one clean sample (label == 0)")
    clean_values = np.asarray(raw_features[clean_indices], dtype=np.float64)
    mean = np.asarray(np.mean(clean_values, axis=0), dtype=np.float64)
    std = np.asarray(np.std(clean_values, axis=0), dtype=np.float64)
    std = np.where(std > 0.0, std, 1.0)
    return mean, std


def entropy_feature_mask(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [str(name).endswith(".spectral_entropy") for name in feature_names],
        dtype=bool,
    )


def normalize_paper_features(
    raw_features: np.ndarray,
    *,
    benign_mean: np.ndarray,
    benign_std: np.ndarray,
    entropy_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(raw_features, dtype=np.float64)
    z = (values - benign_mean.reshape(1, -1)) / benign_std.reshape(1, -1)
    if np.any(entropy_mask):
        z[:, entropy_mask] *= -1.0
    return np.asarray(0.5 * (1.0 + np.tanh(z / 2.0)), dtype=np.float64)


def coefficients_to_normalized_weights(coefficients: np.ndarray) -> np.ndarray:
    raw = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    clipped = np.maximum(raw, 0.0)
    total = float(np.sum(clipped, dtype=np.float64))
    if total <= 0.0:
        raise ValueError(
            "Logistic-regression fitting produced no positive coefficient mass after zero clipping; "
            "cannot form the paper-style normalized weighted sum"
        )
    return np.asarray(clipped / total, dtype=np.float64)


def metric_equal_uniform_weights(feature_names: Sequence[str]) -> np.ndarray:
    grouped_indices: dict[str, list[int]] = {suffix: [] for suffix in PAPER_QV_SELECTED_SUFFIXES}
    unknown_suffixes: set[str] = set()
    for idx, raw_name in enumerate(feature_names):
        suffix = _metric_name_from_feature_name(str(raw_name))
        if suffix not in grouped_indices:
            unknown_suffixes.add(suffix)
            continue
        grouped_indices[suffix].append(int(idx))

    if unknown_suffixes:
        preview = ", ".join(sorted(unknown_suffixes)[:5])
        raise ValueError(f"Unexpected paper q+v metric suffix(es) for clean-uniform scoring: {preview}")

    missing = [suffix for suffix, indices in grouped_indices.items() if not indices]
    if missing:
        preview = ", ".join(missing)
        raise ValueError(f"Clean-uniform scoring requires all paper q+v metric groups; missing: {preview}")

    weights = np.zeros(len(feature_names), dtype=np.float64)
    metric_mass = 1.0 / float(len(PAPER_QV_SELECTED_SUFFIXES))
    for indices in grouped_indices.values():
        weights[np.asarray(indices, dtype=np.int64)] = metric_mass / float(len(indices))
    return weights


def fit_paper_weighted_sum_detector(
    normalized_fit_features: np.ndarray,
    fit_labels: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray]:
    fit_matrix = np.asarray(normalized_fit_features, dtype=np.float64)
    fit_targets = np.asarray(fit_labels, dtype=np.int32).reshape(-1)
    model = LogisticRegression(
        fit_intercept=False,
        solver="lbfgs",
        max_iter=5000,
        C=1.0,
        class_weight=None,
    )
    model.fit(fit_matrix, fit_targets)
    weights = coefficients_to_normalized_weights(np.asarray(model.coef_[0], dtype=np.float64))
    return model, weights


def score_with_weights(normalized_features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(normalized_features, dtype=np.float64) @ np.asarray(weights, dtype=np.float64), dtype=np.float64)


def evaluate_binary_threshold(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    flagged = values >= float(threshold)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    tp = int(np.sum((labels == 1) & flagged))
    fp = int(np.sum((labels == 0) & flagged))
    tn = int(np.sum((labels == 0) & (~flagged)))
    fn = int(np.sum((labels == 1) & (~flagged)))
    recall = float(tp / positives) if positives > 0 else 0.0
    false_positive_rate = float(fp / negatives) if negatives > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    specificity = float(tn / negatives) if negatives > 0 else 0.0
    accuracy = float((tp + tn) / max(1, labels.shape[0]))
    balanced_accuracy = float((recall + specificity) / 2.0)
    youden_j = float(recall - false_positive_rate)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_positive": positives,
        "n_negative": negatives,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "fraction_flagged": float(np.mean(flagged)) if flagged.size > 0 else 0.0,
        "youden_j": youden_j,
    }


def select_calibration_threshold(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    clean_scores = np.asarray(values[labels == 0], dtype=np.float64)
    poison_scores = np.asarray(values[labels == 1], dtype=np.float64)
    if clean_scores.size == 0 or poison_scores.size == 0:
        raise ValueError("Threshold selection requires both clean and poisoned calibration samples")

    max_clean = float(np.max(clean_scores))
    min_poison = float(np.min(poison_scores))
    if max_clean < min_poison:
        threshold = float(max_clean + 0.25 * (min_poison - max_clean))
        metrics = evaluate_binary_threshold(labels_true=labels, scores=values, threshold=threshold)
        metrics["selection_method"] = "perfect_separation_gap_25pct"
        metrics["max_clean_score"] = max_clean
        metrics["min_poison_score"] = min_poison
        return metrics

    candidates = np.unique(values)[::-1]
    rows = [
        evaluate_binary_threshold(labels_true=labels, scores=values, threshold=float(candidate))
        for candidate in candidates.tolist()
    ]
    best = max(
        rows,
        key=lambda row: (
            float(row["youden_j"]),
            float(row["recall"]),
            -float(row["false_positive_rate"]),
            float(row["threshold"]),
        ),
    )
    selected = dict(best)
    selected["selection_method"] = "youden_j"
    return selected


def select_clean_percentile_threshold(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    percentile: float,
    source_partition: str = "train_clean",
) -> dict[str, Any]:
    pct = float(percentile)
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"clean_threshold_percentile must be between 0 and 100, got {percentile}")

    labels = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    clean_scores = np.asarray(values[labels == 0], dtype=np.float64)
    if clean_scores.size == 0:
        raise ValueError("Clean-percentile threshold selection requires at least one clean source sample")

    threshold = float(np.percentile(clean_scores, pct))
    selected = evaluate_binary_threshold(labels_true=labels, scores=values, threshold=threshold)
    resolved_source = str(source_partition)
    if resolved_source == "train_clean":
        selection_method = "clean_train_percentile"
    elif resolved_source == "calibration_clean":
        selection_method = "clean_calibration_percentile"
    else:
        selection_method = "clean_percentile"
    selected["selection_method"] = selection_method
    selected["percentile"] = pct
    selected["target_fpr"] = float(max(0.0, min(1.0, (100.0 - pct) / 100.0)))
    selected["source_partition"] = resolved_source
    selected["n_clean_samples"] = int(clean_scores.size)
    selected["clean_score_min"] = float(np.min(clean_scores))
    selected["clean_score_max"] = float(np.max(clean_scores))
    return selected


def _score_summary(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _score_metrics(labels_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    metrics: dict[str, float | None] = {
        "score_min": float(np.min(values)),
        "score_max": float(np.max(values)),
        "score_mean": float(np.mean(values)),
        "score_std": float(np.std(values)),
        "roc_auc": None,
        "average_precision": None,
    }
    unique = np.unique(labels)
    if unique.size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, values))
        metrics["average_precision"] = float(average_precision_score(labels, values))
    return metrics


def _metric_name_from_feature_name(feature_name: str) -> str:
    return str(feature_name).rpartition(".")[2]


def _weights_by_metric(feature_names: Sequence[str], weights: np.ndarray) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for name, value in zip(feature_names, np.asarray(weights, dtype=np.float64).tolist()):
        metric = _metric_name_from_feature_name(str(name))
        grouped[metric] = float(grouped.get(metric, 0.0) + float(value))
    return {key: float(grouped[key]) for key in sorted(grouped)}


def _save_score_rows(
    *,
    output_path: Path,
    bundle: LoadedFeatureBundle,
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "model_name",
                "label",
                "score",
                "flagged",
                "dataset_name",
                "subset_name",
                "model_family",
                "attack_name",
            ],
        )
        writer.writeheader()
        for row_index, score in zip(indices.tolist(), np.asarray(scores, dtype=np.float64).tolist()):
            model_name = str(bundle.model_names[int(row_index)])
            writer.writerow(
                {
                    "row_index": int(row_index),
                    "model_name": model_name,
                    "label": int(bundle.labels[int(row_index)]) if bundle.labels is not None else None,
                    "score": float(score),
                    "flagged": bool(float(score) >= float(threshold)),
                    "dataset_name": _dataset_reference_value(bundle, model_name, "dataset_name"),
                    "subset_name": _dataset_reference_value(bundle, model_name, "subset_name"),
                    "model_family": _dataset_reference_value(bundle, model_name, "model_family"),
                    "attack_name": _dataset_reference_value(bundle, model_name, "attack_name"),
                }
            )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and evaluate a paper-style q+v reference detector on an existing merged spectral feature bundle."
        )
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=DEFAULT_FEATURE_FILE,
        help="Input feature run name or explicit merged spectral_features.npy file",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help="Base directory used to resolve bare feature run names",
    )
    parser.add_argument(
        "--feature-output-run",
        type=str,
        default=None,
        help="Run id for the derived paper-style feature bundle under runs/feature_extract/<run>/merged",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for experiment artifacts",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional experiment run id. Defaults to the shared run-context timestamp format for single runs; "
            "for suites this is used as a prefix before each stable suite run id."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=list(PAPER_QV_SUITE_CHOICES),
        default=None,
        help="Run a predefined paper q+v baseline suite over existing CNN holdout manifests.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="For --suite, skip fitting and regenerate the suite summary from completed runs.",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for paper q+v suite summaries.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help=(
            "Optional joint train/infer manifest. When provided, train defines the detector training pool "
            "and infer defines the held-out test partition."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Optional directory of joint manifests to run sequentially.",
    )
    parser.add_argument(
        "--manifest-glob",
        type=str,
        default="*.json",
        help="Glob used with --manifest-root.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Deterministic seed used for all splits and fitting.",
    )
    parser.add_argument(
        "--test-split-percent",
        type=int,
        default=DEFAULT_TEST_SPLIT_PERCENT,
        help="Held-out test split percentage.",
    )
    parser.add_argument(
        "--calibration-split-percent",
        type=int,
        default=DEFAULT_CALIBRATION_SPLIT_PERCENT,
        help="Calibration split percentage applied to the post-test remaining pool.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=list(PAPER_QV_SCORING_MODE_CHOICES),
        default=SCORING_MODE_SUPERVISED_LOGREG,
        help=(
            "Score aggregation mode. supervised-logreg fits labeled logistic weights; "
            "clean-uniform uses metric-equal fixed weights and clean-only percentile thresholding."
        ),
    )
    parser.add_argument(
        "--clean-threshold-percentile",
        type=float,
        default=DEFAULT_CLEAN_THRESHOLD_PERCENTILE,
        help="Clean training score percentile used as the threshold for --scoring-mode clean-uniform.",
    )
    return parser


def run_paper_qv_reference_experiment(
    *,
    feature_file: Path,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    feature_output_run: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    manifest_json: Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_split_percent: int = DEFAULT_TEST_SPLIT_PERCENT,
    calibration_split_percent: int = DEFAULT_CALIBRATION_SPLIT_PERCENT,
    scoring_mode: str = SCORING_MODE_SUPERVISED_LOGREG,
    clean_threshold_percentile: float = DEFAULT_CLEAN_THRESHOLD_PERCENTILE,
) -> dict[str, Path]:
    started_at = perf_counter()
    resolved_scoring_mode = str(scoring_mode)
    if resolved_scoring_mode not in PAPER_QV_SCORING_MODE_CHOICES:
        raise ValueError(
            f"Unsupported paper q+v scoring mode {scoring_mode!r}; "
            f"supported modes: {list(PAPER_QV_SCORING_MODE_CHOICES)}"
        )

    bundle = load_feature_bundle(feature_file=feature_file, feature_root=feature_root)
    if bundle.labels is None:
        raise ValueError("The selected feature bundle does not expose labels required for fitting")

    feature_derivation_started_at = perf_counter()
    derived = export_paper_qv_feature_bundle(
        bundle=bundle,
        feature_root=feature_root,
        feature_output_run=feature_output_run,
    )
    feature_derivation_seconds = float(perf_counter() - feature_derivation_started_at)
    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_id,
    )

    splits = build_experiment_split_assignments(
        bundle=bundle,
        manifest_json=manifest_json,
        random_state=int(random_state),
        test_split_percent=int(test_split_percent),
        calibration_split_percent=int(calibration_split_percent),
    )
    labels = np.asarray(bundle.labels, dtype=np.int32)
    reference_source_partition = (
        "train" if resolved_scoring_mode == SCORING_MODE_CLEAN_UNIFORM else "fit"
    )
    reference_source_indices = (
        splits.train_indices if resolved_scoring_mode == SCORING_MODE_CLEAN_UNIFORM else splits.fit_indices
    )
    reference_indices = clean_reference_indices(labels, reference_source_indices)
    benign_mean, benign_std = compute_reference_bank_stats(
        derived.features,
        labels,
        reference_indices=reference_source_indices,
    )
    entropy_mask = entropy_feature_mask(derived.feature_names)
    normalized_features = normalize_paper_features(
        derived.features,
        benign_mean=benign_mean,
        benign_std=benign_std,
        entropy_mask=entropy_mask,
    )

    fit_labels = np.asarray(labels[splits.fit_indices], dtype=np.int32)
    calibration_labels = np.asarray(labels[splits.calibration_indices], dtype=np.int32)
    test_labels = np.asarray(labels[splits.test_indices], dtype=np.int32)
    raw_logistic_coefficients: np.ndarray | None = None
    logistic_regression_params: dict[str, Any] | None = None
    if resolved_scoring_mode == SCORING_MODE_SUPERVISED_LOGREG:
        model, normalized_weights = fit_paper_weighted_sum_detector(
            normalized_features[splits.fit_indices],
            fit_labels,
        )
        raw_logistic_coefficients = np.asarray(model.coef_[0], dtype=np.float64)
        logistic_regression_params = {
            "fit_intercept": False,
            "solver": "lbfgs",
            "max_iter": 5000,
            "C": 1.0,
            "class_weight": None,
        }
        score_aggregation = SCORE_AGGREGATION_SUPERVISED_LOGREG
    else:
        normalized_weights = metric_equal_uniform_weights(derived.feature_names)
        score_aggregation = SCORE_AGGREGATION_METRIC_EQUAL_UNIFORM

    train_scores = score_with_weights(normalized_features[splits.train_indices], normalized_weights)
    fit_scores = score_with_weights(normalized_features[splits.fit_indices], normalized_weights)
    calibration_scores = score_with_weights(
        normalized_features[splits.calibration_indices],
        normalized_weights,
    )
    test_scores = score_with_weights(normalized_features[splits.test_indices], normalized_weights)
    if resolved_scoring_mode == SCORING_MODE_SUPERVISED_LOGREG:
        selected_threshold = select_calibration_threshold(
            labels_true=calibration_labels,
            scores=calibration_scores,
        )
    else:
        selected_threshold = select_clean_percentile_threshold(
            labels_true=np.asarray(labels[splits.train_indices], dtype=np.int32),
            scores=train_scores,
            percentile=float(clean_threshold_percentile),
            source_partition="train_clean",
        )
    threshold = float(selected_threshold["threshold"])

    detector_payload = {
        "script_version": SCRIPT_VERSION,
        "scoring_mode": resolved_scoring_mode,
        "score_aggregation": score_aggregation,
        "feature_names": list(derived.feature_names),
        "block_names": list(derived.block_names),
        "feature_mapping": dict(PAPER_QV_FEATURE_MAPPING),
        "benign_mean": benign_mean.astype(np.float64),
        "benign_std": benign_std.astype(np.float64),
        "entropy_mask": entropy_mask.astype(bool),
        "normalized_weights": normalized_weights.astype(np.float64),
        "threshold": threshold,
        "selection_method": str(selected_threshold["selection_method"]),
        "feature_file": str(bundle.feature_file.resolve()),
        "derived_feature_file": str(derived.feature_path.resolve()),
        "split_strategy": splits.split_strategy,
        "manifest_json": str(splits.manifest_json) if splits.manifest_json is not None else None,
        "reference_bank_source_partition": reference_source_partition,
        "reference_bank_indices": [int(x) for x in reference_indices.tolist()],
        "random_state": int(random_state),
        "test_split_percent": int(test_split_percent),
        "calibration_split_percent": int(calibration_split_percent),
        "clean_threshold_percentile": float(clean_threshold_percentile),
        "test_stratify_kind": splits.test_stratify_kind,
        "calibration_stratify_kind": splits.calibration_stratify_kind,
        "fit_warnings": list(splits.warnings),
    }
    if logistic_regression_params is not None:
        detector_payload["logistic_regression_params"] = logistic_regression_params
    if raw_logistic_coefficients is not None:
        detector_payload["raw_logistic_coefficients"] = raw_logistic_coefficients
    detector_path = ctx.models_dir / DETECTOR_FILENAME
    joblib.dump(detector_payload, detector_path)

    selected_threshold_path = _write_json(
        ctx.reports_dir / SELECTED_THRESHOLD_FILENAME,
        selected_threshold,
    )
    metrics_payload = {
        "script_version": SCRIPT_VERSION,
        "scoring_mode": resolved_scoring_mode,
        "score_aggregation": score_aggregation,
        "clean_threshold_percentile": float(clean_threshold_percentile),
        "source_feature_file": str(bundle.feature_file.resolve()),
        "derived_feature_file": str(derived.feature_path.resolve()),
        "feature_dim": int(len(derived.feature_names)),
        "selected_metric_suffixes": list(PAPER_QV_SELECTED_SUFFIXES),
        "feature_mapping": dict(PAPER_QV_FEATURE_MAPPING),
        "weights_by_metric": _weights_by_metric(derived.feature_names, normalized_weights),
        "selected_threshold": dict(selected_threshold),
        "split_strategy": splits.split_strategy,
        "manifest_json": str(splits.manifest_json) if splits.manifest_json is not None else None,
        "reference_bank": {
            "source_partition": reference_source_partition,
            "n_clean_samples": int(reference_indices.size),
            "indices": [int(x) for x in reference_indices.tolist()],
        },
        "train": {
            "n_samples": int(splits.train_indices.size),
            "label_counts": _label_counts(labels[splits.train_indices].tolist()),
            **_score_metrics(labels[splits.train_indices], train_scores),
        },
        "fit": {
            "n_samples": int(splits.fit_indices.size),
            "label_counts": _label_counts(fit_labels.tolist()),
            **_score_metrics(fit_labels, fit_scores),
        },
        "calibration": {
            "n_samples": int(splits.calibration_indices.size),
            "label_counts": _label_counts(calibration_labels.tolist()),
            **_score_metrics(calibration_labels, calibration_scores),
        },
        "test": {
            "n_samples": int(splits.test_indices.size),
            "label_counts": _label_counts(test_labels.tolist()),
            **_score_metrics(test_labels, test_scores),
            "threshold_metrics": evaluate_binary_threshold(
                labels_true=test_labels,
                scores=test_scores,
                threshold=threshold,
            ),
        },
    }
    metrics_path = _write_json(ctx.reports_dir / METRICS_FILENAME, metrics_payload)

    split_manifest_path = _write_json(
        ctx.reports_dir / SPLIT_MANIFEST_FILENAME,
        {
            "script_version": SCRIPT_VERSION,
            "split_strategy": splits.split_strategy,
            "manifest_json": str(splits.manifest_json) if splits.manifest_json is not None else None,
            "random_state": int(random_state),
            "test_split_percent": int(test_split_percent),
            "calibration_split_percent": int(calibration_split_percent),
            "test_stratify_kind": splits.test_stratify_kind,
            "calibration_stratify_kind": splits.calibration_stratify_kind,
            "warnings": list(splits.warnings),
            "train_indices": [int(x) for x in splits.train_indices.tolist()],
            "fit_indices": [int(x) for x in splits.fit_indices.tolist()],
            "calibration_indices": [int(x) for x in splits.calibration_indices.tolist()],
            "test_indices": [int(x) for x in splits.test_indices.tolist()],
            "train_label_counts": _label_counts(labels[splits.train_indices].tolist()),
            "fit_label_counts": _label_counts(fit_labels.tolist()),
            "calibration_label_counts": _label_counts(calibration_labels.tolist()),
            "test_label_counts": _label_counts(test_labels.tolist()),
            "reference_bank_indices": [int(x) for x in reference_indices.tolist()],
        },
    )
    reference_bank_summary_path = _write_json(
        ctx.reports_dir / REFERENCE_BANK_SUMMARY_FILENAME,
        {
            "script_version": SCRIPT_VERSION,
            "source_feature_file": str(bundle.feature_file.resolve()),
            "source_partition": reference_source_partition,
            "split_strategy": splits.split_strategy,
            "manifest_json": str(splits.manifest_json) if splits.manifest_json is not None else None,
            "n_reference_clean_samples": int(reference_indices.size),
            "reference_bank_indices": [int(x) for x in reference_indices.tolist()],
            "source_label_counts": _label_counts(labels.tolist()),
            "train_label_counts": _label_counts(labels[splits.train_indices].tolist()),
            "fit_label_counts": _label_counts(fit_labels.tolist()),
            "score_summary_train": _score_summary(train_scores),
            "mean_zero_count": int(np.sum(np.isclose(benign_mean, 0.0))),
            "safe_std_unit_count": int(np.sum(np.isclose(benign_std, 1.0))),
            "score_summary_fit": _score_summary(fit_scores),
            "score_summary_calibration": _score_summary(calibration_scores),
            "score_summary_test": _score_summary(test_scores),
        },
    )

    calibration_scores_path = _save_score_rows(
        output_path=ctx.reports_dir / CALIBRATION_SCORES_FILENAME,
        bundle=bundle,
        indices=splits.calibration_indices,
        scores=calibration_scores,
        threshold=threshold,
    )
    test_scores_path = _save_score_rows(
        output_path=ctx.reports_dir / TEST_SCORES_FILENAME,
        bundle=bundle,
        indices=splits.test_indices,
        scores=test_scores,
        threshold=threshold,
    )

    ctx.add_artifact("derived_feature_file", derived.feature_path)
    ctx.add_artifact("derived_model_names", derived.model_names_path)
    if derived.labels_path is not None:
        ctx.add_artifact("derived_labels", derived.labels_path)
    ctx.add_artifact("derived_metadata", derived.metadata_path)
    ctx.add_artifact("derived_dataset_reference_report", derived.dataset_reference_report_path)
    ctx.add_artifact("detector_model", detector_path)
    ctx.add_artifact("selected_threshold", selected_threshold_path)
    ctx.add_artifact("metrics", metrics_path)
    ctx.add_artifact("split_manifest", split_manifest_path)
    ctx.add_artifact("reference_bank_summary", reference_bank_summary_path)
    ctx.add_artifact("calibration_scores_csv", calibration_scores_path)
    ctx.add_artifact("test_scores_csv", test_scores_path)
    total_seconds = float(perf_counter() - started_at)
    ctx.add_timing("feature_derivation_seconds", feature_derivation_seconds)
    ctx.add_timing("paper_qv_method_seconds", max(0.0, total_seconds - feature_derivation_seconds))
    ctx.add_timing("total_seconds", total_seconds)
    ctx.finalize(
        {
            "script_version": SCRIPT_VERSION,
            "pipeline": PIPELINE_NAME,
            "feature_file": str(bundle.feature_file.resolve()),
            "feature_root": str(Path(feature_root).expanduser().resolve()),
            "feature_output_run": str(derived.output_dir.parent.name),
            "output_root": str(Path(output_root).expanduser().resolve()),
            "scoring_mode": resolved_scoring_mode,
            "score_aggregation": score_aggregation,
            "clean_threshold_percentile": float(clean_threshold_percentile),
            "split_strategy": splits.split_strategy,
            "manifest_json": str(splits.manifest_json) if splits.manifest_json is not None else None,
            "random_state": int(random_state),
            "test_split_percent": int(test_split_percent),
            "calibration_split_percent": int(calibration_split_percent),
        }
    )
    return {
        "derived_feature_path": derived.feature_path,
        "derived_metadata_path": derived.metadata_path,
        "derived_dataset_reference_report_path": derived.dataset_reference_report_path,
        "detector_model_path": detector_path,
        "selected_threshold_path": selected_threshold_path,
        "metrics_path": metrics_path,
        "split_manifest_path": split_manifest_path,
        "reference_bank_summary_path": reference_bank_summary_path,
        "calibration_scores_csv": calibration_scores_path,
        "test_scores_csv": test_scores_path,
        "run_dir": ctx.run_dir,
    }


def _manifest_paths_from_root(manifest_root: Path, manifest_glob: str) -> list[Path]:
    resolved_root = manifest_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Manifest root not found: {resolved_root}")
    paths = sorted(path.resolve() for path in resolved_root.glob(str(manifest_glob)) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No manifests matched {manifest_glob!r} under {resolved_root}")
    return paths


def _rank_manifest_sort_key(path: Path) -> tuple[int, str]:
    match = _RANK_LEAVE_ONE_OUT_RE.fullmatch(path.stem)
    if match is None:
        return (10**12, path.name)
    rank = int(match.group(1).removeprefix("rank"))
    return (rank, path.name)


def _tbh_tba_rank_zero_shot_sort_key(path: Path) -> tuple[int, str]:
    match = _TBH_TBA_RANK_ZERO_SHOT_RE.fullmatch(path.stem)
    if match is None:
        return (10**12, path.name)
    rank = int(match.group(1).removeprefix("rank"))
    return (rank, path.name)


def _default_suite_manifest_root(suite: str, repo_root: Path) -> Path:
    if suite == "rank-leave-one-out":
        return repo_root / "manifests" / "leave_one_out"
    if suite == "adapter-leave-one-out":
        return repo_root / "runs" / "generated_manifests" / "leave_one_out_adapter"
    if suite == "architecture-leave-one-out":
        return repo_root / "runs" / "generated_manifests" / "leave_one_out_architecture"
    if suite == "tbh-tba-rank-zero-shot":
        return repo_root / "manifests" / "zero_shots" / "tbh+tba_rank_wise"
    raise ValueError(f"Unsupported paper q+v reference suite: {suite}")


def _suite_slug(suite: str) -> str:
    return str(suite).replace("-", "_")


def _percentile_slug(percentile: float) -> str:
    value = float(percentile)
    if value.is_integer():
        return f"p{int(value)}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "p" + text.replace(".", "_")


def _scoring_variant_suffix(*, scoring_mode: str, clean_threshold_percentile: float) -> str:
    if str(scoring_mode) == SCORING_MODE_SUPERVISED_LOGREG:
        return ""
    if str(scoring_mode) == SCORING_MODE_CLEAN_UNIFORM:
        return f"clean_uniform_{_percentile_slug(clean_threshold_percentile)}"
    raise ValueError(f"Unsupported paper q+v scoring mode: {scoring_mode}")


def _variant_run_id(
    run_id: str,
    *,
    scoring_mode: str,
    clean_threshold_percentile: float,
) -> str:
    suffix = _scoring_variant_suffix(
        scoring_mode=scoring_mode,
        clean_threshold_percentile=clean_threshold_percentile,
    )
    if not suffix or str(run_id).endswith(f"_{suffix}"):
        return str(run_id)
    return f"{run_id}_{suffix}"


def _summary_output_dir(output_root: Path, summary_output_dir: Path | None) -> Path:
    if summary_output_dir is not None:
        return summary_output_dir.expanduser().resolve()
    return (Path(output_root).expanduser() / PIPELINE_NAME / "summaries").resolve()


def _suite_summary_filename(
    suite: str,
    *,
    scoring_mode: str = SCORING_MODE_SUPERVISED_LOGREG,
    clean_threshold_percentile: float = DEFAULT_CLEAN_THRESHOLD_PERCENTILE,
) -> str:
    suffix = _scoring_variant_suffix(
        scoring_mode=scoring_mode,
        clean_threshold_percentile=clean_threshold_percentile,
    )
    if suite == "all-leave-one-out":
        if not suffix:
            return PAPER_QV_COMBINED_SUMMARY_FILENAME
        return f"paper_qv_reference_leave_one_out_{suffix}_summary.csv"
    suite_slug = _suite_slug(suite)
    if suffix:
        suite_slug = f"{suite_slug}_{suffix}"
    return PAPER_QV_SUMMARY_FILENAME_TEMPLATE.format(suite_slug=suite_slug)


def discover_paper_qv_reference_suite_manifests(
    suite: str,
    *,
    repo_root: Path | None = None,
    manifest_root: Path | None = None,
) -> tuple[PaperQvSuiteManifestSpec, ...]:
    """Discover existing CNN holdout manifests used by paper_qv_reference suites."""
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    suite_name = str(suite)
    if suite_name == "all-leave-one-out":
        if manifest_root is not None:
            raise ValueError("--manifest-root cannot be used with suite=all-leave-one-out")
        specs: list[PaperQvSuiteManifestSpec] = []
        for child_suite in ("rank-leave-one-out", "adapter-leave-one-out", "architecture-leave-one-out"):
            specs.extend(
                discover_paper_qv_reference_suite_manifests(
                    child_suite,
                    repo_root=resolved_repo_root,
                )
            )
        return tuple(specs)

    resolved_manifest_root = (
        manifest_root.expanduser().resolve()
        if manifest_root is not None
        else _default_suite_manifest_root(suite_name, resolved_repo_root)
    )

    if suite_name == "rank-leave-one-out":
        manifest_paths = _manifest_paths_from_root(resolved_manifest_root, RANK_LEAVE_ONE_OUT_GLOB)
        rank_paths = [
            path
            for path in manifest_paths
            if _RANK_LEAVE_ONE_OUT_RE.fullmatch(path.stem) is not None
        ]
        if not rank_paths:
            raise FileNotFoundError(
                f"No CNN rank holdout manifests matched {RANK_LEAVE_ONE_OUT_GLOB!r} under {resolved_manifest_root}"
            )
        specs = []
        for manifest_path in sorted(rank_paths, key=_rank_manifest_sort_key):
            match = _RANK_LEAVE_ONE_OUT_RE.fullmatch(manifest_path.stem)
            assert match is not None
            heldout_group = match.group(1)
            specs.append(
                PaperQvSuiteManifestSpec(
                    suite=suite_name,
                    heldout_group=heldout_group,
                    run_id=f"rank_loo_paper_qv_holdout_{heldout_group}",
                    manifest_path=manifest_path,
                )
            )
        return tuple(specs)

    if suite_name == "adapter-leave-one-out":
        manifest_paths = _manifest_paths_from_root(resolved_manifest_root, "holdout_adapter_*.json")
        return tuple(
            PaperQvSuiteManifestSpec(
                suite=suite_name,
                heldout_group=path.stem.removeprefix("holdout_adapter_"),
                run_id=f"adapter_loo_paper_qv_{path.stem}",
                manifest_path=path,
            )
            for path in manifest_paths
        )

    if suite_name == "architecture-leave-one-out":
        manifest_paths = _manifest_paths_from_root(resolved_manifest_root, "holdout_architecture_*.json")
        return tuple(
            PaperQvSuiteManifestSpec(
                suite=suite_name,
                heldout_group=path.stem.removeprefix("holdout_architecture_"),
                run_id=f"architecture_loo_paper_qv_{path.stem}",
                manifest_path=path,
            )
            for path in manifest_paths
        )

    if suite_name == "tbh-tba-rank-zero-shot":
        manifest_paths = _manifest_paths_from_root(resolved_manifest_root, TBH_TBA_RANK_ZERO_SHOT_GLOB)
        rank_paths = [
            path
            for path in manifest_paths
            if _TBH_TBA_RANK_ZERO_SHOT_RE.fullmatch(path.stem) is not None
        ]
        if not rank_paths:
            raise FileNotFoundError(
                f"No TBH+TBA rank zero-shot manifests matched {TBH_TBA_RANK_ZERO_SHOT_GLOB!r} "
                f"under {resolved_manifest_root}"
            )
        specs = []
        for manifest_path in sorted(rank_paths, key=_tbh_tba_rank_zero_shot_sort_key):
            match = _TBH_TBA_RANK_ZERO_SHOT_RE.fullmatch(manifest_path.stem)
            assert match is not None
            heldout_group = match.group(1)
            specs.append(
                PaperQvSuiteManifestSpec(
                    suite=suite_name,
                    heldout_group=heldout_group,
                    run_id=f"tbh_tba_rank_zero_shot_paper_qv_r256_to_{heldout_group}",
                    manifest_path=manifest_path,
                )
            )
        return tuple(specs)

    raise ValueError(f"Unsupported paper q+v reference suite: {suite_name}")


PAPER_QV_SUMMARY_FIELDNAMES = [
    "suite",
    "heldout_group",
    "manifest_path",
    "run_id",
    "test_roc_auc",
    "test_average_precision",
    "test_recall",
    "test_specificity",
    "test_balanced_accuracy",
    "threshold",
    "threshold_selection_method",
    "threshold_source_partition",
    "threshold_percentile",
    "threshold_target_fpr",
    "threshold_clean_count",
    "test_n_samples",
    "reference_clean_count",
    "feature_dim",
]


def _paper_qv_summary_row(
    spec: PaperQvSuiteManifestSpec,
    *,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = Path(output_root).expanduser().resolve() / PIPELINE_NAME / spec.run_id
    metrics_path = run_dir / "reports" / METRICS_FILENAME
    threshold_path = run_dir / "reports" / SELECTED_THRESHOLD_FILENAME
    metrics = _read_json(metrics_path)
    selected_threshold = _read_json(threshold_path)

    test_metrics = metrics.get("test", {})
    if not isinstance(test_metrics, dict):
        test_metrics = {}
    threshold_metrics = test_metrics.get("threshold_metrics", {})
    if not isinstance(threshold_metrics, dict):
        threshold_metrics = {}
    reference_bank = metrics.get("reference_bank", {})
    if not isinstance(reference_bank, dict):
        reference_bank = {}
    balanced_accuracy = threshold_metrics.get("balanced_accuracy")
    if balanced_accuracy is None and threshold_metrics.get("recall") is not None and threshold_metrics.get("specificity") is not None:
        balanced_accuracy = (float(threshold_metrics["recall"]) + float(threshold_metrics["specificity"])) / 2.0

    return {
        "suite": spec.suite,
        "heldout_group": spec.heldout_group,
        "manifest_path": str(spec.manifest_path),
        "run_id": spec.run_id,
        "test_roc_auc": test_metrics.get("roc_auc"),
        "test_average_precision": test_metrics.get("average_precision"),
        "test_recall": threshold_metrics.get("recall"),
        "test_specificity": threshold_metrics.get("specificity"),
        "test_balanced_accuracy": balanced_accuracy,
        "threshold": selected_threshold.get("threshold", threshold_metrics.get("threshold")),
        "threshold_selection_method": selected_threshold.get("selection_method"),
        "threshold_source_partition": selected_threshold.get("source_partition"),
        "threshold_percentile": selected_threshold.get("percentile"),
        "threshold_target_fpr": selected_threshold.get("target_fpr"),
        "threshold_clean_count": selected_threshold.get("n_clean_samples"),
        "test_n_samples": test_metrics.get("n_samples"),
        "reference_clean_count": reference_bank.get("n_clean_samples"),
        "feature_dim": metrics.get("feature_dim"),
    }


def _filter_manifest_items_to_available_rows(
    items: Sequence[ManifestItem],
    *,
    available_model_names: set[str],
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    missing: list[str] = []
    for item in items:
        model_name = str(item.model_name)
        if model_name in available_model_names:
            kept.append(str(item.raw_entry))
        else:
            missing.append(model_name)
    return kept, missing


def _materialize_available_suite_manifest(
    spec: PaperQvSuiteManifestSpec,
    *,
    available_model_names: set[str],
    output_root: Path,
) -> tuple[PaperQvSuiteManifestSpec, list[str]]:
    train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=spec.manifest_path)
    train_entries, missing_train = _filter_manifest_items_to_available_rows(
        train_items,
        available_model_names=available_model_names,
    )
    infer_entries, missing_infer = _filter_manifest_items_to_available_rows(
        infer_items,
        available_model_names=available_model_names,
    )
    missing = [*missing_train, *missing_infer]
    if not missing:
        return spec, []
    if not train_entries:
        raise ValueError(
            f"Filtering unavailable feature rows would empty the train partition for {spec.manifest_path}"
        )
    if not infer_entries:
        raise ValueError(
            f"Filtering unavailable feature rows would empty the infer partition for {spec.manifest_path}"
        )

    filtered_root = (
        Path(output_root).expanduser().resolve()
        / PIPELINE_NAME
        / "generated_manifests"
        / _suite_slug(spec.suite)
    )
    filtered_path = filtered_root / spec.manifest_path.name
    filtered_payload = {
        "source_manifest": str(spec.manifest_path),
        "filtered_for_feature_bundle": True,
        "missing_model_names": sorted(set(missing)),
        "train": train_entries,
        "infer": infer_entries,
    }
    _write_json(filtered_path, filtered_payload)
    return (
        PaperQvSuiteManifestSpec(
            suite=spec.suite,
            heldout_group=spec.heldout_group,
            run_id=spec.run_id,
            manifest_path=filtered_path,
        ),
        missing,
    )


def _materialize_available_suite_manifests(
    specs: Sequence[PaperQvSuiteManifestSpec],
    *,
    feature_file: Path,
    feature_root: Path,
    output_root: Path,
) -> tuple[PaperQvSuiteManifestSpec, ...]:
    bundle = load_feature_bundle(feature_file=feature_file, feature_root=feature_root)
    available_model_names = {str(name) for name in bundle.model_names}
    filtered_specs: list[PaperQvSuiteManifestSpec] = []
    for spec in specs:
        filtered_spec, missing = _materialize_available_suite_manifest(
            spec,
            available_model_names=available_model_names,
            output_root=output_root,
        )
        filtered_specs.append(filtered_spec)
        if missing:
            preview = ", ".join(sorted(set(missing))[:5])
            print(
                f"Filtered {len(set(missing))} unavailable model(s) from {spec.manifest_path}: {preview}"
            )
    return tuple(filtered_specs)


def generate_paper_qv_reference_summary(
    specs: Sequence[PaperQvSuiteManifestSpec],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    summary_output_dir: Path | None = None,
    filename: str | None = None,
    scoring_mode: str = SCORING_MODE_SUPERVISED_LOGREG,
    clean_threshold_percentile: float = DEFAULT_CLEAN_THRESHOLD_PERCENTILE,
) -> Path:
    if not specs:
        raise ValueError("No paper q+v suite specs were provided for summary generation")

    suite_names = sorted({spec.suite for spec in specs})
    suite = suite_names[0] if len(suite_names) == 1 else "all-leave-one-out"
    output_dir = _summary_output_dir(output_root, summary_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / (
        filename
        or _suite_summary_filename(
            suite,
            scoring_mode=scoring_mode,
            clean_threshold_percentile=clean_threshold_percentile,
        )
    )

    rows = [_paper_qv_summary_row(spec, output_root=output_root) for spec in specs]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_QV_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def run_paper_qv_reference_suite(
    *,
    suite: str,
    feature_file: Path,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    feature_output_run: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    manifest_root: Path | None = None,
    summary_output_dir: Path | None = None,
    run_id_prefix: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_split_percent: int = DEFAULT_TEST_SPLIT_PERCENT,
    calibration_split_percent: int = DEFAULT_CALIBRATION_SPLIT_PERCENT,
    scoring_mode: str = SCORING_MODE_SUPERVISED_LOGREG,
    clean_threshold_percentile: float = DEFAULT_CLEAN_THRESHOLD_PERCENTILE,
    summary_only: bool = False,
) -> PaperQvSuiteRunOutputs:
    specs = discover_paper_qv_reference_suite_manifests(
        suite,
        manifest_root=manifest_root,
    )
    resolved_specs = tuple(
        PaperQvSuiteManifestSpec(
            suite=spec.suite,
            heldout_group=spec.heldout_group,
            run_id=_variant_run_id(
                f"{run_id_prefix}_{spec.run_id}" if run_id_prefix else spec.run_id,
                scoring_mode=scoring_mode,
                clean_threshold_percentile=clean_threshold_percentile,
            ),
            manifest_path=spec.manifest_path,
        )
        for spec in specs
    )

    outputs_by_run_id: dict[str, dict[str, Path]] = {}
    run_specs = resolved_specs
    if not summary_only:
        run_specs = _materialize_available_suite_manifests(
            resolved_specs,
            feature_file=feature_file,
            feature_root=feature_root,
            output_root=output_root,
        )
        for spec in run_specs:
            outputs_by_run_id[spec.run_id] = run_paper_qv_reference_experiment(
                feature_file=feature_file,
                feature_root=feature_root,
                feature_output_run=feature_output_run,
                output_root=output_root,
                run_id=spec.run_id,
                manifest_json=spec.manifest_path,
                random_state=random_state,
                test_split_percent=test_split_percent,
                calibration_split_percent=calibration_split_percent,
                scoring_mode=scoring_mode,
                clean_threshold_percentile=clean_threshold_percentile,
            )

    summary_path = generate_paper_qv_reference_summary(
        run_specs,
        output_root=output_root,
        summary_output_dir=summary_output_dir,
        scoring_mode=scoring_mode,
        clean_threshold_percentile=clean_threshold_percentile,
    )
    combined_summary_path: Path | None = None
    if str(suite) == "all-leave-one-out":
        combined_summary_path = summary_path
        for child_suite in ("rank-leave-one-out", "adapter-leave-one-out", "architecture-leave-one-out"):
            child_specs = [spec for spec in run_specs if spec.suite == child_suite]
            if child_specs:
                generate_paper_qv_reference_summary(
                    child_specs,
                    output_root=output_root,
                    summary_output_dir=summary_output_dir,
                    scoring_mode=scoring_mode,
                    clean_threshold_percentile=clean_threshold_percentile,
                )

    return PaperQvSuiteRunOutputs(
        suite=str(suite),
        run_outputs=outputs_by_run_id,
        summary_path=summary_path,
        combined_summary_path=combined_summary_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.suite is not None:
        if args.manifest_json is not None:
            parser.error("--manifest-json cannot be used with --suite")
        outputs = run_paper_qv_reference_suite(
            suite=args.suite,
            feature_file=args.feature_file,
            feature_root=args.feature_root,
            feature_output_run=args.feature_output_run,
            output_root=args.output_root,
            manifest_root=args.manifest_root,
            summary_output_dir=args.summary_output_dir,
            run_id_prefix=args.run_id,
            random_state=args.random_state,
            test_split_percent=args.test_split_percent,
            calibration_split_percent=args.calibration_split_percent,
            scoring_mode=args.scoring_mode,
            clean_threshold_percentile=args.clean_threshold_percentile,
            summary_only=bool(args.summary_only),
        )
        mode = "summary regenerated" if args.summary_only else "complete"
        print(f"Paper-style q+v reference suite {mode}: {args.suite}")
        print(f"summary_path: {outputs.summary_path}")
        if outputs.combined_summary_path is not None:
            print(f"combined_summary_path: {outputs.combined_summary_path}")
        for run_id, run_outputs in outputs.run_outputs.items():
            print(f"run_id: {run_id}")
            for key, path in run_outputs.items():
                print(f"{key}: {path}")
        return 0

    if args.summary_only:
        parser.error("--summary-only requires --suite")

    if args.manifest_root is not None:
        if args.manifest_json is not None:
            parser.error("--manifest-json and --manifest-root are mutually exclusive")
        manifest_paths = _manifest_paths_from_root(args.manifest_root, args.manifest_glob)
        print(f"Running paper-style q+v reference detector for {len(manifest_paths)} manifest(s)")
        for manifest_path in manifest_paths:
            child_run_id = f"{args.run_id}_{manifest_path.stem}" if args.run_id else manifest_path.stem
            child_run_id = _variant_run_id(
                child_run_id,
                scoring_mode=args.scoring_mode,
                clean_threshold_percentile=args.clean_threshold_percentile,
            )
            outputs = run_paper_qv_reference_experiment(
                feature_file=args.feature_file,
                feature_root=args.feature_root,
                feature_output_run=args.feature_output_run,
                output_root=args.output_root,
                run_id=child_run_id,
                manifest_json=manifest_path,
                random_state=args.random_state,
                test_split_percent=args.test_split_percent,
                calibration_split_percent=args.calibration_split_percent,
                scoring_mode=args.scoring_mode,
                clean_threshold_percentile=args.clean_threshold_percentile,
            )
            print(f"manifest: {manifest_path}")
            for key, path in outputs.items():
                print(f"{key}: {path}")
        return 0

    outputs = run_paper_qv_reference_experiment(
        feature_file=args.feature_file,
        feature_root=args.feature_root,
        feature_output_run=args.feature_output_run,
        output_root=args.output_root,
        run_id=args.run_id,
        manifest_json=args.manifest_json,
        random_state=args.random_state,
        test_split_percent=args.test_split_percent,
        calibration_split_percent=args.calibration_split_percent,
        scoring_mode=args.scoring_mode,
        clean_threshold_percentile=args.clean_threshold_percentile,
    )
    print("Paper-style q+v reference detector complete")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0
