from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
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
    fit_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    test_stratify_kind: str
    calibration_stratify_kind: str
    warnings: list[str]


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
        fit_indices=np.asarray(sorted(int(x) for x in fit_indices), dtype=np.int64),
        calibration_indices=np.asarray(sorted(int(x) for x in calibration_indices), dtype=np.int64),
        test_indices=np.asarray(sorted(int(x) for x in test_indices), dtype=np.int64),
        test_stratify_kind=str(test_kind),
        calibration_stratify_kind=str(calibration_kind),
        warnings=warnings,
    )


def compute_reference_bank_stats(
    raw_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    clean_mask = labels == 0
    if not bool(np.any(clean_mask)):
        raise ValueError("Reference-bank fitting requires at least one clean sample (label == 0)")
    clean_values = np.asarray(raw_features[clean_mask], dtype=np.float64)
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
        help="Optional experiment run id. Defaults to the shared run-context timestamp format.",
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
    return parser


def run_paper_qv_reference_experiment(
    *,
    feature_file: Path,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    feature_output_run: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_split_percent: int = DEFAULT_TEST_SPLIT_PERCENT,
    calibration_split_percent: int = DEFAULT_CALIBRATION_SPLIT_PERCENT,
) -> dict[str, Path]:
    started_at = perf_counter()
    bundle = load_feature_bundle(feature_file=feature_file, feature_root=feature_root)
    if bundle.labels is None:
        raise ValueError("The selected feature bundle does not expose labels required for fitting")

    derived = export_paper_qv_feature_bundle(
        bundle=bundle,
        feature_root=feature_root,
        feature_output_run=feature_output_run,
    )
    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_id,
    )

    benign_mean, benign_std = compute_reference_bank_stats(derived.features, np.asarray(bundle.labels, dtype=np.int32))
    entropy_mask = entropy_feature_mask(derived.feature_names)
    normalized_features = normalize_paper_features(
        derived.features,
        benign_mean=benign_mean,
        benign_std=benign_std,
        entropy_mask=entropy_mask,
    )
    splits = build_split_assignments(
        bundle=bundle,
        random_state=int(random_state),
        test_split_percent=int(test_split_percent),
        calibration_split_percent=int(calibration_split_percent),
    )

    fit_labels = np.asarray(bundle.labels[splits.fit_indices], dtype=np.int32)
    calibration_labels = np.asarray(bundle.labels[splits.calibration_indices], dtype=np.int32)
    test_labels = np.asarray(bundle.labels[splits.test_indices], dtype=np.int32)
    model, normalized_weights = fit_paper_weighted_sum_detector(
        normalized_features[splits.fit_indices],
        fit_labels,
    )

    fit_scores = score_with_weights(normalized_features[splits.fit_indices], normalized_weights)
    calibration_scores = score_with_weights(
        normalized_features[splits.calibration_indices],
        normalized_weights,
    )
    test_scores = score_with_weights(normalized_features[splits.test_indices], normalized_weights)
    selected_threshold = select_calibration_threshold(
        labels_true=calibration_labels,
        scores=calibration_scores,
    )
    threshold = float(selected_threshold["threshold"])

    detector_payload = {
        "script_version": SCRIPT_VERSION,
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
        "random_state": int(random_state),
        "test_split_percent": int(test_split_percent),
        "calibration_split_percent": int(calibration_split_percent),
        "test_stratify_kind": splits.test_stratify_kind,
        "calibration_stratify_kind": splits.calibration_stratify_kind,
        "fit_warnings": list(splits.warnings),
        "logistic_regression_params": {
            "fit_intercept": False,
            "solver": "lbfgs",
            "max_iter": 5000,
            "C": 1.0,
            "class_weight": None,
        },
        "raw_logistic_coefficients": np.asarray(model.coef_[0], dtype=np.float64),
    }
    detector_path = ctx.models_dir / DETECTOR_FILENAME
    joblib.dump(detector_payload, detector_path)

    selected_threshold_path = _write_json(
        ctx.reports_dir / SELECTED_THRESHOLD_FILENAME,
        selected_threshold,
    )
    metrics_payload = {
        "script_version": SCRIPT_VERSION,
        "source_feature_file": str(bundle.feature_file.resolve()),
        "derived_feature_file": str(derived.feature_path.resolve()),
        "feature_dim": int(len(derived.feature_names)),
        "selected_metric_suffixes": list(PAPER_QV_SELECTED_SUFFIXES),
        "feature_mapping": dict(PAPER_QV_FEATURE_MAPPING),
        "weights_by_metric": _weights_by_metric(derived.feature_names, normalized_weights),
        "fit": {
            "n_samples": int(splits.fit_indices.size),
            "label_counts": _label_counts(fit_labels.tolist()),
            **_score_metrics(fit_labels, fit_scores),
        },
        "calibration": {
            "n_samples": int(splits.calibration_indices.size),
            "label_counts": _label_counts(calibration_labels.tolist()),
            **_score_metrics(calibration_labels, calibration_scores),
            "selected_threshold": dict(selected_threshold),
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
            "random_state": int(random_state),
            "test_split_percent": int(test_split_percent),
            "calibration_split_percent": int(calibration_split_percent),
            "test_stratify_kind": splits.test_stratify_kind,
            "calibration_stratify_kind": splits.calibration_stratify_kind,
            "warnings": list(splits.warnings),
            "fit_indices": [int(x) for x in splits.fit_indices.tolist()],
            "calibration_indices": [int(x) for x in splits.calibration_indices.tolist()],
            "test_indices": [int(x) for x in splits.test_indices.tolist()],
            "fit_label_counts": _label_counts(fit_labels.tolist()),
            "calibration_label_counts": _label_counts(calibration_labels.tolist()),
            "test_label_counts": _label_counts(test_labels.tolist()),
        },
    )
    clean_mask = np.asarray(bundle.labels, dtype=np.int32) == 0
    reference_bank_summary_path = _write_json(
        ctx.reports_dir / REFERENCE_BANK_SUMMARY_FILENAME,
        {
            "script_version": SCRIPT_VERSION,
            "source_feature_file": str(bundle.feature_file.resolve()),
            "n_reference_clean_samples": int(np.sum(clean_mask)),
            "source_label_counts": _label_counts(np.asarray(bundle.labels, dtype=np.int32).tolist()),
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
    ctx.add_timing("total_seconds", perf_counter() - started_at)
    ctx.finalize(
        {
            "script_version": SCRIPT_VERSION,
            "pipeline": PIPELINE_NAME,
            "feature_file": str(bundle.feature_file.resolve()),
            "feature_root": str(Path(feature_root).expanduser().resolve()),
            "feature_output_run": str(derived.output_dir.parent.name),
            "output_root": str(Path(output_root).expanduser().resolve()),
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = run_paper_qv_reference_experiment(
        feature_file=args.feature_file,
        feature_root=args.feature_root,
        feature_output_run=args.feature_output_run,
        output_root=args.output_root,
        run_id=args.run_id,
        random_state=args.random_state,
        test_split_percent=args.test_split_percent,
        calibration_split_percent=args.calibration_split_percent,
    )
    print("Paper-style q+v reference detector complete")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0
