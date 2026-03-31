from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import joblib
except Exception:  # pragma: no cover - fallback for minimal environments
    joblib = None
import pickle

from ..features.spectral import (
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    build_spectral_feature_names,
    resolve_spectral_features,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
    sanitize_spectral_metadata,
    spectral_block_lora_dims_by_block,
    spectral_extractor_params,
)
from ..unsupervised.reporting import (
    compute_infer_threshold_rows,
    compute_infer_threshold_rows_from_inference,
    compute_offline_metrics,
    save_score_csv,
    summarize_scores,
)
from ..utilities.artifacts.export_winner_feature_weights import (
    cleanup_winner_feature_weights_export,
    export_winner_feature_weights,
    merge_winner_feature_weights_export,
    prepare_winner_feature_weights_export,
    run_winner_feature_weights_export_worker,
)
from ..utilities.artifacts.spectral_metadata import (
    dataset_layouts_from_source,
    load_spectral_metadata,
    resolve_dataset_reference_for_metadata,
    write_spectral_metadata,
)
from ..utilities.core.manifest import (
    AttackSampleIdentity,
    infer_attack_sample_identities,
    parse_joint_manifest_json,
    parse_joint_manifest_json_by_model_name,
    parse_single_manifest_json,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
)
from ..utilities.core.run_context import RunContext, create_run_context
from ..utilities.core.serialization import json_ready
from ..utilities.merge.merge_feature_files import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    _resolve_existing_companion_path,
    _resolve_feature_extract_root,
    _resolve_input_feature_path,
)
from .distributed import (
    build_slurm_array_next_steps,
    resolve_slurm_cpus_per_task,
    resolve_slurm_max_concurrent,
)
from .registry import (
    candidate_params,
    create,
    model_complexity_rank,
    normalization_policy,
    registered_models,
)


SCRIPT_VERSION = "1.0.0"
PIPELINE_NAME = "supervised"
FINALIZE_STATE_FILENAME = "finalize_state.json"
FINALIZE_EXPORT_TRAIN_FEATURES_FILENAME = "winner_feature_weights_train_features.npy"
FINALIZE_EXPORT_TRAIN_LABELS_FILENAME = "winner_feature_weights_train_labels.npy"
SELECTED_THRESHOLD_FILENAME = "selected_threshold.json"


def _detect_manifest_mode(manifest_json: Path) -> str:
    resolved_manifest_json = resolve_manifest_path(manifest_json)
    with open(resolved_manifest_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "train" in payload and "infer" in payload:
        return "joint"
    return "single"


def _labels_from_items(items: list[Any]) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    raw_labels = [item.label for item in items]
    values = np.asarray([int(label) if label is not None else -1 for label in raw_labels], dtype=np.int32)
    known = np.asarray([label is not None for label in raw_labels], dtype=bool)
    return values, known, raw_labels


def _resolve_train_split_percent(train_split_percent: int) -> int:
    resolved = int(train_split_percent)
    if resolved < 1 or resolved > 100:
        raise ValueError(f"train_split must be in the range [1, 100], got {resolved}")
    return resolved


def _resolve_calibration_split_percent(calibration_split_percent: int | None) -> int | None:
    if calibration_split_percent is None:
        return None
    resolved = int(calibration_split_percent)
    if resolved < 1 or resolved > 99:
        raise ValueError(f"calibration_split must be in the range [1, 99], got {resolved}")
    return resolved


def _resolve_accepted_fprs(
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    if accepted_fpr is None:
        return None

    raw_values: list[float]
    if isinstance(accepted_fpr, (list, tuple)):
        raw_values = [float(value) for value in accepted_fpr]
    else:
        raw_values = [float(accepted_fpr)]

    if not raw_values:
        raise ValueError("accepted_fpr must include at least one value when provided")

    resolved_values: list[float] = []
    seen_values: set[float] = set()
    for value in raw_values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"accepted_fpr must be in the range [0.0, 1.0], got {value}")
        if value in seen_values:
            continue
        seen_values.add(value)
        resolved_values.append(float(value))
    return resolved_values


def _resolve_supervised_feature_bundle_paths(
    feature_spec: Path,
    *,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
) -> tuple[Path, Path, Path | None]:
    resolved_feature_root = _resolve_feature_extract_root(feature_root)
    candidate = feature_spec.expanduser()
    local_candidate = candidate if candidate.is_absolute() else (Path.cwd().resolve() / candidate)
    resolved_local_candidate = local_candidate.resolve()

    if resolved_local_candidate.exists():
        resolved_feature_path = (
            resolved_local_candidate / "spectral_features.npy"
            if resolved_local_candidate.is_dir()
            else resolved_local_candidate
        )
    else:
        resolved_feature_path = _resolve_input_feature_path(
            feature_spec,
            feature_root=resolved_feature_root,
        )

    resolved_feature_path = resolved_feature_path.expanduser().resolve()
    if not resolved_feature_path.exists():
        raise FileNotFoundError(f"Feature bundle not found: {resolved_feature_path}")

    resolved_model_names_path = _resolve_existing_companion_path(
        resolved_feature_path,
        "_model_names.json",
        required=True,
    ).expanduser().resolve()
    metadata_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        "_metadata.json",
        required=False,
    )
    resolved_metadata_path = (
        metadata_candidate.expanduser().resolve() if metadata_candidate.exists() else None
    )
    return resolved_feature_path, resolved_model_names_path, resolved_metadata_path


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _resolve_holdout_subset_count(
    *,
    bucket_size: int,
    subset_percent: int,
    bucket_name: str,
    split_name: str,
    subset_label: str,
) -> tuple[int, bool]:
    if bucket_size < 2:
        raise ValueError(
            f"{split_name} requires at least two samples in each split bucket when creating a holdout split, "
            f"but '{bucket_name}' has {bucket_size}"
        )

    raw_subset = (int(bucket_size) * int(subset_percent)) / 100.0
    rounded_subset = _round_half_up(raw_subset)
    subset_count = min(max(rounded_subset, 1), int(bucket_size) - 1)
    return int(subset_count), bool(subset_count != rounded_subset)


def _label_count_rows(labels: np.ndarray) -> list[dict[str, int]]:
    if labels.size == 0:
        return []
    unique, counts = np.unique(labels, return_counts=True)
    return [
        {
            "label": int(label),
            "count": int(count),
        }
        for label, count in zip(unique.tolist(), counts.tolist())
    ]


def _build_single_manifest_split_summary(
    *,
    labels: np.ndarray,
    train_indices: np.ndarray,
    infer_indices: np.ndarray,
    train_split_percent: int,
    random_state: int,
    strategy: str,
) -> dict[str, Any]:
    train_labels = labels[train_indices]
    infer_labels = labels[infer_indices] if infer_indices.size > 0 else np.asarray([], dtype=np.int32)
    return {
        "strategy": strategy,
        "requested_train_split_percent": int(train_split_percent),
        "random_state": int(random_state),
        "n_train": int(train_indices.size),
        "n_inference": int(infer_indices.size),
        "train_label_counts": _label_count_rows(np.asarray(train_labels, dtype=np.int32)),
        "inference_label_counts": _label_count_rows(np.asarray(infer_labels, dtype=np.int32)),
    }


def _build_calibration_split_summary(
    *,
    labels: np.ndarray,
    fit_train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
    strategy: str,
) -> dict[str, Any]:
    fit_train_labels = labels[fit_train_indices]
    calibration_labels = labels[calibration_indices]
    return {
        "strategy": strategy,
        "requested_calibration_split_percent": int(calibration_split_percent),
        "random_state": int(random_state),
        "n_fit_train": int(fit_train_indices.size),
        "n_calibration": int(calibration_indices.size),
        "fit_train_label_counts": _label_count_rows(np.asarray(fit_train_labels, dtype=np.int32)),
        "calibration_label_counts": _label_count_rows(np.asarray(calibration_labels, dtype=np.int32)),
    }


def _resolve_holdout_train_count(
    *,
    bucket_size: int,
    train_split_percent: int,
    bucket_name: str,
    split_name: str,
) -> tuple[int, bool]:
    if bucket_size < 2:
        raise ValueError(
            f"{split_name} requires at least two samples in each split bucket when creating a holdout split, "
            f"but '{bucket_name}' has {bucket_size}"
        )

    raw_train = (int(bucket_size) * int(train_split_percent)) / 100.0
    rounded_train = _round_half_up(raw_train)
    train_count = min(max(rounded_train, 1), int(bucket_size) - 1)
    return int(train_count), bool(train_count != rounded_train)


def _build_single_manifest_stratified_split(
    *,
    labels: np.ndarray,
    train_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    if train_split_percent == 100:
        split_summary = _build_single_manifest_split_summary(
            labels=labels,
            train_indices=all_indices,
            infer_indices=np.asarray([], dtype=np.int64),
            train_split_percent=train_split_percent,
            random_state=random_state,
            strategy="single_manifest_all_train",
        )
        split_summary["split_by_folder"] = False
        return (
            all_indices,
            np.asarray([], dtype=np.int64),
            [],
            split_summary,
        )

    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Stratified --train-split requires at least two classes in a single manifest")

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    infer_parts: list[np.ndarray] = []
    adjusted_labels: list[int] = []
    per_class_rows: list[dict[str, Any]] = []

    for label, count in zip(unique.tolist(), counts.tolist()):
        label_value = int(label)
        label_count = int(count)
        train_count, adjusted = _resolve_holdout_train_count(
            bucket_size=label_count,
            train_split_percent=train_split_percent,
            bucket_name=f"label {label_value}",
            split_name="Stratified --train-split",
        )
        infer_count = label_count - train_count
        if adjusted:
            adjusted_labels.append(label_value)

        label_indices = all_indices[labels == label_value]
        shuffled = label_indices[rng.permutation(label_indices.shape[0])]
        train_parts.append(shuffled[:train_count])
        infer_parts.append(shuffled[train_count:])
        per_class_rows.append(
            {
                "label": label_value,
                "total": label_count,
                "n_train": int(train_count),
                "n_inference": int(infer_count),
            }
        )

    train_indices = np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
    infer_indices = np.sort(np.concatenate(infer_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_labels:
        adjusted_preview = ", ".join(str(label) for label in adjusted_labels[:5])
        warnings.append(
            "Adjusted rounded train counts to keep at least one sample in both train and inference "
            f"for labels: {adjusted_preview}"
        )

    split_summary = _build_single_manifest_split_summary(
        labels=labels,
        train_indices=train_indices,
        infer_indices=infer_indices,
        train_split_percent=train_split_percent,
        random_state=random_state,
        strategy="single_manifest_stratified_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["per_class"] = per_class_rows
    return train_indices, infer_indices, warnings, split_summary


def _split_folder_name(item: Any) -> str:
    parent_name = item.model_dir.parent.name.strip()
    if parent_name:
        return parent_name
    return item.model_dir.name


def _build_single_manifest_folder_label_split(
    *,
    items: list[Any],
    labels: np.ndarray,
    train_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if labels.shape[0] != len(items):
        raise ValueError("Folder-based split requires one label per manifest item")

    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    if train_split_percent < 100 and np.unique(labels).size < 2:
        raise ValueError("Folder-based --train-split requires at least two classes in a single manifest")

    bucket_to_indices: dict[tuple[str, int], list[int]] = {}
    for idx, item in enumerate(items):
        folder_name = _split_folder_name(item)
        label_value = int(labels[idx])
        bucket_to_indices.setdefault((folder_name, label_value), []).append(int(idx))

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    infer_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for folder_name, label_value in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(folder_name, label_value)], dtype=np.int64)
        if train_split_percent == 100:
            train_count = int(bucket_indices.size)
            infer_count = 0
            train_parts.append(bucket_indices)
            infer_parts.append(np.asarray([], dtype=np.int64))
        else:
            train_count, adjusted = _resolve_holdout_train_count(
                bucket_size=int(bucket_indices.size),
                train_split_percent=train_split_percent,
                bucket_name=f"{folder_name} [label {label_value}]",
                split_name="Folder-based --train-split",
            )
            infer_count = int(bucket_indices.size) - train_count
            if adjusted:
                adjusted_buckets.append(f"{folder_name} [label {label_value}]")

            shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
            train_parts.append(shuffled[:train_count])
            infer_parts.append(shuffled[train_count:])

        per_bucket_rows.append(
            {
                "folder": folder_name,
                "label": int(label_value),
                "total": int(bucket_indices.size),
                "n_train": int(train_count),
                "n_inference": int(infer_count),
            }
        )

    train_indices = (
        np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
        if train_parts
        else np.asarray([], dtype=np.int64)
    )
    infer_indices = (
        np.sort(np.concatenate(infer_parts).astype(np.int64, copy=False))
        if infer_parts
        else np.asarray([], dtype=np.int64)
    )

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded train counts to keep at least one sample in both train and inference "
            f"for folder/label buckets: {adjusted_preview}"
        )

    strategy = (
        "single_manifest_folder_label_holdout"
        if train_split_percent < 100
        else "single_manifest_folder_label_all_train"
    )
    split_summary = _build_single_manifest_split_summary(
        labels=labels,
        train_indices=train_indices,
        infer_indices=infer_indices,
        train_split_percent=train_split_percent,
        random_state=random_state,
        strategy=strategy,
    )
    split_summary["split_by_folder"] = True
    split_summary["folder_count"] = int(len({row["folder"] for row in per_bucket_rows}))
    split_summary["folder_label_buckets"] = per_bucket_rows
    return train_indices, infer_indices, warnings, split_summary


def _build_calibration_stratified_split(
    *,
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    unique, counts = np.unique(candidate_labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Calibration split requires at least two classes in the training partition")

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_labels: list[int] = []
    per_class_rows: list[dict[str, Any]] = []

    for label, count in zip(unique.tolist(), counts.tolist()):
        label_value = int(label)
        label_indices = candidate_indices[candidate_labels == label_value]
        calibration_count, adjusted = _resolve_holdout_subset_count(
            bucket_size=int(count),
            subset_percent=calibration_split_percent,
            bucket_name=f"label {label_value}",
            split_name="Stratified --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(label_indices.size) - calibration_count
        if adjusted:
            adjusted_labels.append(label_value)

        shuffled = label_indices[rng.permutation(label_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_class_rows.append(
            {
                "label": label_value,
                "total": int(count),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_labels:
        adjusted_preview = ", ".join(str(label) for label in adjusted_labels[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for labels: {adjusted_preview}"
        )

    split_summary = _build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_stratified_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["per_class"] = per_class_rows
    return fit_train_indices, calibration_indices, warnings, split_summary


def _build_calibration_folder_label_split(
    *,
    items: list[Any],
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if labels.shape[0] != len(items):
        raise ValueError("Folder-based calibration split requires one label per manifest item")

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    if np.unique(candidate_labels).size < 2:
        raise ValueError("Folder-based calibration split requires at least two classes in the training partition")

    bucket_to_indices: dict[tuple[str, int], list[int]] = {}
    for idx in candidate_indices.tolist():
        folder_name = _split_folder_name(items[int(idx)])
        label_value = int(labels[int(idx)])
        bucket_to_indices.setdefault((folder_name, label_value), []).append(int(idx))

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for folder_name, label_value in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(folder_name, label_value)], dtype=np.int64)
        calibration_count, adjusted = _resolve_holdout_subset_count(
            bucket_size=int(bucket_indices.size),
            subset_percent=calibration_split_percent,
            bucket_name=f"{folder_name} [label {label_value}]",
            split_name="Folder-based --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(bucket_indices.size) - calibration_count
        if adjusted:
            adjusted_buckets.append(f"{folder_name} [label {label_value}]")

        shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_bucket_rows.append(
            {
                "folder": folder_name,
                "label": int(label_value),
                "total": int(bucket_indices.size),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for folder/label buckets: {adjusted_preview}"
        )

    split_summary = _build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_folder_label_holdout",
    )
    split_summary["split_by_folder"] = True
    split_summary["folder_count"] = int(len({row["folder"] for row in per_bucket_rows}))
    split_summary["folder_label_buckets"] = per_bucket_rows
    return fit_train_indices, calibration_indices, warnings, split_summary


def _threshold_candidate_values(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Threshold selection requires at least one score")
    unique_scores = np.unique(scores)
    above_max = np.nextafter(float(np.max(unique_scores)), float("inf"))
    return np.concatenate((np.asarray([above_max], dtype=np.float64), unique_scores[::-1]))


def _evaluate_binary_threshold(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels_true = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_true.shape[0] != scores.shape[0]:
        raise ValueError("Threshold evaluation requires labels and scores with the same length")

    positives = int(np.sum(labels_true == 1))
    negatives = int(np.sum(labels_true == 0))
    flagged = scores >= float(threshold)

    tp = int(np.sum((labels_true == 1) & flagged))
    fp = int(np.sum((labels_true == 0) & flagged))
    tn = int(np.sum((labels_true == 0) & (~flagged)))
    fn = int(np.sum((labels_true == 1) & (~flagged)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / positives) if positives > 0 else 0.0
    false_positive_rate = float(fp / negatives) if negatives > 0 else 0.0
    specificity = float(tn / negatives) if negatives > 0 else 0.0
    accuracy = float((tp + tn) / max(1, labels_true.shape[0]))
    balanced_accuracy = float((recall + specificity) / 2.0)
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

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
        "f1": f1,
        "fraction_flagged": float(np.mean(flagged)) if flagged.size > 0 else 0.0,
    }


def _select_threshold_max_recall_under_fpr(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    accepted_fpr: float,
) -> dict[str, Any]:
    labels_true = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_true.shape[0] != scores.shape[0]:
        raise ValueError("Threshold selection requires labels and scores with the same length")

    candidates = [
        _evaluate_binary_threshold(labels_true=labels_true, scores=scores, threshold=float(threshold))
        for threshold in _threshold_candidate_values(scores)
    ]
    feasible = [
        row for row in candidates
        if float(row["false_positive_rate"]) <= float(accepted_fpr) + 1e-12
    ]
    if not feasible:
        raise RuntimeError("No feasible threshold satisfied the accepted_fpr constraint")

    best = max(
        feasible,
        key=lambda row: (
            float(row["recall"]),
            -float(row["false_positive_rate"]),
            float(row["precision"]),
            float(row["threshold"]),
        ),
    )
    selected = dict(best)
    selected["selection_method"] = "maximize_recall_subject_to_fpr"
    selected["accepted_fpr"] = float(accepted_fpr)
    return selected


def _build_selected_threshold_summary(
    *,
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "method": "maximize_recall_subject_to_fpr",
        "source_partition": "calibration",
        "accepted_fprs": [float(selection["accepted_fpr"]) for selection in selections],
        "selections": selections,
    }
    if len(selections) == 1:
        summary.update(selections[0])
    return summary


def _build_selected_threshold_specs(
    selected_threshold_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(selected_threshold_summary, dict):
        return []

    selections = selected_threshold_summary.get("selections", [])
    if not isinstance(selections, list):
        return []

    specs: list[dict[str, Any]] = []
    selection_method = selected_threshold_summary.get("method")
    source_partition = selected_threshold_summary.get("source_partition")
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        if "threshold" not in selection or "accepted_fpr" not in selection:
            continue
        spec: dict[str, Any] = {
            "accepted_fpr": float(selection["accepted_fpr"]),
            "threshold": float(selection["threshold"]),
        }
        if selection_method is not None:
            spec["selection_method"] = str(selection_method)
        if source_partition is not None:
            spec["source_partition"] = str(source_partition)
        specs.append(spec)
    return specs


def _unique_index_by_name(names: list[str], *, context: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for i, name in enumerate(names):
        if name in index:
            duplicates.append(name)
            continue
        index[name] = int(i)
    if duplicates:
        dup_preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"Duplicate model names in {context}; cannot align features safely. "
            f"Examples: {dup_preview}"
        )
    return index


def _feature_block_name(feature_name: str) -> str:
    block_name, sep, _ = str(feature_name).rpartition(".")
    if not sep or not block_name:
        raise ValueError(f"Invalid spectral feature name in external metadata: {feature_name}")
    return block_name


def _ordered_block_names_from_feature_names(feature_names: list[str]) -> list[str]:
    block_names: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        block_name = _feature_block_name(feature_name)
        if block_name in seen:
            continue
        seen.add(block_name)
        block_names.append(block_name)
    return block_names


def _build_requested_feature_names(
    *,
    block_names: list[str],
    spectral_features: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
) -> list[str]:
    return build_spectral_feature_names(
        block_names=block_names,
        selected_features=spectral_features,
        sv_top_k=sv_top_k,
        spectral_moment_source=spectral_moment_source,
        shorten_block_names=False,
    )


def _filter_external_spectral_columns(
    *,
    features: np.ndarray,
    metadata: dict[str, Any],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    raw_feature_names = metadata.get("feature_names")
    if not isinstance(raw_feature_names, list) or not raw_feature_names:
        raise ValueError(
            "External spectral metadata must include non-empty 'feature_names' to honor "
            "--features/--spectral-sv-top-k/--spectral-qv-sum-mode"
        )

    feature_names = [str(x) for x in raw_feature_names]
    if len(feature_names) != int(features.shape[1]):
        raise ValueError(
            f"External feature metadata column count ({len(feature_names)}) does not match "
            f"feature matrix width ({features.shape[1]})"
        )

    selected_features = resolve_spectral_features(spectral_features)
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    resolved_qv_sum_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    all_block_names = _ordered_block_names_from_feature_names(feature_names)

    if resolved_qv_sum_mode == "none":
        selected_block_names = [name for name in all_block_names if ".qv_sum" not in name]
    elif resolved_qv_sum_mode == "only":
        selected_block_names = [name for name in all_block_names if ".qv_sum" in name]
    else:
        selected_block_names = list(all_block_names)

    if not selected_block_names:
        raise ValueError(
            "External feature matrix does not contain any blocks compatible with "
            f"--spectral-qv-sum-mode={resolved_qv_sum_mode}"
        )

    expected_feature_names = _build_requested_feature_names(
        block_names=selected_block_names,
        spectral_features=selected_features,
        sv_top_k=int(spectral_sv_top_k),
        spectral_moment_source=resolved_moment_source,
    )
    if not expected_feature_names:
        raise ValueError("Requested external spectral feature selection resolved to zero columns")

    feature_index = _unique_index_by_name(feature_names, context="external spectral feature names")
    missing = [name for name in expected_feature_names if name not in feature_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "External feature matrix does not contain the requested spectral columns. "
            f"Examples: {preview}"
        )

    warnings: list[str] = []
    if feature_names != expected_feature_names:
        column_indices = np.asarray([feature_index[name] for name in expected_feature_names], dtype=np.int64)
        features = features[:, column_indices]
        warnings.append(
            "Filtered/reordered external feature columns to match requested spectral configuration"
        )

    filtered_metadata = dict(metadata)
    filtered_metadata["resolved_features"] = list(selected_features)
    filtered_metadata["spectral_moment_source"] = resolved_moment_source
    filtered_metadata["spectral_qv_sum_mode"] = resolved_qv_sum_mode
    filtered_metadata["sv_top_k"] = int(spectral_sv_top_k)
    filtered_metadata["feature_dim"] = int(features.shape[1])
    filtered_metadata["feature_names"] = list(expected_feature_names)
    filtered_metadata["block_names"] = list(selected_block_names)
    filtered_metadata["n_blocks"] = int(len(selected_block_names))
    source_lora_dims = spectral_block_lora_dims_by_block(metadata)
    if all(block_name in source_lora_dims for block_name in selected_block_names):
        filtered_metadata["lora_adapter_dims"] = [
            source_lora_dims[block_name] for block_name in selected_block_names
        ]

    filtered_metadata["base_block_names"] = [
        name for name in selected_block_names if ".qv_sum" not in name
    ]
    filtered_metadata["qv_sum_block_names"] = [
        name for name in selected_block_names if ".qv_sum" in name
    ]

    extractor_params = filtered_metadata.get("extractor_params")
    if isinstance(extractor_params, dict):
        filtered_params = dict(extractor_params)
        filtered_params["spectral_features"] = list(selected_features)
        filtered_params["spectral_sv_top_k"] = int(spectral_sv_top_k)
        filtered_params["spectral_moment_source"] = resolved_moment_source
        filtered_params["spectral_qv_sum_mode"] = resolved_qv_sum_mode
        filtered_metadata["extractor_params"] = spectral_extractor_params(filtered_params)

    return features, sanitize_spectral_metadata(filtered_metadata), warnings


def _load_external_spectral_bundle(
    *,
    feature_file: Path,
    model_names_file: Path,
    metadata_file: Path | None,
    expected_model_names: list[str],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    allow_manifest_subset: bool = False,
) -> tuple[np.ndarray, dict[str, Any], list[str], np.ndarray]:
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    if not model_names_file.exists():
        raise FileNotFoundError(f"Model names file not found: {model_names_file}")
    if metadata_file is not None and not metadata_file.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {metadata_file}")

    features_mmap = np.load(feature_file, mmap_mode="r")
    if features_mmap.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix at {feature_file}, got shape={features_mmap.shape}")

    with open(model_names_file, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(features_mmap.shape[0]):
        raise ValueError(
            f"Model names length ({len(model_names)}) does not match feature rows ({features_mmap.shape[0]}) "
            f"for external features"
        )

    metadata: dict[str, Any] = {}
    if metadata_file is not None:
        loaded = load_spectral_metadata(metadata_file)
        if isinstance(loaded, dict):
            metadata = dict(loaded)

    ext_index = _unique_index_by_name(model_names, context=str(model_names_file))
    expected_index = _unique_index_by_name(expected_model_names, context="manifest model names")

    missing = sorted(name for name in expected_index if name not in ext_index)
    extra = sorted(name for name in ext_index if name not in expected_index)
    if missing and not allow_manifest_subset:
        details: list[str] = [f"missing={missing[:5]}"]
        if extra:
            details.append(f"extra={extra[:5]}")
        raise ValueError(
            "External feature/model-name set does not cover the manifest model-name set: "
            + "; ".join(details)
        )

    selected_expected_indices = np.asarray(
        [i for i, name in enumerate(expected_model_names) if name in ext_index],
        dtype=np.int64,
    )
    if selected_expected_indices.size == 0:
        details: list[str] = []
        if expected_model_names:
            details.append(f"expected={expected_model_names[:5]}")
        if model_names:
            details.append(f"external={model_names[:5]}")
        raise ValueError(
            "External feature/model-name set has no overlap with the manifest model-name set"
            + (": " + "; ".join(details) if details else "")
        )

    selected_expected_model_names = [expected_model_names[int(i)] for i in selected_expected_indices.tolist()]
    row_indices = np.asarray([ext_index[name] for name in selected_expected_model_names], dtype=np.int64)
    features = np.asarray(features_mmap[row_indices], dtype=np.float32)

    requested_rows = int(len(expected_model_names))
    expected_rows = int(len(selected_expected_model_names))
    source_rows = int(features_mmap.shape[0])

    warnings: list[str] = []
    if allow_manifest_subset and missing:
        warning = (
            "Selected external feature rows by manifest model names from a source bundle "
            f"with {source_rows} rows; retained {expected_rows}/{requested_rows} manifest models "
            f"and skipped {len(missing)} missing model names"
        )
        if extra:
            warning += f"; ignored {len(extra)} external-only model names"
        warning += f". Missing example(s): {missing[:5]}"
        warnings.append(warning)
    elif source_rows != expected_rows:
        warnings.append(
            "Selected external feature rows by manifest model names from a source bundle "
            f"with {source_rows} rows"
        )
    elif selected_expected_model_names != model_names:
        warnings.append("Reordered external features to match manifest model order using model names")

    metadata = dict(metadata)
    metadata["n_models"] = int(features.shape[0])
    metadata["external_manifest_requested_n_models"] = int(requested_rows)
    metadata["external_manifest_selected_n_models"] = int(features.shape[0])
    metadata["external_manifest_missing_model_count"] = int(len(missing))
    if source_rows != int(features.shape[0]):
        metadata["external_source_n_models"] = int(source_rows)

    features, metadata, column_warnings = _filter_external_spectral_columns(
        features=features,
        metadata=metadata,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
    )
    warnings.extend(column_warnings)

    return features, metadata, warnings, selected_expected_indices


def _load_features_for_tuning_manifest(manifest: dict[str, Any]) -> np.ndarray:
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("Tuning manifest is missing data configuration")

    feature_loading_mode = str(data.get("feature_loading_mode", "materialized"))
    feature_path_value = data.get("feature_path")
    if not isinstance(feature_path_value, str) or not feature_path_value:
        raise ValueError("Tuning manifest is missing data.feature_path")

    if feature_loading_mode == "materialized":
        return np.asarray(np.load(feature_path_value), dtype=np.float32)

    if feature_loading_mode != "external_source":
        raise ValueError(f"Unsupported data.feature_loading_mode={feature_loading_mode!r}")

    model_names_path_value = data.get("model_names_path")
    if not isinstance(model_names_path_value, str) or not model_names_path_value:
        raise ValueError("Tuning manifest is missing data.model_names_path")

    with open(model_names_path_value, "r", encoding="utf-8") as f:
        expected_model_names = [str(x) for x in json.load(f)]

    extractor = manifest.get("extractor")
    if not isinstance(extractor, dict):
        raise ValueError("Tuning manifest is missing extractor configuration")

    extractor_params = extractor.get("params")
    if not isinstance(extractor_params, dict):
        raise ValueError("Tuning manifest is missing extractor.params")

    extractor_metadata = extractor.get("metadata")
    if not isinstance(extractor_metadata, dict):
        raise ValueError("Tuning manifest is missing extractor.metadata")

    external_feature_source = extractor_metadata.get("external_feature_source")
    external_model_names_source = extractor_metadata.get("external_model_names_source")
    external_metadata_source = extractor_metadata.get("external_metadata_source")
    if not isinstance(external_feature_source, str) or not external_feature_source:
        raise ValueError("Tuning manifest is missing extractor.metadata.external_feature_source")
    if not isinstance(external_model_names_source, str) or not external_model_names_source:
        raise ValueError("Tuning manifest is missing extractor.metadata.external_model_names_source")

    spectral_features_value = extractor_params.get("spectral_features")
    if isinstance(spectral_features_value, list) and spectral_features_value:
        spectral_features = [str(x) for x in spectral_features_value]
    else:
        spectral_features = resolve_spectral_features(None)

    features, _, _, selected_expected_indices = _load_external_spectral_bundle(
        feature_file=Path(external_feature_source),
        model_names_file=Path(external_model_names_source),
        metadata_file=(
            Path(external_metadata_source)
            if isinstance(external_metadata_source, str) and external_metadata_source
            else None
        ),
        expected_model_names=expected_model_names,
        spectral_features=spectral_features,
        spectral_sv_top_k=int(extractor_params.get("spectral_sv_top_k", 8)),
        spectral_moment_source=str(
            extractor_params.get("spectral_moment_source", DEFAULT_SPECTRAL_MOMENT_SOURCE)
        ),
        spectral_qv_sum_mode=str(
            extractor_params.get("spectral_qv_sum_mode", DEFAULT_SPECTRAL_QV_SUM_MODE)
        ),
    )
    if int(selected_expected_indices.size) != int(len(expected_model_names)):
        raise ValueError(
            "External feature source no longer covers the prepared tuning-manifest model-name set; "
            "regenerate the supervised prepare stage to refresh the subset of available models"
        )
    return features


def _compact_indices_to_selected_scope(
    indices: np.ndarray,
    *,
    selected_expected_indices: np.ndarray,
) -> np.ndarray:
    remap = {int(old): int(new) for new, old in enumerate(selected_expected_indices.tolist())}
    compacted = [remap[int(idx)] for idx in indices.tolist() if int(idx) in remap]
    return np.asarray(compacted, dtype=np.int64)


def _sanitize_cv_folds(labels: np.ndarray, requested_folds: int) -> tuple[int, list[str]]:
    if requested_folds < 2:
        raise ValueError(f"cv_folds must be >=2, got {requested_folds}")

    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Binary classification requires at least two classes in the training set")

    min_count = int(np.min(counts))
    warnings: list[str] = []
    if min_count < 2:
        warnings.append(
            "Minority class count <2; falling back to a single train-as-validation split for tuning"
        )
        return 1, warnings

    resolved = min(requested_folds, min_count)
    if resolved != requested_folds:
        warnings.append(
            f"Reduced cv_folds from {requested_folds} to {resolved} due to minority class size={min_count}"
        )
    return int(resolved), warnings


def _build_cv_splits(
    *,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if cv_folds == 1:
        return [
            {
                "split_index": 0,
                "train_indices": [int(x) for x in train_indices.tolist()],
                "valid_indices": [int(x) for x in train_indices.tolist()],
            }
        ]

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits: list[dict[str, Any]] = []
    dummy = np.zeros((train_indices.shape[0], 1), dtype=np.float32)
    for split_idx, (tr_rel, val_rel) in enumerate(splitter.split(dummy, train_labels)):
        tr_abs = train_indices[np.asarray(tr_rel, dtype=np.int64)]
        val_abs = train_indices[np.asarray(val_rel, dtype=np.int64)]
        splits.append(
            {
                "split_index": int(split_idx),
                "train_indices": [int(x) for x in tr_abs.tolist()],
                "valid_indices": [int(x) for x in val_abs.tolist()],
            }
        )
    return splits


def _estimate_total_fit_count(
    *,
    n_tasks: int,
    cv_split_groups: list[dict[str, Any]],
) -> int:
    total_splits = sum(len(group.get("cv_splits", [])) for group in cv_split_groups)
    return int(n_tasks * total_splits)


def _grid_search_warnings(
    *,
    n_tasks: int,
    estimated_total_fits: int,
) -> list[str]:
    warnings: list[str] = []
    if n_tasks > 100:
        warnings.append(
            "Large supervised grid search: "
            f"{n_tasks} tasks and approximately {estimated_total_fits} total model fits across CV splits"
        )
    return warnings


def _predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=np.float64)
        if decision.ndim == 2 and decision.shape[1] >= 2:
            return decision[:, 1]
        return decision.reshape(-1)

    pred = np.asarray(model.predict(x), dtype=np.float64)
    return pred.reshape(-1)


def _evaluate_fold(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    model_name: str,
    params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    model = create(model_name, params=params, random_state=random_state)
    model.fit(features[train_indices], labels[train_indices])
    scores = _predict_scores(model, features[valid_indices])
    auc = float(roc_auc_score(labels[valid_indices], scores))
    return {
        "n_train": int(train_indices.size),
        "n_valid": int(valid_indices.size),
        "roc_auc": auc,
    }


def _evaluate_candidate(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    task: dict[str, Any],
    cv_split_groups: list[dict[str, Any]],
    n_jobs: int,
) -> dict[str, Any]:
    eval_jobs: list[tuple[int, dict[str, Any]]] = []
    for group in cv_split_groups:
        seed = int(group["random_state"])
        for split in group["cv_splits"]:
            eval_jobs.append((seed, split))

    if n_jobs == 1 or len(eval_jobs) <= 1 or joblib is None:
        evaluated: list[tuple[int, dict[str, Any]]] = []
        for seed, split in eval_jobs:
            tr = np.asarray(split["train_indices"], dtype=np.int64)
            val = np.asarray(split["valid_indices"], dtype=np.int64)
            row = _evaluate_fold(
                features=features,
                labels=labels,
                train_indices=tr,
                valid_indices=val,
                model_name=str(task["model_name"]),
                params=dict(task["params"]),
                random_state=seed,
            )
            row["cv_random_state"] = int(seed)
            evaluated.append((seed, row))
    else:
        raw_rows = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_evaluate_fold)(
                features=features,
                labels=labels,
                train_indices=np.asarray(split["train_indices"], dtype=np.int64),
                valid_indices=np.asarray(split["valid_indices"], dtype=np.int64),
                model_name=str(task["model_name"]),
                params=dict(task["params"]),
                random_state=seed,
            )
            for seed, split in eval_jobs
        )
        evaluated = []
        for (seed, _), row in zip(eval_jobs, raw_rows):
            row_with_seed = dict(row)
            row_with_seed["cv_random_state"] = int(seed)
            evaluated.append((seed, row_with_seed))

    fold_rows = [row for _, row in evaluated]
    scores = [float(row["roc_auc"]) for row in fold_rows]

    seed_results: list[dict[str, Any]] = []
    unique_seeds = sorted({int(seed) for seed, _ in evaluated})
    for seed in unique_seeds:
        seed_fold_rows = [row for seed_value, row in evaluated if int(seed_value) == int(seed)]
        seed_scores = [float(row["roc_auc"]) for row in seed_fold_rows]
        seed_results.append(
            {
                "random_state": int(seed),
                "fold_results": seed_fold_rows,
                "roc_auc_mean": float(np.mean(seed_scores)) if seed_scores else None,
                "roc_auc_std": float(np.std(seed_scores)) if seed_scores else None,
            }
        )

    return {
        "task_index": int(task["task_index"]),
        "model_name": str(task["model_name"]),
        "params": dict(task["params"]),
        "complexity_rank": int(task["complexity_rank"]),
        "normalization_policy": str(task["normalization_policy"]),
        "status": "ok",
        "fold_results": fold_rows,
        "seed_results": seed_results,
        "roc_auc_mean": float(np.mean(scores)) if scores else None,
        "roc_auc_std": float(np.std(scores)) if scores else None,
    }


def _task_result_path(task_dir: Path, task_index: int) -> Path:
    return task_dir / f"task_{task_index:04d}.json"


def _compact_extractor_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    drop_keys = {
        "feature_names",
        "incoming_metadata",
        "merge_stats",
        "merged_with_existing_output",
        "merge_existing_output_dir",
        "block_names_raw",
        "base_block_names_raw",
        "qv_sum_block_names_raw",
    }
    extractor_name = str(metadata.get("extractor") or metadata.get("extractor_name") or "")
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in drop_keys or key == "shapes" or key.endswith("_shapes"):
            continue
        if key == "extractor_params" and isinstance(value, dict) and extractor_name == "spectral":
            compact[str(key)] = spectral_extractor_params(value)
            continue
        if isinstance(value, dict):
            nested = _compact_extractor_metadata(value)
            if nested:
                compact[str(key)] = nested
            continue
        compact[str(key)] = value
    return compact


def _score_percentile_ranks(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.float64)
    ranks = np.argsort(np.argsort(scores))
    return np.asarray(ranks / max(1, scores.size - 1), dtype=np.float64)


def _resolve_manifest_path_for_reporting(manifest: dict[str, Any]) -> Path:
    raw_value = str(manifest["manifest_json"])
    resolved = resolve_manifest_path(raw_value)
    if resolved.exists():
        return resolved

    raw_path = Path(raw_value).expanduser()

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        run_dir = Path(str(manifest["run_dir"])).expanduser().resolve()
        if len(run_dir.parents) >= 3:
            candidates.append((run_dir.parents[2] / raw_path).resolve())
        candidates.append(raw_path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_manifest_items_for_tuning_manifest(manifest: dict[str, Any]) -> list[Any]:
    manifest_path = _resolve_manifest_path_for_reporting(manifest)
    feature_loading_mode = str(manifest["data"].get("feature_loading_mode", "materialized"))
    mode = str(manifest["mode"])

    if feature_loading_mode == "external_source":
        if mode == "joint":
            train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_path)
            return train_items + infer_items
        return parse_single_manifest_json_by_model_name(
            manifest_path=manifest_path,
            section_key="path",
        )

    dataset_root = Path(str(manifest["dataset_root"])).expanduser().resolve()
    if mode == "joint":
        train_items, infer_items = parse_joint_manifest_json(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
        )
        return train_items + infer_items
    return parse_single_manifest_json(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        section_key="path",
    )


def _build_threshold_specs(train_scores: np.ndarray, percentiles: list[float]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for pct in percentiles:
        specs.append(
            {
                "percentile_from_train": float(pct),
                "threshold": float(np.percentile(train_scores, float(pct))),
            }
        )
    return specs


def _evaluate_attack_threshold_rows(
    *,
    group_scores: np.ndarray,
    known_mask: np.ndarray,
    known_labels: np.ndarray | None,
    threshold_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threshold_rows: list[dict[str, Any]] = []
    for spec in threshold_specs:
        threshold = float(spec["threshold"])
        flagged = group_scores >= threshold
        n_flagged = int(np.sum(flagged))
        row: dict[str, Any] = {key: value for key, value in spec.items() if key != "threshold"}
        row.update(
            {
                "threshold": threshold,
                "n_flagged": n_flagged,
                "fraction_flagged": float(n_flagged / max(1, group_scores.size)),
            }
        )

        if known_labels is not None:
            known_flagged = flagged[known_mask]
            positives = int(np.sum(known_labels == 1))
            negatives = int(np.sum(known_labels == 0))
            tp = int(np.sum((known_labels == 1) & known_flagged))
            fp = int(np.sum((known_labels == 0) & known_flagged))
            if n_flagged > 0:
                row["precision"] = float(tp / n_flagged)
            if positives > 0:
                row["recall"] = float(tp / positives)
            if negatives > 0:
                row["false_positive_rate"] = float(fp / negatives)

        threshold_rows.append(row)

    return threshold_rows


def _summarize_attack_slice(
    *,
    group_scores: np.ndarray,
    group_ranks: np.ndarray,
    group_labels: list[int | None],
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None,
    source_subsets: list[str],
) -> dict[str, Any]:
    clean_count = sum(1 for label in group_labels if label == 0)
    backdoored_count = sum(1 for label in group_labels if label == 1)
    unknown_count = sum(1 for label in group_labels if label is None)

    known_mask = np.asarray([lbl is not None for lbl in group_labels], dtype=bool)
    known_labels: np.ndarray | None = None
    if bool(np.any(known_mask)):
        known_labels = np.asarray(
            [int(group_labels[i]) for i in range(len(group_labels)) if known_mask[i]],
            dtype=np.int32,
        )

    summary = {
        "n_samples": int(group_scores.size),
        "source_subsets": list(source_subsets),
        "label_counts": {
            "clean": int(clean_count),
            "backdoored": int(backdoored_count),
            "unknown": int(unknown_count),
        },
        "score_summary": summarize_scores(group_scores),
        "score_percentile_rank_summary": summarize_scores(group_ranks),
        "threshold_evaluation": _evaluate_attack_threshold_rows(
            group_scores=group_scores,
            known_mask=known_mask,
            known_labels=known_labels,
            threshold_specs=threshold_specs,
        ),
    }
    if selected_threshold_specs:
        summary["selected_threshold_evaluation"] = _evaluate_attack_threshold_rows(
            group_scores=group_scores,
            known_mask=known_mask,
            known_labels=known_labels,
            threshold_specs=selected_threshold_specs,
        )
    return summary


def _summarize_attack_groups(
    *,
    sample_identities: list[AttackSampleIdentity],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(sample_identities) != len(labels) or len(sample_identities) != int(scores.size):
        raise ValueError("Attack analysis inputs must have the same length")

    score_ranks = _score_percentile_ranks(np.asarray(scores, dtype=np.float64))
    all_indices_by_attack: dict[str, list[int]] = {}
    positive_by_attack: dict[str, list[int]] = {}
    unknown_by_attack: dict[str, list[int]] = {}
    attack_subset_names: dict[str, set[str]] = {}
    clean_indices: list[int] = []
    clean_subset_names: set[str] = set()

    for idx, (identity, label) in enumerate(zip(sample_identities, labels)):
        all_indices_by_attack.setdefault(identity.attack_name, []).append(int(idx))
        attack_subset_names.setdefault(identity.attack_name, set()).add(identity.subset_name)
        if label == 1:
            positive_by_attack.setdefault(identity.attack_name, []).append(int(idx))
        elif label == 0:
            clean_indices.append(int(idx))
            clean_subset_names.add(identity.subset_name)
        else:
            unknown_by_attack.setdefault(identity.attack_name, []).append(int(idx))

    grouped_indices: dict[str, list[int]] = {}
    if positive_by_attack:
        for attack in sorted(positive_by_attack):
            combined = (
                list(positive_by_attack[attack])
                + list(clean_indices)
                + list(unknown_by_attack.get(attack, []))
            )
            grouped_indices[attack] = sorted(set(int(i) for i in combined))
    else:
        grouped_indices = {
            attack: sorted(int(i) for i in idx_list)
            for attack, idx_list in all_indices_by_attack.items()
        }

    attacks: dict[str, Any] = {}
    for attack in sorted(grouped_indices):
        idx = np.asarray(grouped_indices[attack], dtype=np.int64)
        attacks[attack] = _summarize_attack_slice(
            group_scores=np.asarray(scores[idx], dtype=np.float64),
            group_ranks=np.asarray(score_ranks[idx], dtype=np.float64),
            group_labels=[labels[int(i)] for i in idx.tolist()],
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
            source_subsets=sorted(attack_subset_names.get(attack, set())),
        )

    return {
        "grouping_rule": (
            "one-vs-clean per attack using a single shared clean pool built from every label0 sample; "
            "known PADBench attacks are canonicalized by name (RIPPLE, syntactic, insertsent, stybkd); "
            "all other folders contribute label1 samples under the folder-derived attack name after "
            "removing model/config tokens"
        ),
        "clean_pool": {
            "n_samples": int(len(clean_indices)),
            "source_subsets": sorted(str(x) for x in clean_subset_names),
        },
        "n_attacks": int(len(attacks)),
        "attacks": attacks,
    }


def _context_from_run_dir(run_dir: Path) -> RunContext:
    output_root = run_dir.parents[1] if len(run_dir.parents) >= 2 else run_dir.parent
    features_dir = run_dir / "features"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"

    for path in [features_dir, models_dir, reports_dir, plots_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_dir.name,
        run_dir=run_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
    )


def _prepare_supervised_run(
    *,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str,
    spectral_features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    random_state: int,
    train_split_percent: int,
    calibration_split_percent: int | None,
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
    split_by_folder: bool,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float],
    feature_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    slurm_max_concurrent: str,
    slurm_cpus_per_task: str,
) -> dict[str, Any]:
    manifest_json = resolve_manifest_path(manifest_json)
    if not manifest_json.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_json}")
    if feature_file is None:
        raise ValueError(
            "Supervised pipeline requires --feature-file. Run feature extraction first, then pass the "
            "extracted feature bundle into supervised."
        )
    if not spectral_features:
        raise ValueError(
            "Supervised pipeline requires --features. Specify the feature groups to select from the "
            "extracted feature bundle."
        )

    mode = _detect_manifest_mode(manifest_json)
    if mode == "joint":
        train_items, infer_items = parse_joint_manifest_json_by_model_name(
            manifest_path=manifest_json,
        )
        all_items = train_items + infer_items
        train_indices = np.arange(0, len(train_items), dtype=np.int64)
        infer_indices = np.arange(len(train_items), len(all_items), dtype=np.int64)
    else:
        all_items = parse_single_manifest_json_by_model_name(
            manifest_path=manifest_json,
            section_key="path",
        )
        train_items = all_items
        infer_items = []
        train_indices = np.arange(0, len(all_items), dtype=np.int64)
        infer_indices = np.asarray([], dtype=np.int64)

    if not train_items:
        raise ValueError("No training items resolved from manifest")

    train_split_percent = _resolve_train_split_percent(train_split_percent)
    calibration_split_percent = _resolve_calibration_split_percent(calibration_split_percent)
    accepted_fprs = _resolve_accepted_fprs(accepted_fpr)
    if (calibration_split_percent is None) != (accepted_fprs is None):
        raise ValueError("--calibration-split and --accepted-fpr must either both be set or both be omitted")

    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_id,
    )

    selected_spectral_features = list(spectral_features)
    feature_params = {
        "dtype": dtype_name,
        "block_size": int(stream_block_size),
        "spectral_features": list(selected_spectral_features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": str(spectral_moment_source),
        "spectral_qv_sum_mode": str(spectral_qv_sum_mode),
        "spectral_entrywise_delta_mode": str(spectral_entrywise_delta_mode),
    }
    extractor_warnings: list[str] = []
    feature_loading_mode = "external_source"
    resolved_feature_file, resolved_model_names_file, resolved_metadata_file = (
        _resolve_supervised_feature_bundle_paths(feature_file)
    )
    features, external_metadata, external_warnings, selected_expected_indices = _load_external_spectral_bundle(
        feature_file=resolved_feature_file,
        model_names_file=resolved_model_names_file,
        metadata_file=resolved_metadata_file,
        expected_model_names=[item.model_name for item in all_items],
        spectral_features=selected_spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        allow_manifest_subset=True,
    )
    extractor_warnings.extend(external_warnings)

    if int(selected_expected_indices.size) != int(len(all_items)):
        all_items = [all_items[int(i)] for i in selected_expected_indices.tolist()]
        train_indices = _compact_indices_to_selected_scope(
            train_indices,
            selected_expected_indices=selected_expected_indices,
        )
        infer_indices = _compact_indices_to_selected_scope(
            infer_indices,
            selected_expected_indices=selected_expected_indices,
        )
        train_items = [all_items[int(i)] for i in train_indices.tolist()]
        infer_items = [all_items[int(i)] for i in infer_indices.tolist()]
        if not train_items:
            raise ValueError(
                "No training items remain after intersecting the manifest with the available "
                "external feature/model-name set"
            )

    run_metadata_path = ctx.features_dir / "spectral_metadata.json"
    extractor_metadata = {
        **external_metadata,
        "external_feature_source": str(resolved_feature_file),
        "external_model_names_source": str(resolved_model_names_file),
        "external_metadata_source": (
            str(resolved_metadata_file) if resolved_metadata_file is not None else None
        ),
        "loaded_external_features": True,
    }
    external_dataset_reference_payload = (
        resolve_dataset_reference_for_metadata(resolved_metadata_file)
        if resolved_metadata_file is not None
        else None
    )
    write_spectral_metadata(
        run_metadata_path,
        internal_metadata=extractor_metadata,
        dataset_layouts=dataset_layouts_from_source(
            metadata=extractor_metadata,
            dataset_reference_payload=external_dataset_reference_payload,
        ),
    )

    artifacts = {
        "feature_path": str(resolved_feature_file),
        "labels_path": None,
        "model_names_path": None,
        "metadata_path": str(run_metadata_path),
    }

    labels_values, labels_known, labels_raw = _labels_from_items(all_items)
    split_warnings: list[str] = []
    if mode == "single":
        if train_split_percent < 100 and not bool(np.all(labels_known)):
            raise ValueError(
                "Single-manifest --train-split requires labels for every sample in the manifest"
            )
        if split_by_folder:
            train_indices, infer_indices, split_warnings, split_summary = _build_single_manifest_folder_label_split(
                items=all_items,
                labels=labels_values,
                train_split_percent=train_split_percent,
                random_state=random_state,
            )
        else:
            train_indices, infer_indices, split_warnings, split_summary = _build_single_manifest_stratified_split(
                labels=labels_values,
                train_split_percent=train_split_percent,
                random_state=random_state,
            )
    else:
        if train_split_percent != 100:
            raise ValueError(
                "--train-split values below 100 are only supported for single manifests; "
                "joint manifests already define train and inference partitions"
            )
        if split_by_folder and calibration_split_percent is None:
            raise ValueError(
                "--split-by-folder requires --calibration-split for joint manifests; "
                "joint manifests already define the outer train and inference partitions"
            )
        split_summary = {
            "strategy": "manifest_defined",
            "split_by_folder": False,
            "requested_train_split_percent": int(train_split_percent),
            "random_state": int(random_state),
            "n_train": int(train_indices.size),
            "n_inference": int(infer_indices.size),
            "train_label_counts": _label_count_rows(labels_values[train_indices]),
            "inference_label_counts": _label_count_rows(labels_values[infer_indices][labels_known[infer_indices]]),
            "n_inference_unknown_label": int(np.sum(~labels_known[infer_indices])),
        }

    train_known = labels_known[train_indices]
    if not bool(np.all(train_known)):
        raise ValueError("Training samples must all have labels (label0/label1) for supervised learning")

    train_pool_indices = np.asarray(train_indices, dtype=np.int64)
    fit_train_indices = np.asarray(train_pool_indices, dtype=np.int64)
    calibration_indices = np.asarray([], dtype=np.int64)
    calibration_summary: dict[str, Any] | None = None
    calibration_warnings: list[str] = []
    if calibration_split_percent is not None:
        if split_by_folder:
            fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = _build_calibration_folder_label_split(
                items=all_items,
                candidate_indices=train_pool_indices,
                labels=labels_values,
                calibration_split_percent=calibration_split_percent,
                random_state=random_state,
            )
        else:
            fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = _build_calibration_stratified_split(
                candidate_indices=train_pool_indices,
                labels=labels_values,
                calibration_split_percent=calibration_split_percent,
                random_state=random_state,
            )

    fit_train_labels = labels_values[fit_train_indices]
    cv_folds_resolved, cv_warnings = _sanitize_cv_folds(fit_train_labels, cv_folds)
    requested_states = list(cv_random_states) if cv_random_states else [int(random_state)]
    dedup_states: list[int] = []
    seen_states: set[int] = set()
    for state in requested_states:
        state_int = int(state)
        if state_int in seen_states:
            continue
        seen_states.add(state_int)
        dedup_states.append(state_int)
    if not dedup_states:
        dedup_states = [int(random_state)]

    cv_split_groups: list[dict[str, Any]] = []
    for state in dedup_states:
        cv_split_groups.append(
            {
                "random_state": int(state),
                "cv_splits": _build_cv_splits(
                    train_indices=fit_train_indices,
                    train_labels=fit_train_labels,
                    cv_folds=cv_folds_resolved,
                    random_state=int(state),
                ),
            }
        )
    cv_splits = list(cv_split_groups[0]["cv_splits"])

    tasks: list[dict[str, Any]] = []
    model_names = registered_models() if model_name == "all" else [model_name]
    task_index = 0
    for selected_model in model_names:
        complexity = model_complexity_rank(selected_model)
        for params in candidate_params(selected_model):
            tasks.append(
                {
                    "task_index": int(task_index),
                    "model_name": selected_model,
                    "params": dict(params),
                    "complexity_rank": int(complexity),
                    "normalization_policy": normalization_policy(selected_model),
                }
            )
            task_index += 1

    estimated_total_fits = _estimate_total_fit_count(
        n_tasks=len(tasks),
        cv_split_groups=cv_split_groups,
    )
    grid_warnings = _grid_search_warnings(
        n_tasks=len(tasks),
        estimated_total_fits=estimated_total_fits,
    )

    labels_value_path = ctx.features_dir / "supervised_label_values.npy"
    labels_known_path = ctx.features_dir / "supervised_label_known.npy"
    np.save(labels_value_path, labels_values)
    np.save(labels_known_path, labels_known.astype(np.int8))

    model_names_path = ctx.features_dir / "supervised_model_names.json"
    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump([item.model_name for item in all_items], f, indent=2)

    slurm_cpus = resolve_slurm_cpus_per_task(slurm_cpus_per_task)
    slurm_concurrency = resolve_slurm_max_concurrent(slurm_max_concurrent, slurm_cpus)

    task_dir = ctx.reports_dir / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    tuning_manifest = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(ctx.run_dir),
        "mode": mode,
        "manifest_json": str(manifest_json.expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "data": {
            "n_samples": int(features.shape[0]),
            "train_indices": [int(x) for x in fit_train_indices.tolist()],
            "train_pool_indices": [int(x) for x in train_pool_indices.tolist()],
            "calibration_indices": [int(x) for x in calibration_indices.tolist()],
            "infer_indices": [int(x) for x in infer_indices.tolist()],
            "split": split_summary,
            "calibration_split": calibration_summary,
            "feature_path": str(Path(artifacts["feature_path"])),
            "feature_loading_mode": feature_loading_mode,
            "labels_value_path": str(labels_value_path),
            "labels_known_path": str(labels_known_path),
            "model_names_path": str(model_names_path),
        },
        "extractor": {
            "name": "spectral",
            "params": feature_params,
            "metadata": extractor_metadata,
            "warnings": extractor_warnings,
            "metadata_path": str(Path(artifacts["metadata_path"])),
        },
        "tuning": {
            "executor": tuning_executor,
            "model_name": model_name,
            "model_names": model_names,
            "metric": "roc_auc",
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
            "train_split_percent": int(train_split_percent),
            "split_by_folder": bool(split_by_folder),
            "calibration_split_percent": (
                int(calibration_split_percent) if calibration_split_percent is not None else None
            ),
            "cv_random_states": [int(x) for x in dedup_states],
            "cv_folds_requested": int(cv_folds),
            "cv_folds_resolved": int(cv_folds_resolved),
            "estimated_total_fits": int(estimated_total_fits),
            "cv_splits": cv_splits,
            "cv_split_groups": cv_split_groups,
            "tasks": tasks,
        },
        "threshold_selection": {
            "method": (
                "maximize_recall_subject_to_fpr"
                if calibration_split_percent is not None
                else None
            ),
            "calibration_split_percent": (
                int(calibration_split_percent) if calibration_split_percent is not None else None
            ),
            "accepted_fprs": list(accepted_fprs) if accepted_fprs is not None else None,
            "accepted_fpr": (
                float(accepted_fprs[0])
                if accepted_fprs is not None and len(accepted_fprs) == 1
                else None
            ),
            "split_by_folder": bool(split_by_folder and calibration_split_percent is not None),
        },
        "runtime": {
            "slurm_partition": slurm_partition,
            "slurm_max_concurrent": int(slurm_concurrency),
            "slurm_cpus_per_task": int(slurm_cpus),
            "score_percentiles": [float(x) for x in score_percentiles],
        },
        "warnings": (
            cv_warnings
            + split_warnings
            + calibration_warnings
            + list(extractor_warnings)
            + grid_warnings
        ),
        "labels_preview": labels_raw,
    }

    tuning_manifest_path = ctx.reports_dir / "tuning_manifest.json"
    with open(tuning_manifest_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(tuning_manifest), f, indent=2)

    return {
        "run_dir": str(ctx.run_dir),
        "tuning_manifest": str(tuning_manifest_path),
        "n_tasks": len(tasks),
        "task_dir": str(task_dir),
        "slurm_partition": slurm_partition,
        "slurm_max_concurrent": int(slurm_concurrency),
        "slurm_cpus_per_task": int(slurm_cpus),
        "warnings": tuning_manifest["warnings"],
    }


def _run_supervised_worker(
    *,
    run_dir: Path,
    task_index: int,
    n_jobs: int | None,
) -> dict[str, Any]:
    tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = manifest["tuning"]["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task_index={task_index} out of range [0, {len(tasks) - 1}]")

    task = tasks[task_index]
    features = _load_features_for_tuning_manifest(manifest)
    labels = np.load(manifest["data"]["labels_value_path"]).astype(np.int32)
    tuning_cfg = manifest["tuning"]
    if "cv_split_groups" in tuning_cfg:
        cv_split_groups = list(tuning_cfg["cv_split_groups"])
    else:
        cv_split_groups = [
            {
                "random_state": int(tuning_cfg["random_state"]),
                "cv_splits": list(tuning_cfg["cv_splits"]),
            }
        ]
    resolved_n_jobs = int(n_jobs) if n_jobs is not None else int(manifest["tuning"]["n_jobs"])

    start = perf_counter()
    try:
        result = _evaluate_candidate(
            features=features,
            labels=labels,
            task=task,
            cv_split_groups=cv_split_groups,
            n_jobs=resolved_n_jobs,
        )
    except Exception as exc:  # pragma: no cover - failure path asserted via output file shape
        result = {
            "task_index": int(task["task_index"]),
            "model_name": str(task["model_name"]),
            "params": dict(task["params"]),
            "complexity_rank": int(task["complexity_rank"]),
            "normalization_policy": str(task["normalization_policy"]),
            "status": "error",
            "error": str(exc),
            "fold_results": [],
            "roc_auc_mean": None,
            "roc_auc_std": None,
        }

    result["elapsed_seconds"] = float(perf_counter() - start)

    task_dir = run_dir / "reports" / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = _task_result_path(task_dir, task_index)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2)

    return {
        "run_dir": str(run_dir),
        "task_index": int(task_index),
        "result_path": str(out_path),
        "status": result.get("status"),
    }


def _select_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in candidates if row.get("status") == "ok" and row.get("roc_auc_mean") is not None]
    if not valid:
        raise RuntimeError("No successful tuning candidates available to select a winner")

    ranked = sorted(
        valid,
        key=lambda row: (
            -float(row["roc_auc_mean"]),
            float(row["roc_auc_std"]) if row.get("roc_auc_std") is not None else float("inf"),
            int(row.get("complexity_rank", 10**9)),
            int(row["task_index"]),
        ),
    )
    return ranked[0]


def _save_model(model: Any, path: Path) -> None:
    if joblib is not None:
        joblib.dump(model, path)
        return
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _finalize_state_path(run_dir: Path) -> Path:
    return run_dir / "reports" / FINALIZE_STATE_FILENAME


def _write_finalize_state(
    *,
    run_dir: Path,
    run_config: dict[str, Any],
    artifacts: dict[str, str | None],
    winner_export_train_features: Path | None,
    winner_export_train_labels: Path | None,
    random_state: int,
    skip_feature_importance: bool,
) -> Path:
    state_path = _finalize_state_path(run_dir)
    payload = {
        "run_config": run_config,
        "artifacts": artifacts,
        "winner_export": {
            "train_feature_matrix_path": (
                str(winner_export_train_features) if winner_export_train_features is not None else None
            ),
            "train_labels_path": (
                str(winner_export_train_labels) if winner_export_train_labels is not None else None
            ),
            "random_state": int(random_state),
            "skip_feature_importance": bool(skip_feature_importance),
        },
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return state_path


def _load_finalize_state(run_dir: Path) -> dict[str, Any]:
    path = _finalize_state_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Finalize state not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in finalize state, got {type(payload).__name__}")
    return payload


def _cleanup_finalize_intermediates(run_dir: Path, state: dict[str, Any]) -> None:
    winner_export = state.get("winner_export", {})
    if isinstance(winner_export, dict):
        for key in ("train_feature_matrix_path", "train_labels_path"):
            value = winner_export.get(key)
            if isinstance(value, str) and value:
                Path(value).expanduser().resolve().unlink(missing_ok=True)
    _finalize_state_path(run_dir).unlink(missing_ok=True)
    cleanup_winner_feature_weights_export(run_dir=run_dir)


def _prepare_supervised_finalize(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
    skip_feature_importance: bool,
) -> dict[str, Any]:
    ctx = _context_from_run_dir(run_dir)
    tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if score_percentiles is None:
        score_percentiles = [float(x) for x in manifest["runtime"]["score_percentiles"]]

    task_dir = run_dir / "reports" / "tuning_tasks"
    task_results: list[dict[str, Any]] = []
    missing: list[int] = []
    for task in manifest["tuning"]["tasks"]:
        task_index = int(task["task_index"])
        path = _task_result_path(task_dir, task_index)
        if not path.exists():
            missing.append(task_index)
            continue
        with open(path, "r", encoding="utf-8") as f:
            task_results.append(json.load(f))

    if missing:
        raise RuntimeError(f"Missing tuning task outputs for indices: {missing[:10]}")

    winner = _select_winner(task_results)
    features = _load_features_for_tuning_manifest(manifest)
    labels_value = np.load(manifest["data"]["labels_value_path"]).astype(np.int32)
    labels_known = np.load(manifest["data"]["labels_known_path"]).astype(bool)
    with open(manifest["data"]["model_names_path"], "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    manifest_items = _load_manifest_items_for_tuning_manifest(manifest)
    manifest_identities = infer_attack_sample_identities(manifest_items)
    identity_by_name = {identity.model_name: identity for identity in manifest_identities}
    missing_identities = [name for name in model_names if name not in identity_by_name]
    if missing_identities:
        preview = ", ".join(missing_identities[:5])
        raise ValueError(
            "Could not align manifest-derived attack identities with stored model names. "
            f"Examples: {preview}"
        )
    all_sample_identities = [identity_by_name[name] for name in model_names]

    train_indices = np.asarray(manifest["data"]["train_indices"], dtype=np.int64)
    train_pool_indices = np.asarray(
        manifest["data"].get("train_pool_indices", manifest["data"]["train_indices"]),
        dtype=np.int64,
    )
    calibration_indices = np.asarray(manifest["data"].get("calibration_indices", []), dtype=np.int64)
    infer_indices = np.asarray(manifest["data"]["infer_indices"], dtype=np.int64)
    x_train = features[train_indices]
    y_train = labels_value[train_indices]

    model = create(
        str(winner["model_name"]),
        params=dict(winner["params"]),
        random_state=int(manifest["tuning"]["random_state"]),
    )
    model.fit(x_train, y_train)
    train_scores = _predict_scores(model, x_train)
    threshold_specs = _build_threshold_specs(
        train_scores=np.asarray(train_scores, dtype=np.float64),
        percentiles=[float(x) for x in score_percentiles],
    )

    model_path = ctx.models_dir / "best_model.joblib"
    _save_model(model, model_path)

    train_model_names = [model_names[int(i)] for i in train_indices.tolist()]
    train_sample_identities = [all_sample_identities[int(i)] for i in train_indices.tolist()]
    train_labels_raw = [int(x) for x in y_train.tolist()]
    train_scores_csv = ctx.reports_dir / "train_scores.csv"
    save_score_csv(
        output_path=train_scores_csv,
        model_names=train_model_names,
        labels=train_labels_raw,
        scores=train_scores,
    )

    calibration_scores_csv: Path | None = None
    calibration_score_summary: dict[str, Any] | None = None
    calibration_offline_metrics: dict[str, Any] | None = None
    calibration_model_names: list[str] = []
    calibration_sample_identities: list[AttackSampleIdentity] = []
    calibration_labels_raw: list[int] = []
    calibration_scores: np.ndarray | None = None
    if calibration_indices.size > 0:
        x_calibration = features[calibration_indices]
        y_calibration = labels_value[calibration_indices]
        calibration_scores = _predict_scores(model, x_calibration)
        calibration_model_names = [model_names[int(i)] for i in calibration_indices.tolist()]
        calibration_sample_identities = [
            all_sample_identities[int(i)] for i in calibration_indices.tolist()
        ]
        calibration_labels_raw = [int(x) for x in y_calibration.tolist()]
        calibration_scores_csv = ctx.reports_dir / "calibration_scores.csv"
        save_score_csv(
            output_path=calibration_scores_csv,
            model_names=calibration_model_names,
            labels=calibration_labels_raw,
            scores=calibration_scores,
        )
        calibration_score_summary = summarize_scores(np.asarray(calibration_scores, dtype=np.float64))
        calibration_offline_metrics = compute_offline_metrics(
            np.asarray(y_calibration, dtype=np.int32),
            np.asarray(calibration_scores, dtype=np.float64),
        )

    infer_scores_csv: Path | None = None
    inference_summary: dict[str, Any] | None = None
    threshold_rows: list[dict[str, Any]] = []
    threshold_rows_from_inference: list[dict[str, Any]] = []
    infer_offline_metrics: dict[str, Any] | None = None
    infer_model_names: list[str] = []
    infer_sample_identities: list[AttackSampleIdentity] = []
    infer_labels_raw: list[int | None] = []
    infer_scores: np.ndarray | None = None
    infer_labels_np: np.ndarray | None = None

    if infer_indices.size > 0:
        x_infer = features[infer_indices]
        infer_scores = _predict_scores(model, x_infer)
        infer_model_names = [model_names[int(i)] for i in infer_indices.tolist()]
        infer_sample_identities = [all_sample_identities[int(i)] for i in infer_indices.tolist()]

        infer_known_mask = labels_known[infer_indices]
        for i in infer_indices.tolist():
            raw = int(labels_value[int(i)])
            infer_labels_raw.append(None if raw < 0 else raw)

        if bool(np.all(infer_known_mask)):
            infer_labels_np = np.asarray([int(x) for x in infer_labels_raw], dtype=np.int32)

        infer_scores_csv = ctx.reports_dir / "inference_scores.csv"
        save_score_csv(
            output_path=infer_scores_csv,
            model_names=infer_model_names,
            labels=infer_labels_raw,
            scores=infer_scores,
        )

        threshold_rows = compute_infer_threshold_rows(
            train_scores=np.asarray(train_scores, dtype=np.float64),
            infer_scores=np.asarray(infer_scores, dtype=np.float64),
            percentiles=[float(x) for x in score_percentiles],
            infer_labels=infer_labels_np,
        )
        threshold_rows_from_inference = compute_infer_threshold_rows_from_inference(
            infer_scores=np.asarray(infer_scores, dtype=np.float64),
            percentiles=[float(x) for x in score_percentiles],
            infer_labels=infer_labels_np,
        )
        infer_offline_metrics = compute_offline_metrics(
            infer_labels_np,
            np.asarray(infer_scores, dtype=np.float64),
        )
        inference_summary = summarize_scores(np.asarray(infer_scores, dtype=np.float64))

    threshold_selection_cfg = manifest.get("threshold_selection", {})
    selected_threshold_summary: dict[str, Any] | None = None
    selected_threshold_path: Path | None = None
    if calibration_indices.size > 0:
        accepted_fpr_values = threshold_selection_cfg.get("accepted_fprs")
        if accepted_fpr_values is None:
            legacy_value = threshold_selection_cfg.get("accepted_fpr")
            accepted_fpr_values = None if legacy_value is None else [legacy_value]
        resolved_accepted_fprs = _resolve_accepted_fprs(accepted_fpr_values)
        if resolved_accepted_fprs is None:
            raise ValueError(
                "Calibration split is present, but tuning manifest is missing "
                "threshold_selection.accepted_fprs"
            )

        selected_threshold_rows: list[dict[str, Any]] = []
        for accepted_fpr_value in resolved_accepted_fprs:
            selected_threshold_metrics = _select_threshold_max_recall_under_fpr(
                labels_true=np.asarray(calibration_labels_raw, dtype=np.int32),
                scores=np.asarray(calibration_scores, dtype=np.float64),
                accepted_fpr=float(accepted_fpr_value),
            )
            inference_selected_metrics: dict[str, Any] | None = None
            if infer_scores is not None and infer_labels_np is not None:
                inference_selected_metrics = _evaluate_binary_threshold(
                    labels_true=infer_labels_np,
                    scores=np.asarray(infer_scores, dtype=np.float64),
                    threshold=float(selected_threshold_metrics["threshold"]),
                )

            selected_threshold_rows.append(
                {
                    "accepted_fpr": float(accepted_fpr_value),
                    "threshold": float(selected_threshold_metrics["threshold"]),
                    "calibration_metrics": selected_threshold_metrics,
                    "inference_metrics": inference_selected_metrics,
                }
            )

        selected_threshold_summary = _build_selected_threshold_summary(
            selections=selected_threshold_rows,
        )
        selected_threshold_path = ctx.reports_dir / SELECTED_THRESHOLD_FILENAME
        with open(selected_threshold_path, "w", encoding="utf-8") as f:
            json.dump(json_ready(selected_threshold_summary), f, indent=2)

    selected_threshold_specs = _build_selected_threshold_specs(selected_threshold_summary)
    train_offline_metrics = compute_offline_metrics(
        np.asarray(y_train, dtype=np.int32),
        np.asarray(train_scores, dtype=np.float64),
    )
    attack_analysis = {
        "train": _summarize_attack_groups(
            sample_identities=train_sample_identities,
            labels=train_labels_raw,
            scores=np.asarray(train_scores, dtype=np.float64),
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        ),
        "inference": (
            _summarize_attack_groups(
                sample_identities=infer_sample_identities,
                labels=infer_labels_raw,
                scores=np.asarray(infer_scores, dtype=np.float64),
                threshold_specs=threshold_specs,
                selected_threshold_specs=selected_threshold_specs,
            )
            if infer_scores is not None
            else None
        ),
    }

    report = {
        "data_info": {
            "mode": manifest["mode"],
            "n_samples": int(features.shape[0]),
            "n_train": int(train_indices.size),
            "n_training_pool": int(train_pool_indices.size),
            "n_calibration": int(calibration_indices.size),
            "n_inference": int(infer_indices.size),
            "split": manifest["data"].get("split"),
            "calibration_split": manifest["data"].get("calibration_split"),
            "n_train_clean": int(np.sum(y_train == 0)),
            "n_train_backdoored": int(np.sum(y_train == 1)),
            "n_calibration_clean": (
                int(np.sum(labels_value[calibration_indices] == 0))
                if calibration_indices.size > 0
                else 0
            ),
            "n_calibration_backdoored": (
                int(np.sum(labels_value[calibration_indices] == 1))
                if calibration_indices.size > 0
                else 0
            ),
            "n_inference_clean": (
                int(np.sum(labels_value[infer_indices] == 0))
                if infer_indices.size > 0
                else 0
            ),
            "n_inference_backdoored": (
                int(np.sum(labels_value[infer_indices] == 1))
                if infer_indices.size > 0
                else 0
            ),
            "n_inference_unknown_label": (
                int(np.sum(labels_value[infer_indices] < 0))
                if infer_indices.size > 0
                else 0
            ),
        },
        "representation": {
            "extractor": manifest["extractor"]["name"],
            "extractor_params": manifest["extractor"]["params"],
            "extractor_metadata": _compact_extractor_metadata(manifest["extractor"]["metadata"]),
            "feature_path": manifest["data"]["feature_path"],
        },
        "tuning": {
            "metric": "roc_auc",
            "model_name": manifest["tuning"]["model_name"],
            "model_names": manifest["tuning"].get("model_names", [manifest["tuning"]["model_name"]]),
            "cv_random_states": manifest["tuning"].get("cv_random_states", [manifest["tuning"]["random_state"]]),
            "cv_folds_resolved": manifest["tuning"]["cv_folds_resolved"],
            "estimated_total_fits": manifest["tuning"].get("estimated_total_fits"),
            "executor": manifest["tuning"]["executor"],
            "tasks_total": len(manifest["tuning"]["tasks"]),
            "candidates": task_results,
            "winner": winner,
        },
        "fit_assessment": {
            "score_definition": "positive_class_score",
            "train_score_summary": summarize_scores(np.asarray(train_scores, dtype=np.float64)),
            "train_offline_metrics": train_offline_metrics,
            "calibration_score_summary": calibration_score_summary,
            "calibration_offline_metrics": calibration_offline_metrics,
            "inference_score_summary": inference_summary,
            "threshold_evaluation": threshold_rows,
            "threshold_evaluation_from_inference": threshold_rows_from_inference,
            "offline_metrics": infer_offline_metrics,
        },
        "threshold_selection": selected_threshold_summary,
        "attack_analysis": attack_analysis,
        "warnings": manifest.get("warnings", []),
    }

    report_path = ctx.reports_dir / "supervised_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    export_train_features_path: Path | None = None
    export_train_labels_path: Path | None = None
    if not skip_feature_importance:
        export_train_features_path = ctx.features_dir / FINALIZE_EXPORT_TRAIN_FEATURES_FILENAME
        export_train_labels_path = ctx.features_dir / FINALIZE_EXPORT_TRAIN_LABELS_FILENAME
        np.save(export_train_features_path, np.asarray(x_train, dtype=np.float32))
        np.save(export_train_labels_path, np.asarray(y_train, dtype=np.int32))

    run_config = {
        "pipeline": PIPELINE_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": manifest["manifest_json"],
        "dataset_root": manifest["dataset_root"],
        "mode": manifest["mode"],
        "tuning_executor": manifest["tuning"]["executor"],
        "model_name": manifest["tuning"]["model_name"],
        "model_names": manifest["tuning"].get("model_names", [manifest["tuning"]["model_name"]]),
        "train_split_percent": int(manifest["tuning"].get("train_split_percent", 100)),
        "calibration_split_percent": manifest["tuning"].get("calibration_split_percent"),
        "accepted_fprs": threshold_selection_cfg.get("accepted_fprs"),
        "accepted_fpr": threshold_selection_cfg.get("accepted_fpr"),
        "split_by_folder": bool(manifest["tuning"].get("split_by_folder", False)),
        "data_split": manifest["data"].get("split"),
        "calibration_split": manifest["data"].get("calibration_split"),
        "cv_random_states": manifest["tuning"].get("cv_random_states", [manifest["tuning"]["random_state"]]),
        "score_percentiles": [float(x) for x in score_percentiles],
        "winner": winner,
        "threshold_selection": selected_threshold_summary,
        "skip_feature_importance": bool(skip_feature_importance),
        "warnings": manifest.get("warnings", []),
    }
    artifacts = {
        "best_model": str(model_path),
        "train_scores_csv": str(train_scores_csv),
        "calibration_scores_csv": str(calibration_scores_csv) if calibration_scores_csv is not None else None,
        "inference_scores_csv": str(infer_scores_csv) if infer_scores_csv is not None else None,
        "selected_threshold": str(selected_threshold_path) if selected_threshold_path is not None else None,
        "report": str(report_path),
        "tuning_manifest": str(tuning_manifest_path),
        "tuning_tasks_dir": str(task_dir),
    }
    state_path = _write_finalize_state(
        run_dir=run_dir,
        run_config=run_config,
        artifacts=artifacts,
        winner_export_train_features=export_train_features_path,
        winner_export_train_labels=export_train_labels_path,
        random_state=int(manifest["tuning"]["random_state"]),
        skip_feature_importance=bool(skip_feature_importance),
    )

    return {
        "run_dir": str(run_dir),
        "report": str(report_path),
        "train_scores_csv": str(train_scores_csv),
        "calibration_scores_csv": str(calibration_scores_csv) if calibration_scores_csv is not None else None,
        "inference_scores_csv": str(infer_scores_csv) if infer_scores_csv is not None else None,
        "best_model": str(model_path),
        "train_features": x_train,
        "train_labels": y_train,
        "random_state": int(manifest["tuning"]["random_state"]),
        "tuning_manifest_path": tuning_manifest_path,
        "finalize_state_path": state_path,
        "winner_export_train_features": export_train_features_path,
        "winner_export_train_labels": export_train_labels_path,
    }


def _complete_supervised_finalize(
    *,
    run_dir: Path,
    winner_exports: dict[str, Path] | None,
) -> dict[str, Any]:
    state = _load_finalize_state(run_dir)
    ctx = _context_from_run_dir(run_dir)

    base_artifacts = state.get("artifacts", {})
    if not isinstance(base_artifacts, dict):
        raise ValueError("Finalize state is missing base artifacts")
    for key, value in base_artifacts.items():
        if value:
            ctx.add_artifact(str(key), Path(str(value)))

    if winner_exports is not None:
        ctx.add_artifact("winner_feature_weights_coefficients_csv", winner_exports["coefficient_csv"])
        ctx.add_artifact("winner_feature_weights_by_metric_csv", winner_exports["metric_csv"])
        ctx.add_artifact("winner_feature_weights_by_block_csv", winner_exports["block_csv"])
        ctx.add_artifact("winner_feature_weights_metadata_json", winner_exports["metadata_json"])

    run_config = state.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("Finalize state is missing run_config")
    ctx.finalize(run_config)
    _cleanup_finalize_intermediates(run_dir, state)

    return {
        "run_dir": str(run_dir),
        "report": str(base_artifacts.get("report")),
        "train_scores_csv": str(base_artifacts.get("train_scores_csv")),
        "inference_scores_csv": (
            str(base_artifacts.get("inference_scores_csv"))
            if base_artifacts.get("inference_scores_csv")
            else None
        ),
        "best_model": str(base_artifacts.get("best_model")),
    }


def _prepare_supervised_finalize_distributed(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
    finalize_export_shards: int,
    skip_feature_importance: bool,
) -> dict[str, Any]:
    prepared = _prepare_supervised_finalize(
        run_dir=run_dir,
        score_percentiles=score_percentiles,
        skip_feature_importance=skip_feature_importance,
    )
    if skip_feature_importance:
        finalized = _complete_supervised_finalize(
            run_dir=run_dir,
            winner_exports=None,
        )
        return {
            **finalized,
            "winner_feature_weights_mode": "skipped",
            "winner_feature_weights_tasks": 0,
        }
    if prepared["winner_export_train_features"] is None or prepared["winner_export_train_labels"] is None:
        raise RuntimeError("Finalize export inputs were not materialized for distributed feature importance")
    export_prepared = prepare_winner_feature_weights_export(
        run_dir=run_dir,
        report_path=Path(prepared["report"]),
        manifest_path=Path(prepared["tuning_manifest_path"]),
        artifact_index_path=run_dir / "artifact_index.json",
        train_feature_matrix_path=Path(prepared["winner_export_train_features"]),
        train_labels_path=Path(prepared["winner_export_train_labels"]),
        random_state=int(prepared["random_state"]),
        n_tasks=int(finalize_export_shards),
    )
    return {
        "run_dir": str(run_dir),
        "report": prepared["report"],
        "train_scores_csv": prepared["train_scores_csv"],
        "inference_scores_csv": prepared["inference_scores_csv"],
        "best_model": prepared["best_model"],
        "winner_feature_weights_mode": export_prepared["mode"],
        "winner_feature_weights_manifest": str(export_prepared["manifest_path"]),
        "winner_feature_weights_tasks": int(export_prepared["n_tasks"]),
    }


def _run_supervised_finalize_worker(
    *,
    run_dir: Path,
    task_index: int,
    n_jobs: int | None,
) -> dict[str, Any]:
    return run_winner_feature_weights_export_worker(
        run_dir=run_dir,
        task_index=task_index,
        n_jobs=1 if n_jobs is None else int(n_jobs),
    )


def _merge_supervised_finalize(
    *,
    run_dir: Path,
) -> dict[str, Any]:
    winner_exports = merge_winner_feature_weights_export(
        run_dir=run_dir,
        artifact_index_path=run_dir / "artifact_index.json",
    )
    return _complete_supervised_finalize(
        run_dir=run_dir,
        winner_exports=winner_exports,
    )


def _finalize_supervised_run(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
    n_jobs: int,
    skip_feature_importance: bool,
) -> dict[str, Any]:
    prepared = _prepare_supervised_finalize(
        run_dir=run_dir,
        score_percentiles=score_percentiles,
        skip_feature_importance=skip_feature_importance,
    )
    if skip_feature_importance:
        return _complete_supervised_finalize(
            run_dir=run_dir,
            winner_exports=None,
        )
    winner_exports = export_winner_feature_weights(
        run_dir=run_dir,
        report_path=Path(prepared["report"]),
        manifest_path=Path(prepared["tuning_manifest_path"]),
        artifact_index_path=run_dir / "artifact_index.json",
        train_features=np.asarray(prepared["train_features"]),
        train_labels=np.asarray(prepared["train_labels"]),
        random_state=int(prepared["random_state"]),
        n_jobs=int(n_jobs),
    )
    return _complete_supervised_finalize(
        run_dir=run_dir,
        winner_exports=winner_exports,
    )


def run_supervised_pipeline(
    *,
    manifest_json: Path | None,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str,
    spectral_features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    random_state: int,
    train_split_percent: int,
    calibration_split_percent: int | None,
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
    split_by_folder: bool,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float] | None,
    feature_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    slurm_max_concurrent: str,
    slurm_cpus_per_task: str,
    finalize_export_shards: int,
    stage: str,
    run_dir: Path | None,
    task_index: int | None,
    skip_feature_importance: bool = False,
) -> dict[str, Any]:
    if stage in {"all", "prepare"} and manifest_json is None:
        raise ValueError("--manifest-json is required for stage=all and stage=prepare")
    if stage in {"all", "prepare"} and feature_file is None:
        raise ValueError(
            "--feature-file is required for stage=all and stage=prepare. Pass a feature run name, "
            "feature output directory, or spectral_features.npy path."
        )
    if stage in {"all", "prepare"} and not spectral_features:
        raise ValueError(
            "--features is required for stage=all and stage=prepare. Specify the feature groups to "
            "select from the extracted feature bundle."
        )
    if stage in {"worker", "finalize", "finalize_prepare", "finalize_worker", "finalize_merge"} and run_dir is None:
        raise ValueError("--run-dir is required for the selected supervised stage")
    if stage in {"worker", "finalize_worker"} and task_index is None:
        raise ValueError("--task-index is required for the selected supervised worker stage")
    resolved_accepted_fprs = _resolve_accepted_fprs(accepted_fpr)
    if (calibration_split_percent is None) != (resolved_accepted_fprs is None):
        raise ValueError("--calibration-split and --accepted-fpr must either both be set or both be omitted")

    if stage == "prepare":
        return _prepare_supervised_run(
            manifest_json=Path(manifest_json) if manifest_json is not None else Path(""),
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=run_id,
            model_name=model_name,
            spectral_features=spectral_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            cv_folds=cv_folds,
            random_state=random_state,
            train_split_percent=train_split_percent,
            calibration_split_percent=calibration_split_percent,
            accepted_fpr=resolved_accepted_fprs,
            split_by_folder=split_by_folder,
            cv_random_states=cv_random_states,
            n_jobs=n_jobs,
            score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
            feature_file=feature_file,
            tuning_executor=tuning_executor,
            slurm_partition=slurm_partition,
            slurm_max_concurrent=slurm_max_concurrent,
            slurm_cpus_per_task=slurm_cpus_per_task,
        )

    if stage == "worker":
        return _run_supervised_worker(
            run_dir=Path(run_dir),
            task_index=int(task_index) if task_index is not None else -1,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
        )

    if stage == "finalize":
        return _finalize_supervised_run(
            run_dir=Path(run_dir),
            score_percentiles=score_percentiles,
            n_jobs=int(n_jobs),
            skip_feature_importance=bool(skip_feature_importance),
        )

    if stage == "finalize_prepare":
        return _prepare_supervised_finalize_distributed(
            run_dir=Path(run_dir),
            score_percentiles=score_percentiles,
            finalize_export_shards=int(finalize_export_shards),
            skip_feature_importance=bool(skip_feature_importance),
        )

    if stage == "finalize_worker":
        return _run_supervised_finalize_worker(
            run_dir=Path(run_dir),
            task_index=int(task_index) if task_index is not None else -1,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
        )

    if stage == "finalize_merge":
        return _merge_supervised_finalize(
            run_dir=Path(run_dir),
        )

    # stage == "all"
    prepared = _prepare_supervised_run(
        manifest_json=Path(manifest_json) if manifest_json is not None else Path(""),
        dataset_root=dataset_root,
        output_root=output_root,
        run_id=run_id,
        model_name=model_name,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        cv_folds=cv_folds,
        random_state=random_state,
        train_split_percent=train_split_percent,
        calibration_split_percent=calibration_split_percent,
        accepted_fpr=resolved_accepted_fprs,
        split_by_folder=split_by_folder,
        cv_random_states=cv_random_states,
        n_jobs=n_jobs,
        score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
        feature_file=feature_file,
        tuning_executor=tuning_executor,
        slurm_partition=slurm_partition,
        slurm_max_concurrent=slurm_max_concurrent,
        slurm_cpus_per_task=slurm_cpus_per_task,
    )
    resolved_run_dir = Path(prepared["run_dir"])

    if tuning_executor == "slurm_array":
        n_tasks = int(prepared["n_tasks"])
        max_concurrent = int(prepared["slurm_max_concurrent"])
        return {
            **prepared,
            "next_steps": build_slurm_array_next_steps(
                run_dir=resolved_run_dir,
                n_tasks=n_tasks,
                max_concurrent=max_concurrent,
                skip_feature_importance=bool(skip_feature_importance),
            ),
        }

    for idx in range(int(prepared["n_tasks"])):
        _run_supervised_worker(
            run_dir=resolved_run_dir,
            task_index=idx,
            n_jobs=n_jobs,
        )

    finalized = _finalize_supervised_run(
        run_dir=resolved_run_dir,
        score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
        n_jobs=n_jobs,
        skip_feature_importance=bool(skip_feature_importance),
    )
    return {
        **finalized,
        "tuning_manifest": prepared["tuning_manifest"],
        "n_tasks": prepared["n_tasks"],
    }
