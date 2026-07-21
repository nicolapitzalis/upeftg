from __future__ import annotations

from typing import Any

import numpy as np


def resolve_accepted_fprs(
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


def threshold_candidate_values(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Threshold selection requires at least one score")
    unique_scores = np.unique(scores)
    above_max = np.nextafter(float(np.max(unique_scores)), float("inf"))
    return np.concatenate((np.asarray([above_max], dtype=np.float64), unique_scores[::-1]))


def evaluate_binary_threshold(
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


def select_threshold_max_recall_under_fpr(
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
        evaluate_binary_threshold(labels_true=labels_true, scores=scores, threshold=float(threshold))
        for threshold in threshold_candidate_values(scores)
    ]
    feasible = [row for row in candidates if float(row["false_positive_rate"]) <= float(accepted_fpr) + 1e-12]
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


def build_selected_threshold_summary(
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


def build_selected_threshold_specs(
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
