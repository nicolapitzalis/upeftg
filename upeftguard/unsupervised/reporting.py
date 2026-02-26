from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def precision_at_k(labels_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(labels_true[idx] == 1))


def compute_offline_metrics(labels_true: np.ndarray | None, scores: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "auroc": None,
        "auprc": None,
        "precision_at_num_positives": None,
        "precision_at_5": None,
        "precision_at_10": None,
    }
    if labels_true is None:
        return metrics

    positives = int(np.sum(labels_true == 1))
    if positives <= 0 or positives >= len(labels_true):
        return metrics

    try:
        metrics["auroc"] = float(roc_auc_score(labels_true, scores))
    except Exception:
        metrics["auroc"] = None

    try:
        metrics["auprc"] = float(average_precision_score(labels_true, scores))
    except Exception:
        metrics["auprc"] = None

    metrics["precision_at_num_positives"] = precision_at_k(labels_true, scores, positives)
    metrics["precision_at_5"] = precision_at_k(labels_true, scores, 5)
    metrics["precision_at_10"] = precision_at_k(labels_true, scores, 10)
    return metrics


def compute_infer_threshold_rows(
    train_scores: np.ndarray,
    infer_scores: np.ndarray,
    percentiles: list[float],
    infer_labels: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_infer = int(infer_scores.size)

    for pct in percentiles:
        threshold = float(np.percentile(train_scores, pct))
        flagged = infer_scores >= threshold
        n_flagged = int(np.sum(flagged))

        row: dict[str, Any] = {
            "percentile_from_train": float(pct),
            "threshold": threshold,
            "n_flagged_in_inference": n_flagged,
            "fraction_flagged_in_inference": float(n_flagged / max(1, n_infer)),
        }

        if infer_labels is not None:
            positives = int(np.sum(infer_labels == 1))
            negatives = int(np.sum(infer_labels == 0))
            tp = int(np.sum((infer_labels == 1) & flagged))
            fp = int(np.sum((infer_labels == 0) & flagged))
            if n_flagged > 0:
                row["precision"] = float(tp / n_flagged)
            if positives > 0:
                row["recall"] = float(tp / positives)
            if negatives > 0:
                row["false_positive_rate"] = float(fp / negatives)

        rows.append(row)

    return rows


def save_score_csv(
    output_path: Path,
    model_names: list[str],
    labels: list[int | None],
    scores: np.ndarray,
) -> None:
    ranks = np.argsort(np.argsort(scores))
    pct = ranks / max(1, len(scores) - 1)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "model_name",
                "label",
                "score",
                "score_percentile_rank",
            ],
        )
        writer.writeheader()
        for i, (name, label, score, rank_pct) in enumerate(zip(model_names, labels, scores, pct)):
            writer.writerow(
                {
                    "index": i,
                    "model_name": name,
                    "label": label,
                    "score": float(score),
                    "score_percentile_rank": float(rank_pct),
                }
            )
