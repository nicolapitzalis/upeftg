from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
    v_measure_score,
)


def format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def sanitize_k_list(k_list: list[int], n_samples: int) -> tuple[list[int], list[str]]:
    valid: list[int] = []
    warnings: list[str] = []
    for k in k_list:
        if k <= 1:
            warnings.append(f"Ignoring invalid k={k}; must be >=2")
            continue
        if k >= n_samples:
            clipped = max(2, n_samples - 1)
            warnings.append(f"Clipped k={k} to {clipped} because n_samples={n_samples}")
            k = clipped
        valid.append(k)
    valid = sorted(set(valid))
    if not valid and n_samples >= 3:
        valid = [2]
        warnings.append("No valid k values left; using fallback [2]")
    return valid, warnings


def sanitize_gmm_components(component_list: list[int], n_samples: int) -> tuple[list[int], list[str]]:
    valid: list[int] = []
    warnings: list[str] = []
    for n in component_list:
        if n <= 0:
            warnings.append(f"Ignoring invalid GMM component count={n}")
            continue
        if n > n_samples:
            warnings.append(f"Ignoring GMM component count={n}; n_samples={n_samples}")
            continue
        valid.append(n)
    valid = sorted(set(valid))
    if not valid:
        valid = [1, 2] if n_samples >= 2 else [1]
        warnings.append(f"No valid GMM component counts left; using fallback {valid}")
    return valid, warnings


def compute_cluster_sizes(labels_pred: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels_pred, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def compute_partition_internal_metrics(z: np.ndarray, labels_pred: np.ndarray) -> dict[str, float | None]:
    unique_labels = np.unique(labels_pred)
    n_unique = unique_labels.size
    n_samples = z.shape[0]

    metrics = {
        "n_clusters_excluding_noise": int(np.sum(unique_labels >= 0)),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if n_unique < 2 or n_unique >= n_samples:
        return metrics

    try:
        metrics["silhouette"] = float(silhouette_score(z, labels_pred))
    except Exception:
        metrics["silhouette"] = None

    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(z, labels_pred))
    except Exception:
        metrics["davies_bouldin"] = None

    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(z, labels_pred))
    except Exception:
        metrics["calinski_harabasz"] = None

    return metrics


def compute_partition_offline_metrics(
    labels_true: np.ndarray | None,
    labels_pred: np.ndarray,
) -> dict[str, float | None]:
    metrics = {
        "adjusted_rand": None,
        "normalized_mutual_info": None,
        "homogeneity": None,
        "v_measure": None,
    }
    if labels_true is None:
        return metrics

    try:
        metrics["adjusted_rand"] = float(adjusted_rand_score(labels_true, labels_pred))
    except Exception:
        pass

    try:
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(labels_true, labels_pred))
    except Exception:
        pass

    try:
        metrics["homogeneity"] = float(homogeneity_score(labels_true, labels_pred))
    except Exception:
        pass

    try:
        metrics["v_measure"] = float(v_measure_score(labels_true, labels_pred))
    except Exception:
        pass

    return metrics


def precision_at_k(labels_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(labels_true[idx] == 1))


def compute_score_offline_metrics(labels_true: np.ndarray | None, scores: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
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


def score_percentile_stats(scores: np.ndarray, percentiles: list[float]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for pct in percentiles:
        threshold = float(np.percentile(scores, pct))
        rows.append(
            {
                "percentile": float(pct),
                "threshold": threshold,
                "n_above_threshold": int(np.sum(scores >= threshold)),
                "fraction_above_threshold": float(np.mean(scores >= threshold)),
            }
        )
    return rows


def pairwise_mean_ari(label_runs: list[np.ndarray]) -> float | None:
    if len(label_runs) < 2:
        return None
    scores = [adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)]
    return float(np.mean(scores)) if scores else None


def pairwise_mean_spearman(score_runs: list[np.ndarray]) -> float | None:
    if len(score_runs) < 2:
        return None
    vals = []
    for a, b in combinations(score_runs, 2):
        corr = spearmanr(a, b).correlation
        if corr is not None and not np.isnan(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else None


def pick_unsupervised_winner(results: list[dict[str, Any]], selection_metric: str) -> dict[str, Any] | None:
    if not results:
        return None

    if selection_metric == "silhouette":
        candidates = [r for r in results if r.get("selection_value") is not None]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: float(x["selection_value"]))
        return {
            "metric": "silhouette",
            "winner": {
                "algorithm": best["algorithm"],
                "variant": best["variant"],
                "selection_value": best["selection_value"],
            },
        }

    if selection_metric == "bic":
        candidates = [
            r for r in results
            if r.get("algorithm") == "gmm" and r.get("selection_value") is not None
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda x: float(x["selection_value"]))
        return {
            "metric": "bic",
            "winner": {
                "algorithm": best["algorithm"],
                "variant": best["variant"],
                "selection_value": best["selection_value"],
            },
        }

    if selection_metric == "stability":
        candidates = [r for r in results if r.get("stability_score") is not None]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: float(x["stability_score"]))
        return {
            "metric": "stability",
            "winner": {
                "algorithm": best["algorithm"],
                "variant": best["variant"],
                "stability_score": best["stability_score"],
            },
        }

    return None


def summarize_offline_eval(results: list[dict[str, Any]]) -> dict[str, Any]:
    score_models = []
    partition_models = []

    for row in results:
        if row.get("score_offline_metrics") is not None:
            score_models.append(row)
        if row.get("partition_offline_metrics") is not None:
            partition_models.append(row)

    best_score = None
    valid_score = [
        r for r in score_models
        if r["score_offline_metrics"].get("auroc") is not None
    ]
    if valid_score:
        best = max(valid_score, key=lambda x: x["score_offline_metrics"]["auroc"])
        best_score = {
            "algorithm": best["algorithm"],
            "variant": best["variant"],
            **best["score_offline_metrics"],
        }

    best_partition = None
    valid_partition = [
        r for r in partition_models
        if r["partition_offline_metrics"].get("adjusted_rand") is not None
    ]
    if valid_partition:
        best = max(valid_partition, key=lambda x: x["partition_offline_metrics"]["adjusted_rand"])
        best_partition = {
            "algorithm": best["algorithm"],
            "variant": best["variant"],
            **best["partition_offline_metrics"],
        }

    return {
        "best_score_model_by_auroc": best_score,
        "best_partition_model_by_ari": best_partition,
    }


def summarize_stability(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in results if r.get("stability_score") is not None]
    if not valid:
        return {"best_by_stability": None}

    best = max(valid, key=lambda x: x["stability_score"])
    return {
        "best_by_stability": {
            "algorithm": best["algorithm"],
            "variant": best["variant"],
            "stability_score": best["stability_score"],
        }
    }
