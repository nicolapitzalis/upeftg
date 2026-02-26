#!/usr/bin/env python3
"""
Phase 2: Unsupervised Baselines in Embedding Space

This script evaluates unsupervised clustering and anomaly-scoring baselines on
SVD embeddings (or an externally provided feature matrix), selects a winner
using unsupervised criteria only, and optionally computes offline label metrics
for benchmarking.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
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
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

SCRIPT_VERSION = "2.0.0"
DEFAULT_DATA_DIR = Path("processed_data")
DEFAULT_OUTPUT_DIR = Path("clustering_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unsupervised baselines on SVD embeddings and report results"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with Z_*.npy artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write reports")
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Embedding dimensionality to load from Z_<k>.npy. If omitted, uses representativeness winner when available",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=None,
        help="Optional .npy feature matrix override. If set, skips Z_<k> loading",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["kmeans", "hierarchical", "dbscan", "gmm", "mahalanobis", "isolation_forest", "lof"],
        help="Subset of algorithms to run",
    )
    parser.add_argument("--k-list", nargs="+", type=int, default=[2, 3, 4, 5], help="Cluster counts for KMeans/Hierarchical")
    parser.add_argument("--eps-list", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0], help="DBSCAN eps values")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples and LOF neighbor floor")
    parser.add_argument(
        "--gmm-components",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="GMM component-count grid",
    )
    parser.add_argument(
        "--gmm-covariance-types",
        nargs="+",
        default=["diag", "full", "tied", "spherical"],
        help="GMM covariance types",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["silhouette", "bic", "stability"],
        default="silhouette",
        help="Unsupervised metric used to pick the best algorithm/config",
    )
    parser.add_argument(
        "--use-offline-label-metrics",
        action="store_true",
        help="Compute label-based offline metrics for benchmarking (not for model selection)",
    )
    parser.add_argument(
        "--stability-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Seeds used for stability checks on stochastic methods",
    )
    parser.add_argument(
        "--score-percentiles",
        nargs="+",
        type=float,
        default=[90, 95, 97, 99],
        help="Percentile thresholds for anomaly-score calibration",
    )
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(x) for x in obj]
    return obj


def format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def load_representation(
    data_dir: Path,
    n_components: int | None,
    feature_file: Path | None,
) -> tuple[np.ndarray, dict[str, Any], int | None]:
    if feature_file is not None:
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        z = np.load(feature_file)
        representation_info = {
            "source": str(feature_file),
            "type": "external_feature_file",
            "shape": [int(z.shape[0]), int(z.shape[1])],
            "n_components": int(z.shape[1]),
        }
        return z, representation_info, None

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    available = sorted(
        int(p.stem.split("_")[1])
        for p in data_dir.glob("Z_*.npy")
        if p.stem.split("_")[1].isdigit()
    )
    if not available:
        raise FileNotFoundError(f"No Z_<k>.npy files found in {data_dir}")

    chosen_k = n_components
    if chosen_k is None:
        summary_path = data_dir / "representativeness_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            winner = summary.get("winner")
            if isinstance(winner, dict) and isinstance(winner.get("k"), int):
                candidate = int(winner["k"])
                if candidate in available:
                    chosen_k = candidate
        if chosen_k is None:
            chosen_k = max(available)

    if chosen_k not in available:
        raise ValueError(f"Requested n-components={chosen_k}, available values={available}")

    z_path = data_dir / f"Z_{chosen_k}.npy"
    info_path = data_dir / f"svd_info_{chosen_k}.json"

    z = np.load(z_path)
    svd_info = None
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            svd_info = json.load(f)

    representation_info = {
        "source": str(z_path),
        "type": "svd_embedding",
        "shape": [int(z.shape[0]), int(z.shape[1])],
        "n_components": int(chosen_k),
        "svd_info": svd_info,
    }
    return z, representation_info, chosen_k


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


def compute_partition_offline_metrics(labels_true: np.ndarray | None, labels_pred: np.ndarray) -> dict[str, float | None]:
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
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(labels_true, labels_pred))
        metrics["homogeneity"] = float(homogeneity_score(labels_true, labels_pred))
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


def score_percentile_stats(
    scores: np.ndarray,
    percentiles: list[float],
    labels_true: np.ndarray | None,
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for pct in percentiles:
        threshold = float(np.percentile(scores, pct))
        flagged = scores >= threshold
        item: dict[str, Any] = {
            "percentile": float(pct),
            "threshold": threshold,
            "n_flagged": int(np.sum(flagged)),
        }
        if labels_true is not None and np.any(flagged):
            precision = float(np.mean(labels_true[flagged] == 1))
            recall = float(np.sum((labels_true == 1) & flagged) / max(1, np.sum(labels_true == 1)))
            item["precision"] = precision
            item["recall"] = recall
        stats.append(item)
    return stats


def pairwise_mean_ari(label_runs: list[np.ndarray]) -> float | None:
    if len(label_runs) < 2:
        return None
    aris = [adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)]
    if not aris:
        return None
    return float(np.mean(aris))


def pairwise_mean_spearman(score_runs: list[np.ndarray]) -> float | None:
    if len(score_runs) < 2:
        return None
    corrs: list[float] = []
    for a, b in combinations(score_runs, 2):
        corr = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
        corrs.append(corr)
    if not corrs:
        return None
    return float(np.mean(corrs))


def run_kmeans(
    z: np.ndarray,
    labels_true: np.ndarray | None,
    k_list: list[int],
    seeds: list[int],
    use_offline_metrics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for k in k_list:
        label_runs: list[np.ndarray] = []
        inertia_runs: list[float] = []
        centers_primary = None

        for seed in seeds:
            km = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels_pred = km.fit_predict(z)
            label_runs.append(labels_pred)
            inertia_runs.append(float(km.inertia_))
            if centers_primary is None:
                centers_primary = km.cluster_centers_.astype(np.float32)

        labels_primary = label_runs[0]
        internal = compute_partition_internal_metrics(z, labels_primary)
        offline = compute_partition_offline_metrics(labels_true, labels_primary) if use_offline_metrics else {}
        stability = pairwise_mean_ari(label_runs)

        results.append(
            {
                "algorithm": "kmeans",
                "variant": f"kmeans_k={k}",
                "kind": "partition",
                "config": {"k": int(k)},
                "cluster_sizes": compute_cluster_sizes(labels_primary),
                "labels_pred": labels_primary.tolist(),
                "centers": centers_primary.tolist() if centers_primary is not None else None,
                "inertia_mean": float(np.mean(inertia_runs)),
                "internal_metrics": internal,
                "offline_metrics": offline,
                "stability_score": stability,
                "selection_values": {
                    "silhouette": internal.get("silhouette"),
                    "bic": None,
                    "stability": stability,
                },
            }
        )

    return results


def run_hierarchical(
    z: np.ndarray,
    labels_true: np.ndarray | None,
    k_list: list[int],
    use_offline_metrics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for linkage in ["ward", "complete", "average"]:
        for k in k_list:
            hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels_pred = hc.fit_predict(z)
            internal = compute_partition_internal_metrics(z, labels_pred)
            offline = compute_partition_offline_metrics(labels_true, labels_pred) if use_offline_metrics else {}

            results.append(
                {
                    "algorithm": "hierarchical",
                    "variant": f"hierarchical_{linkage}_k={k}",
                    "kind": "partition",
                    "config": {"linkage": linkage, "k": int(k)},
                    "cluster_sizes": compute_cluster_sizes(labels_pred),
                    "labels_pred": labels_pred.tolist(),
                    "internal_metrics": internal,
                    "offline_metrics": offline,
                    "stability_score": 1.0,
                    "selection_values": {
                        "silhouette": internal.get("silhouette"),
                        "bic": None,
                        "stability": 1.0,
                    },
                }
            )

    return results


def run_dbscan(
    z: np.ndarray,
    labels_true: np.ndarray | None,
    eps_list: list[float],
    min_samples: int,
    use_offline_metrics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for eps in eps_list:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels_pred = db.fit_predict(z)
        internal = compute_partition_internal_metrics(z, labels_pred)
        offline = compute_partition_offline_metrics(labels_true, labels_pred) if use_offline_metrics else {}

        results.append(
            {
                "algorithm": "dbscan",
                "variant": f"dbscan_eps={eps}",
                "kind": "partition",
                "config": {"eps": float(eps), "min_samples": int(min_samples)},
                "cluster_sizes": compute_cluster_sizes(labels_pred),
                "labels_pred": labels_pred.tolist(),
                "internal_metrics": internal,
                "offline_metrics": offline,
                "stability_score": 1.0,
                "selection_values": {
                    "silhouette": internal.get("silhouette"),
                    "bic": None,
                    "stability": 1.0,
                },
            }
        )

    return results


def run_gmm(
    z: np.ndarray,
    labels_true: np.ndarray | None,
    components_list: list[int],
    covariance_types: list[str],
    seeds: list[int],
    use_offline_metrics: bool,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for n_components in components_list:
        for cov in covariance_types:
            label_runs: list[np.ndarray] = []
            score_runs: list[np.ndarray] = []
            primary = None

            for seed in seeds:
                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=cov,
                        reg_covar=1e-5,
                        n_init=1,
                        random_state=seed,
                    )
                    gmm.fit(z)
                    labels_pred = gmm.predict(z)
                    scores = -gmm.score_samples(z)
                    label_runs.append(labels_pred)
                    score_runs.append(scores)
                    if primary is None:
                        primary = (gmm, labels_pred, scores)
                except Exception:
                    continue

            if primary is None:
                continue

            gmm_primary, labels_pred, scores = primary
            internal = compute_partition_internal_metrics(z, labels_pred)
            offline_partition = compute_partition_offline_metrics(labels_true, labels_pred) if use_offline_metrics else {}
            offline_scores = compute_score_offline_metrics(labels_true, scores) if use_offline_metrics else {}

            stability_labels = pairwise_mean_ari(label_runs)
            stability_scores = pairwise_mean_spearman(score_runs)
            stability = stability_labels if stability_labels is not None else stability_scores

            results.append(
                {
                    "algorithm": "gmm",
                    "variant": f"gmm_{cov}_n={n_components}",
                    "kind": "both",
                    "config": {"n_components": int(n_components), "covariance_type": cov},
                    "cluster_sizes": compute_cluster_sizes(labels_pred),
                    "labels_pred": labels_pred.tolist(),
                    "scores": scores.tolist(),
                    "score_percentile_stats": score_percentile_stats(scores, percentiles, labels_true if use_offline_metrics else None),
                    "internal_metrics": {
                        **internal,
                        "bic": float(gmm_primary.bic(z)),
                        "aic": float(gmm_primary.aic(z)),
                    },
                    "offline_metrics": {
                        "partition": offline_partition,
                        "scores": offline_scores,
                    },
                    "stability_score": stability,
                    "stability_components": {
                        "labels_ari": stability_labels,
                        "scores_rank_spearman": stability_scores,
                    },
                    "selection_values": {
                        "silhouette": internal.get("silhouette"),
                        "bic": -float(gmm_primary.bic(z)),
                        "stability": stability,
                    },
                }
            )

    return results


def run_mahalanobis(
    z: np.ndarray,
    labels_true: np.ndarray | None,
    use_offline_metrics: bool,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    cov = EmpiricalCovariance()
    cov.fit(z)
    scores = cov.mahalanobis(z)

    offline = compute_score_offline_metrics(labels_true, scores) if use_offline_metrics else {}

    return [
        {
            "algorithm": "mahalanobis",
            "variant": "mahalanobis_empirical_covariance",
            "kind": "score",
            "config": {},
            "scores": scores.tolist(),
            "score_percentile_stats": score_percentile_stats(scores, percentiles, labels_true if use_offline_metrics else None),
            "internal_metrics": {},
            "offline_metrics": {"scores": offline},
            "stability_score": 1.0,
            "selection_values": {
                "silhouette": None,
                "bic": None,
                "stability": 1.0,
            },
        }
    ]


def run_isolation_forest(
    z: np.ndarray,
    labels_true: np.ndarray | None,
    seeds: list[int],
    use_offline_metrics: bool,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    score_runs: list[np.ndarray] = []
    for seed in seeds:
        iso = IsolationForest(random_state=seed, contamination="auto")
        iso.fit(z)
        score_runs.append(-iso.score_samples(z))

    scores = score_runs[0]
    offline = compute_score_offline_metrics(labels_true, scores) if use_offline_metrics else {}
    stability = pairwise_mean_spearman(score_runs)

    return [
        {
            "algorithm": "isolation_forest",
            "variant": "isolation_forest_auto_contamination",
            "kind": "score",
            "config": {},
            "scores": scores.tolist(),
            "score_percentile_stats": score_percentile_stats(scores, percentiles, labels_true if use_offline_metrics else None),
            "internal_metrics": {},
            "offline_metrics": {"scores": offline},
            "stability_score": stability,
            "selection_values": {
                "silhouette": None,
                "bic": None,
                "stability": stability,
            },
        }
    ]


def run_lof(
    z: np.ndarray,
    labels_true: np.ndarray | None,
    min_samples: int,
    use_offline_metrics: bool,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    n_neighbors = max(2, min(min_samples, z.shape[0] - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit_predict(z)
    scores = -lof.negative_outlier_factor_

    offline = compute_score_offline_metrics(labels_true, scores) if use_offline_metrics else {}

    return [
        {
            "algorithm": "lof",
            "variant": f"lof_n_neighbors={n_neighbors}",
            "kind": "score",
            "config": {"n_neighbors": int(n_neighbors)},
            "scores": scores.tolist(),
            "score_percentile_stats": score_percentile_stats(scores, percentiles, labels_true if use_offline_metrics else None),
            "internal_metrics": {},
            "offline_metrics": {"scores": offline},
            "stability_score": 1.0,
            "selection_values": {
                "silhouette": None,
                "bic": None,
                "stability": 1.0,
            },
        }
    ]


def pick_unsupervised_winner(results: list[dict[str, Any]], selection_metric: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for result in results:
        value = result.get("selection_values", {}).get(selection_metric)
        if value is None:
            continue
        candidates.append(
            {
                "variant": result["variant"],
                "algorithm": result["algorithm"],
                "selection_metric": selection_metric,
                "selection_value": float(value),
                "config": result.get("config", {}),
            }
        )

    if not candidates:
        return None

    winner = max(candidates, key=lambda x: x["selection_value"])
    return {
        "metric": selection_metric,
        "winner": winner,
        "candidates": candidates,
    }


def summarize_offline_eval(results: list[dict[str, Any]]) -> dict[str, Any]:
    score_rows = []
    partition_rows = []

    for result in results:
        offline = result.get("offline_metrics", {})
        if "scores" in offline and isinstance(offline["scores"], dict):
            row = {
                "variant": result["variant"],
                "algorithm": result["algorithm"],
                **offline["scores"],
            }
            score_rows.append(row)

        if "partition" in offline and isinstance(offline["partition"], dict):
            row = {
                "variant": result["variant"],
                "algorithm": result["algorithm"],
                **offline["partition"],
            }
            partition_rows.append(row)
        elif result.get("kind") in {"partition", "both"} and isinstance(offline, dict):
            row = {
                "variant": result["variant"],
                "algorithm": result["algorithm"],
                **offline,
            }
            partition_rows.append(row)

    best_auroc = None
    valid_score_rows = [r for r in score_rows if r.get("auroc") is not None]
    if valid_score_rows:
        best_auroc = max(valid_score_rows, key=lambda r: r["auroc"])

    best_ari = None
    valid_partition_rows = [r for r in partition_rows if r.get("adjusted_rand") is not None]
    if valid_partition_rows:
        best_ari = max(valid_partition_rows, key=lambda r: r["adjusted_rand"])

    return {
        "score_models": score_rows,
        "partition_models": partition_rows,
        "best_score_model_by_auroc": best_auroc,
        "best_partition_model_by_ari": best_ari,
    }


def summarize_stability(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        rows.append(
            {
                "variant": result["variant"],
                "algorithm": result["algorithm"],
                "stability_score": result.get("stability_score"),
                "stability_labels_ari": (
                    result.get("stability_components", {}).get("labels_ari")
                    if isinstance(result.get("stability_components"), dict)
                    else None
                ),
                "stability_scores_rank_spearman": (
                    result.get("stability_components", {}).get("scores_rank_spearman")
                    if isinstance(result.get("stability_components"), dict)
                    else None
                ),
            }
        )
    valid = [r for r in rows if r["stability_score"] is not None]
    best = max(valid, key=lambda r: r["stability_score"]) if valid else None
    return {
        "per_variant": rows,
        "best_by_stability": best,
    }


def save_score_table(output_path: Path, model_names: list[str], results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []

    for result in results:
        scores = result.get("scores")
        if scores is None:
            continue
        scores_np = np.asarray(scores, dtype=np.float64)
        ranks = np.argsort(np.argsort(scores_np))
        pct = ranks / max(1, len(scores_np) - 1)
        for i, (name, s, p) in enumerate(zip(model_names, scores_np, pct)):
            rows.append(
                {
                    "model_name": name,
                    "index": i,
                    "variant": result["variant"],
                    "algorithm": result["algorithm"],
                    "score": float(s),
                    "score_percentile_rank": float(p),
                }
            )

    if not rows:
        return

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _project_to_2d(z: np.ndarray) -> tuple[np.ndarray, str]:
    if z.shape[1] > 2:
        pca = PCA(n_components=2)
        z2 = pca.fit_transform(z)
        explained = float(np.sum(pca.explained_variance_ratio_))
        return z2, f"PCA(2) explained variance: {explained:.2%}"
    return z, ""


def _find_result_by_summary_item(
    results: list[dict[str, Any]],
    summary_item: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(summary_item, dict):
        return None

    variant = summary_item.get("variant")
    algorithm = summary_item.get("algorithm")
    if not isinstance(variant, str):
        return None

    for result in results:
        if result.get("variant") != variant:
            continue
        if algorithm is not None and result.get("algorithm") != algorithm:
            continue
        return result
    return None


def _extract_partition_offline_metrics(result: dict[str, Any]) -> dict[str, Any]:
    offline = result.get("offline_metrics", {})
    if not isinstance(offline, dict):
        return {}
    if isinstance(offline.get("partition"), dict):
        return offline["partition"]
    return offline


def _extract_score_offline_metrics(result: dict[str, Any]) -> dict[str, Any]:
    offline = result.get("offline_metrics", {})
    if not isinstance(offline, dict):
        return {}
    if isinstance(offline.get("scores"), dict):
        return offline["scores"]
    return {}


def _metric_axis_limits(
    values: np.ndarray,
    default_min: float,
    default_max: float,
) -> tuple[float, float]:
    valid = ~np.isnan(values)
    if not np.any(valid):
        return default_min, default_max
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    return min(default_min, min_val - 0.05), max(default_max, max_val + 0.05)


def render_partition_projection_plot(
    output_path: Path,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    result: dict[str, Any] | None,
    title_prefix: str,
) -> bool:
    if result is None or result.get("labels_pred") is None:
        return False

    labels_pred = np.asarray(result["labels_pred"], dtype=np.float64)
    if labels_pred.shape[0] != z.shape[0]:
        return False

    z2, subtitle = _project_to_2d(z)

    if labels_true is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax0, ax1 = axes

        sc0 = ax0.scatter(z2[:, 0], z2[:, 1], c=labels_true, cmap="RdBu", s=90, alpha=0.75, edgecolors="k")
        ax0.set_title("Offline labels")
        ax0.set_xlabel("Dim 1")
        ax0.set_ylabel("Dim 2")
        fig.colorbar(sc0, ax=ax0, label="label")

        sc1 = ax1.scatter(z2[:, 0], z2[:, 1], c=labels_pred, cmap="Set2", s=90, alpha=0.75, edgecolors="k")
        ax1.set_title(f"{title_prefix}: {result['variant']}\n{subtitle}")
        ax1.set_xlabel("Dim 1")
        ax1.set_ylabel("Dim 2")
        fig.colorbar(sc1, ax=ax1, label="predicted cluster")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        sc = ax.scatter(z2[:, 0], z2[:, 1], c=labels_pred, cmap="Set2", s=90, alpha=0.75, edgecolors="k")
        ax.set_title(f"{title_prefix}: {result['variant']}\n{subtitle}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        fig.colorbar(sc, ax=ax, label="predicted cluster")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def render_partition_metrics_comparison(
    output_path: Path,
    results: list[dict[str, Any]],
) -> bool:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("labels_pred") is None:
            continue
        internal = result.get("internal_metrics", {})
        partition_offline = _extract_partition_offline_metrics(result)
        rows.append(
            {
                "label": f"{result['variant']} [{result['algorithm']}]",
                "silhouette": internal.get("silhouette"),
                "adjusted_rand": partition_offline.get("adjusted_rand"),
                "v_measure": partition_offline.get("v_measure"),
                "normalized_mutual_info": partition_offline.get("normalized_mutual_info"),
            }
        )

    if not rows:
        return False

    rows.sort(
        key=lambda row: (
            float("-inf") if row["adjusted_rand"] is None else float(row["adjusted_rand"]),
            float("-inf") if row["silhouette"] is None else float(row["silhouette"]),
        ),
        reverse=True,
    )

    labels = [row["label"] for row in rows]
    annotation_labels = [label if len(label) <= 56 else f"{label[:53]}..." for label in labels]
    y = np.arange(len(rows))
    metric_specs = [
        ("silhouette", "Silhouette"),
        ("adjusted_rand", "Adjusted Rand Index"),
        ("v_measure", "V-Measure"),
        ("normalized_mutual_info", "Normalized Mutual Info"),
    ]

    fig_height = max(8.0, 0.33 * len(rows) + 2.0)
    fig, axes = plt.subplots(2, 2, figsize=(18, fig_height), sharey=True)
    axes_flat = axes.ravel()

    for idx, (metric_key, metric_name) in enumerate(metric_specs):
        ax = axes_flat[idx]
        values = np.asarray(
            [
                np.nan if row[metric_key] is None else float(row[metric_key])
                for row in rows
            ],
            dtype=np.float64,
        )
        valid = ~np.isnan(values)

        if np.any(valid):
            ax.barh(y[valid], values[valid], color="#1f77b4", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_title(metric_name)

        if metric_key == "silhouette":
            xmin, xmax = _metric_axis_limits(values, -0.2, 1.0)
        else:
            xmin, xmax = _metric_axis_limits(values, 0.0, 1.0)
        xmin = xmin - 0.05
        xmax = xmax + 0.12
        ax.set_xlim(xmin, xmax)
        span = max(1e-9, xmax - xmin)

        for row_idx in np.where(valid)[0]:
            val = float(values[row_idx])
            x_text = val + (0.02 * span if val >= 0 else -0.02 * span)
            ax.text(
                x_text,
                y[row_idx],
                annotation_labels[row_idx],
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=6,
                alpha=0.75,
                clip_on=False,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.invert_yaxis()

    fig.suptitle("Metrics Comparison (Partition)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def render_score_metrics_comparison(
    output_path: Path,
    results: list[dict[str, Any]],
) -> bool:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("scores") is None:
            continue
        score_offline = _extract_score_offline_metrics(result)
        rows.append(
            {
                "label": f"{result['variant']} [{result['algorithm']}]",
                "auroc": score_offline.get("auroc"),
                "auprc": score_offline.get("auprc"),
                "precision_at_num_positives": score_offline.get("precision_at_num_positives"),
                "precision_at_5": score_offline.get("precision_at_5"),
                "precision_at_10": score_offline.get("precision_at_10"),
            }
        )

    if not rows:
        return False

    metric_specs = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("precision_at_num_positives", "Precision@NumPositives"),
        ("precision_at_5", "Precision@5"),
        ("precision_at_10", "Precision@10"),
    ]
    has_any_metric = any(
        row[metric_key] is not None for row in rows for metric_key, _ in metric_specs
    )
    if not has_any_metric:
        return False

    rows.sort(
        key=lambda row: (
            float("-inf") if row["auroc"] is None else float(row["auroc"]),
            float("-inf") if row["auprc"] is None else float(row["auprc"]),
        ),
        reverse=True,
    )

    labels = [row["label"] for row in rows]
    annotation_labels = [label if len(label) <= 56 else f"{label[:53]}..." for label in labels]
    y = np.arange(len(rows))

    fig_height = max(8.0, 0.33 * len(rows) + 2.0)
    fig, axes = plt.subplots(3, 2, figsize=(18, fig_height), sharey=True)
    axes_flat = axes.ravel()

    for idx, (metric_key, metric_name) in enumerate(metric_specs):
        ax = axes_flat[idx]
        values = np.asarray(
            [
                np.nan if row[metric_key] is None else float(row[metric_key])
                for row in rows
            ],
            dtype=np.float64,
        )
        valid = ~np.isnan(values)

        if np.any(valid):
            ax.barh(y[valid], values[valid], color="#1f77b4", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_title(metric_name)

        xmin, xmax = _metric_axis_limits(values, 0.0, 1.0)
        xmin = xmin - 0.05
        xmax = xmax + 0.12
        ax.set_xlim(xmin, xmax)
        span = max(1e-9, xmax - xmin)

        for row_idx in np.where(valid)[0]:
            val = float(values[row_idx])
            x_text = val + (0.02 * span if val >= 0 else -0.02 * span)
            ax.text(
                x_text,
                y[row_idx],
                annotation_labels[row_idx],
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=6,
                alpha=0.75,
                clip_on=False,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.invert_yaxis()

    for idx in range(len(metric_specs), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Metrics Comparison (Score)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()

    z, representation_info, chosen_k = load_representation(
        data_dir=args.data_dir,
        n_components=args.n_components,
        feature_file=args.feature_file,
    )

    labels_true = None
    labels_path = args.data_dir / "labels.npy"
    if labels_path.exists():
        labels_true = np.load(labels_path)

    model_names: list[str]
    names_path = args.data_dir / "model_names.json"
    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as f:
            model_names = json.load(f)
    else:
        model_names = [f"sample_{i}" for i in range(z.shape[0])]

    if len(model_names) != z.shape[0]:
        raise ValueError(
            f"model_names length ({len(model_names)}) does not match feature rows ({z.shape[0]})"
        )

    if args.use_offline_label_metrics:
        if labels_true is None:
            raise ValueError("--use-offline-label-metrics requested but labels.npy was not found")
        if labels_true.shape[0] != z.shape[0]:
            raise ValueError(
                f"labels length ({labels_true.shape[0]}) does not match feature rows ({z.shape[0]})"
            )

    k_list, k_warnings = sanitize_k_list(args.k_list, z.shape[0])
    gmm_components, gmm_warnings = sanitize_gmm_components(args.gmm_components, z.shape[0])

    print("=" * 80)
    print("PHASE 2: Unsupervised Baselines in Embedding Space")
    print("=" * 80)
    print(f"Feature shape: {z.shape}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Selection metric: {args.selection_metric}")
    if chosen_k is not None:
        print(f"Loaded SVD embedding with k={chosen_k}")
    if k_warnings or gmm_warnings:
        print("\nGrid warnings:")
        for warning in k_warnings + gmm_warnings:
            print(f"  - {warning}")

    allowed_algorithms = {
        "kmeans",
        "hierarchical",
        "dbscan",
        "gmm",
        "mahalanobis",
        "isolation_forest",
        "lof",
    }
    unknown = [a for a in args.algorithms if a not in allowed_algorithms]
    if unknown:
        raise ValueError(f"Unknown algorithms requested: {unknown}. Allowed: {sorted(allowed_algorithms)}")

    all_results: list[dict[str, Any]] = []

    if "kmeans" in args.algorithms:
        all_results.extend(
            run_kmeans(
                z=z,
                labels_true=labels_true,
                k_list=k_list,
                seeds=args.stability_seeds,
                use_offline_metrics=args.use_offline_label_metrics,
            )
        )

    if "hierarchical" in args.algorithms:
        all_results.extend(
            run_hierarchical(
                z=z,
                labels_true=labels_true,
                k_list=k_list,
                use_offline_metrics=args.use_offline_label_metrics,
            )
        )

    if "dbscan" in args.algorithms:
        all_results.extend(
            run_dbscan(
                z=z,
                labels_true=labels_true,
                eps_list=args.eps_list,
                min_samples=args.min_samples,
                use_offline_metrics=args.use_offline_label_metrics,
            )
        )

    if "gmm" in args.algorithms:
        all_results.extend(
            run_gmm(
                z=z,
                labels_true=labels_true,
                components_list=gmm_components,
                covariance_types=args.gmm_covariance_types,
                seeds=args.stability_seeds,
                use_offline_metrics=args.use_offline_label_metrics,
                percentiles=args.score_percentiles,
            )
        )

    if "mahalanobis" in args.algorithms:
        all_results.extend(
            run_mahalanobis(
                z=z,
                labels_true=labels_true,
                use_offline_metrics=args.use_offline_label_metrics,
                percentiles=args.score_percentiles,
            )
        )

    if "isolation_forest" in args.algorithms:
        all_results.extend(
            run_isolation_forest(
                z=z,
                labels_true=labels_true,
                seeds=args.stability_seeds,
                use_offline_metrics=args.use_offline_label_metrics,
                percentiles=args.score_percentiles,
            )
        )

    if "lof" in args.algorithms:
        all_results.extend(
            run_lof(
                z=z,
                labels_true=labels_true,
                min_samples=args.min_samples,
                use_offline_metrics=args.use_offline_label_metrics,
                percentiles=args.score_percentiles,
            )
        )

    selection = pick_unsupervised_winner(all_results, args.selection_metric)
    stability_summary = summarize_stability(all_results)
    offline_summary = summarize_offline_eval(all_results) if args.use_offline_label_metrics else {}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Remove deprecated winner plot to avoid stale artifacts.
    legacy_winner_plot_path = args.output_dir / "winner_2d.png"
    if legacy_winner_plot_path.exists():
        legacy_winner_plot_path.unlink()

    best_score_plot_path = args.output_dir / "best_score_model_by_auroc_2d.png"
    best_partition_plot_path = args.output_dir / "best_partition_model_by_ari_2d.png"
    best_silhouette_partition_plot_path = args.output_dir / "best_partition_model_by_silhouette_2d.png"
    metrics_comparison_partition_path = args.output_dir / "metrics_comparison_partition.png"
    metrics_comparison_scores_path = args.output_dir / "metrics_comparison_scores.png"
    legacy_metrics_comparison_path = args.output_dir / "metrics_comparison.png"
    for stale_path in [
        best_score_plot_path,
        best_partition_plot_path,
        best_silhouette_partition_plot_path,
        metrics_comparison_partition_path,
        metrics_comparison_scores_path,
        legacy_metrics_comparison_path,
    ]:
        if stale_path.exists():
            stale_path.unlink()

    # Plot unsupervised winner for silhouette selection when it is a partition-capable model.
    selection_winner_result = None
    if isinstance(selection, dict):
        selection_winner_result = _find_result_by_summary_item(all_results, selection.get("winner"))
    if (
        selection is not None
        and selection.get("metric") == "silhouette"
        and selection_winner_result is not None
        and selection_winner_result.get("labels_pred") is not None
    ):
        render_partition_projection_plot(
            output_path=best_silhouette_partition_plot_path,
            z=z,
            labels_true=labels_true if args.use_offline_label_metrics else None,
            result=selection_winner_result,
            title_prefix="Best partition model by silhouette",
        )

    if args.use_offline_label_metrics:
        best_score_result = _find_result_by_summary_item(
            all_results,
            offline_summary.get("best_score_model_by_auroc"),
        )
        if best_score_result is not None and best_score_result.get("labels_pred") is not None:
            render_partition_projection_plot(
                output_path=best_score_plot_path,
                z=z,
                labels_true=labels_true,
                result=best_score_result,
                title_prefix="Best score model by AUROC",
            )

        best_partition_result = _find_result_by_summary_item(
            all_results,
            offline_summary.get("best_partition_model_by_ari"),
        )
        if best_partition_result is not None and best_partition_result.get("labels_pred") is not None:
            render_partition_projection_plot(
                output_path=best_partition_plot_path,
                z=z,
                labels_true=labels_true,
                result=best_partition_result,
                title_prefix="Best partition model by ARI",
            )

        render_partition_metrics_comparison(
            output_path=metrics_comparison_partition_path,
            results=all_results,
        )
        render_score_metrics_comparison(
            output_path=metrics_comparison_scores_path,
            results=all_results,
        )

    scores_csv_path = args.output_dir / "model_suspicion_scores.csv"
    save_score_table(scores_csv_path, model_names, all_results)

    run_config = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "resolved_k_list": k_list,
        "resolved_gmm_components": gmm_components,
        "warnings": k_warnings + gmm_warnings,
    }

    report = {
        "data_info": {
            "n_samples": int(z.shape[0]),
            "n_features": int(z.shape[1]),
            "n_clean_samples": int(np.sum(labels_true == 0)) if labels_true is not None else None,
            "n_backdoored_samples": int(np.sum(labels_true == 1)) if labels_true is not None else None,
            "model_names": model_names,
        },
        "representation_info": representation_info,
        "unsupervised_selection": selection,
        "offline_eval": offline_summary,
        "stability": stability_summary,
        "artifacts": {
            "best_score_model_by_auroc_plot": (
                str(best_score_plot_path) if best_score_plot_path.exists() else None
            ),
            "best_partition_model_by_ari_plot": (
                str(best_partition_plot_path) if best_partition_plot_path.exists() else None
            ),
            "best_partition_model_by_silhouette_plot": (
                str(best_silhouette_partition_plot_path)
                if best_silhouette_partition_plot_path.exists()
                else None
            ),
            "metrics_comparison_partition": (
                str(metrics_comparison_partition_path)
                if metrics_comparison_partition_path.exists()
                else None
            ),
            "metrics_comparison_scores": (
                str(metrics_comparison_scores_path)
                if metrics_comparison_scores_path.exists()
                else None
            ),
            "score_table": str(scores_csv_path) if scores_csv_path.exists() else None,
            "run_config": str(args.output_dir / "run_config.json"),
            "report": str(args.output_dir / "clustering_report.json"),
        },
        "algorithm_results": all_results,
    }

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(run_config), f, indent=2)

    with open(args.output_dir / "clustering_report.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    print("\n" + "=" * 80)
    print("Unsupervised baseline summary")
    print("=" * 80)

    if selection is None:
        print("No unsupervised winner could be selected for the requested metric")
    else:
        w = selection["winner"]
        print(
            f"Winner ({selection['metric']}): {w['variant']} "
            f"[{w['algorithm']}] -> {format_metric(w['selection_value'])}"
        )

    if args.use_offline_label_metrics:
        best_score = offline_summary.get("best_score_model_by_auroc")
        best_part = offline_summary.get("best_partition_model_by_ari")
        if best_score is not None:
            print(
                f"Best score model by AUROC: {best_score['variant']} "
                f"(AUROC={format_metric(best_score.get('auroc'))})"
            )
        if best_part is not None:
            print(
                f"Best partition model by ARI: {best_part['variant']} "
                f"(ARI={format_metric(best_part.get('adjusted_rand'))})"
            )

    print(f"\nSaved report to {args.output_dir / 'clustering_report.json'}")


if __name__ == "__main__":
    main()
