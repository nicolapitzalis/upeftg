from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from .metrics import (
    compute_cluster_sizes,
    compute_partition_internal_metrics,
    compute_partition_offline_metrics,
    compute_score_offline_metrics,
    pairwise_mean_ari,
    pairwise_mean_spearman,
    score_percentile_stats,
)


def run_kmeans(
    *,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    k_list: list[int],
    seeds: list[int],
    use_offline_metrics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for k in k_list:
        label_runs: list[np.ndarray] = []
        for seed in seeds:
            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels_pred = model.fit_predict(z)
            label_runs.append(labels_pred)

        labels_best = label_runs[0]
        internal = compute_partition_internal_metrics(z, labels_best)
        offline = compute_partition_offline_metrics(labels_true, labels_best) if use_offline_metrics else None
        stability = pairwise_mean_ari(label_runs)

        results.append(
            {
                "algorithm": "kmeans",
                "variant": f"kmeans_k{k}",
                "is_partition_model": True,
                "is_score_model": False,
                "labels_pred": labels_best,
                "scores": None,
                "selection_value": internal.get("silhouette"),
                "stability_score": stability,
                "cluster_sizes": compute_cluster_sizes(labels_best),
                "partition_internal_metrics": internal,
                "partition_offline_metrics": offline,
                "score_offline_metrics": None,
                "score_percentiles": None,
            }
        )
    return results


def run_hierarchical(
    *,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    k_list: list[int],
    use_offline_metrics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for k in k_list:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_pred = model.fit_predict(z)

        internal = compute_partition_internal_metrics(z, labels_pred)
        offline = compute_partition_offline_metrics(labels_true, labels_pred) if use_offline_metrics else None

        results.append(
            {
                "algorithm": "hierarchical",
                "variant": f"hierarchical_k{k}",
                "is_partition_model": True,
                "is_score_model": False,
                "labels_pred": labels_pred,
                "scores": None,
                "selection_value": internal.get("silhouette"),
                "stability_score": None,
                "cluster_sizes": compute_cluster_sizes(labels_pred),
                "partition_internal_metrics": internal,
                "partition_offline_metrics": offline,
                "score_offline_metrics": None,
                "score_percentiles": None,
            }
        )
    return results


def run_dbscan(
    *,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    eps_list: list[float],
    min_samples: int,
    use_offline_metrics: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for eps in eps_list:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels_pred = model.fit_predict(z)

        internal = compute_partition_internal_metrics(z, labels_pred)
        offline = compute_partition_offline_metrics(labels_true, labels_pred) if use_offline_metrics else None

        results.append(
            {
                "algorithm": "dbscan",
                "variant": f"dbscan_eps{eps:g}_min{min_samples}",
                "is_partition_model": True,
                "is_score_model": False,
                "labels_pred": labels_pred,
                "scores": None,
                "selection_value": internal.get("silhouette"),
                "stability_score": None,
                "cluster_sizes": compute_cluster_sizes(labels_pred),
                "partition_internal_metrics": internal,
                "partition_offline_metrics": offline,
                "score_offline_metrics": None,
                "score_percentiles": None,
            }
        )
    return results


def run_gmm(
    *,
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
        for cov_type in covariance_types:
            run_rows: list[dict[str, Any]] = []
            labels_runs: list[np.ndarray] = []
            scores_runs: list[np.ndarray] = []
            bics: list[float] = []

            for seed in seeds:
                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=cov_type,
                        random_state=seed,
                        n_init=1,
                    )
                    gmm.fit(z)
                    labels_pred = gmm.predict(z)
                    scores = -gmm.score_samples(z)
                    bic = float(gmm.bic(z))

                    run_rows.append(
                        {
                            "seed": int(seed),
                            "bic": bic,
                            "converged": bool(getattr(gmm, "converged_", False)),
                            "n_iter": int(getattr(gmm, "n_iter_", 0)),
                        }
                    )
                    labels_runs.append(labels_pred)
                    scores_runs.append(scores)
                    bics.append(bic)
                except Exception as exc:
                    run_rows.append({"seed": int(seed), "error": str(exc)})

            if not scores_runs:
                results.append(
                    {
                        "algorithm": "gmm",
                        "variant": f"gmm_{cov_type}_c{n_components}",
                        "is_partition_model": True,
                        "is_score_model": True,
                        "labels_pred": None,
                        "scores": None,
                        "selection_value": None,
                        "stability_score": None,
                        "cluster_sizes": None,
                        "partition_internal_metrics": None,
                        "partition_offline_metrics": None,
                        "score_offline_metrics": None,
                        "score_percentiles": None,
                        "bic_mean": None,
                        "bic_std": None,
                        "seed_runs": run_rows,
                    }
                )
                continue

            labels_best = labels_runs[0]
            scores_best = scores_runs[0]

            internal = compute_partition_internal_metrics(z, labels_best)
            partition_offline = compute_partition_offline_metrics(labels_true, labels_best) if use_offline_metrics else None
            score_offline = compute_score_offline_metrics(labels_true, scores_best) if use_offline_metrics else None

            stability = pairwise_mean_ari(labels_runs)
            score_stability = pairwise_mean_spearman(scores_runs)

            results.append(
                {
                    "algorithm": "gmm",
                    "variant": f"gmm_{cov_type}_c{n_components}",
                    "is_partition_model": True,
                    "is_score_model": True,
                    "labels_pred": labels_best,
                    "scores": scores_best,
                    "selection_value": float(np.mean(bics)),
                    "stability_score": stability,
                    "score_stability": score_stability,
                    "cluster_sizes": compute_cluster_sizes(labels_best),
                    "partition_internal_metrics": internal,
                    "partition_offline_metrics": partition_offline,
                    "score_offline_metrics": score_offline,
                    "score_percentiles": score_percentile_stats(scores_best, percentiles),
                    "bic_mean": float(np.mean(bics)),
                    "bic_std": float(np.std(bics)),
                    "seed_runs": run_rows,
                }
            )

    return results


def run_mahalanobis(
    *,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    use_offline_metrics: bool,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    cov = EmpiricalCovariance().fit(z)
    scores = cov.mahalanobis(z)

    score_offline = compute_score_offline_metrics(labels_true, scores) if use_offline_metrics else None

    return [
        {
            "algorithm": "mahalanobis",
            "variant": "mahalanobis_empirical_cov",
            "is_partition_model": False,
            "is_score_model": True,
            "labels_pred": None,
            "scores": scores,
            "selection_value": None,
            "stability_score": None,
            "cluster_sizes": None,
            "partition_internal_metrics": None,
            "partition_offline_metrics": None,
            "score_offline_metrics": score_offline,
            "score_percentiles": score_percentile_stats(scores, percentiles),
        }
    ]


def run_isolation_forest(
    *,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    seeds: list[int],
    use_offline_metrics: bool,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    scores_runs: list[np.ndarray] = []
    run_rows: list[dict[str, Any]] = []

    for seed in seeds:
        model = IsolationForest(random_state=seed)
        model.fit(z)
        scores = -model.score_samples(z)
        scores_runs.append(scores)
        run_rows.append({"seed": int(seed)})

    scores_best = scores_runs[0]
    score_offline = compute_score_offline_metrics(labels_true, scores_best) if use_offline_metrics else None

    return [
        {
            "algorithm": "isolation_forest",
            "variant": "isolation_forest_default",
            "is_partition_model": False,
            "is_score_model": True,
            "labels_pred": None,
            "scores": scores_best,
            "selection_value": None,
            "stability_score": pairwise_mean_spearman(scores_runs),
            "cluster_sizes": None,
            "partition_internal_metrics": None,
            "partition_offline_metrics": None,
            "score_offline_metrics": score_offline,
            "score_percentiles": score_percentile_stats(scores_best, percentiles),
            "seed_runs": run_rows,
        }
    ]


def run_lof(
    *,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    min_samples: int,
    use_offline_metrics: bool,
    percentiles: list[float],
) -> list[dict[str, Any]]:
    n_neighbors = max(2, min(min_samples, z.shape[0] - 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    _ = model.fit_predict(z)
    scores = -model.negative_outlier_factor_

    score_offline = compute_score_offline_metrics(labels_true, scores) if use_offline_metrics else None

    return [
        {
            "algorithm": "lof",
            "variant": f"lof_n{n_neighbors}",
            "is_partition_model": False,
            "is_score_model": True,
            "labels_pred": None,
            "scores": scores,
            "selection_value": None,
            "stability_score": None,
            "cluster_sizes": None,
            "partition_internal_metrics": None,
            "partition_offline_metrics": None,
            "score_offline_metrics": score_offline,
            "score_percentiles": score_percentile_stats(scores, percentiles),
        }
    ]
