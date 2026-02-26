from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import (
    run_dbscan,
    run_gmm,
    run_hierarchical,
    run_isolation_forest,
    run_kmeans,
    run_lof,
    run_mahalanobis,
)
from .metrics import (
    pick_unsupervised_winner,
    sanitize_gmm_components,
    sanitize_k_list,
    summarize_offline_eval,
    summarize_stability,
)
from .reporting import (
    _find_result_by_summary_item,
    render_partition_metrics_comparison,
    render_partition_projection_plot,
    render_score_metrics_comparison,
    save_score_table,
)
from ..features.registry import extract_with_cache
from ..utilities.manifest import parse_single_manifest_json
from ..utilities.run_context import create_run_context
from ..utilities.serialization import json_ready


SCRIPT_VERSION = "1.0.0"


def run_clustering_pipeline(
    *,
    manifest_json: Path | None,
    dataset_root: Path,
    feature_file: Path | None,
    extractor_name: str,
    extractor_params: dict[str, Any],
    output_root: Path,
    run_id: str | None,
    algorithms: list[str],
    k_list: list[int],
    eps_list: list[float],
    min_samples: int,
    gmm_components: list[int],
    gmm_covariance_types: list[str],
    selection_metric: str,
    use_offline_label_metrics: bool,
    stability_seeds: list[int],
    score_percentiles: list[float],
    force_recompute_features: bool,
) -> dict[str, Any]:
    ctx = create_run_context(
        pipeline="clustering",
        output_root=output_root,
        run_id=run_id,
    )

    warnings: list[str] = []
    extractor_warnings: list[str] = []

    if feature_file is not None:
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        z = np.load(feature_file)

        labels = None
        model_names: list[str]
        labels_path = feature_file.parent / "labels.npy"
        names_path = feature_file.parent / "model_names.json"

        if labels_path.exists():
            labels = np.load(labels_path)
        if names_path.exists():
            with open(names_path, "r", encoding="utf-8") as f:
                model_names = json.load(f)
        else:
            model_names = [f"sample_{i}" for i in range(z.shape[0])]

        representation_info: dict[str, Any] = {
            "source": str(feature_file),
            "type": "external_feature_file",
            "shape": [int(z.shape[0]), int(z.shape[1])],
            "n_components": int(z.shape[1]),
            "cache_hit": False,
        }

        run_feature_path = ctx.features_dir / "external_features.npy"
        np.save(run_feature_path, z)
        ctx.add_artifact("features", run_feature_path)
        if labels is not None:
            run_labels_path = ctx.features_dir / "external_labels.npy"
            np.save(run_labels_path, labels)
            ctx.add_artifact("labels", run_labels_path)

    else:
        if manifest_json is None:
            raise ValueError("--manifest-json is required unless --feature-file is provided")

        items = parse_single_manifest_json(
            manifest_path=manifest_json,
            dataset_root=dataset_root,
            section_key="path",
        )
        bundle, feature_artifacts, extractor_warnings = extract_with_cache(
            extractor_name=extractor_name,
            items=items,
            params=extractor_params,
            cache_root=ctx.cache_root,
            run_features_dir=ctx.features_dir,
            force_recompute=force_recompute_features,
        )

        z = bundle.features
        labels = bundle.labels
        model_names = bundle.model_names
        representation_info = {
            "source": feature_artifacts["feature_path"],
            "type": "extractor_output",
            "extractor": extractor_name,
            "extractor_metadata": bundle.metadata,
            "cache_hit": feature_artifacts["cache_hit"],
            "cache_key": feature_artifacts["cache_key"],
            "shape": [int(z.shape[0]), int(z.shape[1])],
            "n_components": int(z.shape[1]),
        }

        ctx.add_artifact("features", Path(feature_artifacts["feature_path"]))
        if feature_artifacts.get("labels_path"):
            ctx.add_artifact("labels", Path(str(feature_artifacts["labels_path"])))
        ctx.add_artifact("feature_metadata", Path(feature_artifacts["metadata_path"]))

    if len(model_names) != z.shape[0]:
        raise ValueError(f"model_names length ({len(model_names)}) does not match feature rows ({z.shape[0]})")

    labels_true = labels
    if use_offline_label_metrics:
        if labels_true is None:
            raise ValueError("--use-offline-label-metrics requested but no labels were available")
        if labels_true.shape[0] != z.shape[0]:
            raise ValueError(
                f"labels length ({labels_true.shape[0]}) does not match feature rows ({z.shape[0]})"
            )

    resolved_k_list, k_warnings = sanitize_k_list(k_list, z.shape[0])
    resolved_gmm_components, gmm_warnings = sanitize_gmm_components(gmm_components, z.shape[0])
    warnings.extend(k_warnings + gmm_warnings + extractor_warnings)

    allowed_algorithms = {
        "kmeans",
        "hierarchical",
        "dbscan",
        "gmm",
        "mahalanobis",
        "isolation_forest",
        "lof",
    }
    unknown = [a for a in algorithms if a not in allowed_algorithms]
    if unknown:
        raise ValueError(f"Unknown algorithms requested: {unknown}. Allowed: {sorted(allowed_algorithms)}")

    all_results: list[dict[str, Any]] = []

    if "kmeans" in algorithms:
        all_results.extend(
            run_kmeans(
                z=z,
                labels_true=labels_true,
                k_list=resolved_k_list,
                seeds=stability_seeds,
                use_offline_metrics=use_offline_label_metrics,
            )
        )

    if "hierarchical" in algorithms:
        all_results.extend(
            run_hierarchical(
                z=z,
                labels_true=labels_true,
                k_list=resolved_k_list,
                use_offline_metrics=use_offline_label_metrics,
            )
        )

    if "dbscan" in algorithms:
        all_results.extend(
            run_dbscan(
                z=z,
                labels_true=labels_true,
                eps_list=eps_list,
                min_samples=min_samples,
                use_offline_metrics=use_offline_label_metrics,
            )
        )

    if "gmm" in algorithms:
        all_results.extend(
            run_gmm(
                z=z,
                labels_true=labels_true,
                components_list=resolved_gmm_components,
                covariance_types=gmm_covariance_types,
                seeds=stability_seeds,
                use_offline_metrics=use_offline_label_metrics,
                percentiles=score_percentiles,
            )
        )

    if "mahalanobis" in algorithms:
        all_results.extend(
            run_mahalanobis(
                z=z,
                labels_true=labels_true,
                use_offline_metrics=use_offline_label_metrics,
                percentiles=score_percentiles,
            )
        )

    if "isolation_forest" in algorithms:
        all_results.extend(
            run_isolation_forest(
                z=z,
                labels_true=labels_true,
                seeds=stability_seeds,
                use_offline_metrics=use_offline_label_metrics,
                percentiles=score_percentiles,
            )
        )

    if "lof" in algorithms:
        all_results.extend(
            run_lof(
                z=z,
                labels_true=labels_true,
                min_samples=min_samples,
                use_offline_metrics=use_offline_label_metrics,
                percentiles=score_percentiles,
            )
        )

    selection = pick_unsupervised_winner(all_results, selection_metric)
    stability_summary = summarize_stability(all_results)
    offline_summary = summarize_offline_eval(all_results) if use_offline_label_metrics else {}

    best_score_plot_path = ctx.plots_dir / "best_score_model_by_auroc_2d.png"
    best_partition_plot_path = ctx.plots_dir / "best_partition_model_by_ari_2d.png"
    best_silhouette_partition_plot_path = ctx.plots_dir / "best_partition_model_by_silhouette_2d.png"
    metrics_comparison_partition_path = ctx.plots_dir / "metrics_comparison_partition.png"
    metrics_comparison_scores_path = ctx.plots_dir / "metrics_comparison_scores.png"

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
            labels_true=labels_true if use_offline_label_metrics else None,
            result=selection_winner_result,
            title_prefix="Best partition model by silhouette",
        )
        ctx.add_artifact("best_partition_model_by_silhouette_plot", best_silhouette_partition_plot_path)

    if use_offline_label_metrics:
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
            ctx.add_artifact("best_score_model_by_auroc_plot", best_score_plot_path)

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
            ctx.add_artifact("best_partition_model_by_ari_plot", best_partition_plot_path)

        if render_partition_metrics_comparison(
            output_path=metrics_comparison_partition_path,
            results=all_results,
        ):
            ctx.add_artifact("metrics_comparison_partition", metrics_comparison_partition_path)

        if render_score_metrics_comparison(
            output_path=metrics_comparison_scores_path,
            results=all_results,
        ):
            ctx.add_artifact("metrics_comparison_scores", metrics_comparison_scores_path)

    scores_csv_path = ctx.reports_dir / "model_suspicion_scores.csv"
    save_score_table(scores_csv_path, model_names, all_results)
    if scores_csv_path.exists():
        ctx.add_artifact("score_table", scores_csv_path)

    run_config = {
        "pipeline": "clustering",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": str(manifest_json) if manifest_json is not None else None,
        "dataset_root": str(dataset_root),
        "feature_file": str(feature_file) if feature_file is not None else None,
        "extractor_name": extractor_name,
        "extractor_params": extractor_params,
        "algorithms": algorithms,
        "selection_metric": selection_metric,
        "use_offline_label_metrics": use_offline_label_metrics,
        "resolved_k_list": resolved_k_list,
        "resolved_gmm_components": resolved_gmm_components,
        "warnings": warnings,
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
        "warnings": warnings,
        "algorithm_results": all_results,
    }

    report_path = ctx.reports_dir / "clustering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)
    ctx.add_artifact("report", report_path)

    ctx.finalize(run_config)

    return {
        "run_dir": str(ctx.run_dir),
        "report": str(report_path),
        "score_table": str(scores_csv_path) if scores_csv_path.exists() else None,
    }
