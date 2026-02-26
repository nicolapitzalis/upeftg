from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from ..clustering.metrics import sanitize_gmm_components
from ..features.svd import fit_dual_svd_basis_from_items, project_items_with_dual_basis_streamed
from ..utilities.hashing import compute_dataset_signature, compute_feature_cache_key
from ..utilities.manifest import parse_joint_manifest_json
from ..utilities.run_context import create_run_context
from ..utilities.serialization import json_ready
from .reporting import (
    compute_infer_threshold_rows,
    compute_offline_metrics,
    save_score_csv,
    summarize_scores,
)


SCRIPT_VERSION = "1.0.0"


def _svd_cache_key(
    *,
    train_names: list[str],
    infer_names: list[str],
    component_grid: list[int],
    block_size: int,
    dtype: str,
) -> str:
    signature = compute_dataset_signature(
        model_names=train_names + ["--infer--"] + infer_names,
        extra={
            "train_count": len(train_names),
            "infer_count": len(infer_names),
        },
    )
    return compute_feature_cache_key(
        dataset_signature=signature,
        extractor_name="svd_train_infer",
        extractor_params={
            "component_grid": component_grid,
            "block_size": block_size,
        },
        extractor_version=SCRIPT_VERSION,
        dtype=dtype,
    )


def run_gmm_train_inference_pipeline(
    *,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    svd_components_grid: list[int],
    gmm_components: list[int],
    gmm_covariance_types: list[str],
    stability_seeds: list[int],
    score_percentiles: list[float],
    stream_block_size: int,
    dtype_name: str,
    reg_covar: float,
    n_init: int,
    force_recompute_features: bool,
) -> dict[str, Any]:
    dtype = np.float32 if dtype_name == "float32" else np.float64

    if stream_block_size <= 0:
        raise ValueError(f"stream_block_size must be positive, got {stream_block_size}")
    if n_init <= 0:
        raise ValueError(f"n_init must be positive, got {n_init}")
    if reg_covar < 0.0:
        raise ValueError(f"reg_covar must be >= 0, got {reg_covar}")

    ctx = create_run_context(
        pipeline="gmm_train_inference",
        output_root=output_root,
        run_id=run_id,
    )

    train_items, infer_items = parse_joint_manifest_json(
        manifest_path=manifest_json,
        dataset_root=dataset_root,
    )

    warnings: list[str] = []

    train_labels_list = [item.label for item in train_items]
    infer_labels_list = [item.label for item in infer_items]

    n_train_clean = int(np.sum([label == 0 for label in train_labels_list]))
    n_train_backdoored = int(np.sum([label == 1 for label in train_labels_list]))
    n_infer_unknown = int(np.sum([label is None for label in infer_labels_list]))
    if n_infer_unknown > 0:
        warnings.append(
            f"{n_infer_unknown} inference entries do not match label naming convention; label-based metrics will be omitted"
        )

    train_names = [item.model_name for item in train_items]
    infer_names = [item.model_name for item in infer_items]

    svd_cache_key = _svd_cache_key(
        train_names=train_names,
        infer_names=infer_names,
        component_grid=svd_components_grid,
        block_size=stream_block_size,
        dtype=dtype_name,
    )
    svd_cache_dir = ctx.cache_root / svd_cache_key
    svd_cache_hit = svd_cache_dir.exists() and (svd_cache_dir / "z_train_max.npy").exists()

    if svd_cache_hit and not force_recompute_features:
        z_train_max = np.load(svd_cache_dir / "z_train_max.npy")
        z_infer_max = np.load(svd_cache_dir / "z_infer_max.npy")
        with open(svd_cache_dir / "metadata.json", "r", encoding="utf-8") as f:
            svd_meta = json.load(f)
        resolved_component_grid = [int(x) for x in svd_meta["resolved_component_grid"]]
        max_rank = int(svd_meta["max_rank"])
        layers = [int(x) for x in svd_meta["layers"]]
        n_params = int(svd_meta["n_features_raw"])
        component_warnings = [str(x) for x in svd_meta.get("warnings", [])]
    else:
        basis, component_warnings = fit_dual_svd_basis_from_items(
            items=train_items,
            component_grid=svd_components_grid,
            block_size=stream_block_size,
            dtype=dtype,
        )

        resolved_component_grid = basis.resolved_component_grid
        max_rank = basis.max_rank
        layers = basis.layers
        n_params = basis.n_features

        rank = max(resolved_component_grid)
        z_train_max = basis.z_train_max[:, :rank]
        z_infer_max = project_items_with_dual_basis_streamed(
            train_items=train_items,
            target_items=infer_items,
            basis=basis,
            rank=rank,
            block_size=stream_block_size,
            dtype=dtype,
        )

        svd_cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(svd_cache_dir / "z_train_max.npy", z_train_max)
        np.save(svd_cache_dir / "z_infer_max.npy", z_infer_max)
        with open(svd_cache_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                json_ready(
                    {
                        "resolved_component_grid": resolved_component_grid,
                        "max_rank": max_rank,
                        "layers": layers,
                        "n_features_raw": n_params,
                        "warnings": component_warnings,
                    }
                ),
                f,
                indent=2,
            )

    warnings.extend(component_warnings)

    np.save(ctx.features_dir / "z_train_max.npy", z_train_max)
    np.save(ctx.features_dir / "z_infer_max.npy", z_infer_max)
    ctx.add_artifact("z_train_max", ctx.features_dir / "z_train_max.npy")
    ctx.add_artifact("z_infer_max", ctx.features_dir / "z_infer_max.npy")

    resolved_gmm_components, gmm_warnings = sanitize_gmm_components(gmm_components, len(train_items))
    warnings.extend(gmm_warnings)

    covariance_order = {name: i for i, name in enumerate(gmm_covariance_types)}
    candidate_rows: list[dict[str, Any]] = []
    winner_tuple: tuple[float, float, int, int] | None = None
    winner_candidate: dict[str, Any] | None = None

    for k in resolved_component_grid:
        z_train_k = z_train_max[:, :k]
        for cov in gmm_covariance_types:
            if cov not in {"full", "tied", "diag", "spherical"}:
                warnings.append(f"Skipping unknown covariance type: {cov}")
                continue
            for n_components in resolved_gmm_components:
                run_rows: list[dict[str, Any]] = []
                bics: list[float] = []
                aics: list[float] = []
                for seed in stability_seeds:
                    try:
                        gmm = GaussianMixture(
                            n_components=n_components,
                            covariance_type=cov,
                            reg_covar=reg_covar,
                            n_init=n_init,
                            random_state=seed,
                        )
                        gmm.fit(z_train_k)
                        bic = float(gmm.bic(z_train_k))
                        aic = float(gmm.aic(z_train_k))
                        bics.append(bic)
                        aics.append(aic)
                        run_rows.append(
                            {
                                "seed": int(seed),
                                "bic": bic,
                                "aic": aic,
                                "converged": bool(getattr(gmm, "converged_", False)),
                                "n_iter": int(getattr(gmm, "n_iter_", 0)),
                            }
                        )
                    except Exception as exc:
                        run_rows.append(
                            {
                                "seed": int(seed),
                                "error": str(exc),
                            }
                        )

                if not bics:
                    candidate_rows.append(
                        {
                            "k": int(k),
                            "n_components": int(n_components),
                            "covariance_type": cov,
                            "successful_runs": 0,
                            "total_runs": len(stability_seeds),
                            "mean_bic": None,
                            "std_bic": None,
                            "mean_aic": None,
                            "run_details": run_rows,
                        }
                    )
                    continue

                mean_bic = float(np.mean(bics))
                std_bic = float(np.std(bics))
                mean_aic = float(np.mean(aics))
                candidate = {
                    "k": int(k),
                    "n_components": int(n_components),
                    "covariance_type": cov,
                    "successful_runs": len(bics),
                    "total_runs": len(stability_seeds),
                    "mean_bic": mean_bic,
                    "std_bic": std_bic,
                    "mean_aic": mean_aic,
                    "run_details": run_rows,
                }
                candidate_rows.append(candidate)

                tie_key = (
                    mean_bic,
                    std_bic,
                    int(n_components),
                    covariance_order.get(cov, int(1e9)),
                )
                if winner_tuple is None or tie_key < winner_tuple:
                    winner_tuple = tie_key
                    winner_candidate = candidate

    if winner_candidate is None:
        raise RuntimeError("No valid GMM candidate could be fit from the requested grid")

    k_star = int(winner_candidate["k"])
    n_star = int(winner_candidate["n_components"])
    cov_star = str(winner_candidate["covariance_type"])
    z_train_star = z_train_max[:, :k_star]
    z_infer_star = z_infer_max[:, :k_star]

    np.save(ctx.features_dir / "z_train_star.npy", z_train_star)
    np.save(ctx.features_dir / "z_infer_star.npy", z_infer_star)
    ctx.add_artifact("z_train_star", ctx.features_dir / "z_train_star.npy")
    ctx.add_artifact("z_infer_star", ctx.features_dir / "z_infer_star.npy")

    best_model: GaussianMixture | None = None
    best_seed = None
    best_bic = None
    winner_seed_rows: list[dict[str, Any]] = []

    for seed in stability_seeds:
        try:
            gmm = GaussianMixture(
                n_components=n_star,
                covariance_type=cov_star,
                reg_covar=reg_covar,
                n_init=n_init,
                random_state=seed,
            )
            gmm.fit(z_train_star)
            bic = float(gmm.bic(z_train_star))
            winner_seed_rows.append(
                {
                    "seed": int(seed),
                    "bic": bic,
                    "aic": float(gmm.aic(z_train_star)),
                    "converged": bool(getattr(gmm, "converged_", False)),
                    "n_iter": int(getattr(gmm, "n_iter_", 0)),
                }
            )
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_model = gmm
                best_seed = int(seed)
        except Exception as exc:
            winner_seed_rows.append({"seed": int(seed), "error": str(exc)})

    if best_model is None:
        raise RuntimeError("Winner config selected but refit failed for all seeds")

    train_scores = -best_model.score_samples(z_train_star)
    infer_scores = -best_model.score_samples(z_infer_star)

    train_clean_scores = np.asarray(
        [score for score, label in zip(train_scores, train_labels_list) if label == 0],
        dtype=np.float64,
    )
    train_backdoor_scores = np.asarray(
        [score for score, label in zip(train_scores, train_labels_list) if label == 1],
        dtype=np.float64,
    )

    infer_known_mask = np.asarray([label is not None for label in infer_labels_list], dtype=bool)
    infer_labels_np: np.ndarray | None = None
    if np.all(infer_known_mask):
        infer_labels_np = np.asarray(infer_labels_list, dtype=np.int32)

    threshold_rows = compute_infer_threshold_rows(
        train_scores=train_scores,
        infer_scores=infer_scores,
        percentiles=score_percentiles,
        infer_labels=infer_labels_np,
    )
    offline_metrics = compute_offline_metrics(infer_labels_np, infer_scores)

    infer_clean_scores = np.asarray(
        [score for score, label in zip(infer_scores, infer_labels_list) if label == 0],
        dtype=np.float64,
    )
    infer_backdoor_scores = np.asarray(
        [score for score, label in zip(infer_scores, infer_labels_list) if label == 1],
        dtype=np.float64,
    )

    candidate_rows.sort(
        key=lambda row: (
            float("inf") if row.get("mean_bic") is None else float(row["mean_bic"]),
            float("inf") if row.get("std_bic") is None else float(row["std_bic"]),
            int(row["n_components"]),
            covariance_order.get(str(row["covariance_type"]), int(1e9)),
        )
    )

    report = {
        "data_info": {
            "dataset_root": str(dataset_root),
            "n_train": len(train_items),
            "n_train_clean": n_train_clean,
            "n_train_backdoored": n_train_backdoored,
            "n_train_unknown_label": int(np.sum([label is None for label in train_labels_list])),
            "n_inference": len(infer_items),
            "n_inference_clean": int(np.sum([label == 0 for label in infer_labels_list])),
            "n_inference_backdoored": int(np.sum([label == 1 for label in infer_labels_list])),
            "n_inference_unknown_label": int(np.sum([label is None for label in infer_labels_list])),
            "train_model_names": train_names,
            "inference_model_names": infer_names,
        },
        "representation": {
            "n_features_raw": int(n_params),
            "layers": layers,
            "svd_components_grid_requested": svd_components_grid,
            "svd_components_grid_resolved": resolved_component_grid,
            "chosen_k": k_star,
            "max_available_rank": int(max_rank),
            "svd_cache_key": svd_cache_key,
            "svd_cache_hit": bool(svd_cache_hit and not force_recompute_features),
        },
        "gmm_selection": {
            "criterion": "mean_bic_on_train",
            "tie_breaker": [
                "lower_bic_std",
                "lower_n_components",
                "covariance_order_as_passed",
            ],
            "candidates": candidate_rows,
            "winner": {
                "k": k_star,
                "n_components": n_star,
                "covariance_type": cov_star,
                "mean_bic": winner_candidate["mean_bic"],
                "std_bic": winner_candidate["std_bic"],
                "selected_seed": best_seed,
                "selected_seed_bic": best_bic,
                "seed_runs": winner_seed_rows,
            },
        },
        "fit_assessment": {
            "score_definition": "negative_log_likelihood",
            "train_score_summary": summarize_scores(train_scores),
            "train_clean_score_summary": (
                summarize_scores(train_clean_scores) if train_clean_scores.size > 0 else None
            ),
            "train_backdoor_score_summary": (
                summarize_scores(train_backdoor_scores) if train_backdoor_scores.size > 0 else None
            ),
            "inference_score_summary": summarize_scores(infer_scores),
            "inference_clean_score_summary": (
                summarize_scores(infer_clean_scores) if infer_clean_scores.size > 0 else None
            ),
            "inference_backdoor_score_summary": (
                summarize_scores(infer_backdoor_scores) if infer_backdoor_scores.size > 0 else None
            ),
            "threshold_evaluation": threshold_rows,
            "offline_metrics": offline_metrics,
        },
        "warnings": warnings,
    }

    train_scores_csv = ctx.reports_dir / "train_scores.csv"
    train_clean_scores_csv = ctx.reports_dir / "train_clean_scores.csv"
    inference_scores_csv = ctx.reports_dir / "inference_scores.csv"

    save_score_csv(
        output_path=train_scores_csv,
        model_names=train_names,
        labels=train_labels_list,
        scores=train_scores,
    )
    save_score_csv(
        output_path=train_clean_scores_csv,
        model_names=train_names,
        labels=train_labels_list,
        scores=train_scores,
    )
    save_score_csv(
        output_path=inference_scores_csv,
        model_names=infer_names,
        labels=infer_labels_list,
        scores=infer_scores,
    )

    report_path = ctx.reports_dir / "gmm_train_inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("train_scores_csv", train_scores_csv)
    ctx.add_artifact("train_clean_scores_csv", train_clean_scores_csv)
    ctx.add_artifact("inference_scores_csv", inference_scores_csv)
    ctx.add_artifact("report", report_path)

    run_config = {
        "pipeline": "gmm_train_inference",
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": str(manifest_json),
        "dataset_root": str(dataset_root),
        "svd_components_grid": svd_components_grid,
        "resolved_component_grid": resolved_component_grid,
        "gmm_components": gmm_components,
        "resolved_gmm_components": resolved_gmm_components,
        "gmm_covariance_types": gmm_covariance_types,
        "stability_seeds": stability_seeds,
        "score_percentiles": score_percentiles,
        "stream_block_size": stream_block_size,
        "dtype": dtype_name,
        "reg_covar": reg_covar,
        "n_init": n_init,
        "svd_cache_key": svd_cache_key,
        "svd_cache_hit": bool(svd_cache_hit and not force_recompute_features),
        "warnings": warnings,
    }
    ctx.finalize(run_config)

    return {
        "run_dir": str(ctx.run_dir),
        "report": str(report_path),
        "train_scores_csv": str(train_scores_csv),
        "inference_scores_csv": str(inference_scores_csv),
    }
