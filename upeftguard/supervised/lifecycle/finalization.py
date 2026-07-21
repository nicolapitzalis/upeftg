from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

from ...utilities.core.serialization import json_ready

from ...reporting import (
    PredictionPartition,
    build_binary_evaluation,
    write_reporting_bundle,
)
from ...utilities.core.manifest import resolve_manifest_path
from ..contracts import BINARY_PROJECTION_POSITIVE_CLASS_SCORE
from ..models.artifact import attach_inference_contract
from .refit import fit_winner_and_predict, select_best_cv_fold_model_and_predict
from ...utilities.core.experiment import experiment_context_from_stage_dir
from .run_context import PIPELINE_NAME, context_from_run_dir as _context_from_run_dir
from ..evaluation.thresholds import (
    build_selected_threshold_specs as _build_selected_threshold_specs,
    build_selected_threshold_summary as _build_selected_threshold_summary,
    resolve_accepted_fprs as _resolve_accepted_fprs,
    select_threshold_max_recall_under_fpr as _select_threshold_max_recall_under_fpr,
)
from .timing import append_stage_timing
from .training import (
    select_winner as _select_winner,
    task_result_path as _task_result_path,
)


FINALIZE_STATE_FILENAME = "finalize_state.json"


def finalize_state_path(run_dir: Path) -> Path:
    return run_dir / "reports" / FINALIZE_STATE_FILENAME


def write_finalize_state(
    *,
    run_dir: Path,
    run_config: dict[str, Any],
    artifacts: dict[str, str | None],
) -> Path:
    state_path = finalize_state_path(run_dir)
    payload = {"run_config": run_config, "artifacts": artifacts}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return state_path


def load_finalize_state(run_dir: Path) -> dict[str, Any]:
    path = finalize_state_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Finalize state not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in finalize state, got {type(payload).__name__}")
    return payload


def load_run_config(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_config.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in run config, got {type(payload).__name__}")
    return payload


def completed_supervised_result(run_dir: Path) -> dict[str, Any]:
    artifact_index_path = run_dir / "artifact_index.json"
    if not artifact_index_path.exists():
        raise FileNotFoundError(f"Artifact index not found: {artifact_index_path}")
    with open(artifact_index_path, "r", encoding="utf-8") as f:
        artifact_index = json.load(f)
    if not isinstance(artifact_index, dict):
        raise ValueError(f"Expected JSON object in artifact index, got {type(artifact_index).__name__}")
    return {
        "run_dir": str(run_dir),
        "report": str(artifact_index.get("report")),
        "train_scores_csv": str(artifact_index.get("train_scores_csv")),
        "validation_scores_csv": (
            str(artifact_index.get("validation_predictions") or artifact_index.get("validation_scores_csv"))
            if artifact_index.get("validation_predictions") or artifact_index.get("validation_scores_csv")
            else None
        ),
        "best_model": str(artifact_index.get("best_model")),
    }


def cleanup_finalize_intermediates(run_dir: Path, state: dict[str, Any]) -> None:
    _ = state
    finalize_state_path(run_dir).unlink(missing_ok=True)
    task_dir = run_dir / ".work" / "tuning_tasks"
    if task_dir.exists():
        shutil.rmtree(task_dir)


def _publish_cv_validation_predictions(
    *,
    run_dir: Path,
    task_results: list[dict[str, Any]],
) -> None:
    path_mapping: dict[str, str] = {}
    for candidate in task_results:
        task_index = int(candidate.get("task_index", -1))
        fold_results = candidate.get("fold_results")
        if not isinstance(fold_results, list):
            continue
        for row in fold_results:
            if not isinstance(row, dict) or not row.get("validation_predictions"):
                continue
            source = Path(str(row["validation_predictions"])).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(f"CV validation predictions not found: {source}")
            group_index = int(row.get("cv_group_index", 0))
            fold_index = int(row.get("cv_fold_index", 0))
            seed = int(row.get("cv_random_state", 0))
            destination = (
                run_dir
                / "reports"
                / "tuning"
                / "candidates"
                / f"task_{task_index:04d}"
                / "folds"
                / f"seed_{seed}_group_{group_index:02d}"
                / f"fold_{fold_index:04d}"
                / "validation.csv"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            path_mapping[str(source)] = str(destination.relative_to(run_dir))

    def sanitize(value: Any) -> None:
        if isinstance(value, dict):
            raw_prediction_path = value.get("validation_predictions")
            if raw_prediction_path:
                resolved = str(Path(str(raw_prediction_path)).expanduser().resolve())
                value["validation_predictions"] = path_mapping.get(resolved, str(raw_prediction_path))
            value.pop("model_artifact", None)
            for nested in value.values():
                sanitize(nested)
        elif isinstance(value, list):
            for nested in value:
                sanitize(nested)

    sanitize(task_results)


def prepare_supervised_finalize(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
) -> dict[str, Any]:
    # Kept in the public API for compatibility with existing launch scripts.
    # Percentile-derived decisions are intentionally absent from the unified report.
    _ = score_percentiles

    ctx = _context_from_run_dir(run_dir)
    tuning_manifest_path = run_dir / ".work" / "tuning.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    task_dir = run_dir / ".work" / "tuning_tasks"
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
    no_refit = bool(manifest["tuning"].get("no_refit", False))
    refit = (
        select_best_cv_fold_model_and_predict(manifest=manifest, winner=winner, ctx=ctx)
        if no_refit
        else fit_winner_and_predict(manifest=manifest, winner=winner, ctx=ctx)
    )
    task_spec = refit.task_spec
    input_normalization = refit.input_normalization
    model_path = refit.model_path
    fit_checkpoint_artifacts = refit.fit_checkpoint_artifacts
    transformer_refit_epoch_plan = refit.transformer_refit_epoch_plan
    train_pool_model_names = refit.train_pool_model_names
    train_model_names = refit.train_model_names
    train_labels = refit.train_labels
    train_scores = refit.train_scores
    validation_model_names = refit.validation_model_names
    validation_labels = refit.validation_labels
    validation_scores = refit.validation_scores
    calibration_indices = refit.calibration_indices
    calibration_model_names = refit.calibration_model_names
    calibration_labels = refit.calibration_labels
    calibration_binary_labels = refit.calibration_binary_labels
    calibration_scores = refit.calibration_scores
    threshold_selection = manifest.get("threshold_selection", {})
    selected_thresholds: dict[str, Any] | None = None
    if calibration_binary_labels is not None and calibration_scores is not None:
        accepted_fprs = threshold_selection.get("accepted_fprs")
        if accepted_fprs is None:
            legacy_fpr = threshold_selection.get("accepted_fpr")
            accepted_fprs = None if legacy_fpr is None else [legacy_fpr]
        resolved_fprs = _resolve_accepted_fprs(accepted_fprs)
        if resolved_fprs is None:
            raise ValueError(
                "Calibration split is present, but tuning manifest is missing threshold_selection.accepted_fprs"
            )
        selections = []
        for accepted_fpr in resolved_fprs:
            selected = _select_threshold_max_recall_under_fpr(
                labels_true=calibration_binary_labels,
                scores=calibration_scores,
                accepted_fpr=float(accepted_fpr),
            )
            selections.append(
                {
                    "accepted_fpr": float(accepted_fpr),
                    "threshold": float(selected["threshold"]),
                }
            )
        selected_thresholds = _build_selected_threshold_summary(
            selections=selections,
        )
    threshold_specs = _build_selected_threshold_specs(selected_thresholds)

    evaluations: dict[str, Any] = {"train": build_binary_evaluation(labels=train_labels, scores=train_scores)}
    if validation_scores is not None:
        evaluations["validation"] = build_binary_evaluation(
            labels=validation_labels,
            scores=validation_scores,
        )
    if calibration_scores is not None:
        evaluations["calibration"] = build_binary_evaluation(
            labels=calibration_labels,
            scores=calibration_scores,
            calibrated_thresholds=threshold_specs,
        )

    winner_summary = {
        key: winner[key]
        for key in (
            "task_index",
            "model_name",
            "params",
            "selection_metric_name",
            "selection_metric_mean",
            "selection_metric_std",
        )
        if key in winner
    }
    metric_name = str(manifest["tuning"].get("metric", task_spec.selection_metric_name))
    successful_candidates = [row for row in task_results if str(row.get("status")) == "ok"]
    failed_candidates = [
        {
            "task_index": int(row.get("task_index", -1)),
            "model_name": str(row.get("model_name", "unknown")),
            "error": str(row.get("error", "Unknown tuning failure")),
        }
        for row in task_results
        if str(row.get("status")) != "ok"
    ]
    report_warnings = [str(value) for value in manifest.get("warnings", [])]
    if failed_candidates:
        report_warnings.append(
            f"{len(failed_candidates)} of {len(task_results)} tuning candidates failed; "
            "the winner was selected from the successful candidates only"
        )
    report = {
        "run_id": str(ctx.run_id),
        "status": "complete",
        "task": {
            "training_mode": str(task_spec.task_mode),
            "evaluation_mode": "backdoor_vs_clean",
            "positive_class": "any_backdoor",
            "score": (
                "positive_class_probability"
                if task_spec.binary_projection == BINARY_PROJECTION_POSITIVE_CLASS_SCORE
                else "one_minus_clean_probability"
            ),
        },
        "selection": {
            "metric": metric_name,
            "strategy": manifest["tuning"].get("cv_strategy"),
            "candidate_count": int(len(task_results)),
            "successful_candidate_count": int(len(successful_candidates)),
            "failed_candidate_count": int(len(failed_candidates)),
            "failed_candidates": failed_candidates,
            "winner": winner_summary,
        },
        "training": {
            "strategy": str(refit.training_strategy),
            "refit": not no_refit,
            "selected_fold": refit.selected_fold,
        },
        "evaluation": evaluations,
        "warnings": report_warnings,
    }

    experiment_config = {
        "pipeline": PIPELINE_NAME,
        "task": task_spec.to_dict(),
        "data": {
            "mode": manifest["mode"],
            "manifest_snapshot": "inputs/data_manifest.json",
            "partitions": "inputs/data_partitions.json",
            "split": manifest["data"].get("split"),
            "calibration_split": manifest["data"].get("calibration_split"),
        },
        "features": {
            "extractor": manifest["extractor"]["name"],
            "params": manifest["extractor"]["params"],
            "input_normalization": manifest.get(
                "input_normalization",
                {"mode": str(input_normalization)},
            ),
        },
        "selection": {
            "metric": metric_name,
            "cv_strategy": manifest["tuning"].get("cv_strategy"),
            "cv_folds": manifest["tuning"].get("cv_folds_resolved"),
            "cv_random_states": manifest["tuning"].get(
                "cv_random_states",
                [manifest["tuning"]["random_state"]],
            ),
            "cv_stratification": manifest["tuning"].get("cv_stratification"),
            "model_grid": "inputs/model_grid.json",
        },
        "training": {
            "strategy": str(refit.training_strategy),
            "refit": not no_refit,
            "selected_fold": refit.selected_fold,
            "class_weight_loss": bool(manifest["tuning"].get("class_weight_loss", False)),
            "rank_label_weight_loss": bool(manifest["tuning"].get("rank_label_weight_loss", False)),
            "refit_epoch_plan": transformer_refit_epoch_plan,
            "domain_adaptation": (
                {
                    key: manifest["domain_adaptation"].get(key)
                    for key in (
                        "enabled",
                        "source_rank",
                        "train_ranks",
                        "target_train_ranks",
                        "domain_rank_values",
                        "domain_class_names",
                    )
                }
                if isinstance(manifest.get("domain_adaptation"), dict)
                else None
            ),
        },
        "calibration": {
            "enabled": bool(calibration_indices.size > 0),
            "accepted_fprs": (selected_thresholds.get("accepted_fprs", []) if selected_thresholds is not None else []),
        },
    }

    data_partitions = {
        "training_pool": train_pool_model_names,
        "train": train_model_names,
        "validation": validation_model_names,
        "calibration": calibration_model_names,
    }
    model_grid = {
        "selection_metric": metric_name,
        "models": manifest["tuning"].get(
            "model_names",
            [manifest["tuning"]["model_name"]],
        ),
        "candidates": manifest["tuning"]["tasks"],
    }
    predictions = {
        "train": PredictionPartition(
            model_names=train_model_names,
            labels=train_labels,
            scores=train_scores,
        )
    }
    if validation_scores is not None:
        predictions["validation"] = PredictionPartition(
            model_names=validation_model_names,
            labels=validation_labels,
            scores=validation_scores,
        )
    if calibration_scores is not None:
        predictions["calibration"] = PredictionPartition(
            model_names=calibration_model_names,
            labels=calibration_labels,
            scores=calibration_scores,
        )

    threshold_document = (
        {
            "enabled": True,
            **dict(selected_thresholds or {}),
        }
        if selected_thresholds is not None
        else {
            "enabled": False,
            "method": None,
            "source_partition": None,
            "accepted_fprs": [],
            "selections": [],
        }
    )
    attach_inference_contract(
        model_path,
        {
            "model_name": str(winner["model_name"]),
            "task": task_spec.to_dict(),
            "feature_configuration": dict(manifest["extractor"]["params"]),
            "input_normalization": dict(manifest.get("input_normalization", {})),
            "thresholds": threshold_document,
            "domain_adaptation": manifest.get("domain_adaptation"),
            "training_strategy": str(refit.training_strategy),
            "selected_fold": refit.selected_fold,
        },
    )
    _publish_cv_validation_predictions(run_dir=run_dir, task_results=task_results)
    output_paths = write_reporting_bundle(
        run_dir=run_dir,
        report=report,
        experiment_config=experiment_config,
        source_manifest=resolve_manifest_path(manifest["manifest_json"]),
        data_partitions=data_partitions,
        model_grid=model_grid,
        predictions=predictions,
        thresholds=threshold_document,
        tuning_candidates=task_results,
    )

    artifacts = {
        "best_model": str(model_path),
        **{key: str(path) for key, path in output_paths.items()},
        "tuning_manifest": str(tuning_manifest_path),
        "tuning_tasks_dir": str(task_dir),
        "checkpoint_index": fit_checkpoint_artifacts.get("checkpoint_index"),
        "resume_checkpoint": fit_checkpoint_artifacts.get("resume_checkpoint"),
        "final_training_checkpoint": fit_checkpoint_artifacts.get("final_checkpoint"),
    }
    run_config = {
        "pipeline": PIPELINE_NAME,
        "status": "complete",
        "experiment_config": str(output_paths["experiment_config"]),
    }
    state_path = write_finalize_state(
        run_dir=run_dir,
        run_config=run_config,
        artifacts=artifacts,
    )
    return {
        "run_dir": str(run_dir),
        "report": str(output_paths["report"]),
        "train_scores_csv": str(output_paths["train_predictions"]),
        "calibration_scores_csv": (
            str(output_paths["calibration_predictions"]) if "calibration_predictions" in output_paths else None
        ),
        "validation_scores_csv": (
            str(output_paths["validation_predictions"]) if "validation_predictions" in output_paths else None
        ),
        "best_model": str(model_path),
        "finalize_state_path": state_path,
    }


def complete_supervised_finalize(
    *,
    run_dir: Path,
) -> dict[str, Any]:
    state = load_finalize_state(run_dir)
    ctx = _context_from_run_dir(run_dir)

    base_artifacts = state.get("artifacts", {})
    if not isinstance(base_artifacts, dict):
        raise ValueError("Finalize state is missing base artifacts")
    for key, value in base_artifacts.items():
        if value:
            ctx.add_artifact(str(key), Path(str(value)))

    run_config = state.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("Finalize state is missing run_config")
    experiment = experiment_context_from_stage_dir(run_dir)
    if experiment is None:
        ctx.finalize(run_config)
    else:
        experiment_artifacts: dict[str, Any] = {
            "best_model": {
                "kind": "selected_model",
                "path": experiment.display_path(Path(str(base_artifacts.get("best_model")))),
            }
        }
        if base_artifacts.get("checkpoint_index"):
            experiment_artifacts["interval_checkpoints"] = {
                "kind": "training_checkpoints",
                "path": experiment.display_path(Path(str(base_artifacts["checkpoint_index"])).parent),
                "index": str(base_artifacts["checkpoint_index"]),
                "resume_checkpoint": base_artifacts.get("resume_checkpoint"),
                "final_checkpoint": base_artifacts.get("final_training_checkpoint"),
            }
        experiment.update(
            stage="training",
            stage_status="completed",
            artifacts=experiment_artifacts,
        )

    cleanup_finalize_intermediates(run_dir, state)

    return {
        "run_dir": str(run_dir),
        "report": str(base_artifacts.get("report")),
        "train_scores_csv": str(base_artifacts.get("train_predictions")),
        "validation_scores_csv": (
            str(base_artifacts.get("validation_predictions"))
            if base_artifacts.get("validation_predictions")
            else None
        ),
        "best_model": str(base_artifacts.get("best_model")),
    }


def finalize_supervised_run(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    prepare_supervised_finalize(
        run_dir=run_dir,
        score_percentiles=score_percentiles,
    )
    finalized = complete_supervised_finalize(run_dir=run_dir)
    append_stage_timing(
        run_dir=run_dir,
        stage="supervised_finalize",
        started_at=started_at,
        started_perf=started_perf,
    )
    return finalized
