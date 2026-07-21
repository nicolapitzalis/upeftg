from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from ..features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
)
from ..utilities.core.paths import default_dataset_root
from ..orchestration.slurm.experiment import submit_supervised_training
from .common import (
    timing_finish as _timing_finish,
    timing_start as _timing_start,
)
from .config import (
    coerce_features as _coerce_features,
    resolve_split_by_folder as _resolve_split_by_folder,
    stage_context as _stage_context,
)

from ..supervised.lifecycle.pipeline import run_supervised_pipeline
from ..supervised.models.registry import validate_hyperparams
from .manifests import materialize_manifest_snapshot, resolve_training_manifests


def validate_checkpoint_options(
    *,
    model_name: str,
    checkpoint_interval_hours: float | None,
    resume_checkpoint: Path | None,
    no_refit: bool = False,
) -> None:
    if bool(no_refit) and checkpoint_interval_hours is not None:
        raise ValueError("--no-refit cannot be combined with --checkpoint-interval-hours")
    if checkpoint_interval_hours is not None:
        if str(model_name) != "transformer":
            raise ValueError("--checkpoint-interval-hours currently requires --model transformer")
        if float(checkpoint_interval_hours) <= 0.0:
            raise ValueError("--checkpoint-interval-hours must be positive")
    if resume_checkpoint is not None and checkpoint_interval_hours is None:
        raise ValueError("--resume-checkpoint requires --checkpoint-interval-hours")


def _supervised_common_kwargs(
    *,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    training_run_dir: Path,
    feature_file: Path,
    features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    cv_strategy: str,
    input_normalization: str,
    cv_derived_refit_epochs: bool,
    no_refit: bool,
    random_state: int,
    train_split: int,
    calibration_split: int | None,
    accepted_fpr: list[float] | None,
    split_by_folder: bool | None,
    cv_seeds: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float] | None,
    model_name: str,
    hyperparams: Path | None,
    checkpoint_interval_hours: float | None,
    resume_checkpoint: Path | None,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
    class_weight_loss: bool,
    rank_label_weight_loss: bool,
    selection_metric: str | None,
) -> dict[str, Any]:
    return {
        "manifest_json": manifest_json,
        "dataset_root": dataset_root,
        "output_root": output_root,
        "run_id": run_id,
        "prepared_run_dir": training_run_dir,
        "model_name": str(model_name),
        "spectral_features": _coerce_features(features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": spectral_moment_source,
        "spectral_qv_sum_mode": spectral_qv_sum_mode,
        "spectral_entrywise_delta_mode": spectral_entrywise_delta_mode,
        "spectral_attention_granularity": spectral_attention_granularity,
        "stream_block_size": int(stream_block_size),
        "dtype_name": dtype_name,
        "cv_folds": int(cv_folds),
        "cv_strategy": str(cv_strategy),
        "input_normalization": str(input_normalization),
        "cv_derived_refit_epochs": bool(cv_derived_refit_epochs),
        "no_refit": bool(no_refit),
        "random_state": int(random_state),
        "train_split_percent": int(train_split),
        "calibration_split_percent": calibration_split,
        "accepted_fpr": accepted_fpr,
        "split_by_folder": _resolve_split_by_folder(split_by_folder, calibration_split),
        "cv_random_states": cv_seeds,
        "n_jobs": int(n_jobs),
        "score_percentiles": score_percentiles,
        "feature_file": feature_file,
        "tuning_executor": "local",
        "slurm_partition": "extra",
        "cpus_per_worker": "4",
        "stage": "all",
        "run_dir": None,
        "task_index": None,
        "task_mode": task_mode,
        "multiclass_attack_names": multiclass_attack_names,
        "hyperparams": hyperparams,
        "transformer_checkpoint_dir": (
            training_run_dir / "models" / "interval_checkpoints" if checkpoint_interval_hours is not None else None
        ),
        "transformer_checkpoint_interval_seconds": (
            float(checkpoint_interval_hours) * 3600.0 if checkpoint_interval_hours is not None else None
        ),
        "transformer_resume_checkpoint": resume_checkpoint,
        "class_weight_loss": class_weight_loss,
        "rank_label_weight_loss": rank_label_weight_loss,
        "selection_metric": selection_metric,
    }


def run_supervised_training(
    *,
    manifest_json: Path,
    feature_file: Path,
    dataset_root: Path = default_dataset_root(),
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    nodes: str = "auto",
    cpus_per_worker: str = "4",
    workers_per_node: str = "auto",
    dry_run: bool = False,
    features: list[str] | None = None,
    spectral_sv_top_k: int = 8,
    spectral_moment_source: str = DEFAULT_SPECTRAL_MOMENT_SOURCE,
    spectral_qv_sum_mode: str = "append",
    spectral_entrywise_delta_mode: str = DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    spectral_attention_granularity: str = DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    stream_block_size: int = 131072,
    dtype_name: str = "float32",
    cv_folds: int = 5,
    cv_strategy: str = "stratified",
    input_normalization: str = "none",
    cv_derived_refit_epochs: bool = False,
    no_refit: bool = False,
    random_state: int = 42,
    train_split: int | None = None,
    calibration_split: int | None = None,
    accepted_fpr: list[float] | None = None,
    split_by_folder: bool | None = None,
    cv_seeds: list[int] | None = None,
    n_jobs: int = -1,
    score_percentiles: list[float] | None = None,
    model_name: str,
    hyperparams: Path | None = None,
    checkpoint_interval_hours: float | None = None,
    resume_checkpoint: Path | None = None,
    task_mode: str = "binary",
    multiclass_attack_names: list[str] | None = None,
    class_weight_loss: bool = False,
    rank_label_weight_loss: bool = False,
    selection_metric: str | None = "task_default",
    dependency: str | None = None,
    wait_for_job_index: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    validate_checkpoint_options(
        model_name=model_name,
        checkpoint_interval_hours=checkpoint_interval_hours,
        resume_checkpoint=resume_checkpoint,
        no_refit=no_refit,
    )
    if bool(no_refit) and bool(cv_derived_refit_epochs):
        raise ValueError("--no-refit cannot be combined with --cv-derived-refit-epochs")
    validate_hyperparams(
        model_name=str(model_name),
        hyperparams=hyperparams,
    )
    effective_split_by_folder = _resolve_split_by_folder(split_by_folder, calibration_split)
    experiment, ctx = _stage_context(
        output_root,
        run_id=run_id,
        workflow="train",
        stage="training",
        backend=backend,
    )
    manifest_work_dir = ctx.run_dir / ".work" / "manifest_resolution"
    resolved_manifests = resolve_training_manifests(
        output_dir=manifest_work_dir,
        manifest_json=manifest_json,
        train_split=train_split,
        random_state=random_state,
        split_by_folder=effective_split_by_folder,
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
    )
    stage_inputs_dir = ctx.run_dir / "inputs"
    training_manifest = materialize_manifest_snapshot(
        resolved_manifests.train_manifest,
        stage_inputs_dir / "data_manifest.json",
    )
    full_manifest = None
    inference_manifest = None
    if resolved_manifests.split_summary is not None:
        full_manifest = materialize_manifest_snapshot(
            resolved_manifests.full_manifest,
            stage_inputs_dir / "full_manifest.json",
        )
    if resolved_manifests.inference_manifest is not None:
        inference_manifest = materialize_manifest_snapshot(
            resolved_manifests.inference_manifest,
            stage_inputs_dir / "inference_manifest.json",
        )
    shutil.rmtree(manifest_work_dir)
    try:
        manifest_work_dir.parent.rmdir()
    except OSError:
        pass
    experiment.update(stage="training", stage_status="running")
    experiment.update(
        values={
            "configuration": {
                "manifest_json": str(training_manifest),
                "inference_manifest_json": (str(inference_manifest) if inference_manifest is not None else None),
                "full_manifest_json": str(full_manifest) if full_manifest is not None else None,
                "feature_file": str(Path(feature_file).resolve()),
                "model": str(model_name),
                "features": _coerce_features(features),
                "cv_strategy": str(cv_strategy),
                "no_refit": bool(no_refit),
                "train_split": int(train_split) if train_split is not None else None,
            }
        }
    )
    if backend == "slurm":
        timing, started_perf = _timing_start("training_submit", backend="slurm")
        pipeline_kwargs = _supervised_common_kwargs(
            manifest_json=training_manifest,
            dataset_root=Path(dataset_root).expanduser().resolve(),
            output_root=output_root,
            run_id=ctx.run_id,
            training_run_dir=ctx.run_dir,
            feature_file=Path(feature_file),
            features=features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            spectral_attention_granularity=spectral_attention_granularity,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            input_normalization=input_normalization,
            cv_derived_refit_epochs=cv_derived_refit_epochs,
            no_refit=no_refit,
            random_state=random_state,
            train_split=100,
            calibration_split=calibration_split,
            accepted_fpr=accepted_fpr,
            split_by_folder=effective_split_by_folder,
            cv_seeds=cv_seeds or [42, 43, 44],
            n_jobs=n_jobs,
            score_percentiles=score_percentiles,
            model_name=str(model_name),
            hyperparams=hyperparams,
            checkpoint_interval_hours=checkpoint_interval_hours,
            resume_checkpoint=resume_checkpoint,
            task_mode=task_mode,
            multiclass_attack_names=multiclass_attack_names,
            class_weight_loss=class_weight_loss,
            rank_label_weight_loss=rank_label_weight_loss,
            selection_metric=selection_metric,
        )
        submission = submit_supervised_training(
            run_dir=ctx.run_dir,
            run_id=ctx.run_id,
            pipeline_kwargs=pipeline_kwargs,
            partition=partition,
            nodes=nodes,
            cpus_per_worker=cpus_per_worker,
            workers_per_node=workers_per_node,
            dependency=dependency,
            dry_run=dry_run,
            wait_for_job_index=wait_for_job_index,
        )
        timing = _timing_finish(timing, started_perf)
        checkpoint_artifact = (
            {
                "interval_checkpoints": {
                    "kind": "training_checkpoints",
                    "path": experiment.display_path(ctx.models_dir / "interval_checkpoints"),
                    "interval_hours": float(checkpoint_interval_hours),
                }
            }
            if checkpoint_interval_hours is not None
            else {}
        )
        experiment.update(
            stage="training",
            stage_status="submitted" if not dry_run else "planned",
            stage_values={
                "job_ids": [submission.get("job_id")] if submission.get("job_id") else [],
                "timing": timing,
                "input_feature_path": experiment.display_path(Path(feature_file)),
                "dependency": dependency,
            },
            artifacts=checkpoint_artifact,
        )
        return {
            "run_dir": str(experiment.run_dir),
            "stage_run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "backend": "slurm",
            "slurm": submission,
            "timings": {"submit": timing},
            "train_manifest": str(training_manifest),
            "inference_manifest": (
                str(resolved_manifests.inference_manifest)
                if resolved_manifests.inference_manifest is not None
                else None
            ),
        }

    timing, started_perf = _timing_start("training", backend="local")
    kwargs = _supervised_common_kwargs(
        manifest_json=training_manifest,
        dataset_root=Path(dataset_root).expanduser().resolve(),
        output_root=output_root,
        run_id=ctx.run_id,
        training_run_dir=ctx.run_dir,
        feature_file=Path(feature_file),
        features=features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        cv_folds=cv_folds,
        cv_strategy=cv_strategy,
        input_normalization=input_normalization,
        cv_derived_refit_epochs=cv_derived_refit_epochs,
        no_refit=no_refit,
        random_state=random_state,
        train_split=100,
        calibration_split=calibration_split,
        accepted_fpr=accepted_fpr,
        split_by_folder=effective_split_by_folder,
        cv_seeds=cv_seeds,
        n_jobs=n_jobs,
        score_percentiles=score_percentiles,
        model_name=str(model_name),
        hyperparams=hyperparams,
        checkpoint_interval_hours=checkpoint_interval_hours,
        resume_checkpoint=resume_checkpoint,
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
        class_weight_loss=class_weight_loss,
        rank_label_weight_loss=rank_label_weight_loss,
        selection_metric=selection_metric,
    )
    result = run_supervised_pipeline(**kwargs)
    timing = _timing_finish(timing, started_perf)
    run_dir = Path(str(result["run_dir"]))
    existing_timings: dict[str, Any] = {"training": timing}
    experiment_artifacts = {
        "best_model": {
            "kind": "selected_model",
            "path": experiment.display_path(Path(str(result.get("best_model")))),
        }
    }
    if checkpoint_interval_hours is not None:
        experiment_artifacts["interval_checkpoints"] = {
            "kind": "training_checkpoints",
            "path": experiment.display_path(ctx.models_dir / "interval_checkpoints"),
            "interval_hours": float(checkpoint_interval_hours),
            "index": str(ctx.models_dir / "interval_checkpoints" / "checkpoint_index.json"),
        }
    experiment.update(
        stage="training",
        stage_status="completed",
        stage_values={
            "timing": timing,
            "input_feature_path": experiment.display_path(Path(feature_file)),
        },
        artifacts=experiment_artifacts,
    )
    result.update(
        {
            "run_dir": str(experiment.run_dir),
            "stage_run_dir": str(run_dir),
            "backend": "local",
            "timings": existing_timings,
            "train_manifest": str(training_manifest),
            "inference_manifest": (
                str(resolved_manifests.inference_manifest)
                if resolved_manifests.inference_manifest is not None
                else None
            ),
        }
    )
    return result
