from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from ..features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
)
from ..utilities.core.paths import default_dataset_root
from .common import (
    timing_finish as _timing_finish,
    timing_start as _timing_start,
)
from .config import (
    coerce_features as _coerce_features,
    model_requires_feature_aggregation,
    resolve_split_by_folder as _resolve_split_by_folder,
)
from .experiment import create_experiment_context

from ..orchestration.slurm.experiment import afterok_dependency as _afterok_dependency
from .aggregation import run_feature_aggregation
from .extraction import run_feature_extraction
from .training import run_supervised_training, validate_checkpoint_options
from .inference import run_checkpoint_inference
from .manifests import materialize_manifest_snapshot, resolve_full_manifests
from ..supervised.models.registry import (
    TORCH_SEQUENCE_BACKEND,
    model_backend,
    validate_hyperparams,
)


def run_full_experiment(
    *,
    manifest_json: Path | None = None,
    train_manifest_json: Path | None = None,
    inference_manifest_json: Path | None = None,
    dataset_root: Path = default_dataset_root(),
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    cpus_per_worker: str = "4",
    extract_nodes: str = "auto",
    extract_cpus_per_worker: str | None = None,
    extract_workers_per_node: str = "auto",
    extract_parallelization_settings: list[str] | None = None,
    aggregate_cpus_per_worker: str | None = None,
    train_nodes: str = "auto",
    train_cpus_per_worker: str | None = None,
    train_workers_per_node: str = "auto",
    dry_run: bool = False,
    features: list[str] | None = None,
    spectral_sv_top_k: int = 8,
    spectral_moment_source: str = DEFAULT_SPECTRAL_MOMENT_SOURCE,
    spectral_qv_sum_mode: str = DEFAULT_SPECTRAL_QV_SUM_MODE,
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
    experiment = create_experiment_context(
        output_root=output_root,
        run_id=run_id,
        workflow="full",
        backend=backend,
    )
    for stage in ("extraction", "aggregation", "training", "inference"):
        experiment.update(stage=stage, stage_status="pending")
    full_timing, full_started = _timing_start("full", backend=backend)
    resolved_features = _coerce_features(features)
    aggregation_required = model_requires_feature_aggregation(model_name)
    effective_split_by_folder = _resolve_split_by_folder(split_by_folder, calibration_split)
    resolved_extract_cpus_per_worker = extract_cpus_per_worker or cpus_per_worker
    resolved_aggregate_cpus_per_worker = aggregate_cpus_per_worker or cpus_per_worker
    resolved_train_cpus_per_worker = train_cpus_per_worker or cpus_per_worker
    manifest_work_dir = experiment.run_dir / ".work" / "manifest_resolution"
    resolved_manifests = resolve_full_manifests(
        output_dir=manifest_work_dir,
        manifest_json=manifest_json,
        train_manifest_json=train_manifest_json,
        inference_manifest_json=inference_manifest_json,
        train_split=train_split,
        random_state=random_state,
        split_by_folder=effective_split_by_folder,
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
    )
    assert resolved_manifests.inference_manifest is not None
    full_manifest = materialize_manifest_snapshot(
        resolved_manifests.full_manifest,
        experiment.run_dir / "extraction" / "inputs" / "data_manifest.json",
    )
    training_manifest = materialize_manifest_snapshot(
        resolved_manifests.train_manifest,
        experiment.run_dir / "training" / "inputs" / "data_manifest.json",
    )
    inference_manifest = materialize_manifest_snapshot(
        resolved_manifests.inference_manifest,
        experiment.run_dir / "inference" / "inputs" / "data_manifest.json",
    )
    shutil.rmtree(manifest_work_dir)
    try:
        manifest_work_dir.parent.rmdir()
    except OSError:
        pass
    experiment.update(
        values={
            "configuration": {
                "manifest_json": str(full_manifest),
                "train_manifest_json": str(training_manifest),
                "inference_manifest_json": str(inference_manifest),
                "dataset_root": str(Path(dataset_root).expanduser().resolve()),
                "model": str(model_name),
                "feature_aggregation_required": bool(aggregation_required),
                "features": resolved_features,
                "spectral_moment_source": spectral_moment_source,
                "spectral_qv_sum_mode": spectral_qv_sum_mode,
                "spectral_attention_granularity": spectral_attention_granularity,
                "train_split": int(train_split) if train_split is not None else None,
                "cv_folds": int(cv_folds),
                "cv_seeds": list(cv_seeds or []),
                "no_refit": bool(no_refit),
            }
        }
    )

    if backend == "slurm":
        extract = run_feature_extraction(
            manifest_json=full_manifest,
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=experiment.run_id,
            backend="slurm",
            partition=partition,
            nodes=extract_nodes,
            cpus_per_worker=resolved_extract_cpus_per_worker,
            workers_per_node=extract_workers_per_node,
            dry_run=dry_run,
            features=resolved_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            spectral_attention_granularity=spectral_attention_granularity,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            wait_for_job_index=not dry_run,
            parallelization_settings=extract_parallelization_settings,
        )
        extraction_dependency = _afterok_dependency(
            extract.get("slurm", {}),
            job_index_path=Path(str(extract["slurm_job_index"])),
            dry_run_placeholder="DRYRUN_FEATURE_FINALIZE",
        )
        if aggregation_required:
            aggregate = run_feature_aggregation(
                feature_file=Path(extract["feature_path"]),
                output_filename=None,
                output_root=output_root,
                run_id=experiment.run_id,
                backend="slurm",
                partition=partition,
                cpus_per_worker=resolved_aggregate_cpus_per_worker,
                dry_run=dry_run,
                feature_root=None,
                features=resolved_features,
                spectral_qv_sum_mode=spectral_qv_sum_mode,
                spectral_attention_granularity=spectral_attention_granularity,
                dependency=extraction_dependency,
            )
            train_dependency = _afterok_dependency(
                aggregate.get("slurm", {}),
                dry_run_placeholder="DRYRUN_AGGREGATE",
            )
            training_feature_path = Path(aggregate["feature_path"])
        else:
            training_feature_path = Path(extract["feature_path"])
            train_dependency = extraction_dependency
            aggregate = {
                "skipped": True,
                "reason": "selected model accepts extracted tabular spectral features",
                "feature_path": str(training_feature_path),
                "timings": None,
            }
            experiment.update(
                stage="aggregation",
                stage_status="completed",
                stage_values={"skipped": True, "reason": aggregate["reason"]},
            )
        train = run_supervised_training(
            manifest_json=training_manifest,
            feature_file=training_feature_path,
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=experiment.run_id,
            backend="slurm",
            partition=partition,
            nodes=train_nodes,
            cpus_per_worker=resolved_train_cpus_per_worker,
            workers_per_node=train_workers_per_node,
            dry_run=dry_run,
            features=resolved_features,
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
            train_split=None,
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
            dependency=train_dependency,
            wait_for_job_index=not dry_run,
        )
        train_dependency = _afterok_dependency(
            train.get("slurm", {}),
            job_index_path=Path(str(train["slurm"]["job_index"])),
            dry_run_placeholder="DRYRUN_TRAIN_FINALIZE",
        )
        checkpoint_suffix = ".pt" if model_backend(str(model_name)) == TORCH_SEQUENCE_BACKEND else ".joblib"
        inference = run_checkpoint_inference(
            checkpoint=Path(str(train["stage_run_dir"])) / "models" / f"best_model{checkpoint_suffix}",
            manifest_json=inference_manifest,
            feature_file=training_feature_path,
            output_root=output_root,
            run_id=experiment.run_id,
            backend="slurm",
            partition=partition,
            cpus_per_worker=cpus_per_worker,
            dry_run=dry_run,
            dependency=train_dependency,
        )
        full_timing = _timing_finish(full_timing, full_started)
        timings = {
            "full": full_timing,
            "extract": extract.get("timings"),
            "aggregate": aggregate.get("timings"),
            "train": train.get("timings"),
            "inference": inference.get("timings"),
        }
        artifacts = {
            "feature_path": extract["feature_path"],
            "aggregated_feature_path": (aggregate["feature_path"] if aggregation_required else None),
            "training_feature_path": str(training_feature_path),
            "training_run_dir": train["stage_run_dir"],
            "inference_run_dir": inference["stage_run_dir"],
        }
        experiment.update(
            workflow="full",
            backend="slurm",
            values={
                "status": "planned" if dry_run else "submitted",
                "timing": full_timing,
            },
        )
        return {
            "run_dir": str(experiment.run_dir),
            "run_id": experiment.run_id,
            "backend": "slurm",
            "extract": extract,
            "aggregate": aggregate,
            "train": train,
            "inference": inference,
            "artifacts": artifacts,
            "timings": timings,
        }

    extract = run_feature_extraction(
        manifest_json=full_manifest,
        dataset_root=dataset_root,
        output_root=output_root,
        run_id=experiment.run_id,
        backend="local",
        features=resolved_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
    )
    if aggregation_required:
        aggregate = run_feature_aggregation(
            feature_file=Path(extract["feature_path"]),
            output_filename=None,
            output_root=output_root,
            run_id=experiment.run_id,
            backend="local",
            feature_root=None,
            features=resolved_features,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_attention_granularity=spectral_attention_granularity,
        )
        training_feature_path = Path(aggregate["feature_path"])
    else:
        training_feature_path = Path(extract["feature_path"])
        aggregate = {
            "skipped": True,
            "reason": "selected model accepts extracted tabular spectral features",
            "feature_path": str(training_feature_path),
            "timings": None,
        }
        experiment.update(
            stage="aggregation",
            stage_status="completed",
            stage_values={"skipped": True, "reason": aggregate["reason"]},
        )
    train = run_supervised_training(
        manifest_json=training_manifest,
        feature_file=training_feature_path,
        dataset_root=dataset_root,
        output_root=output_root,
        run_id=experiment.run_id,
        backend="local",
        features=resolved_features,
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
        train_split=None,
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
    inference = run_checkpoint_inference(
        checkpoint=Path(str(train["best_model"])),
        manifest_json=inference_manifest,
        feature_file=training_feature_path,
        output_root=output_root,
        run_id=experiment.run_id,
        backend="local",
    )
    full_timing = _timing_finish(full_timing, full_started)
    timings = {
        "full": full_timing,
        "extract": extract.get("timings"),
        "aggregate": aggregate.get("timings"),
        "train": train.get("timings"),
        "inference": inference.get("timings"),
    }
    artifacts = {
        "feature_path": extract["feature_path"],
        "aggregated_feature_path": (aggregate["feature_path"] if aggregation_required else None),
        "training_feature_path": str(training_feature_path),
        "training_run_dir": train["stage_run_dir"],
        "inference_run_dir": inference["stage_run_dir"],
    }
    experiment.update(
        workflow="full",
        backend="local",
        values={
            "status": "completed",
            "timing": full_timing,
        },
    )
    return {
        "run_dir": str(experiment.run_dir),
        "run_id": experiment.run_id,
        "backend": "local",
        "extract": extract,
        "aggregate": aggregate,
        "train": train,
        "inference": inference,
        "artifacts": artifacts,
        "timings": timings,
    }
