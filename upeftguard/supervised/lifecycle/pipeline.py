from __future__ import annotations

from pathlib import Path
from typing import Any

from ...features.spectral import DEFAULT_SPECTRAL_ATTENTION_GRANULARITY
from ..contracts import (
    SUPERVISED_TASK_MODE_BINARY,
)
from ..data.normalization import (
    INPUT_NORMALIZATION_NONE,
    resolve_input_normalization as _resolve_input_normalization,
)
from ..validation.splits import (
    resolve_split_by_folder as _resolve_split_by_folder,
)
from ..evaluation.thresholds import resolve_accepted_fprs as _resolve_accepted_fprs
from .finalization import finalize_supervised_run as _finalize_supervised_run
from .tuning import (
    DANN_DEFAULT_LAMBDA_GAMMA,
    DANN_DEFAULT_LAMBDA_MAX,
    DANN_DEFAULT_SOURCE_RANK,
    DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
)
from .training import run_supervised_worker as _run_supervised_worker
from .run import prepare_supervised_run as _prepare_supervised_run


def run_supervised_pipeline(
    *,
    manifest_json: Path | None,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str | None,
    spectral_features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str = DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    cv_strategy: str,
    input_normalization: str = INPUT_NORMALIZATION_NONE,
    cv_derived_refit_epochs: bool = False,
    random_state: int,
    train_split_percent: int,
    calibration_split_percent: int | None,
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
    split_by_folder: bool | None,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float] | None,
    feature_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    cpus_per_worker: str,
    stage: str,
    run_dir: Path | None,
    task_index: int | None,
    task_mode: str = SUPERVISED_TASK_MODE_BINARY,
    multiclass_attack_names: list[str] | None = None,
    hyperparams: Path | None = None,
    transformer_checkpoint_dir: Path | None = None,
    transformer_checkpoint_interval_seconds: float | None = None,
    transformer_resume_checkpoint: Path | None = None,
    dann_source_rank: int = DANN_DEFAULT_SOURCE_RANK,
    dann_target_adaptation_percent: int = DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
    dann_lambda_max: float = DANN_DEFAULT_LAMBDA_MAX,
    dann_lambda_gamma: float = DANN_DEFAULT_LAMBDA_GAMMA,
    class_weight_loss: bool = False,
    rank_label_weight_loss: bool = False,
    selection_metric: str | None = None,
    prepared_run_dir: Path | None = None,
    no_refit: bool = False,
) -> dict[str, Any]:
    if stage in {"all", "prepare"} and model_name is None:
        raise ValueError("--model is required for stage=all and stage=prepare")
    if stage in {"all", "prepare"} and manifest_json is None:
        raise ValueError("--manifest-json is required for stage=all and stage=prepare")
    if stage in {"all", "prepare"} and feature_file is None:
        raise ValueError(
            "--feature-file is required for stage=all and stage=prepare. Pass a feature run name, "
            "feature output directory, or spectral_features.npy path."
        )
    if stage in {"all", "prepare"} and not spectral_features:
        raise ValueError(
            "--features is required for stage=all and stage=prepare. Specify the feature groups to "
            "select from the extracted feature bundle."
        )
    if stage in {"worker", "finalize"} and run_dir is None:
        raise ValueError("--run-dir is required for the selected supervised stage")
    if stage == "worker" and task_index is None:
        raise ValueError("task_index must be resolved by the command or orchestration layer")
    resolved_accepted_fprs = _resolve_accepted_fprs(accepted_fpr)
    if bool(no_refit) and bool(cv_derived_refit_epochs):
        raise ValueError("--no-refit cannot be combined with --cv-derived-refit-epochs")
    if (calibration_split_percent is None) != (resolved_accepted_fprs is None):
        raise ValueError("--calibration-split and --accepted-fpr must either both be set or both be omitted")
    effective_split_by_folder = _resolve_split_by_folder(split_by_folder, calibration_split_percent)
    resolved_input_normalization = _resolve_input_normalization(input_normalization)

    prepare_kwargs = {
        "manifest_json": Path(manifest_json) if manifest_json is not None else Path(""),
        "dataset_root": dataset_root,
        "output_root": output_root,
        "run_id": run_id,
        "prepared_run_dir": prepared_run_dir,
        "model_name": str(model_name) if model_name is not None else "",
        "spectral_features": spectral_features,
        "spectral_sv_top_k": spectral_sv_top_k,
        "spectral_moment_source": spectral_moment_source,
        "spectral_qv_sum_mode": spectral_qv_sum_mode,
        "spectral_entrywise_delta_mode": spectral_entrywise_delta_mode,
        "spectral_attention_granularity": spectral_attention_granularity,
        "stream_block_size": stream_block_size,
        "dtype_name": dtype_name,
        "cv_folds": cv_folds,
        "cv_strategy": cv_strategy,
        "input_normalization": resolved_input_normalization,
        "cv_derived_refit_epochs": bool(cv_derived_refit_epochs),
        "no_refit": bool(no_refit),
        "random_state": random_state,
        "train_split_percent": train_split_percent,
        "calibration_split_percent": calibration_split_percent,
        "accepted_fpr": resolved_accepted_fprs,
        "split_by_folder": effective_split_by_folder,
        "cv_random_states": cv_random_states,
        "n_jobs": n_jobs,
        "score_percentiles": score_percentiles or [90.0, 95.0, 99.0],
        "feature_file": feature_file,
        "tuning_executor": tuning_executor,
        "slurm_partition": slurm_partition,
        "cpus_per_worker": cpus_per_worker,
        "task_mode": task_mode,
        "multiclass_attack_names": multiclass_attack_names,
        "hyperparams": hyperparams,
        "transformer_checkpoint_dir": transformer_checkpoint_dir,
        "transformer_checkpoint_interval_seconds": transformer_checkpoint_interval_seconds,
        "transformer_resume_checkpoint": transformer_resume_checkpoint,
        "dann_source_rank": int(dann_source_rank),
        "dann_target_adaptation_percent": int(dann_target_adaptation_percent),
        "dann_lambda_max": float(dann_lambda_max),
        "dann_lambda_gamma": float(dann_lambda_gamma),
        "class_weight_loss": bool(class_weight_loss),
        "rank_label_weight_loss": bool(rank_label_weight_loss),
        "selection_metric": selection_metric,
    }

    if stage == "prepare":
        return _prepare_supervised_run(**prepare_kwargs)

    if stage == "worker":
        return _run_supervised_worker(
            run_dir=Path(run_dir),
            task_index=int(task_index) if task_index is not None else -1,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
        )

    if stage == "finalize":
        return _finalize_supervised_run(
            run_dir=Path(run_dir),
            score_percentiles=score_percentiles,
        )

    # stage == "all"
    prepared = _prepare_supervised_run(**prepare_kwargs)
    resolved_run_dir = Path(prepared["run_dir"])

    if tuning_executor == "slurm_packed":
        return prepared

    for idx in range(int(prepared["n_tasks"])):
        _run_supervised_worker(
            run_dir=resolved_run_dir,
            task_index=idx,
            n_jobs=n_jobs,
        )

    finalized = _finalize_supervised_run(
        run_dir=resolved_run_dir,
        score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
    )
    return {
        **finalized,
        "tuning_manifest": prepared["tuning_manifest"],
        "n_tasks": prepared["n_tasks"],
    }
