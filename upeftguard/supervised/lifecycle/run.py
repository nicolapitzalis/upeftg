from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ...features.spectral import DEFAULT_SPECTRAL_ATTENTION_GRANULARITY
from ...utilities.core.run_context import RunContext, create_run_context
from ...utilities.core.serialization import json_ready
from ..validation.cross_validation import (
    CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT,
    CV_STRATEGY_DATASET_LEAVE_ONE_OUT,
    CV_STRATEGY_STRATIFIED,
    append_cv_always_train_indices as _append_cv_always_train_indices,
    build_attack_family_leave_one_out_cv_splits as _build_attack_family_leave_one_out_cv_splits,
    build_cv_splits as _build_cv_splits,
    build_dataset_leave_one_out_cv_splits as _build_dataset_leave_one_out_cv_splits,
    partition_cv_always_train_indices as _partition_cv_always_train_indices,
    sanitize_cv_folds as _sanitize_cv_folds,
    validated_train_label_counts as _validated_train_label_counts,
)
from ..data.normalization import (
    INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD,
)
from ..data.bundles import supervised_feature_row_count as _supervised_feature_row_count
from .preparation import (
    ManifestPartitions,
    PreparedRunFeatures,
    ResolvedPrepareRequest,
    ResolvedSplitOptions,
    load_manifest_partitions,
    prepare_run_features,
    resolve_prepare_request,
    resolve_split_options,
)
from ..models.registry import (
    CNN_1D_DANN_MODEL_NAME,
    candidate_params,
    model_complexity_rank,
    normalization_policy,
)
from ..validation.splits import (
    build_calibration_folder_label_split as _build_calibration_folder_label_split,
    build_calibration_rank_label_split as _build_calibration_rank_label_split,
    build_calibration_stratified_split as _build_calibration_stratified_split,
    build_single_manifest_folder_label_split as _build_single_manifest_folder_label_split,
    build_single_manifest_stratified_split as _build_single_manifest_stratified_split,
    folder_label_stratification_keys as _folder_label_stratification_keys,
    folder_label_stratification_warning as _folder_label_stratification_warning,
    label_count_rows as _label_count_rows,
    rank_label_stratification_keys as _rank_label_stratification_keys,
    rank_label_stratification_warning as _rank_label_stratification_warning,
    split_folder_name as _split_folder_name,
)
from ..tasks import (
    labels_from_items as _labels_from_items,
)
from .tuning import (
    _build_dann_domain_adaptation_config,
    _estimate_total_fit_count,
    _grid_search_warnings,
    _is_torch_sequence_model_name,
    _parse_rank_from_model_name,
    _resolve_tuning_execution_mode,
)
from .run_context import PIPELINE_NAME


SCRIPT_VERSION = "1.0.0"
_PRE_NORMALIZED_INPUT_PARAM = "_upeftguard_pre_normalized_input"


@dataclass(frozen=True)
class PreparedInputs:
    mode: str
    features: PreparedRunFeatures
    label_values: np.ndarray
    label_known: np.ndarray
    labels_raw: list[int | None]


@dataclass(frozen=True)
class TrainingPartitionPlan:
    train_pool_indices: np.ndarray
    fit_train_indices: np.ndarray
    calibration_indices: np.ndarray
    infer_indices: np.ndarray
    split_summary: dict[str, Any]
    calibration_summary: dict[str, Any] | None
    split_warnings: list[str]
    calibration_warnings: list[str]
    domain_adaptation: dict[str, Any] | None
    domain_warnings: list[str]
    domain_labels: np.ndarray | None


@dataclass(frozen=True)
class CandidatePlan:
    tasks: list[dict[str, Any]]
    random_states: list[int]


@dataclass(frozen=True)
class CrossValidationPlan:
    execution_mode: str
    folds_resolved: int
    splits: list[dict[str, Any]]
    split_groups: list[dict[str, Any]]
    stratification_summary: dict[str, Any]
    estimated_total_fits: int
    warnings: list[str]
    grid_warnings: list[str]


def _prepare_inputs(
    *,
    request: ResolvedPrepareRequest,
    partitions: ManifestPartitions,
    ctx: RunContext,
    feature_file: Path | None,
    model_name: str,
    spectral_features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str,
    stream_block_size: int,
    dtype_name: str,
    rank_label_weight_loss: bool,
    hyperparams: Path | None,
) -> PreparedInputs:
    prepared_features = prepare_run_features(
        ctx=ctx,
        feature_file=Path(feature_file),
        all_items=list(partitions.all_items),
        train_items=list(partitions.train_items),
        infer_items=list(partitions.infer_items),
        sample_identities=list(partitions.sample_identities),
        train_indices=np.asarray(partitions.train_indices, dtype=np.int64),
        infer_indices=np.asarray(partitions.infer_indices, dtype=np.int64),
        cv_always_train_indices=np.asarray(partitions.cv_always_train_indices, dtype=np.int64),
        model_name=model_name,
        spectral_features=list(spectral_features or []),
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        input_normalization=request.input_normalization,
        rank_label_weight_loss=rank_label_weight_loss,
        hyperparams=hyperparams,
    )
    label_values, label_known, labels_raw = _labels_from_items(
        prepared_features.all_items,
        task_spec=request.task_spec,
        sample_identities=prepared_features.sample_identities,
    )
    return PreparedInputs(
        mode=partitions.mode,
        features=prepared_features,
        label_values=label_values,
        label_known=label_known,
        labels_raw=labels_raw,
    )


def _build_training_partition_plan(
    *,
    prepared: PreparedInputs,
    split_options: ResolvedSplitOptions,
    ctx: RunContext,
    random_state: int,
    dann_source_rank: int,
    dann_target_adaptation_percent: int,
    dann_lambda_max: float,
    dann_lambda_gamma: float,
) -> TrainingPartitionPlan:
    bundle = prepared.features
    all_items = bundle.all_items
    label_values = prepared.label_values
    label_known = prepared.label_known
    train_indices = bundle.train_indices
    infer_indices = bundle.infer_indices
    cv_always_train_indices = bundle.cv_always_train_indices

    if cv_always_train_indices.size and prepared.mode == "single" and split_options.train_split_percent < 100:
        raise ValueError(
            "cv_always_train cannot be combined with a generated --train-split; "
            "use explicit training and inference manifests"
        )

    split_warnings: list[str] = []
    if prepared.mode == "single":
        if split_options.train_split_percent < 100 and not bool(np.all(label_known)):
            raise ValueError("Single-manifest --train-split requires labels for every sample in the manifest")
        if split_options.split_by_folder:
            train_indices, infer_indices, split_warnings, split_summary = _build_single_manifest_folder_label_split(
                items=all_items,
                labels=label_values,
                train_split_percent=split_options.train_split_percent,
                random_state=random_state,
            )
        else:
            train_indices, infer_indices, split_warnings, split_summary = _build_single_manifest_stratified_split(
                labels=label_values,
                train_split_percent=split_options.train_split_percent,
                random_state=random_state,
            )
    else:
        if split_options.train_split_percent != 100:
            raise ValueError(
                "--train-split values below 100 are only supported for single manifests; "
                "joint manifests already define train and inference partitions"
            )
        split_summary = {
            "strategy": "manifest_defined",
            "split_by_folder": False,
            "requested_train_split_percent": int(split_options.train_split_percent),
            "random_state": int(random_state),
            "n_train": int(train_indices.size),
            "n_inference": int(infer_indices.size),
            "train_label_counts": _label_count_rows(label_values[train_indices]),
            "inference_label_counts": _label_count_rows(label_values[infer_indices][label_known[infer_indices]]),
            "n_inference_unknown_label": int(np.sum(~label_known[infer_indices])),
        }

    if not bool(np.all(label_known[train_indices])):
        raise ValueError("Training samples must all have known labels for supervised learning")

    train_pool_indices = np.asarray(train_indices, dtype=np.int64)
    unknown_cv_always_train = np.setdiff1d(
        cv_always_train_indices,
        train_pool_indices,
        assume_unique=False,
    )
    if unknown_cv_always_train.size:
        raise ValueError("cv_always_train rows must belong to the resolved training partition")

    fit_train_indices = np.asarray(train_pool_indices, dtype=np.int64)
    domain_adaptation, domain_warnings, domain_labels = _build_dann_domain_adaptation_config(
        model_names=bundle.model_names,
        all_items=all_items,
        train_indices=train_pool_indices,
        labels_values=label_values,
        labels_known=label_known,
        source_rank=int(dann_source_rank),
        target_adaptation_percent=int(dann_target_adaptation_percent),
        lambda_max=float(dann_lambda_max),
        lambda_gamma=float(dann_lambda_gamma),
        output_dir=ctx.run_dir,
    )
    if domain_adaptation is not None:
        split_summary = dict(split_summary)
        split_summary["strategy"] = "training_manifest_dann_rank_adversarial"
        split_summary["n_train"] = int(train_pool_indices.size)
        split_summary["train_ranks"] = list(domain_adaptation["train_ranks"])
        split_summary["target_train_ranks"] = list(domain_adaptation["target_train_ranks"])
        split_summary["train_label_counts"] = _label_count_rows(label_values[train_pool_indices])

    calibration_indices = np.asarray([], dtype=np.int64)
    calibration_summary: dict[str, Any] | None = None
    calibration_warnings: list[str] = []
    if split_options.calibration_split_percent is not None:
        calibration_candidate_indices = np.setdiff1d(
            train_pool_indices,
            cv_always_train_indices,
            assume_unique=False,
        )
        if calibration_candidate_indices.size == 0:
            raise ValueError("cv_always_train cannot consume the complete calibration candidate pool")
        if domain_adaptation is not None:
            all_rank_values = np.asarray(
                [_parse_rank_from_model_name(item.model_name) for item in all_items],
                dtype=np.int64,
            )
            rank_label_warning = _rank_label_stratification_warning(
                candidate_indices=calibration_candidate_indices,
                ranks=all_rank_values,
                labels=label_values,
                split_name="calibration",
            )
            if rank_label_warning is None:
                fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = (
                    _build_calibration_rank_label_split(
                        candidate_indices=calibration_candidate_indices,
                        ranks=all_rank_values,
                        labels=label_values,
                        calibration_split_percent=split_options.calibration_split_percent,
                        random_state=random_state,
                    )
                )
            else:
                calibration_warnings.append(rank_label_warning)
                fit_train_indices, calibration_indices, fallback_warnings, calibration_summary = (
                    _build_calibration_stratified_split(
                        candidate_indices=calibration_candidate_indices,
                        labels=label_values,
                        calibration_split_percent=split_options.calibration_split_percent,
                        random_state=random_state,
                    )
                )
                calibration_warnings.extend(fallback_warnings)
        elif split_options.split_by_folder:
            fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = (
                _build_calibration_folder_label_split(
                    items=all_items,
                    candidate_indices=calibration_candidate_indices,
                    labels=label_values,
                    calibration_split_percent=split_options.calibration_split_percent,
                    random_state=random_state,
                )
            )
        else:
            fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = (
                _build_calibration_stratified_split(
                    candidate_indices=calibration_candidate_indices,
                    labels=label_values,
                    calibration_split_percent=split_options.calibration_split_percent,
                    random_state=random_state,
                )
            )
        if cv_always_train_indices.size:
            fit_train_indices = np.union1d(fit_train_indices, cv_always_train_indices)
            if calibration_summary is not None:
                calibration_summary = dict(calibration_summary)
                calibration_summary["n_cv_always_train"] = int(cv_always_train_indices.size)
                calibration_summary["n_fit_train"] = int(fit_train_indices.size)
            calibration_warnings.append(
                f"Kept {int(cv_always_train_indices.size)} cv_always_train rows out of threshold calibration"
            )

    return TrainingPartitionPlan(
        train_pool_indices=train_pool_indices,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        infer_indices=np.asarray(infer_indices, dtype=np.int64),
        split_summary=split_summary,
        calibration_summary=calibration_summary,
        split_warnings=split_warnings,
        calibration_warnings=calibration_warnings,
        domain_adaptation=domain_adaptation,
        domain_warnings=domain_warnings,
        domain_labels=domain_labels,
    )


def _deduplicate_random_states(cv_random_states: list[int] | None, *, random_state: int) -> list[int]:
    requested_states = list(cv_random_states) if cv_random_states else [int(random_state)]
    deduplicated: list[int] = []
    seen: set[int] = set()
    for state in requested_states:
        state_int = int(state)
        if state_int in seen:
            continue
        seen.add(state_int)
        deduplicated.append(state_int)
    return deduplicated or [int(random_state)]


def _build_candidate_plan(
    *,
    prepared: PreparedInputs,
    request: ResolvedPrepareRequest,
    cv_random_states: list[int] | None,
    random_state: int,
    class_weight_loss: bool,
    dann_source_rank: int,
    dann_lambda_max: float,
    dann_lambda_gamma: float,
) -> CandidatePlan:
    bundle = prepared.features
    tasks: list[dict[str, Any]] = []
    task_index = 0
    for selected_model in bundle.model_names:
        complexity = model_complexity_rank(selected_model)
        base_normalization_policy = normalization_policy(selected_model)
        effective_normalization_policy = (
            "pre_normalized_input_passthrough"
            if request.input_normalization == INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD
            else base_normalization_policy
        )
        model_candidate_params = candidate_params(
            selected_model,
            hyperparams=bundle.hyperparam_axes,
        )
        for params in model_candidate_params:
            task_params = dict(params)
            if request.input_normalization == INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD:
                task_params[_PRE_NORMALIZED_INPUT_PARAM] = True
            if _is_torch_sequence_model_name(selected_model) and bool(class_weight_loss):
                task_params["class_weight_loss"] = True
            if _is_torch_sequence_model_name(selected_model) and bool(bundle.rank_label_weight_loss):
                task_params["rank_label_weight_loss"] = True
            if selected_model == CNN_1D_DANN_MODEL_NAME:
                task_params.update(
                    {
                        "source_rank": int(dann_source_rank),
                        "dann_lambda_max": float(dann_lambda_max),
                        "dann_lambda_gamma": float(dann_lambda_gamma),
                    }
                )
            tasks.append(
                {
                    "task_index": int(task_index),
                    "model_name": selected_model,
                    "params": task_params,
                    "complexity_rank": int(complexity),
                    "normalization_policy": str(effective_normalization_policy),
                    "base_normalization_policy": str(base_normalization_policy),
                    "input_normalization": str(request.input_normalization),
                }
            )
            task_index += 1

    return CandidatePlan(
        tasks=tasks,
        random_states=_deduplicate_random_states(cv_random_states, random_state=random_state),
    )


def _build_cross_validation_plan(
    *,
    prepared: PreparedInputs,
    request: ResolvedPrepareRequest,
    split_options: ResolvedSplitOptions,
    partitions: TrainingPartitionPlan,
    candidates: CandidatePlan,
    cv_folds: int,
    cv_derived_refit_epochs: bool,
    no_refit: bool,
) -> CrossValidationPlan:
    bundle = prepared.features
    label_values = prepared.label_values
    _validated_train_label_counts(label_values[partitions.fit_train_indices])
    execution_mode = _resolve_tuning_execution_mode(
        n_tasks=len(candidates.tasks),
        cv_strategy=request.cv_strategy,
        cv_derived_refit_epochs=bool(cv_derived_refit_epochs),
        no_refit=bool(no_refit),
    )
    cv_split_groups: list[dict[str, Any]] = []
    cv_splits: list[dict[str, Any]] = []
    if execution_mode == "singleton_no_cv":
        cv_folds_resolved = 0
        cv_warnings = [
            "Skipped cross-validation because tuning search contains a single candidate; "
            "singleton_no_cv execution mode will fit only during finalize"
        ]
        cv_stratification_summary = {
            "strategy": None,
            "split_by_folder": False,
            "split_by_rank_label": False,
        }
        estimated_total_fits = 1
    else:
        cv_candidate_indices, active_cv_always_train_indices = _partition_cv_always_train_indices(
            train_indices=partitions.fit_train_indices,
            cv_always_train_indices=bundle.cv_always_train_indices,
        )
        cv_stratification_labels = label_values[cv_candidate_indices]
        cv_stratification_summary = {
            "strategy": "label",
            "split_by_folder": False,
            "split_by_rank_label": False,
        }
        cv_warnings: list[str] = []
        if request.cv_strategy == CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT:
            if split_options.split_by_folder:
                cv_warnings.append(
                    "Ignored --split-by-folder for cross-validation because "
                    "cv_strategy=attack_family_leave_one_out was selected"
                )
            cv_folds_resolved = 0
            for state in candidates.random_states:
                splits, strategy_summary, strategy_warnings = _build_attack_family_leave_one_out_cv_splits(
                    train_indices=cv_candidate_indices,
                    labels=label_values,
                    sample_identities=bundle.sample_identities,
                    random_state=int(state),
                    task_spec=request.task_spec,
                )
                splits = _append_cv_always_train_indices(
                    splits=splits,
                    cv_always_train_indices=active_cv_always_train_indices,
                    labels=label_values,
                    task_spec=request.task_spec,
                )
                cv_split_groups.append(
                    {
                        "random_state": int(state),
                        "cv_splits": splits,
                    }
                )
                cv_folds_resolved = len(splits)
                cv_stratification_summary = strategy_summary
                cv_warnings.extend(strategy_warnings)
        elif request.cv_strategy == CV_STRATEGY_DATASET_LEAVE_ONE_OUT:
            if split_options.split_by_folder:
                cv_warnings.append(
                    "Ignored --split-by-folder for cross-validation because "
                    "cv_strategy=dataset_leave_one_out was selected"
                )
            cv_folds_resolved = 0
            for state in candidates.random_states:
                splits, strategy_summary, strategy_warnings = _build_dataset_leave_one_out_cv_splits(
                    train_indices=cv_candidate_indices,
                    labels=label_values,
                    sample_identities=bundle.sample_identities,
                )
                splits = _append_cv_always_train_indices(
                    splits=splits,
                    cv_always_train_indices=active_cv_always_train_indices,
                    labels=label_values,
                    task_spec=request.task_spec,
                )
                cv_split_groups.append(
                    {
                        "random_state": int(state),
                        "cv_splits": splits,
                    }
                )
                cv_folds_resolved = len(splits)
                cv_stratification_summary = strategy_summary
                cv_warnings.extend(strategy_warnings)
        elif split_options.split_by_folder:
            folder_label_warning = _folder_label_stratification_warning(
                items=bundle.all_items,
                candidate_indices=cv_candidate_indices,
                labels=label_values,
                split_name="cross-validation",
            )
            if folder_label_warning is None:
                cv_stratification_labels = _folder_label_stratification_keys(
                    items=bundle.all_items,
                    candidate_indices=cv_candidate_indices,
                    labels=label_values,
                )
                cv_stratification_summary = {
                    "strategy": "folder_label",
                    "split_by_folder": True,
                    "split_by_rank_label": False,
                    "folder_count": int(
                        len({_split_folder_name(bundle.all_items[int(idx)]) for idx in cv_candidate_indices.tolist()})
                    ),
                }
            else:
                cv_warnings.append(folder_label_warning)
        elif partitions.domain_adaptation is not None:
            all_rank_values = np.asarray(
                [_parse_rank_from_model_name(item.model_name) for item in bundle.all_items],
                dtype=np.int64,
            )
            rank_label_warning = _rank_label_stratification_warning(
                candidate_indices=cv_candidate_indices,
                ranks=all_rank_values,
                labels=label_values,
                split_name="cross-validation",
            )
            if rank_label_warning is None:
                cv_stratification_labels = _rank_label_stratification_keys(
                    candidate_indices=cv_candidate_indices,
                    ranks=all_rank_values,
                    labels=label_values,
                )
                cv_stratification_summary = {
                    "strategy": "rank_label",
                    "split_by_folder": False,
                    "split_by_rank_label": True,
                }
            else:
                cv_warnings.append(rank_label_warning)
        if request.cv_strategy == CV_STRATEGY_STRATIFIED:
            cv_folds_resolved, fold_warnings = _sanitize_cv_folds(cv_stratification_labels, cv_folds)
            cv_warnings.extend(fold_warnings)
            for state in candidates.random_states:
                splits = _build_cv_splits(
                    train_indices=cv_candidate_indices,
                    train_labels=cv_stratification_labels,
                    cv_folds=cv_folds_resolved,
                    random_state=int(state),
                )
                splits = _append_cv_always_train_indices(
                    splits=splits,
                    cv_always_train_indices=active_cv_always_train_indices,
                    labels=label_values,
                    task_spec=request.task_spec,
                )
                cv_split_groups.append(
                    {
                        "random_state": int(state),
                        "cv_splits": splits,
                    }
                )
        if active_cv_always_train_indices.size:
            anchor_dataset_names = sorted(
                {
                    str(bundle.dataset_group_names[int(index)])
                    for index in active_cv_always_train_indices.tolist()
                }
            )
            cv_stratification_summary = dict(cv_stratification_summary)
            cv_stratification_summary.update(
                {
                    "n_fold_eligible": int(cv_candidate_indices.size),
                    "n_cv_always_train": int(active_cv_always_train_indices.size),
                    "cv_always_train_dataset_names": anchor_dataset_names,
                }
            )
            cv_warnings.append(
                f"Pinned {int(active_cv_always_train_indices.size)} cv_always_train rows into every "
                "CV training fold; these rows are never used for validation"
            )
        cv_splits = list(cv_split_groups[0]["cv_splits"])
        estimated_total_fits = _estimate_total_fit_count(
            n_tasks=len(candidates.tasks),
            cv_split_groups=cv_split_groups,
        )

    return CrossValidationPlan(
        execution_mode=execution_mode,
        folds_resolved=int(cv_folds_resolved),
        splits=cv_splits,
        split_groups=cv_split_groups,
        stratification_summary=cv_stratification_summary,
        estimated_total_fits=int(estimated_total_fits),
        warnings=cv_warnings,
        grid_warnings=_grid_search_warnings(
            n_tasks=len(candidates.tasks),
            estimated_total_fits=estimated_total_fits,
        ),
    )


def _write_prepared_arrays(
    *,
    ctx: RunContext,
    prepared: PreparedInputs,
    domain_labels: np.ndarray | None,
) -> Path:
    bundle = prepared.features
    inputs_path = ctx.run_dir / ".work" / "prepared_arrays.npz"
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    input_arrays: dict[str, Any] = {
        "label_values": prepared.label_values.astype(np.int32),
        "label_known": prepared.label_known.astype(np.int8),
        "model_names": np.asarray([item.model_name for item in bundle.all_items], dtype=str),
        "dataset_groups": np.asarray(bundle.dataset_group_names, dtype=str),
    }
    if bool(bundle.rank_label_weight_loss):
        rank_values = np.asarray(
            [_parse_rank_from_model_name(item.model_name) for item in bundle.all_items],
            dtype=np.int64,
        )
        input_arrays["rank_values"] = rank_values
    if domain_labels is not None:
        input_arrays["domain_labels"] = np.asarray(domain_labels, dtype=np.int64)
    np.savez_compressed(inputs_path, **input_arrays)
    return inputs_path


def _resolve_cpus_per_worker(cpus_per_worker: str, *, n_jobs: int) -> int:
    resolved = (
        int(cpus_per_worker)
        if cpus_per_worker != "auto"
        else max(1, int(n_jobs) if int(n_jobs) > 0 else int(os.cpu_count() or 1))
    )
    if resolved <= 0:
        raise ValueError("Resolved CPUs per worker must be positive")
    return int(resolved)


def _build_tuning_manifest(
    *,
    ctx: RunContext,
    request: ResolvedPrepareRequest,
    split_options: ResolvedSplitOptions,
    prepared: PreparedInputs,
    partitions: TrainingPartitionPlan,
    candidates: CandidatePlan,
    cross_validation: CrossValidationPlan,
    inputs_path: Path,
    dataset_root: Path,
    model_name: str,
    tuning_executor: str,
    n_jobs: int,
    random_state: int,
    cv_folds: int,
    cv_derived_refit_epochs: bool,
    no_refit: bool,
    class_weight_loss: bool,
    transformer_checkpoint_dir: Path | None,
    transformer_checkpoint_interval_seconds: float | None,
    transformer_resume_checkpoint: Path | None,
    slurm_partition: str,
    cpus_per_worker: int,
) -> dict[str, Any]:
    bundle = prepared.features
    artifacts = bundle.artifacts
    accepted_fprs = split_options.accepted_fprs
    return {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(ctx.run_dir),
        "mode": prepared.mode,
        "manifest_json": str(request.manifest_json.expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "task": request.task_spec.to_dict(),
        "data": {
            "n_samples": int(_supervised_feature_row_count(bundle.features)),
            "train_indices": [int(x) for x in partitions.fit_train_indices.tolist()],
            "train_pool_indices": [int(x) for x in partitions.train_pool_indices.tolist()],
            "cv_always_train_indices": [int(x) for x in bundle.cv_always_train_indices.tolist()],
            "cv_always_train_model_names": [
                str(bundle.all_items[int(index)].model_name) for index in bundle.cv_always_train_indices.tolist()
            ],
            "calibration_indices": [int(x) for x in partitions.calibration_indices.tolist()],
            "infer_indices": [int(x) for x in partitions.infer_indices.tolist()],
            "split": partitions.split_summary,
            "calibration_split": partitions.calibration_summary,
            "feature_path": str(Path(artifacts["feature_path"])),
            "feature_loading_mode": bundle.feature_loading_mode,
            "inputs_path": str(inputs_path),
            "dataset_group_counts": dict(bundle.dataset_group_counts),
        },
        "extractor": {
            "name": "spectral",
            "params": bundle.feature_params,
            "metadata": bundle.extractor_metadata,
            "warnings": bundle.extractor_warnings,
            "metadata_path": str(Path(artifacts["metadata_path"])),
        },
        "domain_adaptation": partitions.domain_adaptation,
        "input_normalization": bundle.input_normalization_config,
        "tuning": {
            "executor": tuning_executor,
            "model_name": model_name,
            "model_names": bundle.model_names,
            "metric": str(request.selection_metric_name),
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
            "train_split_percent": int(split_options.train_split_percent),
            "split_by_folder": bool(split_options.split_by_folder),
            "cv_strategy": str(request.cv_strategy),
            "input_normalization": str(request.input_normalization),
            "cv_derived_refit_epochs": bool(cv_derived_refit_epochs),
            "no_refit": bool(no_refit),
            "class_weight_loss": bool(class_weight_loss),
            "rank_label_weight_loss": bool(bundle.rank_label_weight_loss),
            "calibration_split_percent": (
                int(split_options.calibration_split_percent)
                if split_options.calibration_split_percent is not None
                else None
            ),
            "cv_random_states": [int(x) for x in candidates.random_states],
            "cv_folds_requested": int(cv_folds),
            "cv_folds_resolved": int(cross_validation.folds_resolved),
            "execution_mode": cross_validation.execution_mode,
            "estimated_total_fits": int(cross_validation.estimated_total_fits),
            "cv_stratification": cross_validation.stratification_summary,
            "hyperparams": bundle.hyperparams_info,
            "transformer_checkpoints": (
                {
                    "checkpoint_dir": str(Path(transformer_checkpoint_dir).expanduser().resolve()),
                    "interval_seconds": float(transformer_checkpoint_interval_seconds),
                    "resume_checkpoint": (
                        str(Path(transformer_resume_checkpoint).expanduser().resolve())
                        if transformer_resume_checkpoint is not None
                        else None
                    ),
                }
                if transformer_checkpoint_dir is not None
                else None
            ),
            "cv_splits": cross_validation.splits,
            "cv_split_groups": cross_validation.split_groups,
            "tasks": candidates.tasks,
        },
        "threshold_selection": {
            "method": (
                "maximize_recall_subject_to_fpr" if split_options.calibration_split_percent is not None else None
            ),
            "calibration_split_percent": (
                int(split_options.calibration_split_percent)
                if split_options.calibration_split_percent is not None
                else None
            ),
            "accepted_fprs": list(accepted_fprs) if accepted_fprs is not None else None,
            "accepted_fpr": (
                float(accepted_fprs[0]) if accepted_fprs is not None and len(accepted_fprs) == 1 else None
            ),
            "split_by_folder": bool(
                split_options.split_by_folder and split_options.calibration_split_percent is not None
            ),
        },
        "runtime": {
            "slurm_partition": slurm_partition,
            "cpus_per_worker": int(cpus_per_worker),
        },
        "warnings": (
            cross_validation.warnings
            + partitions.split_warnings
            + partitions.calibration_warnings
            + partitions.domain_warnings
            + list(bundle.extractor_warnings)
            + cross_validation.grid_warnings
        ),
        "labels_preview": prepared.labels_raw,
    }


def _write_tuning_manifest(*, ctx: RunContext, tuning_manifest: dict[str, Any]) -> Path:
    tuning_manifest_path = ctx.run_dir / ".work" / "tuning.json"
    tuning_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tuning_manifest_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(tuning_manifest), f, indent=2)
    return tuning_manifest_path


def prepare_supervised_run(
    *,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str,
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
    input_normalization: str,
    cv_derived_refit_epochs: bool,
    random_state: int,
    train_split_percent: int,
    calibration_split_percent: int | None,
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
    split_by_folder: bool | None,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float],
    feature_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    cpus_per_worker: str,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
    hyperparams: Path | None,
    transformer_checkpoint_dir: Path | None,
    transformer_checkpoint_interval_seconds: float | None,
    transformer_resume_checkpoint: Path | None,
    dann_source_rank: int,
    dann_target_adaptation_percent: int,
    dann_lambda_max: float,
    dann_lambda_gamma: float,
    class_weight_loss: bool = False,
    rank_label_weight_loss: bool = False,
    selection_metric: str | None = None,
    prepared_run_dir: Path | None = None,
    no_refit: bool = False,
) -> dict[str, Any]:
    if bool(no_refit) and bool(cv_derived_refit_epochs):
        raise ValueError("--no-refit cannot be combined with --cv-derived-refit-epochs")
    request = resolve_prepare_request(
        manifest_json=manifest_json,
        feature_file=feature_file,
        spectral_features=spectral_features,
        model_name=model_name,
        transformer_checkpoint_dir=transformer_checkpoint_dir,
        transformer_checkpoint_interval_seconds=transformer_checkpoint_interval_seconds,
        transformer_resume_checkpoint=transformer_resume_checkpoint,
        class_weight_loss=class_weight_loss,
        rank_label_weight_loss=rank_label_weight_loss,
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
        cv_strategy=cv_strategy,
        input_normalization=input_normalization,
        selection_metric=selection_metric,
    )
    manifest_partitions = load_manifest_partitions(request.manifest_json)
    split_options = resolve_split_options(
        train_split_percent=train_split_percent,
        calibration_split_percent=calibration_split_percent,
        accepted_fpr=accepted_fpr,
        split_by_folder=split_by_folder,
    )
    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_id,
        run_dir=prepared_run_dir,
        create_directories=("models", "reports"),
    )
    prepared = _prepare_inputs(
        request=request,
        partitions=manifest_partitions,
        ctx=ctx,
        feature_file=feature_file,
        model_name=model_name,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        rank_label_weight_loss=rank_label_weight_loss,
        hyperparams=hyperparams,
    )
    training_partitions = _build_training_partition_plan(
        prepared=prepared,
        split_options=split_options,
        ctx=ctx,
        random_state=random_state,
        dann_source_rank=dann_source_rank,
        dann_target_adaptation_percent=dann_target_adaptation_percent,
        dann_lambda_max=dann_lambda_max,
        dann_lambda_gamma=dann_lambda_gamma,
    )
    candidates = _build_candidate_plan(
        prepared=prepared,
        request=request,
        cv_random_states=cv_random_states,
        random_state=random_state,
        class_weight_loss=class_weight_loss,
        dann_source_rank=dann_source_rank,
        dann_lambda_max=dann_lambda_max,
        dann_lambda_gamma=dann_lambda_gamma,
    )

    cross_validation = _build_cross_validation_plan(
        prepared=prepared,
        request=request,
        split_options=split_options,
        partitions=training_partitions,
        candidates=candidates,
        cv_folds=cv_folds,
        cv_derived_refit_epochs=cv_derived_refit_epochs,
        no_refit=no_refit,
    )

    inputs_path = _write_prepared_arrays(
        ctx=ctx,
        prepared=prepared,
        domain_labels=training_partitions.domain_labels,
    )
    resolved_cpus_per_worker = _resolve_cpus_per_worker(cpus_per_worker, n_jobs=n_jobs)

    task_dir = ctx.run_dir / ".work" / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    tuning_manifest = _build_tuning_manifest(
        ctx=ctx,
        request=request,
        split_options=split_options,
        prepared=prepared,
        partitions=training_partitions,
        candidates=candidates,
        cross_validation=cross_validation,
        inputs_path=inputs_path,
        dataset_root=dataset_root,
        model_name=model_name,
        tuning_executor=tuning_executor,
        n_jobs=n_jobs,
        random_state=random_state,
        cv_folds=cv_folds,
        cv_derived_refit_epochs=cv_derived_refit_epochs,
        no_refit=no_refit,
        class_weight_loss=class_weight_loss,
        transformer_checkpoint_dir=transformer_checkpoint_dir,
        transformer_checkpoint_interval_seconds=transformer_checkpoint_interval_seconds,
        transformer_resume_checkpoint=transformer_resume_checkpoint,
        slurm_partition=slurm_partition,
        cpus_per_worker=resolved_cpus_per_worker,
    )
    tuning_manifest_path = _write_tuning_manifest(
        ctx=ctx,
        tuning_manifest=tuning_manifest,
    )

    return {
        "run_dir": str(ctx.run_dir),
        "tuning_manifest": str(tuning_manifest_path),
        "n_tasks": len(candidates.tasks),
        "task_dir": str(task_dir),
        "slurm_partition": slurm_partition,
        "cpus_per_worker": int(resolved_cpus_per_worker),
        "execution_mode": cross_validation.execution_mode,
        "warnings": tuning_manifest["warnings"],
    }
