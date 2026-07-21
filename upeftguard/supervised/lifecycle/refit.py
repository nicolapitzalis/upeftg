from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ...utilities.core.run_context import RunContext
from ..data.preparation import (
    load_dataset_group_names_for_run,
    load_features_for_tuning_manifest,
    load_training_inputs_for_run,
)
from ..contracts import SupervisedTaskSpec
from ..data.normalization import slice_supervised_features_for_input
from ..evaluation.prediction import predict_task_outputs
from ..models.registry import (
    TORCH_SEQUENCE_BACKEND,
    TRANSFORMER_MODEL_NAME,
    create,
    model_backend,
)
from ..tasks import task_spec_from_manifest as _task_spec_from_manifest
from ..validation.splits import project_optional_labels_to_binary
from .model_io import load_model, save_model
from .training import input_normalization_from_manifest
from .tuning import (
    _is_dann_model_name,
    _resolve_dann_fit_inputs,
    _resolve_rank_label_fit_kwargs,
    _transformer_refit_epoch_plan,
)


@dataclass(frozen=True)
class FinalRefitResult:
    task_spec: SupervisedTaskSpec
    input_normalization: str
    model_path: Path
    fit_checkpoint_artifacts: dict[str, Any]
    transformer_refit_epoch_plan: dict[str, Any] | None
    train_pool_model_names: list[str]
    train_model_names: list[str]
    train_labels: list[int]
    train_scores: np.ndarray
    validation_model_names: list[str]
    validation_labels: list[int]
    validation_scores: np.ndarray | None
    calibration_indices: np.ndarray
    calibration_model_names: list[str]
    calibration_labels: list[int]
    calibration_binary_labels: np.ndarray | None
    calibration_scores: np.ndarray | None
    training_strategy: str
    selected_fold: dict[str, Any] | None


def fit_winner_and_predict(
    *,
    manifest: dict[str, Any],
    winner: dict[str, Any],
    ctx: RunContext,
) -> FinalRefitResult:
    task_spec = _task_spec_from_manifest(manifest)
    features = load_features_for_tuning_manifest(manifest)
    prepared_inputs = load_training_inputs_for_run(manifest)
    labels_value = prepared_inputs["label_values"].astype(np.int32)
    rank_values = None
    if "rank_values" in prepared_inputs:
        rank_values = prepared_inputs["rank_values"].astype(np.int64)
    model_names = [str(value) for value in prepared_inputs["model_names"].tolist()]
    dataset_group_names = load_dataset_group_names_for_run(manifest)
    input_normalization = input_normalization_from_manifest(manifest)

    train_indices = np.asarray(manifest["data"]["train_indices"], dtype=np.int64)
    train_pool_indices = np.asarray(
        manifest["data"].get("train_pool_indices", manifest["data"]["train_indices"]),
        dtype=np.int64,
    )
    calibration_indices = np.asarray(manifest["data"].get("calibration_indices", []), dtype=np.int64)
    x_train = slice_supervised_features_for_input(
        features,
        train_indices,
        dataset_group_names=dataset_group_names,
        input_normalization=input_normalization,
    )
    y_train = labels_value[train_indices]

    model = create(
        str(winner["model_name"]),
        params=dict(winner["params"]),
        random_state=int(manifest["tuning"]["random_state"]),
        task_spec=task_spec,
    )
    checkpoint_config = manifest["tuning"].get("transformer_checkpoints")
    if str(winner["model_name"]) == TRANSFORMER_MODEL_NAME and isinstance(checkpoint_config, dict):
        model.configure_training_checkpoints(
            checkpoint_dir=Path(checkpoint_config["checkpoint_dir"]),
            interval_seconds=float(checkpoint_config["interval_seconds"]),
            resume_checkpoint=(
                Path(checkpoint_config["resume_checkpoint"]) if checkpoint_config.get("resume_checkpoint") else None
            ),
        )
    refit_epoch_plan = _transformer_refit_epoch_plan(
        winner,
        enabled=bool(manifest["tuning"].get("cv_derived_refit_epochs", False)),
    )
    if refit_epoch_plan is not None:
        configured_max_epochs = int(getattr(model, "max_epochs"))
        epoch_count = min(int(refit_epoch_plan["epoch_count"]), configured_max_epochs)
        model.max_epochs = epoch_count
        refit_epoch_plan["configured_max_epochs"] = configured_max_epochs
        refit_epoch_plan["epoch_count"] = epoch_count

    winner_backend = model_backend(str(winner["model_name"]))
    if winner_backend == TORCH_SEQUENCE_BACKEND:
        fit_kwargs: dict[str, Any] = {}
        fit_indices = np.asarray(train_indices, dtype=np.int64)
        if _is_dann_model_name(str(winner["model_name"])):
            fit_indices, fit_kwargs = _resolve_dann_fit_inputs(
                manifest=manifest,
                train_indices=train_indices,
            )
        fit_kwargs.update(
            _resolve_rank_label_fit_kwargs(
                params=dict(winner["params"]),
                rank_values=rank_values,
                fit_indices=fit_indices,
            )
        )
        x_fit = slice_supervised_features_for_input(
            features,
            fit_indices,
            dataset_group_names=dataset_group_names,
            input_normalization=input_normalization,
        )
        model.fit(
            x_fit,
            labels_value[fit_indices],
            n_jobs=int(manifest["tuning"].get("n_jobs", 1)),
            **fit_kwargs,
        )
    else:
        model.fit(x_train, y_train)

    fit_checkpoint_artifacts = dict(getattr(model, "_fit_checkpoint_artifacts", {}))
    model_path = ctx.models_dir / ("best_model.pt" if winner_backend == TORCH_SEQUENCE_BACKEND else "best_model.joblib")
    save_model(model, model_path)

    train_outputs = predict_task_outputs(model, x_train, task_spec=task_spec)
    train_scores = np.asarray(train_outputs.backdoor_scores, dtype=np.float64)
    train_model_names = [model_names[int(index)] for index in train_indices.tolist()]
    train_labels = project_optional_labels_to_binary(
        [int(value) for value in y_train.tolist()],
        task_spec=task_spec,
    )

    calibration_model_names: list[str] = []
    calibration_labels: list[int] = []
    calibration_scores: np.ndarray | None = None
    calibration_binary_labels: np.ndarray | None = None
    if calibration_indices.size > 0:
        x_calibration = slice_supervised_features_for_input(
            features,
            calibration_indices,
            dataset_group_names=dataset_group_names,
            input_normalization=input_normalization,
        )
        calibration_outputs = predict_task_outputs(model, x_calibration, task_spec=task_spec)
        calibration_scores = np.asarray(calibration_outputs.backdoor_scores, dtype=np.float64)
        calibration_model_names = [model_names[int(index)] for index in calibration_indices.tolist()]
        calibration_task_labels = [int(value) for value in labels_value[calibration_indices].tolist()]
        calibration_labels = project_optional_labels_to_binary(calibration_task_labels, task_spec=task_spec)
        calibration_binary_labels = task_spec.project_known_labels_to_binary(labels_value[calibration_indices])

    return FinalRefitResult(
        task_spec=task_spec,
        input_normalization=input_normalization,
        model_path=model_path,
        fit_checkpoint_artifacts=fit_checkpoint_artifacts,
        transformer_refit_epoch_plan=refit_epoch_plan,
        train_pool_model_names=[model_names[int(index)] for index in train_pool_indices.tolist()],
        train_model_names=train_model_names,
        train_labels=train_labels,
        train_scores=train_scores,
        validation_model_names=[],
        validation_labels=[],
        validation_scores=None,
        calibration_indices=calibration_indices,
        calibration_model_names=calibration_model_names,
        calibration_labels=calibration_labels,
        calibration_binary_labels=calibration_binary_labels,
        calibration_scores=calibration_scores,
        training_strategy="refit_full_training_partition",
        selected_fold=None,
    )


def select_best_cv_fold_model_and_predict(
    *,
    manifest: dict[str, Any],
    winner: dict[str, Any],
    ctx: RunContext,
) -> FinalRefitResult:
    task_spec = _task_spec_from_manifest(manifest)
    fold_results = winner.get("fold_results")
    if not isinstance(fold_results, list):
        raise ValueError("--no-refit requires cross-validation fold results for the winning candidate")
    selectable_folds = [
        row
        for row in fold_results
        if isinstance(row, dict)
        and row.get("selection_metric") is not None
        and row.get("model_artifact")
    ]
    if not selectable_folds:
        raise ValueError("--no-refit could not find a persisted validation-fold model for the winner")
    selected = sorted(
        selectable_folds,
        key=lambda row: (
            -float(row["selection_metric"]),
            int(row.get("cv_group_index", 0)),
            int(row.get("cv_fold_index", 0)),
            int(row.get("cv_random_state", 0)),
        ),
    )[0]

    group_index = int(selected.get("cv_group_index", 0))
    fold_index = int(selected.get("cv_fold_index", 0))
    cv_split_groups = manifest["tuning"].get("cv_split_groups")
    if not isinstance(cv_split_groups, list):
        cv_split_groups = [
            {
                "random_state": int(manifest["tuning"]["random_state"]),
                "cv_splits": list(manifest["tuning"]["cv_splits"]),
            }
        ]
    try:
        split_group = cv_split_groups[group_index]
        split = split_group["cv_splits"][fold_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            "Selected validation fold does not resolve to the saved CV split: "
            f"group={group_index}, fold={fold_index}"
        ) from exc

    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(split["valid_indices"], dtype=np.int64)
    train_pool_indices = np.asarray(
        manifest["data"].get("train_pool_indices", manifest["data"]["train_indices"]),
        dtype=np.int64,
    )
    calibration_indices = np.asarray(manifest["data"].get("calibration_indices", []), dtype=np.int64)

    features = load_features_for_tuning_manifest(manifest)
    prepared_inputs = load_training_inputs_for_run(manifest)
    labels_value = prepared_inputs["label_values"].astype(np.int32)
    model_names = [str(value) for value in prepared_inputs["model_names"].tolist()]
    dataset_group_names = load_dataset_group_names_for_run(manifest)
    input_normalization = input_normalization_from_manifest(manifest)

    source_model_path = Path(str(selected["model_artifact"])).expanduser().resolve()
    if not source_model_path.exists():
        raise FileNotFoundError(f"Selected CV fold model not found: {source_model_path}")
    winner_backend = model_backend(str(winner["model_name"]))
    model_path = ctx.models_dir / (
        "best_model.pt" if winner_backend == TORCH_SEQUENCE_BACKEND else "best_model.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_model_path, model_path)
    model = load_model(model_path)

    x_train = slice_supervised_features_for_input(
        features,
        train_indices,
        dataset_group_names=dataset_group_names,
        input_normalization=input_normalization,
    )
    train_outputs = predict_task_outputs(model, x_train, task_spec=task_spec)
    train_labels = project_optional_labels_to_binary(
        [int(value) for value in labels_value[train_indices].tolist()],
        task_spec=task_spec,
    )

    x_validation = slice_supervised_features_for_input(
        features,
        validation_indices,
        dataset_group_names=dataset_group_names,
        input_normalization=input_normalization,
    )
    validation_outputs = predict_task_outputs(model, x_validation, task_spec=task_spec)
    validation_labels = project_optional_labels_to_binary(
        [int(value) for value in labels_value[validation_indices].tolist()],
        task_spec=task_spec,
    )

    calibration_model_names: list[str] = []
    calibration_labels: list[int] = []
    calibration_scores: np.ndarray | None = None
    calibration_binary_labels: np.ndarray | None = None
    if calibration_indices.size > 0:
        x_calibration = slice_supervised_features_for_input(
            features,
            calibration_indices,
            dataset_group_names=dataset_group_names,
            input_normalization=input_normalization,
        )
        calibration_outputs = predict_task_outputs(model, x_calibration, task_spec=task_spec)
        calibration_scores = np.asarray(calibration_outputs.backdoor_scores, dtype=np.float64)
        calibration_model_names = [model_names[int(index)] for index in calibration_indices.tolist()]
        calibration_task_labels = [int(value) for value in labels_value[calibration_indices].tolist()]
        calibration_labels = project_optional_labels_to_binary(calibration_task_labels, task_spec=task_spec)
        calibration_binary_labels = task_spec.project_known_labels_to_binary(labels_value[calibration_indices])

    selected_fold = {
        "candidate_task_index": int(winner["task_index"]),
        "cv_group_index": group_index,
        "cv_fold_index": fold_index,
        "cv_random_state": int(selected.get("cv_random_state", split_group["random_state"])),
        "selection_metric_name": str(selected.get("selection_metric_name", winner.get("selection_metric_name"))),
        "selection_metric": float(selected["selection_metric"]),
        "n_train": int(train_indices.size),
        "n_validation": int(validation_indices.size),
    }
    for key, value in selected.items():
        if str(key).startswith("validation_") and key != "validation_predictions":
            selected_fold[str(key)] = value

    return FinalRefitResult(
        task_spec=task_spec,
        input_normalization=input_normalization,
        model_path=model_path,
        fit_checkpoint_artifacts={},
        transformer_refit_epoch_plan=None,
        train_pool_model_names=[model_names[int(index)] for index in train_pool_indices.tolist()],
        train_model_names=[model_names[int(index)] for index in train_indices.tolist()],
        train_labels=train_labels,
        train_scores=np.asarray(train_outputs.backdoor_scores, dtype=np.float64),
        validation_model_names=[model_names[int(index)] for index in validation_indices.tolist()],
        validation_labels=validation_labels,
        validation_scores=np.asarray(validation_outputs.backdoor_scores, dtype=np.float64),
        calibration_indices=calibration_indices,
        calibration_model_names=calibration_model_names,
        calibration_labels=calibration_labels,
        calibration_binary_labels=calibration_binary_labels,
        calibration_scores=calibration_scores,
        training_strategy="selected_validation_fold_model",
        selected_fold=selected_fold,
    )
