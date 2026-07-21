from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal environments
    joblib = None

from ..validation.cross_validation import (
    CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT,
    CV_STRATEGY_DATASET_LEAVE_ONE_OUT,
)
from ...reporting import PredictionPartition, write_prediction_partition
from ..data.bundles import supervised_feature_row_count as _supervised_feature_row_count
from ..evaluation.scoring import evaluate_fold_predictions
from ..contracts import SupervisedFeatureBundle, SupervisedTaskSpec
from ..data.normalization import (
    INPUT_NORMALIZATION_NONE,
    resolve_input_normalization as _resolve_input_normalization,
    slice_supervised_features_for_input as _slice_supervised_features_for_input,
)
from ..evaluation.prediction import predict_task_outputs as _predict_task_outputs
from ..models.registry import (
    CNN_1D_DANN_MODEL_NAME,
    CNN_1D_MODEL_NAME,
    TORCH_SEQUENCE_BACKEND,
    TRANSFORMER_MODEL_NAME,
    create,
    model_backend,
)
from .model_io import save_model


DANN_DEFAULT_SOURCE_RANK = 256
DANN_DEFAULT_LAMBDA_MAX = 1.0
DANN_DEFAULT_LAMBDA_GAMMA = 10.0
DANN_DEFAULT_TARGET_ADAPTATION_PERCENT = 80
_RANK_TOKEN_RE = re.compile(r"(?:^|_)rank(\d+(?:\.\d+)?)(?:_|$)", re.IGNORECASE)


def _is_dann_model_name(model_name: str) -> bool:
    return str(model_name) == CNN_1D_DANN_MODEL_NAME


def _is_torch_sequence_model_name(model_name: str) -> bool:
    return str(model_name) in {
        CNN_1D_MODEL_NAME,
        CNN_1D_DANN_MODEL_NAME,
        TRANSFORMER_MODEL_NAME,
    }


def _parse_rank_from_model_name(model_name: str) -> int:
    match = _RANK_TOKEN_RE.search(str(model_name))
    if match is None:
        raise ValueError(f"Could not parse a rank token from model name: {model_name!r}")
    return int(float(match.group(1)))


def _domain_class_name_for_rank(rank: int) -> str:
    return f"rank_{int(rank)}"


def _build_dann_domain_adaptation_config(
    *,
    model_names: list[str],
    all_items: list[Any],
    train_indices: np.ndarray,
    labels_values: np.ndarray,
    labels_known: np.ndarray,
    source_rank: int,
    target_adaptation_percent: int,
    lambda_max: float,
    lambda_gamma: float,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, list[str], np.ndarray | None]:
    if CNN_1D_DANN_MODEL_NAME not in model_names:
        return None, [], None
    ranks = np.asarray([_parse_rank_from_model_name(item.model_name) for item in all_items], dtype=np.int64)
    train_indices_np = np.asarray(train_indices, dtype=np.int64)
    if train_indices_np.size == 0:
        raise ValueError("cnn_1d_dann requires at least one manifest train row")

    labels_known_np = np.asarray(labels_known, dtype=bool).reshape(-1)
    if not bool(np.all(labels_known_np[train_indices_np])):
        raise ValueError("cnn_1d_dann requires known labels for every manifest train row")

    observed_train_ranks = sorted({int(ranks[int(i)]) for i in train_indices_np.tolist()})
    if int(source_rank) not in observed_train_ranks:
        raise ValueError(
            "cnn_1d_dann requires the configured source rank in the manifest train split; "
            f"{int(source_rank)}; observed train ranks={observed_train_ranks}"
        )
    target_train_ranks = [rank for rank in observed_train_ranks if rank != int(source_rank)]
    if not target_train_ranks:
        raise ValueError("cnn_1d_dann requires at least one non-source rank in the manifest train split")
    train_ranks = [int(source_rank), *target_train_ranks]

    domain_rank_values = [int(source_rank), *target_train_ranks]
    rank_to_domain_index = {int(rank): int(idx) for idx, rank in enumerate(domain_rank_values)}
    domain_labels = np.full(int(ranks.shape[0]), -1, dtype=np.int64)
    for idx in train_indices_np.tolist():
        domain_labels[int(idx)] = int(rank_to_domain_index[int(ranks[int(idx)])])
    labels_np = np.asarray(labels_values, dtype=np.int32).reshape(-1)
    source_train_indices = np.asarray(
        [int(i) for i in train_indices_np.tolist() if int(ranks[int(i)]) == int(source_rank)],
        dtype=np.int64,
    )
    target_train_indices = np.asarray(
        [int(i) for i in train_indices_np.tolist() if int(ranks[int(i)]) != int(source_rank)],
        dtype=np.int64,
    )

    config = {
        "enabled": True,
        "model_name": CNN_1D_DANN_MODEL_NAME,
        "source_rank": int(source_rank),
        "train_ranks": [int(x) for x in train_ranks],
        "target_train_ranks": [int(x) for x in target_train_ranks],
        "domain_rank_values": [int(x) for x in domain_rank_values],
        "domain_class_names": [_domain_class_name_for_rank(rank) for rank in domain_rank_values],
        "rank_to_domain_index": {str(rank): int(idx) for rank, idx in rank_to_domain_index.items()},
        "source_domain_index": int(rank_to_domain_index[int(source_rank)]),
        "train_indices": [int(x) for x in train_indices_np.tolist()],
        "source_train_indices": [int(x) for x in source_train_indices.tolist()],
        "target_train_indices": [int(x) for x in target_train_indices.tolist()],
        "train_rank_label_counts": [
            {
                "rank": int(rank),
                "label": int(label),
                "count": int(
                    np.sum((ranks[train_indices_np] == int(rank)) & (labels_np[train_indices_np] == int(label)))
                ),
            }
            for rank in train_ranks
            for label in sorted(
                {int(labels_np[int(i)]) for i in train_indices_np.tolist() if int(ranks[int(i)]) == int(rank)}
            )
        ],
        "domain_labels_path": str(output_dir / ".work" / "prepared_arrays.npz"),
        "label_loss_scope": "all_training_ranks",
        "domain_loss": "multiclass_rank_cross_entropy",
        "domain_loss_weight": 1.0,
        "lambda_schedule": {
            "type": "dann_paper_logistic",
            "lambda_max": float(lambda_max),
            "gamma": float(lambda_gamma),
        },
        "learning_rate_schedule": {
            "type": "fixed",
        },
    }
    warnings = [
        "cnn_1d_dann is using manifest-defined rank-adversarial supervised training: every train "
        "row contributes label loss and rank-domain loss"
    ]
    if int(target_adaptation_percent) != int(DANN_DEFAULT_TARGET_ADAPTATION_PERCENT):
        warnings.append(
            "Ignored deprecated --dann-target-adaptation-percent because cnn_1d_dann derives "
            "seen ranks from the training manifest"
        )
    return config, warnings, domain_labels


def _resolve_dann_fit_inputs(
    *,
    manifest: dict[str, Any],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    domain_cfg = manifest.get("domain_adaptation")
    if not isinstance(domain_cfg, dict) or not bool(domain_cfg.get("enabled")):
        raise ValueError("cnn_1d_dann task is missing prepared domain_adaptation metadata")

    labels_path = domain_cfg.get("domain_labels_path")
    if not isinstance(labels_path, str) or not labels_path:
        raise ValueError("cnn_1d_dann domain_adaptation metadata is missing domain_labels_path")
    loaded_labels = np.load(labels_path)
    if isinstance(loaded_labels, np.lib.npyio.NpzFile):
        try:
            domain_labels_all = np.asarray(loaded_labels["domain_labels"], dtype=np.int64)
        finally:
            loaded_labels.close()
    else:
        domain_labels_all = np.asarray(loaded_labels, dtype=np.int64)
    fit_indices = np.asarray(train_indices, dtype=np.int64)
    if fit_indices.size == 0:
        raise ValueError("cnn_1d_dann resolved zero training rows for fitting")
    if int(np.max(fit_indices)) >= int(domain_labels_all.shape[0]):
        raise ValueError("cnn_1d_dann training indices exceed the domain-label array length")
    if bool(np.any(domain_labels_all[fit_indices] < 0)):
        raise ValueError(
            "cnn_1d_dann resolved fit rows without training-rank domain labels; "
            "check that only manifest train rows are used for fitting"
        )

    fit_kwargs = {
        "domain_labels": domain_labels_all[fit_indices],
        "domain_class_names": [str(x) for x in domain_cfg.get("domain_class_names", [])],
        "domain_rank_values": [int(x) for x in domain_cfg.get("domain_rank_values", [])],
    }
    return fit_indices, fit_kwargs


def _resolve_rank_label_fit_kwargs(
    *,
    params: dict[str, Any],
    rank_values: np.ndarray | None,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    if not bool(params.get("rank_label_weight_loss", False)):
        return {}
    if rank_values is None:
        raise ValueError("rank_label_weight_loss requires prepared rank values")
    ranks_np = np.asarray(rank_values, dtype=np.int64).reshape(-1)
    fit_indices_np = np.asarray(fit_indices, dtype=np.int64)
    if fit_indices_np.size == 0:
        raise ValueError("rank_label_weight_loss resolved zero training rows for fitting")
    if int(np.max(fit_indices_np)) >= int(ranks_np.shape[0]):
        raise ValueError("rank_label_weight_loss training indices exceed the rank-value array length")
    return {"rank_labels": ranks_np[fit_indices_np]}


def _resolve_tuning_execution_mode(
    *,
    n_tasks: int,
    cv_strategy: str,
    cv_derived_refit_epochs: bool,
    no_refit: bool = False,
) -> str:
    if int(n_tasks) <= 0:
        raise ValueError("supervised tuning requires at least one candidate task")
    if int(n_tasks) > 1:
        return "cross_validation"
    if str(cv_strategy) in {
        CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT,
        CV_STRATEGY_DATASET_LEAVE_ONE_OUT,
    }:
        return "cross_validation"
    if bool(cv_derived_refit_epochs):
        return "cross_validation"
    if bool(no_refit):
        return "cross_validation"
    return "singleton_no_cv"


def _estimate_total_fit_count(
    *,
    n_tasks: int,
    cv_split_groups: list[dict[str, Any]],
) -> int:
    total_splits = sum(len(group.get("cv_splits", [])) for group in cv_split_groups)
    return int(n_tasks * total_splits)


def _grid_search_warnings(
    *,
    n_tasks: int,
    estimated_total_fits: int,
) -> list[str]:
    warnings: list[str] = []
    if n_tasks > 100:
        warnings.append(
            "Large supervised grid search: "
            f"{n_tasks} tasks and approximately {estimated_total_fits} total model fits across CV splits"
        )
    return warnings


def _evaluate_fold(
    *,
    features: np.ndarray | SupervisedFeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    model_name: str,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
    task_spec: SupervisedTaskSpec,
    selection_metric_name: str,
    domain_adaptation: dict[str, Any] | None = None,
    rank_values: np.ndarray | None = None,
    dataset_group_names: list[str] | None = None,
    input_normalization: str = INPUT_NORMALIZATION_NONE,
    model_names: list[str] | None = None,
    artifact_dir: Path | None = None,
    persist_model: bool = False,
) -> dict[str, Any]:
    model = create(model_name, params=params, random_state=random_state, task_spec=task_spec)
    resolved_input_normalization = _resolve_input_normalization(input_normalization)
    resolved_dataset_group_names = (
        list(dataset_group_names)
        if dataset_group_names is not None
        else ["all"] * _supervised_feature_row_count(features)
    )
    valid_features = _slice_supervised_features_for_input(
        features,
        valid_indices,
        dataset_group_names=resolved_dataset_group_names,
        input_normalization=resolved_input_normalization,
    )
    backend = model_backend(model_name)
    if backend == TORCH_SEQUENCE_BACKEND:
        fit_kwargs: dict[str, Any] = {}
        fit_indices = np.asarray(train_indices, dtype=np.int64)
        if _is_dann_model_name(model_name):
            fit_indices, fit_kwargs = _resolve_dann_fit_inputs(
                manifest={"domain_adaptation": domain_adaptation},
                train_indices=train_indices,
            )
        fit_kwargs.update(
            _resolve_rank_label_fit_kwargs(
                params=params,
                rank_values=rank_values,
                fit_indices=fit_indices,
            )
        )
        train_features = _slice_supervised_features_for_input(
            features,
            fit_indices,
            dataset_group_names=resolved_dataset_group_names,
            input_normalization=resolved_input_normalization,
        )
        model.fit(
            train_features,
            labels[fit_indices],
            validation_data=(valid_features, labels[valid_indices]),
            n_jobs=n_jobs,
            **fit_kwargs,
        )
    else:
        train_features = _slice_supervised_features_for_input(
            features,
            train_indices,
            dataset_group_names=resolved_dataset_group_names,
            input_normalization=resolved_input_normalization,
        )
        model.fit(train_features, labels[train_indices])
    training_epoch_metadata = _transformer_training_epoch_metadata(
        model_name=model_name,
        model=model,
    )
    outputs = _predict_task_outputs(model, valid_features, task_spec=task_spec)
    result = {
        "n_train": int(train_indices.size),
        "n_valid": int(valid_indices.size),
        "selection_metric_name": str(selection_metric_name),
        "input_normalization": str(resolved_input_normalization),
        **training_epoch_metadata,
    }
    result.update(
        evaluate_fold_predictions(
            labels=labels,
            valid_indices=valid_indices,
            outputs=outputs,
            task_spec=task_spec,
            selection_metric_name=selection_metric_name,
        )
    )
    if artifact_dir is not None:
        if model_names is None:
            raise ValueError("model_names are required when writing CV validation artifacts")
        artifact_dir = Path(artifact_dir).expanduser().resolve()
        validation_path = write_prediction_partition(
            artifact_dir / "validation.csv",
            PredictionPartition(
                model_names=[model_names[int(index)] for index in valid_indices.tolist()],
                labels=task_spec.project_known_labels_to_binary(labels[valid_indices]).tolist(),
                scores=np.asarray(outputs.backdoor_scores, dtype=np.float64),
            ),
        )
        result["validation_predictions"] = str(validation_path)
        if persist_model:
            model_suffix = ".pt" if backend == TORCH_SEQUENCE_BACKEND else ".joblib"
            model_path = artifact_dir / f"model{model_suffix}"
            save_model(model, model_path)
            result["model_artifact"] = str(model_path)
    return result


def _transformer_training_epoch_metadata(
    *,
    model_name: str,
    model: Any,
) -> dict[str, int]:
    if str(model_name) != TRANSFORMER_MODEL_NAME:
        return {}

    fit_summary = getattr(model, "_fit_summary", None)
    if not isinstance(fit_summary, dict):
        return {}
    try:
        best_epoch = int(fit_summary["best_epoch"])
        epochs_ran = int(fit_summary["epochs_ran"])
    except (KeyError, TypeError, ValueError):
        return {}
    if best_epoch < 0 or epochs_ran <= 0:
        return {}

    return {
        "training_best_epoch": int(best_epoch),
        "training_best_epoch_count": int(best_epoch + 1),
        "training_epochs_ran": int(epochs_ran),
    }


def _transformer_refit_epoch_plan(
    winner: dict[str, Any],
    *,
    enabled: bool = False,
) -> dict[str, Any] | None:
    if not bool(enabled):
        return None
    if str(winner.get("model_name")) != TRANSFORMER_MODEL_NAME:
        return None

    fold_results = winner.get("fold_results")
    if not isinstance(fold_results, list):
        return None

    epoch_counts: list[int] = []
    for row in fold_results:
        if not isinstance(row, dict):
            continue
        try:
            epoch_count = int(row["training_best_epoch_count"])
        except (KeyError, TypeError, ValueError):
            continue
        if epoch_count > 0:
            epoch_counts.append(int(epoch_count))
    if not epoch_counts:
        return None

    median_epoch_count = float(np.median(np.asarray(epoch_counts, dtype=np.float64)))
    return {
        "strategy": "median_cv_validation_best_epoch_count",
        "epoch_count": int(np.ceil(median_epoch_count)),
        "median_epoch_count": float(median_epoch_count),
        "cv_best_epoch_counts": [int(value) for value in epoch_counts],
        "cv_fold_count": int(len(epoch_counts)),
    }


def _evaluate_candidate(
    *,
    features: np.ndarray | SupervisedFeatureBundle,
    labels: np.ndarray,
    task: dict[str, Any],
    cv_split_groups: list[dict[str, Any]],
    n_jobs: int,
    task_spec: SupervisedTaskSpec,
    selection_metric_name: str,
    domain_adaptation: dict[str, Any] | None = None,
    rank_values: np.ndarray | None = None,
    dataset_group_names: list[str] | None = None,
    input_normalization: str = INPUT_NORMALIZATION_NONE,
    model_names: list[str] | None = None,
    artifact_dir: Path | None = None,
    persist_fold_models: bool = False,
) -> dict[str, Any]:
    eval_jobs: list[tuple[int, int, int, dict[str, Any]]] = []
    for group_index, group in enumerate(cv_split_groups):
        seed = int(group["random_state"])
        for fold_index, split in enumerate(group["cv_splits"]):
            eval_jobs.append((group_index, fold_index, seed, split))

    def fold_artifact_dir(group_index: int, fold_index: int, seed: int) -> Path | None:
        if artifact_dir is None:
            return None
        return (
            Path(artifact_dir)
            / f"seed_{int(seed)}_group_{int(group_index):02d}"
            / f"fold_{int(fold_index):04d}"
        )

    backend = model_backend(str(task["model_name"]))
    if backend != "sklearn" or n_jobs == 1 or len(eval_jobs) <= 1 or joblib is None:
        evaluated: list[tuple[int, dict[str, Any]]] = []
        for group_index, fold_index, seed, split in eval_jobs:
            tr = np.asarray(split["train_indices"], dtype=np.int64)
            val = np.asarray(split["valid_indices"], dtype=np.int64)
            row = _evaluate_fold(
                features=features,
                labels=labels,
                train_indices=tr,
                valid_indices=val,
                model_name=str(task["model_name"]),
                params=dict(task["params"]),
                random_state=seed,
                n_jobs=n_jobs,
                task_spec=task_spec,
                selection_metric_name=selection_metric_name,
                domain_adaptation=domain_adaptation,
                rank_values=rank_values,
                dataset_group_names=dataset_group_names,
                input_normalization=input_normalization,
                model_names=model_names,
                artifact_dir=fold_artifact_dir(group_index, fold_index, seed),
                persist_model=persist_fold_models,
            )
            row["cv_group_index"] = int(group_index)
            row["cv_fold_index"] = int(fold_index)
            row["cv_random_state"] = int(seed)
            for key, value in split.items():
                if str(key).startswith("validation_"):
                    row[str(key)] = value
            evaluated.append((seed, row))
    else:
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            raw_rows = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_evaluate_fold)(
                    features=features,
                    labels=labels,
                    train_indices=np.asarray(split["train_indices"], dtype=np.int64),
                    valid_indices=np.asarray(split["valid_indices"], dtype=np.int64),
                    model_name=str(task["model_name"]),
                    params=dict(task["params"]),
                    random_state=seed,
                    n_jobs=n_jobs,
                    task_spec=task_spec,
                    selection_metric_name=selection_metric_name,
                    domain_adaptation=domain_adaptation,
                    rank_values=rank_values,
                    dataset_group_names=dataset_group_names,
                    input_normalization=input_normalization,
                    model_names=model_names,
                    artifact_dir=fold_artifact_dir(group_index, fold_index, seed),
                    persist_model=persist_fold_models,
                )
                for group_index, fold_index, seed, split in eval_jobs
            )
        evaluated = []
        for (group_index, fold_index, seed, split), row in zip(eval_jobs, raw_rows):
            row_with_seed = dict(row)
            row_with_seed["cv_group_index"] = int(group_index)
            row_with_seed["cv_fold_index"] = int(fold_index)
            row_with_seed["cv_random_state"] = int(seed)
            for key, value in split.items():
                if str(key).startswith("validation_"):
                    row_with_seed[str(key)] = value
            evaluated.append((seed, row_with_seed))

    fold_rows = [row for _, row in evaluated]
    selection_scores = [float(row["selection_metric"]) for row in fold_rows if row.get("selection_metric") is not None]

    seed_results: list[dict[str, Any]] = []
    unique_seeds = sorted({int(seed) for seed, _ in evaluated})
    for seed in unique_seeds:
        seed_fold_rows = [row for seed_value, row in evaluated if int(seed_value) == int(seed)]
        seed_scores = [
            float(row["selection_metric"]) for row in seed_fold_rows if row.get("selection_metric") is not None
        ]
        seed_result = {
            "random_state": int(seed),
            "fold_results": seed_fold_rows,
            "selection_metric_name": selection_metric_name,
            "selection_metric_mean": float(np.mean(seed_scores)) if seed_scores else None,
            "selection_metric_std": float(np.std(seed_scores)) if seed_scores else None,
        }
        if task_spec.is_binary:
            seed_result["roc_auc_mean"] = seed_result["selection_metric_mean"]
            seed_result["roc_auc_std"] = seed_result["selection_metric_std"]
            seed_result["binary_auroc_mean"] = seed_result["selection_metric_mean"]
            seed_result["binary_auroc_std"] = seed_result["selection_metric_std"]
        else:
            binary_auroc_scores = [
                float(row["binary_auroc"]) for row in seed_fold_rows if row.get("binary_auroc") is not None
            ]
            seed_result["binary_auroc_mean"] = float(np.mean(binary_auroc_scores)) if binary_auroc_scores else None
            seed_result["binary_auroc_std"] = float(np.std(binary_auroc_scores)) if binary_auroc_scores else None
        seed_results.append(seed_result)

    result = {
        "task_index": int(task["task_index"]),
        "model_name": str(task["model_name"]),
        "params": dict(task["params"]),
        "complexity_rank": int(task["complexity_rank"]),
        "normalization_policy": str(task["normalization_policy"]),
        "base_normalization_policy": task.get("base_normalization_policy"),
        "input_normalization": str(task.get("input_normalization", input_normalization)),
        "status": "ok",
        "selection_metric_name": selection_metric_name,
        "fold_results": fold_rows,
        "seed_results": seed_results,
        "selection_metric_mean": float(np.mean(selection_scores)) if selection_scores else None,
        "selection_metric_std": float(np.std(selection_scores)) if selection_scores else None,
    }
    if task_spec.is_binary:
        result["roc_auc_mean"] = result["selection_metric_mean"]
        result["roc_auc_std"] = result["selection_metric_std"]
        result["binary_auroc_mean"] = result["selection_metric_mean"]
        result["binary_auroc_std"] = result["selection_metric_std"]
    else:
        binary_auroc_scores = [float(row["binary_auroc"]) for row in fold_rows if row.get("binary_auroc") is not None]
        result["binary_auroc_mean"] = float(np.mean(binary_auroc_scores)) if binary_auroc_scores else None
        result["binary_auroc_std"] = float(np.std(binary_auroc_scores)) if binary_auroc_scores else None
    return result


def _task_result_path(task_dir: Path, task_index: int) -> Path:
    return task_dir / f"task_{task_index:04d}.json"
