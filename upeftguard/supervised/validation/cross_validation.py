from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..contracts import SupervisedTaskSpec
from ...utilities.core.manifest import (
    AttackSampleIdentity,
    infer_dataset_group_name,
)


CV_STRATEGY_STRATIFIED = "stratified"
CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT = "attack_family_leave_one_out"
CV_STRATEGY_DATASET_LEAVE_ONE_OUT = "dataset_leave_one_out"
SUPPORTED_CV_STRATEGIES = (
    CV_STRATEGY_STRATIFIED,
    CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT,
    CV_STRATEGY_DATASET_LEAVE_ONE_OUT,
)


def resolve_cv_strategy(cv_strategy: str | None) -> str:
    resolved = str(cv_strategy or CV_STRATEGY_STRATIFIED)
    if resolved not in SUPPORTED_CV_STRATEGIES:
        raise ValueError(f"Unsupported cv_strategy={resolved!r}; supported values={list(SUPPORTED_CV_STRATEGIES)}")
    return resolved


def validated_train_label_counts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Supervised classification requires at least two classes in the training set")
    return unique, counts


def partition_cv_always_train_indices(
    *,
    train_indices: np.ndarray,
    cv_always_train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate fold-eligible rows from rows pinned into every fold's training set."""

    train = np.unique(np.asarray(train_indices, dtype=np.int64).reshape(-1))
    always_train = np.unique(np.asarray(cv_always_train_indices, dtype=np.int64).reshape(-1))
    unknown = np.setdiff1d(always_train, train, assume_unique=True)
    if unknown.size:
        raise ValueError("cv_always_train rows must belong to the active training split")
    fold_eligible = np.setdiff1d(train, always_train, assume_unique=True)
    if fold_eligible.size == 0:
        raise ValueError("cv_always_train cannot consume the complete cross-validation training pool")
    return fold_eligible, always_train


def append_cv_always_train_indices(
    *,
    splits: list[dict[str, Any]],
    cv_always_train_indices: np.ndarray,
    labels: np.ndarray,
    task_spec: SupervisedTaskSpec,
) -> list[dict[str, Any]]:
    """Add fixed rows to every training partition without making them validation-eligible."""

    always_train = np.unique(np.asarray(cv_always_train_indices, dtype=np.int64).reshape(-1))
    if always_train.size == 0:
        return splits
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    decorated: list[dict[str, Any]] = []
    for raw_split in splits:
        split = dict(raw_split)
        valid = np.asarray(split["valid_indices"], dtype=np.int64)
        overlap = np.intersect1d(valid, always_train, assume_unique=False)
        if overlap.size:
            raise ValueError("cv_always_train rows cannot appear in a validation partition")
        train = np.union1d(np.asarray(split["train_indices"], dtype=np.int64), always_train)
        split["train_indices"] = [int(value) for value in train.tolist()]
        split["cv_always_train_indices"] = [int(value) for value in always_train.tolist()]
        split["n_cv_always_train"] = int(always_train.size)
        if "train_class_counts" in split:
            unique, counts = validated_train_label_counts(labels_np[train])
            split["train_class_counts"] = {
                str(int(label)): int(count)
                for label, count in zip(unique.tolist(), counts.tolist())
            }
        if "train_clean_count" in split or "train_positive_count" in split:
            train_binary = np.asarray(
                [task_spec.project_label_to_binary(int(labels_np[int(index)])) for index in train],
                dtype=np.int32,
            )
            split["train_clean_count"] = int(np.sum(train_binary == 0))
            split["train_positive_count"] = int(np.sum(train_binary == 1))
        decorated.append(split)
    return decorated


def sanitize_cv_folds(labels: np.ndarray, requested_folds: int) -> tuple[int, list[str]]:
    if requested_folds < 2:
        raise ValueError(f"cv_folds must be >=2, got {requested_folds}")

    _unique, counts = validated_train_label_counts(labels)
    min_count = int(np.min(counts))
    warnings: list[str] = []
    if min_count < 2:
        warnings.append("Minority class count <2; falling back to a single train-as-validation split for tuning")
        return 1, warnings

    resolved = min(requested_folds, min_count)
    if resolved != requested_folds:
        warnings.append(f"Reduced cv_folds from {requested_folds} to {resolved} due to minority class size={min_count}")
    return int(resolved), warnings


def build_cv_splits(
    *,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if cv_folds == 1:
        return [
            {
                "split_index": 0,
                "train_indices": [int(x) for x in train_indices.tolist()],
                "valid_indices": [int(x) for x in train_indices.tolist()],
            }
        ]

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits: list[dict[str, Any]] = []
    dummy = np.zeros((train_indices.shape[0], 1), dtype=np.float32)
    for split_idx, (tr_rel, val_rel) in enumerate(splitter.split(dummy, train_labels)):
        tr_abs = train_indices[np.asarray(tr_rel, dtype=np.int64)]
        val_abs = train_indices[np.asarray(val_rel, dtype=np.int64)]
        splits.append(
            {
                "split_index": int(split_idx),
                "train_indices": [int(x) for x in tr_abs.tolist()],
                "valid_indices": [int(x) for x in val_abs.tolist()],
            }
        )
    return splits


def build_attack_family_leave_one_out_cv_splits(
    *,
    train_indices: np.ndarray,
    labels: np.ndarray,
    sample_identities: list[AttackSampleIdentity],
    random_state: int,
    task_spec: SupervisedTaskSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if not task_spec.is_binary:
        raise ValueError("cv_strategy=attack_family_leave_one_out is currently supported only for binary tasks")

    candidate_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    if labels_np.shape[0] != len(sample_identities):
        raise ValueError("attack-family CV requires labels and sample identities to be aligned")

    clean_indices: list[int] = []
    positive_by_attack: dict[str, list[int]] = {}
    for raw_idx in candidate_indices.tolist():
        idx = int(raw_idx)
        binary_label = int(task_spec.project_label_to_binary(int(labels_np[idx])))
        if binary_label == 0:
            clean_indices.append(idx)
        else:
            attack_name = str(sample_identities[idx].attack_name or "unknown")
            positive_by_attack.setdefault(attack_name, []).append(idx)

    attack_names = sorted(name for name, idxs in positive_by_attack.items() if idxs)
    if len(attack_names) < 2:
        raise ValueError(
            "cv_strategy=attack_family_leave_one_out requires at least two positive attack families "
            f"in the training split; found {len(attack_names)}"
        )
    if len(clean_indices) < len(attack_names):
        raise ValueError(
            "cv_strategy=attack_family_leave_one_out requires at least one clean validation sample "
            f"per attack fold; found clean={len(clean_indices)}, attacks={len(attack_names)}"
        )

    rng = np.random.default_rng(int(random_state))
    clean_shuffled = np.asarray(clean_indices, dtype=np.int64)
    rng.shuffle(clean_shuffled)
    clean_chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(clean_shuffled, len(attack_names))]
    if any(int(chunk.size) == 0 for chunk in clean_chunks):
        raise ValueError("attack-family CV clean partition produced an empty validation clean chunk")

    splits: list[dict[str, Any]] = []
    for split_idx, attack_name in enumerate(attack_names):
        valid_positive = np.asarray(positive_by_attack[attack_name], dtype=np.int64)
        valid_clean = clean_chunks[split_idx]
        train_positive = np.asarray(
            [
                idx
                for other_attack in attack_names
                if other_attack != attack_name
                for idx in positive_by_attack[other_attack]
            ],
            dtype=np.int64,
        )
        train_clean = np.concatenate(
            [chunk for chunk_idx, chunk in enumerate(clean_chunks) if chunk_idx != split_idx],
            axis=0,
        )
        train_abs = np.sort(np.concatenate([train_positive, train_clean], axis=0))
        valid_abs = np.sort(np.concatenate([valid_positive, valid_clean], axis=0))
        train_binary = np.asarray(
            [task_spec.project_label_to_binary(int(labels_np[int(idx)])) for idx in train_abs],
            dtype=np.int32,
        )
        valid_binary = np.asarray(
            [task_spec.project_label_to_binary(int(labels_np[int(idx)])) for idx in valid_abs],
            dtype=np.int32,
        )
        validated_train_label_counts(train_binary)
        validated_train_label_counts(valid_binary)
        splits.append(
            {
                "split_index": int(split_idx),
                "strategy": CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT,
                "validation_attack_name": str(attack_name),
                "train_attack_names": [str(name) for name in attack_names if name != attack_name],
                "train_indices": [int(x) for x in train_abs.tolist()],
                "valid_indices": [int(x) for x in valid_abs.tolist()],
                "train_clean_count": int(train_clean.size),
                "train_positive_count": int(train_positive.size),
                "valid_clean_count": int(valid_clean.size),
                "valid_positive_count": int(valid_positive.size),
            }
        )

    summary = {
        "strategy": CV_STRATEGY_ATTACK_FAMILY_LEAVE_ONE_OUT,
        "split_by_folder": False,
        "split_by_rank_label": False,
        "split_by_attack_family": True,
        "clean_partition": "seeded_disjoint_chunks",
        "attack_names": [str(name) for name in attack_names],
        "n_attack_families": int(len(attack_names)),
        "n_clean": int(len(clean_indices)),
    }
    warnings = [
        "Using attack-family leave-one-out CV: each fold validates on one seen attack family's "
        "positive samples plus a disjoint clean partition; --cv-folds is ignored"
    ]
    return splits, summary, warnings


def build_dataset_leave_one_out_cv_splits(
    *,
    train_indices: np.ndarray,
    labels: np.ndarray,
    sample_identities: list[AttackSampleIdentity],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    candidate_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    if labels_np.shape[0] != len(sample_identities):
        raise ValueError("dataset CV requires labels and sample identities to be aligned")

    indices_by_dataset: dict[str, list[int]] = {}
    for raw_idx in candidate_indices.tolist():
        idx = int(raw_idx)
        dataset_name = infer_dataset_group_name(sample_identities[idx])
        indices_by_dataset.setdefault(dataset_name, []).append(idx)

    dataset_names = sorted(name for name, idxs in indices_by_dataset.items() if idxs)
    if len(dataset_names) < 2:
        raise ValueError(
            "cv_strategy=dataset_leave_one_out requires at least two datasets in the "
            f"training split; found {len(dataset_names)}"
        )

    splits: list[dict[str, Any]] = []
    for split_idx, dataset_name in enumerate(dataset_names):
        valid_abs = np.asarray(indices_by_dataset[dataset_name], dtype=np.int64)
        train_abs = np.asarray(
            [
                idx
                for other_dataset in dataset_names
                if other_dataset != dataset_name
                for idx in indices_by_dataset[other_dataset]
            ],
            dtype=np.int64,
        )
        train_abs.sort()
        valid_abs.sort()

        try:
            train_unique, train_counts = validated_train_label_counts(labels_np[train_abs])
        except ValueError as exc:
            raise ValueError(
                "cv_strategy=dataset_leave_one_out produced a single-class training fold "
                f"while holding out dataset={dataset_name!r}"
            ) from exc
        try:
            valid_unique, valid_counts = validated_train_label_counts(labels_np[valid_abs])
        except ValueError as exc:
            raise ValueError(
                "cv_strategy=dataset_leave_one_out requires every held-out dataset to contain "
                f"at least two classes; dataset={dataset_name!r} is single-class"
            ) from exc

        splits.append(
            {
                "split_index": int(split_idx),
                "strategy": CV_STRATEGY_DATASET_LEAVE_ONE_OUT,
                "validation_dataset_name": str(dataset_name),
                "train_dataset_names": [str(name) for name in dataset_names if name != dataset_name],
                "train_indices": [int(x) for x in train_abs.tolist()],
                "valid_indices": [int(x) for x in valid_abs.tolist()],
                "train_class_counts": {
                    str(int(label)): int(count) for label, count in zip(train_unique.tolist(), train_counts.tolist())
                },
                "valid_class_counts": {
                    str(int(label)): int(count) for label, count in zip(valid_unique.tolist(), valid_counts.tolist())
                },
            }
        )

    summary = {
        "strategy": CV_STRATEGY_DATASET_LEAVE_ONE_OUT,
        "split_by_folder": False,
        "split_by_rank_label": False,
        "split_by_attack_family": False,
        "split_by_dataset": True,
        "dataset_names": [str(name) for name in dataset_names],
        "n_datasets": int(len(dataset_names)),
    }
    warnings = [
        "Using dataset leave-one-out CV: each fold validates on every sample from one "
        "inferred dataset group; --cv-folds is ignored"
    ]
    return splits, summary, warnings
