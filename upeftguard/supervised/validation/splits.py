from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import SupervisedTaskSpec


def resolve_train_split_percent(train_split_percent: int) -> int:
    resolved = int(train_split_percent)
    if resolved < 1 or resolved > 100:
        raise ValueError(f"train_split must be in the range [1, 100], got {resolved}")
    return resolved


def resolve_calibration_split_percent(calibration_split_percent: int | None) -> int | None:
    if calibration_split_percent is None:
        return None
    resolved = int(calibration_split_percent)
    if resolved < 1 or resolved > 99:
        raise ValueError(f"calibration_split must be in the range [1, 99], got {resolved}")
    return resolved


def resolve_split_by_folder(split_by_folder: bool | None, calibration_split_percent: int | None) -> bool:
    if split_by_folder is None:
        return calibration_split_percent is not None
    return bool(split_by_folder)


def round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def resolve_holdout_subset_count(
    *,
    bucket_size: int,
    subset_percent: int,
    bucket_name: str,
    split_name: str,
    subset_label: str,
) -> tuple[int, bool]:
    if bucket_size < 2:
        raise ValueError(
            f"{split_name} requires at least two samples in each {subset_label} bucket when creating a holdout split, "
            f"but '{bucket_name}' has {bucket_size}"
        )

    raw_subset = (int(bucket_size) * int(subset_percent)) / 100.0
    rounded_subset = round_half_up(raw_subset)
    subset_count = min(max(rounded_subset, 1), int(bucket_size) - 1)
    return int(subset_count), bool(subset_count != rounded_subset)


def label_count_rows(labels: np.ndarray) -> list[dict[str, int]]:
    if labels.size == 0:
        return []
    unique, counts = np.unique(labels, return_counts=True)
    return [
        {
            "label": int(label),
            "count": int(count),
        }
        for label, count in zip(unique.tolist(), counts.tolist())
    ]


def project_optional_labels_to_binary(
    labels: list[int | None],
    *,
    task_spec: SupervisedTaskSpec,
) -> list[int | None]:
    return [task_spec.project_label_to_binary(label) for label in labels]


def build_single_manifest_split_summary(
    *,
    labels: np.ndarray,
    train_indices: np.ndarray,
    infer_indices: np.ndarray,
    train_split_percent: int,
    random_state: int,
    strategy: str,
) -> dict[str, Any]:
    train_labels = labels[train_indices]
    infer_labels = labels[infer_indices] if infer_indices.size > 0 else np.asarray([], dtype=np.int32)
    return {
        "strategy": strategy,
        "requested_train_split_percent": int(train_split_percent),
        "random_state": int(random_state),
        "n_train": int(train_indices.size),
        "n_inference": int(infer_indices.size),
        "train_label_counts": label_count_rows(np.asarray(train_labels, dtype=np.int32)),
        "inference_label_counts": label_count_rows(np.asarray(infer_labels, dtype=np.int32)),
    }


def build_calibration_split_summary(
    *,
    labels: np.ndarray,
    fit_train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
    strategy: str,
) -> dict[str, Any]:
    fit_train_labels = labels[fit_train_indices]
    calibration_labels = labels[calibration_indices]
    return {
        "strategy": strategy,
        "requested_calibration_split_percent": int(calibration_split_percent),
        "random_state": int(random_state),
        "n_fit_train": int(fit_train_indices.size),
        "n_calibration": int(calibration_indices.size),
        "fit_train_label_counts": label_count_rows(np.asarray(fit_train_labels, dtype=np.int32)),
        "calibration_label_counts": label_count_rows(np.asarray(calibration_labels, dtype=np.int32)),
    }


def resolve_holdout_train_count(
    *,
    bucket_size: int,
    train_split_percent: int,
    bucket_name: str,
    split_name: str,
) -> tuple[int, bool]:
    if bucket_size < 2:
        raise ValueError(
            f"{split_name} requires at least two samples in each split bucket when creating a holdout split, "
            f"but '{bucket_name}' has {bucket_size}"
        )

    raw_train = (int(bucket_size) * int(train_split_percent)) / 100.0
    rounded_train = round_half_up(raw_train)
    train_count = min(max(rounded_train, 1), int(bucket_size) - 1)
    return int(train_count), bool(train_count != rounded_train)


def build_single_manifest_stratified_split(
    *,
    labels: np.ndarray,
    train_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    if train_split_percent == 100:
        split_summary = build_single_manifest_split_summary(
            labels=labels,
            train_indices=all_indices,
            infer_indices=np.asarray([], dtype=np.int64),
            train_split_percent=train_split_percent,
            random_state=random_state,
            strategy="single_manifest_all_train",
        )
        split_summary["split_by_folder"] = False
        return (
            all_indices,
            np.asarray([], dtype=np.int64),
            [],
            split_summary,
        )

    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Stratified --train-split requires at least two classes in a single manifest")

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    infer_parts: list[np.ndarray] = []
    adjusted_labels: list[int] = []
    per_class_rows: list[dict[str, Any]] = []

    for label, count in zip(unique.tolist(), counts.tolist()):
        label_value = int(label)
        label_count = int(count)
        train_count, adjusted = resolve_holdout_train_count(
            bucket_size=label_count,
            train_split_percent=train_split_percent,
            bucket_name=f"label {label_value}",
            split_name="Stratified --train-split",
        )
        infer_count = label_count - train_count
        if adjusted:
            adjusted_labels.append(label_value)

        label_indices = all_indices[labels == label_value]
        shuffled = label_indices[rng.permutation(label_indices.shape[0])]
        train_parts.append(shuffled[:train_count])
        infer_parts.append(shuffled[train_count:])
        per_class_rows.append(
            {
                "label": label_value,
                "total": label_count,
                "n_train": int(train_count),
                "n_inference": int(infer_count),
            }
        )

    train_indices = np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
    infer_indices = np.sort(np.concatenate(infer_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_labels:
        adjusted_preview = ", ".join(str(label) for label in adjusted_labels[:5])
        warnings.append(
            "Adjusted rounded train counts to keep at least one sample in both train and inference "
            f"for labels: {adjusted_preview}"
        )

    split_summary = build_single_manifest_split_summary(
        labels=labels,
        train_indices=train_indices,
        infer_indices=infer_indices,
        train_split_percent=train_split_percent,
        random_state=random_state,
        strategy="single_manifest_stratified_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["per_class"] = per_class_rows
    return train_indices, infer_indices, warnings, split_summary


def split_folder_name(item: Any) -> str:
    parent_name = item.model_dir.parent.name.strip()
    if parent_name:
        return parent_name
    return item.model_dir.name


def build_single_manifest_folder_label_split(
    *,
    items: list[Any],
    labels: np.ndarray,
    train_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if labels.shape[0] != len(items):
        raise ValueError("Folder-based split requires one label per manifest item")

    if train_split_percent < 100 and np.unique(labels).size < 2:
        raise ValueError("Folder-based --train-split requires at least two classes in a single manifest")

    bucket_to_indices: dict[tuple[str, int], list[int]] = {}
    for idx, item in enumerate(items):
        folder_name = split_folder_name(item)
        label_value = int(labels[idx])
        bucket_to_indices.setdefault((folder_name, label_value), []).append(int(idx))

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    infer_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for folder_name, label_value in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(folder_name, label_value)], dtype=np.int64)
        if train_split_percent == 100:
            train_count = int(bucket_indices.size)
            infer_count = 0
            train_parts.append(bucket_indices)
            infer_parts.append(np.asarray([], dtype=np.int64))
        else:
            train_count, adjusted = resolve_holdout_train_count(
                bucket_size=int(bucket_indices.size),
                train_split_percent=train_split_percent,
                bucket_name=f"{folder_name} [label {label_value}]",
                split_name="Folder-based --train-split",
            )
            infer_count = int(bucket_indices.size) - train_count
            if adjusted:
                adjusted_buckets.append(f"{folder_name} [label {label_value}]")

            shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
            train_parts.append(shuffled[:train_count])
            infer_parts.append(shuffled[train_count:])

        per_bucket_rows.append(
            {
                "folder": folder_name,
                "label": int(label_value),
                "total": int(bucket_indices.size),
                "n_train": int(train_count),
                "n_inference": int(infer_count),
            }
        )

    train_indices = (
        np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
        if train_parts
        else np.asarray([], dtype=np.int64)
    )
    infer_indices = (
        np.sort(np.concatenate(infer_parts).astype(np.int64, copy=False))
        if infer_parts
        else np.asarray([], dtype=np.int64)
    )

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded train counts to keep at least one sample in both train and inference "
            f"for folder/label buckets: {adjusted_preview}"
        )

    strategy = (
        "single_manifest_folder_label_holdout"
        if train_split_percent < 100
        else "single_manifest_folder_label_all_train"
    )
    split_summary = build_single_manifest_split_summary(
        labels=labels,
        train_indices=train_indices,
        infer_indices=infer_indices,
        train_split_percent=train_split_percent,
        random_state=random_state,
        strategy=strategy,
    )
    split_summary["split_by_folder"] = True
    split_summary["folder_count"] = int(len({row["folder"] for row in per_bucket_rows}))
    split_summary["folder_label_buckets"] = per_bucket_rows
    return train_indices, infer_indices, warnings, split_summary


def build_calibration_stratified_split(
    *,
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    unique, counts = np.unique(candidate_labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Calibration split requires at least two classes in the training partition")

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_labels: list[int] = []
    per_class_rows: list[dict[str, Any]] = []

    for label, count in zip(unique.tolist(), counts.tolist()):
        label_value = int(label)
        label_indices = candidate_indices[candidate_labels == label_value]
        calibration_count, adjusted = resolve_holdout_subset_count(
            bucket_size=int(count),
            subset_percent=calibration_split_percent,
            bucket_name=f"label {label_value}",
            split_name="Stratified --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(label_indices.size) - calibration_count
        if adjusted:
            adjusted_labels.append(label_value)

        shuffled = label_indices[rng.permutation(label_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_class_rows.append(
            {
                "label": label_value,
                "total": int(count),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_labels:
        adjusted_preview = ", ".join(str(label) for label in adjusted_labels[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for labels: {adjusted_preview}"
        )

    split_summary = build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_stratified_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["per_class"] = per_class_rows
    return fit_train_indices, calibration_indices, warnings, split_summary


def rank_label_buckets(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    ranks_np = np.asarray(ranks, dtype=np.int64).reshape(-1)
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    return {
        (rank, label): [
            int(i)
            for i in candidate_indices.tolist()
            if int(ranks_np[int(i)]) == int(rank) and int(labels_np[int(i)]) == int(label)
        ]
        for rank in sorted({int(ranks_np[int(i)]) for i in candidate_indices.tolist()})
        for label in sorted(
            {int(labels_np[int(i)]) for i in candidate_indices.tolist() if int(ranks_np[int(i)]) == int(rank)}
        )
    }


def rank_label_stratification_keys(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    ranks_np = np.asarray(ranks, dtype=np.int64).reshape(-1)
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    return np.asarray(
        [
            f"rank_{int(ranks_np[int(i)])}_label_{int(labels_np[int(i)])}"
            for i in np.asarray(candidate_indices, dtype=np.int64).tolist()
        ],
        dtype=object,
    )


def rank_label_stratification_warning(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> str | None:
    keys = rank_label_stratification_keys(
        candidate_indices=candidate_indices,
        ranks=ranks,
        labels=labels,
    )
    unique, counts = np.unique(keys, return_counts=True)
    if unique.size < 2:
        return (
            f"cnn_1d_dann could not use rank-label stratification for {split_name}: "
            "fewer than two rank-label buckets; falling back to label-only stratification"
        )
    min_count = int(np.min(counts))
    if min_count < 2:
        return (
            f"cnn_1d_dann could not use rank-label stratification for {split_name}: "
            f"smallest rank-label bucket has {min_count} row; falling back to label-only stratification"
        )
    return None


def folder_label_stratification_keys(
    *,
    items: list[Any],
    candidate_indices: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    return np.asarray(
        [
            f"folder_{split_folder_name(items[int(i)])}_label_{int(labels_np[int(i)])}"
            for i in np.asarray(candidate_indices, dtype=np.int64).tolist()
        ],
        dtype=object,
    )


def folder_label_stratification_warning(
    *,
    items: list[Any],
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> str | None:
    keys = folder_label_stratification_keys(
        items=items,
        candidate_indices=candidate_indices,
        labels=labels,
    )
    unique, counts = np.unique(keys, return_counts=True)
    if unique.size < 2:
        return (
            f"Could not use folder-label stratification for {split_name}: "
            "fewer than two folder-label buckets; falling back to label-only stratification"
        )
    min_count = int(np.min(counts))
    if min_count < 2:
        return (
            f"Could not use folder-label stratification for {split_name}: "
            f"smallest folder-label bucket has {min_count} row; falling back to label-only stratification"
        )
    return None


def build_calibration_rank_label_split(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    bucket_to_indices = rank_label_buckets(
        candidate_indices=candidate_indices,
        ranks=ranks,
        labels=labels,
    )
    if len(bucket_to_indices) < 2:
        raise ValueError("Rank-label calibration split requires at least two rank-label buckets")

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for rank, label in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(rank, label)], dtype=np.int64)
        calibration_count, adjusted = resolve_holdout_subset_count(
            bucket_size=int(bucket_indices.size),
            subset_percent=calibration_split_percent,
            bucket_name=f"rank {int(rank)} [label {int(label)}]",
            split_name="Rank-label --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(bucket_indices.size) - calibration_count
        if adjusted:
            adjusted_buckets.append(f"rank {int(rank)} [label {int(label)}]")

        shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_bucket_rows.append(
            {
                "rank": int(rank),
                "label": int(label),
                "total": int(bucket_indices.size),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for rank-label buckets: {adjusted_preview}"
        )

    split_summary = build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_rank_label_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["split_by_rank_label"] = True
    split_summary["rank_label_buckets"] = per_bucket_rows
    return fit_train_indices, calibration_indices, warnings, split_summary


def build_calibration_folder_label_split(
    *,
    items: list[Any],
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if labels.shape[0] != len(items):
        raise ValueError("Folder-based calibration split requires one label per manifest item")

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    if np.unique(candidate_labels).size < 2:
        raise ValueError("Folder-based calibration split requires at least two classes in the training partition")

    bucket_to_indices: dict[tuple[str, int], list[int]] = {}
    for idx in candidate_indices.tolist():
        folder_name = split_folder_name(items[int(idx)])
        label_value = int(labels[int(idx)])
        bucket_to_indices.setdefault((folder_name, label_value), []).append(int(idx))

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for folder_name, label_value in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(folder_name, label_value)], dtype=np.int64)
        calibration_count, adjusted = resolve_holdout_subset_count(
            bucket_size=int(bucket_indices.size),
            subset_percent=calibration_split_percent,
            bucket_name=f"{folder_name} [label {label_value}]",
            split_name="Folder-based --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(bucket_indices.size) - calibration_count
        if adjusted:
            adjusted_buckets.append(f"{folder_name} [label {label_value}]")

        shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_bucket_rows.append(
            {
                "folder": folder_name,
                "label": int(label_value),
                "total": int(bucket_indices.size),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for folder/label buckets: {adjusted_preview}"
        )

    split_summary = build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_folder_label_holdout",
    )
    split_summary["split_by_folder"] = True
    split_summary["folder_count"] = int(len({row["folder"] for row in per_bucket_rows}))
    split_summary["folder_label_buckets"] = per_bucket_rows
    return fit_train_indices, calibration_indices, warnings, split_summary
