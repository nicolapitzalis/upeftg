from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import SupervisedTaskSpec


def compute_balanced_class_loss_config(
    labels: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    labels_np = np.asarray(labels).reshape(-1)
    if sample_mask is None:
        selected = labels_np
    else:
        mask_np = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if mask_np.shape[0] != labels_np.shape[0]:
            raise ValueError("Class-weight loss mask must have the same length as labels")
        selected = labels_np[mask_np]
    selected_int = np.asarray(selected, dtype=np.int64).reshape(-1)
    if selected_int.size == 0:
        raise ValueError("Class-weight loss requires at least one labeled row")

    class_counts = np.asarray(
        [int(np.sum(selected_int == int(class_idx))) for class_idx in range(task_spec.n_classes)],
        dtype=np.int64,
    )
    present_mask = class_counts > 0
    present_count = int(np.sum(present_mask))
    if present_count == 0:
        raise ValueError("Class-weight loss requires at least one observed class")

    class_weights = np.ones(task_spec.n_classes, dtype=np.float32)
    total_present = int(np.sum(class_counts[present_mask]))
    class_weights[present_mask] = float(total_present) / (
        float(present_count) * class_counts[present_mask].astype(np.float32)
    )

    binary_pos_weight: float | None = None
    if task_spec.is_binary:
        negative_count = int(class_counts[0]) if class_counts.shape[0] > 0 else 0
        positive_count = int(class_counts[1]) if class_counts.shape[0] > 1 else 0
        binary_pos_weight = (
            float(negative_count) / float(positive_count) if negative_count > 0 and positive_count > 0 else 1.0
        )

    return {
        "class_counts": [int(x) for x in class_counts.tolist()],
        "class_weights": [float(x) for x in class_weights.tolist()],
        "binary_pos_weight": (None if binary_pos_weight is None else float(binary_pos_weight)),
    }


def compute_balanced_rank_label_loss_config(
    labels: np.ndarray,
    rank_labels: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels_np = np.asarray(labels).reshape(-1)
    ranks_np = np.asarray(rank_labels).reshape(-1)
    if ranks_np.shape[0] != labels_np.shape[0]:
        raise ValueError("Rank-label loss ranks must have the same length as labels")

    if sample_mask is None:
        mask_np = np.ones(labels_np.shape[0], dtype=bool)
    else:
        mask_np = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if mask_np.shape[0] != labels_np.shape[0]:
            raise ValueError("Rank-label loss mask must have the same length as labels")

    selected_labels = np.asarray(labels_np[mask_np], dtype=np.int64).reshape(-1)
    selected_ranks = np.asarray(ranks_np[mask_np], dtype=np.int64).reshape(-1)
    if selected_labels.size == 0:
        raise ValueError("Rank-label loss requires at least one labeled row")
    if bool(np.any(selected_labels < 0)) or bool(np.any(selected_labels >= int(task_spec.n_classes))):
        raise ValueError("Rank-label loss requires labels to be valid supervised class indices")

    bucket_counts: dict[tuple[int, int], int] = {}
    for rank, label in zip(selected_ranks.tolist(), selected_labels.tolist()):
        key = (int(rank), int(label))
        bucket_counts[key] = int(bucket_counts.get(key, 0) + 1)
    if not bucket_counts:
        raise ValueError("Rank-label loss requires at least one observed rank-label bucket")

    total_present = int(selected_labels.size)
    bucket_count = int(len(bucket_counts))
    bucket_weights = {
        key: float(total_present) / (float(bucket_count) * float(count)) for key, count in bucket_counts.items()
    }

    sample_weights = np.zeros(labels_np.shape[0], dtype=np.float32)
    selected_indices = np.flatnonzero(mask_np)
    for idx, rank, label in zip(selected_indices.tolist(), selected_ranks.tolist(), selected_labels.tolist()):
        sample_weights[int(idx)] = np.float32(bucket_weights[(int(rank), int(label))])

    class_counts = [int(np.sum(selected_labels == int(class_idx))) for class_idx in range(int(task_spec.n_classes))]
    rank_counts = [
        {
            "rank": int(rank),
            "count": int(np.sum(selected_ranks == int(rank))),
        }
        for rank in sorted({int(x) for x in selected_ranks.tolist()})
    ]
    rank_label_counts = [
        {
            "rank": int(rank),
            "label": int(label),
            "class_name": (
                str(task_spec.class_names[int(label)]) if 0 <= int(label) < len(task_spec.class_names) else str(label)
            ),
            "count": int(bucket_counts[(int(rank), int(label))]),
            "weight": float(bucket_weights[(int(rank), int(label))]),
        }
        for rank, label in sorted(bucket_counts)
    ]
    selected_weights = sample_weights[mask_np]
    config = {
        "class_counts": class_counts,
        "rank_counts": rank_counts,
        "rank_label_counts": rank_label_counts,
        "sample_weight_min": float(np.min(selected_weights)),
        "sample_weight_max": float(np.max(selected_weights)),
        "sample_weight_mean": float(np.mean(selected_weights)),
    }
    return sample_weights, config
