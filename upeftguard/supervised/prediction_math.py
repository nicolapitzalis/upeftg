from __future__ import annotations

import numpy as np

from .contracts import SupervisedTaskSpec


def sigmoid(values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values_np, dtype=np.float64)
    nonnegative = values_np >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values_np[nonnegative]))
    exp_values = np.exp(values_np[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def softmax(values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    if values_np.ndim != 2:
        raise ValueError(f"Expected a 2D class-score matrix, got shape={values_np.shape}")
    shifted = values_np - np.max(values_np, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    normalizer = np.sum(exp_values, axis=1, keepdims=True)
    normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
    return np.asarray(exp_values / normalizer, dtype=np.float64)


def probabilities_from_logits(
    logits: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
) -> np.ndarray:
    """Convert genuine binary or multiclass logits to class probabilities."""

    if task_spec.is_binary:
        positive = sigmoid(np.asarray(logits, dtype=np.float64).reshape(-1))
        return np.column_stack([1.0 - positive, positive]).astype(
            np.float64,
            copy=False,
        )
    return softmax(logits)
