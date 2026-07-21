"""Prediction and validation scoring shared by torch sequence estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ...contracts import SupervisedTaskSpec
from ...prediction_math import probabilities_from_logits


def selection_auroc_from_logits(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
) -> float:
    """Compute the validation AUROC used for torch-model epoch selection."""

    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    if task_spec.is_binary:
        scores = np.asarray(logits, dtype=np.float64).reshape(-1)
        binary_labels = labels_np
    else:
        probabilities = probabilities_from_logits(logits, task_spec=task_spec)
        scores = 1.0 - probabilities[:, task_spec.clean_class_index]
        binary_labels = task_spec.project_known_labels_to_binary(labels_np)
    return float(roc_auc_score(binary_labels, scores))


class TorchSequencePredictionMixin:
    """Shared sklearn-style prediction surface for torch sequence models."""

    task_spec: SupervisedTaskSpec

    def decision_function(self, features: Any) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, features: Any) -> np.ndarray:
        return probabilities_from_logits(
            self.decision_function(features),
            task_spec=self.task_spec,
        )

    def predict(self, features: Any) -> np.ndarray:
        if self.task_spec.is_binary:
            return (self.decision_function(features) >= 0.0).astype(
                np.int32,
                copy=False,
            )
        return np.argmax(self.predict_proba(features), axis=1).astype(
            np.int32,
            copy=False,
        )
