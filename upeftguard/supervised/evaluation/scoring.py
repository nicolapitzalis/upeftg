from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ..contracts import SupervisedPredictionOutputs, SupervisedTaskSpec
from ..tasks import SELECTION_METRIC_BINARY_AUROC


def evaluate_fold_predictions(
    *,
    labels: np.ndarray,
    valid_indices: np.ndarray,
    outputs: SupervisedPredictionOutputs,
    task_spec: SupervisedTaskSpec,
    selection_metric_name: str,
) -> dict[str, Any]:
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    valid_indices_np = np.asarray(valid_indices, dtype=np.int64).reshape(-1)
    valid_labels = labels_np[valid_indices_np]

    if task_spec.is_binary:
        auroc = float(roc_auc_score(valid_labels, outputs.backdoor_scores))
        return {
            "roc_auc": auroc,
            "binary_auroc": auroc,
            "selection_metric": auroc,
        }

    if selection_metric_name != SELECTION_METRIC_BINARY_AUROC:
        raise ValueError(f"Unsupported selection metric {selection_metric_name!r} for multiclass supervised evaluation")
    binary_labels = task_spec.project_known_labels_to_binary(valid_labels)
    binary_auroc = float(roc_auc_score(binary_labels, outputs.backdoor_scores))
    return {
        "selection_metric": binary_auroc,
        "binary_auroc": binary_auroc,
    }
