from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import SupervisedPredictionOutputs, SupervisedTaskSpec
from ..prediction_math import softmax


def _unwrap_final_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps") and getattr(model, "named_steps", None):
        return list(model.named_steps.values())[-1]
    if hasattr(model, "steps") and getattr(model, "steps", None):
        return model.steps[-1][1]
    return model


def _observed_model_classes(model: Any) -> np.ndarray | None:
    estimator = _unwrap_final_estimator(model)
    for candidate in (estimator, model):
        classes = getattr(candidate, "classes_", None)
        if classes is not None:
            return np.asarray(classes, dtype=np.int32).reshape(-1)
    return None


def align_task_matrix(
    values: np.ndarray,
    *,
    observed_classes: np.ndarray | None,
    task_spec: SupervisedTaskSpec,
    fill_value: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D class-output matrix, got shape={matrix.shape}")
    if matrix.shape[1] == task_spec.n_classes and observed_classes is None:
        return np.asarray(matrix, dtype=np.float64)
    if observed_classes is None:
        observed_classes = np.arange(matrix.shape[1], dtype=np.int32)
    if int(observed_classes.shape[0]) != int(matrix.shape[1]):
        raise ValueError(
            "Model output columns do not align with model classes_: "
            f"shape={matrix.shape}, classes={observed_classes.tolist()}"
        )
    aligned = np.full((matrix.shape[0], task_spec.n_classes), float(fill_value), dtype=np.float64)
    for source_col, class_value in enumerate(observed_classes.tolist()):
        class_index = int(class_value)
        if class_index < 0 or class_index >= task_spec.n_classes:
            raise ValueError(
                f"Observed class index {class_index} is outside the configured task classes "
                f"[0, {task_spec.n_classes - 1}]"
            )
        aligned[:, class_index] = matrix[:, source_col]
    return aligned


def _binary_attack_class_index(task_spec: SupervisedTaskSpec) -> int:
    attack_indices = [index for index in range(task_spec.n_classes) if index != task_spec.clean_class_index]
    if len(attack_indices) != 1:
        raise ValueError(
            "Binary prediction requires exactly one non-clean class, "
            f"got class_names={list(task_spec.class_names)}"
        )
    return int(attack_indices[0])


def _binary_outputs_from_decision_scores(
    raw_decision: np.ndarray,
    *,
    observed_classes: np.ndarray | None,
    task_spec: SupervisedTaskSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attack_class_index = _binary_attack_class_index(task_spec)
    if raw_decision.ndim == 2 and raw_decision.shape[1] == 1:
        raw_decision = raw_decision.reshape(-1)
    if raw_decision.ndim == 2:
        aligned_decision = align_task_matrix(
            raw_decision,
            observed_classes=observed_classes,
            task_spec=task_spec,
            fill_value=-np.inf,
        )
        backdoor_scores = aligned_decision[:, attack_class_index]
        predicted_labels = np.argmax(aligned_decision, axis=1).astype(np.int32, copy=False)
        return backdoor_scores, predicted_labels, aligned_decision

    if raw_decision.ndim != 1:
        raise ValueError(f"Expected 1D or 2D binary decision scores, got shape={raw_decision.shape}")

    if observed_classes is None:
        negative_class_index = int(task_spec.clean_class_index)
        positive_class_index = int(attack_class_index)
    else:
        if observed_classes.shape[0] != 2:
            raise ValueError(
                "Binary decision scores require exactly two observed model classes, "
                f"got classes={observed_classes.tolist()}"
            )
        negative_class_index = int(observed_classes[0])
        positive_class_index = int(observed_classes[1])

    if positive_class_index == attack_class_index:
        backdoor_scores = raw_decision
    elif negative_class_index == attack_class_index:
        backdoor_scores = -raw_decision
    else:
        raise ValueError(
            f"Binary decision classes do not contain attack class index {attack_class_index}: "
            f"negative={negative_class_index}, positive={positive_class_index}"
        )
    predicted_labels = np.where(
        raw_decision >= 0.0,
        positive_class_index,
        negative_class_index,
    ).astype(np.int32, copy=False)
    return backdoor_scores, predicted_labels, raw_decision


def predict_task_outputs(
    model: Any,
    x: Any,
    *,
    task_spec: SupervisedTaskSpec,
) -> SupervisedPredictionOutputs:
    observed_classes = _observed_model_classes(model)
    if hasattr(model, "predict_proba"):
        raw_probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
        if raw_probabilities.ndim != 2:
            raise ValueError(
                "Expected predict_proba() to return a 2D class-probability matrix, "
                f"got shape={raw_probabilities.shape}"
            )
        probabilities = align_task_matrix(
            raw_probabilities,
            observed_classes=observed_classes,
            task_spec=task_spec,
            fill_value=0.0,
        )
        predicted_labels = np.argmax(probabilities, axis=1).astype(np.int32, copy=False)
        backdoor_scores = 1.0 - probabilities[:, task_spec.clean_class_index]
        return SupervisedPredictionOutputs(
            backdoor_scores=np.asarray(backdoor_scores, dtype=np.float64).reshape(-1),
            predicted_labels=predicted_labels,
            probabilities=np.asarray(probabilities, dtype=np.float64),
            decision_scores=None,
        )

    if hasattr(model, "decision_function"):
        raw_decision = np.asarray(model.decision_function(x), dtype=np.float64)
        if task_spec.is_binary:
            backdoor_scores, predicted_labels, decision_scores = _binary_outputs_from_decision_scores(
                raw_decision,
                observed_classes=observed_classes,
                task_spec=task_spec,
            )
        else:
            if raw_decision.ndim == 1:
                raw_decision = np.column_stack([-raw_decision, raw_decision])
            if raw_decision.ndim != 2:
                raise ValueError(
                    "Expected multiclass decision_function() to return a 1D binary margin or "
                    f"2D class-score matrix, got shape={raw_decision.shape}"
                )
            decision_scores = align_task_matrix(
                raw_decision,
                observed_classes=observed_classes,
                task_spec=task_spec,
                fill_value=-np.inf,
            )
            normalized_scores = softmax(decision_scores)
            backdoor_scores = 1.0 - normalized_scores[:, task_spec.clean_class_index]
            predicted_labels = np.argmax(decision_scores, axis=1).astype(np.int32, copy=False)
        return SupervisedPredictionOutputs(
            backdoor_scores=np.asarray(backdoor_scores, dtype=np.float64).reshape(-1),
            predicted_labels=np.asarray(predicted_labels, dtype=np.int32).reshape(-1),
            probabilities=None,
            decision_scores=np.asarray(decision_scores, dtype=np.float64),
        )

    raise TypeError(
        "Supervised prediction requires a model with predict_proba() or decision_function(); "
        f"got {type(model).__name__}"
    )
