from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TABULAR_SPECTRAL_REPRESENTATION_KIND = "spectral_tabular"
ARCHITECTURE_INDEPENDENT_AGGREGATION_KIND = "architecture_independent_aggregation"
ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND = (
    "architecture_independent_layer_sequence_aggregation"
)
SUPERVISED_TASK_MODE_BINARY = "binary"
SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS = "attack_family_multiclass"
BINARY_PROJECTION_POSITIVE_CLASS_SCORE = "positive_class_score"
BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY = "one_minus_clean_probability"
ATTACK_FAMILY_MULTICLASS_ATTACKS = ("RIPPLE", "insertsent", "stybkd", "syntactic")


@dataclass(frozen=True)
class SupervisedDataset:
    features: np.ndarray
    labels: np.ndarray
    model_names: list[str]


@dataclass(frozen=True)
class SupervisedTask:
    task_index: int
    model_name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class SupervisedTaskSpec:
    task_mode: str
    class_names: tuple[str, ...]
    class_to_index: dict[str, int]
    binary_projection: str

    @property
    def n_classes(self) -> int:
        return int(len(self.class_names))

    @property
    def is_binary(self) -> bool:
        return str(self.task_mode) == SUPERVISED_TASK_MODE_BINARY

    @property
    def clean_class_name(self) -> str:
        return str(self.class_names[0])

    @property
    def clean_class_index(self) -> int:
        return int(self.class_to_index[self.clean_class_name])

    @property
    def selection_metric_name(self) -> str:
        if self.is_binary:
            return "roc_auc"
        return "macro_f1"

    def project_label_to_binary(self, label: int | None) -> int | None:
        if label is None:
            return None
        return 0 if int(label) == self.clean_class_index else 1

    def project_known_labels_to_binary(self, labels: np.ndarray) -> np.ndarray:
        labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
        return np.where(labels_np == self.clean_class_index, 0, 1).astype(np.int32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_mode": str(self.task_mode),
            "class_names": [str(x) for x in self.class_names],
            "class_to_index": {str(key): int(value) for key, value in self.class_to_index.items()},
            "binary_projection": str(self.binary_projection),
        }


@dataclass(frozen=True)
class SupervisedPredictionOutputs:
    backdoor_scores: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray | None
    logits: np.ndarray | None


@dataclass(frozen=True)
class CrossValidationSplit:
    split_index: int
    train_indices: np.ndarray
    valid_indices: np.ndarray


@dataclass(frozen=True)
class SupervisedFeatureBundle:
    values: np.ndarray
    representation_kind: str
    metadata: dict[str, Any]
    group_mask: np.ndarray | None = None
    value_mask: np.ndarray | None = None
    group_names: list[list[str]] | None = None

    @property
    def is_tabular(self) -> bool:
        return self.group_mask is None and self.value_mask is None

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    def subset(self, indices: np.ndarray) -> "SupervisedFeatureBundle":
        resolved_indices = np.asarray(indices, dtype=np.int64)
        subset_group_names = None
        if self.group_names is not None:
            subset_group_names = [list(self.group_names[int(i)]) for i in resolved_indices.tolist()]
        return SupervisedFeatureBundle(
            values=np.asarray(self.values[resolved_indices], dtype=np.float32),
            representation_kind=str(self.representation_kind),
            metadata=dict(self.metadata),
            group_mask=(
                None
                if self.group_mask is None
                else np.asarray(self.group_mask[resolved_indices], dtype=bool)
            ),
            value_mask=(
                None
                if self.value_mask is None
                else np.asarray(self.value_mask[resolved_indices], dtype=bool)
            ),
            group_names=subset_group_names,
        )
