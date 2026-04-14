from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TABULAR_SPECTRAL_REPRESENTATION_KIND = "spectral_tabular"
ARCHITECTURE_INDEPENDENT_AGGREGATION_KIND = "architecture_independent_aggregation"
ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND = (
    "architecture_independent_layer_sequence_aggregation"
)


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
