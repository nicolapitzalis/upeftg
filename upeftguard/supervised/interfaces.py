from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
