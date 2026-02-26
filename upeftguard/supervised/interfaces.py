from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SupervisedDataset:
    features: np.ndarray
    labels: np.ndarray
    model_names: list[str]


class SupervisedModel(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...
