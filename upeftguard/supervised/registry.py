from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    complexity_rank: int
    param_grid: tuple[dict[str, Any], ...]
    factory: Callable[[dict[str, Any], int], Any]


_REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        name="logistic_regression",
        complexity_rank=0,
        param_grid=(
            {"C": 0.01},
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ),
        factory=lambda params, random_state: LogisticRegression(
            C=float(params["C"]),
            solver="lbfgs",
            max_iter=5000,
            random_state=int(random_state),
        ),
    ),
    "ridge_classifier": ModelDefinition(
        name="ridge_classifier",
        complexity_rank=1,
        param_grid=(
            {"alpha": 0.1},
            {"alpha": 1.0},
            {"alpha": 10.0},
        ),
        factory=lambda params, random_state: RidgeClassifier(
            alpha=float(params["alpha"]),
            random_state=int(random_state),
        ),
    ),
    "linear_svm": ModelDefinition(
        name="linear_svm",
        complexity_rank=2,
        param_grid=(
            {"C": 0.01},
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ),
        factory=lambda params, random_state: SVC(
            C=float(params["C"]),
            kernel="linear",
            probability=True,
            random_state=int(random_state),
        ),
    ),
}


def create(name: str, params: dict[str, Any], random_state: int) -> Any:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name].factory(params, random_state)


def candidate_params(name: str) -> list[dict[str, Any]]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return [dict(params) for params in _REGISTRY[name].param_grid]


def model_complexity_rank(name: str) -> int:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return int(_REGISTRY[name].complexity_rank)


def registered_models() -> list[str]:
    return sorted(_REGISTRY.keys())
