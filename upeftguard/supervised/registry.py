from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


NormalizationFactory = Callable[[], Any]
EstimatorFactory = Callable[[dict[str, Any], int], Any]


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    complexity_rank: int
    normalization_policy: str
    normalization_factory: NormalizationFactory
    param_grid: tuple[dict[str, Any], ...]
    estimator_factory: EstimatorFactory


def _grid(**axes: list[Any]) -> tuple[dict[str, Any], ...]:
    keys = list(axes.keys())
    values = [list(axes[key]) for key in keys]
    return tuple(
        {key: value for key, value in zip(keys, combo)}
        for combo in product(*values)
    )


def _passthrough() -> str:
    return "passthrough"


def _standard_scaler() -> StandardScaler:
    return StandardScaler()


def _build_pipeline(definition: ModelDefinition, params: dict[str, Any], random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("normalizer", definition.normalization_factory()),
            ("model", definition.estimator_factory(params, random_state)),
        ]
    )


_REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        name="logistic_regression",
        complexity_rank=0,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state: LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            solver="lbfgs",
            max_iter=5000,
            random_state=int(random_state),
        ),
    ),
    "ridge_classifier": ModelDefinition(
        name="ridge_classifier",
        complexity_rank=1,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            alpha=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state: RidgeClassifier(
            alpha=float(params["alpha"]),
            class_weight=params["class_weight"],
            random_state=int(random_state),
        ),
    ),
    "linear_svm": ModelDefinition(
        name="linear_svm",
        complexity_rank=2,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state: SVC(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            kernel="linear",
            probability=True,
            random_state=int(random_state),
        ),
    ),
    "adaboost": ModelDefinition(
        name="adaboost",
        complexity_rank=3,
        normalization_policy="passthrough",
        normalization_factory=_passthrough,
        param_grid=_grid(
            max_depth=[1, 2],
            n_estimators=[50, 100, 200, 400],
            learning_rate=[0.05, 0.1, 0.5, 1.0],
        ),
        estimator_factory=lambda params, random_state: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=int(params["max_depth"]),
                random_state=int(random_state),
            ),
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            random_state=int(random_state),
        ),
    ),
    "kernel_svm": ModelDefinition(
        name="kernel_svm",
        complexity_rank=4,
        normalization_policy="standard_scaler",
        normalization_factory=_standard_scaler,
        param_grid=_grid(
            C=[1e-2, 1e-1, 1.0, 10.0, 100.0],
            gamma=["scale", 1e-2, 1e-1, 1.0],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state: SVC(
            C=float(params["C"]),
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            kernel="rbf",
            probability=False,
            random_state=int(random_state),
        ),
    ),
    "random_forest": ModelDefinition(
        name="random_forest",
        complexity_rank=5,
        normalization_policy="passthrough",
        normalization_factory=_passthrough,
        param_grid=_grid(
            n_estimators=[200, 400, 800],
            max_depth=[None, 8, 16],
            min_samples_leaf=[1, 2, 4],
            class_weight=[None, "balanced"],
        ),
        estimator_factory=lambda params, random_state: RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight=params["class_weight"],
            max_features="sqrt",
            n_jobs=1,
            random_state=int(random_state),
        ),
    ),
}


def create(name: str, params: dict[str, Any], random_state: int) -> Any:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return _build_pipeline(_REGISTRY[name], params, random_state)


def candidate_params(name: str) -> list[dict[str, Any]]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return [dict(params) for params in _REGISTRY[name].param_grid]


def model_complexity_rank(name: str) -> int:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return int(_REGISTRY[name].complexity_rank)


def normalization_policy(name: str) -> str:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return str(_REGISTRY[name].normalization_policy)


def registered_models() -> list[str]:
    return sorted(_REGISTRY.keys())
