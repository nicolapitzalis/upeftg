from __future__ import annotations

from typing import Callable

from .interfaces import SupervisedModel


_REGISTRY: dict[str, Callable[[], SupervisedModel]] = {}


def register(name: str, factory: Callable[[], SupervisedModel]) -> None:
    _REGISTRY[name] = factory


def create(name: str) -> SupervisedModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown supervised model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def registered_models() -> list[str]:
    return sorted(_REGISTRY.keys())
