from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "run_grouped_tsne_analysis",
    "run_grouped_tsne_sweep_analysis",
    "run_layer_value_scatter_analysis",
    "run_unsupervised_layer_scatter_pipeline",
    "run_unsupervised_tsne_pipeline",
    "run_gmm_train_inference_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in {
        "run_grouped_tsne_analysis",
        "run_grouped_tsne_sweep_analysis",
        "run_layer_value_scatter_analysis",
        "run_unsupervised_layer_scatter_pipeline",
        "run_unsupervised_tsne_pipeline",
    }:
        module = import_module(".analysis", __name__)
        return getattr(module, name)

    if name == "run_gmm_train_inference_pipeline":
        module = import_module(".gmm_train_inference", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
