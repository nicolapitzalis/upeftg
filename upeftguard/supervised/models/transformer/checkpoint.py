from __future__ import annotations

from pathlib import Path

import numpy as np

from ...tasks import task_spec_from_payload as _task_spec_from_payload
from ..common.torch_runtime import load_torch_checkpoint_payload
from .data import (
    TransformerNormalizationStats,
    layout_from_payload as _layout_from_payload,
)
from .estimator import (
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_MAX_EPOCHS,
    TRANSFORMER_PATIENCE,
    TransformerSupervisedModel,
)
from .network import TRANSFORMER_POOLING_MEAN


_torch_load_checkpoint = load_torch_checkpoint_payload


def load_transformer_checkpoint(path: Path) -> TransformerSupervisedModel:
    payload = _torch_load_checkpoint(path)
    backend = str(payload.get("backend") or "")
    if backend != "transformer":
        raise ValueError(f"Unsupported transformer checkpoint backend={backend!r}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Transformer checkpoint is missing config")
    task_spec = _task_spec_from_payload(payload.get("task"))
    model = TransformerSupervisedModel(
        d_model=int(config["d_model"]),
        nhead=int(config["nhead"]),
        slot_num_layers=int(config.get("slot_num_layers", 1)),
        layer_num_layers=int(config.get("layer_num_layers", 2)),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        norm_first=bool(config.get("norm_first", True)),
        pooling=str(config.get("pooling", TRANSFORMER_POOLING_MEAN)),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        random_state=int(config.get("random_state", 42)),
        task_spec=task_spec,
        max_epochs=int(config.get("max_epochs", TRANSFORMER_MAX_EPOCHS)),
        batch_size=int(config.get("batch_size", TRANSFORMER_BATCH_SIZE)),
        patience=int(config.get("patience", TRANSFORMER_PATIENCE)),
        class_weight_loss=bool(config.get("class_weight_loss", False)),
        rank_label_weight_loss=bool(config.get("rank_label_weight_loss", False)),
        normalize_input_features=bool(config.get("normalize_input_features", True)),
        include_value_mask_channels=bool(config.get("include_value_mask_channels", False)),
    )

    layout = _layout_from_payload(payload.get("layout"))
    normalization_payload = payload.get("normalization")
    if not isinstance(normalization_payload, dict):
        raise ValueError("Transformer checkpoint is missing normalization")
    feature_mean = np.asarray(normalization_payload.get("feature_mean"), dtype=np.float32)
    feature_std = np.asarray(normalization_payload.get("feature_std"), dtype=np.float32)
    if feature_mean.ndim != 1 or feature_std.ndim != 1 or feature_mean.shape != feature_std.shape:
        raise ValueError("Transformer checkpoint normalization arrays must be aligned 1D arrays")

    model.layout_ = layout
    model.normalization_stats_ = TransformerNormalizationStats(
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    model.feature_mean_ = feature_mean
    model.feature_std_ = feature_std
    model.classes_ = np.asarray(payload.get("classes", np.arange(task_spec.n_classes)), dtype=np.int32)
    model.class_names_ = tuple(str(x) for x in payload.get("class_names", task_spec.class_names))
    fit_summary = payload.get("fit_summary")
    model._fit_summary = dict(fit_summary) if isinstance(fit_summary, dict) else {}
    model.model_ = model._build_model(layout=layout)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Transformer checkpoint is missing state_dict")
    model.model_.load_state_dict(state_dict)
    model.model_.eval()
    return model
