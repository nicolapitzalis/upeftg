from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...tasks import task_spec_from_payload as _task_spec_from_payload
from ..common.torch_runtime import load_torch_checkpoint_payload
from .data import CNNChannelLayout, CNNNormalizationStats
from .estimator import (
    CNN_BATCH_SIZE,
    CNN_MAX_EPOCHS,
    CNN_PATIENCE,
    CNN1DSupervisedModel,
)
from .dann import CNN1DDANNSupervisedModel


_torch_load_checkpoint = load_torch_checkpoint_payload


def _channel_layout_from_payload(payload: Any) -> CNNChannelLayout:
    if not isinstance(payload, dict):
        raise ValueError("CNN checkpoint is missing channel_layout")

    def required_int(name: str) -> int:
        if name not in payload:
            raise ValueError(f"CNN checkpoint channel_layout is missing {name!r}")
        return int(payload[name])

    def optional_int(name: str) -> int | None:
        value = payload.get(name)
        return None if value is None else int(value)

    return CNNChannelLayout(
        slot_names=tuple(str(x) for x in payload.get("slot_names", [])),
        feature_names=tuple(str(x) for x in payload.get("feature_names", [])),
        value_start=required_int("value_start"),
        value_end=required_int("value_end"),
        value_mask_start=required_int("value_mask_start"),
        value_mask_end=required_int("value_mask_end"),
        slot_presence_start=required_int("slot_presence_start"),
        slot_presence_end=required_int("slot_presence_end"),
        self_attention_index=required_int("self_attention_index"),
        cross_attention_index=required_int("cross_attention_index"),
        is_encoder_index=required_int("is_encoder_index"),
        is_decoder_index=required_int("is_decoder_index"),
        layer_index_index=optional_int("layer_index_index"),
        normalized_arch_index_index=optional_int("normalized_arch_index_index"),
        normalized_sequence_index_index=optional_int("normalized_sequence_index_index"),
        total_layers_index=optional_int("total_layers_index"),
        continuous_indices=tuple(int(x) for x in payload.get("continuous_indices", [])),
    )


def load_cnn_checkpoint(path: Path) -> CNN1DSupervisedModel:
    payload = _torch_load_checkpoint(path)
    backend = str(payload.get("backend") or "cnn_1d")
    if backend not in {"cnn_1d", "cnn_1d_dann"}:
        raise ValueError(f"Unsupported CNN checkpoint backend={backend!r}")

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CNN checkpoint is missing config")
    task_spec = _task_spec_from_payload(payload.get("task"))
    domain_payload = payload.get("domain_adaptation")
    domain_config = domain_payload if isinstance(domain_payload, dict) else {}

    common_kwargs = {
        "conv_channels": int(config["conv_channels"]),
        "num_conv_layers": int(config.get("num_conv_layers", 3)),
        "kernel_size": int(config["kernel_size"]),
        "stride": int(config.get("stride", 1)),
        "dilation": int(config.get("dilation", 1)),
        "dropout": float(config["dropout"]),
        "use_residual": bool(config.get("use_residual", True)),
        "normalization": str(config.get("normalization", "layernorm")),
        "pooling": str(config.get("pooling", "mean_max")),
        "include_total_layer_count": bool(config.get("include_total_layer_count", True)),
        "depth_feature_mode": str(config.get("depth_feature_mode", "both")),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "random_state": int(config.get("random_state", 42)),
        "task_spec": task_spec,
        "max_epochs": int(config.get("max_epochs", CNN_MAX_EPOCHS)),
        "batch_size": int(config.get("batch_size", CNN_BATCH_SIZE)),
        "patience": int(config.get("patience", CNN_PATIENCE)),
        "class_weight_loss": bool(config.get("class_weight_loss", False)),
        "rank_label_weight_loss": bool(config.get("rank_label_weight_loss", False)),
        "normalize_input_features": bool(config.get("normalize_input_features", True)),
    }

    if backend == "cnn_1d_dann":
        lambda_schedule = domain_config.get("lambda_schedule")
        if not isinstance(lambda_schedule, dict):
            lambda_schedule = {}
        model: CNN1DSupervisedModel = CNN1DDANNSupervisedModel(
            **common_kwargs,
            source_rank=int(domain_config.get("source_rank", 256)),
            dann_lambda_max=float(lambda_schedule.get("lambda_max", 1.0)),
            dann_lambda_gamma=float(lambda_schedule.get("gamma", 10.0)),
        )
        domain_class_names = tuple(str(x) for x in domain_config.get("domain_class_names", []))
        domain_rank_values = tuple(int(x) for x in domain_config.get("domain_rank_values", []))
        if not domain_class_names:
            domain_output_dim = 0
            state_dict = payload.get("state_dict")
            if isinstance(state_dict, dict):
                domain_weight = state_dict.get("domain_head.weight")
                if domain_weight is not None and hasattr(domain_weight, "shape"):
                    domain_output_dim = int(domain_weight.shape[0])
            domain_class_names = tuple(f"domain_{idx}" for idx in range(domain_output_dim))
        if not domain_rank_values:
            domain_rank_values = tuple(range(len(domain_class_names)))
        dann_model = model
        assert isinstance(dann_model, CNN1DDANNSupervisedModel)
        dann_model.domain_classes_ = np.arange(len(domain_class_names), dtype=np.int64)
        dann_model.domain_class_names_ = domain_class_names
        dann_model.domain_rank_values_ = domain_rank_values
        dann_model.domain_class_to_index_ = {name: int(idx) for idx, name in enumerate(domain_class_names)}
    else:
        model = CNN1DSupervisedModel(**common_kwargs)

    channel_layout = _channel_layout_from_payload(payload.get("channel_layout"))
    normalization_payload = payload.get("normalization")
    if not isinstance(normalization_payload, dict):
        raise ValueError("CNN checkpoint is missing normalization")
    channel_mean = np.asarray(normalization_payload.get("channel_mean"), dtype=np.float32)
    channel_std = np.asarray(normalization_payload.get("channel_std"), dtype=np.float32)
    if channel_mean.ndim != 1 or channel_std.ndim != 1 or channel_mean.shape != channel_std.shape:
        raise ValueError("CNN checkpoint normalization arrays must be aligned 1D arrays")

    model.channel_layout_ = channel_layout
    model.normalization_stats_ = CNNNormalizationStats(
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    model.channel_mean_ = channel_mean
    model.channel_std_ = channel_std
    model.input_channels_ = int(config.get("input_channels") or channel_layout.input_dim)
    model.classes_ = np.asarray(payload.get("classes", np.arange(task_spec.n_classes)), dtype=np.int32)
    model.class_names_ = tuple(str(x) for x in payload.get("class_names", task_spec.class_names))
    fit_summary = payload.get("fit_summary")
    model._fit_summary = dict(fit_summary) if isinstance(fit_summary, dict) else {}

    model.model_ = model._build_model(input_channels=int(model.input_channels_ or channel_layout.input_dim))
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("CNN checkpoint is missing state_dict")
    model.model_.load_state_dict(state_dict)
    model.model_.eval()
    return model
