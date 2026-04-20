from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .interfaces import (
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    SUPERVISED_TASK_MODE_BINARY,
    SupervisedTaskSpec,
    SupervisedFeatureBundle,
)

try:  # pragma: no cover - exercised in environments that install torch
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - soft dependency
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None


CNN_MAX_EPOCHS = 40
CNN_BATCH_SIZE = 16
CNN_PATIENCE = 5
_DEPTH_LABEL_PATTERN = re.compile(r"^(encoder|decoder)\.layer(\d+)$")
_SUPPORTED_ATTENTION_KINDS = ("self", "cross")
TorchModuleBase = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or F is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ModuleNotFoundError(
            "cnn_1d requires the optional 'torch' dependency, but torch is not installed"
        )


def _default_binary_task_spec() -> SupervisedTaskSpec:
    class_names = ("clean", "backdoored")
    return SupervisedTaskSpec(
        task_mode=SUPERVISED_TASK_MODE_BINARY,
        class_names=class_names,
        class_to_index={name: idx for idx, name in enumerate(class_names)},
        binary_projection=BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    )


@dataclass(frozen=True)
class CNNFeatureTensors:
    inputs: np.ndarray
    layer_mask: np.ndarray


@dataclass(frozen=True)
class CNNLayerVectorConfig:
    conv_channels: int = 64
    num_conv_layers: int = 3
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    dropout: float = 0.1
    use_residual: bool = True
    normalization: str = "layernorm"
    pooling: str = "mean_max"
    include_total_layer_count: bool = True
    depth_feature_mode: str = "both"


@dataclass(frozen=True)
class CNNNormalizationStats:
    channel_mean: np.ndarray
    channel_std: np.ndarray


@dataclass(frozen=True)
class CNNChannelLayout:
    slot_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    value_start: int
    value_end: int
    value_mask_start: int
    value_mask_end: int
    slot_presence_start: int
    slot_presence_end: int
    self_attention_index: int
    cross_attention_index: int
    is_encoder_index: int
    is_decoder_index: int
    layer_index_index: int | None
    normalized_arch_index_index: int | None
    normalized_sequence_index_index: int | None
    total_layers_index: int | None
    continuous_indices: tuple[int, ...]

    @property
    def input_dim(self) -> int:
        return int(
            max(
                self.value_end,
                self.value_mask_end,
                self.slot_presence_end,
                self.cross_attention_index + 1,
                self.is_decoder_index + 1,
                (self.layer_index_index + 1) if self.layer_index_index is not None else 0,
                (
                    self.normalized_arch_index_index + 1
                    if self.normalized_arch_index_index is not None
                    else 0
                ),
                (
                    self.normalized_sequence_index_index + 1
                    if self.normalized_sequence_index_index is not None
                    else 0
                ),
                (self.total_layers_index + 1) if self.total_layers_index is not None else 0,
            )
        )


@dataclass(frozen=True)
class CNNLayerVectorBatch:
    sequences: tuple[np.ndarray, ...]
    continuous_valid_masks: tuple[np.ndarray, ...]
    layer_names: tuple[tuple[str, ...], ...]
    channel_layout: CNNChannelLayout
    normalization_stats: CNNNormalizationStats


def _resolve_bundle_metadata(
    bundle: SupervisedFeatureBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    if bundle.representation_kind != ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND:
        raise ValueError(
            "cnn_1d requires architecture_independent_layer_sequence_aggregation input, "
            f"got {bundle.representation_kind!r}"
        )
    if bundle.group_mask is None or bundle.value_mask is None:
        raise ValueError("cnn_1d requires group_mask and value_mask companions")

    values = np.asarray(bundle.values, dtype=np.float32)
    group_mask = np.asarray(bundle.group_mask, dtype=bool)
    value_mask = np.asarray(bundle.value_mask, dtype=bool)
    if values.ndim != 4:
        raise ValueError(f"cnn_1d expects a 4D tensor, got shape={values.shape}")
    if group_mask.shape != values.shape[:2]:
        raise ValueError(
            f"group_mask shape {group_mask.shape} does not match tensor depth shape {values.shape[:2]}"
        )
    if value_mask.shape != values.shape:
        raise ValueError(
            f"value_mask shape {value_mask.shape} does not match tensor shape {values.shape}"
        )

    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    raw_depth_labels = metadata.get("depth_labels")
    if not isinstance(raw_depth_labels, list) or len(raw_depth_labels) != int(values.shape[1]):
        raise ValueError(
            "cnn_1d requires metadata.depth_labels aligned to the canonical depth axis"
        )
    raw_slot_names = metadata.get("slot_names")
    if not isinstance(raw_slot_names, list) or len(raw_slot_names) != int(values.shape[2]):
        raise ValueError(
            "cnn_1d requires metadata.slot_names aligned to the canonical slot axis"
        )
    raw_feature_names = metadata.get("emitted_feature_names")
    if not isinstance(raw_feature_names, list) or len(raw_feature_names) != int(values.shape[3]):
        raw_feature_names = [f"feature_{idx}" for idx in range(int(values.shape[3]))]

    return (
        values,
        group_mask,
        value_mask,
        [str(x) for x in raw_depth_labels],
        [str(x) for x in raw_slot_names],
        [str(x) for x in raw_feature_names],
    )


def _parse_depth_label(depth_label: str) -> tuple[str, int]:
    match = _DEPTH_LABEL_PATTERN.match(str(depth_label).strip())
    if match is None:
        raise ValueError(
            "cnn_1d expected canonical depth labels like 'encoder.layer12' or 'decoder.layer7', "
            f"got {depth_label!r}"
        )
    return str(match.group(1)), int(match.group(2))


def _normalized_position(index: int, upper_bound: int) -> float:
    if upper_bound <= 0:
        return 0.0
    return float(index) / float(upper_bound)


def _resolve_retained_slot_indices(
    slot_names: list[str],
    *,
    group_mask: np.ndarray,
    value_mask: np.ndarray,
) -> list[int]:
    retained_indices: list[int] = []
    populated_other_slots: list[str] = []
    valid_depth_mask = np.asarray(group_mask[:, :, None], dtype=bool)

    for slot_idx, slot_name in enumerate(slot_names):
        attention_kind = str(slot_name).partition(".")[0]
        if attention_kind in _SUPPORTED_ATTENTION_KINDS:
            retained_indices.append(int(slot_idx))
            continue
        slot_populated = bool(
            np.logical_and(valid_depth_mask, value_mask[:, :, int(slot_idx), :]).any()
        )
        if slot_populated:
            populated_other_slots.append(str(slot_name))

    if populated_other_slots:
        preview = ", ".join(populated_other_slots[:5])
        raise ValueError(
            "cnn_1d only supports self/cross attention slots, but the layer-sequence bundle "
            f"contains populated unsupported slot(s): {preview}"
        )
    if not retained_indices:
        raise ValueError("cnn_1d requires at least one populated self/cross attention slot")
    return retained_indices


def _build_channel_layout(
    *,
    retained_slot_names: list[str],
    feature_names: list[str],
    include_total_layer_count: bool,
    depth_feature_mode: str,
) -> CNNChannelLayout:
    if depth_feature_mode not in {"absolute", "normalized", "both"}:
        raise ValueError(
            "cnn_1d depth_feature_mode must be one of {'absolute', 'normalized', 'both'}"
        )

    value_width = int(len(retained_slot_names) * len(feature_names))
    cursor = 0
    value_start = cursor
    value_end = value_start + value_width
    cursor = value_end

    value_mask_start = cursor
    value_mask_end = value_mask_start + value_width
    cursor = value_mask_end

    slot_presence_start = cursor
    slot_presence_end = slot_presence_start + int(len(retained_slot_names))
    cursor = slot_presence_end

    self_attention_index = cursor
    cursor += 1
    cross_attention_index = cursor
    cursor += 1
    is_encoder_index = cursor
    cursor += 1
    is_decoder_index = cursor
    cursor += 1

    layer_index_index = None
    if depth_feature_mode in {"absolute", "both"}:
        layer_index_index = cursor
        cursor += 1

    normalized_arch_index_index = None
    normalized_sequence_index_index = None
    if depth_feature_mode in {"normalized", "both"}:
        normalized_arch_index_index = cursor
        cursor += 1
        normalized_sequence_index_index = cursor
        cursor += 1

    total_layers_index = None
    if include_total_layer_count:
        total_layers_index = cursor
        cursor += 1

    continuous_indices = list(range(value_start, value_end))
    if layer_index_index is not None:
        continuous_indices.append(int(layer_index_index))
    if normalized_arch_index_index is not None:
        continuous_indices.append(int(normalized_arch_index_index))
    if normalized_sequence_index_index is not None:
        continuous_indices.append(int(normalized_sequence_index_index))
    if total_layers_index is not None:
        continuous_indices.append(int(total_layers_index))

    return CNNChannelLayout(
        slot_names=tuple(str(x) for x in retained_slot_names),
        feature_names=tuple(str(x) for x in feature_names),
        value_start=int(value_start),
        value_end=int(value_end),
        value_mask_start=int(value_mask_start),
        value_mask_end=int(value_mask_end),
        slot_presence_start=int(slot_presence_start),
        slot_presence_end=int(slot_presence_end),
        self_attention_index=int(self_attention_index),
        cross_attention_index=int(cross_attention_index),
        is_encoder_index=int(is_encoder_index),
        is_decoder_index=int(is_decoder_index),
        layer_index_index=None if layer_index_index is None else int(layer_index_index),
        normalized_arch_index_index=(
            None
            if normalized_arch_index_index is None
            else int(normalized_arch_index_index)
        ),
        normalized_sequence_index_index=(
            None
            if normalized_sequence_index_index is None
            else int(normalized_sequence_index_index)
        ),
        total_layers_index=None if total_layers_index is None else int(total_layers_index),
        continuous_indices=tuple(int(x) for x in continuous_indices),
    )


def _compute_normalization_stats(
    *,
    sequences: tuple[np.ndarray, ...],
    continuous_valid_masks: tuple[np.ndarray, ...],
    input_dim: int,
) -> CNNNormalizationStats:
    means = np.zeros(int(input_dim), dtype=np.float32)
    std = np.ones(int(input_dim), dtype=np.float32)
    counts = np.zeros(int(input_dim), dtype=np.int64)
    sums = np.zeros(int(input_dim), dtype=np.float64)

    for sequence, valid_mask in zip(sequences, continuous_valid_masks):
        counts += np.asarray(valid_mask.sum(axis=0), dtype=np.int64)
        sums += np.asarray((sequence * valid_mask.astype(np.float32)).sum(axis=0), dtype=np.float64)

    valid_channels = counts > 0
    if np.any(valid_channels):
        means[valid_channels] = (sums[valid_channels] / counts[valid_channels]).astype(np.float32)

    sq_sums = np.zeros(int(input_dim), dtype=np.float64)
    for sequence, valid_mask in zip(sequences, continuous_valid_masks):
        centered = np.where(valid_mask, sequence - means[None, :], 0.0)
        sq_sums += np.square(centered, dtype=np.float64).sum(axis=0, dtype=np.float64)

    if np.any(valid_channels):
        std[valid_channels] = np.sqrt(
            np.maximum(sq_sums[valid_channels] / counts[valid_channels], 1e-6)
        ).astype(np.float32)
    std = np.where(std > 1e-6, std, np.ones_like(std, dtype=np.float32))

    return CNNNormalizationStats(
        channel_mean=means,
        channel_std=std,
    )


def _apply_normalization(
    *,
    sequences: tuple[np.ndarray, ...],
    continuous_valid_masks: tuple[np.ndarray, ...],
    normalization_stats: CNNNormalizationStats,
    channel_layout: CNNChannelLayout,
) -> tuple[np.ndarray, ...]:
    continuous_indices = np.asarray(channel_layout.continuous_indices, dtype=np.int64)
    normalized_sequences: list[np.ndarray] = []

    for sequence, valid_mask in zip(sequences, continuous_valid_masks):
        resolved = np.asarray(sequence, dtype=np.float32).copy()
        if continuous_indices.size > 0:
            resolved[:, continuous_indices] = (
                resolved[:, continuous_indices]
                - normalization_stats.channel_mean[continuous_indices][None, :]
            ) / normalization_stats.channel_std[continuous_indices][None, :]
            resolved[:, continuous_indices] = np.where(
                valid_mask[:, continuous_indices],
                resolved[:, continuous_indices],
                0.0,
            )
        normalized_sequences.append(resolved.astype(np.float32, copy=False))

    return tuple(normalized_sequences)


def build_per_layer_vectors(
    bundle: SupervisedFeatureBundle,
    normalization_stats: CNNNormalizationStats | None = None,
    *,
    include_total_layer_count: bool = True,
    depth_feature_mode: str = "both",
) -> CNNLayerVectorBatch:
    (
        values,
        group_mask,
        value_mask,
        depth_labels,
        slot_names,
        feature_names,
    ) = _resolve_bundle_metadata(bundle)
    retained_slot_indices = _resolve_retained_slot_indices(
        slot_names,
        group_mask=group_mask,
        value_mask=value_mask,
    )
    retained_slot_names = [slot_names[idx] for idx in retained_slot_indices]
    channel_layout = _build_channel_layout(
        retained_slot_names=retained_slot_names,
        feature_names=feature_names,
        include_total_layer_count=bool(include_total_layer_count),
        depth_feature_mode=str(depth_feature_mode),
    )

    retained_kind_array = np.asarray(
        [str(slot_name).partition(".")[0] for slot_name in retained_slot_names],
        dtype=object,
    )
    self_slot_mask = retained_kind_array == "self"
    cross_slot_mask = retained_kind_array == "cross"

    sequences: list[np.ndarray] = []
    continuous_valid_masks: list[np.ndarray] = []
    layer_names: list[tuple[str, ...]] = []

    for sample_idx in range(int(values.shape[0])):
        valid_depth_indices = np.flatnonzero(group_mask[int(sample_idx)])
        if valid_depth_indices.size == 0:
            raise ValueError(
                f"cnn_1d sample at row index {sample_idx} has zero valid layers after applying group_mask"
            )
        sample_depth_labels = [depth_labels[int(depth_idx)] for depth_idx in valid_depth_indices.tolist()]
        parsed_depths = [_parse_depth_label(depth_label) for depth_label in sample_depth_labels]
        arch_layer_max: dict[str, int] = {}
        for architecture_block, layer_index in parsed_depths:
            current_max = arch_layer_max.get(str(architecture_block))
            if current_max is None or int(layer_index) > current_max:
                arch_layer_max[str(architecture_block)] = int(layer_index)

        total_valid_layers = int(valid_depth_indices.size)
        sample_sequence = np.zeros((total_valid_layers, channel_layout.input_dim), dtype=np.float32)
        sample_continuous_valid = np.zeros_like(sample_sequence, dtype=bool)

        for sequence_index, depth_idx in enumerate(valid_depth_indices.tolist()):
            retained_values = np.asarray(
                values[int(sample_idx), int(depth_idx), retained_slot_indices, :],
                dtype=np.float32,
            )
            retained_value_mask = np.asarray(
                value_mask[int(sample_idx), int(depth_idx), retained_slot_indices, :],
                dtype=bool,
            )
            flat_values = retained_values.reshape(-1)
            flat_value_mask = retained_value_mask.reshape(-1)

            row = sample_sequence[int(sequence_index)]
            row[channel_layout.value_start : channel_layout.value_end] = np.where(
                flat_value_mask,
                flat_values,
                0.0,
            )
            sample_continuous_valid[
                int(sequence_index),
                channel_layout.value_start : channel_layout.value_end,
            ] = flat_value_mask

            slot_present = retained_value_mask.any(axis=1).astype(np.float32, copy=False)
            row[channel_layout.value_mask_start : channel_layout.value_mask_end] = flat_value_mask.astype(
                np.float32,
                copy=False,
            )
            row[channel_layout.slot_presence_start : channel_layout.slot_presence_end] = slot_present
            row[channel_layout.self_attention_index] = float(
                bool(np.any(slot_present[self_slot_mask])) if self_slot_mask.size > 0 else False
            )
            row[channel_layout.cross_attention_index] = float(
                bool(np.any(slot_present[cross_slot_mask])) if cross_slot_mask.size > 0 else False
            )

            architecture_block, layer_index = parsed_depths[int(sequence_index)]
            row[channel_layout.is_encoder_index] = float(architecture_block == "encoder")
            row[channel_layout.is_decoder_index] = float(architecture_block == "decoder")

            if channel_layout.layer_index_index is not None:
                row[int(channel_layout.layer_index_index)] = float(layer_index)
                sample_continuous_valid[int(sequence_index), int(channel_layout.layer_index_index)] = True
            if channel_layout.normalized_arch_index_index is not None:
                arch_max_index = int(arch_layer_max.get(str(architecture_block), 0))
                row[int(channel_layout.normalized_arch_index_index)] = _normalized_position(
                    int(layer_index),
                    int(arch_max_index),
                )
                sample_continuous_valid[
                    int(sequence_index),
                    int(channel_layout.normalized_arch_index_index),
                ] = True
            if channel_layout.normalized_sequence_index_index is not None:
                row[int(channel_layout.normalized_sequence_index_index)] = _normalized_position(
                    int(sequence_index),
                    int(total_valid_layers - 1),
                )
                sample_continuous_valid[
                    int(sequence_index),
                    int(channel_layout.normalized_sequence_index_index),
                ] = True
            if channel_layout.total_layers_index is not None:
                row[int(channel_layout.total_layers_index)] = float(total_valid_layers)
                sample_continuous_valid[
                    int(sequence_index),
                    int(channel_layout.total_layers_index),
                ] = True

        sequences.append(sample_sequence)
        continuous_valid_masks.append(sample_continuous_valid)
        layer_names.append(tuple(sample_depth_labels))

    resolved_stats = normalization_stats
    if resolved_stats is None:
        resolved_stats = _compute_normalization_stats(
            sequences=tuple(sequences),
            continuous_valid_masks=tuple(continuous_valid_masks),
            input_dim=int(channel_layout.input_dim),
        )
    normalized_sequences = _apply_normalization(
        sequences=tuple(sequences),
        continuous_valid_masks=tuple(continuous_valid_masks),
        normalization_stats=resolved_stats,
        channel_layout=channel_layout,
    )

    return CNNLayerVectorBatch(
        sequences=normalized_sequences,
        continuous_valid_masks=tuple(continuous_valid_masks),
        layer_names=tuple(layer_names),
        channel_layout=channel_layout,
        normalization_stats=resolved_stats,
    )


def pad_layer_sequence_batch(sequences: tuple[np.ndarray, ...] | list[np.ndarray]) -> CNNFeatureTensors:
    if not sequences:
        raise ValueError("cnn_1d requires at least one layer sequence to build a batch")

    resolved_sequences = [np.asarray(sequence, dtype=np.float32) for sequence in sequences]
    if resolved_sequences[0].ndim != 2:
        raise ValueError(
            f"cnn_1d expected 2D layer sequences, got shape={resolved_sequences[0].shape}"
        )
    input_dim = int(resolved_sequences[0].shape[1])
    max_layers = 0
    for sequence in resolved_sequences:
        if sequence.ndim != 2:
            raise ValueError(f"cnn_1d expected 2D layer sequences, got shape={sequence.shape}")
        if int(sequence.shape[1]) != input_dim:
            raise ValueError("cnn_1d layer sequences must all share the same feature dimension")
        if int(sequence.shape[0]) <= 0:
            raise ValueError("cnn_1d does not support empty layer sequences")
        max_layers = max(max_layers, int(sequence.shape[0]))

    batch_inputs = np.zeros((len(resolved_sequences), max_layers, input_dim), dtype=np.float32)
    layer_mask = np.zeros((len(resolved_sequences), max_layers), dtype=np.float32)
    for sample_idx, sequence in enumerate(resolved_sequences):
        n_layers = int(sequence.shape[0])
        batch_inputs[int(sample_idx), :n_layers, :] = sequence
        layer_mask[int(sample_idx), :n_layers] = 1.0

    return CNNFeatureTensors(inputs=batch_inputs, layer_mask=layer_mask)


def masked_mean_pool(hidden: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
    mask = layer_mask.unsqueeze(1).to(dtype=hidden.dtype)
    valid_counts = torch.clamp(mask.sum(dim=2), min=1.0)
    return (hidden * mask).sum(dim=2) / valid_counts


def masked_max_pool(hidden: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
    mask = layer_mask.unsqueeze(1) > 0
    masked_hidden = hidden.masked_fill(~mask, float("-inf"))
    pooled = masked_hidden.max(dim=2).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


class _LayerNorm1d(TorchModuleBase):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.norm = nn.LayerNorm(int(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _Conv1DBlock(TorchModuleBase):
    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        normalization: str,
        use_residual: bool,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None

        padding = max(0, (int(kernel_size) - 1) * int(dilation) // 2)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.use_residual = bool(use_residual)
        self.conv = nn.Conv1d(
            int(input_channels),
            int(output_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            dilation=int(dilation),
        )
        if str(normalization) == "layernorm":
            self.normalization = _LayerNorm1d(int(output_channels))
        elif str(normalization) == "batchnorm":
            self.normalization = nn.BatchNorm1d(int(output_channels))
        elif str(normalization) == "none":
            self.normalization = None
        else:
            raise ValueError(
                "cnn_1d normalization must be one of {'layernorm', 'batchnorm', 'none'}"
            )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def _propagate_mask(self, layer_mask: torch.Tensor) -> torch.Tensor:
        assert F is not None
        pooled_mask = F.max_pool1d(
            layer_mask.unsqueeze(1),
            kernel_size=int(self.kernel_size),
            stride=int(self.stride),
            padding=int(self.padding),
            dilation=int(self.dilation),
        )
        return (pooled_mask.squeeze(1) > 0).to(dtype=layer_mask.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self.conv(hidden)
        next_layer_mask = self._propagate_mask(layer_mask)
        if self.normalization is not None:
            next_hidden = self.normalization(next_hidden)
        next_hidden = self.activation(next_hidden)
        next_hidden = self.dropout(next_hidden)

        mask = next_layer_mask.unsqueeze(1).to(dtype=next_hidden.dtype)
        next_hidden = next_hidden * mask
        if (
            self.use_residual
            and hidden.shape == next_hidden.shape
            and tuple(layer_mask.shape) == tuple(next_layer_mask.shape)
        ):
            next_hidden = (next_hidden + hidden * mask) * mask
        return next_hidden, next_layer_mask


class Conv1DLayerAggregator(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None

        if str(config.pooling) not in {"mean", "max", "mean_max"}:
            raise ValueError("cnn_1d pooling must be one of {'mean', 'max', 'mean_max'}")

        blocks: list[_Conv1DBlock] = []
        in_channels = int(input_dim)
        for _ in range(int(config.num_conv_layers)):
            blocks.append(
                _Conv1DBlock(
                    input_channels=int(in_channels),
                    output_channels=int(config.conv_channels),
                    kernel_size=int(config.kernel_size),
                    stride=int(config.stride),
                    dilation=int(config.dilation),
                    dropout=float(config.dropout),
                    normalization=str(config.normalization),
                    use_residual=bool(config.use_residual),
                )
            )
            in_channels = int(config.conv_channels)
        self.blocks = nn.ModuleList(blocks)
        self.pooling = str(config.pooling)
        self.output_channels = int(config.conv_channels)
        self.embedding_dim = (
            int(config.conv_channels) * 2
            if str(config.pooling) == "mean_max"
            else int(config.conv_channels)
        )

    def forward(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        hidden = inputs.transpose(1, 2)
        current_mask = layer_mask
        for block in self.blocks:
            hidden, current_mask = block(hidden, current_mask)
        hidden = hidden * current_mask.unsqueeze(1).to(dtype=hidden.dtype)

        if self.pooling == "mean":
            return masked_mean_pool(hidden, current_mask)
        if self.pooling == "max":
            return masked_max_pool(hidden, current_mask)
        return torch.cat(
            [
                masked_mean_pool(hidden, current_mask),
                masked_max_pool(hidden, current_mask),
            ],
            dim=1,
        )


class _CNNLayerSequenceClassifier(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
        output_dim: int,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.aggregator = Conv1DLayerAggregator(
            input_dim=int(input_dim),
            config=config,
        )
        self.output_dim = int(output_dim)
        self.head = nn.Linear(int(self.aggregator.embedding_dim), int(self.output_dim))

    def forward(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        embedding = self.aggregator(inputs, layer_mask)
        logits = self.head(embedding)
        if self.output_dim == 1:
            return logits.squeeze(1)
        return logits


class CNN1DSupervisedModel:
    def __init__(
        self,
        *,
        conv_channels: int,
        num_conv_layers: int = 3,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float,
        use_residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean_max",
        include_total_layer_count: bool = True,
        depth_feature_mode: str = "both",
        learning_rate: float,
        weight_decay: float,
        random_state: int,
        task_spec: SupervisedTaskSpec | None = None,
        max_epochs: int = CNN_MAX_EPOCHS,
        batch_size: int = CNN_BATCH_SIZE,
        patience: int = CNN_PATIENCE,
    ) -> None:
        _require_torch()
        if int(conv_channels) <= 0:
            raise ValueError("cnn_1d conv_channels must be positive")
        if int(num_conv_layers) <= 0:
            raise ValueError("cnn_1d num_conv_layers must be positive")
        if int(kernel_size) <= 0:
            raise ValueError("cnn_1d kernel_size must be positive")
        if int(stride) <= 0:
            raise ValueError("cnn_1d stride must be positive")
        if int(dilation) <= 0:
            raise ValueError("cnn_1d dilation must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("cnn_1d dropout must be in [0, 1)")
        self.layer_vector_config = CNNLayerVectorConfig(
            conv_channels=int(conv_channels),
            num_conv_layers=int(num_conv_layers),
            kernel_size=int(kernel_size),
            stride=int(stride),
            dilation=int(dilation),
            dropout=float(dropout),
            use_residual=bool(use_residual),
            normalization=str(normalization),
            pooling=str(pooling),
            include_total_layer_count=bool(include_total_layer_count),
            depth_feature_mode=str(depth_feature_mode),
        )
        self.conv_channels = int(conv_channels)
        self.num_conv_layers = int(num_conv_layers)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)
        self.normalization = str(normalization)
        self.pooling = str(pooling)
        self.include_total_layer_count = bool(include_total_layer_count)
        self.depth_feature_mode = str(depth_feature_mode)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.task_spec = task_spec if task_spec is not None else _default_binary_task_spec()
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.model_: _CNNLayerSequenceClassifier | None = None
        self.normalization_stats_: CNNNormalizationStats | None = None
        self.channel_layout_: CNNChannelLayout | None = None
        self.channel_mean_: np.ndarray | None = None
        self.channel_std_: np.ndarray | None = None
        self.input_channels_: int | None = None
        self.classes_ = np.arange(self.task_spec.n_classes, dtype=np.int32)
        self.class_names_ = tuple(str(x) for x in self.task_spec.class_names)
        self.backend_name_ = "cnn_1d"
        self._fit_summary: dict[str, Any] = {}

    def _set_random_seeds(self) -> None:
        assert torch is not None
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    def _set_threads(self, n_jobs: int | None) -> None:
        assert torch is not None
        if n_jobs is None:
            return
        resolved = int(n_jobs)
        if resolved <= 0:
            return
        torch.set_num_threads(resolved)

    def _prepare_numpy_inputs(self, bundle: SupervisedFeatureBundle) -> CNNFeatureTensors:
        vector_batch = build_per_layer_vectors(
            bundle,
            normalization_stats=self.normalization_stats_,
            include_total_layer_count=bool(self.include_total_layer_count),
            depth_feature_mode=str(self.depth_feature_mode),
        )
        if self.channel_layout_ is None:
            self.channel_layout_ = vector_batch.channel_layout
        elif self.channel_layout_ != vector_batch.channel_layout:
            raise ValueError(
                "cnn_1d encountered an incompatible layer-sequence channel layout between bundles"
            )

        if self.normalization_stats_ is None:
            self.normalization_stats_ = vector_batch.normalization_stats
            self.channel_mean_ = np.asarray(
                vector_batch.normalization_stats.channel_mean,
                dtype=np.float32,
            )
            self.channel_std_ = np.asarray(
                vector_batch.normalization_stats.channel_std,
                dtype=np.float32,
            )
        elif (
            self.channel_mean_ is None
            or self.channel_std_ is None
        ):
            self.channel_mean_ = np.asarray(
                self.normalization_stats_.channel_mean,
                dtype=np.float32,
            )
            self.channel_std_ = np.asarray(
                self.normalization_stats_.channel_std,
                dtype=np.float32,
            )

        padded_batch = pad_layer_sequence_batch(vector_batch.sequences)
        self.input_channels_ = int(padded_batch.inputs.shape[2])
        return padded_batch

    def _build_model(self, *, input_channels: int) -> _CNNLayerSequenceClassifier:
        _require_torch()
        model = _CNNLayerSequenceClassifier(
            input_dim=int(input_channels),
            config=self.layer_vector_config,
            output_dim=(1 if self.task_spec.is_binary else self.task_spec.n_classes),
        )
        self.input_channels_ = int(input_channels)
        return model

    def _logits_from_loader(self, loader: DataLoader) -> np.ndarray:
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, batch_layer_mask in loader:
                logits = self.model_(batch_inputs, batch_layer_mask)
                outputs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        if not outputs:
            if self.task_spec.is_binary:
                return np.asarray([], dtype=np.float64)
            return np.asarray([], dtype=np.float64).reshape(0, self.task_spec.n_classes)
        combined = np.concatenate(outputs, axis=0)
        if self.task_spec.is_binary:
            return combined.reshape(-1)
        return np.asarray(combined, dtype=np.float64)

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
    ) -> "CNN1DSupervisedModel":
        _require_torch()
        assert torch is not None
        self._set_random_seeds()
        self._set_threads(n_jobs)

        train_tensors = self._prepare_numpy_inputs(bundle)
        if self.task_spec.is_binary:
            y_train = np.asarray(labels, dtype=np.float32).reshape(-1)
        else:
            y_train = np.asarray(labels, dtype=np.int64).reshape(-1)
        if train_tensors.inputs.shape[0] != y_train.shape[0]:
            raise ValueError("cnn_1d training features/labels length mismatch")

        train_dataset = TensorDataset(
            torch.from_numpy(train_tensors.inputs),
            torch.from_numpy(train_tensors.layer_mask),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(1, len(train_dataset))),
            shuffle=True,
        )

        valid_loader = None
        y_valid = None
        if validation_data is not None:
            valid_bundle, valid_labels = validation_data
            valid_tensors = self._prepare_numpy_inputs(valid_bundle)
            y_valid = np.asarray(valid_labels, dtype=np.int32).reshape(-1)
            valid_dataset = TensorDataset(
                torch.from_numpy(valid_tensors.inputs),
                torch.from_numpy(valid_tensors.layer_mask),
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=min(self.batch_size, max(1, len(valid_dataset))),
                shuffle=False,
            )

        self.model_ = self._build_model(input_channels=int(train_tensors.inputs.shape[2]))
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.BCEWithLogitsLoss() if self.task_spec.is_binary else nn.CrossEntropyLoss()

        best_state = None
        best_metric = -math.inf
        best_epoch = -1
        stale_epochs = 0
        history: list[dict[str, float]] = []

        for epoch_idx in range(self.max_epochs):
            self.model_.train()
            train_loss_sum = 0.0
            train_count = 0
            for batch_inputs, batch_layer_mask, batch_labels in train_loader:
                optimizer.zero_grad()
                logits = self.model_(batch_inputs, batch_layer_mask)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                batch_size = int(batch_labels.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(1, train_count)
            metric = -train_loss
            if valid_loader is not None and y_valid is not None:
                valid_logits = self._logits_from_loader(valid_loader)
                if self.task_spec.is_binary:
                    metric = float(roc_auc_score(y_valid, valid_logits))
                else:
                    valid_pred = np.argmax(valid_logits, axis=1).astype(np.int32, copy=False)
                    metric = float(f1_score(y_valid, valid_pred, average="macro"))

            history.append(
                {
                    "epoch": float(epoch_idx),
                    "train_loss": float(train_loss),
                    "selection_metric": float(metric),
                    "selection_metric_name": str(self.task_spec.selection_metric_name),
                }
            )

            if metric > best_metric:
                best_metric = float(metric)
                best_epoch = int(epoch_idx)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if valid_loader is not None and stale_epochs >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("cnn_1d training did not produce a valid checkpoint")
        self.model_.load_state_dict(best_state)
        self._fit_summary = {
            "best_epoch": int(best_epoch),
            "selection_metric": float(best_metric),
            "selection_metric_name": str(self.task_spec.selection_metric_name),
            "epochs_ran": int(len(history)),
            "history": history,
        }
        return self

    def decision_function(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )
        return self._logits_from_loader(loader)

    def predict_proba(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        logits = self.decision_function(bundle)
        if self.task_spec.is_binary:
            probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64).reshape(-1)))
            return np.column_stack([1.0 - probabilities, probabilities]).astype(np.float64, copy=False)
        logits_2d = np.asarray(logits, dtype=np.float64)
        shifted = logits_2d - np.max(logits_2d, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        normalizer = np.sum(exp_logits, axis=1, keepdims=True)
        normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
        return np.asarray(exp_logits / normalizer, dtype=np.float64)

    def predict(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        if self.task_spec.is_binary:
            return (self.decision_function(bundle) >= 0.0).astype(np.int32, copy=False)
        probabilities = self.predict_proba(bundle)
        return np.argmax(probabilities, axis=1).astype(np.int32, copy=False)

    def save(self, path: Path) -> None:
        _require_torch()
        assert torch is not None
        if (
            self.model_ is None
            or self.channel_mean_ is None
            or self.channel_std_ is None
            or self.channel_layout_ is None
        ):
            raise RuntimeError("cnn_1d model has not been fit")
        payload = {
            "backend": self.backend_name_,
            "config": {
                "conv_channels": int(self.conv_channels),
                "num_conv_layers": int(self.num_conv_layers),
                "kernel_size": int(self.kernel_size),
                "stride": int(self.stride),
                "dilation": int(self.dilation),
                "dropout": float(self.dropout),
                "use_residual": bool(self.use_residual),
                "normalization": str(self.normalization),
                "pooling": str(self.pooling),
                "include_total_layer_count": bool(self.include_total_layer_count),
                "depth_feature_mode": str(self.depth_feature_mode),
                "learning_rate": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
                "random_state": int(self.random_state),
                "max_epochs": int(self.max_epochs),
                "batch_size": int(self.batch_size),
                "patience": int(self.patience),
                "input_channels": int(self.input_channels_ or 0),
                "task_mode": str(self.task_spec.task_mode),
                "num_classes": int(self.task_spec.n_classes),
            },
            "channel_layout": asdict(self.channel_layout_),
            "state_dict": self.model_.state_dict(),
            "normalization": {
                "channel_mean": np.asarray(self.channel_mean_, dtype=np.float32),
                "channel_std": np.asarray(self.channel_std_, dtype=np.float32),
            },
            "classes": np.asarray(self.classes_, dtype=np.int32),
            "class_names": list(self.class_names_),
            "task": self.task_spec.to_dict(),
            "fit_summary": dict(self._fit_summary),
        }
        torch.save(payload, Path(path).expanduser().resolve())
