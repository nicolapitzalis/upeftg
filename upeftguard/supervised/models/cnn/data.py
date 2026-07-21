from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from ...contracts import (
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    SupervisedFeatureBundle,
)


_DEPTH_LABEL_PATTERN = re.compile(r"^(encoder|decoder)\.layer(\d+)$")
_SUPPORTED_ATTENTION_KINDS = ("self", "cross")


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
                (self.normalized_arch_index_index + 1 if self.normalized_arch_index_index is not None else 0),
                (self.normalized_sequence_index_index + 1 if self.normalized_sequence_index_index is not None else 0),
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
        raise ValueError(f"group_mask shape {group_mask.shape} does not match tensor depth shape {values.shape[:2]}")
    if value_mask.shape != values.shape:
        raise ValueError(f"value_mask shape {value_mask.shape} does not match tensor shape {values.shape}")

    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    raw_depth_labels = metadata.get("depth_labels")
    if not isinstance(raw_depth_labels, list) or len(raw_depth_labels) != int(values.shape[1]):
        raise ValueError("cnn_1d requires metadata.depth_labels aligned to the canonical depth axis")
    raw_slot_names = metadata.get("slot_names")
    if not isinstance(raw_slot_names, list) or len(raw_slot_names) != int(values.shape[2]):
        raise ValueError("cnn_1d requires metadata.slot_names aligned to the canonical slot axis")
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
            f"cnn_1d expected canonical depth labels like 'encoder.layer12' or 'decoder.layer7', got {depth_label!r}"
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
        slot_populated = bool(np.logical_and(valid_depth_mask, value_mask[:, :, int(slot_idx), :]).any())
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
        raise ValueError("cnn_1d depth_feature_mode must be one of {'absolute', 'normalized', 'both'}")

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
        normalized_arch_index_index=(None if normalized_arch_index_index is None else int(normalized_arch_index_index)),
        normalized_sequence_index_index=(
            None if normalized_sequence_index_index is None else int(normalized_sequence_index_index)
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
        std[valid_channels] = np.sqrt(np.maximum(sq_sums[valid_channels] / counts[valid_channels], 1e-6)).astype(
            np.float32
        )
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
                resolved[:, continuous_indices] - normalization_stats.channel_mean[continuous_indices][None, :]
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
    normalize_continuous_features: bool = True,
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
            raise ValueError(f"cnn_1d sample at row index {sample_idx} has zero valid layers after applying group_mask")
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
        if bool(normalize_continuous_features):
            resolved_stats = _compute_normalization_stats(
                sequences=tuple(sequences),
                continuous_valid_masks=tuple(continuous_valid_masks),
                input_dim=int(channel_layout.input_dim),
            )
        else:
            resolved_stats = CNNNormalizationStats(
                channel_mean=np.zeros(int(channel_layout.input_dim), dtype=np.float32),
                channel_std=np.ones(int(channel_layout.input_dim), dtype=np.float32),
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
        raise ValueError(f"cnn_1d expected 2D layer sequences, got shape={resolved_sequences[0].shape}")
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
