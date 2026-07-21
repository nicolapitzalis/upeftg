from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...contracts import (
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    SupervisedFeatureBundle,
)


@dataclass(frozen=True)
class TransformerNormalizationStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray


@dataclass(frozen=True)
class TransformerLayerSequenceLayout:
    depth_labels: tuple[str, ...]
    slot_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    input_dim: int
    max_layers: int
    max_slots: int


@dataclass(frozen=True)
class TransformerFeatureTensors:
    values: np.ndarray
    slot_mask: np.ndarray
    layer_mask: np.ndarray
    value_mask: np.ndarray
    layout: TransformerLayerSequenceLayout
    normalization_stats: TransformerNormalizationStats


def layout_from_payload(payload: Any) -> TransformerLayerSequenceLayout:
    if not isinstance(payload, dict):
        raise ValueError("Transformer checkpoint is missing layout")
    return TransformerLayerSequenceLayout(
        depth_labels=tuple(str(x) for x in payload.get("depth_labels", [])),
        slot_names=tuple(str(x) for x in payload.get("slot_names", [])),
        feature_names=tuple(str(x) for x in payload.get("feature_names", [])),
        input_dim=int(payload["input_dim"]),
        max_layers=int(payload["max_layers"]),
        max_slots=int(payload["max_slots"]),
    )


def _resolve_bundle_metadata(
    bundle: SupervisedFeatureBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    if bundle.representation_kind != ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND:
        raise ValueError(
            "transformer requires architecture_independent_layer_sequence_aggregation input, "
            f"got {bundle.representation_kind!r}"
        )
    if bundle.group_mask is None or bundle.value_mask is None:
        raise ValueError("transformer requires group_mask and value_mask companions")

    values = np.asarray(bundle.values, dtype=np.float32)
    group_mask = np.asarray(bundle.group_mask, dtype=bool)
    value_mask = np.asarray(bundle.value_mask, dtype=bool)
    if values.ndim != 4:
        raise ValueError(f"transformer expects a 4D tensor, got shape={values.shape}")
    if group_mask.shape != values.shape[:2]:
        raise ValueError(f"group_mask shape {group_mask.shape} does not match tensor layer shape {values.shape[:2]}")
    if value_mask.shape != values.shape:
        raise ValueError(f"value_mask shape {value_mask.shape} does not match tensor shape {values.shape}")

    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    raw_depth_labels = metadata.get("depth_labels")
    if not isinstance(raw_depth_labels, list) or len(raw_depth_labels) != int(values.shape[1]):
        raise ValueError("transformer requires metadata.depth_labels aligned to the layer axis")
    raw_slot_names = metadata.get("slot_names")
    if not isinstance(raw_slot_names, list) or len(raw_slot_names) != int(values.shape[2]):
        raise ValueError("transformer requires metadata.slot_names aligned to the slot axis")
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


def _compute_normalization_stats(
    *,
    values: np.ndarray,
    value_mask: np.ndarray,
) -> TransformerNormalizationStats:
    values_np = np.asarray(values, dtype=np.float32)
    mask_np = np.asarray(value_mask, dtype=bool)
    feature_dim = int(values_np.shape[-1])
    counts = mask_np.sum(axis=(0, 1, 2)).astype(np.int64, copy=False)
    means = np.zeros(feature_dim, dtype=np.float32)
    std = np.ones(feature_dim, dtype=np.float32)

    valid = counts > 0
    if np.any(valid):
        sums = (values_np * mask_np.astype(np.float32)).sum(axis=(0, 1, 2), dtype=np.float64)
        means[valid] = (sums[valid] / counts[valid]).astype(np.float32)

        centered = np.where(mask_np, values_np - means.reshape(1, 1, 1, -1), 0.0)
        sq_sums = np.square(centered, dtype=np.float64).sum(axis=(0, 1, 2), dtype=np.float64)
        std[valid] = np.sqrt(np.maximum(sq_sums[valid] / counts[valid], 1e-6)).astype(np.float32)
    std = np.where(std > 1e-6, std, np.ones_like(std, dtype=np.float32))

    return TransformerNormalizationStats(feature_mean=means, feature_std=std)


def prepare_transformer_layer_sequence(
    bundle: SupervisedFeatureBundle,
    normalization_stats: TransformerNormalizationStats | None = None,
    *,
    normalize_values: bool = True,
    include_value_mask_channels: bool = True,
) -> TransformerFeatureTensors:
    values, group_mask, value_mask, depth_labels, slot_names, feature_names = _resolve_bundle_metadata(bundle)
    slot_mask = np.logical_and(group_mask[:, :, None], value_mask.any(axis=-1))
    layer_mask = slot_mask.any(axis=-1)
    if bool(np.any(~layer_mask.any(axis=1))):
        bad_rows = np.flatnonzero(~layer_mask.any(axis=1))[:5].tolist()
        raise ValueError(
            f"transformer requires every sample to contain at least one valid layer; invalid row examples={bad_rows}"
        )

    resolved_stats = normalization_stats
    if resolved_stats is None:
        if bool(normalize_values):
            resolved_stats = _compute_normalization_stats(values=values, value_mask=value_mask)
        else:
            resolved_stats = TransformerNormalizationStats(
                feature_mean=np.zeros(int(values.shape[3]), dtype=np.float32),
                feature_std=np.ones(int(values.shape[3]), dtype=np.float32),
            )

    normalized = (values - resolved_stats.feature_mean.reshape(1, 1, 1, -1)) / resolved_stats.feature_std.reshape(
        1, 1, 1, -1
    )
    normalized = np.where(value_mask, normalized, 0.0).astype(np.float32, copy=False)
    model_values = normalized
    model_feature_names = list(feature_names)
    if include_value_mask_channels:
        model_values = np.concatenate(
            [normalized, value_mask.astype(np.float32)],
            axis=-1,
        ).astype(np.float32, copy=False)
        model_feature_names.extend(f"observed::{feature_name}" for feature_name in feature_names)

    layout = TransformerLayerSequenceLayout(
        depth_labels=tuple(depth_labels),
        slot_names=tuple(slot_names),
        feature_names=tuple(model_feature_names),
        input_dim=int(model_values.shape[3]),
        max_layers=int(values.shape[1]),
        max_slots=int(values.shape[2]),
    )
    return TransformerFeatureTensors(
        values=model_values,
        slot_mask=slot_mask.astype(np.bool_, copy=False),
        layer_mask=layer_mask.astype(np.bool_, copy=False),
        value_mask=value_mask.astype(np.bool_, copy=False),
        layout=layout,
        normalization_stats=resolved_stats,
    )
