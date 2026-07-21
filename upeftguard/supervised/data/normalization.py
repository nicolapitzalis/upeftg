from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import SupervisedFeatureBundle


INPUT_NORMALIZATION_NONE = "none"
INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD = "dataset_feature_standard"
SUPPORTED_INPUT_NORMALIZATIONS = (
    INPUT_NORMALIZATION_NONE,
    INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD,
)


def resolve_input_normalization(input_normalization: str | None) -> str:
    resolved = str(input_normalization or INPUT_NORMALIZATION_NONE)
    if resolved not in SUPPORTED_INPUT_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported input_normalization={resolved!r}; supported values={list(SUPPORTED_INPUT_NORMALIZATIONS)}"
        )
    return resolved


def slice_supervised_features(features: np.ndarray | SupervisedFeatureBundle, indices: np.ndarray) -> Any:
    resolved_indices = np.asarray(indices, dtype=np.int64)
    if isinstance(features, SupervisedFeatureBundle):
        return features.subset(resolved_indices)
    return np.asarray(features[resolved_indices], dtype=np.float32)


def dataset_group_names_for_indices(
    dataset_group_names: list[str],
    indices: np.ndarray,
) -> list[str]:
    resolved_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if resolved_indices.size == 0:
        return []
    if int(np.max(resolved_indices)) >= len(dataset_group_names):
        raise ValueError("Dataset-group names are not aligned with supervised feature rows")
    return [str(dataset_group_names[int(idx)]) for idx in resolved_indices.tolist()]


def normalize_values_by_dataset_feature(
    values: np.ndarray,
    dataset_group_names: list[str],
    *,
    value_mask: np.ndarray | None = None,
) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32)
    if values_np.shape[0] != len(dataset_group_names):
        raise ValueError(
            "Dataset-group name count does not match the feature slice row count: "
            f"{len(dataset_group_names)} != {values_np.shape[0]}"
        )
    normalized = np.asarray(values_np, dtype=np.float32).copy()
    if values_np.shape[0] == 0:
        return normalized

    mask_np = None if value_mask is None else np.asarray(value_mask, dtype=bool)
    if mask_np is not None and mask_np.shape != values_np.shape:
        raise ValueError(f"value_mask shape {mask_np.shape} does not match feature values shape {values_np.shape}")

    unique_dataset_names = sorted({str(name) for name in dataset_group_names})
    for dataset_name in unique_dataset_names:
        row_indices = np.asarray(
            [
                row_idx
                for row_idx, candidate_name in enumerate(dataset_group_names)
                if str(candidate_name) == dataset_name
            ],
            dtype=np.int64,
        )
        if row_indices.size == 0:
            continue

        group_values = values_np[row_indices]
        if mask_np is None:
            means = np.asarray(np.mean(group_values, axis=0, dtype=np.float64), dtype=np.float32)
            std = np.asarray(np.std(group_values, axis=0, dtype=np.float64), dtype=np.float32)
            std = np.where(std > 1e-6, std, np.ones_like(std, dtype=np.float32))
            normalized[row_indices] = ((group_values - means) / std).astype(np.float32, copy=False)
            continue

        group_mask = mask_np[row_indices]
        counts = np.asarray(group_mask.sum(axis=0), dtype=np.int64)
        valid = counts > 0
        means = np.zeros(values_np.shape[1:], dtype=np.float32)
        std = np.ones(values_np.shape[1:], dtype=np.float32)
        if bool(np.any(valid)):
            sums = np.asarray(
                (group_values * group_mask.astype(np.float32)).sum(axis=0, dtype=np.float64),
                dtype=np.float64,
            )
            means[valid] = (sums[valid] / counts[valid]).astype(np.float32)
            centered = np.where(group_mask, group_values - means, 0.0)
            sq_sums = np.asarray(
                np.square(centered, dtype=np.float64).sum(axis=0, dtype=np.float64),
                dtype=np.float64,
            )
            std[valid] = np.sqrt(np.maximum(sq_sums[valid] / counts[valid], 1e-6)).astype(np.float32)
        std = np.where(std > 1e-6, std, np.ones_like(std, dtype=np.float32))
        group_normalized = ((group_values - means) / std).astype(np.float32, copy=False)
        normalized[row_indices] = np.where(group_mask, group_normalized, 0.0).astype(
            np.float32,
            copy=False,
        )

    return normalized


def apply_input_normalization_to_slice(
    features: np.ndarray | SupervisedFeatureBundle,
    *,
    dataset_group_names: list[str],
    input_normalization: str,
) -> np.ndarray | SupervisedFeatureBundle:
    resolved = resolve_input_normalization(input_normalization)
    if resolved == INPUT_NORMALIZATION_NONE:
        return features
    if resolved != INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD:
        raise ValueError(f"Unsupported input normalization mode: {resolved!r}")

    if isinstance(features, SupervisedFeatureBundle):
        normalized_values = normalize_values_by_dataset_feature(
            features.values,
            dataset_group_names,
            value_mask=features.value_mask,
        )
        metadata = dict(features.metadata)
        metadata["input_normalization"] = {
            "mode": INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD,
            "grouping": "inferred_dataset",
        }
        return SupervisedFeatureBundle(
            values=normalized_values,
            representation_kind=str(features.representation_kind),
            metadata=metadata,
            group_mask=features.group_mask,
            value_mask=features.value_mask,
            group_names=features.group_names,
        )

    return normalize_values_by_dataset_feature(
        np.asarray(features, dtype=np.float32),
        dataset_group_names,
    )


def slice_supervised_features_for_input(
    features: np.ndarray | SupervisedFeatureBundle,
    indices: np.ndarray,
    *,
    dataset_group_names: list[str],
    input_normalization: str,
) -> np.ndarray | SupervisedFeatureBundle:
    resolved_indices = np.asarray(indices, dtype=np.int64)
    feature_slice = slice_supervised_features(features, resolved_indices)
    slice_dataset_group_names = dataset_group_names_for_indices(
        dataset_group_names,
        resolved_indices,
    )
    return apply_input_normalization_to_slice(
        feature_slice,
        dataset_group_names=slice_dataset_group_names,
        input_normalization=input_normalization,
    )
