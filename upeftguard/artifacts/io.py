"""Loading and validation for dense feature-table artifacts."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from .metadata.spectral import load_spectral_metadata
from .tables import FeatureTable, unique_index_by_name


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_spectral_metadata(path)
    return dict(payload) if isinstance(payload, dict) else {}


def resolve_feature_names(
    *,
    metadata: dict[str, Any],
    feature_dim: int,
    context: str,
) -> tuple[list[str], bool]:
    raw = metadata.get("feature_names")
    if isinstance(raw, list):
        names = [str(value) for value in raw]
        if len(names) != int(feature_dim):
            raise ValueError(
                f"feature_names length mismatch in {context}: "
                f"metadata has {len(names)}, matrix has {feature_dim} columns"
            )
        unique_index_by_name(names, context=context, entity="feature names")
        return names, False

    inferred = [f"feature_{index:05d}" for index in range(int(feature_dim))]
    return inferred, True


def load_feature_table(
    *,
    source: str,
    feature_path: Path,
    model_names_path: Path,
    labels_path: Path,
    metadata_path: Path,
    context: str,
) -> FeatureTable:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing spectral feature file for {context}: {feature_path}")
    if not model_names_path.exists():
        raise FileNotFoundError(f"Missing spectral model names file for {context}: {model_names_path}")

    features = np.asarray(np.load(feature_path), dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Spectral features must be 2D in {context}: shape={features.shape}")

    with open(model_names_path, "r", encoding="utf-8") as file:
        model_names = [str(value) for value in json.load(file)]
    if len(model_names) != int(features.shape[0]):
        raise ValueError(
            f"Row mismatch in {context}: features rows={features.shape[0]} but model names={len(model_names)}"
        )
    unique_index_by_name(model_names, context=context, entity="model names")

    labels = None
    if labels_path.exists():
        labels = np.asarray(np.load(labels_path), dtype=np.int32)
        if int(labels.shape[0]) != int(features.shape[0]):
            raise ValueError(
                f"Label mismatch in {context}: labels rows={labels.shape[0]} but features rows={features.shape[0]}"
            )

    metadata = load_metadata(metadata_path)
    feature_names, feature_names_inferred = resolve_feature_names(
        metadata=metadata,
        feature_dim=int(features.shape[1]),
        context=context,
    )
    return FeatureTable(
        source=source,
        features=features,
        labels=labels,
        model_names=model_names,
        feature_names=feature_names,
        feature_names_inferred=feature_names_inferred,
        metadata=metadata,
    )
