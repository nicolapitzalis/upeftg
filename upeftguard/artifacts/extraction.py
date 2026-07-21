"""Persistence for in-memory feature-extraction results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..features.registry import FeatureBundle
from ..utilities.core.manifest import ManifestItem
from ..utilities.core.serialization import json_ready
from .metadata.spectral import dataset_layouts_from_source, write_spectral_metadata


def write_extracted_feature_bundle(
    *,
    bundle: FeatureBundle,
    items: list[ManifestItem],
    extractor_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{extractor_name}_features.npy"
    labels_path = output_dir / f"{extractor_name}_labels.npy"
    model_names_path = output_dir / f"{extractor_name}_model_names.json"
    metadata_path = output_dir / f"{extractor_name}_metadata.json"

    np.save(feature_path, bundle.features)
    if bundle.labels is not None:
        np.save(labels_path, bundle.labels)
    else:
        labels_path.unlink(missing_ok=True)
    with open(model_names_path, "w", encoding="utf-8") as file:
        json.dump(bundle.model_names, file, indent=2)

    kept_names = set(bundle.model_names)
    kept_items = [item for item in items if item.model_name in kept_names]
    if extractor_name == "spectral":
        dataset_counts: dict[str, int] = {}
        for item in kept_items:
            dataset_name = str(item.model_dir.parent.name or "unknown")
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        reference = {
            "dataset_groups": [
                {"dataset_name": name, "sample_count": count} for name, count in sorted(dataset_counts.items())
            ]
        }
        write_spectral_metadata(
            metadata_path,
            internal_metadata=bundle.metadata,
            dataset_layouts=dataset_layouts_from_source(
                metadata=bundle.metadata,
                dataset_reference_payload=reference,
            ),
        )
    else:
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(json_ready(bundle.metadata), file, indent=2)

    return {
        "feature_path": str(feature_path),
        "labels_path": str(labels_path) if bundle.labels is not None else None,
        "model_names_path": str(model_names_path),
        "metadata_path": str(metadata_path),
    }
