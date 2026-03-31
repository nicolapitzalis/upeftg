from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .spectral import SPECTRAL_EXTRACTOR_VERSION, extract_spectral_features, spectral_extractor_params
from .svd import SVD_EXTRACTOR_VERSION, extract_svd_embeddings
from ..utilities.artifacts.spectral_metadata import (
    dataset_layouts_from_source,
    write_spectral_metadata,
)
from ..utilities.core.manifest import ManifestItem
from ..utilities.core.serialization import json_ready


@dataclass
class FeatureBundle:
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    metadata: dict[str, Any]


_EXTRACTOR_VERSIONS = {
    "svd": SVD_EXTRACTOR_VERSION,
    "spectral": SPECTRAL_EXTRACTOR_VERSION,
}


def _warning_messages_from_skipped_spectral_models(metadata: dict[str, Any]) -> list[str]:
    raw_skipped = metadata.get("skipped_models")
    if not isinstance(raw_skipped, list) or not raw_skipped:
        return []

    warnings = [f"Skipped {len(raw_skipped)} spectral adapter(s) due to read/consistency errors"]
    for entry in raw_skipped:
        if not isinstance(entry, dict):
            continue
        model_name = str(entry.get("model_name") or "unknown")
        exc_type = str(entry.get("exception_type") or "Error")
        exc_message = str(entry.get("exception_message") or "").strip()
        if exc_message:
            warnings.append(f"Skipped spectral adapter '{model_name}': {exc_type}: {exc_message}")
        else:
            warnings.append(f"Skipped spectral adapter '{model_name}': {exc_type}")
    return warnings


def _extractor_params_for_metadata(
    *,
    extractor_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    if extractor_name == "spectral":
        return spectral_extractor_params(params)

    return {
        "component_grid": list(params.get("component_grid", [20, 25, 30])),
        "n_components": params.get("n_components"),
        "block_size": int(params.get("block_size", 131072)),
        "dtype": str(params.get("dtype", "float32")),
        "acceptance_spearman_threshold": float(params.get("acceptance_spearman_threshold", 0.99)),
        "acceptance_variance_threshold": float(params.get("acceptance_variance_threshold", 0.95)),
        "run_offline_label_diagnostics": bool(params.get("run_offline_label_diagnostics", False)),
    }


def _compute_bundle(
    *,
    extractor_name: str,
    items: list[ManifestItem],
    params: dict[str, Any],
) -> tuple[FeatureBundle, list[str]]:
    if extractor_name == "svd":
        dtype = np.float32 if params.get("dtype", "float32") == "float32" else np.float64
        features, labels, model_names, metadata, warnings = extract_svd_embeddings(
            items=items,
            n_components=params.get("n_components"),
            component_grid=list(params.get("component_grid", [20, 25, 30])),
            block_size=int(params.get("block_size", 131072)),
            dtype=dtype,
            acceptance_spearman_threshold=float(params.get("acceptance_spearman_threshold", 0.99)),
            acceptance_variance_threshold=float(params.get("acceptance_variance_threshold", 0.95)),
            run_offline_label_diagnostics=bool(params.get("run_offline_label_diagnostics", False)),
        )
        return FeatureBundle(features=features, labels=labels, model_names=model_names, metadata=metadata), warnings

    if extractor_name == "spectral":
        dtype = np.float32 if params.get("dtype", "float32") == "float32" else np.float64
        features, labels, model_names, metadata = extract_spectral_features(
            items=items,
            spectral_features=(
                list(params.get("spectral_features"))
                if params.get("spectral_features") is not None
                else None
            ),
            spectral_qv_sum_mode=str(params.get("spectral_qv_sum_mode", "none")),
            spectral_moment_source=str(params.get("spectral_moment_source", "sv")),
            spectral_entrywise_delta_mode=str(params.get("spectral_entrywise_delta_mode", "auto")),
            sv_top_k=int(params.get("spectral_sv_top_k", 8)),
            block_size=int(params.get("block_size", 131072)),
            dtype=dtype,
        )
        return (
            FeatureBundle(features=features, labels=labels, model_names=model_names, metadata=metadata),
            _warning_messages_from_skipped_spectral_models(metadata),
        )

    raise ValueError(
        f"Unknown extractor '{extractor_name}'. Supported: {sorted(_EXTRACTOR_VERSIONS.keys())}"
    )


def extract_features(
    *,
    extractor_name: str,
    items: list[ManifestItem],
    params: dict[str, Any],
    run_features_dir: Path,
) -> tuple[FeatureBundle, dict[str, Any], list[str]]:
    if extractor_name not in _EXTRACTOR_VERSIONS:
        raise ValueError(
            f"Unknown extractor '{extractor_name}'. Supported: {sorted(_EXTRACTOR_VERSIONS.keys())}"
        )

    bundle, warnings = _compute_bundle(
        extractor_name=extractor_name,
        items=items,
        params=params,
    )
    bundle.metadata = {
        **bundle.metadata,
        "extractor_name": extractor_name,
        "extractor_version": _EXTRACTOR_VERSIONS[extractor_name],
        "extractor_params": json_ready(
            _extractor_params_for_metadata(
                extractor_name=extractor_name,
                params=params,
            )
        ),
    }
    kept_model_names = set(bundle.model_names)
    kept_items = [item for item in items if item.model_name in kept_model_names]

    run_features_dir.mkdir(parents=True, exist_ok=True)
    run_feature_path = run_features_dir / f"{extractor_name}_features.npy"
    np.save(run_feature_path, bundle.features)

    run_labels_path: Path | None = None
    if bundle.labels is not None:
        run_labels_path = run_features_dir / f"{extractor_name}_labels.npy"
        np.save(run_labels_path, bundle.labels)

    run_names_path = run_features_dir / f"{extractor_name}_model_names.json"
    with open(run_names_path, "w", encoding="utf-8") as f:
        json.dump(bundle.model_names, f, indent=2)

    run_metadata_path = run_features_dir / f"{extractor_name}_metadata.json"
    if extractor_name == "spectral":
        dataset_counts: dict[str, int] = {}
        for item in kept_items:
            dataset_name = str(item.model_dir.parent.name or "unknown")
            dataset_counts[dataset_name] = int(dataset_counts.get(dataset_name, 0)) + 1
        dataset_reference_payload = {
            "dataset_groups": [
                {
                    "dataset_name": dataset_name,
                    "sample_count": int(sample_count),
                }
                for dataset_name, sample_count in sorted(dataset_counts.items())
            ]
        }
        dataset_layouts = dataset_layouts_from_source(
            metadata=bundle.metadata,
            dataset_reference_payload=dataset_reference_payload,
        )
        write_spectral_metadata(
            run_metadata_path,
            internal_metadata=bundle.metadata,
            dataset_layouts=dataset_layouts,
        )
    else:
        with open(run_metadata_path, "w", encoding="utf-8") as f:
            json.dump(json_ready(bundle.metadata), f, indent=2)

    artifact_info = {
        "feature_path": str(run_feature_path),
        "labels_path": str(run_labels_path) if run_labels_path is not None else None,
        "model_names_path": str(run_names_path),
        "metadata_path": str(run_metadata_path),
    }
    return bundle, artifact_info, warnings


def supported_extractors() -> list[str]:
    return sorted(_EXTRACTOR_VERSIONS.keys())
