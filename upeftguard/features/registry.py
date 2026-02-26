from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .delta import DELTA_EXTRACTOR_VERSION, extract_delta_feature_matrices
from .norms import NORMS_EXTRACTOR_VERSION, extract_norm_features
from .svd import SVD_EXTRACTOR_VERSION, extract_svd_embeddings
from ..utilities.hashing import compute_dataset_signature, compute_feature_cache_key
from ..utilities.manifest import ManifestItem
from ..utilities.serialization import json_ready


@dataclass
class FeatureBundle:
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    metadata: dict[str, Any]


_EXTRACTOR_VERSIONS = {
    "svd": SVD_EXTRACTOR_VERSION,
    "delta_singular_values": DELTA_EXTRACTOR_VERSION,
    "delta_frobenius": DELTA_EXTRACTOR_VERSION,
    "norms": NORMS_EXTRACTOR_VERSION,
}


def _dataset_signature(items: list[ManifestItem]) -> str:
    model_names = [item.model_name for item in items]
    extra = {
        "adapter_paths": [str(item.adapter_path.resolve()) for item in items],
    }
    return compute_dataset_signature(model_names=model_names, extra=extra)


def _write_bundle(cache_dir: Path, bundle: FeatureBundle) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "features.npy", bundle.features)
    if bundle.labels is not None:
        np.save(cache_dir / "labels.npy", bundle.labels)
    with open(cache_dir / "model_names.json", "w", encoding="utf-8") as f:
        json.dump(bundle.model_names, f, indent=2)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(bundle.metadata), f, indent=2)


def _load_bundle(cache_dir: Path) -> FeatureBundle:
    features = np.load(cache_dir / "features.npy")
    labels_path = cache_dir / "labels.npy"
    labels = np.load(labels_path) if labels_path.exists() else None
    with open(cache_dir / "model_names.json", "r", encoding="utf-8") as f:
        model_names = json.load(f)
    with open(cache_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return FeatureBundle(features=features, labels=labels, model_names=model_names, metadata=metadata)


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

    if extractor_name in {"delta_singular_values", "delta_frobenius"}:
        dtype = np.float32 if params.get("dtype", "float32") == "float32" else np.float64
        sv, fro, labels, model_names, metadata = extract_delta_feature_matrices(
            items=items,
            top_k_singular_values=int(params.get("top_k_singular_values", 8)),
            dtype=dtype,
        )
        if extractor_name == "delta_singular_values":
            features = sv
            metadata = {**metadata, "selected_output": "delta_singular_values"}
        else:
            features = fro
            metadata = {**metadata, "selected_output": "delta_frobenius"}
        return FeatureBundle(features=features, labels=labels, model_names=model_names, metadata=metadata), []

    if extractor_name == "norms":
        dtype = np.float32 if params.get("dtype", "float32") == "float32" else np.float64
        features, labels, model_names, metadata = extract_norm_features(
            items=items,
            block_size=int(params.get("block_size", 131072)),
            dtype=dtype,
        )
        return FeatureBundle(features=features, labels=labels, model_names=model_names, metadata=metadata), []

    raise ValueError(
        f"Unknown extractor '{extractor_name}'. Supported: {sorted(_EXTRACTOR_VERSIONS.keys())}"
    )


def extract_with_cache(
    *,
    extractor_name: str,
    items: list[ManifestItem],
    params: dict[str, Any],
    cache_root: Path,
    run_features_dir: Path,
    force_recompute: bool = False,
) -> tuple[FeatureBundle, dict[str, Any], list[str]]:
    if extractor_name not in _EXTRACTOR_VERSIONS:
        raise ValueError(
            f"Unknown extractor '{extractor_name}'. Supported: {sorted(_EXTRACTOR_VERSIONS.keys())}"
        )

    dataset_signature = _dataset_signature(items)
    dtype = str(params.get("dtype", "float32"))
    cache_key = compute_feature_cache_key(
        dataset_signature=dataset_signature,
        extractor_name=extractor_name,
        extractor_params=json_ready(params),
        extractor_version=_EXTRACTOR_VERSIONS[extractor_name],
        dtype=dtype,
    )

    cache_dir = cache_root / cache_key
    cache_hit = cache_dir.exists() and (cache_dir / "features.npy").exists()

    warnings: list[str] = []
    if cache_hit and not force_recompute:
        bundle = _load_bundle(cache_dir)
    else:
        bundle, warnings = _compute_bundle(
            extractor_name=extractor_name,
            items=items,
            params=params,
        )
        bundle.metadata = {
            **bundle.metadata,
            "dataset_signature": dataset_signature,
            "extractor_name": extractor_name,
            "extractor_version": _EXTRACTOR_VERSIONS[extractor_name],
            "extractor_params": json_ready(params),
        }
        _write_bundle(cache_dir, bundle)

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
    with open(run_metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            json_ready(
                {
                    **bundle.metadata,
                    "cache_key": cache_key,
                    "cache_hit": bool(cache_hit and not force_recompute),
                    "cache_dir": str(cache_dir),
                }
            ),
            f,
            indent=2,
        )

    artifact_info = {
        "cache_key": cache_key,
        "cache_hit": bool(cache_hit and not force_recompute),
        "cache_dir": str(cache_dir),
        "feature_path": str(run_feature_path),
        "labels_path": str(run_labels_path) if run_labels_path is not None else None,
        "model_names_path": str(run_names_path),
        "metadata_path": str(run_metadata_path),
        "dataset_signature": dataset_signature,
    }
    return bundle, artifact_info, warnings


def supported_extractors() -> list[str]:
    return sorted(_EXTRACTOR_VERSIONS.keys())
