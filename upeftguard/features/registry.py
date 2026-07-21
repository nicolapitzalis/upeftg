from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spectral import SPECTRAL_EXTRACTOR_VERSION, extract_spectral_features, spectral_extractor_params
from ..utilities.core.manifest import ManifestItem
from ..utilities.core.serialization import json_ready


@dataclass
class FeatureBundle:
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    metadata: dict[str, Any]


_EXTRACTOR_VERSIONS = {
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
    if extractor_name != "spectral":
        raise ValueError(f"Unsupported stable extractor: {extractor_name!r}")
    return spectral_extractor_params(params)


def _compute_bundle(
    *,
    extractor_name: str,
    items: list[ManifestItem],
    params: dict[str, Any],
) -> tuple[FeatureBundle, list[str]]:
    if extractor_name == "spectral":
        dtype = np.float32 if params.get("dtype", "float32") == "float32" else np.float64
        features, labels, model_names, metadata = extract_spectral_features(
            items=items,
            spectral_features=(
                list(params.get("spectral_features")) if params.get("spectral_features") is not None else None
            ),
            spectral_qv_sum_mode=str(params.get("spectral_qv_sum_mode", "none")),
            spectral_moment_source=str(params.get("spectral_moment_source", "sv")),
            spectral_entrywise_delta_mode=str(params.get("spectral_entrywise_delta_mode", "auto")),
            spectral_attention_granularity=str(params.get("spectral_attention_granularity", "module")),
            sv_top_k=int(params.get("spectral_sv_top_k", 8)),
            block_size=int(params.get("block_size", 131072)),
            dtype=dtype,
        )
        return (
            FeatureBundle(features=features, labels=labels, model_names=model_names, metadata=metadata),
            _warning_messages_from_skipped_spectral_models(metadata),
        )

    raise ValueError(f"Unknown extractor '{extractor_name}'. Supported: {sorted(_EXTRACTOR_VERSIONS.keys())}")


def extract_features(
    *,
    extractor_name: str,
    items: list[ManifestItem],
    params: dict[str, Any],
) -> tuple[FeatureBundle, list[str]]:
    if extractor_name not in _EXTRACTOR_VERSIONS:
        raise ValueError(f"Unknown extractor '{extractor_name}'. Supported: {sorted(_EXTRACTOR_VERSIONS.keys())}")

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
    return bundle, warnings


def supported_extractors() -> list[str]:
    return sorted(_EXTRACTOR_VERSIONS.keys())
