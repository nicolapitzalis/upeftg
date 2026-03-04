from __future__ import annotations

from typing import Any

import numpy as np
from safetensors import safe_open

from .delta import (
    DELTA_SCHEMA_VERSION,
    block_delta_singular_values,
    block_spectral_scalars,
    build_schema_metadata,
    check_consistency_reader,
    load_delta_block_schema,
    shorten_block_name,
    top_k_singular_values,
)
from .norms import block_moments_from_factors
from ..utilities.manifest import ManifestItem


SPECTRAL_EXTRACTOR_VERSION = "1.0.0"

DEFAULT_SPECTRAL_FEATURES = [
    "frobenius",
    "energy",
    "kurtosis",
    "l1_norm",
    "linf_norm",
    "sv_topk",
]

SUPPORTED_SPECTRAL_FEATURES = {
    "frobenius",
    "energy",
    "kurtosis",
    "l1_norm",
    "l2_norm",
    "linf_norm",
    "mean_abs",
    "sv_topk",
    "stable_rank",
    "spectral_entropy",
    "effective_rank",
}

_MOMENT_FEATURES = {"kurtosis", "l1_norm", "linf_norm", "mean_abs"}
_SPECTRAL_SCALAR_FEATURES = {
    "frobenius",
    "energy",
    "l2_norm",
    "stable_rank",
    "spectral_entropy",
    "effective_rank",
}


def resolve_spectral_features(requested: list[str] | None) -> list[str]:
    raw = DEFAULT_SPECTRAL_FEATURES if not requested else [str(x).strip() for x in requested if str(x).strip()]
    if not raw:
        raw = list(DEFAULT_SPECTRAL_FEATURES)

    unknown = sorted(set(raw) - SUPPORTED_SPECTRAL_FEATURES)
    if unknown:
        raise ValueError(
            f"Unknown spectral features requested: {unknown}. "
            f"Supported: {sorted(SUPPORTED_SPECTRAL_FEATURES)}"
        )

    dedup: list[str] = []
    seen: set[str] = set()
    for feat in raw:
        if feat in seen:
            continue
        seen.add(feat)
        dedup.append(feat)
    return dedup


def build_spectral_feature_names(
    *,
    block_names: list[str],
    selected_features: list[str],
    sv_top_k: int,
) -> list[str]:
    names: list[str] = []
    for block in block_names:
        prefix = shorten_block_name(block)
        for feat in selected_features:
            if feat == "sv_topk":
                for i in range(sv_top_k):
                    names.append(f"{prefix}.sv_{i + 1}")
            else:
                names.append(f"{prefix}.{feat}")
    return names


def _needs_moments(selected_features: list[str]) -> bool:
    return any(feature in _MOMENT_FEATURES for feature in selected_features)


def _needs_spectral_scalars(selected_features: list[str]) -> bool:
    return any(feature in _SPECTRAL_SCALAR_FEATURES for feature in selected_features)


def _needs_sv_topk(selected_features: list[str]) -> bool:
    return "sv_topk" in selected_features


def _append_block_features(
    *,
    row: list[float],
    selected_features: list[str],
    singular_values: np.ndarray,
    sv_top_k: int,
    need_spectral_scalars: bool,
    need_sv_topk: bool,
    moments: dict[str, float] | None,
) -> None:
    frobenius = energy = stable_rank = spectral_entropy = effective_rank = None
    l2_norm = None
    if need_spectral_scalars:
        frobenius, energy, stable_rank, spectral_entropy, effective_rank = block_spectral_scalars(singular_values)
        l2_norm = float(np.sqrt(max(0.0, float(energy))))

    top_k: np.ndarray | None = None
    if need_sv_topk:
        top_k = top_k_singular_values(singular_values, top_k=sv_top_k)

    for feature in selected_features:
        if feature == "sv_topk":
            if top_k is None:
                raise RuntimeError("Top-k singular values were requested but not computed")
            row.extend(float(x) for x in top_k.tolist())
        elif feature == "frobenius":
            if frobenius is None:
                raise RuntimeError("Spectral scalars were required for 'frobenius' but were not computed")
            row.append(frobenius)
        elif feature == "energy":
            if energy is None:
                raise RuntimeError("Spectral scalars were required for 'energy' but were not computed")
            row.append(energy)
        elif feature == "l2_norm":
            if l2_norm is None:
                raise RuntimeError("Spectral scalars were required for 'l2_norm' but were not computed")
            row.append(l2_norm)
        elif feature == "stable_rank":
            if stable_rank is None:
                raise RuntimeError("Spectral scalars were required for 'stable_rank' but were not computed")
            row.append(stable_rank)
        elif feature == "spectral_entropy":
            if spectral_entropy is None:
                raise RuntimeError("Spectral scalars were required for 'spectral_entropy' but were not computed")
            row.append(spectral_entropy)
        elif feature == "effective_rank":
            if effective_rank is None:
                raise RuntimeError("Spectral scalars were required for 'effective_rank' but were not computed")
            row.append(effective_rank)
        else:
            if moments is None:
                raise RuntimeError(f"Moments are required for feature '{feature}' but were not computed")
            row.append(float(moments[feature]))


def extract_spectral_features(
    *,
    items: list[ManifestItem],
    spectral_features: list[str] | None,
    sv_top_k: int,
    block_size: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray | None, list[str], dict[str, Any]]:
    if not items:
        raise ValueError("No adapters provided for spectral extraction")
    if sv_top_k <= 0:
        raise ValueError("sv_top_k must be positive")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    selected_features = resolve_spectral_features(spectral_features)
    schema = load_delta_block_schema(items[0].adapter_path)

    expected_pairs = [tuple(x) for x in schema.pairs]
    expected_a_shapes = [tuple(x) for x in schema.a_shapes]
    expected_b_shapes = [tuple(x) for x in schema.b_shapes]

    rows: list[list[float]] = []
    model_names: list[str] = []
    labels_list: list[int | None] = []
    need_moments = _needs_moments(selected_features)
    need_spectral_scalars = _needs_spectral_scalars(selected_features)
    need_top_k = _needs_sv_topk(selected_features)

    for item in items:
        row: list[float] = []
        with safe_open(item.adapter_path, framework="numpy") as reader:
            check_consistency_reader(
                reader=reader,
                adapter_path=item.adapter_path,
                expected_pairs=expected_pairs,
                expected_a_shapes=expected_a_shapes,
                expected_b_shapes=expected_b_shapes,
            )
            for a_key, b_key in schema.pairs:
                a = np.asarray(reader.get_tensor(a_key), dtype=dtype)
                b = np.asarray(reader.get_tensor(b_key), dtype=dtype)
                singular_values = block_delta_singular_values(a=a, b=b)

                moments: dict[str, float] | None = None
                if need_moments:
                    summary = block_moments_from_factors(
                        a=a,
                        b=b,
                        block_size=block_size,
                        dtype=dtype,
                    )
                    moments = {
                        "kurtosis": float(summary.kurtosis),
                        "l1_norm": float(summary.l1_norm),
                        "linf_norm": float(summary.linf_norm),
                        "mean_abs": float(summary.mean_abs),
                    }

                _append_block_features(
                    row=row,
                    selected_features=selected_features,
                    singular_values=singular_values,
                    sv_top_k=sv_top_k,
                    need_spectral_scalars=need_spectral_scalars,
                    need_sv_topk=need_top_k,
                    moments=moments,
                )

        rows.append(row)
        model_names.append(item.model_name)
        labels_list.append(item.label)

    features = np.asarray(rows, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.int32) if all(label is not None for label in labels_list) else None

    feature_names = build_spectral_feature_names(
        block_names=list(schema.block_names),
        selected_features=selected_features,
        sv_top_k=sv_top_k,
    )
    if features.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Spectral feature dimension mismatch: matrix has {features.shape[1]} columns, "
            f"but generated {len(feature_names)} feature names"
        )

    metadata: dict[str, Any] = {
        "extractor": "spectral",
        "extractor_version": SPECTRAL_EXTRACTOR_VERSION,
        "delta_schema_version": DELTA_SCHEMA_VERSION,
        "n_models": int(len(items)),
        "resolved_features": selected_features,
        "sv_top_k": int(sv_top_k),
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names,
        **build_schema_metadata(schema),
    }
    return features, labels, model_names, metadata
