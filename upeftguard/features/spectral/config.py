"""Spectral feature defaults and parameter validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from safetensors import SafetensorError

from ..norms import (
    DEFAULT_ENTRYWISE_DELTA_MODE,
    SUPPORTED_ENTRYWISE_DELTA_MODES,
    resolve_entrywise_delta_mode,
)

SPECTRAL_EXTRACTOR_VERSION = "2.6.0"

DEFAULT_SPECTRAL_FEATURES = [
    "energy",
    "kurtosis",
    "l1_norm",
    "l2_norm",
    "linf_norm",
    "mean_abs",
    "concentration_of_energy",
    "sv_topk",
    "stable_rank",
    "spectral_entropy",
    "effective_rank",
]

_RANK_NORMALIZED_FEATURE_REPLACEMENTS = {
    "energy": "energy_per_rank",
    "l1_norm": "sv_l1_norm_per_rank",
    "l2_norm": "l2_norm_per_sqrt_rank",
    "stable_rank": "stable_rank_frac",
    "spectral_entropy": "normalized_spectral_entropy",
    "effective_rank": "effective_rank_frac",
}

_DERIVED_FEATURE_GROUP_SOURCES = {
    "energy_per_rank": "energy",
    "l1_norm_per_rank": "l1_norm",
    "sv_l1_norm_per_rank": "l1_norm",
    "l2_norm_per_sqrt_rank": "l2_norm",
    "mean_abs_per_rank": "mean_abs",
    "stable_rank_frac": "stable_rank",
    "normalized_spectral_entropy": "spectral_entropy",
    "effective_rank_frac": "effective_rank",
}

DEFAULT_SPECTRAL_QV_SUM_MODE = "none"
DEFAULT_SPECTRAL_MOMENT_SOURCE = "sv"
DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE = DEFAULT_ENTRYWISE_DELTA_MODE
DEFAULT_SPECTRAL_ATTENTION_GRANULARITY = "module"
SUPPORTED_SPECTRAL_QV_SUM_MODES = {"none", "append", "only"}
SUPPORTED_SPECTRAL_MOMENT_SOURCES = {"entrywise", "sv", "both"}
SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES = tuple(SUPPORTED_ENTRYWISE_DELTA_MODES)
SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES = ("module", "head")

SUPPORTED_SPECTRAL_FEATURES = {
    "energy",
    "energy_per_rank",
    "block_rank",
    "kurtosis",
    "l1_norm",
    "l1_norm_per_rank",
    "sv_l1_norm_per_rank",
    "l2_norm",
    "l2_norm_per_sqrt_rank",
    "linf_norm",
    "mean_abs",
    "mean_abs_per_rank",
    "concentration_of_energy",
    "sv_topk",
    "stable_rank",
    "stable_rank_frac",
    "spectral_entropy",
    "normalized_spectral_entropy",
    "effective_rank",
    "effective_rank_frac",
}

_MOMENT_FEATURES = (
    "kurtosis",
    "l1_norm",
    "l1_norm_per_rank",
    "linf_norm",
    "mean_abs",
    "mean_abs_per_rank",
)
_MOMENT_FEATURE_SET = set(_MOMENT_FEATURES)
_SV_MOMENT_FEATURES = {
    "sv_kurtosis",
    "sv_l1_norm",
    "sv_l1_norm_per_rank",
    "sv_linf_norm",
    "sv_mean_abs",
    "sv_mean_abs_per_rank",
}
_SV_MOMENT_BY_ENTRYWISE = {
    "kurtosis": "sv_kurtosis",
    "l1_norm": "sv_l1_norm",
    "l1_norm_per_rank": "sv_l1_norm_per_rank",
    "linf_norm": "sv_linf_norm",
    "mean_abs": "sv_mean_abs",
    "mean_abs_per_rank": "sv_mean_abs_per_rank",
}
_SPECTRAL_SCALAR_FEATURES = {
    "block_rank",
    "energy",
    "energy_per_rank",
    "l2_norm",
    "l2_norm_per_sqrt_rank",
    "concentration_of_energy",
    "stable_rank",
    "stable_rank_frac",
    "spectral_entropy",
    "normalized_spectral_entropy",
    "effective_rank",
    "effective_rank_frac",
}
_SPECTRAL_METADATA_DROP_KEYS = {
    "a_shapes",
    "b_shapes",
    "block_names_raw",
    "base_block_names_raw",
    "qv_sum_block_names_raw",
    "incoming_metadata",
    "merge_stats",
    "merged_with_existing_output",
    "merge_existing_output_dir",
}
_QV_ROLE_ALIASES = {
    "q_proj": "q",
    "v_proj": "v",
    "q": "q",
    "v": "v",
    "query": "q",
    "value": "v",
}

_FEATURE_GROUP_BY_SUFFIX = {
    "energy": "energy",
    "energy_per_rank": "energy_per_rank",
    "block_rank": "block_rank",
    "kurtosis": "kurtosis",
    "sv_kurtosis": "kurtosis",
    "l1_norm": "l1_norm",
    "sv_l1_norm": "l1_norm",
    "l1_norm_per_rank": "l1_norm_per_rank",
    "sv_l1_norm_per_rank": "sv_l1_norm_per_rank",
    "l2_norm": "l2_norm",
    "l2_norm_per_sqrt_rank": "l2_norm_per_sqrt_rank",
    "linf_norm": "linf_norm",
    "sv_linf_norm": "linf_norm",
    "mean_abs": "mean_abs",
    "sv_mean_abs": "mean_abs",
    "mean_abs_per_rank": "mean_abs_per_rank",
    "sv_mean_abs_per_rank": "mean_abs_per_rank",
    "concentration_of_energy": "concentration_of_energy",
    "stable_rank": "stable_rank",
    "stable_rank_frac": "stable_rank_frac",
    "spectral_entropy": "spectral_entropy",
    "normalized_spectral_entropy": "normalized_spectral_entropy",
    "effective_rank": "effective_rank",
    "effective_rank_frac": "effective_rank_frac",
}

_SKIPPABLE_SPECTRAL_ITEM_EXCEPTIONS = (OSError, SafetensorError, ValueError)


def resolve_spectral_features(requested: list[str] | None) -> list[str]:
    raw = DEFAULT_SPECTRAL_FEATURES if not requested else [str(x).strip() for x in requested if str(x).strip()]
    if not raw:
        raw = list(DEFAULT_SPECTRAL_FEATURES)

    unknown = sorted(set(raw) - SUPPORTED_SPECTRAL_FEATURES)
    if unknown:
        raise ValueError(
            f"Unknown spectral features requested: {unknown}. Supported: {sorted(SUPPORTED_SPECTRAL_FEATURES)}"
        )

    dedup: list[str] = []
    seen: set[str] = set()
    for feat in raw:
        if feat in seen:
            continue
        seen.add(feat)
        dedup.append(feat)
    return dedup


def resolve_spectral_qv_sum_mode(mode: str | None) -> str:
    resolved = DEFAULT_SPECTRAL_QV_SUM_MODE if mode is None else str(mode).strip().lower()
    if resolved not in SUPPORTED_SPECTRAL_QV_SUM_MODES:
        raise ValueError(f"Unknown spectral_qv_sum_mode '{mode}'. Supported: {sorted(SUPPORTED_SPECTRAL_QV_SUM_MODES)}")
    return resolved


def resolve_spectral_attention_granularity(mode: str | None) -> str:
    resolved = DEFAULT_SPECTRAL_ATTENTION_GRANULARITY if mode is None else str(mode).strip().lower()
    if resolved not in SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES:
        raise ValueError(
            f"Unknown spectral_attention_granularity '{mode}'. "
            f"Supported: {list(SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES)}"
        )
    return resolved


def resolve_spectral_moment_source(mode: str | None) -> str:
    resolved = DEFAULT_SPECTRAL_MOMENT_SOURCE if mode is None else str(mode).strip().lower()
    if resolved not in SUPPORTED_SPECTRAL_MOMENT_SOURCES:
        raise ValueError(
            f"Unknown spectral_moment_source '{mode}'. Supported: {sorted(SUPPORTED_SPECTRAL_MOMENT_SOURCES)}"
        )
    return resolved


def resolve_spectral_entrywise_delta_mode(mode: str | None) -> str:
    return resolve_entrywise_delta_mode(mode)


def expand_spectral_feature_names(
    *,
    selected_features: list[str],
    spectral_moment_source: str,
) -> list[str]:
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    emitted: list[str] = []
    seen: set[str] = set()

    def append_emitted(feature_name: str) -> None:
        if feature_name in seen:
            return
        seen.add(feature_name)
        emitted.append(feature_name)

    for feature in selected_features:
        if feature not in _MOMENT_FEATURE_SET:
            append_emitted(feature)
            continue
        if resolved_moment_source in {"entrywise", "both"}:
            append_emitted(feature)
        if resolved_moment_source in {"sv", "both"}:
            append_emitted(_SV_MOMENT_BY_ENTRYWISE[feature])
    return emitted


def spectral_extractor_params(params: Mapping[str, Any]) -> dict[str, Any]:
    raw_features = params.get("spectral_features")
    if isinstance(raw_features, list):
        spectral_features = [str(x) for x in raw_features]
    elif raw_features is None:
        spectral_features = list(DEFAULT_SPECTRAL_FEATURES)
    else:
        spectral_features = [str(raw_features)]

    raw_qv_sum_mode = params.get("spectral_qv_sum_mode", DEFAULT_SPECTRAL_QV_SUM_MODE)
    try:
        spectral_qv_sum_mode = resolve_spectral_qv_sum_mode(None if raw_qv_sum_mode is None else str(raw_qv_sum_mode))
    except ValueError:
        spectral_qv_sum_mode = str(raw_qv_sum_mode)

    raw_moment_source = params.get("spectral_moment_source", DEFAULT_SPECTRAL_MOMENT_SOURCE)
    try:
        spectral_moment_source = resolve_spectral_moment_source(
            None if raw_moment_source is None else str(raw_moment_source)
        )
    except ValueError:
        spectral_moment_source = str(raw_moment_source)

    raw_entrywise_delta_mode = params.get(
        "spectral_entrywise_delta_mode",
        DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    try:
        spectral_entrywise_delta_mode = resolve_spectral_entrywise_delta_mode(
            None if raw_entrywise_delta_mode is None else str(raw_entrywise_delta_mode)
        )
    except ValueError:
        spectral_entrywise_delta_mode = str(raw_entrywise_delta_mode)

    raw_attention_granularity = params.get(
        "spectral_attention_granularity",
        DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    )
    try:
        spectral_attention_granularity = resolve_spectral_attention_granularity(
            None if raw_attention_granularity is None else str(raw_attention_granularity)
        )
    except ValueError:
        spectral_attention_granularity = str(raw_attention_granularity)

    return {
        "block_size": int(params.get("block_size", 131072)),
        "dtype": str(params.get("dtype", "float32")),
        "spectral_features": spectral_features,
        "spectral_sv_top_k": int(params.get("spectral_sv_top_k", 8)),
        "spectral_moment_source": spectral_moment_source,
        "spectral_qv_sum_mode": spectral_qv_sum_mode,
        "spectral_entrywise_delta_mode": spectral_entrywise_delta_mode,
        "spectral_attention_granularity": spectral_attention_granularity,
    }
