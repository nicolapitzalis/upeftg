"""Numerical feature computation for individual LoRA blocks."""

from __future__ import annotations

import numpy as np

from ..delta import block_delta_singular_values, block_spectral_scalars, top_k_singular_values
from ..norms import block_moments_from_factors, summarize_array_moments
from .config import _MOMENT_FEATURE_SET, _SPECTRAL_SCALAR_FEATURES, _SV_MOMENT_FEATURES
from .layout import _effective_delta_rank_scale


def _needs_entrywise_moments(emitted_features: list[str]) -> bool:
    return any(feature in _MOMENT_FEATURE_SET for feature in emitted_features)


def _needs_sv_moments(emitted_features: list[str]) -> bool:
    return any(feature in _SV_MOMENT_FEATURES for feature in emitted_features)


def _needs_spectral_scalars(emitted_features: list[str]) -> bool:
    return any(feature in _SPECTRAL_SCALAR_FEATURES - {"block_rank"} for feature in emitted_features)


def _needs_sv_topk(emitted_features: list[str]) -> bool:
    return "sv_topk" in emitted_features


def _append_block_features(
    *,
    row: list[float],
    emitted_features: list[str],
    singular_values: np.ndarray | None,
    block_rank: int,
    sv_top_k: int,
    need_spectral_scalars: bool,
    need_sv_topk: bool,
    entrywise_moments: dict[str, float] | None,
    sv_moments: dict[str, float] | None,
) -> None:
    energy = stable_rank = spectral_entropy = effective_rank = None
    l2_norm = None
    concentration_of_energy = None
    rank_scale = max(1, int(block_rank))
    sqrt_rank_scale = float(np.sqrt(rank_scale))
    entropy_rank_scale = float(np.log(max(2, rank_scale)))
    if need_spectral_scalars:
        if singular_values is None:
            raise RuntimeError("Spectral scalars were requested but singular values were not computed")
        energy, stable_rank, spectral_entropy, effective_rank = block_spectral_scalars(singular_values)
        l2_norm = float(np.sqrt(max(0.0, float(energy))))
        sv = np.asarray(singular_values, dtype=np.float64)
        sv_sum = float(np.sum(sv, dtype=np.float64))
        concentration_of_energy = 0.0 if sv_sum <= 0.0 or sv.size == 0 else float(sv[0] / sv_sum)

    top_k: np.ndarray | None = None
    if need_sv_topk:
        if singular_values is None:
            raise RuntimeError("Top-k singular values were requested but singular values were not computed")
        top_k = top_k_singular_values(singular_values, top_k=sv_top_k)

    for feature in emitted_features:
        if feature == "sv_topk":
            if top_k is None:
                raise RuntimeError("Top-k singular values were requested but not computed")
            row.extend(float(x) for x in top_k.tolist())
        elif feature == "energy":
            if energy is None:
                raise RuntimeError("Spectral scalars were required for 'energy' but were not computed")
            row.append(energy)
        elif feature == "energy_per_rank":
            if energy is None:
                raise RuntimeError("Spectral scalars were required for 'energy_per_rank' but were not computed")
            row.append(float(energy) / float(rank_scale))
        elif feature == "block_rank":
            row.append(float(block_rank))
        elif feature == "l2_norm":
            if l2_norm is None:
                raise RuntimeError("Spectral scalars were required for 'l2_norm' but were not computed")
            row.append(l2_norm)
        elif feature == "l2_norm_per_sqrt_rank":
            if l2_norm is None:
                raise RuntimeError("Spectral scalars were required for 'l2_norm_per_sqrt_rank' but were not computed")
            row.append(float(l2_norm) / sqrt_rank_scale)
        elif feature == "stable_rank":
            if stable_rank is None:
                raise RuntimeError("Spectral scalars were required for 'stable_rank' but were not computed")
            row.append(stable_rank)
        elif feature == "stable_rank_frac":
            if stable_rank is None:
                raise RuntimeError("Spectral scalars were required for 'stable_rank_frac' but were not computed")
            row.append(float(stable_rank) / float(rank_scale))
        elif feature == "spectral_entropy":
            if spectral_entropy is None:
                raise RuntimeError("Spectral scalars were required for 'spectral_entropy' but were not computed")
            row.append(spectral_entropy)
        elif feature == "normalized_spectral_entropy":
            if spectral_entropy is None:
                raise RuntimeError(
                    "Spectral scalars were required for 'normalized_spectral_entropy' but were not computed"
                )
            row.append(0.0 if rank_scale <= 1 else float(spectral_entropy) / entropy_rank_scale)
        elif feature == "effective_rank":
            if effective_rank is None:
                raise RuntimeError("Spectral scalars were required for 'effective_rank' but were not computed")
            row.append(effective_rank)
        elif feature == "effective_rank_frac":
            if effective_rank is None:
                raise RuntimeError("Spectral scalars were required for 'effective_rank_frac' but were not computed")
            row.append(float(effective_rank) / float(rank_scale))
        elif feature == "concentration_of_energy":
            if concentration_of_energy is None:
                raise RuntimeError("Spectral scalars were required for 'concentration_of_energy' but were not computed")
            row.append(concentration_of_energy)
        elif feature in _MOMENT_FEATURE_SET:
            if entrywise_moments is None:
                raise RuntimeError(f"Entry-wise moments are required for feature '{feature}' but were not computed")
            row.append(float(entrywise_moments[feature]))
        elif feature in _SV_MOMENT_FEATURES:
            if sv_moments is None:
                raise RuntimeError(f"Singular-value moments are required for feature '{feature}' but were not computed")
            row.append(float(sv_moments[feature]))
        else:
            raise RuntimeError(f"Unknown emitted spectral feature '{feature}'")


def _append_factor_features(
    *,
    row: list[float],
    emitted_features: list[str],
    a: np.ndarray,
    b: np.ndarray,
    a_qr_r: np.ndarray | None = None,
    sv_top_k: int,
    block_size: int,
    dtype: np.dtype,
    spectral_entrywise_delta_mode: str,
    need_spectral_scalars: bool,
    need_sv_topk: bool,
    need_singular_values: bool,
    need_entrywise_moments: bool,
    need_sv_moments: bool,
) -> tuple[int, int]:
    block_rank = _effective_delta_rank_scale(a, b)
    singular_values = block_delta_singular_values(a=a, b=b, a_qr_r=a_qr_r) if need_singular_values else None

    entrywise_dense_block_count = 0
    entrywise_stream_block_count = 0
    entrywise_moments: dict[str, float] | None = None
    if need_entrywise_moments:
        summary, runtime_entrywise_mode = block_moments_from_factors(
            a=a,
            b=b,
            block_size=block_size,
            dtype=dtype,
            entrywise_delta_mode=spectral_entrywise_delta_mode,
        )
        if runtime_entrywise_mode == "dense":
            entrywise_dense_block_count += 1
        else:
            entrywise_stream_block_count += 1
        entrywise_moments = {
            "kurtosis": float(summary.kurtosis),
            "l1_norm": float(summary.l1_norm),
            "l1_norm_per_rank": float(summary.l1_norm) / max(1, block_rank),
            "linf_norm": float(summary.linf_norm),
            "mean_abs": float(summary.mean_abs),
            "mean_abs_per_rank": float(summary.mean_abs) / max(1, block_rank),
        }

    sv_moments: dict[str, float] | None = None
    if need_sv_moments:
        if singular_values is None:
            raise RuntimeError("Singular-value moments were requested but singular values were not computed")
        summary = summarize_array_moments(singular_values)
        sv_moments = {
            "sv_kurtosis": float(summary.kurtosis),
            "sv_l1_norm": float(summary.l1_norm),
            "sv_l1_norm_per_rank": float(summary.l1_norm) / max(1, block_rank),
            "sv_linf_norm": float(summary.linf_norm),
            "sv_mean_abs": float(summary.mean_abs),
            "sv_mean_abs_per_rank": float(summary.mean_abs) / max(1, block_rank),
        }

    _append_block_features(
        row=row,
        emitted_features=emitted_features,
        singular_values=singular_values,
        block_rank=block_rank,
        sv_top_k=sv_top_k,
        need_spectral_scalars=need_spectral_scalars,
        need_sv_topk=need_sv_topk,
        entrywise_moments=entrywise_moments,
        sv_moments=sv_moments,
    )
    return entrywise_dense_block_count, entrywise_stream_block_count
