from __future__ import annotations

from typing import Any

import numpy as np
from safetensors import safe_open

from .delta import (
    DELTA_SCHEMA_VERSION,
    DeltaBlockSchema,
    block_delta_singular_values,
    block_spectral_scalars,
    build_schema_metadata,
    check_consistency_reader,
    load_delta_block_schema,
    shorten_block_name,
    top_k_singular_values,
)
from .norms import block_moments_from_factors, summarize_array_moments
from ..utilities.manifest import ManifestItem


SPECTRAL_EXTRACTOR_VERSION = "2.0.0"

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

DEFAULT_SPECTRAL_QV_SUM_MODE = "none"
DEFAULT_SPECTRAL_MOMENT_SOURCE = "sv"
SUPPORTED_SPECTRAL_QV_SUM_MODES = {"none", "append", "only"}
SUPPORTED_SPECTRAL_MOMENT_SOURCES = {"entrywise", "sv", "both"}

SUPPORTED_SPECTRAL_FEATURES = {
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
}

_MOMENT_FEATURES = ("kurtosis", "l1_norm", "linf_norm", "mean_abs")
_MOMENT_FEATURE_SET = set(_MOMENT_FEATURES)
_SV_MOMENT_FEATURES = {
    "sv_kurtosis",
    "sv_l1_norm",
    "sv_linf_norm",
    "sv_mean_abs",
}
_SV_MOMENT_BY_ENTRYWISE = {
    "kurtosis": "sv_kurtosis",
    "l1_norm": "sv_l1_norm",
    "linf_norm": "sv_linf_norm",
    "mean_abs": "sv_mean_abs",
}
_SPECTRAL_SCALAR_FEATURES = {
    "energy",
    "l2_norm",
    "concentration_of_energy",
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


def resolve_spectral_qv_sum_mode(mode: str | None) -> str:
    resolved = DEFAULT_SPECTRAL_QV_SUM_MODE if mode is None else str(mode).strip().lower()
    if resolved not in SUPPORTED_SPECTRAL_QV_SUM_MODES:
        raise ValueError(
            f"Unknown spectral_qv_sum_mode '{mode}'. "
            f"Supported: {sorted(SUPPORTED_SPECTRAL_QV_SUM_MODES)}"
        )
    return resolved


def resolve_spectral_moment_source(mode: str | None) -> str:
    resolved = DEFAULT_SPECTRAL_MOMENT_SOURCE if mode is None else str(mode).strip().lower()
    if resolved not in SUPPORTED_SPECTRAL_MOMENT_SOURCES:
        raise ValueError(
            f"Unknown spectral_moment_source '{mode}'. "
            f"Supported: {sorted(SUPPORTED_SPECTRAL_MOMENT_SOURCES)}"
        )
    return resolved


def expand_spectral_feature_names(
    *,
    selected_features: list[str],
    spectral_moment_source: str,
) -> list[str]:
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    emitted: list[str] = []
    for feature in selected_features:
        if feature not in _MOMENT_FEATURE_SET:
            emitted.append(feature)
            continue
        if resolved_moment_source in {"entrywise", "both"}:
            emitted.append(feature)
        if resolved_moment_source in {"sv", "both"}:
            emitted.append(_SV_MOMENT_BY_ENTRYWISE[feature])
    return emitted


def _qv_sum_group_for_block_name(block_name: str) -> tuple[str, str] | None:
    parts = block_name.split(".")
    if "layers" not in parts or "self_attn" not in parts:
        return None

    sa_idx = parts.index("self_attn")
    if sa_idx + 1 >= len(parts):
        return None
    module = parts[sa_idx + 1]
    if module not in {"q_proj", "v_proj"}:
        return None

    group = ".".join(parts[: sa_idx + 1])
    return f"{group}.qv_sum", module


def _build_qv_sum_specs(schema: DeltaBlockSchema) -> list[tuple[str, int, int]]:
    groups: dict[str, dict[str, int]] = {}
    order: list[str] = []

    for pair_idx, block_name in enumerate(schema.block_names):
        parsed = _qv_sum_group_for_block_name(block_name)
        if parsed is None:
            continue
        qv_block_name_raw, module = parsed
        if qv_block_name_raw not in groups:
            groups[qv_block_name_raw] = {}
            order.append(qv_block_name_raw)
        if module in groups[qv_block_name_raw]:
            raise ValueError(
                f"Duplicate '{module}' block while building q+v sum for {qv_block_name_raw}"
            )
        groups[qv_block_name_raw][module] = int(pair_idx)

    missing_roles: list[str] = []
    specs: list[tuple[str, int, int]] = []
    for qv_block_name_raw in order:
        roles = groups[qv_block_name_raw]
        if "q_proj" not in roles or "v_proj" not in roles:
            missing = [name for name in ("q_proj", "v_proj") if name not in roles]
            missing_roles.append(
                f"{shorten_block_name(qv_block_name_raw)} missing {', '.join(missing)}"
            )
            continue

        q_idx = int(roles["q_proj"])
        v_idx = int(roles["v_proj"])
        a_q_shape = tuple(int(x) for x in schema.a_shapes[q_idx])
        a_v_shape = tuple(int(x) for x in schema.a_shapes[v_idx])
        b_q_shape = tuple(int(x) for x in schema.b_shapes[q_idx])
        b_v_shape = tuple(int(x) for x in schema.b_shapes[v_idx])

        # q+v can only be formed when in/out dims match; rank may differ.
        if a_q_shape[1] != a_v_shape[1] or b_q_shape[0] != b_v_shape[0]:
            raise ValueError(
                "Cannot build q+v sum for "
                f"{shorten_block_name(qv_block_name_raw)}: incompatible q/v shapes "
                f"A_q={a_q_shape}, B_q={b_q_shape}, A_v={a_v_shape}, B_v={b_v_shape}"
            )

        specs.append((qv_block_name_raw, q_idx, v_idx))

    if missing_roles:
        preview = "; ".join(missing_roles[:5])
        raise ValueError(
            "Requested q+v-per-layer spectral features, but some layers do not expose "
            f"both q_proj and v_proj LoRA blocks: {preview}"
        )
    return specs


def build_spectral_feature_names(
    *,
    block_names: list[str],
    selected_features: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
    shorten_block_names: bool = True,
) -> list[str]:
    emitted_features = expand_spectral_feature_names(
        selected_features=selected_features,
        spectral_moment_source=spectral_moment_source,
    )
    names: list[str] = []
    for block in block_names:
        prefix = shorten_block_name(block) if shorten_block_names else str(block)
        for feat in emitted_features:
            if feat == "sv_topk":
                for i in range(sv_top_k):
                    names.append(f"{prefix}.sv_{i + 1}")
            else:
                names.append(f"{prefix}.{feat}")
    return names


def _needs_entrywise_moments(emitted_features: list[str]) -> bool:
    return any(feature in _MOMENT_FEATURE_SET for feature in emitted_features)


def _needs_sv_moments(emitted_features: list[str]) -> bool:
    return any(feature in _SV_MOMENT_FEATURES for feature in emitted_features)


def _needs_spectral_scalars(emitted_features: list[str]) -> bool:
    return any(feature in _SPECTRAL_SCALAR_FEATURES for feature in emitted_features)


def _needs_sv_topk(emitted_features: list[str]) -> bool:
    return "sv_topk" in emitted_features


def _append_block_features(
    *,
    row: list[float],
    emitted_features: list[str],
    singular_values: np.ndarray,
    sv_top_k: int,
    need_spectral_scalars: bool,
    need_sv_topk: bool,
    entrywise_moments: dict[str, float] | None,
    sv_moments: dict[str, float] | None,
) -> None:
    energy = stable_rank = spectral_entropy = effective_rank = None
    l2_norm = None
    concentration_of_energy = None
    if need_spectral_scalars:
        energy, stable_rank, spectral_entropy, effective_rank = block_spectral_scalars(singular_values)
        l2_norm = float(np.sqrt(max(0.0, float(energy))))
        sv = np.asarray(singular_values, dtype=np.float64)
        sv_sum = float(np.sum(sv, dtype=np.float64))
        concentration_of_energy = 0.0 if sv_sum <= 0.0 or sv.size == 0 else float(sv[0] / sv_sum)

    top_k: np.ndarray | None = None
    if need_sv_topk:
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
        elif feature == "concentration_of_energy":
            if concentration_of_energy is None:
                raise RuntimeError(
                    "Spectral scalars were required for 'concentration_of_energy' but were not computed"
                )
            row.append(concentration_of_energy)
        elif feature in _MOMENT_FEATURE_SET:
            if entrywise_moments is None:
                raise RuntimeError(f"Entry-wise moments are required for feature '{feature}' but were not computed")
            row.append(float(entrywise_moments[feature]))
        elif feature in _SV_MOMENT_FEATURES:
            if sv_moments is None:
                raise RuntimeError(
                    f"Singular-value moments are required for feature '{feature}' but were not computed"
                )
            row.append(float(sv_moments[feature]))
        else:
            raise RuntimeError(f"Unknown emitted spectral feature '{feature}'")


def extract_spectral_features(
    *,
    items: list[ManifestItem],
    spectral_features: list[str] | None,
    spectral_qv_sum_mode: str = DEFAULT_SPECTRAL_QV_SUM_MODE,
    spectral_moment_source: str = DEFAULT_SPECTRAL_MOMENT_SOURCE,
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
    resolved_qv_sum_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    emitted_features = expand_spectral_feature_names(
        selected_features=selected_features,
        spectral_moment_source=resolved_moment_source,
    )
    schema = load_delta_block_schema(items[0].adapter_path)

    expected_pairs = [tuple(x) for x in schema.pairs]
    expected_a_shapes = [tuple(x) for x in schema.a_shapes]
    expected_b_shapes = [tuple(x) for x in schema.b_shapes]

    include_base_blocks = resolved_qv_sum_mode in {"none", "append"}
    include_qv_sum_blocks = resolved_qv_sum_mode in {"append", "only"}

    qv_specs: list[tuple[str, int, int]] = []
    qv_pair_lookup: dict[int, tuple[str, str]] = {}
    if include_qv_sum_blocks:
        qv_specs = _build_qv_sum_specs(schema)
        if not qv_specs:
            raise ValueError(
                "spectral_qv_sum_mode requested q+v features, but no q_proj/v_proj layer pairs were found"
            )
        for qv_block_name_raw, q_idx, v_idx in qv_specs:
            qv_pair_lookup[q_idx] = (qv_block_name_raw, "q_proj")
            qv_pair_lookup[v_idx] = (qv_block_name_raw, "v_proj")

    rows: list[list[float]] = []
    model_names: list[str] = []
    labels_list: list[int | None] = []
    need_entrywise_moments = _needs_entrywise_moments(emitted_features)
    need_sv_moments = _needs_sv_moments(emitted_features)
    need_spectral_scalars = _needs_spectral_scalars(emitted_features)
    need_top_k = _needs_sv_topk(emitted_features)

    for item in items:
        row: list[float] = []
        qv_factors: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        with safe_open(item.adapter_path, framework="numpy") as reader:
            check_consistency_reader(
                reader=reader,
                adapter_path=item.adapter_path,
                expected_pairs=expected_pairs,
                expected_a_shapes=expected_a_shapes,
                expected_b_shapes=expected_b_shapes,
            )
            for pair_idx, (a_key, b_key) in enumerate(schema.pairs):
                a = np.asarray(reader.get_tensor(a_key), dtype=dtype)
                b = np.asarray(reader.get_tensor(b_key), dtype=dtype)
                if include_base_blocks:
                    singular_values = block_delta_singular_values(a=a, b=b)

                    entrywise_moments: dict[str, float] | None = None
                    if need_entrywise_moments:
                        summary = block_moments_from_factors(
                            a=a,
                            b=b,
                            block_size=block_size,
                            dtype=dtype,
                        )
                        entrywise_moments = {
                            "kurtosis": float(summary.kurtosis),
                            "l1_norm": float(summary.l1_norm),
                            "linf_norm": float(summary.linf_norm),
                            "mean_abs": float(summary.mean_abs),
                        }

                    sv_moments: dict[str, float] | None = None
                    if need_sv_moments:
                        summary = summarize_array_moments(singular_values)
                        sv_moments = {
                            "sv_kurtosis": float(summary.kurtosis),
                            "sv_l1_norm": float(summary.l1_norm),
                            "sv_linf_norm": float(summary.linf_norm),
                            "sv_mean_abs": float(summary.mean_abs),
                        }

                    _append_block_features(
                        row=row,
                        emitted_features=emitted_features,
                        singular_values=singular_values,
                        sv_top_k=sv_top_k,
                        need_spectral_scalars=need_spectral_scalars,
                        need_sv_topk=need_top_k,
                        entrywise_moments=entrywise_moments,
                        sv_moments=sv_moments,
                    )

                if include_qv_sum_blocks and pair_idx in qv_pair_lookup:
                    qv_block_name_raw, role = qv_pair_lookup[pair_idx]
                    bucket = qv_factors.setdefault(qv_block_name_raw, {})
                    bucket[role] = (a, b)

        if include_qv_sum_blocks:
            for qv_block_name_raw, _, _ in qv_specs:
                pair = qv_factors.get(qv_block_name_raw)
                if pair is None or "q_proj" not in pair or "v_proj" not in pair:
                    raise RuntimeError(
                        f"Missing q/v factors for {shorten_block_name(qv_block_name_raw)} in {item.adapter_path}"
                    )

                a_q, b_q = pair["q_proj"]
                a_v, b_v = pair["v_proj"]
                a_qv = np.concatenate([a_q, a_v], axis=0)
                b_qv = np.concatenate([b_q, b_v], axis=1)
                singular_values = block_delta_singular_values(a=a_qv, b=b_qv)

                entrywise_moments: dict[str, float] | None = None
                if need_entrywise_moments:
                    summary = block_moments_from_factors(
                        a=a_qv,
                        b=b_qv,
                        block_size=block_size,
                        dtype=dtype,
                    )
                    entrywise_moments = {
                        "kurtosis": float(summary.kurtosis),
                        "l1_norm": float(summary.l1_norm),
                        "linf_norm": float(summary.linf_norm),
                        "mean_abs": float(summary.mean_abs),
                    }

                sv_moments: dict[str, float] | None = None
                if need_sv_moments:
                    summary = summarize_array_moments(singular_values)
                    sv_moments = {
                        "sv_kurtosis": float(summary.kurtosis),
                        "sv_l1_norm": float(summary.l1_norm),
                        "sv_linf_norm": float(summary.linf_norm),
                        "sv_mean_abs": float(summary.mean_abs),
                    }

                _append_block_features(
                    row=row,
                    emitted_features=emitted_features,
                    singular_values=singular_values,
                    sv_top_k=sv_top_k,
                    need_spectral_scalars=need_spectral_scalars,
                    need_sv_topk=need_top_k,
                    entrywise_moments=entrywise_moments,
                    sv_moments=sv_moments,
                )

        rows.append(row)
        model_names.append(item.model_name)
        labels_list.append(item.label)

    features = np.asarray(rows, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.int32) if all(label is not None for label in labels_list) else None

    feature_block_names_raw: list[str] = []
    if include_base_blocks:
        feature_block_names_raw.extend(str(x) for x in schema.block_names)
    if include_qv_sum_blocks:
        feature_block_names_raw.extend(name for name, _, _ in qv_specs)

    feature_names = build_spectral_feature_names(
        block_names=feature_block_names_raw,
        selected_features=selected_features,
        sv_top_k=sv_top_k,
        spectral_moment_source=resolved_moment_source,
    )
    if features.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Spectral feature dimension mismatch: matrix has {features.shape[1]} columns, "
            f"but generated {len(feature_names)} feature names"
        )

    schema_metadata = build_schema_metadata(schema)
    qv_sum_block_names_raw = [name for name, _, _ in qv_specs]
    qv_sum_block_names = [shorten_block_name(name) for name in qv_sum_block_names_raw]
    base_block_names = [str(x) for x in schema_metadata.get("block_names", [])]
    base_block_names_raw = [str(x) for x in schema_metadata.get("block_names_raw", [])]
    feature_block_names = [shorten_block_name(name) for name in feature_block_names_raw]

    metadata: dict[str, Any] = {
        "extractor": "spectral",
        "extractor_version": SPECTRAL_EXTRACTOR_VERSION,
        "delta_schema_version": DELTA_SCHEMA_VERSION,
        "n_models": int(len(items)),
        "resolved_features": selected_features,
        "spectral_qv_sum_mode": resolved_qv_sum_mode,
        "spectral_moment_source": resolved_moment_source,
        "sv_top_k": int(sv_top_k),
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names,
        **schema_metadata,
        "base_block_names": base_block_names,
        "base_block_names_raw": base_block_names_raw,
        "qv_sum_block_names": qv_sum_block_names,
        "qv_sum_block_names_raw": qv_sum_block_names_raw,
        "block_names": feature_block_names,
        "block_names_raw": feature_block_names_raw,
        "n_blocks": int(len(feature_block_names)),
    }
    return features, labels, model_names, metadata
