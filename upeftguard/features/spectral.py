from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open

from .delta import (
    DELTA_SCHEMA_VERSION,
    DeltaBlockSchema,
    block_delta_singular_values,
    block_spectral_scalars,
    build_schema_metadata,
    check_consistency_reader,
    iter_block_factors,
    lora_adapter_dims_from_shapes,
    load_delta_block_schema,
    schema_has_adalora_scaling,
    shorten_block_name,
    top_k_singular_values,
)
from .norms import (
    DEFAULT_ENTRYWISE_DELTA_MODE,
    SUPPORTED_ENTRYWISE_DELTA_MODES,
    block_moments_from_factors,
    resolve_entrywise_delta_mode,
    summarize_array_moments,
)
from ..utilities.core.manifest import ManifestItem


SPECTRAL_EXTRACTOR_VERSION = "2.5.0"

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
DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE = DEFAULT_ENTRYWISE_DELTA_MODE
SUPPORTED_SPECTRAL_QV_SUM_MODES = {"none", "append", "only"}
SUPPORTED_SPECTRAL_MOMENT_SOURCES = {"entrywise", "sv", "both"}
SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES = tuple(SUPPORTED_ENTRYWISE_DELTA_MODES)

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

_SKIPPABLE_SPECTRAL_ITEM_EXCEPTIONS = (OSError, SafetensorError, ValueError)


@dataclass(frozen=True)
class QvSumSpec:
    qv_block_name_raw: str
    q_index: int
    v_index: int
    q_block_name_raw: str
    v_block_name_raw: str


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


def resolve_spectral_entrywise_delta_mode(mode: str | None) -> str:
    return resolve_entrywise_delta_mode(mode)


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
    if len(parts) < 2:
        return None

    module = parts[-1]
    role = _QV_ROLE_ALIASES.get(module)
    if role is None:
        return None
    if not any("attn" in part.lower() or "attention" in part.lower() for part in parts[:-1]):
        return None

    group = ".".join(parts[:-1])
    return f"{group}.qv_sum", role


def build_qv_sum_specs(schema: DeltaBlockSchema) -> list[QvSumSpec]:
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
    specs: list[QvSumSpec] = []
    for qv_block_name_raw in order:
        roles = groups[qv_block_name_raw]
        if "q" not in roles or "v" not in roles:
            missing = [name for name in ("q", "v") if name not in roles]
            missing_roles.append(
                f"{shorten_block_name(qv_block_name_raw)} missing {', '.join(missing)}"
            )
            continue

        q_idx = int(roles["q"])
        v_idx = int(roles["v"])
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

        specs.append(
            QvSumSpec(
                qv_block_name_raw=qv_block_name_raw,
                q_index=q_idx,
                v_index=v_idx,
                q_block_name_raw=str(schema.block_names[q_idx]),
                v_block_name_raw=str(schema.block_names[v_idx]),
            )
        )

    if missing_roles:
        preview = "; ".join(missing_roles[:5])
        raise ValueError(
            "Requested q+v-per-layer spectral features, but some attention groups do not expose "
            f"both q and v LoRA blocks: {preview}"
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
        spectral_qv_sum_mode = resolve_spectral_qv_sum_mode(
            None if raw_qv_sum_mode is None else str(raw_qv_sum_mode)
        )
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

    return {
        "block_size": int(params.get("block_size", 131072)),
        "dtype": str(params.get("dtype", "float32")),
        "spectral_features": spectral_features,
        "spectral_sv_top_k": int(params.get("spectral_sv_top_k", 8)),
        "spectral_moment_source": spectral_moment_source,
        "spectral_qv_sum_mode": spectral_qv_sum_mode,
        "spectral_entrywise_delta_mode": spectral_entrywise_delta_mode,
    }


def feature_block_name(feature_name: str) -> str:
    block_name, sep, _ = str(feature_name).rpartition(".")
    if not sep or not block_name:
        raise ValueError(f"Invalid spectral feature name: {feature_name}")
    return block_name


def ordered_block_names_from_feature_names(feature_names: list[str]) -> list[str]:
    block_names: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        block_name = feature_block_name(feature_name)
        if block_name in seen:
            continue
        seen.add(block_name)
        block_names.append(block_name)
    return block_names


def _layer_identifier_for_block_name(block_name: str) -> str:
    parts = [part for part in str(block_name).split(".") if part]
    if len(parts) <= 2:
        return str(block_name)
    prefix = ".".join(parts[:-2]).strip()
    return prefix or str(block_name)


def _summarize_adapter_dims(
    dims: list[dict[str, int]],
    *,
    variable_lora_rank: bool,
) -> dict[str, Any] | None:
    normalized: list[tuple[int, int, int]] = []
    for raw_dims in dims:
        normalized_dims = _normalize_lora_dims_entry(raw_dims)
        if normalized_dims is None:
            continue
        normalized.append(
            (
                int(normalized_dims["m"]),
                int(normalized_dims["n"]),
                int(normalized_dims["r"]),
            )
        )
    if not normalized:
        return None

    unique = sorted(set(normalized))
    unique_mn = sorted({(m, n) for m, n, _r in unique})
    if len(unique) == 1:
        m, n, r = unique[0]
        return {"m": int(m), "n": int(n), "r": int(r)}

    if len(unique_mn) == 1:
        m, n = unique_mn[0]
        r_values = sorted({int(r) for _m, _n, r in unique})
        return {
            "m": int(m),
            "n": int(n),
            "r": {
                "mode": "adaptive" if variable_lora_rank else "mixed",
                "values": [int(x) for x in r_values],
                "min": int(min(r_values)),
                "max": int(max(r_values)),
            },
        }

    return {
        "mode": "mixed",
        "values": [{"m": int(m), "n": int(n), "r": int(r)} for m, n, r in unique],
    }


def summarize_schema_layout(
    *,
    base_block_names: list[str],
    base_lora_adapter_dims: list[dict[str, int]],
    variable_lora_rank: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "layer_count": int(len({_layer_identifier_for_block_name(name) for name in base_block_names})),
    }
    adapter_dims = _summarize_adapter_dims(
        list(base_lora_adapter_dims),
        variable_lora_rank=bool(variable_lora_rank),
    )
    if adapter_dims is not None:
        summary["adapter_dims"] = adapter_dims
    return summary


def _normalize_lora_dims_entry(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {
            "m": int(value["m"]),
            "n": int(value["n"]),
            "r": int(value["r"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _normalize_shape_list(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(int(x) for x in value)
    except (TypeError, ValueError):
        return None


def _update_lora_dims_from_explicit_metadata(
    dim_map: dict[str, dict[str, int]],
    *,
    block_names: Any,
    dims: Any,
) -> None:
    if not isinstance(block_names, list) or not isinstance(dims, list) or len(block_names) != len(dims):
        return

    for block_name, raw_dims in zip(block_names, dims):
        normalized = _normalize_lora_dims_entry(raw_dims)
        if normalized is None:
            continue
        dim_map[str(block_name)] = normalized


def _update_lora_dims_from_shapes(
    dim_map: dict[str, dict[str, int]],
    *,
    block_names: Any,
    a_shapes: Any,
    b_shapes: Any,
) -> None:
    if (
        not isinstance(block_names, list)
        or not isinstance(a_shapes, list)
        or not isinstance(b_shapes, list)
        or len(block_names) != len(a_shapes)
        or len(block_names) != len(b_shapes)
    ):
        return

    for block_name, raw_a_shape, raw_b_shape in zip(block_names, a_shapes, b_shapes):
        a_shape = _normalize_shape_list(raw_a_shape)
        b_shape = _normalize_shape_list(raw_b_shape)
        if a_shape is None or b_shape is None:
            continue
        try:
            dim_map[str(block_name)] = lora_adapter_dims_from_shapes(a_shape, b_shape)
        except ValueError:
            continue


def _derive_qv_sum_lora_dims(
    *,
    dim_map: dict[str, dict[str, int]],
    qv_sum_block_names: Any,
) -> None:
    if not isinstance(qv_sum_block_names, list):
        return

    for raw_qv_name in qv_sum_block_names:
        qv_name = str(raw_qv_name)
        prefix, sep, _ = qv_name.rpartition(".qv_sum")
        if not sep or not prefix:
            continue
        q_dims = (
            dim_map.get(prefix + ".q_proj")
            or dim_map.get(prefix + ".q")
            or dim_map.get(prefix + ".query")
        )
        v_dims = (
            dim_map.get(prefix + ".v_proj")
            or dim_map.get(prefix + ".v")
            or dim_map.get(prefix + ".value")
        )
        if q_dims is None or v_dims is None:
            continue
        if q_dims["m"] != v_dims["m"] or q_dims["n"] != v_dims["n"]:
            continue
        dim_map[qv_name] = {
            "m": int(q_dims["m"]),
            "n": int(q_dims["n"]),
            "r": int(q_dims["r"] + v_dims["r"]),
        }


def spectral_block_lora_dims_by_block(metadata: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    dim_map: dict[str, dict[str, int]] = {}
    raw_a_shapes = metadata.get("a_shapes")
    raw_b_shapes = metadata.get("b_shapes")

    _update_lora_dims_from_explicit_metadata(
        dim_map,
        block_names=metadata.get("block_names"),
        dims=metadata.get("lora_adapter_dims"),
    )
    _update_lora_dims_from_explicit_metadata(
        dim_map,
        block_names=metadata.get("base_block_names"),
        dims=metadata.get("base_lora_adapter_dims"),
    )
    _update_lora_dims_from_explicit_metadata(
        dim_map,
        block_names=metadata.get("qv_sum_block_names"),
        dims=metadata.get("qv_sum_lora_adapter_dims"),
    )

    base_block_names = metadata.get("base_block_names")
    legacy_block_names = (
        base_block_names
        if isinstance(base_block_names, list)
        and isinstance(raw_a_shapes, list)
        and len(base_block_names) == len(raw_a_shapes)
        else metadata.get("block_names")
    )
    _update_lora_dims_from_shapes(
        dim_map,
        block_names=legacy_block_names,
        a_shapes=raw_a_shapes,
        b_shapes=raw_b_shapes,
    )
    _derive_qv_sum_lora_dims(
        dim_map=dim_map,
        qv_sum_block_names=metadata.get("qv_sum_block_names"),
    )
    return dim_map


def sanitize_spectral_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        key_str = str(key)
        if key_str in _SPECTRAL_METADATA_DROP_KEYS:
            continue
        if key_str == "extractor_params" and isinstance(value, Mapping):
            cleaned[key_str] = spectral_extractor_params(value)
            continue
        cleaned[key_str] = value

    raw_block_names = cleaned.get("block_names")
    block_names = [str(x) for x in raw_block_names] if isinstance(raw_block_names, list) else None
    if block_names is not None:
        cleaned["block_names"] = block_names
        cleaned["n_blocks"] = int(len(block_names))

    for key in ("base_block_names", "qv_sum_block_names"):
        value = cleaned.get(key)
        if isinstance(value, list):
            cleaned[key] = [str(x) for x in value]

    extractor_params = cleaned.get("extractor_params")
    if isinstance(extractor_params, dict):
        if "resolved_features" not in cleaned and isinstance(extractor_params.get("spectral_features"), list):
            cleaned["resolved_features"] = [str(x) for x in extractor_params["spectral_features"]]
        if "sv_top_k" not in cleaned and "spectral_sv_top_k" in extractor_params:
            cleaned["sv_top_k"] = int(extractor_params["spectral_sv_top_k"])
        if "spectral_moment_source" not in cleaned and "spectral_moment_source" in extractor_params:
            cleaned["spectral_moment_source"] = str(extractor_params["spectral_moment_source"])
        if "spectral_qv_sum_mode" not in cleaned and "spectral_qv_sum_mode" in extractor_params:
            cleaned["spectral_qv_sum_mode"] = str(extractor_params["spectral_qv_sum_mode"])
        if (
            "spectral_entrywise_delta_mode" not in cleaned
            and "spectral_entrywise_delta_mode" in extractor_params
        ):
            cleaned["spectral_entrywise_delta_mode"] = str(extractor_params["spectral_entrywise_delta_mode"])

    if block_names:
        dim_map = spectral_block_lora_dims_by_block(metadata)
        if all(block_name in dim_map for block_name in block_names):
            cleaned["lora_adapter_dims"] = [dim_map[block_name] for block_name in block_names]
        else:
            cleaned.pop("lora_adapter_dims", None)

    return cleaned


def _skip_entry_for_item(item: ManifestItem, exc: Exception) -> dict[str, Any]:
    return {
        "model_name": str(item.model_name),
        "adapter_path": str(item.adapter_path),
        "label": int(item.label) if item.label is not None else None,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


def _resolve_readable_schema(
    items: list[ManifestItem],
) -> tuple[DeltaBlockSchema, list[dict[str, Any]], set[str]]:
    skipped_models: list[dict[str, Any]] = []
    skipped_adapter_paths: set[str] = set()
    for item in items:
        try:
            return load_delta_block_schema(item.adapter_path), skipped_models, skipped_adapter_paths
        except _SKIPPABLE_SPECTRAL_ITEM_EXCEPTIONS as exc:
            skipped_models.append(_skip_entry_for_item(item, exc))
            skipped_adapter_paths.add(str(item.adapter_path))

    preview = "; ".join(
        f"{entry['model_name']} ({entry['exception_type']}: {entry['exception_message']})"
        for entry in skipped_models[:3]
    )
    raise RuntimeError(
        "No readable adapters were available for spectral extraction"
        + (f". Examples: {preview}" if preview else "")
    )


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
    singular_values: np.ndarray | None,
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
    spectral_entrywise_delta_mode: str = DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
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
    resolved_entrywise_delta_mode = resolve_spectral_entrywise_delta_mode(spectral_entrywise_delta_mode)
    emitted_features = expand_spectral_feature_names(
        selected_features=selected_features,
        spectral_moment_source=resolved_moment_source,
    )
    schema, skipped_models, skipped_adapter_paths = _resolve_readable_schema(items)

    expected_pairs = [tuple(x) for x in schema.pairs]
    expected_a_shapes = [tuple(x) for x in schema.a_shapes]
    expected_b_shapes = [tuple(x) for x in schema.b_shapes]
    expected_e_keys = [str(x) if x is not None else None for x in schema.e_keys]
    expected_e_shapes = [
        tuple(int(y) for y in x) if x is not None else None
        for x in schema.e_shapes
    ]
    allow_rank_variation = bool(schema_has_adalora_scaling(schema))

    include_base_blocks = resolved_qv_sum_mode in {"none", "append"}
    include_qv_sum_blocks = resolved_qv_sum_mode in {"append", "only"}

    qv_specs: list[QvSumSpec] = []
    qv_pair_lookup: dict[int, tuple[str, str]] = {}
    if include_qv_sum_blocks:
        qv_specs = build_qv_sum_specs(schema)
        if not qv_specs:
            raise ValueError(
                "spectral_qv_sum_mode requested q+v features, but no supported attention q/v layer pairs were found"
            )
        for spec in qv_specs:
            qv_pair_lookup[spec.q_index] = (spec.qv_block_name_raw, "q")
            qv_pair_lookup[spec.v_index] = (spec.qv_block_name_raw, "v")

    rows: list[list[float]] = []
    model_names: list[str] = []
    labels_list: list[int | None] = []
    need_entrywise_moments = _needs_entrywise_moments(emitted_features)
    need_sv_moments = _needs_sv_moments(emitted_features)
    need_spectral_scalars = _needs_spectral_scalars(emitted_features)
    need_top_k = _needs_sv_topk(emitted_features)
    need_singular_values = need_spectral_scalars or need_top_k or need_sv_moments
    entrywise_dense_block_count = 0
    entrywise_stream_block_count = 0

    for item in items:
        if str(item.adapter_path) in skipped_adapter_paths:
            continue

        row: list[float] = []
        qv_factors: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        try:
            with safe_open(item.adapter_path, framework="numpy") as reader:
                check_consistency_reader(
                    reader=reader,
                    adapter_path=item.adapter_path,
                    expected_pairs=expected_pairs,
                    expected_a_shapes=expected_a_shapes,
                    expected_b_shapes=expected_b_shapes,
                    expected_e_keys=expected_e_keys,
                    expected_e_shapes=expected_e_shapes,
                    allow_rank_variation=allow_rank_variation,
                )
                for pair_idx, (_, a, b) in enumerate(iter_block_factors(reader=reader, schema=schema, dtype=dtype)):
                    if include_base_blocks:
                        singular_values = (
                            block_delta_singular_values(a=a, b=b) if need_singular_values else None
                        )

                        entrywise_moments: dict[str, float] | None = None
                        if need_entrywise_moments:
                            summary, runtime_entrywise_mode = block_moments_from_factors(
                                a=a,
                                b=b,
                                block_size=block_size,
                                dtype=dtype,
                                entrywise_delta_mode=resolved_entrywise_delta_mode,
                            )
                            if runtime_entrywise_mode == "dense":
                                entrywise_dense_block_count += 1
                            else:
                                entrywise_stream_block_count += 1
                            entrywise_moments = {
                                "kurtosis": float(summary.kurtosis),
                                "l1_norm": float(summary.l1_norm),
                                "linf_norm": float(summary.linf_norm),
                                "mean_abs": float(summary.mean_abs),
                            }

                        sv_moments: dict[str, float] | None = None
                        if need_sv_moments:
                            if singular_values is None:
                                raise RuntimeError(
                                    "Singular-value moments were requested but singular values were not computed"
                                )
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
                for spec in qv_specs:
                    qv_block_name_raw = spec.qv_block_name_raw
                    pair = qv_factors.get(qv_block_name_raw)
                    if pair is None or "q" not in pair or "v" not in pair:
                        raise RuntimeError(
                            f"Missing q/v factors for {shorten_block_name(qv_block_name_raw)} in {item.adapter_path}"
                        )

                    a_q, b_q = pair["q"]
                    a_v, b_v = pair["v"]
                    a_qv = np.concatenate([a_q, a_v], axis=0)
                    b_qv = np.concatenate([b_q, b_v], axis=1)
                    singular_values = (
                        block_delta_singular_values(a=a_qv, b=b_qv) if need_singular_values else None
                    )

                    entrywise_moments: dict[str, float] | None = None
                    if need_entrywise_moments:
                        summary, runtime_entrywise_mode = block_moments_from_factors(
                            a=a_qv,
                            b=b_qv,
                            block_size=block_size,
                            dtype=dtype,
                            entrywise_delta_mode=resolved_entrywise_delta_mode,
                        )
                        if runtime_entrywise_mode == "dense":
                            entrywise_dense_block_count += 1
                        else:
                            entrywise_stream_block_count += 1
                        entrywise_moments = {
                            "kurtosis": float(summary.kurtosis),
                            "l1_norm": float(summary.l1_norm),
                            "linf_norm": float(summary.linf_norm),
                            "mean_abs": float(summary.mean_abs),
                        }

                    sv_moments: dict[str, float] | None = None
                    if need_sv_moments:
                        if singular_values is None:
                            raise RuntimeError(
                                "Singular-value moments were requested but singular values were not computed"
                            )
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
        except _SKIPPABLE_SPECTRAL_ITEM_EXCEPTIONS as exc:
            skipped_models.append(_skip_entry_for_item(item, exc))
            continue

        rows.append(row)
        model_names.append(item.model_name)
        labels_list.append(item.label)
    labels = np.asarray(labels_list, dtype=np.int32) if all(label is not None for label in labels_list) else None

    feature_block_names_raw: list[str] = []
    if include_base_blocks:
        feature_block_names_raw.extend(str(x) for x in schema.block_names)
    if include_qv_sum_blocks:
        feature_block_names_raw.extend(spec.qv_block_name_raw for spec in qv_specs)

    feature_names = build_spectral_feature_names(
        block_names=feature_block_names_raw,
        selected_features=selected_features,
        sv_top_k=sv_top_k,
        spectral_moment_source=resolved_moment_source,
    )
    if rows:
        features = np.asarray(rows, dtype=np.float32)
    else:
        features = np.zeros((0, len(feature_names)), dtype=np.float32)
    if features.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Spectral feature dimension mismatch: matrix has {features.shape[1]} columns, "
            f"but generated {len(feature_names)} feature names"
        )

    schema_metadata = build_schema_metadata(schema)
    variable_lora_rank = bool(schema_metadata.get("variable_lora_rank"))
    qv_sum_block_names = [shorten_block_name(spec.qv_block_name_raw) for spec in qv_specs]
    base_block_names = [str(x) for x in schema_metadata.get("block_names", [])]
    feature_block_names = [shorten_block_name(name) for name in feature_block_names_raw]
    base_lora_adapter_dims = [dict(x) for x in schema_metadata.get("lora_adapter_dims", [])]
    base_lora_adapter_dims_for_summary = (
        list(base_lora_adapter_dims)
        if base_lora_adapter_dims
        else [
            lora_adapter_dims_from_shapes(
                tuple(int(x) for x in a_shape),
                tuple(int(x) for x in b_shape),
            )
            for a_shape, b_shape in zip(schema.a_shapes, schema.b_shapes)
        ]
    )
    qv_sum_lora_adapter_dims = (
        []
        if variable_lora_rank
        else [
            lora_adapter_dims_from_shapes(
                (
                    int(schema.a_shapes[spec.q_index][0]) + int(schema.a_shapes[spec.v_index][0]),
                    int(schema.a_shapes[spec.q_index][1]),
                ),
                (
                    int(schema.b_shapes[spec.q_index][0]),
                    int(schema.b_shapes[spec.q_index][1]) + int(schema.b_shapes[spec.v_index][1]),
                ),
            )
            for spec in qv_specs
        ]
    )
    feature_lora_adapter_dims: list[dict[str, int]] = []
    if include_base_blocks:
        feature_lora_adapter_dims.extend(base_lora_adapter_dims)
    if include_qv_sum_blocks:
        feature_lora_adapter_dims.extend(qv_sum_lora_adapter_dims)
    schema_layout_summary = summarize_schema_layout(
        base_block_names=base_block_names,
        base_lora_adapter_dims=base_lora_adapter_dims_for_summary,
        variable_lora_rank=variable_lora_rank,
    )

    metadata: dict[str, Any] = {
        "extractor": "spectral",
        "extractor_version": SPECTRAL_EXTRACTOR_VERSION,
        "delta_schema_version": DELTA_SCHEMA_VERSION,
        "input_n_models": int(len(items)),
        "n_models": int(len(model_names)),
        "resolved_features": selected_features,
        "spectral_qv_sum_mode": resolved_qv_sum_mode,
        "spectral_moment_source": resolved_moment_source,
        "spectral_entrywise_delta_mode": resolved_entrywise_delta_mode,
        "entrywise_dense_block_count": int(entrywise_dense_block_count),
        "entrywise_stream_block_count": int(entrywise_stream_block_count),
        "sv_top_k": int(sv_top_k),
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names,
        **schema_metadata,
        "schema_layout_summary": schema_layout_summary,
        "base_block_names": base_block_names,
        "qv_sum_block_names": qv_sum_block_names,
        "block_names": feature_block_names,
        "n_blocks": int(len(feature_block_names)),
    }
    if skipped_models:
        metadata["skipped_model_count"] = int(len(skipped_models))
        metadata["skipped_models"] = skipped_models
    if not variable_lora_rank:
        metadata["base_lora_adapter_dims"] = base_lora_adapter_dims
        metadata["qv_sum_lora_adapter_dims"] = qv_sum_lora_adapter_dims
        metadata["lora_adapter_dims"] = feature_lora_adapter_dims
    return features, labels, model_names, sanitize_spectral_metadata(metadata)
