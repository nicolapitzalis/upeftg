"""Spectral block metadata normalization and schema summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts.spectral import feature_block_name, layer_identifier_for_block_name
from ..delta import lora_adapter_dims_from_shapes
from .config import _SPECTRAL_METADATA_DROP_KEYS, spectral_extractor_params


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
        "layer_count": int(len({layer_identifier_for_block_name(name) for name in base_block_names})),
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
        q_dims = dim_map.get(prefix + ".q_proj") or dim_map.get(prefix + ".q") or dim_map.get(prefix + ".query")
        v_dims = dim_map.get(prefix + ".v_proj") or dim_map.get(prefix + ".v") or dim_map.get(prefix + ".value")
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
        if "spectral_entrywise_delta_mode" not in cleaned and "spectral_entrywise_delta_mode" in extractor_params:
            cleaned["spectral_entrywise_delta_mode"] = str(extractor_params["spectral_entrywise_delta_mode"])
        if "spectral_attention_granularity" not in cleaned and "spectral_attention_granularity" in extractor_params:
            cleaned["spectral_attention_granularity"] = str(extractor_params["spectral_attention_granularity"])

    if block_names:
        dim_map = spectral_block_lora_dims_by_block(metadata)
        if all(block_name in dim_map for block_name in block_names):
            cleaned["lora_adapter_dims"] = [dim_map[block_name] for block_name in block_names]
        else:
            cleaned.pop("lora_adapter_dims", None)

    return cleaned
