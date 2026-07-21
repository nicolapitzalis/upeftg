"""Pure feature aggregation layout and naming calculations."""

from __future__ import annotations

import re

import numpy as np

from ...contracts.spectral import feature_block_name, layer_identifier_for_block_name
from ...features.spectral import (
    build_spectral_feature_names,
    resolve_spectral_qv_sum_mode,
)
from ..schema import feature_group_for_feature_name, resolve_output_feature_names


ROLE_BUCKET_ORDER = ("q", "v", "qv_sum", "other")
SUPPORTED_AGGREGATION_OPERATORS = ("avg", "max", "min")
SUPPORTED_AGGREGATION_LAYOUTS = ("flat", "layer_sequence")
GROUP_MASK_SUFFIX = "_group_mask.npy"
VALUE_MASK_SUFFIX = "_value_mask.npy"
GROUP_NAMES_SUFFIX = "_group_names.json"
ARCHITECTURE_BLOCK_ORDER = ("encoder", "decoder")
ATTENTION_KIND_ORDER = ("self", "cross", "other")
ADAPTER_BLOCK_ORDER = ("q", "v", "qv_sum", "other")

_Q_ROLE_ALIASES = {"q", "q_proj", "query"}
_V_ROLE_ALIASES = {"v", "v_proj", "value"}
_HEAD_SUFFIX_RE = re.compile(r"^head(\d+)$", re.IGNORECASE)


def _normalize_aggregation_operator(operator: str) -> str:
    resolved = str(operator).strip().lower()
    if resolved not in SUPPORTED_AGGREGATION_OPERATORS:
        raise ValueError(
            f"Unsupported aggregation operator '{operator}'. Supported: {list(SUPPORTED_AGGREGATION_OPERATORS)}"
        )
    return resolved


def _normalize_aggregation_layout(layout: str) -> str:
    resolved = str(layout).strip().lower()
    if resolved not in SUPPORTED_AGGREGATION_LAYOUTS:
        raise ValueError(f"Unsupported aggregation layout '{layout}'. Supported: {list(SUPPORTED_AGGREGATION_LAYOUTS)}")
    return resolved


def _emitted_feature_name(feature_name: str) -> str:
    return str(feature_name).rpartition(".")[2]


def _split_head_suffix(block_name: str) -> tuple[str, str | None, int | None]:
    text = str(block_name).strip()
    base, sep, suffix = text.rpartition(".")
    if not sep:
        return text, None, None
    matched = _HEAD_SUFFIX_RE.match(suffix.strip())
    if matched is None:
        return text, None, None
    return base, suffix.strip().lower(), int(matched.group(1))


def _role_bucket_for_block_name(block_name: str) -> str:
    text, _head_label, _head_idx = _split_head_suffix(block_name)
    if text.endswith(".qv_sum"):
        return "qv_sum"
    module = text.rsplit(".", 1)[-1].strip().lower()
    if module in _Q_ROLE_ALIASES:
        return "q"
    if module in _V_ROLE_ALIASES:
        return "v"
    return "other"


def _role_bucket_for_feature_name(feature_name: str) -> str:
    return _role_bucket_for_block_name(feature_block_name(feature_name))


def _indexed_part_span(parts: list[str], *, prefix: str) -> tuple[int, int] | None:
    prefix_text = str(prefix).strip().lower()
    for idx, raw_part in enumerate(parts):
        part = str(raw_part).strip().lower()
        if part.startswith(prefix_text) and part[len(prefix_text) :].isdigit():
            return idx, idx + 1
        if part == prefix_text and idx + 1 < len(parts) and str(parts[idx + 1]).strip().isdigit():
            return idx, idx + 2
    return None


def _structural_group_for_block_name(block_name: str) -> str:
    parts = [part for part in str(block_name).split(".") if part]
    if not parts:
        return str(block_name)

    block_span = _indexed_part_span(parts, prefix="block")
    if block_span is not None:
        _, end_idx = block_span
        return ".".join(parts[:end_idx]).strip() or str(block_name)

    layer_span = _indexed_part_span(parts, prefix="layer")
    if layer_span is not None:
        _, end_idx = layer_span
        return ".".join(parts[:end_idx]).strip() or str(block_name)

    return layer_identifier_for_block_name(str(block_name))


def _structural_group_for_feature_name(feature_name: str) -> str:
    return _structural_group_for_block_name(feature_block_name(feature_name))


def _relative_block_name_for_block_name(block_name: str) -> str:
    resolved_block_name = str(block_name)
    structural_group = _structural_group_for_block_name(resolved_block_name)
    prefix = f"{structural_group}."
    if resolved_block_name.startswith(prefix):
        relative = resolved_block_name[len(prefix) :].strip()
        if relative:
            return relative
    return "__self__"


def _relative_block_name_for_feature_name(feature_name: str) -> str:
    return _relative_block_name_for_block_name(feature_block_name(feature_name))


def _filter_selected_input_feature_names(
    *,
    root_feature_names: list[str],
    requested_features: list[str] | None,
    spectral_qv_sum_mode: str,
) -> list[str]:
    resolved_qv_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    available_feature_names = list(root_feature_names)
    if resolved_qv_mode == "none":
        available_feature_names = [
            name for name in available_feature_names if _role_bucket_for_feature_name(name) != "qv_sum"
        ]
    elif resolved_qv_mode == "only":
        available_feature_names = [
            name for name in available_feature_names if _role_bucket_for_feature_name(name) == "qv_sum"
        ]

    if not available_feature_names:
        raise ValueError(
            "Input feature bundle does not contain any columns compatible with "
            f"--spectral-qv-sum-mode={resolved_qv_mode}"
        )

    selected_feature_names = resolve_output_feature_names(
        available_feature_names=available_feature_names,
        requested_features=requested_features,
    )
    if not selected_feature_names:
        raise ValueError("Requested aggregation resolved to zero input columns")
    return selected_feature_names


def _ordered_feature_groups(feature_names: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        feature_group = feature_group_for_feature_name(feature_name)
        if feature_group is None or feature_group in seen:
            continue
        seen.add(feature_group)
        ordered.append(feature_group)
    return ordered


def _ordered_structural_groups(feature_names: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        structural_group = _structural_group_for_feature_name(feature_name)
        if structural_group in seen:
            continue
        seen.add(structural_group)
        ordered.append(structural_group)
    return ordered


def _ordered_relative_block_names(feature_names: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        relative_block = _relative_block_name_for_feature_name(feature_name)
        if relative_block in seen:
            continue
        seen.add(relative_block)
        ordered.append(relative_block)
    return ordered


def _architecture_block_for_block_name(block_name: str) -> str:
    parts = [str(part).strip().lower() for part in str(block_name).split(".") if str(part).strip()]
    if "encoder" in parts:
        return "encoder"
    if "decoder" in parts:
        return "decoder"
    if "roberta" in parts:
        return "encoder"
    return "decoder"


def _indexed_value_for_parts(parts: list[str], *, prefixes: tuple[str, ...]) -> int | None:
    lowered_prefixes = tuple(str(prefix).strip().lower() for prefix in prefixes)
    for idx, raw_part in enumerate(parts):
        part = str(raw_part).strip().lower()
        for prefix in lowered_prefixes:
            if part.startswith(prefix) and part[len(prefix) :].isdigit():
                return int(part[len(prefix) :])
            if part == prefix and idx + 1 < len(parts) and str(parts[idx + 1]).strip().isdigit():
                return int(str(parts[idx + 1]).strip())
    return None


def _layer_index_for_block_name(block_name: str) -> int:
    parts = [part for part in str(block_name).split(".") if part]
    lowered = [str(part).strip().lower() for part in parts]
    if "encoder" in lowered or "decoder" in lowered:
        block_idx = _indexed_value_for_parts(parts, prefixes=("block",))
        if block_idx is not None:
            return int(block_idx)
    layer_idx = _indexed_value_for_parts(parts, prefixes=("layer", "layers"))
    if layer_idx is not None:
        return int(layer_idx)
    block_idx = _indexed_value_for_parts(parts, prefixes=("block",))
    if block_idx is not None:
        return int(block_idx)
    raise ValueError(f"Could not determine canonical layer index for block name: {block_name}")


def _attention_kind_for_block_name(block_name: str) -> str:
    lowered = str(block_name).strip().lower()
    if "encdecattention" in lowered or "crossattention" in lowered or ".cross." in lowered:
        return "cross"
    if (
        "selfattention" in lowered
        or "self_attn" in lowered
        or "self_attention" in lowered
        or ".attention.self." in lowered
        or ".self." in lowered
    ):
        return "self"
    return "other"


def _adapter_block_for_block_name(block_name: str) -> str:
    text, head_label, _head_idx = _split_head_suffix(block_name)
    if text.endswith(".qv_sum"):
        adapter_block = "qv_sum"
        return adapter_block if head_label is None else f"{adapter_block}.{head_label}"
    module = text.rsplit(".", 1)[-1].strip().lower()
    if module in _Q_ROLE_ALIASES:
        adapter_block = "q"
        return adapter_block if head_label is None else f"{adapter_block}.{head_label}"
    if module in _V_ROLE_ALIASES:
        adapter_block = "v"
        return adapter_block if head_label is None else f"{adapter_block}.{head_label}"
    adapter_block = "other"
    return adapter_block if head_label is None else f"{adapter_block}.{head_label}"


def _canonical_depth_tuple_for_block_name(block_name: str) -> tuple[str, int]:
    return (
        _architecture_block_for_block_name(block_name),
        int(_layer_index_for_block_name(block_name)),
    )


def _canonical_depth_label_for_block_name(block_name: str) -> str:
    architecture_block, layer_idx = _canonical_depth_tuple_for_block_name(block_name)
    return f"{architecture_block}.layer{layer_idx}"


def _canonical_depth_label_for_feature_name(feature_name: str) -> str:
    return _canonical_depth_label_for_block_name(feature_block_name(feature_name))


def _canonical_slot_tuple_for_block_name(block_name: str) -> tuple[str, str]:
    return (
        _attention_kind_for_block_name(block_name),
        _adapter_block_for_block_name(block_name),
    )


def _canonical_slot_name_for_block_name(block_name: str) -> str:
    attention_kind, adapter_block = _canonical_slot_tuple_for_block_name(block_name)
    return f"{attention_kind}.{adapter_block}"


def _canonical_slot_name_for_feature_name(feature_name: str) -> str:
    return _canonical_slot_name_for_block_name(feature_block_name(feature_name))


def _architecture_block_sort_key(architecture_block: str) -> tuple[int, str]:
    try:
        return (ARCHITECTURE_BLOCK_ORDER.index(str(architecture_block)), "")
    except ValueError:
        return (len(ARCHITECTURE_BLOCK_ORDER), str(architecture_block))


def _attention_kind_sort_key(attention_kind: str) -> tuple[int, str]:
    try:
        return (ATTENTION_KIND_ORDER.index(str(attention_kind)), "")
    except ValueError:
        return (len(ATTENTION_KIND_ORDER), str(attention_kind))


def _adapter_block_sort_key(adapter_block: str) -> tuple[int, int, int, str]:
    adapter_text = str(adapter_block).strip().lower()
    role_text, sep, head_label = adapter_text.partition(".")
    matched = _HEAD_SUFFIX_RE.match(head_label) if sep else None
    role_for_order = role_text if matched is not None else adapter_text
    try:
        role_order = ADAPTER_BLOCK_ORDER.index(role_for_order)
    except ValueError:
        role_order = len(ADAPTER_BLOCK_ORDER)
    if matched is None:
        return (role_order, 0, -1, adapter_text)
    return (role_order, 1, int(matched.group(1)), adapter_text)


def _ordered_canonical_depth_labels(feature_names: list[str]) -> list[str]:
    unique_depths = {
        _canonical_depth_tuple_for_block_name(feature_block_name(feature_name)) for feature_name in feature_names
    }
    return [
        f"{architecture_block}.layer{layer_idx}"
        for architecture_block, layer_idx in sorted(
            unique_depths,
            key=lambda item: (_architecture_block_sort_key(item[0]), int(item[1])),
        )
    ]


def _ordered_canonical_slot_names(feature_names: list[str]) -> list[str]:
    unique_slots = {
        _canonical_slot_tuple_for_block_name(feature_block_name(feature_name)) for feature_name in feature_names
    }
    return [
        f"{attention_kind}.{adapter_block}"
        for attention_kind, adapter_block in sorted(
            unique_slots,
            key=lambda item: (
                _attention_kind_sort_key(item[0]),
                _adapter_block_sort_key(item[1]),
            ),
        )
    ]


def _resolved_output_roles(feature_names: list[str]) -> list[str]:
    present_roles = {_role_bucket_for_feature_name(feature_name) for feature_name in feature_names}
    ordered = [role for role in ROLE_BUCKET_ORDER if role in present_roles]
    if not ordered:
        raise ValueError("Requested aggregation resolved to zero role buckets")
    return ordered


def _sv_top_k_from_feature_names(feature_names: list[str], *, default: int) -> int:
    max_rank = 0
    for feature_name in feature_names:
        emitted = _emitted_feature_name(feature_name)
        if not emitted.startswith("sv_"):
            continue
        suffix = emitted[3:]
        if suffix.isdigit():
            max_rank = max(max_rank, int(suffix))
    return max_rank if max_rank > 0 else int(default)


def _aggregation_value(
    grouped_values: dict[str, np.ndarray],
    *,
    operator: str,
) -> float:
    if operator == "avg":
        # Give each higher-level structural group one vote so architectures with
        # more repeated substructure do not dominate the average by column count.
        group_means = [
            float(np.mean(np.asarray(values, dtype=np.float32), dtype=np.float64)) for values in grouped_values.values()
        ]
        return float(np.mean(np.asarray(group_means, dtype=np.float64), dtype=np.float64))

    all_values = np.concatenate(
        [np.asarray(values, dtype=np.float32).reshape(-1) for values in grouped_values.values()],
        axis=0,
    )
    if operator == "min":
        return float(np.min(all_values))
    if operator == "max":
        return float(np.max(all_values))
    raise ValueError(f"Unsupported aggregation operator '{operator}'")


def _build_output_feature_names(
    *,
    output_block_names: list[str],
    selected_feature_groups: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
) -> list[str]:
    return build_spectral_feature_names(
        block_names=output_block_names,
        selected_features=selected_feature_groups,
        sv_top_k=int(sv_top_k),
        spectral_moment_source=spectral_moment_source,
        shorten_block_names=False,
    )


def _build_output_emitted_feature_names(
    *,
    selected_feature_groups: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
) -> list[str]:
    prefixed_feature_names = build_spectral_feature_names(
        block_names=["__emitted__"],
        selected_features=selected_feature_groups,
        sv_top_k=int(sv_top_k),
        spectral_moment_source=spectral_moment_source,
        shorten_block_names=False,
    )
    prefix = "__emitted__."
    return [name[len(prefix) :] for name in prefixed_feature_names]


def _aggregation_value_for_single_group(values: np.ndarray, *, operator: str) -> float:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    if flattened.size == 0:
        return 0.0
    if operator == "avg":
        return float(np.mean(flattened, dtype=np.float64))
    if operator == "min":
        return float(np.min(flattened))
    if operator == "max":
        return float(np.max(flattened))
    raise ValueError(f"Unsupported aggregation operator '{operator}'")
