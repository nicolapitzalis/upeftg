"""Spectral feature naming and attention/QV block layouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..delta import DeltaBlockSchema, shorten_block_name
from .config import (
    _DERIVED_FEATURE_GROUP_SOURCES,
    _FEATURE_GROUP_BY_SUFFIX,
    _QV_ROLE_ALIASES,
    _RANK_NORMALIZED_FEATURE_REPLACEMENTS,
    expand_spectral_feature_names,
)


@dataclass(frozen=True)
class QvSumSpec:
    qv_block_name_raw: str
    q_index: int
    v_index: int
    q_block_name_raw: str
    v_block_name_raw: str


@dataclass(frozen=True)
class AttentionHeadLayout:
    n_heads: int
    head_dim: int


def feature_group_for_spectral_feature_name(feature_name: str) -> str | None:
    suffix = str(feature_name).rpartition(".")[2]
    if not suffix:
        return None
    if suffix.startswith("sv_") and suffix[3:].isdigit():
        return "sv_topk"
    return _FEATURE_GROUP_BY_SUFFIX.get(suffix)


def rank_normalized_feature_group(feature_group: str) -> str:
    resolved = str(feature_group).strip()
    return _RANK_NORMALIZED_FEATURE_REPLACEMENTS.get(resolved, resolved)


def provenance_source_feature_group(feature_group: str) -> str | None:
    resolved = str(feature_group).strip()
    if resolved == "block_rank":
        return None
    return _DERIVED_FEATURE_GROUP_SOURCES.get(resolved, resolved)


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


def _attention_role_for_block_name(block_name: str) -> str | None:
    parsed = _qv_sum_group_for_block_name(block_name)
    if parsed is None:
        return None
    return parsed[1]


def _expected_head_dim_for_block_name(block_name: str) -> int | None:
    parts = [part for part in str(block_name).split(".") if part]
    if not parts:
        return None
    module = parts[-1].strip().lower()
    lowered = str(block_name).strip().lower()

    # Architectures currently used by the list2 manifest family:
    # LLaMA/Qwen-style q_proj/v_proj split hidden states into 128-wide heads.
    if module in {"q_proj", "v_proj"} and ("self_attn" in lowered or "attention" in lowered or "attn" in lowered):
        return 128

    # T5 uses q/v module names in SelfAttention and EncDecAttention blocks.
    if module in {"q", "v"} and ("selfattention" in lowered or "encdecattention" in lowered):
        return 64

    # RoBERTa/BERT-style attention projections use query/value names and 64-wide heads.
    if module in {"query", "value"} and (".attention.self." in lowered or "roberta" in lowered or "bert" in lowered):
        return 64

    return None


def _infer_attention_head_layout(
    *,
    block_name: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
) -> AttentionHeadLayout:
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(
            f"Cannot split {shorten_block_name(block_name)} by attention head: "
            f"expected rank-2 LoRA factors, got A{a_shape}, B{b_shape}"
        )
    if int(a_shape[0]) != int(b_shape[1]):
        raise ValueError(
            f"Cannot split {shorten_block_name(block_name)} by attention head: "
            f"LoRA rank mismatch A{a_shape}, B{b_shape}"
        )

    head_dim = _expected_head_dim_for_block_name(block_name)
    if head_dim is None:
        raise ValueError(
            "Cannot infer attention head layout for "
            f"{shorten_block_name(block_name)}. Head granularity currently supports "
            "LLaMA/Qwen q_proj/v_proj blocks, T5 q/v blocks, and RoBERTa query/value blocks."
        )

    out_dim = int(b_shape[0])
    if out_dim <= 0 or out_dim % int(head_dim) != 0:
        raise ValueError(
            f"Cannot split {shorten_block_name(block_name)} into {head_dim}-wide heads: B output dimension is {out_dim}"
        )

    return AttentionHeadLayout(n_heads=int(out_dim // int(head_dim)), head_dim=int(head_dim))


def _head_suffix_width(layouts: list[AttentionHeadLayout]) -> int:
    max_head_index = max((int(layout.n_heads) - 1 for layout in layouts), default=0)
    return max(2, len(str(max(0, max_head_index))))


def _head_block_name(block_name: str, head_idx: int, *, width: int) -> str:
    return f"{block_name}.head{int(head_idx):0{int(width)}d}"


def _slice_head_b(b: np.ndarray, *, layout: AttentionHeadLayout, head_idx: int) -> np.ndarray:
    start = int(head_idx) * int(layout.head_dim)
    end = start + int(layout.head_dim)
    return b[start:end, :]


def _effective_delta_rank_scale(a: np.ndarray, b: np.ndarray) -> int:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected rank-2 factors for rank scale, got A{a.shape}, B{b.shape}")
    return max(1, min(int(a.shape[0]), int(a.shape[1]), int(b.shape[0])))


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
            raise ValueError(f"Duplicate '{module}' block while building q+v sum for {qv_block_name_raw}")
        groups[qv_block_name_raw][module] = int(pair_idx)

    missing_roles: list[str] = []
    specs: list[QvSumSpec] = []
    for qv_block_name_raw in order:
        roles = groups[qv_block_name_raw]
        if "q" not in roles or "v" not in roles:
            missing = [name for name in ("q", "v") if name not in roles]
            missing_roles.append(f"{shorten_block_name(qv_block_name_raw)} missing {', '.join(missing)}")
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
