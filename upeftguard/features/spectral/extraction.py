"""Spectral extraction orchestration across manifest items."""

from __future__ import annotations

from typing import Any

import numpy as np
from safetensors import safe_open

from ..delta import (
    DELTA_SCHEMA_VERSION,
    DeltaBlockSchema,
    build_schema_metadata,
    check_consistency_reader,
    iter_block_factors,
    lora_adapter_dims_from_shapes,
    load_delta_block_schema,
    schema_has_adalora_scaling,
    shorten_block_name,
)
from ...utilities.core.manifest import ManifestItem
from .computation import (
    _append_factor_features,
    _needs_entrywise_moments,
    _needs_spectral_scalars,
    _needs_sv_moments,
    _needs_sv_topk,
)
from .config import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    SPECTRAL_EXTRACTOR_VERSION,
    _SKIPPABLE_SPECTRAL_ITEM_EXCEPTIONS,
    expand_spectral_feature_names,
    resolve_spectral_attention_granularity,
    resolve_spectral_entrywise_delta_mode,
    resolve_spectral_features,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
)
from .layout import (
    AttentionHeadLayout,
    QvSumSpec,
    _attention_role_for_block_name,
    _head_block_name,
    _head_suffix_width,
    _infer_attention_head_layout,
    _slice_head_b,
    build_qv_sum_specs,
    build_spectral_feature_names,
)
from .metadata import sanitize_spectral_metadata, summarize_schema_layout


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
        "No readable adapters were available for spectral extraction" + (f". Examples: {preview}" if preview else "")
    )


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
    spectral_attention_granularity: str = DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
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
    resolved_attention_granularity = resolve_spectral_attention_granularity(spectral_attention_granularity)
    emitted_features = expand_spectral_feature_names(
        selected_features=selected_features,
        spectral_moment_source=resolved_moment_source,
    )
    schema, skipped_models, skipped_adapter_paths = _resolve_readable_schema(items)

    expected_pairs = [tuple(x) for x in schema.pairs]
    expected_a_shapes = [tuple(x) for x in schema.a_shapes]
    expected_b_shapes = [tuple(x) for x in schema.b_shapes]
    expected_e_keys = [str(x) if x is not None else None for x in schema.e_keys]
    expected_e_shapes = [tuple(int(y) for y in x) if x is not None else None for x in schema.e_shapes]
    allow_rank_variation = bool(schema_has_adalora_scaling(schema))

    include_base_blocks = resolved_qv_sum_mode in {"none", "append"}
    include_qv_sum_blocks = resolved_qv_sum_mode in {"append", "only"}
    use_head_blocks = resolved_attention_granularity == "head"

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

    head_layout_by_pair: dict[int, AttentionHeadLayout] = {}
    if use_head_blocks:
        layout_pair_indices: set[int] = set(range(len(schema.block_names))) if include_base_blocks else set()
        if include_qv_sum_blocks:
            for spec in qv_specs:
                layout_pair_indices.add(int(spec.q_index))
                layout_pair_indices.add(int(spec.v_index))

        for pair_idx in sorted(layout_pair_indices):
            block_name = str(schema.block_names[pair_idx])
            if _attention_role_for_block_name(block_name) not in {"q", "v"}:
                raise ValueError(
                    "--spectral-attention-granularity=head requires q/v attention LoRA blocks; "
                    f"got {shorten_block_name(block_name)}"
                )
            head_layout_by_pair[pair_idx] = _infer_attention_head_layout(
                block_name=block_name,
                a_shape=tuple(int(x) for x in schema.a_shapes[pair_idx]),
                b_shape=tuple(int(x) for x in schema.b_shapes[pair_idx]),
            )

        for spec in qv_specs:
            q_layout = head_layout_by_pair[int(spec.q_index)]
            v_layout = head_layout_by_pair[int(spec.v_index)]
            if q_layout != v_layout:
                raise ValueError(
                    "Cannot build head-wise q+v sum for "
                    f"{shorten_block_name(spec.qv_block_name_raw)}: q layout "
                    f"{q_layout.n_heads}x{q_layout.head_dim} differs from v layout "
                    f"{v_layout.n_heads}x{v_layout.head_dim}"
                )

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
                        if use_head_blocks:
                            layout = head_layout_by_pair[int(pair_idx)]
                            a_qr_r = None
                            if need_singular_values:
                                _, a_qr_r = np.linalg.qr(a.T, mode="reduced")
                            for head_idx in range(int(layout.n_heads)):
                                dense_count, stream_count = _append_factor_features(
                                    row=row,
                                    emitted_features=emitted_features,
                                    a=a,
                                    b=_slice_head_b(b, layout=layout, head_idx=head_idx),
                                    a_qr_r=a_qr_r,
                                    sv_top_k=sv_top_k,
                                    block_size=block_size,
                                    dtype=dtype,
                                    spectral_entrywise_delta_mode=resolved_entrywise_delta_mode,
                                    need_spectral_scalars=need_spectral_scalars,
                                    need_sv_topk=need_top_k,
                                    need_singular_values=need_singular_values,
                                    need_entrywise_moments=need_entrywise_moments,
                                    need_sv_moments=need_sv_moments,
                                )
                                entrywise_dense_block_count += int(dense_count)
                                entrywise_stream_block_count += int(stream_count)
                        else:
                            dense_count, stream_count = _append_factor_features(
                                row=row,
                                emitted_features=emitted_features,
                                a=a,
                                b=b,
                                sv_top_k=sv_top_k,
                                block_size=block_size,
                                dtype=dtype,
                                spectral_entrywise_delta_mode=resolved_entrywise_delta_mode,
                                need_spectral_scalars=need_spectral_scalars,
                                need_sv_topk=need_top_k,
                                need_singular_values=need_singular_values,
                                need_entrywise_moments=need_entrywise_moments,
                                need_sv_moments=need_sv_moments,
                            )
                            entrywise_dense_block_count += int(dense_count)
                            entrywise_stream_block_count += int(stream_count)

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
                    if use_head_blocks:
                        layout = head_layout_by_pair[int(spec.q_index)]
                        a_qv_qr_r = None
                        if need_singular_values:
                            _, a_qv_qr_r = np.linalg.qr(a_qv.T, mode="reduced")
                        for head_idx in range(int(layout.n_heads)):
                            b_qv = np.concatenate(
                                [
                                    _slice_head_b(b_q, layout=layout, head_idx=head_idx),
                                    _slice_head_b(b_v, layout=layout, head_idx=head_idx),
                                ],
                                axis=1,
                            )
                            dense_count, stream_count = _append_factor_features(
                                row=row,
                                emitted_features=emitted_features,
                                a=a_qv,
                                b=b_qv,
                                a_qr_r=a_qv_qr_r,
                                sv_top_k=sv_top_k,
                                block_size=block_size,
                                dtype=dtype,
                                spectral_entrywise_delta_mode=resolved_entrywise_delta_mode,
                                need_spectral_scalars=need_spectral_scalars,
                                need_sv_topk=need_top_k,
                                need_singular_values=need_singular_values,
                                need_entrywise_moments=need_entrywise_moments,
                                need_sv_moments=need_sv_moments,
                            )
                            entrywise_dense_block_count += int(dense_count)
                            entrywise_stream_block_count += int(stream_count)
                    else:
                        b_qv = np.concatenate([b_q, b_v], axis=1)
                        dense_count, stream_count = _append_factor_features(
                            row=row,
                            emitted_features=emitted_features,
                            a=a_qv,
                            b=b_qv,
                            sv_top_k=sv_top_k,
                            block_size=block_size,
                            dtype=dtype,
                            spectral_entrywise_delta_mode=resolved_entrywise_delta_mode,
                            need_spectral_scalars=need_spectral_scalars,
                            need_sv_topk=need_top_k,
                            need_singular_values=need_singular_values,
                            need_entrywise_moments=need_entrywise_moments,
                            need_sv_moments=need_sv_moments,
                        )
                        entrywise_dense_block_count += int(dense_count)
                        entrywise_stream_block_count += int(stream_count)
        except _SKIPPABLE_SPECTRAL_ITEM_EXCEPTIONS as exc:
            skipped_models.append(_skip_entry_for_item(item, exc))
            continue

        rows.append(row)
        model_names.append(item.model_name)
        labels_list.append(item.label)
    labels = np.asarray(labels_list, dtype=np.int32) if all(label is not None for label in labels_list) else None

    all_base_block_names_raw: list[str] = []
    qv_sum_feature_block_names_raw: list[str] = []
    head_name_width = _head_suffix_width(list(head_layout_by_pair.values())) if use_head_blocks else 0
    if use_head_blocks:
        for pair_idx, block_name in enumerate(str(x) for x in schema.block_names):
            layout = head_layout_by_pair.get(int(pair_idx))
            if layout is None:
                continue
            for head_idx in range(int(layout.n_heads)):
                all_base_block_names_raw.append(_head_block_name(block_name, head_idx, width=head_name_width))
        if include_qv_sum_blocks:
            for spec in qv_specs:
                layout = head_layout_by_pair[int(spec.q_index)]
                for head_idx in range(int(layout.n_heads)):
                    qv_sum_feature_block_names_raw.append(
                        _head_block_name(spec.qv_block_name_raw, head_idx, width=head_name_width)
                    )
    else:
        all_base_block_names_raw = [str(x) for x in schema.block_names]
        if include_qv_sum_blocks:
            qv_sum_feature_block_names_raw = [spec.qv_block_name_raw for spec in qv_specs]

    feature_block_names_raw: list[str] = []
    if include_base_blocks:
        feature_block_names_raw.extend(all_base_block_names_raw)
    if include_qv_sum_blocks:
        feature_block_names_raw.extend(qv_sum_feature_block_names_raw)

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
    qv_sum_block_names = [shorten_block_name(name) for name in qv_sum_feature_block_names_raw]
    base_block_names = [shorten_block_name(name) for name in all_base_block_names_raw]
    feature_block_names = [shorten_block_name(name) for name in feature_block_names_raw]
    if use_head_blocks:
        base_lora_adapter_dims_for_summary: list[dict[str, int]] = []
        for pair_idx, _block_name in enumerate(schema.block_names):
            layout = head_layout_by_pair.get(int(pair_idx))
            if layout is None:
                continue
            a_shape = tuple(int(x) for x in schema.a_shapes[pair_idx])
            for _head_idx in range(int(layout.n_heads)):
                base_lora_adapter_dims_for_summary.append(
                    {
                        "m": int(layout.head_dim),
                        "n": int(a_shape[1]),
                        "r": int(a_shape[0]),
                    }
                )
        base_lora_adapter_dims = [] if variable_lora_rank else list(base_lora_adapter_dims_for_summary)
        qv_sum_lora_adapter_dims_for_summary: list[dict[str, int]] = []
        for spec in qv_specs:
            layout = head_layout_by_pair[int(spec.q_index)]
            a_q_shape = tuple(int(x) for x in schema.a_shapes[spec.q_index])
            a_v_shape = tuple(int(x) for x in schema.a_shapes[spec.v_index])
            for _head_idx in range(int(layout.n_heads)):
                qv_sum_lora_adapter_dims_for_summary.append(
                    {
                        "m": int(layout.head_dim),
                        "n": int(a_q_shape[1]),
                        "r": int(a_q_shape[0]) + int(a_v_shape[0]),
                    }
                )
        qv_sum_lora_adapter_dims = [] if variable_lora_rank else list(qv_sum_lora_adapter_dims_for_summary)
    else:
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
        "spectral_attention_granularity": resolved_attention_granularity,
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
    if use_head_blocks:
        metadata["attention_head_layouts"] = [
            {
                "block_name": shorten_block_name(str(schema.block_names[pair_idx])),
                "n_heads": int(layout.n_heads),
                "head_dim": int(layout.head_dim),
            }
            for pair_idx, layout in sorted(head_layout_by_pair.items(), key=lambda item: int(item[0]))
        ]
        metadata["attention_head_name_width"] = int(head_name_width)
    if not variable_lora_rank:
        metadata["base_lora_adapter_dims"] = base_lora_adapter_dims
        metadata["qv_sum_lora_adapter_dims"] = qv_sum_lora_adapter_dims
        metadata["lora_adapter_dims"] = feature_lora_adapter_dims
    return features, labels, model_names, sanitize_spectral_metadata(metadata)
