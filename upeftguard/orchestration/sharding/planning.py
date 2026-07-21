from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ...features.delta import (
    DeltaBlockSchema,
    load_delta_block_schema,
    schema_has_adalora_scaling,
    shorten_block_name,
)
from ...features.spectral import build_qv_sum_specs, resolve_spectral_qv_sum_mode
from ...utilities.core.manifest import (
    CV_ALWAYS_TRAIN_SECTION_KEY,
    ManifestItem,
    _expand_sources_to_entries,
    _load_manifest_payload,
    _parse_json_section_sources,
    parse_manifest_entries,
    resolve_manifest_path,
)
from ...utilities.core.serialization import json_ready

_RANK_TOKEN_RE = re.compile(r"(?:^|_)rank(\d+(?:\.\d+)?)(?:_|$)", re.IGNORECASE)


def _model_rank(model_name: str) -> int | None:
    match = _RANK_TOKEN_RE.search(str(model_name))
    return int(float(match.group(1))) if match is not None else None


def _schema_signature_payload(schema: DeltaBlockSchema) -> dict[str, Any]:
    if schema_has_adalora_scaling(schema):
        return {
            "pairs": [list(pair) for pair in schema.pairs],
            "block_names": list(schema.block_names),
            "in_dims": [int(shape[1]) for shape in schema.a_shapes],
            "out_dims": [int(shape[0]) for shape in schema.b_shapes],
            "has_e": [bool(key) for key in schema.e_keys],
            "signature_mode": "adalora_rank_tolerant",
        }
    return {
        "pairs": [list(pair) for pair in schema.pairs],
        "block_names": list(schema.block_names),
        "a_shapes": [list(shape) for shape in schema.a_shapes],
        "b_shapes": [list(shape) for shape in schema.b_shapes],
        "e_keys": [str(key) if key is not None else None for key in schema.e_keys],
        "e_shapes": [list(shape) if shape is not None else None for shape in schema.e_shapes],
        "signature_mode": "exact",
    }


def _schema_signature_key(schema: DeltaBlockSchema) -> str:
    return json.dumps(_schema_signature_payload(schema), sort_keys=True, separators=(",", ":"))


def _schema_digest(schema: DeltaBlockSchema) -> str:
    return hashlib.sha1(_schema_signature_key(schema).encode("utf-8")).hexdigest()[:12]


def _label_counts(items: list[ManifestItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = "unknown" if item.label is None else str(int(item.label))
        counts[key] = int(counts.get(key, 0) + 1)
    return counts


def _write_manifest(path: Path, items: list[ManifestItem]) -> None:
    payload = {"path": [str(item.adapter_path.resolve()) for item in items]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _shard_slices(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    if n_items <= 0:
        return []
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    n_resolved = min(int(n_shards), int(n_items))
    base_size, remainder = divmod(int(n_items), n_resolved)
    slices: list[tuple[int, int]] = []
    start = 0
    for shard in range(n_resolved):
        shard_size = base_size + (1 if shard < remainder else 0)
        end = start + shard_size
        slices.append((start, end))
        start = end
    return slices


def _schema_preview(schema: DeltaBlockSchema, *, limit: int = 3) -> list[str]:
    return [shorten_block_name(str(name)) for name in list(schema.block_names)[:limit]]


def _parse_single_manifest_sources(
    *,
    manifest_path: Path,
    dataset_root: Path,
    section_key: str = "path",
) -> list[tuple[str, list[ManifestItem]]]:
    resolved_manifest_path, payload = _load_manifest_payload(manifest_path)
    if section_key in payload:
        section_payloads = [(section_key, payload[section_key])]
        if section_key == "path" and CV_ALWAYS_TRAIN_SECTION_KEY in payload:
            section_payloads.append((CV_ALWAYS_TRAIN_SECTION_KEY, payload[CV_ALWAYS_TRAIN_SECTION_KEY]))
    elif "train" in payload and "infer" in payload:
        section_payloads = [("train", payload["train"]), ("infer", payload["infer"])]
    else:
        raise ValueError(
            f"Manifest JSON {resolved_manifest_path} must include key '{section_key}' "
            "or both 'train' and 'infer' keys for schema-aware shard preparation"
        )

    parsed_sources: list[tuple[str, list[ManifestItem]]] = []
    for current_section_key, section_payload in section_payloads:
        sources = _parse_json_section_sources(
            section_payload,
            section_name=current_section_key,
            manifest_path=resolved_manifest_path,
        )
        for source_index, (path_pattern, indices) in enumerate(sources):
            entries = _expand_sources_to_entries([(path_pattern, indices)])
            items = parse_manifest_entries(
                entries,
                dataset_root=dataset_root,
                manifest_name=f"{resolved_manifest_path}::{current_section_key}[{source_index}]",
            )
            parsed_sources.append((path_pattern, items))
    return parsed_sources


def _dataset_key_for_source(source_path: str, items: list[ManifestItem]) -> str:
    parent_dirs = {str(item.model_dir.parent.resolve()) for item in items}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs))
    return f"source::{source_path}"


def _prepare_group_summary(
    *,
    group_index: int,
    schema: DeltaBlockSchema,
    items: list[ManifestItem],
    output_dir: Path,
    worker_capacity: int,
    requested_qv_sum_mode: str,
    adapter_rank: int | None,
) -> tuple[dict[str, Any], list[str]]:
    qv_specs: list[Any] = []
    effective_qv_sum_mode = requested_qv_sum_mode
    warnings: list[str] = []
    if requested_qv_sum_mode in {"append", "only"}:
        qv_specs = build_qv_sum_specs(schema)

    if requested_qv_sum_mode == "append" and not qv_specs:
        effective_qv_sum_mode = "none"
        warnings.append(
            "Requested spectral_qv_sum_mode=append, but this schema has no supported attention q/v pairs; "
            "falling back to base spectral blocks only."
        )
    if requested_qv_sum_mode == "only" and not qv_specs:
        sample_models = ", ".join(item.model_name for item in items[:3])
        raise ValueError(
            "Requested spectral_qv_sum_mode=only, but schema group has no supported attention q/v pairs. "
            f"Example models: {sample_models}"
        )

    schema_digest = _schema_digest(schema)
    group_id = f"group_{group_index:03d}_{schema_digest}"
    group_dir = output_dir / group_id
    manifest_path = group_dir / "manifest.json"
    shard_manifest_dir = group_dir / "shard_manifests"
    shard_output_root = group_dir / "shards"
    merged_output_dir = group_dir / "merged"

    _write_manifest(manifest_path, items)
    shard_ranges = _shard_slices(len(items), worker_capacity)
    for shard_index, (start, end) in enumerate(shard_ranges):
        _write_manifest(shard_manifest_dir / f"shard_{shard_index}.json", items[start:end])

    shard_output_root.mkdir(parents=True, exist_ok=True)
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "group_id": group_id,
        "schema_digest": schema_digest,
        "schema_signature_mode": str(_schema_signature_payload(schema).get("signature_mode", "exact")),
        "manifest_path": str(manifest_path.resolve()),
        "shard_manifest_dir": str(shard_manifest_dir.resolve()),
        "shard_output_root": str(shard_output_root.resolve()),
        "merged_output_dir": str(merged_output_dir.resolve()),
        "n_items": int(len(items)),
        "n_shards": int(len(shard_ranges)),
        "adapter_rank": int(adapter_rank) if adapter_rank is not None else None,
        "requested_spectral_qv_sum_mode": requested_qv_sum_mode,
        "effective_spectral_qv_sum_mode": effective_qv_sum_mode,
        "qv_pair_count": int(len(qv_specs)),
        "qv_pairs_supported": bool(qv_specs),
        "n_blocks": int(len(schema.block_names)),
        "variable_lora_rank": bool(schema_has_adalora_scaling(schema)),
        "block_name_preview": _schema_preview(schema),
        "model_name_preview": [item.model_name for item in items[:5]],
        "label_counts": _label_counts(items),
    }
    return summary, warnings


def prepare_schema_sharded_manifests(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    worker_capacity: int,
    spectral_qv_sum_mode: str,
    report_path: Path | None = None,
    worker_capacity_by_rank: dict[int, int] | None = None,
) -> dict[str, Any]:
    if worker_capacity <= 0:
        raise ValueError(f"worker_capacity must be positive, got {worker_capacity}")

    resolved_manifest_path = resolve_manifest_path(manifest_path)
    resolved_dataset_root = dataset_root.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    requested_qv_sum_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    source_items = _parse_single_manifest_sources(
        manifest_path=resolved_manifest_path,
        dataset_root=resolved_dataset_root,
        section_key="path",
    )
    all_items = [item for _, items in source_items for item in items]
    if not all_items:
        raise ValueError("Manifest resolved to zero items")

    grouped_by_signature: dict[tuple[str, int | None], dict[str, Any]] = {}
    signature_order: list[tuple[str, int | None]] = []
    schema_by_dataset: dict[str, tuple[DeltaBlockSchema, str]] = {}
    sampled_dataset_count = 0
    for source_path, items in source_items:
        dataset_key = _dataset_key_for_source(source_path, items)
        cached = schema_by_dataset.get(dataset_key)
        if cached is None:
            schema = load_delta_block_schema(items[0].adapter_path)
            signature_key = _schema_signature_key(schema)
            schema_by_dataset[dataset_key] = (schema, signature_key)
            sampled_dataset_count += 1
        else:
            schema, signature_key = cached

        signature_key = _schema_signature_key(schema)
        items_by_rank: dict[int | None, list[ManifestItem]] = {}
        if worker_capacity_by_rank is None:
            items_by_rank[None] = list(items)
        else:
            for item in items:
                items_by_rank.setdefault(_model_rank(item.model_name), []).append(item)
        for adapter_rank, rank_items in items_by_rank.items():
            group_key = (signature_key, adapter_rank)
            if group_key not in grouped_by_signature:
                grouped_by_signature[group_key] = {
                    "schema": schema,
                    "items": [],
                    "adapter_rank": adapter_rank,
                }
                signature_order.append(group_key)
            grouped_by_signature[group_key]["items"].extend(rank_items)

    groups: list[dict[str, Any]] = []
    warnings: list[str] = []
    if len(signature_order) > 1:
        warnings.append(
            "Multiple adapter schemas were detected. Shards and merged outputs will be prepared per schema group."
        )

    for group_index, group_key in enumerate(signature_order):
        record = grouped_by_signature[group_key]
        adapter_rank = record["adapter_rank"]
        group_capacity = (
            int(worker_capacity_by_rank.get(adapter_rank, worker_capacity))
            if worker_capacity_by_rank is not None and adapter_rank is not None
            else int(worker_capacity)
        )
        group_summary, group_warnings = _prepare_group_summary(
            group_index=group_index,
            schema=record["schema"],
            items=list(record["items"]),
            output_dir=resolved_output_dir,
            worker_capacity=group_capacity,
            requested_qv_sum_mode=requested_qv_sum_mode,
            adapter_rank=adapter_rank,
        )
        groups.append(group_summary)
        for warning in group_warnings:
            warnings.append(f"{group_summary['group_id']}: {warning}")

    report = {
        "manifest_path": str(resolved_manifest_path),
        "dataset_root": str(resolved_dataset_root),
        "output_dir": str(resolved_output_dir),
        "requested_spectral_qv_sum_mode": requested_qv_sum_mode,
        "worker_capacity": int(worker_capacity),
        "worker_capacity_by_rank": {
            str(rank): int(count) for rank, count in sorted((worker_capacity_by_rank or {}).items())
        },
        "n_items": int(len(all_items)),
        "n_manifest_sources": int(len(source_items)),
        "n_schema_samples": int(sampled_dataset_count),
        "schema_inference_mode": "per_dataset_sample",
        "group_count": int(len(groups)),
        "groups": groups,
        "warnings": warnings,
    }

    resolved_report_path = (
        report_path.expanduser().resolve()
        if report_path is not None
        else (resolved_output_dir / "schema_partition_report.json").resolve()
    )
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")

    result = dict(report)
    result["report_path"] = str(resolved_report_path)
    return result
