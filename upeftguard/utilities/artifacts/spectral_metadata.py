from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from ..core.manifest import _resolve_adapter_path_for_model_dir
from ..core.serialization import json_ready
from .dataset_references import DATASET_REFERENCE_REPORT_NAME, load_dataset_reference_report


SPECTRAL_METADATA_STATE_NAME = ".spectral_metadata_state.json"

_PUBLIC_METADATA_DROP_KEYS = {
    "feature_names",
    "block_names",
    "base_block_names",
    "qv_sum_block_names",
    "lora_adapter_dims",
    "base_lora_adapter_dims",
    "qv_sum_lora_adapter_dims",
    "n_blocks",
    "has_adalora_scaling",
    "e_shapes",
    "schema_layout_summary",
}


def default_spectral_metadata_state_path(metadata_path: Path) -> Path:
    return metadata_path.with_name(SPECTRAL_METADATA_STATE_NAME)


def load_spectral_metadata(metadata_path: Path) -> dict[str, Any]:
    state_path = default_spectral_metadata_state_path(metadata_path)
    target = state_path if state_path.exists() else metadata_path
    with open(target, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {target}, got {type(payload).__name__}")
    if state_path.exists() and metadata_path.exists() and state_path != metadata_path:
        with open(metadata_path, "r", encoding="utf-8") as f:
            public_payload = json.load(f)
        if isinstance(public_payload, dict):
            public_dataset_layouts = public_payload.get("dataset_layouts")
            if isinstance(public_dataset_layouts, list):
                payload["dataset_layouts"] = public_dataset_layouts
            for key, value in public_payload.items():
                if key == "dataset_layouts":
                    continue
                payload.setdefault(str(key), value)
    return dict(payload)


def _layer_identifier(block_name: str) -> str:
    parts = [part for part in str(block_name).split(".") if part]
    if len(parts) <= 2:
        return str(block_name)
    prefix = ".".join(parts[:-2]).strip()
    return prefix or str(block_name)


def _base_block_names(metadata: Mapping[str, Any]) -> list[str]:
    raw_base = metadata.get("base_block_names")
    if isinstance(raw_base, list):
        return [str(x) for x in raw_base]

    raw_blocks = metadata.get("block_names")
    if isinstance(raw_blocks, list):
        return [str(x) for x in raw_blocks if ".qv_sum" not in str(x)]

    return []


def _adapter_dims_summary(dim_entries: list[dict[str, int]], *, variable_lora_rank: bool) -> dict[str, Any] | None:
    normalized: list[tuple[int, int, int]] = []
    for entry in dim_entries:
        if not isinstance(entry, dict):
            continue
        try:
            normalized.append((int(entry["m"]), int(entry["n"]), int(entry["r"])))
        except (KeyError, TypeError, ValueError):
            continue
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


@lru_cache(maxsize=512)
def _schema_layout_summary_from_adapter_path(adapter_path_str: str) -> dict[str, Any] | None:
    from ...features.delta import (
        build_schema_metadata,
        load_delta_block_schema,
        lora_adapter_dims_from_shapes,
    )
    from ...features.spectral import summarize_schema_layout

    adapter_path = Path(adapter_path_str).expanduser()
    if not adapter_path.exists():
        return None

    schema = load_delta_block_schema(adapter_path)
    schema_metadata = build_schema_metadata(schema)
    base_block_names = [str(x) for x in schema_metadata.get("block_names", [])]
    if not base_block_names:
        return None

    base_lora_adapter_dims = [dict(x) for x in schema_metadata.get("lora_adapter_dims", [])]
    if not base_lora_adapter_dims:
        base_lora_adapter_dims = [
            lora_adapter_dims_from_shapes(
                tuple(int(x) for x in a_shape),
                tuple(int(x) for x in b_shape),
            )
            for a_shape, b_shape in zip(schema.a_shapes, schema.b_shapes)
        ]

    return summarize_schema_layout(
        base_block_names=base_block_names,
        base_lora_adapter_dims=base_lora_adapter_dims,
        variable_lora_rank=bool(schema_metadata.get("variable_lora_rank", False)),
    )


def _layout_summaries_from_dataset_reference_payload(
    dataset_reference_payload: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(dataset_reference_payload, Mapping):
        return {}

    model_index = dataset_reference_payload.get("model_index")
    if not isinstance(model_index, Mapping):
        return {}

    grouped_entries: dict[str, list[Mapping[str, Any]]] = {}
    for model_name, raw_entry in model_index.items():
        if not isinstance(raw_entry, Mapping):
            continue
        dataset_name = str(raw_entry.get("dataset_name") or "unknown")
        entry = dict(raw_entry)
        entry["_model_name"] = str(model_name)
        grouped_entries.setdefault(dataset_name, []).append(entry)

    summaries: dict[str, dict[str, Any]] = {}
    for dataset_name, entries in grouped_entries.items():
        for entry in entries:
            dataset_path = entry.get("dataset_path")
            model_name = entry.get("_model_name")
            if not dataset_path or not model_name:
                continue
            adapter_path = _resolve_adapter_path_for_model_dir(
                Path(str(dataset_path)).expanduser() / str(model_name)
            )
            summary = _schema_layout_summary_from_adapter_path(str(adapter_path.resolve()))
            if summary is None:
                continue
            summaries[dataset_name] = dict(summary)
            break
    return summaries


def schema_layout_summary_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    from ...features.spectral import spectral_block_lora_dims_by_block

    raw_summary = metadata.get("schema_layout_summary")
    if isinstance(raw_summary, dict) and raw_summary:
        return dict(raw_summary)

    base_block_names = _base_block_names(metadata)
    if not base_block_names:
        return None

    layer_count = int(len({_layer_identifier(name) for name in base_block_names}))
    dim_map = spectral_block_lora_dims_by_block(metadata)
    base_dims = [dict(dim_map[name]) for name in base_block_names if name in dim_map]
    adapter_dims = _adapter_dims_summary(
        base_dims,
        variable_lora_rank=bool(metadata.get("variable_lora_rank", False)),
    )
    summary: dict[str, Any] = {"layer_count": layer_count}
    if adapter_dims is not None:
        summary["adapter_dims"] = adapter_dims
    return summary


def dataset_layouts_from_source(
    *,
    metadata: Mapping[str, Any],
    dataset_reference_payload: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    raw_dataset_layouts = metadata.get("dataset_layouts")
    if isinstance(raw_dataset_layouts, list):
        explicit_layouts: list[dict[str, Any]] = []
        for raw_entry in raw_dataset_layouts:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = {
                key: value
                for key, value in raw_entry.items()
                if key not in {"dataset_name", "sample_count"}
            }
            entry["dataset_name"] = str(raw_entry.get("dataset_name") or "unknown")
            entry["sample_count"] = int(raw_entry.get("sample_count", 0))
            explicit_layouts.append(entry)
        if explicit_layouts:
            return explicit_layouts

    summary = schema_layout_summary_from_metadata(metadata)
    inferred_summaries = _layout_summaries_from_dataset_reference_payload(dataset_reference_payload)
    if summary is None:
        if not inferred_summaries:
            return []

    dataset_groups = dataset_reference_payload.get("dataset_groups", []) if isinstance(dataset_reference_payload, Mapping) else []
    layouts: list[dict[str, Any]] = []
    if isinstance(dataset_groups, list) and dataset_groups:
        for group in dataset_groups:
            if not isinstance(group, Mapping):
                continue
            dataset_name = str(group.get("dataset_name") or "unknown")
            entry = {
                "dataset_name": dataset_name,
                "sample_count": int(group.get("sample_count", 0)),
            }
            resolved_summary = inferred_summaries.get(dataset_name) or summary
            if resolved_summary is not None:
                entry.update(dict(resolved_summary))
            layouts.append(entry)
        return layouts

    if inferred_summaries:
        return [
            {
                "dataset_name": dataset_name,
                "sample_count": 0,
                **dict(dataset_summary),
            }
            for dataset_name, dataset_summary in sorted(inferred_summaries.items())
        ]

    return [{"dataset_name": "unknown", "sample_count": 0, **dict(summary)}]


def merge_dataset_layouts(
    *,
    output_dataset_reference_payload: Mapping[str, Any] | None,
    source_layouts: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    variants_by_name: dict[str, list[dict[str, Any]]] = {}
    seen_by_name: dict[str, set[str]] = {}
    for layout_group in source_layouts:
        for raw_entry in layout_group:
            if not isinstance(raw_entry, Mapping):
                continue
            dataset_name = str(raw_entry.get("dataset_name") or "unknown")
            entry = {
                key: value
                for key, value in raw_entry.items()
                if key not in {"dataset_name", "sample_count"}
            }
            entry_key = json.dumps(json_ready(entry), sort_keys=True)
            if dataset_name not in variants_by_name:
                variants_by_name[dataset_name] = []
                seen_by_name[dataset_name] = set()
            if entry_key in seen_by_name[dataset_name]:
                continue
            seen_by_name[dataset_name].add(entry_key)
            variants_by_name[dataset_name].append(entry)

    dataset_groups = (
        output_dataset_reference_payload.get("dataset_groups", [])
        if isinstance(output_dataset_reference_payload, Mapping)
        else []
    )
    merged: list[dict[str, Any]] = []
    if isinstance(dataset_groups, list) and dataset_groups:
        for group in dataset_groups:
            if not isinstance(group, Mapping):
                continue
            dataset_name = str(group.get("dataset_name") or "unknown")
            entry: dict[str, Any] = {
                "dataset_name": dataset_name,
                "sample_count": int(group.get("sample_count", 0)),
            }
            variants = variants_by_name.get(dataset_name, [])
            if len(variants) == 1:
                entry.update(dict(variants[0]))
            elif len(variants) > 1:
                entry["layout_variants"] = [dict(variant) for variant in variants]
            merged.append(entry)
        return merged

    for dataset_name in sorted(variants_by_name):
        entry = {"dataset_name": dataset_name, "sample_count": 0}
        variants = variants_by_name[dataset_name]
        if len(variants) == 1:
            entry.update(dict(variants[0]))
        elif len(variants) > 1:
            entry["layout_variants"] = [dict(variant) for variant in variants]
        merged.append(entry)
    return merged


def build_public_spectral_metadata(
    *,
    internal_metadata: Mapping[str, Any],
    dataset_layouts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from ...features.spectral import sanitize_spectral_metadata

    cleaned = sanitize_spectral_metadata(internal_metadata)
    public = {
        key: value
        for key, value in cleaned.items()
        if key not in _PUBLIC_METADATA_DROP_KEYS
    }
    if dataset_layouts is not None:
        public["dataset_layouts"] = [dict(entry) for entry in dataset_layouts]
    return public


def write_spectral_metadata(
    metadata_path: Path,
    *,
    internal_metadata: Mapping[str, Any],
    dataset_layouts: list[dict[str, Any]] | None = None,
) -> Path:
    resolved = metadata_path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd().resolve() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    state_payload = dict(internal_metadata)
    if dataset_layouts is not None:
        state_payload["dataset_layouts"] = [dict(entry) for entry in dataset_layouts]

    state_path = default_spectral_metadata_state_path(resolved)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(state_payload), f, indent=2)

    public_metadata = build_public_spectral_metadata(
        internal_metadata=state_payload,
        dataset_layouts=dataset_layouts,
    )
    with open(resolved, "w", encoding="utf-8") as f:
        json.dump(json_ready(public_metadata), f, indent=2)
    return resolved


def resolve_dataset_reference_for_metadata(metadata_path: Path) -> dict[str, Any] | None:
    metadata_dir = metadata_path.parent
    if metadata_dir.name == "features":
        report_path = metadata_dir.parent / "reports" / DATASET_REFERENCE_REPORT_NAME
    else:
        report_path = metadata_dir / DATASET_REFERENCE_REPORT_NAME
    if not report_path.exists():
        return None
    return load_dataset_reference_report(report_path)
