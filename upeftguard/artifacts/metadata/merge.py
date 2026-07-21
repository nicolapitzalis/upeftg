"""Metadata reconciliation shared by feature merge workflows."""

from __future__ import annotations

from typing import Any

from ...features.spectral import spectral_block_lora_dims_by_block


def merge_lora_dim_maps(*metadata_payloads: dict[str, Any]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for payload in metadata_payloads:
        for block_name, dims in spectral_block_lora_dims_by_block(payload).items():
            existing = merged.get(block_name)
            if existing is not None and existing != dims:
                raise ValueError(f"Conflicting LoRA dimensions for block '{block_name}': {existing} vs {dims}")
            merged[block_name] = dict(dims)
    return merged


def resolved_qv_sum_mode(block_names: list[str]) -> str:
    has_qv_sum = any(".qv_sum" in name for name in block_names)
    has_base = any(".qv_sum" not in name for name in block_names)
    if has_qv_sum and has_base:
        return "append"
    if has_qv_sum:
        return "only"
    return "none"


def merge_skipped_models(*metadata_payloads: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for payload in metadata_payloads:
        raw_entries = payload.get("skipped_models")
        if not isinstance(raw_entries, list):
            continue
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            model_name = str(raw_entry.get("model_name") or "").strip()
            if not model_name:
                continue
            merged[model_name] = {
                "model_name": model_name,
                "adapter_path": str(raw_entry.get("adapter_path") or ""),
                "label": raw_entry.get("label"),
                "exception_type": str(raw_entry.get("exception_type") or ""),
                "exception_message": str(raw_entry.get("exception_message") or ""),
            }
    return [merged[name] for name in sorted(merged)]
