from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestItem:
    raw_entry: str
    model_dir: Path
    adapter_path: Path
    model_name: str
    label: int | None


def parse_label(model_dir_name: str) -> int | None:
    if "label0" in model_dir_name:
        return 0
    if "label1" in model_dir_name:
        return 1
    return None


def _parse_indices_spec(spec: str, *, manifest_path: Path, key: str) -> list[int]:
    text = spec.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(
            f"Invalid index spec for '{key}' in {manifest_path}. Expected bracket form like [0,10], got '{spec}'"
        )

    body = text[1:-1].strip()
    if not body:
        raise ValueError(f"Empty index spec for '{key}' in {manifest_path}")

    parts = [p.strip() for p in body.split(",") if p.strip()]
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"Non-integer index value in '{key}' in {manifest_path}: '{spec}'"
        ) from exc

    if any(v < 0 for v in values):
        raise ValueError(f"Negative indices are not allowed for '{key}' in {manifest_path}: '{spec}'")

    if len(values) == 2 and values[1] >= values[0]:
        return list(range(values[0], values[1] + 1))

    dedup: list[int] = []
    seen: set[int] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def parse_indices_value(value: Any, *, manifest_path: Path, key: str) -> list[int]:
    if isinstance(value, str):
        return _parse_indices_spec(value, manifest_path=manifest_path, key=key)

    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty index list for '{key}' in {manifest_path}")
        try:
            ints = [int(v) for v in value]
        except Exception as exc:
            raise ValueError(f"Non-integer index value in '{key}' in {manifest_path}: {value}") from exc
        if any(v < 0 for v in ints):
            raise ValueError(f"Negative indices are not allowed for '{key}' in {manifest_path}: {value}")
        if len(ints) == 2 and ints[1] >= ints[0]:
            return list(range(ints[0], ints[1] + 1))

        dedup: list[int] = []
        seen: set[int] = set()
        for v in ints:
            if v in seen:
                continue
            seen.add(v)
            dedup.append(v)
        return dedup

    raise ValueError(f"Unsupported index format for '{key}' in {manifest_path}: {value}")


def expand_structured_paths(path_pattern: str, indices: list[int]) -> list[str]:
    entries: list[str] = []
    for i in indices:
        if "{i}" in path_pattern:
            entries.append(path_pattern.format(i=i))
        else:
            entries.append(f"{path_pattern}{i}")
    return entries


def _resolve_manifest_entry(entry: str, dataset_root: Path) -> tuple[Path, Path]:
    raw = Path(entry)
    if raw.is_absolute():
        resolved = raw
    else:
        resolved = dataset_root / raw
    resolved = resolved.expanduser().resolve()

    if resolved.is_dir():
        model_dir = resolved
        adapter_path = model_dir / "adapter_model.safetensors"
    else:
        model_dir = resolved.parent
        adapter_path = resolved

    if adapter_path.name != "adapter_model.safetensors":
        raise ValueError(
            f"Manifest entry must resolve to a model directory or adapter_model.safetensors: '{entry}'"
        )
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter file not found for manifest entry '{entry}': {adapter_path}")

    return model_dir, adapter_path


def parse_manifest_entries(
    entries: list[str],
    *,
    dataset_root: Path,
    manifest_name: str,
) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    seen: set[Path] = set()
    for line_no, entry in enumerate(entries, start=1):
        model_dir, adapter_path = _resolve_manifest_entry(entry, dataset_root=dataset_root)
        key = adapter_path.resolve()
        if key in seen:
            raise ValueError(f"Duplicate adapter in manifest {manifest_name}:{line_no} -> {adapter_path}")
        seen.add(key)
        items.append(
            ManifestItem(
                raw_entry=entry,
                model_dir=model_dir,
                adapter_path=adapter_path,
                model_name=model_dir.name,
                label=parse_label(model_dir.name),
            )
        )
    return items


def _parse_json_section_sources(
    section: Any,
    *,
    section_name: str,
    manifest_path: Path,
) -> list[tuple[str, list[int]]]:
    if isinstance(section, list):
        source_list = section
    elif isinstance(section, dict):
        if "path" in section and "indices" in section:
            source_list = [section]
        elif isinstance(section.get("sources"), list):
            source_list = section["sources"]
        else:
            raise ValueError(
                f"Section '{section_name}' in {manifest_path} must be a list, a path/indices object, "
                "or an object with a 'sources' list"
            )
    else:
        raise ValueError(
            f"Section '{section_name}' in {manifest_path} must be a list or object"
        )

    if not source_list:
        raise ValueError(f"Section '{section_name}' in {manifest_path} is empty")

    sources: list[tuple[str, list[int]]] = []
    for i, item in enumerate(source_list):
        if not isinstance(item, dict):
            raise ValueError(
                f"{section_name}[{i}] in {manifest_path} must be an object with path/indices"
            )
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{section_name}[{i}] in {manifest_path} is missing non-empty 'path'")
        if "indices" not in item:
            raise ValueError(f"{section_name}[{i}] in {manifest_path} is missing 'indices'")
        indices = parse_indices_value(
            item["indices"],
            manifest_path=manifest_path,
            key=f"{section_name}[{i}].indices",
        )
        sources.append((path.strip(), indices))
    return sources


def parse_single_manifest_json(
    *,
    manifest_path: Path,
    dataset_root: Path,
    section_key: str = "path",
) -> list[ManifestItem]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in manifest file {manifest_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON {manifest_path} must be an object")
    if section_key not in payload:
        raise ValueError(f"Manifest JSON {manifest_path} must include key '{section_key}'")

    sources = _parse_json_section_sources(
        payload[section_key],
        section_name=section_key,
        manifest_path=manifest_path,
    )
    entries: list[str] = []
    for path, indices in sources:
        entries.extend(expand_structured_paths(path, indices))

    return parse_manifest_entries(
        entries,
        dataset_root=dataset_root,
        manifest_name=f"{manifest_path}::{section_key}",
    )


def parse_joint_manifest_json(
    *,
    manifest_path: Path,
    dataset_root: Path,
) -> tuple[list[ManifestItem], list[ManifestItem]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in manifest file {manifest_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON {manifest_path} must be an object")
    if "train" not in payload or "infer" not in payload:
        raise ValueError(f"Manifest JSON {manifest_path} must include 'train' and 'infer' keys")

    train_sources = _parse_json_section_sources(
        payload["train"],
        section_name="train",
        manifest_path=manifest_path,
    )
    infer_sources = _parse_json_section_sources(
        payload["infer"],
        section_name="infer",
        manifest_path=manifest_path,
    )

    train_entries: list[str] = []
    infer_entries: list[str] = []
    for path, indices in train_sources:
        train_entries.extend(expand_structured_paths(path, indices))
    for path, indices in infer_sources:
        infer_entries.extend(expand_structured_paths(path, indices))

    train_items = parse_manifest_entries(
        train_entries,
        dataset_root=dataset_root,
        manifest_name=f"{manifest_path}::train",
    )
    infer_items = parse_manifest_entries(
        infer_entries,
        dataset_root=dataset_root,
        manifest_name=f"{manifest_path}::infer",
    )
    validate_disjoint(train_items, infer_items)
    return train_items, infer_items


def validate_disjoint(train_items: list[ManifestItem], infer_items: list[ManifestItem]) -> None:
    if not train_items:
        raise ValueError("Training manifest is empty after parsing")
    if not infer_items:
        raise ValueError("Inference manifest is empty after parsing")

    train_paths = {item.adapter_path.resolve() for item in train_items}
    infer_paths = {item.adapter_path.resolve() for item in infer_items}
    overlap = sorted(train_paths & infer_paths)
    if overlap:
        preview = ", ".join(str(p) for p in overlap[:5])
        raise ValueError(f"Training and inference manifests must be disjoint. Overlap examples: {preview}")
