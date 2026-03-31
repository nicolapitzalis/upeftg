from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class ManifestItem:
    raw_entry: str
    model_dir: Path
    adapter_path: Path
    model_name: str
    label: int | None


@dataclass(frozen=True)
class AttackSampleIdentity:
    model_name: str
    subset_name: str
    model_family: str
    attack_name: str
    subset_has_clean: bool
    subset_has_backdoor: bool
    attack_name_source: str


_SAMPLE_SUFFIX_RE = re.compile(r"_label\d+_\d+$")
_RANK_TOKEN_RE = re.compile(r"^rank\d+(?:\.\d+)?$", re.IGNORECASE)
_MODEL_SIZE_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?[bm]$", re.IGNORECASE)
_KNOWN_ATTACK_NAMES = {
    "ripple": "RIPPLE",
    "syntactic": "syntactic",
    "insertsent": "insertsent",
    "stybkd": "stybkd",
    "stykbd": "stybkd",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_dir() -> Path:
    return _project_root() / "manifests"


def _find_manifest_by_name(manifest_name: str) -> Path | None:
    manifest_root = default_manifest_dir().resolve()
    if not manifest_root.exists():
        return None

    matches = sorted(path.resolve() for path in manifest_root.rglob(manifest_name) if path.is_file())
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    choices = ", ".join(path.relative_to(manifest_root).as_posix() for path in matches)
    raise ValueError(f"Manifest name '{manifest_name}' is ambiguous under {manifest_root}: {choices}")


def resolve_manifest_path(manifest_path: Path | str) -> Path:
    raw_path = Path(manifest_path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    manifest_root = default_manifest_dir().resolve()
    project_root = _project_root().resolve()
    candidates: list[Path] = []
    search_by_name = raw_path.parent == Path(".")
    if raw_path.parts and raw_path.parts[0] == manifest_root.name:
        candidates.append((project_root / raw_path).resolve())
        search_by_name = len(raw_path.parts) == 2
    else:
        candidates.append((manifest_root / raw_path).resolve())
    candidates.append((Path.cwd().resolve() / raw_path).resolve())
    candidates.append((project_root / raw_path).resolve())

    seen: set[Path] = set()
    deduped_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.exists():
            return candidate
    if search_by_name:
        resolved = _find_manifest_by_name(raw_path.name)
        if resolved is not None:
            return resolved
    return deduped_candidates[0]


def parse_label(model_dir_name: str) -> int | None:
    if "label0" in model_dir_name:
        return 0
    if "label1" in model_dir_name:
        return 1
    return None


def _subset_name_from_item(item: ManifestItem) -> str:
    parent_name = item.model_dir.parent.name.strip()
    if parent_name and parent_name != item.model_dir.name:
        return parent_name

    candidate = _SAMPLE_SUFFIX_RE.sub("", item.model_name)
    return candidate or item.model_name


def _strip_rank_config_tokens(subset_name: str) -> list[str]:
    tokens = [token for token in subset_name.split("_") if token]
    for i, token in enumerate(tokens):
        if _RANK_TOKEN_RE.fullmatch(token):
            return tokens[:i]
    return tokens


def _longest_common_token_prefix(token_rows: list[list[str]]) -> list[str]:
    if not token_rows:
        return []

    prefix = list(token_rows[0])
    for row in token_rows[1:]:
        keep = 0
        for left, right in zip(prefix, row):
            if left != right:
                break
            keep += 1
        prefix = prefix[:keep]
        if not prefix:
            break
    return prefix


def _split_model_family_tokens(tokens: list[str], *, cohort_prefix: list[str]) -> tuple[list[str], list[str]]:
    for i, token in enumerate(tokens):
        if _MODEL_SIZE_TOKEN_RE.fullmatch(token):
            return tokens[: i + 1], tokens[i + 1 :]

    if cohort_prefix and len(cohort_prefix) < len(tokens):
        return tokens[: len(cohort_prefix)], tokens[len(cohort_prefix) :]

    if len(tokens) >= 2:
        return tokens[:2], tokens[2:]
    if tokens:
        return tokens[:1], []
    return [], []


def _known_attack_name(tokens: list[str]) -> str | None:
    for token in tokens:
        canonical = _KNOWN_ATTACK_NAMES.get(token.lower())
        if canonical is not None:
            return canonical
    return None


def infer_attack_sample_identities(items: list[ManifestItem]) -> list[AttackSampleIdentity]:
    prepared: list[dict[str, Any]] = []
    token_rows: list[list[str]] = []
    subset_labels: dict[str, set[int | None]] = defaultdict(set)
    for item in items:
        subset_name = _subset_name_from_item(item)
        stem_tokens = _strip_rank_config_tokens(subset_name)
        prepared.append(
            {
                "item": item,
                "subset_name": subset_name,
                "stem_tokens": stem_tokens,
            }
        )
        subset_labels[subset_name].add(item.label)
        if stem_tokens:
            token_rows.append(stem_tokens)

    cohort_prefix = _longest_common_token_prefix(token_rows)
    by_model_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prepared:
        model_tokens, remainder_tokens = _split_model_family_tokens(
            list(record["stem_tokens"]),
            cohort_prefix=cohort_prefix,
        )
        model_family = "_".join(model_tokens) if model_tokens else "unknown"
        record["model_family"] = model_family
        record["remainder_tokens"] = remainder_tokens
        by_model_family[model_family].append(record)

    identities: list[AttackSampleIdentity] = []
    for record in prepared:
        model_family = str(record["model_family"])
        remainder_tokens = list(record["remainder_tokens"])
        subset_name = str(record["subset_name"])
        labels_in_subset = subset_labels.get(subset_name, set())
        subset_has_clean = 0 in labels_in_subset
        subset_has_backdoor = 1 in labels_in_subset

        known_attack = _known_attack_name(remainder_tokens)
        if known_attack is not None:
            attack_name = known_attack
            attack_name_source = "known_attack_name"
        else:
            attack_name = "_".join(remainder_tokens) if remainder_tokens else "unknown"
            attack_name_source = (
                "mixed_clean_backdoor_folder"
                if subset_has_clean and subset_has_backdoor
                else "folder_name"
            )
        identities.append(
            AttackSampleIdentity(
                model_name=record["item"].model_name,
                subset_name=subset_name,
                model_family=model_family,
                attack_name=attack_name,
                subset_has_clean=subset_has_clean,
                subset_has_backdoor=subset_has_backdoor,
                attack_name_source=attack_name_source,
            )
        )

    return identities


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


def _adapter_path_candidates_for_model_dir(model_dir: Path) -> tuple[Path, ...]:
    return (
        model_dir / "adapter_model.safetensors",
        model_dir / "best_model" / "adapter_model.safetensors",
    )


def _resolve_adapter_path_for_model_dir(model_dir: Path) -> Path:
    candidates = _adapter_path_candidates_for_model_dir(model_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _model_dir_from_adapter_path(adapter_path: Path) -> Path:
    parent = adapter_path.parent
    if parent.name == "best_model":
        return parent.parent
    return parent


def _resolve_manifest_entry(entry: str, dataset_root: Path) -> tuple[Path, Path]:
    raw = Path(entry)
    if raw.is_absolute():
        resolved = raw
    else:
        resolved = dataset_root / raw
    resolved = resolved.expanduser().resolve()

    if resolved.is_dir():
        if resolved.name == "best_model":
            model_dir = resolved.parent
            adapter_path = resolved / "adapter_model.safetensors"
        else:
            model_dir = resolved
            adapter_path = _resolve_adapter_path_for_model_dir(model_dir)
    else:
        adapter_path = resolved
        model_dir = _model_dir_from_adapter_path(adapter_path)

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


def _manifest_item_from_name_entry(entry: str) -> ManifestItem:
    raw = Path(entry).expanduser()
    if raw.exists():
        if raw.is_dir():
            if raw.name == "best_model":
                model_dir = raw.parent
                adapter_path = raw / "adapter_model.safetensors"
            else:
                model_dir = raw
                adapter_path = _resolve_adapter_path_for_model_dir(model_dir)
        else:
            model_dir = _model_dir_from_adapter_path(raw)
            adapter_path = raw
    else:
        if raw.name == "adapter_model.safetensors":
            model_dir = _model_dir_from_adapter_path(raw)
            adapter_path = raw
        elif raw.name == "best_model":
            model_dir = raw.parent
            adapter_path = raw / "adapter_model.safetensors"
        else:
            model_dir = raw
            adapter_path = raw / "adapter_model.safetensors"

    model_name = model_dir.name
    return ManifestItem(
        raw_entry=entry,
        model_dir=model_dir,
        adapter_path=adapter_path,
        model_name=model_name,
        label=parse_label(model_name),
    )


def parse_manifest_entries_by_model_name(
    entries: list[str],
    *,
    manifest_name: str,
) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    seen_names: set[str] = set()
    for line_no, entry in enumerate(entries, start=1):
        item = _manifest_item_from_name_entry(entry)
        if item.model_name in seen_names:
            raise ValueError(
                f"Duplicate model name in manifest {manifest_name}:{line_no} -> {item.model_name}"
            )
        seen_names.add(item.model_name)
        items.append(item)
    return items


def _parse_json_section_sources(
    section: Any,
    *,
    section_name: str,
    manifest_path: Path,
) -> list[tuple[str, list[int] | None]]:
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

    sources: list[tuple[str, list[int] | None]] = []
    for i, item in enumerate(source_list):
        if isinstance(item, str):
            entry = item.strip()
            if not entry:
                raise ValueError(f"{section_name}[{i}] in {manifest_path} is an empty string")
            sources.append((entry, None))
            continue

        if isinstance(item, dict):
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
            continue

        raise ValueError(
            f"{section_name}[{i}] in {manifest_path} must be either a string entry "
            "or an object with path/indices"
        )
    return sources


def _expand_sources_to_entries(sources: list[tuple[str, list[int] | None]]) -> list[str]:
    entries: list[str] = []
    for path, indices in sources:
        if indices is None:
            entries.append(path)
        else:
            entries.extend(expand_structured_paths(path, indices))
    return entries


def _load_manifest_payload(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved_manifest_path = resolve_manifest_path(manifest_path)
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {resolved_manifest_path}")

    with open(resolved_manifest_path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in manifest file {resolved_manifest_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON {resolved_manifest_path} must be an object")
    return resolved_manifest_path, payload


def parse_single_manifest_json(
    *,
    manifest_path: Path,
    dataset_root: Path,
    section_key: str = "path",
) -> list[ManifestItem]:
    resolved_manifest_path, payload = _load_manifest_payload(manifest_path)
    if section_key in payload:
        sources = _parse_json_section_sources(
            payload[section_key],
            section_name=section_key,
            manifest_path=resolved_manifest_path,
        )
        entries = _expand_sources_to_entries(sources)
        return parse_manifest_entries(
            entries,
            dataset_root=dataset_root,
            manifest_name=f"{resolved_manifest_path}::{section_key}",
        )

    if "train" in payload and "infer" in payload:
        train_sources = _parse_json_section_sources(
            payload["train"],
            section_name="train",
            manifest_path=resolved_manifest_path,
        )
        infer_sources = _parse_json_section_sources(
            payload["infer"],
            section_name="infer",
            manifest_path=resolved_manifest_path,
        )
        entries = _expand_sources_to_entries(train_sources) + _expand_sources_to_entries(infer_sources)
        return parse_manifest_entries(
            entries,
            dataset_root=dataset_root,
            manifest_name=f"{resolved_manifest_path}::train+infer",
        )

    raise ValueError(
        f"Manifest JSON {resolved_manifest_path} must include key '{section_key}' "
        "or both 'train' and 'infer' keys"
    )


def parse_single_manifest_json_by_model_name(
    *,
    manifest_path: Path,
    section_key: str = "path",
) -> list[ManifestItem]:
    resolved_manifest_path, payload = _load_manifest_payload(manifest_path)
    if section_key in payload:
        sources = _parse_json_section_sources(
            payload[section_key],
            section_name=section_key,
            manifest_path=resolved_manifest_path,
        )
        entries = _expand_sources_to_entries(sources)
        return parse_manifest_entries_by_model_name(
            entries,
            manifest_name=f"{resolved_manifest_path}::{section_key}",
        )

    if "train" in payload and "infer" in payload:
        train_sources = _parse_json_section_sources(
            payload["train"],
            section_name="train",
            manifest_path=resolved_manifest_path,
        )
        infer_sources = _parse_json_section_sources(
            payload["infer"],
            section_name="infer",
            manifest_path=resolved_manifest_path,
        )
        entries = _expand_sources_to_entries(train_sources) + _expand_sources_to_entries(infer_sources)
        return parse_manifest_entries_by_model_name(
            entries,
            manifest_name=f"{resolved_manifest_path}::train+infer",
        )

    raise ValueError(
        f"Manifest JSON {resolved_manifest_path} must include key '{section_key}' "
        "or both 'train' and 'infer' keys"
    )


def parse_joint_manifest_json(
    *,
    manifest_path: Path,
    dataset_root: Path,
) -> tuple[list[ManifestItem], list[ManifestItem]]:
    resolved_manifest_path, payload = _load_manifest_payload(manifest_path)
    if "train" not in payload or "infer" not in payload:
        raise ValueError(f"Manifest JSON {resolved_manifest_path} must include 'train' and 'infer' keys")

    train_sources = _parse_json_section_sources(
        payload["train"],
        section_name="train",
        manifest_path=resolved_manifest_path,
    )
    infer_sources = _parse_json_section_sources(
        payload["infer"],
        section_name="infer",
        manifest_path=resolved_manifest_path,
    )

    train_entries = _expand_sources_to_entries(train_sources)
    infer_entries = _expand_sources_to_entries(infer_sources)

    train_items = parse_manifest_entries(
        train_entries,
        dataset_root=dataset_root,
        manifest_name=f"{resolved_manifest_path}::train",
    )
    infer_items = parse_manifest_entries(
        infer_entries,
        dataset_root=dataset_root,
        manifest_name=f"{resolved_manifest_path}::infer",
    )
    validate_disjoint(train_items, infer_items)
    return train_items, infer_items


def parse_joint_manifest_json_by_model_name(
    *,
    manifest_path: Path,
) -> tuple[list[ManifestItem], list[ManifestItem]]:
    resolved_manifest_path, payload = _load_manifest_payload(manifest_path)
    if "train" not in payload or "infer" not in payload:
        raise ValueError(f"Manifest JSON {resolved_manifest_path} must include 'train' and 'infer' keys")

    train_sources = _parse_json_section_sources(
        payload["train"],
        section_name="train",
        manifest_path=resolved_manifest_path,
    )
    infer_sources = _parse_json_section_sources(
        payload["infer"],
        section_name="infer",
        manifest_path=resolved_manifest_path,
    )

    train_entries = _expand_sources_to_entries(train_sources)
    infer_entries = _expand_sources_to_entries(infer_sources)

    train_items = parse_manifest_entries_by_model_name(
        train_entries,
        manifest_name=f"{resolved_manifest_path}::train",
    )
    infer_items = parse_manifest_entries_by_model_name(
        infer_entries,
        manifest_name=f"{resolved_manifest_path}::infer",
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
