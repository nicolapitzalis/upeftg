#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from ..utilities.core.manifest import (
    AttackSampleIdentity,
    ManifestItem,
    infer_attack_sample_identities,
    parse_indices_value,
    parse_label,
)


GroupBy = Literal["adapter", "architecture"]

_RANK_TOKEN_RE = re.compile(r"^rank\d+(?:\.\d+)?$", re.IGNORECASE)
_ADAPTER_LABEL_BY_TOKEN = {
    "adalora": "adalora",
    "dora": "dora",
    "qlora": "qlora",
}
_ADAPTER_LABEL_BY_PAIR = {
    ("lora", "plus"): "lora+",
    ("lora", "only"): "lora-only",
}


@dataclass(frozen=True)
class ManifestSource:
    path: str
    indices: Any | None = None

    def to_json(self) -> str | dict[str, Any]:
        if self.indices is None:
            return self.path
        return {"path": self.path, "indices": self.indices}


@dataclass(frozen=True)
class GroupedSource:
    source: ManifestSource
    group_name: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON {path} must be an object")
    return payload


def _section_sources(section: Any, *, section_name: str, manifest_path: Path) -> list[ManifestSource]:
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
        raise ValueError(f"Section '{section_name}' in {manifest_path} must be a list or object")

    if not source_list:
        raise ValueError(f"Section '{section_name}' in {manifest_path} is empty")

    sources: list[ManifestSource] = []
    for index, item in enumerate(source_list):
        if isinstance(item, str):
            entry = item.strip()
            if not entry:
                raise ValueError(f"{section_name}[{index}] in {manifest_path} is an empty string")
            sources.append(ManifestSource(path=entry))
            continue

        if isinstance(item, dict):
            path = item.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{section_name}[{index}] in {manifest_path} is missing non-empty 'path'")
            if "indices" not in item:
                raise ValueError(f"{section_name}[{index}] in {manifest_path} is missing 'indices'")
            parse_indices_value(
                item["indices"],
                manifest_path=manifest_path,
                key=f"{section_name}[{index}].indices",
            )
            sources.append(ManifestSource(path=path.strip(), indices=item["indices"]))
            continue

        raise ValueError(
            f"{section_name}[{index}] in {manifest_path} must be either a string entry "
            "or an object with path/indices"
        )
    return sources


def _load_manifest_sources(manifest_path: Path) -> list[ManifestSource]:
    payload = _load_json(manifest_path)
    if "path" in payload:
        return _section_sources(payload["path"], section_name="path", manifest_path=manifest_path)
    if "train" in payload and "infer" in payload:
        return [
            *_section_sources(payload["train"], section_name="train", manifest_path=manifest_path),
            *_section_sources(payload["infer"], section_name="infer", manifest_path=manifest_path),
        ]
    raise ValueError(f"Manifest JSON {manifest_path} must include 'path' or both 'train' and 'infer'")


def _item_from_source(source: ManifestSource) -> ManifestItem:
    raw = Path(source.path).expanduser()
    if raw.name == "adapter_model.safetensors":
        model_dir = raw.parent.parent if raw.parent.name == "best_model" else raw.parent
    elif raw.name == "best_model":
        model_dir = raw.parent
    else:
        model_dir = raw
    model_name = model_dir.name
    return ManifestItem(
        raw_entry=source.path,
        model_dir=model_dir,
        adapter_path=model_dir / "adapter_model.safetensors",
        model_name=model_name,
        label=parse_label(model_name),
    )


def _subset_tokens(identity: AttackSampleIdentity) -> list[str]:
    tokens = [token for token in str(identity.subset_name or "unknown").split("_") if token]
    for index, token in enumerate(tokens):
        if _RANK_TOKEN_RE.fullmatch(token):
            return tokens[:index]
    return tokens


def _longest_common_token_suffix(token_rows: list[list[str]]) -> list[str]:
    if not token_rows:
        return []
    reversed_rows = [list(reversed(row)) for row in token_rows]
    suffix_reversed: list[str] = list(reversed_rows[0])
    for row in reversed_rows[1:]:
        keep = 0
        for left, right in zip(suffix_reversed, row):
            if left != right:
                break
            keep += 1
        suffix_reversed = suffix_reversed[:keep]
        if not suffix_reversed:
            break
    return list(reversed(suffix_reversed))


def adapter_group_name(identity: AttackSampleIdentity) -> str:
    subset_tokens = [token.lower() for token in str(identity.subset_name or "unknown").split("_") if token]
    model_tokens = [token.lower() for token in str(identity.model_family or "").split("_") if token]
    remainder = subset_tokens
    if model_tokens and subset_tokens[: len(model_tokens)] == model_tokens:
        remainder = subset_tokens[len(model_tokens) :]

    best_match: tuple[int, str] | None = None
    for index, token in enumerate(remainder):
        label = _ADAPTER_LABEL_BY_TOKEN.get(token)
        if label is not None:
            best_match = (index, label)

    for index in range(max(0, len(remainder) - 1)):
        label = _ADAPTER_LABEL_BY_PAIR.get((remainder[index], remainder[index + 1]))
        if label is not None:
            best_match = (index + 1, label)

    if best_match is not None:
        return best_match[1]
    return "lora"


def architecture_group_name(identity: AttackSampleIdentity, *, common_dataset_suffix: list[str]) -> str:
    tokens = _subset_tokens(identity)
    if common_dataset_suffix and tokens[-len(common_dataset_suffix) :] == common_dataset_suffix:
        architecture_tokens = tokens[: -len(common_dataset_suffix)]
    else:
        architecture_tokens = []
    if architecture_tokens:
        return "_".join(architecture_tokens)
    return str(identity.model_family or identity.subset_name or "unknown")


def _group_sources(sources: list[ManifestSource], *, group_by: GroupBy) -> list[GroupedSource]:
    items = [_item_from_source(source) for source in sources]
    identities = infer_attack_sample_identities(items)
    common_suffix = _longest_common_token_suffix([_subset_tokens(identity) for identity in identities])

    grouped: list[GroupedSource] = []
    for source, identity in zip(sources, identities):
        if group_by == "adapter":
            group_name = adapter_group_name(identity)
        elif group_by == "architecture":
            group_name = architecture_group_name(identity, common_dataset_suffix=common_suffix)
        else:
            raise ValueError(f"Unsupported group_by value: {group_by}")
        grouped.append(GroupedSource(source=source, group_name=group_name))
    return grouped


def _slugify_group_name(group_name: str) -> str:
    slug = str(group_name).strip().lower().replace("+", "_plus")
    slug = re.sub(r"[^a-z0-9._-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def build_leave_one_out_manifests(source_manifest: Path, *, group_by: GroupBy) -> dict[str, dict[str, Any]]:
    source_manifest = source_manifest.expanduser().resolve()
    if not source_manifest.exists():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest}")

    grouped_sources = _group_sources(_load_manifest_sources(source_manifest), group_by=group_by)
    group_names = sorted({row.group_name for row in grouped_sources})
    if len(group_names) < 2:
        raise ValueError(
            f"{group_by} leave-one-out requires at least two groups; found {len(group_names)} in {source_manifest}"
        )

    manifests: dict[str, dict[str, Any]] = {}
    for heldout_group in group_names:
        train = [
            row.source.to_json()
            for row in grouped_sources
            if row.group_name != heldout_group
        ]
        infer = [
            row.source.to_json()
            for row in grouped_sources
            if row.group_name == heldout_group
        ]
        if not train or not infer:
            raise ValueError(f"Generated empty train/infer split for held-out {group_by} group {heldout_group}")
        manifests[heldout_group] = {"train": train, "infer": infer}
    return manifests


def prepare_manifests(source_manifest: Path, output_root: Path, *, group_by: GroupBy) -> list[Path]:
    source_manifest = source_manifest.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_manifest.parent == output_root:
        raise ValueError("source_manifest parent and output_root must be different directories")

    output_root.mkdir(parents=True, exist_ok=True)
    for stale_path in output_root.glob("*.json"):
        stale_path.unlink()

    generated_payloads = build_leave_one_out_manifests(source_manifest, group_by=group_by)
    generated_paths: list[Path] = []
    for heldout_group, payload in generated_payloads.items():
        output_path = output_root / f"holdout_{group_by}_{_slugify_group_name(heldout_group)}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        generated_paths.append(output_path)

    return sorted(generated_paths)


def _default_source_manifest(group_by: GroupBy) -> Path:
    if group_by == "adapter":
        return _repo_root() / "manifests" / "adapter_exploration" / "llama2_7b_tbh_all_adapters.json"
    if group_by == "architecture":
        return _repo_root() / "manifests" / "architecture_exploration" / "tbh_all_architectures.json"
    raise ValueError(f"Unsupported group_by value: {group_by}")


def _default_output_root(group_by: GroupBy) -> Path:
    return _repo_root() / "runs" / "generated_manifests" / f"leave_one_out_{group_by}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build adapter or architecture leave-one-out manifests from a grouped source manifest."
    )
    parser.add_argument("--group-by", choices=["adapter", "architecture"], required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Grouped source manifest. Defaults to the committed all-adapters/all-architectures manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory to write generated leave-one-out manifests.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the generated manifest root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    group_by = str(args.group_by)
    source_manifest = args.source_manifest or _default_source_manifest(group_by)  # type: ignore[arg-type]
    output_root = args.output_root or _default_output_root(group_by)  # type: ignore[arg-type]
    generated_paths = prepare_manifests(source_manifest, output_root, group_by=group_by)  # type: ignore[arg-type]
    output_root = output_root.expanduser().resolve()
    if args.quiet:
        print(output_root)
    else:
        print(f"Generated {len(generated_paths)} {group_by} leave-one-out manifests in {output_root}")
        for path in generated_paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
