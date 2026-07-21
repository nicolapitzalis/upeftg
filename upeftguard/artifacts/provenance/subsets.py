"""Feature ownership and leaf-source provenance traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...contracts.spectral import feature_block_name as _feature_block_name
from ...features.spectral import provenance_source_feature_group
from ..io import load_feature_table as _load_feature_table
from ..tables import unique_index_by_name as _unique_index_by_name
from .datasets import resolve_dataset_reference_payload_for_artifact
from ..paths import resolve_existing_companion_path as _resolve_existing_companion_path
from ..schema import feature_group_for_feature_name as _feature_group_for_feature_name


@dataclass(frozen=True)
class LeafFeatureSource:
    feature_path: Path
    model_names: frozenset[str]
    feature_names: tuple[str, ...]


def _resolve_companion_paths(feature_path: Path) -> tuple[Path, Path, Path]:
    model_names_path = _resolve_existing_companion_path(
        feature_path,
        "_model_names.json",
        required=True,
    )
    labels_path = _resolve_existing_companion_path(
        feature_path,
        "_labels.npy",
        required=False,
    )
    metadata_path = _resolve_existing_companion_path(
        feature_path,
        "_metadata.json",
        required=False,
    )
    return model_names_path, labels_path, metadata_path


def _load_table_from_feature_path(feature_path: Path):
    resolved_feature_path = feature_path.expanduser()
    if not resolved_feature_path.is_absolute():
        resolved_feature_path = (Path.cwd().resolve() / resolved_feature_path).resolve()
    else:
        resolved_feature_path = resolved_feature_path.resolve()
    model_names_path, labels_path, metadata_path = _resolve_companion_paths(resolved_feature_path)
    return _load_feature_table(
        source=str(resolved_feature_path),
        feature_path=resolved_feature_path,
        model_names_path=model_names_path,
        labels_path=labels_path,
        metadata_path=metadata_path,
        context=f"feature subset input {resolved_feature_path}",
    )


def _resolve_child_feature_path(raw_path: Any, *, parent_feature_path: Path) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    parent_relative = (parent_feature_path.parent / candidate).resolve()
    if parent_relative.exists():
        return parent_relative
    return (Path.cwd().resolve() / candidate).resolve()


def _collect_leaf_feature_sources(
    feature_path: Path,
    *,
    memo: dict[Path, dict[Path, LeafFeatureSource]],
    active: set[Path],
) -> dict[Path, LeafFeatureSource]:
    resolved_feature_path = feature_path.resolve()
    if resolved_feature_path in memo:
        return memo[resolved_feature_path]
    if resolved_feature_path in active:
        raise ValueError(f"Detected recursive merge-source reference while resolving {resolved_feature_path}")

    active.add(resolved_feature_path)
    try:
        table = _load_table_from_feature_path(resolved_feature_path)
        raw_sources = table.metadata.get("merge_source_feature_files")
        source_paths = (
            [
                _resolve_child_feature_path(raw_path, parent_feature_path=resolved_feature_path)
                for raw_path in raw_sources
                if str(raw_path).strip()
            ]
            if isinstance(raw_sources, list)
            else []
        )

        if not source_paths:
            payload = resolve_dataset_reference_payload_for_artifact(resolved_feature_path)
            raw_model_index = payload.get("model_index")
            if isinstance(raw_model_index, dict):
                model_names = frozenset(str(name) for name in raw_model_index)
            else:
                model_names = frozenset(str(name) for name in table.model_names)
            result = {
                resolved_feature_path: LeafFeatureSource(
                    feature_path=resolved_feature_path,
                    model_names=model_names,
                    feature_names=tuple(str(name) for name in table.feature_names),
                )
            }
        else:
            result: dict[Path, LeafFeatureSource] = {}
            for source_path in source_paths:
                result.update(
                    _collect_leaf_feature_sources(
                        source_path,
                        memo=memo,
                        active=active,
                    )
                )

        memo[resolved_feature_path] = result
        return result
    finally:
        active.discard(resolved_feature_path)


def _load_source_payload(feature_path: Path) -> dict[str, Any]:
    payload = resolve_dataset_reference_payload_for_artifact(feature_path)
    raw_model_index = payload.get("model_index")
    if not isinstance(raw_model_index, dict) or not raw_model_index:
        raise ValueError(
            f"Feature subset export requires dataset-reference state with a non-empty model_index for {feature_path}"
        )
    return payload


def _resolve_provenance_feature_names(
    *,
    root_feature_path: Path,
    root_feature_names: list[str],
    selected_model_names: list[str],
) -> tuple[list[str], list[Path]]:
    owned_feature_names_by_model, matched_leaf_paths = _resolve_model_owned_feature_names(
        root_feature_path=root_feature_path,
        root_feature_names=root_feature_names,
        selected_model_names=selected_model_names,
    )
    required_feature_names = {
        feature_name
        for owned_feature_names in owned_feature_names_by_model.values()
        for feature_name in owned_feature_names
    }
    available_feature_names = [name for name in root_feature_names if name in required_feature_names]
    if not available_feature_names:
        raise ValueError("No provenance-backed columns were available for the selected rows")
    return available_feature_names, matched_leaf_paths


def _resolve_model_owned_feature_names(
    *,
    root_feature_path: Path,
    root_feature_names: list[str],
    selected_model_names: list[str],
) -> tuple[dict[str, list[str]], list[Path]]:
    _unique_index_by_name(root_feature_names, context=str(root_feature_path), entity="feature names")

    leaf_sources = _collect_leaf_feature_sources(
        root_feature_path,
        memo={},
        active=set(),
    )
    selected_model_name_set = set(selected_model_names)
    model_owned_feature_names: dict[str, set[str]] = {}
    covered_model_names: set[str] = set()
    matched_leaf_paths: list[Path] = []

    for leaf_path in sorted(leaf_sources):
        leaf = leaf_sources[leaf_path]
        overlap = selected_model_name_set & set(leaf.model_names)
        if not overlap:
            continue
        matched_leaf_paths.append(leaf.feature_path)
        covered_model_names.update(overlap)
        owned_feature_names = {str(name) for name in leaf.feature_names}
        for model_name in overlap:
            model_owned_feature_names.setdefault(str(model_name), set()).update(owned_feature_names)

    uncovered = sorted(selected_model_name_set - covered_model_names)
    if uncovered:
        preview = ", ".join(uncovered[:5])
        raise ValueError(
            "Could not resolve provenance-owning source feature files for "
            f"{len(uncovered)} selected model(s). Examples: {preview}"
        )

    root_feature_name_set = set(root_feature_names)
    ordered_feature_names_by_model: dict[str, list[str]] = {}
    for model_name in selected_model_names:
        owned_feature_name_set = model_owned_feature_names.get(str(model_name), set())
        if not owned_feature_name_set:
            raise ValueError(
                f"Resolved provenance leaf sources but no feature names were discovered for model '{model_name}'"
            )
        missing_in_root = sorted(name for name in owned_feature_name_set if name not in root_feature_name_set)
        if missing_in_root:
            ordered_owned_feature_names = _resolved_owned_feature_names_by_equivalent_groups(
                root_feature_names=root_feature_names,
                owned_feature_name_set=owned_feature_name_set,
            )
            if not ordered_owned_feature_names:
                preview = ", ".join(missing_in_root[:5])
                raise ValueError(
                    "Provenance source feature names are not present in the requested feature bundle. "
                    f"Examples: {preview}"
                )
        else:
            ordered_owned_feature_names = [name for name in root_feature_names if name in owned_feature_name_set]
        if not ordered_owned_feature_names:
            raise ValueError(f"No provenance-backed columns were available for model '{model_name}'")
        ordered_feature_names_by_model[str(model_name)] = ordered_owned_feature_names

    return ordered_feature_names_by_model, matched_leaf_paths


def _resolved_owned_feature_names_by_equivalent_groups(
    *,
    root_feature_names: list[str],
    owned_feature_name_set: set[str],
) -> list[str]:
    owned_blocks: set[str] = set()
    owned_block_groups: set[tuple[str, str]] = set()
    for feature_name in owned_feature_name_set:
        block_name = _feature_block_name(feature_name)
        owned_blocks.add(block_name)
        group = _feature_group_for_feature_name(feature_name)
        if group is None:
            continue
        owned_block_groups.add((block_name, group))

    resolved: list[str] = []
    for feature_name in root_feature_names:
        block_name = _feature_block_name(feature_name)
        if block_name not in owned_blocks:
            continue
        group = _feature_group_for_feature_name(feature_name)
        if group is None:
            continue
        if group == "block_rank":
            resolved.append(feature_name)
            continue
        source_group = provenance_source_feature_group(group)
        if source_group is None:
            continue
        if (block_name, source_group) in owned_block_groups:
            resolved.append(feature_name)
    return resolved
