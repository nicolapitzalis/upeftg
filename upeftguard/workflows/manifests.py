from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ..supervised.tasks import labels_from_items, resolve_supervised_task_spec
from ..supervised.validation.splits import (
    build_single_manifest_folder_label_split,
    build_single_manifest_stratified_split,
)
from ..utilities.core.manifest import (
    CV_ALWAYS_TRAIN_SECTION_KEY,
    infer_attack_sample_identities,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
    validate_disjoint,
)


@dataclass(frozen=True)
class ResolvedExperimentManifests:
    full_manifest: Path
    train_manifest: Path
    inference_manifest: Path | None
    split_summary: dict[str, Any] | None


def _write_manifest(path: Path, entries: list[str]) -> Path:
    if not entries:
        raise ValueError(f"Cannot write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"path": [str(entry) for entry in entries]}, handle, indent=2)
        handle.write("\n")
    return path


def _copy_manifest(source: Path, destination: Path) -> Path:
    source = resolve_manifest_path(source)
    if not source.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination:
        shutil.copyfile(source, destination)
    return destination


def materialize_manifest_snapshot(source: Path, destination: Path) -> Path:
    """Copy one resolved workflow manifest to its canonical stage input path."""
    return _copy_manifest(source, destination)


def _items(path: Path) -> list[Any]:
    resolved = resolve_manifest_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "path" not in payload:
        raise ValueError(f"Workflow manifests must contain one 'path' section: {resolved}")
    return parse_single_manifest_json_by_model_name(
        manifest_path=resolved,
        section_key="path",
    )


def _split_complete_manifest(
    *,
    manifest_json: Path,
    train_split: int,
    random_state: int,
    split_by_folder: bool,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if int(train_split) < 1 or int(train_split) > 99:
        raise ValueError("--train-split must be in the range [1, 99] when creating inference data")
    resolved_manifest = resolve_manifest_path(manifest_json)
    with resolved_manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and CV_ALWAYS_TRAIN_SECTION_KEY in payload:
        raise ValueError(
            f"{CV_ALWAYS_TRAIN_SECTION_KEY!r} cannot be combined with --train-split; "
            "use explicit training and inference manifests"
        )
    items = _items(manifest_json)
    identities = infer_attack_sample_identities(items)
    task_spec = resolve_supervised_task_spec(
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
    )
    labels, known, _ = labels_from_items(
        items,
        task_spec=task_spec,
        sample_identities=identities,
    )
    if not bool(np.all(known)):
        raise ValueError("--train-split requires labels for every sample in the complete manifest")
    if split_by_folder:
        train_indices, infer_indices, warnings, summary = build_single_manifest_folder_label_split(
            items=items,
            labels=labels,
            train_split_percent=int(train_split),
            random_state=int(random_state),
        )
    else:
        train_indices, infer_indices, warnings, summary = build_single_manifest_stratified_split(
            labels=labels,
            train_split_percent=int(train_split),
            random_state=int(random_state),
        )
    summary = dict(summary)
    summary["warnings"] = [str(value) for value in warnings]
    return (
        [items[int(index)] for index in train_indices.tolist()],
        [items[int(index)] for index in infer_indices.tolist()],
        summary,
    )


def resolve_full_manifests(
    *,
    output_dir: Path,
    manifest_json: Path | None,
    train_manifest_json: Path | None,
    inference_manifest_json: Path | None,
    train_split: int | None,
    random_state: int,
    split_by_folder: bool,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
) -> ResolvedExperimentManifests:
    output_dir = Path(output_dir).expanduser().resolve()
    explicit_train_manifest_source: Path | None = None
    if manifest_json is not None:
        if train_manifest_json is not None or inference_manifest_json is not None:
            raise ValueError("--manifest-json cannot be combined with --train-manifest-json or --infer-manifest-json")
        if train_split is None:
            raise ValueError("experiment full requires --train-split with --manifest-json")
        full_path = _copy_manifest(manifest_json, output_dir / "full.json")
        train_items, infer_items, split_summary = _split_complete_manifest(
            manifest_json=full_path,
            train_split=int(train_split),
            random_state=int(random_state),
            split_by_folder=bool(split_by_folder),
            task_mode=task_mode,
            multiclass_attack_names=multiclass_attack_names,
        )
    else:
        if train_split is not None:
            raise ValueError("--train-split can only be used with --manifest-json")
        if train_manifest_json is None or inference_manifest_json is None:
            raise ValueError(
                "experiment full requires either --manifest-json with --train-split or both "
                "--train-manifest-json and --infer-manifest-json"
            )
        explicit_train_manifest_source = resolve_manifest_path(train_manifest_json)
        train_items = _items(explicit_train_manifest_source)
        infer_items = _items(inference_manifest_json)
        validate_disjoint(train_items, infer_items)
        split_summary = None
        full_path = _write_manifest(
            output_dir / "full.json",
            [item.raw_entry for item in [*train_items, *infer_items]],
        )

    train_path = (
        _copy_manifest(explicit_train_manifest_source, output_dir / "train.json")
        if explicit_train_manifest_source is not None
        else _write_manifest(
            output_dir / "train.json",
            [item.raw_entry for item in train_items],
        )
    )
    inference_path = _write_manifest(
        output_dir / "inference.json",
        [item.raw_entry for item in infer_items],
    )
    return ResolvedExperimentManifests(
        full_manifest=full_path,
        train_manifest=train_path,
        inference_manifest=inference_path,
        split_summary=split_summary,
    )


def resolve_training_manifests(
    *,
    output_dir: Path,
    manifest_json: Path,
    train_split: int | None,
    random_state: int,
    split_by_folder: bool,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
) -> ResolvedExperimentManifests:
    output_dir = Path(output_dir).expanduser().resolve()
    if train_split is None:
        train_items = _items(manifest_json)
        if not train_items:
            raise ValueError("Training manifest is empty")
        train_path = _copy_manifest(manifest_json, output_dir / "train.json")
        return ResolvedExperimentManifests(
            full_manifest=train_path,
            train_manifest=train_path,
            inference_manifest=None,
            split_summary=None,
        )

    full_path = _copy_manifest(manifest_json, output_dir / "full.json")
    train_items, infer_items, split_summary = _split_complete_manifest(
        manifest_json=full_path,
        train_split=int(train_split),
        random_state=int(random_state),
        split_by_folder=bool(split_by_folder),
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
    )
    train_path = _write_manifest(
        output_dir / "train.json",
        [item.raw_entry for item in train_items],
    )
    inference_path = _write_manifest(
        output_dir / "inference.json",
        [item.raw_entry for item in infer_items],
    )
    return ResolvedExperimentManifests(
        full_manifest=full_path,
        train_manifest=train_path,
        inference_manifest=inference_path,
        split_summary=split_summary,
    )
