from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from ..core.manifest import (
    ManifestItem,
    infer_attack_sample_identities,
    parse_single_manifest_json,
    resolve_manifest_path,
)
from ..core.serialization import json_ready


DATASET_REFERENCE_REPORT_NAME = "dataset_reference_report.json"
DATASET_REFERENCE_STATE_NAME = ".dataset_reference_state.json"
DATASET_REFERENCE_REPORT_VERSION = 1
_MODEL_NAME_SUFFIX_RE = re.compile(r"_label\d+_\d+$")


def default_dataset_reference_report_path(output_dir: Path) -> Path:
    return output_dir / DATASET_REFERENCE_REPORT_NAME


def default_dataset_reference_state_path(output_dir: Path) -> Path:
    return output_dir / DATASET_REFERENCE_STATE_NAME


def _state_path_for_report(path: Path) -> Path:
    return path.with_name(DATASET_REFERENCE_STATE_NAME)


def _public_dataset_reference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "model_index"}


def load_dataset_reference_report(path: Path) -> dict[str, Any]:
    state_path = _state_path_for_report(path)
    target_path = state_path if state_path.exists() else path
    with open(target_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset reference report must be a JSON object: {target_path}")
    return dict(payload)


def write_dataset_reference_report(path: Path, payload: dict[str, Any]) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd().resolve() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        json.dump(json_ready(_public_dataset_reference_payload(payload)), f, indent=2)
    state_path = _state_path_for_report(resolved)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return resolved


def _resolve_path_maybe(raw: Any) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd().resolve() / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_manifest_path_maybe(raw: Any) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return resolve_manifest_path(text)


def _normalize_label(label: Any) -> int | None:
    if label is None:
        return None
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def _label_key(label: int | None) -> str:
    return "unknown" if label is None else str(int(label))


def _add_label_count(dst: dict[str, int], label: int | None) -> None:
    key = _label_key(label)
    dst[key] = int(dst.get(key, 0)) + 1


def _dataset_name_and_path_for_item(*, adapter_path: Path, model_dir: Path, dataset_root: Path) -> tuple[str, str]:
    resolved_root = dataset_root.expanduser()
    if not resolved_root.is_absolute():
        resolved_root = (Path.cwd().resolve() / resolved_root).resolve()
    else:
        resolved_root = resolved_root.resolve()

    try:
        rel = model_dir.resolve().relative_to(resolved_root)
    except ValueError:
        dataset_path = str(model_dir.resolve().parent)
        dataset_name = model_dir.resolve().parent.name or model_dir.resolve().name
        return dataset_name, dataset_path

    dataset_name = rel.parts[0] if rel.parts else model_dir.resolve().parent.name
    dataset_path = str((resolved_root / dataset_name).resolve())
    return dataset_name, dataset_path


def _subset_name_from_model_name(model_name: str) -> str:
    stripped = _MODEL_NAME_SUFFIX_RE.sub("", str(model_name))
    return stripped or str(model_name)


def _load_model_names_and_labels(
    *,
    model_names_path: Path,
    labels_path: Path | None,
) -> tuple[list[str], dict[str, int | None]]:
    with open(model_names_path, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]

    labels_by_name: dict[str, int | None] = {name: None for name in model_names}
    if labels_path is not None and labels_path.exists():
        import numpy as np

        labels = np.asarray(np.load(labels_path), dtype=np.int32)
        if int(labels.shape[0]) == len(model_names):
            for model_name, label in zip(model_names, labels.tolist()):
                labels_by_name[model_name] = int(label)
    return model_names, labels_by_name


def _build_incomplete_payload_from_model_names(
    *,
    model_names: list[str],
    labels_by_name: dict[str, int | None],
    artifact_kind: str,
    artifact_model_count: int | None,
    provenance_gaps: list[str],
    source_artifacts: list[str],
) -> dict[str, Any]:
    pseudo_items = [
        ManifestItem(
            raw_entry=str(model_name),
            model_dir=Path("unknown") / _subset_name_from_model_name(str(model_name)) / str(model_name),
            adapter_path=Path("unknown") / _subset_name_from_model_name(str(model_name)) / str(model_name),
            model_name=str(model_name),
            label=labels_by_name.get(model_name),
        )
        for model_name in model_names
    ]
    identities = infer_attack_sample_identities(pseudo_items)
    identity_by_model_name = {identity.model_name: identity for identity in identities}
    model_index = {
        str(model_name): {
            "dataset_name": _subset_name_from_model_name(model_name),
            "dataset_path": str(Path("unknown") / _subset_name_from_model_name(model_name)),
            "label": labels_by_name.get(model_name),
            "subset_name": identity_by_model_name.get(model_name).subset_name
            if model_name in identity_by_model_name
            else _subset_name_from_model_name(model_name),
            "model_family": identity_by_model_name.get(model_name).model_family
            if model_name in identity_by_model_name
            else "unknown",
            "attack_name": identity_by_model_name.get(model_name).attack_name
            if model_name in identity_by_model_name
            else "unknown",
        }
        for model_name in model_names
    }
    return _finalize_payload(
        artifact_kind=artifact_kind,
        model_index=model_index,
        artifact_model_count=artifact_model_count if artifact_model_count is not None else len(model_names),
        manifest_json=None,
        dataset_root=None,
        source_artifacts=source_artifacts,
        provenance_gaps=provenance_gaps,
        is_complete=False,
    )


def _merge_model_indices(payloads: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    merged: dict[str, dict[str, Any]] = {}
    gaps: list[str] = []
    for payload in payloads:
        model_index = payload.get("model_index", {})
        if not isinstance(model_index, dict):
            gaps.append(f"Missing model_index in source payload for artifact_kind={payload.get('artifact_kind')}")
            continue
        for model_name, raw_entry in model_index.items():
            entry = dict(raw_entry) if isinstance(raw_entry, dict) else {}
            if model_name not in merged:
                merged[model_name] = entry
                continue
            existing = merged[model_name]
            for key in [
                "dataset_name",
                "dataset_path",
                "subset_name",
                "model_family",
                "attack_name",
            ]:
                left = existing.get(key)
                right = entry.get(key)
                if left is not None and right is not None and left != right:
                    gaps.append(
                        f"Conflicting {key} for model '{model_name}': {left!r} vs {right!r}"
                    )
            left_label = _normalize_label(existing.get("label"))
            right_label = _normalize_label(entry.get("label"))
            if left_label is None and right_label is not None:
                existing["label"] = right_label
            elif left_label is not None and right_label is not None and left_label != right_label:
                gaps.append(
                    f"Conflicting label for model '{model_name}': {left_label!r} vs {right_label!r}"
                )
    return merged, gaps


def _build_dataset_groups(model_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for model_name in sorted(model_index):
        entry = dict(model_index[model_name])
        dataset_name = str(entry.get("dataset_name") or "unknown")
        dataset_path = str(entry.get("dataset_path") or "unknown")
        key = (dataset_path, dataset_name)
        group = grouped.setdefault(
            key,
            {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "sample_count": 0,
                "label_counts": {},
                "model_family_counts": {},
                "attack_name_counts": {},
                "subset_names": set(),
            },
        )
        group["sample_count"] = int(group["sample_count"]) + 1
        _add_label_count(group["label_counts"], _normalize_label(entry.get("label")))

        model_family = str(entry.get("model_family") or "unknown")
        attack_name = str(entry.get("attack_name") or "unknown")
        subset_name = str(entry.get("subset_name") or "unknown")

        group["model_family_counts"][model_family] = int(group["model_family_counts"].get(model_family, 0)) + 1
        group["attack_name_counts"][attack_name] = int(group["attack_name_counts"].get(attack_name, 0)) + 1
        group["subset_names"].add(subset_name)

    finalized: list[dict[str, Any]] = []
    for (_dataset_path, _dataset_name), group in sorted(grouped.items(), key=lambda item: item[0][1]):
        subset_names = sorted(str(x) for x in group.pop("subset_names"))
        group["subset_count"] = int(len(subset_names))
        group["subset_names_preview"] = subset_names[:10]
        group["model_family_counts"] = {
            key: int(value) for key, value in sorted(group["model_family_counts"].items())
        }
        group["attack_name_counts"] = {
            key: int(value) for key, value in sorted(group["attack_name_counts"].items())
        }
        group["label_counts"] = {key: int(value) for key, value in sorted(group["label_counts"].items())}
        finalized.append(group)
    return finalized


def _build_summary(
    *,
    dataset_groups: list[dict[str, Any]],
    resolved_model_count: int,
    artifact_model_count: int | None,
    is_complete: bool,
    provenance_gaps: list[str],
) -> str:
    if artifact_model_count is None:
        coverage = f"{resolved_model_count}"
    else:
        coverage = f"{resolved_model_count}/{artifact_model_count}"

    if not dataset_groups:
        if provenance_gaps:
            return (
                f"Could not fully resolve dataset references for this artifact; "
                f"resolved 0/{artifact_model_count or 0} models."
            )
        return "No dataset references were resolved for this artifact."

    preview_chunks: list[str] = []
    for group in dataset_groups[:4]:
        labels = ", ".join(f"{key}={value}" for key, value in sorted(group.get("label_counts", {}).items()))
        if labels:
            preview_chunks.append(
                f"{group['dataset_name']} ({int(group['sample_count'])} models; labels {labels})"
            )
        else:
            preview_chunks.append(f"{group['dataset_name']} ({int(group['sample_count'])} models)")
    if len(dataset_groups) > 4:
        preview_chunks.append(f"... +{len(dataset_groups) - 4} more")

    status = "References" if is_complete else "Partially resolves"
    return (
        f"{status} {len(dataset_groups)} dataset group(s) covering {coverage} models: "
        + "; ".join(preview_chunks)
    )


def _finalize_payload(
    *,
    artifact_kind: str,
    model_index: dict[str, dict[str, Any]],
    artifact_model_count: int | None,
    manifest_json: Path | None,
    dataset_root: Path | None,
    source_artifacts: list[str],
    provenance_gaps: list[str],
    is_complete: bool,
) -> dict[str, Any]:
    dataset_groups = _build_dataset_groups(model_index)
    resolved_model_count = int(len(model_index))
    if artifact_model_count is None:
        artifact_model_count = resolved_model_count
    if resolved_model_count != int(artifact_model_count):
        is_complete = False
    normalized_gaps = [str(gap) for gap in provenance_gaps if str(gap).strip()]
    if normalized_gaps:
        is_complete = False

    aggregate_label_counts: dict[str, int] = {}
    for entry in model_index.values():
        _add_label_count(aggregate_label_counts, _normalize_label(entry.get("label")))

    return {
        "report_schema_version": DATASET_REFERENCE_REPORT_VERSION,
        "report_type": "dataset_reference_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": str(artifact_kind),
        "artifact_model_count": int(artifact_model_count),
        "resolved_model_count": resolved_model_count,
        "dataset_group_count": int(len(dataset_groups)),
        "is_complete": bool(is_complete),
        "provenance_gaps": normalized_gaps,
        "manifest_json": str(manifest_json) if manifest_json is not None else None,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "source_artifacts": [str(x) for x in source_artifacts],
        "label_counts": {key: int(value) for key, value in sorted(aggregate_label_counts.items())},
        "summary": _build_summary(
            dataset_groups=dataset_groups,
            resolved_model_count=resolved_model_count,
            artifact_model_count=int(artifact_model_count),
            is_complete=bool(is_complete),
            provenance_gaps=normalized_gaps,
        ),
        "dataset_groups": dataset_groups,
        "model_index": {key: model_index[key] for key in sorted(model_index)},
    }


def build_dataset_reference_payload_from_items(
    *,
    items: list[Any],
    artifact_kind: str,
    manifest_json: Path | None = None,
    dataset_root: Path | None = None,
    artifact_model_count: int | None = None,
    source_artifacts: list[str] | None = None,
    provenance_gaps: list[str] | None = None,
    is_complete: bool = True,
) -> dict[str, Any]:
    if dataset_root is None:
        raise ValueError("dataset_root is required to build a dataset reference payload from manifest items")

    identities = infer_attack_sample_identities(items)
    identity_by_model_name = {identity.model_name: identity for identity in identities}
    model_index: dict[str, dict[str, Any]] = {}
    for item in items:
        identity = identity_by_model_name.get(item.model_name)
        dataset_name, dataset_path = _dataset_name_and_path_for_item(
            adapter_path=item.adapter_path,
            model_dir=item.model_dir,
            dataset_root=dataset_root,
        )
        model_index[str(item.model_name)] = {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "label": _normalize_label(item.label),
            "subset_name": identity.subset_name if identity is not None else item.model_dir.parent.name,
            "model_family": identity.model_family if identity is not None else "unknown",
            "attack_name": identity.attack_name if identity is not None else "unknown",
        }

    return _finalize_payload(
        artifact_kind=artifact_kind,
        model_index=model_index,
        artifact_model_count=artifact_model_count if artifact_model_count is not None else len(items),
        manifest_json=manifest_json,
        dataset_root=dataset_root,
        source_artifacts=list(source_artifacts or ([str(manifest_json)] if manifest_json is not None else [])),
        provenance_gaps=list(provenance_gaps or []),
        is_complete=bool(is_complete),
    )


def merge_dataset_reference_payloads(
    *,
    payloads: list[dict[str, Any]],
    artifact_kind: str,
    artifact_model_count: int | None = None,
    source_artifacts: list[str] | None = None,
    provenance_gaps: list[str] | None = None,
) -> dict[str, Any]:
    merged_index, merge_gaps = _merge_model_indices(payloads)
    source_paths = list(source_artifacts or [])
    declared_manifest = None
    declared_dataset_root = None
    complete = True
    if artifact_model_count is None:
        artifact_model_count = len(merged_index)
    for payload in payloads:
        complete = complete and bool(payload.get("is_complete", False))
    return _finalize_payload(
        artifact_kind=artifact_kind,
        model_index=merged_index,
        artifact_model_count=artifact_model_count,
        manifest_json=declared_manifest,
        dataset_root=declared_dataset_root,
        source_artifacts=source_paths,
        provenance_gaps=[*merge_gaps, *(provenance_gaps or [])],
        is_complete=complete,
    )


def _resolve_extraction_payload_from_run_dir(run_dir: Path) -> dict[str, Any]:
    report_path = default_dataset_reference_report_path(run_dir / "reports")
    if report_path.exists():
        return load_dataset_reference_report(report_path)

    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json for feature extraction run: {run_dir}")
    with open(run_config_path, "r", encoding="utf-8") as f:
        run_config = json.load(f)
    manifest_json = _resolve_manifest_path_maybe(run_config.get("manifest_json"))
    dataset_root = _resolve_path_maybe(run_config.get("dataset_root"))
    model_names_candidates = sorted((run_dir / "features").glob("*_model_names.json"))
    labels_candidates = sorted((run_dir / "features").glob("*_labels.npy"))
    model_names_path = model_names_candidates[0] if model_names_candidates else None
    labels_path = labels_candidates[0] if labels_candidates else None

    if manifest_json is None or dataset_root is None or not manifest_json.exists():
        if model_names_path is None:
            raise ValueError(f"Feature extraction run missing manifest_json/dataset_root: {run_dir}")
        model_names, labels_by_name = _load_model_names_and_labels(
            model_names_path=model_names_path,
            labels_path=labels_path,
        )
        return _build_incomplete_payload_from_model_names(
            model_names=model_names,
            labels_by_name=labels_by_name,
            artifact_kind="feature_extract",
            artifact_model_count=len(model_names),
            provenance_gaps=[
                f"Could not load manifest provenance for feature extraction run {run_dir}; "
                f"manifest_json={manifest_json}"
            ],
            source_artifacts=[str(run_config_path)],
        )

    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=dataset_root,
        section_key="path",
    )
    return build_dataset_reference_payload_from_items(
        items=items,
        artifact_kind="feature_extract",
        manifest_json=manifest_json,
        dataset_root=dataset_root,
        artifact_model_count=len(items),
        source_artifacts=[str(run_config_path), str(manifest_json)],
    )


def _resolve_merge_payload_from_report(
    report_path: Path,
    memo: dict[str, dict[str, Any]],
    *,
    prefer_existing_report: bool,
) -> dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if not isinstance(report, dict):
        raise ValueError(f"Merge report must be a JSON object: {report_path}")

    artifact_model_count = report.get("n_rows")
    if artifact_model_count is None:
        output = report.get("output", {})
        if isinstance(output, dict):
            feature_path = _resolve_path_maybe(output.get("feature_path"))
            if feature_path is not None and feature_path.exists():
                # Avoid importing numpy here just to count rows from features.
                pass

    inputs = report.get("inputs")
    if isinstance(inputs, list) and inputs:
        source_payloads: list[dict[str, Any]] = []
        source_artifacts: list[str] = []
        source_gaps: list[str] = []
        for entry in inputs:
            if not isinstance(entry, dict):
                continue
            feature_path = _resolve_path_maybe(entry.get("feature_path"))
            if feature_path is None:
                continue
            source_artifacts.append(str(feature_path))
            try:
                source_payloads.append(
                    resolve_dataset_reference_payload_for_artifact(
                        feature_path,
                        memo=memo,
                        prefer_existing_report=prefer_existing_report,
                    )
                )
            except Exception as exc:
                source_gaps.append(f"Could not resolve source artifact {feature_path}: {exc}")
        kind = "finalize_schema_group_merge" if report_path.name == "schema_group_merge_report.json" else "merge"
        return merge_dataset_reference_payloads(
            payloads=source_payloads,
            artifact_kind=kind,
            artifact_model_count=(
                int(artifact_model_count)
                if artifact_model_count is not None
                else None
            ),
            source_artifacts=source_artifacts,
            provenance_gaps=source_gaps,
        )

    manifest_json = _resolve_manifest_path_maybe(report.get("manifest_json"))
    dataset_root = _resolve_path_maybe(report.get("dataset_root"))
    if manifest_json is None or dataset_root is None or not manifest_json.exists():
        output = report.get("output", {})
        model_names_path = None
        labels_path = None
        if isinstance(output, dict):
            model_names_path = _resolve_path_maybe(output.get("model_names_path"))
            labels_path = _resolve_path_maybe(output.get("labels_path"))
        if model_names_path is None or not model_names_path.exists():
            fallback_model_names_path = report_path.parent / "spectral_model_names.json"
            if fallback_model_names_path.exists():
                model_names_path = fallback_model_names_path
        if labels_path is None or (labels_path is not None and not labels_path.exists()):
            fallback_labels_path = report_path.parent / "spectral_labels.npy"
            if fallback_labels_path.exists():
                labels_path = fallback_labels_path
        if model_names_path is not None and model_names_path.exists():
            model_names, labels_by_name = _load_model_names_and_labels(
                model_names_path=model_names_path,
                labels_path=labels_path if labels_path is not None and labels_path.exists() else None,
            )
            return _build_incomplete_payload_from_model_names(
                model_names=model_names,
                labels_by_name=labels_by_name,
                artifact_kind="merge_spectral_shards",
                artifact_model_count=(
                    int(artifact_model_count)
                    if artifact_model_count is not None
                    else len(model_names)
                ),
                provenance_gaps=[
                    f"Could not load manifest provenance for merge report {report_path}; "
                    f"manifest_json={manifest_json}"
                ],
                source_artifacts=[str(report_path)],
            )
        raise ValueError(f"Cannot reconstruct dataset references from merge report: {report_path}")
    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=dataset_root,
        section_key="path",
    )
    provenance_gaps: list[str] = []
    is_complete = True
    if bool(report.get("merged_with_existing_output")):
        provenance_gaps.append(
            "This merge consumed a pre-existing output in place, but the prior source provenance is not "
            "fully recoverable from the legacy artifacts alone."
        )
        is_complete = False
    return build_dataset_reference_payload_from_items(
        items=items,
        artifact_kind="merge_spectral_shards",
        manifest_json=manifest_json,
        dataset_root=dataset_root,
        artifact_model_count=(
            int(artifact_model_count)
            if artifact_model_count is not None
            else len(items)
        ),
        source_artifacts=[str(report_path), str(manifest_json)],
        provenance_gaps=provenance_gaps,
        is_complete=is_complete,
    )


def resolve_dataset_reference_payload_for_artifact(
    artifact_path: Path,
    *,
    memo: dict[str, dict[str, Any]] | None = None,
    prefer_existing_report: bool = True,
) -> dict[str, Any]:
    cache = memo if memo is not None else {}
    resolved_input = artifact_path.expanduser()
    if not resolved_input.is_absolute():
        resolved_input = (Path.cwd().resolve() / resolved_input).resolve()
    else:
        resolved_input = resolved_input.resolve()

    cache_key = str(resolved_input)
    if cache_key in cache:
        return cache[cache_key]

    if resolved_input.is_file() and resolved_input.name == DATASET_REFERENCE_REPORT_NAME:
        payload = load_dataset_reference_report(resolved_input)
        cache[cache_key] = payload
        return payload

    if resolved_input.is_file() and resolved_input.name.endswith("_features.npy"):
        parent = resolved_input.parent
        if parent.name == "features":
            payload = _resolve_extraction_payload_from_run_dir(parent.parent)
        else:
            payload = resolve_dataset_reference_payload_for_artifact(parent, memo=cache)
        cache[cache_key] = payload
        return payload

    if resolved_input.is_dir():
        direct_report = default_dataset_reference_report_path(
            resolved_input / "reports" if (resolved_input / "run_config.json").exists() else resolved_input
        )
        if prefer_existing_report and direct_report.exists():
            payload = load_dataset_reference_report(direct_report)
            if isinstance(payload.get("model_index"), dict):
                cache[cache_key] = payload
                return payload

        if (resolved_input / "run_config.json").exists():
            payload = _resolve_extraction_payload_from_run_dir(resolved_input)
            cache[cache_key] = payload
            return payload

        for report_name in ["schema_group_merge_report.json", "spectral_merge_report.json"]:
            merge_report_path = resolved_input / report_name
            if merge_report_path.exists():
                payload = _resolve_merge_payload_from_report(
                    merge_report_path,
                    cache,
                    prefer_existing_report=prefer_existing_report,
                )
                cache[cache_key] = payload
                return payload

    if resolved_input.is_file() and resolved_input.name.endswith("_merge_report.json"):
        payload = _resolve_merge_payload_from_report(
            resolved_input,
            cache,
            prefer_existing_report=prefer_existing_report,
        )
        cache[cache_key] = payload
        return payload

    raise FileNotFoundError(f"Could not resolve dataset reference provenance for artifact: {resolved_input}")
