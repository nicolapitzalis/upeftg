from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    resolve_spectral_features,
)
from ...utilities.core.manifest import (
    infer_attack_sample_identities,
    infer_dataset_group_name,
    parse_joint_manifest_json,
    parse_joint_manifest_json_by_model_name,
    parse_single_manifest_json,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
)
from ..contracts import SupervisedFeatureBundle


from .bundles import (
    ResolvedFeatureBundlePaths,
    load_external_spectral_bundle,
    resolve_supervised_feature_bundle_paths,
    supervised_feature_row_count,
)


__all__ = [
    "compact_indices_to_selected_scope",
    "load_external_spectral_bundle",
    "load_features_for_tuning_manifest",
    "load_training_inputs_for_run",
    "prepare_supervised_dataset",
    "supervised_feature_row_count",
]


@dataclass(frozen=True)
class PreparedSupervisedDataset:
    """A manifest-aligned feature dataset ready for experiment policy and splitting."""

    features: np.ndarray | SupervisedFeatureBundle
    items: tuple[Any, ...]
    sample_identities: tuple[Any, ...]
    model_names: tuple[str, ...]
    dataset_group_names: tuple[str, ...]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]
    source_paths: ResolvedFeatureBundlePaths
    selected_manifest_indices: np.ndarray

    @property
    def n_samples(self) -> int:
        return supervised_feature_row_count(self.features)


def prepare_supervised_dataset(
    *,
    feature_spec: Path,
    items: list[Any],
    sample_identities: list[Any],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_attention_granularity: str,
    allow_manifest_subset: bool = True,
) -> PreparedSupervisedDataset:
    """Load and align a feature bundle to manifest items as one coherent dataset."""
    if len(items) != len(sample_identities):
        raise ValueError("Manifest items and sample identities must be row-aligned")

    source_paths = resolve_supervised_feature_bundle_paths(feature_spec)
    features, metadata, warnings, selected_indices = load_external_spectral_bundle(
        feature_file=source_paths.feature_path,
        model_names_file=source_paths.model_names_path,
        metadata_file=source_paths.metadata_path,
        group_mask_file=source_paths.group_mask_path,
        value_mask_file=source_paths.value_mask_path,
        group_names_file=source_paths.group_names_path,
        expected_model_names=[str(item.model_name) for item in items],
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        allow_manifest_subset=allow_manifest_subset,
    )
    selected_items = tuple(items[int(index)] for index in selected_indices.tolist())
    selected_identities = tuple(sample_identities[int(index)] for index in selected_indices.tolist())
    return PreparedSupervisedDataset(
        features=features,
        items=selected_items,
        sample_identities=selected_identities,
        model_names=tuple(str(item.model_name) for item in selected_items),
        dataset_group_names=tuple(infer_dataset_group_name(identity) for identity in selected_identities),
        metadata=dict(metadata),
        warnings=tuple(str(item) for item in warnings),
        source_paths=source_paths,
        selected_manifest_indices=np.asarray(selected_indices, dtype=np.int64),
    )


def load_features_for_tuning_manifest(
    manifest: dict[str, Any],
) -> np.ndarray | SupervisedFeatureBundle:
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("Tuning manifest is missing data configuration")

    feature_loading_mode = str(data.get("feature_loading_mode", "materialized"))
    feature_path_value = data.get("feature_path")
    if not isinstance(feature_path_value, str) or not feature_path_value:
        raise ValueError("Tuning manifest is missing data.feature_path")

    if feature_loading_mode == "materialized":
        return np.asarray(np.load(feature_path_value), dtype=np.float32)

    if feature_loading_mode != "external_source":
        raise ValueError(f"Unsupported data.feature_loading_mode={feature_loading_mode!r}")

    inputs_path = data.get("inputs_path")
    if isinstance(inputs_path, str) and inputs_path:
        prepared_inputs = load_training_inputs_for_run(manifest)
        expected_model_names = [str(x) for x in prepared_inputs["model_names"].tolist()]
    else:
        model_names_path_value = data.get("model_names_path")
        if not isinstance(model_names_path_value, str) or not model_names_path_value:
            raise ValueError("Tuning manifest is missing aligned model names")
        with open(model_names_path_value, "r", encoding="utf-8") as f:
            expected_model_names = [str(x) for x in json.load(f)]

    extractor = manifest.get("extractor")
    if not isinstance(extractor, dict):
        raise ValueError("Tuning manifest is missing extractor configuration")

    extractor_params = extractor.get("params")
    if not isinstance(extractor_params, dict):
        raise ValueError("Tuning manifest is missing extractor.params")

    extractor_metadata = extractor.get("metadata")
    if not isinstance(extractor_metadata, dict):
        raise ValueError("Tuning manifest is missing extractor.metadata")

    external_feature_source = extractor_metadata.get("external_feature_source")
    external_model_names_source = extractor_metadata.get("external_model_names_source")
    external_metadata_source = extractor_metadata.get("external_metadata_source")
    external_group_mask_source = extractor_metadata.get("external_group_mask_source")
    external_value_mask_source = extractor_metadata.get("external_value_mask_source")
    external_group_names_source = extractor_metadata.get("external_group_names_source")
    if not isinstance(external_feature_source, str) or not external_feature_source:
        raise ValueError("Tuning manifest is missing extractor.metadata.external_feature_source")
    if not isinstance(external_model_names_source, str) or not external_model_names_source:
        raise ValueError("Tuning manifest is missing extractor.metadata.external_model_names_source")

    spectral_features_value = extractor_params.get("spectral_features")
    if isinstance(spectral_features_value, list) and spectral_features_value:
        spectral_features = [str(x) for x in spectral_features_value]
    else:
        spectral_features = resolve_spectral_features(None)

    features, _, _, selected_expected_indices = load_external_spectral_bundle(
        feature_file=Path(external_feature_source),
        model_names_file=Path(external_model_names_source),
        metadata_file=(
            Path(external_metadata_source)
            if isinstance(external_metadata_source, str) and external_metadata_source
            else None
        ),
        group_mask_file=(
            Path(external_group_mask_source)
            if isinstance(external_group_mask_source, str) and external_group_mask_source
            else None
        ),
        value_mask_file=(
            Path(external_value_mask_source)
            if isinstance(external_value_mask_source, str) and external_value_mask_source
            else None
        ),
        group_names_file=(
            Path(external_group_names_source)
            if isinstance(external_group_names_source, str) and external_group_names_source
            else None
        ),
        expected_model_names=expected_model_names,
        spectral_features=spectral_features,
        spectral_sv_top_k=int(extractor_params.get("spectral_sv_top_k", 8)),
        spectral_moment_source=str(extractor_params.get("spectral_moment_source", DEFAULT_SPECTRAL_MOMENT_SOURCE)),
        spectral_qv_sum_mode=str(extractor_params.get("spectral_qv_sum_mode", DEFAULT_SPECTRAL_QV_SUM_MODE)),
        spectral_attention_granularity=str(
            extractor_params.get(
                "spectral_attention_granularity",
                DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
            )
        ),
    )
    if int(selected_expected_indices.size) != int(len(expected_model_names)):
        raise ValueError(
            "External feature source no longer covers the prepared tuning-manifest model-name set; "
            "regenerate the supervised prepare stage to refresh the subset of available models"
        )
    return features


def compact_indices_to_selected_scope(
    indices: np.ndarray,
    *,
    selected_expected_indices: np.ndarray,
) -> np.ndarray:
    remap = {int(old): int(new) for new, old in enumerate(selected_expected_indices.tolist())}
    compacted = [remap[int(idx)] for idx in indices.tolist() if int(idx) in remap]
    return np.asarray(compacted, dtype=np.int64)


def load_training_inputs_for_run(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("Tuning manifest is missing data configuration")
    inputs_path = data.get("inputs_path")
    if isinstance(inputs_path, str) and inputs_path:
        with np.load(inputs_path) as bundle:
            return {name: np.asarray(bundle[name]) for name in bundle.files}

    raise ValueError("Tuning manifest is missing data.inputs_path")


def resolve_manifest_path_for_run(manifest: dict[str, Any]) -> Path:
    raw_value = str(manifest["manifest_json"])
    resolved = resolve_manifest_path(raw_value)
    if resolved.exists():
        return resolved

    raw_path = Path(raw_value).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        run_dir = Path(str(manifest["run_dir"])).expanduser().resolve()
        if len(run_dir.parents) >= 3:
            candidates.append((run_dir.parents[2] / raw_path).resolve())
        candidates.append(raw_path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_manifest_items_for_run(manifest: dict[str, Any]) -> list[Any]:
    manifest_path = resolve_manifest_path_for_run(manifest)
    feature_loading_mode = str(manifest["data"].get("feature_loading_mode", "materialized"))
    mode = str(manifest["mode"])

    if feature_loading_mode == "external_source":
        if mode == "joint":
            train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_path)
            return train_items + infer_items
        return parse_single_manifest_json_by_model_name(
            manifest_path=manifest_path,
            section_key="path",
        )

    dataset_root = Path(str(manifest["dataset_root"])).expanduser().resolve()
    if mode == "joint":
        train_items, infer_items = parse_joint_manifest_json(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
        )
        return train_items + infer_items
    return parse_single_manifest_json(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        section_key="path",
    )


def load_dataset_group_names_for_run(manifest: dict[str, Any]) -> list[str]:
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("Tuning manifest is missing data configuration")

    inputs_path = data.get("inputs_path")
    if isinstance(inputs_path, str) and inputs_path:
        inputs = load_training_inputs_for_run(manifest)
        if "dataset_groups" in inputs:
            return [str(name) for name in inputs["dataset_groups"].tolist()]

    dataset_group_names_path = data.get("dataset_group_names_path")
    if isinstance(dataset_group_names_path, str) and dataset_group_names_path:
        with open(dataset_group_names_path, "r", encoding="utf-8") as f:
            raw_names = json.load(f)
        if not isinstance(raw_names, list):
            raise ValueError("Stored dataset-group names must be a JSON list")
        return [str(name) for name in raw_names]

    manifest_items = load_manifest_items_for_run(manifest)
    identities = infer_attack_sample_identities(manifest_items)
    model_names_path = data.get("model_names_path")
    if isinstance(model_names_path, str) and model_names_path:
        with open(model_names_path, "r", encoding="utf-8") as f:
            model_names = [str(x) for x in json.load(f)]
        identity_by_name = {identity.model_name: identity for identity in identities}
        missing = [name for name in model_names if name not in identity_by_name]
        if missing:
            raise ValueError(
                f"Could not align manifest-derived dataset groups with stored model names. Examples: {missing[:5]}"
            )
        return [infer_dataset_group_name(identity_by_name[name]) for name in model_names]

    return [infer_dataset_group_name(identity) for identity in identities]
