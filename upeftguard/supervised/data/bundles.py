"""External feature-bundle path resolution, loading, and representation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.spectral import feature_block_name
from ...artifacts.paths import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    resolve_existing_companion_path as _resolve_existing_companion_path,
    resolve_feature_extract_root as _resolve_feature_extract_root,
    resolve_input_feature_path as _resolve_input_feature_path,
)
from ...artifacts.metadata.spectral import load_spectral_metadata
from ...features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    build_spectral_feature_names,
    resolve_spectral_attention_granularity,
    resolve_spectral_features,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
    sanitize_spectral_metadata,
    spectral_block_lora_dims_by_block,
    spectral_extractor_params,
)
from ..contracts import (
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    SupervisedFeatureBundle,
    TABULAR_SPECTRAL_REPRESENTATION_KIND,
)
from ..models.registry import CNN_1D_DANN_MODEL_NAME, registered_models, supported_representation_kinds


GROUP_MASK_SUFFIX = "_group_mask.npy"
VALUE_MASK_SUFFIX = "_value_mask.npy"
GROUP_NAMES_SUFFIX = "_group_names.json"


@dataclass(frozen=True)
class ResolvedFeatureBundlePaths:
    feature_path: Path
    model_names_path: Path
    metadata_path: Path | None
    group_mask_path: Path | None
    value_mask_path: Path | None
    group_names_path: Path | None


def resolve_supervised_feature_bundle_paths(
    feature_spec: Path,
    *,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
) -> ResolvedFeatureBundlePaths:
    resolved_feature_root = _resolve_feature_extract_root(feature_root)
    candidate = feature_spec.expanduser()
    local_candidate = candidate if candidate.is_absolute() else (Path.cwd().resolve() / candidate)
    resolved_local_candidate = local_candidate.resolve()

    if resolved_local_candidate.exists():
        resolved_feature_path = (
            resolved_local_candidate / "spectral_features.npy"
            if resolved_local_candidate.is_dir()
            else resolved_local_candidate
        )
    else:
        resolved_feature_path = _resolve_input_feature_path(
            feature_spec,
            feature_root=resolved_feature_root,
        )

    resolved_feature_path = resolved_feature_path.expanduser().resolve()
    if not resolved_feature_path.exists():
        raise FileNotFoundError(f"Feature bundle not found: {resolved_feature_path}")

    resolved_model_names_path = (
        _resolve_existing_companion_path(
            resolved_feature_path,
            "_model_names.json",
            required=True,
        )
        .expanduser()
        .resolve()
    )
    metadata_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        "_metadata.json",
        required=False,
    )
    group_mask_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        GROUP_MASK_SUFFIX,
        required=False,
    )
    value_mask_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        VALUE_MASK_SUFFIX,
        required=False,
    )
    group_names_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        GROUP_NAMES_SUFFIX,
        required=False,
    )
    return ResolvedFeatureBundlePaths(
        feature_path=resolved_feature_path,
        model_names_path=resolved_model_names_path,
        metadata_path=(metadata_candidate.expanduser().resolve() if metadata_candidate.exists() else None),
        group_mask_path=(group_mask_candidate.expanduser().resolve() if group_mask_candidate.exists() else None),
        value_mask_path=(value_mask_candidate.expanduser().resolve() if value_mask_candidate.exists() else None),
        group_names_path=(group_names_candidate.expanduser().resolve() if group_names_candidate.exists() else None),
    )


def _unique_index_by_name(names: list[str], *, context: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for i, name in enumerate(names):
        if name in index:
            duplicates.append(name)
            continue
        index[name] = int(i)
    if duplicates:
        dup_preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate model names in {context}; cannot align features safely. Examples: {dup_preview}")
    return index


def _ordered_block_names_from_feature_names(feature_names: list[str]) -> list[str]:
    block_names: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        block_name = feature_block_name(feature_name)
        if block_name in seen:
            continue
        seen.add(block_name)
        block_names.append(block_name)
    return block_names


def _build_requested_feature_names(
    *,
    block_names: list[str],
    spectral_features: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
) -> list[str]:
    return build_spectral_feature_names(
        block_names=block_names,
        selected_features=spectral_features,
        sv_top_k=sv_top_k,
        spectral_moment_source=spectral_moment_source,
        shorten_block_names=False,
    )


def normalized_representation_kind(metadata: dict[str, Any] | None, *, feature_ndim: int | None = None) -> str:
    if isinstance(metadata, dict):
        raw_kind = metadata.get("representation_kind")
        if isinstance(raw_kind, str) and raw_kind.strip():
            return str(raw_kind).strip()
    if feature_ndim == 4:
        return ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND
    return TABULAR_SPECTRAL_REPRESENTATION_KIND


def tabular_array_or_raise(features: np.ndarray | SupervisedFeatureBundle) -> np.ndarray:
    if isinstance(features, SupervisedFeatureBundle):
        raise ValueError("This operation requires a tabular feature matrix, but the run uses a structured bundle")
    return np.asarray(features, dtype=np.float32)


def supervised_feature_row_count(features: np.ndarray | SupervisedFeatureBundle) -> int:
    if isinstance(features, SupervisedFeatureBundle):
        return int(features.n_samples)
    return int(features.shape[0])


def compatible_model_names_for_representation(
    *,
    requested_model_name: str,
    representation_kind: str,
) -> tuple[list[str], list[str]]:
    registered = registered_models()
    compatible = [
        name
        for name in registered
        if str(representation_kind) in supported_representation_kinds(name) and name != CNN_1D_DANN_MODEL_NAME
    ]
    incompatible = [name for name in registered if name not in compatible]
    if requested_model_name == "all":
        if not compatible:
            raise ValueError(f"No registered supervised models support representation_kind={representation_kind!r}")
        return compatible, incompatible

    if requested_model_name not in registered:
        raise ValueError(f"Unknown supervised model '{requested_model_name}'. Registered: {registered}")
    if str(representation_kind) not in supported_representation_kinds(requested_model_name):
        raise ValueError(
            f"Supervised model '{requested_model_name}' does not support representation_kind={representation_kind!r}"
        )
    return [requested_model_name], incompatible


def _validate_layer_sequence_external_bundle(
    *,
    metadata: dict[str, Any],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_attention_granularity: str = DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
) -> dict[str, Any]:
    selected_features = resolve_spectral_features(spectral_features)
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    resolved_qv_sum_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    resolved_attention_granularity = resolve_spectral_attention_granularity(spectral_attention_granularity)

    metadata_features = metadata.get("resolved_features")
    if not isinstance(metadata_features, list) or [str(x) for x in metadata_features] != list(selected_features):
        raise ValueError(
            "Structured external feature bundle was already aggregated with a different --features selection"
        )
    if "sv_topk" in selected_features and int(metadata.get("sv_top_k", spectral_sv_top_k)) != int(spectral_sv_top_k):
        raise ValueError("Structured external feature bundle was aggregated with a different --spectral-sv-top-k")
    if str(metadata.get("spectral_moment_source", resolved_moment_source)) != str(resolved_moment_source):
        raise ValueError("Structured external feature bundle was aggregated with a different --spectral-moment-source")
    if str(metadata.get("spectral_qv_sum_mode", resolved_qv_sum_mode)) != str(resolved_qv_sum_mode):
        raise ValueError("Structured external feature bundle was aggregated with a different --spectral-qv-sum-mode")
    if str(metadata.get("spectral_attention_granularity", resolved_attention_granularity)) != str(
        resolved_attention_granularity
    ):
        raise ValueError(
            "Structured external feature bundle was aggregated with a different --spectral-attention-granularity"
        )
    return sanitize_spectral_metadata(dict(metadata))


def _filter_external_spectral_columns(
    *,
    features: np.ndarray,
    metadata: dict[str, Any],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_attention_granularity: str,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    raw_feature_names = metadata.get("feature_names")
    if not isinstance(raw_feature_names, list) or not raw_feature_names:
        raise ValueError(
            "External spectral metadata must include non-empty 'feature_names' to honor "
            "--features/--spectral-sv-top-k/--spectral-qv-sum-mode"
        )

    feature_names = [str(x) for x in raw_feature_names]
    if len(feature_names) != int(features.shape[1]):
        raise ValueError(
            f"External feature metadata column count ({len(feature_names)}) does not match "
            f"feature matrix width ({features.shape[1]})"
        )

    selected_features = resolve_spectral_features(spectral_features)
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    resolved_qv_sum_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    resolved_attention_granularity = resolve_spectral_attention_granularity(spectral_attention_granularity)
    all_block_names = _ordered_block_names_from_feature_names(feature_names)

    if resolved_qv_sum_mode == "none":
        selected_block_names = [name for name in all_block_names if ".qv_sum" not in name]
    elif resolved_qv_sum_mode == "only":
        selected_block_names = [name for name in all_block_names if ".qv_sum" in name]
    else:
        selected_block_names = list(all_block_names)

    if not selected_block_names:
        raise ValueError(
            "External feature matrix does not contain any blocks compatible with "
            f"--spectral-qv-sum-mode={resolved_qv_sum_mode}"
        )

    expected_feature_names = _build_requested_feature_names(
        block_names=selected_block_names,
        spectral_features=selected_features,
        sv_top_k=int(spectral_sv_top_k),
        spectral_moment_source=resolved_moment_source,
    )
    if not expected_feature_names:
        raise ValueError("Requested external spectral feature selection resolved to zero columns")

    feature_index = _unique_index_by_name(feature_names, context="external spectral feature names")
    missing = [name for name in expected_feature_names if name not in feature_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"External feature matrix does not contain the requested spectral columns. Examples: {preview}"
        )

    warnings: list[str] = []
    if feature_names != expected_feature_names:
        column_indices = np.asarray([feature_index[name] for name in expected_feature_names], dtype=np.int64)
        features = features[:, column_indices]
        warnings.append("Filtered/reordered external feature columns to match requested spectral configuration")

    filtered_metadata = dict(metadata)
    filtered_metadata["resolved_features"] = list(selected_features)
    filtered_metadata["spectral_moment_source"] = resolved_moment_source
    filtered_metadata["spectral_qv_sum_mode"] = resolved_qv_sum_mode
    filtered_metadata["spectral_attention_granularity"] = resolved_attention_granularity
    filtered_metadata["sv_top_k"] = int(spectral_sv_top_k)
    filtered_metadata["feature_dim"] = int(features.shape[1])
    filtered_metadata["feature_names"] = list(expected_feature_names)
    filtered_metadata["block_names"] = list(selected_block_names)
    filtered_metadata["n_blocks"] = int(len(selected_block_names))
    source_lora_dims = spectral_block_lora_dims_by_block(metadata)
    if all(block_name in source_lora_dims for block_name in selected_block_names):
        filtered_metadata["lora_adapter_dims"] = [source_lora_dims[block_name] for block_name in selected_block_names]

    filtered_metadata["base_block_names"] = [name for name in selected_block_names if ".qv_sum" not in name]
    filtered_metadata["qv_sum_block_names"] = [name for name in selected_block_names if ".qv_sum" in name]

    extractor_params = filtered_metadata.get("extractor_params")
    if isinstance(extractor_params, dict):
        filtered_params = dict(extractor_params)
        filtered_params["spectral_features"] = list(selected_features)
        filtered_params["spectral_sv_top_k"] = int(spectral_sv_top_k)
        filtered_params["spectral_moment_source"] = resolved_moment_source
        filtered_params["spectral_qv_sum_mode"] = resolved_qv_sum_mode
        filtered_params["spectral_attention_granularity"] = resolved_attention_granularity
        filtered_metadata["extractor_params"] = spectral_extractor_params(filtered_params)

    return features, sanitize_spectral_metadata(filtered_metadata), warnings


def load_external_spectral_bundle(
    *,
    feature_file: Path,
    model_names_file: Path,
    metadata_file: Path | None,
    group_mask_file: Path | None,
    value_mask_file: Path | None,
    group_names_file: Path | None,
    expected_model_names: list[str],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_attention_granularity: str,
    allow_manifest_subset: bool = False,
) -> tuple[np.ndarray | SupervisedFeatureBundle, dict[str, Any], list[str], np.ndarray]:
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    if not model_names_file.exists():
        raise FileNotFoundError(f"Model names file not found: {model_names_file}")
    if metadata_file is not None and not metadata_file.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {metadata_file}")

    features_mmap = np.load(feature_file, mmap_mode="r")
    if features_mmap.ndim not in {2, 4}:
        raise ValueError(f"Expected a 2D or 4D feature bundle at {feature_file}, got shape={features_mmap.shape}")

    with open(model_names_file, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(features_mmap.shape[0]):
        raise ValueError(
            f"Model names length ({len(model_names)}) does not match feature rows ({features_mmap.shape[0]}) "
            f"for external features"
        )

    metadata: dict[str, Any] = {}
    if metadata_file is not None:
        loaded = load_spectral_metadata(metadata_file)
        if isinstance(loaded, dict):
            metadata = dict(loaded)
    representation_kind = normalized_representation_kind(metadata, feature_ndim=int(features_mmap.ndim))
    if str(representation_kind) == ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND and features_mmap.ndim != 4:
        raise ValueError(
            "Structured layer-sequence metadata requires a 4D feature tensor, "
            f"but {feature_file} has shape={features_mmap.shape}"
        )
    if str(representation_kind) != ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND and features_mmap.ndim != 2:
        raise ValueError(
            "Tabular supervised representations require a 2D feature matrix, "
            f"but {feature_file} has shape={features_mmap.shape}"
        )

    ext_index = _unique_index_by_name(model_names, context=str(model_names_file))
    expected_index = _unique_index_by_name(expected_model_names, context="manifest model names")

    missing = sorted(name for name in expected_index if name not in ext_index)
    extra = sorted(name for name in ext_index if name not in expected_index)
    if missing and not allow_manifest_subset:
        details: list[str] = [f"missing={missing[:5]}"]
        if extra:
            details.append(f"extra={extra[:5]}")
        raise ValueError(
            "External feature/model-name set does not cover the manifest model-name set: " + "; ".join(details)
        )

    selected_expected_indices = np.asarray(
        [i for i, name in enumerate(expected_model_names) if name in ext_index],
        dtype=np.int64,
    )
    if selected_expected_indices.size == 0:
        details: list[str] = []
        if expected_model_names:
            details.append(f"expected={expected_model_names[:5]}")
        if model_names:
            details.append(f"external={model_names[:5]}")
        raise ValueError(
            "External feature/model-name set has no overlap with the manifest model-name set"
            + (": " + "; ".join(details) if details else "")
        )

    selected_expected_model_names = [expected_model_names[int(i)] for i in selected_expected_indices.tolist()]
    row_indices = np.asarray([ext_index[name] for name in selected_expected_model_names], dtype=np.int64)
    features = np.asarray(features_mmap[row_indices], dtype=np.float32)

    requested_rows = int(len(expected_model_names))
    expected_rows = int(len(selected_expected_model_names))
    source_rows = int(features_mmap.shape[0])

    warnings: list[str] = []
    if allow_manifest_subset and missing:
        warning = (
            "Selected external feature rows by manifest model names from a source bundle "
            f"with {source_rows} rows; retained {expected_rows}/{requested_rows} manifest models "
            f"and skipped {len(missing)} missing model names"
        )
        if extra:
            warning += f"; ignored {len(extra)} external-only model names"
        warning += f". Missing example(s): {missing[:5]}"
        warnings.append(warning)
    elif source_rows != expected_rows:
        warnings.append(
            f"Selected external feature rows by manifest model names from a source bundle with {source_rows} rows"
        )
    elif selected_expected_model_names != model_names:
        warnings.append("Reordered external features to match manifest model order using model names")

    metadata = dict(metadata)
    metadata["representation_kind"] = str(representation_kind)
    metadata["n_models"] = int(features.shape[0])
    metadata["external_manifest_requested_n_models"] = int(requested_rows)
    metadata["external_manifest_selected_n_models"] = int(features.shape[0])
    metadata["external_manifest_missing_model_count"] = int(len(missing))
    if source_rows != int(features.shape[0]):
        metadata["external_source_n_models"] = int(source_rows)

    if str(representation_kind) == ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND:
        if group_mask_file is None or not group_mask_file.exists():
            raise FileNotFoundError("Structured external feature bundle is missing the required group-mask companion")
        if value_mask_file is None or not value_mask_file.exists():
            raise FileNotFoundError("Structured external feature bundle is missing the required value-mask companion")
        if group_names_file is None or not group_names_file.exists():
            raise FileNotFoundError("Structured external feature bundle is missing the required group-names companion")

        group_mask_mmap = np.load(group_mask_file, mmap_mode="r")
        value_mask_mmap = np.load(value_mask_file, mmap_mode="r")
        if group_mask_mmap.shape != features_mmap.shape[:2]:
            raise ValueError(
                f"group_mask shape {group_mask_mmap.shape} does not match feature tensor shape {features_mmap.shape[:2]}"
            )
        if value_mask_mmap.shape != features_mmap.shape:
            raise ValueError(
                f"value_mask shape {value_mask_mmap.shape} does not match feature tensor shape {features_mmap.shape}"
            )
        with open(group_names_file, "r", encoding="utf-8") as f:
            raw_group_names = json.load(f)
        if not isinstance(raw_group_names, list) or len(raw_group_names) != source_rows:
            raise ValueError("Structured external feature bundle group-names companion must be a row-aligned JSON list")

        selected_group_names = [
            [str(x) for x in raw_group_names[int(source_idx)]] for source_idx in row_indices.tolist()
        ]
        metadata = _validate_layer_sequence_external_bundle(
            metadata=metadata,
            spectral_features=spectral_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_attention_granularity=spectral_attention_granularity,
        )
        metadata["structural_group_names"] = [list(names) for names in selected_group_names]
        return (
            SupervisedFeatureBundle(
                values=features,
                representation_kind=str(representation_kind),
                metadata=dict(metadata),
                group_mask=np.asarray(group_mask_mmap[row_indices], dtype=bool),
                value_mask=np.asarray(value_mask_mmap[row_indices], dtype=bool),
                group_names=selected_group_names,
            ),
            metadata,
            warnings,
            selected_expected_indices,
        )

    features, metadata, column_warnings = _filter_external_spectral_columns(
        features=features,
        metadata=metadata,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_attention_granularity=spectral_attention_granularity,
    )
    warnings.extend(column_warnings)

    return features, metadata, warnings, selected_expected_indices
