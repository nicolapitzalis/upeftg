from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...utilities.core.manifest import (
    infer_attack_sample_identities,
    parse_cv_always_train_manifest_json_by_model_name,
    parse_joint_manifest_json_by_model_name,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
)
from ...utilities.core.run_context import RunContext
from ..validation.cross_validation import resolve_cv_strategy
from ..data.preparation import (
    compact_indices_to_selected_scope,
    prepare_supervised_dataset,
)
from ..data.bundles import (
    compatible_model_names_for_representation,
    normalized_representation_kind,
)
from ..contracts import SupervisedFeatureBundle, SupervisedTaskSpec
from ..data.normalization import INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD, resolve_input_normalization
from ..models.registry import (
    TRANSFORMER_MODEL_NAME,
    resolve_model_hyperparams,
)
from ..validation.splits import (
    resolve_calibration_split_percent,
    resolve_split_by_folder,
    resolve_train_split_percent,
)
from ..tasks import resolve_selection_metric, resolve_supervised_task_spec
from ..evaluation.thresholds import resolve_accepted_fprs
from .tuning import _is_torch_sequence_model_name


@dataclass(frozen=True)
class ResolvedPrepareRequest:
    manifest_json: Path
    task_spec: SupervisedTaskSpec
    cv_strategy: str
    input_normalization: str
    selection_metric_name: str


@dataclass(frozen=True)
class ManifestPartitions:
    mode: str
    all_items: tuple[Any, ...]
    train_items: tuple[Any, ...]
    infer_items: tuple[Any, ...]
    sample_identities: tuple[Any, ...]
    train_indices: np.ndarray
    infer_indices: np.ndarray
    cv_always_train_indices: np.ndarray


@dataclass(frozen=True)
class ResolvedSplitOptions:
    train_split_percent: int
    calibration_split_percent: int | None
    accepted_fprs: tuple[float, ...] | None
    split_by_folder: bool


@dataclass(frozen=True)
class PreparedRunFeatures:
    features: np.ndarray | SupervisedFeatureBundle
    all_items: list[Any]
    train_items: list[Any]
    infer_items: list[Any]
    sample_identities: list[Any]
    train_indices: np.ndarray
    infer_indices: np.ndarray
    cv_always_train_indices: np.ndarray
    dataset_group_names: list[str]
    dataset_group_names_path: Path
    dataset_group_counts: dict[str, int]
    input_normalization_config: dict[str, Any]
    feature_params: dict[str, Any]
    feature_loading_mode: str
    extractor_metadata: dict[str, Any]
    extractor_warnings: list[str]
    artifacts: dict[str, str | None]
    representation_kind: str
    model_names: list[str]
    rank_label_weight_loss: bool
    hyperparam_axes: dict[str, list[Any]]
    hyperparams_info: dict[str, Any]


def detect_manifest_mode(manifest_json: Path) -> str:
    resolved_manifest_json = resolve_manifest_path(manifest_json)
    with open(resolved_manifest_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "train" in payload and "infer" in payload:
        return "joint"
    return "single"


def resolve_prepare_request(
    *,
    manifest_json: Path,
    feature_file: Path | None,
    spectral_features: list[str] | None,
    model_name: str,
    transformer_checkpoint_dir: Path | None,
    transformer_checkpoint_interval_seconds: float | None,
    transformer_resume_checkpoint: Path | None,
    class_weight_loss: bool,
    rank_label_weight_loss: bool,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
    cv_strategy: str,
    input_normalization: str,
    selection_metric: str | None,
) -> ResolvedPrepareRequest:
    resolved_manifest = resolve_manifest_path(manifest_json)
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {resolved_manifest}")
    if feature_file is None:
        raise ValueError(
            "Supervised pipeline requires --feature-file. Run feature extraction first, then pass the "
            "extracted feature bundle into supervised."
        )
    if not spectral_features:
        raise ValueError(
            "Supervised pipeline requires --features. Specify the feature groups to select from the "
            "extracted feature bundle."
        )
    checkpoint_values = (transformer_checkpoint_dir, transformer_checkpoint_interval_seconds)
    if any(value is not None for value in checkpoint_values) and not all(
        value is not None for value in checkpoint_values
    ):
        raise ValueError("Periodic checkpointing requires both its internal directory and interval")
    if (
        any(value is not None for value in (*checkpoint_values, transformer_resume_checkpoint))
        and model_name != TRANSFORMER_MODEL_NAME
    ):
        raise ValueError("Interval checkpointing currently requires --model transformer")
    if transformer_checkpoint_interval_seconds is not None and float(transformer_checkpoint_interval_seconds) <= 0.0:
        raise ValueError("Transformer checkpoint interval must be positive")
    if transformer_resume_checkpoint is not None and transformer_checkpoint_interval_seconds is None:
        raise ValueError("--resume-checkpoint requires --checkpoint-interval-hours")
    if bool(class_weight_loss) and bool(rank_label_weight_loss):
        raise ValueError("--class-weight-loss and --rank-label-weight-loss are mutually exclusive")

    task_spec = resolve_supervised_task_spec(
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
    )
    return ResolvedPrepareRequest(
        manifest_json=resolved_manifest,
        task_spec=task_spec,
        cv_strategy=resolve_cv_strategy(cv_strategy),
        input_normalization=resolve_input_normalization(input_normalization),
        selection_metric_name=resolve_selection_metric(selection_metric, task_spec=task_spec),
    )


def load_manifest_partitions(manifest_json: Path) -> ManifestPartitions:
    mode = detect_manifest_mode(manifest_json)
    cv_always_train_items = parse_cv_always_train_manifest_json_by_model_name(
        manifest_path=manifest_json,
    )
    if mode == "joint":
        if cv_always_train_items:
            raise ValueError("cv_always_train is supported only in single training manifests")
        train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_json)
        all_items = train_items + infer_items
        train_indices = np.arange(0, len(train_items), dtype=np.int64)
        infer_indices = np.arange(len(train_items), len(all_items), dtype=np.int64)
    else:
        all_items = parse_single_manifest_json_by_model_name(
            manifest_path=manifest_json,
            section_key="path",
        )
        train_items = all_items
        infer_items = []
        train_indices = np.arange(0, len(all_items), dtype=np.int64)
        infer_indices = np.asarray([], dtype=np.int64)
    cv_always_train_names = {str(item.model_name) for item in cv_always_train_items}
    cv_always_train_indices = np.asarray(
        [index for index, item in enumerate(all_items) if str(item.model_name) in cv_always_train_names],
        dtype=np.int64,
    )
    if int(cv_always_train_indices.size) != len(cv_always_train_items):
        raise ValueError("Could not align every cv_always_train row to the parsed training manifest")
    if not train_items:
        raise ValueError("No training items resolved from manifest")
    return ManifestPartitions(
        mode=mode,
        all_items=tuple(all_items),
        train_items=tuple(train_items),
        infer_items=tuple(infer_items),
        sample_identities=tuple(infer_attack_sample_identities(all_items)),
        train_indices=train_indices,
        infer_indices=infer_indices,
        cv_always_train_indices=cv_always_train_indices,
    )


def resolve_split_options(
    *,
    train_split_percent: int,
    calibration_split_percent: int | None,
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
    split_by_folder: bool | None,
) -> ResolvedSplitOptions:
    resolved_train = resolve_train_split_percent(train_split_percent)
    resolved_calibration = resolve_calibration_split_percent(calibration_split_percent)
    accepted_fprs = resolve_accepted_fprs(accepted_fpr)
    if (resolved_calibration is None) != (accepted_fprs is None):
        raise ValueError("--calibration-split and --accepted-fpr must either both be set or both be omitted")
    return ResolvedSplitOptions(
        train_split_percent=resolved_train,
        calibration_split_percent=resolved_calibration,
        accepted_fprs=accepted_fprs,
        split_by_folder=resolve_split_by_folder(split_by_folder, resolved_calibration),
    )


def prepare_run_features(
    *,
    ctx: RunContext,
    feature_file: Path,
    all_items: list[Any],
    train_items: list[Any],
    infer_items: list[Any],
    sample_identities: list[Any],
    train_indices: np.ndarray,
    infer_indices: np.ndarray,
    cv_always_train_indices: np.ndarray,
    model_name: str,
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str,
    stream_block_size: int,
    dtype_name: str,
    input_normalization: str,
    rank_label_weight_loss: bool,
    hyperparams: Path | None,
) -> PreparedRunFeatures:
    feature_params = {
        "dtype": dtype_name,
        "block_size": int(stream_block_size),
        "spectral_features": list(spectral_features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": str(spectral_moment_source),
        "spectral_qv_sum_mode": str(spectral_qv_sum_mode),
        "spectral_entrywise_delta_mode": str(spectral_entrywise_delta_mode),
        "spectral_attention_granularity": str(spectral_attention_granularity),
    }
    manifest_item_count = len(all_items)
    prepared = prepare_supervised_dataset(
        feature_spec=feature_file,
        items=all_items,
        sample_identities=sample_identities,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        allow_manifest_subset=True,
    )
    features = prepared.features
    external_metadata = dict(prepared.metadata)
    warnings = list(prepared.warnings)
    paths = prepared.source_paths
    selected_indices = prepared.selected_manifest_indices
    all_items = list(prepared.items)
    sample_identities = list(prepared.sample_identities)
    if int(selected_indices.size) != int(manifest_item_count):
        train_indices = compact_indices_to_selected_scope(train_indices, selected_expected_indices=selected_indices)
        infer_indices = compact_indices_to_selected_scope(infer_indices, selected_expected_indices=selected_indices)
        compacted_cv_always_train_indices = compact_indices_to_selected_scope(
            cv_always_train_indices,
            selected_expected_indices=selected_indices,
        )
        if int(compacted_cv_always_train_indices.size) != int(cv_always_train_indices.size):
            raise ValueError("External feature source is missing one or more cv_always_train rows")
        cv_always_train_indices = compacted_cv_always_train_indices
        train_items = [all_items[int(index)] for index in train_indices.tolist()]
        infer_items = [all_items[int(index)] for index in infer_indices.tolist()]
        if not train_items:
            raise ValueError(
                "No training items remain after intersecting the manifest with the available "
                "external feature/model-name set"
            )

    dataset_group_names = list(prepared.dataset_group_names)
    dataset_group_names_path = ctx.run_dir / ".work" / "prepared_arrays.npz"
    dataset_group_counts = {
        str(name): int(dataset_group_names.count(name)) for name in sorted(set(dataset_group_names))
    }
    input_normalization_config = {
        "mode": str(input_normalization),
        "grouping": (
            "inferred_dataset" if input_normalization == INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD else None
        ),
        "statistics": (
            "per_slice_dataset_feature_standard"
            if input_normalization == INPUT_NORMALIZATION_DATASET_FEATURE_STANDARD
            else None
        ),
        "dataset_group_names_path": str(dataset_group_names_path),
        "dataset_group_counts": dict(dataset_group_counts),
    }

    extractor_metadata = {
        **external_metadata,
        "external_feature_source": str(paths.feature_path),
        "external_model_names_source": str(paths.model_names_path),
        "external_metadata_source": str(paths.metadata_path) if paths.metadata_path else None,
        "external_group_mask_source": str(paths.group_mask_path) if paths.group_mask_path else None,
        "external_value_mask_source": str(paths.value_mask_path) if paths.value_mask_path else None,
        "external_group_names_source": str(paths.group_names_path) if paths.group_names_path else None,
        "loaded_external_features": True,
    }
    artifacts = {
        "feature_path": str(paths.feature_path),
        "labels_path": None,
        "model_names_path": None,
        "metadata_path": str(paths.metadata_path) if paths.metadata_path else None,
    }
    representation_kind = normalized_representation_kind(
        external_metadata,
        feature_ndim=(
            int(features.values.ndim) if isinstance(features, SupervisedFeatureBundle) else int(features.ndim)
        ),
    )
    model_names, _incompatible = compatible_model_names_for_representation(
        requested_model_name=model_name,
        representation_kind=representation_kind,
    )
    if len(model_names) != 1:
        raise ValueError("Supervised preparation requires one concrete --model selection")
    torch_sequence_selected = any(_is_torch_sequence_model_name(name) for name in model_names)
    resolved_rank_weight = bool(rank_label_weight_loss and torch_sequence_selected)
    if rank_label_weight_loss and not torch_sequence_selected:
        warnings.append(
            "Ignored --rank-label-weight-loss because no torch sequence model is selected for this supervised run"
        )

    hyperparam_axes, hyperparams_info = resolve_model_hyperparams(model_names[0], hyperparams)

    return PreparedRunFeatures(
        features=features,
        all_items=all_items,
        train_items=train_items,
        infer_items=infer_items,
        sample_identities=sample_identities,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        infer_indices=np.asarray(infer_indices, dtype=np.int64),
        cv_always_train_indices=np.asarray(cv_always_train_indices, dtype=np.int64),
        dataset_group_names=dataset_group_names,
        dataset_group_names_path=dataset_group_names_path,
        dataset_group_counts=dataset_group_counts,
        input_normalization_config=input_normalization_config,
        feature_params=feature_params,
        feature_loading_mode="external_source",
        extractor_metadata=extractor_metadata,
        extractor_warnings=warnings,
        artifacts=artifacts,
        representation_kind=representation_kind,
        model_names=model_names,
        rank_label_weight_loss=resolved_rank_weight,
        hyperparam_axes=hyperparam_axes,
        hyperparams_info=hyperparams_info,
    )
