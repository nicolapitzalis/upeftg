from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import joblib
except Exception:  # pragma: no cover - fallback for minimal environments
    joblib = None
import pickle

from ..features.registry import extract_with_cache
from ..features.spectral import (
    build_spectral_feature_names,
    resolve_spectral_features,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
)
from ..unsupervised.reporting import (
    compute_infer_threshold_rows,
    compute_offline_metrics,
    save_score_csv,
    summarize_scores,
)
from ..utilities.manifest import (
    parse_joint_manifest_json,
    parse_joint_manifest_json_by_model_name,
    parse_single_manifest_json,
    parse_single_manifest_json_by_model_name,
)
from ..utilities.run_context import RunContext, create_run_context
from ..utilities.serialization import json_ready
from ..utilities.export_winner_feature_weights import export_winner_feature_weights
from .distributed import (
    build_slurm_array_next_steps,
    resolve_slurm_cpus_per_task,
    resolve_slurm_max_concurrent,
)
from .registry import candidate_params, create, model_complexity_rank, registered_models


SCRIPT_VERSION = "1.0.0"
PIPELINE_NAME = "supervised"


def _detect_manifest_mode(manifest_json: Path) -> str:
    with open(manifest_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "train" in payload and "infer" in payload:
        return "joint"
    return "single"


def _labels_from_items(items: list[Any]) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    raw_labels = [item.label for item in items]
    values = np.asarray([int(label) if label is not None else -1 for label in raw_labels], dtype=np.int32)
    known = np.asarray([label is not None for label in raw_labels], dtype=bool)
    return values, known, raw_labels


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
        raise ValueError(
            f"Duplicate model names in {context}; cannot align features safely. "
            f"Examples: {dup_preview}"
        )
    return index


def _feature_block_name(feature_name: str) -> str:
    block_name, sep, _ = str(feature_name).rpartition(".")
    if not sep or not block_name:
        raise ValueError(f"Invalid spectral feature name in external metadata: {feature_name}")
    return block_name


def _ordered_block_names_from_feature_names(feature_names: list[str]) -> list[str]:
    block_names: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        block_name = _feature_block_name(feature_name)
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


def _filter_external_spectral_columns(
    *,
    features: np.ndarray,
    metadata: dict[str, Any],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
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
            "External feature matrix does not contain the requested spectral columns. "
            f"Examples: {preview}"
        )

    warnings: list[str] = []
    if feature_names != expected_feature_names:
        column_indices = np.asarray([feature_index[name] for name in expected_feature_names], dtype=np.int64)
        features = features[:, column_indices]
        warnings.append(
            "Filtered/reordered external feature columns to match requested spectral configuration"
        )

    filtered_metadata = dict(metadata)
    filtered_metadata["resolved_features"] = list(selected_features)
    filtered_metadata["spectral_moment_source"] = resolved_moment_source
    filtered_metadata["spectral_qv_sum_mode"] = resolved_qv_sum_mode
    filtered_metadata["sv_top_k"] = int(spectral_sv_top_k)
    filtered_metadata["feature_dim"] = int(features.shape[1])
    filtered_metadata["feature_names"] = list(expected_feature_names)
    filtered_metadata["block_names"] = list(selected_block_names)
    filtered_metadata["n_blocks"] = int(len(selected_block_names))

    source_block_names = metadata.get("block_names")
    source_block_names_raw = metadata.get("block_names_raw")
    if (
        isinstance(source_block_names, list)
        and isinstance(source_block_names_raw, list)
        and len(source_block_names) == len(source_block_names_raw)
    ):
        raw_by_block = {str(block): str(raw) for block, raw in zip(source_block_names, source_block_names_raw)}
        if all(block_name in raw_by_block for block_name in selected_block_names):
            selected_block_names_raw = [raw_by_block[block_name] for block_name in selected_block_names]
            filtered_metadata["block_names_raw"] = selected_block_names_raw
            filtered_metadata["base_block_names_raw"] = [
                name for name in selected_block_names_raw if ".qv_sum" not in name
            ]
            filtered_metadata["qv_sum_block_names_raw"] = [
                name for name in selected_block_names_raw if ".qv_sum" in name
            ]
        else:
            filtered_metadata.pop("block_names_raw", None)
            filtered_metadata.pop("base_block_names_raw", None)
            filtered_metadata.pop("qv_sum_block_names_raw", None)
    else:
        filtered_metadata.pop("block_names_raw", None)
        filtered_metadata.pop("base_block_names_raw", None)
        filtered_metadata.pop("qv_sum_block_names_raw", None)

    filtered_metadata["base_block_names"] = [
        name for name in selected_block_names if ".qv_sum" not in name
    ]
    filtered_metadata["qv_sum_block_names"] = [
        name for name in selected_block_names if ".qv_sum" in name
    ]

    extractor_params = filtered_metadata.get("extractor_params")
    if isinstance(extractor_params, dict):
        filtered_params = dict(extractor_params)
        filtered_params["spectral_features"] = list(selected_features)
        filtered_params["spectral_sv_top_k"] = int(spectral_sv_top_k)
        filtered_params["spectral_moment_source"] = resolved_moment_source
        filtered_params["spectral_qv_sum_mode"] = resolved_qv_sum_mode
        filtered_metadata["extractor_params"] = filtered_params

    return features, filtered_metadata, warnings


def _load_external_spectral_bundle(
    *,
    feature_file: Path,
    model_names_file: Path,
    metadata_file: Path | None,
    expected_model_names: list[str],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    if not model_names_file.exists():
        raise FileNotFoundError(f"Model names file not found: {model_names_file}")
    if metadata_file is not None and not metadata_file.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {metadata_file}")

    features = np.asarray(np.load(feature_file), dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix at {feature_file}, got shape={features.shape}")

    with open(model_names_file, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(features.shape[0]):
        raise ValueError(
            f"Model names length ({len(model_names)}) does not match feature rows ({features.shape[0]}) "
            f"for external features"
        )

    metadata: dict[str, Any] = {}
    if metadata_file is not None:
        with open(metadata_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            metadata = dict(loaded)

    expected_rows = int(len(expected_model_names))
    if int(features.shape[0]) != expected_rows:
        raise ValueError(
            f"External features row count ({features.shape[0]}) does not match manifest size ({expected_rows})"
        )

    warnings: list[str] = []
    if model_names != expected_model_names:
        ext_index = _unique_index_by_name(model_names, context=str(model_names_file))
        expected_index = _unique_index_by_name(expected_model_names, context="manifest model names")

        missing = sorted(name for name in expected_index if name not in ext_index)
        extra = sorted(name for name in ext_index if name not in expected_index)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing[:5]}")
            if extra:
                details.append(f"extra={extra[:5]}")
            raise ValueError(
                "External feature/model-name set does not match manifest model-name set: "
                + "; ".join(details)
            )

        reorder = np.asarray([ext_index[name] for name in expected_model_names], dtype=np.int64)
        features = features[reorder]
        warnings.append("Reordered external features to match manifest model order using model names")

    features, metadata, column_warnings = _filter_external_spectral_columns(
        features=features,
        metadata=metadata,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
    )
    warnings.extend(column_warnings)

    return features, metadata, warnings


def _sanitize_cv_folds(labels: np.ndarray, requested_folds: int) -> tuple[int, list[str]]:
    if requested_folds < 2:
        raise ValueError(f"cv_folds must be >=2, got {requested_folds}")

    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Binary classification requires at least two classes in the training set")

    min_count = int(np.min(counts))
    warnings: list[str] = []
    if min_count < 2:
        warnings.append(
            "Minority class count <2; falling back to a single train-as-validation split for tuning"
        )
        return 1, warnings

    resolved = min(requested_folds, min_count)
    if resolved != requested_folds:
        warnings.append(
            f"Reduced cv_folds from {requested_folds} to {resolved} due to minority class size={min_count}"
        )
    return int(resolved), warnings


def _build_cv_splits(
    *,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if cv_folds == 1:
        return [
            {
                "split_index": 0,
                "train_indices": [int(x) for x in train_indices.tolist()],
                "valid_indices": [int(x) for x in train_indices.tolist()],
            }
        ]

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits: list[dict[str, Any]] = []
    dummy = np.zeros((train_indices.shape[0], 1), dtype=np.float32)
    for split_idx, (tr_rel, val_rel) in enumerate(splitter.split(dummy, train_labels)):
        tr_abs = train_indices[np.asarray(tr_rel, dtype=np.int64)]
        val_abs = train_indices[np.asarray(val_rel, dtype=np.int64)]
        splits.append(
            {
                "split_index": int(split_idx),
                "train_indices": [int(x) for x in tr_abs.tolist()],
                "valid_indices": [int(x) for x in val_abs.tolist()],
            }
        )
    return splits


def _predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=np.float64)
        if decision.ndim == 2 and decision.shape[1] >= 2:
            return decision[:, 1]
        return decision.reshape(-1)

    pred = np.asarray(model.predict(x), dtype=np.float64)
    return pred.reshape(-1)


def _evaluate_fold(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    model_name: str,
    params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    model = create(model_name, params=params, random_state=random_state)
    model.fit(features[train_indices], labels[train_indices])
    scores = _predict_scores(model, features[valid_indices])
    auc = float(roc_auc_score(labels[valid_indices], scores))
    return {
        "n_train": int(train_indices.size),
        "n_valid": int(valid_indices.size),
        "roc_auc": auc,
    }


def _evaluate_candidate(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    task: dict[str, Any],
    cv_split_groups: list[dict[str, Any]],
    n_jobs: int,
) -> dict[str, Any]:
    eval_jobs: list[tuple[int, dict[str, Any]]] = []
    for group in cv_split_groups:
        seed = int(group["random_state"])
        for split in group["cv_splits"]:
            eval_jobs.append((seed, split))

    if n_jobs == 1 or len(eval_jobs) <= 1 or joblib is None:
        evaluated: list[tuple[int, dict[str, Any]]] = []
        for seed, split in eval_jobs:
            tr = np.asarray(split["train_indices"], dtype=np.int64)
            val = np.asarray(split["valid_indices"], dtype=np.int64)
            row = _evaluate_fold(
                features=features,
                labels=labels,
                train_indices=tr,
                valid_indices=val,
                model_name=str(task["model_name"]),
                params=dict(task["params"]),
                random_state=seed,
            )
            row["cv_random_state"] = int(seed)
            evaluated.append((seed, row))
    else:
        raw_rows = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_evaluate_fold)(
                features=features,
                labels=labels,
                train_indices=np.asarray(split["train_indices"], dtype=np.int64),
                valid_indices=np.asarray(split["valid_indices"], dtype=np.int64),
                model_name=str(task["model_name"]),
                params=dict(task["params"]),
                random_state=seed,
            )
            for seed, split in eval_jobs
        )
        evaluated = []
        for (seed, _), row in zip(eval_jobs, raw_rows):
            row_with_seed = dict(row)
            row_with_seed["cv_random_state"] = int(seed)
            evaluated.append((seed, row_with_seed))

    fold_rows = [row for _, row in evaluated]
    scores = [float(row["roc_auc"]) for row in fold_rows]

    seed_results: list[dict[str, Any]] = []
    unique_seeds = sorted({int(seed) for seed, _ in evaluated})
    for seed in unique_seeds:
        seed_fold_rows = [row for seed_value, row in evaluated if int(seed_value) == int(seed)]
        seed_scores = [float(row["roc_auc"]) for row in seed_fold_rows]
        seed_results.append(
            {
                "random_state": int(seed),
                "fold_results": seed_fold_rows,
                "roc_auc_mean": float(np.mean(seed_scores)) if seed_scores else None,
                "roc_auc_std": float(np.std(seed_scores)) if seed_scores else None,
            }
        )

    return {
        "task_index": int(task["task_index"]),
        "model_name": str(task["model_name"]),
        "params": dict(task["params"]),
        "complexity_rank": int(task["complexity_rank"]),
        "status": "ok",
        "fold_results": fold_rows,
        "seed_results": seed_results,
        "roc_auc_mean": float(np.mean(scores)) if scores else None,
        "roc_auc_std": float(np.std(scores)) if scores else None,
    }


def _task_result_path(task_dir: Path, task_index: int) -> Path:
    return task_dir / f"task_{task_index:04d}.json"


def _compact_extractor_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if key == "feature_names" or key == "shapes" or key.endswith("_shapes"):
            continue
        compact[str(key)] = value
    return compact


def _score_percentile_ranks(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.float64)
    ranks = np.argsort(np.argsort(scores))
    return np.asarray(ranks / max(1, scores.size - 1), dtype=np.float64)


def _infer_attack_name(model_name: str) -> str:
    parts = model_name.split("_")
    if "imdb" in parts:
        idx = parts.index("imdb")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1].strip()
            if candidate:
                return candidate
    return "unknown"


def _build_threshold_specs(train_scores: np.ndarray, percentiles: list[float]) -> list[dict[str, float]]:
    specs: list[dict[str, float]] = []
    for pct in percentiles:
        specs.append(
            {
                "percentile_from_train": float(pct),
                "threshold": float(np.percentile(train_scores, float(pct))),
            }
        )
    return specs


def _summarize_attack_groups(
    *,
    model_names: list[str],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, float]],
) -> dict[str, Any]:
    if len(model_names) != len(labels) or len(model_names) != int(scores.size):
        raise ValueError("Attack analysis inputs must have the same length")

    score_ranks = _score_percentile_ranks(np.asarray(scores, dtype=np.float64))
    grouped_by_token: dict[str, list[int]] = {}
    positive_by_attack: dict[str, list[int]] = {}
    unknown_by_attack: dict[str, list[int]] = {}
    clean_indices: list[int] = []

    for idx, (name, label) in enumerate(zip(model_names, labels)):
        attack = _infer_attack_name(name)
        grouped_by_token.setdefault(attack, []).append(idx)
        if label == 1:
            positive_by_attack.setdefault(attack, []).append(idx)
        elif label == 0:
            clean_indices.append(idx)
        else:
            unknown_by_attack.setdefault(attack, []).append(idx)

    grouped_indices: dict[str, list[int]] = {}
    if positive_by_attack:
        # Attack-specific view: each attack is evaluated against the shared clean pool.
        for attack in sorted(positive_by_attack):
            combined = (
                positive_by_attack[attack]
                + clean_indices
                + unknown_by_attack.get(attack, [])
            )
            grouped_indices[attack] = sorted(set(int(i) for i in combined))
        grouping_rule = (
            "one-vs-clean per attack: positives grouped by token parsed after 'imdb'; "
            "all clean samples are reused as negatives for every attack; fallback='unknown'"
        )
    else:
        # Fallback for unusual datasets with no known positives.
        grouped_indices = {
            attack: sorted(int(i) for i in idx_list)
            for attack, idx_list in grouped_by_token.items()
        }
        grouping_rule = "attack token parsed from model_name as segment after 'imdb'; fallback='unknown'"

    attacks: dict[str, Any] = {}
    for attack in sorted(grouped_indices):
        idx = np.asarray(grouped_indices[attack], dtype=np.int64)
        group_scores = np.asarray(scores[idx], dtype=np.float64)
        group_ranks = np.asarray(score_ranks[idx], dtype=np.float64)
        group_labels = [labels[int(i)] for i in idx.tolist()]
        clean_count = sum(1 for label in group_labels if label == 0)
        backdoored_count = sum(1 for label in group_labels if label == 1)
        unknown_count = sum(1 for label in group_labels if label is None)

        known_mask = np.asarray([lbl is not None for lbl in group_labels], dtype=bool)
        known_labels: np.ndarray | None = None
        if bool(np.any(known_mask)):
            known_labels = np.asarray(
                [int(group_labels[i]) for i in range(len(group_labels)) if known_mask[i]],
                dtype=np.int32,
            )

        threshold_rows: list[dict[str, Any]] = []
        for spec in threshold_specs:
            threshold = float(spec["threshold"])
            flagged = group_scores >= threshold
            n_flagged = int(np.sum(flagged))
            row: dict[str, Any] = {
                "percentile_from_train": float(spec["percentile_from_train"]),
                "threshold": threshold,
                "n_flagged": n_flagged,
                "fraction_flagged": float(n_flagged / max(1, group_scores.size)),
            }

            if known_labels is not None:
                known_flagged = flagged[known_mask]
                positives = int(np.sum(known_labels == 1))
                negatives = int(np.sum(known_labels == 0))
                tp = int(np.sum((known_labels == 1) & known_flagged))
                fp = int(np.sum((known_labels == 0) & known_flagged))
                if n_flagged > 0:
                    row["precision"] = float(tp / n_flagged)
                if positives > 0:
                    row["recall"] = float(tp / positives)
                if negatives > 0:
                    row["false_positive_rate"] = float(fp / negatives)

            threshold_rows.append(row)

        attacks[attack] = {
            "n_samples": int(group_scores.size),
            "label_counts": {
                "clean": int(clean_count),
                "backdoored": int(backdoored_count),
                "unknown": int(unknown_count),
            },
            "score_summary": summarize_scores(group_scores),
            "score_percentile_rank_summary": summarize_scores(group_ranks),
            "threshold_evaluation": threshold_rows,
        }

    return {
        "grouping_rule": grouping_rule,
        "n_attacks": int(len(attacks)),
        "attacks": attacks,
    }


def _context_from_run_dir(run_dir: Path) -> RunContext:
    output_root = run_dir.parents[1] if len(run_dir.parents) >= 2 else run_dir.parent
    features_dir = run_dir / "features"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"
    cache_root = output_root / "cache" / "features"

    for path in [features_dir, models_dir, reports_dir, plots_dir, logs_dir, cache_root]:
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_dir.name,
        run_dir=run_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
        cache_root=cache_root,
    )


def _prepare_supervised_run(
    *,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str,
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    random_state: int,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float],
    force_recompute_features: bool,
    feature_file: Path | None,
    feature_model_names_file: Path | None,
    feature_metadata_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    slurm_max_concurrent: str,
    slurm_cpus_per_task: str,
) -> dict[str, Any]:
    if not manifest_json.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_json}")

    mode = _detect_manifest_mode(manifest_json)
    if feature_file is None:
        if mode == "joint":
            train_items, infer_items = parse_joint_manifest_json(
                manifest_path=manifest_json,
                dataset_root=dataset_root,
            )
            all_items = train_items + infer_items
            train_indices = np.arange(0, len(train_items), dtype=np.int64)
            infer_indices = np.arange(len(train_items), len(all_items), dtype=np.int64)
        else:
            all_items = parse_single_manifest_json(
                manifest_path=manifest_json,
                dataset_root=dataset_root,
                section_key="path",
            )
            train_items = all_items
            infer_items = []
            train_indices = np.arange(0, len(all_items), dtype=np.int64)
            infer_indices = np.asarray([], dtype=np.int64)
    else:
        if mode == "joint":
            train_items, infer_items = parse_joint_manifest_json_by_model_name(
                manifest_path=manifest_json,
            )
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

    if not train_items:
        raise ValueError("No training items resolved from manifest")

    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_id,
    )

    feature_params = {
        "dtype": dtype_name,
        "block_size": int(stream_block_size),
        "spectral_features": list(spectral_features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": str(spectral_moment_source),
        "spectral_qv_sum_mode": str(spectral_qv_sum_mode),
    }
    extractor_warnings: list[str] = []
    if feature_file is None:
        bundle, artifacts, extractor_warnings = extract_with_cache(
            extractor_name="spectral",
            items=all_items,
            params=feature_params,
            cache_root=ctx.cache_root,
            run_features_dir=ctx.features_dir,
            force_recompute=force_recompute_features,
        )
        features = np.asarray(bundle.features, dtype=np.float32)
        extractor_metadata = dict(bundle.metadata)
    else:
        if force_recompute_features:
            extractor_warnings.append(
                "--force-recompute-features was ignored because --feature-file was provided"
            )

        resolved_model_names_file = (
            feature_model_names_file
            if feature_model_names_file is not None
            else feature_file.parent / "spectral_model_names.json"
        )
        resolved_metadata_file = (
            feature_metadata_file
            if feature_metadata_file is not None
            else (
                feature_file.parent / "spectral_metadata.json"
                if (feature_file.parent / "spectral_metadata.json").exists()
                else None
            )
        )
        features, external_metadata, external_warnings = _load_external_spectral_bundle(
            feature_file=feature_file,
            model_names_file=resolved_model_names_file,
            metadata_file=resolved_metadata_file,
            expected_model_names=[item.model_name for item in all_items],
            spectral_features=spectral_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
        )
        extractor_warnings.extend(external_warnings)

        run_feature_path = ctx.features_dir / "spectral_features.npy"
        np.save(run_feature_path, features.astype(np.float32, copy=False))

        run_metadata_path = ctx.features_dir / "spectral_metadata.json"
        extractor_metadata = {
            **external_metadata,
            "external_feature_source": str(feature_file),
            "external_model_names_source": str(resolved_model_names_file),
            "external_metadata_source": (
                str(resolved_metadata_file) if resolved_metadata_file is not None else None
            ),
            "loaded_external_features": True,
        }
        with open(run_metadata_path, "w", encoding="utf-8") as f:
            json.dump(json_ready(extractor_metadata), f, indent=2)

        artifacts = {
            "cache_key": None,
            "cache_hit": False,
            "feature_path": str(run_feature_path),
            "labels_path": None,
            "model_names_path": None,
            "metadata_path": str(run_metadata_path),
        }

    labels_values, labels_known, labels_raw = _labels_from_items(all_items)
    train_known = labels_known[train_indices]
    if not bool(np.all(train_known)):
        raise ValueError("Training samples must all have labels (label0/label1) for supervised learning")

    train_labels = labels_values[train_indices]
    cv_folds_resolved, cv_warnings = _sanitize_cv_folds(train_labels, cv_folds)
    requested_states = list(cv_random_states) if cv_random_states else [int(random_state)]
    dedup_states: list[int] = []
    seen_states: set[int] = set()
    for state in requested_states:
        state_int = int(state)
        if state_int in seen_states:
            continue
        seen_states.add(state_int)
        dedup_states.append(state_int)
    if not dedup_states:
        dedup_states = [int(random_state)]

    cv_split_groups: list[dict[str, Any]] = []
    for state in dedup_states:
        cv_split_groups.append(
            {
                "random_state": int(state),
                "cv_splits": _build_cv_splits(
                    train_indices=train_indices,
                    train_labels=train_labels,
                    cv_folds=cv_folds_resolved,
                    random_state=int(state),
                ),
            }
        )
    cv_splits = list(cv_split_groups[0]["cv_splits"])

    tasks: list[dict[str, Any]] = []
    model_names = registered_models() if model_name == "all" else [model_name]
    task_index = 0
    for selected_model in model_names:
        complexity = model_complexity_rank(selected_model)
        for params in candidate_params(selected_model):
            tasks.append(
                {
                    "task_index": int(task_index),
                    "model_name": selected_model,
                    "params": dict(params),
                    "complexity_rank": int(complexity),
                }
            )
            task_index += 1

    labels_value_path = ctx.features_dir / "supervised_label_values.npy"
    labels_known_path = ctx.features_dir / "supervised_label_known.npy"
    np.save(labels_value_path, labels_values)
    np.save(labels_known_path, labels_known.astype(np.int8))

    model_names_path = ctx.features_dir / "supervised_model_names.json"
    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump([item.model_name for item in all_items], f, indent=2)

    slurm_cpus = resolve_slurm_cpus_per_task(slurm_cpus_per_task)
    slurm_concurrency = resolve_slurm_max_concurrent(slurm_max_concurrent, slurm_cpus)

    task_dir = ctx.reports_dir / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    tuning_manifest = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(ctx.run_dir),
        "mode": mode,
        "manifest_json": str(manifest_json),
        "dataset_root": str(dataset_root),
        "data": {
            "n_samples": int(features.shape[0]),
            "train_indices": [int(x) for x in train_indices.tolist()],
            "infer_indices": [int(x) for x in infer_indices.tolist()],
            "feature_path": str(Path(artifacts["feature_path"])),
            "labels_value_path": str(labels_value_path),
            "labels_known_path": str(labels_known_path),
            "model_names_path": str(model_names_path),
        },
        "extractor": {
            "name": "spectral",
            "params": feature_params,
            "metadata": extractor_metadata,
            "warnings": extractor_warnings,
            "cache_key": artifacts["cache_key"],
            "cache_hit": artifacts["cache_hit"],
            "metadata_path": str(Path(artifacts["metadata_path"])),
        },
        "tuning": {
            "executor": tuning_executor,
            "model_name": model_name,
            "model_names": model_names,
            "metric": "roc_auc",
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
            "cv_random_states": [int(x) for x in dedup_states],
            "cv_folds_requested": int(cv_folds),
            "cv_folds_resolved": int(cv_folds_resolved),
            "cv_splits": cv_splits,
            "cv_split_groups": cv_split_groups,
            "tasks": tasks,
        },
        "runtime": {
            "slurm_partition": slurm_partition,
            "slurm_max_concurrent": int(slurm_concurrency),
            "slurm_cpus_per_task": int(slurm_cpus),
            "score_percentiles": [float(x) for x in score_percentiles],
        },
        "warnings": cv_warnings + list(extractor_warnings),
        "labels_preview": labels_raw,
    }

    tuning_manifest_path = ctx.reports_dir / "tuning_manifest.json"
    with open(tuning_manifest_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(tuning_manifest), f, indent=2)

    return {
        "run_dir": str(ctx.run_dir),
        "tuning_manifest": str(tuning_manifest_path),
        "n_tasks": len(tasks),
        "task_dir": str(task_dir),
        "slurm_partition": slurm_partition,
        "slurm_max_concurrent": int(slurm_concurrency),
        "slurm_cpus_per_task": int(slurm_cpus),
        "warnings": tuning_manifest["warnings"],
    }


def _run_supervised_worker(
    *,
    run_dir: Path,
    task_index: int,
    n_jobs: int | None,
) -> dict[str, Any]:
    tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = manifest["tuning"]["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task_index={task_index} out of range [0, {len(tasks) - 1}]")

    task = tasks[task_index]
    features = np.load(manifest["data"]["feature_path"])
    labels = np.load(manifest["data"]["labels_value_path"]).astype(np.int32)
    tuning_cfg = manifest["tuning"]
    if "cv_split_groups" in tuning_cfg:
        cv_split_groups = list(tuning_cfg["cv_split_groups"])
    else:
        cv_split_groups = [
            {
                "random_state": int(tuning_cfg["random_state"]),
                "cv_splits": list(tuning_cfg["cv_splits"]),
            }
        ]
    resolved_n_jobs = int(n_jobs) if n_jobs is not None else int(manifest["tuning"]["n_jobs"])

    start = perf_counter()
    try:
        result = _evaluate_candidate(
            features=features,
            labels=labels,
            task=task,
            cv_split_groups=cv_split_groups,
            n_jobs=resolved_n_jobs,
        )
    except Exception as exc:  # pragma: no cover - failure path asserted via output file shape
        result = {
            "task_index": int(task["task_index"]),
            "model_name": str(task["model_name"]),
            "params": dict(task["params"]),
            "complexity_rank": int(task["complexity_rank"]),
            "status": "error",
            "error": str(exc),
            "fold_results": [],
            "roc_auc_mean": None,
            "roc_auc_std": None,
        }

    result["elapsed_seconds"] = float(perf_counter() - start)

    task_dir = run_dir / "reports" / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = _task_result_path(task_dir, task_index)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2)

    return {
        "run_dir": str(run_dir),
        "task_index": int(task_index),
        "result_path": str(out_path),
        "status": result.get("status"),
    }


def _select_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in candidates if row.get("status") == "ok" and row.get("roc_auc_mean") is not None]
    if not valid:
        raise RuntimeError("No successful tuning candidates available to select a winner")

    ranked = sorted(
        valid,
        key=lambda row: (
            -float(row["roc_auc_mean"]),
            float(row["roc_auc_std"]) if row.get("roc_auc_std") is not None else float("inf"),
            int(row.get("complexity_rank", 10**9)),
            int(row["task_index"]),
        ),
    )
    return ranked[0]


def _save_model(model: Any, path: Path) -> None:
    if joblib is not None:
        joblib.dump(model, path)
        return
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _finalize_supervised_run(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
) -> dict[str, Any]:
    ctx = _context_from_run_dir(run_dir)
    tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if score_percentiles is None:
        score_percentiles = [float(x) for x in manifest["runtime"]["score_percentiles"]]

    task_dir = run_dir / "reports" / "tuning_tasks"
    task_results: list[dict[str, Any]] = []
    missing: list[int] = []
    for task in manifest["tuning"]["tasks"]:
        task_index = int(task["task_index"])
        path = _task_result_path(task_dir, task_index)
        if not path.exists():
            missing.append(task_index)
            continue
        with open(path, "r", encoding="utf-8") as f:
            task_results.append(json.load(f))

    if missing:
        raise RuntimeError(f"Missing tuning task outputs for indices: {missing[:10]}")

    winner = _select_winner(task_results)
    features = np.load(manifest["data"]["feature_path"])
    labels_value = np.load(manifest["data"]["labels_value_path"]).astype(np.int32)
    labels_known = np.load(manifest["data"]["labels_known_path"]).astype(bool)
    with open(manifest["data"]["model_names_path"], "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]

    train_indices = np.asarray(manifest["data"]["train_indices"], dtype=np.int64)
    infer_indices = np.asarray(manifest["data"]["infer_indices"], dtype=np.int64)
    x_train = features[train_indices]
    y_train = labels_value[train_indices]

    model = create(
        str(winner["model_name"]),
        params=dict(winner["params"]),
        random_state=int(manifest["tuning"]["random_state"]),
    )
    model.fit(x_train, y_train)
    train_scores = _predict_scores(model, x_train)
    threshold_specs = _build_threshold_specs(
        train_scores=np.asarray(train_scores, dtype=np.float64),
        percentiles=[float(x) for x in score_percentiles],
    )

    model_path = ctx.models_dir / "best_model.joblib"
    _save_model(model, model_path)
    ctx.add_artifact("best_model", model_path)

    train_model_names = [model_names[int(i)] for i in train_indices.tolist()]
    train_labels_raw = [int(x) for x in y_train.tolist()]
    train_scores_csv = ctx.reports_dir / "train_scores.csv"
    save_score_csv(
        output_path=train_scores_csv,
        model_names=train_model_names,
        labels=train_labels_raw,
        scores=train_scores,
    )
    ctx.add_artifact("train_scores_csv", train_scores_csv)

    infer_scores_csv: Path | None = None
    inference_summary: dict[str, Any] | None = None
    threshold_rows: list[dict[str, Any]] = []
    infer_offline_metrics: dict[str, Any] | None = None
    infer_model_names: list[str] = []
    infer_labels_raw: list[int | None] = []
    infer_scores: np.ndarray | None = None

    if infer_indices.size > 0:
        x_infer = features[infer_indices]
        infer_scores = _predict_scores(model, x_infer)
        infer_model_names = [model_names[int(i)] for i in infer_indices.tolist()]

        infer_known_mask = labels_known[infer_indices]
        for i in infer_indices.tolist():
            raw = int(labels_value[int(i)])
            infer_labels_raw.append(None if raw < 0 else raw)

        infer_labels_np: np.ndarray | None = None
        if bool(np.all(infer_known_mask)):
            infer_labels_np = np.asarray([int(x) for x in infer_labels_raw], dtype=np.int32)

        infer_scores_csv = ctx.reports_dir / "inference_scores.csv"
        save_score_csv(
            output_path=infer_scores_csv,
            model_names=infer_model_names,
            labels=infer_labels_raw,
            scores=infer_scores,
        )
        ctx.add_artifact("inference_scores_csv", infer_scores_csv)

        threshold_rows = compute_infer_threshold_rows(
            train_scores=np.asarray(train_scores, dtype=np.float64),
            infer_scores=np.asarray(infer_scores, dtype=np.float64),
            percentiles=[float(x) for x in score_percentiles],
            infer_labels=infer_labels_np,
        )
        infer_offline_metrics = compute_offline_metrics(infer_labels_np, np.asarray(infer_scores, dtype=np.float64))
        inference_summary = summarize_scores(np.asarray(infer_scores, dtype=np.float64))

    train_offline_metrics = compute_offline_metrics(np.asarray(y_train, dtype=np.int32), np.asarray(train_scores, dtype=np.float64))
    attack_analysis = {
        "train": _summarize_attack_groups(
            model_names=train_model_names,
            labels=train_labels_raw,
            scores=np.asarray(train_scores, dtype=np.float64),
            threshold_specs=threshold_specs,
        ),
        "inference": (
            _summarize_attack_groups(
                model_names=infer_model_names,
                labels=infer_labels_raw,
                scores=np.asarray(infer_scores, dtype=np.float64),
                threshold_specs=threshold_specs,
            )
            if infer_scores is not None
            else None
        ),
    }

    report = {
        "data_info": {
            "mode": manifest["mode"],
            "n_samples": int(features.shape[0]),
            "n_train": int(train_indices.size),
            "n_inference": int(infer_indices.size),
            "n_train_clean": int(np.sum(y_train == 0)),
            "n_train_backdoored": int(np.sum(y_train == 1)),
            "n_inference_clean": (
                int(np.sum(labels_value[infer_indices] == 0))
                if infer_indices.size > 0
                else 0
            ),
            "n_inference_backdoored": (
                int(np.sum(labels_value[infer_indices] == 1))
                if infer_indices.size > 0
                else 0
            ),
            "n_inference_unknown_label": (
                int(np.sum(labels_value[infer_indices] < 0))
                if infer_indices.size > 0
                else 0
            ),
        },
        "representation": {
            "extractor": manifest["extractor"]["name"],
            "extractor_params": manifest["extractor"]["params"],
            "extractor_metadata": _compact_extractor_metadata(manifest["extractor"]["metadata"]),
            "feature_path": manifest["data"]["feature_path"],
        },
        "tuning": {
            "metric": "roc_auc",
            "model_name": manifest["tuning"]["model_name"],
            "model_names": manifest["tuning"].get("model_names", [manifest["tuning"]["model_name"]]),
            "cv_random_states": manifest["tuning"].get("cv_random_states", [manifest["tuning"]["random_state"]]),
            "cv_folds_resolved": manifest["tuning"]["cv_folds_resolved"],
            "executor": manifest["tuning"]["executor"],
            "tasks_total": len(manifest["tuning"]["tasks"]),
            "candidates": task_results,
            "winner": winner,
        },
        "fit_assessment": {
            "score_definition": "positive_class_score",
            "train_score_summary": summarize_scores(np.asarray(train_scores, dtype=np.float64)),
            "train_offline_metrics": train_offline_metrics,
            "inference_score_summary": inference_summary,
            "threshold_evaluation": threshold_rows,
            "offline_metrics": infer_offline_metrics,
        },
        "attack_analysis": attack_analysis,
        "warnings": manifest.get("warnings", []),
    }

    report_path = ctx.reports_dir / "supervised_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)
    ctx.add_artifact("report", report_path)

    winner_exports = export_winner_feature_weights(
        run_dir=run_dir,
        report_path=report_path,
        manifest_path=tuning_manifest_path,
        artifact_index_path=run_dir / "artifact_index.json",
    )
    ctx.add_artifact("winner_feature_weights_coefficients_csv", winner_exports["coefficient_csv"])
    ctx.add_artifact("winner_feature_weights_by_metric_csv", winner_exports["metric_csv"])
    ctx.add_artifact("winner_feature_weights_by_block_csv", winner_exports["block_csv"])
    ctx.add_artifact("winner_feature_weights_metadata_json", winner_exports["metadata_json"])

    ctx.add_artifact("tuning_manifest", tuning_manifest_path)
    ctx.add_artifact("tuning_tasks_dir", task_dir)

    run_config = {
        "pipeline": PIPELINE_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": manifest["manifest_json"],
        "dataset_root": manifest["dataset_root"],
        "mode": manifest["mode"],
        "tuning_executor": manifest["tuning"]["executor"],
        "model_name": manifest["tuning"]["model_name"],
        "model_names": manifest["tuning"].get("model_names", [manifest["tuning"]["model_name"]]),
        "cv_random_states": manifest["tuning"].get("cv_random_states", [manifest["tuning"]["random_state"]]),
        "score_percentiles": [float(x) for x in score_percentiles],
        "winner": winner,
        "warnings": manifest.get("warnings", []),
    }
    ctx.finalize(run_config)

    return {
        "run_dir": str(run_dir),
        "report": str(report_path),
        "train_scores_csv": str(train_scores_csv),
        "inference_scores_csv": str(infer_scores_csv) if infer_scores_csv is not None else None,
        "best_model": str(model_path),
    }


def run_supervised_pipeline(
    *,
    manifest_json: Path | None,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str,
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    random_state: int,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float] | None,
    force_recompute_features: bool,
    feature_file: Path | None,
    feature_model_names_file: Path | None,
    feature_metadata_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    slurm_max_concurrent: str,
    slurm_cpus_per_task: str,
    stage: str,
    run_dir: Path | None,
    task_index: int | None,
) -> dict[str, Any]:
    if stage in {"all", "prepare"} and manifest_json is None:
        raise ValueError("--manifest-json is required for stage=all and stage=prepare")
    if stage in {"worker", "finalize"} and run_dir is None:
        raise ValueError("--run-dir is required for stage=worker and stage=finalize")
    if stage == "worker" and task_index is None:
        raise ValueError("--task-index is required for stage=worker")

    if stage == "prepare":
        return _prepare_supervised_run(
            manifest_json=Path(manifest_json) if manifest_json is not None else Path(""),
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=run_id,
            model_name=model_name,
            spectral_features=spectral_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            cv_folds=cv_folds,
            random_state=random_state,
            cv_random_states=cv_random_states,
            n_jobs=n_jobs,
            score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
            force_recompute_features=force_recompute_features,
            feature_file=feature_file,
            feature_model_names_file=feature_model_names_file,
            feature_metadata_file=feature_metadata_file,
            tuning_executor=tuning_executor,
            slurm_partition=slurm_partition,
            slurm_max_concurrent=slurm_max_concurrent,
            slurm_cpus_per_task=slurm_cpus_per_task,
        )

    if stage == "worker":
        return _run_supervised_worker(
            run_dir=Path(run_dir),
            task_index=int(task_index) if task_index is not None else -1,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
        )

    if stage == "finalize":
        return _finalize_supervised_run(
            run_dir=Path(run_dir),
            score_percentiles=score_percentiles,
        )

    # stage == "all"
    prepared = _prepare_supervised_run(
        manifest_json=Path(manifest_json) if manifest_json is not None else Path(""),
        dataset_root=dataset_root,
        output_root=output_root,
        run_id=run_id,
        model_name=model_name,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        cv_folds=cv_folds,
        random_state=random_state,
        cv_random_states=cv_random_states,
        n_jobs=n_jobs,
        score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
        force_recompute_features=force_recompute_features,
        feature_file=feature_file,
        feature_model_names_file=feature_model_names_file,
        feature_metadata_file=feature_metadata_file,
        tuning_executor=tuning_executor,
        slurm_partition=slurm_partition,
        slurm_max_concurrent=slurm_max_concurrent,
        slurm_cpus_per_task=slurm_cpus_per_task,
    )
    resolved_run_dir = Path(prepared["run_dir"])

    if tuning_executor == "slurm_array":
        n_tasks = int(prepared["n_tasks"])
        max_concurrent = int(prepared["slurm_max_concurrent"])
        return {
            **prepared,
            "next_steps": build_slurm_array_next_steps(
                run_dir=resolved_run_dir,
                n_tasks=n_tasks,
                max_concurrent=max_concurrent,
            ),
        }

    for idx in range(int(prepared["n_tasks"])):
        _run_supervised_worker(
            run_dir=resolved_run_dir,
            task_index=idx,
            n_jobs=n_jobs,
        )

    finalized = _finalize_supervised_run(
        run_dir=resolved_run_dir,
        score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
    )
    return {
        **finalized,
        "tuning_manifest": prepared["tuning_manifest"],
        "n_tasks": prepared["n_tasks"],
    }
