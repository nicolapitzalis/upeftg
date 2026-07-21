from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ...utilities.core.serialization import json_ready
from ..data.preparation import (
    load_dataset_group_names_for_run as _load_dataset_group_names_for_tuning_manifest,
    load_features_for_tuning_manifest as _load_features_for_tuning_manifest,
    load_training_inputs_for_run,
)
from ..data.normalization import (
    INPUT_NORMALIZATION_NONE,
    resolve_input_normalization as _resolve_input_normalization,
)
from ..tasks import resolve_selection_metric as _resolve_selection_metric, task_spec_from_payload
from .tuning import _evaluate_candidate


def select_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in candidates if row.get("status") == "ok" and row.get("selection_metric_mean") is not None]
    if not valid:
        singleton_no_cv = [
            row for row in candidates if row.get("status") == "ok" and row.get("execution_mode") == "singleton_no_cv"
        ]
        if len(singleton_no_cv) == 1:
            return singleton_no_cv[0]
        raise RuntimeError("No successful tuning candidates available to select a winner")

    ranked = sorted(
        valid,
        key=lambda row: (
            -float(row["selection_metric_mean"]),
            (float(row["selection_metric_std"]) if row.get("selection_metric_std") is not None else float("inf")),
            int(row.get("complexity_rank", 10**9)),
            int(row["task_index"]),
        ),
    )
    return ranked[0]


def task_result_path(task_dir: Path, task_index: int) -> Path:
    return task_dir / f"task_{task_index:04d}.json"


def input_normalization_from_manifest(manifest: dict[str, Any]) -> str:
    tuning = manifest.get("tuning")
    if isinstance(tuning, dict) and tuning.get("input_normalization"):
        return _resolve_input_normalization(str(tuning["input_normalization"]))
    config = manifest.get("input_normalization")
    if isinstance(config, dict) and config.get("mode"):
        return _resolve_input_normalization(str(config["mode"]))
    return INPUT_NORMALIZATION_NONE


def run_supervised_worker(
    *,
    run_dir: Path,
    task_index: int,
    n_jobs: int | None,
) -> dict[str, Any]:
    tuning_manifest_path = run_dir / ".work" / "tuning.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    tasks = manifest["tuning"]["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task_index={task_index} out of range [0, {len(tasks) - 1}]")

    task = tasks[task_index]
    task_spec = task_spec_from_payload(manifest.get("task"))
    tuning_cfg = manifest["tuning"]
    selection_metric_name = _resolve_selection_metric(
        tuning_cfg.get("metric"),
        task_spec=task_spec,
    )
    execution_mode = str(tuning_cfg.get("execution_mode", "cross_validation"))
    input_normalization = input_normalization_from_manifest(manifest)
    if execution_mode == "singleton_no_cv":
        started_at = datetime.now(timezone.utc)
        start = perf_counter()
        ended_at = datetime.now(timezone.utc)
        result = {
            "task_index": int(task["task_index"]),
            "model_name": str(task["model_name"]),
            "params": dict(task["params"]),
            "complexity_rank": int(task["complexity_rank"]),
            "normalization_policy": str(task["normalization_policy"]),
            "base_normalization_policy": task.get("base_normalization_policy"),
            "input_normalization": str(task.get("input_normalization", input_normalization)),
            "status": "ok",
            "execution_mode": execution_mode,
            "selection_metric_name": str(selection_metric_name),
            "selection_metric_mean": None,
            "selection_metric_std": None,
            "fold_results": [],
            "seed_results": [],
            "elapsed_seconds": float(perf_counter() - start),
            "start_timestamp_utc": started_at.isoformat(),
            "end_timestamp_utc": ended_at.isoformat(),
        }
        if task_spec.is_binary:
            result["roc_auc_mean"] = None
            result["roc_auc_std"] = None
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None
        else:
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None

        task_dir = run_dir / ".work" / "tuning_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_result_path(task_dir, task_index)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_ready(result), f, indent=2)

        return {
            "run_dir": str(run_dir),
            "task_index": int(task_index),
            "result_path": str(out_path),
            "status": result.get("status"),
        }

    features = _load_features_for_tuning_manifest(manifest)
    prepared_inputs = load_training_inputs_for_run(manifest)
    labels = prepared_inputs["label_values"].astype(np.int32)
    model_names = [str(value) for value in prepared_inputs["model_names"].tolist()]
    dataset_group_names = _load_dataset_group_names_for_tuning_manifest(manifest)
    rank_values = None
    if "rank_values" in prepared_inputs:
        rank_values = prepared_inputs["rank_values"].astype(np.int64)
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

    started_at = datetime.now(timezone.utc)
    start = perf_counter()
    try:
        result = _evaluate_candidate(
            features=features,
            labels=labels,
            task=task,
            cv_split_groups=cv_split_groups,
            n_jobs=resolved_n_jobs,
            task_spec=task_spec,
            selection_metric_name=selection_metric_name,
            domain_adaptation=manifest.get("domain_adaptation"),
            rank_values=rank_values,
            dataset_group_names=dataset_group_names,
            input_normalization=input_normalization,
            model_names=model_names,
            artifact_dir=(
                run_dir
                / ".work"
                / "tuning_tasks"
                / f"task_{int(task_index):04d}"
                / "folds"
            ),
            persist_fold_models=bool(tuning_cfg.get("no_refit", False)),
        )
    except Exception as exc:  # pragma: no cover - failure path asserted via output file shape
        result = {
            "task_index": int(task["task_index"]),
            "model_name": str(task["model_name"]),
            "params": dict(task["params"]),
            "complexity_rank": int(task["complexity_rank"]),
            "normalization_policy": str(task["normalization_policy"]),
            "base_normalization_policy": task.get("base_normalization_policy"),
            "input_normalization": str(task.get("input_normalization", input_normalization)),
            "status": "error",
            "error": str(exc),
            "execution_mode": execution_mode,
            "selection_metric_name": str(selection_metric_name),
            "selection_metric_mean": None,
            "selection_metric_std": None,
            "fold_results": [],
            "seed_results": [],
        }
        if task_spec.is_binary:
            result["roc_auc_mean"] = None
            result["roc_auc_std"] = None
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None
        else:
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None

    ended_at = datetime.now(timezone.utc)
    result["elapsed_seconds"] = float(perf_counter() - start)
    result["start_timestamp_utc"] = started_at.isoformat()
    result["end_timestamp_utc"] = ended_at.isoformat()

    task_dir = run_dir / ".work" / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_result_path(task_dir, task_index)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2)

    return {
        "run_dir": str(run_dir),
        "task_index": int(task_index),
        "result_path": str(out_path),
        "status": result.get("status"),
    }
