from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

from ..features.registry import extract_features
from ..features.spectral import (
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
)
from ..supervised.pipeline import run_supervised_pipeline
from ..utilities.artifacts.aggregate_features import aggregate_features
from ..utilities.artifacts.dataset_references import (
    build_dataset_reference_payload_from_items,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from ..utilities.core.manifest import parse_single_manifest_json, resolve_manifest_path
from ..utilities.core.paths import default_dataset_root
from ..utilities.core.run_context import create_run_context
from ..utilities.core.serialization import json_ready
from ..utilities.merge.merge_feature_files import _resolve_output_feature_path


DEFAULT_CNN_FEATURES = list(DEFAULT_SPECTRAL_FEATURES)
DEFAULT_CNN_AGGREGATION_LAYOUT = "layer_sequence"
DEFAULT_CNN_MODEL = "cnn_1d"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _json_write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return path


def _timing_start(stage: str, *, backend: str) -> tuple[dict[str, Any], float]:
    now = _utc_now()
    return (
        {
            "stage": stage,
            "backend": backend,
            "start_timestamp_utc": now.isoformat(),
            "start_epoch_seconds": float(now.timestamp()),
        },
        perf_counter(),
    )


def _timing_finish(timing: dict[str, Any], started_perf: float) -> dict[str, Any]:
    now = _utc_now()
    timing.update(
        {
            "end_timestamp_utc": now.isoformat(),
            "end_epoch_seconds": float(now.timestamp()),
            "elapsed_seconds": float(perf_counter() - started_perf),
        }
    )
    return timing


def _coerce_features(features: list[str] | tuple[str, ...] | None) -> list[str]:
    if features is None:
        return list(DEFAULT_CNN_FEATURES)
    return [str(feature) for feature in features]


def _feature_params(
    *,
    features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    stream_block_size: int,
    dtype_name: str,
) -> dict[str, Any]:
    return {
        "block_size": int(stream_block_size),
        "dtype": str(dtype_name),
        "spectral_features": list(features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": str(spectral_moment_source),
        "spectral_qv_sum_mode": str(spectral_qv_sum_mode),
        "spectral_entrywise_delta_mode": str(spectral_entrywise_delta_mode),
    }


def _feature_extract_path(output_root: Path, run_id: str, *, slurm: bool) -> Path:
    subdir = "merged" if slurm else "features"
    return (Path(output_root).expanduser().resolve() / "feature_extract" / run_id / subdir / "spectral_features.npy")


def _aggregate_output_name(run_id: str) -> Path:
    return Path(f"{run_id}_cnn_layer_sequence")


def _aggregate_feature_root(output_root: Path) -> Path:
    return Path(output_root).expanduser().resolve() / "feature_extract"


def _submission_context(output_root: Path, pipeline: str, run_id: str | None):
    return create_run_context(
        pipeline=pipeline,
        output_root=Path(output_root).expanduser().resolve(),
        run_id=run_id,
    )


def _stringify_env(env: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in env.items() if value is not None}


def _sbatch_command(
    script: Path,
    *,
    partition: str,
    dependency: str | None = None,
    wait: bool = False,
    cpus_per_task: str | None = None,
) -> list[str]:
    command = ["sbatch", "--parsable", "--partition", str(partition)]
    if cpus_per_task:
        command.extend(["--cpus-per-task", str(cpus_per_task)])
    if dependency:
        command.extend(["--dependency", dependency])
    if wait:
        command.append("--wait")
    command.append(str(script))
    return command


def _run_sbatch(
    script_name: str,
    *,
    env: dict[str, Any],
    partition: str,
    dependency: str | None = None,
    wait: bool = False,
    cpus_per_task: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    script = _project_root() / "sbatch" / script_name
    command = _sbatch_command(
        script,
        partition=partition,
        dependency=dependency,
        wait=wait,
        cpus_per_task=cpus_per_task,
    )
    payload = {
        "command": command,
        "env": _stringify_env(env),
        "dependency": dependency,
        "wait": bool(wait),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        payload["job_id"] = None
        return payload

    process_env = os.environ.copy()
    process_env.update(payload["env"])
    completed = subprocess.run(
        command,
        cwd=_project_root(),
        env=process_env,
        check=True,
        text=True,
        capture_output=True,
    )
    stdout = completed.stdout.strip()
    payload["stdout"] = stdout
    payload["stderr"] = completed.stderr.strip()
    payload["job_id"] = stdout.splitlines()[-1].split(";", 1)[0] if stdout else None
    return payload


def _resolve_partition_cpus_for_submit(partition: str, requested: str) -> str:
    requested_text = str(requested).strip()
    if requested_text and requested_text != "auto":
        return requested_text
    try:
        completed = subprocess.run(
            ["sinfo", "-h", "-p", str(partition), "-o", "%c"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return "32"
    values: list[int] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            values.append(int(text))
    return str(max(values)) if values else "32"


def _read_slurm_dependency_job_id(index_path: Path) -> str | None:
    if not index_path.exists():
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    value = payload.get("final_dependency_job_id")
    return str(value) if value else None


def run_cnn_extract(
    *,
    manifest_json: Path,
    dataset_root: Path = default_dataset_root(),
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    worker_cpus: str = "auto",
    max_concurrent: str = "auto",
    dry_run: bool = False,
    features: list[str] | None = None,
    spectral_sv_top_k: int = 8,
    spectral_moment_source: str = DEFAULT_SPECTRAL_MOMENT_SOURCE,
    spectral_qv_sum_mode: str = DEFAULT_SPECTRAL_QV_SUM_MODE,
    spectral_entrywise_delta_mode: str = DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    stream_block_size: int = 131072,
    dtype_name: str = "float32",
    n_shards: int = 8,
    wait_for_job_index: bool = False,
) -> dict[str, Any]:
    resolved_features = _coerce_features(features)
    if backend == "slurm":
        ctx = _submission_context(output_root, "cnn_extract", run_id)
        timing, started_perf = _timing_start("cnn_extract_submit", backend="slurm")
        slurm_index_path = ctx.reports_dir / "slurm_job_index.json"
        env = {
            "REPO_ROOT": _project_root(),
            "MANIFEST_JSON": resolve_manifest_path(manifest_json),
            "DATASET_ROOT": Path(dataset_root).expanduser().resolve(),
            "OUTPUT_ROOT": Path(output_root).expanduser().resolve(),
            "RUN_ID": ctx.run_id,
            "FEATURES": " ".join(resolved_features),
            "SV_TOP_K": int(spectral_sv_top_k),
            "SPECTRAL_MOMENT_SOURCE": spectral_moment_source,
            "SPECTRAL_QV_SUM_MODE": spectral_qv_sum_mode,
            "SPECTRAL_ENTRYWISE_DELTA_MODE": spectral_entrywise_delta_mode,
            "STREAM_BLOCK_SIZE": int(stream_block_size),
            "DTYPE": dtype_name,
            "N_SHARDS": int(n_shards),
            "SLURM_PARTITION": partition,
            "SLURM_CPUS_PER_TASK": worker_cpus,
            "SLURM_MAX_CONCURRENT": max_concurrent,
            "SLURM_JOB_INDEX_PATH": slurm_index_path,
        }
        submission = _run_sbatch(
            "feature_extract_array.sh",
            env=env,
            partition=partition,
            wait=wait_for_job_index,
            dry_run=dry_run,
        )
        timing = _timing_finish(timing, started_perf)
        timings = {"submit": timing}
        artifacts = {
            "slurm_job_index": str(slurm_index_path),
            "feature_path": str(_feature_extract_path(Path(output_root), ctx.run_id, slurm=True)),
        }
        run_config = {
            "pipeline": "cnn_extract",
            "backend": "slurm",
            "manifest_json": str(resolve_manifest_path(manifest_json)),
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "features": resolved_features,
            "slurm": submission,
        }
        _json_write(ctx.run_dir / "run_config.json", run_config)
        _json_write(ctx.run_dir / "artifact_index.json", artifacts)
        _json_write(ctx.run_dir / "timings.json", timings)
        return {
            "run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "feature_path": artifacts["feature_path"],
            "slurm_job_index": artifacts["slurm_job_index"],
            "backend": "slurm",
            "slurm": submission,
            "timings": timings,
        }

    timing, started_perf = _timing_start("cnn_extract", backend="local")
    manifest_json = resolve_manifest_path(manifest_json)
    output_root = Path(output_root).expanduser().resolve()
    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=Path(dataset_root).expanduser().resolve(),
        section_key="path",
    )
    ctx = create_run_context(
        pipeline="feature_extract",
        output_root=output_root,
        run_id=run_id,
    )
    params = _feature_params(
        features=resolved_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
    )
    bundle, artifacts, warnings = extract_features(
        extractor_name="spectral",
        items=items,
        params=params,
        run_features_dir=ctx.features_dir,
    )
    kept_names = set(bundle.model_names)
    kept_items = [item for item in items if item.model_name in kept_names]
    dataset_reference_payload = build_dataset_reference_payload_from_items(
        items=kept_items,
        artifact_kind="feature_extract",
        manifest_json=manifest_json,
        dataset_root=Path(dataset_root).expanduser().resolve(),
        artifact_model_count=len(bundle.model_names),
        source_artifacts=[str(manifest_json)],
    )
    dataset_reference_report = write_dataset_reference_report(
        default_dataset_reference_report_path(ctx.reports_dir),
        dataset_reference_payload,
    )
    timing = _timing_finish(timing, started_perf)
    report = {
        "timestamp_utc": _iso_now(),
        "timing": timing,
        "elapsed_seconds": timing["elapsed_seconds"],
        "extractor": "spectral",
        "feature_shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        "labels_available": bool(bundle.labels is not None),
        "model_count": len(bundle.model_names),
        "warnings": warnings,
        "metadata": bundle.metadata,
        "dataset_reference_report_path": str(dataset_reference_report),
    }
    report_path = _json_write(ctx.reports_dir / "feature_extraction_report.json", report)
    artifact_index = {
        "features": artifacts["feature_path"],
        "labels": artifacts.get("labels_path"),
        "feature_metadata": artifacts["metadata_path"],
        "dataset_reference_report": str(dataset_reference_report),
        "report": str(report_path),
    }
    run_config = {
        "pipeline": "feature_extract",
        "stable_pipeline": "cnn_extract",
        "backend": "local",
        "timestamp_utc": _iso_now(),
        "manifest_json": str(manifest_json),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "extractor": "spectral",
        "extractor_params": params,
        "warnings": warnings,
    }
    timings = {"cnn_extract": timing}
    _json_write(ctx.run_dir / "run_config.json", run_config)
    _json_write(ctx.run_dir / "artifact_index.json", artifact_index)
    _json_write(ctx.run_dir / "timings.json", timings)
    return {
        "run_dir": str(ctx.run_dir),
        "run_id": ctx.run_id,
        "feature_path": artifacts["feature_path"],
        "report": str(report_path),
        "dataset_reference_report": str(dataset_reference_report),
        "warnings": warnings,
        "backend": "local",
        "timings": timings,
    }


def run_cnn_aggregate(
    *,
    feature_file: Path,
    output_filename: Path | None = None,
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    worker_cpus: str = "auto",
    max_concurrent: str = "auto",
    dry_run: bool = False,
    feature_root: Path | None = None,
    features: list[str] | None = None,
    spectral_qv_sum_mode: str = "append",
    dependency: str | None = None,
) -> dict[str, Any]:
    resolved_features = _coerce_features(features)
    output_root = Path(output_root).expanduser().resolve()
    ctx = _submission_context(output_root, "cnn_aggregate", run_id)
    feature_root = Path(feature_root) if feature_root is not None else _aggregate_feature_root(output_root)
    output_filename = Path(output_filename) if output_filename is not None else _aggregate_output_name(ctx.run_id)

    if backend == "slurm":
        timing, started_perf = _timing_start("cnn_aggregate_submit", backend="slurm")
        env = {
            "REPO_ROOT": _project_root(),
            "FEATURE_FILE": Path(feature_file),
            "OUTPUT_FILENAME": output_filename,
            "OUTPUT_ROOT": output_root,
            "RUN_ID": ctx.run_id,
            "FEATURE_ROOT": feature_root,
            "FEATURES": " ".join(resolved_features),
            "SPECTRAL_QV_SUM_MODE": spectral_qv_sum_mode,
            "SLURM_CPUS_PER_TASK": worker_cpus,
            "SLURM_MAX_CONCURRENT": max_concurrent,
            "SLURM_PARTITION": partition,
        }
        submission = _run_sbatch(
            "cnn_aggregate.sh",
            env=env,
            partition=partition,
            dependency=dependency,
            cpus_per_task=_resolve_partition_cpus_for_submit(partition, worker_cpus),
            dry_run=dry_run,
        )
        timing = _timing_finish(timing, started_perf)
        artifacts = {
            "feature_path": str(_resolve_output_feature_path(output_filename, feature_root=feature_root)),
        }
        _json_write(
            ctx.run_dir / "run_config.json",
            {
                "pipeline": "cnn_aggregate",
                "backend": "slurm",
                "feature_file": str(feature_file),
                "output_filename": str(output_filename),
                "feature_root": str(feature_root),
                "features": resolved_features,
                "spectral_qv_sum_mode": spectral_qv_sum_mode,
                "slurm": submission,
            },
        )
        _json_write(ctx.run_dir / "artifact_index.json", artifacts)
        _json_write(ctx.run_dir / "timings.json", {"submit": timing})
        return {
            "run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "feature_path": artifacts["feature_path"],
            "backend": "slurm",
            "slurm": submission,
            "timings": {"submit": timing},
        }

    timing, started_perf = _timing_start("cnn_aggregate", backend="local")
    outputs = aggregate_features(
        feature_file=Path(feature_file),
        output_filename=output_filename,
        feature_root=feature_root,
        operator="avg",
        features=resolved_features,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        layout=DEFAULT_CNN_AGGREGATION_LAYOUT,
    )
    timing = _timing_finish(timing, started_perf)
    artifacts = {key: (str(value) if value is not None else None) for key, value in outputs.items()}
    _json_write(
        ctx.run_dir / "run_config.json",
        {
            "pipeline": "cnn_aggregate",
            "backend": "local",
            "feature_file": str(feature_file),
            "output_filename": str(output_filename),
            "feature_root": str(feature_root),
            "features": resolved_features,
            "spectral_qv_sum_mode": spectral_qv_sum_mode,
            "layout": DEFAULT_CNN_AGGREGATION_LAYOUT,
        },
    )
    _json_write(ctx.run_dir / "artifact_index.json", artifacts)
    _json_write(ctx.run_dir / "timings.json", {"cnn_aggregate": timing})
    return {
        "run_dir": str(ctx.run_dir),
        "run_id": ctx.run_id,
        "feature_path": str(outputs["feature_path"]),
        "outputs": artifacts,
        "backend": "local",
        "timings": {"cnn_aggregate": timing},
    }


def _supervised_common_kwargs(
    *,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    feature_file: Path,
    features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    random_state: int,
    train_split: int,
    calibration_split: int | None,
    accepted_fpr: list[float] | None,
    split_by_folder: bool,
    cv_seeds: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float] | None,
    cnn_hyperparams: Path | None,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
    class_weight_loss: bool,
    rank_label_weight_loss: bool,
    selection_metric: str | None,
    skip_feature_importance: bool,
) -> dict[str, Any]:
    return {
        "manifest_json": manifest_json,
        "dataset_root": dataset_root,
        "output_root": output_root,
        "run_id": run_id,
        "model_name": DEFAULT_CNN_MODEL,
        "spectral_features": _coerce_features(features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": spectral_moment_source,
        "spectral_qv_sum_mode": spectral_qv_sum_mode,
        "spectral_entrywise_delta_mode": spectral_entrywise_delta_mode,
        "stream_block_size": int(stream_block_size),
        "dtype_name": dtype_name,
        "cv_folds": int(cv_folds),
        "random_state": int(random_state),
        "train_split_percent": int(train_split),
        "calibration_split_percent": calibration_split,
        "accepted_fpr": accepted_fpr,
        "split_by_folder": bool(split_by_folder),
        "cv_random_states": cv_seeds,
        "n_jobs": int(n_jobs),
        "score_percentiles": score_percentiles,
        "feature_file": feature_file,
        "tuning_executor": "local",
        "slurm_partition": "extra",
        "slurm_max_concurrent": "auto",
        "slurm_cpus_per_task": "auto",
        "finalize_export_shards": 1,
        "stage": "all",
        "run_dir": None,
        "task_index": None,
        "task_mode": task_mode,
        "multiclass_attack_names": multiclass_attack_names,
        "cnn_hyperparams": cnn_hyperparams,
        "class_weight_loss": class_weight_loss,
        "rank_label_weight_loss": rank_label_weight_loss,
        "skip_feature_importance": skip_feature_importance,
        "selection_metric": selection_metric,
    }


def run_cnn_train(
    *,
    manifest_json: Path,
    feature_file: Path,
    dataset_root: Path = default_dataset_root(),
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    worker_cpus: str = "auto",
    max_concurrent: str = "auto",
    dry_run: bool = False,
    features: list[str] | None = None,
    spectral_sv_top_k: int = 8,
    spectral_moment_source: str = DEFAULT_SPECTRAL_MOMENT_SOURCE,
    spectral_qv_sum_mode: str = "append",
    spectral_entrywise_delta_mode: str = DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    stream_block_size: int = 131072,
    dtype_name: str = "float32",
    cv_folds: int = 5,
    random_state: int = 42,
    train_split: int = 100,
    calibration_split: int | None = None,
    accepted_fpr: list[float] | None = None,
    split_by_folder: bool = False,
    cv_seeds: list[int] | None = None,
    n_jobs: int = -1,
    score_percentiles: list[float] | None = None,
    cnn_hyperparams: Path | None = None,
    task_mode: str = "binary",
    multiclass_attack_names: list[str] | None = None,
    class_weight_loss: bool = False,
    rank_label_weight_loss: bool = False,
    selection_metric: str | None = "task_default",
    skip_feature_importance: bool = False,
    dependency: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    if backend == "slurm":
        ctx = _submission_context(output_root, "cnn_train", run_id)
        timing, started_perf = _timing_start("cnn_train_submit", backend="slurm")
        env = {
            "REPO_ROOT": _project_root(),
            "MANIFEST_JSON": resolve_manifest_path(manifest_json),
            "DATASET_ROOT": Path(dataset_root).expanduser().resolve(),
            "OUTPUT_ROOT": output_root,
            "RUN_ID": ctx.run_id,
            "FEATURE_FILE": Path(feature_file),
            "FEATURES": " ".join(_coerce_features(features)),
            "MODEL": DEFAULT_CNN_MODEL,
            "TASK_MODE": task_mode,
            "MULTICLASS_ATTACK_NAMES": " ".join(multiclass_attack_names or []),
            "CNN_HYPERPARAMS": cnn_hyperparams,
            "SV_TOP_K": int(spectral_sv_top_k),
            "SPECTRAL_MOMENT_SOURCE": spectral_moment_source,
            "SPECTRAL_QV_SUM_MODE": spectral_qv_sum_mode,
            "SPECTRAL_ENTRYWISE_DELTA_MODE": spectral_entrywise_delta_mode,
            "CV_FOLDS": int(cv_folds),
            "CV_SEEDS": " ".join(str(x) for x in (cv_seeds or [42, 43, 44])),
            "RANDOM_STATE": int(random_state),
            "TRAIN_SPLIT": int(train_split),
            "CALIBRATION_SPLIT": calibration_split,
            "ACCEPTED_FPR": " ".join(str(x) for x in accepted_fpr) if accepted_fpr else None,
            "SPLIT_BY_FOLDER": int(bool(split_by_folder)),
            "CLASS_WEIGHT_LOSS": int(bool(class_weight_loss)),
            "RANK_LABEL_WEIGHT_LOSS": int(bool(rank_label_weight_loss)),
            "SELECTION_METRIC": selection_metric or "task_default",
            "SCORE_PERCENTILES": " ".join(str(x) for x in score_percentiles) if score_percentiles else None,
            "SLURM_PARTITION": partition,
            "SLURM_MAX_CONCURRENT_REQUEST": max_concurrent,
            "SLURM_CPUS_PER_TASK_REQUEST": worker_cpus,
            "SKIP_FEATURE_IMPORTANCE": int(bool(skip_feature_importance)),
        }
        submission = _run_sbatch(
            "supervised_array.sh",
            env=env,
            partition=partition,
            dependency=dependency,
            dry_run=dry_run,
        )
        timing = _timing_finish(timing, started_perf)
        artifacts = {
            "supervised_run_dir": str(output_root / "supervised" / ctx.run_id),
        }
        _json_write(
            ctx.run_dir / "run_config.json",
            {
                "pipeline": "cnn_train",
                "backend": "slurm",
                "manifest_json": str(resolve_manifest_path(manifest_json)),
                "feature_file": str(feature_file),
                "features": _coerce_features(features),
                "slurm": submission,
            },
        )
        _json_write(ctx.run_dir / "artifact_index.json", artifacts)
        _json_write(ctx.run_dir / "timings.json", {"submit": timing})
        return {
            "run_dir": artifacts["supervised_run_dir"],
            "submission_run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "backend": "slurm",
            "slurm": submission,
            "timings": {"submit": timing},
        }

    timing, started_perf = _timing_start("cnn_train", backend="local")
    kwargs = _supervised_common_kwargs(
        manifest_json=resolve_manifest_path(manifest_json),
        dataset_root=Path(dataset_root).expanduser().resolve(),
        output_root=output_root,
        run_id=run_id,
        feature_file=Path(feature_file),
        features=features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        cv_folds=cv_folds,
        random_state=random_state,
        train_split=train_split,
        calibration_split=calibration_split,
        accepted_fpr=accepted_fpr,
        split_by_folder=split_by_folder,
        cv_seeds=cv_seeds,
        n_jobs=n_jobs,
        score_percentiles=score_percentiles,
        cnn_hyperparams=cnn_hyperparams,
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
        class_weight_loss=class_weight_loss,
        rank_label_weight_loss=rank_label_weight_loss,
        selection_metric=selection_metric,
        skip_feature_importance=skip_feature_importance,
    )
    result = run_supervised_pipeline(**kwargs)
    timing = _timing_finish(timing, started_perf)
    run_dir = Path(str(result["run_dir"]))
    timings_path = run_dir / "timings.json"
    existing_timings: dict[str, Any] = {}
    if timings_path.exists():
        with open(timings_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            existing_timings.update(existing)
    existing_timings["cnn_train"] = timing
    _json_write(timings_path, existing_timings)
    result.update({"backend": "local", "timings": existing_timings})
    return result


def run_cnn_infer(
    *,
    checkpoint: Path | None = None,
    run_dir: Path | None = None,
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    worker_cpus: str = "auto",
    max_concurrent: str = "auto",
    dry_run: bool = False,
    dependency: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    if backend == "slurm":
        ctx = _submission_context(output_root, "cnn_infer", run_id)
        timing, started_perf = _timing_start("cnn_infer_submit", backend="slurm")
        env = {
            "REPO_ROOT": _project_root(),
            "CHECKPOINT": checkpoint,
            "RUN_DIR": run_dir,
            "OUTPUT_ROOT": output_root,
            "RUN_ID": ctx.run_id,
            "SLURM_PARTITION": partition,
            "SLURM_CPUS_PER_TASK": worker_cpus,
            "SLURM_MAX_CONCURRENT": max_concurrent,
        }
        submission = _run_sbatch(
            "cnn_infer.sh",
            env=env,
            partition=partition,
            dependency=dependency,
            cpus_per_task=_resolve_partition_cpus_for_submit(partition, worker_cpus),
            dry_run=dry_run,
        )
        timing = _timing_finish(timing, started_perf)
        _json_write(
            ctx.run_dir / "run_config.json",
            {
                "pipeline": "cnn_infer",
                "backend": "slurm",
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "run_dir": str(run_dir) if run_dir is not None else None,
                "slurm": submission,
            },
        )
        _json_write(ctx.run_dir / "artifact_index.json", {})
        _json_write(ctx.run_dir / "timings.json", {"submit": timing})
        return {
            "run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "backend": "slurm",
            "slurm": submission,
            "timings": {"submit": timing},
        }

    from ..supervised.pipeline import run_supervised_checkpoint_inference

    timing, started_perf = _timing_start("cnn_infer", backend="local")
    result = run_supervised_checkpoint_inference(
        checkpoint=checkpoint,
        reference_run_dir=run_dir,
        output_root=output_root,
        run_id=run_id,
    )
    timing = _timing_finish(timing, started_perf)
    result_run_dir = Path(str(result["run_dir"]))
    timings = {"cnn_infer": timing}
    _json_write(result_run_dir / "timings.json", timings)
    result.update({"backend": "local", "timings": timings})
    return result


def run_cnn_full(
    *,
    manifest_json: Path,
    dataset_root: Path = default_dataset_root(),
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    worker_cpus: str = "auto",
    max_concurrent: str = "auto",
    dry_run: bool = False,
    features: list[str] | None = None,
    spectral_sv_top_k: int = 8,
    spectral_moment_source: str = DEFAULT_SPECTRAL_MOMENT_SOURCE,
    spectral_qv_sum_mode: str = DEFAULT_SPECTRAL_QV_SUM_MODE,
    spectral_entrywise_delta_mode: str = DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    stream_block_size: int = 131072,
    dtype_name: str = "float32",
    n_shards: int = 8,
    cv_folds: int = 5,
    random_state: int = 42,
    train_split: int = 100,
    calibration_split: int | None = None,
    accepted_fpr: list[float] | None = None,
    split_by_folder: bool = False,
    cv_seeds: list[int] | None = None,
    n_jobs: int = -1,
    score_percentiles: list[float] | None = None,
    cnn_hyperparams: Path | None = None,
    task_mode: str = "binary",
    multiclass_attack_names: list[str] | None = None,
    class_weight_loss: bool = False,
    rank_label_weight_loss: bool = False,
    selection_metric: str | None = "task_default",
    skip_feature_importance: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    ctx = _submission_context(output_root, "cnn_full", run_id)
    full_timing, full_started = _timing_start("cnn_full", backend=backend)
    resolved_features = _coerce_features(features)

    if backend == "slurm":
        extract = run_cnn_extract(
            manifest_json=manifest_json,
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=ctx.run_id,
            backend="slurm",
            partition=partition,
            worker_cpus=worker_cpus,
            max_concurrent=max_concurrent,
            dry_run=dry_run,
            features=resolved_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            n_shards=n_shards,
            wait_for_job_index=not dry_run,
        )
        if not dry_run:
            dependency_job_id = _read_slurm_dependency_job_id(Path(str(extract["slurm_job_index"])))
        else:
            dependency_job_id = "DRYRUN_FEATURE_FINALIZE"
        if dependency_job_id is None:
            dependency_job_id = extract.get("slurm", {}).get("job_id")
        aggregate = run_cnn_aggregate(
            feature_file=Path(extract["feature_path"]),
            output_filename=_aggregate_output_name(ctx.run_id),
            output_root=output_root,
            run_id=ctx.run_id,
            backend="slurm",
            partition=partition,
            worker_cpus=worker_cpus,
            max_concurrent=max_concurrent,
            dry_run=dry_run,
            feature_root=_aggregate_feature_root(output_root),
            features=resolved_features,
            spectral_qv_sum_mode="append",
            dependency=f"afterok:{dependency_job_id}" if dependency_job_id else None,
        )
        train_dependency = aggregate.get("slurm", {}).get("job_id")
        if dry_run and not train_dependency:
            train_dependency = "DRYRUN_AGGREGATE"
        train = run_cnn_train(
            manifest_json=manifest_json,
            feature_file=Path(aggregate["feature_path"]),
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=ctx.run_id,
            backend="slurm",
            partition=partition,
            worker_cpus=worker_cpus,
            max_concurrent=max_concurrent,
            dry_run=dry_run,
            features=resolved_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode="append",
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            cv_folds=cv_folds,
            random_state=random_state,
            train_split=train_split,
            calibration_split=calibration_split,
            accepted_fpr=accepted_fpr,
            split_by_folder=split_by_folder,
            cv_seeds=cv_seeds,
            n_jobs=n_jobs,
            score_percentiles=score_percentiles,
            cnn_hyperparams=cnn_hyperparams,
            task_mode=task_mode,
            multiclass_attack_names=multiclass_attack_names,
            class_weight_loss=class_weight_loss,
            rank_label_weight_loss=rank_label_weight_loss,
            selection_metric=selection_metric,
            skip_feature_importance=skip_feature_importance,
            dependency=f"afterok:{train_dependency}" if train_dependency else None,
        )
        full_timing = _timing_finish(full_timing, full_started)
        timings = {
            "cnn_full": full_timing,
            "extract": extract.get("timings"),
            "aggregate": aggregate.get("timings"),
            "train": train.get("timings"),
        }
        artifacts = {
            "feature_path": extract["feature_path"],
            "aggregated_feature_path": aggregate["feature_path"],
            "supervised_run_dir": train["run_dir"],
        }
        _json_write(
            ctx.run_dir / "run_config.json",
            {
                "pipeline": "cnn_full",
                "backend": "slurm",
                "manifest_json": str(resolve_manifest_path(manifest_json)),
                "dataset_root": str(Path(dataset_root).expanduser().resolve()),
                "features": resolved_features,
                "extract": extract,
                "aggregate": aggregate,
                "train": train,
            },
        )
        _json_write(ctx.run_dir / "artifact_index.json", artifacts)
        _json_write(ctx.run_dir / "timings.json", timings)
        return {
            "run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "backend": "slurm",
            "extract": extract,
            "aggregate": aggregate,
            "train": train,
            "artifacts": artifacts,
            "timings": timings,
        }

    extract = run_cnn_extract(
        manifest_json=manifest_json,
        dataset_root=dataset_root,
        output_root=output_root,
        run_id=ctx.run_id,
        backend="local",
        features=resolved_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        n_shards=n_shards,
    )
    aggregate = run_cnn_aggregate(
        feature_file=Path(extract["feature_path"]),
        output_filename=_aggregate_output_name(ctx.run_id),
        output_root=output_root,
        run_id=ctx.run_id,
        backend="local",
        feature_root=_aggregate_feature_root(output_root),
        features=resolved_features,
        spectral_qv_sum_mode="append",
    )
    train = run_cnn_train(
        manifest_json=manifest_json,
        feature_file=Path(aggregate["feature_path"]),
        dataset_root=dataset_root,
        output_root=output_root,
        run_id=ctx.run_id,
        backend="local",
        features=resolved_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode="append",
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        cv_folds=cv_folds,
        random_state=random_state,
        train_split=train_split,
        calibration_split=calibration_split,
        accepted_fpr=accepted_fpr,
        split_by_folder=split_by_folder,
        cv_seeds=cv_seeds,
        n_jobs=n_jobs,
        score_percentiles=score_percentiles,
        cnn_hyperparams=cnn_hyperparams,
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
        class_weight_loss=class_weight_loss,
        rank_label_weight_loss=rank_label_weight_loss,
        selection_metric=selection_metric,
        skip_feature_importance=skip_feature_importance,
    )
    full_timing = _timing_finish(full_timing, full_started)
    timings = {
        "cnn_full": full_timing,
        "extract": extract.get("timings"),
        "aggregate": aggregate.get("timings"),
        "train": train.get("timings"),
    }
    artifacts = {
        "feature_path": extract["feature_path"],
        "aggregated_feature_path": aggregate["feature_path"],
        "supervised_run_dir": train["run_dir"],
        "results_summary_md": train.get("results_summary_md"),
    }
    _json_write(
        ctx.run_dir / "run_config.json",
        {
            "pipeline": "cnn_full",
            "backend": "local",
            "manifest_json": str(resolve_manifest_path(manifest_json)),
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "features": resolved_features,
            "extract": extract,
            "aggregate": aggregate,
            "train": {k: v for k, v in train.items() if k not in {"train_features", "train_labels"}},
        },
    )
    _json_write(ctx.run_dir / "artifact_index.json", artifacts)
    _json_write(ctx.run_dir / "timings.json", timings)
    return {
        "run_dir": str(ctx.run_dir),
        "run_id": ctx.run_id,
        "backend": "local",
        "extract": extract,
        "aggregate": aggregate,
        "train": train,
        "artifacts": artifacts,
        "timings": timings,
    }
