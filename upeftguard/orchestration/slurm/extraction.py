"""Python orchestration for schema-aware feature extraction on Slurm."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from .client import project_root, run_sbatch_command
from .resources import discover_partition_capacity, resolve_positive_slurm_value
from .extraction_parallelization import (
    RankParallelizationSetting,
    resolve_rank_parallelization_settings,
)
from ...utilities.core.serialization import json_ready
from ...utilities.core.experiment import experiment_context_from_stage_dir
from ..sharding.planning import prepare_schema_sharded_manifests


def write_extraction_slurm_config(
    path: Path,
    *,
    extraction_kwargs: dict[str, Any],
    partition: str,
    nodes: str,
    cpus_per_worker: str,
    workers_per_node: str,
    log_dir: Path,
    job_index_path: Path,
) -> Path:
    payload = {
        "extraction_kwargs": extraction_kwargs,
        "slurm": {
            "partition": str(partition),
            "nodes": str(nodes),
            "cpus_per_worker": str(cpus_per_worker),
            "workers_per_node": str(workers_per_node),
            "log_dir": str(log_dir),
            "job_index_path": str(job_index_path),
            "working_directory": str(project_root()),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2)
    return path


def extraction_controller_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "upeftguard.cli",
        "run",
        "slurm-extraction-controller",
        str(config_path),
    ]


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    kwargs = dict(payload["extraction_kwargs"])
    for name in ("manifest_json", "dataset_root", "output_root", "output_run_dir"):
        kwargs[name] = Path(kwargs[name])
    return kwargs, dict(payload["slurm"])


def _safe_job_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_job_index(path: Path, payload: dict[str, Any]) -> None:
    persisted = {
        key: payload.get(key)
        for key in (
            "controller_job_id",
            "worker_job_id",
            "merge_job_id",
            "final_dependency_job_id",
            "partition",
            "worker_submission_mode",
            "cpus_per_worker",
            "nodes",
            "workers_per_node",
            "schema_group_count",
            "n_shards",
            "workers",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(json_ready(persisted), file, indent=2)
    temporary.replace(path)


def submit_extraction_worker_graph(
    *,
    run_root: Path,
    run_id: str,
    schema_report_path: Path,
    dataset_root: Path,
    merged_dir: Path,
    features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str,
    stream_block_size: int,
    dtype_name: str,
    n_shards: int,
    group_count: int,
    pipeline_start_epoch_seconds: float,
    partition: str,
    nodes: int,
    cpus_per_worker: int,
    workers_per_node: int,
    log_dir: Path,
    job_index_path: Path,
    working_directory: Path | None = None,
    dry_run: bool = False,
    worker_plans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if n_shards <= 0:
        raise ValueError(f"Feature extraction produced no shards: {n_shards}")

    workdir = Path(working_directory or run_root).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = _safe_job_name(run_id)
    python = sys.executable
    base_worker_command = [
        python,
        "-m",
        "upeftguard.cli",
        "run",
        "schema-shard-worker",
        "--schema-report-path",
        str(schema_report_path),
        "--dataset-root",
        str(dataset_root),
        "--features",
        *features,
        "--spectral-sv-top-k",
        str(spectral_sv_top_k),
        "--spectral-moment-source",
        spectral_moment_source,
        "--spectral-entrywise-delta-mode",
        spectral_entrywise_delta_mode,
        "--spectral-attention-granularity",
        spectral_attention_granularity,
        "--stream-block-size",
        str(stream_block_size),
        "--dtype",
        dtype_name,
    ]
    plans = worker_plans or [
        {
            "label": "all_schema_groups",
            "group_ids": [],
            "n_shards": n_shards,
            "nodes": nodes,
            "cpus_per_worker": cpus_per_worker,
            "workers_per_node": workers_per_node,
        }
    ]
    worker_submissions: list[dict[str, Any]] = []
    for plan in plans:
        plan_label = _safe_job_name(str(plan.get("label") or "schema_groups"))
        plan_shards = int(plan["n_shards"])
        cpus_per_worker = int(plan.get("cpus_per_worker", cpus_per_worker))
        nodes = int(plan.get("nodes", nodes))
        workers_per_node = int(plan.get("workers_per_node", workers_per_node))
        if plan_shards <= 0:
            raise ValueError(f"Worker plan {plan_label} has no shards")
        group_ids = [str(value) for value in plan.get("group_ids", [])]
        worker_command = list(base_worker_command)
        if group_ids:
            worker_command.extend(["--group-ids", *group_ids])
        thread_env = {
            name: str(cpus_per_worker)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
        worker_options: dict[str, Any] = {
            "partition": partition,
            "cpus_per_task": cpus_per_worker,
            "env": thread_env,
            "job_name": f"upeftguard_feature_extract_{plan_label}_{safe_run_id}",
            "chdir": workdir,
            "dry_run": dry_run,
        }
        capacity = nodes * workers_per_node
        if capacity < plan_shards:
            raise ValueError(f"Worker capacity for {plan_label} must cover its shard count: {capacity} < {plan_shards}")
        worker_command = [
            "srun",
            "--kill-on-bad-exit=1",
            "--distribution",
            "cyclic",
            "--nodes",
            str(nodes),
            "--ntasks",
            str(plan_shards),
            "--ntasks-per-node",
            str(workers_per_node),
            "--cpus-per-task",
            str(cpus_per_worker),
            "--output",
            str(log_dir / f"feature_extract_{plan_label}_{safe_run_id}_%j_%t.out"),
            "--error",
            str(log_dir / f"feature_extract_{plan_label}_{safe_run_id}_%j_%t.err"),
            *worker_command,
        ]
        worker_options.update(
            {
                "nodes": nodes,
                "ntasks": plan_shards,
                "ntasks_per_node": workers_per_node,
                "output": log_dir / f"feature_extract_{plan_label}_{safe_run_id}_packed_%j.out",
                "error": log_dir / f"feature_extract_{plan_label}_{safe_run_id}_packed_%j.err",
            }
        )
        submission = run_sbatch_command(worker_command, **worker_options)
        job_id = submission.get("job_id")
        if not dry_run and not job_id:
            raise RuntimeError(f"Slurm did not return a feature worker job id for {plan_label}")
        worker_submissions.append(
            {
                **plan,
                "label": plan_label,
                "submission_mode": "packed",
                "job_id": job_id,
                "submission": submission,
            }
        )

    worker = worker_submissions[0]["submission"]
    worker_job_id = worker_submissions[0].get("job_id")
    submission_modes = {str(item["submission_mode"]) for item in worker_submissions}
    submission_mode = next(iter(submission_modes)) if len(submission_modes) == 1 else "mixed"

    index: dict[str, Any] = {
        "controller_job_id": os.getenv("SLURM_JOB_ID"),
        "run_root": str(run_root),
        "schema_partition_report": str(schema_report_path),
        "final_feature_output_dir": str(merged_dir),
        "feature_path": str(merged_dir / "spectral_features.npy"),
        "worker_job_id": worker_job_id,
        "worker_job_ids": [f"{item['label']}:{item.get('job_id')}" for item in worker_submissions],
        "merge_job_id": None,
        "merge_job_ids": [],
        "finalize_job_id": None,
        "final_dependency_job_id": None,
        "partition": partition,
        "worker_submission_mode": submission_mode,
        "cpus_per_worker": (int(plans[0]["cpus_per_worker"]) if len(plans) == 1 else None),
        "nodes": int(plans[0]["nodes"]) if len(plans) == 1 else None,
        "workers_per_node": (int(plans[0]["workers_per_node"]) if len(plans) == 1 else None),
        "packed_task_distribution": "cyclic",
        "schema_group_count": group_count,
        "n_shards": n_shards,
        "worker": worker,
        "workers": worker_submissions,
    }
    if not dry_run:
        _write_job_index(job_index_path, index)

    resolved_worker_ids = [str(item["job_id"]) for item in worker_submissions if item.get("job_id")]
    dependency = "afterok:" + ":".join(resolved_worker_ids) if resolved_worker_ids else "afterok:DRYRUN_WORKERS"
    merge = run_sbatch_command(
        [
            python,
            "-m",
            "upeftguard.cli",
            "run",
            "merge-schema-group-shards",
            "--schema-report-path",
            str(schema_report_path),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(merged_dir),
            "--pipeline-start-epoch-seconds",
            str(pipeline_start_epoch_seconds),
        ],
        partition=partition,
        dependency=dependency,
        cpus_per_task=4,
        job_name=f"upeftguard_feature_extract_merge_{safe_run_id}",
        output=log_dir / f"feature_extract_merge_{safe_run_id}_%j.out",
        error=log_dir / f"feature_extract_merge_{safe_run_id}_%j.err",
        chdir=workdir,
        dry_run=dry_run,
    )
    merge_job_id = merge.get("job_id")
    if not dry_run and not merge_job_id:
        raise RuntimeError("Slurm did not return a feature merge job id")
    index.update(
        {
            "merge_job_id": merge_job_id,
            "merge_job_ids": [f"all_schema_groups:{merge_job_id}"],
            "final_dependency_job_id": merge_job_id,
            "merge": merge,
        }
    )
    if not dry_run:
        _write_job_index(job_index_path, index)
    return {**index, "job_index": str(job_index_path)}


def run_extraction_slurm_controller(config_path: Path) -> dict[str, Any]:
    kwargs, slurm = _load_config(config_path)
    partition = str(slurm["partition"])
    partition_capacity = discover_partition_capacity(partition)
    nodes = (
        partition_capacity.nodes
        if str(slurm["nodes"]).strip() == "auto"
        else resolve_positive_slurm_value(str(slurm["nodes"]), name="nodes")
    )
    cpus_per_worker = resolve_positive_slurm_value(str(slurm["cpus_per_worker"]), name="cpus_per_worker")
    workers_per_node = (
        partition_capacity.cpus_per_node // cpus_per_worker
        if str(slurm["workers_per_node"]).strip() == "auto"
        else resolve_positive_slurm_value(str(slurm["workers_per_node"]), name="workers_per_node")
    )
    default_setting = RankParallelizationSetting(
        rank=-1,
        nodes=nodes,
        cpus_per_worker=cpus_per_worker,
        workers_per_node=workers_per_node,
    )
    raw_parallelization_settings = list(kwargs.get("parallelization_settings") or [])
    rank_settings = resolve_rank_parallelization_settings(raw_parallelization_settings)

    def validate_setting(setting: RankParallelizationSetting, *, label: str) -> None:
        if setting.nodes > partition_capacity.nodes:
            raise ValueError(
                f"{label} requests {setting.nodes} nodes, but partition {partition!r} has "
                f"{partition_capacity.nodes} non-DOWN nodes"
            )
        requested_cpus = setting.cpus_per_worker * setting.workers_per_node
        if requested_cpus > partition_capacity.cpus_per_node:
            raise ValueError(
                f"{label} requests {requested_cpus} CPUs per node "
                f"({setting.workers_per_node} workers * {setting.cpus_per_worker} CPUs), but "
                f"partition nodes have {partition_capacity.cpus_per_node} CPUs"
            )

    validate_setting(default_setting, label="Default extraction setting")
    for rank, setting in rank_settings.items():
        validate_setting(setting, label=f"Rank {rank} extraction setting")

    run_root = Path(kwargs["output_run_dir"]).expanduser().resolve()
    work_root = run_root / ".work"
    schema_group_root = work_root / "schema_groups"
    merged_dir = run_root / "features"
    schema_report_path = work_root / "schema_partition_report.json"
    pipeline_start = float(kwargs.get("pipeline_start_epoch_seconds") or time.time())
    report = prepare_schema_sharded_manifests(
        manifest_path=Path(kwargs["manifest_json"]),
        dataset_root=Path(kwargs["dataset_root"]),
        output_dir=schema_group_root,
        worker_capacity=nodes * workers_per_node,
        spectral_qv_sum_mode=str(kwargs["spectral_qv_sum_mode"]),
        report_path=schema_report_path,
        worker_capacity_by_rank={
            rank: setting.nodes * setting.workers_per_node for rank, setting in rank_settings.items()
        },
    )
    groups = list(report.get("groups", []))
    if not groups:
        raise ValueError("No schema groups were prepared")
    n_shards = max(int(group["n_shards"]) for group in groups)
    observed_ranks = {int(group["adapter_rank"]) for group in groups if group.get("adapter_rank") is not None}
    missing_ranks = sorted(set(rank_settings) - observed_ranks)
    if missing_ranks:
        raise ValueError(
            "Parallelization settings reference ranks that are absent from the manifest: "
            + ", ".join(str(rank) for rank in missing_ranks)
        )
    warnings = list(report.get("warnings", []))
    for rank in sorted(observed_ranks - set(rank_settings)):
        warning = (
            f"Rank {rank} is using the default Slurm allocation. Different ranks may have "
            "different optimal CPU and memory requirements."
        )
        print(f"WARNING: {warning}", flush=True)
        warnings.append(warning)
    if any(group.get("adapter_rank") is None for group in groups):
        warning = (
            "At least one schema group has no parseable adapter rank and is using the default "
            "Slurm allocation. Different ranks may have different optimal CPU and memory requirements."
        )
        print(f"WARNING: {warning}", flush=True)
        warnings.append(warning)
    report["warnings"] = warnings
    schema_report_path.parent.mkdir(parents=True, exist_ok=True)
    schema_report_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")

    worker_plans: list[dict[str, Any]] = []
    for group in groups:
        raw_rank = group.get("adapter_rank")
        adapter_rank = int(raw_rank) if raw_rank is not None else None
        setting = rank_settings.get(adapter_rank, default_setting)
        worker_plans.append(
            {
                "label": (
                    f"rank_{adapter_rank}_{group['group_id']}"
                    if adapter_rank is not None
                    else f"unknown_rank_{group['group_id']}"
                ),
                "rank": adapter_rank,
                "group_ids": [str(group["group_id"])],
                "n_shards": int(group["n_shards"]),
                "nodes": setting.nodes,
                "cpus_per_worker": setting.cpus_per_worker,
                "workers_per_node": setting.workers_per_node,
            }
        )
    merged_dir.mkdir(parents=True, exist_ok=True)
    result = submit_extraction_worker_graph(
        run_root=run_root,
        run_id=str(kwargs["run_id"]),
        schema_report_path=schema_report_path,
        dataset_root=Path(kwargs["dataset_root"]),
        merged_dir=merged_dir,
        features=list(kwargs["features"]),
        spectral_sv_top_k=int(kwargs["spectral_sv_top_k"]),
        spectral_moment_source=str(kwargs["spectral_moment_source"]),
        spectral_entrywise_delta_mode=str(kwargs["spectral_entrywise_delta_mode"]),
        spectral_attention_granularity=str(kwargs["spectral_attention_granularity"]),
        stream_block_size=int(kwargs["stream_block_size"]),
        dtype_name=str(kwargs["dtype_name"]),
        n_shards=n_shards,
        group_count=len(groups),
        pipeline_start_epoch_seconds=pipeline_start,
        partition=partition,
        nodes=nodes,
        cpus_per_worker=cpus_per_worker,
        workers_per_node=workers_per_node,
        log_dir=Path(slurm["log_dir"]),
        job_index_path=Path(slurm["job_index_path"]),
        working_directory=Path(slurm["working_directory"]),
        worker_plans=worker_plans,
    )
    experiment = experiment_context_from_stage_dir(run_root)
    if experiment is not None:
        job_ids = [result.get("controller_job_id")]
        job_ids.extend(worker.get("job_id") for worker in result.get("workers", []))
        job_ids.append(result.get("merge_job_id"))
        job_ids = list(dict.fromkeys(value for value in job_ids if value))
        experiment.update(
            stage="extraction",
            stage_status="submitted",
            stage_values={"job_ids": job_ids},
        )
    return result
