"""Python orchestration for supervised Slurm jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from .client import project_root, run_sbatch_command
from .resources import discover_partition_capacity, resolve_positive_slurm_value
from ...utilities.core.serialization import json_ready
from ...utilities.core.experiment import experiment_context_from_stage_dir


_PATH_ARGUMENTS = {
    "manifest_json",
    "dataset_root",
    "output_root",
    "feature_file",
    "run_dir",
    "prepared_run_dir",
    "hyperparams",
    "transformer_checkpoint_dir",
    "transformer_resume_checkpoint",
}


def write_supervised_slurm_config(
    path: Path,
    *,
    pipeline_kwargs: dict[str, Any],
    partition: str,
    nodes: str,
    cpus_per_worker: str,
    workers_per_node: str,
    log_dir: Path,
) -> Path:
    payload = {
        "pipeline_kwargs": pipeline_kwargs,
        "slurm": {
            "partition": str(partition),
            "nodes": str(nodes),
            "cpus_per_worker": str(cpus_per_worker),
            "workers_per_node": str(workers_per_node),
            "log_dir": str(log_dir),
            "working_directory": str(project_root()),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2)
    return path


def controller_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "upeftguard.cli",
        "run",
        "slurm-supervised-controller",
        str(config_path),
    ]


def _load_controller_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    pipeline_kwargs = dict(payload["pipeline_kwargs"])
    for name in _PATH_ARGUMENTS:
        value = pipeline_kwargs.get(name)
        if value is not None:
            pipeline_kwargs[name] = Path(value)
    return pipeline_kwargs, dict(payload["slurm"])


def _safe_job_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_job_index(path: Path, payload: dict[str, Any]) -> None:
    persisted = {
        key: payload.get(key)
        for key in (
            "controller_job_id",
            "worker_job_id",
            "finalize_job_id",
            "final_dependency_job_id",
            "partition",
            "worker_submission_mode",
            "nodes",
            "cpus_per_worker",
            "workers_per_node",
            "n_tasks",
            "worker_batches",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(json_ready(persisted), file, indent=2)
    temporary.replace(path)


def submit_supervised_worker_graph(
    *,
    run_dir: Path,
    run_id: str,
    n_tasks: int,
    partition: str,
    nodes: int,
    cpus_per_worker: int,
    workers_per_node: int,
    log_dir: Path,
    dry_run: bool = False,
    working_directory: Path | None = None,
) -> dict[str, Any]:
    if n_tasks <= 0:
        raise ValueError(f"Supervised tuning produced no tasks: {n_tasks}")
    capacity = nodes * workers_per_node
    if capacity <= 0:
        raise ValueError("Supervised worker capacity must be positive")

    workdir = Path(working_directory or run_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = _safe_job_name(run_id)
    python = sys.executable
    job_index_path = run_dir / ".work" / "job_index.json"

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
    worker_batches: list[dict[str, Any]] = []
    for batch_index, task_offset in enumerate(range(0, n_tasks, capacity)):
        batch_tasks = min(capacity, n_tasks - task_offset)
        worker_command = [
            python,
            "-m",
            "upeftguard.cli",
            "run",
            "supervised",
            "--stage",
            "worker",
            "--run-dir",
            str(run_dir),
            "--n-jobs",
            str(cpus_per_worker),
            "--task-index-offset",
            str(task_offset),
        ]
        worker_command = [
            "srun",
            "--kill-on-bad-exit=1",
            "--distribution",
            "block",
            "--nodes",
            str(nodes),
            "--ntasks",
            str(batch_tasks),
            "--ntasks-per-node",
            str(workers_per_node),
            "--cpus-per-task",
            str(cpus_per_worker),
            "--output",
            str(log_dir / f"supervised_worker_{safe_run_id}_b{batch_index}_%j_%t.out"),
            "--error",
            str(log_dir / f"supervised_worker_{safe_run_id}_b{batch_index}_%j_%t.err"),
            *worker_command,
        ]
        worker = run_sbatch_command(
            worker_command,
            partition=partition,
            nodes=nodes,
            ntasks=batch_tasks,
            ntasks_per_node=workers_per_node,
            cpus_per_task=cpus_per_worker,
            env=thread_env,
            job_name=f"upeftguard_supervised_worker_b{batch_index}_{safe_run_id}",
            output=log_dir / f"supervised_worker_{safe_run_id}_b{batch_index}_packed_%j.out",
            error=log_dir / f"supervised_worker_{safe_run_id}_b{batch_index}_packed_%j.err",
            chdir=workdir,
            dry_run=dry_run,
        )
        job_id = worker.get("job_id")
        if not dry_run and not job_id:
            raise RuntimeError(f"Slurm did not return a worker job id for batch {batch_index}")
        worker_batches.append(
            {
                "batch_index": batch_index,
                "task_offset": task_offset,
                "n_tasks": batch_tasks,
                "job_id": job_id,
                "submission": worker,
            }
        )
    worker = worker_batches[0]["submission"]
    worker_job_id = worker_batches[0]["job_id"]

    index: dict[str, Any] = {
        "controller_job_id": os.getenv("SLURM_JOB_ID"),
        "worker_job_id": worker_job_id,
        "finalize_job_id": None,
        "final_dependency_job_id": None,
        "partition": partition,
        "nodes": int(nodes),
        "cpus_per_worker": int(cpus_per_worker),
        "workers_per_node": int(workers_per_node),
        "worker_submission_mode": "packed_batches",
        "n_tasks": int(n_tasks),
        "worker": worker,
        "worker_batches": worker_batches,
    }
    if not dry_run:
        _write_job_index(job_index_path, index)

    worker_job_ids = [str(batch["job_id"]) for batch in worker_batches if batch.get("job_id")]
    finalize_dependency = "afterok:" + ":".join(worker_job_ids) if worker_job_ids else "afterok:DRYRUN_WORKERS"
    finalize = run_sbatch_command(
        [
            python,
            "-m",
            "upeftguard.cli",
            "run",
            "supervised",
            "--stage",
            "finalize",
            "--run-dir",
            str(run_dir),
        ],
        partition=partition,
        dependency=finalize_dependency,
        cpus_per_task=4,
        job_name=f"upeftguard_supervised_finalize_{safe_run_id}",
        output=log_dir / f"supervised_finalize_{safe_run_id}_%j.out",
        error=log_dir / f"supervised_finalize_{safe_run_id}_%j.err",
        chdir=workdir,
        dry_run=dry_run,
    )
    finalize_job_id = finalize.get("job_id")
    if not dry_run and not finalize_job_id:
        raise RuntimeError("Slurm did not return a finalize job id")

    index.update(
        {
            "finalize_job_id": finalize_job_id,
            "final_dependency_job_id": finalize_job_id,
            "finalize": finalize,
        }
    )
    if not dry_run:
        _write_job_index(job_index_path, index)
    return {**index, "job_index": str(job_index_path)}


def run_supervised_slurm_controller(config_path: Path) -> dict[str, Any]:
    pipeline_kwargs, slurm = _load_controller_config(config_path)
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
    if nodes > partition_capacity.nodes:
        raise ValueError(
            f"Training requests {nodes} nodes, but partition {partition!r} has "
            f"{partition_capacity.nodes} non-DOWN nodes"
        )
    if cpus_per_worker * workers_per_node > partition_capacity.cpus_per_node:
        raise ValueError("Training workers request more CPUs per node than the partition provides")
    pipeline_kwargs.update(
        {
            "stage": "prepare",
            "tuning_executor": "slurm_packed",
            "slurm_partition": partition,
            "cpus_per_worker": str(cpus_per_worker),
            "run_dir": None,
            "task_index": None,
        }
    )

    from ...supervised.lifecycle.pipeline import run_supervised_pipeline

    prepared = run_supervised_pipeline(**pipeline_kwargs)
    result = submit_supervised_worker_graph(
        run_dir=Path(prepared["run_dir"]),
        run_id=str(pipeline_kwargs.get("run_id") or Path(prepared["run_dir"]).name),
        n_tasks=int(prepared["n_tasks"]),
        partition=partition,
        nodes=nodes,
        cpus_per_worker=cpus_per_worker,
        workers_per_node=workers_per_node,
        log_dir=Path(slurm["log_dir"]),
        working_directory=Path(slurm["working_directory"]),
    )
    training_run_dir = Path(prepared["run_dir"])
    experiment = experiment_context_from_stage_dir(training_run_dir)
    if experiment is not None:
        job_ids = [
            value
            for value in (
                result.get("controller_job_id"),
                result.get("finalize_job_id"),
            )
            if value
        ]
        job_ids.extend(batch.get("job_id") for batch in result.get("worker_batches", []))
        experiment.update(
            stage="training",
            stage_status="submitted",
            stage_values={"job_ids": job_ids},
        )
    return result
