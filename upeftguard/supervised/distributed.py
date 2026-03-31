from __future__ import annotations

import os
from pathlib import Path


def resolve_slurm_cpus_per_task(value: str) -> int:
    if value != "auto":
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"slurm_cpus_per_task must be positive, got {parsed}")
        return parsed

    env_candidates = [
        os.getenv("SLURM_CPUS_PER_TASK"),
        os.getenv("SLURM_CPUS_ON_NODE"),
    ]
    for item in env_candidates:
        if item and item.isdigit() and int(item) > 0:
            return int(item)
    return 32


def resolve_slurm_max_concurrent(value: str, cpus_per_task: int) -> int:
    if value != "auto":
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"slurm_max_concurrent must be positive, got {parsed}")
        return parsed

    nnodes_env = os.getenv("SLURM_NNODES")
    cpus_node_env = os.getenv("SLURM_CPUS_ON_NODE")
    if nnodes_env and cpus_node_env and nnodes_env.isdigit() and cpus_node_env.isdigit():
        nnodes = int(nnodes_env)
        cpus_node = int(cpus_node_env)
        total_cpus = max(1, nnodes * cpus_node)
        return max(1, total_cpus // max(1, cpus_per_task))
    return 8


def build_slurm_array_next_steps(
    *,
    run_dir: Path,
    n_tasks: int,
    max_concurrent: int,
    skip_feature_importance: bool = False,
) -> list[str]:
    finalize_cmd = f"python -m upeftguard.cli run supervised --stage finalize --run-dir {run_dir}"
    if skip_feature_importance:
        finalize_cmd += " --skip-feature-importance"
    return [
        f"Submit array workers for task indices 0..{max(0, n_tasks - 1)} with max concurrency {max_concurrent}",
        f"After workers complete, run: {finalize_cmd}",
    ]
