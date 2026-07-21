"""Slurm resource discovery and worker-index resolution."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionCapacity:
    nodes: int
    cpus_per_node: int


def discover_partition_capacity(partition: str) -> PartitionCapacity:
    """Return homogeneous non-DOWN node capacity for a Slurm partition."""

    try:
        completed = subprocess.run(
            ["sinfo", "-N", "-h", "-p", str(partition), "-o", "%N|%c|%T"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not inspect partition {partition!r}; specify resources only after sinfo is available"
        ) from exc

    available: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 3 or not fields[1].isdigit():
            continue
        node_name, cpu_text, state = fields
        normalized_state = state.lower().rstrip("*~#+")
        if normalized_state.startswith("down"):
            continue
        available[node_name] = int(cpu_text)
    if not available:
        raise ValueError(f"Partition {partition!r} has no non-DOWN nodes")
    cpu_counts = set(available.values())
    if len(cpu_counts) != 1:
        raise ValueError(f"Partition {partition!r} is not CPU-homogeneous across non-DOWN nodes: {sorted(cpu_counts)}")
    return PartitionCapacity(nodes=len(available), cpus_per_node=next(iter(cpu_counts)))


def resolve_partition_cpus(partition: str, requested: str) -> str:
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
    values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip().isdigit()]
    return str(max(values)) if values else "32"


def resolve_positive_slurm_value(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be 'auto' or a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def resolve_slurm_task_index(task_index: int | None) -> int:
    if task_index is not None:
        return int(task_index)
    value = os.getenv("SLURM_PROCID")
    if value is None:
        raise ValueError("--task-index is required outside a packed srun worker")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"SLURM_PROCID must be an integer, got {value!r}") from exc
