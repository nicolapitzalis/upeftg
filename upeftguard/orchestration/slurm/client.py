"""Slurm command submission and job-index mechanics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def stringify_env(env: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in env.items() if value is not None}


def run_sbatch_command(
    wrapped_command: Sequence[str],
    *,
    partition: str,
    env: dict[str, Any] | None = None,
    dependency: str | None = None,
    wait: bool = False,
    cpus_per_task: int | str | None = None,
    memory: str | None = None,
    nodes: int | str | None = None,
    ntasks: int | str | None = None,
    ntasks_per_node: int | str | None = None,
    job_name: str | None = None,
    output: Path | str | None = None,
    error: Path | str | None = None,
    chdir: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit an argv-style command without requiring a workflow shell script."""

    if not wrapped_command:
        raise ValueError("wrapped_command must not be empty")

    command = ["sbatch", "--parsable", "--partition", str(partition)]
    if dependency:
        command.extend(["--dependency", str(dependency)])
    if wait:
        command.append("--wait")
    if cpus_per_task is not None:
        command.extend(["--cpus-per-task", str(cpus_per_task)])
    if memory:
        command.extend(["--mem", str(memory)])
    if nodes is not None:
        command.extend(["--nodes", str(nodes)])
    if ntasks is not None:
        command.extend(["--ntasks", str(ntasks)])
    if ntasks_per_node is not None:
        command.extend(["--ntasks-per-node", str(ntasks_per_node)])
    if job_name:
        command.extend(["--job-name", str(job_name)])
    if output is not None:
        command.extend(["--output", str(output)])
    if error is not None:
        command.extend(["--error", str(error)])
    if chdir is not None:
        command.extend(["--chdir", str(chdir)])
    command.extend(["--wrap", shlex.join(str(token) for token in wrapped_command)])

    payload: dict[str, Any] = {
        "command": command,
        "wrapped_command": [str(token) for token in wrapped_command],
        "env": stringify_env(env or {}),
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
        cwd=Path(chdir) if chdir is not None else project_root(),
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


def read_dependency_job_id(index_path: Path) -> str | None:
    if not index_path.exists():
        return None
    with open(index_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    value = payload.get("final_dependency_job_id")
    return str(value) if value else None
