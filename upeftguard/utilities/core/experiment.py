"""Authoritative filesystem record for one logical experiment."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .serialization import json_ready


EXPERIMENT_SCHEMA_VERSION = 1


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(json_ready(payload), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ExperimentContext:
    output_root: Path
    run_id: str
    run_dir: Path

    @property
    def record_path(self) -> Path:
        return self.run_dir / "experiment.json"

    def stage_dir(self, stage: str) -> Path:
        path = self.run_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def display_path(self, path: Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return str(resolved.relative_to(self.run_dir))
        except ValueError:
            return str(resolved)

    def _load(self) -> dict[str, Any]:
        if not self.record_path.exists():
            return {
                "schema_version": EXPERIMENT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "stages": {},
                "artifacts": {},
            }
        with self.record_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid experiment record: {self.record_path}")
        return payload

    def update(
        self,
        *,
        workflow: str | None = None,
        backend: str | None = None,
        stage: str | None = None,
        stage_status: str | None = None,
        stage_values: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        values: dict[str, Any] | None = None,
    ) -> None:
        lock_path = self.record_path.with_name(f".{self.record_path.name}.lock")
        with _exclusive_file_lock(lock_path):
            self._update_unlocked(
                workflow=workflow,
                backend=backend,
                stage=stage,
                stage_status=stage_status,
                stage_values=stage_values,
                artifacts=artifacts,
                values=values,
            )

    def _update_unlocked(
        self,
        *,
        workflow: str | None = None,
        backend: str | None = None,
        stage: str | None = None,
        stage_status: str | None = None,
        stage_values: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        values: dict[str, Any] | None = None,
    ) -> None:
        payload = self._load()
        if workflow is not None:
            current_workflow = payload.get("workflow")
            if current_workflow is None or current_workflow == workflow or workflow == "full":
                payload["workflow"] = str(workflow)
        if backend is not None:
            current_backend = payload.get("backend")
            if current_backend is None or current_backend == backend or backend == "slurm":
                payload["backend"] = str(backend)
        if stage is not None:
            stages = payload.setdefault("stages", {})
            entry = dict(stages.get(stage, {}))
            entry.setdefault("directory", stage)
            if stage_status is not None:
                entry["status"] = stage_status
            if stage_values:
                entry.update(stage_values)
            if isinstance(entry.get("timing"), dict):
                timing = entry["timing"]
                entry["timing"] = {
                    "started_at_utc": timing.get("start_timestamp_utc"),
                    "ended_at_utc": timing.get("end_timestamp_utc"),
                    "elapsed_seconds": timing.get("elapsed_seconds"),
                }
            stages[stage] = entry
        if artifacts:
            registered = payload.setdefault("artifacts", {})
            for name, value in artifacts.items():
                if isinstance(registered.get(name), dict) and isinstance(value, dict):
                    merged = dict(registered[name])
                    merged.update(value)
                    registered[name] = merged
                else:
                    registered[name] = value
        if values:
            for name, value in values.items():
                if name == "configuration" and name in payload:
                    continue
                payload[name] = value
        if isinstance(payload.get("timing"), dict) and "start_timestamp_utc" in payload["timing"]:
            timing = payload["timing"]
            payload["timing"] = {
                "started_at_utc": timing.get("start_timestamp_utc"),
                "ended_at_utc": timing.get("end_timestamp_utc"),
                "elapsed_seconds": timing.get("elapsed_seconds"),
            }
        stages = payload.get("stages", {})
        if stages and all(isinstance(value, dict) and value.get("status") == "completed" for value in stages.values()):
            payload["status"] = "completed"
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json_write(self.record_path, payload)


def create_experiment_context(
    *,
    output_root: Path,
    run_id: str | None,
    workflow: str,
    backend: str,
) -> ExperimentContext:
    resolved_root = Path(output_root).expanduser().resolve()
    resolved_id = str(run_id or _default_run_id())
    run_dir = resolved_root / resolved_id
    run_dir.mkdir(parents=True, exist_ok=True)
    context = ExperimentContext(
        output_root=resolved_root,
        run_id=resolved_id,
        run_dir=run_dir,
    )
    context.update(workflow=workflow, backend=backend)
    return context


def experiment_context_from_stage_dir(stage_dir: Path) -> ExperimentContext | None:
    resolved_stage = Path(stage_dir).expanduser().resolve()
    run_dir = resolved_stage.parent
    record_path = run_dir / "experiment.json"
    if not record_path.exists():
        return None
    return ExperimentContext(
        output_root=run_dir.parent,
        run_id=run_dir.name,
        run_dir=run_dir,
    )
