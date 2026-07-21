from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ...utilities.core.serialization import json_ready
from ...utilities.core.experiment import experiment_context_from_stage_dir


def append_stage_timing(
    *,
    run_dir: Path,
    stage: str,
    started_at: datetime,
    started_perf: float,
) -> None:
    ended_at = datetime.now(timezone.utc)
    timing = {
        "stage": str(stage),
        "backend": "local_or_slurm_worker",
        "start_timestamp_utc": started_at.isoformat(),
        "end_timestamp_utc": ended_at.isoformat(),
        "elapsed_seconds": float(perf_counter() - started_perf),
    }
    experiment = experiment_context_from_stage_dir(run_dir)
    if experiment is not None:
        stage_name = run_dir.name
        experiment.update(stage=stage_name, stage_values={"timing": timing})
        return
    timings_path = run_dir / "timings.json"
    timings: dict[str, Any] = {}
    if timings_path.exists():
        with open(timings_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            timings.update(payload)
    timings[str(stage)] = timing
    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(timings), f, indent=2)
