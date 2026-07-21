"""Small serialization and timing primitives for workflow coordinators."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ..utilities.core.serialization import json_ready


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def json_write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2)
    return path


def timing_start(stage: str, *, backend: str) -> tuple[dict[str, Any], float]:
    now = utc_now()
    return (
        {
            "stage": stage,
            "backend": backend,
            "start_timestamp_utc": now.isoformat(),
            "start_epoch_seconds": float(now.timestamp()),
        },
        perf_counter(),
    )


def timing_finish(timing: dict[str, Any], started_perf: float) -> dict[str, Any]:
    now = utc_now()
    timing.update(
        {
            "end_timestamp_utc": now.isoformat(),
            "end_epoch_seconds": float(now.timestamp()),
            "elapsed_seconds": float(perf_counter() - started_perf),
        }
    )
    return timing
