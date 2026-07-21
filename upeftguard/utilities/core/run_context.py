from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from .serialization import json_ready


@dataclass
class RunContext:
    pipeline: str
    output_root: Path
    run_id: str
    run_dir: Path
    features_dir: Path
    models_dir: Path
    reports_dir: Path
    logs_dir: Path
    artifacts: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def add_artifact(self, key: str, path: Path) -> None:
        self.artifacts[key] = str(path)

    def add_timing(self, key: str, value_seconds: float) -> None:
        self.timings[key] = float(value_seconds)

    def finalize(self, run_config: dict[str, Any]) -> None:
        with open(self.run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(json_ready(run_config), f, indent=2)
        with open(self.run_dir / "artifact_index.json", "w", encoding="utf-8") as f:
            json.dump(json_ready(self.artifacts), f, indent=2)
        timings = {}
        timings_path = self.run_dir / "timings.json"
        if timings_path.exists():
            with open(timings_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                timings.update(existing)
        timings.update(self.timings)
        with open(self.run_dir / "timings.json", "w", encoding="utf-8") as f:
            json.dump(json_ready(timings), f, indent=2)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_context(
    *,
    pipeline: str,
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    run_dir: Path | None = None,
    create_directories: tuple[str, ...] = ("features", "models", "reports", "logs"),
) -> RunContext:
    rid = run_id or _default_run_id()
    output_root = Path(output_root).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve() if run_dir is not None else output_root / pipeline / rid
    features_dir = run_dir / "features"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"

    directories = {
        "features": features_dir,
        "models": models_dir,
        "reports": reports_dir,
        "logs": logs_dir,
    }
    unknown = set(create_directories) - set(directories)
    if unknown:
        raise ValueError(f"Unknown run directories: {sorted(unknown)}")
    for name in create_directories:
        directories[name].mkdir(parents=True, exist_ok=True)

    return RunContext(
        pipeline=pipeline,
        output_root=output_root,
        run_id=rid,
        run_dir=run_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )
