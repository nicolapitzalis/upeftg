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
    plots_dir: Path
    logs_dir: Path
    cache_root: Path
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
        with open(self.run_dir / "timings.json", "w", encoding="utf-8") as f:
            json.dump(json_ready(self.timings), f, indent=2)



def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_context(
    *,
    pipeline: str,
    output_root: Path = Path("runs"),
    run_id: str | None = None,
) -> RunContext:
    rid = run_id or _default_run_id()
    run_dir = output_root / pipeline / rid
    features_dir = run_dir / "features"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"
    cache_root = output_root / "cache" / "features"

    for path in [features_dir, models_dir, reports_dir, plots_dir, logs_dir, cache_root]:
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        pipeline=pipeline,
        output_root=output_root,
        run_id=rid,
        run_dir=run_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
        cache_root=cache_root,
    )
