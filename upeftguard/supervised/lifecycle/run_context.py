from __future__ import annotations

from pathlib import Path

from ...utilities.core.run_context import RunContext


PIPELINE_NAME = "training"


def context_from_run_dir(run_dir: Path) -> RunContext:
    output_root = run_dir.parents[1] if len(run_dir.parents) >= 2 else run_dir.parent
    features_dir = run_dir / "features"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"

    for path in [models_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_dir.name,
        run_dir=run_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )
