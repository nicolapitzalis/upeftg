from __future__ import annotations

from pathlib import Path
from typing import Any

from ..orchestration.slurm.experiment import submit_checkpoint_inference
from .common import (
    timing_finish as _timing_finish,
    timing_start as _timing_start,
)
from .config import stage_context as _stage_context


def run_checkpoint_inference(
    *,
    checkpoint: Path,
    manifest_json: Path,
    feature_file: Path,
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    cpus_per_worker: str = "4",
    dry_run: bool = False,
    dependency: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    experiment, ctx = _stage_context(
        output_root,
        run_id=run_id,
        workflow="infer",
        stage="inference",
        backend=backend,
    )
    experiment.update(stage="inference", stage_status="running")
    if backend == "slurm":
        timing, started_perf = _timing_start("inference_submit", backend="slurm")
        submission = submit_checkpoint_inference(
            run_dir=ctx.run_dir,
            run_id=ctx.run_id,
            checkpoint=checkpoint,
            manifest_json=manifest_json,
            feature_file=feature_file,
            output_root=output_root,
            partition=partition,
            cpus_per_worker=cpus_per_worker,
            dependency=dependency,
            dry_run=dry_run,
        )
        timing = _timing_finish(timing, started_perf)
        experiment.update(
            stage="inference",
            stage_status="submitted" if not dry_run else "planned",
            stage_values={
                "job_ids": [submission.get("job_id")] if submission.get("job_id") else [],
                "timing": timing,
                "dependency": dependency,
                "input_feature_path": experiment.display_path(Path(feature_file)),
                "checkpoint": experiment.display_path(Path(checkpoint)),
            },
        )
        return {
            "run_dir": str(experiment.run_dir),
            "stage_run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "backend": "slurm",
            "slurm": submission,
            "timings": {"submit": timing},
        }

    from ..supervised.lifecycle.inference import run_supervised_checkpoint_inference

    timing, started_perf = _timing_start("inference", backend="local")
    result = run_supervised_checkpoint_inference(
        checkpoint=checkpoint,
        manifest_json=manifest_json,
        feature_file=feature_file,
        output_root=output_root,
        run_id=ctx.run_id,
        output_run_dir=ctx.run_dir,
    )
    timing = _timing_finish(timing, started_perf)
    result_run_dir = Path(str(result["run_dir"]))
    timings = {"inference": timing}
    experiment.update(
        stage="inference",
        stage_status="completed",
        stage_values={"timing": timing},
        artifacts={
            "inference_report": {
                "kind": "inference_report",
                "path": experiment.display_path(Path(str(result.get("report")))),
            }
        },
    )
    result.update(
        {
            "run_dir": str(experiment.run_dir),
            "stage_run_dir": str(result_run_dir),
            "backend": "local",
            "timings": timings,
        }
    )
    return result
