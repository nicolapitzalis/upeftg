from __future__ import annotations

from pathlib import Path
from typing import Any

from ..orchestration.slurm.experiment import submit_feature_aggregation
from .common import (
    timing_finish as _timing_finish,
    timing_start as _timing_start,
)
from .config import (
    DEFAULT_EXPERIMENT_AGGREGATION_LAYOUT,
    coerce_features as _coerce_features,
    stage_context as _stage_context,
)

from ..artifacts.aggregation import aggregate_features
from ..artifacts.paths import resolve_output_feature_path as _resolve_output_feature_path


def run_feature_aggregation(
    *,
    feature_file: Path,
    output_filename: Path | None = None,
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    cpus_per_worker: str = "4",
    dry_run: bool = False,
    feature_root: Path | None = None,
    features: list[str] | None = None,
    spectral_qv_sum_mode: str = "append",
    spectral_attention_granularity: str | None = None,
    dependency: str | None = None,
) -> dict[str, Any]:
    resolved_features = _coerce_features(features)
    output_root = Path(output_root).expanduser().resolve()
    experiment, ctx = _stage_context(
        output_root,
        run_id=run_id,
        workflow="aggregate",
        stage="aggregation",
        backend=backend,
    )
    experiment.update(stage="aggregation", stage_status="running")
    feature_root = Path(feature_root) if feature_root is not None else ctx.features_dir
    output_filename = (
        Path(output_filename) if output_filename is not None else ctx.features_dir / "spectral_features.npy"
    )
    experiment.update(
        values={
            "configuration": {
                "feature_file": str(Path(feature_file).resolve()),
                "features": resolved_features,
                "spectral_qv_sum_mode": spectral_qv_sum_mode,
                "spectral_attention_granularity": spectral_attention_granularity,
            }
        }
    )

    if backend == "slurm":
        timing, started_perf = _timing_start("aggregation_submit", backend="slurm")
        submission = submit_feature_aggregation(
            run_dir=ctx.run_dir,
            run_id=ctx.run_id,
            feature_file=Path(feature_file),
            output_filename=output_filename,
            output_root=output_root,
            feature_root=feature_root,
            features=resolved_features,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_attention_granularity=spectral_attention_granularity,
            partition=partition,
            cpus_per_worker=cpus_per_worker,
            dependency=dependency,
            dry_run=dry_run,
        )
        timing = _timing_finish(timing, started_perf)
        artifacts = {
            "feature_path": str(_resolve_output_feature_path(output_filename, feature_root=feature_root)),
        }
        experiment.update(
            stage="aggregation",
            stage_status="submitted" if not dry_run else "planned",
            stage_values={
                "job_ids": [submission.get("job_id")] if submission.get("job_id") else [],
                "timing": timing,
                "input_feature_path": experiment.display_path(Path(feature_file)),
                "dependency": dependency,
                "cpus_per_worker": str(cpus_per_worker),
            },
            artifacts={
                "aggregated_features": {
                    "kind": "layer_sequence_feature_bundle",
                    "path": experiment.display_path(Path(artifacts["feature_path"])),
                }
            },
        )
        return {
            "run_dir": str(experiment.run_dir),
            "stage_run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "feature_path": artifacts["feature_path"],
            "backend": "slurm",
            "slurm": submission,
            "timings": {"submit": timing},
        }

    timing, started_perf = _timing_start("aggregation", backend="local")
    outputs = aggregate_features(
        feature_file=Path(feature_file),
        output_filename=output_filename,
        feature_root=feature_root,
        operator="avg",
        features=resolved_features,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        layout=DEFAULT_EXPERIMENT_AGGREGATION_LAYOUT,
    )
    timing = _timing_finish(timing, started_perf)
    artifacts = {key: (str(value) if value is not None else None) for key, value in outputs.items()}
    experiment.update(
        stage="aggregation",
        stage_status="completed",
        stage_values={
            "timing": timing,
            "input_feature_path": experiment.display_path(Path(feature_file)),
            "cpus_per_worker": str(cpus_per_worker),
        },
        artifacts={
            "aggregated_features": {
                "kind": "layer_sequence_feature_bundle",
                "path": experiment.display_path(Path(outputs["feature_path"])),
            }
        },
    )
    return {
        "run_dir": str(experiment.run_dir),
        "stage_run_dir": str(ctx.run_dir),
        "run_id": ctx.run_id,
        "feature_path": str(outputs["feature_path"]),
        "outputs": artifacts,
        "backend": "local",
        "timings": {"aggregation": timing},
    }
