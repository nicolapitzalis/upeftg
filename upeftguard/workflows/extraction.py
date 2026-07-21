from __future__ import annotations

from pathlib import Path
from typing import Any

from ..features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
)
from ..utilities.core.paths import default_dataset_root
from ..orchestration.slurm.experiment import submit_feature_extraction
from .common import (
    timing_finish as _timing_finish,
    timing_start as _timing_start,
)
from .config import (
    coerce_features as _coerce_features,
    feature_params as _feature_params,
    stage_context as _stage_context,
)

from ..features.registry import extract_features
from ..artifacts.extraction import write_extracted_feature_bundle
from ..artifacts.provenance.datasets import (
    build_dataset_reference_payload_from_items,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from ..utilities.core.manifest import parse_single_manifest_json, resolve_manifest_path


def run_feature_extraction(
    *,
    manifest_json: Path,
    dataset_root: Path = default_dataset_root(),
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    backend: str = "slurm",
    partition: str = "extra",
    nodes: str = "auto",
    cpus_per_worker: str = "4",
    workers_per_node: str = "auto",
    dry_run: bool = False,
    features: list[str] | None = None,
    spectral_sv_top_k: int = 8,
    spectral_moment_source: str = DEFAULT_SPECTRAL_MOMENT_SOURCE,
    spectral_qv_sum_mode: str = DEFAULT_SPECTRAL_QV_SUM_MODE,
    spectral_entrywise_delta_mode: str = DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    spectral_attention_granularity: str = DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    stream_block_size: int = 131072,
    dtype_name: str = "float32",
    wait_for_job_index: bool = False,
    parallelization_settings: list[str] | None = None,
) -> dict[str, Any]:
    resolved_features = _coerce_features(features)
    output_root = Path(output_root).expanduser().resolve()
    experiment, ctx = _stage_context(
        output_root,
        run_id=run_id,
        workflow="extract",
        stage="extraction",
        backend=backend,
    )
    experiment.update(stage="extraction", stage_status="running")
    experiment.update(
        values={
            "configuration": {
                "manifest_json": str(resolve_manifest_path(manifest_json)),
                "dataset_root": str(Path(dataset_root).expanduser().resolve()),
                "features": resolved_features,
                "spectral_moment_source": spectral_moment_source,
                "spectral_qv_sum_mode": spectral_qv_sum_mode,
                "spectral_attention_granularity": spectral_attention_granularity,
                "parallelization_settings": list(parallelization_settings or []),
            }
        }
    )
    if backend == "slurm":
        timing, started_perf = _timing_start("extraction_submit", backend="slurm")
        submission = submit_feature_extraction(
            run_dir=ctx.run_dir,
            run_id=ctx.run_id,
            manifest_json=resolve_manifest_path(manifest_json),
            dataset_root=Path(dataset_root).expanduser().resolve(),
            output_root=Path(output_root).expanduser().resolve(),
            output_run_dir=ctx.run_dir,
            features=resolved_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            spectral_attention_granularity=spectral_attention_granularity,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            partition=partition,
            nodes=nodes,
            cpus_per_worker=cpus_per_worker,
            workers_per_node=workers_per_node,
            wait_for_job_index=wait_for_job_index,
            parallelization_settings=parallelization_settings,
            dry_run=dry_run,
        )
        timing = _timing_finish(timing, started_perf)
        timings = {"submit": timing}
        feature_path = str(ctx.features_dir / "spectral_features.npy")
        experiment.update(
            stage="extraction",
            stage_status="submitted" if not dry_run else "planned",
            stage_values={
                "job_ids": [submission.get("job_id")] if submission.get("job_id") else [],
                "timing": timing,
            },
            artifacts={
                "extracted_features": {
                    "kind": "spectral_feature_bundle",
                    "path": experiment.display_path(Path(feature_path)),
                }
            },
        )
        return {
            "run_dir": str(experiment.run_dir),
            "stage_run_dir": str(ctx.run_dir),
            "run_id": ctx.run_id,
            "feature_path": feature_path,
            "slurm_job_index": str(submission["job_index"]),
            "backend": "slurm",
            "slurm": submission,
            "timings": timings,
        }

    timing, started_perf = _timing_start("extraction", backend="local")
    manifest_json = resolve_manifest_path(manifest_json)
    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=Path(dataset_root).expanduser().resolve(),
        section_key="path",
    )
    params = _feature_params(
        features=resolved_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        spectral_attention_granularity=spectral_attention_granularity,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
    )
    bundle, warnings = extract_features(
        extractor_name="spectral",
        items=items,
        params=params,
    )
    artifacts = write_extracted_feature_bundle(
        bundle=bundle,
        items=items,
        extractor_name="spectral",
        output_dir=ctx.features_dir,
    )
    kept_names = set(bundle.model_names)
    kept_items = [item for item in items if item.model_name in kept_names]
    dataset_reference_payload = build_dataset_reference_payload_from_items(
        items=kept_items,
        artifact_kind="feature_extract",
        manifest_json=manifest_json,
        dataset_root=Path(dataset_root).expanduser().resolve(),
        artifact_model_count=len(bundle.model_names),
        source_artifacts=[str(manifest_json)],
    )
    dataset_reference_report = write_dataset_reference_report(
        default_dataset_reference_report_path(ctx.features_dir),
        dataset_reference_payload,
    )
    timing = _timing_finish(timing, started_perf)
    timings = {"extraction": timing}
    experiment.update(
        stage="extraction",
        stage_status="completed",
        stage_values={"timing": timing, "warnings": warnings},
        artifacts={
            "extracted_features": {
                "kind": "spectral_feature_bundle",
                "path": experiment.display_path(Path(artifacts["feature_path"])),
            }
        },
    )
    return {
        "run_dir": str(experiment.run_dir),
        "stage_run_dir": str(ctx.run_dir),
        "run_id": ctx.run_id,
        "feature_path": artifacts["feature_path"],
        "dataset_reference_report": str(dataset_reference_report),
        "warnings": warnings,
        "backend": "local",
        "timings": timings,
    }
