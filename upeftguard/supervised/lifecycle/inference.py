from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ...features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
)
from ...reporting import (
    PredictionPartition,
    build_binary_evaluation,
    build_grouped_evaluations,
    write_reporting_bundle,
)
from ...utilities.core.manifest import (
    infer_attack_sample_identities,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
)
from ...utilities.core.run_context import create_run_context
from ..data.normalization import apply_input_normalization_to_slice
from ..data.preparation import prepare_supervised_dataset
from ..evaluation.prediction import predict_task_outputs
from ..evaluation.thresholds import build_selected_threshold_specs
from ..models.artifact import load_supervised_artifact
from ..tasks import labels_from_items, task_spec_from_payload
from ..validation.splits import project_optional_labels_to_binary
from .timing import append_stage_timing
from .tuning import _parse_rank_from_model_name


def run_supervised_checkpoint_inference(
    *,
    checkpoint: Path,
    manifest_json: Path,
    feature_file: Path,
    output_root: Path,
    run_id: str | None,
    output_run_dir: Path | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    checkpoint = Path(checkpoint).expanduser().resolve()
    manifest_json = resolve_manifest_path(manifest_json)
    feature_file = Path(feature_file).expanduser().resolve()

    artifact = load_supervised_artifact(checkpoint)
    contract = artifact.inference_contract
    task_spec = task_spec_from_payload(contract.get("task"))
    feature_config = contract.get("feature_configuration")
    if not isinstance(feature_config, dict):
        raise ValueError("Checkpoint inference_contract is missing feature_configuration")

    items = parse_single_manifest_json_by_model_name(
        manifest_path=manifest_json,
        section_key="path",
    )
    identities = infer_attack_sample_identities(items)
    prepared = prepare_supervised_dataset(
        feature_spec=feature_file,
        items=items,
        sample_identities=identities,
        spectral_features=[str(value) for value in feature_config.get("spectral_features", [])],
        spectral_sv_top_k=int(feature_config.get("spectral_sv_top_k", 8)),
        spectral_moment_source=str(feature_config.get("spectral_moment_source", DEFAULT_SPECTRAL_MOMENT_SOURCE)),
        spectral_qv_sum_mode=str(feature_config.get("spectral_qv_sum_mode", DEFAULT_SPECTRAL_QV_SUM_MODE)),
        spectral_attention_granularity=str(
            feature_config.get(
                "spectral_attention_granularity",
                DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
            )
        ),
        allow_manifest_subset=False,
    )

    normalized_config = contract.get("input_normalization")
    input_normalization = str(normalized_config.get("mode", "none")) if isinstance(normalized_config, dict) else "none"
    inference_features = apply_input_normalization_to_slice(
        prepared.features,
        dataset_group_names=list(prepared.dataset_group_names),
        input_normalization=input_normalization,
    )
    outputs = predict_task_outputs(artifact.model, inference_features, task_spec=task_spec)
    scores = np.asarray(outputs.backdoor_scores, dtype=np.float64)

    selected_items = list(prepared.items)
    selected_identities = list(prepared.sample_identities)
    domain_adaptation = contract.get("domain_adaptation")
    if isinstance(domain_adaptation, dict) and bool(domain_adaptation.get("enabled")):
        training_ranks = {int(value) for value in domain_adaptation.get("train_ranks", [])}
        inference_ranks = {_parse_rank_from_model_name(str(item.model_name)) for item in selected_items}
        overlap = sorted(training_ranks & inference_ranks)
        if overlap:
            raise ValueError(f"DANN inference ranks must be disjoint from training ranks; overlap={overlap}")
    labels_value, labels_known, _ = labels_from_items(
        selected_items,
        task_spec=task_spec,
        sample_identities=selected_identities,
    )
    optional_task_labels = [
        int(value) if bool(known) else None for value, known in zip(labels_value.tolist(), labels_known.tolist())
    ]
    labels = project_optional_labels_to_binary(optional_task_labels, task_spec=task_spec)
    model_names = [str(item.model_name) for item in selected_items]

    thresholds = contract.get("thresholds")
    threshold_document = (
        dict(thresholds)
        if isinstance(thresholds, dict)
        else {
            "enabled": False,
            "method": None,
            "source_partition": None,
            "accepted_fprs": [],
            "selections": [],
        }
    )
    threshold_specs = build_selected_threshold_specs(threshold_document)
    evaluation = build_binary_evaluation(
        labels=labels,
        scores=scores,
        calibrated_thresholds=threshold_specs,
    )
    grouped = build_grouped_evaluations(
        identities=selected_identities,
        labels=labels,
        scores=scores,
        calibrated_thresholds=threshold_specs,
    )
    if grouped:
        evaluation["groups"] = grouped

    ctx = create_run_context(
        pipeline="inference",
        output_root=Path(output_root).expanduser().resolve(),
        run_id=run_id,
        run_dir=output_run_dir,
        create_directories=("reports",),
    )
    work_dir = ctx.run_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared_arrays = work_dir / "prepared_arrays.npz"
    np.savez_compressed(
        prepared_arrays,
        label_values=labels_value.astype(np.int32),
        label_known=labels_known.astype(np.int8),
        model_names=np.asarray(model_names, dtype=str),
        dataset_groups=np.asarray(prepared.dataset_group_names, dtype=str),
    )

    model_name = str(contract.get("model_name", "unknown"))
    report = {
        "run_id": str(ctx.run_id),
        "status": "complete",
        "task": {
            "training_mode": str(task_spec.task_mode),
            "evaluation_mode": "backdoor_vs_clean",
            "positive_class": "any_backdoor",
        },
        "selection": {
            "metric": "loaded_checkpoint",
            "strategy": "loaded_checkpoint",
            "candidate_count": 0,
            "winner": {"model_name": model_name, "status": "loaded_checkpoint"},
        },
        "evaluation": {"inference": evaluation},
        "warnings": [str(value) for value in prepared.warnings],
    }
    experiment_config = {
        "pipeline": "inference",
        "task": task_spec.to_dict(),
        "data": {
            "manifest_snapshot": "inputs/data_manifest.json",
            "partitions": "inputs/data_partitions.json",
        },
        "features": {
            "extractor": "spectral",
            "params": dict(feature_config),
            "input_normalization": (
                dict(normalized_config) if isinstance(normalized_config, dict) else {"mode": "none"}
            ),
        },
        "model": {
            "checkpoint": str(checkpoint),
            "model_name": model_name,
        },
        "calibration": {
            "enabled": bool(threshold_document.get("enabled", False)),
            "accepted_fprs": threshold_document.get("accepted_fprs", []),
        },
    }
    output_paths = write_reporting_bundle(
        run_dir=ctx.run_dir,
        report=report,
        experiment_config=experiment_config,
        source_manifest=manifest_json,
        data_partitions={"inference": model_names},
        model_grid={"selection_metric": "loaded_checkpoint", "models": [model_name], "candidates": []},
        predictions={
            "inference": PredictionPartition(
                model_names=model_names,
                labels=labels,
                scores=scores,
            )
        },
        thresholds=threshold_document,
        tuning_candidates=[],
    )
    artifacts = {
        "checkpoint": str(checkpoint),
        "prepared_arrays": str(prepared_arrays),
        **{key: str(path) for key, path in output_paths.items()},
    }
    for key, value in artifacts.items():
        ctx.add_artifact(key, Path(value))
    ctx.finalize(
        {
            "pipeline": "inference",
            "status": "complete",
            "experiment_config": str(output_paths["experiment_config"]),
        }
    )
    append_stage_timing(
        run_dir=ctx.run_dir,
        stage="checkpoint_inference",
        started_at=started_at,
        started_perf=started_perf,
    )
    return {
        "run_dir": str(ctx.run_dir),
        "report": str(output_paths["report"]),
        "inference_scores_csv": str(output_paths["inference_predictions"]),
        "checkpoint": str(checkpoint),
    }
