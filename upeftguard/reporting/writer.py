from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np

from ..utilities.core.serialization import json_ready


REPORT_SCHEMA_VERSION = 1
EXPERIMENT_CONFIG_SCHEMA_VERSION = 1
PARTITIONS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredictionPartition:
    model_names: Sequence[str]
    labels: Sequence[int | None]
    scores: np.ndarray


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)
        handle.write("\n")
    return path


def write_prediction_partition(path: Path, partition: PredictionPartition) -> Path:
    scores = np.asarray(partition.scores, dtype=np.float64).reshape(-1)
    if len(partition.model_names) != len(partition.labels) or len(partition.labels) != int(scores.size):
        raise ValueError("Prediction partition names, labels, and scores must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "model_name", "label", "backdoor_score"])
        writer.writeheader()
        for index, (name, label, score) in enumerate(zip(partition.model_names, partition.labels, scores)):
            writer.writerow(
                {
                    "index": int(index),
                    "model_name": str(name),
                    "label": "" if label is None else int(label),
                    "backdoor_score": float(score),
                }
            )
    return path


def write_reporting_bundle(
    *,
    run_dir: Path,
    report: dict[str, Any],
    experiment_config: dict[str, Any],
    source_manifest: Path,
    data_partitions: dict[str, Sequence[str]],
    model_grid: dict[str, Any],
    predictions: dict[str, PredictionPartition],
    thresholds: dict[str, Any] | None,
    tuning_candidates: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    run_dir = Path(run_dir).expanduser().resolve()
    inputs_dir = run_dir / "inputs"
    reports_dir = run_dir / "reports"
    prediction_dir = reports_dir / "predictions"
    tuning_dir = reports_dir / "tuning"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = Path(source_manifest).expanduser().resolve()
    manifest_snapshot = inputs_dir / "data_manifest.json"
    if source_manifest != manifest_snapshot:
        shutil.copyfile(source_manifest, manifest_snapshot)

    experiment_path = _write_json(
        run_dir / "experiment_config.json",
        {"schema_version": EXPERIMENT_CONFIG_SCHEMA_VERSION, **dict(experiment_config)},
    )
    partitions_path = _write_json(
        inputs_dir / "data_partitions.json",
        {
            "schema_version": PARTITIONS_SCHEMA_VERSION,
            "partitions": {name: [str(value) for value in values] for name, values in data_partitions.items()},
        },
    )
    grid_path = _write_json(inputs_dir / "model_grid.json", dict(model_grid))

    prediction_paths = {
        name: write_prediction_partition(prediction_dir / f"{name}.csv", partition)
        for name, partition in predictions.items()
    }
    thresholds_path = _write_json(reports_dir / "thresholds.json", dict(thresholds or {}))
    tuning_summary_path = _write_json(
        tuning_dir / "summary.json",
        {
            "metric": report.get("selection", {}).get("metric"),
            "winner": report.get("selection", {}).get("winner"),
            "candidate_count": int(len(tuning_candidates)),
            "successful_candidate_count": report.get("selection", {}).get("successful_candidate_count"),
            "failed_candidate_count": report.get("selection", {}).get("failed_candidate_count"),
            "failed_candidates": report.get("selection", {}).get(
                "failed_candidates",
                [],
            ),
        },
    )
    candidates_dir = tuning_dir / "candidates"
    candidate_paths: list[Path] = []
    for index, candidate in enumerate(tuning_candidates):
        task_index = int(candidate.get("task_index", index))
        candidate_paths.append(_write_json(candidates_dir / f"task_{task_index:04d}.json", dict(candidate)))

    canonical_report = {"schema_version": REPORT_SCHEMA_VERSION, **dict(report)}
    report_path = _write_json(reports_dir / "report.json", canonical_report)
    return {
        "experiment_config": experiment_path,
        "data_manifest": manifest_snapshot,
        "data_partitions": partitions_path,
        "model_grid": grid_path,
        "report": report_path,
        "thresholds": thresholds_path,
        "tuning_summary": tuning_summary_path,
        **{f"{name}_predictions": path for name, path in prediction_paths.items()},
    }
