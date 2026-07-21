from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _legacy_evaluation(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    decision = {
        key: metrics.get(key) for key in ("accuracy", "precision", "recall", "confusion_matrix") if key in metrics
    }
    if decision:
        decision["threshold"] = 0.5
    return {
        "samples": metrics.get("samples", {}),
        "auroc": metrics.get("auroc", metrics.get("roc_auc")),
        "auprc": metrics.get("auprc", metrics.get("average_precision")),
        "decisions": {"default": decision} if decision else {},
    }


def load_report(run_dir: Path) -> dict[str, Any]:
    """Load the canonical report, adapting pre-schema runs when necessary."""

    run_dir = Path(run_dir).expanduser().resolve()
    canonical_path = run_dir / "reports" / "report.json"
    if canonical_path.exists():
        return _read_json(canonical_path)

    legacy_path = run_dir / "reports" / "supervised_report.json"
    if not legacy_path.exists():
        raise FileNotFoundError(f"No reports/report.json or legacy reports/supervised_report.json in {run_dir}")
    legacy = _read_json(legacy_path)
    fit = legacy.get("fit_assessment", {})
    evaluations = {
        name: evaluation
        for name, evaluation in (
            ("train", _legacy_evaluation(fit.get("train_offline_metrics"))),
            ("calibration", _legacy_evaluation(fit.get("calibration_offline_metrics"))),
            ("inference", _legacy_evaluation(fit.get("offline_metrics"))),
        )
        if evaluation is not None
    }
    tuning = legacy.get("tuning", {})
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "legacy_adapted",
        "task": legacy.get("task", {}),
        "selection": {
            "metric": tuning.get("metric"),
            "winner": tuning.get("winner", {}),
        },
        "evaluation": evaluations,
        "warnings": [
            "Adapted from the legacy reports/supervised_report.json schema.",
            *[str(value) for value in legacy.get("warnings", [])],
        ],
    }
