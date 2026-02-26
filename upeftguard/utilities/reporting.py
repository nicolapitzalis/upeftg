from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_compare_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Aggregate multiple clustering reports"
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Report entries formatted as name=/path/to/clustering_report.json",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("runs/report/compare_representations/representation_comparison.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--target-auroc",
        type=float,
        default=0.80,
        help="Minimum AUROC target to consider flow handoff",
    )
    parser.add_argument(
        "--target-stability",
        type=float,
        default=0.80,
        help="Minimum stability score target to consider flow handoff",
    )


def parse_report_entry(entry: str) -> tuple[str, Path]:
    if "=" not in entry:
        raise ValueError(f"Invalid --reports entry '{entry}'. Expected format name=path")
    name, path = entry.split("=", 1)
    if not name:
        raise ValueError(f"Invalid report name in entry '{entry}'")
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report path not found: {report_path}")
    return name, report_path


def safe_get_best_auroc(report: dict[str, Any]) -> float | None:
    item = report.get("offline_eval", {}).get("best_score_model_by_auroc")
    if not isinstance(item, dict):
        return None
    value = item.get("auroc")
    return float(value) if isinstance(value, (float, int)) else None


def safe_get_best_ari(report: dict[str, Any]) -> float | None:
    item = report.get("offline_eval", {}).get("best_partition_model_by_ari")
    if not isinstance(item, dict):
        return None
    value = item.get("adjusted_rand")
    return float(value) if isinstance(value, (float, int)) else None


def safe_get_best_stability(report: dict[str, Any]) -> float | None:
    item = report.get("stability", {}).get("best_by_stability")
    if not isinstance(item, dict):
        return None
    value = item.get("stability_score")
    return float(value) if isinstance(value, (float, int)) else None


def run_compare_reports(
    *,
    reports: list[str],
    output_file: Path,
    target_auroc: float,
    target_stability: float,
) -> Path:
    records = []
    for entry in reports:
        name, report_path = parse_report_entry(entry)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        unsup = report.get("unsupervised_selection", {})
        winner = unsup.get("winner") if isinstance(unsup, dict) else None

        record = {
            "name": name,
            "report_path": str(report_path),
            "representation_info": report.get("representation_info", {}),
            "unsupervised_metric": unsup.get("metric") if isinstance(unsup, dict) else None,
            "unsupervised_winner": winner,
            "best_auroc": safe_get_best_auroc(report),
            "best_ari": safe_get_best_ari(report),
            "best_stability": safe_get_best_stability(report),
        }
        records.append(record)

    best_by_auroc = None
    valid_aurocs = [r for r in records if r["best_auroc"] is not None]
    if valid_aurocs:
        best_by_auroc = max(valid_aurocs, key=lambda r: r["best_auroc"])

    best_by_stability = None
    valid_stability = [r for r in records if r["best_stability"] is not None]
    if valid_stability:
        best_by_stability = max(valid_stability, key=lambda r: r["best_stability"])

    global_auroc = best_by_auroc["best_auroc"] if best_by_auroc is not None else None
    global_stability = best_by_stability["best_stability"] if best_by_stability is not None else None

    pass_auroc = global_auroc is not None and global_auroc >= target_auroc
    pass_stability = global_stability is not None and global_stability >= target_stability

    go_to_flow = bool(pass_auroc and pass_stability)

    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {
            "target_auroc": target_auroc,
            "target_stability": target_stability,
        },
        "records": records,
        "best_by_auroc": best_by_auroc,
        "best_by_stability": best_by_stability,
        "go_to_flow": go_to_flow,
        "reason": (
            "Proceed to flow modeling"
            if go_to_flow
            else "Do not proceed to flow yet; iterate representation/baselines"
        ),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    return output_file
