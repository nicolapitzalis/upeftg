from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..artifacts.dataset_references import (
    default_dataset_reference_report_path,
    resolve_dataset_reference_payload_for_artifact,
    write_dataset_reference_report,
)
from ..core.serialization import json_ready


def _feature_extract_run_dirs(root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for run_config_path in sorted(root.rglob("run_config.json")):
        run_dir = run_config_path.parent
        try:
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("pipeline")) != "feature_extract":
            continue
        run_dirs.append(run_dir.resolve())
    return run_dirs


def _merge_output_dirs(root: Path) -> list[Path]:
    dirs: set[Path] = set()
    for report_name in ["spectral_merge_report.json", "schema_group_merge_report.json"]:
        for report_path in root.rglob(report_name):
            dirs.add(report_path.parent.resolve())
    return sorted(dirs)


def _update_artifact_index(run_dir: Path, report_path: Path) -> None:
    artifact_index_path = run_dir / "artifact_index.json"
    if not artifact_index_path.exists():
        return
    payload: dict[str, Any]
    try:
        payload = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["dataset_reference_report"] = str(report_path)
    with open(artifact_index_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def backfill_dataset_reference_reports(root: Path) -> dict[str, int]:
    resolved_root = root.expanduser()
    if not resolved_root.is_absolute():
        resolved_root = (Path.cwd().resolve() / resolved_root).resolve()
    else:
        resolved_root = resolved_root.resolve()

    extraction_written = 0
    merge_written = 0

    for run_dir in _feature_extract_run_dirs(resolved_root):
        payload = resolve_dataset_reference_payload_for_artifact(run_dir, prefer_existing_report=False)
        report_path = write_dataset_reference_report(
            default_dataset_reference_report_path(run_dir / "reports"),
            payload,
        )
        _update_artifact_index(run_dir, report_path)
        extraction_written += 1

    for output_dir in _merge_output_dirs(resolved_root):
        payload = resolve_dataset_reference_payload_for_artifact(output_dir, prefer_existing_report=False)
        write_dataset_reference_report(default_dataset_reference_report_path(output_dir), payload)
        merge_written += 1

    return {
        "feature_extract_reports_written": int(extraction_written),
        "merge_reports_written": int(merge_written),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill dataset_reference_report.json files for existing feature extraction and merge artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Root directory to scan for feature extraction and merge artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stats = backfill_dataset_reference_reports(args.root)
    print("Dataset reference backfill complete")
    print(f"Feature extraction reports written: {stats['feature_extract_reports_written']}")
    print(f"Merge reports written: {stats['merge_reports_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
