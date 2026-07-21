from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from ...utilities.core.experiment import experiment_context_from_stage_dir


def merge_schema_group_shards(
    *,
    schema_report_path: Path,
    dataset_root: Path,
    output_dir: Path,
    pipeline_start_epoch_seconds: float,
) -> Path:
    report_path = schema_report_path.expanduser().resolve()
    final_output_dir = output_dir.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Schema report contains no groups: {report_path}")

    multiple_groups = len(groups) > 1
    for group in groups:
        group_output_dir = (
            Path(group["merged_output_dir"]).expanduser().resolve() if multiple_groups else final_output_dir
        )
        shard_glob = str(Path(group["shard_output_root"]).expanduser().resolve() / "shard_*")
        command = [
            sys.executable,
            "-m",
            "upeftguard.cli",
            "run",
            "merge-spectral-shards",
            "--manifest-json",
            str(group["manifest_path"]),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(group_output_dir),
            "--pipeline-start-epoch-seconds",
            str(pipeline_start_epoch_seconds),
            "--shard-run-dir-glob",
            shard_glob,
        ]
        print(f"Merging {group['group_id']} into {group_output_dir}", flush=True)
        subprocess.run(command, check=True)

    if multiple_groups:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "upeftguard.cli",
                "run",
                "finalize-schema-group-merge",
                "--schema-report-path",
                str(report_path),
                "--output-dir",
                str(final_output_dir),
            ],
            check=True,
        )
    experiment = experiment_context_from_stage_dir(final_output_dir.parent)
    if experiment is not None:
        experiment.update(
            stage="extraction",
            stage_status="completed",
            artifacts={
                "extracted_features": {
                    "kind": "spectral_feature_bundle",
                    "path": str(final_output_dir / "spectral_features.npy"),
                    "producer": "extraction",
                }
            },
        )
    schema_groups_dir = report_path.parent / "schema_groups"
    if schema_groups_dir.exists():
        shutil.rmtree(schema_groups_dir)
    report_path.unlink(missing_ok=True)
    print(f"Final merged feature output: {final_output_dir}")
    return final_output_dir
