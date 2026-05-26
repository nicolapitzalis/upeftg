from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from upeftguard import cli


MANIFEST = Path("manifests/single_datasets/llama2_7b_toxic_backdoors_hard.json")


class CnnCliTests(unittest.TestCase):
    def test_slurm_full_dry_run_records_dependency_chain_and_auto_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "cnn",
                    "full",
                    "--dry-run",
                    "--manifest-json",
                    str(MANIFEST),
                    "--output-root",
                    tmp,
                    "--run-id",
                    "dry_full",
                    "--skip-feature-importance",
                ]
            )
            self.assertEqual(code, 0)
            run_config = json.loads(
                (Path(tmp) / "cnn_full" / "dry_full" / "run_config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(run_config["backend"], "slurm")
        extract = run_config["extract"]["slurm"]
        aggregate = run_config["aggregate"]["slurm"]
        train = run_config["train"]["slurm"]
        self.assertIn("--partition", extract["command"])
        self.assertIn("extra", extract["command"])
        self.assertEqual(extract["env"]["SLURM_CPUS_PER_TASK"], "auto")
        self.assertEqual(extract["env"]["SLURM_MAX_CONCURRENT"], "auto")
        self.assertEqual(aggregate["dependency"], "afterok:DRYRUN_FEATURE_FINALIZE")
        self.assertEqual(train["dependency"], "afterok:DRYRUN_AGGREGATE")
        self.assertEqual(train["env"]["SLURM_CPUS_PER_TASK_REQUEST"], "auto")
        self.assertEqual(train["env"]["SLURM_MAX_CONCURRENT_REQUEST"], "auto")

    def test_slurm_extract_dry_run_writes_timing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "cnn",
                    "extract",
                    "--dry-run",
                    "--manifest-json",
                    str(MANIFEST),
                    "--output-root",
                    tmp,
                    "--run-id",
                    "dry_extract",
                ]
            )
            self.assertEqual(code, 0)
            timings = json.loads(
                (Path(tmp) / "cnn_extract" / "dry_extract" / "timings.json").read_text(encoding="utf-8")
            )

        submit = timings["submit"]
        self.assertIn("start_timestamp_utc", submit)
        self.assertIn("end_timestamp_utc", submit)
        self.assertIn("elapsed_seconds", submit)
        self.assertGreaterEqual(float(submit["elapsed_seconds"]), 0.0)

    def test_local_cnn_commands_dispatch_to_pipeline_functions(self) -> None:
        fake = {"run_dir": "/tmp/run", "feature_path": "/tmp/features.npy"}
        with patch("upeftguard.cli.run_cnn_extract", return_value=fake) as extract:
            self.assertEqual(
                cli.main(
                    [
                        "cnn",
                        "extract",
                        "--backend",
                        "local",
                        "--manifest-json",
                        str(MANIFEST),
                    ]
                ),
                0,
            )
            self.assertEqual(extract.call_args.kwargs["backend"], "local")

        with patch("upeftguard.cli.run_cnn_aggregate", return_value=fake) as aggregate:
            self.assertEqual(
                cli.main(
                    [
                        "cnn",
                        "aggregate",
                        "--backend",
                        "local",
                        "--feature-file",
                        "/tmp/source.npy",
                    ]
                ),
                0,
            )
            self.assertEqual(aggregate.call_args.kwargs["backend"], "local")

        with patch("upeftguard.cli.run_cnn_train", return_value={"run_dir": "/tmp/supervised"}) as train:
            self.assertEqual(
                cli.main(
                    [
                        "cnn",
                        "train",
                        "--backend",
                        "local",
                        "--manifest-json",
                        str(MANIFEST),
                        "--feature-file",
                        "/tmp/agg.npy",
                    ]
                ),
                0,
            )
            self.assertEqual(train.call_args.kwargs["backend"], "local")

        with patch("upeftguard.cli.run_cnn_infer", return_value={"run_dir": "/tmp/infer"}) as infer:
            self.assertEqual(
                cli.main(
                    [
                        "cnn",
                        "infer",
                        "--backend",
                        "local",
                        "--run-dir",
                        "/tmp/supervised",
                    ]
                ),
                0,
            )
            self.assertEqual(infer.call_args.kwargs["backend"], "local")

    def test_local_full_dispatches_to_pipeline_function(self) -> None:
        with patch("upeftguard.cli.run_cnn_full", return_value={"run_dir": "/tmp/full"}) as full:
            self.assertEqual(
                cli.main(
                    [
                        "cnn",
                        "full",
                        "--backend",
                        "local",
                        "--manifest-json",
                        str(MANIFEST),
                    ]
                ),
                0,
            )
            self.assertEqual(full.call_args.kwargs["backend"], "local")


if __name__ == "__main__":
    unittest.main()
