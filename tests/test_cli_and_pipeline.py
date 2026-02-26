import csv
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import upeftguard.cli as cli_mod


REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def make_tiny_adapter_dataset(root: Path) -> Path:
    data_dir = root / "tiny_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)

    for label in [0, 1]:
        for idx in [0, 1, 2, 3]:
            model_dir = data_dir / f"tiny_label{label}_{idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensors = {}
            for layer in range(2):
                for module_name, out_dim in [("q_proj", 4), ("v_proj", 3)]:
                    a_key = f"base_model.model.model.layers.{layer}.self_attn.{module_name}.lora_A.weight"
                    b_key = f"base_model.model.model.layers.{layer}.self_attn.{module_name}.lora_B.weight"

                    a = rng.standard_normal((2, 4), dtype=np.float32)
                    b = rng.standard_normal((out_dim, 2), dtype=np.float32)

                    if label == 1:
                        a = a + 0.5
                        b = b + 0.25

                    tensors[a_key] = a
                    tensors[b_key] = b

            save_file(tensors, str(model_dir / "adapter_model.safetensors"))

    return data_dir


def write_single_manifest(path: Path) -> None:
    payload = {
        "path": [
            {"path": "tiny_data/tiny_label0_", "indices": [0, 3]},
            {"path": "tiny_data/tiny_label1_", "indices": [0, 3]},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_joint_manifest(path: Path) -> None:
    payload = {
        "train": [
            {"path": "tiny_data/tiny_label0_", "indices": [0, 1]},
            {"path": "tiny_data/tiny_label1_", "indices": [0, 0]},
        ],
        "infer": [
            {"path": "tiny_data/tiny_label0_", "indices": [2, 3]},
            {"path": "tiny_data/tiny_label1_", "indices": [1, 3]},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "upeftguard.cli", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


class TestCliAndPipeline(unittest.TestCase):
    def test_output_path_guards_redirect_filesystem_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with working_directory(tmp_path):
                resolved_output_root = cli_mod._resolve_output_root(Path("/"), "feature_extract")
                resolved_report_file = cli_mod._resolve_report_output_file(Path("/representation_comparison.json"))
                rewritten_download = cli_mod._rewrite_download_local_dir(["--local-dir", "/"])

            self.assertEqual(resolved_output_root, (tmp_path / "runs" / "feature_extract").resolve())
            self.assertEqual(
                resolved_report_file,
                (tmp_path / "runs" / "report" / "compare_representations" / "representation_comparison.json").resolve(),
            )
            self.assertEqual(rewritten_download, ["--local-dir", str((tmp_path / "runs" / "util" / "download_dataset").resolve())])

    def test_feature_extract_svd_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "feature_svd_test"

            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "svd",
                    "--svd-components-grid",
                    "2",
                    "3",
                    "--svd-n-components",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            run_dir = runs_root / "feature_extract" / run_id
            self.assertTrue((run_dir / "run_config.json").exists())
            self.assertTrue((run_dir / "artifact_index.json").exists())
            self.assertTrue((run_dir / "reports" / "feature_extraction_report.json").exists())
            self.assertTrue((run_dir / "features" / "svd_features.npy").exists())

    def test_clustering_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "clustering_test"

            proc = run_cli(
                [
                    "run",
                    "clustering",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "svd",
                    "--svd-components-grid",
                    "2",
                    "3",
                    "--svd-n-components",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--algorithms",
                    "kmeans",
                    "gmm",
                    "mahalanobis",
                    "isolation_forest",
                    "lof",
                    "--k-list",
                    "2",
                    "--gmm-components",
                    "1",
                    "2",
                    "--selection-metric",
                    "silhouette",
                    "--use-offline-label-metrics",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            report_path = runs_root / "clustering" / run_id / "reports" / "clustering_report.json"
            self.assertTrue(report_path.exists())
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            for key in [
                "data_info",
                "representation_info",
                "unsupervised_selection",
                "offline_eval",
                "stability",
                "algorithm_results",
            ]:
                self.assertIn(key, report)

    def test_gmm_train_inference_allows_mixed_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "gmm_manifest.json"
            write_joint_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "gmm_mixed_train_test"

            proc = run_cli(
                [
                    "run",
                    "gmm-train-inference",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--svd-components-grid",
                    "1",
                    "2",
                    "--gmm-components",
                    "1",
                    "2",
                    "--gmm-covariance-types",
                    "diag",
                    "spherical",
                    "--stability-seeds",
                    "42",
                    "43",
                    "--score-percentiles",
                    "90",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            report_path = runs_root / "gmm_train_inference" / run_id / "reports" / "gmm_train_inference_report.json"
            self.assertTrue(report_path.exists())

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            self.assertGreater(report["data_info"]["n_train_backdoored"], 0)
            self.assertIn("fit_assessment", report)

    def test_gmm_thresholds_use_all_train_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "gmm_manifest.json"
            write_joint_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "gmm_threshold_test"

            proc = run_cli(
                [
                    "run",
                    "gmm-train-inference",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--svd-components-grid",
                    "1",
                    "--gmm-components",
                    "1",
                    "--gmm-covariance-types",
                    "diag",
                    "--stability-seeds",
                    "42",
                    "--score-percentiles",
                    "90",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            run_dir = runs_root / "gmm_train_inference" / run_id
            report_path = run_dir / "reports" / "gmm_train_inference_report.json"
            train_scores_path = run_dir / "reports" / "train_scores.csv"

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            rows = report["fit_assessment"]["threshold_evaluation"]
            self.assertEqual(len(rows), 1)
            reported_threshold = float(rows[0]["threshold"])

            scores = []
            with open(train_scores_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scores.append(float(row["score"]))

            expected = float(np.percentile(np.asarray(scores, dtype=np.float64), 90))
            self.assertAlmostEqual(reported_threshold, expected, places=6)


if __name__ == "__main__":
    unittest.main()
