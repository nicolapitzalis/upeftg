import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

import cluster_z_space
import prepare_data

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def write_manifest(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCliAndPipeline(unittest.TestCase):
    def test_format_metric_zero(self):
        self.assertEqual(cluster_z_space.format_metric(0.0), "0.0000")

    def test_k_and_component_bounds(self):
        k_list, k_warnings = cluster_z_space.sanitize_k_list([1, 2, 9], n_samples=4)
        self.assertEqual(k_list, [2, 3])
        self.assertTrue(any("Clipped" in w for w in k_warnings))

        comp_list, comp_warnings, max_rank = prepare_data.sanitize_component_grid([0, 5], n_samples=4, n_features=10)
        self.assertEqual(max_rank, 3)
        self.assertEqual(comp_list, [3])
        self.assertTrue(any("Clipped" in w for w in comp_warnings))

    def test_prepare_data_missing_path(self):
        cmd = [
            sys.executable,
            "prepare_data.py",
            "--data-dir",
            "does_not_exist",
            "--output-dir",
            "tmp_out",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Data directory not found", proc.stderr + proc.stdout)

    def test_cluster_missing_path(self):
        cmd = [
            sys.executable,
            "cluster_z_space.py",
            "--data-dir",
            "does_not_exist",
            "--output-dir",
            "tmp_cluster_out",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Data directory not found", proc.stderr + proc.stdout)

    def test_smoke_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adapter_dataset(tmp_path)
            processed_dir = tmp_path / "processed"
            cluster_dir = tmp_path / "cluster"

            prep_cmd = [
                sys.executable,
                "prepare_data.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(processed_dir),
                "--n-per-label",
                "2",
                "--sample-mode",
                "first",
                "--trunc-svds-components",
                "2",
                "5",
            ]
            subprocess.run(prep_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            self.assertTrue((processed_dir / "run_config.json").exists())
            self.assertTrue((processed_dir / "representativeness_summary.json").exists())
            self.assertTrue((processed_dir / "Z_2.npy").exists())
            # n=4 => max rank 3, so requested 5 should be clipped and saved as Z_3
            self.assertTrue((processed_dir / "Z_3.npy").exists())

            cluster_cmd = [
                sys.executable,
                "cluster_z_space.py",
                "--data-dir",
                str(processed_dir),
                "--output-dir",
                str(cluster_dir),
                "--n-components",
                "2",
                "--algorithms",
                "kmeans",
                "gmm",
                "mahalanobis",
                "isolation_forest",
                "lof",
                "--k-list",
                "2",
                "5",
                "--selection-metric",
                "silhouette",
                "--use-offline-label-metrics",
            ]
            subprocess.run(cluster_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            report_path = cluster_dir / "clustering_report.json"
            self.assertTrue(report_path.exists())

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            for key in [
                "data_info",
                "representation_info",
                "unsupervised_selection",
                "offline_eval",
                "stability",
                "artifacts",
            ]:
                self.assertIn(key, report)

    def test_delta_feature_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adapter_dataset(tmp_path)
            delta_dir = tmp_path / "delta"
            cluster_dir = tmp_path / "delta_cluster"

            delta_cmd = [
                sys.executable,
                "extract_delta_features.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(delta_dir),
                "--n-per-label",
                "2",
                "--top-k-singular-values",
                "3",
            ]
            subprocess.run(delta_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            self.assertTrue((delta_dir / "delta_singular_values.npy").exists())
            self.assertTrue((delta_dir / "delta_frobenius.npy").exists())
            self.assertTrue((delta_dir / "feature_metadata.json").exists())

            cluster_cmd = [
                sys.executable,
                "cluster_z_space.py",
                "--data-dir",
                str(delta_dir),
                "--feature-file",
                str(delta_dir / "delta_singular_values.npy"),
                "--output-dir",
                str(cluster_dir),
                "--algorithms",
                "gmm",
                "mahalanobis",
                "isolation_forest",
                "--selection-metric",
                "stability",
                "--use-offline-label-metrics",
            ]
            subprocess.run(cluster_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            report_path = cluster_dir / "clustering_report.json"
            self.assertTrue(report_path.exists())
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["representation_info"]["type"], "external_feature_file")

    def test_gmm_clean_inference_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adapter_dataset(tmp_path)
            out_dir = tmp_path / "gmm_out"
            manifest_json = tmp_path / "gmm_manifest.json"
            manifest_payload = {
                "train": [
                    {"path": "tiny_data/tiny_label0_", "indices": [0, 1]},
                    {"path": "tiny_data/tiny_label0_", "indices": [2, 2]},
                ],
                "infer": [
                    {"path": "tiny_data/tiny_label0_", "indices": [3, 3]},
                    {"path": "tiny_data/tiny_label1_", "indices": [0, 0]},
                    {"path": "tiny_data/tiny_label1_", "indices": [1, 1]},
                ],
            }
            manifest_json.write_text(json.dumps(manifest_payload), encoding="utf-8")

            cmd = [
                sys.executable,
                "gmm_clean_inference.py",
                "--dataset-root",
                str(tmp_path),
                "--manifest-json",
                str(manifest_json),
                "--output-dir",
                str(out_dir),
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
                "95",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            report_path = out_dir / "gmm_clean_inference_report.json"
            self.assertTrue(report_path.exists())
            self.assertTrue((out_dir / "inference_scores.csv").exists())
            self.assertTrue((out_dir / "train_clean_scores.csv").exists())
            self.assertTrue((out_dir / "run_config.json").exists())

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertIn("gmm_selection", report)
            self.assertIn("winner", report["gmm_selection"])
            self.assertIn("fit_assessment", report)
            self.assertIn("offline_metrics", report["fit_assessment"])

    def test_gmm_clean_inference_rejects_non_clean_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            out_dir = tmp_path / "gmm_out"
            train_manifest = tmp_path / "train_clean.txt"
            infer_manifest = tmp_path / "infer_mixed.txt"

            write_manifest(
                train_manifest,
                [
                    "tiny_data/tiny_label0_0",
                    "tiny_data/tiny_label1_0",
                ],
            )
            write_manifest(
                infer_manifest,
                [
                    "tiny_data/tiny_label0_1",
                    "tiny_data/tiny_label1_1",
                ],
            )

            cmd = [
                sys.executable,
                "gmm_clean_inference.py",
                "--dataset-root",
                str(tmp_path),
                "--train-list",
                str(train_manifest),
                "--infer-list",
                str(infer_manifest),
                "--output-dir",
                str(out_dir),
                "--svd-components-grid",
                "1",
                "--gmm-components",
                "1",
                "--gmm-covariance-types",
                "diag",
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("clean models only", proc.stderr + proc.stdout)

    def test_gmm_clean_inference_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            out_dir = tmp_path / "gmm_out"
            train_manifest = tmp_path / "train_clean.txt"
            infer_manifest = tmp_path / "infer_mixed.txt"

            write_manifest(
                train_manifest,
                [
                    "tiny_data/tiny_label0_0",
                    "tiny_data/tiny_label0_1",
                ],
            )
            write_manifest(
                infer_manifest,
                [
                    "tiny_data/tiny_label0_0",
                    "tiny_data/tiny_label1_0",
                ],
            )

            cmd = [
                sys.executable,
                "gmm_clean_inference.py",
                "--dataset-root",
                str(tmp_path),
                "--train-list",
                str(train_manifest),
                "--infer-list",
                str(infer_manifest),
                "--output-dir",
                str(out_dir),
                "--svd-components-grid",
                "1",
                "--gmm-components",
                "1",
                "--gmm-covariance-types",
                "diag",
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("must be disjoint", proc.stderr + proc.stdout)

    def test_gmm_clean_inference_unknown_label_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adapter_dataset(tmp_path)
            mystery_dir = data_dir / "mystery_model"
            mystery_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                data_dir / "tiny_label0_3" / "adapter_model.safetensors",
                mystery_dir / "adapter_model.safetensors",
            )

            out_dir = tmp_path / "gmm_out"
            train_manifest = tmp_path / "train_clean.txt"
            infer_manifest = tmp_path / "infer_mixed.txt"

            write_manifest(
                train_manifest,
                [
                    "tiny_data/tiny_label0_0",
                    "tiny_data/tiny_label0_1",
                    "tiny_data/tiny_label0_2",
                ],
            )
            write_manifest(
                infer_manifest,
                [
                    "tiny_data/tiny_label1_0",
                    "tiny_data/mystery_model",
                ],
            )

            cmd = [
                sys.executable,
                "gmm_clean_inference.py",
                "--dataset-root",
                str(tmp_path),
                "--train-list",
                str(train_manifest),
                "--infer-list",
                str(infer_manifest),
                "--output-dir",
                str(out_dir),
                "--svd-components-grid",
                "1",
                "--gmm-components",
                "1",
                "--gmm-covariance-types",
                "diag",
                "--stability-seeds",
                "42",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            with open(out_dir / "gmm_clean_inference_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)

            self.assertIsNone(report["fit_assessment"]["offline_metrics"]["auroc"])
            warnings = report.get("warnings", [])
            self.assertTrue(any("label-based metrics will be omitted" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
