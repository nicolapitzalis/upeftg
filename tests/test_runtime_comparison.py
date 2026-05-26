from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from upeftguard.experiments.runtime_comparison import collect_and_plot, generate_joint_manifest


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict)
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def label_from_entry(entry: str) -> int:
    if "label0" in entry:
        return 0
    if "label1" in entry:
        return 1
    raise AssertionError(f"Missing label token in {entry}")


class RuntimeComparisonTests(unittest.TestCase):
    def test_generate_joint_manifest_preserves_folder_label_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"
            entries = []
            for label in [0, 1]:
                for idx in range(10):
                    entries.append(f"toy_toxic_backdoors_hard_rank256_qv/toy_label{label}_{idx}")
            write_json(
                manifest_path,
                {
                    "path": [
                        {
                            "path": "toy_toxic_backdoors_hard_rank256_qv/toy_label0_",
                            "indices": [0, 9],
                        },
                        {
                            "path": "toy_toxic_backdoors_hard_rank256_qv/toy_label1_",
                            "indices": [0, 9],
                        },
                    ]
                },
            )

            output_path = tmp_path / "generated" / "split.json"
            payload = generate_joint_manifest(
                manifest_json=manifest_path,
                output_path=output_path,
                train_split_percent=80,
                random_state=42,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(payload["train"]), 16)
            self.assertEqual(len(payload["infer"]), 4)
            self.assertEqual(sum(label_from_entry(entry) == 0 for entry in payload["train"]), 8)
            self.assertEqual(sum(label_from_entry(entry) == 1 for entry in payload["train"]), 8)
            self.assertEqual(sum(label_from_entry(entry) == 0 for entry in payload["infer"]), 2)
            self.assertEqual(sum(label_from_entry(entry) == 1 for entry in payload["infer"]), 2)
            self.assertEqual(payload["split_summary"]["strategy"], "single_manifest_folder_label_holdout")

    def test_collect_and_plot_skips_external_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_run = tmp_path / "runs" / "feature_extract" / "runtime_tbh_rank256_features"
            cnn_feature_run = tmp_path / "runs" / "feature_extract" / "runtime_tbh_rank256_features_cnn"
            cnn_run = tmp_path / "runs" / "supervised" / "runtime_tbh_rank256_cnn_winner"
            paper_feature_run = tmp_path / "runs" / "feature_extract" / "runtime_tbh_rank256_paper_qv_spectral_features"
            paper_run = tmp_path / "runs" / "paper_qv_reference" / "runtime_tbh_rank256_paper_qv"
            output_dir = tmp_path / "runs" / "runtime_comparison" / "toxic_backdoors_hard_rank256"

            write_json(feature_run / "merged" / "spectral_merge_report.json", {"pipeline_elapsed_seconds": 100.0})
            write_json(paper_feature_run / "merged" / "spectral_merge_report.json", {"pipeline_elapsed_seconds": 40.0})
            write_json(
                cnn_feature_run / "merged" / "spectral_aggregation_report.json",
                {"aggregation_elapsed_seconds": 20.0},
            )
            write_json(cnn_run / "timings.json", {"supervised_training_method_seconds": 30.0})
            write_json(
                paper_run / "timings.json",
                {
                    "feature_derivation_seconds": 5.0,
                    "paper_qv_method_seconds": 10.0,
                    "total_seconds": 15.0,
                },
            )

            outputs = collect_and_plot(
                output_dir=output_dir,
                feature_run=feature_run,
                cnn_feature_run=cnn_feature_run,
                cnn_run=cnn_run,
                paper_qv_base_feature_run=paper_feature_run,
                paper_qv_run=paper_run,
            )

            for key in ["csv", "json", "png", "pdf", "manual_external"]:
                self.assertTrue(outputs[key].exists(), key)
            self.assertGreater(outputs["png"].stat().st_size, 0)
            self.assertGreater(outputs["pdf"].stat().st_size, 0)

            payload = read_json(outputs["json"])
            rows = {row["method"]: row for row in payload["rows"]}
            self.assertEqual(rows["Z-PEFT"]["feature_seconds"], 120.0)
            self.assertEqual(rows["Z-PEFT"]["training_seconds"], 30.0)
            self.assertEqual(rows["WSD-in-LoRA"]["feature_seconds"], 45.0)
            self.assertEqual(rows["WSD-in-LoRA"]["training_seconds"], 10.0)
            self.assertEqual(rows["External method"]["status"], "placeholder")

            csv_rows = {row["method"]: row for row in read_csv_rows(outputs["csv"])}
            self.assertEqual(csv_rows["External method"]["status"], "placeholder")

    def test_collect_and_plot_includes_complete_external_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_run = tmp_path / "feature"
            cnn_feature_run = tmp_path / "cnn_feature"
            cnn_run = tmp_path / "cnn"
            paper_feature_run = tmp_path / "paper_feature"
            paper_run = tmp_path / "paper"
            output_dir = tmp_path / "out"
            manual_path = tmp_path / "manual_external_method.json"

            write_json(feature_run / "merged" / "spectral_merge_report.json", {"pipeline_elapsed_seconds": 100.0})
            write_json(paper_feature_run / "merged" / "spectral_merge_report.json", {"pipeline_elapsed_seconds": 40.0})
            write_json(cnn_feature_run / "merged" / "spectral_aggregation_report.json", {"aggregation_elapsed_seconds": 20.0})
            write_json(cnn_run / "timings.json", {"supervised_training_method_seconds": 30.0})
            write_json(paper_run / "timings.json", {"feature_derivation_seconds": 5.0, "paper_qv_method_seconds": 10.0})
            write_json(
                manual_path,
                {
                    "method": "External method",
                    "feature_extraction_seconds": 7.0,
                    "training_seconds": 11.0,
                },
            )

            outputs = collect_and_plot(
                output_dir=output_dir,
                feature_run=feature_run,
                cnn_feature_run=cnn_feature_run,
                cnn_run=cnn_run,
                paper_qv_base_feature_run=paper_feature_run,
                paper_qv_run=paper_run,
                manual_external_file=manual_path,
            )

            payload = read_json(outputs["json"])
            rows = {row["method"]: row for row in payload["rows"]}
            self.assertEqual(rows["External method"]["status"], "ok")
            self.assertEqual(rows["External method"]["total_seconds"], 18.0)


if __name__ == "__main__":
    unittest.main()
