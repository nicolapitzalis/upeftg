from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from upeftguard.experiments.supervised_architecture_breakdown import (
    build_architecture_analysis,
    build_supervised_results_summary,
    run_supervised_results_summary,
)
from upeftguard.supervised.interfaces import SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS
from upeftguard.utilities.core.manifest import AttackSampleIdentity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_score_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "model_name", "label", "score", "score_percentile_rank"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_synthetic_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "supervised" / "synthetic_cnn"
    reports_dir = run_dir / "reports"
    manifest_path = tmp_path / "manifests" / "synthetic_list.json"

    manifest_payload = {
        "path": [
            {"path": "llama2_7b_imdb_syntactic_rank256_qv/llama2_7b_imdb_syntactic_rank256_qv_label0_", "indices": [0, 1]},
            {"path": "llama2_7b_imdb_syntactic_rank256_qv/llama2_7b_imdb_syntactic_rank256_qv_label1_", "indices": [0, 1]},
            {"path": "roberta_base_imdb_insertsent_rank16_qv/roberta_base_imdb_insertsent_rank16_qv_label0_", "indices": [0, 1]},
            {"path": "roberta_base_imdb_insertsent_rank16_qv/roberta_base_imdb_insertsent_rank16_qv_label1_", "indices": [0, 1]},
        ]
    }
    _write_json(manifest_path, manifest_payload)

    tuning_manifest = {
        "script_version": "test",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "run_dir": str(run_dir),
        "mode": "single",
        "manifest_json": str(manifest_path),
        "dataset_root": str(tmp_path / "data"),
        "data": {
            "feature_loading_mode": "external_source",
        },
    }
    _write_json(reports_dir / "tuning_manifest.json", tuning_manifest)

    report = {
        "fit_assessment": {
            "threshold_evaluation": [
                {"percentile_from_train": 90.0, "threshold": 0.60},
                {"percentile_from_train": 95.0, "threshold": 0.80},
            ]
        },
        "attack_analysis": {
            "inference": {
                "grouping_rule": "per-attack evaluation against the shared clean pool.",
                "clean_pool": "shared",
                "n_attacks": 2,
                "attacks": {
                    "syntactic": {
                        "n_samples": 2,
                        "source_subsets": ["llama2_7b_imdb_syntactic_rank256_qv"],
                        "label_counts": {"clean": 1, "backdoored": 1, "unknown": 0},
                        "selected_threshold_evaluation": [
                            {
                                "accepted_fpr": 0.05,
                                "threshold": 0.65,
                                "recall": 1.0,
                                "precision": 1.0,
                                "false_positive_rate": 0.0,
                            }
                        ],
                    },
                    "insertsent": {
                        "n_samples": 2,
                        "source_subsets": ["roberta_base_imdb_insertsent_rank16_qv"],
                        "label_counts": {"clean": 1, "backdoored": 1, "unknown": 0},
                        "selected_threshold_evaluation": [
                            {
                                "accepted_fpr": 0.05,
                                "threshold": 0.65,
                                "recall": 1.0,
                                "precision": 0.5,
                                "false_positive_rate": 1.0,
                            }
                        ],
                    },
                },
            }
        },
        "tuning": {
            "model_name": "cnn_1d",
            "winner": {"model_name": "cnn_1d"},
        },
    }
    _write_json(reports_dir / "supervised_report.json", report)

    selected_threshold = {
        "method": "maximize_recall_subject_to_fpr",
        "source_partition": "calibration",
        "accepted_fprs": [0.05],
        "selections": [
            {
                "accepted_fpr": 0.05,
                "threshold": 0.65,
            }
        ],
    }
    _write_json(reports_dir / "selected_threshold.json", selected_threshold)

    _write_score_csv(
        reports_dir / "train_scores.csv",
        [
            {"index": 0, "model_name": "llama2_7b_imdb_syntactic_rank256_qv_label0_0", "label": 0, "score": 0.10, "score_percentile_rank": 0.00},
            {"index": 1, "model_name": "llama2_7b_imdb_syntactic_rank256_qv_label1_0", "label": 1, "score": 0.90, "score_percentile_rank": 1.00},
            {"index": 2, "model_name": "roberta_base_imdb_insertsent_rank16_qv_label0_0", "label": 0, "score": 0.70, "score_percentile_rank": 0.66},
            {"index": 3, "model_name": "roberta_base_imdb_insertsent_rank16_qv_label1_0", "label": 1, "score": 0.80, "score_percentile_rank": 0.90},
        ],
    )
    _write_score_csv(
        reports_dir / "calibration_scores.csv",
        [
            {"index": 0, "model_name": "llama2_7b_imdb_syntactic_rank256_qv_label0_1", "label": 0, "score": 0.20, "score_percentile_rank": 0.00},
            {"index": 1, "model_name": "llama2_7b_imdb_syntactic_rank256_qv_label1_1", "label": 1, "score": 0.85, "score_percentile_rank": 1.00},
            {"index": 2, "model_name": "roberta_base_imdb_insertsent_rank16_qv_label0_1", "label": 0, "score": 0.75, "score_percentile_rank": 0.66},
            {"index": 3, "model_name": "roberta_base_imdb_insertsent_rank16_qv_label1_1", "label": 1, "score": 0.78, "score_percentile_rank": 0.90},
        ],
    )
    _write_score_csv(
        reports_dir / "inference_scores.csv",
        [
            {"index": 0, "model_name": "llama2_7b_imdb_syntactic_rank256_qv_label0_0", "label": 0, "score": 0.15, "score_percentile_rank": 0.00},
            {"index": 1, "model_name": "llama2_7b_imdb_syntactic_rank256_qv_label1_1", "label": 1, "score": 0.88, "score_percentile_rank": 1.00},
            {"index": 2, "model_name": "roberta_base_imdb_insertsent_rank16_qv_label0_0", "label": 0, "score": 0.74, "score_percentile_rank": 0.66},
            {"index": 3, "model_name": "roberta_base_imdb_insertsent_rank16_qv_label1_1", "label": 1, "score": 0.77, "score_percentile_rank": 0.90},
        ],
    )

    _write_json(run_dir / "artifact_index.json", {"report": str(reports_dir / "supervised_report.json")})
    return run_dir


def _build_synthetic_multiclass_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "supervised" / "synthetic_multiclass_cnn"
    reports_dir = run_dir / "reports"

    report = {
        "task": {
            "task_mode": SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
            "class_names": ["clean", "RIPPLE", "insertsent", "stybkd", "syntactic"],
            "binary_projection": "one_minus_clean_probability",
        },
        "data_info": {
            "n_train": 10,
            "n_train_clean": 2,
            "n_train_backdoored": 8,
            "n_calibration": 0,
            "n_calibration_clean": 0,
            "n_calibration_backdoored": 0,
            "n_inference": 5,
            "n_inference_clean": 1,
            "n_inference_backdoored": 4,
            "n_inference_unknown_label": 0,
        },
        "fit_assessment": {
            "score_definition": "backdoor_score",
            "binary_projection": "one_minus_clean_probability",
            "train_score_summary": {"mean": 0.75},
            "train_offline_metrics": {
                "auroc": 0.990,
                "auprc": 0.995,
                "precision_at_num_positives": 0.980,
            },
            "calibration_score_summary": None,
            "calibration_offline_metrics": None,
            "inference_score_summary": {"mean": 0.70},
            "offline_metrics": {
                "auroc": 0.950,
                "auprc": 0.970,
                "precision_at_num_positives": 0.900,
            },
            "threshold_evaluation": [
                {
                    "percentile_from_train": 90.0,
                    "threshold": 0.80,
                    "recall": 0.50,
                    "precision": 1.0,
                    "false_positive_rate": 0.0,
                    "fraction_flagged": 0.20,
                }
            ],
        },
        "multiclass_assessment": {
            "train": {
                "accuracy": 0.960,
                "macro_f1": 0.955,
                "micro_f1": 0.960,
                "per_class": [
                    {
                        "class_name": "clean",
                        "support": 2,
                        "predicted_count": 2,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                    {
                        "class_name": "syntactic",
                        "support": 2,
                        "predicted_count": 2,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                ],
            },
            "inference": {
                "accuracy": 0.800,
                "macro_f1": 0.790,
                "micro_f1": 0.800,
                "per_class": [
                    {
                        "class_name": "clean",
                        "support": 1,
                        "predicted_count": 2,
                        "precision": 0.500,
                        "recall": 1.0,
                        "f1": 0.667,
                    },
                    {
                        "class_name": "RIPPLE",
                        "support": 1,
                        "predicted_count": 1,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                    {
                        "class_name": "syntactic",
                        "support": 1,
                        "predicted_count": 1,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                ],
            },
        },
        "attack_analysis": {
            "inference": {
                "attacks": {
                    "syntactic": {
                        "offline_metrics": {"auroc": 1.0},
                    }
                }
            }
        },
        "tuning": {
            "model_name": "cnn_1d",
            "winner": {"model_name": "cnn_1d"},
        },
    }
    _write_json(reports_dir / "supervised_report.json", report)
    _write_json(run_dir / "artifact_index.json", {"report": str(reports_dir / "supervised_report.json")})
    return run_dir


class TestSupervisedArchitectureBreakdown(unittest.TestCase):
    def test_adapter_group_name_defaults_to_lora_and_detects_variants(self):
        from upeftguard.experiments.supervised_architecture_breakdown import _adapter_group_name

        self.assertEqual(
            _adapter_group_name(
                AttackSampleIdentity(
                    model_name="dummy",
                    subset_name="llama2_7b_toxic_backdoors_hard_rank256_qv",
                    model_family="llama2_7b",
                    attack_name="toxic_backdoors_hard",
                    subset_has_clean=True,
                    subset_has_backdoor=True,
                    attack_name_source="folder_name",
                )
            ),
            "lora",
        )
        self.assertEqual(
            _adapter_group_name(
                AttackSampleIdentity(
                    model_name="dummy",
                    subset_name="llama2_7b_lora_plus_toxic_backdoors_hard_rank8_qv",
                    model_family="llama2_7b",
                    attack_name="toxic_backdoors_hard",
                    subset_has_clean=True,
                    subset_has_backdoor=True,
                    attack_name_source="folder_name",
                )
            ),
            "lora+",
        )
        self.assertEqual(
            _adapter_group_name(
                AttackSampleIdentity(
                    model_name="dummy",
                    subset_name="llama2_7b_tbh_zero_shot_lora_to_adalora",
                    model_family="llama2_7b",
                    attack_name="tbh_zero_shot_lora_to_adalora",
                    subset_has_clean=True,
                    subset_has_backdoor=True,
                    attack_name_source="folder_name",
                )
            ),
            "adalora",
        )

    def test_dataset_group_name_strips_variant_prefixes(self):
        from upeftguard.experiments.supervised_architecture_breakdown import _dataset_group_name

        self.assertEqual(
            _dataset_group_name(
                AttackSampleIdentity(
                    model_name="dummy",
                    subset_name="llama2_7b_adalora_toxic_backdoors_hard",
                    model_family="llama2_7b",
                    attack_name="toxic_backdoors_hard",
                    subset_has_clean=True,
                    subset_has_backdoor=True,
                    attack_name_source="folder_name",
                )
            ),
            "toxic_backdoors_hard",
        )
        self.assertEqual(
            _dataset_group_name(
                AttackSampleIdentity(
                    model_name="dummy",
                    subset_name="flan_t5_xl_toxic_backdoors_hard",
                    model_family="flan_t5",
                    attack_name="toxic_backdoors_hard",
                    subset_has_clean=True,
                    subset_has_backdoor=True,
                    attack_name_source="folder_name",
                )
            ),
            "toxic_backdoors_hard",
        )

    def test_build_architecture_analysis_groups_within_model_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_synthetic_run(Path(tmp))
            analysis = build_architecture_analysis(run_dir=run_dir)

            inference = analysis["inference"]["architectures"]
            self.assertCountEqual(list(inference.keys()), ["llama2_7b", "roberta_base"])
            self.assertEqual(inference["llama2_7b"]["n_samples"], 2)
            self.assertEqual(inference["llama2_7b"]["label_counts"]["clean"], 1)
            self.assertEqual(inference["llama2_7b"]["label_counts"]["backdoored"], 1)
            self.assertEqual(
                inference["llama2_7b"]["source_subsets"],
                ["llama2_7b_imdb_syntactic_rank256_qv"],
            )
            self.assertEqual(inference["roberta_base"]["n_samples"], 2)
            self.assertEqual(
                inference["roberta_base"]["source_subsets"],
                ["roberta_base_imdb_insertsent_rank16_qv"],
            )

    def test_build_supervised_results_summary_includes_dataset_attack_and_architecture_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_synthetic_run(Path(tmp))
            summary = build_supervised_results_summary(run_dir=run_dir)

            inference = summary["inference_overview"]
            self.assertEqual(inference["n_samples"], 4)
            self.assertEqual(inference["label_counts"]["clean"], 2)
            self.assertEqual(inference["label_counts"]["backdoored"], 2)
            self.assertIn("imdb", summary["dataset_analysis"]["datasets"])
            self.assertIn("lora", summary["adapter_analysis"]["adapters"])
            self.assertIn("llama2_7b", summary["architecture_analysis"]["architectures"])
            self.assertIn("syntactic", summary["attack_analysis"]["attacks"])
            self.assertEqual(summary["attack_analysis"]["clean_pool"]["n_samples"], 2)
            self.assertEqual(summary["attack_analysis"]["attacks"]["syntactic"]["n_samples"], 3)
            self.assertEqual(
                summary["attack_analysis"]["attacks"]["syntactic"]["offline_metrics"]["auroc"],
                1.0,
            )
            self.assertEqual(
                summary["attack_analysis"]["attacks"]["syntactic"]["offline_metrics"]["precision_at_num_positives"],
                1.0,
            )

    def test_run_supervised_results_summary_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _build_synthetic_run(tmp_path)
            reports_dir = run_dir / "reports"
            (reports_dir / "architecture_analysis.md").write_text("legacy", encoding="utf-8")
            (reports_dir / "architecture_analysis.json").write_text("{}", encoding="utf-8")

            outputs = run_supervised_results_summary(
                run_spec=run_dir,
                update_report=True,
            )

            md_path = outputs["results_summary_md"]
            self.assertTrue(md_path.exists())
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("# Results Summary", markdown)
            self.assertIn("## Entire Inference Set", markdown)
            self.assertIn("## Per Architecture", markdown)
            self.assertIn("## Per Dataset", markdown)
            self.assertIn("## Per Adapter", markdown)
            self.assertIn("## Per Attack", markdown)
            self.assertIn("| syntactic | 3 | 2 | 1 | 0 | 1.000 | 1.000 | 1.000 |", markdown)

            with open(run_dir / "artifact_index.json", "r", encoding="utf-8") as f:
                artifact_index = json.load(f)
            self.assertIn("results_summary_md", artifact_index)
            self.assertNotIn("architecture_analysis_json", artifact_index)
            self.assertNotIn("architecture_analysis_md", artifact_index)

            self.assertFalse((reports_dir / "architecture_analysis.md").exists())
            self.assertFalse((reports_dir / "architecture_analysis.json").exists())

            with open(run_dir / "reports" / "supervised_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertIn("results_summary", report)
            self.assertIn("dataset_analysis", report)
            self.assertIn("adapter_analysis", report)
            self.assertIn("attack_analysis", report)
            self.assertIn("architecture_analysis", report)
            self.assertEqual(
                report["attack_analysis"]["inference"]["attacks"]["syntactic"]["offline_metrics"]["auroc"],
                1.0,
            )

    def test_build_supervised_results_summary_uses_multiclass_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_synthetic_multiclass_run(Path(tmp))
            summary = build_supervised_results_summary(run_dir=run_dir)

            self.assertEqual(summary["summary_mode"], "multiclass")
            self.assertEqual(summary["task_mode"], SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS)
            self.assertIn("binary_classification", summary)
            self.assertIn("class_results", summary)
            self.assertNotIn("dataset_analysis", summary)
            self.assertEqual(
                summary["binary_classification"]["partitions"]["inference"]["offline_metrics"]["auroc"],
                0.95,
            )
            self.assertEqual(
                summary["class_results"]["partitions"]["inference"]["macro_f1"],
                0.79,
            )

    def test_run_supervised_results_summary_writes_multiclass_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_synthetic_multiclass_run(Path(tmp))

            outputs = run_supervised_results_summary(
                run_spec=run_dir,
                update_report=True,
            )

            markdown = outputs["results_summary_md"].read_text(encoding="utf-8")
            self.assertIn("# Results Summary", markdown)
            self.assertIn("## Binary Classification", markdown)
            self.assertIn("## Multiclass Overview", markdown)
            self.assertIn("## Per-Class Inference", markdown)
            self.assertNotIn("## Per Architecture", markdown)
            self.assertNotIn("## Per Dataset", markdown)
            self.assertNotIn("## Per Adapter", markdown)
            self.assertNotIn("## Per Attack", markdown)
            self.assertIn("| inference | 5 | 1 | 4 | 0 | 0.950 | 0.970 | 0.900 |", markdown)
            self.assertIn("| clean | 1 | 2 | 0.500 | 1.000 | 0.667 |", markdown)

            with open(run_dir / "reports" / "supervised_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["results_summary"]["summary_mode"], "multiclass")
            self.assertIn("binary_classification", report["results_summary"])
            self.assertIn("class_results", report["results_summary"])
            self.assertIn("attack_analysis", report)
            self.assertEqual(
                report["attack_analysis"]["inference"]["attacks"]["syntactic"]["offline_metrics"]["auroc"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
