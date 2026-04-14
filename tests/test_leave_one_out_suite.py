from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from upeftguard.utilities.core.manifest import parse_joint_manifest_json_by_model_name


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_backdoor_detection_summaries.py"
SUMMARY_SPEC = importlib.util.spec_from_file_location("generate_backdoor_detection_summaries", SUMMARY_SCRIPT_PATH)
if SUMMARY_SPEC is None or SUMMARY_SPEC.loader is None:
    raise RuntimeError(f"Could not load summary script from {SUMMARY_SCRIPT_PATH}")
SUMMARY_MODULE = importlib.util.module_from_spec(SUMMARY_SPEC)
sys.modules[SUMMARY_SPEC.name] = SUMMARY_MODULE
SUMMARY_SPEC.loader.exec_module(SUMMARY_MODULE)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestLeaveOneOutSuite(unittest.TestCase):
    def test_committed_leave_one_out_manifests_exist_and_parse(self):
        manifest_dir = REPO_ROOT / "manifests" / "leave_one_out"
        manifest_paths = sorted(manifest_dir.glob("holdout_*.json"))

        self.assertEqual(len(manifest_paths), 27)

        held_out_names = {path.stem.removeprefix("holdout_") for path in manifest_paths}
        self.assertNotIn("chatglm6b_toxic_backdoors_hard_rank256_qv", held_out_names)
        self.assertNotIn("llama3_8b_toxic_backdoors_hard_rank256_qv", held_out_names)
        self.assertNotIn("qwen2_vl_vqav2_insertsent_rank16_qv", held_out_names)

        for manifest_path in manifest_paths:
            train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_path)
            self.assertGreater(len(train_items), 0, msg=str(manifest_path))
            self.assertGreater(len(infer_items), 0, msg=str(manifest_path))

    def test_backdoor_only_holdout_uses_shared_clean_pool(self):
        manifest_path = (
            REPO_ROOT
            / "manifests"
            / "leave_one_out"
            / "holdout_llama2_7b_ag_news_RIPPLE_rank256_qv.json"
        )
        payload = _load_json(manifest_path)

        infer_entries = {entry["path"]: entry["indices"] for entry in payload["infer"]}
        train_entries = {entry["path"]: entry["indices"] for entry in payload["train"]}

        self.assertEqual(
            infer_entries["llama2_7b_ag_news_RIPPLE_rank256_qv/llama2_7b_ag_news_RIPPLE_rank256_qv_label1_"],
            [0, 249],
        )
        self.assertEqual(
            infer_entries["llama2_7b_imdb_insertsent_rank256_qv/llama2_7b_imdb_insertsent_rank256_qv_label0_"],
            [125, 249],
        )
        self.assertEqual(
            train_entries["llama2_7b_imdb_insertsent_rank256_qv/llama2_7b_imdb_insertsent_rank256_qv_label0_"],
            [0, 124],
        )
        self.assertEqual(len(payload["infer"]), 22)

    def test_clean_holdout_keeps_inference_to_held_out_dataset(self):
        manifest_path = (
            REPO_ROOT
            / "manifests"
            / "leave_one_out"
            / "holdout_roberta_base_imdb_insertsent_rank16_qv.json"
        )
        payload = _load_json(manifest_path)

        self.assertEqual(len(payload["infer"]), 2)
        self.assertTrue(
            all(entry["path"].startswith("roberta_base_imdb_insertsent_rank16_qv/") for entry in payload["infer"])
        )
        self.assertTrue(
            all(
                not entry["path"].startswith("roberta_base_imdb_insertsent_rank16_qv/")
                for entry in payload["train"]
            )
        )

    def test_discover_leave_one_out_run_specs_matches_committed_manifests(self):
        run_specs = SUMMARY_MODULE.discover_leave_one_out_run_specs(REPO_ROOT)

        self.assertEqual(len(run_specs), 27)
        self.assertEqual(run_specs[0].dataset, SUMMARY_MODULE.LEAVE_ONE_OUT_DATASET_LABEL)
        self.assertEqual(run_specs[0].attack_type, "flan_t5_xl_toxic_backdoors_hard_rank256_qv")
        self.assertEqual(run_specs[0].run_id, "holdout_flan_t5_xl_toxic_backdoors_hard_rank256_qv")
        self.assertEqual(run_specs[-1].attack_type, "roberta_base_imdb_insertsent_rank16_qv")

    def test_generate_suite_csvs_for_leave_one_out_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            manifest_dir = repo_root / "manifests" / "leave_one_out"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            _write_json(manifest_dir / "holdout_alpha.json", {"train": [], "infer": []})
            _write_json(manifest_dir / "holdout_beta.json", {"train": [], "infer": []})

            run_specs = SUMMARY_MODULE.discover_leave_one_out_run_specs(repo_root)
            self.assertEqual([spec.attack_type for spec in run_specs], ["alpha", "beta"])

            params = {
                "conv_channels": 64,
                "dropout": 0.1,
                "kernel_size": 5,
                "learning_rate": 0.0003,
                "num_conv_layers": 3,
                "weight_decay": 0.0,
            }

            for idx, spec in enumerate(run_specs):
                run_dir = repo_root / "runs" / "supervised" / "leave_one_out_cnn" / spec.run_id
                reports_dir = run_dir / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)

                _write_json(
                    reports_dir / "supervised_report.json",
                    {
                        "fit_assessment": {"offline_metrics": {"auroc": 0.91 + (0.01 * idx)}},
                        "tuning": {
                            "candidates": [
                                {
                                    "model_name": "cnn_1d",
                                    "normalization_policy": "masked_train_only",
                                    "params": params,
                                    "roc_auc_mean": 0.91 + (0.01 * idx),
                                    "status": "ok",
                                }
                            ]
                        },
                    },
                )
                _write_json(
                    reports_dir / "selected_threshold.json",
                    {
                        "method": "maximize_recall_subject_to_fpr",
                        "source_partition": "calibration",
                        "accepted_fprs": [0.05],
                        "selections": [
                            {
                                "accepted_fpr": 0.05,
                                "threshold": 0.77 + (0.01 * idx),
                                "inference_metrics": {
                                    "recall": 0.70 + (0.05 * idx),
                                    "specificity": 0.90 - (0.05 * idx),
                                },
                            }
                        ],
                    },
                )
                _write_json(
                    reports_dir / "tuning_manifest.json",
                    {"runtime": {}, "tuning": {}, "threshold_selection": {"accepted_fprs": [0.05]}},
                )
                _write_json(
                    run_dir / "run_config.json",
                    {
                        "winner": {
                            "model_name": "cnn_1d",
                            "normalization_policy": "masked_train_only",
                            "params": params,
                        }
                    },
                )

            suite = SUMMARY_MODULE.SummarySuite(
                key="leave_one_out_cnn_test",
                description="synthetic leave-one-out test suite",
                run_specs=tuple(run_specs),
                summary_filename_template="backdoor_detection_leave_one_out_cnn_summary_fpr_{fpr_slug}.csv",
                winner_summary_filename_template="backdoor_detection_leave_one_out_cnn_winner_summary_fpr_{fpr_slug}.csv",
                universal_filename="universal_config_leave_one_out_cnn_ranking_cv.csv",
                run_subdir=("leave_one_out_cnn",),
                summary_subdir=("leave_one_out_cnn", "summaries"),
            )

            output_root = repo_root / "runs" / "supervised"
            SUMMARY_MODULE.generate_suite_csvs(
                repo_root=repo_root,
                output_root=output_root,
                suite=suite,
                accepted_fprs=[0.05],
            )

            summary_path = (
                output_root
                / "leave_one_out_cnn"
                / "summaries"
                / "backdoor_detection_leave_one_out_cnn_summary_fpr_0_05.csv"
            )
            winner_path = (
                output_root
                / "leave_one_out_cnn"
                / "summaries"
                / "backdoor_detection_leave_one_out_cnn_winner_summary_fpr_0_05.csv"
            )
            universal_path = (
                output_root
                / "leave_one_out_cnn"
                / "summaries"
                / "universal_config_leave_one_out_cnn_ranking_cv.csv"
            )

            self.assertTrue(summary_path.exists())
            self.assertTrue(winner_path.exists())
            self.assertTrue(universal_path.exists())

            summary_rows = _load_csv_rows(summary_path)
            self.assertEqual([row["type of attack"] for row in summary_rows], ["alpha", "beta"])
            self.assertEqual(
                summary_rows[0]["dataset"],
                SUMMARY_MODULE.LEAVE_ONE_OUT_DATASET_LABEL,
            )

            winner_rows = _load_csv_rows(winner_path)
            self.assertEqual(len(winner_rows), 2)
            self.assertEqual(winner_rows[1]["type of attack"], "beta")

            universal_rows = _load_csv_rows(universal_path)
            self.assertEqual(len(universal_rows), 1)
            self.assertEqual(universal_rows[0]["model"], "cnn_1d")


if __name__ == "__main__":
    unittest.main()
