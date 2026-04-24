from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from upeftguard.experiments import attack_family_leave_one_out as GENERATOR_MODULE
from upeftguard.experiments import backdoor_detection_summaries as SUMMARY_MODULE
from upeftguard.experiments import supervised_cnn_suite as SUITE_MODULE
from upeftguard.utilities.core.manifest import (
    infer_attack_sample_identities,
    parse_joint_manifest_json_by_model_name,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    def test_attack_family_multiclass_leave_one_out_generator_filters_to_canonical_cohort(self):
        expected_names = {
            "holdout_attack_family_RIPPLE_rank256_qv.json",
            "holdout_attack_family_insertsent_rank256_qv.json",
            "holdout_attack_family_stybkd_rank256_qv.json",
            "holdout_attack_family_syntactic_rank256_qv.json",
        }
        allowed_positive_attacks = {"RIPPLE", "insertsent", "stybkd", "syntactic"}

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "generated_manifests"
            generated_paths = GENERATOR_MODULE.prepare_manifests(
                REPO_ROOT / "manifests" / "leave_one_out",
                output_root,
            )

            self.assertEqual({path.name for path in generated_paths}, expected_names)
            self.assertEqual(len(generated_paths), 4)

            for manifest_path in generated_paths:
                heldout_attack = manifest_path.stem.removeprefix("holdout_attack_family_").removesuffix(
                    "_rank256_qv"
                )
                payload = _load_json(manifest_path)
                self.assertEqual(len(payload["train"]), 8)
                self.assertEqual(len(payload["infer"]), 4)
                for section_name in ("train", "infer"):
                    self.assertGreater(len(payload[section_name]), 0)
                    for entry in payload[section_name]:
                        self.assertRegex(
                            entry["path"],
                            r"^llama2_7b_(ag_news|imdb)_(RIPPLE|insertsent|stybkd|syntactic)_rank256_qv/",
                            msg=f"{manifest_path}:{section_name}",
                        )

                train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_path)
                train_identities = infer_attack_sample_identities(train_items)
                infer_identities = infer_attack_sample_identities(infer_items)
                train_positive_attack_names = {
                    identity.attack_name
                    for item, identity in zip(train_items, train_identities)
                    if item.label == 1
                }
                infer_positive_attack_names = {
                    identity.attack_name
                    for item, identity in zip(infer_items, infer_identities)
                    if item.label == 1
                }
                self.assertEqual(train_positive_attack_names, allowed_positive_attacks - {heldout_attack})
                self.assertEqual(infer_positive_attack_names, {heldout_attack})
                self.assertEqual(sum(1 for item in train_items if item.label == 0), 250)
                self.assertEqual(sum(1 for item in infer_items if item.label == 0), 250)

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

    def test_supervised_cnn_suite_dry_run_enumerates_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            manifest_dir = repo_root / "manifests" / "zero_shots"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            _write_json(manifest_dir / "toy_transfer.json", {"train": [], "infer": []})

            reference_dir = repo_root / "runs" / "supervised" / "reference_cnn"
            reports_dir = reference_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            params = {
                "conv_channels": 64,
                "dropout": 0.1,
                "kernel_size": 5,
                "learning_rate": 0.0003,
                "num_conv_layers": 3,
                "weight_decay": 0.0,
            }
            _write_json(reference_dir / "run_config.json", {"pipeline": "supervised"})
            _write_json(
                reports_dir / "supervised_report.json",
                {
                    "tuning": {
                        "winner": {
                            "model_name": "cnn_1d",
                            "task_index": 3,
                            "params": params,
                        }
                    }
                },
            )
            _write_json(
                reports_dir / "tuning_manifest.json",
                {
                    "extractor": {
                        "params": {
                            "spectral_features": ["energy"],
                            "spectral_sv_top_k": 8,
                            "spectral_moment_source": "both",
                            "spectral_qv_sum_mode": "append",
                            "spectral_entrywise_delta_mode": "dense",
                        },
                        "metadata": {"external_feature_source": "toy_features"},
                    },
                    "threshold_selection": {},
                    "tuning": {"cv_folds_requested": 2, "cv_random_states": [42]},
                },
            )

            with mock.patch.object(SUITE_MODULE, "_repo_root", return_value=repo_root):
                rc = SUITE_MODULE.main(
                    [
                        "--suite",
                        "zero-shot",
                        "--hyperparam-config",
                        "reference_cnn",
                        "--dry-run",
                    ]
                )

            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
