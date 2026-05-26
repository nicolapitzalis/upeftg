from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from upeftguard.experiments.paper_qv_reference import (
    PAPER_QV_SELECTED_SUFFIXES,
    coefficients_to_normalized_weights,
    normalize_paper_features,
    run_paper_qv_reference_experiment,
    select_calibration_threshold,
    select_paper_qv_feature_indices,
)
from upeftguard.utilities.artifacts.dataset_references import (
    _finalize_payload,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from upeftguard.utilities.artifacts.spectral_metadata import load_spectral_metadata, write_spectral_metadata


def load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_public_and_internal_spectral_metadata(metadata_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json_file(metadata_path), load_spectral_metadata(metadata_path)


def write_merged_feature_artifacts(
    output_dir: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray | None,
    model_names: list[str],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "spectral_features.npy", np.asarray(features, dtype=np.float32))
    with open(output_dir / "spectral_model_names.json", "w", encoding="utf-8") as f:
        json.dump(list(model_names), f, indent=2)
    if labels is not None:
        np.save(output_dir / "spectral_labels.npy", np.asarray(labels, dtype=np.int32))
    write_spectral_metadata(output_dir / "spectral_metadata.json", internal_metadata=metadata)


def write_synthetic_source_bundle(tmp_path: Path) -> dict[str, Any]:
    feature_root = tmp_path / "runs" / "feature_extract"
    output_dir = feature_root / "synthetic_source" / "merged"

    feature_names = [
        "layer0.self_attn.q_proj.energy",
        "layer0.self_attn.v_proj.energy",
        "layer0.self_attn.qv_sum.kurtosis",
        "layer0.self_attn.qv_sum.l2_norm",
        "layer0.self_attn.qv_sum.concentration_of_energy",
        "layer0.self_attn.qv_sum.sv_1",
        "layer0.self_attn.qv_sum.spectral_entropy",
        "layer0.self_attn.qv_sum.sv_2",
        "layer1.self_attn.q_proj.energy",
        "layer1.self_attn.v_proj.energy",
        "layer1.self_attn.qv_sum.kurtosis",
        "layer1.self_attn.qv_sum.l2_norm",
        "layer1.self_attn.qv_sum.concentration_of_energy",
        "layer1.self_attn.qv_sum.sv_1",
        "layer1.self_attn.qv_sum.spectral_entropy",
        "layer1.self_attn.qv_sum.sv_2",
        "layer2.self_attn.q_proj.energy",
        "layer2.self_attn.v_proj.energy",
        "layer2.self_attn.qv_sum.kurtosis",
        "layer2.self_attn.qv_sum.l2_norm",
        "layer2.self_attn.qv_sum.concentration_of_energy",
        "layer2.self_attn.qv_sum.sv_1",
        "layer2.self_attn.qv_sum.spectral_entropy",
        "layer2.self_attn.qv_sum.sv_2",
    ]
    block_names = [
        "layer0.self_attn.q_proj",
        "layer0.self_attn.v_proj",
        "layer0.self_attn.qv_sum",
        "layer1.self_attn.q_proj",
        "layer1.self_attn.v_proj",
        "layer1.self_attn.qv_sum",
        "layer2.self_attn.q_proj",
        "layer2.self_attn.v_proj",
        "layer2.self_attn.qv_sum",
    ]
    lora_adapter_dims = [
        {"m": 4, "n": 4, "r": 2},
        {"m": 4, "n": 4, "r": 2},
        {"m": 4, "n": 4, "r": 4},
        {"m": 4, "n": 4, "r": 2},
        {"m": 4, "n": 4, "r": 2},
        {"m": 4, "n": 4, "r": 4},
        {"m": 4, "n": 4, "r": 2},
        {"m": 4, "n": 4, "r": 2},
        {"m": 4, "n": 4, "r": 4},
    ]
    metadata = {
        "extractor": "spectral",
        "extractor_name": "spectral",
        "extractor_version": "test",
        "delta_schema_version": "test",
        "feature_dim": len(feature_names),
        "feature_names": list(feature_names),
        "block_names": list(block_names),
        "base_block_names": [name for name in block_names if ".qv_sum" not in name],
        "qv_sum_block_names": [name for name in block_names if ".qv_sum" in name],
        "lora_adapter_dims": list(lora_adapter_dims),
        "qv_sum_lora_adapter_dims": [dict(entry) for name, entry in zip(block_names, lora_adapter_dims) if ".qv_sum" in name],
        "n_blocks": len(block_names),
        "resolved_features": [
            "energy",
            "kurtosis",
            "l2_norm",
            "concentration_of_energy",
            "sv_topk",
            "spectral_entropy",
        ],
        "sv_top_k": 2,
        "spectral_moment_source": "entrywise",
        "spectral_qv_sum_mode": "append",
        "spectral_entrywise_delta_mode": "dense",
        "extractor_params": {
            "spectral_features": [
                "energy",
                "kurtosis",
                "l2_norm",
                "concentration_of_energy",
                "sv_topk",
                "spectral_entropy",
            ],
            "spectral_sv_top_k": 2,
            "spectral_moment_source": "entrywise",
            "spectral_qv_sum_mode": "append",
            "spectral_entrywise_delta_mode": "dense",
        },
        "dataset_layouts": [
            {
                "dataset_name": "arch_a_dataset",
                "sample_count": 10,
                "layer_count": 2,
                "adapter_dims": {"m": 4, "n": 4, "r": 4},
            },
            {
                "dataset_name": "arch_b_dataset",
                "sample_count": 10,
                "layer_count": 3,
                "adapter_dims": {"m": 4, "n": 4, "r": 4},
            },
        ],
    }

    model_names: list[str] = []
    labels: list[int] = []
    rows: list[np.ndarray] = []
    model_index: dict[str, dict[str, Any]] = {}

    for dataset_name, model_family, layer1_present, layer2_present in [
        ("arch_a_dataset", "family_a", True, False),
        ("arch_b_dataset", "family_b", False, True),
    ]:
        for label in [0, 1]:
            for sample_idx in range(5):
                model_name = f"{dataset_name}_label{label}_{sample_idx}"
                model_names.append(model_name)
                labels.append(label)
                base = 0.10 + 0.01 * sample_idx
                score_shift = 0.65 if label == 1 else 0.0
                entropy = 2.5 - 0.05 * sample_idx - (1.0 if label == 1 else 0.0)

                row = np.zeros(len(feature_names), dtype=np.float32)
                row[0] = 0.3 + base
                row[1] = 0.2 + base
                row[2] = base + score_shift
                row[3] = 1.0 + base + score_shift
                row[4] = 0.2 + base + score_shift
                row[5] = 0.4 + base + score_shift
                row[6] = entropy
                row[7] = 0.05 + base

                row[8] = 0.25 + base if layer1_present else 0.0
                row[9] = 0.15 + base if layer1_present else 0.0
                row[10] = (base + score_shift + 0.05) if layer1_present else 0.0
                row[11] = (0.9 + base + score_shift) if layer1_present else 0.0
                row[12] = (0.25 + base + score_shift) if layer1_present else 0.0
                row[13] = (0.35 + base + score_shift) if layer1_present else 0.0
                row[14] = (entropy - 0.15) if layer1_present else 0.0
                row[15] = (0.04 + base) if layer1_present else 0.0

                row[16] = 0.22 + base if layer2_present else 0.0
                row[17] = 0.12 + base if layer2_present else 0.0
                row[18] = (base + score_shift + 0.08) if layer2_present else 0.0
                row[19] = (1.1 + base + score_shift) if layer2_present else 0.0
                row[20] = (0.18 + base + score_shift) if layer2_present else 0.0
                row[21] = (0.45 + base + score_shift) if layer2_present else 0.0
                row[22] = (entropy - 0.25) if layer2_present else 0.0
                row[23] = (0.03 + base) if layer2_present else 0.0
                rows.append(row)

                model_index[model_name] = {
                    "dataset_name": dataset_name,
                    "dataset_path": str((tmp_path / dataset_name).resolve()),
                    "label": label,
                    "subset_name": dataset_name,
                    "model_family": model_family,
                    "attack_name": "clean" if label == 0 else "paper_backdoor",
                }

    features = np.vstack(rows).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int32)
    write_merged_feature_artifacts(
        output_dir,
        features=features,
        labels=label_array,
        model_names=model_names,
        metadata=metadata,
    )
    dataset_reference_payload = _finalize_payload(
        artifact_kind="merge_feature_files",
        model_index=model_index,
        artifact_model_count=len(model_names),
        manifest_json=None,
        dataset_root=tmp_path,
        source_artifacts=[str(output_dir / "spectral_features.npy")],
        provenance_gaps=[],
        is_complete=True,
    )
    write_dataset_reference_report(
        default_dataset_reference_report_path(output_dir),
        dataset_reference_payload,
    )
    return {
        "feature_root": feature_root,
        "output_dir": output_dir,
        "feature_names": feature_names,
        "features": features,
        "labels": label_array,
        "model_names": model_names,
    }


class TestPaperQvReferenceExperiment(unittest.TestCase):
    def test_select_paper_qv_feature_indices_keeps_only_expected_columns(self):
        feature_names = [
            "layer0.self_attn.q_proj.energy",
            "layer0.self_attn.v_proj.energy",
            "layer0.self_attn.qv_sum.kurtosis",
            "layer0.self_attn.qv_sum.l2_norm",
            "layer0.self_attn.qv_sum.concentration_of_energy",
            "layer0.self_attn.qv_sum.sv_1",
            "layer0.self_attn.qv_sum.spectral_entropy",
            "layer0.self_attn.qv_sum.sv_2",
            "layer1.self_attn.qv_sum.kurtosis",
        ]
        selected = select_paper_qv_feature_indices(feature_names)
        selected_names = [feature_names[int(idx)] for idx in selected.tolist()]
        self.assertEqual(
            selected_names,
            [
                "layer0.self_attn.qv_sum.kurtosis",
                "layer0.self_attn.qv_sum.l2_norm",
                "layer0.self_attn.qv_sum.concentration_of_energy",
                "layer0.self_attn.qv_sum.sv_1",
                "layer0.self_attn.qv_sum.spectral_entropy",
                "layer1.self_attn.qv_sum.kurtosis",
            ],
        )
        self.assertFalse(any(".q_proj." in name or ".v_proj." in name for name in selected_names))
        self.assertTrue(all(".qv_sum." in name for name in selected_names))

    def test_normalize_paper_features_flips_entropy_and_uses_safe_std(self):
        raw = np.asarray(
            [
                [2.0, 4.0, 1.0],
                [3.0, 5.0, 3.0],
            ],
            dtype=np.float64,
        )
        benign_mean = np.asarray([1.0, 4.0, 2.0], dtype=np.float64)
        benign_std = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        entropy_mask = np.asarray([False, False, True], dtype=bool)
        normalized = normalize_paper_features(
            raw,
            benign_mean=benign_mean,
            benign_std=benign_std,
            entropy_mask=entropy_mask,
        )
        expected_z = np.asarray(
            [
                [1.0, 0.0, 1.0],
                [2.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        )
        expected = 0.5 * (1.0 + np.tanh(expected_z / 2.0))
        np.testing.assert_allclose(normalized, expected, rtol=1e-7, atol=1e-7)

    def test_coefficients_to_normalized_weights_zero_clips_and_normalizes(self):
        weights = coefficients_to_normalized_weights(np.asarray([-2.0, 1.0, 3.0], dtype=np.float64))
        np.testing.assert_allclose(weights, np.asarray([0.0, 0.25, 0.75], dtype=np.float64))

    def test_select_calibration_threshold_uses_perfect_separation_formula(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        scores = np.asarray([0.10, 0.20, 0.50, 0.70], dtype=np.float64)
        selected = select_calibration_threshold(labels_true=labels, scores=scores)
        self.assertEqual(selected["selection_method"], "perfect_separation_gap_25pct")
        self.assertAlmostEqual(float(selected["threshold"]), 0.275, places=6)

    def test_select_calibration_threshold_falls_back_to_youden(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        scores = np.asarray([0.10, 0.35, 0.50, 0.40, 0.55, 0.80], dtype=np.float64)
        selected = select_calibration_threshold(labels_true=labels, scores=scores)
        self.assertEqual(selected["selection_method"], "youden_j")
        self.assertAlmostEqual(float(selected["threshold"]), 0.40, places=6)

    def test_run_paper_qv_reference_experiment_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = write_synthetic_source_bundle(tmp_path)
            runs_root = tmp_path / "runs"

            outputs = run_paper_qv_reference_experiment(
                feature_file=Path("synthetic_source"),
                feature_root=source["feature_root"],
                feature_output_run="synthetic_paper_qv",
                output_root=runs_root,
                run_id="paper_qv_test_run",
                random_state=7,
                test_split_percent=20,
                calibration_split_percent=20,
            )

            derived_feature_path = outputs["derived_feature_path"]
            derived_features = np.asarray(np.load(derived_feature_path), dtype=np.float32)
            self.assertEqual(int(derived_features.shape[0]), int(source["features"].shape[0]))
            self.assertEqual(int(derived_features.shape[1]), 15)

            with open(derived_feature_path.with_name("spectral_model_names.json"), "r", encoding="utf-8") as f:
                derived_model_names = [str(x) for x in json.load(f)]
            self.assertEqual(derived_model_names, source["model_names"])
            derived_labels = np.asarray(np.load(derived_feature_path.with_name("spectral_labels.npy")), dtype=np.int32)
            np.testing.assert_array_equal(derived_labels, source["labels"])

            public_metadata, internal_metadata = load_public_and_internal_spectral_metadata(
                derived_feature_path.with_name("spectral_metadata.json")
            )
            derived_feature_names = [str(x) for x in internal_metadata["feature_names"]]
            self.assertEqual(len(derived_feature_names), 15)
            self.assertEqual(
                [str(x) for x in internal_metadata["block_names"]],
                [
                    "layer0.self_attn.qv_sum",
                    "layer1.self_attn.qv_sum",
                    "layer2.self_attn.qv_sum",
                ],
            )
            self.assertEqual(public_metadata["spectral_qv_sum_mode"], "only")
            self.assertNotIn("feature_names", public_metadata)
            self.assertTrue(all(".qv_sum." in name for name in derived_feature_names))
            self.assertTrue(
                all(name.rpartition(".")[2] in set(PAPER_QV_SELECTED_SUFFIXES) for name in derived_feature_names)
            )
            self.assertFalse(any(".q_proj." in name or ".v_proj." in name for name in derived_feature_names))

            feature_index = {name: idx for idx, name in enumerate(derived_feature_names)}
            layer2_cols = np.asarray(
                [
                    feature_index["layer2.self_attn.qv_sum.kurtosis"],
                    feature_index["layer2.self_attn.qv_sum.l2_norm"],
                    feature_index["layer2.self_attn.qv_sum.concentration_of_energy"],
                    feature_index["layer2.self_attn.qv_sum.sv_1"],
                    feature_index["layer2.self_attn.qv_sum.spectral_entropy"],
                ],
                dtype=np.int64,
            )
            layer1_cols = np.asarray(
                [
                    feature_index["layer1.self_attn.qv_sum.kurtosis"],
                    feature_index["layer1.self_attn.qv_sum.l2_norm"],
                    feature_index["layer1.self_attn.qv_sum.concentration_of_energy"],
                    feature_index["layer1.self_attn.qv_sum.sv_1"],
                    feature_index["layer1.self_attn.qv_sum.spectral_entropy"],
                ],
                dtype=np.int64,
            )
            np.testing.assert_allclose(derived_features[:10][:, layer2_cols], 0.0, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(derived_features[10:][:, layer1_cols], 0.0, rtol=0.0, atol=0.0)

            detector_payload = joblib.load(outputs["detector_model_path"])
            self.assertEqual(len(detector_payload["feature_names"]), 15)
            self.assertAlmostEqual(float(np.sum(detector_payload["normalized_weights"])), 1.0, places=8)
            self.assertTrue(np.isfinite(float(detector_payload["threshold"])))

            selected_threshold = load_json_file(outputs["selected_threshold_path"])
            metrics = load_json_file(outputs["metrics_path"])
            split_manifest = load_json_file(outputs["split_manifest_path"])
            reference_summary = load_json_file(outputs["reference_bank_summary_path"])
            self.assertIn("threshold", selected_threshold)
            self.assertIn("test", metrics)
            self.assertEqual(metrics["feature_dim"], 15)
            self.assertEqual(split_manifest["test_stratify_kind"], "dataset_name_x_label")
            self.assertEqual(split_manifest["calibration_stratify_kind"], "dataset_name_x_label")
            expected_reference_indices = [
                int(idx) for idx in split_manifest["fit_indices"] if int(source["labels"][int(idx)]) == 0
            ]
            self.assertEqual(detector_payload["reference_bank_indices"], expected_reference_indices)
            self.assertEqual(reference_summary["source_partition"], "fit")
            self.assertEqual(reference_summary["reference_bank_indices"], expected_reference_indices)
            self.assertEqual(reference_summary["n_reference_clean_samples"], len(expected_reference_indices))
            self.assertLess(reference_summary["n_reference_clean_samples"], 10)

            for csv_path in [outputs["calibration_scores_csv"], outputs["test_scores_csv"]]:
                self.assertTrue(Path(csv_path).exists())

            artifact_index = load_json_file(outputs["run_dir"] / "artifact_index.json")
            self.assertEqual(artifact_index["derived_feature_file"], str(derived_feature_path))
            self.assertEqual(artifact_index["detector_model"], str(outputs["detector_model_path"]))

    def test_run_paper_qv_reference_experiment_uses_joint_manifest_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = write_synthetic_source_bundle(tmp_path)
            runs_root = tmp_path / "runs"

            train_indices = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]
            infer_indices = [4, 9, 14, 19]
            manifest_path = tmp_path / "holdout_synthetic_attack.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train": [source["model_names"][idx] for idx in train_indices],
                        "infer": [source["model_names"][idx] for idx in infer_indices],
                    },
                    f,
                    indent=2,
                )

            outputs = run_paper_qv_reference_experiment(
                feature_file=Path("synthetic_source"),
                feature_root=source["feature_root"],
                feature_output_run="synthetic_paper_qv_manifest",
                output_root=runs_root,
                run_id="paper_qv_manifest_run",
                manifest_json=manifest_path,
                random_state=11,
                calibration_split_percent=25,
            )

            split_manifest = load_json_file(outputs["split_manifest_path"])
            reference_summary = load_json_file(outputs["reference_bank_summary_path"])
            metrics = load_json_file(outputs["metrics_path"])
            detector_payload = joblib.load(outputs["detector_model_path"])

            self.assertEqual(split_manifest["split_strategy"], "manifest_defined")
            self.assertEqual(split_manifest["manifest_json"], str(manifest_path.resolve()))
            self.assertEqual(split_manifest["train_indices"], train_indices)
            self.assertEqual(split_manifest["test_indices"], infer_indices)
            self.assertEqual(split_manifest["test_stratify_kind"], "manifest_defined")
            self.assertEqual(split_manifest["calibration_stratify_kind"], "dataset_name_x_label")
            self.assertTrue(set(split_manifest["fit_indices"]).issubset(set(train_indices)))
            self.assertTrue(set(split_manifest["calibration_indices"]).issubset(set(train_indices)))
            self.assertFalse(set(split_manifest["test_indices"]) & set(split_manifest["fit_indices"]))
            self.assertFalse(set(split_manifest["test_indices"]) & set(split_manifest["calibration_indices"]))

            expected_reference_indices = [
                int(idx) for idx in split_manifest["fit_indices"] if int(source["labels"][int(idx)]) == 0
            ]
            self.assertEqual(detector_payload["reference_bank_indices"], expected_reference_indices)
            self.assertEqual(reference_summary["source_partition"], "fit")
            self.assertEqual(reference_summary["reference_bank_indices"], expected_reference_indices)
            self.assertEqual(reference_summary["n_reference_clean_samples"], len(expected_reference_indices))
            self.assertEqual(metrics["split_strategy"], "manifest_defined")
            self.assertEqual(metrics["test"]["n_samples"], len(infer_indices))


if __name__ == "__main__":
    unittest.main()
