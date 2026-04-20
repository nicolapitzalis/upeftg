from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from upeftguard.experiments.rank_normalization_study import (
    StudyArmSpec,
    StudyConfig,
    aggregate_arm_feature_bundle,
    args_to_config,
    block_rank_scale_for_sample,
    build_arm_feature_groups,
    build_parser,
    collect_manifest_model_names,
    collect_zero_shot_expectations,
    derive_arm_feature_bundle,
    launch_zero_shot_suite,
    parse_fixed_rank_from_model_name,
    run_rank_normalization_study,
    zero_shot_target_rank_from_name,
)
from upeftguard.features.spectral import build_spectral_feature_names
from upeftguard.unsupervised.analysis import load_feature_bundle
from upeftguard.utilities.artifacts.dataset_references import (
    _finalize_payload,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from upeftguard.utilities.artifacts.spectral_metadata import load_spectral_metadata, write_spectral_metadata


BASELINE_SELECTED_FEATURES = [
    "energy",
    "kurtosis",
    "l1_norm",
    "l2_norm",
    "linf_norm",
    "mean_abs",
    "concentration_of_energy",
    "sv_topk",
    "stable_rank",
    "spectral_entropy",
    "effective_rank",
]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _write_source_bundle(tmp_path: Path) -> dict[str, Path | list[str]]:
    feature_root = tmp_path / "runs" / "feature_extract"
    output_dir = feature_root / "source_bundle" / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    block_names = [
        "layer0.self_attn.q_proj",
        "layer0.self_attn.qv_sum",
    ]
    feature_names = build_spectral_feature_names(
        block_names=block_names,
        selected_features=list(BASELINE_SELECTED_FEATURES),
        sv_top_k=2,
        spectral_moment_source="both",
        shorten_block_names=False,
    )

    model_names = [
        "llama2_7b_toxic_backdoors_hard_rank8_qv_label0_0",
        "llama2_7b_toxic_backdoors_hard_rank8_qv_label1_0",
        "llama2_7b_toxic_backdoors_hard_rank16_qv_label0_0",
        "llama2_7b_toxic_backdoors_hard_rank16_qv_label1_0",
    ]
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    rows: list[list[float]] = []
    for rank, label in [(8, 0), (8, 1), (16, 0), (16, 1)]:
        shift = 0.5 if label == 1 else 0.0
        rows.append(
            [
                80.0 + shift + rank,   # q_proj.energy
                1.0 + shift,           # q_proj.kurtosis
                1.25 + shift,          # q_proj.sv_kurtosis
                16.0 + shift,          # q_proj.l1_norm
                32.0 + shift,          # q_proj.sv_l1_norm
                24.0 + shift,          # q_proj.l2_norm
                8.0 + shift,           # q_proj.linf_norm
                12.0 + shift,          # q_proj.sv_linf_norm
                4.0 + shift,           # q_proj.mean_abs
                6.0 + shift,           # q_proj.sv_mean_abs
                0.7 + shift,           # q_proj.concentration_of_energy
                10.0 + shift,          # q_proj.sv_1
                5.0 + shift,           # q_proj.sv_2
                7.0 + shift,           # q_proj.stable_rank
                2.0 + shift,           # q_proj.spectral_entropy
                6.0 + shift,           # q_proj.effective_rank
                160.0 + shift + rank,  # qv_sum.energy
                1.5 + shift,           # qv_sum.kurtosis
                1.75 + shift,          # qv_sum.sv_kurtosis
                32.0 + shift,          # qv_sum.l1_norm
                48.0 + shift,          # qv_sum.sv_l1_norm
                40.0 + shift,          # qv_sum.l2_norm
                16.0 + shift,          # qv_sum.linf_norm
                20.0 + shift,          # qv_sum.sv_linf_norm
                8.0 + shift,           # qv_sum.mean_abs
                10.0 + shift,          # qv_sum.sv_mean_abs
                0.8 + shift,           # qv_sum.concentration_of_energy
                12.0 + shift,          # qv_sum.sv_1
                6.0 + shift,           # qv_sum.sv_2
                12.0 + shift,          # qv_sum.stable_rank
                3.0 + shift,           # qv_sum.spectral_entropy
                10.0 + shift,          # qv_sum.effective_rank
            ]
        )
    features = np.asarray(rows, dtype=np.float32)

    np.save(output_dir / "spectral_features.npy", features)
    _write_json(output_dir / "spectral_model_names.json", model_names)
    np.save(output_dir / "spectral_labels.npy", labels)

    metadata = {
        "extractor": "spectral",
        "extractor_name": "spectral",
        "extractor_version": "test",
        "delta_schema_version": "test",
        "feature_dim": len(feature_names),
        "feature_names": list(feature_names),
        "block_names": list(block_names),
        "base_block_names": ["layer0.self_attn.q_proj"],
        "qv_sum_block_names": ["layer0.self_attn.qv_sum"],
        "n_blocks": len(block_names),
        "resolved_features": list(BASELINE_SELECTED_FEATURES),
        "sv_top_k": 2,
        "spectral_moment_source": "both",
        "spectral_qv_sum_mode": "append",
        "spectral_entrywise_delta_mode": "dense",
        "extractor_params": {
            "spectral_features": list(BASELINE_SELECTED_FEATURES),
            "spectral_sv_top_k": 2,
            "spectral_moment_source": "both",
            "spectral_qv_sum_mode": "append",
            "spectral_entrywise_delta_mode": "dense",
        },
        "dataset_layouts": [
            {
                "dataset_name": "llama2_7b_toxic_backdoors_hard_rank8_qv",
                "sample_count": 2,
                "layer_count": 1,
            },
            {
                "dataset_name": "llama2_7b_toxic_backdoors_hard_rank16_qv",
                "sample_count": 2,
                "layer_count": 1,
            },
        ],
    }
    write_spectral_metadata(output_dir / "spectral_metadata.json", internal_metadata=metadata)

    model_index = {
        model_name: {
            "dataset_name": (
                "llama2_7b_toxic_backdoors_hard_rank8_qv"
                if "rank8" in model_name
                else "llama2_7b_toxic_backdoors_hard_rank16_qv"
            ),
            "dataset_path": str((tmp_path / "data").resolve()),
            "label": int(labels[idx]),
            "subset_name": (
                "llama2_7b_toxic_backdoors_hard_rank8_qv"
                if "rank8" in model_name
                else "llama2_7b_toxic_backdoors_hard_rank16_qv"
            ),
            "model_family": "llama2_7b",
            "attack_name": "tbh",
        }
        for idx, model_name in enumerate(model_names)
    }
    payload = _finalize_payload(
        artifact_kind="feature_extract",
        model_index=model_index,
        artifact_model_count=len(model_names),
        manifest_json=None,
        dataset_root=tmp_path / "data",
        source_artifacts=["synthetic_source"],
        provenance_gaps=[],
        is_complete=True,
    )
    write_dataset_reference_report(default_dataset_reference_report_path(output_dir), payload)

    return {
        "feature_root": feature_root,
        "feature_dir": output_dir,
        "feature_names": feature_names,
        "model_names": model_names,
    }


def _write_baseline_reference_run(tmp_path: Path) -> dict[str, Path]:
    run_dir = tmp_path / "runs" / "supervised" / "baseline_reference"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_config.json",
        {
            "model_name": "cnn_1d",
            "train_split_percent": 80,
            "split_by_folder": True,
            "cv_random_states": [42, 43],
            "random_state": 42,
            "score_percentiles": [90.0, 95.0, 99.0],
        },
    )
    _write_json(
        reports_dir / "tuning_manifest.json",
        {
            "extractor": {
                "params": {
                    "spectral_features": list(BASELINE_SELECTED_FEATURES),
                    "spectral_sv_top_k": 2,
                    "spectral_moment_source": "both",
                    "spectral_qv_sum_mode": "append",
                    "spectral_entrywise_delta_mode": "dense",
                },
            },
            "threshold_selection": {
                "calibration_split_percent": 20,
                "accepted_fprs": [0.01, 0.05],
                "split_by_folder": True,
            },
            "tuning": {
                "cv_folds_requested": 3,
                "cv_random_states": [42, 43],
            },
        },
    )
    _write_json(reports_dir / "supervised_report.json", {"tuning": {"winner": {"model_name": "cnn_1d"}}})
    return {"run_dir": run_dir}


def _write_baseline_cnn_bundle(tmp_path: Path) -> Path:
    output_dir = tmp_path / "runs" / "feature_extract" / "baseline_cnn_bundle" / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "spectral_features.npy", np.zeros((4, 1, 2, 16), dtype=np.float32))
    _write_json(output_dir / "spectral_model_names.json", [])
    write_spectral_metadata(
        output_dir / "spectral_metadata.json",
        internal_metadata={
            "feature_dim": 16,
            "feature_names": [],
            "resolved_features": list(BASELINE_SELECTED_FEATURES),
            "sv_top_k": 2,
            "spectral_moment_source": "both",
            "spectral_qv_sum_mode": "append",
        },
    )
    return output_dir / "spectral_features.npy"


def _write_manifests(tmp_path: Path) -> dict[str, Path]:
    tuning_manifest = tmp_path / "manifests" / "others" / "list2.json"
    zero_shot_root = tmp_path / "manifests" / "zero_shots" / "rank_wise"
    zero_shot_manifest = zero_shot_root / "llama2_7b_tbh_zero_shot_r256_to_rank16.json"

    _write_json(
        tuning_manifest,
        {
            "path": [
                {
                    "path": "llama2_7b_toxic_backdoors_hard_rank8_qv/llama2_7b_toxic_backdoors_hard_rank8_qv_label0_",
                    "indices": [0, 0],
                },
                {
                    "path": "llama2_7b_toxic_backdoors_hard_rank8_qv/llama2_7b_toxic_backdoors_hard_rank8_qv_label1_",
                    "indices": [0, 0],
                },
            ]
        },
    )
    _write_json(
        zero_shot_manifest,
        {
            "train": [
                {
                    "path": "llama2_7b_toxic_backdoors_hard_rank8_qv/llama2_7b_toxic_backdoors_hard_rank8_qv_label0_",
                    "indices": [0, 0],
                },
                {
                    "path": "llama2_7b_toxic_backdoors_hard_rank8_qv/llama2_7b_toxic_backdoors_hard_rank8_qv_label1_",
                    "indices": [0, 0],
                },
            ],
            "infer": [
                {
                    "path": "llama2_7b_toxic_backdoors_hard_rank16_qv/llama2_7b_toxic_backdoors_hard_rank16_qv_label0_",
                    "indices": [0, 0],
                },
                {
                    "path": "llama2_7b_toxic_backdoors_hard_rank16_qv/llama2_7b_toxic_backdoors_hard_rank16_qv_label1_",
                    "indices": [0, 0],
                },
            ],
        },
    )
    return {
        "tuning_manifest": tuning_manifest,
        "zero_shot_root": zero_shot_root,
        "zero_shot_manifest": zero_shot_manifest,
    }


def _write_baseline_zero_shot_results(tmp_path: Path) -> Path:
    run_dir = (
        tmp_path
        / "runs"
        / "supervised"
        / "zero_shot_cnn"
        / "llama2_7b_tbh_zero_shot_cnn_r256_to_rank16"
        / "reports"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "inference_scores.csv", "w", encoding="utf-8", newline="") as f:
        f.write("index,model_name,label,score,score_percentile_rank\n")
        f.write("0,llama2_7b_toxic_backdoors_hard_rank16_qv_label0_0,0,0.1,10\n")
        f.write("1,llama2_7b_toxic_backdoors_hard_rank16_qv_label1_0,1,0.9,90\n")
    return run_dir.parent


def _build_config(
    *,
    tmp_path: Path,
    source_feature_root: Path,
    manifests: dict[str, Path],
    **overrides: object,
) -> StudyConfig:
    payload: dict[str, object] = {
        "stage": "all",
        "source_feature_file": Path("source_bundle"),
        "baseline_feature_file": Path("baseline_cnn_bundle"),
        "baseline_reference_run": "baseline_reference",
        "baseline_transfer_results_root": tmp_path / "runs" / "supervised" / "zero_shot_cnn",
        "tuning_manifest": manifests["tuning_manifest"],
        "zero_shot_manifest_root": manifests["zero_shot_root"],
        "zero_shot_manifest_filter": "llama2_7b_tbh_zero_shot_r256_to_rank",
        "feature_root": source_feature_root,
        "output_root": tmp_path / "runs",
        "dataset_root": tmp_path / "data",
        "run_id": "study",
        "rank_norm_feature_output_run": "rank_norm_bundle",
        "rank_norm_cnn_output_run": "rank_norm_bundle_cnn",
        "rank_norm_reference_run": "rank_norm_reference",
        "rank_norm_zero_shot_prefix": "zero_shot_cnn_rank_norm",
        "raw_rank_feature_output_run": "raw_rank_bundle",
        "raw_rank_cnn_output_run": "raw_rank_bundle_cnn",
        "raw_rank_reference_run": "raw_rank_reference",
        "raw_rank_zero_shot_prefix": "zero_shot_cnn_raw_rank",
        "reference_tuning_executor": "local",
        "reference_n_jobs": 1,
        "reference_dry_run": False,
        "zero_shot_dry_run": True,
        "slurm_partition": "extra",
        "slurm_log_dir": tmp_path / "logs",
        "conda_sh": Path("/tmp/fake_conda.sh"),
        "conda_env": "upeftg",
        "skip_derive": False,
        "skip_aggregate": False,
        "skip_reference": False,
        "skip_zero_shot_launch": False,
        "report_only": False,
    }
    payload.update(overrides)
    return StudyConfig(**payload)


class RankNormalizationStudyTests(unittest.TestCase):
    def test_parse_fixed_rank_and_qv_sum_scale(self):
        self.assertEqual(
            parse_fixed_rank_from_model_name("llama2_7b_toxic_backdoors_hard_rank256_qv_label1_0"),
            256,
        )
        self.assertEqual(
            block_rank_scale_for_sample(
                model_name="llama2_7b_toxic_backdoors_hard_rank16_qv_label0_0",
                block_name="layer0.self_attn.q_proj",
            ),
            16,
        )
        self.assertEqual(
            block_rank_scale_for_sample(
                model_name="llama2_7b_toxic_backdoors_hard_rank16_qv_label0_0",
                block_name="layer0.self_attn.qv_sum",
            ),
            32,
        )
        self.assertEqual(zero_shot_target_rank_from_name("llama2_7b_tbh_zero_shot_r256_to_rank16"), 16)

    def test_collect_manifest_model_names_is_manifest_driven(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifests = _write_manifests(tmp_path)
            expectations = collect_zero_shot_expectations(
                manifest_root=manifests["zero_shot_root"],
                manifest_filter="llama2_7b_tbh_zero_shot_r256_to_rank",
            )
            model_names, manifest_paths = collect_manifest_model_names(
                tuning_manifest=manifests["tuning_manifest"],
                zero_shot_expectations=expectations,
            )
            self.assertEqual(
                model_names,
                [
                    "llama2_7b_toxic_backdoors_hard_rank8_qv_label0_0",
                    "llama2_7b_toxic_backdoors_hard_rank8_qv_label1_0",
                    "llama2_7b_toxic_backdoors_hard_rank16_qv_label0_0",
                    "llama2_7b_toxic_backdoors_hard_rank16_qv_label1_0",
                ],
            )
            self.assertEqual(len(manifest_paths), 2)

    def test_derive_arm_feature_bundle_applies_rank_normalization_and_block_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = _write_source_bundle(tmp_path)
            manifests = _write_manifests(tmp_path)
            bundle = load_feature_bundle(
                feature_file=Path("source_bundle"),
                feature_root=source["feature_root"],
            )
            expectations = collect_zero_shot_expectations(
                manifest_root=manifests["zero_shot_root"],
                manifest_filter="llama2_7b_tbh_zero_shot_r256_to_rank",
            )
            selected_model_names, manifest_paths = collect_manifest_model_names(
                tuning_manifest=manifests["tuning_manifest"],
                zero_shot_expectations=expectations,
            )
            rank_norm_features, raw_rank_features = build_arm_feature_groups(
                baseline_features=BASELINE_SELECTED_FEATURES
            )
            rank_norm_arm = StudyArmSpec(
                name="rank_norm",
                selected_features=tuple(rank_norm_features),
                feature_output_run="rank_norm_bundle",
                cnn_output_run="rank_norm_bundle_cnn",
                reference_run_id="rank_norm_reference",
                zero_shot_run_prefix="zero_shot_cnn_rank_norm",
            )
            raw_rank_arm = StudyArmSpec(
                name="raw_rank",
                selected_features=tuple(raw_rank_features),
                feature_output_run="raw_rank_bundle",
                cnn_output_run="raw_rank_bundle_cnn",
                reference_run_id="raw_rank_reference",
                zero_shot_run_prefix="zero_shot_cnn_raw_rank",
            )

            rank_norm_bundle = derive_arm_feature_bundle(
                bundle=bundle,
                arm=rank_norm_arm,
                feature_root=source["feature_root"],
                selected_model_names=selected_model_names,
                manifest_paths=manifest_paths,
                spectral_sv_top_k=2,
                spectral_moment_source="both",
                spectral_qv_sum_mode="append",
            )
            rank_norm_matrix = np.asarray(np.load(rank_norm_bundle.feature_path), dtype=np.float32)
            rank_norm_index = {
                name: idx for idx, name in enumerate(rank_norm_bundle.feature_names)
            }
            self.assertIn("layer0.self_attn.q_proj.energy_per_rank", rank_norm_index)
            self.assertIn("layer0.self_attn.q_proj.sv_l1_norm_per_rank", rank_norm_index)
            self.assertIn("layer0.self_attn.qv_sum.normalized_spectral_entropy", rank_norm_index)
            self.assertAlmostEqual(
                float(rank_norm_matrix[0, rank_norm_index["layer0.self_attn.q_proj.energy_per_rank"]]),
                88.0 / 8.0,
                places=6,
            )
            self.assertAlmostEqual(
                float(rank_norm_matrix[0, rank_norm_index["layer0.self_attn.qv_sum.energy_per_rank"]]),
                168.0 / 16.0,
                places=6,
            )

            raw_rank_bundle = derive_arm_feature_bundle(
                bundle=bundle,
                arm=raw_rank_arm,
                feature_root=source["feature_root"],
                selected_model_names=selected_model_names,
                manifest_paths=manifest_paths,
                spectral_sv_top_k=2,
                spectral_moment_source="both",
                spectral_qv_sum_mode="append",
            )
            raw_rank_matrix = np.asarray(np.load(raw_rank_bundle.feature_path), dtype=np.float32)
            raw_rank_index = {
                name: idx for idx, name in enumerate(raw_rank_bundle.feature_names)
            }
            self.assertEqual(
                float(raw_rank_matrix[0, raw_rank_index["layer0.self_attn.q_proj.block_rank"]]),
                8.0,
            )
            self.assertEqual(
                float(raw_rank_matrix[2, raw_rank_index["layer0.self_attn.qv_sum.block_rank"]]),
                32.0,
            )

    def test_derive_arm_feature_bundle_skips_missing_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = _write_source_bundle(tmp_path)
            manifests = _write_manifests(tmp_path)
            bundle = load_feature_bundle(
                feature_file=Path("source_bundle"),
                feature_root=source["feature_root"],
            )
            expectations = collect_zero_shot_expectations(
                manifest_root=manifests["zero_shot_root"],
                manifest_filter="llama2_7b_tbh_zero_shot_r256_to_rank",
            )
            selected_model_names, manifest_paths = collect_manifest_model_names(
                tuning_manifest=manifests["tuning_manifest"],
                zero_shot_expectations=expectations,
            )
            selected_model_names = [*selected_model_names, "llama2_7b_toxic_backdoors_hard_rank512_qv_label0_2"]
            rank_norm_features, _ = build_arm_feature_groups(
                baseline_features=BASELINE_SELECTED_FEATURES
            )
            arm = StudyArmSpec(
                name="rank_norm",
                selected_features=tuple(rank_norm_features),
                feature_output_run="rank_norm_bundle",
                cnn_output_run="rank_norm_bundle_cnn",
                reference_run_id="rank_norm_reference",
                zero_shot_run_prefix="zero_shot_cnn_rank_norm",
            )
            derived = derive_arm_feature_bundle(
                bundle=bundle,
                arm=arm,
                feature_root=source["feature_root"],
                selected_model_names=selected_model_names,
                manifest_paths=manifest_paths,
                spectral_sv_top_k=2,
                spectral_moment_source="both",
                spectral_qv_sum_mode="append",
            )
            self.assertEqual(derived.requested_model_count, 5)
            self.assertEqual(len(derived.model_names), 4)
            self.assertEqual(derived.missing_model_names, ["llama2_7b_toxic_backdoors_hard_rank512_qv_label0_2"])

    def test_aggregate_arm_feature_bundle_creates_layer_sequence_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = _write_source_bundle(tmp_path)
            manifests = _write_manifests(tmp_path)
            bundle = load_feature_bundle(
                feature_file=Path("source_bundle"),
                feature_root=source["feature_root"],
            )
            expectations = collect_zero_shot_expectations(
                manifest_root=manifests["zero_shot_root"],
                manifest_filter="llama2_7b_tbh_zero_shot_r256_to_rank",
            )
            selected_model_names, manifest_paths = collect_manifest_model_names(
                tuning_manifest=manifests["tuning_manifest"],
                zero_shot_expectations=expectations,
            )
            rank_norm_features, _ = build_arm_feature_groups(
                baseline_features=BASELINE_SELECTED_FEATURES
            )
            arm = StudyArmSpec(
                name="rank_norm",
                selected_features=tuple(rank_norm_features),
                feature_output_run="rank_norm_bundle",
                cnn_output_run="rank_norm_bundle_cnn",
                reference_run_id="rank_norm_reference",
                zero_shot_run_prefix="zero_shot_cnn_rank_norm",
            )
            derived = derive_arm_feature_bundle(
                bundle=bundle,
                arm=arm,
                feature_root=source["feature_root"],
                selected_model_names=selected_model_names,
                manifest_paths=manifest_paths,
                spectral_sv_top_k=2,
                spectral_moment_source="both",
                spectral_qv_sum_mode="append",
            )
            outputs = aggregate_arm_feature_bundle(
                arm=arm,
                feature_root=source["feature_root"],
                feature_path=derived.feature_path,
                selected_features=arm.selected_features,
                spectral_qv_sum_mode="append",
            )
            tensor = np.asarray(np.load(outputs["feature_path"]), dtype=np.float32)
            self.assertEqual(tensor.ndim, 4)
            metadata = load_spectral_metadata(Path(str(outputs["metadata_path"])))
            self.assertEqual(metadata["resolved_features"], list(arm.selected_features))

    def test_launch_zero_shot_suite_uses_existing_launcher_and_explicit_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifests = {
                "tuning_manifest": tmp_path / "manifests" / "others" / "list2.json",
                "zero_shot_root": tmp_path / "manifests" / "zero_shots" / "rank_wise",
            }
            config = _build_config(
                tmp_path=tmp_path,
                source_feature_root=tmp_path / "runs" / "feature_extract",
                manifests=manifests,
            )
            arm = StudyArmSpec(
                name="rank_norm",
                selected_features=tuple(BASELINE_SELECTED_FEATURES),
                feature_output_run="rank_norm_bundle",
                cnn_output_run="rank_norm_bundle_cnn",
                reference_run_id="rank_norm_reference",
                zero_shot_run_prefix="zero_shot_cnn_rank_norm",
            )
            (tmp_path / "manifests" / "zero_shots" / "rank_wise").mkdir(parents=True, exist_ok=True)
            with patch("upeftguard.experiments.rank_normalization_study.subprocess.run") as run_mock:
                run_mock.return_value = object()
                launch_zero_shot_suite(
                    arm=arm,
                    config=config,
                    reference_run_dir=tmp_path / "runs" / "supervised" / "rank_norm_reference",
                    cnn_feature_path=tmp_path / "runs" / "feature_extract" / "rank_norm_bundle_cnn" / "merged" / "spectral_features.npy",
                )
                args, kwargs = run_mock.call_args
                cmd = args[0]
                self.assertIn("--hyperparam_config", cmd)
                self.assertIn("--manifest-filter", cmd)
                self.assertIn("llama2_7b_tbh_zero_shot_r256_to_rank", cmd)
                self.assertEqual(kwargs["env"]["RUN_ID_PREFIX"], "zero_shot_cnn_rank_norm")
                self.assertEqual(
                    kwargs["env"]["ZERO_SHOT_MANIFEST_ROOT"],
                    str((tmp_path / "manifests" / "zero_shots" / "rank_wise").resolve()),
                )

    def test_run_rank_normalization_study_keeps_baseline_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = _write_source_bundle(tmp_path)
            baseline_reference = _write_baseline_reference_run(tmp_path)
            _write_baseline_cnn_bundle(tmp_path)
            manifests = _write_manifests(tmp_path)
            _write_baseline_zero_shot_results(tmp_path)

            config = _build_config(
                tmp_path=tmp_path,
                source_feature_root=source["feature_root"],
                manifests=manifests,
            )

            with patch("upeftguard.experiments.rank_normalization_study.run_supervised_pipeline") as run_pipeline_mock, patch(
                "upeftguard.experiments.rank_normalization_study.subprocess.run"
            ) as subprocess_mock, patch(
                "upeftguard.experiments.rank_normalization_study._assert_reference_run_completed"
            ) as assert_completed_mock:
                run_pipeline_mock.side_effect = [
                    {"run_dir": str(tmp_path / "runs" / "supervised" / "rank_norm_reference")},
                    {"run_dir": str(tmp_path / "runs" / "supervised" / "raw_rank_reference")},
                ]
                subprocess_mock.return_value = object()
                result = run_rank_normalization_study(config)

                self.assertEqual(run_pipeline_mock.call_count, 2)
                self.assertEqual(assert_completed_mock.call_count, 2)
                requested_run_ids = [call.kwargs["run_id"] for call in run_pipeline_mock.call_args_list]
                self.assertEqual(requested_run_ids, ["rank_norm_reference", "raw_rank_reference"])
                self.assertEqual(subprocess_mock.call_count, 2)

                with open(result["study_report"], "r", encoding="utf-8") as f:
                    study_report = json.load(f)
                baseline_metrics = study_report["metrics_by_arm"]["baseline"]["rows"]
                self.assertEqual(len(baseline_metrics), 1)
                self.assertEqual(baseline_metrics[0]["status"], "ok")
                self.assertAlmostEqual(float(baseline_metrics[0]["roc_auc"]), 1.0, places=6)
                self.assertEqual(
                    study_report["baseline_reference_run"],
                    str((baseline_reference["run_dir"]).resolve()),
                )

    def test_prepare_reference_stage_submits_slurm_array_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = _write_source_bundle(tmp_path)
            _write_baseline_reference_run(tmp_path)
            _write_baseline_cnn_bundle(tmp_path)
            manifests = _write_manifests(tmp_path)

            config = _build_config(
                tmp_path=tmp_path,
                source_feature_root=source["feature_root"],
                manifests=manifests,
                stage="prepare_reference",
                reference_tuning_executor="slurm_array",
                reference_dry_run=False,
                zero_shot_dry_run=False,
            )

            prepared_rank_norm_dir = tmp_path / "runs" / "supervised" / "rank_norm_reference"
            prepared_raw_rank_dir = tmp_path / "runs" / "supervised" / "raw_rank_reference"
            prepared_payloads = [
                {
                    "run_dir": str(prepared_rank_norm_dir),
                    "tuning_manifest": str(prepared_rank_norm_dir / "reports" / "tuning_manifest.json"),
                    "n_tasks": 64,
                },
                {
                    "run_dir": str(prepared_raw_rank_dir),
                    "tuning_manifest": str(prepared_raw_rank_dir / "reports" / "tuning_manifest.json"),
                    "n_tasks": 64,
                },
            ]
            runtime_payload = {
                "runtime": {
                    "slurm_max_concurrent": 14,
                    "slurm_cpus_per_task": 4,
                    "score_percentiles": [90.0, 95.0, 99.0],
                }
            }
            for run_dir in [prepared_rank_norm_dir, prepared_raw_rank_dir]:
                (run_dir / "reports").mkdir(parents=True, exist_ok=True)
                _write_json(run_dir / "reports" / "tuning_manifest.json", runtime_payload)

            completed = [
                type("Proc", (), {"stdout": "111\n"})(),
                type("Proc", (), {"stdout": "222\n"})(),
                type("Proc", (), {"stdout": "333\n"})(),
                type("Proc", (), {"stdout": "444\n"})(),
            ]
            with patch("upeftguard.experiments.rank_normalization_study.run_supervised_pipeline") as run_pipeline_mock, patch(
                "upeftguard.experiments.rank_normalization_study.subprocess.run"
            ) as subprocess_mock:
                run_pipeline_mock.side_effect = prepared_payloads
                subprocess_mock.side_effect = completed
                result = run_rank_normalization_study(config)

                self.assertEqual(run_pipeline_mock.call_count, 2)
                for call in run_pipeline_mock.call_args_list:
                    self.assertEqual(call.kwargs["stage"], "prepare")
                    self.assertEqual(call.kwargs["tuning_executor"], "slurm_array")
                self.assertEqual(subprocess_mock.call_count, 4)
                first_worker_cmd = subprocess_mock.call_args_list[0].args[0]
                self.assertEqual(first_worker_cmd[0], "sbatch")
                self.assertIn("--array", first_worker_cmd)
                self.assertIn("0-63%14", first_worker_cmd)
                self.assertIsNone(result["comparison_json"])

                with open(result["study_report"], "r", encoding="utf-8") as f:
                    study_report = json.load(f)
                self.assertEqual(study_report["stage"], "prepare_reference")
                self.assertIn("reference_submission", study_report["arm_artifacts"]["rank_norm"])
                self.assertEqual(
                    study_report["arm_artifacts"]["raw_rank"]["reference_submission"]["finalize_job_id"],
                    "444",
                )

    def test_parser_report_only_maps_to_report_stage(self):
        parser = build_parser()
        args = parser.parse_args(["--report-only"])
        config = args_to_config(args)
        self.assertEqual(config.stage, "report")
        self.assertTrue(config.skip_reference)
        self.assertTrue(config.skip_zero_shot_launch)


if __name__ == "__main__":
    unittest.main()
