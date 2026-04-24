import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

try:
    import torch
except Exception:
    torch = None

from upeftguard.supervised import pipeline as pipeline_mod
from upeftguard.supervised import registry as registry_mod
from upeftguard.supervised.interfaces import (
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    ATTACK_FAMILY_MULTICLASS_ATTACKS,
    BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
    SupervisedPredictionOutputs,
)
from upeftguard.supervised.pipeline import (
    OPEN_SET_UNKNOWN_ATTACK_NAME,
    _apply_open_set_unknown_attack_rule,
    _build_open_set_unknown_attack_config,
    _labels_from_items,
    _open_set_true_labels,
    _resolve_supervised_task_spec,
    run_supervised_pipeline,
)
from upeftguard.utilities.artifacts.spectral_metadata import write_spectral_metadata
from upeftguard.utilities.core.manifest import ManifestItem, infer_attack_sample_identities


def make_manifest_item(subset_name: str, label: int, index: int) -> ManifestItem:
    model_name = f"{subset_name}_label{label}_{index}"
    model_dir = Path("/tmp/data") / subset_name / model_name
    return ManifestItem(
        raw_entry=str(model_dir),
        model_dir=model_dir,
        adapter_path=model_dir / "adapter_model.safetensors",
        model_name=model_name,
        label=label,
    )


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_joint_manifest_by_name(path: Path, *, train_entries: list[str], infer_entries: list[str]) -> None:
    payload = {"train": list(train_entries), "infer": list(infer_entries)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_external_feature_bundle(
    output_dir: Path,
    *,
    features: np.ndarray,
    model_names: list[str],
    metadata: dict[str, Any],
    group_mask: np.ndarray | None = None,
    value_mask: np.ndarray | None = None,
    group_names: list[list[str]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "spectral_features.npy", np.asarray(features, dtype=np.float32))
    with open(output_dir / "spectral_model_names.json", "w", encoding="utf-8") as f:
        json.dump(list(model_names), f, indent=2)
    if group_mask is not None:
        np.save(output_dir / "spectral_group_mask.npy", np.asarray(group_mask, dtype=bool))
    if value_mask is not None:
        np.save(output_dir / "spectral_value_mask.npy", np.asarray(value_mask, dtype=bool))
    if group_names is not None:
        with open(output_dir / "spectral_group_names.json", "w", encoding="utf-8") as f:
            json.dump([list(names) for names in group_names], f, indent=2)
    write_spectral_metadata(output_dir / "spectral_metadata.json", internal_metadata=metadata)


def build_multiclass_joint_rows(
    *,
    train_per_dataset: int,
    infer_per_dataset: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    train_entries: list[str] = []
    infer_entries: list[str] = []
    rows: list[dict[str, Any]] = []
    dataset_names = ("ag_news", "imdb")
    class_rows = [
        ("clean", "insertsent", 0, 0),
        ("RIPPLE", "RIPPLE", 1, 1),
        ("insertsent", "insertsent", 1, 2),
        ("stybkd", "stybkd", 1, 3),
        ("syntactic", "syntactic", 1, 4),
    ]

    for dataset_name in dataset_names:
        for class_name, subset_attack_name, label, class_index in class_rows:
            subset_name = f"llama2_7b_{dataset_name}_{subset_attack_name}_rank256_qv"
            for sample_idx in range(train_per_dataset):
                row_index = sample_idx
                model_name = f"{subset_name}_label{label}_{row_index}"
                train_entries.append(f"{subset_name}/{model_name}")
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "class_name": class_name,
                        "class_index": class_index,
                    }
                )
            for sample_idx in range(infer_per_dataset):
                row_index = 200 + sample_idx
                model_name = f"{subset_name}_label{label}_{row_index}"
                infer_entries.append(f"{subset_name}/{model_name}")
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "class_name": class_name,
                        "class_index": class_index,
                    }
                )

    return train_entries, infer_entries, rows


def build_tabular_bundle(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    block_names = [
        "base_model.model.model.layers.0.self_attn.q_proj",
        "base_model.model.model.layers.0.self_attn.v_proj",
    ]
    feature_names = [
        f"{block_name}.{feature_name}"
        for block_name in block_names
        for feature_name in ("energy", "sv_kurtosis")
    ]
    centroids = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    dataset_shift = {"ag_news": 0.05, "imdb": -0.05}
    features = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        base = np.asarray(centroids[int(row["class_index"])], dtype=np.float32)
        offset = np.asarray(
            [
                0.01 * row_idx,
                -0.02 * row_idx,
                0.015 * row_idx,
                -0.01 * row_idx,
            ],
            dtype=np.float32,
        )
        features[row_idx] = base + offset + float(dataset_shift[str(row["dataset_name"])])

    metadata = {
        "representation_kind": "spectral_tabular",
        "resolved_features": ["energy", "kurtosis"],
        "spectral_moment_source": "sv",
        "spectral_qv_sum_mode": "none",
        "sv_top_k": 1,
        "feature_dim": int(features.shape[1]),
        "feature_names": list(feature_names),
        "block_names": list(block_names),
        "n_blocks": int(len(block_names)),
    }
    model_names = [str(row["model_name"]) for row in rows]
    write_external_feature_bundle(
        output_dir,
        features=features,
        model_names=model_names,
        metadata=metadata,
    )
    return output_dir / "spectral_features.npy"


def build_layer_sequence_bundle(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    depth_labels = ["encoder.layer0", "encoder.layer1"]
    slot_names = ["self.q", "self.v"]
    emitted_feature_names = ["energy", "stable_rank"]
    values = np.zeros((len(rows), len(depth_labels), len(slot_names), len(emitted_feature_names)), dtype=np.float32)
    group_mask = np.ones((len(rows), len(depth_labels)), dtype=bool)
    value_mask = np.ones_like(values, dtype=bool)
    group_names = [list(depth_labels) for _ in rows]

    for row_idx, row in enumerate(rows):
        class_index = int(row["class_index"])
        values[row_idx, :, :, :] = 0.05 * row_idx
        if class_index == 0:
            values[row_idx, 0, 0, :] += np.asarray([0.1, 0.05], dtype=np.float32)
        elif class_index == 1:
            values[row_idx, 0, 0, :] += np.asarray([3.0, 1.5], dtype=np.float32)
        elif class_index == 2:
            values[row_idx, 0, 1, :] += np.asarray([3.0, 1.5], dtype=np.float32)
        elif class_index == 3:
            values[row_idx, 1, 0, :] += np.asarray([3.0, 1.5], dtype=np.float32)
        else:
            values[row_idx, 1, 1, :] += np.asarray([3.0, 1.5], dtype=np.float32)

    metadata = {
        "representation_kind": ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
        "resolved_features": ["energy", "stable_rank"],
        "spectral_moment_source": "sv",
        "spectral_qv_sum_mode": "append",
        "sv_top_k": 1,
        "depth_labels": list(depth_labels),
        "slot_names": list(slot_names),
        "emitted_feature_names": list(emitted_feature_names),
    }
    model_names = [str(row["model_name"]) for row in rows]
    write_external_feature_bundle(
        output_dir,
        features=values,
        model_names=model_names,
        metadata=metadata,
        group_mask=group_mask,
        value_mask=value_mask,
        group_names=group_names,
    )
    return output_dir / "spectral_features.npy"


class TestSupervisedMulticlassTaskMode(unittest.TestCase):
    def test_multiclass_label_mapping_uses_fixed_class_order_and_canonical_aliases(self):
        task_spec = _resolve_supervised_task_spec(
            task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
            multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
        )
        self.assertEqual(
            list(task_spec.class_names),
            ["clean", "RIPPLE", "insertsent", "stybkd", "syntactic"],
        )
        self.assertEqual(task_spec.binary_projection, BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY)

        items = [
            make_manifest_item("llama2_7b_imdb_insertsent_rank256_qv", 0, 0),
            make_manifest_item("llama2_7b_imdb_RIPPLE_rank256_qv", 1, 1),
            make_manifest_item("llama2_7b_imdb_insertsent_rank256_qv", 1, 2),
            make_manifest_item("llama2_7b_imdb_stykbd_rank256_qv", 1, 3),
            make_manifest_item("llama2_7b_imdb_syntactic_rank256_qv", 1, 4),
        ]
        identities = infer_attack_sample_identities(items)
        values, known, raw_labels = _labels_from_items(
            items,
            task_spec=task_spec,
            sample_identities=identities,
        )

        self.assertEqual([identity.attack_name for identity in identities], ["insertsent", "RIPPLE", "insertsent", "stybkd", "syntactic"])
        self.assertEqual(values.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(known.tolist(), [True, True, True, True, True])
        self.assertEqual(raw_labels, [0, 1, 2, 3, 4])

    def test_multiclass_label_mapping_rejects_unsupported_positive_attack(self):
        task_spec = _resolve_supervised_task_spec(
            task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
            multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
        )
        items = [make_manifest_item("llama2_7b_toxic_backdoors_hard_rank256_qv", 1, 0)]
        identities = infer_attack_sample_identities(items)

        with self.assertRaisesRegex(ValueError, "outside the configured attack vocabulary"):
            _labels_from_items(
                items,
                task_spec=task_spec,
                sample_identities=identities,
            )

    def test_custom_multiclass_attack_vocabulary_is_supported(self):
        task_spec = _resolve_supervised_task_spec(
            task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
            multiclass_attack_names=["toxic_backdoors_hard", "toxic_backdoors_alpaca"],
        )
        self.assertEqual(
            list(task_spec.class_names),
            ["clean", "toxic_backdoors_hard", "toxic_backdoors_alpaca"],
        )
        self.assertEqual(task_spec.binary_projection, BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY)

        items = [
            make_manifest_item("llama2_7b_toxic_backdoors_hard_rank256_qv", 0, 0),
            make_manifest_item("llama2_7b_toxic_backdoors_hard_rank256_qv", 1, 1),
            make_manifest_item("llama2_7b_toxic_backdoors_alpaca_rank256_qv", 1, 2),
        ]
        identities = infer_attack_sample_identities(items)
        values, known, raw_labels = _labels_from_items(
            items,
            task_spec=task_spec,
            sample_identities=identities,
        )

        self.assertEqual(
            [identity.attack_name for identity in identities],
            ["toxic_backdoors_hard", "toxic_backdoors_hard", "toxic_backdoors_alpaca"],
        )
        self.assertEqual(values.tolist(), [0, 1, 2])
        self.assertEqual(known.tolist(), [True, True, True])
        self.assertEqual(raw_labels, [0, 1, 2])

    def test_custom_attack_vocabulary_open_set_maps_other_attack_to_unknown(self):
        task_spec = _resolve_supervised_task_spec(
            task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
            multiclass_attack_names=["toxic_backdoors_hard", "toxic_backdoors_alpaca"],
        )
        train_probabilities = np.asarray(
            [
                [0.98, 0.02, 0.00],
                [0.03, 0.97, 0.00],
                [0.04, 0.96, 0.00],
            ],
            dtype=np.float64,
        )
        train_outputs = SupervisedPredictionOutputs(
            backdoor_scores=1.0 - train_probabilities[:, 0],
            predicted_labels=np.argmax(train_probabilities, axis=1).astype(np.int32),
            probabilities=train_probabilities,
            logits=None,
        )
        train_labels = np.asarray([0, 1, 1], dtype=np.int32)

        config = _build_open_set_unknown_attack_config(
            train_labels=train_labels,
            train_outputs=train_outputs,
            task_spec=task_spec,
        )
        self.assertIsNotNone(config)
        self.assertEqual(config["known_attack_names"], ["toxic_backdoors_hard"])

        infer_probabilities = np.asarray(
            [
                [0.04, 0.90, 0.06],
                [0.02, 0.97, 0.01],
                [0.99, 0.01, 0.00],
            ],
            dtype=np.float64,
        )
        infer_outputs = SupervisedPredictionOutputs(
            backdoor_scores=1.0 - infer_probabilities[:, 0],
            predicted_labels=np.argmax(infer_probabilities, axis=1).astype(np.int32),
            probabilities=infer_probabilities,
            logits=None,
        )
        open_set = _apply_open_set_unknown_attack_rule(
            outputs=infer_outputs,
            task_spec=task_spec,
            config=config,
        )
        self.assertIsNotNone(open_set)
        unknown_index = int(config["unknown_class_index"])
        self.assertEqual(open_set["predicted_labels"].tolist(), [unknown_index, 1, 0])

        true_open_labels = _open_set_true_labels(
            np.asarray([2, 1, 0], dtype=np.int32),
            task_spec=task_spec,
            config=config,
        )
        self.assertEqual(true_open_labels.tolist(), [unknown_index, 1, 0])

    def test_open_set_unknown_attack_rule_maps_unseen_attack_to_unknown(self):
        task_spec = _resolve_supervised_task_spec(
            task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
            multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
        )
        train_probabilities = np.asarray(
            [
                [0.98, 0.01, 0.01, 0.00, 0.00],
                [0.02, 0.96, 0.02, 0.00, 0.00],
                [0.02, 0.03, 0.95, 0.00, 0.00],
                [0.03, 0.02, 0.95, 0.00, 0.00],
            ],
            dtype=np.float64,
        )
        train_outputs = SupervisedPredictionOutputs(
            backdoor_scores=1.0 - train_probabilities[:, 0],
            predicted_labels=np.argmax(train_probabilities, axis=1).astype(np.int32),
            probabilities=train_probabilities,
            logits=None,
        )
        train_labels = np.asarray([0, 1, 2, 2], dtype=np.int32)

        config = _build_open_set_unknown_attack_config(
            train_labels=train_labels,
            train_outputs=train_outputs,
            task_spec=task_spec,
        )
        self.assertIsNotNone(config)
        self.assertEqual(config["known_attack_names"], ["RIPPLE", "insertsent"])

        infer_probabilities = np.asarray(
            [
                [0.03, 0.10, 0.15, 0.02, 0.70],
                [0.02, 0.97, 0.01, 0.00, 0.00],
                [0.99, 0.01, 0.00, 0.00, 0.00],
            ],
            dtype=np.float64,
        )
        infer_outputs = SupervisedPredictionOutputs(
            backdoor_scores=1.0 - infer_probabilities[:, 0],
            predicted_labels=np.argmax(infer_probabilities, axis=1).astype(np.int32),
            probabilities=infer_probabilities,
            logits=None,
        )
        open_set = _apply_open_set_unknown_attack_rule(
            outputs=infer_outputs,
            task_spec=task_spec,
            config=config,
        )
        self.assertIsNotNone(open_set)
        unknown_index = int(config["unknown_class_index"])
        self.assertEqual(open_set["predicted_labels"].tolist(), [unknown_index, 1, 0])
        self.assertEqual(config["unknown_class_name"], OPEN_SET_UNKNOWN_ATTACK_NAME)

        true_open_labels = _open_set_true_labels(
            np.asarray([4, 1, 0], dtype=np.int32),
            task_spec=task_spec,
            config=config,
        )
        self.assertEqual(true_open_labels.tolist(), [unknown_index, 1, 0])

    def test_default_task_spec_remains_binary(self):
        task_spec = _resolve_supervised_task_spec(task_mode=None, multiclass_attack_names=None)
        self.assertTrue(task_spec.is_binary)
        self.assertEqual(task_spec.selection_metric_name, "roc_auc")

    def test_cnn_candidate_params_can_be_driven_by_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixed_path = tmp_path / "cnn_fixed.json"
            fixed_path.write_text(
                json.dumps(
                    {
                        "conv_channels": [48],
                        "num_conv_layers": [2],
                        "kernel_size": [5],
                        "dropout": [0.15],
                        "learning_rate": [0.0007],
                        "weight_decay": [0.0002],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            fixed_params = registry_mod.candidate_params("cnn_1d", cnn_hyperparams=fixed_path)
            self.assertEqual(
                fixed_params,
                [
                    {
                        "conv_channels": 48,
                        "num_conv_layers": 2,
                        "kernel_size": 5,
                        "dropout": 0.15,
                        "learning_rate": 0.0007,
                        "weight_decay": 0.0002,
                    }
                ],
            )

            grid_path = tmp_path / "cnn_grid.json"
            grid_path.write_text(
                json.dumps(
                    {
                        "conv_channels": [32, 64],
                        "num_conv_layers": [2],
                        "kernel_size": [3],
                        "dropout": [0.1],
                        "learning_rate": [0.001],
                        "weight_decay": [0.0],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            grid_params = registry_mod.candidate_params("cnn_1d", cnn_hyperparams=grid_path)
            self.assertEqual(len(grid_params), 2)

    def test_supervised_prepare_uses_fixed_cnn_hyperparams_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "multiclass_joint_manifest.json"
            train_entries, infer_entries, rows = build_multiclass_joint_rows(
                train_per_dataset=2,
                infer_per_dataset=1,
            )
            write_joint_manifest_by_name(
                manifest_path,
                train_entries=train_entries,
                infer_entries=infer_entries,
            )

            external_dir = tmp_path / "external_layer_sequence"
            feature_path = build_layer_sequence_bundle(external_dir, rows)
            cnn_hyperparams_path = tmp_path / "cnn_fixed.json"
            cnn_hyperparams_path.write_text(
                json.dumps(
                    {
                        "conv_channels": [48],
                        "num_conv_layers": [2],
                        "kernel_size": [5],
                        "dropout": [0.15],
                        "learning_rate": [0.0007],
                        "weight_decay": [0.0002],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            runs_root = tmp_path / "runs"
            prepared = run_supervised_pipeline(
                manifest_json=manifest_path,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id="multiclass_cnn_fixed_prepare",
                model_name="cnn_1d",
                spectral_features=["energy", "stable_rank"],
                spectral_sv_top_k=1,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="append",
                spectral_entrywise_delta_mode="none",
                stream_block_size=131072,
                dtype_name="float32",
                cv_folds=2,
                random_state=7,
                train_split_percent=100,
                calibration_split_percent=None,
                accepted_fpr=None,
                split_by_folder=False,
                cv_random_states=None,
                n_jobs=1,
                score_percentiles=[90.0, 95.0],
                feature_file=feature_path,
                tuning_executor="local",
                slurm_partition="cpu",
                slurm_max_concurrent="1",
                slurm_cpus_per_task="1",
                finalize_export_shards=1,
                stage="prepare",
                run_dir=None,
                task_index=None,
                task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
                multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
                cnn_hyperparams=cnn_hyperparams_path,
                skip_feature_importance=False,
            )

            run_dir = Path(prepared["run_dir"])
            tuning_manifest = load_json(run_dir / "reports" / "tuning_manifest.json")
            self.assertEqual(len(tuning_manifest["tuning"]["tasks"]), 1)
            self.assertEqual(
                tuning_manifest["tuning"]["tasks"][0]["params"],
                {
                    "conv_channels": 48,
                    "num_conv_layers": 2,
                    "kernel_size": 5,
                    "dropout": 0.15,
                    "learning_rate": 0.0007,
                    "weight_decay": 0.0002,
                },
            )
            self.assertEqual(tuning_manifest["tuning"]["cnn_hyperparams"]["source"], "file")
            self.assertEqual(tuning_manifest["tuning"]["cnn_hyperparams"]["n_candidates"], 1)
            self.assertEqual(tuning_manifest["tuning"]["execution_mode"], "singleton_no_cv")
            self.assertEqual(tuning_manifest["tuning"]["cv_folds_resolved"], 0)
            self.assertEqual(tuning_manifest["tuning"]["cv_splits"], [])
            self.assertEqual(tuning_manifest["tuning"]["cv_split_groups"], [])
            self.assertEqual(tuning_manifest["tuning"]["estimated_total_fits"], 1)

            worker_result = run_supervised_pipeline(
                manifest_json=None,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id=None,
                model_name="cnn_1d",
                spectral_features=["energy", "stable_rank"],
                spectral_sv_top_k=1,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="append",
                spectral_entrywise_delta_mode="none",
                stream_block_size=131072,
                dtype_name="float32",
                cv_folds=2,
                random_state=7,
                train_split_percent=100,
                calibration_split_percent=None,
                accepted_fpr=None,
                split_by_folder=False,
                cv_random_states=None,
                n_jobs=1,
                score_percentiles=[90.0, 95.0],
                feature_file=None,
                tuning_executor="local",
                slurm_partition="cpu",
                slurm_max_concurrent="1",
                slurm_cpus_per_task="1",
                finalize_export_shards=1,
                stage="worker",
                run_dir=run_dir,
                task_index=0,
                task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
                multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
                cnn_hyperparams=None,
                skip_feature_importance=False,
            )

            task_result = load_json(Path(worker_result["result_path"]))
            self.assertEqual(task_result["status"], "ok")
            self.assertEqual(task_result["execution_mode"], "singleton_no_cv")
            self.assertEqual(task_result["fold_results"], [])
            self.assertEqual(task_result["seed_results"], [])
            self.assertIsNone(task_result["selection_metric_mean"])

    def test_supervised_multiclass_pipeline_runs_with_sklearn(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "multiclass_joint_manifest.json"
            train_entries, infer_entries, rows = build_multiclass_joint_rows(
                train_per_dataset=2,
                infer_per_dataset=1,
            )
            write_joint_manifest_by_name(
                manifest_path,
                train_entries=train_entries,
                infer_entries=infer_entries,
            )

            external_dir = tmp_path / "external_tabular"
            feature_path = build_tabular_bundle(external_dir, rows)
            runs_root = tmp_path / "runs"

            result = run_supervised_pipeline(
                manifest_json=manifest_path,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id="multiclass_sklearn",
                model_name="logistic_regression",
                spectral_features=["energy", "kurtosis"],
                spectral_sv_top_k=1,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="none",
                spectral_entrywise_delta_mode="none",
                stream_block_size=131072,
                dtype_name="float32",
                cv_folds=2,
                random_state=42,
                train_split_percent=100,
                calibration_split_percent=None,
                accepted_fpr=None,
                split_by_folder=False,
                cv_random_states=None,
                n_jobs=1,
                score_percentiles=[90.0, 95.0],
                feature_file=feature_path,
                tuning_executor="local",
                slurm_partition="cpu",
                slurm_max_concurrent="1",
                slurm_cpus_per_task="1",
                finalize_export_shards=1,
                stage="all",
                run_dir=None,
                task_index=None,
                task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
                multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
                skip_feature_importance=False,
            )

            run_dir = Path(result["run_dir"])
            report = load_json(run_dir / "reports" / "supervised_report.json")
            run_config = load_json(run_dir / "run_config.json")
            artifact_index = load_json(run_dir / "artifact_index.json")
            inference_rows = load_csv_rows(run_dir / "reports" / "inference_scores.csv")
            summary_markdown = (run_dir / "reports" / "results_summary.md").read_text(encoding="utf-8")

            self.assertEqual(report["task"]["task_mode"], SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS)
            self.assertEqual(report["task"]["class_names"], ["clean", "RIPPLE", "insertsent", "stybkd", "syntactic"])
            self.assertEqual(report["tuning"]["metric"], "macro_f1")
            self.assertEqual(
                report["fit_assessment"]["binary_projection"],
                BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
            )
            self.assertEqual(report["fit_assessment"]["score_definition"], "backdoor_score")
            self.assertIn("confusion_matrix", report["multiclass_assessment"]["train"])
            self.assertIn("per_class", report["multiclass_assessment"]["inference"])
            self.assertIn("predicted_class_distribution", report["multiclass_assessment"]["inference"])
            self.assertEqual(report["open_set_assessment"]["config"]["unknown_class_name"], OPEN_SET_UNKNOWN_ATTACK_NAME)
            self.assertIn("per_class", report["open_set_assessment"]["inference"])
            self.assertIn("open_set_unknown", report["results_summary"])
            self.assertIn("Open-Set Unknown Attack", summary_markdown)
            self.assertIn(OPEN_SET_UNKNOWN_ATTACK_NAME, summary_markdown)
            self.assertIn("prob_clean", inference_rows[0])
            self.assertIn("prob_RIPPLE", inference_rows[0])
            self.assertIn("open_set_prediction_name", inference_rows[0])
            self.assertIn("attacks", report["attack_analysis"]["inference"])
            self.assertEqual(run_config["winner_feature_weights_mode"], "unsupported_for_task_mode")
            self.assertTrue(
                any("does not yet support feature-importance export" in str(x) for x in report["warnings"])
            )
            self.assertNotIn("winner_feature_weights_coefficients_csv", artifact_index)
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_coefficients.csv").exists())

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_supervised_multiclass_pipeline_runs_with_cnn(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "multiclass_joint_manifest.json"
            train_entries, infer_entries, rows = build_multiclass_joint_rows(
                train_per_dataset=2,
                infer_per_dataset=1,
            )
            write_joint_manifest_by_name(
                manifest_path,
                train_entries=train_entries,
                infer_entries=infer_entries,
            )

            external_dir = tmp_path / "external_layer_sequence"
            feature_path = build_layer_sequence_bundle(external_dir, rows)
            cnn_hyperparams_path = tmp_path / "cnn_fixed.json"
            cnn_hyperparams_path.write_text(
                json.dumps(
                    {
                        "conv_channels": [48],
                        "num_conv_layers": [2],
                        "kernel_size": [5],
                        "dropout": [0.15],
                        "learning_rate": [0.0007],
                        "weight_decay": [0.0002],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            runs_root = tmp_path / "runs"

            original_create = pipeline_mod.create

            def fast_create(name: str, params: dict[str, Any], random_state: int, task_spec=None):
                model = original_create(name, params=params, random_state=random_state, task_spec=task_spec)
                if name == "cnn_1d":
                    model.max_epochs = 3
                    model.patience = 1
                    model.batch_size = 8
                return model

            with mock.patch.object(pipeline_mod, "create", side_effect=fast_create):
                result = run_supervised_pipeline(
                    manifest_json=manifest_path,
                    dataset_root=tmp_path,
                    output_root=runs_root,
                    run_id="multiclass_cnn",
                    model_name="cnn_1d",
                    spectral_features=["energy", "stable_rank"],
                    spectral_sv_top_k=1,
                    spectral_moment_source="sv",
                    spectral_qv_sum_mode="append",
                    spectral_entrywise_delta_mode="none",
                    stream_block_size=131072,
                    dtype_name="float32",
                    cv_folds=2,
                    random_state=7,
                    train_split_percent=100,
                    calibration_split_percent=None,
                    accepted_fpr=None,
                    split_by_folder=False,
                    cv_random_states=None,
                    n_jobs=1,
                    score_percentiles=[90.0, 95.0],
                    feature_file=feature_path,
                    tuning_executor="local",
                    slurm_partition="cpu",
                    slurm_max_concurrent="1",
                    slurm_cpus_per_task="1",
                    finalize_export_shards=1,
                    stage="all",
                    run_dir=None,
                    task_index=None,
                    task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
                    multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
                    cnn_hyperparams=cnn_hyperparams_path,
                    skip_feature_importance=False,
                )

            run_dir = Path(result["run_dir"])
            report = load_json(run_dir / "reports" / "supervised_report.json")
            run_config = load_json(run_dir / "run_config.json")
            try:
                checkpoint = torch.load(
                    run_dir / "models" / "best_model.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(run_dir / "models" / "best_model.pt", map_location="cpu")

            self.assertEqual(run_config["winner_feature_weights_mode"], "unsupported_for_task_mode")
            self.assertEqual(run_config["execution_mode"], "singleton_no_cv")
            self.assertEqual(checkpoint["task"]["task_mode"], SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS)
            self.assertEqual(checkpoint["config"]["num_classes"], 5)
            self.assertEqual(checkpoint["class_names"], ["clean", "RIPPLE", "insertsent", "stybkd", "syntactic"])
            self.assertEqual(report["tuning"]["metric"], "macro_f1")
            self.assertEqual(report["tuning"]["execution_mode"], "singleton_no_cv")
            self.assertEqual(report["tuning"]["tasks_total"], 1)
            self.assertEqual(report["tuning"]["estimated_total_fits"], 1)
            self.assertIsNone(report["tuning"]["winner"]["selection_metric_mean"])
            self.assertEqual(report["tuning"]["candidates"][0]["fold_results"], [])
            self.assertIsNotNone(report["multiclass_assessment"])
            self.assertIn("confusion_matrix", report["multiclass_assessment"]["train"])
            self.assertIn("per_class", report["multiclass_assessment"]["inference"])
            self.assertTrue(
                any("does not yet support feature-importance export" in str(x) for x in report["warnings"])
            )


if __name__ == "__main__":
    unittest.main()
