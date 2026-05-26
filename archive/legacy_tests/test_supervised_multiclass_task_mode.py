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
from upeftguard.supervised.cnn import (
    compute_balanced_class_loss_config,
    compute_balanced_rank_label_loss_config,
)
from upeftguard.supervised.interfaces import (
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    ATTACK_FAMILY_MULTICLASS_ATTACKS,
    BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
    SupervisedFeatureBundle,
    SupervisedPredictionOutputs,
)
from upeftguard.supervised.pipeline import (
    SELECTION_METRIC_BINARY_AUROC,
    SELECTION_METRIC_MACRO_F1,
    OPEN_SET_UNKNOWN_ATTACK_NAME,
    _apply_open_set_unknown_attack_rule,
    _build_cv_splits,
    _build_open_set_unknown_attack_config,
    _folder_label_stratification_keys,
    _labels_from_items,
    _open_set_true_labels,
    _resolve_selection_metric,
    _resolve_supervised_task_spec,
    run_supervised_pipeline,
)
from upeftguard.unsupervised.analysis import run_supervised_cnn_feature_tsne_pipeline
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


def build_rank_joint_rows(
    *,
    train_ranks: tuple[int, ...],
    infer_ranks: tuple[int, ...],
    train_per_label: int = 2,
    infer_per_label: int = 1,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    train_entries: list[str] = []
    infer_entries: list[str] = []
    rows: list[dict[str, Any]] = []

    for rank in train_ranks:
        subset_name = f"llama2_7b_toxic_backdoors_hard_rank{int(rank)}_qv"
        for label in (0, 1):
            for sample_idx in range(train_per_label):
                model_name = f"{subset_name}_label{label}_{sample_idx}"
                train_entries.append(f"{subset_name}/{model_name}")
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": "tbh",
                        "class_name": "clean" if label == 0 else "backdoored",
                        "class_index": label,
                    }
                )

    for rank in infer_ranks:
        subset_name = f"llama2_7b_toxic_backdoors_hard_rank{int(rank)}_qv"
        for label in (0, 1):
            for sample_idx in range(infer_per_label):
                row_index = 100 + sample_idx
                model_name = f"{subset_name}_label{label}_{row_index}"
                infer_entries.append(f"{subset_name}/{model_name}")
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": "tbh",
                        "class_name": "clean" if label == 0 else "backdoored",
                        "class_index": label,
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
    def test_split_by_folder_cv_uses_folder_label_strata(self):
        items: list[ManifestItem] = []
        labels: list[int] = []
        for subset_name in ("rank16", "rank256"):
            for label in (0, 1):
                for sample_idx in range(6):
                    items.append(make_manifest_item(subset_name, label, sample_idx))
                    labels.append(label)

        train_indices = np.arange(len(items), dtype=np.int64)
        label_array = np.asarray(labels, dtype=np.int32)
        strata = _folder_label_stratification_keys(
            items=items,
            candidate_indices=train_indices,
            labels=label_array,
        )
        splits = _build_cv_splits(
            train_indices=train_indices,
            train_labels=strata,
            cv_folds=3,
            random_state=7,
        )

        for split in splits:
            valid_indices = np.asarray(split["valid_indices"], dtype=np.int64)
            valid_strata = strata[valid_indices]
            unique, counts = np.unique(valid_strata, return_counts=True)
            self.assertEqual(len(unique), 4)
            self.assertEqual(set(counts.tolist()), {2})

    def test_class_weight_loss_config_balances_binary_counts(self):
        task_spec = _resolve_supervised_task_spec(task_mode="binary", multiclass_attack_names=None)
        config = compute_balanced_class_loss_config(
            np.asarray([0, 0, 0, 1], dtype=np.int32),
            task_spec=task_spec,
        )

        self.assertEqual(config["class_counts"], [3, 1])
        self.assertAlmostEqual(config["binary_pos_weight"], 3.0)
        self.assertAlmostEqual(config["class_weights"][0], 2.0 / 3.0)
        self.assertAlmostEqual(config["class_weights"][1], 2.0)

    def test_rank_label_weight_loss_config_balances_rank_label_buckets(self):
        task_spec = _resolve_supervised_task_spec(task_mode="binary", multiclass_attack_names=None)
        weights, config = compute_balanced_rank_label_loss_config(
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
            np.asarray([256, 256, 256, 8, 8, 8], dtype=np.int32),
            task_spec=task_spec,
        )

        self.assertEqual(config["class_counts"], [3, 3])
        self.assertEqual(
            [(row["rank"], row["label"], row["count"]) for row in config["rank_label_counts"]],
            [(8, 1, 3), (256, 0, 3)],
        )
        self.assertTrue(np.allclose(weights, np.ones(6, dtype=np.float32)))

        weights, config = compute_balanced_rank_label_loss_config(
            np.asarray([0, 0, 0, 1], dtype=np.int32),
            np.asarray([256, 256, 256, 8], dtype=np.int32),
            task_spec=task_spec,
        )
        self.assertEqual(
            [(row["rank"], row["label"], row["count"]) for row in config["rank_label_counts"]],
            [(8, 1, 1), (256, 0, 3)],
        )
        self.assertTrue(np.allclose(weights, np.asarray([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0])))

    def _prepare_rank_dann_manifest(
        self,
        tmp_path: Path,
        *,
        train_ranks: tuple[int, ...],
        infer_ranks: tuple[int, ...],
        run_id: str,
    ) -> dict[str, Any]:
        manifest_path = tmp_path / f"{run_id}.json"
        train_entries, infer_entries, rows = build_rank_joint_rows(
            train_ranks=train_ranks,
            infer_ranks=infer_ranks,
        )
        write_joint_manifest_by_name(
            manifest_path,
            train_entries=train_entries,
            infer_entries=infer_entries,
        )
        feature_path = build_layer_sequence_bundle(tmp_path / f"{run_id}_features", rows)
        cnn_hyperparams_path = tmp_path / f"{run_id}_cnn_fixed.json"
        cnn_hyperparams_path.write_text(
            json.dumps(
                {
                    "conv_channels": [32],
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
        return run_supervised_pipeline(
            manifest_json=manifest_path,
            dataset_root=tmp_path,
            output_root=tmp_path / "runs",
            run_id=run_id,
            model_name="cnn_1d_dann",
            spectral_features=["energy", "stable_rank"],
            spectral_sv_top_k=1,
            spectral_moment_source="sv",
            spectral_qv_sum_mode="append",
            spectral_entrywise_delta_mode="auto",
            stream_block_size=131072,
            dtype_name="float32",
            cv_folds=2,
            random_state=42,
            train_split_percent=100,
            calibration_split_percent=None,
            accepted_fpr=None,
            split_by_folder=False,
            cv_random_states=[42],
            n_jobs=1,
            score_percentiles=[90.0],
            feature_file=feature_path,
            tuning_executor="local",
            slurm_partition="extra",
            slurm_max_concurrent="auto",
            slurm_cpus_per_task="auto",
            finalize_export_shards=1,
            stage="prepare",
            run_dir=None,
            task_index=None,
            cnn_hyperparams=cnn_hyperparams_path,
        )

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

    def test_multiclass_can_select_by_binary_auroc(self):
        task_spec = _resolve_supervised_task_spec(
            task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
            multiclass_attack_names=list(ATTACK_FAMILY_MULTICLASS_ATTACKS),
        )
        self.assertEqual(_resolve_selection_metric(None, task_spec=task_spec), SELECTION_METRIC_MACRO_F1)
        self.assertEqual(
            _resolve_selection_metric(SELECTION_METRIC_BINARY_AUROC, task_spec=task_spec),
            SELECTION_METRIC_BINARY_AUROC,
        )
        self.assertEqual(
            _resolve_selection_metric("roc_auc", task_spec=task_spec),
            SELECTION_METRIC_BINARY_AUROC,
        )

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

    def test_supervised_prepare_builds_cnn_dann_rank_adversarial_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "rank_dann_joint_manifest.json"
            train_entries: list[str] = []
            infer_entries: list[str] = []
            rows: list[dict[str, Any]] = []

            for rank in (256, 8):
                subset_name = f"llama2_7b_toxic_backdoors_hard_rank{rank}_qv"
                for label in (0, 1):
                    for sample_idx in range(2):
                        model_name = f"{subset_name}_label{label}_{sample_idx}"
                        train_entries.append(f"{subset_name}/{model_name}")
                        rows.append(
                            {
                                "model_name": model_name,
                                "dataset_name": "tbh",
                                "class_name": "clean" if label == 0 else "backdoored",
                                "class_index": label,
                            }
                        )

            subset_name = "llama2_7b_toxic_backdoors_hard_rank16_qv"
            for label in (0, 1):
                model_name = f"{subset_name}_label{label}_0"
                infer_entries.append(f"{subset_name}/{model_name}")
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": "tbh",
                        "class_name": "clean" if label == 0 else "backdoored",
                        "class_index": label,
                    }
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
                        "conv_channels": [32],
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

            prepared = run_supervised_pipeline(
                manifest_json=manifest_path,
                dataset_root=tmp_path,
                output_root=tmp_path / "runs",
                run_id="rank_dann_prepare",
                model_name="cnn_1d_dann",
                spectral_features=["energy", "stable_rank"],
                spectral_sv_top_k=1,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="append",
                spectral_entrywise_delta_mode="auto",
                stream_block_size=131072,
                dtype_name="float32",
                cv_folds=2,
                random_state=42,
                train_split_percent=100,
                calibration_split_percent=None,
                accepted_fpr=None,
                split_by_folder=False,
                cv_random_states=[42],
                n_jobs=1,
                score_percentiles=[90.0],
                feature_file=feature_path,
                tuning_executor="local",
                slurm_partition="extra",
                slurm_max_concurrent="auto",
                slurm_cpus_per_task="auto",
                finalize_export_shards=1,
                stage="prepare",
                run_dir=None,
                task_index=None,
                cnn_hyperparams=cnn_hyperparams_path,
            )

            tuning_manifest = load_json(Path(prepared["tuning_manifest"]))
            domain = tuning_manifest["domain_adaptation"]
            self.assertTrue(domain["enabled"])
            self.assertEqual(domain["source_rank"], 256)
            self.assertEqual(domain["train_ranks"], [256, 8])
            self.assertEqual(domain["target_train_ranks"], [8])
            self.assertEqual(domain["zero_shot_inference_ranks"], [16])
            self.assertEqual(domain["domain_rank_values"], [256, 8])
            self.assertEqual(domain["domain_class_names"], ["rank_256", "rank_8"])
            self.assertEqual(domain["label_loss_scope"], "all_training_ranks")
            self.assertEqual(domain["domain_loss_weight"], 1.0)
            self.assertEqual(domain["lambda_schedule"]["gamma"], 10.0)
            self.assertEqual(domain["learning_rate_schedule"]["type"], "fixed")
            self.assertEqual(tuning_manifest["tuning"]["model_names"], ["cnn_1d_dann"])
            self.assertEqual(len(tuning_manifest["tuning"]["tasks"]), 1)
            self.assertEqual(tuning_manifest["tuning"]["execution_mode"], "singleton_no_cv")
            self.assertEqual(tuning_manifest["tuning"]["tasks"][0]["params"]["source_rank"], 256)
            self.assertNotIn("dann_target_adaptation_percent", tuning_manifest["tuning"]["tasks"][0]["params"])

            domain_labels = np.load(domain["domain_labels_path"]).astype(np.int64)
            self.assertEqual(domain_labels.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
            self.assertEqual(domain["train_indices"], list(range(8)))
            self.assertEqual(domain["source_train_indices"], [0, 1, 2, 3])
            self.assertEqual(domain["target_train_indices"], [4, 5, 6, 7])
            self.assertEqual(domain["inference_indices"], [8, 9])
            self.assertEqual(tuning_manifest["data"]["train_indices"], list(range(8)))
            self.assertEqual(tuning_manifest["data"]["infer_indices"], [8, 9])
            self.assertFalse(
                set(tuning_manifest["data"]["train_indices"]) & set(tuning_manifest["data"]["infer_indices"])
            )
            self.assertTrue(
                any("rank-adversarial supervised training" in str(x) for x in tuning_manifest["warnings"])
            )

    def test_supervised_prepare_rejects_cnn_dann_without_source_rank_in_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "source rank"):
                self._prepare_rank_dann_manifest(
                    tmp_path,
                    train_ranks=(8,),
                    infer_ranks=(16,),
                    run_id="rank_dann_missing_source",
                )

    def test_supervised_prepare_rejects_cnn_dann_without_target_train_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "at least one non-source rank"):
                self._prepare_rank_dann_manifest(
                    tmp_path,
                    train_ranks=(256,),
                    infer_ranks=(16,),
                    run_id="rank_dann_missing_target_train",
                )

    def test_supervised_prepare_rejects_cnn_dann_without_zero_shot_inference_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "infer.*empty"):
                self._prepare_rank_dann_manifest(
                    tmp_path,
                    train_ranks=(256, 8),
                    infer_ranks=(),
                    run_id="rank_dann_missing_zero_shot",
                )

    def test_supervised_prepare_rejects_cnn_dann_train_infer_rank_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "disjoint.*overlap"):
                self._prepare_rank_dann_manifest(
                    tmp_path,
                    train_ranks=(256, 8),
                    infer_ranks=(8,),
                    run_id="rank_dann_rank_overlap",
                )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_cnn_checkpoint_roundtrip_restores_feature_extractor_embeddings(self):
        from upeftguard.supervised.cnn import CNN1DSupervisedModel, load_cnn_checkpoint

        rng = np.random.default_rng(321)
        values = rng.normal(size=(6, 3, 2, 2)).astype(np.float32)
        group_mask = np.ones((6, 3), dtype=bool)
        value_mask = np.ones_like(values, dtype=bool)
        bundle = SupervisedFeatureBundle(
            values=values,
            representation_kind=ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
            metadata={
                "representation_kind": ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
                "depth_labels": [f"encoder.layer{i}" for i in range(3)],
                "slot_names": ["self.q", "self.v"],
                "emitted_feature_names": ["energy", "stable_rank"],
            },
            group_mask=group_mask,
            value_mask=value_mask,
            group_names=[[f"encoder.layer{i}" for i in range(3)] for _ in range(6)],
        )
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
        model = CNN1DSupervisedModel(
            conv_channels=4,
            num_conv_layers=1,
            kernel_size=3,
            dropout=0.0,
            learning_rate=0.001,
            weight_decay=0.0,
            random_state=11,
            max_epochs=2,
            batch_size=3,
            patience=2,
        )
        model.fit(bundle, labels)
        before = model.extract_features(bundle)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best_model.pt"
            model.save(checkpoint_path)
            loaded = load_cnn_checkpoint(checkpoint_path)
            after = loaded.extract_features(bundle)

        self.assertEqual(before.shape, (6, 8))
        self.assertEqual(after.shape, before.shape)
        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_cnn_dann_tiny_fit_uses_all_labels_by_default_and_rank_domains(self):
        from upeftguard.supervised.cnn import CNN1DDANNSupervisedModel, load_cnn_checkpoint

        rng = np.random.default_rng(123)
        values = rng.normal(size=(8, 4, 2, 3)).astype(np.float32)
        group_mask = np.ones((8, 4), dtype=bool)
        value_mask = np.ones_like(values, dtype=bool)
        bundle = SupervisedFeatureBundle(
            values=values,
            representation_kind=ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
            metadata={
                "representation_kind": ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
                "depth_labels": [f"encoder.layer{i}" for i in range(4)],
                "slot_names": ["self.q", "self.v"],
                "emitted_feature_names": ["energy", "stable_rank", "sv1"],
            },
            group_mask=group_mask,
            value_mask=value_mask,
            group_names=[[f"encoder.layer{i}" for i in range(4)] for _ in range(8)],
        )
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        domain_labels = np.asarray([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64)

        model = CNN1DDANNSupervisedModel(
            conv_channels=4,
            num_conv_layers=1,
            kernel_size=3,
            dropout=0.0,
            learning_rate=0.001,
            weight_decay=0.0,
            random_state=7,
            max_epochs=2,
            batch_size=4,
            patience=2,
        )
        model.fit(
            bundle,
            labels,
            domain_labels=domain_labels,
            domain_class_names=("rank_256", "rank_8", "rank_16"),
            domain_rank_values=(256, 8, 16),
        )

        logits = model.decision_function(bundle)
        self.assertEqual(logits.shape, (8,))
        self.assertEqual(model.domain_class_names_, ("rank_256", "rank_8", "rank_16"))
        self.assertEqual(model._fit_summary["domain_loss_weight"], 1.0)
        self.assertEqual(model._fit_summary["label_loss_scope"], "all_training_ranks")
        self.assertEqual(model._fit_summary["history"][-1]["domain_rows"], 8.0)
        self.assertEqual(model._fit_summary["history"][-1]["label_rows"], 8.0)

        before = model.extract_features(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best_model.pt"
            model.save(checkpoint_path)
            loaded = load_cnn_checkpoint(checkpoint_path)
            after = loaded.extract_features(bundle)
        self.assertEqual(loaded.domain_class_names_, ("rank_256", "rank_8", "rank_16"))
        self.assertEqual(after.shape, before.shape)

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
            with self.assertRaisesRegex(ValueError, "requires a CNN winner"):
                run_supervised_cnn_feature_tsne_pipeline(
                    run_dir=run_dir,
                    output_root=tmp_path / "analysis_runs",
                    run_id="not_cnn",
                    max_iter=250,
                )

    def test_supervised_multiclass_pipeline_can_select_binary_auroc(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "multiclass_binary_auroc_manifest.json"
            train_entries, infer_entries, rows = build_multiclass_joint_rows(
                train_per_dataset=2,
                infer_per_dataset=1,
            )
            write_joint_manifest_by_name(
                manifest_path,
                train_entries=train_entries,
                infer_entries=infer_entries,
            )

            result = run_supervised_pipeline(
                manifest_json=manifest_path,
                dataset_root=tmp_path,
                output_root=tmp_path / "runs",
                run_id="multiclass_binary_auroc",
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
                cv_random_states=[42],
                n_jobs=1,
                score_percentiles=[90.0],
                feature_file=build_tabular_bundle(tmp_path / "features", rows),
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
                selection_metric=SELECTION_METRIC_BINARY_AUROC,
            )

            report = load_json(Path(result["run_dir"]) / "reports" / "supervised_report.json")
            winner = report["tuning"]["winner"]
            self.assertEqual(report["tuning"]["metric"], SELECTION_METRIC_BINARY_AUROC)
            self.assertEqual(winner["selection_metric_name"], SELECTION_METRIC_BINARY_AUROC)
            self.assertIn("binary_auroc_mean", winner)
            self.assertIn("macro_f1_mean", winner)
            self.assertAlmostEqual(winner["selection_metric_mean"], winner["binary_auroc_mean"])

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

            tsne_result = run_supervised_cnn_feature_tsne_pipeline(
                run_dir=run_dir,
                output_root=tmp_path / "analysis_runs",
                run_id="multiclass_cnn_feature_tsne",
                perplexity=5.0,
                max_iter=250,
                random_state=7,
            )
            tsne_run_dir = Path(tsne_result["run_dir"])
            tsne_report = load_json(Path(tsne_result["report"]))
            tsne_artifacts = load_json(tsne_run_dir / "artifact_index.json")
            embeddings = np.load(tsne_run_dir / "features" / "cnn_feature_embeddings.npy")
            labels = np.load(tsne_run_dir / "features" / "cnn_feature_embedding_labels.npy")
            rows = load_csv_rows(tsne_run_dir / "reports" / "cnn_feature_embedding_rows.csv")

            self.assertEqual(embeddings.shape[0], report["data_info"]["n_train"] + report["data_info"]["n_inference"])
            self.assertEqual(labels.shape[0], embeddings.shape[0])
            self.assertEqual(len(rows), embeddings.shape[0])
            clean_rows = [row for row in rows if row["task_label_name"] == "clean"]
            backdoor_rows = [row for row in rows if row["task_label_name"] != "clean"]
            self.assertTrue(clean_rows)
            self.assertTrue(backdoor_rows)
            self.assertTrue(all(row["effective_attack_name"] == "clean" for row in clean_rows))
            self.assertTrue(
                any(row["attack_name"] != "clean" and row["effective_attack_name"] == "clean" for row in clean_rows)
            )
            self.assertTrue(all(row["effective_attack_name"] == row["attack_name"] for row in backdoor_rows))
            self.assertIn("head_projection", rows[0])
            self.assertIn("head_margin", rows[0])
            self.assertIn("prediction_entropy", rows[0])
            self.assertEqual(tsne_report["analysis"], "supervised_cnn_feature_tsne")
            self.assertEqual({view["view"] for view in tsne_report["views"]}, {"combined", "train", "inference"})
            expected_attack_plot_counts = {
                "combined": sum(row["binary_label"] == "1" for row in rows),
                "train": sum(row["partition"] == "train" and row["binary_label"] == "1" for row in rows),
                "inference": sum(row["partition"] == "inference" and row["binary_label"] == "1" for row in rows),
            }
            self.assertIn("cnn_feature_embeddings", tsne_artifacts)
            self.assertIn("cnn_feature_tsne_report", tsne_artifacts)
            self.assertIn("cnn_feature_score_plots", tsne_artifacts)
            self.assertIn("cnn_feature_margin_plots", tsne_artifacts)
            self.assertIn("cnn_feature_head_projection_plots", tsne_artifacts)
            self.assertTrue(Path(tsne_report["artifacts"]["tsne_plot_dir"]).name == "tsne")
            self.assertTrue(Path(tsne_report["artifacts"]["plot_dir"]).name == "cnn_feature_extractor")
            behavior_plots = tsne_report["behavior_plots"]
            self.assertTrue(Path(behavior_plots["scores"]["backdoor_score_histogram"]).exists())
            self.assertTrue(Path(behavior_plots["scores"]["backdoor_score_ecdf"]).exists())
            self.assertTrue(Path(behavior_plots["scores"]["backdoor_score_by_partition_and_label"]).exists())
            self.assertTrue(Path(behavior_plots["margins"]["head_margin_histogram"]).exists())
            self.assertTrue(Path(behavior_plots["head_projection"]["head_projection_scatter"]).exists())
            for view in tsne_report["views"]:
                self.assertIn("embedding_csv", view)
                self.assertTrue(Path(view["embedding_csv"]).exists())
                self.assertEqual(view["attack_plot_field"], "attack_name")
                self.assertEqual(view["attack_plot_scope"], "backdoor_rows_only")
                self.assertEqual(view["attack_plot_n_samples"], expected_attack_plot_counts[view["view"]])
                coordinate_rows = load_csv_rows(Path(view["embedding_csv"]))
                self.assertIn("effective_attack_name", coordinate_rows[0])
                self.assertTrue(Path(view["plots"]["partition"]).exists())


if __name__ == "__main__":
    unittest.main()
