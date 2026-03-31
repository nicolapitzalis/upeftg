import csv
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from safetensors.numpy import load_file, save_file
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import upeftguard.cli as cli_mod
from upeftguard.features.delta import block_delta_singular_values, load_delta_block_schema, top_k_singular_values
from upeftguard.features.norms import summarize_array_moments
from upeftguard.features.spectral import extract_spectral_features
from upeftguard.features.svd import extract_svd_embeddings
from upeftguard.supervised.pipeline import _load_features_for_tuning_manifest, run_supervised_pipeline
from upeftguard.supervised.registry import candidate_params, create, normalization_policy, registered_models
from upeftguard.utilities.artifacts.dataset_references import (
    build_dataset_reference_payload_from_items,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from upeftguard.utilities.artifacts.spectral_metadata import (
    dataset_layouts_from_source,
    load_spectral_metadata,
    write_spectral_metadata,
)
from upeftguard.utilities.core.manifest import parse_joint_manifest_json, parse_single_manifest_json
from upeftguard.utilities.maintenance.backfill_dataset_reference_reports import (
    backfill_dataset_reference_reports,
)
from upeftguard.utilities.merge.merge_feature_files import (
    finalize_schema_group_merge,
    merge_feature_files,
)
from upeftguard.utilities.merge.merge_spectral_shards import merge_spectral_shards
from upeftguard.utilities.merge.prepare_spectral_shards import prepare_schema_sharded_manifests


REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def make_tiny_adapter_dataset(
    root: Path,
    *,
    q_out_dim: int = 4,
    v_out_dim: int = 3,
) -> Path:
    data_dir = root / "tiny_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)

    for label in [0, 1]:
        for idx in [0, 1, 2, 3]:
            model_dir = data_dir / f"tiny_label{label}_{idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensors = {}
            for layer in range(2):
                for module_name, out_dim in [("q_proj", q_out_dim), ("v_proj", v_out_dim)]:
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


def make_tiny_roberta_adapter_dataset(
    root: Path,
    *,
    out_dim: int = 4,
) -> Path:
    data_dir = root / "tiny_roberta_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(124)

    for label in [0, 1]:
        for idx in [0, 1]:
            model_dir = data_dir / f"tiny_roberta_label{label}_{idx}"
            best_model_dir = model_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)

            tensors = {}
            for layer in range(2):
                for module_name in ["query", "value"]:
                    a_key = (
                        f"base_model.model.roberta.encoder.layer.{layer}."
                        f"attention.self.{module_name}.lora_A.weight"
                    )
                    b_key = (
                        f"base_model.model.roberta.encoder.layer.{layer}."
                        f"attention.self.{module_name}.lora_B.weight"
                    )

                    a = rng.standard_normal((2, 4), dtype=np.float32)
                    b = rng.standard_normal((out_dim, 2), dtype=np.float32)

                    if label == 1:
                        a = a + 0.35
                        b = b + 0.15

                    tensors[a_key] = a
                    tensors[b_key] = b

            save_file(tensors, str(best_model_dir / "adapter_model.safetensors"))

    return data_dir


def make_tiny_t5_adapter_dataset(
    root: Path,
    *,
    out_dim: int = 4,
) -> Path:
    data_dir = root / "tiny_t5_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(456)

    for label in [0, 1]:
        for idx in [0, 1]:
            model_dir = data_dir / f"tiny_t5_label{label}_{idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensors = {}
            for block in range(2):
                for layer_idx, attention_name in [(0, "SelfAttention"), (1, "EncDecAttention")]:
                    for module_name in ["q", "v"]:
                        a_key = (
                            f"base_model.model.decoder.block.{block}.layer.{layer_idx}."
                            f"{attention_name}.{module_name}.lora_A.weight"
                        )
                        b_key = (
                            f"base_model.model.decoder.block.{block}.layer.{layer_idx}."
                            f"{attention_name}.{module_name}.lora_B.weight"
                        )

                        a = rng.standard_normal((2, 4), dtype=np.float32)
                        b = rng.standard_normal((out_dim, 2), dtype=np.float32)

                        if label == 1:
                            a = a + 0.25
                            b = b + 0.5

                        tensors[a_key] = a
                        tensors[b_key] = b

            save_file(tensors, str(model_dir / "adapter_model.safetensors"))

    return data_dir


def make_tiny_t5_encoder_decoder_adapter_dataset(
    root: Path,
    *,
    out_dim: int = 4,
) -> Path:
    data_dir = root / "tiny_t5_encoder_decoder_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(457)

    for label in [0, 1]:
        model_dir = data_dir / f"tiny_t5_encdec_label{label}_0"
        model_dir.mkdir(parents=True, exist_ok=True)

        tensors = {}
        for scope_name in ["encoder", "decoder"]:
            for module_name in ["q", "v"]:
                a_key = (
                    f"base_model.model.{scope_name}.block.0.layer.0."
                    f"SelfAttention.{module_name}.lora_A.weight"
                )
                b_key = (
                    f"base_model.model.{scope_name}.block.0.layer.0."
                    f"SelfAttention.{module_name}.lora_B.weight"
                )

                a = rng.standard_normal((2, 4), dtype=np.float32)
                b = rng.standard_normal((out_dim, 2), dtype=np.float32)
                if label == 1:
                    a = a + 0.2
                    b = b + 0.35

                tensors[a_key] = a
                tensors[b_key] = b

        for module_name in ["q", "v"]:
            a_key = (
                "base_model.model.decoder.block.0.layer.1."
                f"EncDecAttention.{module_name}.lora_A.weight"
            )
            b_key = (
                "base_model.model.decoder.block.0.layer.1."
                f"EncDecAttention.{module_name}.lora_B.weight"
            )

            a = rng.standard_normal((2, 4), dtype=np.float32)
            b = rng.standard_normal((out_dim, 2), dtype=np.float32)
            if label == 1:
                a = a + 0.15
                b = b + 0.1

            tensors[a_key] = a
            tensors[b_key] = b

        save_file(tensors, str(model_dir / "adapter_model.safetensors"))

    return data_dir


def make_tiny_chatglm_adapter_dataset(
    root: Path,
    *,
    out_dim: int = 6,
) -> Path:
    data_dir = root / "tiny_chatglm_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(789)

    for label in [0, 1]:
        for idx in [0, 1]:
            model_dir = data_dir / f"tiny_chatglm_label{label}_{idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensors = {}
            for layer in range(2):
                a_key = (
                    f"base_model.model.transformer.encoder.layers.{layer}."
                    "self_attention.query_key_value.lora_A.weight"
                )
                b_key = (
                    f"base_model.model.transformer.encoder.layers.{layer}."
                    "self_attention.query_key_value.lora_B.weight"
                )
                a = rng.standard_normal((2, 4), dtype=np.float32)
                b = rng.standard_normal((out_dim, 2), dtype=np.float32)

                if label == 1:
                    a = a + 0.1
                    b = b - 0.2

                tensors[a_key] = a
                tensors[b_key] = b

            save_file(tensors, str(model_dir / "adapter_model.safetensors"))

    return data_dir


def make_tiny_adalora_dataset(
    root: Path,
    *,
    out_dim: int = 4,
) -> Path:
    data_dir = root / "tiny_adalora_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(321)

    for label in [0, 1]:
        for idx in [0, 1]:
            model_dir = data_dir / f"tiny_adalora_label{label}_{idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensors = {}
            for layer in range(2):
                for module_name, rank in [("q_proj", 2), ("v_proj", 3)]:
                    prefix = f"base_model.model.model.layers.{layer}.self_attn.{module_name}"
                    a_key = prefix + ".lora_A"
                    b_key = prefix + ".lora_B"
                    e_key = prefix + ".lora_E"

                    a = rng.standard_normal((rank, 4), dtype=np.float32)
                    b = rng.standard_normal((out_dim, rank), dtype=np.float32)
                    e = rng.standard_normal((rank, 1), dtype=np.float32)

                    if label == 1:
                        a = a + 0.15
                        b = b + 0.2
                        e = e - 0.1

                    tensors[a_key] = a
                    tensors[b_key] = b
                    tensors[e_key] = e

            save_file(tensors, str(model_dir / "adapter_model.safetensors"))

    return data_dir


def make_tiny_variable_rank_adalora_dataset(
    root: Path,
    *,
    out_dim: int = 4,
) -> Path:
    data_dir = root / "tiny_adalora_variable_rank_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(654)
    rank_patterns = {
        0: [("q_proj", 2), ("v_proj", 3)],
        1: [("q_proj", 4), ("v_proj", 1)],
        2: [("q_proj", 2), ("v_proj", 3)],
        3: [("q_proj", 4), ("v_proj", 1)],
    }

    for idx in [0, 1, 2, 3]:
        label = idx % 2
        model_dir = data_dir / f"tiny_adalora_var_label{label}_{idx}"
        model_dir.mkdir(parents=True, exist_ok=True)

        tensors = {}
        for layer in range(2):
            for module_name, rank in rank_patterns[idx]:
                prefix = f"base_model.model.model.layers.{layer}.self_attn.{module_name}"
                a_key = prefix + ".lora_A"
                b_key = prefix + ".lora_B"
                e_key = prefix + ".lora_E"

                a = rng.standard_normal((rank, 4), dtype=np.float32)
                b = rng.standard_normal((out_dim, rank), dtype=np.float32)
                e = rng.standard_normal((rank, 1), dtype=np.float32)

                if label == 1:
                    a = a + 0.1
                    b = b - 0.05
                    e = e + 0.2

                tensors[a_key] = a
                tensors[b_key] = b
                tensors[e_key] = e

        save_file(tensors, str(model_dir / "adapter_model.safetensors"))

    return data_dir


def make_tiny_zero_rank_adalora_dataset(
    root: Path,
    *,
    out_dim: int = 4,
) -> Path:
    data_dir = root / "tiny_adalora_zero_rank_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(655)
    model_dir = data_dir / "tiny_adalora_zero_rank_label0_0"
    model_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for layer in range(2):
        for module_name in ["q_proj", "v_proj"]:
            rank = 0 if layer == 1 and module_name == "q_proj" else 2
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module_name}"
            a_key = prefix + ".lora_A"
            b_key = prefix + ".lora_B"
            e_key = prefix + ".lora_E"

            a = rng.standard_normal((rank, 4), dtype=np.float32)
            b = rng.standard_normal((out_dim, rank), dtype=np.float32)
            e = rng.standard_normal((rank, 1), dtype=np.float32)

            tensors[a_key] = a
            tensors[b_key] = b
            tensors[e_key] = e

    save_file(tensors, str(model_dir / "adapter_model.safetensors"))

    return data_dir


def corrupt_safetensor_file(path: Path) -> None:
    raw = path.read_bytes()
    keep = max(8, len(raw) // 2)
    path.write_bytes(raw[:keep])


def write_path_manifest(path: Path, entries: list[str]) -> None:
    payload = {"path": list(entries)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_tiny_adapter(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones((2, 4), dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.ones((4, 2), dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": np.ones((2, 4), dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": np.ones((4, 2), dtype=np.float32),
    }
    save_file(tensors, str(model_dir / "adapter_model.safetensors"))


def load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_public_and_internal_spectral_metadata(metadata_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    public_metadata = load_json_file(metadata_path)
    internal_metadata = load_spectral_metadata(metadata_path)
    return dict(public_metadata), dict(internal_metadata)


def clone_spectral_metadata_with_state(src_metadata_path: Path, dst_metadata_path: Path) -> None:
    public_metadata, internal_metadata = load_public_and_internal_spectral_metadata(src_metadata_path)
    raw_dataset_layouts = public_metadata.get("dataset_layouts")
    dataset_layouts = (
        [dict(entry) for entry in raw_dataset_layouts]
        if isinstance(raw_dataset_layouts, list)
        else None
    )
    write_spectral_metadata(
        dst_metadata_path,
        internal_metadata=internal_metadata,
        dataset_layouts=dataset_layouts,
    )


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
    write_spectral_metadata(
        output_dir / "spectral_metadata.json",
        internal_metadata=metadata,
    )


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


def extract_feature_bundle(
    *,
    manifest_path: Path,
    dataset_root: Path,
    runs_root: Path,
    run_id: str,
    spectral_features: list[str] | tuple[str, ...] = ("energy", "kurtosis", "sv_topk"),
    spectral_sv_top_k: int = 2,
    spectral_moment_source: str = "sv",
    spectral_qv_sum_mode: str = "none",
) -> Path:
    proc = run_cli(
        [
            "feature",
            "extract",
            "--manifest-json",
            str(manifest_path),
            "--dataset-root",
            str(dataset_root),
            "--extractor",
            "spectral",
            "--spectral-features",
            *[str(x) for x in spectral_features],
            "--spectral-sv-top-k",
            str(spectral_sv_top_k),
            "--spectral-moment-source",
            spectral_moment_source,
            "--spectral-qv-sum-mode",
            spectral_qv_sum_mode,
            "--output-root",
            str(runs_root),
            "--run-id",
            run_id,
        ],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr + proc.stdout)
    return runs_root / "feature_extract" / run_id / "features" / "spectral_features.npy"


def build_tiny_provenance_merge_bundle(tmp_path: Path) -> dict[str, Any]:
    llama_dir = make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
    t5_dir = make_tiny_t5_encoder_decoder_adapter_dataset(tmp_path, out_dim=4)

    llama_manifest = tmp_path / "llama_manifest.json"
    t5_manifest = tmp_path / "t5_manifest.json"
    write_path_manifest(
        llama_manifest,
        [str((llama_dir / "tiny_label0_0" / "adapter_model.safetensors").resolve())],
    )
    write_path_manifest(
        t5_manifest,
        [str((t5_dir / "tiny_t5_encdec_label0_0" / "adapter_model.safetensors").resolve())],
    )

    llama_items = parse_single_manifest_json(
        manifest_path=llama_manifest,
        dataset_root=tmp_path,
        section_key="path",
    )
    t5_items = parse_single_manifest_json(
        manifest_path=t5_manifest,
        dataset_root=tmp_path,
        section_key="path",
    )

    llama_features, llama_labels, llama_names, llama_metadata = extract_spectral_features(
        items=llama_items,
        spectral_features=["energy", "stable_rank"],
        spectral_qv_sum_mode="append",
        sv_top_k=1,
        block_size=64,
        dtype=np.float32,
    )
    t5_features, t5_labels, t5_names, t5_metadata = extract_spectral_features(
        items=t5_items,
        spectral_features=["energy", "stable_rank"],
        spectral_qv_sum_mode="append",
        sv_top_k=1,
        block_size=64,
        dtype=np.float32,
    )

    feature_root = tmp_path / "runs" / "feature_extract"
    llama_merged_dir = feature_root / "run_llama" / "merged"
    t5_merged_dir = feature_root / "run_t5" / "merged"
    write_merged_feature_artifacts(
        llama_merged_dir,
        features=llama_features,
        labels=llama_labels,
        model_names=llama_names,
        metadata=llama_metadata,
    )
    write_merged_feature_artifacts(
        t5_merged_dir,
        features=t5_features,
        labels=t5_labels,
        model_names=t5_names,
        metadata=t5_metadata,
    )
    write_dataset_reference_report(
        default_dataset_reference_report_path(llama_merged_dir),
        build_dataset_reference_payload_from_items(
            items=llama_items,
            artifact_kind="merge_spectral_shards",
            manifest_json=llama_manifest,
            dataset_root=tmp_path,
            artifact_model_count=len(llama_names),
            source_artifacts=[str(llama_manifest)],
        ),
    )
    write_dataset_reference_report(
        default_dataset_reference_report_path(t5_merged_dir),
        build_dataset_reference_payload_from_items(
            items=t5_items,
            artifact_kind="merge_spectral_shards",
            manifest_json=t5_manifest,
            dataset_root=tmp_path,
            artifact_model_count=len(t5_names),
            source_artifacts=[str(t5_manifest)],
        ),
    )

    proc = run_cli(
        [
            "util",
            "merge-features",
            "--merge",
            "run_llama",
            "run_t5",
            "--output-filename",
            "merged_combo",
            "--feature-root",
            str(feature_root),
        ],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr + proc.stdout)

    merged_dir = feature_root / "merged_combo" / "merged"
    return {
        "feature_root": feature_root,
        "merged_dir": merged_dir,
        "llama_merged_dir": llama_merged_dir,
        "t5_merged_dir": t5_merged_dir,
        "llama_features": np.asarray(llama_features, dtype=np.float32),
        "t5_features": np.asarray(t5_features, dtype=np.float32),
        "llama_labels": None if llama_labels is None else np.asarray(llama_labels, dtype=np.int32),
        "t5_labels": None if t5_labels is None else np.asarray(t5_labels, dtype=np.int32),
        "llama_names": [str(x) for x in llama_names],
        "t5_names": [str(x) for x in t5_names],
        "llama_metadata": dict(llama_metadata),
        "t5_metadata": dict(t5_metadata),
    }


class TestCliAndPipeline(unittest.TestCase):
    def _run_supervised_local_smoke(
        self,
        *,
        model_name: str,
        expected_importance_type: str,
        expected_normalization_policy: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            merged_dir = tmp_path / "merged"
            run_id = f"supervised_{model_name}_test"
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id=f"{run_id}_features",
            )

            proc = run_cli(
                [
                    "run",
                    "supervised",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    model_name,
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--feature-file",
                    str(feature_path),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                    "--tuning-executor",
                    "local",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            run_dir = runs_root / "supervised" / run_id
            report_path = run_dir / "reports" / "supervised_report.json"
            metadata_path = run_dir / "reports" / "winner_feature_weights_metadata.json"
            self.assertTrue(report_path.exists())
            self.assertTrue(metadata_path.exists())

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(report["tuning"]["winner"]["model_name"], model_name)
            self.assertEqual(report["tuning"]["winner"]["normalization_policy"], expected_normalization_policy)
            self.assertEqual(metadata["importance_type"], expected_importance_type)
            return report, metadata

    def test_output_path_guards_redirect_filesystem_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with working_directory(tmp_path):
                resolved_output_root = cli_mod._resolve_output_root(Path("/"), "feature_extract")
                rewritten_download = cli_mod._rewrite_download_local_dir(["--local-dir", "/"])
                normalized_download = cli_mod._normalize_download_args(["--", "--backdoored", "60"])

            self.assertEqual(resolved_output_root, (tmp_path / "runs" / "feature_extract").resolve())
            self.assertEqual(rewritten_download, ["--local-dir", str((tmp_path / "data").resolve())])
            self.assertEqual(normalized_download, ["--backdoored", "60"])

    def test_supervised_finalize_distributed_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "supervised_kernel_finalize_distributed"
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id=f"{run_id}_features",
            )

            prepared = run_supervised_pipeline(
                manifest_json=manifest,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id=run_id,
                model_name="kernel_svm",
                spectral_features=["energy", "kurtosis", "sv_topk"],
                spectral_sv_top_k=2,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="none",
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
                score_percentiles=[90.0, 95.0],
                feature_file=feature_path,
                tuning_executor="local",
                slurm_partition="extra",
                slurm_max_concurrent="auto",
                slurm_cpus_per_task="auto",
                finalize_export_shards=2,
                stage="prepare",
                run_dir=None,
                task_index=None,
            )
            run_dir = Path(prepared["run_dir"])

            for task_idx in range(int(prepared["n_tasks"])):
                run_supervised_pipeline(
                    manifest_json=None,
                    dataset_root=tmp_path,
                    output_root=runs_root,
                    run_id=None,
                    model_name="kernel_svm",
                    spectral_features=["energy", "kurtosis", "sv_topk"],
                    spectral_sv_top_k=2,
                    spectral_moment_source="sv",
                    spectral_qv_sum_mode="none",
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
                    score_percentiles=[90.0, 95.0],
                    feature_file=None,
                    tuning_executor="local",
                    slurm_partition="extra",
                    slurm_max_concurrent="auto",
                    slurm_cpus_per_task="auto",
                    finalize_export_shards=2,
                    stage="worker",
                    run_dir=run_dir,
                    task_index=task_idx,
                )

            finalize_prepared = run_supervised_pipeline(
                manifest_json=None,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id=None,
                model_name="kernel_svm",
                spectral_features=["energy", "kurtosis", "sv_topk"],
                spectral_sv_top_k=2,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="none",
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
                score_percentiles=[90.0, 95.0],
                feature_file=None,
                tuning_executor="local",
                slurm_partition="extra",
                slurm_max_concurrent="auto",
                slurm_cpus_per_task="auto",
                finalize_export_shards=2,
                stage="finalize_prepare",
                run_dir=run_dir,
                task_index=None,
            )
            self.assertEqual(finalize_prepared["winner_feature_weights_mode"], "permutation")
            self.assertEqual(finalize_prepared["winner_feature_weights_tasks"], 2)

            for task_idx in range(int(finalize_prepared["winner_feature_weights_tasks"])):
                worker_result = run_supervised_pipeline(
                    manifest_json=None,
                    dataset_root=tmp_path,
                    output_root=runs_root,
                    run_id=None,
                    model_name="kernel_svm",
                    spectral_features=["energy", "kurtosis", "sv_topk"],
                    spectral_sv_top_k=2,
                    spectral_moment_source="sv",
                    spectral_qv_sum_mode="none",
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
                    score_percentiles=[90.0, 95.0],
                    feature_file=None,
                    tuning_executor="local",
                    slurm_partition="extra",
                    slurm_max_concurrent="auto",
                    slurm_cpus_per_task="auto",
                    finalize_export_shards=2,
                    stage="finalize_worker",
                    run_dir=run_dir,
                    task_index=task_idx,
                )
                self.assertEqual(worker_result["status"], "ok")

            finalized = run_supervised_pipeline(
                manifest_json=None,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id=None,
                model_name="kernel_svm",
                spectral_features=["energy", "kurtosis", "sv_topk"],
                spectral_sv_top_k=2,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="none",
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
                score_percentiles=[90.0, 95.0],
                feature_file=None,
                tuning_executor="local",
                slurm_partition="extra",
                slurm_max_concurrent="auto",
                slurm_cpus_per_task="auto",
                finalize_export_shards=2,
                stage="finalize_merge",
                run_dir=run_dir,
                task_index=None,
            )

            metadata_path = run_dir / "reports" / "winner_feature_weights_metadata.json"
            artifact_index_path = run_dir / "artifact_index.json"
            finalize_state_path = run_dir / "reports" / "finalize_state.json"
            finalize_manifest_path = run_dir / "reports" / "winner_feature_weights_manifest.json"
            finalize_parts_dir = run_dir / "reports" / "winner_feature_weights_parts"
            finalize_train_features_path = run_dir / "features" / "winner_feature_weights_train_features.npy"
            finalize_train_labels_path = run_dir / "features" / "winner_feature_weights_train_labels.npy"
            self.assertEqual(finalized["report"], str(run_dir / "reports" / "supervised_report.json"))
            self.assertTrue(metadata_path.exists())
            self.assertTrue(artifact_index_path.exists())
            self.assertFalse(finalize_state_path.exists())
            self.assertFalse(finalize_manifest_path.exists())
            self.assertFalse(finalize_parts_dir.exists())
            self.assertFalse(finalize_train_features_path.exists())
            self.assertFalse(finalize_train_labels_path.exists())

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(metadata["importance_type"], "permutation_importance")
            self.assertEqual(metadata["execution_mode"], "distributed_permutation")

    def test_supervised_cli_skip_feature_importance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "supervised_skip_feature_importance"
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id=f"{run_id}_features",
            )

            proc = run_cli(
                [
                    "run",
                    "supervised",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "kernel_svm",
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--feature-file",
                    str(feature_path),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                    "--tuning-executor",
                    "local",
                    "--skip-feature-importance",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            run_dir = runs_root / "supervised" / run_id
            self.assertTrue((run_dir / "reports" / "supervised_report.json").exists())
            self.assertTrue((run_dir / "artifact_index.json").exists())
            self.assertTrue((run_dir / "run_config.json").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_coefficients.csv").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_by_metric.csv").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_by_block.csv").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_metadata.json").exists())
            self.assertFalse((run_dir / "reports" / "finalize_state.json").exists())
            self.assertFalse((run_dir / "features" / "winner_feature_weights_train_features.npy").exists())
            self.assertFalse((run_dir / "features" / "winner_feature_weights_train_labels.npy").exists())

            with open(run_dir / "run_config.json", "r", encoding="utf-8") as f:
                run_config = json.load(f)
            self.assertTrue(run_config["skip_feature_importance"])

    def test_supervised_finalize_prepare_skip_feature_importance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "supervised_finalize_prepare_skip"
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id=f"{run_id}_features",
            )

            prepared = run_supervised_pipeline(
                manifest_json=manifest,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id=run_id,
                model_name="kernel_svm",
                spectral_features=["energy", "kurtosis", "sv_topk"],
                spectral_sv_top_k=2,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="none",
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
                score_percentiles=[90.0, 95.0],
                feature_file=feature_path,
                tuning_executor="local",
                slurm_partition="extra",
                slurm_max_concurrent="auto",
                slurm_cpus_per_task="auto",
                finalize_export_shards=2,
                stage="prepare",
                run_dir=None,
                task_index=None,
            )
            run_dir = Path(prepared["run_dir"])

            for task_idx in range(int(prepared["n_tasks"])):
                run_supervised_pipeline(
                    manifest_json=None,
                    dataset_root=tmp_path,
                    output_root=runs_root,
                    run_id=None,
                    model_name="kernel_svm",
                    spectral_features=["energy", "kurtosis", "sv_topk"],
                    spectral_sv_top_k=2,
                    spectral_moment_source="sv",
                    spectral_qv_sum_mode="none",
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
                    score_percentiles=[90.0, 95.0],
                    feature_file=None,
                    tuning_executor="local",
                    slurm_partition="extra",
                    slurm_max_concurrent="auto",
                    slurm_cpus_per_task="auto",
                    finalize_export_shards=2,
                    stage="worker",
                    run_dir=run_dir,
                    task_index=task_idx,
                )

            finalized = run_supervised_pipeline(
                manifest_json=None,
                dataset_root=tmp_path,
                output_root=runs_root,
                run_id=None,
                model_name="kernel_svm",
                spectral_features=["energy", "kurtosis", "sv_topk"],
                spectral_sv_top_k=2,
                spectral_moment_source="sv",
                spectral_qv_sum_mode="none",
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
                score_percentiles=[90.0, 95.0],
                feature_file=None,
                tuning_executor="local",
                slurm_partition="extra",
                slurm_max_concurrent="auto",
                slurm_cpus_per_task="auto",
                finalize_export_shards=2,
                stage="finalize_prepare",
                run_dir=run_dir,
                task_index=None,
                skip_feature_importance=True,
            )

            self.assertEqual(finalized["winner_feature_weights_mode"], "skipped")
            self.assertEqual(finalized["winner_feature_weights_tasks"], 0)
            self.assertTrue((run_dir / "reports" / "supervised_report.json").exists())
            self.assertTrue((run_dir / "artifact_index.json").exists())
            self.assertTrue((run_dir / "run_config.json").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_coefficients.csv").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_by_metric.csv").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_by_block.csv").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_metadata.json").exists())
            self.assertFalse((run_dir / "reports" / "finalize_state.json").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_manifest.json").exists())
            self.assertFalse((run_dir / "reports" / "winner_feature_weights_parts").exists())
            self.assertFalse((run_dir / "features" / "winner_feature_weights_train_features.npy").exists())
            self.assertFalse((run_dir / "features" / "winner_feature_weights_train_labels.npy").exists())

    def test_download_dataset_passthrough_accepts_unknown_flags(self):
        captured_args: list[str] = []

        def _capture(args):
            nonlocal captured_args
            captured_args = list(args.download_args)
            return 0

        with mock.patch.object(cli_mod, "_cmd_download_dataset", side_effect=_capture):
            rc = cli_mod.main(["util", "download-dataset", "--backdoored", "60", "--dry-run"])

        self.assertEqual(rc, 0)
        self.assertEqual(captured_args, ["--backdoored", "60", "--dry-run"])

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

    def test_feature_extract_spectral_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "feature_spectral_test"

            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
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
            metadata_path = run_dir / "features" / "spectral_metadata.json"
            self.assertTrue((run_dir / "features" / "spectral_features.npy").exists())
            self.assertTrue(metadata_path.exists())

            with open(run_dir / "reports" / "feature_extraction_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            with open(run_dir / "run_config.json", "r", encoding="utf-8") as f:
                run_config = json.load(f)
            with open(run_dir / "timings.json", "r", encoding="utf-8") as f:
                timings = json.load(f)
            public_metadata, internal_metadata = load_public_and_internal_spectral_metadata(metadata_path)

            self.assertGreaterEqual(float(report["elapsed_seconds"]), 0.0)
            self.assertAlmostEqual(float(run_config["elapsed_seconds"]), float(report["elapsed_seconds"]), places=6)
            self.assertAlmostEqual(
                float(timings["feature_extract_elapsed_seconds"]),
                float(report["elapsed_seconds"]),
                places=6,
            )
            self.assertNotIn("feature_names", public_metadata)
            self.assertNotIn("block_names", public_metadata)
            self.assertNotIn("lora_adapter_dims", public_metadata)
            self.assertIn("feature_names", internal_metadata)
            self.assertIn("block_names", internal_metadata)
            self.assertEqual(public_metadata["dataset_layouts"][0]["dataset_name"], "tiny_data")
            self.assertEqual(int(public_metadata["dataset_layouts"][0]["sample_count"]), 8)
            self.assertEqual(int(public_metadata["dataset_layouts"][0]["layer_count"]), 2)
            self.assertIn("adapter_dims", public_metadata["dataset_layouts"][0])

            dataset_reference_report_path = run_dir / "reports" / "dataset_reference_report.json"
            self.assertTrue(dataset_reference_report_path.exists())
            with open(dataset_reference_report_path, "r", encoding="utf-8") as f:
                dataset_report = json.load(f)
            with open(run_dir / "artifact_index.json", "r", encoding="utf-8") as f:
                artifact_index = json.load(f)

            self.assertEqual(report["dataset_reference_report_path"], str(dataset_reference_report_path))
            self.assertEqual(artifact_index["dataset_reference_report"], str(dataset_reference_report_path))
            self.assertEqual(dataset_report["artifact_kind"], "feature_extract")
            self.assertTrue(bool(dataset_report["is_complete"]))
            self.assertEqual(int(dataset_report["artifact_model_count"]), 8)
            self.assertEqual(int(dataset_report["resolved_model_count"]), 8)
            self.assertEqual(int(dataset_report["dataset_group_count"]), 1)
            self.assertEqual(dataset_report["dataset_groups"][0]["dataset_name"], "tiny_data")
            self.assertEqual(dataset_report["label_counts"], {"0": 4, "1": 4})
            self.assertNotIn("model_index", dataset_report)

    def test_feature_extract_spectral_skips_corrupt_safetensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adapter_dataset(tmp_path)
            corrupt_safetensor_file(data_dir / "tiny_label0_0" / "adapter_model.safetensors")
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "feature_spectral_skip_corrupt"

            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "sv_topk",
                    "--spectral-sv-top-k",
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
            metadata_path = run_dir / "features" / "spectral_metadata.json"
            with open(run_dir / "reports" / "feature_extraction_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            public_metadata, internal_metadata = load_public_and_internal_spectral_metadata(metadata_path)
            with open(run_dir / "reports" / "dataset_reference_report.json", "r", encoding="utf-8") as f:
                dataset_report = json.load(f)

            self.assertEqual(report["feature_shape"][0], 7)
            self.assertEqual(int(internal_metadata["input_n_models"]), 8)
            self.assertEqual(int(internal_metadata["n_models"]), 7)
            self.assertEqual(int(internal_metadata["skipped_model_count"]), 1)
            self.assertEqual(internal_metadata["skipped_models"][0]["model_name"], "tiny_label0_0")
            self.assertIn("Skipped spectral adapter 'tiny_label0_0'", " ".join(report["warnings"]))
            self.assertEqual(int(dataset_report["artifact_model_count"]), 7)
            self.assertEqual(int(dataset_report["resolved_model_count"]), 7)
            self.assertEqual(dataset_report["label_counts"], {"0": 3, "1": 4})
            self.assertEqual(int(public_metadata["dataset_layouts"][0]["sample_count"]), 7)

    def test_feature_extract_accepts_joint_manifest_and_extracts_all_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "gmm_manifest.json"
            write_joint_manifest(manifest)

            train_items, infer_items = parse_joint_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
            )
            expected_count = len(train_items) + len(infer_items)

            runs_root = tmp_path / "runs"
            run_id = "feature_spectral_joint_manifest_test"

            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
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
            features = np.load(run_dir / "features" / "spectral_features.npy")
            with open(run_dir / "features" / "spectral_model_names.json", "r", encoding="utf-8") as f:
                model_names = [str(x) for x in json.load(f)]
            with open(run_dir / "reports" / "feature_extraction_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)

            self.assertEqual(int(features.shape[0]), expected_count)
            self.assertEqual(len(model_names), expected_count)
            self.assertEqual(int(report["model_count"]), expected_count)
            self.assertEqual(int(report["feature_shape"][0]), expected_count)

    def test_backfill_dataset_reference_reports_recreates_missing_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "backfill_source"
            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
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
            merged_dir = runs_root / "feature_extract" / "backfill_merged" / "merged"
            merge_spectral_shards(
                manifest_json=manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[run_dir],
                output_dir=merged_dir,
            )

            (run_dir / "reports" / "dataset_reference_report.json").unlink()
            (merged_dir / "dataset_reference_report.json").unlink()

            stats = backfill_dataset_reference_reports(runs_root / "feature_extract")
            self.assertGreaterEqual(int(stats["feature_extract_reports_written"]), 1)
            self.assertGreaterEqual(int(stats["merge_reports_written"]), 1)
            self.assertTrue((run_dir / "reports" / "dataset_reference_report.json").exists())
            self.assertTrue((merged_dir / "dataset_reference_report.json").exists())

            with open(run_dir / "artifact_index.json", "r", encoding="utf-8") as f:
                artifact_index = json.load(f)
            self.assertEqual(
                artifact_index["dataset_reference_report"],
                str(run_dir / "reports" / "dataset_reference_report.json"),
            )

            with open(merged_dir / "dataset_reference_report.json", "r", encoding="utf-8") as f:
                merged_dataset_report = json.load(f)
            self.assertEqual(merged_dataset_report["artifact_kind"], "merge_spectral_shards")
            self.assertTrue(bool(merged_dataset_report["is_complete"]))
            self.assertEqual(int(merged_dataset_report["artifact_model_count"]), 8)

    def test_feature_extract_rejects_removed_frobenius_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "frobenius",
                ],
                cwd=REPO_ROOT,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Unknown spectral features requested", proc.stderr + proc.stdout)

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
            rows_from_inference = report["fit_assessment"]["threshold_evaluation_from_inference"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows_from_inference), 1)
            reported_threshold = float(rows[0]["threshold"])
            reported_threshold_from_inference = float(rows_from_inference[0]["threshold"])
            reported_percentile_from_inference = float(rows_from_inference[0]["percentile_from_inference"])

            scores = []
            infer_scores = []
            with open(train_scores_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scores.append(float(row["score"]))

            infer_scores_path = run_dir / "reports" / "inference_scores.csv"
            with open(infer_scores_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    infer_scores.append(float(row["score"]))

            expected = float(np.percentile(np.asarray(scores, dtype=np.float64), 90))
            expected_from_inference = float(np.percentile(np.asarray(infer_scores, dtype=np.float64), 90))
            self.assertAlmostEqual(reported_threshold, expected, places=6)
            self.assertAlmostEqual(reported_threshold_from_inference, expected_from_inference, places=6)
            self.assertAlmostEqual(reported_percentile_from_inference, 90.0, places=6)

    def test_spectral_per_block_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                sv_top_k=4,
                block_size=64,
                dtype=np.float32,
            )

            self.assertIsNotNone(labels)
            self.assertEqual(int(features.shape[1]), int(metadata["feature_dim"]))
            feature_names = [str(x) for x in metadata["feature_names"]]
            block_names = [str(x) for x in metadata["block_names"]]

            for block in block_names:
                idx_energy = feature_names.index(f"{block}.energy")
                idx_l2 = feature_names.index(f"{block}.l2_norm")
                np.testing.assert_allclose(
                    features[:, idx_energy],
                    np.square(features[:, idx_l2]),
                    rtol=1e-4,
                    atol=1e-4,
                )
                idx_sv3 = feature_names.index(f"{block}.sv_3")
                idx_sv4 = feature_names.index(f"{block}.sv_4")
                self.assertTrue(np.allclose(features[:, idx_sv3], 0.0))
                self.assertTrue(np.allclose(features[:, idx_sv4], 0.0))

    def test_spectral_t5_encoder_decoder_names_remain_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_t5_encoder_decoder_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_path_manifest(
                manifest,
                [str(path.resolve()) for path in sorted(data_dir.glob("*/adapter_model.safetensors"))],
            )

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            _, _, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "sv_topk"],
                spectral_qv_sum_mode="append",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            block_names = [str(x) for x in metadata["block_names"]]
            feature_names = [str(x) for x in metadata["feature_names"]]

            self.assertEqual(len(block_names), len(set(block_names)))
            self.assertEqual(len(feature_names), len(set(feature_names)))
            self.assertIn("encoder.block0.layer0.SelfAttention.q", block_names)
            self.assertIn("decoder.block0.layer0.SelfAttention.q", block_names)
            self.assertIn("encoder.block0.layer0.SelfAttention.qv_sum", block_names)
            self.assertIn("decoder.block0.layer0.SelfAttention.qv_sum", block_names)
            self.assertIn("encoder.block0.layer0.SelfAttention.q.energy", feature_names)
            self.assertIn("decoder.block0.layer0.SelfAttention.q.energy", feature_names)
            self.assertIn("encoder.block0.layer0.SelfAttention.qv_sum.energy", feature_names)
            self.assertIn("decoder.block0.layer0.SelfAttention.qv_sum.energy", feature_names)

    def test_manifest_resolves_nested_best_model_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_roberta_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            model_dir = (data_dir / "tiny_roberta_label0_0").resolve()
            write_path_manifest(manifest, [str(model_dir)])

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].model_dir, model_dir)
            self.assertEqual(items[0].model_name, "tiny_roberta_label0_0")
            self.assertEqual(items[0].label, 0)
            self.assertEqual(
                items[0].adapter_path,
                model_dir / "best_model" / "adapter_model.safetensors",
            )

    def test_spectral_accepts_zero_rank_adalora_scaling_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_zero_rank_adalora_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_path_manifest(
                manifest,
                [str(path.resolve()) for path in sorted(data_dir.glob("*/adapter_model.safetensors"))],
            )

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "sv_topk"],
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            self.assertIsNotNone(labels)
            np.testing.assert_array_equal(labels, np.asarray([0], dtype=np.int32))
            self.assertEqual(features.shape, (1, int(metadata["feature_dim"])))
            feature_names = [str(x) for x in metadata["feature_names"]]
            idx_energy = feature_names.index("layer1.self_attn.q_proj.energy")
            idx_sv1 = feature_names.index("layer1.self_attn.q_proj.sv_1")
            idx_sv2 = feature_names.index("layer1.self_attn.q_proj.sv_2")
            self.assertAlmostEqual(float(features[0, idx_energy]), 0.0, places=6)
            self.assertAlmostEqual(float(features[0, idx_sv1]), 0.0, places=6)
            self.assertAlmostEqual(float(features[0, idx_sv2]), 0.0, places=6)

    def test_finalize_schema_group_merge_zero_fills_cross_schema_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llama_dir = make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            t5_dir = make_tiny_t5_encoder_decoder_adapter_dataset(tmp_path, out_dim=4)

            llama_manifest = tmp_path / "llama_manifest.json"
            t5_manifest = tmp_path / "t5_manifest.json"
            write_path_manifest(
                llama_manifest,
                [str((llama_dir / "tiny_label0_0" / "adapter_model.safetensors").resolve())],
            )
            write_path_manifest(
                t5_manifest,
                [str((t5_dir / "tiny_t5_encdec_label0_0" / "adapter_model.safetensors").resolve())],
            )

            llama_items = parse_single_manifest_json(
                manifest_path=llama_manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            t5_items = parse_single_manifest_json(
                manifest_path=t5_manifest,
                dataset_root=tmp_path,
                section_key="path",
            )

            llama_features, llama_labels, llama_names, llama_metadata = extract_spectral_features(
                items=llama_items,
                spectral_features=["energy"],
                spectral_qv_sum_mode="append",
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )
            t5_features, t5_labels, t5_names, t5_metadata = extract_spectral_features(
                items=t5_items,
                spectral_features=["energy"],
                spectral_qv_sum_mode="append",
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )

            llama_merged_dir = tmp_path / "schema_groups" / "group_llama" / "merged"
            t5_merged_dir = tmp_path / "schema_groups" / "group_t5" / "merged"
            write_merged_feature_artifacts(
                llama_merged_dir,
                features=llama_features,
                labels=llama_labels,
                model_names=llama_names,
                metadata=llama_metadata,
            )
            write_merged_feature_artifacts(
                t5_merged_dir,
                features=t5_features,
                labels=t5_labels,
                model_names=t5_names,
                metadata=t5_metadata,
            )
            write_dataset_reference_report(
                default_dataset_reference_report_path(llama_merged_dir),
                build_dataset_reference_payload_from_items(
                    items=llama_items,
                    artifact_kind="merge_spectral_shards",
                    manifest_json=llama_manifest,
                    dataset_root=tmp_path,
                    artifact_model_count=len(llama_names),
                    source_artifacts=[str(llama_manifest)],
                ),
            )
            write_dataset_reference_report(
                default_dataset_reference_report_path(t5_merged_dir),
                build_dataset_reference_payload_from_items(
                    items=t5_items,
                    artifact_kind="merge_spectral_shards",
                    manifest_json=t5_manifest,
                    dataset_root=tmp_path,
                    artifact_model_count=len(t5_names),
                    source_artifacts=[str(t5_manifest)],
                ),
            )

            schema_report = tmp_path / "schema_partition_report.json"
            schema_report.write_text(
                json.dumps(
                    {
                        "groups": [
                            {
                                "group_id": "group_llama",
                                "merged_output_dir": str(llama_merged_dir),
                            },
                            {
                                "group_id": "group_t5",
                                "merged_output_dir": str(t5_merged_dir),
                            },
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            output_dir = tmp_path / "finalized"
            outputs = finalize_schema_group_merge(
                schema_report_path=schema_report,
                output_dir=output_dir,
            )

            merged_features = np.load(outputs["feature_path"])
            with open(outputs["model_names_path"], "r", encoding="utf-8") as f:
                model_names = json.load(f)
            public_merged_metadata, merged_metadata = load_public_and_internal_spectral_metadata(
                outputs["metadata_path"]
            )
            with open(output_dir / "dataset_reference_report.json", "r", encoding="utf-8") as f:
                dataset_report = json.load(f)

            feature_names = [str(x) for x in merged_metadata["feature_names"]]
            self.assertEqual(merged_features.shape[0], 2)
            self.assertEqual(merged_features.shape[1], len(feature_names))
            self.assertNotIn("feature_names", public_merged_metadata)
            self.assertNotIn("block_names", public_merged_metadata)
            self.assertNotIn("lora_adapter_dims", public_merged_metadata)
            self.assertEqual(len(public_merged_metadata["dataset_layouts"]), 2)

            llama_row = model_names.index("tiny_label0_0")
            t5_row = model_names.index("tiny_t5_encdec_label0_0")
            llama_feature = feature_names.index("layer0.self_attn.q_proj.energy")
            t5_feature = feature_names.index("encoder.block0.layer0.SelfAttention.q.energy")

            self.assertGreater(float(merged_features[llama_row, llama_feature]), 0.0)
            self.assertGreater(float(merged_features[t5_row, t5_feature]), 0.0)
            self.assertAlmostEqual(float(merged_features[llama_row, t5_feature]), 0.0, places=6)
            self.assertAlmostEqual(float(merged_features[t5_row, llama_feature]), 0.0, places=6)
            self.assertEqual(dataset_report["artifact_kind"], "finalize_schema_group_merge")
            self.assertTrue(bool(dataset_report["is_complete"]))
            self.assertEqual(int(dataset_report["artifact_model_count"]), 2)
            self.assertEqual(int(dataset_report["resolved_model_count"]), 2)
            self.assertEqual(int(dataset_report["dataset_group_count"]), 2)

    def test_util_merge_features_zero_fills_disjoint_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llama_dir = make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            t5_dir = make_tiny_t5_encoder_decoder_adapter_dataset(tmp_path, out_dim=4)

            llama_manifest = tmp_path / "llama_manifest.json"
            t5_manifest = tmp_path / "t5_manifest.json"
            write_path_manifest(
                llama_manifest,
                [str((llama_dir / "tiny_label0_0" / "adapter_model.safetensors").resolve())],
            )
            write_path_manifest(
                t5_manifest,
                [str((t5_dir / "tiny_t5_encdec_label0_0" / "adapter_model.safetensors").resolve())],
            )

            llama_items = parse_single_manifest_json(
                manifest_path=llama_manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            t5_items = parse_single_manifest_json(
                manifest_path=t5_manifest,
                dataset_root=tmp_path,
                section_key="path",
            )

            llama_features, llama_labels, llama_names, llama_metadata = extract_spectral_features(
                items=llama_items,
                spectral_features=["energy"],
                spectral_qv_sum_mode="append",
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )
            t5_features, t5_labels, t5_names, t5_metadata = extract_spectral_features(
                items=t5_items,
                spectral_features=["energy"],
                spectral_qv_sum_mode="append",
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )

            feature_root = tmp_path / "runs" / "feature_extract"
            llama_merged_dir = feature_root / "run_llama" / "merged"
            t5_merged_dir = feature_root / "run_t5" / "merged"
            write_merged_feature_artifacts(
                llama_merged_dir,
                features=llama_features,
                labels=llama_labels,
                model_names=llama_names,
                metadata=llama_metadata,
            )
            write_merged_feature_artifacts(
                t5_merged_dir,
                features=t5_features,
                labels=t5_labels,
                model_names=t5_names,
                metadata=t5_metadata,
            )
            write_dataset_reference_report(
                default_dataset_reference_report_path(llama_merged_dir),
                build_dataset_reference_payload_from_items(
                    items=llama_items,
                    artifact_kind="merge_spectral_shards",
                    manifest_json=llama_manifest,
                    dataset_root=tmp_path,
                    artifact_model_count=len(llama_names),
                    source_artifacts=[str(llama_manifest)],
                ),
            )
            write_dataset_reference_report(
                default_dataset_reference_report_path(t5_merged_dir),
                build_dataset_reference_payload_from_items(
                    items=t5_items,
                    artifact_kind="merge_spectral_shards",
                    manifest_json=t5_manifest,
                    dataset_root=tmp_path,
                    artifact_model_count=len(t5_names),
                    source_artifacts=[str(t5_manifest)],
                ),
            )

            proc = run_cli(
                [
                    "util",
                    "merge-features",
                    "--merge",
                    "run_llama",
                    "run_t5",
                    "--output-filename",
                    "merged_combo",
                    "--feature-root",
                    str(feature_root),
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            output_dir = feature_root / "merged_combo" / "merged"
            merged_features = np.load(output_dir / "spectral_features.npy")
            with open(output_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                model_names = json.load(f)
            _, merged_metadata = load_public_and_internal_spectral_metadata(
                output_dir / "spectral_metadata.json"
            )
            with open(output_dir / "dataset_reference_report.json", "r", encoding="utf-8") as f:
                dataset_report = json.load(f)
            with open(output_dir / "spectral_merge_report.json", "r", encoding="utf-8") as f:
                merge_report = json.load(f)

            feature_names = [str(x) for x in merged_metadata["feature_names"]]
            self.assertEqual(merged_features.shape[0], 2)
            self.assertEqual(merged_features.shape[1], len(feature_names))
            self.assertEqual(merge_report["merge_stats"]["merge_mode"], "zero_fill_disjoint_rows")
            self.assertGreater(int(merge_report["merge_stats"]["zero_filled_cells"]), 0)

            llama_row = model_names.index("tiny_label0_0")
            t5_row = model_names.index("tiny_t5_encdec_label0_0")
            llama_feature = feature_names.index("layer0.self_attn.q_proj.energy")
            t5_feature = feature_names.index("encoder.block0.layer0.SelfAttention.q.energy")

            self.assertGreater(float(merged_features[llama_row, llama_feature]), 0.0)
            self.assertGreater(float(merged_features[t5_row, t5_feature]), 0.0)
            self.assertAlmostEqual(float(merged_features[llama_row, t5_feature]), 0.0, places=6)
            self.assertAlmostEqual(float(merged_features[t5_row, llama_feature]), 0.0, places=6)
            self.assertEqual(dataset_report["artifact_kind"], "merge_feature_files")
            self.assertTrue(bool(dataset_report["is_complete"]))
            self.assertEqual(int(dataset_report["artifact_model_count"]), 2)
            self.assertEqual(int(dataset_report["resolved_model_count"]), 2)

    def test_util_export_feature_subset_uses_provenance_owned_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_tiny_provenance_merge_bundle(tmp_path)
            feature_root = bundle["feature_root"]
            output_dir = feature_root / "t5_only" / "merged"

            proc = run_cli(
                [
                    "util",
                    "export-feature-subset",
                    "--feature-file",
                    "merged_combo",
                    "--output-filename",
                    "t5_only",
                    "--feature-root",
                    str(feature_root),
                    "--dataset-name",
                    "tiny_t5_encoder_decoder_data",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            exported_features = np.asarray(np.load(output_dir / "spectral_features.npy"), dtype=np.float32)
            exported_labels = np.asarray(np.load(output_dir / "spectral_labels.npy"), dtype=np.int32)
            _, exported_metadata = load_public_and_internal_spectral_metadata(
                output_dir / "spectral_metadata.json"
            )
            with open(output_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                exported_model_names = json.load(f)
            with open(output_dir / "spectral_feature_subset_report.json", "r", encoding="utf-8") as f:
                subset_report = json.load(f)
            with open(output_dir / "dataset_reference_report.json", "r", encoding="utf-8") as f:
                dataset_report = json.load(f)

            np.testing.assert_allclose(exported_features, bundle["t5_features"], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(exported_labels, bundle["t5_labels"])
            self.assertEqual(exported_model_names, bundle["t5_names"])
            self.assertEqual(
                [str(x) for x in exported_metadata["feature_names"]],
                [str(x) for x in bundle["t5_metadata"]["feature_names"]],
            )
            self.assertEqual(
                [str(path) for path in subset_report["selection"]["selected_source_feature_files"]],
                [str(bundle["t5_merged_dir"] / "spectral_features.npy")],
            )
            self.assertEqual(dataset_report["artifact_kind"], "export_feature_subset")
            self.assertEqual(int(dataset_report["artifact_model_count"]), 1)
            self.assertEqual(int(dataset_report["dataset_group_count"]), 1)
            self.assertEqual(dataset_report["dataset_groups"][0]["dataset_name"], "tiny_t5_encoder_decoder_data")

    def test_util_export_feature_subset_respects_requested_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_tiny_provenance_merge_bundle(tmp_path)
            feature_root = bundle["feature_root"]
            output_dir = feature_root / "t5_energy_subset" / "merged"
            source_feature_names = [str(x) for x in bundle["t5_metadata"]["feature_names"]]
            requested_features = ["energy"]
            expected_feature_names = [name for name in source_feature_names if name.endswith(".energy")]
            source_index = {name: i for i, name in enumerate(source_feature_names)}

            proc = run_cli(
                [
                    "util",
                    "export-feature-subset",
                    "--feature-file",
                    "merged_combo",
                    "--output-filename",
                    "t5_energy_subset",
                    "--feature-root",
                    str(feature_root),
                    "--dataset-name",
                    "tiny_t5_encoder_decoder_data",
                    "--features",
                    *requested_features,
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            exported_features = np.asarray(np.load(output_dir / "spectral_features.npy"), dtype=np.float32)
            _, exported_metadata = load_public_and_internal_spectral_metadata(
                output_dir / "spectral_metadata.json"
            )
            with open(output_dir / "spectral_feature_subset_report.json", "r", encoding="utf-8") as f:
                subset_report = json.load(f)

            expected = bundle["t5_features"][:, [source_index[name] for name in expected_feature_names]]
            np.testing.assert_allclose(exported_features, expected, rtol=1e-6, atol=1e-6)
            self.assertEqual([str(x) for x in exported_metadata["feature_names"]], expected_feature_names)
            self.assertEqual(subset_report["selection"]["requested_features"], requested_features)

    def test_util_export_feature_subset_columns_alias_uses_feature_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_tiny_provenance_merge_bundle(tmp_path)
            feature_root = bundle["feature_root"]
            output_dir = feature_root / "t5_energy_subset_alias" / "merged"
            source_feature_names = [str(x) for x in bundle["t5_metadata"]["feature_names"]]
            expected_feature_names = [name for name in source_feature_names if name.endswith(".energy")]
            source_index = {name: i for i, name in enumerate(source_feature_names)}

            proc = run_cli(
                [
                    "util",
                    "export-feature-subset",
                    "--feature-file",
                    "merged_combo",
                    "--output-filename",
                    "t5_energy_subset_alias",
                    "--feature-root",
                    str(feature_root),
                    "--dataset-name",
                    "tiny_t5_encoder_decoder_data",
                    "--columns",
                    "energy",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            exported_features = np.asarray(np.load(output_dir / "spectral_features.npy"), dtype=np.float32)
            _, exported_metadata = load_public_and_internal_spectral_metadata(
                output_dir / "spectral_metadata.json"
            )
            expected = bundle["t5_features"][:, [source_index[name] for name in expected_feature_names]]
            np.testing.assert_allclose(exported_features, expected, rtol=1e-6, atol=1e-6)
            self.assertEqual([str(x) for x in exported_metadata["feature_names"]], expected_feature_names)

    def test_spectral_rejects_removed_frobenius_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            with self.assertRaisesRegex(ValueError, "Unknown spectral features requested"):
                extract_spectral_features(
                    items=items,
                    spectral_features=["frobenius"],
                    sv_top_k=2,
                    block_size=64,
                    dtype=np.float32,
                )

    def test_spectral_moment_source_controls_emitted_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )

            _, _, _, entrywise_metadata = extract_spectral_features(
                items=items,
                spectral_features=["kurtosis", "l1_norm"],
                spectral_moment_source="entrywise",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )
            _, _, _, sv_metadata = extract_spectral_features(
                items=items,
                spectral_features=["kurtosis", "l1_norm"],
                spectral_moment_source="sv",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )
            _, _, _, both_metadata = extract_spectral_features(
                items=items,
                spectral_features=["kurtosis", "l1_norm"],
                spectral_moment_source="both",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            entrywise_names = [str(x) for x in entrywise_metadata["feature_names"]]
            sv_names = [str(x) for x in sv_metadata["feature_names"]]
            both_names = [str(x) for x in both_metadata["feature_names"]]
            block = str(both_metadata["block_names"][0])

            self.assertEqual(entrywise_metadata["spectral_moment_source"], "entrywise")
            self.assertEqual(sv_metadata["spectral_moment_source"], "sv")
            self.assertEqual(both_metadata["spectral_moment_source"], "both")
            self.assertIn(f"{block}.kurtosis", entrywise_names)
            self.assertNotIn(f"{block}.sv_kurtosis", entrywise_names)
            self.assertIn(f"{block}.sv_kurtosis", sv_names)
            self.assertNotIn(f"{block}.kurtosis", sv_names)

            both_block_names = [name for name in both_names if name.startswith(f"{block}.")]
            self.assertEqual(
                both_block_names[:4],
                [
                    f"{block}.kurtosis",
                    f"{block}.sv_kurtosis",
                    f"{block}.l1_norm",
                    f"{block}.sv_l1_norm",
                ],
            )

    def test_spectral_entrywise_only_skips_svd_for_base_and_qv_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )

            with mock.patch(
                "upeftguard.features.spectral.block_delta_singular_values",
                side_effect=AssertionError("entrywise-only extraction should not compute singular values"),
            ) as svd_mock:
                features, labels, _, metadata = extract_spectral_features(
                    items=items,
                    spectral_features=["kurtosis", "l1_norm", "linf_norm", "mean_abs"],
                    spectral_qv_sum_mode="append",
                    spectral_moment_source="entrywise",
                    sv_top_k=2,
                    block_size=64,
                    dtype=np.float32,
                )

            self.assertEqual(svd_mock.call_count, 0)
            self.assertIsNotNone(labels)
            self.assertEqual(int(features.shape[0]), len(items))
            self.assertEqual(int(features.shape[1]), int(metadata["feature_dim"]))
            self.assertEqual(metadata["spectral_moment_source"], "entrywise")
            self.assertEqual(metadata["spectral_qv_sum_mode"], "append")

            feature_names = [str(x) for x in metadata["feature_names"]]
            self.assertTrue(feature_names)
            self.assertFalse(any(".sv_" in name for name in feature_names))
            self.assertFalse(any(name.endswith(".energy") for name in feature_names))

    def test_spectral_sv_moments_and_concentration_match_manual_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, _, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["kurtosis", "l1_norm", "linf_norm", "mean_abs", "concentration_of_energy"],
                spectral_moment_source="sv",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            block = str(metadata["block_names"][0])
            feature_names = [str(x) for x in metadata["feature_names"]]

            tensors = load_file(str(items[0].adapter_path))
            a = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
                dtype=np.float32,
            )
            b = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"],
                dtype=np.float32,
            )
            sv = block_delta_singular_values(a=a, b=b)
            summary = summarize_array_moments(sv)
            concentration = float(sv[0] / np.sum(sv, dtype=np.float64))

            self.assertAlmostEqual(
                float(features[0, feature_names.index(f"{block}.sv_kurtosis")]),
                float(summary.kurtosis),
                places=5,
            )
            self.assertAlmostEqual(
                float(features[0, feature_names.index(f"{block}.sv_l1_norm")]),
                float(summary.l1_norm),
                places=5,
            )
            self.assertAlmostEqual(
                float(features[0, feature_names.index(f"{block}.sv_linf_norm")]),
                float(summary.linf_norm),
                places=5,
            )
            self.assertAlmostEqual(
                float(features[0, feature_names.index(f"{block}.sv_mean_abs")]),
                float(summary.mean_abs),
                places=5,
            )
            self.assertAlmostEqual(
                float(features[0, feature_names.index(f"{block}.concentration_of_energy")]),
                concentration,
                places=5,
            )

    def test_spectral_qv_sum_only_matches_manual_layer_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                spectral_qv_sum_mode="only",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            self.assertIsNotNone(labels)
            block_names = [str(x) for x in metadata["block_names"]]
            self.assertEqual(block_names, ["layer0.self_attn.qv_sum", "layer1.self_attn.qv_sum"])

            feature_names = [str(x) for x in metadata["feature_names"]]
            idx_energy = feature_names.index("layer0.self_attn.qv_sum.energy")
            idx_l2 = feature_names.index("layer0.self_attn.qv_sum.l2_norm")
            np.testing.assert_allclose(
                features[:, idx_energy],
                np.square(features[:, idx_l2]),
                rtol=1e-5,
                atol=1e-5,
            )

            item0 = items[0]
            tensors = load_file(str(item0.adapter_path))
            a_q = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
                dtype=np.float32,
            )
            b_q = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"],
                dtype=np.float32,
            )
            a_v = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"],
                dtype=np.float32,
            )
            b_v = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"],
                dtype=np.float32,
            )

            a_qv = np.concatenate([a_q, a_v], axis=0)
            b_qv = np.concatenate([b_q, b_v], axis=1)
            sv = block_delta_singular_values(a=a_qv, b=b_qv)
            top2 = top_k_singular_values(sv, top_k=2)
            energy = float(np.sum(np.square(sv), dtype=np.float64))
            l2_norm = float(np.sqrt(max(0.0, energy)))

            idx_sv1 = feature_names.index("layer0.self_attn.qv_sum.sv_1")
            idx_sv2 = feature_names.index("layer0.self_attn.qv_sum.sv_2")
            self.assertAlmostEqual(float(features[0, idx_sv1]), float(top2[0]), places=5)
            self.assertAlmostEqual(float(features[0, idx_sv2]), float(top2[1]), places=5)
            self.assertAlmostEqual(float(features[0, idx_energy]), energy, places=5)
            self.assertAlmostEqual(float(features[0, idx_l2]), l2_norm, places=5)

    def test_spectral_qv_sum_only_matches_manual_t5_layer_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_t5_adapter_dataset(tmp_path, out_dim=4)
            manifest = tmp_path / "prepare_t5_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((data_dir / f"tiny_t5_label{label}_{idx}").resolve())
                    for label in [0, 1]
                    for idx in [0, 1]
                ],
            )

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                spectral_qv_sum_mode="only",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            self.assertIsNotNone(labels)
            self.assertEqual(
                [str(x) for x in metadata["block_names"]],
                [
                    "decoder.block0.layer0.SelfAttention.qv_sum",
                    "decoder.block0.layer1.EncDecAttention.qv_sum",
                    "decoder.block1.layer0.SelfAttention.qv_sum",
                    "decoder.block1.layer1.EncDecAttention.qv_sum",
                ],
            )

            feature_names = [str(x) for x in metadata["feature_names"]]
            idx_energy = feature_names.index("decoder.block0.layer0.SelfAttention.qv_sum.energy")
            idx_l2 = feature_names.index("decoder.block0.layer0.SelfAttention.qv_sum.l2_norm")
            np.testing.assert_allclose(
                features[:, idx_energy],
                np.square(features[:, idx_l2]),
                rtol=1e-5,
                atol=1e-5,
            )

            tensors = load_file(str(items[0].adapter_path))
            a_q = np.asarray(
                tensors["base_model.model.decoder.block.0.layer.0.SelfAttention.q.lora_A.weight"],
                dtype=np.float32,
            )
            b_q = np.asarray(
                tensors["base_model.model.decoder.block.0.layer.0.SelfAttention.q.lora_B.weight"],
                dtype=np.float32,
            )
            a_v = np.asarray(
                tensors["base_model.model.decoder.block.0.layer.0.SelfAttention.v.lora_A.weight"],
                dtype=np.float32,
            )
            b_v = np.asarray(
                tensors["base_model.model.decoder.block.0.layer.0.SelfAttention.v.lora_B.weight"],
                dtype=np.float32,
            )

            a_qv = np.concatenate([a_q, a_v], axis=0)
            b_qv = np.concatenate([b_q, b_v], axis=1)
            sv = block_delta_singular_values(a=a_qv, b=b_qv)
            top2 = top_k_singular_values(sv, top_k=2)
            energy = float(np.sum(np.square(sv), dtype=np.float64))
            l2_norm = float(np.sqrt(max(0.0, energy)))

            idx_sv1 = feature_names.index("decoder.block0.layer0.SelfAttention.qv_sum.sv_1")
            idx_sv2 = feature_names.index("decoder.block0.layer0.SelfAttention.qv_sum.sv_2")
            self.assertAlmostEqual(float(features[0, idx_sv1]), float(top2[0]), places=5)
            self.assertAlmostEqual(float(features[0, idx_sv2]), float(top2[1]), places=5)
            self.assertAlmostEqual(float(features[0, idx_energy]), energy, places=5)
            self.assertAlmostEqual(float(features[0, idx_l2]), l2_norm, places=5)

    def test_spectral_qv_sum_only_matches_manual_adalora_layer_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adalora_dataset(tmp_path, out_dim=4)
            manifest = tmp_path / "prepare_adalora_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((data_dir / f"tiny_adalora_label{label}_{idx}").resolve())
                    for label in [0, 1]
                    for idx in [0, 1]
                ],
            )

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                spectral_qv_sum_mode="only",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            self.assertIsNotNone(labels)
            self.assertEqual(
                [str(x) for x in metadata["block_names"]],
                [
                    "layer0.self_attn.qv_sum",
                    "layer1.self_attn.qv_sum",
                ],
            )

            feature_names = [str(x) for x in metadata["feature_names"]]
            idx_energy = feature_names.index("layer0.self_attn.qv_sum.energy")
            idx_l2 = feature_names.index("layer0.self_attn.qv_sum.l2_norm")
            np.testing.assert_allclose(
                features[:, idx_energy],
                np.square(features[:, idx_l2]),
                rtol=1e-5,
                atol=1e-5,
            )

            tensors = load_file(str(items[0].adapter_path))
            a_q = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A"],
                dtype=np.float32,
            )
            b_q = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B"],
                dtype=np.float32,
            )
            e_q = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_E"],
                dtype=np.float32,
            ).reshape(-1, 1)
            a_v = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_A"],
                dtype=np.float32,
            )
            b_v = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_B"],
                dtype=np.float32,
            )
            e_v = np.asarray(
                tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_E"],
                dtype=np.float32,
            ).reshape(-1, 1)

            a_qv = np.concatenate([a_q * e_q, a_v * e_v], axis=0)
            b_qv = np.concatenate([b_q, b_v], axis=1)
            sv = block_delta_singular_values(a=a_qv, b=b_qv)
            top2 = top_k_singular_values(sv, top_k=2)
            energy = float(np.sum(np.square(sv), dtype=np.float64))
            l2_norm = float(np.sqrt(max(0.0, energy)))

            idx_sv1 = feature_names.index("layer0.self_attn.qv_sum.sv_1")
            idx_sv2 = feature_names.index("layer0.self_attn.qv_sum.sv_2")
            self.assertAlmostEqual(float(features[0, idx_sv1]), float(top2[0]), places=5)
            self.assertAlmostEqual(float(features[0, idx_sv2]), float(top2[1]), places=5)
            self.assertAlmostEqual(float(features[0, idx_energy]), energy, places=5)
            self.assertAlmostEqual(float(features[0, idx_l2]), l2_norm, places=5)

    def test_spectral_qv_sum_only_matches_manual_roberta_layer_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_roberta_adapter_dataset(tmp_path, out_dim=4)
            manifest = tmp_path / "prepare_roberta_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((data_dir / f"tiny_roberta_label{label}_{idx}").resolve())
                    for label in [0, 1]
                    for idx in [0, 1]
                ],
            )

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                spectral_qv_sum_mode="only",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            self.assertIsNotNone(labels)
            self.assertEqual(
                [str(x) for x in metadata["block_names"]],
                [
                    "base_model.model.roberta.encoder.layer.0.attention.self.qv_sum",
                    "base_model.model.roberta.encoder.layer.1.attention.self.qv_sum",
                ],
            )
            self.assertEqual(
                [dict(x) for x in metadata["qv_sum_lora_adapter_dims"]],
                [
                    {"m": 4, "n": 4, "r": 4},
                    {"m": 4, "n": 4, "r": 4},
                ],
            )

            feature_names = [str(x) for x in metadata["feature_names"]]
            idx_energy = feature_names.index(
                "base_model.model.roberta.encoder.layer.0.attention.self.qv_sum.energy"
            )
            idx_l2 = feature_names.index(
                "base_model.model.roberta.encoder.layer.0.attention.self.qv_sum.l2_norm"
            )
            np.testing.assert_allclose(
                features[:, idx_energy],
                np.square(features[:, idx_l2]),
                rtol=1e-5,
                atol=1e-5,
            )

            tensors = load_file(str(items[0].adapter_path))
            a_q = np.asarray(
                tensors["base_model.model.roberta.encoder.layer.0.attention.self.query.lora_A.weight"],
                dtype=np.float32,
            )
            b_q = np.asarray(
                tensors["base_model.model.roberta.encoder.layer.0.attention.self.query.lora_B.weight"],
                dtype=np.float32,
            )
            a_v = np.asarray(
                tensors["base_model.model.roberta.encoder.layer.0.attention.self.value.lora_A.weight"],
                dtype=np.float32,
            )
            b_v = np.asarray(
                tensors["base_model.model.roberta.encoder.layer.0.attention.self.value.lora_B.weight"],
                dtype=np.float32,
            )

            a_qv = np.concatenate([a_q, a_v], axis=0)
            b_qv = np.concatenate([b_q, b_v], axis=1)
            sv = block_delta_singular_values(a=a_qv, b=b_qv)
            top2 = top_k_singular_values(sv, top_k=2)
            energy = float(np.sum(np.square(sv), dtype=np.float64))
            l2_norm = float(np.sqrt(max(0.0, energy)))

            idx_sv1 = feature_names.index(
                "base_model.model.roberta.encoder.layer.0.attention.self.qv_sum.sv_1"
            )
            idx_sv2 = feature_names.index(
                "base_model.model.roberta.encoder.layer.0.attention.self.qv_sum.sv_2"
            )
            self.assertAlmostEqual(float(features[0, idx_sv1]), float(top2[0]), places=5)
            self.assertAlmostEqual(float(features[0, idx_sv2]), float(top2[1]), places=5)
            self.assertAlmostEqual(float(features[0, idx_energy]), energy, places=5)
            self.assertAlmostEqual(float(features[0, idx_l2]), l2_norm, places=5)

    def test_prepare_schema_sharded_manifests_groups_mixed_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llama_dir = make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            t5_dir = make_tiny_t5_adapter_dataset(tmp_path, out_dim=4)
            chatglm_dir = make_tiny_chatglm_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_mixed_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((llama_dir / "tiny_label0_0").resolve()),
                    str((llama_dir / "tiny_label1_0").resolve()),
                    str((t5_dir / "tiny_t5_label0_0").resolve()),
                    str((t5_dir / "tiny_t5_label1_0").resolve()),
                    str((chatglm_dir / "tiny_chatglm_label0_0").resolve()),
                    str((chatglm_dir / "tiny_chatglm_label1_0").resolve()),
                ],
            )

            report = prepare_schema_sharded_manifests(
                manifest_path=manifest,
                dataset_root=tmp_path,
                output_dir=tmp_path / "prepared",
                n_shards=4,
                spectral_qv_sum_mode="append",
            )

            self.assertEqual(report["group_count"], 3)
            self.assertTrue(any("Multiple adapter schemas" in warning for warning in report["warnings"]))
            effective_modes = sorted(str(group["effective_spectral_qv_sum_mode"]) for group in report["groups"])
            self.assertEqual(effective_modes, ["append", "append", "none"])

            none_groups = [
                group for group in report["groups"] if str(group["effective_spectral_qv_sum_mode"]) == "none"
            ]
            self.assertEqual(len(none_groups), 1)
            self.assertFalse(bool(none_groups[0]["qv_pairs_supported"]))
            self.assertEqual(int(none_groups[0]["qv_pair_count"]), 0)

            for group in report["groups"]:
                manifest_path = Path(str(group["manifest_path"]))
                shard_manifest_dir = Path(str(group["shard_manifest_dir"]))
                self.assertTrue(manifest_path.exists())
                self.assertTrue(shard_manifest_dir.exists())
                shard_paths = sorted(shard_manifest_dir.glob("shard_*.json"))
                self.assertEqual(len(shard_paths), int(group["n_shards"]))

                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_payload = json.load(f)
                self.assertEqual(len(manifest_payload["path"]), int(group["n_items"]))

    def test_prepare_schema_sharded_manifests_samples_one_schema_per_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llama_dir = make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            t5_dir = make_tiny_t5_adapter_dataset(tmp_path, out_dim=4)
            chatglm_dir = make_tiny_chatglm_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_mixed_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((llama_dir / "tiny_label0_0").resolve()),
                    str((llama_dir / "tiny_label1_0").resolve()),
                    str((llama_dir / "tiny_label0_1").resolve()),
                    str((t5_dir / "tiny_t5_label0_0").resolve()),
                    str((t5_dir / "tiny_t5_label1_0").resolve()),
                    str((chatglm_dir / "tiny_chatglm_label0_0").resolve()),
                    str((chatglm_dir / "tiny_chatglm_label1_0").resolve()),
                ],
            )

            import upeftguard.utilities.merge.prepare_spectral_shards as prep_mod

            real_load = prep_mod.load_delta_block_schema
            seen_paths: list[str] = []

            def wrapped(path: Path):
                seen_paths.append(str(path))
                return real_load(path)

            with mock.patch.object(prep_mod, "load_delta_block_schema", side_effect=wrapped):
                report = prepare_schema_sharded_manifests(
                    manifest_path=manifest,
                    dataset_root=tmp_path,
                    output_dir=tmp_path / "prepared",
                    n_shards=4,
                    spectral_qv_sum_mode="append",
                )

            self.assertEqual(report["n_schema_samples"], 3)
            self.assertEqual(len(seen_paths), 3)
            self.assertEqual(report["group_count"], 3)

    def test_prepare_schema_sharded_manifests_recognizes_adalora_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adalora_dataset(tmp_path, out_dim=4)
            manifest = tmp_path / "prepare_adalora_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((data_dir / f"tiny_adalora_label{label}_{idx}").resolve())
                    for label in [0, 1]
                    for idx in [0, 1]
                ],
            )

            report = prepare_schema_sharded_manifests(
                manifest_path=manifest,
                dataset_root=tmp_path,
                output_dir=tmp_path / "prepared",
                n_shards=4,
                spectral_qv_sum_mode="append",
            )

            self.assertEqual(report["group_count"], 1)
            self.assertEqual(report["n_schema_samples"], 1)
            group = report["groups"][0]
            self.assertEqual(int(group["n_blocks"]), 4)
            self.assertEqual(int(group["qv_pair_count"]), 2)
            self.assertTrue(bool(group["qv_pairs_supported"]))
            self.assertEqual(str(group["effective_spectral_qv_sum_mode"]), "append")
            self.assertEqual(
                [str(x) for x in group["block_name_preview"]],
                [
                    "layer0.self_attn.q_proj",
                    "layer0.self_attn.v_proj",
                    "layer1.self_attn.q_proj",
                ],
            )

            sample_adapter = (data_dir / "tiny_adalora_label0_0" / "adapter_model.safetensors").resolve()
            schema = load_delta_block_schema(Path(str(sample_adapter)))
            self.assertEqual([bool(x) for x in schema.e_keys], [True, True, True, True])

    def test_prepare_schema_sharded_manifests_keeps_variable_rank_adalora_in_one_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_variable_rank_adalora_dataset(tmp_path, out_dim=4)
            manifest = tmp_path / "prepare_adalora_variable_manifest.json"
            entries = [
                str((data_dir / f"tiny_adalora_var_label{idx % 2}_{idx}").resolve())
                for idx in [0, 1, 2, 3]
            ]
            write_path_manifest(manifest, entries)

            exact_signatures = {
                (
                    tuple(schema.a_shapes),
                    tuple(schema.b_shapes),
                    tuple(schema.e_shapes),
                )
                for schema in [
                    load_delta_block_schema(Path(entry) / "adapter_model.safetensors")
                    for entry in entries
                ]
            }
            self.assertGreater(len(exact_signatures), 1)

            report = prepare_schema_sharded_manifests(
                manifest_path=manifest,
                dataset_root=tmp_path,
                output_dir=tmp_path / "prepared",
                n_shards=4,
                spectral_qv_sum_mode="append",
            )

            self.assertEqual(report["group_count"], 1)
            self.assertEqual(report["n_schema_samples"], 1)
            group = report["groups"][0]
            self.assertEqual(str(group["schema_signature_mode"]), "adalora_rank_tolerant")
            self.assertTrue(bool(group["variable_lora_rank"]))
            self.assertEqual(int(group["n_items"]), 4)
            self.assertEqual(int(group["qv_pair_count"]), 2)

    def test_extract_spectral_features_supports_variable_rank_adalora(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_variable_rank_adalora_dataset(tmp_path, out_dim=4)
            manifest = tmp_path / "prepare_adalora_variable_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((data_dir / f"tiny_adalora_var_label{idx % 2}_{idx}").resolve())
                    for idx in [0, 1, 2, 3]
                ],
            )

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, _, metadata = extract_spectral_features(
                items=items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                spectral_qv_sum_mode="append",
                sv_top_k=2,
                block_size=64,
                dtype=np.float32,
            )

            self.assertEqual(features.shape[0], 4)
            self.assertIsNotNone(labels)
            self.assertEqual(int(metadata["n_blocks"]), 6)
            self.assertTrue(bool(metadata["variable_lora_rank"]))
            self.assertNotIn("lora_adapter_dims", metadata)
            self.assertNotIn("base_lora_adapter_dims", metadata)
            self.assertNotIn("qv_sum_lora_adapter_dims", metadata)
            self.assertEqual(
                [str(x) for x in metadata["qv_sum_block_names"]],
                ["layer0.self_attn.qv_sum", "layer1.self_attn.qv_sum"],
            )

    def test_svd_embeddings_support_adalora(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adalora_dataset(tmp_path, out_dim=4)
            manifest = tmp_path / "prepare_adalora_manifest.json"
            write_path_manifest(
                manifest,
                [
                    str((data_dir / f"tiny_adalora_label{label}_{idx}").resolve())
                    for label in [0, 1]
                    for idx in [0, 1]
                ],
            )

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            features, labels, model_names, metadata, warnings = extract_svd_embeddings(
                items=items,
                n_components=2,
                component_grid=[1, 2],
                block_size=64,
                dtype=np.float32,
                acceptance_spearman_threshold=0.0,
                acceptance_variance_threshold=0.0,
                run_offline_label_diagnostics=False,
            )

            self.assertEqual(features.shape, (4, 2))
            self.assertIsNotNone(labels)
            self.assertEqual(len(model_names), 4)
            self.assertEqual(int(metadata["chosen_k"]), 2)
            self.assertEqual(warnings, [])

    def test_merge_spectral_shards_supports_variable_rank_adalora(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_variable_rank_adalora_dataset(tmp_path, out_dim=4)
            manifest_a = tmp_path / "manifest_adalora_a.json"
            manifest_b = tmp_path / "manifest_adalora_b.json"
            write_path_manifest(
                manifest_a,
                [
                    str((data_dir / "tiny_adalora_var_label0_0").resolve()),
                    str((data_dir / "tiny_adalora_var_label1_1").resolve()),
                ],
            )
            write_path_manifest(
                manifest_b,
                [
                    str((data_dir / "tiny_adalora_var_label1_3").resolve()),
                    str((data_dir / "tiny_adalora_var_label0_2").resolve()),
                ],
            )
            merged_manifest = tmp_path / "merged_manifest.json"
            write_path_manifest(
                merged_manifest,
                [
                    str((data_dir / "tiny_adalora_var_label0_0").resolve()),
                    str((data_dir / "tiny_adalora_var_label1_1").resolve()),
                    str((data_dir / "tiny_adalora_var_label1_3").resolve()),
                    str((data_dir / "tiny_adalora_var_label0_2").resolve()),
                ],
            )

            runs_root = tmp_path / "runs"
            merged_dir = tmp_path / "merged"

            run_a = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest_a),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--spectral-qv-sum-mode",
                    "append",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    "adalora_rank_var_a",
                ],
                cwd=REPO_ROOT,
            )
            if run_a.returncode != 0:
                self.fail(run_a.stderr + run_a.stdout)

            run_b = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest_b),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--spectral-qv-sum-mode",
                    "append",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    "adalora_rank_var_b",
                ],
                cwd=REPO_ROOT,
            )
            if run_b.returncode != 0:
                self.fail(run_b.stderr + run_b.stdout)

            merge_spectral_shards(
                manifest_json=merged_manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[
                    runs_root / "feature_extract" / "adalora_rank_var_a",
                    runs_root / "feature_extract" / "adalora_rank_var_b",
                ],
                output_dir=merged_dir,
            )

            public_merged_metadata, merged_metadata = load_public_and_internal_spectral_metadata(
                merged_dir / "spectral_metadata.json"
            )
            merged_features = np.load(merged_dir / "spectral_features.npy")

            self.assertEqual(merged_features.shape[0], 4)
            self.assertTrue(bool(merged_metadata["variable_lora_rank"]))
            self.assertNotIn("lora_adapter_dims", merged_metadata)
            self.assertEqual(int(public_merged_metadata["dataset_layouts"][0]["layer_count"]), 2)
            adapter_dims = public_merged_metadata["dataset_layouts"][0]["adapter_dims"]
            self.assertEqual(adapter_dims["m"], 4)
            self.assertEqual(adapter_dims["n"], 4)
            self.assertEqual(adapter_dims["r"]["mode"], "adaptive")
            self.assertGreaterEqual(int(adapter_dims["r"]["min"]), 1)
            self.assertLessEqual(int(adapter_dims["r"]["max"]), 4)
            self.assertTrue(adapter_dims["r"]["values"])

    def test_merge_spectral_shards_appends_new_feature_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            base_merged_dir = tmp_path / "merged_base"
            qv_merged_dir = tmp_path / "merged_qv_only"
            final_merged_dir = tmp_path / "merged_final"

            base = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    "base_features",
                ],
                cwd=REPO_ROOT,
            )
            if base.returncode != 0:
                self.fail(base.stderr + base.stdout)

            qv_only = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--spectral-qv-sum-mode",
                    "only",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    "qv_only_features",
                ],
                cwd=REPO_ROOT,
            )
            if qv_only.returncode != 0:
                self.fail(qv_only.stderr + qv_only.stdout)

            merge_spectral_shards(
                manifest_json=manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[runs_root / "feature_extract" / "base_features"],
                output_dir=base_merged_dir,
            )
            base_merged = np.load(base_merged_dir / "spectral_features.npy")

            merge_spectral_shards(
                manifest_json=manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[runs_root / "feature_extract" / "qv_only_features"],
                output_dir=qv_merged_dir,
            )
            qv_features = np.load(qv_merged_dir / "spectral_features.npy")
            merge_feature_files(
                feature_paths=[
                    base_merged_dir / "spectral_features.npy",
                    qv_merged_dir / "spectral_features.npy",
                ],
                output_filename=final_merged_dir / "spectral_features.npy",
            )

            merged_features = np.load(final_merged_dir / "spectral_features.npy")
            public_merged_metadata, merged_metadata = load_public_and_internal_spectral_metadata(
                final_merged_dir / "spectral_metadata.json"
            )
            base_dim = int(base_merged.shape[1])
            qv_dim = int(qv_features.shape[1])

            self.assertEqual(int(merged_features.shape[1]), base_dim + qv_dim)
            np.testing.assert_allclose(merged_features[:, :base_dim], base_merged, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(merged_features[:, base_dim:], qv_features, rtol=1e-6, atol=1e-6)

            feature_names = [str(x) for x in merged_metadata["feature_names"]]
            self.assertEqual(len(feature_names), int(merged_features.shape[1]))
            self.assertTrue(any(".qv_sum." in name for name in feature_names))
            self.assertNotIn("feature_names", public_merged_metadata)
            self.assertNotIn("block_names", public_merged_metadata)
            self.assertNotIn("base_block_names", public_merged_metadata)
            self.assertNotIn("qv_sum_block_names", public_merged_metadata)
            self.assertNotIn("lora_adapter_dims", public_merged_metadata)
            self.assertEqual(
                public_merged_metadata["dataset_layouts"],
                [
                    {
                        "dataset_name": "tiny_data",
                        "sample_count": 8,
                        "layer_count": 2,
                        "adapter_dims": {"m": 4, "n": 4, "r": 2},
                    }
                ],
            )
            self.assertNotIn("incoming_metadata", merged_metadata)
            self.assertNotIn("merge_stats", merged_metadata)
            self.assertNotIn("merged_with_existing_output", merged_metadata)
            self.assertNotIn("a_shapes", merged_metadata)
            self.assertNotIn("b_shapes", merged_metadata)
            self.assertNotIn("block_names_raw", merged_metadata)
            self.assertNotIn("base_block_names_raw", merged_metadata)
            self.assertNotIn("qv_sum_block_names_raw", merged_metadata)
            self.assertNotIn("component_grid", merged_metadata.get("extractor_params", {}))
            self.assertNotIn("n_components", merged_metadata.get("extractor_params", {}))
            self.assertTrue(any(".qv_sum" in name for name in merged_metadata["block_names"]))
            self.assertEqual(len(merged_metadata["block_names"]), len(merged_metadata["lora_adapter_dims"]))
            qv_idx = merged_metadata["block_names"].index("layer0.self_attn.qv_sum")
            self.assertEqual(merged_metadata["lora_adapter_dims"][qv_idx], {"m": 4, "n": 4, "r": 4})

            with open(final_merged_dir / "dataset_reference_report.json", "r", encoding="utf-8") as f:
                dataset_report = json.load(f)
            self.assertEqual(dataset_report["artifact_kind"], "merge_feature_files")
            self.assertTrue(bool(dataset_report["is_complete"]))
            self.assertEqual(int(dataset_report["artifact_model_count"]), 8)
            self.assertEqual(int(dataset_report["resolved_model_count"]), 8)
            self.assertEqual(int(dataset_report["dataset_group_count"]), 1)
            self.assertEqual(dataset_report["dataset_groups"][0]["dataset_name"], "tiny_data")
            self.assertEqual(dataset_report["label_counts"], {"0": 4, "1": 4})

    def test_merge_spectral_shards_report_includes_elapsed_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            merged_dir = tmp_path / "merged"

            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "l2_norm",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    "timed_merge_base",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            pipeline_start_epoch_seconds = time.time() - 5.0
            merge_spectral_shards(
                manifest_json=manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[runs_root / "feature_extract" / "timed_merge_base"],
                output_dir=merged_dir,
                pipeline_start_epoch_seconds=pipeline_start_epoch_seconds,
            )

            with open(merged_dir / "spectral_merge_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)

            self.assertGreaterEqual(float(report["merge_elapsed_seconds"]), 0.0)
            self.assertEqual(report["pipeline_start_source"], "launcher")
            self.assertIsNotNone(report["pipeline_start_timestamp_utc"])
            self.assertGreaterEqual(float(report["pipeline_elapsed_seconds"]), 4.0)
            self.assertAlmostEqual(
                float(report["pipeline_start_epoch_seconds"]),
                float(pipeline_start_epoch_seconds),
                places=3,
            )
            shard_runtime = report["shard_runtime"]
            self.assertEqual(int(shard_runtime["reported_elapsed_count"]), 1)
            self.assertEqual(int(shard_runtime["missing_elapsed_count"]), 0)
            self.assertGreaterEqual(float(shard_runtime["elapsed_seconds_sum"]), 0.0)
            self.assertIsNotNone(shard_runtime["earliest_shard_start_timestamp_utc"])
            self.assertIsNotNone(shard_runtime["latest_shard_end_timestamp_utc"])

    def test_merge_spectral_shards_allows_skipped_corrupt_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = make_tiny_adapter_dataset(tmp_path)
            corrupt_safetensor_file(data_dir / "tiny_label0_0" / "adapter_model.safetensors")
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)
            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )

            shard_a_items = items[:4]
            shard_b_items = items[4:]
            shard_a_features, shard_a_labels, shard_a_names, shard_a_metadata = extract_spectral_features(
                items=shard_a_items,
                spectral_features=["energy"],
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )
            shard_b_features, shard_b_labels, shard_b_names, shard_b_metadata = extract_spectral_features(
                items=shard_b_items,
                spectral_features=["energy"],
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )

            runs_root = tmp_path / "shards"
            shard_a_run = runs_root / "feature_extract" / "shard_0"
            shard_b_run = runs_root / "feature_extract" / "shard_1"
            write_merged_feature_artifacts(
                shard_a_run / "features",
                features=shard_a_features,
                labels=shard_a_labels,
                model_names=shard_a_names,
                metadata=shard_a_metadata,
            )
            write_merged_feature_artifacts(
                shard_b_run / "features",
                features=shard_b_features,
                labels=shard_b_labels,
                model_names=shard_b_names,
                metadata=shard_b_metadata,
            )

            merged_dir = tmp_path / "merged"
            outputs = merge_spectral_shards(
                manifest_json=manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[shard_a_run, shard_b_run],
                output_dir=merged_dir,
            )

            merged_features = np.load(outputs["feature_path"])
            with open(outputs["model_names_path"], "r", encoding="utf-8") as f:
                merged_names = json.load(f)
            _, merged_metadata = load_public_and_internal_spectral_metadata(outputs["metadata_path"])
            with open(outputs["dataset_reference_report_path"], "r", encoding="utf-8") as f:
                dataset_report = json.load(f)
            with open(outputs["merge_report_path"], "r", encoding="utf-8") as f:
                merge_report = json.load(f)

            self.assertEqual(merged_features.shape[0], 7)
            self.assertEqual(len(merged_names), 7)
            self.assertNotIn("tiny_label0_0", merged_names)
            self.assertEqual(int(merged_metadata["input_n_models"]), 8)
            self.assertEqual(int(merged_metadata["n_models"]), 7)
            self.assertEqual(int(merged_metadata["skipped_model_count"]), 1)
            self.assertEqual(merged_metadata["skipped_models"][0]["model_name"], "tiny_label0_0")
            self.assertEqual(int(dataset_report["artifact_model_count"]), 7)
            self.assertEqual(int(dataset_report["resolved_model_count"]), 7)
            self.assertEqual(int(merge_report["manifest_model_count"]), 8)
            self.assertEqual(int(merge_report["skipped_model_count"]), 1)

    def test_merge_spectral_shards_allows_version_mismatch_when_schema_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)
            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )

            shard_a_items = items[:4]
            shard_b_items = items[4:]
            shard_a_features, shard_a_labels, shard_a_names, shard_a_metadata = extract_spectral_features(
                items=shard_a_items,
                spectral_features=["energy"],
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )
            shard_b_features, shard_b_labels, shard_b_names, shard_b_metadata = extract_spectral_features(
                items=shard_b_items,
                spectral_features=["energy"],
                sv_top_k=1,
                block_size=64,
                dtype=np.float32,
            )

            runs_root = tmp_path / "shards"
            shard_a_run = runs_root / "feature_extract" / "shard_0"
            shard_b_run = runs_root / "feature_extract" / "shard_1"
            write_merged_feature_artifacts(
                shard_a_run / "features",
                features=shard_a_features,
                labels=shard_a_labels,
                model_names=shard_a_names,
                metadata=shard_a_metadata,
            )
            write_merged_feature_artifacts(
                shard_b_run / "features",
                features=shard_b_features,
                labels=shard_b_labels,
                model_names=shard_b_names,
                metadata=shard_b_metadata,
            )

            shard_a_metadata_path = shard_a_run / "features" / "spectral_metadata.json"
            public_shard_a_metadata, shard_a_internal_metadata = (
                load_public_and_internal_spectral_metadata(shard_a_metadata_path)
            )
            shard_a_internal_metadata["extractor_version"] = "2.4.0"
            raw_dataset_layouts = public_shard_a_metadata.get("dataset_layouts")
            dataset_layouts = (
                [dict(entry) for entry in raw_dataset_layouts]
                if isinstance(raw_dataset_layouts, list)
                else None
            )
            write_spectral_metadata(
                shard_a_metadata_path,
                internal_metadata=shard_a_internal_metadata,
                dataset_layouts=dataset_layouts,
            )

            merged_dir = tmp_path / "merged"
            outputs = merge_spectral_shards(
                manifest_json=manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[shard_a_run, shard_b_run],
                output_dir=merged_dir,
            )

            merged_features = np.load(outputs["feature_path"])
            with open(outputs["model_names_path"], "r", encoding="utf-8") as f:
                merged_names = json.load(f)

            self.assertEqual(merged_features.shape[0], 8)
            self.assertEqual(len(merged_names), 8)

    def test_merge_spectral_shards_appends_new_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)

            base_manifest = tmp_path / "manifest_base.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "path": [
                            {"path": "tiny_data/tiny_label0_", "indices": [0, 1]},
                            {"path": "tiny_data/tiny_label1_", "indices": [0, 1]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            new_manifest = tmp_path / "manifest_new.json"
            new_manifest.write_text(
                json.dumps(
                    {
                        "path": [
                            {"path": "tiny_data/tiny_label0_", "indices": [2, 3]},
                            {"path": "tiny_data/tiny_label1_", "indices": [2, 3]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            full_manifest = tmp_path / "manifest_full.json"
            write_single_manifest(full_manifest)

            runs_root = tmp_path / "runs"
            base_merged_dir = tmp_path / "merged_rows_base"
            new_merged_dir = tmp_path / "merged_rows_new"
            final_merged_dir = tmp_path / "merged_rows_final"
            extract_args = [
                "feature",
                "extract",
                "--dataset-root",
                str(tmp_path),
                "--extractor",
                "spectral",
                "--spectral-features",
                "energy",
                "l2_norm",
                "sv_topk",
                "--spectral-sv-top-k",
                "2",
                "--output-root",
                str(runs_root),
            ]

            first = run_cli(
                [*extract_args, "--manifest-json", str(base_manifest), "--run-id", "rows_base"],
                cwd=REPO_ROOT,
            )
            if first.returncode != 0:
                self.fail(first.stderr + first.stdout)

            second = run_cli(
                [*extract_args, "--manifest-json", str(new_manifest), "--run-id", "rows_new"],
                cwd=REPO_ROOT,
            )
            if second.returncode != 0:
                self.fail(second.stderr + second.stdout)

            merge_spectral_shards(
                manifest_json=base_manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[runs_root / "feature_extract" / "rows_base"],
                output_dir=base_merged_dir,
            )
            merge_spectral_shards(
                manifest_json=new_manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[runs_root / "feature_extract" / "rows_new"],
                output_dir=new_merged_dir,
            )
            merge_feature_files(
                feature_paths=[
                    base_merged_dir / "spectral_features.npy",
                    new_merged_dir / "spectral_features.npy",
                ],
                output_filename=final_merged_dir / "spectral_features.npy",
            )

            merged_features = np.load(final_merged_dir / "spectral_features.npy")
            with open(final_merged_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                merged_names = [str(x) for x in json.load(f)]
            merged_index = {name: i for i, name in enumerate(merged_names)}

            full_items = parse_single_manifest_json(
                manifest_path=full_manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            expected_features, _, expected_names, _ = extract_spectral_features(
                items=full_items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                sv_top_k=2,
                block_size=131072,
                dtype=np.float32,
            )

            self.assertEqual(set(merged_names), set(expected_names))
            reorder = np.asarray([merged_index[name] for name in expected_names], dtype=np.int64)
            np.testing.assert_allclose(merged_features[reorder], expected_features, rtol=1e-5, atol=1e-5)

    def test_merge_spectral_shards_with_string_entry_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            items = parse_single_manifest_json(
                manifest_path=manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            all_entries = [str(item.adapter_path.resolve()) for item in items]
            mid = len(all_entries) // 2

            shard0_manifest = tmp_path / "shard_0.json"
            shard1_manifest = tmp_path / "shard_1.json"
            shard0_manifest.write_text(json.dumps({"path": all_entries[:mid]}), encoding="utf-8")
            shard1_manifest.write_text(json.dumps({"path": all_entries[mid:]}), encoding="utf-8")

            shard_runs_root = tmp_path / "runs_shards"
            common_extract_args = [
                "feature",
                "extract",
                "--dataset-root",
                str(tmp_path),
                "--extractor",
                "spectral",
                "--spectral-features",
                "energy",
                "l2_norm",
                "sv_topk",
                "--spectral-sv-top-k",
                "2",
                "--output-root",
                str(shard_runs_root),
            ]

            shard0 = run_cli(
                [
                    *common_extract_args,
                    "--manifest-json",
                    str(shard0_manifest),
                    "--run-id",
                    "shard_0",
                ],
                cwd=REPO_ROOT,
            )
            if shard0.returncode != 0:
                self.fail(shard0.stderr + shard0.stdout)

            shard1 = run_cli(
                [
                    *common_extract_args,
                    "--manifest-json",
                    str(shard1_manifest),
                    "--run-id",
                    "shard_1",
                ],
                cwd=REPO_ROOT,
            )
            if shard1.returncode != 0:
                self.fail(shard1.stderr + shard1.stdout)

            merged_dir = tmp_path / "merged"
            merge_spectral_shards(
                manifest_json=manifest,
                dataset_root=tmp_path,
                shard_run_dirs=[
                    shard_runs_root / "feature_extract" / "shard_0",
                    shard_runs_root / "feature_extract" / "shard_1",
                ],
                output_dir=merged_dir,
            )

            merged_features = np.load(merged_dir / "spectral_features.npy")
            with open(merged_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                merged_names = [str(x) for x in json.load(f)]

            expected_features, _, expected_names, _ = extract_spectral_features(
                items=items,
                spectral_features=["energy", "l2_norm", "sv_topk"],
                sv_top_k=2,
                block_size=131072,
                dtype=np.float32,
            )
            np.testing.assert_allclose(merged_features, expected_features, rtol=1e-5, atol=1e-5)
            self.assertEqual(merged_names, expected_names)

    def test_merge_spectral_shards_rejects_mismatched_moment_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            common_args = [
                "feature",
                "extract",
                "--manifest-json",
                str(manifest),
                "--dataset-root",
                str(tmp_path),
                "--extractor",
                "spectral",
                "--spectral-features",
                "kurtosis",
                "sv_topk",
                "--spectral-sv-top-k",
                "2",
                "--output-root",
                str(runs_root),
            ]

            entrywise = run_cli(
                [*common_args, "--spectral-moment-source", "entrywise", "--run-id", "entrywise_run"],
                cwd=REPO_ROOT,
            )
            if entrywise.returncode != 0:
                self.fail(entrywise.stderr + entrywise.stdout)

            sv = run_cli(
                [*common_args, "--spectral-moment-source", "sv", "--run-id", "sv_run"],
                cwd=REPO_ROOT,
            )
            if sv.returncode != 0:
                self.fail(sv.stderr + sv.stdout)

            with self.assertRaisesRegex(ValueError, "Incompatible shard metadata"):
                merge_spectral_shards(
                    manifest_json=manifest,
                    dataset_root=tmp_path,
                    shard_run_dirs=[
                        runs_root / "feature_extract" / "entrywise_run",
                        runs_root / "feature_extract" / "sv_run",
                    ],
                    output_dir=tmp_path / "merged",
                )

    def test_supervised_local_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "supervised_local_test"
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id=f"{run_id}_features",
            )

            proc = run_cli(
                [
                    "run",
                    "supervised",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "logistic_regression",
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--feature-file",
                    str(feature_path),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                    "--tuning-executor",
                    "local",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            run_dir = runs_root / "supervised" / run_id
            report_path = run_dir / "reports" / "supervised_report.json"
            self.assertTrue(report_path.exists())
            self.assertTrue((run_dir / "models" / "best_model.joblib").exists())
            self.assertTrue((run_dir / "reports" / "train_scores.csv").exists())

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            with open(run_dir / "reports" / "winner_feature_weights_metadata.json", "r", encoding="utf-8") as f:
                weight_metadata = json.load(f)
            metadata = report["representation"]["extractor_metadata"]
            self.assertNotIn("feature_names", metadata)
            self.assertNotIn("a_shapes", metadata)
            self.assertNotIn("b_shapes", metadata)
            self.assertNotIn("block_names_raw", metadata)
            self.assertNotIn("base_block_names_raw", metadata)
            self.assertNotIn("qv_sum_block_names_raw", metadata)
            self.assertNotIn("incoming_metadata", metadata)
            self.assertNotIn("merge_stats", metadata)
            self.assertNotIn("merged_with_existing_output", metadata)
            self.assertNotIn("component_grid", metadata.get("extractor_params", {}))
            self.assertNotIn("n_components", metadata.get("extractor_params", {}))
            self.assertEqual(len(metadata["block_names"]), len(metadata["lora_adapter_dims"]))
            self.assertEqual(metadata["lora_adapter_dims"][0], {"m": 4, "n": 4, "r": 2})
            self.assertEqual(report["tuning"]["winner"]["normalization_policy"], "standard_scaler")
            self.assertEqual(weight_metadata["importance_type"], "coefficient")

    def test_supervised_registry_expanded_models_and_pipelines(self):
        expected_counts = {
            "adaboost": 32,
            "kernel_svm": 40,
            "linear_svm": 12,
            "logistic_regression": 12,
            "random_forest": 54,
            "ridge_classifier": 12,
        }
        expected_policies = {
            "adaboost": "passthrough",
            "kernel_svm": "standard_scaler",
            "linear_svm": "standard_scaler",
            "logistic_regression": "standard_scaler",
            "random_forest": "passthrough",
            "ridge_classifier": "standard_scaler",
        }

        self.assertEqual(registered_models(), sorted(expected_counts))

        for model_name, expected_count in expected_counts.items():
            params = candidate_params(model_name)
            self.assertEqual(len(params), expected_count)
            self.assertEqual(normalization_policy(model_name), expected_policies[model_name])

            pipeline = create(model_name, params=params[0], random_state=42)
            self.assertIsInstance(pipeline, Pipeline)
            self.assertEqual(pipeline.steps[0][0], "normalizer")
            self.assertEqual(pipeline.steps[1][0], "model")
            if expected_policies[model_name] == "standard_scaler":
                self.assertIsInstance(pipeline.steps[0][1], StandardScaler)
            else:
                self.assertEqual(pipeline.steps[0][1], "passthrough")

    def test_supervised_prepare_all_has_expanded_task_count_and_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "supervised_all_prepare_test"
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id=f"{run_id}_features",
            )
            proc = run_cli(
                [
                    "run",
                    "supervised",
                    "--stage",
                    "prepare",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "all",
                    "--feature-file",
                    str(feature_path),
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--cv-folds",
                    "2",
                    "--cv-seeds",
                    "42",
                    "43",
                    "44",
                    "--n-jobs",
                    "1",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            tuning_manifest = runs_root / "supervised" / run_id / "reports" / "tuning_manifest.json"
            with open(tuning_manifest, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(len(payload["tuning"]["tasks"]), 162)
            self.assertEqual(payload["tuning"]["estimated_total_fits"], 972)
            self.assertTrue(
                any("Large supervised grid search" in str(x) for x in payload.get("warnings", []))
            )
            self.assertEqual(
                sorted({task["normalization_policy"] for task in payload["tuning"]["tasks"]}),
                ["passthrough", "standard_scaler"],
            )

    def test_supervised_adaboost_local_smoke(self):
        report, metadata = self._run_supervised_local_smoke(
            model_name="adaboost",
            expected_importance_type="feature_importance",
            expected_normalization_policy="passthrough",
        )
        self.assertEqual(report["tuning"]["tasks_total"], 32)
        self.assertFalse(metadata["has_signed_direction"])

    def test_supervised_random_forest_local_smoke(self):
        report, metadata = self._run_supervised_local_smoke(
            model_name="random_forest",
            expected_importance_type="feature_importance",
            expected_normalization_policy="passthrough",
        )
        self.assertEqual(report["tuning"]["tasks_total"], 54)
        self.assertFalse(metadata["has_signed_direction"])

    def test_supervised_kernel_svm_local_smoke(self):
        report, metadata = self._run_supervised_local_smoke(
            model_name="kernel_svm",
            expected_importance_type="permutation_importance",
            expected_normalization_policy="standard_scaler",
        )
        self.assertEqual(report["tuning"]["tasks_total"], 40)
        self.assertFalse(metadata["has_signed_direction"])

    def test_supervised_uses_external_precomputed_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            feature_run_id = "feature_spectral_for_supervised"
            extract = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    feature_run_id,
                ],
                cwd=REPO_ROOT,
            )
            if extract.returncode != 0:
                self.fail(extract.stderr + extract.stdout)

            feature_dir = runs_root / "feature_extract" / feature_run_id / "features"
            src_features = np.load(feature_dir / "spectral_features.npy")
            with open(feature_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                src_names = [str(x) for x in json.load(f)]
            public_src_metadata, src_metadata = load_public_and_internal_spectral_metadata(
                feature_dir / "spectral_metadata.json"
            )
            self.assertNotIn("feature_names", public_src_metadata)
            self.assertNotIn("block_names", public_src_metadata)
            self.assertNotIn("lora_adapter_dims", public_src_metadata)
            self.assertNotIn("a_shapes", src_metadata)
            self.assertNotIn("b_shapes", src_metadata)
            self.assertNotIn("block_names_raw", src_metadata)
            self.assertNotIn("base_block_names_raw", src_metadata)
            self.assertNotIn("qv_sum_block_names_raw", src_metadata)
            self.assertNotIn("component_grid", src_metadata.get("extractor_params", {}))
            self.assertNotIn("n_components", src_metadata.get("extractor_params", {}))
            self.assertEqual(len(src_metadata["block_names"]), len(src_metadata["lora_adapter_dims"]))

            external_dir = tmp_path / "external_features"
            external_dir.mkdir(parents=True, exist_ok=True)
            np.save(external_dir / "spectral_features.npy", src_features[::-1].copy())
            with open(external_dir / "spectral_model_names.json", "w", encoding="utf-8") as f:
                json.dump(list(reversed(src_names)), f, indent=2)
            clone_spectral_metadata_with_state(
                feature_dir / "spectral_metadata.json",
                external_dir / "spectral_metadata.json",
            )

            run_id = "supervised_external_features_test"
            proc = run_cli(
                [
                    "run",
                    "supervised",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "logistic_regression",
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--feature-file",
                    str(external_dir / "spectral_features.npy"),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                    "--tuning-executor",
                    "local",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            run_dir = runs_root / "supervised" / run_id
            report_path = run_dir / "reports" / "supervised_report.json"
            self.assertTrue(report_path.exists())
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            warnings = report.get("warnings", [])
            self.assertTrue(
                any("Reordered external features to match manifest model order" in str(x) for x in warnings)
            )
            extractor_metadata = report["representation"]["extractor_metadata"]
            self.assertTrue(bool(extractor_metadata.get("loaded_external_features")))

    def test_load_spectral_metadata_preserves_public_dataset_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_path = tmp_path / "spectral_metadata.json"
            internal_metadata = {
                "extractor": "spectral",
                "extractor_name": "spectral",
                "extractor_version": "test",
                "extractor_params": {
                    "spectral_features": ["energy"],
                    "spectral_sv_top_k": 1,
                    "spectral_moment_source": "sv",
                    "spectral_qv_sum_mode": "separate",
                },
                "feature_dim": 2,
                "feature_names": [
                    "base_model.model.model.layers.0.self_attn.q_proj.energy",
                    "base_model.model.model.layers.0.self_attn.v_proj.energy",
                ],
                "block_names": [
                    "base_model.model.model.layers.0.self_attn.q_proj",
                    "base_model.model.model.layers.0.self_attn.v_proj",
                ],
                "base_block_names": [
                    "base_model.model.model.layers.0.self_attn.q_proj",
                    "base_model.model.model.layers.0.self_attn.v_proj",
                ],
                "qv_sum_block_names": [],
                "lora_adapter_dims": [
                    {"m": 4, "n": 4, "r": 2},
                    {"m": 3, "n": 4, "r": 2},
                ],
            }
            dataset_layouts = [
                {
                    "dataset_name": "llama2_7b_imdb_syntactic_rank256_qv",
                    "sample_count": 250,
                    "layer_count": 32,
                    "adapter_dims": {"m": 4096, "n": 4096, "r": 256},
                }
            ]
            write_spectral_metadata(
                metadata_path,
                internal_metadata=internal_metadata,
                dataset_layouts=dataset_layouts,
            )

            loaded_metadata = load_spectral_metadata(metadata_path)
            self.assertEqual(loaded_metadata.get("dataset_layouts"), dataset_layouts)
            self.assertEqual(
                dataset_layouts_from_source(
                    metadata=loaded_metadata,
                    dataset_reference_payload={
                        "dataset_groups": [
                            {
                                "dataset_name": "llama2_7b_imdb_syntactic_rank256_qv",
                                "sample_count": 250,
                            }
                        ],
                        "model_index": {},
                    },
                ),
                dataset_layouts,
            )

    def test_load_spectral_metadata_prefers_public_dataset_layouts_over_stale_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_path = tmp_path / "spectral_metadata.json"
            state_path = metadata_path.with_name(".spectral_metadata_state.json")

            stale_internal_metadata = {
                "extractor": "spectral",
                "extractor_name": "spectral",
                "extractor_version": "test",
                "dataset_layouts": [
                    {
                        "dataset_name": "stale_dataset",
                        "sample_count": 1,
                    }
                ],
            }
            public_metadata = {
                "extractor": "spectral",
                "extractor_name": "spectral",
                "extractor_version": "test",
                "dataset_layouts": [
                    {
                        "dataset_name": "fresh_dataset",
                        "sample_count": 2,
                        "layer_count": 32,
                        "adapter_dims": {"m": 4096, "n": 4096, "r": 256},
                    }
                ],
            }

            state_path.write_text(json.dumps(stale_internal_metadata), encoding="utf-8")
            metadata_path.write_text(json.dumps(public_metadata), encoding="utf-8")

            loaded_metadata = load_spectral_metadata(metadata_path)
            self.assertEqual(loaded_metadata.get("dataset_layouts"), public_metadata["dataset_layouts"])

    def test_write_spectral_metadata_overwrites_stale_internal_dataset_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_path = tmp_path / "spectral_metadata.json"
            internal_metadata = {
                "extractor": "spectral",
                "extractor_name": "spectral",
                "extractor_version": "test",
                "dataset_layouts": [
                    {
                        "dataset_name": "stale_dataset",
                        "sample_count": 1,
                    }
                ],
            }
            dataset_layouts = [
                {
                    "dataset_name": "fresh_dataset",
                    "sample_count": 2,
                    "layer_count": 32,
                    "adapter_dims": {"m": 4096, "n": 4096, "r": 256},
                }
            ]

            write_spectral_metadata(
                metadata_path,
                internal_metadata=internal_metadata,
                dataset_layouts=dataset_layouts,
            )

            loaded_metadata = load_spectral_metadata(metadata_path)
            self.assertEqual(loaded_metadata.get("dataset_layouts"), dataset_layouts)

    def test_supervised_prepare_subsets_external_feature_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path, v_out_dim=4)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            runs_root = tmp_path / "runs"
            feature_run_id = "feature_spectral_external_subset_source"
            extract = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "l2_norm",
                    "energy",
                    "sv_topk",
                    "stable_rank",
                    "--spectral-sv-top-k",
                    "4",
                    "--spectral-qv-sum-mode",
                    "append",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    feature_run_id,
                ],
                cwd=REPO_ROOT,
            )
            if extract.returncode != 0:
                self.fail(extract.stderr + extract.stdout)

            feature_dir = runs_root / "feature_extract" / feature_run_id / "features"
            src_features = np.load(feature_dir / "spectral_features.npy")
            with open(feature_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                src_names = [str(x) for x in json.load(f)]
            public_src_metadata, src_metadata = load_public_and_internal_spectral_metadata(
                feature_dir / "spectral_metadata.json"
            )
            self.assertNotIn("feature_names", public_src_metadata)
            self.assertNotIn("block_names", public_src_metadata)
            self.assertNotIn("lora_adapter_dims", public_src_metadata)

            requested_blocks = [
                str(name) for name in src_metadata["block_names"] if ".qv_sum" in str(name)
            ]
            expected_feature_names: list[str] = []
            for block_name in requested_blocks:
                expected_feature_names.extend(
                    [
                        f"{block_name}.sv_1",
                        f"{block_name}.sv_2",
                        f"{block_name}.energy",
                    ]
                )

            feature_index = {str(name): i for i, name in enumerate(src_metadata["feature_names"])}
            expected_columns = np.asarray([feature_index[name] for name in expected_feature_names], dtype=np.int64)
            expected_features = src_features[:, expected_columns]

            external_dir = tmp_path / "external_features"
            external_dir.mkdir(parents=True, exist_ok=True)
            np.save(external_dir / "spectral_features.npy", src_features)
            with open(external_dir / "spectral_model_names.json", "w", encoding="utf-8") as f:
                json.dump(src_names, f, indent=2)
            clone_spectral_metadata_with_state(
                feature_dir / "spectral_metadata.json",
                external_dir / "spectral_metadata.json",
            )

            run_id = "supervised_external_feature_subset_test"
            run_dir = runs_root / "supervised" / run_id
            prep = run_cli(
                [
                    "run",
                    "supervised",
                    "--stage",
                    "prepare",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "logistic_regression",
                    "--features",
                    "sv_topk",
                    "energy",
                    "--spectral-sv-top-k",
                    "2",
                    "--spectral-qv-sum-mode",
                    "only",
                    "--feature-file",
                    str(external_dir / "spectral_features.npy"),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                ],
                cwd=REPO_ROOT,
            )
            if prep.returncode != 0:
                self.fail(prep.stderr + prep.stdout)

            tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
            with open(tuning_manifest_path, "r", encoding="utf-8") as f:
                tuning_manifest = json.load(f)
            used_features = _load_features_for_tuning_manifest(tuning_manifest)
            np.testing.assert_allclose(used_features, expected_features, rtol=1e-6, atol=1e-6)
            self.assertLess(int(used_features.shape[1]), int(src_features.shape[1]))
            self.assertFalse((run_dir / "features" / "spectral_features.npy").exists())
            self.assertEqual(tuning_manifest["data"]["feature_loading_mode"], "external_source")
            self.assertEqual(
                tuning_manifest["data"]["feature_path"],
                str((external_dir / "spectral_features.npy").resolve()),
            )

            public_used_metadata, used_metadata = load_public_and_internal_spectral_metadata(
                run_dir / "features" / "spectral_metadata.json"
            )
            self.assertEqual(public_used_metadata["resolved_features"], ["sv_topk", "energy"])
            self.assertEqual(int(public_used_metadata["sv_top_k"]), 2)
            self.assertEqual(public_used_metadata["spectral_moment_source"], "sv")
            self.assertEqual(public_used_metadata["spectral_qv_sum_mode"], "only")
            self.assertNotIn("feature_names", public_used_metadata)
            self.assertNotIn("block_names", public_used_metadata)
            self.assertNotIn("lora_adapter_dims", public_used_metadata)
            self.assertEqual([str(x) for x in used_metadata["feature_names"]], expected_feature_names)
            self.assertEqual([str(x) for x in used_metadata["block_names"]], requested_blocks)
            self.assertNotIn("a_shapes", used_metadata)
            self.assertNotIn("b_shapes", used_metadata)
            self.assertNotIn("block_names_raw", used_metadata)
            self.assertNotIn("base_block_names_raw", used_metadata)
            self.assertNotIn("qv_sum_block_names_raw", used_metadata)
            self.assertNotIn("incoming_metadata", used_metadata)
            self.assertNotIn("merge_stats", used_metadata)
            self.assertNotIn("merged_with_existing_output", used_metadata)
            self.assertNotIn("component_grid", used_metadata.get("extractor_params", {}))
            self.assertNotIn("n_components", used_metadata.get("extractor_params", {}))
            self.assertEqual(len(used_metadata["lora_adapter_dims"]), len(requested_blocks))
            self.assertEqual(used_metadata["lora_adapter_dims"][0], {"m": 4, "n": 4, "r": 4})

            with open(run_dir / "reports" / "tuning_manifest.json", "r", encoding="utf-8") as f:
                tuning_manifest = json.load(f)
            warnings = [str(x) for x in tuning_manifest.get("warnings", [])]
            self.assertTrue(
                any(
                    "Filtered/reordered external feature columns to match requested spectral configuration" in warning
                    for warning in warnings
                )
            )
            tuning_metadata = tuning_manifest["extractor"]["metadata"]
            self.assertNotIn("a_shapes", tuning_metadata)
            self.assertNotIn("b_shapes", tuning_metadata)
            self.assertNotIn("block_names_raw", tuning_metadata)
            self.assertNotIn("base_block_names_raw", tuning_metadata)
            self.assertNotIn("qv_sum_block_names_raw", tuning_metadata)
            self.assertNotIn("incoming_metadata", tuning_metadata)
            self.assertNotIn("merge_stats", tuning_metadata)
            self.assertNotIn("merged_with_existing_output", tuning_metadata)
            self.assertNotIn("component_grid", tuning_metadata.get("extractor_params", {}))
            self.assertNotIn("n_components", tuning_metadata.get("extractor_params", {}))

    def test_supervised_external_features_select_noncontiguous_manifest_subset_without_materializing_feature_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)

            full_manifest = tmp_path / "manifest_full.json"
            write_single_manifest(full_manifest)
            subset_manifest = tmp_path / "manifest_subset.json"
            subset_manifest.write_text(
                json.dumps(
                    {
                        "path": [
                            {"path": "tiny_data/tiny_label0_", "indices": [3, 0]},
                            {"path": "tiny_data/tiny_label1_", "indices": [2, 1]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            runs_root = tmp_path / "runs"
            feature_run_id = "feature_spectral_superset_source"
            extract = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(full_manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    feature_run_id,
                ],
                cwd=REPO_ROOT,
            )
            if extract.returncode != 0:
                self.fail(extract.stderr + extract.stdout)

            feature_dir = runs_root / "feature_extract" / feature_run_id / "features"
            src_features = np.load(feature_dir / "spectral_features.npy")
            with open(feature_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                src_names = [str(x) for x in json.load(f)]

            subset_items = parse_single_manifest_json(
                manifest_path=subset_manifest,
                dataset_root=tmp_path,
                section_key="path",
            )
            expected_names = [item.model_name for item in subset_items]
            src_index = {name: i for i, name in enumerate(src_names)}
            expected_features = src_features[
                np.asarray([src_index[name] for name in expected_names], dtype=np.int64)
            ]

            run_id = "supervised_external_subset_rows_test"
            proc = run_cli(
                [
                    "run",
                    "supervised",
                    "--manifest-json",
                    str(subset_manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "logistic_regression",
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--feature-file",
                    str(feature_dir / "spectral_features.npy"),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                    "--tuning-executor",
                    "local",
                ],
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)

            run_dir = runs_root / "supervised" / run_id
            self.assertFalse((run_dir / "features" / "spectral_features.npy").exists())

            tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
            with open(tuning_manifest_path, "r", encoding="utf-8") as f:
                tuning_manifest = json.load(f)
            loaded_features = _load_features_for_tuning_manifest(tuning_manifest)
            np.testing.assert_allclose(loaded_features, expected_features, rtol=1e-6, atol=1e-6)
            self.assertEqual(tuning_manifest["data"]["feature_loading_mode"], "external_source")
            self.assertEqual(
                tuning_manifest["data"]["feature_path"],
                str((feature_dir / "spectral_features.npy").resolve()),
            )
            self.assertTrue(
                any(
                    "Selected external feature rows by manifest model names" in str(x)
                    for x in tuning_manifest.get("warnings", [])
                )
            )

            report_path = run_dir / "reports" / "supervised_report.json"
            self.assertTrue(report_path.exists())
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(int(report["data_info"]["n_samples"]), len(expected_names))
            self.assertTrue(bool(report["representation"]["extractor_metadata"].get("loaded_external_features")))

    def test_supervised_prepare_external_features_compacts_to_available_manifest_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "gmm_manifest.json"
            write_joint_manifest(manifest)

            runs_root = tmp_path / "runs"
            feature_run_id = "feature_spectral_manifest_subset_source"
            extract = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    feature_run_id,
                ],
                cwd=REPO_ROOT,
            )
            if extract.returncode != 0:
                self.fail(extract.stderr + extract.stdout)

            feature_dir = runs_root / "feature_extract" / feature_run_id / "features"
            src_features = np.load(feature_dir / "spectral_features.npy")
            with open(feature_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                src_names = [str(x) for x in json.load(f)]

            missing_name = src_names[-1]
            kept_names = src_names[:-1]
            kept_features = src_features[:-1].copy()

            external_dir = tmp_path / "external_features_missing_one"
            external_dir.mkdir(parents=True, exist_ok=True)
            np.save(external_dir / "spectral_features.npy", kept_features)
            with open(external_dir / "spectral_model_names.json", "w", encoding="utf-8") as f:
                json.dump(kept_names, f, indent=2)
            clone_spectral_metadata_with_state(
                feature_dir / "spectral_metadata.json",
                external_dir / "spectral_metadata.json",
            )

            run_id = "supervised_external_missing_manifest_rows_test"
            prep = run_cli(
                [
                    "run",
                    "supervised",
                    "--stage",
                    "prepare",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "logistic_regression",
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--feature-file",
                    str(external_dir / "spectral_features.npy"),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                    "--tuning-executor",
                    "local",
                ],
                cwd=REPO_ROOT,
            )
            if prep.returncode != 0:
                self.fail(prep.stderr + prep.stdout)

            run_dir = runs_root / "supervised" / run_id
            tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
            with open(tuning_manifest_path, "r", encoding="utf-8") as f:
                tuning_manifest = json.load(f)

            loaded_features = _load_features_for_tuning_manifest(tuning_manifest)
            np.testing.assert_allclose(loaded_features, kept_features, rtol=1e-6, atol=1e-6)

            with open(tuning_manifest["data"]["model_names_path"], "r", encoding="utf-8") as f:
                selected_model_names = [str(x) for x in json.load(f)]

            self.assertEqual(int(tuning_manifest["data"]["n_samples"]), len(kept_names))
            self.assertEqual(len(selected_model_names), len(kept_names))
            self.assertNotIn(missing_name, selected_model_names)
            self.assertEqual(len(tuning_manifest["data"]["train_indices"]), 3)
            self.assertEqual(len(tuning_manifest["data"]["infer_indices"]), 4)

            warnings = [str(x) for x in tuning_manifest.get("warnings", [])]
            self.assertTrue(
                any(
                    "retained 7/8 manifest models and skipped 1 missing model names" in warning
                    for warning in warnings
                )
            )

            extractor_metadata = tuning_manifest["extractor"]["metadata"]
            self.assertEqual(int(extractor_metadata["external_manifest_requested_n_models"]), 8)
            self.assertEqual(int(extractor_metadata["external_manifest_selected_n_models"]), 7)
            self.assertEqual(int(extractor_metadata["external_manifest_missing_model_count"]), 1)
            self.assertTrue(bool(extractor_metadata.get("loaded_external_features")))

    def test_supervised_staged_external_features_do_not_require_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "gmm_manifest.json"
            write_joint_manifest(manifest)

            runs_root = tmp_path / "runs"
            feature_run_id = "feature_spectral_for_staged_supervised"
            extract = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "spectral",
                    "--spectral-features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    feature_run_id,
                ],
                cwd=REPO_ROOT,
            )
            if extract.returncode != 0:
                self.fail(extract.stderr + extract.stdout)

            feature_dir = runs_root / "feature_extract" / feature_run_id / "features"
            src_features = np.load(feature_dir / "spectral_features.npy")
            with open(feature_dir / "spectral_model_names.json", "r", encoding="utf-8") as f:
                src_names = [str(x) for x in json.load(f)]
            _public_src_metadata, _src_metadata = load_public_and_internal_spectral_metadata(
                feature_dir / "spectral_metadata.json"
            )

            external_dir = tmp_path / "external_features"
            external_dir.mkdir(parents=True, exist_ok=True)
            np.save(external_dir / "spectral_features.npy", src_features[::-1].copy())
            with open(external_dir / "spectral_model_names.json", "w", encoding="utf-8") as f:
                json.dump(list(reversed(src_names)), f, indent=2)
            clone_spectral_metadata_with_state(
                feature_dir / "spectral_metadata.json",
                external_dir / "spectral_metadata.json",
            )

            run_id = "supervised_staged_external_features_test"
            run_dir = runs_root / "supervised" / run_id
            missing_root = tmp_path / "models_not_here"

            prep = run_cli(
                [
                    "run",
                    "supervised",
                    "--stage",
                    "prepare",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(missing_root),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "logistic_regression",
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--feature-file",
                    str(external_dir / "spectral_features.npy"),
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                ],
                cwd=REPO_ROOT,
            )
            if prep.returncode != 0:
                self.fail(prep.stderr + prep.stdout)

            tuning_manifest = run_dir / "reports" / "tuning_manifest.json"
            self.assertTrue(tuning_manifest.exists())
            with open(tuning_manifest, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["dataset_root"], str(missing_root))

            n_tasks = len(payload["tuning"]["tasks"])
            self.assertGreater(n_tasks, 0)

            for idx in range(n_tasks):
                worker = run_cli(
                    [
                        "run",
                        "supervised",
                        "--stage",
                        "worker",
                        "--run-dir",
                        str(run_dir),
                        "--task-index",
                        str(idx),
                        "--n-jobs",
                        "1",
                    ],
                    cwd=REPO_ROOT,
                )
                if worker.returncode != 0:
                    self.fail(worker.stderr + worker.stdout)

            finalize = run_cli(
                [
                    "run",
                    "supervised",
                    "--stage",
                    "finalize",
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=REPO_ROOT,
            )
            if finalize.returncode != 0:
                self.fail(finalize.stderr + finalize.stdout)

            report_path = run_dir / "reports" / "supervised_report.json"
            self.assertTrue(report_path.exists())
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            warnings = report.get("warnings", [])
            self.assertTrue(
                any("Reordered external features to match manifest model order" in str(x) for x in warnings)
            )
            self.assertTrue(bool(report["representation"]["extractor_metadata"].get("loaded_external_features")))

    def test_supervised_array_prepare_only_tolerates_missing_companion_env_vars(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "gmm_manifest.json"
            write_joint_manifest(manifest)

            runs_root = tmp_path / "runs"
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id="feature_spectral_for_supervised_array",
            )

            fake_conda_sh = tmp_path / "fake_conda.sh"
            fake_conda_sh.write_text(
                "conda() {\n"
                "  return 0\n"
                "}\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "CONDA_SH": str(fake_conda_sh),
                    "CONDA_ENV": "upeftg",
                    "MANIFEST_JSON": str(manifest),
                    "DATASET_ROOT": str(tmp_path),
                    "FEATURE_FILE": str(feature_path),
                    "FEATURES": "energy kurtosis sv_topk",
                    "OUTPUT_ROOT": str(runs_root),
                    "RUN_ID": "supervised_array_prepare_only_test",
                    "MODEL": "logistic_regression",
                    "SV_TOP_K": "2",
                    "CV_FOLDS": "2",
                    "PIPELINE_MODE": "prepare_only",
                    "SKIP_FEATURE_IMPORTANCE": "1",
                    "SLURM_LOG_DIR": str(tmp_path / "logs"),
                }
            )

            proc = subprocess.run(
                ["bash", "sbatch/supervised_array.sh"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                self.fail(proc.stderr + proc.stdout)
            self.assertIn("Feature companions: auto-resolved from FEATURE_FILE", proc.stdout)
            self.assertIn("Preparation completed; skipping worker/finalize submission.", proc.stdout)

    def test_supervised_prepare_worker_finalize_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "gmm_manifest.json"
            write_joint_manifest(manifest)

            runs_root = tmp_path / "runs"
            run_id = "supervised_staged_test"
            run_dir = runs_root / "supervised" / run_id
            feature_path = extract_feature_bundle(
                manifest_path=manifest,
                dataset_root=tmp_path,
                runs_root=runs_root,
                run_id=f"{run_id}_features",
            )

            prep = run_cli(
                [
                    "run",
                    "supervised",
                    "--stage",
                    "prepare",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--output-root",
                    str(runs_root),
                    "--run-id",
                    run_id,
                    "--model",
                    "logistic_regression",
                    "--feature-file",
                    str(feature_path),
                    "--features",
                    "energy",
                    "kurtosis",
                    "sv_topk",
                    "--spectral-sv-top-k",
                    "2",
                    "--cv-folds",
                    "2",
                    "--n-jobs",
                    "1",
                ],
                cwd=REPO_ROOT,
            )
            if prep.returncode != 0:
                self.fail(prep.stderr + prep.stdout)

            tuning_manifest = run_dir / "reports" / "tuning_manifest.json"
            self.assertTrue(tuning_manifest.exists())
            with open(tuning_manifest, "r", encoding="utf-8") as f:
                payload = json.load(f)
            n_tasks = len(payload["tuning"]["tasks"])
            self.assertGreater(n_tasks, 0)
            self.assertTrue(
                all("normalization_policy" in task for task in payload["tuning"]["tasks"])
            )

            for idx in range(n_tasks):
                worker = run_cli(
                    [
                        "run",
                        "supervised",
                        "--stage",
                        "worker",
                        "--run-dir",
                        str(run_dir),
                        "--task-index",
                        str(idx),
                        "--n-jobs",
                        "1",
                    ],
                    cwd=REPO_ROOT,
                )
                if worker.returncode != 0:
                    self.fail(worker.stderr + worker.stdout)

            finalize = run_cli(
                [
                    "run",
                    "supervised",
                    "--stage",
                    "finalize",
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=REPO_ROOT,
            )
            if finalize.returncode != 0:
                self.fail(finalize.stderr + finalize.stdout)

            self.assertTrue((run_dir / "reports" / "supervised_report.json").exists())
            self.assertTrue((run_dir / "reports" / "inference_scores.csv").exists())
            self.assertTrue((run_dir / "reports" / "winner_feature_weights_coefficients.csv").exists())
            self.assertTrue((run_dir / "reports" / "winner_feature_weights_by_metric.csv").exists())
            self.assertTrue((run_dir / "reports" / "winner_feature_weights_by_block.csv").exists())
            self.assertTrue((run_dir / "reports" / "winner_feature_weights_metadata.json").exists())
            with open(run_dir / "reports" / "supervised_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertTrue(
                all("normalization_policy" in candidate for candidate in report["tuning"]["candidates"])
            )
            self.assertIn("normalization_policy", report["tuning"]["winner"])

    def test_removed_legacy_extractors_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_tiny_adapter_dataset(tmp_path)
            manifest = tmp_path / "prepare_manifest.json"
            write_single_manifest(manifest)

            proc = run_cli(
                [
                    "feature",
                    "extract",
                    "--manifest-json",
                    str(manifest),
                    "--dataset-root",
                    str(tmp_path),
                    "--extractor",
                    "delta_frobenius",
                ],
                cwd=REPO_ROOT,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("invalid choice", (proc.stderr + proc.stdout).lower())


if __name__ == "__main__":
    unittest.main()
