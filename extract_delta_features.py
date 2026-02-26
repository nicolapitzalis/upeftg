#!/usr/bin/env python3
"""
Phase 4 baseline: Delta-feature extraction for LoRA adapters.

Extracts low-rank invariant features from Delta weights (Delta W = B @ A):
1. Per-block top singular values (using low-rank algebra, no dense Delta materialization)
2. Per-block Frobenius norms

Outputs can be evaluated by cluster_z_space.py via --feature-file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

SCRIPT_VERSION = "1.0.0"
DEFAULT_DATA_DIR = Path("data/llama3_8b_toxic_backdoors_hard_rank256_qv")
DEFAULT_OUTPUT_DIR = Path("delta_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Delta-based spectral features from LoRA adapters")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Adapter dataset root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for extracted features")
    parser.add_argument("--n-per-label", type=int, default=20, help="Adapters to sample per label")
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="Sampling mode per label",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for random sampling")
    parser.add_argument("--top-k-singular-values", type=int, default=8, help="Top-k singular values per Delta block")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="Computation dtype")
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(x) for x in obj]
    return obj


def parse_label(name: str) -> int | None:
    if "label0" in name:
        return 0
    if "label1" in name:
        return 1
    return None


def select_model_dirs(data_dir: Path, n_per_label: int, sample_mode: str, sample_seed: int) -> list[Path]:
    model_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    label0 = [d for d in model_dirs if parse_label(d.name) == 0]
    label1 = [d for d in model_dirs if parse_label(d.name) == 1]

    if not label0 or not label1:
        raise ValueError(f"Missing label0/label1 directories in {data_dir}")

    if n_per_label > min(len(label0), len(label1)):
        raise ValueError(
            f"Requested n-per-label={n_per_label}, but available label0={len(label0)} label1={len(label1)}"
        )

    if sample_mode == "first":
        selected0 = label0[:n_per_label]
        selected1 = label1[:n_per_label]
    else:
        rng = np.random.default_rng(sample_seed)
        selected0 = [label0[i] for i in rng.choice(len(label0), n_per_label, replace=False)]
        selected1 = [label1[i] for i in rng.choice(len(label1), n_per_label, replace=False)]
        selected0 = sorted(selected0)
        selected1 = sorted(selected1)

    return selected0 + selected1


def discover_delta_pairs(adapter_path: Path) -> tuple[list[tuple[str, str]], list[str], list[tuple[int, ...]], list[tuple[int, ...]]]:
    with safe_open(adapter_path, framework="numpy") as f:
        keys = sorted(f.keys())
        a_keys = [k for k in keys if k.endswith(".lora_A.weight")]
        pairs: list[tuple[str, str]] = []
        block_names: list[str] = []
        a_shapes: list[tuple[int, ...]] = []
        b_shapes: list[tuple[int, ...]] = []

        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            if b_key not in keys:
                raise ValueError(f"Missing matching B tensor for {a_key}")
            pairs.append((a_key, b_key))
            block_names.append(a_key.replace(".lora_A.weight", ""))
            a_shapes.append(tuple(int(x) for x in f.get_tensor(a_key).shape))
            b_shapes.append(tuple(int(x) for x in f.get_tensor(b_key).shape))

    return pairs, block_names, a_shapes, b_shapes


def check_consistency(
    adapter_path: Path,
    expected_pairs: list[tuple[str, str]],
    expected_a_shapes: list[tuple[int, ...]],
    expected_b_shapes: list[tuple[int, ...]],
) -> None:
    with safe_open(adapter_path, framework="numpy") as f:
        keys = set(f.keys())
        for i, (a_key, b_key) in enumerate(expected_pairs):
            if a_key not in keys or b_key not in keys:
                raise ValueError(f"Key mismatch in {adapter_path}: expected {a_key} and {b_key}")
            a_shape = tuple(int(x) for x in f.get_tensor(a_key).shape)
            b_shape = tuple(int(x) for x in f.get_tensor(b_key).shape)
            if a_shape != expected_a_shapes[i] or b_shape != expected_b_shapes[i]:
                raise ValueError(
                    f"Shape mismatch in {adapter_path} for block index {i}: "
                    f"A expected {expected_a_shapes[i]} got {a_shape}; "
                    f"B expected {expected_b_shapes[i]} got {b_shape}"
                )


def block_delta_spectrum_and_fro(
    a: np.ndarray,
    b: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, float]:
    # B = Qb Rb, A = Ra^T Qa^T  => non-zero singular values of (B@A)
    # equal singular values of small square matrix (Rb @ Ra^T).
    _, rb = np.linalg.qr(b, mode="reduced")
    _, ra = np.linalg.qr(a.T, mode="reduced")
    s_small = rb @ ra.T

    singular_values = np.linalg.svd(s_small, compute_uv=False)
    if singular_values.size < top_k:
        singular_values = np.pad(singular_values, (0, top_k - singular_values.size))
    top = singular_values[:top_k]

    fro = float(np.linalg.norm(s_small, ord="fro"))
    return top, fro


def compute_dataset_signature(model_names: list[str], pairs: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for name in model_names:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
    for a_key, b_key in pairs:
        h.update(a_key.encode("utf-8"))
        h.update(b"||")
        h.update(b_key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.top_k_singular_values <= 0:
        raise ValueError("--top-k-singular-values must be > 0")

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    selected = select_model_dirs(
        data_dir=args.data_dir,
        n_per_label=args.n_per_label,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
    )

    if not selected:
        raise RuntimeError("No models selected")

    first_adapter = selected[0] / "adapter_model.safetensors"
    if not first_adapter.exists():
        raise FileNotFoundError(f"Missing adapter file: {first_adapter}")

    pairs, block_names, a_shapes, b_shapes = discover_delta_pairs(first_adapter)

    all_sv: list[list[float]] = []
    all_fro: list[list[float]] = []
    labels: list[int] = []
    model_names: list[str] = []

    print("=" * 80)
    print("Phase 4 baseline: Delta feature extraction")
    print("=" * 80)
    print(f"Selected models: {len(selected)}")
    print(f"Delta blocks per model: {len(pairs)}")
    print(f"Top-k singular values per block: {args.top_k_singular_values}")

    for idx, model_dir in enumerate(selected, start=1):
        adapter_path = model_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing adapter file: {adapter_path}")

        label = parse_label(model_dir.name)
        if label is None:
            raise ValueError(f"Unable to parse label from directory name: {model_dir.name}")

        check_consistency(adapter_path, pairs, a_shapes, b_shapes)

        sv_vec: list[float] = []
        fro_vec: list[float] = []

        with safe_open(adapter_path, framework="numpy") as f:
            for a_key, b_key in pairs:
                a = np.asarray(f.get_tensor(a_key), dtype=dtype)
                b = np.asarray(f.get_tensor(b_key), dtype=dtype)
                top, fro = block_delta_spectrum_and_fro(a=a, b=b, top_k=args.top_k_singular_values)
                sv_vec.extend(top.tolist())
                fro_vec.append(fro)

        all_sv.append(sv_vec)
        all_fro.append(fro_vec)
        labels.append(label)
        model_names.append(model_dir.name)

        print(f"[{idx}/{len(selected)}] Processed {model_dir.name}")

    sv_matrix = np.asarray(all_sv, dtype=np.float32)
    fro_matrix = np.asarray(all_fro, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "delta_singular_values.npy", sv_matrix)
    np.save(args.output_dir / "delta_frobenius.npy", fro_matrix)
    np.save(args.output_dir / "labels.npy", labels_np)

    with open(args.output_dir / "model_names.json", "w", encoding="utf-8") as f:
        json.dump(model_names, f, indent=2)

    metadata = {
        "n_models": int(len(model_names)),
        "n_blocks": int(len(block_names)),
        "top_k_singular_values": int(args.top_k_singular_values),
        "singular_feature_dim": int(sv_matrix.shape[1]),
        "frobenius_feature_dim": int(fro_matrix.shape[1]),
        "block_names": block_names,
        "a_shapes": [list(s) for s in a_shapes],
        "b_shapes": [list(s) for s in b_shapes],
    }

    with open(args.output_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(metadata), f, indent=2)

    run_config = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "dataset_signature": compute_dataset_signature(model_names, pairs),
        "selected_models": model_names,
    }

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(run_config), f, indent=2)

    print("\nSaved artifacts:")
    for file in sorted(args.output_dir.iterdir()):
        size = file.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 ** 2:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / 1024 ** 2:.1f} MB"
        print(f"  {file.name}: {size_str}")


if __name__ == "__main__":
    main()
