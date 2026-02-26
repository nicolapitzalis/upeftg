from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from ..utilities.manifest import ManifestItem


DELTA_EXTRACTOR_VERSION = "1.0.0"


def discover_delta_pairs(adapter_path: Path) -> tuple[list[tuple[str, str]], list[str], list[tuple[int, ...]], list[tuple[int, ...]]]:
    with safe_open(adapter_path, framework="numpy") as reader:
        keys = sorted(reader.keys())
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
            a_shapes.append(tuple(int(x) for x in reader.get_tensor(a_key).shape))
            b_shapes.append(tuple(int(x) for x in reader.get_tensor(b_key).shape))

    return pairs, block_names, a_shapes, b_shapes


def check_consistency(
    adapter_path: Path,
    expected_pairs: list[tuple[str, str]],
    expected_a_shapes: list[tuple[int, ...]],
    expected_b_shapes: list[tuple[int, ...]],
) -> None:
    with safe_open(adapter_path, framework="numpy") as reader:
        keys = set(reader.keys())
        for i, (a_key, b_key) in enumerate(expected_pairs):
            if a_key not in keys or b_key not in keys:
                raise ValueError(f"Key mismatch in {adapter_path}: expected {a_key} and {b_key}")
            a_shape = tuple(int(x) for x in reader.get_tensor(a_key).shape)
            b_shape = tuple(int(x) for x in reader.get_tensor(b_key).shape)
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
    _, rb = np.linalg.qr(b, mode="reduced")
    _, ra = np.linalg.qr(a.T, mode="reduced")
    s_small = rb @ ra.T

    singular_values = np.linalg.svd(s_small, compute_uv=False)
    if singular_values.size < top_k:
        singular_values = np.pad(singular_values, (0, top_k - singular_values.size))
    top = singular_values[:top_k]

    fro = float(np.linalg.norm(s_small, ord="fro"))
    return top, fro


def extract_delta_feature_matrices(
    *,
    items: list[ManifestItem],
    top_k_singular_values: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str], dict[str, Any]]:
    if top_k_singular_values <= 0:
        raise ValueError("top_k_singular_values must be > 0")
    if not items:
        raise ValueError("No adapters provided for delta extraction")

    first_adapter = items[0].adapter_path
    pairs, block_names, a_shapes, b_shapes = discover_delta_pairs(first_adapter)

    all_sv: list[list[float]] = []
    all_fro: list[list[float]] = []
    labels_list = [item.label for item in items]
    model_names = [item.model_name for item in items]

    for item in items:
        adapter_path = item.adapter_path
        check_consistency(adapter_path, pairs, a_shapes, b_shapes)

        sv_vec: list[float] = []
        fro_vec: list[float] = []

        with safe_open(adapter_path, framework="numpy") as reader:
            for a_key, b_key in pairs:
                a = np.asarray(reader.get_tensor(a_key), dtype=dtype)
                b = np.asarray(reader.get_tensor(b_key), dtype=dtype)
                top, fro = block_delta_spectrum_and_fro(a=a, b=b, top_k=top_k_singular_values)
                sv_vec.extend(top.tolist())
                fro_vec.append(fro)

        all_sv.append(sv_vec)
        all_fro.append(fro_vec)

    sv_matrix = np.asarray(all_sv, dtype=np.float32)
    fro_matrix = np.asarray(all_fro, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.int32) if all(x is not None for x in labels_list) else None

    metadata: dict[str, Any] = {
        "extractor": "delta",
        "extractor_version": DELTA_EXTRACTOR_VERSION,
        "n_models": int(len(model_names)),
        "n_blocks": int(len(block_names)),
        "top_k_singular_values": int(top_k_singular_values),
        "singular_feature_dim": int(sv_matrix.shape[1]),
        "frobenius_feature_dim": int(fro_matrix.shape[1]),
        "block_names": block_names,
        "a_shapes": [list(s) for s in a_shapes],
        "b_shapes": [list(s) for s in b_shapes],
    }
    return sv_matrix, fro_matrix, labels, model_names, metadata
