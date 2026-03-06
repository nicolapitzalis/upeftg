from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from safetensors import safe_open

DELTA_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class DeltaBlockSchema:
    pairs: tuple[tuple[str, str], ...]
    block_names: tuple[str, ...]
    a_shapes: tuple[tuple[int, ...], ...]
    b_shapes: tuple[tuple[int, ...], ...]


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


def load_delta_block_schema(adapter_path: Path) -> DeltaBlockSchema:
    pairs, block_names, a_shapes, b_shapes = discover_delta_pairs(adapter_path)
    return DeltaBlockSchema(
        pairs=tuple((str(a_key), str(b_key)) for a_key, b_key in pairs),
        block_names=tuple(str(name) for name in block_names),
        a_shapes=tuple(tuple(int(x) for x in shape) for shape in a_shapes),
        b_shapes=tuple(tuple(int(x) for x in shape) for shape in b_shapes),
    )


def shorten_block_name(block_name: str) -> str:
    parts = block_name.split(".")
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            layer = parts[idx + 1]
            tail = parts[idx + 2 :]
            if tail:
                return "layer" + str(layer) + "." + ".".join(tail)
            return "layer" + str(layer)
    return block_name


def check_consistency(
    adapter_path: Path,
    expected_pairs: list[tuple[str, str]],
    expected_a_shapes: list[tuple[int, ...]],
    expected_b_shapes: list[tuple[int, ...]],
) -> None:
    with safe_open(adapter_path, framework="numpy") as reader:
        check_consistency_reader(
            reader=reader,
            adapter_path=adapter_path,
            expected_pairs=expected_pairs,
            expected_a_shapes=expected_a_shapes,
            expected_b_shapes=expected_b_shapes,
        )


def _tensor_shape(reader: Any, key: str) -> tuple[int, ...]:
    if hasattr(reader, "get_slice"):
        return tuple(int(x) for x in reader.get_slice(key).get_shape())
    return tuple(int(x) for x in reader.get_tensor(key).shape)


def check_consistency_reader(
    *,
    reader: Any,
    adapter_path: Path,
    expected_pairs: list[tuple[str, str]],
    expected_a_shapes: list[tuple[int, ...]],
    expected_b_shapes: list[tuple[int, ...]],
) -> None:
    keys = set(reader.keys())
    for i, (a_key, b_key) in enumerate(expected_pairs):
        if a_key not in keys or b_key not in keys:
            raise ValueError(f"Key mismatch in {adapter_path}: expected {a_key} and {b_key}")
        a_shape = _tensor_shape(reader, a_key)
        b_shape = _tensor_shape(reader, b_key)
        if a_shape != expected_a_shapes[i] or b_shape != expected_b_shapes[i]:
            raise ValueError(
                f"Shape mismatch in {adapter_path} for block index {i}: "
                f"A expected {expected_a_shapes[i]} got {a_shape}; "
                f"B expected {expected_b_shapes[i]} got {b_shape}"
            )


def block_delta_singular_values(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    _, rb = np.linalg.qr(b, mode="reduced")
    _, ra = np.linalg.qr(a.T, mode="reduced")
    s_small = rb @ ra.T
    return np.asarray(np.linalg.svd(s_small, compute_uv=False), dtype=np.float64)


def top_k_singular_values(singular_values: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    out = np.asarray(singular_values, dtype=np.float64)
    if out.size < top_k:
        out = np.pad(out, (0, top_k - out.size))
    return out[:top_k]


def block_delta_spectrum_and_fro(
    a: np.ndarray,
    b: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, float]:
    singular_values = block_delta_singular_values(a=a, b=b)
    top = top_k_singular_values(singular_values, top_k=top_k)
    fro = float(np.linalg.norm(singular_values))
    return top, fro


def iter_block_factors(
    *,
    reader: Any,
    schema: DeltaBlockSchema,
    dtype: np.dtype,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    for block_name, (a_key, b_key) in zip(schema.block_names, schema.pairs):
        a = np.asarray(reader.get_tensor(a_key), dtype=dtype)
        b = np.asarray(reader.get_tensor(b_key), dtype=dtype)
        yield block_name, a, b


def build_schema_metadata(schema: DeltaBlockSchema) -> dict[str, Any]:
    return {
        "delta_schema_version": DELTA_SCHEMA_VERSION,
        "n_blocks": int(len(schema.block_names)),
        "block_names": [shorten_block_name(name) for name in schema.block_names],
        "block_names_raw": list(schema.block_names),
        "a_shapes": [list(shape) for shape in schema.a_shapes],
        "b_shapes": [list(shape) for shape in schema.b_shapes],
    }


def block_spectral_scalars(singular_values: np.ndarray) -> tuple[float, float, float, float]:
    sv = np.asarray(singular_values, dtype=np.float64)
    if sv.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    energy = float(np.sum(np.square(sv), dtype=np.float64))
    max_sv = float(np.max(sv))
    if max_sv <= 0.0:
        stable_rank = 0.0
    else:
        stable_rank = float(energy / max(1e-12, max_sv * max_sv))

    if energy <= 0.0:
        spectral_entropy = 0.0
        effective_rank = 0.0
    else:
        p = np.square(sv) / energy
        p = np.clip(p, 1e-12, 1.0)
        spectral_entropy = float(-np.sum(p * np.log(p), dtype=np.float64))
        effective_rank = float(np.exp(spectral_entropy))

    return energy, stable_rank, spectral_entropy, effective_rank
