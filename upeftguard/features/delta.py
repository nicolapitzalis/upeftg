from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from safetensors import safe_open

DELTA_SCHEMA_VERSION = "2.1.0"


@dataclass(frozen=True)
class DeltaBlockSchema:
    pairs: tuple[tuple[str, str], ...]
    block_names: tuple[str, ...]
    a_shapes: tuple[tuple[int, ...], ...]
    b_shapes: tuple[tuple[int, ...], ...]
    e_keys: tuple[str | None, ...]
    e_shapes: tuple[tuple[int, ...] | None, ...]


def schema_has_adalora_scaling(schema: DeltaBlockSchema) -> bool:
    return any(key is not None for key in schema.e_keys)


def _match_lora_a_key(key: str) -> tuple[str, str] | None:
    if key.endswith(".lora_A.weight"):
        return key[: -len(".lora_A.weight")], ".lora_A.weight"
    if key.endswith(".lora_A"):
        return key[: -len(".lora_A")], ".lora_A"
    return None


def _matching_factor_keys(
    block_name: str,
    a_suffix: str,
    keys: set[str],
) -> tuple[str, str | None]:
    if a_suffix == ".lora_A.weight":
        return block_name + ".lora_B.weight", None
    if a_suffix == ".lora_A":
        e_key = block_name + ".lora_E"
        return block_name + ".lora_B", e_key if e_key in keys else None
    raise ValueError(f"Unsupported LoRA A suffix for {block_name}: {a_suffix}")


def _normalize_optional_shape(shape: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return tuple(int(x) for x in shape)


def discover_delta_pairs(adapter_path: Path) -> tuple[list[tuple[str, str]], list[str], list[tuple[int, ...]], list[tuple[int, ...]]]:
    schema = load_delta_block_schema(adapter_path)
    return (
        [tuple(pair) for pair in schema.pairs],
        [str(name) for name in schema.block_names],
        [tuple(int(x) for x in shape) for shape in schema.a_shapes],
        [tuple(int(x) for x in shape) for shape in schema.b_shapes],
    )


def _tensor_shape(reader: Any, key: str) -> tuple[int, ...]:
    if hasattr(reader, "get_slice"):
        return tuple(int(x) for x in reader.get_slice(key).get_shape())
    return tuple(int(x) for x in reader.get_tensor(key).shape)


def _load_schema_fields(
    adapter_path: Path,
) -> tuple[
    list[tuple[str, str]],
    list[str],
    list[tuple[int, ...]],
    list[tuple[int, ...]],
    list[str | None],
    list[tuple[int, ...] | None],
]:
    with safe_open(adapter_path, framework="numpy") as reader:
        keys = sorted(reader.keys())
        key_set = set(keys)
        a_keys = [k for k in keys if _match_lora_a_key(k) is not None]
        pairs: list[tuple[str, str]] = []
        block_names: list[str] = []
        a_shapes: list[tuple[int, ...]] = []
        b_shapes: list[tuple[int, ...]] = []
        e_keys: list[str | None] = []
        e_shapes: list[tuple[int, ...] | None] = []

        for a_key in a_keys:
            matched = _match_lora_a_key(a_key)
            if matched is None:
                continue
            block_name, a_suffix = matched
            b_key, e_key = _matching_factor_keys(block_name, a_suffix, key_set)
            if b_key not in key_set:
                raise ValueError(f"Missing matching B tensor for {a_key}")
            pairs.append((a_key, b_key))
            block_names.append(block_name)
            a_shapes.append(_tensor_shape(reader, a_key))
            b_shapes.append(_tensor_shape(reader, b_key))
            e_keys.append(e_key)
            e_shapes.append(_tensor_shape(reader, e_key) if e_key is not None else None)

    return pairs, block_names, a_shapes, b_shapes, e_keys, e_shapes


def load_delta_block_schema(adapter_path: Path) -> DeltaBlockSchema:
    pairs, block_names, a_shapes, b_shapes, e_keys, e_shapes = _load_schema_fields(adapter_path)
    return DeltaBlockSchema(
        pairs=tuple((str(a_key), str(b_key)) for a_key, b_key in pairs),
        block_names=tuple(str(name) for name in block_names),
        a_shapes=tuple(tuple(int(x) for x in shape) for shape in a_shapes),
        b_shapes=tuple(tuple(int(x) for x in shape) for shape in b_shapes),
        e_keys=tuple(str(key) if key is not None else None for key in e_keys),
        e_shapes=tuple(_normalize_optional_shape(shape) for shape in e_shapes),
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
    if "block" in parts:
        idx = parts.index("block")
        if idx + 1 < len(parts):
            scope_prefix = ""
            for scope_name in reversed(parts[:idx]):
                if scope_name in {"encoder", "decoder"}:
                    scope_prefix = scope_name + "."
                    break
            block = parts[idx + 1]
            next_idx = idx + 2
            if next_idx + 1 < len(parts) and parts[next_idx] == "layer":
                layer = parts[next_idx + 1]
                tail = parts[next_idx + 2 :]
                prefix = scope_prefix + "block" + str(block) + ".layer" + str(layer)
                if tail:
                    return prefix + "." + ".".join(tail)
                return prefix
            tail = parts[next_idx:]
            if tail:
                return scope_prefix + "block" + str(block) + "." + ".".join(tail)
            return scope_prefix + "block" + str(block)
    return block_name


def check_consistency(
    adapter_path: Path,
    expected_pairs: list[tuple[str, str]],
    expected_a_shapes: list[tuple[int, ...]],
    expected_b_shapes: list[tuple[int, ...]],
    expected_e_keys: list[str | None] | None = None,
    expected_e_shapes: list[tuple[int, ...] | None] | None = None,
    allow_rank_variation: bool = False,
) -> None:
    with safe_open(adapter_path, framework="numpy") as reader:
        check_consistency_reader(
            reader=reader,
            adapter_path=adapter_path,
            expected_pairs=expected_pairs,
            expected_a_shapes=expected_a_shapes,
            expected_b_shapes=expected_b_shapes,
            expected_e_keys=expected_e_keys,
            expected_e_shapes=expected_e_shapes,
            allow_rank_variation=allow_rank_variation,
        )


def _reshape_vector_tensor(
    tensor: np.ndarray,
    *,
    key: str,
    tensor_kind: str,
) -> np.ndarray:
    if tensor.ndim == 2 and 1 in tensor.shape:
        return tensor.reshape(-1)
    if tensor.ndim == 1:
        return tensor
    raise ValueError(
        f"Expected {tensor_kind} {key} to be rank-1 or singleton rank-2, got shape {tensor.shape}"
    )


def load_effective_block_factors(
    *,
    reader: Any,
    a_key: str,
    b_key: str,
    e_key: str | None,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(reader.get_tensor(a_key), dtype=dtype)
    b = np.asarray(reader.get_tensor(b_key), dtype=dtype)
    if e_key is not None:
        e = _reshape_vector_tensor(
            np.asarray(reader.get_tensor(e_key), dtype=dtype),
            key=e_key,
            tensor_kind="AdaLoRA scaling tensor",
        )
        if int(e.shape[0]) != int(a.shape[0]) or int(e.shape[0]) != int(b.shape[1]):
            raise ValueError(
                f"Incompatible AdaLoRA factor shapes for {a_key}: A{a.shape}, B{b.shape}, E{tuple(int(x) for x in e.shape)}"
            )
        a = a * e.reshape(-1, 1)

    return a, b


def check_consistency_reader(
    *,
    reader: Any,
    adapter_path: Path,
    expected_pairs: list[tuple[str, str]],
    expected_a_shapes: list[tuple[int, ...]],
    expected_b_shapes: list[tuple[int, ...]],
    expected_e_keys: list[str | None] | None = None,
    expected_e_shapes: list[tuple[int, ...] | None] | None = None,
    allow_rank_variation: bool = False,
) -> None:
    if len(expected_pairs) != len(expected_a_shapes) or len(expected_pairs) != len(expected_b_shapes):
        raise ValueError(
            "Expected delta schema lengths must match: "
            f"pairs={len(expected_pairs)}, A={len(expected_a_shapes)}, B={len(expected_b_shapes)}"
        )
    if expected_e_keys is not None and len(expected_e_keys) != len(expected_pairs):
        raise ValueError(
            f"Expected E-key count must match pairs: {len(expected_e_keys)} vs {len(expected_pairs)}"
        )
    if expected_e_shapes is not None and len(expected_e_shapes) != len(expected_pairs):
        raise ValueError(
            f"Expected E-shape count must match pairs: {len(expected_e_shapes)} vs {len(expected_pairs)}"
        )
    keys = set(reader.keys())
    for i, (a_key, b_key) in enumerate(expected_pairs):
        if a_key not in keys or b_key not in keys:
            raise ValueError(f"Key mismatch in {adapter_path}: expected {a_key} and {b_key}")
        a_shape = _tensor_shape(reader, a_key)
        b_shape = _tensor_shape(reader, b_key)
        if allow_rank_variation:
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise ValueError(
                    f"Expected rank-2 LoRA tensors in {adapter_path} for block index {i}, got A{a_shape}, B{b_shape}"
                )
            if int(a_shape[1]) != int(expected_a_shapes[i][1]) or int(b_shape[0]) != int(expected_b_shapes[i][0]):
                raise ValueError(
                    f"Shape mismatch in {adapter_path} for block index {i}: "
                    f"expected in/out dims {(expected_a_shapes[i][1], expected_b_shapes[i][0])} "
                    f"but got {(a_shape[1], b_shape[0])}"
                )
            if int(a_shape[0]) != int(b_shape[1]):
                raise ValueError(
                    f"LoRA rank mismatch in {adapter_path} for block index {i}: A{a_shape}, B{b_shape}"
                )
        elif a_shape != expected_a_shapes[i] or b_shape != expected_b_shapes[i]:
            raise ValueError(
                f"Shape mismatch in {adapter_path} for block index {i}: "
                f"A expected {expected_a_shapes[i]} got {a_shape}; "
                f"B expected {expected_b_shapes[i]} got {b_shape}"
            )

        if expected_e_keys is not None:
            e_key = expected_e_keys[i]
            if e_key is not None:
                if e_key not in keys:
                    raise ValueError(f"Key mismatch in {adapter_path}: expected AdaLoRA scaling tensor {e_key}")

                if expected_e_shapes is not None:
                    expected_e_shape = expected_e_shapes[i]
                    if expected_e_shape is not None:
                        e_shape = _tensor_shape(reader, e_key)
                        if allow_rank_variation:
                            if len(e_shape) == 2 and 1 in e_shape:
                                e_len = int(np.prod(e_shape, dtype=np.int64))
                            elif len(e_shape) == 1:
                                e_len = int(e_shape[0])
                            else:
                                raise ValueError(
                                    f"Expected AdaLoRA scaling tensor {e_key} in {adapter_path} to be rank-1 or singleton rank-2, "
                                    f"got shape {e_shape}"
                                )
                            if e_len != int(a_shape[0]) or e_len != int(b_shape[1]):
                                raise ValueError(
                                    f"Incompatible AdaLoRA factor shapes in {adapter_path} for block index {i}: "
                                    f"A{a_shape}, B{b_shape}, E{e_shape}"
                                )
                        elif e_shape != expected_e_shape:
                            raise ValueError(
                                f"Shape mismatch in {adapter_path} for block index {i}: "
                                f"E expected {expected_e_shape} got {e_shape}"
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
    for block_name, (a_key, b_key), e_key in zip(schema.block_names, schema.pairs, schema.e_keys):
        a, b = load_effective_block_factors(
            reader=reader,
            a_key=a_key,
            b_key=b_key,
            e_key=e_key,
            dtype=dtype,
        )
        yield block_name, a, b


def lora_adapter_dims_from_shapes(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
) -> dict[str, int]:
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"Expected rank-2 LoRA shapes, got A{a_shape}, B{b_shape}")

    rank_a, in_dim = (int(x) for x in a_shape)
    out_dim, rank_b = (int(x) for x in b_shape)
    if rank_a != rank_b:
        raise ValueError(f"LoRA rank mismatch for shapes A{a_shape}, B{b_shape}")

    return {
        "m": int(out_dim),
        "n": int(in_dim),
        "r": int(rank_a),
    }


def build_schema_metadata(schema: DeltaBlockSchema) -> dict[str, Any]:
    has_adalora_scaling = [bool(key) for key in schema.e_keys]
    return {
        "delta_schema_version": DELTA_SCHEMA_VERSION,
        "n_blocks": int(len(schema.block_names)),
        "block_names": [shorten_block_name(name) for name in schema.block_names],
        "has_adalora_scaling": has_adalora_scaling,
        "variable_lora_rank": bool(any(has_adalora_scaling)),
        **(
            {}
            if any(has_adalora_scaling)
            else {
                "lora_adapter_dims": [
                    lora_adapter_dims_from_shapes(
                        tuple(int(x) for x in a_shape),
                        tuple(int(x) for x in b_shape),
                    )
                    for a_shape, b_shape in zip(schema.a_shapes, schema.b_shapes)
                ],
                "e_shapes": [
                    list(shape) if shape is not None else None
                    for shape in schema.e_shapes
                ],
            }
        ),
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
