from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from safetensors import safe_open


ADAPTER_FILENAME = "adapter_model.safetensors"


def extract_layers_from_keys(keys: tuple[str, ...]) -> list[int]:
    layers: set[int] = set()
    for key in keys:
        if ".layers." not in key:
            continue
        parts = key.split(".")
        layer_idx = parts.index("layers") + 1
        layers.add(int(parts[layer_idx]))
    return sorted(layers)


def get_tensor_shape_safe(reader: Any, key: str) -> tuple[int, ...]:
    if hasattr(reader, "get_slice"):
        return tuple(int(x) for x in reader.get_slice(key).get_shape())
    return tuple(int(x) for x in reader.get_tensor(key).shape)


def inspect_adapter_schema(
    adapter_path: Path,
    expected_keys: tuple[str, ...] | None,
    expected_shapes: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], list[int], int]:
    with safe_open(adapter_path, framework="numpy") as reader:
        keys = tuple(sorted(reader.keys()))
        if expected_keys is not None and keys != expected_keys:
            raise ValueError(f"Key mismatch for {adapter_path}. Expected exactly the same tensor key set")

        shapes: list[tuple[int, ...]] = []
        n_params = 0
        for i, key in enumerate(keys):
            shape = get_tensor_shape_safe(reader, key)
            shapes.append(shape)
            if expected_shapes is not None and shape != expected_shapes[i]:
                raise ValueError(
                    f"Shape mismatch for {adapter_path} at key {key}: "
                    f"expected {expected_shapes[i]}, found {shape}"
                )
            n_params += int(np.prod(shape, dtype=np.int64))

    return keys, tuple(shapes), extract_layers_from_keys(keys), n_params


def iter_tensor_flat_chunks(
    tensor_slice: Any,
    shape: tuple[int, ...],
    block_size: int,
    dtype: np.dtype,
) -> Iterator[np.ndarray]:
    if len(shape) == 0:
        chunk = np.asarray(tensor_slice[()], dtype=dtype).reshape(-1)
        if chunk.size:
            yield chunk
        return

    if len(shape) == 1:
        step = max(1, block_size)
        for start in range(0, shape[0], step):
            end = min(start + step, shape[0])
            chunk = np.asarray(tensor_slice[start:end], dtype=dtype).reshape(-1)
            if chunk.size:
                yield chunk
        return

    row_width = int(np.prod(shape[1:], dtype=np.int64))
    rows_per_chunk = max(1, block_size // max(1, row_width))
    for row_start in range(0, shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, shape[0])
        chunk = np.asarray(tensor_slice[row_start:row_end], dtype=dtype).reshape(-1)
        if chunk.size == 0:
            continue
        for offset in range(0, chunk.size, block_size):
            sub = chunk[offset : offset + block_size]
            if sub.size:
                yield sub


def iter_reader_flat_chunks(
    reader: Any,
    *,
    adapter_path: Path,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    block_size: int,
    dtype: np.dtype,
) -> Iterator[np.ndarray]:
    keys = tuple(sorted(reader.keys()))
    if keys != expected_keys:
        raise ValueError(f"Key mismatch for {adapter_path}. Expected exactly the same tensor key set")

    for i, key in enumerate(keys):
        expected_shape = expected_shapes[i]
        actual_shape = get_tensor_shape_safe(reader, key)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {adapter_path} at key {key}: "
                f"expected {expected_shape}, found {actual_shape}"
            )

        if hasattr(reader, "get_slice"):
            tensor_reader = reader.get_slice(key)
            chunks = iter_tensor_flat_chunks(
                tensor_slice=tensor_reader,
                shape=actual_shape,
                block_size=block_size,
                dtype=dtype,
            )
        else:
            full = np.asarray(reader.get_tensor(key), dtype=dtype).reshape(-1)
            chunks = (full[j : j + block_size] for j in range(0, full.size, block_size))

        for chunk in chunks:
            yield np.asarray(chunk, dtype=dtype).reshape(-1)


def stream_matrix_blocks(
    adapter_paths: list[Path],
    *,
    expected_keys: tuple[str, ...],
    expected_shapes: tuple[tuple[int, ...], ...],
    block_size: int,
    dtype: np.dtype,
    n_features: int,
) -> Iterator[tuple[int, np.ndarray]]:
    if not adapter_paths:
        raise ValueError("adapter_paths must be non-empty")
    if block_size <= 0:
        raise ValueError(f"stream_block_size must be positive, got {block_size}")

    with ExitStack() as stack:
        readers = [stack.enter_context(safe_open(path, framework="numpy")) for path in adapter_paths]
        iters = [
            iter_reader_flat_chunks(
                reader,
                adapter_path=path,
                expected_keys=expected_keys,
                expected_shapes=expected_shapes,
                block_size=block_size,
                dtype=dtype,
            )
            for reader, path in zip(readers, adapter_paths)
        ]

        write_offset = 0
        while True:
            chunks = [next(it, None) for it in iters]
            n_done = sum(chunk is None for chunk in chunks)
            if n_done == len(chunks):
                break
            if n_done != 0:
                raise RuntimeError("Adapter chunk streams are misaligned across inputs")

            first = chunks[0]
            if first is None:
                raise RuntimeError("Unexpected None chunk for first iterator")
            chunk_size = int(first.size)
            if chunk_size <= 0:
                continue

            block = np.empty((len(chunks), chunk_size), dtype=np.float64)
            for i, chunk in enumerate(chunks):
                if chunk is None:
                    raise RuntimeError("Unexpected None chunk for non-finished iterator")
                if int(chunk.size) != chunk_size:
                    raise RuntimeError("Chunk-size mismatch across adapters in streamed block")
                block[i, :] = np.asarray(chunk, dtype=np.float64)

            end = write_offset + chunk_size
            if end > n_features:
                raise RuntimeError(
                    "Feature write overflow in streamed blocks: "
                    f"attempted end={end}, n_features={n_features}"
                )
            yield write_offset, block
            write_offset = end

        if write_offset != n_features:
            raise RuntimeError(
                f"Feature write underflow in streamed blocks: wrote {write_offset}, expected {n_features}"
            )
