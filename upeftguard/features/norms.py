from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterator

import numpy as np


DEFAULT_ENTRYWISE_DELTA_MODE = "auto"
SUPPORTED_ENTRYWISE_DELTA_MODES = ("auto", "dense", "stream")
_AUTO_DENSE_MEMORY_FRACTION = 0.25
_DENSE_MOMENT_WORKING_SET_BYTES_PER_ENTRY = 24


@dataclass(frozen=True)
class BlockMomentSummary:
    count: int
    mean: float
    variance: float
    l1_norm: float
    l2_norm: float
    linf_norm: float
    mean_abs: float
    kurtosis: float


@dataclass
class _MomentAccumulator:
    count: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_x3: float = 0.0
    sum_x4: float = 0.0
    sum_abs: float = 0.0
    max_abs: float = 0.0

    def update(self, flat: np.ndarray) -> None:
        vals = np.asarray(flat, dtype=np.float64).reshape(-1)
        if vals.size == 0:
            return

        self.count += int(vals.size)
        self.sum_x += float(np.sum(vals, dtype=np.float64))
        sq = np.square(vals, dtype=np.float64)
        self.sum_x2 += float(np.sum(sq, dtype=np.float64))
        self.sum_x3 += float(np.dot(sq, vals))
        self.sum_x4 += float(np.dot(sq, sq))
        abs_vals = np.abs(vals)
        self.sum_abs += float(np.sum(abs_vals, dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(abs_vals)))

    def finalize(self) -> BlockMomentSummary:
        if self.count <= 0:
            return BlockMomentSummary(
                count=0,
                mean=0.0,
                variance=0.0,
                l1_norm=0.0,
                l2_norm=0.0,
                linf_norm=0.0,
                mean_abs=0.0,
                kurtosis=0.0,
            )

        n = float(self.count)
        ex1 = self.sum_x / n
        ex2 = self.sum_x2 / n
        ex3 = self.sum_x3 / n
        ex4 = self.sum_x4 / n

        variance = max(0.0, ex2 - (ex1 * ex1))
        centered_fourth = ex4 - (4.0 * ex1 * ex3) + (6.0 * (ex1 ** 2) * ex2) - (3.0 * (ex1 ** 4))
        if variance <= 1e-18:
            kurtosis = 0.0
        else:
            kurtosis = float(centered_fourth / max(1e-18, variance * variance) - 3.0)

        return BlockMomentSummary(
            count=self.count,
            mean=float(ex1),
            variance=float(variance),
            l1_norm=float(self.sum_abs),
            l2_norm=float(np.sqrt(max(0.0, self.sum_x2))),
            linf_norm=float(self.max_abs),
            mean_abs=float(self.sum_abs / n),
            kurtosis=kurtosis,
        )


def resolve_entrywise_delta_mode(mode: str | None) -> str:
    resolved = DEFAULT_ENTRYWISE_DELTA_MODE if mode is None else str(mode).strip().lower()
    if resolved not in SUPPORTED_ENTRYWISE_DELTA_MODES:
        raise ValueError(
            f"Unknown entrywise_delta_mode '{mode}'. "
            f"Supported: {list(SUPPORTED_ENTRYWISE_DELTA_MODES)}"
        )
    return resolved


def _validate_lora_factors(a: np.ndarray, b: np.ndarray) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected rank-2 arrays for LoRA factors, got A{a.shape}, B{b.shape}")
    if int(a.shape[0]) != int(b.shape[1]):
        raise ValueError(f"LoRA rank mismatch for factors A{a.shape}, B{b.shape}")


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
                break
    except OSError:
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and avail_pages > 0:
            return page_size * avail_pages
    except (AttributeError, OSError, ValueError):
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and total_pages > 0:
            return page_size * total_pages
    except (AttributeError, OSError, ValueError):
        return None
    return None


def _estimate_dense_moment_working_set_bytes(a: np.ndarray, b: np.ndarray) -> int:
    n_entries = int(b.shape[0]) * int(a.shape[1])
    return n_entries * _DENSE_MOMENT_WORKING_SET_BYTES_PER_ENTRY


def _resolve_runtime_entrywise_delta_mode(
    *,
    a: np.ndarray,
    b: np.ndarray,
    requested_mode: str,
    available_memory_bytes: int | None,
) -> str:
    if requested_mode != "auto":
        return requested_mode

    available = _available_memory_bytes() if available_memory_bytes is None else int(available_memory_bytes)
    if available is None or available <= 0:
        return "stream"

    estimated = _estimate_dense_moment_working_set_bytes(a=a, b=b)
    dense_budget = int(float(available) * _AUTO_DENSE_MEMORY_FRACTION)
    if estimated <= 0 or dense_budget <= 0:
        return "stream"
    return "dense" if estimated <= dense_budget else "stream"


def iter_delta_matrix_chunks(
    *,
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
    dtype: np.dtype,
) -> Iterator[np.ndarray]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    _validate_lora_factors(a=a, b=b)

    out_dim = int(b.shape[0])
    in_dim = int(a.shape[1])
    cols_per_chunk = max(1, block_size // max(1, out_dim))

    for col_start in range(0, in_dim, cols_per_chunk):
        col_end = min(col_start + cols_per_chunk, in_dim)
        chunk = np.asarray(b @ a[:, col_start:col_end], dtype=dtype)
        if chunk.size:
            yield chunk


def block_moments_from_factors(
    *,
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
    dtype: np.dtype,
    entrywise_delta_mode: str = DEFAULT_ENTRYWISE_DELTA_MODE,
    available_memory_bytes: int | None = None,
) -> tuple[BlockMomentSummary, str]:
    requested_mode = resolve_entrywise_delta_mode(entrywise_delta_mode)
    _validate_lora_factors(a=a, b=b)
    runtime_mode = _resolve_runtime_entrywise_delta_mode(
        a=a,
        b=b,
        requested_mode=requested_mode,
        available_memory_bytes=available_memory_bytes,
    )
    acc = _MomentAccumulator()
    if runtime_mode == "dense":
        acc.update(np.asarray(b @ a, dtype=dtype))
        return acc.finalize(), runtime_mode

    for chunk in iter_delta_matrix_chunks(
        a=a,
        b=b,
        block_size=block_size,
        dtype=dtype,
    ):
        acc.update(chunk)
    return acc.finalize(), runtime_mode


def summarize_array_moments(values: np.ndarray) -> BlockMomentSummary:
    acc = _MomentAccumulator()
    acc.update(values)
    return acc.finalize()
