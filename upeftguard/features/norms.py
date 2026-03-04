from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


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

        abs_vals = np.abs(vals)
        self.count += int(vals.size)
        self.sum_x += float(np.sum(vals, dtype=np.float64))
        self.sum_x2 += float(np.sum(np.square(vals), dtype=np.float64))
        self.sum_x3 += float(np.sum(np.power(vals, 3), dtype=np.float64))
        self.sum_x4 += float(np.sum(np.power(vals, 4), dtype=np.float64))
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


def iter_delta_matrix_chunks(
    *,
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
    dtype: np.dtype,
) -> Iterator[np.ndarray]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected rank-2 arrays for LoRA factors, got A{a.shape}, B{b.shape}")
    if int(a.shape[0]) != int(b.shape[1]):
        raise ValueError(f"LoRA rank mismatch for factors A{a.shape}, B{b.shape}")

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
) -> BlockMomentSummary:
    acc = _MomentAccumulator()
    for chunk in iter_delta_matrix_chunks(
        a=a,
        b=b,
        block_size=block_size,
        dtype=dtype,
    ):
        acc.update(chunk)
    return acc.finalize()
