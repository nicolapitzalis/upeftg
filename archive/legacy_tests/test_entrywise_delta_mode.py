from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

_NORMS_PATH = Path(__file__).resolve().parents[1] / "upeftguard" / "features" / "norms.py"
_NORMS_SPEC = importlib.util.spec_from_file_location("upeftguard.features.norms", _NORMS_PATH)
if _NORMS_SPEC is None or _NORMS_SPEC.loader is None:
    raise RuntimeError(f"Failed to load norms module from {_NORMS_PATH}")
_NORMS_MODULE = importlib.util.module_from_spec(_NORMS_SPEC)
sys.modules[_NORMS_SPEC.name] = _NORMS_MODULE
_NORMS_SPEC.loader.exec_module(_NORMS_MODULE)

block_moments_from_factors = _NORMS_MODULE.block_moments_from_factors


class EntrywiseDeltaModeTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.a = rng.standard_normal((4, 13)).astype(np.float32)
        self.b = rng.standard_normal((11, 4)).astype(np.float32)

    def test_dense_and_stream_match(self) -> None:
        dense_summary, dense_mode = block_moments_from_factors(
            a=self.a,
            b=self.b,
            block_size=7,
            dtype=np.float32,
            entrywise_delta_mode="dense",
        )
        stream_summary, stream_mode = block_moments_from_factors(
            a=self.a,
            b=self.b,
            block_size=7,
            dtype=np.float32,
            entrywise_delta_mode="stream",
        )

        self.assertEqual(dense_mode, "dense")
        self.assertEqual(stream_mode, "stream")
        self.assertEqual(dense_summary.count, stream_summary.count)
        for field in ("mean", "variance", "l1_norm", "l2_norm", "linf_norm", "mean_abs", "kurtosis"):
            self.assertAlmostEqual(
                getattr(dense_summary, field),
                getattr(stream_summary, field),
                places=6,
                msg=field,
            )

    def test_auto_uses_dense_when_memory_budget_is_large(self) -> None:
        _, runtime_mode = block_moments_from_factors(
            a=self.a,
            b=self.b,
            block_size=7,
            dtype=np.float32,
            entrywise_delta_mode="auto",
            available_memory_bytes=1 << 30,
        )
        self.assertEqual(runtime_mode, "dense")

    def test_auto_falls_back_to_stream_when_memory_budget_is_small(self) -> None:
        _, runtime_mode = block_moments_from_factors(
            a=self.a,
            b=self.b,
            block_size=7,
            dtype=np.float32,
            entrywise_delta_mode="auto",
            available_memory_bytes=128,
        )
        self.assertEqual(runtime_mode, "stream")


if __name__ == "__main__":
    unittest.main()
