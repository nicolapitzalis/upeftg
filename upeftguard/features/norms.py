from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import inspect_adapter_schema, stream_matrix_blocks
from ..utilities.manifest import ManifestItem


NORMS_EXTRACTOR_VERSION = "1.0.0"


def extract_norm_features(
    *,
    items: list[ManifestItem],
    block_size: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray | None, list[str], dict[str, Any]]:
    if not items:
        raise ValueError("No adapters provided for norm extraction")

    first_adapter = items[0].adapter_path
    expected_keys, expected_shapes, layers, n_features = inspect_adapter_schema(
        adapter_path=first_adapter,
        expected_keys=None,
        expected_shapes=None,
    )

    adapter_paths = [item.adapter_path for item in items]
    n_samples = len(items)

    l1 = np.zeros(n_samples, dtype=np.float64)
    l2_sq = np.zeros(n_samples, dtype=np.float64)
    linf = np.zeros(n_samples, dtype=np.float64)
    mean_abs_acc = np.zeros(n_samples, dtype=np.float64)

    for _, block in stream_matrix_blocks(
        adapter_paths=adapter_paths,
        expected_keys=expected_keys,
        expected_shapes=expected_shapes,
        block_size=block_size,
        dtype=dtype,
        n_features=n_features,
    ):
        abs_block = np.abs(block)
        l1 += abs_block.sum(axis=1)
        l2_sq += np.square(block).sum(axis=1)
        linf = np.maximum(linf, abs_block.max(axis=1))
        mean_abs_acc += abs_block.sum(axis=1)

    mean_abs = mean_abs_acc / max(1, n_features)
    l2 = np.sqrt(l2_sq)

    features = np.stack([l1, l2, linf, mean_abs], axis=1).astype(np.float32)

    labels_list = [item.label for item in items]
    labels = np.asarray(labels_list, dtype=np.int32) if all(x is not None for x in labels_list) else None
    model_names = [item.model_name for item in items]

    metadata: dict[str, Any] = {
        "extractor": "norms",
        "extractor_version": NORMS_EXTRACTOR_VERSION,
        "n_models": int(n_samples),
        "n_features_raw": int(n_features),
        "layers": layers,
        "feature_names": ["l1_norm", "l2_norm", "linf_norm", "mean_abs"],
        "feature_dim": int(features.shape[1]),
    }
    return features, labels, model_names, metadata
