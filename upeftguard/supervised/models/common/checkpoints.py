from __future__ import annotations

from pathlib import Path
from typing import Any


def load_torch_sequence_checkpoint(path: Path) -> Any:
    from ..cnn.checkpoint import load_cnn_checkpoint
    from ..registry import CNN_1D_DANN_MODEL_NAME, CNN_1D_MODEL_NAME, TRANSFORMER_MODEL_NAME
    from .torch_runtime import load_torch_checkpoint_payload

    payload = load_torch_checkpoint_payload(path)
    backend = str(payload.get("backend") or CNN_1D_MODEL_NAME)
    if backend in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME}:
        return load_cnn_checkpoint(path)
    if backend == TRANSFORMER_MODEL_NAME:
        from ..transformer.checkpoint import load_transformer_checkpoint

        return load_transformer_checkpoint(path)
    raise ValueError(f"Unsupported torch sequence checkpoint backend={backend!r}")
