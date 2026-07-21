from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised in environments that install torch
    import torch
except ImportError:  # pragma: no cover - soft dependency
    torch = None


def require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError("Torch-backed supervised models require the optional 'torch' dependency")
    return torch


def set_torch_random_seeds(random_state: int) -> None:
    torch_module = require_torch()
    resolved_seed = int(random_state)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch_module.manual_seed(resolved_seed)
    if hasattr(torch_module, "use_deterministic_algorithms"):
        try:
            torch_module.use_deterministic_algorithms(True)
        except Exception:
            pass


def set_torch_threads(n_jobs: int | None) -> None:
    if n_jobs is None or int(n_jobs) <= 0:
        return
    require_torch().set_num_threads(int(n_jobs))


def load_torch_checkpoint_payload(path: Path) -> dict[str, Any]:
    torch_module = require_torch()
    resolved_path = Path(path).expanduser().resolve()
    try:
        payload = torch_module.load(
            resolved_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - older torch
        payload = torch_module.load(resolved_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Torch checkpoint must contain a dictionary payload: {resolved_path}")
    return payload
