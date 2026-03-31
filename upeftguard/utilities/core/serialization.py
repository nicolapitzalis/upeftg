from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(x) for x in obj]
    if isinstance(obj, tuple):
        return [json_ready(x) for x in obj]
    return obj
