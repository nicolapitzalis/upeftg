from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal environments
    joblib = None

from ..models.common.checkpoints import load_torch_sequence_checkpoint


def save_model(model: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save"):
        model.save(path)
        return
    if joblib is not None:
        joblib.dump(model, path)
        return
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(path: Path) -> Any:
    path = Path(path).expanduser().resolve()
    if path.suffix == ".pt":
        return load_torch_sequence_checkpoint(path)
    if joblib is not None:
        return joblib.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)
