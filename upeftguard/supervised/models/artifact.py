from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

from .common.checkpoints import load_torch_sequence_checkpoint
from .common.torch_runtime import load_torch_checkpoint_payload, require_torch


INFERENCE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class LoadedSupervisedArtifact:
    model: Any
    inference_contract: dict[str, Any]


def attach_inference_contract(path: Path, contract: dict[str, Any]) -> None:
    resolved = Path(path).expanduser().resolve()
    payload_contract = {
        "schema_version": INFERENCE_CONTRACT_VERSION,
        **dict(contract),
    }
    if resolved.suffix == ".pt":
        payload = load_torch_checkpoint_payload(resolved)
        payload["inference_contract"] = payload_contract
        require_torch().save(payload, resolved)
        return

    if joblib is not None:
        model = joblib.load(resolved)
        joblib.dump(
            {
                "upeftguard_model": model,
                "inference_contract": payload_contract,
            },
            resolved,
        )
        return
    with resolved.open("rb") as handle:
        model = pickle.load(handle)
    with resolved.open("wb") as handle:
        pickle.dump(
            {
                "upeftguard_model": model,
                "inference_contract": payload_contract,
            },
            handle,
        )


def load_supervised_artifact(path: Path) -> LoadedSupervisedArtifact:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    if resolved.suffix == ".pt":
        payload = load_torch_checkpoint_payload(resolved)
        contract = payload.get("inference_contract")
        if not isinstance(contract, dict):
            raise ValueError(f"Checkpoint is missing inference_contract: {resolved}")
        return LoadedSupervisedArtifact(
            model=load_torch_sequence_checkpoint(resolved),
            inference_contract=dict(contract),
        )

    if joblib is not None:
        payload = joblib.load(resolved)
    else:
        with resolved.open("rb") as handle:
            payload = pickle.load(handle)
    if not isinstance(payload, dict) or "upeftguard_model" not in payload:
        raise ValueError(f"Joblib artifact is missing the UPEFTGuard model wrapper: {resolved}")
    contract = payload.get("inference_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Joblib artifact is missing inference_contract: {resolved}")
    return LoadedSupervisedArtifact(
        model=payload["upeftguard_model"],
        inference_contract=dict(contract),
    )
