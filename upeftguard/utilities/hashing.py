from __future__ import annotations

import hashlib
import json
from typing import Any


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(payload: Any) -> str:
    data = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_dataset_signature(model_names: list[str], extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "model_names": list(model_names),
    }
    if extra:
        payload["extra"] = extra
    return hash_payload(payload)


def compute_feature_cache_key(
    *,
    dataset_signature: str,
    extractor_name: str,
    extractor_params: dict[str, Any],
    extractor_version: str,
    dtype: str,
) -> str:
    payload = {
        "dataset_signature": dataset_signature,
        "extractor_name": extractor_name,
        "extractor_params": extractor_params,
        "extractor_version": extractor_version,
        "dtype": dtype,
    }
    return hash_payload(payload)
