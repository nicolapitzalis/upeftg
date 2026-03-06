from __future__ import annotations

import getpass
import os
from pathlib import Path

PROJECT_STORAGE_ENV = "UPEFTGUARD_STORAGE_ROOT"
DATASET_ROOT_ENV = "UPEFTGUARD_DATA_ROOT"
PROJECT_STORAGE_NAME = "unsupervised-peftguard"


def _current_user() -> str:
    return os.getenv("USER") or getpass.getuser()


def default_storage_root() -> Path:
    override = os.getenv(PROJECT_STORAGE_ENV)
    if override:
        return Path(override).expanduser()
    return Path("/models") / _current_user() / PROJECT_STORAGE_NAME


def default_dataset_root() -> Path:
    override = os.getenv(DATASET_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return default_storage_root() / "data"


def dataset_root_help() -> str:
    return (
        "Dataset root "
        f"(default: {default_dataset_root()}; override with {DATASET_ROOT_ENV} "
        f"or {PROJECT_STORAGE_ENV})."
    )
