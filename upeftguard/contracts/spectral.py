"""Canonical spectral naming and tensor-introspection contracts."""

from __future__ import annotations

from typing import Any


def feature_block_name(feature_name: str) -> str:
    block_name, separator, _ = str(feature_name).rpartition(".")
    if not separator or not block_name:
        raise ValueError(f"Invalid spectral feature name: {feature_name}")
    return block_name


def layer_identifier_for_block_name(block_name: str) -> str:
    parts = [part for part in str(block_name).split(".") if part]
    if len(parts) <= 2:
        return str(block_name)
    prefix = ".".join(parts[:-2]).strip()
    return prefix or str(block_name)


def tensor_shape(reader: Any, key: str) -> tuple[int, ...]:
    if hasattr(reader, "get_slice"):
        return tuple(int(value) for value in reader.get_slice(key).get_shape())
    return tuple(int(value) for value in reader.get_tensor(key).shape)
