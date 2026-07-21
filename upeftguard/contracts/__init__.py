"""Cross-package contracts shared by feature, artifact, and model layers."""

from .spectral import feature_block_name, layer_identifier_for_block_name, tensor_shape

__all__ = [
    "feature_block_name",
    "layer_identifier_for_block_name",
    "tensor_shape",
]
