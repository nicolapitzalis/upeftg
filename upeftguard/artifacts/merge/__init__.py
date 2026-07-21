"""Feature-artifact merge operations."""

from .files import finalize_schema_group_merge, merge_feature_files
from .shards import merge_spectral_shards

__all__ = [
    "finalize_schema_group_merge",
    "merge_feature_files",
    "merge_spectral_shards",
]
