"""Generic in-memory feature-table contracts and deterministic merging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class FeatureTable:
    source: str
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    feature_names: list[str]
    feature_names_inferred: bool
    metadata: dict[str, Any]


class FeatureTableLike(Protocol):
    source: str
    features: np.ndarray
    labels: np.ndarray | None
    model_names: list[str]
    feature_names: list[str]
    feature_names_inferred: bool


@dataclass(frozen=True)
class ZeroFillMergeResult:
    model_names: list[str]
    feature_names: list[str]
    features: np.ndarray
    labels: np.ndarray | None
    feature_names_inferred: bool
    stats: dict[str, Any]


def _resolved_feature_name_views(
    *,
    base: FeatureTableLike,
    incoming: FeatureTableLike,
) -> tuple[list[str], list[str], bool]:
    used_positional_feature_names = bool(base.feature_names_inferred or incoming.feature_names_inferred)
    if not used_positional_feature_names:
        return list(base.feature_names), list(incoming.feature_names), False

    if len(base.feature_names) != len(incoming.feature_names):
        raise ValueError(
            "Cannot merge feature schemas with inferred positional names when dimensions differ. "
            "Provide metadata with explicit feature_names for both sources."
        )
    positional = [f"feature_{i:05d}" for i in range(len(base.feature_names))]
    return positional, positional, True


def _merge_feature_name_union(
    *,
    base_feature_names: list[str],
    incoming_feature_names: list[str],
) -> list[str]:
    merged_feature_names = list(base_feature_names)
    base_feature_name_set = set(base_feature_names)
    merged_feature_names.extend(name for name in incoming_feature_names if name not in base_feature_name_set)
    return merged_feature_names


def _place_table_values(
    *,
    merged_features: np.ndarray,
    coverage: np.ndarray,
    merged_row_index: dict[str, int],
    merged_col_index: dict[str, int],
    table: FeatureTableLike,
    table_feature_names: list[str],
) -> None:
    row_idx = np.asarray([merged_row_index[name] for name in table.model_names], dtype=np.int64)
    col_idx = np.asarray([merged_col_index[name] for name in table_feature_names], dtype=np.int64)
    table_block = np.asarray(table.features, dtype=np.float32)
    current = merged_features[np.ix_(row_idx, col_idx)]
    seen = coverage[np.ix_(row_idx, col_idx)]

    if np.any(seen):
        conflict = seen & (~np.isclose(current, table_block, rtol=1e-5, atol=1e-6, equal_nan=True))
        if np.any(conflict):
            conflict_idx = np.argwhere(conflict)[0]
            i = int(conflict_idx[0])
            j = int(conflict_idx[1])
            row_name = table.model_names[i]
            feature_name = table_feature_names[j]
            raise ValueError(
                "Conflicting feature values for overlapping row/feature cell: "
                f"model='{row_name}', feature='{feature_name}', "
                f"existing={float(current[i, j])}, incoming={float(table_block[i, j])}"
            )

    merged_features[np.ix_(row_idx, col_idx)] = np.where(seen, current, table_block)
    coverage[np.ix_(row_idx, col_idx)] = True


def _merge_labels_for_dense_rows(
    *,
    base: FeatureTableLike,
    incoming: FeatureTableLike,
    merged_row_index: dict[str, int],
) -> np.ndarray | None:
    labels_unknown = np.iinfo(np.int32).min
    merged_label_values = np.full(len(merged_row_index), labels_unknown, dtype=np.int32)
    merged_label_known = np.zeros(len(merged_row_index), dtype=bool)

    def _place_labels(table: FeatureTableLike) -> None:
        if table.labels is None:
            return
        row_idx = np.asarray([merged_row_index[name] for name in table.model_names], dtype=np.int64)
        vals = np.asarray(table.labels, dtype=np.int32)
        known = merged_label_known[row_idx]
        existing_vals = merged_label_values[row_idx]

        conflict = known & (existing_vals != vals)
        if np.any(conflict):
            i = int(np.argwhere(conflict)[0, 0])
            raise ValueError(
                "Conflicting labels for overlapping model: "
                f"model='{table.model_names[i]}', existing={int(existing_vals[i])}, incoming={int(vals[i])}"
            )

        write_vals = existing_vals.copy()
        write_vals[~known] = vals[~known]
        merged_label_values[row_idx] = write_vals
        merged_label_known[row_idx] = True

    _place_labels(base)
    _place_labels(incoming)
    return merged_label_values if bool(np.all(merged_label_known)) else None


def merge_feature_tables_dense(
    *,
    base: FeatureTable,
    incoming: FeatureTable,
) -> tuple[FeatureTable, dict[str, Any]]:
    """Merge overlapping feature tables only when every output cell is defined."""
    base_feature_names, incoming_feature_names, used_positional_feature_names = _resolved_feature_name_views(
        base=base, incoming=incoming
    )

    base_row_index = unique_index_by_name(
        base.model_names,
        context=base.source,
        entity="model names",
    )
    unique_index_by_name(
        incoming.model_names,
        context=incoming.source,
        entity="model names",
    )
    unique_index_by_name(
        base_feature_names,
        context=f"{base.source} feature schema",
        entity="feature names",
    )
    unique_index_by_name(
        incoming_feature_names,
        context=f"{incoming.source} feature schema",
        entity="feature names",
    )

    merged_model_names = list(base.model_names)
    merged_model_names.extend(name for name in incoming.model_names if name not in base_row_index)
    merged_feature_names = _merge_feature_name_union(
        base_feature_names=base_feature_names,
        incoming_feature_names=incoming_feature_names,
    )
    merged_row_index = unique_index_by_name(
        merged_model_names,
        context="merged output",
        entity="model names",
    )
    merged_col_index = unique_index_by_name(
        merged_feature_names,
        context="merged output",
        entity="feature names",
    )

    merged_features = np.zeros(
        (len(merged_model_names), len(merged_feature_names)),
        dtype=np.float32,
    )
    coverage = np.zeros_like(merged_features, dtype=bool)
    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=base,
        table_feature_names=base_feature_names,
    )
    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=incoming,
        table_feature_names=incoming_feature_names,
    )
    if not bool(np.all(coverage)):
        missing = np.argwhere(~coverage)
        preview = [
            f"{merged_model_names[int(row)]}:{merged_feature_names[int(column)]}"
            for row, column in missing[:5].tolist()
        ]
        raise ValueError(
            "Merged output would contain missing feature cells; this usually means the new run "
            "does not cover all required rows/features. "
            f"Missing example(s): {preview}"
        )

    merged_labels = _merge_labels_for_dense_rows(
        base=base,
        incoming=incoming,
        merged_row_index=merged_row_index,
    )
    overlap_rows = sorted(set(base.model_names) & set(incoming.model_names))
    overlap_features = sorted(set(base_feature_names) & set(incoming_feature_names))
    stats = {
        "base_rows": int(len(base.model_names)),
        "base_feature_dim": int(len(base_feature_names)),
        "incoming_rows": int(len(incoming.model_names)),
        "incoming_feature_dim": int(len(incoming_feature_names)),
        "merged_rows": int(len(merged_model_names)),
        "merged_feature_dim": int(len(merged_feature_names)),
        "rows_added": int(len(merged_model_names) - len(base.model_names)),
        "features_added": int(len(merged_feature_names) - len(base_feature_names)),
        "row_overlap": int(len(overlap_rows)),
        "feature_overlap": int(len(overlap_features)),
        "labels_complete": bool(merged_labels is not None),
        "used_positional_feature_names": used_positional_feature_names,
    }
    return (
        FeatureTable(
            source="merged_output",
            features=merged_features,
            labels=merged_labels,
            model_names=merged_model_names,
            feature_names=merged_feature_names,
            feature_names_inferred=used_positional_feature_names,
            metadata={},
        ),
        stats,
    )


def merge_disjoint_feature_tables_zero_fill(
    *,
    base: FeatureTableLike,
    incoming: FeatureTableLike,
    index_by_name,
    overlap_error_prefix: str,
) -> ZeroFillMergeResult:
    base_feature_names, incoming_feature_names, used_positional_feature_names = _resolved_feature_name_views(
        base=base,
        incoming=incoming,
    )

    base_row_index = index_by_name(base.model_names, context=base.source, entity="model names")
    incoming_row_index = index_by_name(
        incoming.model_names,
        context=incoming.source,
        entity="model names",
    )
    index_by_name(base_feature_names, context=f"{base.source} feature schema", entity="feature names")
    index_by_name(incoming_feature_names, context=f"{incoming.source} feature schema", entity="feature names")

    overlap_rows = sorted(set(base_row_index) & set(incoming_row_index))
    if overlap_rows:
        preview = ", ".join(overlap_rows[:5])
        raise ValueError(f"{overlap_error_prefix}. Examples: {preview}")

    merged_model_names = list(base.model_names)
    merged_model_names.extend(incoming.model_names)
    merged_feature_names = _merge_feature_name_union(
        base_feature_names=base_feature_names,
        incoming_feature_names=incoming_feature_names,
    )

    merged_row_index = index_by_name(merged_model_names, context="merged output", entity="model names")
    merged_col_index = index_by_name(merged_feature_names, context="merged output", entity="feature names")

    merged_features = np.zeros((len(merged_model_names), len(merged_feature_names)), dtype=np.float32)
    coverage = np.zeros_like(merged_features, dtype=bool)

    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=base,
        table_feature_names=base_feature_names,
    )
    _place_table_values(
        merged_features=merged_features,
        coverage=coverage,
        merged_row_index=merged_row_index,
        merged_col_index=merged_col_index,
        table=incoming,
        table_feature_names=incoming_feature_names,
    )

    merged_label_values = np.empty(len(merged_model_names), dtype=np.int32)
    labels_complete = base.labels is not None and incoming.labels is not None
    if labels_complete:
        merged_label_values[: len(base.model_names)] = np.asarray(base.labels, dtype=np.int32)
        merged_label_values[len(base.model_names) :] = np.asarray(incoming.labels, dtype=np.int32)
        merged_labels: np.ndarray | None = merged_label_values
    else:
        merged_labels = None

    overlap_features = sorted(set(base_feature_names) & set(incoming_feature_names))
    zero_filled_cells = int(merged_features.size - int(np.count_nonzero(coverage)))
    stats = {
        "merge_mode": "zero_fill_disjoint_rows",
        "base_rows": int(len(base.model_names)),
        "base_feature_dim": int(len(base_feature_names)),
        "incoming_rows": int(len(incoming.model_names)),
        "incoming_feature_dim": int(len(incoming_feature_names)),
        "merged_rows": int(len(merged_model_names)),
        "merged_feature_dim": int(len(merged_feature_names)),
        "rows_added": int(len(incoming.model_names)),
        "features_added": int(len(merged_feature_names) - len(base_feature_names)),
        "row_overlap": 0,
        "feature_overlap": int(len(overlap_features)),
        "labels_complete": bool(merged_labels is not None),
        "zero_filled_cells": zero_filled_cells,
    }
    return ZeroFillMergeResult(
        model_names=merged_model_names,
        feature_names=merged_feature_names,
        features=merged_features,
        labels=merged_labels,
        feature_names_inferred=used_positional_feature_names,
        stats=stats,
    )


def unique_index_by_name(names: list[str], *, context: str, entity: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for i, name in enumerate(names):
        if name in index:
            duplicates.append(name)
            continue
        index[name] = int(i)
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate {entity} in {context}; cannot align merged data safely. Examples: {preview}")
    return index
