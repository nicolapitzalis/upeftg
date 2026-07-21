"""Feature-name schema selection shared by aggregation and subset export."""

from __future__ import annotations

from ..features.spectral import (
    feature_group_for_spectral_feature_name,
    resolve_spectral_features,
)


def normalize_requested_features(
    features: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if not features:
        return None
    cleaned = [str(value).strip().lower() for value in features if str(value).strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1 and cleaned[0] == "all":
        return None
    if "all" in cleaned:
        raise ValueError(
            "--features/--columns must either be omitted, set to 'all', or list supported spectral feature groups"
        )
    return resolve_spectral_features(cleaned)


def feature_group_for_feature_name(feature_name: str) -> str | None:
    return feature_group_for_spectral_feature_name(feature_name)


def resolve_output_feature_names(
    *,
    available_feature_names: list[str],
    requested_features: list[str] | None,
) -> list[str]:
    if requested_features is None:
        return list(available_feature_names)

    requested_feature_set = set(requested_features)
    selected_feature_names = [
        name for name in available_feature_names if feature_group_for_feature_name(name) in requested_feature_set
    ]
    available_feature_groups = {
        group
        for group in (feature_group_for_feature_name(name) for name in available_feature_names)
        if group is not None
    }
    missing_features = [name for name in requested_features if name not in available_feature_groups]
    if missing_features:
        preview = ", ".join(missing_features[:5])
        raise ValueError(
            f"Requested --features/--columns are not available for the selected provenance subset. Examples: {preview}"
        )
    return selected_feature_names
