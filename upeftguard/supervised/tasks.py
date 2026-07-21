from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import (
    BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
    SUPERVISED_TASK_MODE_BINARY,
    SupervisedTaskSpec,
)


UNKNOWN_ATTACK_CLASS_NAME = "unknown_attack"
SELECTION_METRIC_TASK_DEFAULT = "task_default"
SELECTION_METRIC_ROC_AUC = "roc_auc"
SELECTION_METRIC_BINARY_AUROC = "binary_auroc"
SUPPORTED_SELECTION_METRICS = (
    SELECTION_METRIC_TASK_DEFAULT,
    SELECTION_METRIC_ROC_AUC,
    SELECTION_METRIC_BINARY_AUROC,
)


def default_binary_task_spec() -> SupervisedTaskSpec:
    class_names = ("clean", "backdoored")
    return SupervisedTaskSpec(
        task_mode=SUPERVISED_TASK_MODE_BINARY,
        class_names=class_names,
        class_to_index={name: index for index, name in enumerate(class_names)},
        binary_projection=BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    )


def attack_family_multiclass_task_spec(
    attack_names: Sequence[str] | None,
) -> SupervisedTaskSpec:
    if not attack_names:
        raise ValueError(
            "task_mode=attack_family_multiclass requires --multiclass-attack-names "
            "with at least one positive attack class name"
        )

    normalized_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in attack_names:
        name = str(raw_name).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        normalized_names.append(name)
    if not normalized_names:
        raise ValueError("task_mode=attack_family_multiclass requires at least one non-empty attack class name")

    reserved_names = {"clean", UNKNOWN_ATTACK_CLASS_NAME}
    conflicting_names = [name for name in normalized_names if name.lower() in reserved_names]
    if conflicting_names:
        raise ValueError(
            "attack_family_multiclass class names cannot reuse reserved labels "
            f"{sorted(reserved_names)}; got {conflicting_names}"
        )

    class_names = ("clean", *normalized_names)
    return SupervisedTaskSpec(
        task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
        class_names=class_names,
        class_to_index={name: index for index, name in enumerate(class_names)},
        binary_projection=BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    )


def resolve_supervised_task_spec(
    *,
    task_mode: str | None,
    multiclass_attack_names: Sequence[str] | None,
) -> SupervisedTaskSpec:
    resolved_mode = str(task_mode or SUPERVISED_TASK_MODE_BINARY)
    if resolved_mode == SUPERVISED_TASK_MODE_BINARY:
        return default_binary_task_spec()
    if resolved_mode == SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS:
        return attack_family_multiclass_task_spec(multiclass_attack_names)
    raise ValueError(f"Unsupported supervised task_mode={resolved_mode!r}")


def task_spec_from_payload(payload: Any) -> SupervisedTaskSpec:
    if not isinstance(payload, dict):
        return default_binary_task_spec()

    task_mode = str(payload.get("task_mode") or SUPERVISED_TASK_MODE_BINARY)
    class_names_raw = payload.get("class_names")
    if not isinstance(class_names_raw, list) or not class_names_raw:
        return default_binary_task_spec()
    class_names = tuple(str(value) for value in class_names_raw)

    class_to_index_raw = payload.get("class_to_index")
    if isinstance(class_to_index_raw, dict) and class_to_index_raw:
        class_to_index = {str(key): int(value) for key, value in class_to_index_raw.items()}
    else:
        class_to_index = {name: index for index, name in enumerate(class_names)}

    default_projection = (
        BINARY_PROJECTION_POSITIVE_CLASS_SCORE
        if task_mode == SUPERVISED_TASK_MODE_BINARY
        else BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY
    )
    return SupervisedTaskSpec(
        task_mode=task_mode,
        class_names=class_names,
        class_to_index=class_to_index,
        binary_projection=str(payload.get("binary_projection", default_projection)),
    )


def resolve_selection_metric(
    selection_metric: str | None,
    *,
    task_spec: SupervisedTaskSpec,
) -> str:
    raw_metric = str(selection_metric or SELECTION_METRIC_TASK_DEFAULT).strip().lower()
    if raw_metric in {"", "default", SELECTION_METRIC_TASK_DEFAULT}:
        return str(task_spec.selection_metric_name)
    if raw_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported supervised selection metric {selection_metric!r}; "
            f"expected one of {list(SUPPORTED_SELECTION_METRICS)}"
        )
    if raw_metric == SELECTION_METRIC_ROC_AUC:
        return SELECTION_METRIC_ROC_AUC if task_spec.is_binary else SELECTION_METRIC_BINARY_AUROC
    return raw_metric


def task_spec_from_manifest(manifest: dict[str, Any]) -> SupervisedTaskSpec:
    return task_spec_from_payload(manifest.get("task"))


def labels_from_items(
    items: list[Any],
    *,
    task_spec: SupervisedTaskSpec,
    sample_identities: list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    if task_spec.is_binary:
        raw_labels = [item.label for item in items]
        values = np.asarray(
            [int(label) if label is not None else -1 for label in raw_labels],
            dtype=np.int32,
        )
        known = np.asarray([label is not None for label in raw_labels], dtype=bool)
        return values, known, raw_labels

    if sample_identities is None or len(sample_identities) != len(items):
        raise ValueError("Multiclass supervised label derivation requires aligned attack sample identities")

    raw_labels: list[int | None] = []
    for item, identity in zip(items, sample_identities):
        if item.label is None:
            raw_labels.append(None)
        elif int(item.label) == 0:
            raw_labels.append(int(task_spec.clean_class_index))
        else:
            attack_name = str(identity.attack_name)
            if attack_name not in task_spec.class_to_index:
                raise ValueError(
                    "task_mode=attack_family_multiclass encountered a positive sample outside the configured "
                    f"attack vocabulary: model={item.model_name!r}, attack_name={attack_name!r}, "
                    f"supported={list(task_spec.class_names[1:])}"
                )
            raw_labels.append(int(task_spec.class_to_index[attack_name]))

    values = np.asarray(
        [int(label) if label is not None else -1 for label in raw_labels],
        dtype=np.int32,
    )
    known = np.asarray([label is not None for label in raw_labels], dtype=bool)
    return values, known, raw_labels
