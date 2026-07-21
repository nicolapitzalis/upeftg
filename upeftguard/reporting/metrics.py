from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..utilities.core.manifest import AttackSampleIdentity, infer_dataset_group_name


DEFAULT_DECISION_THRESHOLD = 0.5

_ADAPTER_LABEL_BY_TOKEN = {
    "adalora": "adalora",
    "dora": "dora",
    "qlora": "qlora",
}
_ADAPTER_LABEL_BY_PAIR = {
    ("lora", "plus"): "lora_plus",
    ("lora", "only"): "lora_only",
}


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _decision_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = np.asarray(scores >= float(threshold), dtype=np.int32)
    truth = np.asarray(labels, dtype=np.int32)
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    return {
        "threshold": float(threshold),
        "accuracy": _optional_ratio(tp + tn, int(truth.size)),
        "precision": _optional_ratio(tp, tp + fp),
        "recall": _optional_ratio(tp, tp + fn),
        "confusion_matrix": {
            "labels": ["clean", "backdoored"],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def build_binary_evaluation(
    *,
    labels: Sequence[int | None],
    scores: np.ndarray,
    calibrated_thresholds: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scores_np = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != int(scores_np.size):
        raise ValueError("Binary evaluation labels and scores must have the same length")

    known_mask = np.asarray([label is not None for label in labels], dtype=bool)
    known_labels = np.asarray([int(label) for label in labels if label is not None], dtype=np.int32)
    known_scores = np.asarray(scores_np[known_mask], dtype=np.float64)
    negative_count = int(np.sum(known_labels == 0))
    positive_count = int(np.sum(known_labels == 1))

    auroc: float | None = None
    auprc: float | None = None
    if negative_count > 0 and positive_count > 0:
        auroc = float(roc_auc_score(known_labels, known_scores))
        auprc = float(average_precision_score(known_labels, known_scores))

    decisions: dict[str, Any] = {"default": _decision_metrics(known_labels, known_scores, DEFAULT_DECISION_THRESHOLD)}
    calibrated_rows: list[dict[str, Any]] = []
    for raw in calibrated_thresholds or ():
        if not isinstance(raw, dict) or raw.get("threshold") is None:
            continue
        row = _decision_metrics(known_labels, known_scores, float(raw["threshold"]))
        if raw.get("accepted_fpr") is not None:
            row["accepted_fpr"] = float(raw["accepted_fpr"])
        if raw.get("selection_method") is not None:
            row["selection_method"] = str(raw["selection_method"])
        if raw.get("source_partition") is not None:
            row["source_partition"] = str(raw["source_partition"])
        calibrated_rows.append(row)
    if calibrated_rows:
        decisions["calibrated"] = calibrated_rows

    return {
        "samples": {
            "total": int(scores_np.size),
            "known": int(known_labels.size),
            "clean": negative_count,
            "backdoored": positive_count,
            "unknown": int(scores_np.size - known_labels.size),
        },
        "auroc": auroc,
        "auprc": auprc,
        "decisions": decisions,
    }


def _adapter_group(identity: AttackSampleIdentity) -> str:
    subset_tokens = [token.lower() for token in str(identity.subset_name or "unknown").split("_") if token]
    model_tokens = [token.lower() for token in str(identity.model_family or "").split("_") if token]
    remainder = (
        subset_tokens[len(model_tokens) :]
        if model_tokens and subset_tokens[: len(model_tokens)] == model_tokens
        else subset_tokens
    )
    best_match: tuple[int, str] | None = None
    for index, token in enumerate(remainder):
        if token in _ADAPTER_LABEL_BY_TOKEN:
            best_match = (index, _ADAPTER_LABEL_BY_TOKEN[token])
    for index in range(max(0, len(remainder) - 1)):
        pair = (remainder[index], remainder[index + 1])
        if pair in _ADAPTER_LABEL_BY_PAIR:
            best_match = (index + 1, _ADAPTER_LABEL_BY_PAIR[pair])
    return best_match[1] if best_match is not None else "lora"


def _evaluate_regular_groups(
    *,
    identities: Sequence[AttackSampleIdentity],
    labels: Sequence[int | None],
    scores: np.ndarray,
    key_fn: Callable[[AttackSampleIdentity], str],
    calibrated_thresholds: Sequence[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, identity in enumerate(identities):
        indices[str(key_fn(identity) or "unknown")].append(index)
    if len(indices) <= 1:
        return None
    return {
        name: build_binary_evaluation(
            labels=[labels[index] for index in rows],
            scores=np.asarray(scores[rows], dtype=np.float64),
            calibrated_thresholds=calibrated_thresholds,
        )
        for name, rows in sorted(indices.items())
    }


def _evaluate_attack_groups(
    *,
    identities: Sequence[AttackSampleIdentity],
    labels: Sequence[int | None],
    scores: np.ndarray,
    calibrated_thresholds: Sequence[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    clean_indices = [index for index, label in enumerate(labels) if label == 0]
    positive_indices: dict[str, list[int]] = defaultdict(list)
    unknown_indices: dict[str, list[int]] = defaultdict(list)
    for index, (identity, label) in enumerate(zip(identities, labels)):
        attack = str(identity.attack_name or "unknown")
        if label == 1:
            positive_indices[attack].append(index)
        elif label is None:
            unknown_indices[attack].append(index)
    if len(positive_indices) <= 1:
        return None
    result: dict[str, Any] = {}
    for attack in sorted(positive_indices):
        rows = sorted(set(clean_indices + positive_indices[attack] + unknown_indices.get(attack, [])))
        result[attack] = build_binary_evaluation(
            labels=[labels[index] for index in rows],
            scores=np.asarray(scores[rows], dtype=np.float64),
            calibrated_thresholds=calibrated_thresholds,
        )
    return result


def build_grouped_evaluations(
    *,
    identities: Sequence[AttackSampleIdentity],
    labels: Sequence[int | None],
    scores: np.ndarray,
    calibrated_thresholds: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scores_np = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(identities) != len(labels) or len(labels) != int(scores_np.size):
        raise ValueError("Grouped evaluation identities, labels, and scores must have the same length")
    candidates = {
        "architecture": _evaluate_regular_groups(
            identities=identities,
            labels=labels,
            scores=scores_np,
            key_fn=lambda identity: str(identity.model_family or "unknown"),
            calibrated_thresholds=calibrated_thresholds,
        ),
        "dataset": _evaluate_regular_groups(
            identities=identities,
            labels=labels,
            scores=scores_np,
            key_fn=infer_dataset_group_name,
            calibrated_thresholds=calibrated_thresholds,
        ),
        "adapter": _evaluate_regular_groups(
            identities=identities,
            labels=labels,
            scores=scores_np,
            key_fn=_adapter_group,
            calibrated_thresholds=calibrated_thresholds,
        ),
        "attack": _evaluate_attack_groups(
            identities=identities,
            labels=labels,
            scores=scores_np,
            calibrated_thresholds=calibrated_thresholds,
        ),
    }
    return {name: value for name, value in candidates.items() if value is not None}
