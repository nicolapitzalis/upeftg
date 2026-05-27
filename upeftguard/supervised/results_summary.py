from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .score_reporting import compute_offline_metrics, summarize_scores
from ..utilities.core.manifest import (
    AttackSampleIdentity,
    parse_joint_manifest_json_by_model_name,
    parse_manifest_entries_by_model_name,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
    infer_attack_sample_identities,
)
from ..utilities.core.serialization import json_ready


SCRIPT_VERSION = "1.0.0"
DEFAULT_SUPERVISED_RUNS_ROOT = Path("runs") / "supervised"
RESULTS_SUMMARY_MD_FILENAME = "results_summary.md"
LEGACY_ARCHITECTURE_ANALYSIS_JSON_FILENAME = "architecture_analysis.json"
LEGACY_ARCHITECTURE_ANALYSIS_MD_FILENAME = "architecture_analysis.md"
_SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS = "attack_family_multiclass"
_DATASET_GROUP_ATTACK_TOKENS = frozenset({"ripple", "syntactic", "insertsent", "stybkd", "stykbd"})
_RANKISH_TOKEN_RE = re.compile(r"^rank\d", re.IGNORECASE)
_DATASET_VARIANT_PREFIX_TOKENS = frozenset({"adalora", "dora", "qlora", "xl", "xxl", "large", "medium", "small", "mini", "tiny"})
_DATASET_VARIANT_PREFIX_PAIRS = frozenset({("lora", "plus"), ("lora", "only")})
_ADAPTER_LABEL_BY_TOKEN = {
    "adalora": "adalora",
    "dora": "dora",
    "qlora": "qlora",
}
_ADAPTER_LABEL_BY_PAIR = {
    ("lora", "plus"): "lora+",
    ("lora", "only"): "lora-only",
}


@dataclass(frozen=True)
class ScorePartition:
    name: str
    model_names: list[str]
    labels: list[int | None]
    scores: np.ndarray
    sample_identities: list[AttackSampleIdentity]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return path


def _looks_like_explicit_path(path_spec: str | Path) -> bool:
    candidate = Path(path_spec).expanduser()
    return candidate.is_absolute() or len(candidate.parts) > 1


def resolve_supervised_run_dir(
    run_spec: str | Path,
    *,
    runs_root: Path = DEFAULT_SUPERVISED_RUNS_ROOT,
) -> Path:
    candidate = Path(run_spec).expanduser()
    if _looks_like_explicit_path(candidate):
        resolved = candidate if candidate.is_absolute() else (Path.cwd().resolve() / candidate)
        return resolved.resolve()
    return (Path.cwd().resolve() / runs_root / candidate).resolve()


def _resolve_manifest_path_for_reporting(manifest: dict[str, Any]) -> Path:
    raw_value = str(manifest["manifest_json"])
    resolved = resolve_manifest_path(raw_value)
    if resolved.exists():
        return resolved

    raw_path = Path(raw_value).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path.resolve())
    else:
        run_dir = Path(str(manifest["run_dir"])).expanduser().resolve()
        candidates.append((Path.cwd().resolve() / raw_path).resolve())
        if len(run_dir.parents) >= 3:
            candidates.append((run_dir.parents[2] / raw_path).resolve())
        candidates.append(raw_path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_manifest_items_for_tuning_manifest(manifest: dict[str, Any]) -> list[Any]:
    manifest_path = _resolve_manifest_path_for_reporting(manifest)
    feature_loading_mode = str(manifest.get("data", {}).get("feature_loading_mode", "materialized"))
    mode = str(manifest["mode"])

    if feature_loading_mode == "external_source":
        if mode == "joint":
            train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_path)
            return train_items + infer_items
        return parse_single_manifest_json_by_model_name(
            manifest_path=manifest_path,
            section_key="path",
        )

    raise ValueError(
        "This backfill tool currently supports supervised runs that were loaded from an external feature source"
    )


def _collect_scored_model_names(reports_dir: Path) -> list[str]:
    model_names: list[str] = []
    seen: set[str] = set()
    for filename in ("train_scores.csv", "calibration_scores.csv", "inference_scores.csv"):
        csv_path = reports_dir / filename
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                model_name = str(row.get("model_name", "")).strip()
                if not model_name or model_name in seen:
                    continue
                seen.add(model_name)
                model_names.append(model_name)
    return model_names


def _infer_sample_identities_for_run(
    *,
    reports_dir: Path,
    tuning_manifest: dict[str, Any] | None,
) -> list[AttackSampleIdentity]:
    manifest_error: Exception | None = None
    if isinstance(tuning_manifest, dict):
        try:
            manifest_items = load_manifest_items_for_tuning_manifest(tuning_manifest)
        except Exception as exc:  # pragma: no cover - exercised by integration paths
            manifest_error = exc
        else:
            return infer_attack_sample_identities(manifest_items)

    scored_model_names = _collect_scored_model_names(reports_dir)
    if scored_model_names:
        manifest_items = parse_manifest_entries_by_model_name(
            scored_model_names,
            manifest_name=str(reports_dir / "scores"),
        )
        return infer_attack_sample_identities(manifest_items)

    if manifest_error is not None:
        raise manifest_error
    raise FileNotFoundError(
        f"Could not infer sample identities for {reports_dir}. No compatible manifest data or score CSVs were found."
    )


def _score_percentile_ranks(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.float64)
    ranks = np.argsort(np.argsort(scores))
    return np.asarray(ranks / max(1, scores.size - 1), dtype=np.float64)


def _coerce_label(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text or text.lower() == "none":
        return None
    value = int(float(text))
    return None if value < 0 else value


def _load_score_partition(
    *,
    partition_name: str,
    csv_path: Path,
    identity_by_name: dict[str, AttackSampleIdentity],
) -> ScorePartition:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Score CSV is empty: {csv_path}")

    model_names = [str(row["model_name"]) for row in rows]
    labels = [_coerce_label(row.get("label")) for row in rows]
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)

    missing_names = [name for name in model_names if name not in identity_by_name]
    if missing_names:
        preview = ", ".join(missing_names[:5])
        raise KeyError(
            f"Could not resolve {len(missing_names)} scored model name(s) in the manifest identities. "
            f"Examples: {preview}"
        )

    sample_identities = [identity_by_name[name] for name in model_names]
    return ScorePartition(
        name=partition_name,
        model_names=model_names,
        labels=labels,
        scores=scores,
        sample_identities=sample_identities,
    )


def _extract_threshold_specs_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("fit_assessment", {}).get("threshold_evaluation", [])
    if not isinstance(rows, list):
        return []
    specs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or "threshold" not in row:
            continue
        spec = {"threshold": float(row["threshold"])}
        for key in ("percentile_from_train", "percentile_from_inference"):
            if key in row:
                spec[key] = float(row[key])
        specs.append(spec)
    return specs


def _build_selected_threshold_specs(
    selected_threshold_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(selected_threshold_summary, dict):
        return []

    selections = selected_threshold_summary.get("selections", [])
    if not isinstance(selections, list):
        return []

    selection_method = selected_threshold_summary.get("method")
    source_partition = selected_threshold_summary.get("source_partition")
    specs: list[dict[str, Any]] = []
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        if "threshold" not in selection or "accepted_fpr" not in selection:
            continue
        spec: dict[str, Any] = {
            "accepted_fpr": float(selection["accepted_fpr"]),
            "threshold": float(selection["threshold"]),
        }
        if selection_method is not None:
            spec["selection_method"] = str(selection_method)
        if source_partition is not None:
            spec["source_partition"] = str(source_partition)
        specs.append(spec)
    return specs


def _evaluate_group_threshold_rows(
    *,
    group_scores: np.ndarray,
    known_mask: np.ndarray,
    known_labels: np.ndarray | None,
    threshold_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threshold_rows: list[dict[str, Any]] = []
    for spec in threshold_specs:
        threshold = float(spec["threshold"])
        flagged = np.asarray(group_scores >= threshold, dtype=bool)
        n_flagged = int(np.sum(flagged))
        row: dict[str, Any] = {key: value for key, value in spec.items() if key != "threshold"}
        row.update(
            {
                "threshold": threshold,
                "n_flagged": n_flagged,
                "fraction_flagged": float(n_flagged / max(1, group_scores.size)),
            }
        )

        if known_labels is not None:
            known_flagged = flagged[known_mask]
            positives = int(np.sum(known_labels == 1))
            negatives = int(np.sum(known_labels == 0))
            tp = int(np.sum((known_labels == 1) & known_flagged))
            fp = int(np.sum((known_labels == 0) & known_flagged))
            if n_flagged > 0:
                row["precision"] = float(tp / n_flagged)
            if positives > 0:
                row["recall"] = float(tp / positives)
            if negatives > 0:
                row["false_positive_rate"] = float(fp / negatives)

        threshold_rows.append(row)
    return threshold_rows


def _summarize_group_slice(
    *,
    group_scores: np.ndarray,
    group_ranks: np.ndarray,
    group_labels: Sequence[int | None],
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None,
    source_subsets: list[str],
) -> dict[str, Any]:
    clean_count = sum(1 for label in group_labels if label == 0)
    backdoored_count = sum(1 for label in group_labels if label == 1)
    unknown_count = sum(1 for label in group_labels if label is None)

    known_mask = np.asarray([label is not None for label in group_labels], dtype=bool)
    known_labels: np.ndarray | None = None
    if bool(np.any(known_mask)):
        known_labels = np.asarray(
            [int(group_labels[i]) for i in range(len(group_labels)) if known_mask[i]],
            dtype=np.int32,
        )
    known_scores = np.asarray(group_scores[known_mask], dtype=np.float64)

    summary = {
        "n_samples": int(group_scores.size),
        "source_subsets": list(source_subsets),
        "label_counts": {
            "clean": int(clean_count),
            "backdoored": int(backdoored_count),
            "unknown": int(unknown_count),
        },
        "score_summary": summarize_scores(np.asarray(group_scores, dtype=np.float64)),
        "score_percentile_rank_summary": summarize_scores(np.asarray(group_ranks, dtype=np.float64)),
        "offline_metrics": compute_offline_metrics(known_labels, known_scores),
        "threshold_evaluation": _evaluate_group_threshold_rows(
            group_scores=np.asarray(group_scores, dtype=np.float64),
            known_mask=known_mask,
            known_labels=known_labels,
            threshold_specs=threshold_specs,
        ),
    }
    if selected_threshold_specs:
        summary["selected_threshold_evaluation"] = _evaluate_group_threshold_rows(
            group_scores=np.asarray(group_scores, dtype=np.float64),
            known_mask=known_mask,
            known_labels=known_labels,
            threshold_specs=selected_threshold_specs,
        )
    return summary


def summarize_architecture_groups(
    *,
    sample_identities: list[AttackSampleIdentity],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(sample_identities) != len(labels) or len(sample_identities) != int(scores.size):
        raise ValueError("Architecture analysis inputs must have the same length")

    score_ranks = _score_percentile_ranks(np.asarray(scores, dtype=np.float64))
    indices_by_architecture: dict[str, list[int]] = {}
    subset_names_by_architecture: dict[str, set[str]] = {}
    for idx, identity in enumerate(sample_identities):
        architecture = str(identity.model_family or "unknown")
        indices_by_architecture.setdefault(architecture, []).append(int(idx))
        subset_names_by_architecture.setdefault(architecture, set()).add(str(identity.subset_name))

    architectures: dict[str, Any] = {}
    for architecture in sorted(indices_by_architecture):
        idx = np.asarray(indices_by_architecture[architecture], dtype=np.int64)
        architectures[architecture] = _summarize_group_slice(
            group_scores=np.asarray(scores[idx], dtype=np.float64),
            group_ranks=np.asarray(score_ranks[idx], dtype=np.float64),
            group_labels=[labels[int(i)] for i in idx.tolist()],
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
            source_subsets=sorted(subset_names_by_architecture.get(architecture, set())),
        )

    return {
        "grouping_rule": (
            "within-architecture evaluation by model_family. Each architecture summary uses only samples whose "
            "manifest-derived model_family matches that architecture, so clean/backdoored comparisons stay "
            "architecture-local rather than sharing a global clean pool."
        ),
        "clean_pool_mode": "within_architecture",
        "n_architectures": int(len(architectures)),
        "architectures": architectures,
    }


def summarize_dataset_groups(
    *,
    sample_identities: list[AttackSampleIdentity],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(sample_identities) != len(labels) or len(sample_identities) != int(scores.size):
        raise ValueError("Dataset analysis inputs must have the same length")

    score_ranks = _score_percentile_ranks(np.asarray(scores, dtype=np.float64))
    indices_by_dataset: dict[str, list[int]] = {}
    subset_names_by_dataset: dict[str, set[str]] = {}
    for idx, identity in enumerate(sample_identities):
        dataset_name = _dataset_group_name(identity)
        indices_by_dataset.setdefault(dataset_name, []).append(int(idx))
        subset_names_by_dataset.setdefault(dataset_name, set()).add(str(identity.subset_name or "unknown"))

    datasets: dict[str, Any] = {}
    for dataset_name in sorted(indices_by_dataset):
        idx = np.asarray(indices_by_dataset[dataset_name], dtype=np.int64)
        datasets[dataset_name] = _summarize_group_slice(
            group_scores=np.asarray(scores[idx], dtype=np.float64),
            group_ranks=np.asarray(score_ranks[idx], dtype=np.float64),
            group_labels=[labels[int(i)] for i in idx.tolist()],
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
            source_subsets=sorted(subset_names_by_dataset.get(dataset_name, set())),
        )

    return {
        "grouping_rule": (
            "within-dataset evaluation by an inferred dataset key parsed from subset_name after removing the "
            "model-family prefix and truncating before attack or rank tokens. Each dataset summary keeps only the "
            "clean/backdoored/unknown samples assigned to that dataset key."
        ),
        "clean_pool_mode": "within_dataset",
        "n_datasets": int(len(datasets)),
        "datasets": datasets,
    }


def summarize_adapter_groups(
    *,
    sample_identities: list[AttackSampleIdentity],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(sample_identities) != len(labels) or len(sample_identities) != int(scores.size):
        raise ValueError("Adapter analysis inputs must have the same length")

    score_ranks = _score_percentile_ranks(np.asarray(scores, dtype=np.float64))
    indices_by_adapter: dict[str, list[int]] = {}
    subset_names_by_adapter: dict[str, set[str]] = {}
    for idx, identity in enumerate(sample_identities):
        adapter_name = _adapter_group_name(identity)
        indices_by_adapter.setdefault(adapter_name, []).append(int(idx))
        subset_names_by_adapter.setdefault(adapter_name, set()).add(str(identity.subset_name or "unknown"))

    adapters: dict[str, Any] = {}
    for adapter_name in sorted(indices_by_adapter):
        idx = np.asarray(indices_by_adapter[adapter_name], dtype=np.int64)
        adapters[adapter_name] = _summarize_group_slice(
            group_scores=np.asarray(scores[idx], dtype=np.float64),
            group_ranks=np.asarray(score_ranks[idx], dtype=np.float64),
            group_labels=[labels[int(i)] for i in idx.tolist()],
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
            source_subsets=sorted(subset_names_by_adapter.get(adapter_name, set())),
        )

    return {
        "grouping_rule": (
            "within-adapter evaluation by an inferred PEFT adapter family parsed from subset_name. Explicit variants "
            "such as adalora, dora, qlora, lora+, and lora-only are kept separate; subsets without an explicit "
            "adapter token default to lora."
        ),
        "clean_pool_mode": "within_adapter",
        "n_adapters": int(len(adapters)),
        "adapters": adapters,
    }


def summarize_attack_groups(
    *,
    sample_identities: list[AttackSampleIdentity],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(sample_identities) != len(labels) or len(sample_identities) != int(scores.size):
        raise ValueError("Attack analysis inputs must have the same length")

    score_ranks = _score_percentile_ranks(np.asarray(scores, dtype=np.float64))
    all_indices_by_attack: dict[str, list[int]] = {}
    positive_by_attack: dict[str, list[int]] = {}
    unknown_by_attack: dict[str, list[int]] = {}
    attack_subset_names: dict[str, set[str]] = {}
    clean_indices: list[int] = []
    clean_subset_names: set[str] = set()

    for idx, (identity, label) in enumerate(zip(sample_identities, labels)):
        attack_name = str(identity.attack_name or "unknown")
        subset_name = str(identity.subset_name or "unknown")
        all_indices_by_attack.setdefault(attack_name, []).append(int(idx))
        attack_subset_names.setdefault(attack_name, set()).add(subset_name)
        if label == 1:
            positive_by_attack.setdefault(attack_name, []).append(int(idx))
        elif label == 0:
            clean_indices.append(int(idx))
            clean_subset_names.add(subset_name)
        else:
            unknown_by_attack.setdefault(attack_name, []).append(int(idx))

    grouped_indices: dict[str, list[int]] = {}
    if positive_by_attack:
        for attack_name in sorted(positive_by_attack):
            combined = (
                list(positive_by_attack[attack_name])
                + list(clean_indices)
                + list(unknown_by_attack.get(attack_name, []))
            )
            grouped_indices[attack_name] = sorted(set(int(i) for i in combined))
    else:
        grouped_indices = {
            attack_name: sorted(int(i) for i in idx_list)
            for attack_name, idx_list in all_indices_by_attack.items()
        }

    attacks: dict[str, Any] = {}
    for attack_name in sorted(grouped_indices):
        idx = np.asarray(grouped_indices[attack_name], dtype=np.int64)
        attacks[attack_name] = _summarize_group_slice(
            group_scores=np.asarray(scores[idx], dtype=np.float64),
            group_ranks=np.asarray(score_ranks[idx], dtype=np.float64),
            group_labels=[labels[int(i)] for i in idx.tolist()],
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
            source_subsets=sorted(attack_subset_names.get(attack_name, set())),
        )

    return {
        "grouping_rule": (
            "one-vs-clean per attack using a single shared clean pool built from every label0 sample; "
            "known PADBench attacks are canonicalized by name (RIPPLE, syntactic, insertsent, stybkd); "
            "all other folders contribute label1 samples under the folder-derived attack name after "
            "removing model/config tokens"
        ),
        "clean_pool": {
            "n_samples": int(len(clean_indices)),
            "source_subsets": sorted(str(x) for x in clean_subset_names),
        },
        "n_attacks": int(len(attacks)),
        "attacks": attacks,
    }


def _dataset_group_name(identity: AttackSampleIdentity) -> str:
    subset_tokens = [token for token in str(identity.subset_name or "unknown").split("_") if token]
    model_tokens = [token for token in str(identity.model_family or "").split("_") if token]
    remainder = subset_tokens
    if model_tokens and subset_tokens[: len(model_tokens)] == model_tokens:
        remainder = subset_tokens[len(model_tokens) :]

    while remainder:
        lowered = [token.lower() for token in remainder]
        if len(lowered) >= 2 and tuple(lowered[:2]) in _DATASET_VARIANT_PREFIX_PAIRS:
            remainder = remainder[2:]
            continue
        if lowered[0] in _DATASET_VARIANT_PREFIX_TOKENS:
            remainder = remainder[1:]
            continue
        break

    dataset_tokens: list[str] = []
    for token in remainder:
        token_lower = token.lower()
        if _RANKISH_TOKEN_RE.match(token_lower):
            break
        if token_lower in _DATASET_GROUP_ATTACK_TOKENS:
            break
        dataset_tokens.append(token)

    if dataset_tokens:
        return "_".join(dataset_tokens)
    if remainder:
        filtered = [token for token in remainder if not _RANKISH_TOKEN_RE.match(token.lower())]
        if filtered:
            return "_".join(filtered)
    return str(identity.subset_name or "unknown")


def _adapter_group_name(identity: AttackSampleIdentity) -> str:
    subset_tokens = [token.lower() for token in str(identity.subset_name or "unknown").split("_") if token]
    model_tokens = [token.lower() for token in str(identity.model_family or "").split("_") if token]
    remainder = subset_tokens
    if model_tokens and subset_tokens[: len(model_tokens)] == model_tokens:
        remainder = subset_tokens[len(model_tokens) :]

    best_match: tuple[int, str] | None = None
    for index, token in enumerate(remainder):
        label = _ADAPTER_LABEL_BY_TOKEN.get(token)
        if label is not None:
            best_match = (index, label)

    for index in range(max(0, len(remainder) - 1)):
        pair = (remainder[index], remainder[index + 1])
        label = _ADAPTER_LABEL_BY_PAIR.get(pair)
        if label is not None:
            best_match = (index + 1, label)

    if best_match is not None:
        return best_match[1]
    return "lora"


def _selected_threshold_rows_by_fpr(summary: dict[str, Any]) -> dict[float, dict[str, Any]]:
    rows = summary.get("selected_threshold_evaluation", [])
    if not isinstance(rows, list):
        return {}
    grouped: dict[float, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "accepted_fpr" not in row:
            continue
        grouped[float(row["accepted_fpr"])] = dict(row)
    return grouped


def _offline_metric(summary: dict[str, Any], key: str) -> str:
    metrics = summary.get("offline_metrics", {})
    value = metrics.get(key) if isinstance(metrics, dict) else None
    return "-" if value is None else f"{float(value):.3f}"


def _threshold_metric(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return "-"
    value = row.get(key)
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _raw_metric_text(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _params_text(params: Any) -> str:
    if not isinstance(params, dict) or not params:
        return "-"
    return ", ".join(f"{key}={params[key]}" for key in sorted(params))


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_tuning_selection_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    tuning = report.get("tuning")
    if not isinstance(tuning, dict):
        return None
    winner = tuning.get("winner")
    if not isinstance(winner, dict):
        return None

    candidates = tuning.get("candidates")
    candidate_rows = [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []
    ok_candidates = [
        row
        for row in candidate_rows
        if row.get("status") == "ok" and _metric_value(row, "selection_metric_mean") is not None
    ]
    ok_candidates.sort(
        key=lambda row: (
            -float(row["selection_metric_mean"]),
            float(row.get("selection_metric_std") or 0.0),
            int(row.get("complexity_rank") or 0),
            int(row.get("task_index") or 0),
        )
    )

    cnn_hyperparams = tuning.get("cnn_hyperparams")
    n_candidates = None
    if isinstance(cnn_hyperparams, dict):
        n_candidates = cnn_hyperparams.get("n_candidates")
    if n_candidates is None:
        n_candidates = tuning.get("tasks_total")
    if n_candidates is None and candidate_rows:
        n_candidates = len(candidate_rows)

    metric_name = (
        winner.get("selection_metric_name")
        or tuning.get("metric")
        or tuning.get("selection_metric_name")
    )

    summary = {
        "metric": metric_name,
        "execution_mode": tuning.get("execution_mode") or winner.get("execution_mode"),
        "n_candidates": n_candidates,
        "cv_folds_resolved": tuning.get("cv_folds_resolved"),
        "estimated_total_fits": tuning.get("estimated_total_fits"),
        "winner": {
            "task_index": winner.get("task_index"),
            "model_name": winner.get("model_name") or tuning.get("model_name"),
            "params": winner.get("params"),
            "status": winner.get("status"),
            "selection_metric_name": metric_name,
            "selection_metric_mean": winner.get("selection_metric_mean"),
            "selection_metric_std": winner.get("selection_metric_std"),
            "accuracy_mean": winner.get("accuracy_mean"),
            "accuracy_std": winner.get("accuracy_std"),
            "fold_results": winner.get("fold_results") if isinstance(winner.get("fold_results"), list) else [],
        },
        "top_candidates": [
            {
                "rank": index + 1,
                "task_index": row.get("task_index"),
                "model_name": row.get("model_name"),
                "params": row.get("params"),
                "selection_metric_name": row.get("selection_metric_name") or metric_name,
                "selection_metric_mean": row.get("selection_metric_mean"),
                "selection_metric_std": row.get("selection_metric_std"),
                "accuracy_mean": row.get("accuracy_mean"),
                "accuracy_std": row.get("accuracy_std"),
            }
            for index, row in enumerate(ok_candidates[:5])
        ],
    }
    return summary


def _append_tuning_selection_section(
    *,
    lines: list[str],
    tuning_selection: dict[str, Any] | None,
) -> None:
    if not isinstance(tuning_selection, dict):
        return

    winner = tuning_selection.get("winner")
    if not isinstance(winner, dict):
        return

    lines.append("## Cross-Validation Selection")
    lines.append("")
    lines.append("| Model | Task | Candidates | Mode | Metric | CV Mean | CV Std | CV Accuracy |")
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |")
    lines.append(
        "| {model} | {task_index} | {n_candidates} | {mode} | {metric} | {mean} | {std} | {accuracy} |".format(
            model=str(winner.get("model_name") or "-"),
            task_index="-" if winner.get("task_index") is None else int(winner.get("task_index")),
            n_candidates="-" if tuning_selection.get("n_candidates") is None else int(tuning_selection["n_candidates"]),
            mode=str(tuning_selection.get("execution_mode") or "-"),
            metric=str(winner.get("selection_metric_name") or tuning_selection.get("metric") or "-"),
            mean=_raw_metric_text(winner.get("selection_metric_mean")),
            std=_raw_metric_text(winner.get("selection_metric_std")),
            accuracy=_raw_metric_text(winner.get("accuracy_mean")),
        )
    )
    lines.append("")

    params = winner.get("params")
    if isinstance(params, dict) and params:
        lines.append(f"Selected params: `{_params_text(params)}`.")
        lines.append("")

    fold_results = winner.get("fold_results")
    if isinstance(fold_results, list) and fold_results:
        lines.append("| Fold | Train N | Valid N | Metric | Accuracy |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for index, row in enumerate(fold_results, start=1):
            if not isinstance(row, dict):
                continue
            metric_value = row.get("selection_metric")
            if metric_value is None:
                metric_value = row.get(str(winner.get("selection_metric_name") or ""))
            lines.append(
                "| {fold} | {n_train} | {n_valid} | {metric} | {accuracy} |".format(
                    fold=index,
                    n_train=int(row.get("n_train", 0)),
                    n_valid=int(row.get("n_valid", 0)),
                    metric=_raw_metric_text(metric_value),
                    accuracy=_raw_metric_text(row.get("accuracy")),
                )
            )
        lines.append("")

    top_candidates = tuning_selection.get("top_candidates")
    if isinstance(top_candidates, list) and top_candidates:
        lines.append("### Top CV Candidates")
        lines.append("")
        lines.append("| Rank | Task | CV Mean | CV Std | CV Accuracy | Params |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for row in top_candidates:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {rank} | {task_index} | {mean} | {std} | {accuracy} | `{params}` |".format(
                    rank=int(row.get("rank", 0)),
                    task_index="-" if row.get("task_index") is None else int(row.get("task_index")),
                    mean=_raw_metric_text(row.get("selection_metric_mean")),
                    std=_raw_metric_text(row.get("selection_metric_std")),
                    accuracy=_raw_metric_text(row.get("accuracy_mean")),
                    params=_params_text(row.get("params")),
                )
            )
        lines.append("")


def _partition_counts_from_report(
    report: dict[str, Any],
    *,
    partition_name: str,
) -> dict[str, int]:
    data_info = report.get("data_info", {})
    key_map = {
        "train": ("n_train", "n_train_clean", "n_train_backdoored", "n_train_unknown_label"),
        "calibration": (
            "n_calibration",
            "n_calibration_clean",
            "n_calibration_backdoored",
            "n_calibration_unknown_label",
        ),
        "inference": (
            "n_inference",
            "n_inference_clean",
            "n_inference_backdoored",
            "n_inference_unknown_label",
        ),
    }
    n_key, clean_key, backdoor_key, unknown_key = key_map[partition_name]
    clean = int(data_info.get(clean_key, 0) or 0)
    backdoored = int(data_info.get(backdoor_key, 0) or 0)
    unknown = int(data_info.get(unknown_key, 0) or 0)
    n_samples = data_info.get(n_key)
    if n_samples is None:
        n_samples = clean + backdoored + unknown

    if int(n_samples) == 0:
        partition = (report.get("multiclass_assessment") or {}).get(partition_name)
        if isinstance(partition, dict):
            true_distribution = partition.get("true_class_distribution", [])
            if isinstance(true_distribution, list):
                n_samples = int(sum(int(row.get("count", 0) or 0) for row in true_distribution if isinstance(row, dict)))
                clean = int(
                    sum(
                        int(row.get("count", 0) or 0)
                        for row in true_distribution
                        if isinstance(row, dict) and str(row.get("class_name")) == "clean"
                    )
                )
                backdoored = max(0, int(n_samples) - int(clean))
                unknown = 0

    return {
        "n_samples": int(n_samples),
        "clean": int(clean),
        "backdoored": int(backdoored),
        "unknown": int(unknown),
    }


def _binary_partition_summary_from_report(
    report: dict[str, Any],
    *,
    partition_name: str,
) -> dict[str, Any] | None:
    fit_assessment = report.get("fit_assessment", {})
    key_map = {
        "train": ("train_score_summary", "train_offline_metrics", None),
        "calibration": ("calibration_score_summary", "calibration_offline_metrics", None),
        "inference": ("inference_score_summary", "offline_metrics", "threshold_evaluation"),
    }
    score_key, metrics_key, threshold_key = key_map[partition_name]
    score_summary = fit_assessment.get(score_key)
    offline_metrics = fit_assessment.get(metrics_key)
    if score_summary is None and offline_metrics is None:
        return None

    summary = {
        **_partition_counts_from_report(report, partition_name=partition_name),
        "score_summary": score_summary,
        "offline_metrics": offline_metrics,
    }
    if threshold_key is not None:
        threshold_rows = fit_assessment.get(threshold_key)
        if isinstance(threshold_rows, list):
            summary["threshold_evaluation"] = [dict(row) for row in threshold_rows if isinstance(row, dict)]
    return summary


def _multiclass_partition_summary_from_report(
    report: dict[str, Any],
    *,
    partition_name: str,
) -> dict[str, Any] | None:
    multiclass_assessment = report.get("multiclass_assessment", {})
    partition = multiclass_assessment.get(partition_name)
    if not isinstance(partition, dict):
        return None
    return dict(partition)


def _open_set_partition_summary_from_report(
    report: dict[str, Any],
    *,
    partition_name: str,
) -> dict[str, Any] | None:
    open_set_assessment = report.get("open_set_assessment", {})
    if not isinstance(open_set_assessment, dict):
        return None
    partition = open_set_assessment.get(partition_name)
    if not isinstance(partition, dict):
        return None
    return dict(partition)


def summarize_full_partition(
    *,
    sample_identities: list[AttackSampleIdentity],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    subset_names = sorted({str(identity.subset_name) for identity in sample_identities})
    return _summarize_group_slice(
        group_scores=np.asarray(scores, dtype=np.float64),
        group_ranks=_score_percentile_ranks(np.asarray(scores, dtype=np.float64)),
        group_labels=labels,
        threshold_specs=threshold_specs,
        selected_threshold_specs=selected_threshold_specs,
        source_subsets=subset_names,
    )


def _selected_threshold_rows_table(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = summary.get("selected_threshold_evaluation", [])
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _append_group_table(
    *,
    lines: list[str],
    title: str | None,
    group_label: str,
    grouped: dict[str, Any],
    note: str | None = None,
) -> None:
    if not grouped:
        return
    if title:
        lines.append(f"## {title}")
        lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    lines.append(
        f"| {group_label} | N | Clean | Backdoor | Unknown | AUROC | AUPRC | P@P | Rec@1% | Prec@1% | Rec@5% | Prec@5% | Rec@10% | Prec@10% |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for group_name in sorted(grouped):
        summary = grouped[group_name]
        label_counts = summary.get("label_counts", {})
        selected_rows = _selected_threshold_rows_by_fpr(summary)
        row_1 = selected_rows.get(0.01)
        row_5 = selected_rows.get(0.05)
        row_10 = selected_rows.get(0.10)
        lines.append(
            "| {group_name} | {n_samples} | {clean} | {backdoored} | {unknown} | {auroc} | {auprc} | {p_at_p} | {r1} | {p1} | {r5} | {p5} | {r10} | {p10} |".format(
                group_name=group_name,
                n_samples=int(summary.get("n_samples", 0)),
                clean=int(label_counts.get("clean", 0)),
                backdoored=int(label_counts.get("backdoored", 0)),
                unknown=int(label_counts.get("unknown", 0)),
                auroc=_offline_metric(summary, "auroc"),
                auprc=_offline_metric(summary, "auprc"),
                p_at_p=_offline_metric(summary, "precision_at_num_positives"),
                r1=_threshold_metric(row_1, "recall"),
                p1=_threshold_metric(row_1, "precision"),
                r5=_threshold_metric(row_5, "recall"),
                p5=_threshold_metric(row_5, "precision"),
                r10=_threshold_metric(row_10, "recall"),
                p10=_threshold_metric(row_10, "precision"),
            )
        )
    lines.append("")


def build_architecture_markdown(
    *,
    analysis: dict[str, Any],
    selected_threshold_summary: dict[str, Any] | None,
    run_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Architecture Analysis")
    lines.append("")
    lines.append(f"Run: `{run_dir}`")
    if isinstance(selected_threshold_summary, dict):
        method = str(selected_threshold_summary.get("method", "unknown"))
        accepted_fprs = selected_threshold_summary.get("accepted_fprs", [])
        fpr_text = ", ".join(f"{100.0 * float(value):.0f}%" for value in accepted_fprs) if isinstance(accepted_fprs, list) else "-"
        lines.append(f"Selected-threshold method: `{method}` with accepted FPRs `{fpr_text}`.")
    lines.append("")

    for partition_name in ("train", "calibration", "inference"):
        partition = analysis.get(partition_name)
        if not isinstance(partition, dict):
            continue
        architectures = partition.get("architectures", {})
        if not isinstance(architectures, dict) or not architectures:
            continue

        lines.append(f"## {partition_name.capitalize()}")
        lines.append("")
        lines.append(
            "| Architecture | N | Clean | Backdoor | Unknown | AUROC | AUPRC | P@P | Rec@1% | Prec@1% | Rec@5% | Prec@5% | Rec@10% | Prec@10% |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for architecture in sorted(architectures):
            summary = architectures[architecture]
            label_counts = summary.get("label_counts", {})
            selected_rows = _selected_threshold_rows_by_fpr(summary)
            row_1 = selected_rows.get(0.01)
            row_5 = selected_rows.get(0.05)
            row_10 = selected_rows.get(0.10)
            lines.append(
                "| {architecture} | {n_samples} | {clean} | {backdoored} | {unknown} | {auroc} | {auprc} | {p_at_p} | {r1} | {p1} | {r5} | {p5} | {r10} | {p10} |".format(
                    architecture=architecture,
                    n_samples=int(summary.get("n_samples", 0)),
                    clean=int(label_counts.get("clean", 0)),
                    backdoored=int(label_counts.get("backdoored", 0)),
                    unknown=int(label_counts.get("unknown", 0)),
                    auroc=_offline_metric(summary, "auroc"),
                    auprc=_offline_metric(summary, "auprc"),
                    p_at_p=_offline_metric(summary, "precision_at_num_positives"),
                    r1=_threshold_metric(row_1, "recall"),
                    p1=_threshold_metric(row_1, "precision"),
                    r5=_threshold_metric(row_5, "recall"),
                    p5=_threshold_metric(row_5, "precision"),
                    r10=_threshold_metric(row_10, "recall"),
                    p10=_threshold_metric(row_10, "precision"),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_architecture_analysis(
    *,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    reports_dir = run_dir / "reports"
    report = _load_json(reports_dir / "supervised_report.json")
    tuning_manifest = _load_optional_json(reports_dir / "tuning_manifest.json")
    selected_threshold_summary = _load_optional_json(reports_dir / "selected_threshold.json")

    sample_identities = _infer_sample_identities_for_run(
        reports_dir=reports_dir,
        tuning_manifest=tuning_manifest,
    )
    identity_by_name = {identity.model_name: identity for identity in sample_identities}

    threshold_specs = _extract_threshold_specs_from_report(report)
    selected_threshold_specs = _build_selected_threshold_specs(selected_threshold_summary)

    partitions: dict[str, Any] = {}
    for partition_name, filename in (
        ("train", "train_scores.csv"),
        ("calibration", "calibration_scores.csv"),
        ("inference", "inference_scores.csv"),
    ):
        csv_path = reports_dir / filename
        if not csv_path.exists():
            continue
        partition = _load_score_partition(
            partition_name=partition_name,
            csv_path=csv_path,
            identity_by_name=identity_by_name,
        )
        partitions[partition_name] = summarize_architecture_groups(
            sample_identities=partition.sample_identities,
            labels=partition.labels,
            scores=partition.scores,
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        )

    return {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(run_dir),
        "model_name": report.get("tuning", {}).get("winner", {}).get("model_name")
        or report.get("tuning", {}).get("model_name"),
        "selected_threshold_summary": selected_threshold_summary,
        **partitions,
    }


def build_supervised_results_summary(
    *,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    reports_dir = run_dir / "reports"
    report = _load_json(reports_dir / "supervised_report.json")
    tuning_manifest = _load_optional_json(reports_dir / "tuning_manifest.json")
    selected_threshold_summary = _load_optional_json(reports_dir / "selected_threshold.json")
    task_mode = str(report.get("task", {}).get("task_mode") or "")

    if task_mode == _SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS:
        binary_partitions: dict[str, Any] = {}
        class_partitions: dict[str, Any] = {}
        open_set_partitions: dict[str, Any] = {}
        for partition_name in ("train", "calibration", "inference"):
            binary_partition = _binary_partition_summary_from_report(
                report,
                partition_name=partition_name,
            )
            if isinstance(binary_partition, dict):
                binary_partitions[partition_name] = binary_partition
            class_partition = _multiclass_partition_summary_from_report(
                report,
                partition_name=partition_name,
            )
            if isinstance(class_partition, dict):
                class_partitions[partition_name] = class_partition
            open_set_partition = _open_set_partition_summary_from_report(
                report,
                partition_name=partition_name,
            )
            if isinstance(open_set_partition, dict):
                open_set_partitions[partition_name] = open_set_partition

        open_set_assessment = report.get("open_set_assessment", {})
        open_set_config = (
            open_set_assessment.get("config", {})
            if isinstance(open_set_assessment, dict)
            else {}
        )

        return {
            "script_version": SCRIPT_VERSION,
            "summary_mode": "multiclass",
            "run_dir": str(run_dir),
            "model_name": report.get("tuning", {}).get("winner", {}).get("model_name")
            or report.get("tuning", {}).get("model_name"),
            "task_mode": task_mode,
            "selected_threshold_summary": selected_threshold_summary,
            "tuning_selection": _build_tuning_selection_summary(report),
            "binary_classification": {
                "score_definition": report.get("fit_assessment", {}).get("score_definition"),
                "binary_projection": report.get("fit_assessment", {}).get("binary_projection"),
                "partitions": binary_partitions,
            },
            "class_results": {
                "class_names": list(report.get("task", {}).get("class_names", [])),
                "partitions": class_partitions,
            },
            "open_set_unknown": {
                "config": dict(open_set_config) if isinstance(open_set_config, dict) else {},
                "partitions": open_set_partitions,
            },
            "attack_analysis": (
                report.get("attack_analysis", {}).get("inference")
                if isinstance(report.get("attack_analysis"), dict)
                else None
            ),
        }

    sample_identities = _infer_sample_identities_for_run(
        reports_dir=reports_dir,
        tuning_manifest=tuning_manifest,
    )
    identity_by_name = {identity.model_name: identity for identity in sample_identities}

    threshold_specs = _extract_threshold_specs_from_report(report)
    selected_threshold_specs = _build_selected_threshold_specs(selected_threshold_summary)

    inference_partition: ScorePartition | None = None
    inference_csv_path = reports_dir / "inference_scores.csv"
    if inference_csv_path.exists():
        inference_partition = _load_score_partition(
            partition_name="inference",
            csv_path=inference_csv_path,
            identity_by_name=identity_by_name,
        )

    inference_overview = None
    dataset_analysis = None
    adapter_analysis = None
    attack_analysis = None
    architecture_analysis = None
    if inference_partition is not None:
        inference_overview = summarize_full_partition(
            sample_identities=inference_partition.sample_identities,
            labels=inference_partition.labels,
            scores=inference_partition.scores,
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        )
        dataset_analysis = summarize_dataset_groups(
            sample_identities=inference_partition.sample_identities,
            labels=inference_partition.labels,
            scores=inference_partition.scores,
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        )
        adapter_analysis = summarize_adapter_groups(
            sample_identities=inference_partition.sample_identities,
            labels=inference_partition.labels,
            scores=inference_partition.scores,
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        )
        attack_analysis = summarize_attack_groups(
            sample_identities=inference_partition.sample_identities,
            labels=inference_partition.labels,
            scores=inference_partition.scores,
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        )
        architecture_analysis = summarize_architecture_groups(
            sample_identities=inference_partition.sample_identities,
            labels=inference_partition.labels,
            scores=inference_partition.scores,
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        )
    return {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(run_dir),
        "model_name": report.get("tuning", {}).get("winner", {}).get("model_name")
        or report.get("tuning", {}).get("model_name"),
        "selected_threshold_summary": selected_threshold_summary,
        "tuning_selection": _build_tuning_selection_summary(report),
        "inference_overview": inference_overview,
        "dataset_analysis": dataset_analysis,
        "attack_analysis": attack_analysis,
        "adapter_analysis": adapter_analysis,
        "architecture_analysis": architecture_analysis,
    }


def _append_multiclass_binary_partition_table(
    *,
    lines: list[str],
    binary_classification: dict[str, Any],
) -> None:
    partitions = binary_classification.get("partitions", {})
    if not isinstance(partitions, dict) or not partitions:
        return
    lines.append("## Binary Classification")
    lines.append("")
    projection = binary_classification.get("binary_projection")
    score_definition = binary_classification.get("score_definition")
    if projection is not None or score_definition is not None:
        lines.append(
            "Projected clean-vs-backdoor view using "
            f"`{score_definition or 'score'}` with projection `{projection or 'unknown'}`."
        )
        lines.append("")
    lines.append("| Partition | N | Clean | Backdoor | Unknown | AUROC | AUPRC | P@P |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for partition_name in ("train", "calibration", "inference"):
        summary = partitions.get(partition_name)
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("offline_metrics", {})
        lines.append(
            "| {partition} | {n_samples} | {clean} | {backdoored} | {unknown} | {auroc} | {auprc} | {p_at_p} |".format(
                partition=partition_name,
                n_samples=int(summary.get("n_samples", 0)),
                clean=int(summary.get("clean", 0)),
                backdoored=int(summary.get("backdoored", 0)),
                unknown=int(summary.get("unknown", 0)),
                auroc=_raw_metric_text((metrics or {}).get("auroc") if isinstance(metrics, dict) else None),
                auprc=_raw_metric_text((metrics or {}).get("auprc") if isinstance(metrics, dict) else None),
                p_at_p=_raw_metric_text(
                    (metrics or {}).get("precision_at_num_positives") if isinstance(metrics, dict) else None
                ),
            )
        )
    lines.append("")

    inference_summary = partitions.get("inference")
    threshold_rows = (
        inference_summary.get("threshold_evaluation")
        if isinstance(inference_summary, dict)
        else None
    )
    if isinstance(threshold_rows, list) and threshold_rows:
        lines.append("## Binary Threshold Sweep")
        lines.append("")
        lines.append("| Source Percentile | Threshold | Recall | Precision | FPR | Fraction Flagged |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in threshold_rows:
            if not isinstance(row, dict):
                continue
            percentile = row.get("percentile_from_train")
            if percentile is None:
                percentile = row.get("percentile_from_inference")
            percentile_text = "-" if percentile is None else f"{float(percentile):.0f}%"
            lines.append(
                "| {percentile} | {threshold} | {recall} | {precision} | {fpr} | {fraction_flagged} |".format(
                    percentile=percentile_text,
                    threshold=_raw_metric_text(row.get("threshold")),
                    recall=_raw_metric_text(row.get("recall")),
                    precision=_raw_metric_text(row.get("precision")),
                    fpr=_raw_metric_text(row.get("false_positive_rate")),
                    fraction_flagged=_raw_metric_text(row.get("fraction_flagged")),
                )
            )
        lines.append("")


def _append_multiclass_overview_table(
    *,
    lines: list[str],
    class_results: dict[str, Any],
) -> None:
    partitions = class_results.get("partitions", {})
    if not isinstance(partitions, dict) or not partitions:
        return
    lines.append("## Multiclass Overview")
    lines.append("")
    lines.append("| Partition | Accuracy | Macro F1 | Micro F1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for partition_name in ("train", "calibration", "inference"):
        summary = partitions.get(partition_name)
        if not isinstance(summary, dict):
            continue
        lines.append(
            "| {partition} | {accuracy} | {macro_f1} | {micro_f1} |".format(
                partition=partition_name,
                accuracy=_raw_metric_text(summary.get("accuracy")),
                macro_f1=_raw_metric_text(summary.get("macro_f1")),
                micro_f1=_raw_metric_text(summary.get("micro_f1")),
            )
        )
    lines.append("")


def _append_multiclass_per_class_table(
    *,
    lines: list[str],
    partition_name: str,
    partition_summary: dict[str, Any],
) -> None:
    per_class = partition_summary.get("per_class")
    if not isinstance(per_class, list) or not per_class:
        return
    lines.append(f"## Per-Class {partition_name.capitalize()}")
    lines.append("")
    lines.append("| Class | Support | Predicted | Precision | Recall | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in per_class:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {class_name} | {support} | {predicted_count} | {precision} | {recall} | {f1} |".format(
                class_name=str(row.get("class_name", "unknown")),
                support=int(row.get("support", 0)),
                predicted_count=int(row.get("predicted_count", 0)),
                precision=_raw_metric_text(row.get("precision")),
                recall=_raw_metric_text(row.get("recall")),
                f1=_raw_metric_text(row.get("f1")),
            )
        )
    lines.append("")


def _append_open_set_overview_table(
    *,
    lines: list[str],
    open_set_unknown: dict[str, Any],
) -> None:
    partitions = open_set_unknown.get("partitions", {})
    if not isinstance(partitions, dict) or not partitions:
        return
    lines.append("## Open-Set Unknown Attack")
    lines.append("")
    config = open_set_unknown.get("config", {})
    if isinstance(config, dict) and config:
        lines.append(
            "Unknown attack rule: clean below backdoor-score threshold "
            f"`{_raw_metric_text(config.get('attack_threshold'))}`; otherwise `unknown_attack` "
            "when known-attack confidence is below "
            f"`{_raw_metric_text(config.get('known_attack_confidence_threshold'))}`."
        )
        lines.append("")
    lines.append("| Partition | Accuracy | Macro F1 | Micro F1 | Unknown Predictions |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for partition_name in ("train", "calibration", "inference"):
        summary = partitions.get(partition_name)
        if not isinstance(summary, dict):
            continue
        lines.append(
            "| {partition} | {accuracy} | {macro_f1} | {micro_f1} | {unknown_predictions} |".format(
                partition=partition_name,
                accuracy=_raw_metric_text(summary.get("accuracy")),
                macro_f1=_raw_metric_text(summary.get("macro_f1")),
                micro_f1=_raw_metric_text(summary.get("micro_f1")),
                unknown_predictions=int(summary.get("n_unknown_attack_predictions", 0)),
            )
        )
    lines.append("")


def _append_open_set_per_class_table(
    *,
    lines: list[str],
    partition_name: str,
    partition_summary: dict[str, Any],
) -> None:
    per_class = partition_summary.get("per_class")
    if not isinstance(per_class, list) or not per_class:
        return
    lines.append(f"## Open-Set Per-Class {partition_name.capitalize()}")
    lines.append("")
    lines.append("| Class | Support | Predicted | Precision | Recall | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in per_class:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {class_name} | {support} | {predicted_count} | {precision} | {recall} | {f1} |".format(
                class_name=str(row.get("class_name", "unknown")),
                support=int(row.get("support", 0)),
                predicted_count=int(row.get("predicted_count", 0)),
                precision=_raw_metric_text(row.get("precision")),
                recall=_raw_metric_text(row.get("recall")),
                f1=_raw_metric_text(row.get("f1")),
            )
        )
    lines.append("")


def _build_multiclass_supervised_results_markdown(
    *,
    summary: dict[str, Any],
    run_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append(f"Run: `{run_dir}`")
    lines.append("")

    _append_tuning_selection_section(
        lines=lines,
        tuning_selection=summary.get("tuning_selection"),
    )
    _append_multiclass_binary_partition_table(
        lines=lines,
        binary_classification=summary.get("binary_classification", {}),
    )
    attack_analysis = summary.get("attack_analysis", {})
    attacks = attack_analysis.get("attacks", {}) if isinstance(attack_analysis, dict) else {}
    if attacks:
        _append_group_table(
            lines=lines,
            title="Per Attack",
            group_label="Attack",
            grouped=attacks,
            note=attack_analysis.get("grouping_rule")
            if isinstance(attack_analysis, dict)
            else None,
        )
    _append_multiclass_overview_table(
        lines=lines,
        class_results=summary.get("class_results", {}),
    )
    open_set_unknown = summary.get("open_set_unknown", {})
    if isinstance(open_set_unknown, dict):
        _append_open_set_overview_table(
            lines=lines,
            open_set_unknown=open_set_unknown,
        )
    partitions = summary.get("class_results", {}).get("partitions", {})
    if isinstance(partitions, dict):
        for partition_name in ("train", "inference"):
            partition_summary = partitions.get(partition_name)
            if isinstance(partition_summary, dict):
                _append_multiclass_per_class_table(
                    lines=lines,
                    partition_name=partition_name,
                    partition_summary=partition_summary,
                )
    open_set_partitions = (
        open_set_unknown.get("partitions", {})
        if isinstance(open_set_unknown, dict)
        else {}
    )
    if isinstance(open_set_partitions, dict):
        partition_summary = open_set_partitions.get("inference")
        if isinstance(partition_summary, dict):
            _append_open_set_per_class_table(
                lines=lines,
                partition_name="inference",
                partition_summary=partition_summary,
            )

    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def build_supervised_results_markdown(
    *,
    summary: dict[str, Any],
    run_dir: Path,
) -> str:
    if str(summary.get("summary_mode") or "") == "multiclass":
        return _build_multiclass_supervised_results_markdown(
            summary=summary,
            run_dir=run_dir,
        )

    lines: list[str] = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append(f"Run: `{run_dir}`")
    selected_threshold_summary = summary.get("selected_threshold_summary")
    if isinstance(selected_threshold_summary, dict):
        method = str(selected_threshold_summary.get("method", "unknown"))
        accepted_fprs = selected_threshold_summary.get("accepted_fprs", [])
        fpr_text = (
            ", ".join(f"{100.0 * float(value):.0f}%" for value in accepted_fprs)
            if isinstance(accepted_fprs, list)
            else "-"
        )
        lines.append(f"Selected-threshold method: `{method}` with accepted FPRs `{fpr_text}`.")
    lines.append("")

    _append_tuning_selection_section(
        lines=lines,
        tuning_selection=summary.get("tuning_selection"),
    )

    inference_overview = summary.get("inference_overview")
    if isinstance(inference_overview, dict):
        label_counts = inference_overview.get("label_counts", {})
        lines.append("## Entire Inference Set")
        lines.append("")
        lines.append(
            "N={n}, clean={clean}, backdoor={backdoored}, unknown={unknown}, "
            "AUROC={auroc}, AUPRC={auprc}, P@P={p_at_p}.".format(
                n=int(inference_overview.get("n_samples", 0)),
                clean=int(label_counts.get("clean", 0)),
                backdoored=int(label_counts.get("backdoored", 0)),
                unknown=int(label_counts.get("unknown", 0)),
                auroc=_offline_metric(inference_overview, "auroc"),
                auprc=_offline_metric(inference_overview, "auprc"),
                p_at_p=_offline_metric(inference_overview, "precision_at_num_positives"),
            )
        )
        lines.append("")
        threshold_rows = _selected_threshold_rows_table(inference_overview)
        if threshold_rows:
            lines.append("| Accepted FPR | Threshold | Recall | Precision | FPR | Fraction Flagged |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in threshold_rows:
                lines.append(
                    "| {accepted_fpr} | {threshold:.6f} | {recall} | {precision} | {false_positive_rate} | {fraction_flagged:.3f} |".format(
                        accepted_fpr=f"{100.0 * float(row['accepted_fpr']):.0f}%",
                        threshold=float(row["threshold"]),
                        recall=_threshold_metric(row, "recall"),
                        precision=_threshold_metric(row, "precision"),
                        false_positive_rate=_threshold_metric(row, "false_positive_rate"),
                        fraction_flagged=float(row.get("fraction_flagged", 0.0)),
                    )
                )
            lines.append("")

    architecture_analysis = summary.get("architecture_analysis", {})
    architectures = architecture_analysis.get("architectures", {}) if isinstance(architecture_analysis, dict) else {}
    _append_group_table(
        lines=lines,
        title="Per Architecture",
        group_label="Architecture",
        grouped=architectures,
        note=architecture_analysis.get("grouping_rule") if isinstance(architecture_analysis, dict) else None,
    )

    dataset_analysis = summary.get("dataset_analysis", {})
    datasets = dataset_analysis.get("datasets", {}) if isinstance(dataset_analysis, dict) else {}
    _append_group_table(
        lines=lines,
        title="Per Dataset",
        group_label="Dataset",
        grouped=datasets,
        note=dataset_analysis.get("grouping_rule") if isinstance(dataset_analysis, dict) else None,
    )

    adapter_analysis = summary.get("adapter_analysis", {})
    adapters = adapter_analysis.get("adapters", {}) if isinstance(adapter_analysis, dict) else {}
    _append_group_table(
        lines=lines,
        title="Per Adapter",
        group_label="Adapter",
        grouped=adapters,
        note=adapter_analysis.get("grouping_rule") if isinstance(adapter_analysis, dict) else None,
    )

    attack_analysis = summary.get("attack_analysis", {})
    attacks = attack_analysis.get("attacks", {}) if isinstance(attack_analysis, dict) else {}
    if attacks:
        _append_group_table(
            lines=lines,
            title="Per Attack",
            group_label="Attack",
            grouped=attacks,
            note=attack_analysis.get("grouping_rule")
            if isinstance(attack_analysis, dict)
            else (
                "This section reuses the existing attack-analysis semantics from the supervised report."
            ),
        )

    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def _summary_payload_for_report(summary: dict[str, Any], markdown_path: Path) -> dict[str, Any]:
    payload = {
        "script_version": summary.get("script_version"),
        "model_name": summary.get("model_name"),
        "summary_mode": summary.get("summary_mode"),
        "task_mode": summary.get("task_mode"),
        "results_summary_md": str(markdown_path),
        "selected_threshold_summary": summary.get("selected_threshold_summary"),
    }
    if "inference_overview" in summary:
        payload["inference_overview"] = summary.get("inference_overview")
    if "binary_classification" in summary:
        payload["binary_classification"] = summary.get("binary_classification")
    if "class_results" in summary:
        payload["class_results"] = summary.get("class_results")
    if "open_set_unknown" in summary:
        payload["open_set_unknown"] = summary.get("open_set_unknown")
    if "attack_analysis" in summary:
        payload["attack_analysis"] = summary.get("attack_analysis")
    if "tuning_selection" in summary:
        payload["tuning_selection"] = summary.get("tuning_selection")
    return payload


def write_supervised_results_summary_outputs(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    update_report: bool = False,
    update_artifact_index: bool = True,
    remove_legacy_outputs: bool = True,
) -> dict[str, Path]:
    run_dir = run_dir.expanduser().resolve()
    reports_dir = run_dir / "reports"
    markdown_path = reports_dir / RESULTS_SUMMARY_MD_FILENAME
    markdown = build_supervised_results_markdown(
        summary=summary,
        run_dir=run_dir,
    )
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    if remove_legacy_outputs:
        for legacy_name in (
            LEGACY_ARCHITECTURE_ANALYSIS_JSON_FILENAME,
            LEGACY_ARCHITECTURE_ANALYSIS_MD_FILENAME,
        ):
            legacy_path = reports_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

    artifact_index_path = run_dir / "artifact_index.json"
    if update_artifact_index:
        artifact_index: dict[str, Any] = _load_optional_json(artifact_index_path) or {}
        artifact_index.pop("architecture_analysis_json", None)
        artifact_index.pop("architecture_analysis_md", None)
        artifact_index["results_summary_md"] = str(markdown_path)
        _write_json(artifact_index_path, artifact_index)

    updated_report_path: Path | None = None
    if update_report:
        report_path = reports_dir / "supervised_report.json"
        report = _load_json(report_path)
        dataset_analysis = summary.get("dataset_analysis")
        adapter_analysis = summary.get("adapter_analysis")
        attack_analysis = summary.get("attack_analysis")
        architecture_analysis = summary.get("architecture_analysis")
        report["results_summary"] = _summary_payload_for_report(summary, markdown_path)
        if str(summary.get("summary_mode") or "") != "multiclass":
            if isinstance(dataset_analysis, dict):
                report["dataset_analysis"] = {"inference": dataset_analysis}
            else:
                report.pop("dataset_analysis", None)
            if isinstance(adapter_analysis, dict):
                report["adapter_analysis"] = {"inference": adapter_analysis}
            else:
                report.pop("adapter_analysis", None)
            if isinstance(attack_analysis, dict):
                prior_attack_analysis = report.get("attack_analysis")
                if isinstance(prior_attack_analysis, dict):
                    updated_attack_analysis = dict(prior_attack_analysis)
                else:
                    updated_attack_analysis = {}
                updated_attack_analysis["inference"] = attack_analysis
                report["attack_analysis"] = updated_attack_analysis
            else:
                report.pop("attack_analysis", None)
            if isinstance(architecture_analysis, dict):
                report["architecture_analysis"] = {"inference": architecture_analysis}
            else:
                report.pop("architecture_analysis", None)
        updated_report_path = _write_json(report_path, report)

    outputs = {
        "results_summary_md": markdown_path,
    }
    if updated_report_path is not None:
        outputs["updated_report"] = updated_report_path
    return outputs


def run_supervised_results_summary(
    *,
    run_spec: str | Path,
    runs_root: Path = DEFAULT_SUPERVISED_RUNS_ROOT,
    update_report: bool = False,
) -> dict[str, Path]:
    run_dir = resolve_supervised_run_dir(run_spec, runs_root=runs_root)
    summary = build_supervised_results_summary(run_dir=run_dir)
    return write_supervised_results_summary_outputs(
        run_dir=run_dir,
        summary=summary,
        update_report=bool(update_report),
    )


def run_supervised_results_summary_for_all_runs(
    *,
    runs_root: Path = DEFAULT_SUPERVISED_RUNS_ROOT,
    update_report: bool = False,
) -> dict[str, dict[str, str]]:
    runs_root = runs_root.expanduser().resolve()
    processed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    failed: dict[str, str] = {}

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        reports_dir = run_dir / "reports"
        if not (reports_dir / "supervised_report.json").exists():
            skipped[run_dir.name] = "missing reports/supervised_report.json"
            continue
        if not (reports_dir / "inference_scores.csv").exists():
            skipped[run_dir.name] = "missing reports/inference_scores.csv"
            continue
        try:
            outputs = run_supervised_results_summary(
                run_spec=run_dir,
                runs_root=runs_root,
                update_report=update_report,
            )
        except Exception as exc:  # pragma: no cover - integration path
            failed[run_dir.name] = str(exc)
            continue
        processed[run_dir.name] = str(outputs["results_summary_md"])

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a single supervised results summary markdown for one run or backfill it across existing runs."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run",
        dest="run_spec",
        type=str,
        help="Supervised run name under runs/supervised or an explicit run directory path.",
    )
    target.add_argument(
        "--all-runs",
        action="store_true",
        help="Backfill results_summary.md for every eligible supervised run under --runs-root.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_SUPERVISED_RUNS_ROOT,
        help="Base directory used to resolve bare supervised run names.",
    )
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="Also embed dataset/adapter/architecture summary payloads and the markdown pointer into reports/supervised_report.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.all_runs):
        results = run_supervised_results_summary_for_all_runs(
            runs_root=args.runs_root,
            update_report=args.update_report,
        )
        print("Supervised results summary backfill complete")
        print(f"processed: {len(results['processed'])}")
        print(f"skipped: {len(results['skipped'])}")
        print(f"failed: {len(results['failed'])}")
        for run_name, path in sorted(results["processed"].items()):
            print(f"processed {run_name}: {path}")
        for run_name, reason in sorted(results["skipped"].items()):
            print(f"skipped {run_name}: {reason}")
        for run_name, reason in sorted(results["failed"].items()):
            print(f"failed {run_name}: {reason}")
        return 0

    outputs = run_supervised_results_summary(
        run_spec=args.run_spec,
        runs_root=args.runs_root,
        update_report=args.update_report,
    )
    print("Supervised results summary complete")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0
