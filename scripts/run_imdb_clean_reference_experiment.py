#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((Path("/tmp") / "upeftguard_mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from upeftguard.unsupervised.reporting import (
    compute_infer_threshold_rows,
    compute_offline_metrics,
    summarize_scores,
)
from upeftguard.utilities.core.run_context import create_run_context
from upeftguard.utilities.core.serialization import json_ready


SCRIPT_VERSION = "1.1.0"
PIPELINE_NAME = "imdb_clean_reference"
DEFAULT_FEATURE_FILE = Path("runs") / "feature_extract" / "llama2_7b_full" / "merged" / "spectral_features.npy"
DEFAULT_OUTPUT_ROOT = Path("runs")
DEFAULT_THRESHOLD_PERCENTILES = [90.0, 95.0, 97.5, 99.0]
DEFAULT_MAX_PCA_COMPONENTS = 20
DEFAULT_NUM_SPLITS = 1
FIT_CLEAN_COUNT = 150
CALIBRATION_CLEAN_COUNT = 50
EVAL_CLEAN_COUNT = 50

MODEL_NAME_RE = re.compile(r"^(?P<prefix>.+?)_label(?P<label>-?\d+)_(?P<sample_index>\d+)$")
IMDB_PREFIX = "llama2_7b_imdb_"
DATASET_SUFFIX = "_rank256_qv"
CLEAN_PREFIX = f"{IMDB_PREFIX}insertsent{DATASET_SUFFIX}"
ATTACK_PREFIXES = {
    "insertsent": CLEAN_PREFIX,
    "ripple": f"{IMDB_PREFIX}RIPPLE{DATASET_SUFFIX}",
    "stybkd": f"{IMDB_PREFIX}stybkd{DATASET_SUFFIX}",
    "syntactic": f"{IMDB_PREFIX}syntactic{DATASET_SUFFIX}",
}
HISTOGRAM_COLORS = {
    "fit_clean": "#4c78a8",
    "calibration_clean": "#72b7b2",
    "eval_clean": "#a0cbe8",
    "insertsent": "#f58518",
    "ripple": "#e45756",
    "stybkd": "#54a24b",
    "syntactic": "#b279a2",
}
OVERALL_SPLIT_METRIC_KEYS = [
    "reconstruction_auroc",
    "reconstruction_auprc",
    "reconstruction_precision_at_num_positives",
    "reconstruction_default_threshold",
    "reconstruction_default_precision",
    "reconstruction_default_recall",
    "reconstruction_default_false_positive_rate",
    "mahalanobis_auroc",
    "mahalanobis_auprc",
]
PER_ATTACK_SPLIT_METRIC_KEYS = [
    "reconstruction_auroc",
    "reconstruction_auprc",
    "reconstruction_default_precision",
    "reconstruction_default_recall",
    "reconstruction_default_false_positive_rate",
    "mahalanobis_auroc",
    "mahalanobis_auprc",
]


@dataclass(frozen=True)
class FeatureBundle:
    feature_file: Path
    model_names_file: Path
    labels_file: Path
    features: np.ndarray
    model_names: list[str]
    labels: np.ndarray


@dataclass(frozen=True)
class SampleRecord:
    row_index: int
    model_name: str
    prefix: str
    label: int
    attack_name: str
    sample_index: int


@dataclass(frozen=True)
class ScoreOutputs:
    reconstruction: np.ndarray
    mahalanobis: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype clean-reference anomaly detector for the llama2_7b IMDb subset.",
    )
    parser.add_argument(
        "--feature-file",
        default=str(DEFAULT_FEATURE_FILE),
        help=(
            "Feature bundle path, merged feature directory, or feature run name. "
            "Defaults to runs/feature_extract/llama2_7b_full/merged/spectral_features.npy."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory for the imdb_clean_reference run.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to the shared run-context timestamp format.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Deterministic seed used for the clean split.",
    )
    parser.add_argument(
        "--max-pca-components",
        type=int,
        default=DEFAULT_MAX_PCA_COMPONENTS,
        help="Upper bound on the number of PCA components fit on the clean reference set.",
    )
    parser.add_argument(
        "--threshold-percentiles",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLD_PERCENTILES,
        help="Calibration-clean score percentiles used to define detection thresholds.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=DEFAULT_NUM_SPLITS,
        help=(
            "Number of repeated clean splits to evaluate. "
            "Split seeds are derived deterministically as random_state + split_index."
        ),
    )
    return parser.parse_args()


def resolve_feature_file(path_spec: str | Path) -> Path:
    raw = Path(path_spec).expanduser()
    cwd = Path.cwd().resolve()

    if raw.suffix == ".npy":
        candidate = raw if raw.is_absolute() else (cwd / raw)
        return candidate.resolve()

    if raw.is_absolute():
        candidate_dir = raw
    elif len(raw.parts) > 1:
        candidate_dir = (cwd / raw).resolve()
    else:
        candidate_dir = (cwd / "runs" / "feature_extract" / raw / "merged").resolve()

    if candidate_dir.is_dir():
        return (candidate_dir / "spectral_features.npy").resolve()

    candidate_file = candidate_dir.resolve()
    if candidate_file.name == "spectral_features.npy":
        return candidate_file

    raise FileNotFoundError(f"Could not resolve feature bundle from: {path_spec}")


def load_feature_bundle(feature_path: Path) -> FeatureBundle:
    resolved_feature_path = resolve_feature_file(feature_path)
    if not resolved_feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {resolved_feature_path}")

    model_names_path = resolved_feature_path.with_name("spectral_model_names.json")
    labels_path = resolved_feature_path.with_name("spectral_labels.npy")
    if not model_names_path.exists():
        raise FileNotFoundError(f"Model names file not found: {model_names_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    features = np.load(resolved_feature_path, mmap_mode="r")
    labels = np.asarray(np.load(labels_path), dtype=np.int32)
    with open(model_names_path, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]

    if len(model_names) != int(features.shape[0]):
        raise ValueError(
            f"model_names length ({len(model_names)}) does not match feature rows ({features.shape[0]})"
        )
    if int(labels.shape[0]) != int(features.shape[0]):
        raise ValueError(
            f"labels length ({labels.shape[0]}) does not match feature rows ({features.shape[0]})"
        )

    return FeatureBundle(
        feature_file=resolved_feature_path,
        model_names_file=model_names_path.resolve(),
        labels_file=labels_path.resolve(),
        features=features,
        model_names=model_names,
        labels=labels,
    )


def parse_model_name(model_name: str) -> tuple[str, int, int]:
    match = MODEL_NAME_RE.fullmatch(str(model_name))
    if match is None:
        raise ValueError(f"Unrecognized model name format: {model_name}")
    prefix = str(match.group("prefix"))
    label = int(match.group("label"))
    sample_index = int(match.group("sample_index"))
    return prefix, label, sample_index


def attack_name_from_prefix(prefix: str) -> str:
    if not prefix.startswith(IMDB_PREFIX) or not prefix.endswith(DATASET_SUFFIX):
        raise ValueError(f"Unexpected IMDb prefix: {prefix}")
    attack_token = prefix[len(IMDB_PREFIX) : -len(DATASET_SUFFIX)]
    return str(attack_token).lower()


def collect_imdb_records(bundle: FeatureBundle) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for row_index, (model_name, label_value) in enumerate(zip(bundle.model_names, bundle.labels.tolist())):
        prefix, model_label, sample_index = parse_model_name(model_name)
        if not prefix.startswith(IMDB_PREFIX):
            continue
        if model_label != int(label_value):
            raise ValueError(
                f"Label mismatch for {model_name}: name label={model_label}, labels array={label_value}"
            )
        attack_name = attack_name_from_prefix(prefix)
        records.append(
            SampleRecord(
                row_index=row_index,
                model_name=model_name,
                prefix=prefix,
                label=int(label_value),
                attack_name=attack_name,
                sample_index=sample_index,
            )
        )
    if not records:
        raise RuntimeError("No llama2_7b IMDb samples were found in the selected feature bundle")
    return records


def split_clean_records(clean_records: list[SampleRecord], random_state: int) -> dict[str, list[SampleRecord]]:
    if len(clean_records) != FIT_CLEAN_COUNT + CALIBRATION_CLEAN_COUNT + EVAL_CLEAN_COUNT:
        raise ValueError(
            "Expected exactly 250 clean reference records, "
            f"found {len(clean_records)} for {CLEAN_PREFIX}"
        )

    ordered = sorted(clean_records, key=lambda record: record.sample_index)
    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(len(ordered))

    fit_records = [ordered[i] for i in permutation[:FIT_CLEAN_COUNT]]
    calibration_records = [
        ordered[i]
        for i in permutation[FIT_CLEAN_COUNT : FIT_CLEAN_COUNT + CALIBRATION_CLEAN_COUNT]
    ]
    eval_records = [ordered[i] for i in permutation[-EVAL_CLEAN_COUNT:]]
    return {
        "fit_clean": sorted(fit_records, key=lambda record: record.sample_index),
        "calibration_clean": sorted(calibration_records, key=lambda record: record.sample_index),
        "eval_clean": sorted(eval_records, key=lambda record: record.sample_index),
    }


def records_to_matrix(features: np.ndarray, records: list[SampleRecord]) -> np.ndarray:
    row_indices = np.asarray([record.row_index for record in records], dtype=np.int64)
    return np.asarray(features[row_indices], dtype=np.float64)


def fit_clean_reference_model(
    *,
    clean_fit_matrix: np.ndarray,
    max_pca_components: int,
    random_state: int,
) -> tuple[StandardScaler, PCA, EmpiricalCovariance]:
    scaler = StandardScaler().fit(clean_fit_matrix)
    clean_fit_scaled = scaler.transform(clean_fit_matrix)
    n_components = min(max_pca_components, clean_fit_scaled.shape[0] - 1, clean_fit_scaled.shape[1])
    if n_components <= 0:
        raise ValueError(f"Resolved PCA component count must be positive, got {n_components}")

    pca = PCA(n_components=n_components, random_state=random_state)
    clean_fit_latent = pca.fit_transform(clean_fit_scaled)
    covariance = EmpiricalCovariance().fit(clean_fit_latent)
    return scaler, pca, covariance


def score_samples(
    matrix: np.ndarray,
    *,
    scaler: StandardScaler,
    pca: PCA,
    covariance: EmpiricalCovariance,
) -> ScoreOutputs:
    scaled = scaler.transform(matrix)
    latent = pca.transform(scaled)
    reconstructed = pca.inverse_transform(latent)
    reconstruction = np.sum((scaled - reconstructed) ** 2, axis=1, dtype=np.float64)
    mahalanobis = np.asarray(covariance.mahalanobis(latent), dtype=np.float64)
    return ScoreOutputs(reconstruction=reconstruction, mahalanobis=mahalanobis)


def save_eval_score_csv(
    output_path: Path,
    *,
    model_names: list[str],
    attack_names: list[str],
    labels: list[int],
    splits: list[str],
    reconstruction_scores: np.ndarray,
    mahalanobis_scores: np.ndarray,
) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "model_name",
                "attack_name",
                "label",
                "split",
                "reconstruction_score",
                "mahalanobis_score",
            ],
        )
        writer.writeheader()
        for i, (model_name, attack_name, label, split, reconstruction, mahalanobis) in enumerate(
            zip(
                model_names,
                attack_names,
                labels,
                splits,
                reconstruction_scores.tolist(),
                mahalanobis_scores.tolist(),
            )
        ):
            writer.writerow(
                {
                    "index": int(i),
                    "model_name": str(model_name),
                    "attack_name": str(attack_name),
                    "label": int(label),
                    "split": str(split),
                    "reconstruction_score": float(reconstruction),
                    "mahalanobis_score": float(mahalanobis),
                }
            )


def render_histogram_plot(
    output_path: Path,
    *,
    clean_split_scores: dict[str, np.ndarray],
    per_attack_scores: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for split_name, scores in clean_split_scores.items():
        axes[0].hist(
            scores,
            bins=20,
            alpha=0.55,
            label=split_name.replace("_", " "),
            color=HISTOGRAM_COLORS[split_name],
        )
    axes[0].set_title("Clean Split Reconstruction Scores")
    axes[0].set_xlabel("Reconstruction Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(
        clean_split_scores["eval_clean"],
        bins=20,
        alpha=0.45,
        label="eval clean",
        color=HISTOGRAM_COLORS["eval_clean"],
    )
    for attack_name, scores in per_attack_scores.items():
        axes[1].hist(
            scores,
            bins=20,
            alpha=0.45,
            label=attack_name,
            color=HISTOGRAM_COLORS[attack_name],
        )
    axes[1].set_title("Eval Clean vs IMDb Backdoor Attacks")
    axes[1].set_xlabel("Reconstruction Score")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_boxplot_plot(
    output_path: Path,
    *,
    eval_clean_scores: np.ndarray,
    per_attack_scores: dict[str, np.ndarray],
) -> None:
    ordered_labels = ["eval_clean", "insertsent", "ripple", "stybkd", "syntactic"]
    data = [eval_clean_scores] + [per_attack_scores[label] for label in ordered_labels[1:]]
    colors = [
        HISTOGRAM_COLORS["eval_clean"],
        HISTOGRAM_COLORS["insertsent"],
        HISTOGRAM_COLORS["ripple"],
        HISTOGRAM_COLORS["stybkd"],
        HISTOGRAM_COLORS["syntactic"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    boxplot = ax.boxplot(data, patch_artist=True, tick_labels=ordered_labels)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("IMDb Eval Reconstruction Scores by Attack")
    ax.set_ylabel("Reconstruction Score")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def select_row_by_percentile(rows: list[dict[str, Any]], percentile: float) -> dict[str, Any]:
    for row in rows:
        if abs(float(row["percentile_from_calibration_clean"]) - float(percentile)) < 1e-9:
            return dict(row)
    raise KeyError(f"Could not find percentile row for {percentile}")


def counter_dict(records: list[SampleRecord]) -> dict[str, int]:
    counts = Counter(record.attack_name for record in records)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def compute_calibration_threshold_rows(
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
    percentiles: list[float],
    eval_labels: np.ndarray,
) -> list[dict[str, Any]]:
    raw_rows = compute_infer_threshold_rows(
        calibration_scores,
        eval_scores,
        percentiles,
        eval_labels,
    )
    remapped_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        remapped = dict(row)
        remapped["percentile_from_calibration_clean"] = float(remapped.pop("percentile_from_train"))
        remapped_rows.append(remapped)
    return remapped_rows


def resolve_default_percentile(percentiles: list[float]) -> float:
    if not percentiles:
        raise ValueError("threshold percentiles must not be empty")
    for candidate in percentiles:
        if abs(float(candidate) - 95.0) < 1e-9:
            return float(candidate)
    return float(percentiles[0])


def summarize_numeric_series(values: list[float | None]) -> dict[str, float | int | None]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "max": None,
        }
    array = np.asarray(filtered, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "max": float(np.max(array)),
    }


def save_rows_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot save empty row set to {output_path}")

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_ready(row))


def build_metric_summary(
    rows: list[dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, dict[str, float | int | None]]:
    return {
        metric_key: summarize_numeric_series([row.get(metric_key) for row in rows])
        for metric_key in metric_keys
    }


def render_repeated_split_metrics_plot(
    output_path: Path,
    *,
    overall_rows: list[dict[str, Any]],
    per_attack_rows: list[dict[str, Any]],
) -> None:
    grouped_by_attack: dict[str, list[dict[str, Any]]] = {}
    for row in per_attack_rows:
        grouped_by_attack.setdefault(str(row["attack_name"]), []).append(row)

    auroc_labels = ["overall", "insertsent", "ripple", "stybkd", "syntactic"]
    auroc_data = [[float(row["reconstruction_auroc"]) for row in overall_rows]]
    auroc_data.extend(
        [[float(row["reconstruction_auroc"]) for row in grouped_by_attack.get(attack_name, [])] for attack_name in auroc_labels[1:]]
    )

    recall_labels = ["overall", "insertsent", "ripple", "stybkd", "syntactic"]
    recall_data = [[float(row["reconstruction_default_recall"]) for row in overall_rows]]
    recall_data.extend(
        [
            [float(row["reconstruction_default_recall"]) for row in grouped_by_attack.get(attack_name, [])]
            for attack_name in recall_labels[1:]
        ]
    )

    colors = [
        "#4c78a8",
        HISTOGRAM_COLORS["insertsent"],
        HISTOGRAM_COLORS["ripple"],
        HISTOGRAM_COLORS["stybkd"],
        HISTOGRAM_COLORS["syntactic"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    auroc_boxplot = axes[0].boxplot(auroc_data, patch_artist=True, tick_labels=auroc_labels)
    for patch, color in zip(auroc_boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Repeated-Split Reconstruction AUROC")
    axes[0].set_ylabel("AUROC")

    recall_boxplot = axes[1].boxplot(recall_data, patch_artist=True, tick_labels=recall_labels)
    for patch, color in zip(recall_boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Repeated-Split Recall at Default Threshold")
    axes[1].set_ylabel("Recall")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_single_split(
    *,
    bundle: FeatureBundle,
    clean_records: list[SampleRecord],
    attack_records: dict[str, list[SampleRecord]],
    split_seed: int,
    max_pca_components: int,
    threshold_percentiles: list[float],
) -> dict[str, Any]:
    clean_splits = split_clean_records(clean_records, split_seed)
    fit_records = clean_splits["fit_clean"]
    calibration_records = clean_splits["calibration_clean"]
    eval_clean_records = clean_splits["eval_clean"]
    eval_backdoor_records = [
        record
        for attack_name in ["insertsent", "ripple", "stybkd", "syntactic"]
        for record in attack_records[attack_name]
    ]

    fit_matrix = records_to_matrix(bundle.features, fit_records)
    calibration_matrix = records_to_matrix(bundle.features, calibration_records)
    eval_clean_matrix = records_to_matrix(bundle.features, eval_clean_records)
    eval_backdoor_matrix = records_to_matrix(bundle.features, eval_backdoor_records)
    scaler, pca, covariance = fit_clean_reference_model(
        clean_fit_matrix=fit_matrix,
        max_pca_components=max_pca_components,
        random_state=split_seed,
    )

    fit_scores = score_samples(fit_matrix, scaler=scaler, pca=pca, covariance=covariance)
    calibration_scores = score_samples(
        calibration_matrix,
        scaler=scaler,
        pca=pca,
        covariance=covariance,
    )
    eval_clean_scores = score_samples(
        eval_clean_matrix,
        scaler=scaler,
        pca=pca,
        covariance=covariance,
    )
    eval_backdoor_scores = score_samples(
        eval_backdoor_matrix,
        scaler=scaler,
        pca=pca,
        covariance=covariance,
    )

    eval_records = eval_clean_records + eval_backdoor_records
    eval_labels = np.asarray([0] * len(eval_clean_records) + [1] * len(eval_backdoor_records), dtype=np.int32)
    eval_reconstruction = np.concatenate(
        [eval_clean_scores.reconstruction, eval_backdoor_scores.reconstruction],
        axis=0,
    )
    eval_mahalanobis = np.concatenate(
        [eval_clean_scores.mahalanobis, eval_backdoor_scores.mahalanobis],
        axis=0,
    )

    threshold_rows = compute_calibration_threshold_rows(
        calibration_scores.reconstruction,
        eval_reconstruction,
        threshold_percentiles,
        eval_labels,
    )
    default_percentile = resolve_default_percentile(threshold_percentiles)
    default_threshold_row = select_row_by_percentile(threshold_rows, default_percentile)

    overall_metrics = {
        "reconstruction": compute_offline_metrics(eval_labels, eval_reconstruction),
        "mahalanobis": compute_offline_metrics(eval_labels, eval_mahalanobis),
    }

    per_attack_metrics: dict[str, Any] = {}
    per_attack_scores_for_plot: dict[str, np.ndarray] = {}
    attack_offset = 0
    for attack_name in ["insertsent", "ripple", "stybkd", "syntactic"]:
        attack_size = len(attack_records[attack_name])
        attack_slice = slice(attack_offset, attack_offset + attack_size)
        attack_offset += attack_size

        attack_reconstruction = eval_backdoor_scores.reconstruction[attack_slice]
        attack_mahalanobis = eval_backdoor_scores.mahalanobis[attack_slice]
        per_attack_scores_for_plot[attack_name] = attack_reconstruction

        attack_labels = np.asarray([0] * len(eval_clean_records) + [1] * attack_size, dtype=np.int32)
        attack_eval_reconstruction = np.concatenate(
            [eval_clean_scores.reconstruction, attack_reconstruction],
            axis=0,
        )
        attack_eval_mahalanobis = np.concatenate(
            [eval_clean_scores.mahalanobis, attack_mahalanobis],
            axis=0,
        )

        per_attack_metrics[attack_name] = {
            "sample_counts": {
                "eval_clean": int(len(eval_clean_records)),
                "backdoor": int(attack_size),
            },
            "reconstruction": {
                "offline_metrics": compute_offline_metrics(attack_labels, attack_eval_reconstruction),
                "score_summary": {
                    "eval_clean": summarize_scores(eval_clean_scores.reconstruction),
                    "backdoor": summarize_scores(attack_reconstruction),
                },
                "threshold_rows": compute_calibration_threshold_rows(
                    calibration_scores.reconstruction,
                    attack_eval_reconstruction,
                    threshold_percentiles,
                    attack_labels,
                ),
            },
            "mahalanobis": {
                "offline_metrics": compute_offline_metrics(attack_labels, attack_eval_mahalanobis),
                "score_summary": {
                    "eval_clean": summarize_scores(eval_clean_scores.mahalanobis),
                    "backdoor": summarize_scores(attack_mahalanobis),
                },
                "threshold_rows": compute_calibration_threshold_rows(
                    calibration_scores.mahalanobis,
                    attack_eval_mahalanobis,
                    threshold_percentiles,
                    attack_labels,
                ),
            },
        }

    return {
        "split_seed": int(split_seed),
        "fit_records": fit_records,
        "calibration_records": calibration_records,
        "eval_clean_records": eval_clean_records,
        "eval_backdoor_records": eval_backdoor_records,
        "eval_records": eval_records,
        "scaler": scaler,
        "pca": pca,
        "covariance": covariance,
        "fit_scores": fit_scores,
        "calibration_scores": calibration_scores,
        "eval_clean_scores": eval_clean_scores,
        "eval_backdoor_scores": eval_backdoor_scores,
        "eval_labels": eval_labels,
        "eval_reconstruction": eval_reconstruction,
        "eval_mahalanobis": eval_mahalanobis,
        "threshold_rows": threshold_rows,
        "default_threshold_percentile": float(default_percentile),
        "default_threshold_row": default_threshold_row,
        "overall_metrics": overall_metrics,
        "per_attack_metrics": per_attack_metrics,
        "per_attack_scores_for_plot": per_attack_scores_for_plot,
        "split_counts": {
            "fit_clean": int(len(fit_records)),
            "calibration_clean": int(len(calibration_records)),
            "eval_clean": int(len(eval_clean_records)),
            "eval_backdoor_total": int(len(eval_backdoor_records)),
            "eval_rows_total": int(len(eval_records)),
        },
        "split_attack_counts": {
            "fit_clean": counter_dict(fit_records),
            "calibration_clean": counter_dict(calibration_records),
            "eval_clean": counter_dict(eval_clean_records),
            "eval_backdoors": counter_dict(eval_backdoor_records),
        },
        "score_summaries": {
            "reconstruction": {
                "fit_clean": summarize_scores(fit_scores.reconstruction),
                "calibration_clean": summarize_scores(calibration_scores.reconstruction),
                "eval_clean": summarize_scores(eval_clean_scores.reconstruction),
                "eval_backdoors": summarize_scores(eval_backdoor_scores.reconstruction),
            },
            "mahalanobis": {
                "fit_clean": summarize_scores(fit_scores.mahalanobis),
                "calibration_clean": summarize_scores(calibration_scores.mahalanobis),
                "eval_clean": summarize_scores(eval_clean_scores.mahalanobis),
                "eval_backdoors": summarize_scores(eval_backdoor_scores.mahalanobis),
            },
        },
        "pca_summary": {
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        },
    }


def build_overall_split_row(split_index: int, split_result: dict[str, Any]) -> dict[str, Any]:
    reconstruction_metrics = split_result["overall_metrics"]["reconstruction"]
    mahalanobis_metrics = split_result["overall_metrics"]["mahalanobis"]
    default_threshold_row = split_result["default_threshold_row"]
    return {
        "split_index": int(split_index),
        "split_seed": int(split_result["split_seed"]),
        "reconstruction_auroc": reconstruction_metrics["auroc"],
        "reconstruction_auprc": reconstruction_metrics["auprc"],
        "reconstruction_precision_at_num_positives": reconstruction_metrics["precision_at_num_positives"],
        "reconstruction_default_threshold": default_threshold_row["threshold"],
        "reconstruction_default_precision": default_threshold_row.get("precision"),
        "reconstruction_default_recall": default_threshold_row.get("recall"),
        "reconstruction_default_false_positive_rate": default_threshold_row.get("false_positive_rate"),
        "mahalanobis_auroc": mahalanobis_metrics["auroc"],
        "mahalanobis_auprc": mahalanobis_metrics["auprc"],
    }


def build_per_attack_split_rows(split_index: int, split_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack_name, payload in split_result["per_attack_metrics"].items():
        reconstruction_metrics = payload["reconstruction"]["offline_metrics"]
        mahalanobis_metrics = payload["mahalanobis"]["offline_metrics"]
        default_threshold_row = select_row_by_percentile(
            payload["reconstruction"]["threshold_rows"],
            split_result["default_threshold_percentile"],
        )
        rows.append(
            {
                "split_index": int(split_index),
                "split_seed": int(split_result["split_seed"]),
                "attack_name": str(attack_name),
                "reconstruction_auroc": reconstruction_metrics["auroc"],
                "reconstruction_auprc": reconstruction_metrics["auprc"],
                "reconstruction_default_precision": default_threshold_row.get("precision"),
                "reconstruction_default_recall": default_threshold_row.get("recall"),
                "reconstruction_default_false_positive_rate": default_threshold_row.get("false_positive_rate"),
                "mahalanobis_auroc": mahalanobis_metrics["auroc"],
                "mahalanobis_auprc": mahalanobis_metrics["auprc"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    overall_start = perf_counter()
    load_start = perf_counter()
    bundle = load_feature_bundle(args.feature_file)
    imdb_records = collect_imdb_records(bundle)
    load_seconds = perf_counter() - load_start

    grouped: dict[tuple[str, int], list[SampleRecord]] = {}
    for record in imdb_records:
        grouped.setdefault((record.prefix, record.label), []).append(record)

    clean_records = grouped.get((CLEAN_PREFIX, 0), [])
    if len(clean_records) != 250:
        raise ValueError(
            f"Expected 250 clean records for {CLEAN_PREFIX} label 0, found {len(clean_records)}"
        )

    attack_records: dict[str, list[SampleRecord]] = {}
    for attack_name, prefix in ATTACK_PREFIXES.items():
        records = grouped.get((prefix, 1), [])
        if len(records) != 250:
            raise ValueError(
                f"Expected 250 backdoor records for {prefix} label 1, found {len(records)}"
            )
        attack_records[attack_name] = sorted(records, key=lambda record: record.sample_index)

    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=Path(args.output_root),
        run_id=args.run_id,
    )

    if int(args.num_splits) <= 0:
        raise ValueError(f"num_splits must be positive, got {args.num_splits}")

    split_seeds = [int(args.random_state) + i for i in range(int(args.num_splits))]
    evaluation_start = perf_counter()
    representative_result: dict[str, Any] | None = None
    overall_split_rows: list[dict[str, Any]] = []
    per_attack_split_rows: list[dict[str, Any]] = []
    for split_index, split_seed in enumerate(split_seeds):
        split_result = run_single_split(
            bundle=bundle,
            clean_records=clean_records,
            attack_records=attack_records,
            split_seed=split_seed,
            max_pca_components=int(args.max_pca_components),
            threshold_percentiles=list(args.threshold_percentiles),
        )
        if representative_result is None:
            representative_result = split_result
        overall_split_rows.append(build_overall_split_row(split_index, split_result))
        per_attack_split_rows.extend(build_per_attack_split_rows(split_index, split_result))
    evaluation_seconds = perf_counter() - evaluation_start

    if representative_result is None:
        raise RuntimeError("No split results were generated")

    fit_records = representative_result["fit_records"]
    calibration_records = representative_result["calibration_records"]
    eval_clean_records = representative_result["eval_clean_records"]
    eval_backdoor_records = representative_result["eval_backdoor_records"]
    eval_records = representative_result["eval_records"]
    scaler = representative_result["scaler"]
    pca = representative_result["pca"]
    covariance = representative_result["covariance"]
    fit_scores = representative_result["fit_scores"]
    calibration_scores = representative_result["calibration_scores"]
    eval_clean_scores = representative_result["eval_clean_scores"]
    eval_backdoor_scores = representative_result["eval_backdoor_scores"]
    eval_reconstruction = representative_result["eval_reconstruction"]
    eval_mahalanobis = representative_result["eval_mahalanobis"]
    threshold_rows = representative_result["threshold_rows"]
    default_threshold_row = representative_result["default_threshold_row"]
    overall_metrics = representative_result["overall_metrics"]
    per_attack_metrics = representative_result["per_attack_metrics"]
    per_attack_scores_for_plot = representative_result["per_attack_scores_for_plot"]

    grouped_per_attack_rows: dict[str, list[dict[str, Any]]] = {}
    for row in per_attack_split_rows:
        grouped_per_attack_rows.setdefault(str(row["attack_name"]), []).append(row)

    repeated_split_summary = {
        "num_splits": int(len(split_seeds)),
        "split_seeds": split_seeds,
        "representative_split_index": 0,
        "representative_split_seed": int(representative_result["split_seed"]),
        "default_threshold_percentile": float(representative_result["default_threshold_percentile"]),
        "overall": build_metric_summary(overall_split_rows, OVERALL_SPLIT_METRIC_KEYS),
        "per_attack": {
            attack_name: build_metric_summary(rows, PER_ATTACK_SPLIT_METRIC_KEYS)
            for attack_name, rows in sorted(grouped_per_attack_rows.items())
        },
    }

    report = {
        "script_version": SCRIPT_VERSION,
        "feature_bundle": {
            "feature_file": bundle.feature_file,
            "model_names_file": bundle.model_names_file,
            "labels_file": bundle.labels_file,
            "shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        },
        "selection": {
            "imdb_prefix": IMDB_PREFIX,
            "clean_prefix": CLEAN_PREFIX,
            "attack_prefixes": ATTACK_PREFIXES,
        },
        "repeated_split_evaluation": repeated_split_summary,
        "sample_counts": {
            "clean_pool": int(len(clean_records)),
            "attack_backdoors": {attack_name: int(len(records)) for attack_name, records in attack_records.items()},
            "all_imdb_records": int(len(imdb_records)),
            "imdb_label_counts": {
                str(label): int(sum(record.label == label for record in imdb_records))
                for label in [0, 1]
            },
        },
        "representative_split_seed": int(representative_result["split_seed"]),
        "split_counts": representative_result["split_counts"],
        "split_attack_counts": representative_result["split_attack_counts"],
        "pca": representative_result["pca_summary"],
        "score_summaries": representative_result["score_summaries"],
        "threshold_rows": threshold_rows,
        "default_threshold": {
            "percentile": float(representative_result["default_threshold_percentile"]),
            "reconstruction": default_threshold_row,
        },
        "overall_metrics": overall_metrics,
        "per_attack_metrics": per_attack_metrics,
    }

    threshold_rows_path = ctx.reports_dir / "threshold_rows.json"
    per_attack_metrics_path = ctx.reports_dir / "per_attack_metrics.json"
    report_path = ctx.reports_dir / "imdb_clean_reference_report.json"
    repeated_split_summary_path = ctx.reports_dir / "repeated_split_summary.json"
    repeated_split_overall_path = ctx.reports_dir / "repeated_split_overall_metrics.csv"
    repeated_split_per_attack_path = ctx.reports_dir / "repeated_split_per_attack_metrics.csv"
    scores_eval_path = ctx.reports_dir / "scores_eval.csv"
    histogram_path = ctx.plots_dir / "score_histogram_by_attack.png"
    boxplot_path = ctx.plots_dir / "score_boxplot_by_attack.png"
    repeated_split_plot_path = ctx.plots_dir / "repeated_split_reconstruction_metrics.png"
    model_path = ctx.models_dir / "clean_reference_model.npz"

    save_json(threshold_rows_path, {"reconstruction": threshold_rows})
    save_json(per_attack_metrics_path, per_attack_metrics)
    save_json(repeated_split_summary_path, repeated_split_summary)
    save_json(report_path, report)
    save_rows_csv(repeated_split_overall_path, overall_split_rows)
    save_rows_csv(repeated_split_per_attack_path, per_attack_split_rows)

    save_eval_score_csv(
        scores_eval_path,
        model_names=[record.model_name for record in eval_records],
        attack_names=[
            "clean" if record.label == 0 else record.attack_name
            for record in eval_records
        ],
        labels=[record.label for record in eval_records],
        splits=[
            "eval_clean" if record.label == 0 else f"eval_{record.attack_name}"
            for record in eval_records
        ],
        reconstruction_scores=eval_reconstruction,
        mahalanobis_scores=eval_mahalanobis,
    )

    render_histogram_plot(
        histogram_path,
        clean_split_scores={
            "fit_clean": fit_scores.reconstruction,
            "calibration_clean": calibration_scores.reconstruction,
            "eval_clean": eval_clean_scores.reconstruction,
        },
        per_attack_scores=per_attack_scores_for_plot,
    )
    render_boxplot_plot(
        boxplot_path,
        eval_clean_scores=eval_clean_scores.reconstruction,
        per_attack_scores=per_attack_scores_for_plot,
    )
    render_repeated_split_metrics_plot(
        repeated_split_plot_path,
        overall_rows=overall_split_rows,
        per_attack_rows=per_attack_split_rows,
    )

    np.savez(
        model_path,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        pca_explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        pca_explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        covariance_location=np.asarray(covariance.location_, dtype=np.float64),
        covariance_covariance=np.asarray(covariance.covariance_, dtype=np.float64),
        covariance_precision=np.asarray(covariance.precision_, dtype=np.float64),
    )

    ctx.add_artifact("report", report_path)
    ctx.add_artifact("threshold_rows", threshold_rows_path)
    ctx.add_artifact("per_attack_metrics", per_attack_metrics_path)
    ctx.add_artifact("repeated_split_summary", repeated_split_summary_path)
    ctx.add_artifact("repeated_split_overall_metrics", repeated_split_overall_path)
    ctx.add_artifact("repeated_split_per_attack_metrics", repeated_split_per_attack_path)
    ctx.add_artifact("scores_eval", scores_eval_path)
    ctx.add_artifact("score_histogram", histogram_path)
    ctx.add_artifact("score_boxplot", boxplot_path)
    ctx.add_artifact("repeated_split_reconstruction_metrics", repeated_split_plot_path)
    ctx.add_artifact("clean_reference_model", model_path)

    total_seconds = perf_counter() - overall_start
    ctx.add_timing("load_bundle_seconds", load_seconds)
    ctx.add_timing("evaluate_splits_seconds", evaluation_seconds)
    ctx.add_timing("total_seconds", total_seconds)
    ctx.finalize(
        run_config={
            "script_version": SCRIPT_VERSION,
            "feature_file": bundle.feature_file,
            "output_root": Path(args.output_root),
            "run_id": ctx.run_id,
            "random_state": int(args.random_state),
            "max_pca_components": int(args.max_pca_components),
            "threshold_percentiles": [float(x) for x in args.threshold_percentiles],
            "num_splits": int(args.num_splits),
            "split_seeds": split_seeds,
            "clean_split_counts": {
                "fit_clean": FIT_CLEAN_COUNT,
                "calibration_clean": CALIBRATION_CLEAN_COUNT,
                "eval_clean": EVAL_CLEAN_COUNT,
            },
        }
    )


if __name__ == "__main__":
    main()
