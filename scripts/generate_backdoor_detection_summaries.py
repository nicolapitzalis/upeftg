#!/usr/bin/env python3
"""Maintenance script for aggregating completed supervised run outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    attack_type: str
    run_id: str


@dataclass(frozen=True)
class SummarySuite:
    key: str
    description: str
    run_specs: tuple[RunSpec, ...]
    summary_filename_template: str
    winner_summary_filename_template: str
    universal_filename: str
    required: bool = False


RUN_SPECS = [
    RunSpec("squad", "insertsent", "llama2_7b_squad_insertsent"),
    RunSpec("toxic backdoors alpaca", "word", "llama2_7b_toxic_backdoors_alpaca"),
    RunSpec("toxic backdoors hard", "sentence", "llama2_7b_toxic_backdoors_hard"),
    RunSpec("ag news", "ripple", "llama2_7b_ag_news_ripple"),
    RunSpec("ag news", "insertsent", "llama2_7b_ag_news_insertsent"),
    RunSpec("ag news", "syntactic", "llama2_7b_ag_news_syntactic"),
    RunSpec("ag news", "stybkd", "llama2_7b_ag_news_stybkd"),
    RunSpec("imdb", "ripple", "llama2_7b_imdb_ripple"),
    RunSpec("imdb", "insertsent", "llama2_7b_imdb_insertsent"),
    RunSpec("imdb", "syntactic", "llama2_7b_imdb_syntactic"),
    RunSpec("imdb", "stybkd", "llama2_7b_imdb_stybkd"),
]

LLAMA2_7B_TBH_RANK_RUN_SPECS = [
    RunSpec("llama2_7b tbh", "rank 8", "llama2_7b_tbh_rank8"),
    RunSpec("llama2_7b tbh", "rank 16", "llama2_7b_tbh_rank16"),
    RunSpec("llama2_7b tbh", "rank 32", "llama2_7b_tbh_rank32"),
    RunSpec("llama2_7b tbh", "rank 64", "llama2_7b_tbh_rank64"),
    RunSpec("llama2_7b tbh", "rank 128", "llama2_7b_tbh_rank128"),
    RunSpec("llama2_7b tbh", "rank 256", "llama2_7b_tbh_rank256"),
    RunSpec("llama2_7b tbh", "rank 512", "llama2_7b_tbh_rank512"),
    RunSpec("llama2_7b tbh", "rank 1024", "llama2_7b_tbh_rank1024"),
    RunSpec("llama2_7b tbh", "rank 2048", "llama2_7b_tbh_rank2048"),
]

LLAMA2_7B_TBH_PEFT_METHOD_RUN_SPECS = [
    RunSpec("llama2_7b tbh", "adalora", "llama2_7b_adalora_tbh"),
    RunSpec("llama2_7b tbh", "dora", "llama2_7b_dora_tbh"),
    RunSpec("llama2_7b tbh", "qlora", "llama2_7b_qlora_tbh"),
    RunSpec("llama2_7b tbh", "lora+", "llama2_7b_lora_plus_tbh"),
    RunSpec("llama2_7b tbh", "lora rank 256", "llama2_7b_tbh_rank256"),
]

LLAMA2_7B_TBH_ZERO_SHOT_RANK_RUN_SPECS = [
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 8", "llama2_7b_tbh_zero_shot_r256_to_rank8"),
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 16", "llama2_7b_tbh_zero_shot_r256_to_rank16"),
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 32", "llama2_7b_tbh_zero_shot_r256_to_rank32"),
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 64", "llama2_7b_tbh_zero_shot_r256_to_rank64"),
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 128", "llama2_7b_tbh_zero_shot_r256_to_rank128"),
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 512", "llama2_7b_tbh_zero_shot_r256_to_rank512"),
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 1024", "llama2_7b_tbh_zero_shot_r256_to_rank1024"),
    RunSpec("llama2_7b tbh zero-shot from rank 256", "rank 2048", "llama2_7b_tbh_zero_shot_r256_to_rank2048"),
]

LLAMA2_7B_TBH_ZERO_SHOT_ADAPTER_RUN_SPECS = [
    RunSpec("llama2_7b tbh zero-shot from lora", "dora", "llama2_7b_tbh_zero_shot_lora_to_dora"),
    RunSpec(
        "llama2_7b tbh zero-shot from lora",
        "lora+",
        "llama2_7b_tbh_zero_shot_lora_to_lora_plus",
    ),
    RunSpec("llama2_7b tbh zero-shot from lora", "qlora", "llama2_7b_tbh_zero_shot_lora_to_qlora"),
    RunSpec(
        "llama2_7b tbh zero-shot from lora",
        "adalora",
        "llama2_7b_tbh_zero_shot_lora_to_adalora",
    ),
]

LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_INSERTSENT_RUN_SPECS = [
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from insertsent",
        "ripple",
        "llama2_7b_ag_news_imdb_zero_shot_insertsent_to_ripple",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from insertsent",
        "stybkd",
        "llama2_7b_ag_news_imdb_zero_shot_insertsent_to_stybkd",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from insertsent",
        "syntactic",
        "llama2_7b_ag_news_imdb_zero_shot_insertsent_to_syntactic",
    ),
]

LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_RIPPLE_RUN_SPECS = [
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from ripple",
        "insertsent",
        "llama2_7b_ag_news_imdb_zero_shot_ripple_to_insertsent",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from ripple",
        "stybkd",
        "llama2_7b_ag_news_imdb_zero_shot_ripple_to_stybkd",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from ripple",
        "syntactic",
        "llama2_7b_ag_news_imdb_zero_shot_ripple_to_syntactic",
    ),
]

LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_STYBKD_RUN_SPECS = [
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from stybkd",
        "insertsent",
        "llama2_7b_ag_news_imdb_zero_shot_stybkd_to_insertsent",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from stybkd",
        "ripple",
        "llama2_7b_ag_news_imdb_zero_shot_stybkd_to_ripple",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from stybkd",
        "syntactic",
        "llama2_7b_ag_news_imdb_zero_shot_stybkd_to_syntactic",
    ),
]

LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_SYNTACTIC_RUN_SPECS = [
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from syntactic",
        "insertsent",
        "llama2_7b_ag_news_imdb_zero_shot_syntactic_to_insertsent",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from syntactic",
        "ripple",
        "llama2_7b_ag_news_imdb_zero_shot_syntactic_to_ripple",
    ),
    RunSpec(
        "llama2_7b ag_news/imdb zero-shot from syntactic",
        "stybkd",
        "llama2_7b_ag_news_imdb_zero_shot_syntactic_to_stybkd",
    ),
]

ARCHITECTURE_RUN_SPECS = [
    RunSpec("toxic backdoors hard rank 256", "flan_t5_xl", "flan_t5_xl_architecture_tbh"),
    RunSpec("toxic backdoors hard rank 256", "llama2_13b", "llama2_13b_architecture_tbh"),
    RunSpec("toxic backdoors hard rank 256", "llama3_8b", "llama3_8b_architecture_tbh"),
    RunSpec("toxic backdoors hard rank 256", "qwen1.5_7b", "qwen1.5_7b_architecture_tbh"),
    RunSpec("imdb insertsent rank 16", "roberta_base", "roberta_base_architecture"),
    RunSpec("toxic backdoors hard rank 256", "llama2_7b", "llama2_7b_tbh_rank256"),
]

SUMMARY_SUITES = (
    SummarySuite(
        key="backdoor_detection",
        description="baseline supervised backdoor detection runs",
        run_specs=tuple(RUN_SPECS),
        summary_filename_template="backdoor_detection_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template="backdoor_detection_winner_summary_fpr_{fpr_slug}.csv",
        universal_filename="universal_config_ranking_cv.csv",
        required=True,
    ),
    SummarySuite(
        key="llama2_7b_tbh_ranks",
        description="llama2_7b toxic backdoors hard rank sweep",
        run_specs=tuple(LLAMA2_7B_TBH_RANK_RUN_SPECS),
        summary_filename_template="backdoor_detection_tbh_rank_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template="backdoor_detection_tbh_rank_winner_summary_fpr_{fpr_slug}.csv",
        universal_filename="universal_config_tbh_rank_ranking_cv.csv",
    ),
    SummarySuite(
        key="llama2_7b_tbh_peft_methods",
        description="llama2_7b toxic backdoors hard PEFT method comparison",
        run_specs=tuple(LLAMA2_7B_TBH_PEFT_METHOD_RUN_SPECS),
        summary_filename_template="backdoor_detection_tbh_peft_method_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template="backdoor_detection_tbh_peft_method_winner_summary_fpr_{fpr_slug}.csv",
        universal_filename="universal_config_tbh_peft_method_ranking_cv.csv",
    ),
    SummarySuite(
        key="llama2_7b_tbh_zero_shot_ranks",
        description="llama2_7b toxic backdoors hard zero-shot rank sweep from rank 256",
        run_specs=tuple(LLAMA2_7B_TBH_ZERO_SHOT_RANK_RUN_SPECS),
        summary_filename_template="backdoor_detection_tbh_zero_shot_rank_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template="backdoor_detection_tbh_zero_shot_rank_winner_summary_fpr_{fpr_slug}.csv",
        universal_filename="universal_config_tbh_zero_shot_rank_ranking_cv.csv",
    ),
    SummarySuite(
        key="llama2_7b_tbh_zero_shot_adapters",
        description="llama2_7b toxic backdoors hard zero-shot adapter sweep from lora",
        run_specs=tuple(LLAMA2_7B_TBH_ZERO_SHOT_ADAPTER_RUN_SPECS),
        summary_filename_template="backdoor_detection_tbh_zero_shot_adapter_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template="backdoor_detection_tbh_zero_shot_adapter_winner_summary_fpr_{fpr_slug}.csv",
        universal_filename="universal_config_tbh_zero_shot_adapter_ranking_cv.csv",
    ),
    SummarySuite(
        key="llama2_7b_ag_news_imdb_zero_shot_insertsent",
        description="llama2_7b ag_news/imdb zero-shot attack transfer sweep from insertsent",
        run_specs=tuple(LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_INSERTSENT_RUN_SPECS),
        summary_filename_template="backdoor_detection_ag_news_imdb_zero_shot_insertsent_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template=(
            "backdoor_detection_ag_news_imdb_zero_shot_insertsent_winner_summary_fpr_{fpr_slug}.csv"
        ),
        universal_filename="universal_config_ag_news_imdb_zero_shot_insertsent_ranking_cv.csv",
    ),
    SummarySuite(
        key="llama2_7b_ag_news_imdb_zero_shot_ripple",
        description="llama2_7b ag_news/imdb zero-shot attack transfer sweep from ripple",
        run_specs=tuple(LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_RIPPLE_RUN_SPECS),
        summary_filename_template="backdoor_detection_ag_news_imdb_zero_shot_ripple_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template=(
            "backdoor_detection_ag_news_imdb_zero_shot_ripple_winner_summary_fpr_{fpr_slug}.csv"
        ),
        universal_filename="universal_config_ag_news_imdb_zero_shot_ripple_ranking_cv.csv",
    ),
    SummarySuite(
        key="llama2_7b_ag_news_imdb_zero_shot_stybkd",
        description="llama2_7b ag_news/imdb zero-shot attack transfer sweep from stybkd",
        run_specs=tuple(LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_STYBKD_RUN_SPECS),
        summary_filename_template="backdoor_detection_ag_news_imdb_zero_shot_stybkd_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template=(
            "backdoor_detection_ag_news_imdb_zero_shot_stybkd_winner_summary_fpr_{fpr_slug}.csv"
        ),
        universal_filename="universal_config_ag_news_imdb_zero_shot_stybkd_ranking_cv.csv",
    ),
    SummarySuite(
        key="llama2_7b_ag_news_imdb_zero_shot_syntactic",
        description="llama2_7b ag_news/imdb zero-shot attack transfer sweep from syntactic",
        run_specs=tuple(LLAMA2_7B_AG_NEWS_IMDB_ZERO_SHOT_SYNTACTIC_RUN_SPECS),
        summary_filename_template="backdoor_detection_ag_news_imdb_zero_shot_syntactic_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template=(
            "backdoor_detection_ag_news_imdb_zero_shot_syntactic_winner_summary_fpr_{fpr_slug}.csv"
        ),
        universal_filename="universal_config_ag_news_imdb_zero_shot_syntactic_ranking_cv.csv",
    ),
    SummarySuite(
        key="architecture_comparison",
        description="architecture comparison sweep",
        run_specs=tuple(ARCHITECTURE_RUN_SPECS),
        summary_filename_template="backdoor_detection_architecture_summary_fpr_{fpr_slug}.csv",
        winner_summary_filename_template="backdoor_detection_architecture_winner_summary_fpr_{fpr_slug}.csv",
        universal_filename="universal_config_architecture_ranking_cv.csv",
    ),
)

DEFAULT_FPRS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-FPR CSV summaries for the supervised backdoor detection runs."
        )
    )
    parser.add_argument(
        "--fpr",
        dest="fprs",
        type=float,
        nargs="*",
        default=list(DEFAULT_FPRS),
        help="Accepted FPR values to export. Defaults to 0.01 0.05 0.10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the CSV files will be written. Defaults to runs/supervised.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_scalar(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def format_params(params: dict[str, Any]) -> str:
    if not params:
        return "none"
    return ", ".join(
        f"{key}={format_scalar(params[key])}" for key in sorted(params.keys())
    )


def format_note(winner: dict[str, Any]) -> str:
    params = winner.get("params") or {}
    params_text = format_params(params)
    normalization = winner.get("normalization_policy", "unknown")
    return (
        f"model={winner['model_name']}; "
        f"normalization={normalization}; "
        f"params={params_text}"
    )


def format_fpr_for_filename(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def dataset_label(spec: RunSpec) -> str:
    return f"{spec.dataset} ({spec.attack_type})"


def candidate_key(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        candidate["model_name"],
        candidate.get("normalization_policy", "unknown"),
        json.dumps(candidate.get("params", {}), sort_keys=True, default=str),
    )


def create_model(name: str, params: dict[str, Any], random_state: int) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("normalizer", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params["C"]),
                        class_weight=params["class_weight"],
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    if name == "ridge_classifier":
        return Pipeline(
            steps=[
                ("normalizer", StandardScaler()),
                (
                    "model",
                    RidgeClassifier(
                        alpha=float(params["alpha"]),
                        class_weight=params["class_weight"],
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    if name == "linear_svm":
        return Pipeline(
            steps=[
                ("normalizer", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(params["C"]),
                        class_weight=params["class_weight"],
                        kernel="linear",
                        probability=True,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    if name == "adaboost":
        return Pipeline(
            steps=[
                ("normalizer", "passthrough"),
                (
                    "model",
                    AdaBoostClassifier(
                        estimator=DecisionTreeClassifier(
                            max_depth=int(params["max_depth"]),
                            random_state=int(random_state),
                        ),
                        n_estimators=int(params["n_estimators"]),
                        learning_rate=float(params["learning_rate"]),
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    if name == "kernel_svm":
        return Pipeline(
            steps=[
                ("normalizer", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(params["C"]),
                        gamma=params["gamma"],
                        class_weight=params["class_weight"],
                        kernel="rbf",
                        probability=False,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            steps=[
                ("normalizer", "passthrough"),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=(
                            None if params["max_depth"] is None else int(params["max_depth"])
                        ),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        class_weight=params["class_weight"],
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model name: {name}")


def _threshold_candidate_values(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Threshold selection requires at least one score")
    unique_scores = np.unique(scores)
    above_max = np.nextafter(float(np.max(unique_scores)), float("inf"))
    return np.concatenate((np.asarray([above_max], dtype=np.float64), unique_scores[::-1]))


def evaluate_binary_threshold(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels_true = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(labels_true == 1))
    negatives = int(np.sum(labels_true == 0))
    flagged = scores >= float(threshold)

    tp = int(np.sum((labels_true == 1) & flagged))
    fp = int(np.sum((labels_true == 0) & flagged))
    tn = int(np.sum((labels_true == 0) & (~flagged)))
    fn = int(np.sum((labels_true == 1) & (~flagged)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / positives) if positives > 0 else 0.0
    false_positive_rate = float(fp / negatives) if negatives > 0 else 0.0
    specificity = float(tn / negatives) if negatives > 0 else 0.0
    accuracy = float((tp + tn) / max(1, labels_true.shape[0]))
    balanced_accuracy = float((recall + specificity) / 2.0)
    f1 = (
        float((2.0 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_positive": positives,
        "n_negative": negatives,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "fraction_flagged": float(np.mean(flagged)) if flagged.size > 0 else 0.0,
    }


def select_threshold_max_recall_under_fpr(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    accepted_fpr: float,
) -> dict[str, Any]:
    candidates = [
        evaluate_binary_threshold(
            labels_true=np.asarray(labels_true, dtype=np.int32),
            scores=np.asarray(scores, dtype=np.float64),
            threshold=float(threshold),
        )
        for threshold in _threshold_candidate_values(scores)
    ]
    feasible = [
        row
        for row in candidates
        if float(row["false_positive_rate"]) <= float(accepted_fpr) + 1e-12
    ]
    if not feasible:
        raise RuntimeError("No feasible threshold satisfied the accepted_fpr constraint")

    best = max(
        feasible,
        key=lambda row: (
            float(row["recall"]),
            -float(row["false_positive_rate"]),
            float(row["precision"]),
            float(row["threshold"]),
        ),
    )
    selected = dict(best)
    selected["selection_method"] = "maximize_recall_subject_to_fpr"
    selected["accepted_fpr"] = float(accepted_fpr)
    return selected


def predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=np.float64)
        if decision.ndim == 2 and decision.shape[1] >= 2:
            return decision[:, 1]
        return decision.reshape(-1)

    pred = np.asarray(model.predict(x), dtype=np.float64)
    return pred.reshape(-1)


def _unique_index_by_name(names: list[str], *, context: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for i, name in enumerate(names):
        if name in index:
            duplicates.append(name)
            continue
        index[name] = int(i)
    if duplicates:
        dup_preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"Duplicate model names in {context}; cannot align features safely. Examples: {dup_preview}"
        )
    return index


def load_features_for_tuning_manifest(
    manifest: dict[str, Any],
    *,
    external_feature_cache: dict[str, Any],
    external_model_names_cache: dict[str, list[str]],
    external_index_cache: dict[str, dict[str, int]],
) -> np.ndarray:
    data = manifest["data"]
    feature_loading_mode = str(data.get("feature_loading_mode", "materialized"))
    feature_path_value = data.get("feature_path")
    if not isinstance(feature_path_value, str) or not feature_path_value:
        raise ValueError("Tuning manifest is missing data.feature_path")

    if feature_loading_mode == "materialized":
        return np.asarray(np.load(feature_path_value), dtype=np.float32)

    if feature_loading_mode != "external_source":
        raise ValueError(f"Unsupported data.feature_loading_mode={feature_loading_mode!r}")

    model_names_path_value = data.get("model_names_path")
    if not isinstance(model_names_path_value, str) or not model_names_path_value:
        raise ValueError("Tuning manifest is missing data.model_names_path")
    with open(model_names_path_value, "r", encoding="utf-8") as f:
        expected_model_names = [str(x) for x in json.load(f)]

    extractor_metadata = manifest["extractor"]["metadata"]
    external_feature_source = extractor_metadata["external_feature_source"]
    external_model_names_source = extractor_metadata["external_model_names_source"]

    if external_feature_source not in external_feature_cache:
        external_feature_cache[external_feature_source] = np.load(
            external_feature_source,
            mmap_mode="r",
        )
    if external_model_names_source not in external_model_names_cache:
        with open(external_model_names_source, "r", encoding="utf-8") as f:
            external_model_names_cache[external_model_names_source] = [
                str(x) for x in json.load(f)
            ]
        external_index_cache[external_model_names_source] = _unique_index_by_name(
            external_model_names_cache[external_model_names_source],
            context=external_model_names_source,
        )

    external_index = external_index_cache[external_model_names_source]
    row_indices = np.asarray(
        [external_index[name] for name in expected_model_names],
        dtype=np.int64,
    )
    return np.asarray(external_feature_cache[external_feature_source][row_indices], dtype=np.float32)


def build_rows(
    repo_root: Path,
    accepted_fpr: float,
    *,
    run_specs: Sequence[RunSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = repo_root / "runs" / "supervised"

    for spec in run_specs:
        run_dir = base / spec.run_id
        report = load_json(run_dir / "reports" / "supervised_report.json")
        threshold = load_json(run_dir / "reports" / "selected_threshold.json")
        run_config = load_json(run_dir / "run_config.json")

        selection = next(
            (
                entry
                for entry in threshold["selections"]
                if abs(float(entry["accepted_fpr"]) - accepted_fpr) < 1e-12
            ),
            None,
        )
        if selection is None:
            raise ValueError(
                f"Missing accepted_fpr={accepted_fpr} in {run_dir / 'reports' / 'selected_threshold.json'}"
            )

        inference_metrics = selection["inference_metrics"]
        rows.append(
            {
                "dataset": spec.dataset,
                "type of attack": spec.attack_type,
                "backdoor detection acc": inference_metrics["recall"],
                "clean detection acc": inference_metrics["specificity"],
                "detection auc": report["fit_assessment"]["offline_metrics"]["auroc"],
                "note": format_note(run_config["winner"]),
            }
        )

    return rows


def build_universal_config_entries(
    repo_root: Path,
    *,
    run_specs: Sequence[RunSpec],
) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    base = repo_root / "runs" / "supervised"
    ordered_dataset_labels = [dataset_label(spec) for spec in run_specs]
    best_auc_by_dataset: dict[str, float] = {}

    for spec in run_specs:
        label = dataset_label(spec)
        run_dir = base / spec.run_id
        report = load_json(run_dir / "reports" / "supervised_report.json")
        candidates = report["tuning"]["candidates"]
        dataset_best_auc = max(
            float(candidate["roc_auc_mean"])
            for candidate in candidates
            if candidate.get("status") == "ok" and candidate.get("roc_auc_mean") is not None
        )
        best_auc_by_dataset[label] = dataset_best_auc

        for candidate in candidates:
            if candidate.get("status") != "ok" or candidate.get("roc_auc_mean") is None:
                continue
            key = candidate_key(candidate)
            params = candidate.get("params") or {}
            group = grouped_rows.setdefault(
                key,
                {
                    "model": candidate["model_name"],
                    "normalization": candidate.get("normalization_policy", "unknown"),
                    "hyperparameters": format_params(params),
                    "params": dict(params),
                    "scores": {},
                },
            )
            group["scores"][label] = float(candidate["roc_auc_mean"])

    complete_groups = [
        row
        for row in grouped_rows.values()
        if len(row["scores"]) == len(ordered_dataset_labels)
    ]

    score_vectors = [
        [row["scores"][label] for label in ordered_dataset_labels] for row in complete_groups
    ]

    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(complete_groups):
        scores = [row["scores"][label] for label in ordered_dataset_labels]
        mean_cv_auc = sum(scores) / len(scores)
        variance = sum((score - mean_cv_auc) ** 2 for score in scores) / len(scores)
        std_cv_auc = math.sqrt(variance)
        regrets = [best_auc_by_dataset[label] - row["scores"][label] for label in ordered_dataset_labels]
        mean_regret = sum(regrets) / len(regrets)
        max_regret = max(regrets)
        cv_win_datasets = [
            label
            for label in ordered_dataset_labels
            if abs(best_auc_by_dataset[label] - row["scores"][label]) <= 1e-12
        ]
        pairwise_distances = []
        this_vector = score_vectors[row_index]
        for other_index, other_vector in enumerate(score_vectors):
            if other_index == row_index:
                continue
            squared_distance = sum(
                (left - right) ** 2 for left, right in zip(this_vector, other_vector)
            )
            pairwise_distances.append(math.sqrt(squared_distance))
        medoid_distance = (
            sum(pairwise_distances) / len(pairwise_distances) if pairwise_distances else 0.0
        )
        rows.append(
            {
                "model": row["model"],
                "normalization": row["normalization"],
                "hyperparameters": row["hyperparameters"],
                "params": dict(row["params"]),
                "mean_cv_auc": mean_cv_auc,
                "std_cv_auc": std_cv_auc,
                "mean_regret": mean_regret,
                "max_regret": max_regret,
                "medoid_distance": medoid_distance,
                "n_dataset_cv_wins": len(cv_win_datasets),
                "cv_win_datasets": "; ".join(cv_win_datasets),
            }
        )

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["medoid_distance"],
            row["mean_regret"],
            -row["mean_cv_auc"],
            row["std_cv_auc"],
            row["model"],
            row["hyperparameters"],
        ),
    )

    ranked_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        ranked_rows.append(
            {
                "rank": rank,
                "model": row["model"],
                "normalization": row["normalization"],
                "hyperparameters": row["hyperparameters"],
                "params": dict(row["params"]),
                "mean_cv_auc": row["mean_cv_auc"],
                "std_cv_auc": row["std_cv_auc"],
                "mean_regret": row["mean_regret"],
                "max_regret": row["max_regret"],
                "medoid_distance": row["medoid_distance"],
                "n_dataset_cv_wins": row["n_dataset_cv_wins"],
                "cv_win_datasets": row["cv_win_datasets"],
            }
        )

    return ranked_rows


def build_universal_config_rows(
    repo_root: Path,
    *,
    run_specs: Sequence[RunSpec],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": row["rank"],
            "model": row["model"],
            "normalization": row["normalization"],
            "hyperparameters": row["hyperparameters"],
            "mean_cv_auc": row["mean_cv_auc"],
            "std_cv_auc": row["std_cv_auc"],
            "mean_regret": row["mean_regret"],
            "max_regret": row["max_regret"],
            "medoid_distance": row["medoid_distance"],
            "n_dataset_cv_wins": row["n_dataset_cv_wins"],
            "cv_win_datasets": row["cv_win_datasets"],
        }
        for row in build_universal_config_entries(repo_root, run_specs=run_specs)
    ]


def build_universal_top_threshold_rows(
    repo_root: Path,
    *,
    run_specs: Sequence[RunSpec],
) -> list[dict[str, Any]]:
    universal_entries = build_universal_config_entries(repo_root, run_specs=run_specs)
    if not universal_entries:
        raise ValueError("No universal config entries available")
    top_entry = universal_entries[0]

    base = repo_root / "runs" / "supervised"
    external_feature_cache: dict[str, Any] = {}
    external_model_names_cache: dict[str, list[str]] = {}
    external_index_cache: dict[str, dict[str, int]] = {}

    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        run_dir = base / spec.run_id
        manifest = load_json(run_dir / "reports" / "tuning_manifest.json")
        features = load_features_for_tuning_manifest(
            manifest,
            external_feature_cache=external_feature_cache,
            external_model_names_cache=external_model_names_cache,
            external_index_cache=external_index_cache,
        )
        labels_value = np.load(manifest["data"]["labels_value_path"]).astype(np.int32)
        labels_known = np.load(manifest["data"]["labels_known_path"]).astype(bool)
        train_indices = np.asarray(manifest["data"]["train_indices"], dtype=np.int64)
        calibration_indices = np.asarray(
            manifest["data"].get("calibration_indices", []),
            dtype=np.int64,
        )
        infer_indices = np.asarray(manifest["data"]["infer_indices"], dtype=np.int64)

        if calibration_indices.size == 0:
            raise ValueError(f"No calibration split available for {run_dir}")
        if not bool(np.all(labels_known[infer_indices])):
            raise ValueError(f"Inference labels are not fully known for {run_dir}")

        model = create_model(
            str(top_entry["model"]),
            params=dict(top_entry["params"]),
            random_state=int(manifest["tuning"]["random_state"]),
        )
        model.fit(features[train_indices], labels_value[train_indices])

        calibration_labels = labels_value[calibration_indices]
        calibration_scores = predict_scores(model, features[calibration_indices])
        infer_labels = labels_value[infer_indices]
        infer_scores = predict_scores(model, features[infer_indices])
        detection_auc = float(roc_auc_score(infer_labels, infer_scores))
        winner_note = (
            f"model={top_entry['model']}; "
            f"normalization={top_entry['normalization']}; "
            f"params={top_entry['hyperparameters']}"
        )

        accepted_fprs = manifest["threshold_selection"].get("accepted_fprs", list(DEFAULT_FPRS))
        for accepted_fpr in accepted_fprs:
            selection = select_threshold_max_recall_under_fpr(
                labels_true=calibration_labels,
                scores=calibration_scores,
                accepted_fpr=float(accepted_fpr),
            )
            inference_metrics = evaluate_binary_threshold(
                labels_true=infer_labels,
                scores=infer_scores,
                threshold=float(selection["threshold"]),
            )
            rows.append(
                {
                    "dataset": spec.dataset,
                    "type of attack": spec.attack_type,
                    "accepted_fpr": float(accepted_fpr),
                    "backdoor detection acc": float(inference_metrics["recall"]),
                    "clean detection acc": float(inference_metrics["specificity"]),
                    "detection auc": detection_auc,
                    "note": winner_note,
                }
            )

    return rows


def suite_required_paths(repo_root: Path, suite: SummarySuite) -> list[Path]:
    base = repo_root / "runs" / "supervised"
    required_rel_paths = (
        Path("reports") / "supervised_report.json",
        Path("reports") / "selected_threshold.json",
        Path("reports") / "tuning_manifest.json",
        Path("run_config.json"),
    )
    return [
        base / spec.run_id / rel_path
        for spec in suite.run_specs
        for rel_path in required_rel_paths
    ]


def suite_is_ready(repo_root: Path, suite: SummarySuite) -> bool:
    missing_paths = [path for path in suite_required_paths(repo_root, suite) if not path.exists()]
    if not missing_paths:
        return True

    preview = ", ".join(str(path) for path in missing_paths[:3])
    message = (
        f"Summary suite '{suite.key}' is missing required artifacts for {suite.description}. "
        f"Example missing path(s): {preview}"
    )
    if suite.required:
        raise FileNotFoundError(message)
    print(f"Skipping {suite.key}: {message}", file=sys.stderr)
    return False


def generate_suite_csvs(
    *,
    repo_root: Path,
    output_dir: Path,
    suite: SummarySuite,
    accepted_fprs: Sequence[float],
) -> None:
    if not suite_is_ready(repo_root, suite):
        return

    for accepted_fpr in accepted_fprs:
        rows = build_rows(
            repo_root=repo_root,
            accepted_fpr=float(accepted_fpr),
            run_specs=suite.run_specs,
        )
        filename = suite.summary_filename_template.format(
            fpr_slug=format_fpr_for_filename(float(accepted_fpr))
        )
        write_csv(output_dir / filename, rows)
        print(f"Wrote {output_dir / filename}")

    universal_rows = build_universal_config_rows(
        repo_root=repo_root,
        run_specs=suite.run_specs,
    )
    write_csv(output_dir / suite.universal_filename, universal_rows)
    print(f"Wrote {output_dir / suite.universal_filename}")

    universal_top_threshold_rows = build_universal_top_threshold_rows(
        repo_root=repo_root,
        run_specs=suite.run_specs,
    )
    for accepted_fpr in accepted_fprs:
        winner_rows = [
            {
                "dataset": row["dataset"],
                "type of attack": row["type of attack"],
                "backdoor detection acc": row["backdoor detection acc"],
                "clean detection acc": row["clean detection acc"],
                "detection auc": row["detection auc"],
                "note": row["note"],
            }
            for row in universal_top_threshold_rows
            if abs(float(row["accepted_fpr"]) - float(accepted_fpr)) < 1e-12
        ]
        winner_filename = suite.winner_summary_filename_template.format(
            fpr_slug=format_fpr_for_filename(float(accepted_fpr))
        )
        write_csv(output_dir / winner_filename, winner_rows)
        print(f"Wrote {output_dir / winner_filename}")


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    resolved_fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (repo_root / "runs" / "supervised" / "summaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite in SUMMARY_SUITES:
        generate_suite_csvs(
            repo_root=repo_root,
            output_dir=output_dir,
            suite=suite,
            accepted_fprs=args.fprs,
        )


if __name__ == "__main__":
    main()
