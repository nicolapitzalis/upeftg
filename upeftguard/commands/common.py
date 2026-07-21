"""Shared CLI output, path safety, and argument definitions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES,
    SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES,
)
from ..supervised.validation.cross_validation import SUPPORTED_CV_STRATEGIES
from ..supervised.data.normalization import SUPPORTED_INPUT_NORMALIZATIONS
from ..supervised.lifecycle.tuning import (
    DANN_DEFAULT_LAMBDA_GAMMA,
    DANN_DEFAULT_LAMBDA_MAX,
    DANN_DEFAULT_SOURCE_RANK,
    DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
)
from ..supervised.models.registry import registered_models
from ..supervised.tasks import SUPPORTED_SELECTION_METRICS
from ..utilities.core.paths import default_dataset_root, dataset_root_help


def _execution_root() -> Path:
    return Path.cwd().resolve()


def _is_filesystem_root(path: Path) -> bool:
    return path == path.parent


def _resolve_output_root(output_root: Path) -> Path:
    root = _execution_root()
    candidate = output_root.expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved == root or _is_filesystem_root(resolved):
        return (root / "runs").resolve()
    return resolved


def _has_option(tokens: list[str], option: str) -> bool:
    return any(tok == option or tok.startswith(option + "=") for tok in tokens)


def _rewrite_download_local_dir(tokens: list[str]) -> list[str]:
    root = _execution_root()
    default_download_root = default_dataset_root()
    safe_local_root = (root / "data").resolve()
    out = list(tokens)

    if not _has_option(out, "--local-dir"):
        return ["--local-dir", str(default_download_root), *out]

    for idx, tok in enumerate(out):
        if tok == "--local-dir" and idx + 1 < len(out):
            current = Path(out[idx + 1]).expanduser()
            resolved = (current if current.is_absolute() else root / current).resolve()
            if resolved == root or _is_filesystem_root(resolved):
                out[idx + 1] = str(safe_local_root)
            return out
        if tok.startswith("--local-dir="):
            raw = tok.split("=", 1)[1]
            current = Path(raw).expanduser()
            resolved = (current if current.is_absolute() else root / current).resolve()
            if resolved == root or _is_filesystem_root(resolved):
                out[idx] = f"--local-dir={safe_local_root}"
            return out
    return out


def _normalize_download_args(tokens: list[str]) -> list[str]:
    out = list(tokens)
    while out and out[0] == "--":
        out = out[1:]
    return out


def _print_result(result: dict[str, Any]) -> None:
    print("Run complete")
    for label, key in (
        ("Run dir", "run_dir"),
        ("Submission run dir", "submission_run_dir"),
        ("Feature file", "feature_path"),
        ("Report", "report"),
        ("Inference scores", "inference_scores_csv"),
        ("Tuning manifest", "tuning_manifest"),
    ):
        value = result.get(key)
        if value:
            print(f"{label}: {value}")
    slurm = result.get("slurm")
    if isinstance(slurm, dict):
        if slurm.get("job_id"):
            print(f"Slurm job id: {slurm['job_id']}")
        if slurm.get("dry_run"):
            print("Dry-run sbatch command:")
            print(" ".join(str(x) for x in slurm.get("command", [])))


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["slurm", "local"], default="slurm")
    parser.add_argument("--partition", default="extra")
    parser.add_argument("--cpus-per-worker", default="4")
    parser.add_argument("--dry-run", action="store_true")


def _add_common_run_args(parser: argparse.ArgumentParser, *, manifest_required: bool = True) -> None:
    parser.add_argument("--manifest-json", type=Path, required=manifest_required)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(), help=dataset_root_help())
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", type=str, default=None)


def _add_spectral_args(parser: argparse.ArgumentParser, *, default_qv_mode: str) -> None:
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_SPECTRAL_FEATURES))
    parser.add_argument("--spectral-sv-top-k", type=int, default=8)
    parser.add_argument(
        "--spectral-moment-source",
        choices=["entrywise", "sv", "both"],
        default=DEFAULT_SPECTRAL_MOMENT_SOURCE,
    )
    parser.add_argument(
        "--spectral-qv-sum-mode",
        choices=["none", "append", "only"],
        default=default_qv_mode,
    )
    parser.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    parser.add_argument(
        "--spectral-attention-granularity",
        choices=list(SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES),
        default=DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    )
    parser.add_argument("--stream-block-size", type=int, default=131072)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=registered_models(),
        required=True,
        help="Supervised model to train; there is no implicit model default.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--cv-strategy",
        choices=list(SUPPORTED_CV_STRATEGIES),
        default="stratified",
        help=(
            "Cross-validation split strategy. Leave-one-out strategies derive folds from "
            "attack families or datasets and ignore --cv-folds."
        ),
    )
    parser.add_argument(
        "--input-normalization",
        choices=list(SUPPORTED_INPUT_NORMALIZATIONS),
        default="none",
        help="Optional experiment-level input normalization applied before supervised models.",
    )
    parser.add_argument(
        "--cv-derived-refit-epochs",
        action="store_true",
        help="Refit the selected transformer for the median validation-selected CV epoch count.",
    )
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Promote the winning candidate's best validation-fold model instead of fitting on all training rows.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--train-split",
        "--train_split",
        dest="train_split",
        type=int,
        default=None,
        help="Percent of a complete manifest assigned to training; omitted means the manifest is training-only.",
    )
    parser.add_argument("--calibration-split", "--calibration_split", dest="calibration_split", type=int, default=None)
    parser.add_argument("--accepted-fpr", "--accepted_fpr", dest="accepted_fpr", type=float, nargs="+", default=None)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--split-by-folder", dest="split_by_folder", action="store_true")
    split_group.add_argument("--no-split-by-folder", dest="split_by_folder", action="store_false")
    parser.set_defaults(split_by_folder=None)
    parser.add_argument("--class-weight-loss", action="store_true")
    parser.add_argument("--rank-label-weight-loss", action="store_true")
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--hyperparams",
        type=Path,
        default=None,
        help=(
            "Model-specific hyperparameter-grid JSON. Required for CNN, DANN, and Transformer; "
            "optional for overriding a classical model's registered grid."
        ),
    )
    parser.add_argument("--checkpoint-interval-hours", type=float, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--task-mode", choices=["binary", "attack_family_multiclass"], default="binary")
    parser.add_argument("--multiclass-attack-names", nargs="+", default=None)
    parser.add_argument("--selection-metric", choices=list(SUPPORTED_SELECTION_METRICS), default="task_default")


def _add_supervised_internal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(), help=dataset_root_help())
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--prepared-run-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=registered_models(),
        default=None,
        help="Required for stage=all and stage=prepare; worker/finalize read the prepared run.",
    )
    parser.add_argument("--task-mode", choices=["binary", "attack_family_multiclass"], default="binary")
    parser.add_argument("--multiclass-attack-names", nargs="+", default=None)
    parser.add_argument(
        "--hyperparams",
        type=Path,
        default=None,
        help="Selected model's hyperparameter-grid JSON.",
    )
    parser.add_argument("--checkpoint-interval-hours", type=float, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--dann-source-rank", type=int, default=DANN_DEFAULT_SOURCE_RANK)
    parser.add_argument(
        "--dann-target-adaptation-percent",
        "--dann-adaptation-percent",
        dest="dann_target_adaptation_percent",
        type=int,
        default=DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
    )
    parser.add_argument("--dann-lambda-max", type=float, default=DANN_DEFAULT_LAMBDA_MAX)
    parser.add_argument("--dann-lambda-gamma", type=float, default=DANN_DEFAULT_LAMBDA_GAMMA)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--spectral-sv-top-k", type=int, default=8)
    parser.add_argument(
        "--spectral-moment-source", choices=["entrywise", "sv", "both"], default=DEFAULT_SPECTRAL_MOMENT_SOURCE
    )
    parser.add_argument(
        "--spectral-qv-sum-mode", choices=["none", "append", "only"], default=DEFAULT_SPECTRAL_QV_SUM_MODE
    )
    parser.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    parser.add_argument(
        "--spectral-attention-granularity",
        choices=list(SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES),
        default=DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    )
    parser.add_argument("--stream-block-size", type=int, default=131072)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--cv-strategy",
        choices=list(SUPPORTED_CV_STRATEGIES),
        default="stratified",
        help=(
            "Cross-validation split strategy. Leave-one-out strategies derive folds from "
            "attack families or datasets and ignore --cv-folds."
        ),
    )
    parser.add_argument(
        "--input-normalization",
        choices=list(SUPPORTED_INPUT_NORMALIZATIONS),
        default="none",
        help="Optional experiment-level input normalization applied before supervised models.",
    )
    parser.add_argument(
        "--cv-derived-refit-epochs",
        action="store_true",
        help="Refit the selected transformer for the median validation-selected CV epoch count.",
    )
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Promote the winning candidate's best validation-fold model instead of fitting on all training rows.",
    )
    parser.add_argument("--selection-metric", choices=list(SUPPORTED_SELECTION_METRICS), default="task_default")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-split", "--train_split", dest="train_split", type=int, default=100)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--split-by-folder", dest="split_by_folder", action="store_true")
    split_group.add_argument("--no-split-by-folder", dest="split_by_folder", action="store_false")
    parser.set_defaults(split_by_folder=None)
    parser.add_argument("--class-weight-loss", action="store_true")
    parser.add_argument("--rank-label-weight-loss", action="store_true")
    parser.add_argument("--calibration-split", "--calibration_split", dest="calibration_split", type=int, default=None)
    parser.add_argument("--accepted-fpr", "--accepted_fpr", dest="accepted_fpr", type=float, nargs="+", default=None)
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tuning-executor", choices=["local", "slurm_packed"], default="local")
    parser.add_argument("--slurm-partition", type=str, default="extra")
    parser.add_argument("--cpus-per-worker", type=str, default="4")
    parser.add_argument("--feature-file", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "worker", "finalize"],
        default="all",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--task-index-offset", type=int, default=0)
