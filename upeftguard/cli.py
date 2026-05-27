from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from .features.spectral import (
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES,
)
from .cnn_pipeline import (
    run_cnn_aggregate,
    run_cnn_extract,
    run_cnn_full,
    run_cnn_infer,
    run_cnn_train,
)
from .supervised.pipeline import (
    DANN_DEFAULT_LAMBDA_GAMMA,
    DANN_DEFAULT_LAMBDA_MAX,
    DANN_DEFAULT_LR_ALPHA,
    DANN_DEFAULT_LR_BETA,
    DANN_DEFAULT_SOURCE_RANK,
    DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
    SUPPORTED_SELECTION_METRICS,
    run_supervised_pipeline,
)
from .supervised.registry import default_cnn_hyperparams_path, registered_models
from .utilities.artifacts.aggregate_features import aggregate_features
from .utilities.artifacts.export_feature_subset import export_feature_subset
from .utilities.core.paths import default_dataset_root, dataset_root_help
from .utilities.merge.merge_feature_files import merge_feature_files


def _execution_root() -> Path:
    return Path.cwd().resolve()


def _is_filesystem_root(path: Path) -> bool:
    return path == path.parent


def _resolve_output_root(output_root: Path, caller: str) -> Path:
    root = _execution_root()
    candidate = output_root.expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved == root or _is_filesystem_root(resolved):
        return (root / "runs" / caller).resolve()
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
        ("Results summary", "results_summary_md"),
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


def _cmd_cnn_extract(args: argparse.Namespace) -> int:
    result = run_cnn_extract(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root, "cnn_extract"),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        worker_cpus=args.worker_cpus,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
        features=args.features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        n_shards=args.n_shards,
    )
    _print_result(result)
    return 0


def _cmd_cnn_aggregate(args: argparse.Namespace) -> int:
    result = run_cnn_aggregate(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        output_root=_resolve_output_root(args.output_root, "cnn_aggregate"),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        worker_cpus=args.worker_cpus,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
        feature_root=args.feature_root,
        features=args.features,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
    )
    _print_result(result)
    return 0


def _cmd_cnn_train(args: argparse.Namespace) -> int:
    result = run_cnn_train(
        manifest_json=args.manifest_json,
        feature_file=args.feature_file,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root, "cnn_train"),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        worker_cpus=args.worker_cpus,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
        features=args.features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        train_split=args.train_split,
        calibration_split=args.calibration_split,
        accepted_fpr=args.accepted_fpr,
        split_by_folder=args.split_by_folder,
        cv_seeds=args.cv_seeds,
        n_jobs=args.n_jobs,
        score_percentiles=args.score_percentiles,
        cnn_hyperparams=args.cnn_hyperparams,
        task_mode=args.task_mode,
        multiclass_attack_names=args.multiclass_attack_names,
        class_weight_loss=args.class_weight_loss,
        rank_label_weight_loss=args.rank_label_weight_loss,
        selection_metric=args.selection_metric,
        skip_feature_importance=args.skip_feature_importance,
    )
    _print_result(result)
    return 0


def _cmd_cnn_infer(args: argparse.Namespace) -> int:
    result = run_cnn_infer(
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        output_root=_resolve_output_root(args.output_root, "cnn_infer"),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        worker_cpus=args.worker_cpus,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
    )
    _print_result(result)
    return 0


def _cmd_cnn_full(args: argparse.Namespace) -> int:
    result = run_cnn_full(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root, "cnn_full"),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        worker_cpus=args.worker_cpus,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
        features=args.features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        n_shards=args.n_shards,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        train_split=args.train_split,
        calibration_split=args.calibration_split,
        accepted_fpr=args.accepted_fpr,
        split_by_folder=args.split_by_folder,
        cv_seeds=args.cv_seeds,
        n_jobs=args.n_jobs,
        score_percentiles=args.score_percentiles,
        cnn_hyperparams=args.cnn_hyperparams,
        task_mode=args.task_mode,
        multiclass_attack_names=args.multiclass_attack_names,
        class_weight_loss=args.class_weight_loss,
        rank_label_weight_loss=args.rank_label_weight_loss,
        selection_metric=args.selection_metric,
        skip_feature_importance=args.skip_feature_importance,
    )
    _print_result(result)
    train = result.get("train")
    if isinstance(train, dict) and train.get("run_dir"):
        print(f"Supervised run dir: {train['run_dir']}")
    return 0


def _cmd_feature_extract(args: argparse.Namespace) -> int:
    if args.extractor != "spectral":
        raise ValueError("Only the stable spectral extractor is available in the active CLI")
    result = run_cnn_extract(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root, "feature_extract"),
        run_id=args.run_id,
        backend="local",
        features=args.spectral_features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
    )
    _print_result(result)
    return 0


def _cmd_run_supervised(args: argparse.Namespace) -> int:
    result = run_supervised_pipeline(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root, "run_supervised"),
        run_id=args.run_id,
        model_name=args.model,
        spectral_features=(list(args.features) if args.features is not None else None),
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        train_split_percent=args.train_split,
        calibration_split_percent=args.calibration_split,
        accepted_fpr=args.accepted_fpr,
        split_by_folder=args.split_by_folder,
        cv_random_states=args.cv_seeds,
        n_jobs=args.n_jobs,
        score_percentiles=args.score_percentiles,
        tuning_executor=args.tuning_executor,
        slurm_partition=args.slurm_partition,
        slurm_max_concurrent=args.slurm_max_concurrent,
        slurm_cpus_per_task=args.slurm_cpus_per_task,
        finalize_export_shards=args.finalize_export_shards,
        stage=args.stage,
        run_dir=args.run_dir,
        task_index=args.task_index,
        task_mode=args.task_mode,
        multiclass_attack_names=(
            list(args.multiclass_attack_names) if args.multiclass_attack_names is not None else None
        ),
        cnn_hyperparams=args.cnn_hyperparams,
        dann_source_rank=args.dann_source_rank,
        dann_target_adaptation_percent=args.dann_target_adaptation_percent,
        dann_lambda_max=args.dann_lambda_max,
        dann_lambda_gamma=args.dann_lambda_gamma,
        dann_lr_alpha=args.dann_lr_alpha,
        dann_lr_beta=args.dann_lr_beta,
        class_weight_loss=args.class_weight_loss,
        rank_label_weight_loss=args.rank_label_weight_loss,
        skip_feature_importance=args.skip_feature_importance,
        selection_metric=args.selection_metric,
        feature_file=args.feature_file,
    )
    _print_result(result)
    if result.get("next_steps"):
        print("Next steps:")
        for step in result["next_steps"]:
            print(f"- {step}")
    return 0


def _cmd_util_aggregate_features(args: argparse.Namespace) -> int:
    outputs = aggregate_features(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        feature_root=args.feature_root,
        operator=args.operator,
        features=args.features,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        layout=args.layout,
    )
    print("Feature aggregation complete")
    for key, value in outputs.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


def _cmd_util_merge_features(args: argparse.Namespace) -> int:
    outputs = merge_feature_files(
        feature_paths=list(args.merge),
        output_filename=args.output_filename,
        feature_root=args.feature_root,
    )
    print("Feature merge complete")
    for key, value in outputs.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


def _cmd_util_export_feature_subset(args: argparse.Namespace) -> int:
    outputs = export_feature_subset(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        feature_root=args.feature_root,
        dataset_names=args.dataset_names,
        subset_names=args.subset_names,
        model_families=args.model_families,
        attack_names=args.attack_names,
        model_names=args.model_names,
        features=args.features,
    )
    print("Feature subset export complete")
    for key, value in outputs.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


def _cmd_download_dataset(args: argparse.Namespace) -> int:
    from .utilities.data import dataset_download

    passthrough = _normalize_download_args(list(args.download_args))
    passthrough = _rewrite_download_local_dir(passthrough)
    old_argv = sys.argv
    try:
        sys.argv = ["download_dataset"] + passthrough
        dataset_download.main()
    finally:
        sys.argv = old_argv
    return 0


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["slurm", "local"], default="slurm")
    parser.add_argument("--partition", default="extra")
    parser.add_argument("--worker-cpus", default="auto")
    parser.add_argument("--max-concurrent", default="auto")
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
    parser.add_argument("--stream-block-size", type=int, default=131072)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-split", "--train_split", dest="train_split", type=int, default=100)
    parser.add_argument("--calibration-split", "--calibration_split", dest="calibration_split", type=int, default=None)
    parser.add_argument("--accepted-fpr", "--accepted_fpr", dest="accepted_fpr", type=float, nargs="+", default=None)
    parser.add_argument("--split-by-folder", action="store_true")
    parser.add_argument("--class-weight-loss", action="store_true")
    parser.add_argument("--rank-label-weight-loss", action="store_true")
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--score-percentiles", nargs="+", type=float, default=None)
    parser.add_argument("--cnn-hyperparams", type=Path, default=None)
    parser.add_argument("--task-mode", choices=["binary", "attack_family_multiclass"], default="binary")
    parser.add_argument("--multiclass-attack-names", nargs="+", default=None)
    parser.add_argument("--selection-metric", choices=list(SUPPORTED_SELECTION_METRICS), default="task_default")
    parser.add_argument("--skip-feature-importance", action="store_true")


def _add_supervised_internal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(), help=dataset_root_help())
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model", choices=["all", *registered_models()], default="logistic_regression")
    parser.add_argument("--task-mode", choices=["binary", "attack_family_multiclass"], default="binary")
    parser.add_argument("--multiclass-attack-names", nargs="+", default=None)
    parser.add_argument(
        "--cnn-hyperparams",
        type=Path,
        default=None,
        help=f"Optional CNN hyperparameter JSON. Defaults to {default_cnn_hyperparams_path()}.",
    )
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
    parser.add_argument("--dann-lr-alpha", type=float, default=DANN_DEFAULT_LR_ALPHA)
    parser.add_argument("--dann-lr-beta", type=float, default=DANN_DEFAULT_LR_BETA)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--spectral-sv-top-k", type=int, default=8)
    parser.add_argument("--spectral-moment-source", choices=["entrywise", "sv", "both"], default=DEFAULT_SPECTRAL_MOMENT_SOURCE)
    parser.add_argument("--spectral-qv-sum-mode", choices=["none", "append", "only"], default=DEFAULT_SPECTRAL_QV_SUM_MODE)
    parser.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    parser.add_argument("--stream-block-size", type=int, default=131072)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--selection-metric", choices=list(SUPPORTED_SELECTION_METRICS), default="task_default")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-split", "--train_split", dest="train_split", type=int, default=100)
    parser.add_argument("--split-by-folder", action="store_true")
    parser.add_argument("--class-weight-loss", action="store_true")
    parser.add_argument("--rank-label-weight-loss", action="store_true")
    parser.add_argument("--calibration-split", "--calibration_split", dest="calibration_split", type=int, default=None)
    parser.add_argument("--accepted-fpr", "--accepted_fpr", dest="accepted_fpr", type=float, nargs="+", default=None)
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--score-percentiles", nargs="+", type=float, default=None)
    parser.add_argument("--tuning-executor", choices=["local", "slurm_array"], default="local")
    parser.add_argument("--slurm-partition", type=str, default="extra")
    parser.add_argument("--slurm-max-concurrent", type=str, default="auto")
    parser.add_argument("--slurm-cpus-per-task", type=str, default="auto")
    parser.add_argument("--finalize-export-shards", type=int, default=1)
    parser.add_argument("--skip-feature-importance", action="store_true")
    parser.add_argument("--feature-file", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "worker", "finalize", "finalize_prepare", "finalize_worker", "finalize_merge"],
        default="all",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--task-index", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="upeftguard stable CNN CLI")
    sub = parser.add_subparsers(dest="command_group", required=True)

    cnn = sub.add_parser("cnn", help="Run the stable CNN pipeline")
    cnn_sub = cnn.add_subparsers(dest="cnn_command", required=True)

    full = cnn_sub.add_parser("full", help="Run feature extraction, CNN aggregation, and CNN training")
    _add_backend_args(full)
    _add_common_run_args(full)
    _add_spectral_args(full, default_qv_mode=DEFAULT_SPECTRAL_QV_SUM_MODE)
    _add_training_args(full)
    full.add_argument("--n-shards", type=int, default=8)
    full.set_defaults(func=_cmd_cnn_full)

    extract = cnn_sub.add_parser("extract", help="Run spectral feature extraction")
    _add_backend_args(extract)
    _add_common_run_args(extract)
    _add_spectral_args(extract, default_qv_mode=DEFAULT_SPECTRAL_QV_SUM_MODE)
    extract.add_argument("--n-shards", type=int, default=8)
    extract.set_defaults(func=_cmd_cnn_extract)

    aggregate = cnn_sub.add_parser("aggregate", help="Build a CNN layer-sequence feature bundle")
    _add_backend_args(aggregate)
    aggregate.add_argument("--feature-file", type=Path, required=True)
    aggregate.add_argument("--output-filename", type=Path, default=None)
    aggregate.add_argument("--output-root", type=Path, default=Path("runs"))
    aggregate.add_argument("--run-id", type=str, default=None)
    aggregate.add_argument("--feature-root", type=Path, default=None)
    aggregate.add_argument("--features", nargs="+", default=list(DEFAULT_SPECTRAL_FEATURES))
    aggregate.add_argument("--spectral-qv-sum-mode", choices=["none", "append", "only"], default="append")
    aggregate.set_defaults(func=_cmd_cnn_aggregate)

    train = cnn_sub.add_parser("train", help="Train/finalize the CNN from an aggregated feature bundle")
    _add_backend_args(train)
    _add_common_run_args(train)
    train.add_argument("--feature-file", type=Path, required=True)
    _add_spectral_args(train, default_qv_mode="append")
    _add_training_args(train)
    train.set_defaults(func=_cmd_cnn_train)

    infer = cnn_sub.add_parser("infer", help="Run checkpoint inference for a prepared CNN run")
    _add_backend_args(infer)
    infer.add_argument("--checkpoint", type=Path, default=None)
    infer.add_argument("--run-dir", type=Path, default=None)
    infer.add_argument("--output-root", type=Path, default=Path("runs"))
    infer.add_argument("--run-id", type=str, default=None)
    infer.set_defaults(func=_cmd_cnn_infer)

    run_parser = sub.add_parser("run", help="Internal compatibility runners")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)
    supervised = run_sub.add_parser("supervised", help="Internal supervised runner")
    _add_supervised_internal_args(supervised)
    supervised.set_defaults(func=_cmd_run_supervised)

    feature_parser = sub.add_parser("feature", help="Internal feature commands")
    feature_sub = feature_parser.add_subparsers(dest="feature_command", required=True)
    feature_extract = feature_sub.add_parser("extract", help="Internal spectral feature extraction")
    feature_extract.add_argument("--manifest-json", type=Path, required=True)
    feature_extract.add_argument("--dataset-root", type=Path, default=default_dataset_root(), help=dataset_root_help())
    feature_extract.add_argument("--extractor", choices=["spectral"], default="spectral")
    feature_extract.add_argument("--output-root", type=Path, default=Path("runs"))
    feature_extract.add_argument("--run-id", type=str, default=None)
    feature_extract.add_argument("--spectral-features", nargs="+", default=list(DEFAULT_SPECTRAL_FEATURES))
    feature_extract.add_argument("--spectral-sv-top-k", type=int, default=8)
    feature_extract.add_argument("--spectral-moment-source", choices=["entrywise", "sv", "both"], default=DEFAULT_SPECTRAL_MOMENT_SOURCE)
    feature_extract.add_argument("--spectral-qv-sum-mode", choices=["none", "append", "only"], default=DEFAULT_SPECTRAL_QV_SUM_MODE)
    feature_extract.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    feature_extract.add_argument("--stream-block-size", type=int, default=131072)
    feature_extract.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    feature_extract.set_defaults(func=_cmd_feature_extract)

    util_parser = sub.add_parser("util", help="Internal utilities")
    util_sub = util_parser.add_subparsers(dest="util_command", required=True)
    merge_features = util_sub.add_parser("merge-features", help="Merge spectral feature files")
    merge_features.add_argument("--merge", type=Path, nargs=2, metavar=("FILE1", "FILE2"), required=True)
    merge_features.add_argument("--output-filename", type=Path, required=True)
    merge_features.add_argument("--feature-root", type=Path, default=Path("runs") / "feature_extract")
    merge_features.set_defaults(func=_cmd_util_merge_features)

    export_subset = util_sub.add_parser("export-feature-subset", help="Export a provenance-backed feature subset")
    export_subset.add_argument("--feature-file", type=Path, required=True)
    export_subset.add_argument("--output-filename", type=Path, required=True)
    export_subset.add_argument("--feature-root", type=Path, default=Path("runs") / "feature_extract")
    export_subset.add_argument("--dataset-name", dest="dataset_names", nargs="+", default=None)
    export_subset.add_argument("--subset-name", dest="subset_names", nargs="+", default=None)
    export_subset.add_argument("--model-family", dest="model_families", nargs="+", default=None)
    export_subset.add_argument("--attack-name", dest="attack_names", nargs="+", default=None)
    export_subset.add_argument("--model-name", dest="model_names", nargs="+", default=None)
    export_subset.add_argument("--features", "--columns", dest="features", nargs="+", default=None)
    export_subset.set_defaults(func=_cmd_util_export_feature_subset)

    aggregate_features_parser = util_sub.add_parser("aggregate-features", help="Aggregate a spectral feature file")
    aggregate_features_parser.add_argument("--feature-file", type=Path, required=True)
    aggregate_features_parser.add_argument("--output-filename", type=Path, required=True)
    aggregate_features_parser.add_argument("--operator", choices=["avg", "max", "min"], default="avg")
    aggregate_features_parser.add_argument("--layout", choices=["flat", "layer_sequence"], default="flat")
    aggregate_features_parser.add_argument("--feature-root", type=Path, default=Path("runs") / "feature_extract")
    aggregate_features_parser.add_argument("--features", "--columns", dest="features", nargs="+", default=None)
    aggregate_features_parser.add_argument("--spectral-qv-sum-mode", choices=["none", "append", "only"], default="append")
    aggregate_features_parser.set_defaults(func=_cmd_util_aggregate_features)

    download = util_sub.add_parser("download-dataset", help="Download PADBench subsets")
    download.add_argument("download_args", nargs=argparse.REMAINDER)
    download.set_defaults(func=_cmd_download_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    if len(argv_tokens) >= 2 and argv_tokens[0] == "util" and argv_tokens[1] == "download-dataset":
        args = argparse.Namespace(
            command_group="util",
            util_command="download-dataset",
            download_args=argv_tokens[2:],
            func=_cmd_download_dataset,
        )
    else:
        args = parser.parse_args(argv_tokens)
    result = args.func(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
