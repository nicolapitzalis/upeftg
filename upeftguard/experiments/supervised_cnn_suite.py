from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence

from .attack_family_leave_one_out import prepare_manifests as prepare_attack_family_manifests
from .group_leave_one_out import prepare_manifests as prepare_group_leave_one_out_manifests
from ..supervised.pipeline import SUPPORTED_SELECTION_METRICS, run_supervised_pipeline


DEFAULT_MULTICLASS_ATTACK_NAMES = ("RIPPLE", "insertsent", "stybkd", "syntactic")


@dataclass(frozen=True)
class SuiteDefaults:
    manifest_root: Path
    run_id_prefix: str
    suite_label: str
    suite_label_lower: str
    task_mode: str = "binary"
    manifest_filter: str = ""
    multiclass_attack_names: tuple[str, ...] = ()
    hyperparam_config: str = ""
    skip_feature_importance: bool = False
    attack_family_generated_subdir: str | None = None
    group_leave_one_out: str | None = None
    group_source_manifest: Path | None = None
    group_generated_subdir: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _truthy(value: bool | None, *, default: bool = False) -> bool:
    return default if value is None else bool(value)


def _env_words(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return str(raw).split()


def _resolve_path(path: str | Path, *, base: Path) -> Path:
    raw = Path(path).expanduser()
    return (raw if raw.is_absolute() else base / raw).resolve()


def _resolve_feature_spec(path: str | Path, *, repo_root: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    if len(raw.parts) == 1:
        return (repo_root / "runs" / "feature_extract" / raw / "merged" / "spectral_features.npy").resolve()
    return (repo_root / raw).resolve()


def _resolve_cnn_hyperparams_spec(path: str | Path, *, repo_root: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    if len(raw.parts) == 1:
        filename = raw.name if raw.suffix else f"{raw.name}.json"
        return (repo_root / "manifests" / "cnn_hyperparams" / filename).resolve()
    return (repo_root / raw).resolve()


def _default_dataset_root() -> Path:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    storage_root = Path(
        os.environ.get(
            "UPEFTGUARD_STORAGE_ROOT",
            f"/models/{user}/unsupervised-peftguard",
        )
    ).expanduser()
    return Path(
        os.environ.get(
            "DATASET_ROOT",
            os.environ.get("UPEFTGUARD_DATA_ROOT", str(storage_root / "data")),
        )
    ).expanduser()


def _suite_defaults(suite: str, repo_root: Path) -> SuiteDefaults:
    if suite == "leave-one-out":
        return SuiteDefaults(
            manifest_root=repo_root / "manifests" / "leave_one_out",
            run_id_prefix="leave_one_out_cnn",
            suite_label="Leave-one-out",
            suite_label_lower="leave-one-out",
        )
    if suite == "attack-family-leave-one-out-multiclass":
        return SuiteDefaults(
            manifest_root=repo_root / "runs" / "generated_manifests" / "leave_one_out_attack_family_multiclass",
            run_id_prefix="leave_one_out_attack_family_multiclass_cnn",
            suite_label="Attack-family multiclass leave-one-out",
            suite_label_lower="attack-family multiclass leave-one-out",
            task_mode="attack_family_multiclass",
            multiclass_attack_names=DEFAULT_MULTICLASS_ATTACK_NAMES,
            skip_feature_importance=True,
            attack_family_generated_subdir="leave_one_out_attack_family_multiclass",
        )
    if suite == "attack-family-leave-one-out-binary":
        return SuiteDefaults(
            manifest_root=repo_root / "runs" / "generated_manifests" / "leave_one_out_attack_family_binary",
            run_id_prefix="leave_one_out_attack_family_binary_cnn",
            suite_label="Attack-family binary leave-one-out",
            suite_label_lower="attack-family binary leave-one-out",
            hyperparam_config="cnn_ag_news_imdb_attack_family_binary_tuning",
            skip_feature_importance=True,
            attack_family_generated_subdir="leave_one_out_attack_family_binary",
        )
    if suite == "adapter-leave-one-out":
        return SuiteDefaults(
            manifest_root=repo_root / "runs" / "generated_manifests" / "leave_one_out_adapter",
            run_id_prefix="leave_one_out_adapter_cnn",
            suite_label="Adapter leave-one-out",
            suite_label_lower="adapter leave-one-out",
            skip_feature_importance=True,
            group_leave_one_out="adapter",
            group_source_manifest=repo_root / "manifests" / "adapter_exploration" / "llama2_7b_tbh_all_adapters.json",
            group_generated_subdir="leave_one_out_adapter",
        )
    if suite == "architecture-leave-one-out":
        return SuiteDefaults(
            manifest_root=repo_root / "runs" / "generated_manifests" / "leave_one_out_architecture",
            run_id_prefix="leave_one_out_architecture_cnn",
            suite_label="Architecture leave-one-out",
            suite_label_lower="architecture leave-one-out",
            skip_feature_importance=True,
            group_leave_one_out="architecture",
            group_source_manifest=repo_root / "manifests" / "architecture_exploration" / "tbh_all_architectures.json",
            group_generated_subdir="leave_one_out_architecture",
        )
    if suite == "tbh-tba-zero-shot":
        return SuiteDefaults(
            manifest_root=repo_root / "manifests" / "zero_shots" / "attack_wise",
            run_id_prefix="tbh_tba_zero_shot_cnn",
            suite_label="TBH/TBA zero-shot",
            suite_label_lower="tbh/tba zero-shot",
            manifest_filter="llama2_7b_tbh_tba_zero_shot_",
            skip_feature_importance=True,
        )
    if suite == "tbh-tba-open-set-zero-shot":
        return SuiteDefaults(
            manifest_root=repo_root / "manifests" / "zero_shots" / "attack_wise",
            run_id_prefix="tbh_tba_open_set_zero_shot_cnn",
            suite_label="TBH/TBA open-set zero-shot",
            suite_label_lower="tbh/tba open-set zero-shot",
            task_mode="attack_family_multiclass",
            manifest_filter="llama2_7b_tbh_tba_zero_shot_",
            multiclass_attack_names=("toxic_backdoors_hard", "toxic_backdoors_alpaca"),
            skip_feature_importance=True,
        )
    if suite == "zero-shot":
        return SuiteDefaults(
            manifest_root=repo_root / "manifests" / "zero_shots",
            run_id_prefix="",
            suite_label="Zero-shot",
            suite_label_lower="zero-shot",
        )
    raise ValueError(f"Unknown supervised CNN suite: {suite}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a supervised CNN transfer suite by preparing each target run, "
            "then submitting worker and dependent finalize Slurm jobs per manifest. "
            "Use --hyperparam-config to freeze to a reference winner, or "
            "--cnn-hyperparams to run a fresh CNN grid per held-out manifest."
        )
    )
    parser.add_argument(
        "--suite",
        choices=[
            "zero-shot",
            "leave-one-out",
            "adapter-leave-one-out",
            "architecture-leave-one-out",
            "attack-family-leave-one-out-binary",
            "attack-family-leave-one-out-multiclass",
            "tbh-tba-zero-shot",
            "tbh-tba-open-set-zero-shot",
        ],
        default=os.environ.get("SUPERVISED_CNN_SUITE", "zero-shot"),
    )
    parser.add_argument("--hyperparam-config", "--hyperparam_config", dest="hyperparam_config", default=None)
    parser.add_argument(
        "--cnn-hyperparams",
        "--cnn_hyperparams",
        dest="cnn_hyperparams",
        type=Path,
        default=None,
        help=(
            "CNN hyperparameter grid JSON. When set, each held-out manifest runs its own grid search "
            "instead of freezing a reference winner."
        ),
    )
    parser.add_argument("--manifest-root", "--zero-shot-manifest-root", dest="manifest_root", type=Path, default=None)
    parser.add_argument("--manifest-filter", "--manifest_filter", dest="manifest_filter", default=None)
    parser.add_argument("--run-id-prefix", dest="run_id_prefix", default=None)
    parser.add_argument("--suite-label", dest="suite_label", default=None)
    parser.add_argument("--suite-label-lower", dest="suite_label_lower", default=None)
    parser.add_argument("--feature-file", dest="feature_file", default=None)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--task-mode",
        choices=["binary", "attack_family_multiclass"],
        default=None,
    )
    parser.add_argument("--multiclass-attack-names", nargs="+", default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--train-split", "--train_split", dest="train_split", type=int, default=None)
    parser.add_argument("--calibration-split", "--calibration_split", dest="calibration_split", type=int, default=None)
    parser.add_argument("--accepted-fpr", "--accepted_fpr", dest="accepted_fpr", nargs="+", type=float, default=None)
    parser.add_argument("--split-by-folder", dest="split_by_folder", action="store_true", default=None)
    parser.add_argument("--no-split-by-folder", dest="split_by_folder", action="store_false")
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=list(SUPPORTED_SELECTION_METRICS),
        default=None,
        help=(
            "Metric used to choose the winning hyperparameters. Use binary_auroc to train multiclass "
            "but select by clean-vs-any-attack AUROC."
        ),
    )
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--spectral-sv-top-k", type=int, default=None)
    parser.add_argument("--spectral-moment-source", choices=["entrywise", "sv", "both"], default=None)
    parser.add_argument("--spectral-qv-sum-mode", choices=["none", "append", "only"], default=None)
    parser.add_argument("--spectral-entrywise-delta-mode", choices=["auto", "dense", "stream"], default=None)
    parser.add_argument("--score-percentiles", nargs="+", type=float, default=None)
    parser.add_argument("--conda-sh", type=Path, default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--slurm-partition", "--partition", dest="slurm_partition", default=None)
    parser.add_argument("--slurm-log-dir", "--log-dir", dest="slurm_log_dir", type=Path, default=None)
    parser.add_argument("--worker-cpus-per-task", "--worker-cpus", dest="worker_cpus_per_task", type=int, default=None)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", default=None)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--skip-feature-importance", dest="skip_feature_importance", action="store_true", default=None)
    parser.add_argument("--keep-feature-importance", dest="skip_feature_importance", action="store_false")
    parser.add_argument("--rank-label-weight-loss", action="store_true", default=None)
    parser.add_argument("--no-rank-label-weight-loss", dest="rank_label_weight_loss", action="store_false")
    parser.add_argument(
        "--source-leave-one-out-manifest-root",
        type=Path,
        default=None,
        help="Source full leave-one-out manifest root for attack-family generated suites.",
    )
    parser.add_argument(
        "--generated-manifest-root",
        type=Path,
        default=None,
        help="Output root for generated attack-family, adapter, or architecture leave-one-out manifests.",
    )
    parser.add_argument(
        "--source-group-manifest",
        type=Path,
        default=None,
        help="Source grouped manifest for adapter/architecture generated leave-one-out suites.",
    )
    return parser


def _read_reference_defaults(reference_run_dir: Path) -> dict[str, Any]:
    run_config_path = reference_run_dir / "run_config.json"
    report_path = reference_run_dir / "reports" / "supervised_report.json"
    tuning_manifest_path = reference_run_dir / "reports" / "tuning_manifest.json"
    for path, label in [
        (run_config_path, "Reference run_config"),
        (report_path, "Reference supervised_report"),
        (tuning_manifest_path, "Reference tuning_manifest"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    json.loads(run_config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tuning_manifest = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))

    winner = report.get("tuning", {}).get("winner")
    if not isinstance(winner, dict):
        raise ValueError("Reference supervised_report.json is missing tuning.winner")
    winner_params = winner.get("params")
    if not isinstance(winner_params, dict):
        raise ValueError("Reference winner is missing params")

    extractor = tuning_manifest.get("extractor", {})
    extractor_params = extractor.get("params", {}) if isinstance(extractor, dict) else {}
    extractor_metadata = extractor.get("metadata", {}) if isinstance(extractor, dict) else {}
    threshold_selection = tuning_manifest.get("threshold_selection", {})
    if not isinstance(threshold_selection, dict):
        threshold_selection = {}
    tuning = tuning_manifest.get("tuning", {})
    if not isinstance(tuning, dict):
        tuning = {}

    feature_file = extractor_metadata.get("external_feature_source")
    if not isinstance(feature_file, str) or not feature_file:
        raise ValueError("Reference tuning manifest is missing extractor.metadata.external_feature_source")
    spectral_features = extractor_params.get("spectral_features")
    if not isinstance(spectral_features, list) or not spectral_features:
        raise ValueError("Reference tuning manifest is missing extractor.params.spectral_features")

    accepted_fprs = threshold_selection.get("accepted_fprs")
    if accepted_fprs is None:
        accepted_fprs = threshold_selection.get("accepted_fpr")
    if accepted_fprs is None:
        accepted_fprs = []
    elif not isinstance(accepted_fprs, list):
        accepted_fprs = [accepted_fprs]

    return {
        "winner_model_name": str(winner.get("model_name", "cnn_1d")),
        "winner_task_index": int(winner.get("task_index", 0)),
        "winner_params": winner_params,
        "feature_file": feature_file,
        "features": [str(x) for x in spectral_features],
        "spectral_sv_top_k": int(extractor_params.get("spectral_sv_top_k", 8)),
        "spectral_moment_source": str(extractor_params.get("spectral_moment_source", "both")),
        "spectral_qv_sum_mode": str(extractor_params.get("spectral_qv_sum_mode", "append")),
        "spectral_entrywise_delta_mode": str(extractor_params.get("spectral_entrywise_delta_mode", "dense")),
        "cv_folds": int(tuning.get("cv_folds_requested", 5)),
        "cv_seeds": [int(x) for x in tuning.get("cv_random_states", [42])],
        "selection_metric": str(tuning.get("metric", "task_default")),
        "calibration_split": threshold_selection.get("calibration_split_percent"),
        "accepted_fprs": [float(x) for x in accepted_fprs],
        "split_by_folder": bool(threshold_selection.get("split_by_folder", False)),
    }


def _resolve_reference_run_dir(hyperparam_config: str, repo_root: Path) -> tuple[str, Path]:
    spec = Path(str(hyperparam_config)).expanduser()
    if spec.is_absolute() or "/" in str(hyperparam_config):
        resolved = (spec if spec.is_absolute() else repo_root / spec).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Reference run directory not found: {resolved}")
        return resolved.name, resolved
    return str(hyperparam_config), repo_root / "runs" / "supervised" / str(hyperparam_config)


def _lock_tuning_manifest(
    *,
    tuning_manifest_path: Path,
    reference_run_id: str,
    reference_task_index: int,
    winner_params: dict[str, Any],
    extra_task_params: dict[str, Any] | None = None,
) -> None:
    payload = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    extra_task_params = dict(extra_task_params or {})

    def params_match_reference(task_params: Any) -> bool:
        if not isinstance(task_params, dict):
            return False
        comparable = dict(task_params)
        for key in extra_task_params:
            comparable.pop(key, None)
        return comparable == winner_params and all(
            task_params.get(key) == value for key, value in extra_task_params.items()
        )

    tasks = payload.get("tuning", {}).get("tasks", [])
    matches = [
        task
        for task in tasks
        if isinstance(task, dict)
        and str(task.get("model_name")) == "cnn_1d"
        and params_match_reference(task.get("params"))
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Expected at most one matching cnn_1d task for reference winner params in {tuning_manifest_path}, "
            f"found {len(matches)}"
        )

    if matches:
        selected = dict(matches[0])
    else:
        template = next(
            (
                task
                for task in tasks
                if isinstance(task, dict) and str(task.get("model_name")) == "cnn_1d"
            ),
            {},
        )
        selected = dict(template)
        selected["model_name"] = "cnn_1d"
        selected["params"] = dict(winner_params)
    selected["params"] = {**dict(selected.get("params", {})), **extra_task_params}
    selected.setdefault("complexity_rank", 6)
    selected.setdefault("normalization_policy", "masked_train_only")

    selected["task_index"] = 0
    payload["tuning"]["tasks"] = [selected]

    cv_split_groups = payload.get("tuning", {}).get("cv_split_groups")
    if isinstance(cv_split_groups, list):
        total_splits = sum(
            len(group.get("cv_splits", []))
            for group in cv_split_groups
            if isinstance(group, dict)
        )
    else:
        total_splits = len(payload.get("tuning", {}).get("cv_splits", []))
    payload["tuning"]["estimated_total_fits"] = int(max(1, total_splits))
    payload["tuning"]["fixed_reference_winner"] = {
        "source_run_id": reference_run_id,
        "source_task_index": int(reference_task_index),
        "model_name": "cnn_1d",
        "params": {**dict(winner_params), **extra_task_params},
    }

    warnings = [
        str(row)
        for row in payload.get("warnings", [])
        if not str(row).startswith("Large supervised grid search:")
    ]
    warnings.append(
        "Fixed supervised tuning to the cnn_1d winner from "
        f"{reference_run_id} (source task_index={reference_task_index})"
    )
    if not matches:
        warnings.append(
            "Injected reference winner params directly because they were not present "
            "in the target run's candidate grid"
        )
    payload["warnings"] = warnings

    tuning_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "Locked tuning manifest to reference winner:",
        json.dumps(payload["tuning"]["tasks"][0], sort_keys=True),
    )


def _runtime_defaults(tuning_manifest_path: Path) -> tuple[int, int, list[float]]:
    payload = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    runtime = payload.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    return (
        int(runtime.get("slurm_cpus_per_task", 4)),
        int(runtime.get("slurm_max_concurrent", 1)),
        [float(x) for x in runtime.get("score_percentiles", [])],
    )


def _tuning_task_count(tuning_manifest_path: Path) -> int:
    payload = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    tasks = payload.get("tuning", {}).get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Tuning manifest has no tasks: {tuning_manifest_path}")
    return int(len(tasks))


def _submit_sbatch(command: list[str], *, repo_root: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _wrap_command(parts: Sequence[str]) -> str:
    return " ".join(
        str(part) if str(part) in {"&&", "||", ";"} else shlex.quote(str(part))
        for part in parts
    )


def _split_by_folder_from(
    args: argparse.Namespace,
    reference_defaults: dict[str, Any] | None,
) -> bool:
    if args.split_by_folder is not None:
        return bool(args.split_by_folder)
    env_value = _env_bool("SPLIT_BY_FOLDER")
    if env_value is not None:
        return env_value
    if reference_defaults is not None:
        return bool(reference_defaults["split_by_folder"])
    return False


def _reference_default(
    reference_defaults: dict[str, Any] | None,
    key: str,
    fallback: Any,
) -> Any:
    if reference_defaults is not None and key in reference_defaults:
        return reference_defaults[key]
    return fallback


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo_root = _repo_root()
    suite_defaults = _suite_defaults(args.suite, repo_root)
    generated_manifest_root_for_suite: Path | None = None

    if suite_defaults.attack_family_generated_subdir is not None:
        source_root = _resolve_path(
            args.source_leave_one_out_manifest_root
            or os.environ.get("SOURCE_LEAVE_ONE_OUT_MANIFEST_ROOT")
            or (repo_root / "manifests" / "leave_one_out"),
            base=repo_root,
        )
        generated_root = _resolve_path(
            args.generated_manifest_root
            or os.environ.get("GENERATED_MANIFEST_ROOT")
            or (repo_root / "runs" / "generated_manifests" / suite_defaults.attack_family_generated_subdir),
            base=repo_root,
        )
        prepare_attack_family_manifests(source_root, generated_root)
        generated_manifest_root_for_suite = generated_root

    if suite_defaults.group_leave_one_out is not None:
        source_manifest = _resolve_path(
            args.source_group_manifest
            or os.environ.get("SOURCE_GROUP_MANIFEST")
            or suite_defaults.group_source_manifest,
            base=repo_root,
        )
        generated_root = _resolve_path(
            args.generated_manifest_root
            or os.environ.get("GENERATED_MANIFEST_ROOT")
            or (repo_root / "runs" / "generated_manifests" / suite_defaults.group_generated_subdir),
            base=repo_root,
        )
        prepare_group_leave_one_out_manifests(
            source_manifest,
            generated_root,
            group_by=suite_defaults.group_leave_one_out,
        )
        generated_manifest_root_for_suite = generated_root

    manifest_root = _resolve_path(
        args.manifest_root
        or os.environ.get("ZERO_SHOT_MANIFEST_ROOT")
        or generated_manifest_root_for_suite
        or suite_defaults.manifest_root,
        base=repo_root,
    )
    output_root = _resolve_path(args.output_root or os.environ.get("OUTPUT_ROOT") or "runs", base=repo_root)
    dataset_root = Path(args.dataset_root or os.environ.get("DATASET_ROOT") or _default_dataset_root()).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (repo_root / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    conda_sh = _resolve_path(
        args.conda_sh
        or os.environ.get("CONDA_SH")
        or "/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh",
        base=repo_root,
    )
    conda_env = str(args.conda_env or os.environ.get("CONDA_ENV") or "upeftg")
    slurm_partition = str(args.slurm_partition or os.environ.get("SLURM_PARTITION") or "extra")
    slurm_log_dir = _resolve_path(
        args.slurm_log_dir or os.environ.get("SLURM_LOG_DIR") or (repo_root / "logs"),
        base=repo_root,
    )
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    cnn_hyperparams_raw = args.cnn_hyperparams or os.environ.get("CNN_HYPERPARAMS")
    cnn_hyperparams = (
        _resolve_cnn_hyperparams_spec(cnn_hyperparams_raw, repo_root=repo_root)
        if cnn_hyperparams_raw
        else None
    )
    tuning_mode = "per-holdout-grid" if cnn_hyperparams is not None else "fixed-reference"

    hyperparam_config = (
        args.hyperparam_config
        or os.environ.get("HYPERPARAM_CONFIG")
        or suite_defaults.hyperparam_config
    )
    if tuning_mode == "fixed-reference" and not hyperparam_config:
        raise SystemExit("--hyperparam-config is required unless --cnn-hyperparams is set.")

    reference_run_id: str | None = None
    reference_defaults: dict[str, Any] | None = None
    if hyperparam_config:
        reference_run_id, reference_run_dir = _resolve_reference_run_dir(str(hyperparam_config), repo_root)
        reference_defaults = _read_reference_defaults(reference_run_dir)

    if not manifest_root.is_dir():
        raise SystemExit(f"{suite_defaults.suite_label} manifest root not found: {manifest_root}")
    if not conda_sh.exists():
        raise SystemExit(f"Conda activation script not found: {conda_sh}")
    if cnn_hyperparams is not None and not cnn_hyperparams.exists():
        raise SystemExit(f"CNN hyperparams file not found: {cnn_hyperparams}")

    manifest_filter = (
        args.manifest_filter
        if args.manifest_filter is not None
        else os.environ.get("MANIFEST_FILTER", suite_defaults.manifest_filter)
    )
    run_id_prefix = (
        args.run_id_prefix
        if args.run_id_prefix is not None
        else os.environ.get("RUN_ID_PREFIX", suite_defaults.run_id_prefix)
    )
    suite_label = (
        args.suite_label
        if args.suite_label is not None
        else os.environ.get("SUITE_LABEL", suite_defaults.suite_label)
    )
    suite_label_lower = (
        args.suite_label_lower
        if args.suite_label_lower is not None
        else os.environ.get("SUITE_LABEL_LOWER", suite_defaults.suite_label_lower)
    )
    task_mode = str(args.task_mode or os.environ.get("TASK_MODE") or suite_defaults.task_mode)
    multiclass_attack_names = (
        list(args.multiclass_attack_names)
        if args.multiclass_attack_names is not None
        else (_env_words("MULTICLASS_ATTACK_NAMES") or list(suite_defaults.multiclass_attack_names))
    )
    if task_mode == "attack_family_multiclass" and not multiclass_attack_names:
        multiclass_attack_names = list(DEFAULT_MULTICLASS_ATTACK_NAMES)

    feature_file = (
        args.feature_file
        or os.environ.get("FEATURE_FILE")
        or _reference_default(reference_defaults, "feature_file", None)
    )
    if not feature_file:
        raise SystemExit("--feature-file is required when --cnn-hyperparams is set without --hyperparam-config.")
    features = (
        list(args.features)
        if args.features is not None
        else (_env_words("FEATURES") or list(_reference_default(reference_defaults, "features", [])))
    )
    if not features:
        raise SystemExit("--features is required when --cnn-hyperparams is set without --hyperparam-config.")
    model = str(args.model or os.environ.get("MODEL") or _reference_default(reference_defaults, "winner_model_name", "cnn_1d"))
    spectral_sv_top_k = int(args.spectral_sv_top_k or os.environ.get("SV_TOP_K") or _reference_default(reference_defaults, "spectral_sv_top_k", 8))
    spectral_moment_source = str(
        args.spectral_moment_source
        or os.environ.get("SPECTRAL_MOMENT_SOURCE")
        or _reference_default(reference_defaults, "spectral_moment_source", "both")
    )
    spectral_qv_sum_mode = str(
        args.spectral_qv_sum_mode
        or os.environ.get("SPECTRAL_QV_SUM_MODE")
        or _reference_default(reference_defaults, "spectral_qv_sum_mode", "append")
    )
    spectral_entrywise_delta_mode = str(
        args.spectral_entrywise_delta_mode
        or os.environ.get("SPECTRAL_ENTRYWISE_DELTA_MODE")
        or _reference_default(reference_defaults, "spectral_entrywise_delta_mode", "dense")
    )
    cv_folds = int(args.cv_folds or os.environ.get("CV_FOLDS") or _reference_default(reference_defaults, "cv_folds", 5))
    selection_metric = str(
        args.selection_metric
        or os.environ.get("SELECTION_METRIC")
        or _reference_default(reference_defaults, "selection_metric", "task_default")
    )
    cv_seeds = (
        list(args.cv_seeds)
        if args.cv_seeds is not None
        else [int(x) for x in (_env_words("CV_SEEDS") or _reference_default(reference_defaults, "cv_seeds", [42]))]
    )
    if not cv_seeds:
        raise SystemExit("CV_SEEDS must include at least one value.")
    train_split = int(args.train_split or os.environ.get("TRAIN_SPLIT") or 100)
    calibration_split_raw = (
        args.calibration_split
        if args.calibration_split is not None
        else os.environ.get("CALIBRATION_SPLIT", _reference_default(reference_defaults, "calibration_split", None))
    )
    calibration_split = None if calibration_split_raw in {None, ""} else int(calibration_split_raw)
    accepted_fpr = (
        list(args.accepted_fpr)
        if args.accepted_fpr is not None
        else [float(x) for x in (_env_words("ACCEPTED_FPR") or _reference_default(reference_defaults, "accepted_fprs", []))]
    )
    if calibration_split is None and not accepted_fpr:
        accepted_fpr = None
    if (calibration_split is None) != (accepted_fpr is None):
        raise SystemExit("CALIBRATION_SPLIT and ACCEPTED_FPR must either both be set or both be empty.")
    split_by_folder = _split_by_folder_from(args, reference_defaults)
    rank_label_weight_loss = _truthy(
        args.rank_label_weight_loss
        if args.rank_label_weight_loss is not None
        else _env_bool("RANK_LABEL_WEIGHT_LOSS"),
        default=False,
    )
    dry_run = _truthy(args.dry_run if args.dry_run is not None else _env_bool("DRY_RUN"), default=False)
    skip_feature_importance = _truthy(
        args.skip_feature_importance
        if args.skip_feature_importance is not None
        else _env_bool("SKIP_FEATURE_IMPORTANCE"),
        default=suite_defaults.skip_feature_importance,
    )
    score_percentiles = (
        list(args.score_percentiles)
        if args.score_percentiles is not None
        else ([float(x) for x in _env_words("SCORE_PERCENTILES")] if _env_words("SCORE_PERCENTILES") else None)
    )

    manifests = sorted(manifest_root.glob("**/*.json"))
    if manifest_filter:
        manifests = [path for path in manifests if str(path).find(str(manifest_filter)) >= 0]
    if not manifests:
        raise SystemExit(f"No {suite_label_lower} manifests found under {manifest_root}")

    if task_mode == "attack_family_multiclass" and not multiclass_attack_names:
        raise SystemExit("MULTICLASS_ATTACK_NAMES is required when TASK_MODE=attack_family_multiclass.")

    print(f"Repository root: {repo_root}")
    print(f"{suite_label} manifest root: {manifest_root}")
    print(f"Tuning mode: {tuning_mode}")
    if reference_run_id is not None:
        print(f"Reference run: {reference_run_id}")
        print(f"Reference winner task_index: {reference_defaults['winner_task_index'] if reference_defaults else 'unknown'}")
    if cnn_hyperparams is not None:
        print(f"CNN hyperparams: {cnn_hyperparams}")
    print(f"Feature file: {feature_file}")
    print(f"Model: {model}")
    print(f"Task mode: {task_mode}")
    if multiclass_attack_names:
        print(f"Multiclass attack names: {' '.join(multiclass_attack_names)}")
    print(f"Features: {' '.join(features)}")
    print(f"SV_TOP_K: {spectral_sv_top_k}")
    print(f"SPECTRAL_MOMENT_SOURCE: {spectral_moment_source}")
    print(f"SPECTRAL_QV_SUM_MODE: {spectral_qv_sum_mode}")
    print(f"SPECTRAL_ENTRYWISE_DELTA_MODE: {spectral_entrywise_delta_mode}")
    print(f"CV_FOLDS: {cv_folds}")
    print(f"SELECTION_METRIC: {selection_metric}")
    print(f"CV_SEEDS: {' '.join(str(x) for x in cv_seeds)}")
    print(f"TRAIN_SPLIT: {train_split}")
    print(f"CALIBRATION_SPLIT: {calibration_split if calibration_split is not None else 'none'}")
    print(f"ACCEPTED_FPR: {' '.join(str(x) for x in accepted_fpr) if accepted_fpr else 'none'}")
    print(f"SPLIT_BY_FOLDER: {1 if split_by_folder else 0}")
    print(f"RANK_LABEL_WEIGHT_LOSS: {1 if rank_label_weight_loss else 0}")
    print(f"Manifest count: {len(manifests)}")
    if manifest_filter:
        print(f"Manifest filter: {manifest_filter}")
    print(f"DRY_RUN: {'enabled' if dry_run else 'disabled'}")

    submitted = 0
    for idx, manifest in enumerate(manifests, start=1):
        base_run_id = manifest.stem
        run_id = f"{run_id_prefix}/{base_run_id}" if run_id_prefix else base_run_id
        slurm_safe_run_id = run_id.replace("/", "__")
        rel_manifest = manifest.relative_to(repo_root) if manifest.is_relative_to(repo_root) else manifest
        print(f"[{idx}/{len(manifests)}] {base_run_id} <- {rel_manifest}")

        if dry_run:
            print("  prepare: python -m upeftguard.cli run supervised --stage prepare ...")
            if tuning_mode == "fixed-reference":
                print(f"  freeze:  reference winner from {reference_run_id}")
                print("  worker:  sbatch single-task worker")
            else:
                print(f"  grid:    CNN hyperparams from {cnn_hyperparams}")
                print("  worker:  sbatch worker array over all grid tasks")
            print("  finalize: sbatch dependent finalize")
            continue

        run_supervised_pipeline(
            manifest_json=manifest,
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=run_id,
            model_name=model,
            spectral_features=features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            stream_block_size=131072,
            dtype_name="float32",
            cv_folds=cv_folds,
            random_state=42,
            train_split_percent=train_split,
            calibration_split_percent=calibration_split,
            accepted_fpr=accepted_fpr,
            split_by_folder=split_by_folder,
            cv_random_states=cv_seeds,
            n_jobs=-1,
            score_percentiles=None,
            feature_file=_resolve_feature_spec(str(feature_file), repo_root=repo_root),
            tuning_executor="slurm_array",
            slurm_partition=slurm_partition,
            slurm_max_concurrent="auto",
            slurm_cpus_per_task="auto",
            finalize_export_shards=1,
            stage="prepare",
            run_dir=None,
            task_index=None,
            task_mode=task_mode,
            multiclass_attack_names=multiclass_attack_names or None,
            cnn_hyperparams=cnn_hyperparams,
            rank_label_weight_loss=rank_label_weight_loss,
            skip_feature_importance=False,
            selection_metric=selection_metric,
        )

        run_dir = output_root / "supervised" / run_id
        tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
        if tuning_mode == "fixed-reference":
            assert reference_run_id is not None
            assert reference_defaults is not None
            _lock_tuning_manifest(
                tuning_manifest_path=tuning_manifest_path,
                reference_run_id=reference_run_id,
                reference_task_index=int(reference_defaults["winner_task_index"]),
                winner_params=dict(reference_defaults["winner_params"]),
                extra_task_params=({"rank_label_weight_loss": True} if rank_label_weight_loss else None),
            )

        default_cpus, default_max_concurrent, default_score_percentiles = _runtime_defaults(tuning_manifest_path)
        worker_cpus = int(
            args.worker_cpus_per_task
            or os.environ.get("SLURM_CPUS_PER_TASK_OVERRIDE")
            or default_cpus
            or 4
        )
        worker_task_count = 1 if tuning_mode == "fixed-reference" else _tuning_task_count(tuning_manifest_path)
        worker_max_concurrent = int(max(1, min(worker_task_count, int(default_max_concurrent or worker_task_count))))
        finalize_extra_args: list[str] = []
        if score_percentiles:
            finalize_extra_args.extend(["--score-percentiles", *[str(float(x)) for x in score_percentiles]])
        if skip_feature_importance:
            finalize_extra_args.append("--skip-feature-importance")
        elif default_score_percentiles:
            finalize_extra_args.extend(
                ["--score-percentiles", *[str(float(x)) for x in default_score_percentiles]]
            )

        worker_wrap = _wrap_command(
            [
                "source",
                str(conda_sh),
                "&&",
                "conda",
                "activate",
                conda_env,
                "&&",
                "cd",
                str(repo_root),
                "&&",
                "python",
                "-m",
                "upeftguard.cli",
                "run",
                "supervised",
                "--stage",
                "worker",
                "--run-dir",
                str(run_dir),
                "--task-index",
                "0",
                "--n-jobs",
                str(worker_cpus),
            ]
        )
        if tuning_mode == "per-holdout-grid":
            worker_wrap = worker_wrap.replace("--task-index 0", "--task-index ${SLURM_ARRAY_TASK_ID}")
        finalize_wrap = _wrap_command(
            [
                "source",
                str(conda_sh),
                "&&",
                "conda",
                "activate",
                conda_env,
                "&&",
                "cd",
                str(repo_root),
                "&&",
                "python",
                "-m",
                "upeftguard.cli",
                "run",
                "supervised",
                "--stage",
                "finalize",
                "--run-dir",
                str(run_dir),
                *finalize_extra_args,
            ]
        )
        worker_command = [
            "sbatch",
            "--parsable",
            "--partition",
            slurm_partition,
            "--cpus-per-task",
            str(worker_cpus),
            "--job-name",
            f"upeftguard_supervised_worker_{slurm_safe_run_id}",
            "--output",
            str(
                slurm_log_dir
                / (
                    f"supervised_worker_{slurm_safe_run_id}_%j.out"
                    if tuning_mode == "fixed-reference"
                    else f"supervised_worker_{slurm_safe_run_id}_%A_%a.out"
                )
            ),
            "--error",
            str(
                slurm_log_dir
                / (
                    f"supervised_worker_{slurm_safe_run_id}_%j.err"
                    if tuning_mode == "fixed-reference"
                    else f"supervised_worker_{slurm_safe_run_id}_%A_%a.err"
                )
            ),
            "--wrap",
            worker_wrap,
        ]
        if tuning_mode == "per-holdout-grid":
            worker_command[worker_command.index("--job-name"):worker_command.index("--job-name")] = [
                "--array",
                f"0-{worker_task_count - 1}%{worker_max_concurrent}",
            ]
        worker_job_id = _submit_sbatch(worker_command, repo_root=repo_root)
        finalize_job_id = _submit_sbatch(
            [
                "sbatch",
                "--parsable",
                "--partition",
                slurm_partition,
                "--cpus-per-task",
                "4",
                "--dependency",
                f"afterok:{worker_job_id}",
                "--job-name",
                f"upeftguard_supervised_finalize_{slurm_safe_run_id}",
                "--output",
                str(slurm_log_dir / f"supervised_finalize_{slurm_safe_run_id}_%j.out"),
                "--error",
                str(slurm_log_dir / f"supervised_finalize_{slurm_safe_run_id}_%j.err"),
                "--wrap",
                finalize_wrap,
            ],
            repo_root=repo_root,
        )
        print(f"  prepared run_dir {run_dir}")
        if tuning_mode == "fixed-reference":
            print(f"  locked to reference winner from {reference_run_id}")
        else:
            print(f"  kept CNN grid with {worker_task_count} task(s)")
        print(f"  worker job id: {worker_job_id}")
        print(f"  finalize job id: {finalize_job_id}")
        submitted += 1

    if dry_run:
        print(f"Dry run complete: {len(manifests)} manifest(s) enumerated.")
    else:
        print(f"Submission complete: {submitted} {suite_label_lower} job(s) submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
