#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


DEFAULT_MULTICLASS_ATTACK_NAMES = ["RIPPLE", "insertsent", "stybkd", "syntactic"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    storage_root = Path(
        os.environ.get(
            "UPEFTGUARD_STORAGE_ROOT",
            f"/models/{user}/unsupervised-peftguard",
        )
    )
    return Path(
        os.environ.get(
            "DATASET_ROOT",
            os.environ.get("UPEFTGUARD_DATA_ROOT", str(storage_root / "data")),
        )
    )


def _default_run_id(manifest_json: Path, model: str, task_mode: str) -> str:
    stem = manifest_json.stem
    suffix = f"_{task_mode}" if task_mode != "binary" else ""
    return f"{stem}_{model}{suffix}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the generic sbatch/supervised_array.sh workflow with the important supervised flags."
        )
    )
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model", type=str, default="cnn_1d")
    parser.add_argument(
        "--task-mode",
        choices=["binary", "attack_family_multiclass"],
        default="binary",
    )
    parser.add_argument(
        "--multiclass-attack-names",
        nargs="+",
        default=None,
        help=(
            "Required for attack_family_multiclass. Defaults to the canonical set "
            "RIPPLE insertsent stybkd syntactic."
        ),
    )
    parser.add_argument("--cnn-hyperparams", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--output-root", type=Path, default=_repo_root() / "runs")
    parser.add_argument("--partition", type=str, default="extra")
    parser.add_argument(
        "--worker-cpus",
        type=int,
        default=None,
        help="Worker CPUs per tuning task. Default: use all CPUs on a node in the selected partition.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent tuning workers. Default: use all nodes in the selected partition.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-split", type=int, default=100)
    parser.add_argument("--calibration-split", type=int, default=None)
    parser.add_argument("--accepted-fpr", nargs="+", type=float, default=None)
    parser.add_argument("--split-by-folder", action="store_true")
    parser.add_argument("--score-percentiles", nargs="+", type=float, default=None)
    parser.add_argument("--spectral-sv-top-k", type=int, default=8)
    parser.add_argument(
        "--spectral-moment-source",
        choices=["entrywise", "sv", "both"],
        default="sv",
    )
    parser.add_argument(
        "--spectral-qv-sum-mode",
        choices=["none", "append", "only"],
        default="none",
    )
    parser.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=["auto", "dense", "stream"],
        default="auto",
    )
    parser.add_argument("--conda-sh", type=Path, default=Path("/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh"))
    parser.add_argument("--conda-env", type=str, default="upeftg")
    parser.add_argument("--log-dir", type=Path, default=_repo_root() / "logs")
    parser.add_argument("--sbatch-script", type=Path, default=_repo_root() / "sbatch" / "supervised_array.sh")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-feature-importance",
        dest="skip_feature_importance",
        action="store_true",
        help="Skip winner feature-importance export during finalize.",
    )
    parser.add_argument(
        "--keep-feature-importance",
        dest="skip_feature_importance",
        action="store_false",
        help="Do not force-skip winner feature-importance export.",
    )
    parser.set_defaults(skip_feature_importance=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    repo_root = _repo_root()

    manifest_json = args.manifest_json.expanduser().resolve()
    feature_file = args.feature_file.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    sbatch_script = args.sbatch_script.expanduser().resolve()
    cnn_hyperparams = args.cnn_hyperparams.expanduser().resolve() if args.cnn_hyperparams else None

    if not manifest_json.exists():
        raise SystemExit(f"Manifest not found: {manifest_json}")
    if not feature_file.exists():
        raise SystemExit(f"Feature file not found: {feature_file}")
    if not sbatch_script.exists():
        raise SystemExit(f"sbatch script not found: {sbatch_script}")
    if cnn_hyperparams is not None and not cnn_hyperparams.exists():
        raise SystemExit(f"CNN hyperparams file not found: {cnn_hyperparams}")
    if args.task_mode == "attack_family_multiclass":
        attack_names = list(args.multiclass_attack_names or DEFAULT_MULTICLASS_ATTACK_NAMES)
    else:
        attack_names = []
    if args.task_mode == "attack_family_multiclass" and not attack_names:
        raise SystemExit("--multiclass-attack-names is required for attack_family_multiclass")
    if (args.calibration_split is None) != (args.accepted_fpr is None):
        raise SystemExit("--calibration-split and --accepted-fpr must either both be set or both be omitted")

    run_id = args.run_id or _default_run_id(
        manifest_json=manifest_json,
        model=str(args.model),
        task_mode=str(args.task_mode),
    )
    skip_feature_importance = args.skip_feature_importance
    if skip_feature_importance is None:
        skip_feature_importance = str(args.model) == "cnn_1d"

    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(repo_root),
            "CONDA_SH": str(args.conda_sh.expanduser().resolve()),
            "CONDA_ENV": str(args.conda_env),
            "MANIFEST_JSON": str(manifest_json),
            "FEATURE_FILE": str(feature_file),
            "FEATURES": " ".join(str(x) for x in args.features),
            "DATASET_ROOT": str(dataset_root),
            "OUTPUT_ROOT": str(output_root),
            "RUN_ID": str(run_id),
            "MODEL": str(args.model),
            "TASK_MODE": str(args.task_mode),
            "MULTICLASS_ATTACK_NAMES": " ".join(str(x) for x in attack_names),
            "SV_TOP_K": str(int(args.spectral_sv_top_k)),
            "SPECTRAL_MOMENT_SOURCE": str(args.spectral_moment_source),
            "SPECTRAL_QV_SUM_MODE": str(args.spectral_qv_sum_mode),
            "SPECTRAL_ENTRYWISE_DELTA_MODE": str(args.spectral_entrywise_delta_mode),
            "CV_FOLDS": str(int(args.cv_folds)),
            "CV_SEEDS": " ".join(str(int(x)) for x in args.cv_seeds),
            "RANDOM_STATE": str(int(args.random_state)),
            "TRAIN_SPLIT": str(int(args.train_split)),
            "SPLIT_BY_FOLDER": "1" if bool(args.split_by_folder) else "0",
            "SLURM_PARTITION": str(args.partition),
            "SLURM_CPUS_PER_TASK_REQUEST": (
                str(int(args.worker_cpus)) if args.worker_cpus is not None else "auto"
            ),
            "SLURM_MAX_CONCURRENT_REQUEST": (
                str(int(args.max_concurrent)) if args.max_concurrent is not None else "auto"
            ),
            "SLURM_LOG_DIR": str(log_dir),
            "PIPELINE_MODE": "prepare_only" if bool(args.prepare_only) else "full",
            "SKIP_FEATURE_IMPORTANCE": "1" if bool(skip_feature_importance) else "0",
        }
    )
    if cnn_hyperparams is not None:
        env["CNN_HYPERPARAMS"] = str(cnn_hyperparams)
    if args.calibration_split is not None:
        env["CALIBRATION_SPLIT"] = str(int(args.calibration_split))
        env["ACCEPTED_FPR"] = " ".join(str(float(x)) for x in args.accepted_fpr or [])
    if args.score_percentiles:
        env["SCORE_PERCENTILES"] = " ".join(str(float(x)) for x in args.score_percentiles)

    command = ["sbatch", str(sbatch_script)]
    print("Submitting supervised Slurm workflow with:")
    print(f"  run_id={run_id}")
    print(f"  manifest_json={manifest_json}")
    print(f"  model={args.model}")
    print(f"  task_mode={args.task_mode}")
    if attack_names:
        print(f"  multiclass_attack_names={' '.join(attack_names)}")
    if cnn_hyperparams is not None:
        print(f"  cnn_hyperparams={cnn_hyperparams}")
    print(f"  feature_file={feature_file}")
    print(
        "  worker_cpus="
        + (str(args.worker_cpus) if args.worker_cpus is not None else "auto(use all CPUs on a partition node)")
    )
    print(
        "  max_concurrent="
        + (str(args.max_concurrent) if args.max_concurrent is not None else "auto(use all nodes in the partition)")
    )
    print(f"  skip_feature_importance={bool(skip_feature_importance)}")
    print(f"  command={shlex.join(command)}")

    if args.dry_run:
        return 0

    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
