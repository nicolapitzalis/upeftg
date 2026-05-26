#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_FILE = (
    REPO_ROOT / "runs/feature_extract/list2_features-merged-cnn-sv-only/merged/spectral_features.npy"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs"
DEFAULT_DATASET_ROOT = Path("/models/n.pitzalis/unsupervised-peftguard/data")

SINGLE_DATASET_GRID = REPO_ROOT / "manifests/cnn_hyperparams/cnn_1d_single_dataset_small_grid.json"
ATTACK_LOO_GRID = REPO_ROOT / "manifests/cnn_hyperparams/cnn_1d_attack_family_loo_capped24.json"
RANK_LOO_GRID = REPO_ROOT / "manifests/cnn_hyperparams/cnn_1d_dann_rank_leave_one_out_medium.json"

CNN_SV_ONLY_FEATURES = (
    "energy",
    "kurtosis",
    "l1_norm",
    "l2_norm",
    "linf_norm",
    "mean_abs",
    "concentration_of_energy",
    "sv_topk",
    "stable_rank",
    "spectral_entropy",
    "effective_rank",
)

SINGLE_DATASET_RUNS = (
    "single_dataset_cnn_ag_news_ripple",
    "single_dataset_cnn_ag_news_insertsent",
    "single_dataset_cnn_ag_news_stybkd",
    "single_dataset_cnn_ag_news_syntactic",
    "single_dataset_cnn_imdb_ripple",
    "single_dataset_cnn_imdb_insertsent",
    "single_dataset_cnn_imdb_stybkd",
    "single_dataset_cnn_imdb_syntactic",
    "single_dataset_cnn_squad_insertsent",
    "single_dataset_cnn_tba",
    "single_dataset_cnn_tbh",
    "single_dataset_cnn_tbh_adalora",
    "single_dataset_cnn_tbh_dora",
    "single_dataset_cnn_tbh_lora_plus",
    "single_dataset_cnn_tbh_qlora",
    "single_dataset_cnn_tbh_rank8",
    "single_dataset_cnn_tbh_rank16",
    "single_dataset_cnn_tbh_rank32",
    "single_dataset_cnn_tbh_rank64",
    "single_dataset_cnn_tbh_rank128",
    "single_dataset_cnn_tbh_rank512",
    "single_dataset_cnn_tbh_rank1024",
    "single_dataset_cnn_tbh_rank2048",
    "single_dataset_cnn_tbh_flan_t5_xl",
    "single_dataset_cnn_tbh_llama2_13b",
    "single_dataset_cnn_tbh_qwen1_5_7b",
    "single_dataset_cnn_roberta_base_imdb_insertsent",
)

LEAVE_ONE_OUT_ATTACK_BINARY_RUNS = (
    "leave_one_out_attack_family_binary_cnn_cal20_ts100__holdout_RIPPLE",
    "leave_one_out_attack_family_binary_cnn_cal20_ts100__holdout_insertsent",
    "leave_one_out_attack_family_binary_cnn_cal20_ts100__holdout_stybkd",
    "leave_one_out_attack_family_binary_cnn_cal20_ts100__holdout_syntactic",
    "leave_one_out_attack_family_binary_cnn/holdout_attack_family_RIPPLE_rank256_qv",
    "leave_one_out_attack_family_binary_cnn/holdout_attack_family_insertsent_rank256_qv",
    "leave_one_out_attack_family_binary_cnn/holdout_attack_family_stybkd_rank256_qv",
    "leave_one_out_attack_family_binary_cnn/holdout_attack_family_syntactic_rank256_qv",
)

LEAVE_ONE_OUT_ATTACK_MULTICLASS_RUNS = (
    "leave_one_out_attack_family_multiclass_cnn/holdout_attack_family_RIPPLE_rank256_qv",
    "leave_one_out_attack_family_multiclass_cnn/holdout_attack_family_insertsent_rank256_qv",
    "leave_one_out_attack_family_multiclass_cnn/holdout_attack_family_stybkd_rank256_qv",
    "leave_one_out_attack_family_multiclass_cnn/holdout_attack_family_syntactic_rank256_qv",
)

# The all_requested report names the standard-loss rank holdouts as no_dann.
# Weighted-rank-loss holdouts are intentionally not included.
TBH_STANDARD_RANK_LOO_RUNS = (
    "tbh_cnn_no_dann_train_all_except_rank8_medium_cal20_ts100",
    "tbh_cnn_no_dann_train_all_except_rank16_medium_cal20_ts100",
    "tbh_cnn_no_dann_train_all_except_rank32_medium_cal20_ts100",
    "tbh_cnn_no_dann_train_all_except_rank64_medium_cal20_ts100",
    "tbh_cnn_no_dann_train_all_except_rank128_medium_cal20_ts100",
    "tbh_cnn_no_dann_train_all_except_rank512_medium_cal20_ts100",
    "tbh_cnn_no_dann_train_all_except_rank1024_medium_cal20_ts100",
    "tbh_cnn_no_dann_train_all_except_rank2048_medium_cal20_ts100",
)

LEAVE_ONE_OUT_ADAPTER_RUNS = (
    "leave_one_out_adapter_cnn_split_by_folder_cal20_ts100__holdout_adalora",
    "leave_one_out_adapter_cnn_split_by_folder_cal20_ts100__holdout_dora",
    "leave_one_out_adapter_cnn_split_by_folder_cal20_ts100__holdout_lora",
    "leave_one_out_adapter_cnn_split_by_folder_cal20_ts100__holdout_lora_plus",
    "leave_one_out_adapter_cnn_split_by_folder_cal20_ts100__holdout_qlora",
)

LEAVE_ONE_OUT_ARCHITECTURE_RUNS = (
    "leave_one_out_architecture_cnn_split_by_folder_cal20_ts100__holdout_flan_t5_xl",
    "leave_one_out_architecture_cnn_split_by_folder_cal20_ts100__holdout_llama2_13b",
    "leave_one_out_architecture_cnn_split_by_folder_cal20_ts100__holdout_llama2_7b",
    "leave_one_out_architecture_cnn_split_by_folder_cal20_ts100__holdout_qwen1.5_7b",
)

TBH_TBA_ZERO_SHOT_RANK_TRANSFER_RUNS = (
    "cnn_tbh_tba_zero_shot_r256_to_rank8",
    "cnn_tbh_tba_zero_shot_r256_to_rank16",
    "cnn_tbh_tba_zero_shot_r256_to_rank32",
    "cnn_tbh_tba_zero_shot_r256_to_rank64",
    "cnn_tbh_tba_zero_shot_r256_to_rank128",
    "cnn_tbh_tba_zero_shot_r256_to_rank512",
    "cnn_tbh_tba_zero_shot_r256_to_rank1024",
    "cnn_tbh_tba_zero_shot_r256_to_rank2048",
)


@dataclass(frozen=True)
class RunSpec:
    source_run_id: str
    family: str
    cnn_hyperparams: Path


@dataclass(frozen=True)
class PlannedRun:
    spec: RunSpec
    destination_run_id: str
    output_dir: Path
    command: tuple[str, ...]
    n_candidates: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the hardcoded SV-only CNN comparison batch requested from "
            "all_requested_cnn_results_report.md, with every CNN grid capped at 24 candidates."
        )
    )
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--partition", default="extra")
    parser.add_argument("--worker-cpus", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based plan index to start submitting from. Useful for resuming an interrupted submitter.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only-family",
        choices=[
            "single_dataset",
            "loo_attack_binary",
            "loo_attack_multiclass",
            "tbh_rank_standard",
            "loo_adapter",
            "loo_architecture",
            "tbh_tba_zero_shot_rank_transfer",
        ],
        nargs="+",
        default=None,
    )
    return parser


def _specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    specs.extend(RunSpec(run, "single_dataset", SINGLE_DATASET_GRID) for run in SINGLE_DATASET_RUNS)
    specs.extend(RunSpec(run, "loo_attack_binary", ATTACK_LOO_GRID) for run in LEAVE_ONE_OUT_ATTACK_BINARY_RUNS)
    specs.extend(RunSpec(run, "loo_attack_multiclass", ATTACK_LOO_GRID) for run in LEAVE_ONE_OUT_ATTACK_MULTICLASS_RUNS)
    specs.extend(RunSpec(run, "tbh_rank_standard", RANK_LOO_GRID) for run in TBH_STANDARD_RANK_LOO_RUNS)
    specs.extend(RunSpec(run, "loo_adapter", SINGLE_DATASET_GRID) for run in LEAVE_ONE_OUT_ADAPTER_RUNS)
    specs.extend(RunSpec(run, "loo_architecture", SINGLE_DATASET_GRID) for run in LEAVE_ONE_OUT_ARCHITECTURE_RUNS)
    specs.extend(
        RunSpec(run, "tbh_tba_zero_shot_rank_transfer", SINGLE_DATASET_GRID)
        for run in TBH_TBA_ZERO_SHOT_RANK_TRANSFER_RUNS
    )
    return specs


def _candidate_count(hyperparams_path: Path) -> int:
    axes = json.loads(hyperparams_path.read_text())
    count = 1
    for values in axes.values():
        if isinstance(values, list):
            count *= len(values)
    return int(count)


def _source_config_path(source_run_id: str) -> Path:
    return REPO_ROOT / "runs/supervised" / source_run_id / "run_config.json"


def _destination_run_id(source_run_id: str) -> str:
    if "/" in source_run_id:
        suite, child = source_run_id.split("/", 1)
        return f"{suite}_sv_only_capped24/{child}"
    return f"{source_run_id}_sv_only_capped24"


def _task_mode(cfg: dict) -> str:
    task = cfg.get("task") or {}
    return str(task.get("task_mode") or "binary")


def _multiclass_attack_names(cfg: dict) -> list[str]:
    task = cfg.get("task") or {}
    return [str(name) for name in task.get("class_names", []) if str(name) != "clean"]


def _random_state(cfg: dict) -> int:
    data_split = cfg.get("data_split")
    if isinstance(data_split, dict) and data_split.get("random_state") is not None:
        return int(data_split["random_state"])
    calibration_split = cfg.get("calibration_split")
    if isinstance(calibration_split, dict) and calibration_split.get("random_state") is not None:
        return int(calibration_split["random_state"])
    return 42


def _list_arg(values: Sequence[object] | None) -> list[str]:
    if not values:
        return []
    return [str(value) for value in values]


def _validate_feature_bundle(feature_file: Path) -> None:
    if not feature_file.exists():
        raise SystemExit(f"SV-only feature file not found: {feature_file}")
    metadata_path = feature_file.with_name("spectral_metadata.json")
    if not metadata_path.exists():
        raise SystemExit(f"SV-only feature metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    params = metadata.get("extractor_params") or {}
    expected = {
        "spectral_features": list(CNN_SV_ONLY_FEATURES),
        "spectral_sv_top_k": 8,
        "spectral_moment_source": "sv",
        "spectral_qv_sum_mode": "append",
    }
    mismatches = {
        key: {"expected": value, "actual": params.get(key)}
        for key, value in expected.items()
        if params.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"Feature bundle is not the expected SV-only CNN bundle: {mismatches}")


def _command_for_spec(
    *,
    spec: RunSpec,
    cfg: dict,
    destination_run_id: str,
    feature_file: Path,
    output_root: Path,
    dataset_root: Path,
    partition: str,
    worker_cpus: int | None,
    max_concurrent: int | None,
) -> tuple[str, ...]:
    task_mode = _task_mode(cfg)
    command: list[str] = [
        sys.executable,
        "-m",
        "upeftguard.cli",
        "experiment",
        "supervised-slurm",
        "--manifest-json",
        str(Path(str(cfg["manifest_json"]))),
        "--feature-file",
        str(feature_file),
        "--features",
        *CNN_SV_ONLY_FEATURES,
        "--run-id",
        destination_run_id,
        "--model",
        str(cfg.get("model_name") or "cnn_1d"),
        "--task-mode",
        task_mode,
        "--cnn-hyperparams",
        str(spec.cnn_hyperparams),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--partition",
        str(partition),
        "--cv-seeds",
        *_list_arg(cfg.get("cv_random_states") or [42]),
        "--random-state",
        str(_random_state(cfg)),
        "--train-split",
        str(int(cfg.get("train_split_percent") or 100)),
        "--spectral-moment-source",
        "sv",
        "--spectral-qv-sum-mode",
        "append",
        "--spectral-sv-top-k",
        "8",
        "--skip-feature-importance",
    ]

    if task_mode == "attack_family_multiclass":
        command.extend(["--multiclass-attack-names", *_multiclass_attack_names(cfg)])

    calibration_split = cfg.get("calibration_split_percent")
    if calibration_split is not None:
        command.extend(["--calibration-split", str(int(calibration_split))])
        accepted_fprs = _list_arg(cfg.get("accepted_fprs"))
        if accepted_fprs:
            command.extend(["--accepted-fpr", *accepted_fprs])

    score_percentiles = _list_arg(cfg.get("score_percentiles"))
    if score_percentiles:
        command.extend(["--score-percentiles", *score_percentiles])

    if bool(cfg.get("split_by_folder")):
        command.append("--split-by-folder")
    if bool(cfg.get("class_weight_loss")):
        command.append("--class-weight-loss")
    if bool(cfg.get("rank_label_weight_loss")):
        command.append("--rank-label-weight-loss")
    if worker_cpus is not None:
        command.extend(["--worker-cpus", str(worker_cpus)])
    if max_concurrent is not None:
        command.extend(["--max-concurrent", str(max_concurrent)])

    return tuple(command)


def _build_plan(args: argparse.Namespace) -> list[PlannedRun]:
    feature_file = args.feature_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    _validate_feature_bundle(feature_file)

    plan: list[PlannedRun] = []
    families = set(args.only_family or [])
    for spec in _specs():
        if families and spec.family not in families:
            continue
        source_config_path = _source_config_path(spec.source_run_id)
        if not source_config_path.exists():
            raise SystemExit(f"Missing source run config for {spec.source_run_id}: {source_config_path}")
        if not spec.cnn_hyperparams.exists():
            raise SystemExit(f"Missing hyperparameter grid: {spec.cnn_hyperparams}")
        n_candidates = _candidate_count(spec.cnn_hyperparams)
        if n_candidates > 24:
            raise SystemExit(
                f"Grid exceeds 24 candidates for {spec.source_run_id}: "
                f"{n_candidates} candidates in {spec.cnn_hyperparams}"
            )

        cfg = json.loads(source_config_path.read_text())
        destination_run_id = _destination_run_id(spec.source_run_id)
        output_dir = output_root / "supervised" / destination_run_id
        command = _command_for_spec(
            spec=spec,
            cfg=cfg,
            destination_run_id=destination_run_id,
            feature_file=feature_file,
            output_root=output_root,
            dataset_root=dataset_root,
            partition=args.partition,
            worker_cpus=args.worker_cpus,
            max_concurrent=args.max_concurrent,
        )
        plan.append(
            PlannedRun(
                spec=spec,
                destination_run_id=destination_run_id,
                output_dir=output_dir,
                command=command,
                n_candidates=n_candidates,
            )
        )

    start_index = max(1, int(args.start_index))
    if start_index > 1:
        plan = plan[start_index - 1 :]
    if args.limit is not None:
        plan = plan[: max(0, int(args.limit))]
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _build_plan(args)
    skip_existing = not args.no_skip_existing

    family_counts: dict[str, int] = {}
    for item in plan:
        family_counts[item.spec.family] = family_counts.get(item.spec.family, 0) + 1

    print(f"Planned hardcoded SV-only capped24 submissions: {len(plan)}")
    print("Families: " + ", ".join(f"{name}={count}" for name, count in sorted(family_counts.items())))

    submitted = 0
    skipped = 0
    for idx, item in enumerate(plan, start=1):
        report_json = item.output_dir / "reports/supervised_report.json"
        if skip_existing and report_json.exists():
            skipped += 1
            print(f"[{idx:02d}/{len(plan):02d}] skip existing {item.destination_run_id}")
            continue
        print(
            f"[{idx:02d}/{len(plan):02d}] {item.spec.family}: "
            f"{item.spec.source_run_id} -> {item.destination_run_id} "
            f"({item.n_candidates} candidates)"
        )
        print(f"  {shlex.join(item.command)}")
        if args.dry_run:
            continue
        completed = subprocess.run(item.command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
        submitted += 1

    if args.dry_run:
        print("Dry run only; no jobs submitted.")
    else:
        print(f"Submitted {submitted} jobs; skipped {skipped} existing completed runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
