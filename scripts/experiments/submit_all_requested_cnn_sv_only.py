#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "runs/supervised/summaries/all_requested_cnn_results_report.md"
DEFAULT_FEATURE_FILE = (
    REPO_ROOT / "runs/feature_extract/list2_features-merged-cnn-sv-only/merged/spectral_features.npy"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs"
DEFAULT_DATASET_ROOT = Path("/models/n.pitzalis/unsupervised-peftguard/data")
SUPERVISED_PATH_RE = re.compile(r"`(runs/supervised/[^`]+)`")
DEFAULT_CNN_SPECTRAL_FEATURES = (
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

SKIPPED_SECTION_TITLES = {
    "Per-Folder Grouped CNN Results",
    "Paper QV Reference Baseline Comparisons",
}
SKIPPED_SOURCE_RUNS = {
    "runs/supervised/single_dataset_cnn_all_datasets",
}
RUN_ID_RENAMES = {
    "supervised_list2_cnn_all_features_small_grid_split_by_folder_train80_cal20": (
        "supervised_list2_cnn_sv_only_small_grid_split_by_folder_train80_cal20"
    ),
}


@dataclass(frozen=True)
class PlannedRun:
    source_path: str
    source_config: Path
    run_id: str
    command: tuple[str, ...]
    output_dir: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit SV-only CNN counterparts for the supervised experiments listed in "
            "all_requested_cnn_results_report.md. The one-for-all grouped-dataset run "
            "is skipped."
        )
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--partition", default="extra")
    parser.add_argument("--worker-cpus", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_CNN_SPECTRAL_FEATURES),
        help="Concrete spectral feature groups to select from the CNN feature bundle.",
    )
    parser.add_argument("--sv-top-k", type=int, default=8)
    parser.add_argument(
        "--qv-sum-mode",
        choices=["none", "append", "only"],
        default="append",
        help="Must match the SV-only CNN feature bundle. The existing bundle uses append.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned submissions only.")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Submit even when the destination run already has reports/supervised_report.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Submit at most this many planned jobs after filtering. Useful for a small test wave.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(
            "Optional source run-id/path filters. Matches exact values or substrings; "
            "prefix with exact: to force exact matching."
        ),
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=None,
        help=(
            "Optional source run-id/path filters to skip. Matches exact values or substrings; "
            "prefix with exact: to force exact matching."
        ),
    )
    return parser


def _strip_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
    if not match:
        return None
    return len(match.group(1)), match.group(2)


def _is_under_skipped_section(section_stack: list[tuple[int, str]]) -> bool:
    for level, title in section_stack:
        if level == 2 and title in SKIPPED_SECTION_TITLES:
            return True
    return False


def _source_paths_from_report(report: Path) -> list[str]:
    section_stack: list[tuple[int, str]] = []
    paths: list[str] = []
    seen: set[str] = set()

    for raw_line in report.read_text().splitlines():
        heading = _strip_heading(raw_line)
        if heading is not None:
            level, title = heading
            section_stack = [(lvl, name) for lvl, name in section_stack if lvl < level]
            section_stack.append((level, title))
            continue

        if _is_under_skipped_section(section_stack) or not raw_line.startswith("|"):
            continue

        for source_path in SUPERVISED_PATH_RE.findall(raw_line):
            source_path = source_path.strip()
            if source_path in SKIPPED_SOURCE_RUNS:
                continue
            if source_path not in seen:
                seen.add(source_path)
                paths.append(source_path)

    return paths


def _source_config_path(source_path: str) -> Path:
    return REPO_ROOT / source_path / "run_config.json"


def _sv_only_run_id(source_path: str) -> str:
    if not source_path.startswith("runs/supervised/"):
        raise ValueError(f"Unexpected source path: {source_path}")
    run_id = source_path.removeprefix("runs/supervised/")
    if run_id in RUN_ID_RENAMES:
        return RUN_ID_RENAMES[run_id]
    if "/" in run_id:
        suite, child = run_id.split("/", 1)
        return f"{suite}_sv_only/{child}"
    return f"{run_id}_sv_only"


def _filter_matches(value: str, item: str) -> bool:
    if item.startswith("exact:"):
        return value == item.removeprefix("exact:")
    if item.startswith("contains:"):
        return item.removeprefix("contains:") in value
    return value == item or item in value


def _matches_any(value: str, filters: Sequence[str] | None) -> bool:
    if not filters:
        return True
    return any(_filter_matches(value, item) for item in filters)


def _is_skipped(value: str, filters: Sequence[str] | None) -> bool:
    if not filters:
        return False
    return any(_filter_matches(value, item) for item in filters)


def _task_mode(cfg: dict) -> str:
    task = cfg.get("task") or {}
    return str(task.get("task_mode") or "binary")


def _multiclass_attack_names(cfg: dict) -> list[str]:
    task = cfg.get("task") or {}
    names = [str(name) for name in task.get("class_names") or []]
    return [name for name in names if name != "clean"]


def _cnn_hyperparams_path(cfg: dict) -> Path | None:
    hyperparams = cfg.get("cnn_hyperparams")
    if not isinstance(hyperparams, dict):
        return None
    path = hyperparams.get("path")
    return Path(path) if path else None


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


def _command_for_run(
    *,
    cfg: dict,
    run_id: str,
    feature_file: Path,
    output_root: Path,
    dataset_root: Path | None,
    partition: str,
    worker_cpus: int | None,
    max_concurrent: int | None,
    sv_top_k: int,
    qv_sum_mode: str,
    features: Sequence[str],
) -> tuple[str, ...]:
    manifest_json = Path(str(cfg["manifest_json"]))
    model_name = str(cfg.get("model_name") or "cnn_1d")
    task_mode = _task_mode(cfg)
    train_split = int(cfg.get("train_split_percent") or 100)
    cv_seeds = _list_arg(cfg.get("cv_random_states") or [42])
    score_percentiles = _list_arg(cfg.get("score_percentiles"))
    accepted_fprs = _list_arg(cfg.get("accepted_fprs"))
    calibration_split = cfg.get("calibration_split_percent")
    cnn_hyperparams = _cnn_hyperparams_path(cfg)
    resolved_dataset_root = dataset_root or Path(str(cfg.get("dataset_root") or DEFAULT_DATASET_ROOT))

    command: list[str] = [
        sys.executable,
        "-m",
        "upeftguard.cli",
        "experiment",
        "supervised-slurm",
        "--manifest-json",
        str(manifest_json),
        "--feature-file",
        str(feature_file),
        "--features",
        *[str(feature) for feature in features],
        "--run-id",
        run_id,
        "--model",
        model_name,
        "--task-mode",
        task_mode,
        "--dataset-root",
        str(resolved_dataset_root),
        "--output-root",
        str(output_root),
        "--partition",
        partition,
        "--cv-seeds",
        *cv_seeds,
        "--random-state",
        str(_random_state(cfg)),
        "--train-split",
        str(train_split),
        "--spectral-moment-source",
        "sv",
        "--spectral-qv-sum-mode",
        qv_sum_mode,
        "--spectral-sv-top-k",
        str(sv_top_k),
        "--skip-feature-importance",
    ]

    if task_mode == "attack_family_multiclass":
        command.extend(["--multiclass-attack-names", *_multiclass_attack_names(cfg)])
    if cnn_hyperparams is not None:
        command.extend(["--cnn-hyperparams", str(cnn_hyperparams)])
    if calibration_split is not None:
        command.extend(["--calibration-split", str(int(calibration_split))])
        if accepted_fprs:
            command.extend(["--accepted-fpr", *accepted_fprs])
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
    report = args.report.expanduser().resolve()
    feature_file = args.feature_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None

    if not report.exists():
        raise SystemExit(f"Report not found: {report}")
    if not feature_file.exists():
        raise SystemExit(f"SV-only feature file not found: {feature_file}")

    planned: list[PlannedRun] = []
    for source_path in _source_paths_from_report(report):
        run_id = source_path.removeprefix("runs/supervised/")
        if not _matches_any(source_path, args.only) and not _matches_any(run_id, args.only):
            continue
        if _is_skipped(source_path, args.skip) or _is_skipped(run_id, args.skip):
            continue

        source_config = _source_config_path(source_path)
        if not source_config.exists():
            print(f"warning: skipping missing config: {source_config}", file=sys.stderr)
            continue

        cfg = json.loads(source_config.read_text())
        sv_run_id = _sv_only_run_id(source_path)
        output_dir = output_root / "supervised" / sv_run_id
        command = _command_for_run(
            cfg=cfg,
            run_id=sv_run_id,
            feature_file=feature_file,
            output_root=output_root,
            dataset_root=dataset_root,
            partition=args.partition,
            worker_cpus=args.worker_cpus,
            max_concurrent=args.max_concurrent,
            sv_top_k=args.sv_top_k,
            qv_sum_mode=args.qv_sum_mode,
            features=args.features,
        )
        planned.append(
            PlannedRun(
                source_path=source_path,
                source_config=source_config,
                run_id=sv_run_id,
                command=command,
                output_dir=output_dir,
            )
        )

    if args.limit is not None:
        planned = planned[: max(0, int(args.limit))]
    return planned


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _build_plan(args)
    skip_existing = not bool(args.no_skip_existing)

    print(f"Planned SV-only submissions: {len(plan)}")
    submitted = 0
    skipped_existing = 0
    for index, item in enumerate(plan, start=1):
        report_json = item.output_dir / "reports/supervised_report.json"
        if skip_existing and report_json.exists():
            skipped_existing += 1
            print(f"[{index:02d}/{len(plan):02d}] skip existing {item.run_id}")
            continue

        print(f"[{index:02d}/{len(plan):02d}] {item.source_path} -> {item.run_id}")
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
        print(f"Submitted {submitted} jobs; skipped {skipped_existing} existing runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
