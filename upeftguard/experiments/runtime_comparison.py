from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..features.spectral import DEFAULT_SPECTRAL_FEATURES
from .paper_qv_reference import PAPER_QV_SELECTED_FEATURES
from ..supervised.interfaces import SUPERVISED_TASK_MODE_BINARY
from ..supervised.pipeline import (
    _build_single_manifest_folder_label_split,
    _default_binary_task_spec,
    _labels_from_items,
)
from ..utilities.core.manifest import parse_single_manifest_json_by_model_name, resolve_manifest_path
from ..utilities.core.serialization import json_ready


DEFAULT_MANIFEST_JSON = Path("manifests") / "single_datasets" / "llama2_7b_toxic_backdoors_hard.json"
DEFAULT_OUTPUT_DIR = Path("runs") / "runtime_comparison" / "toxic_backdoors_hard_rank256"
DEFAULT_GENERATED_MANIFEST_NAME = "runtime_tbh_rank256_train80_infer20.json"
DEFAULT_FEATURE_RUN_ID = "runtime_tbh_rank256_features"
DEFAULT_CNN_FEATURE_RUN_ID = "runtime_tbh_rank256_features_cnn"
DEFAULT_CNN_RUN_ID = "runtime_tbh_rank256_cnn_winner"
DEFAULT_PAPER_QV_BASE_FEATURE_RUN_ID = "runtime_tbh_rank256_paper_qv_spectral_features"
DEFAULT_PAPER_QV_RUN_ID = "runtime_tbh_rank256_paper_qv"
DEFAULT_PAPER_QV_FEATURE_RUN_ID = "runtime_tbh_rank256_paper_qv_features"
DEFAULT_CNN_HYPERPARAMS = Path("manifests") / "cnn_hyperparams" / "cnn_1d_list2_features_cnn_winner.json"
DEFAULT_TRAIN_SPLIT_PERCENT = 80
DEFAULT_CALIBRATION_SPLIT_PERCENT = 20
DEFAULT_RANDOM_STATE = 42
DEFAULT_ACCEPTED_FPRS = (0.01, 0.05, 0.1)
MANUAL_EXTERNAL_FILENAME = "manual_external_method.json"


@dataclass(frozen=True)
class RuntimeSegment:
    method: str
    feature_seconds: float | None
    training_seconds: float | None
    status: str
    source: dict[str, Any]

    @property
    def complete(self) -> bool:
        return (
            self.status == "ok"
            and _valid_seconds(self.feature_seconds) is not None
            and _valid_seconds(self.training_seconds) is not None
        )

    @property
    def total_seconds(self) -> float | None:
        if not self.complete:
            return None
        return float(self.feature_seconds or 0.0) + float(self.training_seconds or 0.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    return raw.resolve() if raw.is_absolute() else (_repo_root() / raw).resolve()


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return path


def _valid_seconds(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


def _first_valid_seconds(*values: Any) -> float | None:
    for value in values:
        parsed = _valid_seconds(value)
        if parsed is not None:
            return parsed
    return None


def _safe_read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json_object(path)


def _resolve_run_dir(raw: Path | str, *, default_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    direct = (_repo_root() / candidate).resolve()
    if direct.exists():
        return direct
    return (_repo_root() / default_root / candidate).resolve()


def _parse_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _elapsed_between(start_raw: Any, end_raw: Any) -> float | None:
    start = _parse_timestamp(start_raw)
    end = _parse_timestamp(end_raw)
    if start is None or end is None:
        return None
    elapsed = float((end - start).total_seconds())
    return elapsed if elapsed >= 0.0 else None


def _read_group_pipeline_seconds(feature_run_dir: Path) -> float | None:
    group_reports = sorted(feature_run_dir.glob("schema_groups/*/merged/spectral_merge_report.json"))
    values = [
        _valid_seconds(_safe_read_json_object(path).get("pipeline_elapsed_seconds"))
        for path in group_reports
    ]
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    schema_group_report = _safe_read_json_object(feature_run_dir / "merged" / "schema_group_merge_report.json")
    return float(max(values) + float(_valid_seconds(schema_group_report.get("merge_elapsed_seconds")) or 0.0))


def feature_extraction_wall_seconds(feature_run_dir: Path) -> tuple[float, dict[str, Any]]:
    merge_report_path = feature_run_dir / "merged" / "spectral_merge_report.json"
    merge_report = _safe_read_json_object(merge_report_path)
    value = _first_valid_seconds(
        merge_report.get("pipeline_elapsed_seconds"),
        _read_group_pipeline_seconds(feature_run_dir),
        _safe_read_json_object(feature_run_dir / "timings.json").get("feature_extract_elapsed_seconds"),
        _safe_read_json_object(feature_run_dir / "reports" / "feature_extraction_report.json").get("elapsed_seconds"),
    )
    if value is None:
        raise FileNotFoundError(
            "Could not resolve feature extraction wall-clock seconds from "
            f"{feature_run_dir}"
        )
    return value, {
        "run_dir": str(feature_run_dir),
        "merge_report": str(merge_report_path) if merge_report_path.exists() else None,
        "timing_field": "pipeline_elapsed_seconds",
    }


def aggregation_wall_seconds(cnn_feature_run_dir: Path) -> tuple[float, dict[str, Any]]:
    report_path = cnn_feature_run_dir / "merged" / "spectral_aggregation_report.json"
    report = _safe_read_json_object(report_path)
    value = _valid_seconds(report.get("aggregation_elapsed_seconds"))
    if value is None:
        raise FileNotFoundError(f"Could not resolve aggregation_elapsed_seconds from {report_path}")
    return value, {"run_dir": str(cnn_feature_run_dir), "report": str(report_path)}


def supervised_training_wall_seconds(supervised_run_dir: Path) -> tuple[float, dict[str, Any]]:
    timings_path = supervised_run_dir / "timings.json"
    timings = _safe_read_json_object(timings_path)
    value = _first_valid_seconds(
        timings.get("supervised_training_method_seconds"),
        timings.get("finalize_elapsed_seconds"),
        timings.get("final_model_fit_seconds"),
    )
    source: dict[str, Any] = {
        "run_dir": str(supervised_run_dir),
        "timings": str(timings_path) if timings_path.exists() else None,
        "timing_field": None,
    }
    if value is not None:
        source["timing_field"] = (
            "supervised_training_method_seconds"
            if "supervised_training_method_seconds" in timings
            else "finalize_elapsed_seconds"
        )
        return value, source

    task_values: list[float] = []
    for task_path in sorted((supervised_run_dir / "reports" / "tuning_tasks").glob("task_*.json")):
        elapsed = _valid_seconds(_safe_read_json_object(task_path).get("elapsed_seconds"))
        if elapsed is not None:
            task_values.append(elapsed)
    if task_values:
        source["timing_field"] = "max_tuning_task_elapsed_seconds"
        return float(max(task_values)), source

    raise FileNotFoundError(f"Could not resolve supervised training wall-clock seconds from {supervised_run_dir}")


def paper_qv_wall_segments(paper_qv_run_dir: Path) -> tuple[float, float, dict[str, Any]]:
    timings_path = paper_qv_run_dir / "timings.json"
    timings = _safe_read_json_object(timings_path)
    feature_seconds = _valid_seconds(timings.get("feature_derivation_seconds"))
    method_seconds = _valid_seconds(timings.get("paper_qv_method_seconds"))
    if method_seconds is None:
        total_seconds = _valid_seconds(timings.get("total_seconds"))
        method_seconds = total_seconds
        if total_seconds is not None and feature_seconds is not None:
            method_seconds = max(0.0, float(total_seconds) - float(feature_seconds))
    if feature_seconds is None:
        feature_seconds = 0.0
    if method_seconds is None:
        raise FileNotFoundError(f"Could not resolve Paper-QV timing fields from {timings_path}")
    return float(feature_seconds), float(method_seconds), {
        "run_dir": str(paper_qv_run_dir),
        "timings": str(timings_path) if timings_path.exists() else None,
    }


def ensure_manual_external_file(path: Path) -> Path:
    if path.exists():
        return path
    payload = {
        "method": "External method",
        "feature_extraction_seconds": None,
        "training_seconds": None,
        "notes": "Fill both timing fields to include this method in the runtime plot.",
    }
    return _write_json(path, payload)


def manual_external_segment(path: Path) -> RuntimeSegment:
    ensure_manual_external_file(path)
    payload = _safe_read_json_object(path)
    feature_seconds = _valid_seconds(payload.get("feature_extraction_seconds"))
    training_seconds = _valid_seconds(payload.get("training_seconds"))
    return RuntimeSegment(
        method=str(payload.get("method") or "External method"),
        feature_seconds=feature_seconds,
        training_seconds=training_seconds,
        status="ok" if feature_seconds is not None and training_seconds is not None else "placeholder",
        source={"manual_file": str(path)},
    )


def generate_joint_manifest(
    *,
    manifest_json: Path,
    output_path: Path,
    train_split_percent: int = DEFAULT_TRAIN_SPLIT_PERCENT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    resolved_manifest = resolve_manifest_path(manifest_json)
    items = parse_single_manifest_json_by_model_name(manifest_path=resolved_manifest)
    labels, labels_known, _ = _labels_from_items(
        items,
        task_spec=_default_binary_task_spec(),
    )
    if not bool(np.all(labels_known)):
        raise ValueError("Runtime comparison split generation requires known binary labels for every sample")

    train_indices, infer_indices, warnings, split_summary = _build_single_manifest_folder_label_split(
        items=items,
        labels=labels,
        train_split_percent=int(train_split_percent),
        random_state=int(random_state),
    )
    train_entries = [str(items[int(idx)].raw_entry) for idx in train_indices.tolist()]
    infer_entries = [str(items[int(idx)].raw_entry) for idx in infer_indices.tolist()]
    payload = {
        "source_manifest": str(resolved_manifest),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_mode": SUPERVISED_TASK_MODE_BINARY,
        "train_split_percent": int(train_split_percent),
        "random_state": int(random_state),
        "split_summary": split_summary,
        "warnings": warnings,
        "train": train_entries,
        "infer": infer_entries,
    }
    _write_json(output_path, payload)
    return payload


def collect_runtime_segments(
    *,
    base_feature_run_dir: Path,
    cnn_feature_run_dir: Path,
    cnn_run_dir: Path,
    paper_qv_base_feature_run_dir: Path | None = None,
    paper_qv_run_dir: Path,
    manual_external_path: Path,
) -> list[RuntimeSegment]:
    base_feature_seconds, base_feature_source = feature_extraction_wall_seconds(base_feature_run_dir)
    aggregation_seconds, aggregation_source = aggregation_wall_seconds(cnn_feature_run_dir)
    cnn_training_seconds, cnn_training_source = supervised_training_wall_seconds(cnn_run_dir)
    paper_base_feature_dir = paper_qv_base_feature_run_dir or base_feature_run_dir
    paper_base_feature_seconds, paper_base_feature_source = feature_extraction_wall_seconds(paper_base_feature_dir)
    paper_feature_seconds, paper_method_seconds, paper_source = paper_qv_wall_segments(paper_qv_run_dir)

    return [
        RuntimeSegment(
            method="Z-PEFT",
            feature_seconds=float(base_feature_seconds + aggregation_seconds),
            training_seconds=cnn_training_seconds,
            status="ok",
            source={
                "base_feature": base_feature_source,
                "aggregation": aggregation_source,
                "training": cnn_training_source,
            },
        ),
        RuntimeSegment(
            method="WSD-in-LoRA",
            feature_seconds=float(paper_base_feature_seconds + paper_feature_seconds),
            training_seconds=paper_method_seconds,
            status="ok",
            source={
                "base_feature": paper_base_feature_source,
                "paper_qv": paper_source,
            },
        ),
        manual_external_segment(manual_external_path),
    ]


def _segment_to_row(segment: RuntimeSegment) -> dict[str, Any]:
    total = segment.total_seconds
    return {
        "method": segment.method,
        "feature_seconds": segment.feature_seconds,
        "training_seconds": segment.training_seconds,
        "total_seconds": total,
        "feature_hours": (float(segment.feature_seconds) / 3600.0 if segment.feature_seconds is not None else None),
        "training_hours": (float(segment.training_seconds) / 3600.0 if segment.training_seconds is not None else None),
        "total_hours": (float(total) / 3600.0 if total is not None else None),
        "status": segment.status,
        "source": segment.source,
    }


def write_runtime_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    fieldnames = [
        "method",
        "feature_seconds",
        "training_seconds",
        "total_seconds",
        "feature_hours",
        "training_hours",
        "total_hours",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _format_duration(seconds: float) -> str:
    value = float(seconds)
    if value < 1.0:
        return f"{value:.2f} s"
    if value < 60.0:
        return f"{value:.1f} s"
    if value < 3600.0:
        return f"{value / 60.0:.1f} min"
    return f"{value / 3600.0:.2f} h"


def plot_runtime_breakdown(*, rows: list[dict[str, Any]], output_png: Path, output_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete_rows = [row for row in rows if row.get("status") == "ok" and row.get("total_seconds") is not None]
    if not complete_rows:
        raise ValueError("No complete runtime rows available for plotting")

    methods = [str(row["method"]) for row in complete_rows]
    feature_seconds = np.asarray([float(row["feature_seconds"]) for row in complete_rows], dtype=np.float64)
    training_seconds = np.asarray([float(row["training_seconds"]) for row in complete_rows], dtype=np.float64)
    positive_values = np.concatenate(
        [
            feature_seconds[feature_seconds > 0.0],
            training_seconds[training_seconds > 0.0],
        ]
    )
    min_positive = float(np.min(positive_values)) if positive_values.size else 1.0
    display_floor = max(min_positive / 3.0, 1e-3)

    fig_width = max(6.8, 1.9 * len(methods) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4.9))
    x = np.arange(len(methods))
    width = 0.34
    feature_color = "#4C78A8"
    training_color = "#F58518"

    feature_display = np.maximum(feature_seconds, display_floor)
    training_display = np.maximum(training_seconds, display_floor)
    feature_zero = feature_seconds <= 0.0
    training_zero = training_seconds <= 0.0

    feature_bars = ax.bar(
        x - width / 2,
        feature_display,
        width=width,
        color=feature_color,
        label="Feature extraction",
    )
    training_bars = ax.bar(
        x + width / 2,
        training_display,
        width=width,
        color=training_color,
        label="Training",
    )
    for is_zero, bar in zip(feature_zero.tolist(), feature_bars):
        if is_zero:
            bar.set_alpha(0.25)
            bar.set_hatch("//")
    for is_zero, bar in zip(training_zero.tolist(), training_bars):
        if is_zero:
            bar.set_alpha(0.25)
            bar.set_hatch("//")
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time (seconds, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha="center")
    upper = max(float(np.max(np.concatenate([feature_display, training_display]))) * 4.0, 1.0)
    ax.set_ylim(display_floor / 1.6, upper)
    tick_candidates = [0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
    tick_values = [tick for tick in tick_candidates if display_floor / 1.6 <= tick <= upper]
    if tick_values:
        ax.set_yticks(tick_values)
        ax.set_yticklabels(
            [
                "0.1" if tick < 1.0 else (f"{tick / 1000.0:g}k" if tick >= 1000.0 else f"{tick:g}")
                for tick in tick_values
            ]
        )
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    for idx, seconds in enumerate(feature_seconds.tolist()):
        shown = float(feature_display[idx])
        ax.text(
            idx - width / 2,
            shown * 1.22,
            _format_duration(float(seconds)),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for idx, seconds in enumerate(training_seconds.tolist()):
        shown = float(training_display[idx])
        ax.text(
            idx + width / 2,
            shown * 1.22,
            _format_duration(float(seconds)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.16)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def collect_and_plot(
    *,
    output_dir: Path,
    feature_run: Path | str,
    cnn_feature_run: Path | str,
    cnn_run: Path | str,
    paper_qv_base_feature_run: Path | str | None = None,
    paper_qv_run: Path | str,
    manual_external_file: Path | None = None,
) -> dict[str, Path]:
    resolved_output_dir = _resolve_path(output_dir)
    manual_path = (
        _resolve_path(manual_external_file)
        if manual_external_file is not None
        else resolved_output_dir / MANUAL_EXTERNAL_FILENAME
    )
    segments = collect_runtime_segments(
        base_feature_run_dir=_resolve_run_dir(feature_run, default_root=Path("runs") / "feature_extract"),
        cnn_feature_run_dir=_resolve_run_dir(cnn_feature_run, default_root=Path("runs") / "feature_extract"),
        cnn_run_dir=_resolve_run_dir(cnn_run, default_root=Path("runs") / "supervised"),
        paper_qv_base_feature_run_dir=(
            _resolve_run_dir(paper_qv_base_feature_run, default_root=Path("runs") / "feature_extract")
            if paper_qv_base_feature_run is not None
            else None
        ),
        paper_qv_run_dir=_resolve_run_dir(paper_qv_run, default_root=Path("runs") / "paper_qv_reference"),
        manual_external_path=manual_path,
    )
    rows = [_segment_to_row(segment) for segment in segments]

    csv_path = resolved_output_dir / "runtime_breakdown.csv"
    json_path = resolved_output_dir / "runtime_breakdown.json"
    png_path = resolved_output_dir / "runtime_breakdown.png"
    pdf_path = resolved_output_dir / "runtime_breakdown.pdf"
    write_runtime_csv(csv_path, rows)
    _write_json(
        json_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "timing_basis": "wall_clock_seconds",
            "rows": rows,
        },
    )
    plot_runtime_breakdown(rows=rows, output_png=png_path, output_pdf=pdf_path)
    return {
        "csv": csv_path,
        "json": json_path,
        "png": png_path,
        "pdf": pdf_path,
        "manual_external": manual_path,
    }


def _feature_words() -> str:
    return " ".join(str(x) for x in DEFAULT_SPECTRAL_FEATURES)


def _paper_qv_feature_words() -> str:
    return " ".join(str(x) for x in PAPER_QV_SELECTED_FEATURES)


def workflow_commands(*, output_dir: Path) -> list[str]:
    generated_manifest = Path(output_dir) / "generated_manifests" / DEFAULT_GENERATED_MANIFEST_NAME
    feature_path = Path("runs") / "feature_extract" / DEFAULT_FEATURE_RUN_ID / "merged" / "spectral_features.npy"
    cnn_feature_path = Path("runs") / "feature_extract" / DEFAULT_CNN_FEATURE_RUN_ID / "merged" / "spectral_features.npy"
    paper_qv_feature_path = (
        Path("runs") / "feature_extract" / DEFAULT_PAPER_QV_BASE_FEATURE_RUN_ID / "merged" / "spectral_features.npy"
    )
    accepted_fprs = " ".join(str(x) for x in DEFAULT_ACCEPTED_FPRS)
    features = _feature_words()
    paper_qv_features = _paper_qv_feature_words()
    return [
        (
            "python -m upeftguard.cli experiment runtime-comparison generate-manifest "
            f"--output-dir {output_dir}"
        ),
        (
            f"MANIFEST_JSON={DEFAULT_MANIFEST_JSON} RUN_ID={DEFAULT_FEATURE_RUN_ID} "
            "SPECTRAL_MOMENT_SOURCE=both "
            "SPECTRAL_QV_SUM_MODE=append "
            "SPECTRAL_ENTRYWISE_DELTA_MODE=dense "
            "sbatch sbatch/feature_extract_array.sh"
        ),
        (
            "python -m upeftguard.cli util aggregate-features "
            f"--feature-file {feature_path} "
            f"--output-filename {DEFAULT_CNN_FEATURE_RUN_ID} "
            "--layout layer_sequence "
            f"--features {features} "
            "--spectral-qv-sum-mode append"
        ),
        (
            f"MANIFEST_JSON={generated_manifest} FEATURE_FILE={cnn_feature_path} "
            f"FEATURES=\"{features}\" RUN_ID={DEFAULT_CNN_RUN_ID} MODEL=cnn_1d "
            "SPECTRAL_MOMENT_SOURCE=both SPECTRAL_QV_SUM_MODE=append "
            "SPECTRAL_ENTRYWISE_DELTA_MODE=dense "
            f"CNN_HYPERPARAMS={DEFAULT_CNN_HYPERPARAMS} TRAIN_SPLIT=100 "
            f"CALIBRATION_SPLIT={DEFAULT_CALIBRATION_SPLIT_PERCENT} ACCEPTED_FPR=\"{accepted_fprs}\" "
            "SPLIT_BY_FOLDER=0 SKIP_FEATURE_IMPORTANCE=1 "
            "sbatch sbatch/supervised_array.sh"
        ),
        (
            f"MANIFEST_JSON={DEFAULT_MANIFEST_JSON} RUN_ID={DEFAULT_PAPER_QV_BASE_FEATURE_RUN_ID} "
            f"FEATURES=\"{paper_qv_features}\" SV_TOP_K=1 "
            "SPECTRAL_MOMENT_SOURCE=entrywise "
            "SPECTRAL_QV_SUM_MODE=only "
            "SPECTRAL_ENTRYWISE_DELTA_MODE=dense "
            "sbatch sbatch/feature_extract_array.sh"
        ),
        (
            "python -m upeftguard.cli experiment paper-qv-reference "
            f"--feature-file {paper_qv_feature_path} "
            f"--feature-output-run {DEFAULT_PAPER_QV_FEATURE_RUN_ID} "
            f"--run-id {DEFAULT_PAPER_QV_RUN_ID} "
            f"--manifest-json {generated_manifest} "
            f"--calibration-split-percent {DEFAULT_CALIBRATION_SPLIT_PERCENT}"
        ),
        (
            "python -m upeftguard.cli experiment runtime-comparison collect "
            f"--output-dir {output_dir}"
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and plot runtime comparisons for the paper methods.")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("generate-manifest", help="Generate the shared train/infer manifest")
    manifest.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    manifest.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    manifest.add_argument("--output-path", type=Path, default=None)
    manifest.add_argument("--train-split", type=int, default=DEFAULT_TRAIN_SPLIT_PERCENT)
    manifest.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    collect = sub.add_parser("collect", help="Collect timings and write the runtime plot")
    collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    collect.add_argument("--feature-run", type=Path, default=Path(DEFAULT_FEATURE_RUN_ID))
    collect.add_argument("--cnn-feature-run", type=Path, default=Path(DEFAULT_CNN_FEATURE_RUN_ID))
    collect.add_argument("--cnn-run", type=Path, default=Path(DEFAULT_CNN_RUN_ID))
    collect.add_argument("--paper-qv-base-feature-run", type=Path, default=Path(DEFAULT_PAPER_QV_BASE_FEATURE_RUN_ID))
    collect.add_argument("--paper-qv-run", type=Path, default=Path(DEFAULT_PAPER_QV_RUN_ID))
    collect.add_argument("--manual-external-file", type=Path, default=None)

    commands = sub.add_parser("print-commands", help="Print the recommended run workflow")
    commands.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-manifest":
        output_dir = _resolve_path(args.output_dir)
        output_path = (
            _resolve_path(args.output_path)
            if args.output_path is not None
            else output_dir / "generated_manifests" / DEFAULT_GENERATED_MANIFEST_NAME
        )
        payload = generate_joint_manifest(
            manifest_json=args.manifest_json,
            output_path=output_path,
            train_split_percent=int(args.train_split),
            random_state=int(args.random_state),
        )
        print(f"Generated manifest: {output_path}")
        print(f"Train rows: {len(payload['train'])}")
        print(f"Infer rows: {len(payload['infer'])}")
        return 0

    if args.command == "collect":
        outputs = collect_and_plot(
            output_dir=args.output_dir,
            feature_run=args.feature_run,
            cnn_feature_run=args.cnn_feature_run,
            cnn_run=args.cnn_run,
            paper_qv_base_feature_run=args.paper_qv_base_feature_run,
            paper_qv_run=args.paper_qv_run,
            manual_external_file=args.manual_external_file,
        )
        print(f"Runtime CSV: {outputs['csv']}")
        print(f"Runtime JSON: {outputs['json']}")
        print(f"Runtime plot PNG: {outputs['png']}")
        print(f"Runtime plot PDF: {outputs['pdf']}")
        print(f"Manual external timing file: {outputs['manual_external']}")
        return 0

    if args.command == "print-commands":
        for command in workflow_commands(output_dir=args.output_dir):
            print(command)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
