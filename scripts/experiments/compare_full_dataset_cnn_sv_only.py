#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from upeftguard.unsupervised.reporting import compute_offline_metrics


RUNS_ROOT = REPO_ROOT / "runs" / "supervised"
SUMMARY_DIR = RUNS_ROOT / "summaries"
BASELINE_RUN = RUNS_ROOT / "supervised_list2_cnn_all_features_small_grid_split_by_folder_train80_cal20"
SV_ONLY_RUN = RUNS_ROOT / "supervised_list2_cnn_sv_only_small_grid_split_by_folder_train80_cal20"
OUTPUT_MD = SUMMARY_DIR / "full_dataset_cnn_sv_only_direct_comparison.md"
OUTPUT_CSV = SUMMARY_DIR / "full_dataset_cnn_sv_only_direct_comparison.csv"
ACCEPTED_FPR = 0.05


@dataclass(frozen=True)
class GroupSpec:
    section: str
    item: str
    positive_subsets: tuple[str, ...]
    clean_subsets: tuple[str, ...] | None = None


def _subset_from_model_name(model_name: str) -> str:
    return model_name.split("_label", 1)[0]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "reports" / "inference_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing inference scores: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            model_name = str(row["model_name"])
            rows.append(
                {
                    "model_name": model_name,
                    "subset": _subset_from_model_name(model_name),
                    "label": int(float(row["label"])),
                    "score": float(row["score"]),
                }
            )
    return rows


def _selected_threshold(run_dir: Path, accepted_fpr: float) -> float:
    path = run_dir / "reports" / "selected_threshold.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected threshold: {path}")
    payload = _load_json(path)
    for row in payload.get("selections", []):
        if isinstance(row, dict) and abs(float(row.get("accepted_fpr")) - accepted_fpr) < 1e-12:
            return float(row["threshold"])
    raise ValueError(f"Could not find accepted_fpr={accepted_fpr} in {path}")


def _select_group_rows(rows: list[dict[str, Any]], spec: GroupSpec | None) -> list[dict[str, Any]]:
    if spec is None:
        return list(rows)
    positive_subsets = set(spec.positive_subsets)
    clean_subsets = set(spec.clean_subsets) if spec.clean_subsets is not None else positive_subsets
    selected = []
    for row in rows:
        subset = str(row["subset"])
        label = int(row["label"])
        if label == 1 and subset in positive_subsets:
            selected.append(row)
        elif label == 0 and subset in clean_subsets:
            selected.append(row)
    return selected


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute metrics for an empty group")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    offline = compute_offline_metrics(labels, scores)
    flagged = scores >= float(threshold)
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(flagged & positives))
    fp = int(np.sum(flagged & negatives))
    tn = int(np.sum((~flagged) & negatives))
    fn = int(np.sum((~flagged) & positives))
    n_flagged = tp + fp
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    return {
        "n": int(labels.size),
        "clean": n_neg,
        "backdoor": n_pos,
        "auroc": offline.get("auroc"),
        "auprc": offline.get("auprc"),
        "precision_at_num_positives": offline.get("precision_at_num_positives"),
        "recall_at_5_fpr": float(tp / n_pos) if n_pos else None,
        "precision_at_5_fpr": float(tp / n_flagged) if n_flagged else None,
        "false_positive_rate_at_5_fpr": float(fp / n_neg) if n_neg else None,
        "accuracy_at_5_fpr": float((tp + tn) / max(1, labels.size)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _run_metrics(run_dir: Path, specs: list[GroupSpec]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _load_rows(run_dir)
    threshold = _selected_threshold(run_dir, ACCEPTED_FPR)
    output: dict[tuple[str, str], dict[str, Any]] = {
        ("Overall", "All inference samples"): _metrics(rows, threshold)
    }
    for spec in specs:
        output[(spec.section, spec.item)] = _metrics(_select_group_rows(rows, spec), threshold)
    return output


def _specs() -> list[GroupSpec]:
    ag_news_clean = ("llama2_7b_ag_news_insertsent_rank256_qv",)
    imdb_clean = ("llama2_7b_imdb_insertsent_rank256_qv",)
    attack_specs = [
        GroupSpec("Attack", "ag_news RIPPLE", ("llama2_7b_ag_news_RIPPLE_rank256_qv",), ag_news_clean),
        GroupSpec("Attack", "ag_news insertsent", ("llama2_7b_ag_news_insertsent_rank256_qv",)),
        GroupSpec("Attack", "ag_news stybkd", ("llama2_7b_ag_news_stybkd_rank256_qv",), ag_news_clean),
        GroupSpec("Attack", "ag_news syntactic", ("llama2_7b_ag_news_syntactic_rank256_qv",), ag_news_clean),
        GroupSpec("Attack", "imdb RIPPLE", ("llama2_7b_imdb_RIPPLE_rank256_qv",), imdb_clean),
        GroupSpec("Attack", "imdb insertsent", ("llama2_7b_imdb_insertsent_rank256_qv",)),
        GroupSpec("Attack", "imdb stybkd", ("llama2_7b_imdb_stybkd_rank256_qv",), imdb_clean),
        GroupSpec("Attack", "imdb syntactic", ("llama2_7b_imdb_syntactic_rank256_qv",), imdb_clean),
        GroupSpec("Attack", "squad insertsent", ("llama2_7b_squad_insertsent_rank256_qv",)),
        GroupSpec("Attack", "toxic backdoors alpaca", ("llama2_7b_toxic_backdoors_alpaca_rank256_qv",)),
        GroupSpec("Attack", "toxic backdoors hard", ("llama2_7b_toxic_backdoors_hard_rank256_qv",)),
    ]
    adapter_specs = [
        GroupSpec("Adapter", "lora", ("llama2_7b_toxic_backdoors_hard_rank256_qv",)),
        GroupSpec("Adapter", "adalora", ("llama2_7b_adalora_toxic_backdoors_hard_rank8_qv",)),
        GroupSpec("Adapter", "dora", ("llama2_7b_dora_toxic_backdoors_hard_rank256_qv",)),
        GroupSpec("Adapter", "lora+", ("llama2_7b_lora_plus_toxic_backdoors_hard_rank8_qv",)),
        GroupSpec("Adapter", "qlora", ("llama2_7b_qlora_toxic_backdoors_hard_rank256_qv",)),
    ]
    rank_specs = [
        GroupSpec("Rank", str(rank), (f"llama2_7b_toxic_backdoors_hard_rank{rank}_qv",))
        for rank in (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    ]
    architecture_specs = [
        GroupSpec("Architecture", "flan_t5_xl", ("flan_t5_xl_toxic_backdoors_hard_rank256_qv",)),
        GroupSpec("Architecture", "llama2_13b", ("llama2_13b_toxic_backdoors_hard_rank256_qv",)),
        GroupSpec("Architecture", "llama2_7b", ("llama2_7b_toxic_backdoors_hard_rank256_qv",)),
        GroupSpec("Architecture", "qwen1.5_7b", ("qwen1.5_7b_toxic_backdoors_hard_rank256_qv",)),
        GroupSpec("Architecture", "roberta_base", ("roberta_base_imdb_insertsent_rank16_qv",)),
    ]
    return attack_specs + adapter_specs + rank_specs + architecture_specs


def _metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "features" / "spectral_metadata.json"
    metadata = _load_json(metadata_path)
    report = _load_json(run_dir / "reports" / "supervised_report.json")
    winner = report.get("tuning", {}).get("winner", {})
    return {
        "tensor_shape": metadata.get("tensor_shape"),
        "feature_dim": metadata.get("feature_dim"),
        "spectral_moment_source": metadata.get("spectral_moment_source"),
        "winner_task": winner.get("task_index"),
        "winner_params": winner.get("params"),
        "cv_mean": winner.get("selection_metric_mean"),
        "cv_std": winner.get("selection_metric_std"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _delta(new_value: Any, old_value: Any) -> str:
    if new_value is None or old_value is None:
        return "-"
    return f"{float(new_value) - float(old_value):+.3f}"


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _comparison_rows(
    baseline: dict[tuple[str, str], dict[str, Any]],
    sv_only: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for key in baseline:
        base = baseline[key]
        sv = sv_only[key]
        section, item = key
        row: dict[str, Any] = {
            "section": section,
            "item": item,
            "n": base["n"],
            "clean": base["clean"],
            "backdoor": base["backdoor"],
        }
        for metric in (
            "auroc",
            "recall_at_5_fpr",
            "precision_at_5_fpr",
            "accuracy_at_5_fpr",
            "false_positive_rate_at_5_fpr",
            "auprc",
            "precision_at_num_positives",
        ):
            row[f"all_features_{metric}"] = base.get(metric)
            row[f"sv_only_{metric}"] = sv.get(metric)
            row[f"delta_{metric}"] = (
                None
                if base.get(metric) is None or sv.get(metric) is None
                else float(sv[metric]) - float(base[metric])
            )
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "section",
        "item",
        "n",
        "clean",
        "backdoor",
        "all_features_auroc",
        "sv_only_auroc",
        "delta_auroc",
        "all_features_recall_at_5_fpr",
        "sv_only_recall_at_5_fpr",
        "delta_recall_at_5_fpr",
        "all_features_precision_at_5_fpr",
        "sv_only_precision_at_5_fpr",
        "delta_precision_at_5_fpr",
        "all_features_accuracy_at_5_fpr",
        "sv_only_accuracy_at_5_fpr",
        "delta_accuracy_at_5_fpr",
        "all_features_false_positive_rate_at_5_fpr",
        "sv_only_false_positive_rate_at_5_fpr",
        "delta_false_positive_rate_at_5_fpr",
        "all_features_auprc",
        "sv_only_auprc",
        "delta_auprc",
        "all_features_precision_at_num_positives",
        "sv_only_precision_at_num_positives",
        "delta_precision_at_num_positives",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _append_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| Item | N | Clean | Backdoor | AUROC all | AUROC SV | Delta | Rec@5 all | Rec@5 SV | Delta | Prec@5 all | Prec@5 SV | Delta | Acc@5 all | Acc@5 SV | Delta |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        lines.append(
            "| {item} | {n} | {clean} | {backdoor} | {auroc_all} | {auroc_sv} | {auroc_delta} | {rec_all} | {rec_sv} | {rec_delta} | {prec_all} | {prec_sv} | {prec_delta} | {acc_all} | {acc_sv} | {acc_delta} |".format(
                item=row["item"],
                n=int(row["n"]),
                clean=int(row["clean"]),
                backdoor=int(row["backdoor"]),
                auroc_all=_fmt(row["all_features_auroc"]),
                auroc_sv=_fmt(row["sv_only_auroc"]),
                auroc_delta=_delta(row["sv_only_auroc"], row["all_features_auroc"]),
                rec_all=_fmt(row["all_features_recall_at_5_fpr"]),
                rec_sv=_fmt(row["sv_only_recall_at_5_fpr"]),
                rec_delta=_delta(row["sv_only_recall_at_5_fpr"], row["all_features_recall_at_5_fpr"]),
                prec_all=_fmt(row["all_features_precision_at_5_fpr"]),
                prec_sv=_fmt(row["sv_only_precision_at_5_fpr"]),
                prec_delta=_delta(row["sv_only_precision_at_5_fpr"], row["all_features_precision_at_5_fpr"]),
                acc_all=_fmt(row["all_features_accuracy_at_5_fpr"]),
                acc_sv=_fmt(row["sv_only_accuracy_at_5_fpr"]),
                acc_delta=_delta(row["sv_only_accuracy_at_5_fpr"], row["all_features_accuracy_at_5_fpr"]),
            )
        )
    lines.append("")


def _write_markdown(rows: list[dict[str, Any]], baseline_meta: dict[str, Any], sv_meta: dict[str, Any]) -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Full-Dataset CNN SV-Only Direct Comparison",
        "",
        f"Baseline run: `{BASELINE_RUN.relative_to(REPO_ROOT)}`",
        f"SV-only run: `{SV_ONLY_RUN.relative_to(REPO_ROOT)}`",
        "",
        f"Metrics use each run's calibration-selected threshold for accepted FPR {100 * ACCEPTED_FPR:.0f}%.",
        "",
        "## Feature Inputs",
        "",
        "| Variant | Moment source | Tensor shape | Feature dim | Winner task | CV AUROC | Winner params |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
        "| All features | {moment} | `{shape}` | {dim} | {task} | {cv} | `{params}` |".format(
            moment=baseline_meta.get("spectral_moment_source"),
            shape=baseline_meta.get("tensor_shape"),
            dim=baseline_meta.get("feature_dim"),
            task=baseline_meta.get("winner_task"),
            cv=_fmt(baseline_meta.get("cv_mean")),
            params=baseline_meta.get("winner_params"),
        ),
        "| SV-only | {moment} | `{shape}` | {dim} | {task} | {cv} | `{params}` |".format(
            moment=sv_meta.get("spectral_moment_source"),
            shape=sv_meta.get("tensor_shape"),
            dim=sv_meta.get("feature_dim"),
            task=sv_meta.get("winner_task"),
            cv=_fmt(sv_meta.get("cv_mean")),
            params=sv_meta.get("winner_params"),
        ),
        "",
    ]
    by_section: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_section.setdefault(str(row["section"]), []).append(row)
    for section in ("Overall", "Attack", "Adapter", "Rank", "Architecture"):
        if section in by_section:
            _append_table(lines, section, by_section[section])
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    specs = _specs()
    baseline = _run_metrics(BASELINE_RUN, specs)
    sv_only = _run_metrics(SV_ONLY_RUN, specs)
    rows = _comparison_rows(baseline, sv_only)
    _write_csv(rows)
    _write_markdown(rows, _metadata(BASELINE_RUN), _metadata(SV_ONLY_RUN))
    print(f"Wrote {OUTPUT_MD.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
