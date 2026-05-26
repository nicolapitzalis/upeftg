from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def save_score_table(output_path: Path, model_names: list[str], results: list[dict[str, Any]]) -> None:
    score_models = [r for r in results if r.get("scores") is not None]
    if not score_models:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["index", "model_name"] + [row["variant"] for row in score_models]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, name in enumerate(model_names):
            row = {
                "index": i,
                "model_name": name,
            }
            for result in score_models:
                row[result["variant"]] = float(result["scores"][i])
            writer.writerow(row)


def _project_to_2d(z: np.ndarray) -> tuple[np.ndarray, str]:
    if z.shape[1] == 1:
        out = np.concatenate([z, np.zeros((z.shape[0], 1), dtype=z.dtype)], axis=1)
        return out, "identity-1d"

    if z.shape[1] == 2:
        return z, "identity-2d"

    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(z), "pca"


def _find_result_by_summary_item(
    results: list[dict[str, Any]],
    summary_item: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if summary_item is None:
        return None
    algo = summary_item.get("algorithm")
    variant = summary_item.get("variant")
    for row in results:
        if row.get("algorithm") == algo and row.get("variant") == variant:
            return row
    return None


def _extract_partition_offline_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("partition_offline_metrics") or {}
    return {
        "ari": metrics.get("adjusted_rand"),
        "nmi": metrics.get("normalized_mutual_info"),
        "homogeneity": metrics.get("homogeneity"),
        "v_measure": metrics.get("v_measure"),
    }


def _extract_score_offline_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("score_offline_metrics") or {}
    return {
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "precision_at_num_positives": metrics.get("precision_at_num_positives"),
        "precision_at_5": metrics.get("precision_at_5"),
        "precision_at_10": metrics.get("precision_at_10"),
    }


def _metric_axis_limits(values: np.ndarray, default_min: float, default_max: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default_min, default_max
    return float(np.min(finite)), float(np.max(finite))


def render_partition_projection_plot(
    *,
    output_path: Path,
    z: np.ndarray,
    labels_true: np.ndarray | None,
    result: dict[str, Any],
    title_prefix: str,
) -> bool:
    labels_pred = result.get("labels_pred")
    if labels_pred is None:
        return False

    z2, projection = _project_to_2d(z)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    scatter = ax.scatter(z2[:, 0], z2[:, 1], c=labels_pred, cmap="tab10", s=28, alpha=0.85)
    ax.set_title(f"{title_prefix}: {result['variant']} [{projection}]")
    ax.set_xlabel("component-1")
    ax.set_ylabel("component-2")
    plt.colorbar(scatter, ax=ax, label="predicted cluster")

    if labels_true is not None:
        mismatched = labels_true != labels_pred
        if np.any(mismatched):
            ax.scatter(
                z2[mismatched, 0],
                z2[mismatched, 1],
                marker="x",
                c="black",
                s=32,
                linewidths=0.8,
                label="label mismatch",
            )
            ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def render_partition_metrics_comparison(
    *,
    output_path: Path,
    results: list[dict[str, Any]],
) -> bool:
    rows = [
        {
            "label": f"{r['variant']} [{r['algorithm']}]",
            **_extract_partition_offline_metrics(r),
        }
        for r in results
        if r.get("partition_offline_metrics") is not None
    ]
    if not rows:
        return False

    metric_specs = [
        ("ari", "ARI"),
        ("nmi", "NMI"),
        ("homogeneity", "Homogeneity"),
        ("v_measure", "V-Measure"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, 0.45 * len(rows))))
    axes_flat = axes.flatten()

    y = np.arange(len(rows))
    labels = [row["label"] for row in rows]

    for ax, (metric_key, metric_name) in zip(axes_flat, metric_specs):
        values = np.asarray([np.nan if row[metric_key] is None else float(row[metric_key]) for row in rows], dtype=np.float64)
        valid = ~np.isnan(values)

        if np.any(valid):
            ax.barh(y[valid], values[valid], color="#2ca02c", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_title(metric_name)

        xmin, xmax = _metric_axis_limits(values, 0.0, 1.0)
        xmin = xmin - 0.05
        xmax = xmax + 0.12
        ax.set_xlim(xmin, xmax)

        ax.set_yticks(y)
        ax.set_yticklabels(labels if metric_key == "ari" else [])
        ax.invert_yaxis()

    fig.suptitle("Metrics Comparison (Partition)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def render_score_metrics_comparison(
    *,
    output_path: Path,
    results: list[dict[str, Any]],
) -> bool:
    rows = [
        {
            "label": f"{r['variant']} [{r['algorithm']}]",
            **_extract_score_offline_metrics(r),
        }
        for r in results
        if r.get("score_offline_metrics") is not None
    ]
    if not rows:
        return False

    metric_specs = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("precision_at_num_positives", "P@N+"),
        ("precision_at_5", "P@5"),
        ("precision_at_10", "P@10"),
    ]

    n_cols = 3
    n_rows = int(np.ceil(len(metric_specs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, max(5, 0.45 * len(rows))))
    axes_flat = np.array(axes).reshape(-1)

    y = np.arange(len(rows))
    labels = [row["label"] for row in rows]

    for idx, (metric_key, metric_name) in enumerate(metric_specs):
        ax = axes_flat[idx]
        values = np.asarray([np.nan if row[metric_key] is None else float(row[metric_key]) for row in rows], dtype=np.float64)
        valid = ~np.isnan(values)

        if np.any(valid):
            ax.barh(y[valid], values[valid], color="#1f77b4", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_title(metric_name)

        xmin, xmax = _metric_axis_limits(values, 0.0, 1.0)
        xmin = xmin - 0.05
        xmax = xmax + 0.12
        ax.set_xlim(xmin, xmax)

        ax.set_yticks(y)
        ax.set_yticklabels(labels if idx == 0 else [])
        ax.invert_yaxis()

    for idx in range(len(metric_specs), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Metrics Comparison (Score)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
