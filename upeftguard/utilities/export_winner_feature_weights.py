from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class GroupStat:
    count: int = 0
    abs_sum: float = 0.0
    signed_sum: float = 0.0
    top_feature_name: str = ""
    top_feature_coef: float = 0.0
    top_feature_abs: float = -1.0

    def update(self, feature_name: str, coefficient: float) -> None:
        abs_coef = float(abs(coefficient))
        self.count += 1
        self.abs_sum += abs_coef
        self.signed_sum += float(coefficient)
        if abs_coef > self.top_feature_abs:
            self.top_feature_name = feature_name
            self.top_feature_coef = float(coefficient)
            self.top_feature_abs = abs_coef


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _extract_estimator_with_coefficients(model_obj: Any) -> Any:
    if hasattr(model_obj, "coef_"):
        return model_obj

    if hasattr(model_obj, "named_steps") and model_obj.named_steps:
        last = list(model_obj.named_steps.values())[-1]
        if hasattr(last, "coef_"):
            return last

    if hasattr(model_obj, "steps") and model_obj.steps:
        last = model_obj.steps[-1][1]
        if hasattr(last, "coef_"):
            return last

    if isinstance(model_obj, dict):
        for key in ("model", "estimator", "classifier", "best_model"):
            maybe = model_obj.get(key)
            if hasattr(maybe, "coef_"):
                return maybe

    raise TypeError(
        "Could not locate an estimator with coefficients in best model artifact; "
        f"loaded object type: {type(model_obj).__name__}"
    )


def _candidate_feature_names(payload: dict[str, Any], expected_len: int) -> list[str] | None:
    paths: tuple[tuple[str, ...], ...] = (
        ("extractor", "metadata", "feature_names"),
        ("representation", "extractor_metadata", "feature_names"),
        ("feature_names",),
    )
    for path in paths:
        cur: Any = payload
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and isinstance(cur, list) and len(cur) == expected_len:
            return [str(x) for x in cur]

    # Fallback: scan for any feature_names list with matching dimensionality.
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "feature_names" and isinstance(value, list) and len(value) == expected_len:
                    return [str(x) for x in value]
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node[:50])
    return None


def _resolve_feature_names(
    *,
    manifest_payload: dict[str, Any],
    report_payload: dict[str, Any],
    estimator: Any,
    expected_len: int,
) -> list[str]:
    names = _candidate_feature_names(manifest_payload, expected_len)
    if names is None:
        names = _candidate_feature_names(report_payload, expected_len)
    if names is None and hasattr(estimator, "feature_names_in_"):
        candidate = getattr(estimator, "feature_names_in_")
        if len(candidate) == expected_len:
            names = [str(x) for x in candidate]
    if names is None:
        raise ValueError(
            "Unable to find feature names in tuning manifest/report/model metadata "
            f"for feature dimension {expected_len}"
        )
    return names


def _write_feature_csv(
    *,
    out_path: Path,
    feature_names: list[str],
    coefficients: np.ndarray,
    class0: str,
    class1: str,
) -> None:
    n_features = int(coefficients.shape[0])
    order_abs = np.argsort(np.abs(coefficients))[::-1]
    order_pos = np.argsort(coefficients)[::-1]
    order_neg = np.argsort(coefficients)

    rank_abs = np.empty(n_features, dtype=np.int64)
    rank_pos = np.empty(n_features, dtype=np.int64)
    rank_neg = np.empty(n_features, dtype=np.int64)
    rank_abs[order_abs] = np.arange(1, n_features + 1, dtype=np.int64)
    rank_pos[order_pos] = np.arange(1, n_features + 1, dtype=np.int64)
    rank_neg[order_neg] = np.arange(1, n_features + 1, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rank_abs",
            "rank_positive",
            "rank_negative",
            "feature_index",
            "feature_name",
            "coefficient",
            "abs_coefficient",
            "direction",
            "toward_class",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in order_abs:
            coef = float(coefficients[idx])
            direction = "zero"
            toward_class = "neutral"
            if coef > 0:
                direction = "positive"
                toward_class = class1
            elif coef < 0:
                direction = "negative"
                toward_class = class0
            writer.writerow(
                {
                    "rank_abs": int(rank_abs[idx]),
                    "rank_positive": int(rank_pos[idx]),
                    "rank_negative": int(rank_neg[idx]),
                    "feature_index": int(idx),
                    "feature_name": feature_names[idx],
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                    "direction": direction,
                    "toward_class": toward_class,
                }
            )


def _write_group_csv(
    *,
    out_path: Path,
    feature_names: list[str],
    coefficients: np.ndarray,
    mode: str,
) -> None:
    if mode not in {"metric", "block"}:
        raise ValueError(f"Unsupported grouping mode: {mode}")

    stats: dict[str, GroupStat] = {}
    for name, coef in zip(feature_names, coefficients):
        if mode == "metric":
            group = name.rsplit(".", maxsplit=1)[-1]
        else:
            group = name.rsplit(".", maxsplit=1)[0] if "." in name else name
        stat = stats.setdefault(group, GroupStat())
        stat.update(name, float(coef))

    rows = sorted(stats.items(), key=lambda item: item[1].abs_sum, reverse=True)
    total_abs_weight_sum = float(sum(stat.abs_sum for _, stat in rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group",
            "feature_count",
            "abs_weight_sum",
            "abs_weight_mean",
            "signed_weight_sum",
            "signed_weight_mean",
            "top_feature_name",
            "top_feature_coefficient",
            "top_feature_abs_coefficient",
        ]
        if mode == "metric":
            fieldnames.extend(
                [
                    "abs_weight_share_total",
                    "abs_weight_share_total_pct",
                    "total_abs_weight_sum",
                ]
            )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, stat in rows:
            count = max(stat.count, 1)
            row = {
                "group": group,
                "feature_count": stat.count,
                "abs_weight_sum": stat.abs_sum,
                "abs_weight_mean": stat.abs_sum / count,
                "signed_weight_sum": stat.signed_sum,
                "signed_weight_mean": stat.signed_sum / count,
                "top_feature_name": stat.top_feature_name,
                "top_feature_coefficient": stat.top_feature_coef,
                "top_feature_abs_coefficient": stat.top_feature_abs,
            }
            if mode == "metric":
                share = (stat.abs_sum / total_abs_weight_sum) if total_abs_weight_sum > 0 else 0.0
                row["abs_weight_share_total"] = share
                row["abs_weight_share_total_pct"] = 100.0 * share
                row["total_abs_weight_sum"] = total_abs_weight_sum
            writer.writerow(row)


def export_winner_feature_weights(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    output_prefix: str = "winner_feature_weights",
    report_path: Path | None = None,
    manifest_path: Path | None = None,
    artifact_index_path: Path | None = None,
) -> dict[str, Path]:
    run_dir = run_dir.expanduser().resolve()
    report_path = (report_path or (run_dir / "reports" / "supervised_report.json")).expanduser().resolve()
    manifest_path = (manifest_path or (run_dir / "reports" / "tuning_manifest.json")).expanduser().resolve()
    artifact_index_path = (
        artifact_index_path or (run_dir / "artifact_index.json")
    ).expanduser().resolve()
    output_dir = (output_dir or (run_dir / "reports")).expanduser().resolve()

    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    artifact_index = _load_json(artifact_index_path) if artifact_index_path.exists() else {}

    best_model_path = Path(
        artifact_index.get("best_model", run_dir / "models" / "best_model.joblib")
    ).expanduser().resolve()
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model artifact not found: {best_model_path}")

    estimator = _extract_estimator_with_coefficients(joblib.load(best_model_path))
    coef_matrix = np.asarray(estimator.coef_, dtype=np.float64)
    if coef_matrix.ndim == 1:
        coefficients = coef_matrix
    elif coef_matrix.ndim == 2 and coef_matrix.shape[0] == 1:
        coefficients = coef_matrix[0]
    else:
        raise ValueError(
            "This exporter currently supports binary linear models with a single coefficient row. "
            f"Found coefficient shape: {coef_matrix.shape}"
        )

    feature_names = _resolve_feature_names(
        manifest_payload=manifest,
        report_payload=report,
        estimator=estimator,
        expected_len=int(coefficients.shape[0]),
    )

    classes = [str(x) for x in getattr(estimator, "classes_", ["0", "1"])]
    class0 = classes[0] if len(classes) >= 1 else "0"
    class1 = classes[1] if len(classes) >= 2 else "1"

    coef_csv = output_dir / f"{output_prefix}_coefficients.csv"
    metric_csv = output_dir / f"{output_prefix}_by_metric.csv"
    block_csv = output_dir / f"{output_prefix}_by_block.csv"
    metadata_json = output_dir / f"{output_prefix}_metadata.json"

    _write_feature_csv(
        out_path=coef_csv,
        feature_names=feature_names,
        coefficients=coefficients,
        class0=class0,
        class1=class1,
    )
    _write_group_csv(
        out_path=metric_csv,
        feature_names=feature_names,
        coefficients=coefficients,
        mode="metric",
    )
    _write_group_csv(
        out_path=block_csv,
        feature_names=feature_names,
        coefficients=coefficients,
        mode="block",
    )

    winner = report.get("tuning", {}).get("winner", {})
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "winner_model_name": winner.get("model_name"),
                "winner_params": winner.get("params"),
                "winner_task_index": winner.get("task_index"),
                "best_model_path": str(best_model_path),
                "feature_count": int(len(feature_names)),
                "class_labels": [class0, class1],
                "coefficient_csv": str(coef_csv),
                "metric_csv": str(metric_csv),
                "block_csv": str(block_csv),
            },
            f,
            indent=2,
        )

    return {
        "coefficient_csv": coef_csv,
        "metric_csv": metric_csv,
        "block_csv": block_csv,
        "metadata_json": metadata_json,
    }
