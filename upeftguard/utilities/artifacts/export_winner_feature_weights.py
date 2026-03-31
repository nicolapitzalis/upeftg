from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import get_scorer


DEFAULT_PERMUTATION_SCORING = "roc_auc"
DEFAULT_PERMUTATION_REPEATS = 10


@dataclass
class GroupStat:
    count: int = 0
    abs_sum: float = 0.0
    signed_sum: float = 0.0
    top_feature_name: str = ""
    top_feature_coef: float = 0.0
    top_feature_abs: float = -1.0

    def update(self, feature_name: str, value: float, *, signed_value: float) -> None:
        abs_coef = float(abs(value))
        self.count += 1
        self.abs_sum += abs_coef
        self.signed_sum += float(signed_value)
        if abs_coef > self.top_feature_abs:
            self.top_feature_name = feature_name
            self.top_feature_coef = float(value)
            self.top_feature_abs = abs_coef


@dataclass(frozen=True)
class ImportancePayload:
    values: np.ndarray
    importance_type: str
    has_signed_direction: bool


@dataclass(frozen=True)
class ExportContext:
    run_dir: Path
    output_dir: Path
    output_prefix: str
    report_path: Path
    tuning_manifest_path: Path
    best_model_path: Path
    report: dict[str, Any]
    tuning_manifest: dict[str, Any]
    winner: dict[str, Any]
    model_obj: Any
    estimator: Any
    feature_names: list[str]
    class0: str
    class1: str
    coefficient_csv: Path
    metric_csv: Path
    block_csv: Path
    metadata_json: Path


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _unwrap_model_object(model_obj: Any) -> Any:
    if isinstance(model_obj, dict):
        for key in ("model", "estimator", "classifier", "best_model"):
            maybe = model_obj.get(key)
            if maybe is not None:
                return _unwrap_model_object(maybe)
    return model_obj


def _extract_final_estimator(model_obj: Any) -> Any:
    obj = _unwrap_model_object(model_obj)
    if hasattr(obj, "named_steps") and obj.named_steps:
        return list(obj.named_steps.values())[-1]
    if hasattr(obj, "steps") and obj.steps:
        return obj.steps[-1][1]
    return obj


def _native_importance_payload(estimator: Any) -> ImportancePayload | None:
    if hasattr(estimator, "coef_"):
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
        return ImportancePayload(
            values=np.asarray(coefficients, dtype=np.float64),
            importance_type="coefficient",
            has_signed_direction=True,
        )

    if hasattr(estimator, "feature_importances_"):
        return ImportancePayload(
            values=np.asarray(estimator.feature_importances_, dtype=np.float64).reshape(-1),
            importance_type="feature_importance",
            has_signed_direction=False,
        )

    return None


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


def _find_any_feature_names(payload: dict[str, Any]) -> list[str] | None:
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "feature_names" and isinstance(value, list) and value:
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


def _resolve_class_labels(estimator: Any, model_obj: Any) -> tuple[str, str]:
    class_source = estimator if hasattr(estimator, "classes_") else _unwrap_model_object(model_obj)
    classes = [str(x) for x in getattr(class_source, "classes_", ["0", "1"])]
    class0 = classes[0] if len(classes) >= 1 else "0"
    class1 = classes[1] if len(classes) >= 2 else "1"
    return class0, class1


def _resolve_export_context(
    *,
    run_dir: Path,
    output_dir: Path | None,
    output_prefix: str,
    report_path: Path | None,
    manifest_path: Path | None,
    artifact_index_path: Path | None,
) -> ExportContext:
    run_dir = run_dir.expanduser().resolve()
    report_path = (report_path or (run_dir / "reports" / "supervised_report.json")).expanduser().resolve()
    tuning_manifest_path = (manifest_path or (run_dir / "reports" / "tuning_manifest.json")).expanduser().resolve()
    artifact_index_path = (
        artifact_index_path or (run_dir / "artifact_index.json")
    ).expanduser().resolve()
    output_dir = (output_dir or (run_dir / "reports")).expanduser().resolve()

    report = _load_json(report_path)
    tuning_manifest = _load_json(tuning_manifest_path)
    artifact_index = _load_json(artifact_index_path) if artifact_index_path.exists() else {}

    best_model_path = Path(
        artifact_index.get("best_model", run_dir / "models" / "best_model.joblib")
    ).expanduser().resolve()
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model artifact not found: {best_model_path}")

    model_obj = joblib.load(best_model_path)
    estimator = _extract_final_estimator(model_obj)

    native_payload = _native_importance_payload(estimator)
    expected_len = int(native_payload.values.shape[0]) if native_payload is not None else int(
        tuning_manifest.get("representation", {})
        .get("extractor_metadata", {})
        .get("feature_dim", 0)
        or report.get("representation", {})
        .get("extractor_metadata", {})
        .get("feature_dim", 0)
    )
    if expected_len <= 0:
        metadata_names = _find_any_feature_names(tuning_manifest) or _find_any_feature_names(report)
        if metadata_names is None:
            raise ValueError("Unable to determine feature dimensionality for winner feature export")
        expected_len = len(metadata_names)

    feature_names = _resolve_feature_names(
        manifest_payload=tuning_manifest,
        report_payload=report,
        estimator=estimator,
        expected_len=expected_len,
    )
    class0, class1 = _resolve_class_labels(estimator, model_obj)
    winner = report.get("tuning", {}).get("winner", {})

    coefficient_csv = output_dir / f"{output_prefix}_coefficients.csv"
    metric_csv = output_dir / f"{output_prefix}_by_metric.csv"
    block_csv = output_dir / f"{output_prefix}_by_block.csv"
    metadata_json = output_dir / f"{output_prefix}_metadata.json"

    return ExportContext(
        run_dir=run_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
        report_path=report_path,
        tuning_manifest_path=tuning_manifest_path,
        best_model_path=best_model_path,
        report=report,
        tuning_manifest=tuning_manifest,
        winner=winner,
        model_obj=model_obj,
        estimator=estimator,
        feature_names=feature_names,
        class0=class0,
        class1=class1,
        coefficient_csv=coefficient_csv,
        metric_csv=metric_csv,
        block_csv=block_csv,
        metadata_json=metadata_json,
    )


def _write_feature_csv(
    *,
    out_path: Path,
    feature_names: list[str],
    values: np.ndarray,
    importance_type: str,
    has_signed_direction: bool,
    class0: str,
    class1: str,
) -> None:
    n_features = int(values.shape[0])
    order_abs = np.argsort(np.abs(values))[::-1]
    if has_signed_direction:
        order_pos = np.argsort(values)[::-1]
        order_neg = np.argsort(values)
    else:
        order_pos = np.asarray(order_abs, dtype=np.int64)
        order_neg = np.asarray(order_abs, dtype=np.int64)

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
            "importance_type",
            "coefficient",
            "abs_coefficient",
            "direction",
            "toward_class",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in order_abs:
            coef = float(values[idx])
            direction = "neutral"
            toward_class = "neutral"
            if has_signed_direction and coef > 0:
                direction = "positive"
                toward_class = class1
            elif has_signed_direction and coef < 0:
                direction = "negative"
                toward_class = class0
            elif has_signed_direction:
                direction = "zero"
            writer.writerow(
                {
                    "rank_abs": int(rank_abs[idx]),
                    "rank_positive": int(rank_pos[idx]),
                    "rank_negative": int(rank_neg[idx]),
                    "feature_index": int(idx),
                    "feature_name": feature_names[idx],
                    "importance_type": importance_type,
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
    values: np.ndarray,
    importance_type: str,
    has_signed_direction: bool,
    mode: str,
) -> None:
    if mode not in {"metric", "block"}:
        raise ValueError(f"Unsupported grouping mode: {mode}")

    stats: dict[str, GroupStat] = {}
    for name, value in zip(feature_names, values):
        if mode == "metric":
            group = name.rsplit(".", maxsplit=1)[-1]
        else:
            group = name.rsplit(".", maxsplit=1)[0] if "." in name else name
        stat = stats.setdefault(group, GroupStat())
        stat.update(
            name,
            float(value),
            signed_value=float(value) if has_signed_direction else 0.0,
        )

    rows = sorted(stats.items(), key=lambda item: item[1].abs_sum, reverse=True)
    total_abs_weight_sum = float(sum(stat.abs_sum for _, stat in rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "importance_type",
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
                "importance_type": importance_type,
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


def _write_export_outputs(
    *,
    context: ExportContext,
    importance: ImportancePayload,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    _write_feature_csv(
        out_path=context.coefficient_csv,
        feature_names=context.feature_names,
        values=importance.values,
        importance_type=importance.importance_type,
        has_signed_direction=importance.has_signed_direction,
        class0=context.class0,
        class1=context.class1,
    )
    _write_group_csv(
        out_path=context.metric_csv,
        feature_names=context.feature_names,
        values=importance.values,
        importance_type=importance.importance_type,
        has_signed_direction=importance.has_signed_direction,
        mode="metric",
    )
    _write_group_csv(
        out_path=context.block_csv,
        feature_names=context.feature_names,
        values=importance.values,
        importance_type=importance.importance_type,
        has_signed_direction=importance.has_signed_direction,
        mode="block",
    )

    metadata = {
        "run_dir": str(context.run_dir),
        "winner_model_name": context.winner.get("model_name"),
        "winner_params": context.winner.get("params"),
        "winner_task_index": context.winner.get("task_index"),
        "best_model_path": str(context.best_model_path),
        "feature_count": int(len(context.feature_names)),
        "class_labels": [context.class0, context.class1],
        "importance_type": importance.importance_type,
        "source_estimator_type": type(context.estimator).__name__,
        "has_signed_direction": bool(importance.has_signed_direction),
        "coefficient_csv": str(context.coefficient_csv),
        "metric_csv": str(context.metric_csv),
        "block_csv": str(context.block_csv),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    _write_json(context.metadata_json, metadata)

    return {
        "coefficient_csv": context.coefficient_csv,
        "metric_csv": context.metric_csv,
        "block_csv": context.block_csv,
        "metadata_json": context.metadata_json,
    }


def _normalized_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    resolved = int(n_jobs)
    if resolved == 0:
        raise ValueError("n_jobs must not be zero")
    return resolved


def _dispatch_manifest_path(output_dir: Path, output_prefix: str) -> Path:
    return output_dir / f"{output_prefix}_manifest.json"


def _parts_dir(output_dir: Path, output_prefix: str) -> Path:
    return output_dir / f"{output_prefix}_parts"


def _part_path(parts_dir: Path, task_index: int) -> Path:
    return parts_dir / f"part_{int(task_index):05d}.npz"


def _feature_indices_for_task(feature_count: int, n_tasks: int, task_index: int) -> np.ndarray:
    if feature_count <= 0:
        return np.asarray([], dtype=np.int64)
    resolved_tasks = max(1, min(int(n_tasks), int(feature_count)))
    if int(task_index) < 0 or int(task_index) >= resolved_tasks:
        return np.asarray([], dtype=np.int64)
    task_slices = np.array_split(np.arange(feature_count, dtype=np.int64), resolved_tasks)
    return np.asarray(task_slices[int(task_index)], dtype=np.int64)


def _clear_existing_parts(parts_dir: Path) -> None:
    if not parts_dir.exists():
        return
    for path in parts_dir.glob("part_*.npz"):
        path.unlink()


def cleanup_winner_feature_weights_export(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    output_prefix: str = "winner_feature_weights",
) -> None:
    run_dir = Path(run_dir).expanduser().resolve()
    resolved_output_dir = (output_dir or (run_dir / "reports")).expanduser().resolve()
    dispatch_manifest_path = _dispatch_manifest_path(resolved_output_dir, output_prefix)
    parts_dir = _parts_dir(resolved_output_dir, output_prefix)

    if dispatch_manifest_path.exists():
        try:
            payload = _load_json(dispatch_manifest_path)
        except Exception:
            payload = {}
        parts_dir_value = payload.get("parts_dir")
        if isinstance(parts_dir_value, str) and parts_dir_value:
            parts_dir = Path(parts_dir_value).expanduser().resolve()
        dispatch_manifest_path.unlink()

    if parts_dir.exists():
        _clear_existing_parts(parts_dir)
        try:
            parts_dir.rmdir()
        except OSError:
            pass


def _score_estimator(model_obj: Any, train_features: np.ndarray, train_labels: np.ndarray, scoring: str) -> float:
    scorer = get_scorer(scoring)
    return float(scorer(_unwrap_model_object(model_obj), train_features, train_labels))


def _compute_chunk_importances(
    *,
    feature_indices: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    model_obj: Any,
    scoring: str,
    baseline_score: float,
    n_repeats: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_indices.size == 0:
        return np.asarray([], dtype=np.int64), np.empty((0, int(n_repeats)), dtype=np.float64)

    scorer = get_scorer(scoring)
    fitted_model = _unwrap_model_object(model_obj)
    x_base = np.asarray(train_features)
    y_base = np.asarray(train_labels)
    x_work = np.array(x_base, copy=True)
    importances = np.empty((int(feature_indices.size), int(n_repeats)), dtype=np.float64)

    for row_idx, feature_idx in enumerate(feature_indices.tolist()):
        original_column = np.array(x_base[:, int(feature_idx)], copy=True)
        for repeat_idx in range(int(n_repeats)):
            seed = (
                int(random_state)
                + 1_000_003 * (int(feature_idx) + 1)
                + 9_176 * (int(repeat_idx) + 1)
            ) % (2**32 - 1)
            permuted = np.array(original_column, copy=True)
            np.random.default_rng(seed).shuffle(permuted)
            x_work[:, int(feature_idx)] = permuted
            shuffled_score = float(scorer(fitted_model, x_work, y_base))
            importances[row_idx, repeat_idx] = float(baseline_score) - shuffled_score
        x_work[:, int(feature_idx)] = original_column

    return np.asarray(feature_indices, dtype=np.int64), importances


def _compute_permutation_importance_subset(
    *,
    feature_indices: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    model_obj: Any,
    scoring: str,
    baseline_score: float,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    indices = np.asarray(feature_indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, int(n_repeats)), dtype=np.float64)

    resolved_n_jobs = _normalized_n_jobs(n_jobs)
    if resolved_n_jobs == 1 or indices.size == 1:
        _, importances = _compute_chunk_importances(
            feature_indices=indices,
            train_features=train_features,
            train_labels=train_labels,
            model_obj=model_obj,
            scoring=scoring,
            baseline_score=baseline_score,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        return importances

    n_partitions = indices.size if resolved_n_jobs < 0 else min(indices.size, resolved_n_jobs)
    chunks = [
        np.asarray(chunk, dtype=np.int64)
        for chunk in np.array_split(indices, max(1, n_partitions))
        if len(chunk) > 0
    ]

    parallel = joblib.Parallel(n_jobs=resolved_n_jobs)
    results = parallel(
        joblib.delayed(_compute_chunk_importances)(
            feature_indices=chunk,
            train_features=train_features,
            train_labels=train_labels,
            model_obj=model_obj,
            scoring=scoring,
            baseline_score=baseline_score,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        for chunk in chunks
    )

    importance_by_index: dict[int, np.ndarray] = {}
    for chunk_indices, chunk_importances in results:
        for local_row, feature_idx in enumerate(chunk_indices.tolist()):
            importance_by_index[int(feature_idx)] = np.asarray(chunk_importances[local_row], dtype=np.float64)

    matrix = np.empty((indices.size, int(n_repeats)), dtype=np.float64)
    for row_idx, feature_idx in enumerate(indices.tolist()):
        matrix[row_idx] = importance_by_index[int(feature_idx)]
    return matrix


def _compute_permutation_importance_payload(
    *,
    model_obj: Any,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    random_state: int,
    n_jobs: int,
    scoring: str = DEFAULT_PERMUTATION_SCORING,
    n_repeats: int = DEFAULT_PERMUTATION_REPEATS,
) -> ImportancePayload:
    baseline_score = _score_estimator(model_obj, train_features, train_labels, scoring)
    full_importances = _compute_permutation_importance_subset(
        feature_indices=np.arange(train_features.shape[1], dtype=np.int64),
        train_features=train_features,
        train_labels=train_labels,
        model_obj=model_obj,
        scoring=scoring,
        baseline_score=baseline_score,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return ImportancePayload(
        values=np.asarray(full_importances.mean(axis=1), dtype=np.float64).reshape(-1),
        importance_type="permutation_importance",
        has_signed_direction=False,
    )


def export_winner_feature_weights(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    output_prefix: str = "winner_feature_weights",
    report_path: Path | None = None,
    manifest_path: Path | None = None,
    artifact_index_path: Path | None = None,
    train_features: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> dict[str, Path]:
    context = _resolve_export_context(
        run_dir=run_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
        report_path=report_path,
        manifest_path=manifest_path,
        artifact_index_path=artifact_index_path,
    )

    native_importance = _native_importance_payload(context.estimator)
    if native_importance is not None:
        return _write_export_outputs(
            context=context,
            importance=native_importance,
            metadata_extra={"execution_mode": "local_native"},
        )

    if train_features is None or train_labels is None or random_state is None:
        raise TypeError(
            "Permutation importance export requires train_features, train_labels, and random_state "
            "for winners without native coefficients or feature importances"
        )

    importance = _compute_permutation_importance_payload(
        model_obj=context.model_obj,
        train_features=np.asarray(train_features),
        train_labels=np.asarray(train_labels),
        random_state=int(random_state),
        n_jobs=n_jobs,
    )
    return _write_export_outputs(
        context=context,
        importance=importance,
        metadata_extra={
            "execution_mode": "local_permutation",
            "permutation_n_repeats": int(DEFAULT_PERMUTATION_REPEATS),
            "permutation_scoring": DEFAULT_PERMUTATION_SCORING,
            "n_jobs": int(_normalized_n_jobs(n_jobs)),
        },
    )


def prepare_winner_feature_weights_export(
    *,
    run_dir: Path,
    train_feature_matrix_path: Path,
    train_labels_path: Path,
    output_dir: Path | None = None,
    output_prefix: str = "winner_feature_weights",
    report_path: Path | None = None,
    manifest_path: Path | None = None,
    artifact_index_path: Path | None = None,
    random_state: int | None = None,
    n_tasks: int = 1,
) -> dict[str, Any]:
    if int(n_tasks) <= 0:
        raise ValueError(f"n_tasks must be positive, got {n_tasks}")

    context = _resolve_export_context(
        run_dir=run_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
        report_path=report_path,
        manifest_path=manifest_path,
        artifact_index_path=artifact_index_path,
    )
    dispatch_manifest_path = _dispatch_manifest_path(context.output_dir, context.output_prefix)
    parts_dir = _parts_dir(context.output_dir, context.output_prefix)
    parts_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_parts(parts_dir)

    native_importance = _native_importance_payload(context.estimator)
    if native_importance is not None:
        outputs = _write_export_outputs(
            context=context,
            importance=native_importance,
            metadata_extra={"execution_mode": "distributed_native_prepare"},
        )
        payload = {
            "schema_version": 1,
            "mode": "completed",
            "run_dir": str(context.run_dir),
            "output_dir": str(context.output_dir),
            "output_prefix": context.output_prefix,
            "report_path": str(context.report_path),
            "tuning_manifest_path": str(context.tuning_manifest_path),
            "best_model_path": str(context.best_model_path),
            "feature_count": int(len(context.feature_names)),
            "feature_names": list(context.feature_names),
            "class_labels": [context.class0, context.class1],
            "importance_type": native_importance.importance_type,
            "has_signed_direction": bool(native_importance.has_signed_direction),
            "source_estimator_type": type(context.estimator).__name__,
            "winner_model_name": context.winner.get("model_name"),
            "winner_params": context.winner.get("params"),
            "winner_task_index": context.winner.get("task_index"),
            "coefficient_csv": str(outputs["coefficient_csv"]),
            "metric_csv": str(outputs["metric_csv"]),
            "block_csv": str(outputs["block_csv"]),
            "metadata_json": str(outputs["metadata_json"]),
            "parts_dir": str(parts_dir),
            "n_tasks": 0,
        }
        _write_json(dispatch_manifest_path, payload)
        return {
            "mode": "completed",
            "n_tasks": 0,
            "manifest_path": dispatch_manifest_path,
            **outputs,
        }

    if random_state is None:
        raise TypeError("random_state is required to prepare distributed permutation importance export")

    train_features = np.asarray(np.load(train_feature_matrix_path))
    train_labels = np.asarray(np.load(train_labels_path))
    baseline_score = _score_estimator(
        context.model_obj,
        train_features,
        train_labels,
        DEFAULT_PERMUTATION_SCORING,
    )
    resolved_tasks = max(1, min(int(n_tasks), int(train_features.shape[1])))
    payload = {
        "schema_version": 1,
        "mode": "permutation",
        "run_dir": str(context.run_dir),
        "output_dir": str(context.output_dir),
        "output_prefix": context.output_prefix,
        "report_path": str(context.report_path),
        "tuning_manifest_path": str(context.tuning_manifest_path),
        "best_model_path": str(context.best_model_path),
        "train_feature_matrix_path": str(Path(train_feature_matrix_path).expanduser().resolve()),
        "train_labels_path": str(Path(train_labels_path).expanduser().resolve()),
        "feature_count": int(len(context.feature_names)),
        "feature_names": list(context.feature_names),
        "class_labels": [context.class0, context.class1],
        "importance_type": "permutation_importance",
        "has_signed_direction": False,
        "source_estimator_type": type(context.estimator).__name__,
        "winner_model_name": context.winner.get("model_name"),
        "winner_params": context.winner.get("params"),
        "winner_task_index": context.winner.get("task_index"),
        "coefficient_csv": str(context.coefficient_csv),
        "metric_csv": str(context.metric_csv),
        "block_csv": str(context.block_csv),
        "metadata_json": str(context.metadata_json),
        "parts_dir": str(parts_dir),
        "n_tasks": int(resolved_tasks),
        "n_repeats": int(DEFAULT_PERMUTATION_REPEATS),
        "scoring": DEFAULT_PERMUTATION_SCORING,
        "random_state": int(random_state),
        "baseline_score": float(baseline_score),
    }
    _write_json(dispatch_manifest_path, payload)
    return {
        "mode": "permutation",
        "n_tasks": int(resolved_tasks),
        "manifest_path": dispatch_manifest_path,
        "parts_dir": parts_dir,
    }


def run_winner_feature_weights_export_worker(
    *,
    run_dir: Path,
    task_index: int,
    output_dir: Path | None = None,
    output_prefix: str = "winner_feature_weights",
    n_jobs: int = 1,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    resolved_output_dir = (output_dir or (run_dir / "reports")).expanduser().resolve()
    dispatch_manifest = _load_json(_dispatch_manifest_path(resolved_output_dir, output_prefix))
    mode = str(dispatch_manifest.get("mode", ""))
    if mode != "permutation":
        return {
            "status": "skipped",
            "reason": f"winner feature export mode={mode!r} does not require distributed workers",
        }

    resolved_task_index = int(task_index)
    feature_count = int(dispatch_manifest["feature_count"])
    n_tasks = int(dispatch_manifest["n_tasks"])
    feature_indices = _feature_indices_for_task(feature_count, n_tasks, resolved_task_index)
    if feature_indices.size == 0:
        return {
            "status": "skipped",
            "task_index": resolved_task_index,
            "reason": "task index outside active shard range",
        }

    train_features = np.asarray(np.load(dispatch_manifest["train_feature_matrix_path"]))
    train_labels = np.asarray(np.load(dispatch_manifest["train_labels_path"]))
    model_obj = joblib.load(dispatch_manifest["best_model_path"])
    importances = _compute_permutation_importance_subset(
        feature_indices=feature_indices,
        train_features=train_features,
        train_labels=train_labels,
        model_obj=model_obj,
        scoring=str(dispatch_manifest["scoring"]),
        baseline_score=float(dispatch_manifest["baseline_score"]),
        n_repeats=int(dispatch_manifest["n_repeats"]),
        random_state=int(dispatch_manifest["random_state"]),
        n_jobs=n_jobs,
    )
    parts_dir = Path(dispatch_manifest["parts_dir"]).expanduser().resolve()
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = _part_path(parts_dir, resolved_task_index)
    np.savez_compressed(part_path, feature_indices=feature_indices, importances=importances)
    return {
        "status": "ok",
        "task_index": resolved_task_index,
        "part_path": str(part_path),
        "n_features": int(feature_indices.size),
    }


def merge_winner_feature_weights_export(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    output_prefix: str = "winner_feature_weights",
    report_path: Path | None = None,
    manifest_path: Path | None = None,
    artifact_index_path: Path | None = None,
) -> dict[str, Path]:
    context = _resolve_export_context(
        run_dir=run_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
        report_path=report_path,
        manifest_path=manifest_path,
        artifact_index_path=artifact_index_path,
    )
    dispatch_manifest = _load_json(_dispatch_manifest_path(context.output_dir, context.output_prefix))
    mode = str(dispatch_manifest.get("mode", ""))
    if mode == "completed":
        outputs = {
            "coefficient_csv": Path(dispatch_manifest["coefficient_csv"]).expanduser().resolve(),
            "metric_csv": Path(dispatch_manifest["metric_csv"]).expanduser().resolve(),
            "block_csv": Path(dispatch_manifest["block_csv"]).expanduser().resolve(),
            "metadata_json": Path(dispatch_manifest["metadata_json"]).expanduser().resolve(),
        }
        cleanup_winner_feature_weights_export(
            run_dir=context.run_dir,
            output_dir=context.output_dir,
            output_prefix=context.output_prefix,
        )
        return outputs
    if mode != "permutation":
        raise ValueError(f"Unsupported winner feature export mode in dispatch manifest: {mode!r}")

    feature_count = int(dispatch_manifest["feature_count"])
    n_repeats = int(dispatch_manifest["n_repeats"])
    n_tasks = int(dispatch_manifest["n_tasks"])
    parts_dir = Path(dispatch_manifest["parts_dir"]).expanduser().resolve()
    full_importances = np.full((feature_count, n_repeats), np.nan, dtype=np.float64)

    for task_idx in range(n_tasks):
        part_path = _part_path(parts_dir, task_idx)
        if not part_path.exists():
            raise FileNotFoundError(f"Missing permutation export shard: {part_path}")
        with np.load(part_path) as payload:
            feature_indices = np.asarray(payload["feature_indices"], dtype=np.int64)
            importances = np.asarray(payload["importances"], dtype=np.float64)
        if importances.shape != (feature_indices.size, n_repeats):
            raise ValueError(
                f"Unexpected shard shape for {part_path}: {importances.shape}; "
                f"expected ({feature_indices.size}, {n_repeats})"
            )
        full_importances[feature_indices] = importances

    if np.isnan(full_importances).any():
        raise ValueError("Distributed permutation export did not cover all feature importances")

    importance = ImportancePayload(
        values=np.asarray(full_importances.mean(axis=1), dtype=np.float64),
        importance_type="permutation_importance",
        has_signed_direction=False,
    )
    outputs = _write_export_outputs(
        context=context,
        importance=importance,
        metadata_extra={
            "execution_mode": "distributed_permutation",
            "permutation_n_repeats": n_repeats,
            "permutation_scoring": str(dispatch_manifest["scoring"]),
            "distributed_tasks": n_tasks,
            "parts_dir": str(parts_dir),
            "baseline_score": float(dispatch_manifest["baseline_score"]),
        },
    )
    cleanup_winner_feature_weights_export(
        run_dir=context.run_dir,
        output_dir=context.output_dir,
        output_prefix=context.output_prefix,
    )
    return outputs
