from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from .clustering.pipeline import run_clustering_pipeline
from .features.registry import extract_with_cache, supported_extractors
from .unsupervised.gmm_train_inference import run_gmm_train_inference_pipeline
from .utilities.manifest import parse_single_manifest_json
from .utilities.run_context import create_run_context
from .utilities.serialization import json_ready
from .utilities.reporting import build_compare_parser, run_compare_reports


def _execution_root() -> Path:
    return Path.cwd().resolve()


def _is_filesystem_root(path: Path) -> bool:
    return path == path.parent


def _resolve_output_root(output_root: Path, caller: str) -> Path:
    root = _execution_root()
    candidate = output_root.expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved == root or _is_filesystem_root(resolved):
        # Never write caller artifacts directly at execution root or filesystem root.
        return (root / "runs" / caller).resolve()
    return resolved


def _resolve_report_output_file(output_file: Path) -> Path:
    root = _execution_root()
    candidate = output_file.expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved.parent == root or _is_filesystem_root(resolved.parent):
        # Never write report files directly at execution root or filesystem root.
        return (root / "runs" / "report" / "compare_representations" / resolved.name).resolve()
    return resolved


def _has_option(tokens: list[str], option: str) -> bool:
    return any(tok == option or tok.startswith(option + "=") for tok in tokens)


def _rewrite_download_local_dir(tokens: list[str]) -> list[str]:
    root = _execution_root()
    safe_default = (root / "runs" / "util" / "download_dataset").resolve()
    out = list(tokens)

    if not _has_option(out, "--local-dir"):
        return ["--local-dir", str(safe_default), *out]

    for idx, tok in enumerate(out):
        if tok == "--local-dir" and idx + 1 < len(out):
            current = Path(out[idx + 1]).expanduser()
            resolved = (current if current.is_absolute() else root / current).resolve()
            if resolved == root or _is_filesystem_root(resolved):
                out[idx + 1] = str(safe_default)
            return out
        if tok.startswith("--local-dir="):
            raw = tok.split("=", 1)[1]
            current = Path(raw).expanduser()
            resolved = (current if current.is_absolute() else root / current).resolve()
            if resolved == root or _is_filesystem_root(resolved):
                out[idx] = f"--local-dir={safe_default}"
            return out
    return out


def _extractor_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "component_grid": list(args.svd_components_grid),
        "n_components": args.svd_n_components,
        "top_k_singular_values": args.top_k_singular_values,
        "block_size": args.stream_block_size,
        "dtype": args.dtype,
        "acceptance_spearman_threshold": args.acceptance_spearman_threshold,
        "acceptance_variance_threshold": args.acceptance_variance_threshold,
        "run_offline_label_diagnostics": args.run_offline_label_diagnostics,
    }


def _cmd_run_clustering(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_clustering")
    result = run_clustering_pipeline(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        feature_file=args.feature_file,
        extractor_name=args.extractor,
        extractor_params=_extractor_params_from_args(args),
        output_root=output_root,
        run_id=args.run_id,
        algorithms=args.algorithms,
        k_list=args.k_list,
        eps_list=args.eps_list,
        min_samples=args.min_samples,
        gmm_components=args.gmm_components,
        gmm_covariance_types=args.gmm_covariance_types,
        selection_metric=args.selection_metric,
        use_offline_label_metrics=args.use_offline_label_metrics,
        stability_seeds=args.stability_seeds,
        score_percentiles=args.score_percentiles,
        force_recompute_features=args.force_recompute_features,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    if result.get("score_table"):
        print(f"Score table: {result['score_table']}")
    return 0


def _cmd_run_gmm_train_inference(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_gmm_train_inference")
    result = run_gmm_train_inference_pipeline(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=output_root,
        run_id=args.run_id,
        svd_components_grid=args.svd_components_grid,
        gmm_components=args.gmm_components,
        gmm_covariance_types=args.gmm_covariance_types,
        stability_seeds=args.stability_seeds,
        score_percentiles=args.score_percentiles,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        reg_covar=args.reg_covar,
        n_init=args.n_init,
        force_recompute_features=args.force_recompute_features,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    print(f"Train scores: {result['train_scores_csv']}")
    print(f"Inference scores: {result['inference_scores_csv']}")
    return 0


def _cmd_feature_extract(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "feature_extract")
    items = parse_single_manifest_json(
        manifest_path=args.manifest_json,
        dataset_root=args.dataset_root,
        section_key="path",
    )

    ctx = create_run_context(
        pipeline="feature_extract",
        output_root=output_root,
        run_id=args.run_id,
    )

    bundle, artifacts, warnings = extract_with_cache(
        extractor_name=args.extractor,
        items=items,
        params=_extractor_params_from_args(args),
        cache_root=ctx.cache_root,
        run_features_dir=ctx.features_dir,
        force_recompute=args.force_recompute_features,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "extractor": args.extractor,
        "feature_shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        "labels_available": bool(bundle.labels is not None),
        "model_count": len(bundle.model_names),
        "cache_hit": artifacts["cache_hit"],
        "cache_key": artifacts["cache_key"],
        "warnings": warnings,
        "metadata": bundle.metadata,
    }
    report_path = ctx.reports_dir / "feature_extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("features", Path(artifacts["feature_path"]))
    if artifacts.get("labels_path"):
        ctx.add_artifact("labels", Path(str(artifacts["labels_path"])))
    ctx.add_artifact("feature_metadata", Path(artifacts["metadata_path"]))
    ctx.add_artifact("report", report_path)

    run_config = {
        "pipeline": "feature_extract",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": str(args.manifest_json),
        "dataset_root": str(args.dataset_root),
        "extractor": args.extractor,
        "extractor_params": _extractor_params_from_args(args),
        "cache_key": artifacts["cache_key"],
        "cache_hit": artifacts["cache_hit"],
        "warnings": warnings,
    }
    ctx.finalize(run_config)

    print("Feature extraction complete")
    print(f"Run dir: {ctx.run_dir}")
    print(f"Report: {report_path}")
    return 0


def _cmd_download_dataset(args: argparse.Namespace) -> int:
    from .utilities import dataset_download

    passthrough = _rewrite_download_local_dir(list(args.download_args))
    old_argv = sys.argv
    try:
        sys.argv = ["download_dataset"] + passthrough
        dataset_download.main()
    finally:
        sys.argv = old_argv
    return 0


def _cmd_compare_reports(args: argparse.Namespace) -> int:
    output_file = _resolve_report_output_file(args.output_file)
    out = run_compare_reports(
        reports=args.reports,
        output_file=output_file,
        target_auroc=args.target_auroc,
        target_stability=args.target_stability,
    )
    print(f"Saved: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="upeftguard unified CLI")
    sub = parser.add_subparsers(dest="command_group", required=True)

    run_parser = sub.add_parser("run", help="Run end-to-end pipelines")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)

    clustering = run_sub.add_parser("clustering", help="Run clustering pipeline")
    clustering.add_argument("--manifest-json", type=Path, default=None, help="Single-set manifest JSON")
    clustering.add_argument("--dataset-root", type=Path, default=Path("data"), help="Dataset root")
    clustering.add_argument("--feature-file", type=Path, default=None, help="Optional external feature matrix (.npy)")
    clustering.add_argument("--extractor", choices=supported_extractors(), default="svd", help="Feature extractor")
    clustering.add_argument("--output-root", type=Path, default=Path("runs"), help="Output root for run artifacts")
    clustering.add_argument("--run-id", type=str, default=None, help="Optional fixed run id")
    clustering.add_argument("--force-recompute-features", action="store_true", help="Ignore feature cache")

    clustering.add_argument("--svd-components-grid", nargs="+", type=int, default=[20, 25, 30])
    clustering.add_argument("--svd-n-components", type=int, default=None)
    clustering.add_argument("--top-k-singular-values", type=int, default=8)
    clustering.add_argument("--stream-block-size", type=int, default=131072)
    clustering.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    clustering.add_argument("--acceptance-spearman-threshold", type=float, default=0.99)
    clustering.add_argument("--acceptance-variance-threshold", type=float, default=0.95)
    clustering.add_argument("--run-offline-label-diagnostics", action="store_true")

    clustering.add_argument(
        "--algorithms",
        nargs="+",
        default=["kmeans", "hierarchical", "dbscan", "gmm", "mahalanobis", "isolation_forest", "lof"],
    )
    clustering.add_argument("--k-list", nargs="+", type=int, default=[2, 3, 4, 5])
    clustering.add_argument("--eps-list", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    clustering.add_argument("--min-samples", type=int, default=2)
    clustering.add_argument("--gmm-components", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    clustering.add_argument(
        "--gmm-covariance-types",
        nargs="+",
        default=["diag", "full", "tied", "spherical"],
    )
    clustering.add_argument("--selection-metric", choices=["silhouette", "bic", "stability"], default="silhouette")
    clustering.add_argument("--use-offline-label-metrics", action="store_true")
    clustering.add_argument("--stability-seeds", nargs="+", type=int, default=[42, 43, 44])
    clustering.add_argument("--score-percentiles", nargs="+", type=float, default=[90, 95, 97, 99])
    clustering.set_defaults(func=_cmd_run_clustering)

    gmm = run_sub.add_parser("gmm-train-inference", help="Run SVD+GMM train/inference pipeline")
    gmm.add_argument("--manifest-json", type=Path, required=True)
    gmm.add_argument("--dataset-root", type=Path, default=Path("data"))
    gmm.add_argument("--output-root", type=Path, default=Path("runs"))
    gmm.add_argument("--run-id", type=str, default=None)
    gmm.add_argument("--force-recompute-features", action="store_true")
    gmm.add_argument("--svd-components-grid", nargs="+", type=int, default=[20, 25, 30])
    gmm.add_argument("--gmm-components", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    gmm.add_argument(
        "--gmm-covariance-types",
        nargs="+",
        default=["diag", "full", "tied", "spherical"],
    )
    gmm.add_argument("--stability-seeds", nargs="+", type=int, default=[42, 43, 44])
    gmm.add_argument("--score-percentiles", nargs="+", type=float, default=[90, 95, 97, 99])
    gmm.add_argument("--stream-block-size", type=int, default=131072)
    gmm.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    gmm.add_argument("--reg-covar", type=float, default=1e-5)
    gmm.add_argument("--n-init", type=int, default=1)
    gmm.set_defaults(func=_cmd_run_gmm_train_inference)

    feature_parser = sub.add_parser("feature", help="Feature extraction commands")
    feature_sub = feature_parser.add_subparsers(dest="feature_command", required=True)

    extract = feature_sub.add_parser("extract", help="Extract and cache features")
    extract.add_argument("--manifest-json", type=Path, required=True)
    extract.add_argument("--dataset-root", type=Path, default=Path("data"))
    extract.add_argument("--extractor", choices=supported_extractors(), default="svd")
    extract.add_argument("--output-root", type=Path, default=Path("runs"))
    extract.add_argument("--run-id", type=str, default=None)
    extract.add_argument("--force-recompute-features", action="store_true")
    extract.add_argument("--svd-components-grid", nargs="+", type=int, default=[20, 25, 30])
    extract.add_argument("--svd-n-components", type=int, default=None)
    extract.add_argument("--top-k-singular-values", type=int, default=8)
    extract.add_argument("--stream-block-size", type=int, default=131072)
    extract.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    extract.add_argument("--acceptance-spearman-threshold", type=float, default=0.99)
    extract.add_argument("--acceptance-variance-threshold", type=float, default=0.95)
    extract.add_argument("--run-offline-label-diagnostics", action="store_true")
    extract.set_defaults(func=_cmd_feature_extract)

    util_parser = sub.add_parser("util", help="Utility commands")
    util_sub = util_parser.add_subparsers(dest="util_command", required=True)

    download = util_sub.add_parser("download-dataset", help="Download PADBench subsets")
    download.add_argument("download_args", nargs=argparse.REMAINDER)
    download.set_defaults(func=_cmd_download_dataset)

    report_parser = sub.add_parser("report", help="Reporting commands")
    report_sub = report_parser.add_subparsers(dest="report_command", required=True)

    compare = report_sub.add_parser("compare-representations", help="Compare representation reports")
    build_compare_parser(compare)
    compare.set_defaults(func=_cmd_compare_reports)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
