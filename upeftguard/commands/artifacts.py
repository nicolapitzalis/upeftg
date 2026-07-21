"""Feature artifact utility commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..artifacts.aggregation import aggregate_features
from ..artifacts.subset import export_feature_subset
from ..features.spectral import SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES
from ..artifacts.merge.files import merge_feature_files
from .dataset import register_download_dataset_command


def handle_aggregate_features(args: argparse.Namespace) -> int:
    outputs = aggregate_features(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        feature_root=args.feature_root,
        operator=args.operator,
        features=args.features,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_attention_granularity=args.spectral_attention_granularity,
        layout=args.layout,
    )
    print("Feature aggregation complete")
    for key, value in outputs.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


def handle_merge_features(args: argparse.Namespace) -> int:
    outputs = merge_feature_files(
        feature_paths=list(args.merge),
        output_filename=args.output_filename,
        feature_root=args.feature_root,
    )
    print("Feature merge complete")
    for key, value in outputs.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


def handle_export_feature_subset(args: argparse.Namespace) -> int:
    outputs = export_feature_subset(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        feature_root=args.feature_root,
        dataset_names=args.dataset_names,
        subset_names=args.subset_names,
        model_families=args.model_families,
        attack_names=args.attack_names,
        model_names=args.model_names,
        features=args.features,
    )
    print("Feature subset export complete")
    for key, value in outputs.items():
        if value is not None:
            print(f"{key}: {value}")
    return 0


def register_artifact_commands(sub, handlers: dict[str, object]) -> None:
    util_parser = sub.add_parser("util", help="Internal utilities")
    util_sub = util_parser.add_subparsers(dest="util_command", required=True)
    merge_features = util_sub.add_parser("merge-features", help="Merge spectral feature files")
    merge_features.add_argument("--merge", type=Path, nargs=2, metavar=("FILE1", "FILE2"), required=True)
    merge_features.add_argument("--output-filename", type=Path, required=True)
    merge_features.add_argument("--feature-root", type=Path, default=Path("runs"))
    merge_features.set_defaults(func=handlers["merge_features"])

    export_subset = util_sub.add_parser("export-feature-subset", help="Export a provenance-backed feature subset")
    export_subset.add_argument("--feature-file", type=Path, required=True)
    export_subset.add_argument("--output-filename", type=Path, required=True)
    export_subset.add_argument("--feature-root", type=Path, default=Path("runs"))
    export_subset.add_argument("--dataset-name", dest="dataset_names", nargs="+", default=None)
    export_subset.add_argument("--subset-name", dest="subset_names", nargs="+", default=None)
    export_subset.add_argument("--model-family", dest="model_families", nargs="+", default=None)
    export_subset.add_argument("--attack-name", dest="attack_names", nargs="+", default=None)
    export_subset.add_argument("--model-name", dest="model_names", nargs="+", default=None)
    export_subset.add_argument("--features", "--columns", dest="features", nargs="+", default=None)
    export_subset.set_defaults(func=handlers["export_subset"])

    aggregate_features_parser = util_sub.add_parser("aggregate-features", help="Aggregate a spectral feature file")
    aggregate_features_parser.add_argument("--feature-file", type=Path, required=True)
    aggregate_features_parser.add_argument("--output-filename", type=Path, required=True)
    aggregate_features_parser.add_argument("--operator", choices=["avg", "max", "min"], default="avg")
    aggregate_features_parser.add_argument("--layout", choices=["flat", "layer_sequence"], default="flat")
    aggregate_features_parser.add_argument("--feature-root", type=Path, default=Path("runs"))
    aggregate_features_parser.add_argument("--features", "--columns", dest="features", nargs="+", default=None)
    aggregate_features_parser.add_argument(
        "--spectral-qv-sum-mode", choices=["none", "append", "only"], default="append"
    )
    aggregate_features_parser.add_argument(
        "--spectral-attention-granularity",
        choices=list(SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES),
        default=None,
    )
    aggregate_features_parser.set_defaults(func=handlers["aggregate_features"])

    register_download_dataset_command(util_sub)
