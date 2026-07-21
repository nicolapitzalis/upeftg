"""PADBench dataset-download CLI adapter."""

from __future__ import annotations

import argparse

from ..utilities.core.paths import default_dataset_root
from ..utilities.data import dataset_download


def _selection_request(
    raw_values: list[int] | None,
    *,
    argument: str,
) -> dataset_download.SelectionRequest:
    if raw_values is None:
        return dataset_download.SelectionRequest(count=0)
    if len(raw_values) == 1 and raw_values[0] >= 0:
        return dataset_download.SelectionRequest(count=raw_values[0])
    if len(raw_values) == 2 and 0 <= raw_values[0] <= raw_values[1]:
        return dataset_download.SelectionRequest(start=raw_values[0], end=raw_values[1])
    raise ValueError(f"{argument} expects a non-negative count or an inclusive <start> <end> range")


def handle_download_dataset(args: argparse.Namespace) -> int:
    args.datasets = dataset_download.unique_values(args.datasets)
    if args.all and (args.clean is not None or args.backdoored is not None):
        raise ValueError("--all cannot be combined with --clean or --backdoored")
    if args.all:
        args.clean_request = dataset_download.SelectionRequest(select_all=True)
        args.backdoored_request = dataset_download.SelectionRequest(select_all=True)
    else:
        args.clean_request = _selection_request(args.clean, argument="--clean")
        args.backdoored_request = _selection_request(args.backdoored, argument="--backdoored")
    if not args.show_list and not args.datasets:
        raise ValueError("--dataset must be specified unless --show-list is used")
    args.clean_sources = dataset_download.resolve_clean_sources(args.datasets)
    args.backdoored_sources = dataset_download.resolve_backdoored_sources(args.datasets)
    if not args.show_list and args.clean_request.is_empty and args.backdoored_request.is_empty:
        raise ValueError("at least one of --clean or --backdoored must select an adapter")
    dataset_download.download_padbench(args)
    return 0


def register_download_dataset_command(util_sub) -> None:
    parser = util_sub.add_parser("download-dataset", help="Download PADBench subsets")
    parser.add_argument("--clean", type=int, nargs="+", metavar="N")
    parser.add_argument(
        "--backdoored",
        "--backdoor",
        dest="backdoored",
        type=int,
        nargs="+",
        metavar="N",
    )
    parser.add_argument("--repo-id", default=dataset_download.DEFAULT_REPO_ID)
    parser.add_argument("--show-list", action="store_true")
    parser.add_argument("--dataset", dest="datasets", action="append", metavar="FOLDER")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--local-dir", default=str(default_dataset_root()))
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(func=handle_download_dataset)
