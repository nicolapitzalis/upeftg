import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile, RepoFolder

DEFAULT_REPO_ID = "Vincent-HKUSTGZ/PADBench"
MODEL_DIR_RE = re.compile(r"^(?P<prefix>.+)_label(?P<label>\d+)_(?P<index>\d+)$")
BYTES_PER_GB = 1_000_000_000


@dataclass(frozen=True)
class Source:
    subset: str
    label: int

    @property
    def key(self) -> Tuple[str, int]:
        return (self.subset, self.label)

    def model_dir(self, index: int) -> str:
        return f"{self.subset}/{self.subset}_label{self.label}_{index}"


@dataclass(frozen=True)
class SelectionRequest:
    count: int | None = None
    start: int | None = None
    end: int | None = None

    @property
    def uses_count(self) -> bool:
        return self.count is not None

    @property
    def is_empty(self) -> bool:
        if self.uses_count:
            return self.count == 0
        return self.start is None or self.end is None or self.start > self.end


# label0 = clean, label1 = backdoored
CLEAN_SOURCES = [
    Source("llama2_7b_imdb_insertsent_rank256_qv", 0),
]

BACKDOORED_SOURCES = [
    Source("llama2_7b_imdb_RIPPLE_rank256_qv", 1),
    Source("llama2_7b_imdb_insertsent_rank256_qv", 1),
    Source("llama2_7b_imdb_stybkd_rank256_qv", 1),
    Source("llama2_7b_imdb_syntactic_rank256_qv", 1),
]


def parse_selection_request(
    raw_values: Sequence[int] | None,
    arg_name: str,
    parser: argparse.ArgumentParser,
) -> SelectionRequest:
    if raw_values is None:
        return SelectionRequest(count=0)

    if len(raw_values) == 1:
        count = raw_values[0]
        if count < 0:
            parser.error(f"{arg_name} count must be >= 0")
        return SelectionRequest(count=count)

    if len(raw_values) == 2:
        start, end = raw_values
        if start < 0 or end < 0:
            parser.error(f"{arg_name} range values must be >= 0")
        if start > end:
            parser.error(f"{arg_name} range must satisfy start <= end")
        return SelectionRequest(start=start, end=end)

    parser.error(
        f"{arg_name} expects either one integer (<count>) "
        "or two integers (<start> <end>)"
    )
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected PADBench adapters from the specified IMDB subsets. "
            "Clean adapters are taken from label0, backdoored adapters from label1."
        )
    )
    parser.add_argument(
        "--clean",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Clean adapter selection. Pass one value (<count>) or two values "
            "(<start> <end>, inclusive). Example: --clean 0 100."
        ),
    )
    parser.add_argument(
        "--backdoored",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Backdoored adapter selection. Pass one value (<count>, round-robin "
            "across attacks) or two values (<start> <end>, inclusive per attack)."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./data",
        help="Local output directory for downloaded files (default: ./data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected model directories without downloading them.",
    )

    args = parser.parse_args()
    args.clean_request = parse_selection_request(args.clean, "--clean", parser)
    args.backdoored_request = parse_selection_request(
        args.backdoored, "--backdoored", parser
    )
    if args.clean_request.is_empty and args.backdoored_request.is_empty:
        parser.error(
            "at least one of --clean or --backdoored must select at least one adapter"
        )
    return args


def parse_model_dir(subset: str, model_dir: str) -> Tuple[int, int] | None:
    match = MODEL_DIR_RE.fullmatch(model_dir)
    if match is None:
        return None
    if match.group("prefix") != subset:
        return None
    label = int(match.group("label"))
    index = int(match.group("index"))
    return label, index


def file_size_bytes(entry: RepoFile) -> int:
    if entry.size is not None:
        return int(entry.size)
    lfs_info = entry.lfs
    if isinstance(lfs_info, dict):
        lfs_size = lfs_info.get("size")
        if isinstance(lfs_size, int):
            return lfs_size
    return 0


def parse_tree_entry_path(entry_path: str, default_subset: str) -> Tuple[str, str] | None:
    parts = entry_path.split("/", 2)
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return default_subset, parts[0]
    return None


def relevant_subsets() -> List[str]:
    return sorted({source.subset for source in CLEAN_SOURCES + BACKDOORED_SOURCES})


def list_available_indices(
    repo_id: str,
    subsets: Sequence[str],
) -> Dict[Tuple[str, int], List[int]]:
    api = HfApi()

    by_source: Dict[Tuple[str, int], set[int]] = defaultdict(set)
    for subset in subsets:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=subset,
            repo_type="dataset",
            recursive=False,
            expand=False,
        )
        for entry in entries:
            if not isinstance(entry, RepoFolder):
                continue

            parsed_path = parse_tree_entry_path(entry.path, default_subset=subset)
            if parsed_path is None:
                continue

            parsed_subset, model_dir = parsed_path
            parsed = parse_model_dir(parsed_subset, model_dir)
            if parsed is None:
                continue

            label, index = parsed
            by_source[(parsed_subset, label)].add(index)

    return {key: sorted(indices) for key, indices in by_source.items()}


def list_selected_model_sizes(
    repo_id: str,
    selection: Dict[Source, List[int]],
) -> Dict[str, int]:
    api = HfApi()
    size_by_model_dir: Dict[str, int] = defaultdict(int)
    for source, indices in selection.items():
        if not indices:
            continue

        sample_index = indices[0]
        sample_model_dir = source.model_dir(sample_index)
        entries = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=sample_model_dir,
            repo_type="dataset",
            recursive=True,
            expand=True,
        )

        sample_size_bytes = 0
        for entry in entries:
            if not isinstance(entry, RepoFile):
                continue
            sample_size_bytes += file_size_bytes(entry)

        for index in indices:
            size_by_model_dir[source.model_dir(index)] = sample_size_bytes

    return dict(size_by_model_dir)


def allocate_round_robin(total: int, capacities: Sequence[int]) -> List[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return [0] * len(capacities)
    if not capacities:
        raise ValueError("no sources configured for allocation")
    if sum(capacities) < total:
        raise ValueError(
            f"requested {total} models but only {sum(capacities)} are available"
        )

    allocation = [0] * len(capacities)
    remaining = total
    while remaining > 0:
        progressed = False
        for i, capacity in enumerate(capacities):
            if allocation[i] >= capacity:
                continue
            allocation[i] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("allocation stalled before reaching requested total")
    return allocation


def select_patterns(
    request: SelectionRequest,
    sources: Sequence[Source],
    available_by_source: Dict[Tuple[str, int], List[int]],
    name: str,
) -> Tuple[List[str], Dict[Source, List[int]]]:
    allow_patterns: List[str] = []
    selection: Dict[Source, List[int]] = {}
    if request.uses_count:
        requested_total = request.count or 0
        capacities = [len(available_by_source.get(source.key, [])) for source in sources]
        try:
            allocation = allocate_round_robin(requested_total, capacities)
        except ValueError as exc:
            details = ", ".join(
                f"{source.subset} label{source.label}: {capacity}"
                for source, capacity in zip(sources, capacities)
            )
            raise ValueError(
                f"Unable to satisfy --{name}={requested_total}. Available per source -> {details}"
            ) from exc

        for source, count in zip(sources, allocation):
            chosen_indices = available_by_source.get(source.key, [])[:count]
            selection[source] = chosen_indices
            for index in chosen_indices:
                allow_patterns.append(f"{source.model_dir(index)}/*")
        return allow_patterns, selection

    start = request.start
    end = request.end
    if start is None or end is None:
        raise ValueError(f"Invalid --{name} selection request")

    for source in sources:
        available = available_by_source.get(source.key, [])
        chosen_indices = [index for index in available if start <= index <= end]
        selection[source] = chosen_indices
        for index in chosen_indices:
            allow_patterns.append(f"{source.model_dir(index)}/*")

    if not allow_patterns:
        details = ", ".join(
            f"{source.subset} label{source.label}: {len(available_by_source.get(source.key, []))}"
            for source in sources
        )
        raise ValueError(
            f"Unable to satisfy --{name} range {start}..{end}. "
            f"Available counts per source -> {details}"
        )

    return allow_patterns, selection


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / BYTES_PER_GB:.3f} GB"


def print_selection(
    title: str,
    selection: Dict[Source, List[int]],
    size_by_model_dir: Dict[str, int],
) -> int:
    print(f"{title} allocation:")
    total_bytes = 0
    for source, indices in selection.items():
        source_bytes = sum(size_by_model_dir.get(source.model_dir(index), 0) for index in indices)
        total_bytes += source_bytes
        print(
            f"  - {source.subset} label{source.label}: "
            f"{len(indices)} adapters, {format_gb(source_bytes)}"
        )
    return total_bytes


def main() -> None:
    args = parse_args()

    print(f"Listing available model folders for '{args.repo_id}'...")
    available_by_source = list_available_indices(args.repo_id, relevant_subsets())

    clean_patterns, clean_selection = select_patterns(
        request=args.clean_request,
        sources=CLEAN_SOURCES,
        available_by_source=available_by_source,
        name="clean",
    )
    backdoored_patterns, backdoored_selection = select_patterns(
        request=args.backdoored_request,
        sources=BACKDOORED_SOURCES,
        available_by_source=available_by_source,
        name="backdoored",
    )

    allow_patterns = clean_patterns + backdoored_patterns
    combined_selection = {**clean_selection, **backdoored_selection}
    print("Estimating storage from one representative adapter per source...")
    size_by_model_dir = list_selected_model_sizes(args.repo_id, combined_selection)
    clean_bytes = print_selection("Clean", clean_selection, size_by_model_dir)
    backdoored_bytes = print_selection("Backdoored", backdoored_selection, size_by_model_dir)
    print(f"Total adapters selected: {len(allow_patterns)}")
    print(
        "Estimated storage required: "
        f"clean={format_gb(clean_bytes)}, "
        f"backdoored={format_gb(backdoored_bytes)}, "
        f"total={format_gb(clean_bytes + backdoored_bytes)}"
    )

    if args.dry_run:
        print("Dry run enabled; no files were downloaded.")
        return

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=args.local_dir,
    )
    print("Download complete!")


if __name__ == "__main__":
    main()
