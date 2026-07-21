import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import httpx
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
    select_all: bool = False

    @property
    def uses_all(self) -> bool:
        return self.select_all

    @property
    def uses_count(self) -> bool:
        return not self.select_all and self.count is not None

    @property
    def is_empty(self) -> bool:
        if self.select_all:
            return False
        if self.uses_count:
            return self.count == 0
        return self.start is None or self.end is None or self.start > self.end


def unique_values(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []

    seen: set[str] = set()
    unique_values: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def resolve_clean_sources(datasets: Sequence[str]) -> List[Source]:
    return [Source(subset, 0) for subset in datasets]


def resolve_backdoored_sources(datasets: Sequence[str]) -> List[Source]:
    return [Source(subset, 1) for subset in datasets]


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


def relevant_subsets(*source_groups: Sequence[Source]) -> List[str]:
    return sorted({source.subset for sources in source_groups for source in sources})


def _is_slurm_context() -> bool:
    return bool(os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOB_NODELIST"))


def _fatal_hf_connect_error(action: str, repo_id: str, exc: Exception) -> None:
    context = " from a Slurm compute node" if _is_slurm_context() else ""
    raise SystemExit(
        f"{action} failed: unable to reach Hugging Face Hub repo '{repo_id}' "
        f"(DNS/network issue{context}). "
        "Run this command on a login node with internet access, or pre-download and "
        "reuse a shared local dataset directory."
    ) from exc


def list_available_indices(
    repo_id: str,
    subsets: Sequence[str],
) -> Dict[Tuple[str, int], List[int]]:
    api = HfApi()

    by_source: Dict[Tuple[str, int], set[int]] = defaultdict(set)
    for subset in subsets:
        try:
            entries = api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=subset,
                repo_type="dataset",
                recursive=False,
                expand=False,
            )
        except httpx.ConnectError as exc:
            _fatal_hf_connect_error("Listing available model folders", repo_id, exc)

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


def list_padbench_folders(repo_id: str) -> List[str]:
    api = HfApi()
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo="",
            repo_type="dataset",
            recursive=False,
            expand=False,
        )
    except httpx.ConnectError as exc:
        _fatal_hf_connect_error("Listing PADBench folders", repo_id, exc)

    folders: set[str] = set()
    for entry in entries:
        if not isinstance(entry, RepoFolder):
            continue
        folder = entry.path.split("/", 1)[0]
        if folder:
            folders.add(folder)
    return sorted(folders)


def validate_requested_datasets(
    requested_datasets: Sequence[str],
    available_datasets: Sequence[str],
) -> None:
    available = set(available_datasets)
    missing = [dataset for dataset in requested_datasets if dataset not in available]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            f"Unknown PADBench folder(s): {missing_text}. Use --show-list to inspect the available folders."
        )


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
        try:
            entries = api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=sample_model_dir,
                repo_type="dataset",
                recursive=True,
                expand=True,
            )
        except httpx.ConnectError as exc:
            _fatal_hf_connect_error("Estimating adapter storage", repo_id, exc)

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
        raise ValueError(f"requested {total} models but only {sum(capacities)} are available")

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
    if request.uses_all:
        for source in sources:
            chosen_indices = list(available_by_source.get(source.key, []))
            selection[source] = chosen_indices
            for index in chosen_indices:
                allow_patterns.append(f"{source.model_dir(index)}/*")
        return allow_patterns, selection

    if request.uses_count:
        requested_total = request.count or 0
        capacities = [len(available_by_source.get(source.key, [])) for source in sources]
        try:
            allocation = allocate_round_robin(requested_total, capacities)
        except ValueError as exc:
            details = ", ".join(
                f"{source.subset} label{source.label}: {capacity}" for source, capacity in zip(sources, capacities)
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
            f"{source.subset} label{source.label}: {len(available_by_source.get(source.key, []))}" for source in sources
        )
        raise ValueError(f"Unable to satisfy --{name} range {start}..{end}. Available counts per source -> {details}")

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
        print(f"  - {source.subset} label{source.label}: {len(indices)} adapters, {format_gb(source_bytes)}")
    return total_bytes


def print_padbench_folders(repo_id: str, folders: Sequence[str]) -> None:
    print(f"Available PADBench folders in '{repo_id}':")
    if not folders:
        print("  (none)")
        return
    for folder in folders:
        print(f"  - {folder}")


def _format_indices(indices: Sequence[int]) -> str:
    if not indices:
        return "(none)"
    return ", ".join(str(index) for index in indices)


def print_dataset_indices(
    datasets: Sequence[str],
    available_by_source: Dict[Tuple[str, int], List[int]],
) -> None:
    for position, dataset in enumerate(datasets):
        if position:
            print()
        print(f"{dataset}:")
        print(f"  clean (label0): {_format_indices(available_by_source.get((dataset, 0), []))}")
        print(f"  backdoor (label1): {_format_indices(available_by_source.get((dataset, 1), []))}")


def download_padbench(args) -> None:
    if args.show_list:
        if not args.datasets:
            print(f"Listing PADBench folders for '{args.repo_id}'...")
            print_padbench_folders(args.repo_id, list_padbench_folders(args.repo_id))
            return

        available_datasets = list_padbench_folders(args.repo_id)
        try:
            validate_requested_datasets(args.datasets, available_datasets)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        print(f"Listing available indices for '{args.repo_id}'...")
        print_dataset_indices(
            args.datasets,
            list_available_indices(args.repo_id, args.datasets),
        )
        return

    if args.datasets:
        available_datasets = list_padbench_folders(args.repo_id)
        try:
            validate_requested_datasets(args.datasets, available_datasets)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    clean_sources = args.clean_sources
    backdoored_sources = args.backdoored_sources

    print(f"Listing available model folders for '{args.repo_id}'...")
    available_by_source = list_available_indices(
        args.repo_id,
        relevant_subsets(clean_sources, backdoored_sources),
    )

    try:
        clean_patterns, clean_selection = select_patterns(
            request=args.clean_request,
            sources=clean_sources,
            available_by_source=available_by_source,
            name="clean",
        )
        backdoored_patterns, backdoored_selection = select_patterns(
            request=args.backdoored_request,
            sources=backdoored_sources,
            available_by_source=available_by_source,
            name="backdoored",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_dir=args.local_dir,
        )
    except httpx.ConnectError as exc:
        _fatal_hf_connect_error("Downloading selected adapters", args.repo_id, exc)
    print("Download complete!")
