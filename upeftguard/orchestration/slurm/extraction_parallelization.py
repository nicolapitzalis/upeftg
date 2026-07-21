"""Rank-specific packed Slurm settings for spectral extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RankParallelizationSetting:
    rank: int
    nodes: int
    cpus_per_worker: int
    workers_per_node: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_rank_parallelization_settings(
    values: Iterable[str] | None,
) -> dict[int, RankParallelizationSetting]:
    """Parse mandatory ``RANK:NODES:CPUS_PER_WORKER:WORKERS_PER_NODE`` settings."""

    if not values:
        return {}
    resolved: dict[int, RankParallelizationSetting] = {}
    for raw_value in values:
        raw = str(raw_value).strip()
        fields = raw.split(":")
        if len(fields) != 4:
            raise ValueError(
                "Parallelization settings must use RANK:NODES:CPUS_PER_WORKER:WORKERS_PER_NODE, got " + repr(raw)
            )
        try:
            rank, nodes, cpus_per_worker, workers_per_node = (int(value) for value in fields)
        except ValueError as exc:
            raise ValueError(f"Parallelization setting contains a non-integer value: {raw!r}") from exc
        for name, value in (
            ("rank", rank),
            ("nodes", nodes),
            ("cpus_per_worker", cpus_per_worker),
            ("workers_per_node", workers_per_node),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive in parallelization setting {raw!r}")
        if rank in resolved:
            raise ValueError(f"Duplicate parallelization setting for rank {rank}")
        resolved[rank] = RankParallelizationSetting(
            rank=rank,
            nodes=nodes,
            cpus_per_worker=cpus_per_worker,
            workers_per_node=workers_per_node,
        )
    return resolved
