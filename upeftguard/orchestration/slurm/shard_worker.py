from __future__ import annotations

import json
import os
from pathlib import Path

from ...artifacts.extraction import write_extracted_feature_bundle
from ...features.registry import extract_features
from ...utilities.core.manifest import parse_single_manifest_json


def _resolve_shard_index(explicit_index: int | None) -> int:
    if explicit_index is not None:
        return int(explicit_index)
    value = os.environ.get("SLURM_PROCID")
    if value is not None:
        return int(value)
    raise ValueError("Shard index was not provided. Pass --shard-index or run under packed srun with SLURM_PROCID set.")


def run_schema_shard_worker(
    *,
    schema_report_path: Path,
    shard_index: int | None,
    dataset_root: Path,
    features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str,
    stream_block_size: int,
    dtype_name: str,
    group_ids: list[str] | None = None,
) -> int:
    resolved_shard_index = _resolve_shard_index(shard_index)
    report_path = schema_report_path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Schema report contains no groups: {report_path}")

    processed = 0
    selected_group_ids = set(group_ids or [])
    for group in groups:
        if selected_group_ids and str(group["group_id"]) not in selected_group_ids:
            continue
        n_shards = int(group["n_shards"])
        if resolved_shard_index >= n_shards:
            continue
        shard_manifest = Path(group["shard_manifest_dir"]) / f"shard_{resolved_shard_index}.json"
        shard_dir = Path(group["shard_output_root"]) / f"shard_{resolved_shard_index}"
        print(
            f"Extracting {group['group_id']} shard {resolved_shard_index}/{n_shards - 1}",
            flush=True,
        )
        items = parse_single_manifest_json(
            manifest_path=shard_manifest,
            dataset_root=dataset_root,
            section_key="path",
        )
        bundle, _warnings = extract_features(
            extractor_name="spectral",
            items=items,
            params={
                "block_size": int(stream_block_size),
                "dtype": str(dtype_name),
                "spectral_features": list(features),
                "spectral_sv_top_k": int(spectral_sv_top_k),
                "spectral_moment_source": str(spectral_moment_source),
                "spectral_qv_sum_mode": str(group["effective_spectral_qv_sum_mode"]),
                "spectral_entrywise_delta_mode": str(spectral_entrywise_delta_mode),
                "spectral_attention_granularity": str(spectral_attention_granularity),
            },
        )
        write_extracted_feature_bundle(
            bundle=bundle,
            items=items,
            extractor_name="spectral",
            output_dir=shard_dir,
        )
        processed += 1

    if processed == 0:
        raise ValueError(f"Shard index {resolved_shard_index} did not match any selected group in {report_path}")
    print(f"Completed shard index {resolved_shard_index} across {processed} schema groups")
    return processed
