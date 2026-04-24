#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


ATTACKS = ("RIPPLE", "insertsent", "stybkd", "syntactic")
DATASETS = ("ag_news", "imdb")
CLEAN_SOURCE_ATTACK = "insertsent"
CLEAN_TRAIN_INDICES = [0, 124]
CLEAN_INFER_INDICES = [125, 249]
ATTACK_INDICES = [0, 249]
EXPECTED_MANIFEST_COUNT = len(ATTACKS)
SOURCE_HELDOUT_MANIFEST_RE = re.compile(
    r"^holdout_llama2_7b_(?:ag_news|imdb)_(?:RIPPLE|insertsent|stybkd|stykbd|syntactic)_rank256_qv\.json$"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _subset_name(dataset: str, attack: str) -> str:
    return f"llama2_7b_{dataset}_{attack}_rank256_qv"


def _entry(dataset: str, attack: str, label: int, indices: list[int]) -> dict[str, Any]:
    subset_name = _subset_name(dataset, attack)
    return {
        "path": f"{subset_name}/{subset_name}_label{int(label)}_",
        "indices": list(indices),
    }


def _validate_source_suite(source_root: Path) -> None:
    source_names = {
        path.name
        for path in source_root.glob("holdout_*.json")
        if SOURCE_HELDOUT_MANIFEST_RE.fullmatch(path.name) is not None
    }
    expected_source_names = {
        f"holdout_llama2_7b_{dataset}_{attack}_rank256_qv.json"
        for dataset in DATASETS
        for attack in ATTACKS
    }
    missing = sorted(expected_source_names - source_names)
    if missing:
        preview = ", ".join(missing)
        raise FileNotFoundError(
            f"source_root is missing expected attack-family leave-one-out manifests: {preview}"
        )


def _build_attack_holdout_manifest(heldout_attack: str) -> dict[str, list[dict[str, Any]]]:
    train: list[dict[str, Any]] = [
        _entry(dataset, CLEAN_SOURCE_ATTACK, 0, CLEAN_TRAIN_INDICES)
        for dataset in DATASETS
    ]
    infer: list[dict[str, Any]] = [
        _entry(dataset, CLEAN_SOURCE_ATTACK, 0, CLEAN_INFER_INDICES)
        for dataset in DATASETS
    ]

    for dataset in DATASETS:
        for attack in ATTACKS:
            target = infer if attack == heldout_attack else train
            target.append(_entry(dataset, attack, 1, ATTACK_INDICES))

    return {"train": train, "infer": infer}


def prepare_manifests(source_root: Path, output_root: Path) -> list[Path]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("source_root and output_root must be different directories")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    _validate_source_suite(source_root)

    output_root.mkdir(parents=True, exist_ok=True)
    for stale_path in output_root.glob("*.json"):
        stale_path.unlink()

    generated_paths: list[Path] = []
    for heldout_attack in ATTACKS:
        output_payload = _build_attack_holdout_manifest(heldout_attack)
        output_path = output_root / f"holdout_attack_family_{heldout_attack}_rank256_qv.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
            f.write("\n")
        generated_paths.append(output_path)

    if len(generated_paths) != EXPECTED_MANIFEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MANIFEST_COUNT} attack-family leave-one-out manifests, "
            f"generated {len(generated_paths)} from {source_root}"
        )
    return generated_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build multiclass attack-family leave-one-out manifests from the full LOO suite."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_repo_root() / "manifests" / "leave_one_out",
        help="Directory containing the full leave-one-out manifest suite.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "runs" / "generated_manifests" / "leave_one_out_attack_family_multiclass",
        help="Directory to write the filtered multiclass leave-one-out manifests.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the generated manifest root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generated_paths = prepare_manifests(args.source_root, args.output_root)
    output_root = args.output_root.expanduser().resolve()
    if args.quiet:
        print(output_root)
    else:
        print(f"Generated {len(generated_paths)} attack-family leave-one-out manifests in {output_root}")
        for path in generated_paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
