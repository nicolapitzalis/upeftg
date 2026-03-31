from __future__ import annotations

import argparse
from pathlib import Path

from ..artifacts.spectral_metadata import (
    dataset_layouts_from_source,
    load_spectral_metadata,
    resolve_dataset_reference_for_metadata,
    write_spectral_metadata,
)


def backfill_spectral_metadata(root: Path) -> dict[str, int]:
    resolved_root = root.expanduser()
    if not resolved_root.is_absolute():
        resolved_root = (Path.cwd().resolve() / resolved_root).resolve()
    else:
        resolved_root = resolved_root.resolve()

    written = 0
    for metadata_path in sorted(resolved_root.rglob("spectral_metadata.json")):
        internal_metadata = load_spectral_metadata(metadata_path)
        dataset_reference_payload = resolve_dataset_reference_for_metadata(metadata_path)
        write_spectral_metadata(
            metadata_path,
            internal_metadata=internal_metadata,
            dataset_layouts=dataset_layouts_from_source(
                metadata=internal_metadata,
                dataset_reference_payload=dataset_reference_payload,
            ),
        )
        written += 1

    return {"metadata_files_rewritten": int(written)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite spectral_metadata.json files to the slim public format and hidden state sidecar."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Root directory to scan for spectral_metadata.json files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stats = backfill_spectral_metadata(args.root)
    print("Spectral metadata backfill complete")
    print(f"Metadata files rewritten: {stats['metadata_files_rewritten']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
