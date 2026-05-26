from __future__ import annotations

import argparse
from pathlib import Path

from .merge_feature_files import finalize_schema_group_merge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize schema-group spectral feature outputs into one merged feature directory"
    )
    parser.add_argument("--schema-report-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = finalize_schema_group_merge(
        schema_report_path=args.schema_report_path,
        output_dir=args.output_dir,
    )
    print("Finalized schema-group spectral merge")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Merge report: {outputs['merge_report_path']}")
    print(f"Dataset references: {outputs['dataset_reference_report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
