# upeftguard

Stable CNN-only pipeline for spectral PEFT adapter features.

The active codebase exposes one public workflow:

```bash
python -m upeftguard.cli cnn ...
```

Older clustering, GMM, SVD, one-off experiment launchers, and legacy tests are
archived under `archive/`. The complete pre-refactor state is also preserved on
the git branch `archive/pre-cnn-cli-refactor-20260526` and tag
`pre-cnn-cli-refactor-20260526`.

## Environment

```bash
conda env create -f environment.yml
conda activate upeftg
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate upeftg
```

## Active Layout

- `upeftguard/features`: stable spectral feature extraction
- `upeftguard/pipelines`: CNN pipeline orchestration and Slurm submission
- `upeftguard/supervised`: CNN training/finalization and checkpoint inference helpers
- `upeftguard/utilities/artifacts`: aggregation, spectral metadata, and dataset-reference helpers
- `upeftguard/utilities/merge`: spectral sharding and merge/finalize utilities
- `upeftguard/utilities/data`: dataset download helpers
- `upeftguard/utilities/maintenance`: backfill utilities for older artifacts
- `sbatch/`: generic Slurm scripts used by the CLI
- `archive/`: inactive research and trial code

`runs/` is intentionally not reorganized by this refactor because historical
metadata can contain absolute artifact paths.

## Manifest Contract

Use one `--manifest-json`.

Single-pool manifests use a `path` section. Training can create a train/infer
split with `--train-split`:

```json
{
  "path": [
    {
      "path": "llama2_7b_toxic_backdoors_hard_rank256_qv/llama2_7b_toxic_backdoors_hard_rank256_qv_label0_",
      "indices": [0, 249]
    }
  ]
}
```

Joint manifests use `train` and `infer` sections and already define the split:

```json
{
  "train": [{"path": "train_prefix_", "indices": [0, 249]}],
  "infer": [{"path": "heldout_prefix_", "indices": [0, 249]}]
}
```

Leave-one-out experiments should be represented as generated or pre-existing
joint manifests, then submitted one manifest per run.

## CNN CLI

Slurm is the default backend and uses the `extra` partition by default:

```bash
python -m upeftguard.cli cnn full --manifest-json <MANIFEST_JSON>
python -m upeftguard.cli cnn extract --manifest-json <MANIFEST_JSON>
python -m upeftguard.cli cnn aggregate --feature-file <FEATURE_FILE>
python -m upeftguard.cli cnn train --manifest-json <MANIFEST_JSON> --feature-file <AGG_FEATURE_FILE>
python -m upeftguard.cli cnn infer --run-dir runs/supervised/<RUN_ID>
```

Common backend options:

```bash
--backend slurm|local
--partition extra
--worker-cpus auto
--max-concurrent auto
--dry-run
```

Local execution is explicit:

```bash
python -m upeftguard.cli cnn full \
  --backend local \
  --manifest-json manifests/single_datasets/llama2_7b_toxic_backdoors_hard.json \
  --run-id cnn_local_demo
```

## Stages

### Full Pipeline

```bash
python -m upeftguard.cli cnn full \
  --manifest-json <MANIFEST_JSON> \
  --run-id <RUN_ID>
```

This runs:

- spectral feature extraction
- CNN layer-sequence aggregation
- CNN training/finalization

For Slurm, the CLI submits dependent jobs for extraction, aggregation, and
training. Use `--dry-run` to inspect the generated `sbatch` commands.

### Feature Extraction

```bash
python -m upeftguard.cli cnn extract \
  --manifest-json <MANIFEST_JSON> \
  --run-id <RUN_ID>
```

This exposes only the stable spectral extractor. SVD is archived and is not part
of the public CLI.

Slurm extraction writes the final feature bundle under:

```text
runs/feature_extract/<RUN_ID>/merged/spectral_features.npy
```

Local extraction writes under:

```text
runs/feature_extract/<RUN_ID>/features/spectral_features.npy
```

### CNN Aggregation

```bash
python -m upeftguard.cli cnn aggregate \
  --feature-file runs/feature_extract/<RUN_ID>/merged/spectral_features.npy \
  --output-filename <RUN_ID>_cnn_layer_sequence
```

The command creates the `layer_sequence` representation expected by `cnn_1d`,
including the generated mask and group-name companion files.

### CNN Training

```bash
python -m upeftguard.cli cnn train \
  --manifest-json <MANIFEST_JSON> \
  --feature-file runs/feature_extract/<RUN_ID>_cnn_layer_sequence/merged/spectral_features.npy \
  --run-id <RUN_ID>
```

The reporting style is the existing supervised style. Key outputs include:

- `run_config.json`
- `artifact_index.json`
- `timings.json`
- `reports/supervised_report.json`
- `reports/results_summary.md`
- `reports/inference_scores.csv` when the manifest has an inference split

### Checkpoint Inference

```bash
python -m upeftguard.cli cnn infer \
  --run-dir runs/supervised/<RUN_ID>
```

or:

```bash
python -m upeftguard.cli cnn infer \
  --checkpoint runs/supervised/<RUN_ID>/models/best_model.pt
```

Inference loads the saved CNN checkpoint and tuning manifest, scores the
manifest inference split, and writes the same supervised report/summary style
under `runs/supervised_inference/<RUN_ID>/`.

## Timing Metadata

Every CNN command writes timing metadata with:

- `start_timestamp_utc`
- `end_timestamp_utc`
- `elapsed_seconds`
- backend and Slurm submission metadata where applicable

For Slurm, the submitter writes submission metadata immediately. Worker and
finalize stages also write completion timing into their stage outputs and
`timings.json`.

## Internal Compatibility Commands

The root CLI still contains internal compatibility commands used by Slurm
scripts:

- `python -m upeftguard.cli feature extract`
- `python -m upeftguard.cli run supervised`
- `python -m upeftguard.cli util aggregate-features`
- `python -m upeftguard.cli util merge-features`
- `python -m upeftguard.cli util export-feature-subset`
- `python -m upeftguard.cli util download-dataset`

These are not the recommended public workflow. Prefer the `cnn` namespace for
new runs.
