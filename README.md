# upeftguard

Spectral PEFT-adapter feature extraction and supervised backdoor detection.

The public workflow is model-neutral. It supports classical scikit-learn
estimators, CNN, DANN, and Transformer models through one CLI:

```bash
python -m upeftguard.cli experiment <extract|aggregate|train|infer|full>
```

## Environment

Create or update the Conda environment, then activate it:

```bash
conda env create -f environment.yml
conda activate upeftg
```

```bash
conda env update -f environment.yml --prune
conda activate upeftg
```

Run commands from the repository root.

## Quick Start

Slurm is the default backend. This command extracts features, creates an 80/20
train/inference split, trains a logistic-regression model, and evaluates it:

```bash
python -m upeftguard.cli experiment full \
  --manifest-json manifests/rank_exploration/llama2_7b_tbh_rank256.json \
  --train-split 80 \
  --model logistic_regression \
  --run-id logistic_tbh_rank256_train80
```

Both `train` and `full` require an explicit `--model`. Use the single
`--hyperparams` flag for that model's candidate-grid JSON. It is required for
CNN, DANN, and Transformer models and optional for overriding a classical
model's registered grid. See the
[experiment runbook](docs/experiment-runbook.md#model-hyperparameters).

Use `--backend local` to execute in the current process, or add `--dry-run` to
inspect Slurm submissions without launching jobs.

`full` requires either:

- one complete `--manifest-json` plus `--train-split`; or
- both `--train-manifest-json` and `--infer-manifest-json`.

## Public Workflows

| Command | Purpose |
| --- | --- |
| `experiment extract` | Extract spectral features from adapters in a manifest. |
| `experiment aggregate` | Convert extracted features into the layer-sequence representation used by CNN, DANN, and Transformer models. |
| `experiment train` | Train and finalize a selected supervised model. Sequence models consume aggregated features; classical models can consume extracted tabular features directly. |
| `experiment infer` | Evaluate a self-contained `.pt` or `.joblib` checkpoint against an explicit inference manifest and compatible feature artifact. |
| `experiment full` | Run extraction, optional aggregation, training, and checkpoint inference as one local or Slurm workflow. |

The authoritative run root is `runs/<run-id>/`. Each selected stage owns its
inputs, operational state, logs, scientific artifacts, and reports beneath that
root.

## Documentation

- [Architecture](docs/architecture.md): package boundaries, dependency direction, public APIs, and runtime-data policy.
- [Pipeline overview](docs/pipeline-overview.md): end-to-end data flow, artifact representations, supervised lifecycle, and output layout.
- [Experiment CLI runbook](docs/experiment-runbook.md): complete commands, flag defaults, manifest forms, cross-validation, calibration, and experiment variants.
- [Spectral features](docs/spectral-features.md): feature definitions, numerical implementation, spectral flags, schemas, and metadata.
- [Slurm orchestration](docs/slurm-orchestration.md): job graphs, resource discovery, packed workers, dependencies, logs, dry runs, and troubleshooting.

Use the live parser to inspect the installed checkout:

```bash
python -m upeftguard.cli experiment <COMMAND> --help
```

## Verification

Run the same checks as CI from the activated environment:

```bash
python -m compileall -q upeftguard
python -c 'import json, pathlib; [json.loads(path.read_text()) for path in pathlib.Path("manifests").rglob("*.json")]'
ruff check upeftguard scripts
git diff --check
```

## Historical Code

Inactive clustering, GMM, legacy standalone SVD, and one-off experiment code
lives under `archive/`. The complete pre-refactor state is preserved on branch
`archive/pre-cnn-cli-refactor-20260526` and tag
`pre-cnn-cli-refactor-20260526`.

Historical run directories are not reorganized because their metadata can
contain absolute artifact paths.
