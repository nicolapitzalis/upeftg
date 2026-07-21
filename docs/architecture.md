# Architecture

`upeftguard` separates scientific domain logic from execution and storage
mechanics. Public commands coordinate the following flow:

```text
manifest
  -> spectral extraction
  -> feature artifact + provenance
  -> optional layer-sequence aggregation for torch sequence models
  -> supervised preparation and cross-validation
  -> selected backend refit
  -> binary clean-vs-backdoor evaluation
  -> unified reporting bundle
```

## Boundaries

### Contracts

`upeftguard.contracts` owns neutral spectral feature-name parsing, layer
identifiers, and safe tensor-shape inspection shared by features and artifacts.

### Features

`upeftguard.features.spectral` owns spectral configuration, feature naming,
attention/QV layouts, metadata normalization, per-block calculations, and
extraction orchestration. `features.delta` remains focused on LoRA factors and
effective-delta operations.

### Artifacts

`upeftguard.artifacts` owns feature-table contracts, companion paths, loading,
schema selection, metadata reconciliation, dataset references, extraction
persistence, aggregation, subsetting, and all feature-artifact merge algorithms.
Its `aggregation`, `metadata`, `provenance`, and `merge` subpackages separate
those operations by responsibility.

### Supervised Domain

The root of `upeftguard.supervised` contains only the model-neutral contracts
and task vocabulary. Input preparation and normalization live in `data`, split
and cross-validation policy in `validation`, and prediction, scoring, and
threshold selection in `evaluation`.

`upeftguard.supervised.lifecycle` coordinates preparation, tuning workers,
winner selection, refit, finalization, and checkpoint inference.

`upeftguard.supervised.models` contains the registry, model-specific
hyperparameter-grid validation, shared runtime/loss/checkpoint helpers, and
backend implementations:

- `registry.py`: classical scikit-learn estimators and factories for every model
- `cnn`: structured input, networks, estimator, DANN, and checkpoints
- `transformer`: layer-sequence input, hierarchy network, estimator, and checkpoints

Backends satisfy the shared estimator protocol and are selected through the
registry rather than imported by lifecycle code opportunistically.

### Workflows And Execution

`upeftguard.workflows` exposes typed, CLI-independent extraction, aggregation,
training, inference, and full-experiment use cases. `commands` translates CLI
arguments into calls to these workflows.

`upeftguard.orchestration` owns scheduler mechanics and scheduler-neutral
parallel-work planning. Schema-aware shard planning lives in `sharding`; Slurm
resource resolution, dependencies, workers, and `sbatch` submission live in
`slurm`. Shard jobs call artifact merge operations rather than implementing
feature-table manipulation in orchestration.
Its `slurm` package separates the low-level submission client and resource
discovery from the extraction and supervised job-graph controllers.
See [Slurm orchestration](slurm-orchestration.md) for the complete execution
model, resource defaults, shard behavior, and operational guidance.

### Reporting

`upeftguard.reporting` writes a model-neutral bundle. Evaluation is always
projected to clean versus backdoor, including multiclass attack-family tasks.
The supported metrics are AUROC, AUPRC, accuracy, precision, recall, and the
confusion matrix. Threshold-dependent metrics are written at `0.5` and, when
calibration is enabled, at each selected threshold.

The concise experiment configuration is separate from reports. Dataset
partitions, input manifests, and the evaluated model grid are recorded under
the run's input metadata.

### CLI

`upeftguard.cli` only assembles the root parser and dispatches handlers from
`upeftguard.commands`. All flags, including internal Slurm-worker and artifact
maintenance flags, are declared under `commands`. Domain and orchestration
modules expose callable Python operations without standalone parsers.

### Utilities

`upeftguard.utilities` is limited to shared, domain-neutral mechanisms:
manifest parsing, storage paths, run contexts, JSON conversion, and PADBench
data access. Utilities do not own feature-artifact transformations or Slurm
job coordination and do not depend on domain packages.

## Dependency Direction

```text
commands -> workflows -> features / artifacts / supervised / reporting
                       -> orchestration
features / artifacts -> contracts
all domain packages -> utilities
orchestration -> domain public operations
```

`features` produces in-memory bundles. `artifacts` persists and transforms
them. Supervised domain code does not import Slurm orchestration, and utilities
do not import domain packages.

## Public API Policy

Package `__init__.py` modules expose stable, domain-oriented APIs where that
keeps callers independent of implementation layout. Legacy import and command
aliases from before the structural refactor are not part of the active API.
Existing checkpoints and feature artifacts remain readable without migration.

## Verification

Local verification covers:

- package compilation
- manifest JSON parsing
- Ruff static checks
- whitespace/error checks through `git diff --check`

Slurm behavior is validated through dry-run commands; real cluster jobs and
full-dataset training remain deployment-level checks.

## Runtime Data

`runs/` and `logs/` are generated runtime state. `results/` may contain curated
analysis intended for review. `archive/` and `old_version_results/` are
historical material. Removing, deduplicating, or migrating these directories
is deliberately outside the structural refactor and requires a separate
review.
