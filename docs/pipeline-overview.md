# UPEFTGuard Pipeline Overview

This document explains how a request moves through UPEFTGuard, from a CLI
command to extracted features, a trained model, and final reports. It is aimed
at readers who have not worked on the codebase before.

The most important idea is that UPEFTGuard has **one scientific pipeline** and
two ways to execute it:

```text
manifest
   -> feature extraction
   -> optional feature aggregation for sequence models
   -> supervised training
   -> evaluation and reporting
```

The pipeline can run in the current Python process or be distributed through
Slurm. Slurm changes where and how the work runs; it does not provide a second
implementation of the scientific operations.

## 1. The responsibility chain

A public request passes through several layers:

```text
CLI flags
   -> commands
   -> workflows
   -> domain operations
   -> artifacts and reports
```

Slurm adds an execution branch below workflows:

```text
commands
   -> workflows
         |-> local domain operations
         `-> Slurm orchestration -> jobs -> local domain operations
```

Each package has a distinct role.

| Package | Responsibility |
| --- | --- |
| `commands` | Declares CLI flags, parses user input, and calls a workflow. |
| `workflows` | Coordinates a complete use case and chooses local or Slurm execution. |
| `orchestration` | Plans shards and translates workflow requests into Slurm jobs, arrays, resources, logs, and dependencies. |
| `features` | Computes feature values and returns in-memory feature bundles. |
| `artifacts` | Persists, loads, merges, documents, and structurally transforms artifacts. |
| `supervised` | Owns supervised data preparation, validation, models, evaluation, and the training lifecycle. |
| `reporting` | Writes stable experiment reports, summaries, prediction tables, and configuration documents. |
| `utilities` | Provides shared, domain-neutral support such as manifests, serialization, paths, and run contexts. |

Two boundaries are especially useful when reading the code:

- Commands answer **what the user requested**; workflows answer **how to carry
  out that use case**.
- Workflows answer **what should run**; orchestration answers **how to run it on
  Slurm**.

Utilities are not another pipeline stage. They sit underneath the other
packages and must not own extraction, training, scheduling, or reporting
decisions.

## 2. Public workflows

The package exposes five workflows through
[`upeftguard/workflows`](../upeftguard/workflows):

| Workflow | Purpose |
| --- | --- |
| `run_feature_extraction` | Read adapters described by a manifest, compute spectral features, and write a feature artifact. |
| `run_feature_aggregation` | Convert extracted features into a consistent representation for supervised models. |
| `run_supervised_training` | Prepare training data, tune candidates, select a winner, refit it, calibrate it, and export a self-contained model. |
| `run_checkpoint_inference` | Score an explicit inference manifest and feature artifact with an exported checkpoint. |
| `run_full_experiment` | Run extraction, optional sequence aggregation, training, and inference sequentially. |

`workflows/common.py` and `workflows/config.py` support these workflows but are
not workflows themselves.

The usual public CLI equivalents are:

```bash
python -m upeftguard.cli experiment extract ...
python -m upeftguard.cli experiment aggregate ...
python -m upeftguard.cli experiment train ...
python -m upeftguard.cli experiment infer ...
python -m upeftguard.cli experiment full ...
```

The CLI handlers translate flags into typed Python arguments. They should not
contain feature algorithms, training logic, artifact processing, or Slurm job
construction.

## 3. Local and Slurm execution

Every public workflow accepts a `backend` choice.

### Local backend

`backend="local"` means: perform the operation in the current Python process.
It does not necessarily mean a laptop or a login node. A Slurm compute job also
uses the local backend once it starts running.

For example:

```text
run_feature_aggregation(backend="local")
   -> artifacts.aggregation.aggregate_features(...)
```

### Slurm backend

`backend="slurm"` means: ask the Slurm orchestration layer to submit the
operation.

```text
run_feature_aggregation(backend="slurm")
   -> orchestration.slurm.experiment.submit_feature_aggregation(...)
   -> sbatch
   -> compute node runs `experiment aggregate --backend local`
   -> artifacts.aggregation.aggregate_features(...)
```

The high-level submission functions live in
[`orchestration/slurm/experiment.py`](../upeftguard/orchestration/slurm/experiment.py):

```text
submit_feature_extraction
submit_feature_aggregation
submit_checkpoint_inference
submit_supervised_training
```

These functions own submitted CLI command construction, resource resolution,
memory requests, job names, log paths, working directories, and the final call
to the low-level Slurm client. Workflows still choose the backend, pass the
requested execution policy, connect stages, record run metadata, and return a
consistent result.

This is a dependency, not a responsibility overlap: workflows use
orchestration without knowing how `sbatch` is assembled.

## 4. Feature extraction

Feature extraction starts from a manifest describing model adapters and their
labels or dataset membership.

The local flow is:

```text
manifest JSON
   -> parse manifest items
   -> features.registry.extract_features(...)
   -> FeatureBundle in memory
   -> artifacts.extraction.write_extracted_feature_bundle(...)
   -> feature files and provenance
```

The boundary between `features` and `artifacts` is deliberate:

- `features` computes values and returns an in-memory bundle.
- `artifacts` decides how the bundle is named, stored, loaded, and documented.

Feature algorithms therefore do not need to know the run-directory layout or
the public metadata format.

### Extracted feature artifact

The feature bundle is represented on disk by a coordinated set of files:

| File | Meaning |
| --- | --- |
| `spectral_features.npy` | Feature matrix with one row per retained model. |
| `spectral_labels.npy` | Optional labels aligned with feature rows. |
| `spectral_model_names.json` | Model name aligned with each feature row. |
| `spectral_metadata.json` | Public extractor, schema, feature, dataset-layout, skip, and merge metadata. |
| `dataset_reference_report.json` | Public provenance summary for datasets, labels, model families, attacks, and completeness. |
| `.spectral_metadata_state.json` | Full internal metadata required for reliable downstream processing. |
| `.dataset_reference_state.json` | Full internal provenance, including detailed per-model state. |

The public JSON files are intentionally more concise. The hidden state files
preserve information that machines need for later merging and transformation.

### Slurm extraction

Slurm extraction has more execution steps but produces the same logical final
artifact:

```text
manifest
   -> discover compatible schema groups
   -> divide groups into shards
   -> packed extraction workers by rank/schema
   -> merge each schema group's shard artifacts
   -> merge compatible schema-group outputs
   -> final feature bundle
```

While it runs, it keeps shard manifests, shard outputs, the schema partition,
and a compact job index under `extraction/.work/`. These are operational Slurm
state, not scientific outputs. Schema and shard intermediates are deleted
automatically after a successful merge; logs remain under `extraction/logs/`.

## 5. Feature aggregation

Extraction emits measurements tied to adapter blocks, layers, attention kinds,
and feature names. Supervised sequence models need those measurements arranged
along predictable structural axes.

Aggregation converts the extracted table into a canonical representation:

```text
extracted table: models x raw feature columns

                 becomes

model x architecture layer x attention/adapter slot x feature
```

Different architectures can have different depths and structures. Aggregation
therefore creates canonical axes and companion masks or padding so downstream
models can distinguish real values from empty structural positions.

In short:

- Extraction asks: **what measurements were obtained?**
- Aggregation asks: **how should those measurements be arranged for learning?**

Aggregation is artifact processing, not feature extraction and not model
training. Its implementation lives under
[`artifacts/aggregation`](../upeftguard/artifacts/aggregation).

CNN, DANN, and Transformer models require this layer-sequence representation.
Classical scikit-learn models accept the extracted tabular spectral
representation directly, so `run_full_experiment` skips aggregation for them.

## 6. The supervised lifecycle

Supervised training is implemented as a resumable three-stage lifecycle:

```text
PREPARE -> WORKER(S) -> FINALIZE
```

### Prepare

Preparation:

- loads the training manifest and the feature representation required by the
  selected model;
- resolves training and optional calibration partitions;
- derives task labels;
- builds cross-validation splits;
- expands model and hyperparameter candidates into tuning tasks; and
- writes the tuning manifest.

The tuning manifest is internal worker state placed at:

```text
<output-root>/<run-id>/training/.work/tuning.json
```

It is the shared contract between preparation, tuning workers, and
finalization. It records resolved paths, partitions, feature configuration,
task definition, CV splits, candidates, runtime settings, and warnings.

### Workers

Each worker executes one indexed tuning candidate. It:

- creates the requested model through the model registry;
- trains and evaluates it across the prepared CV splits; and
- writes a structured task-result file.

Workers can run sequentially in one process or as packed Slurm batches. The tuning
logic is the same in both cases.

### Finalize

Finalization:

- verifies that every expected worker result exists;
- selects the best candidate using the configured metric;
- refits the winner on the intended training data, or promotes the winning
  candidate's best validation-fold estimator when `--no-refit` is set;
- generates train and calibration predictions when available, plus one
  `validation.csv` for every CV fold and a top-level selected-fold validation
  prediction file in no-refit mode;
- selects calibrated thresholds when calibration is enabled;
- computes overall and grouped evaluations;
- saves the winning model; and
- hands completed results to `reporting`.

Slurm maps directly onto these stages:

```text
supervised prepare job
   -> packed tuning worker batches
   -> supervised finalize job
```

Orchestration distributes the lifecycle; it does not decide how candidates are
trained or which candidate wins.

### Model independence

The lifecycle creates estimators through
[`supervised/models/registry.py`](../upeftguard/supervised/models/registry.py).
Models share the protocol defined in
[`supervised/contracts.py`](../upeftguard/supervised/contracts.py), including
operations such as `fit`, prediction, and `save`.

This makes the pipeline model-extensible, but not completely plug-and-play.
Conventional estimators can usually be added by implementing the shared
surface and registering a factory, parameter grid, backend, and supported
representation. Models requiring special inputs or training behaviour may
also require lifecycle support. The current lifecycle contains explicit logic
for CNN/Transformer training, Transformer epochs and checkpoints, and DANN
domain-adaptation inputs.

## 7. Experiment variants use the same pipeline

Binary detection, attack-family multiclass classification, and leave-one-out
experiments are configurations of the same supervised lifecycle. They are not
separate pipelines.

For complete public CLI templates, explicit defaults, and runnable variant
examples, see the [Experiment CLI Runbook](experiment-runbook.md).

The main experiment axes are:

| Axis | Current choices | What it changes |
| --- | --- | --- |
| Task mode | `binary`, `attack_family_multiclass` | Label vocabulary, model output, and score interpretation. |
| CV strategy | `stratified`, `attack_family_leave_one_out`, `dataset_leave_one_out` | How training data is divided for candidate evaluation. |
| Manifest design | Complete population plus split, or explicit train/inference manifests | Which samples are available for fitting and final inference. |
| Model | Registered classical, CNN, DANN, or Transformer models | The estimator trained by each candidate. |
| Calibration | Optional calibration split and accepted FPRs | Whether operating thresholds are selected under an FPR constraint. |

Examples:

```text
Binary attack-family leave-one-out CV
   task_mode = binary
   cv_strategy = attack_family_leave_one_out

Attack-family multiclass classification
   task_mode = attack_family_multiclass
   multiclass_attack_names = [...]
   cv_strategy = stratified or dataset_leave_one_out
```

`attack_family_leave_one_out` CV currently supports only binary tasks, so it
cannot be combined directly with attack-family multiclass training.

It is also important to distinguish two meanings of "held out":

- A CV strategy holds out data temporarily inside training to compare model
  candidates.
- A resolved inference manifest defines a population that is never used for
  tuning and is evaluated only by the inference workflow.

For example, a zero-shot leave-one-attack experiment can put the unseen attack
in the inference manifest while using a supported CV strategy on the training
manifest. That is an outer experimental holdout, not an inner CV fold.

Regardless of these choices, execution remains:

```text
prepare -> tune candidates -> select winner -> refit -> evaluate -> report
```

The resolved choices are recorded in the internal tuning manifest and the final
experiment report, which makes runs interpretable after the original CLI
command is gone.

## 8. Reporting and final outputs

The end of supervised finalization has three separate responsibilities:

```text
supervised/evaluation
   -> computes metrics and threshold decisions

supervised/lifecycle/finalization.py
   -> assembles the completed experiment result

reporting
   -> writes the stable external representation
```

Reporting does not train a model, choose a winner, or invent evaluation
decisions. It serializes completed results into standardized reports,
configuration snapshots, prediction tables, thresholds, summaries, and tuning
candidate records.

## 9. Checkpoint inference

Checkpoint inference is independent of training worker state:

```text
checkpoint + inference manifest + compatible feature artifact
   -> load the checkpoint's inference contract
   -> align the explicit inference population
   -> predict and evaluate
   -> write a new inference reporting bundle
```

The exported `.pt` or `.joblib` artifact contains model preprocessing, task and
feature contracts, and calibrated thresholds. The inference manifest and its
compatible feature artifact remain separate inputs. Sequence checkpoints use
the aggregated layer-sequence representation; classical checkpoints can use
the extracted tabular representation.

## 10. The full experiment

`run_full_experiment` composes existing workflows rather than implementing new
scientific logic:

```text
run_feature_extraction
   -> run_feature_aggregation when the selected model requires layer sequences
   -> run_supervised_training
   -> run_checkpoint_inference
```

Locally, the selected workflows run sequentially in the current process. On
Slurm, the full workflow connects their job dependencies. Sequence models use:

```text
extraction controller
   -> extraction workers
   -> extraction merge
   -> aggregation job
   -> supervised prepare
   -> tuning workers
   -> supervised finalize
   -> checkpoint inference
```

For classical models, supervised preparation depends directly on the
extraction merge; no aggregation job is submitted.

The full workflow decides this use-case ordering. The helper
`orchestration.slurm.experiment.afterok_dependency` translates stage results
into Slurm dependency expressions, keeping scheduler syntax outside the
workflow package.

Each component workflow remains independently runnable, which is useful when
reusing an existing feature artifact or rerunning only training.

## 11. Run directories

Every command has one authoritative root. Only stages selected by the command
are created:

```text
runs/<run-id>/
|-- experiment.json
|-- extraction/
|-- aggregation/
|-- training/
`-- inference/
```

- `experiment.json` maps the selected stages, Slurm jobs, and important artifacts.
- Each stage owns its configuration, reports, logs, timings, and scientific outputs.
- A standalone command creates only its own stage. `full` creates extraction,
  training, and inference, plus aggregation when the selected model requires
  the layer-sequence representation.
- Reusable feature bundles can be consumed from another run without copying them.

Local and Slurm execution use the same stage directories. Slurm controller
configuration and logs live beside the outputs of the stage they coordinate.

## 12. A practical code-reading path

For a first pass through the implementation, read these files in order:

1. [`commands/experiment.py`](../upeftguard/commands/experiment.py) to see how
   public CLI arguments become workflow calls.
2. [`workflows/full.py`](../upeftguard/workflows/full.py) to see the end-to-end
   composition.
3. The individual files in
   [`workflows`](../upeftguard/workflows) to compare local and Slurm branches.
4. [`orchestration/slurm/experiment.py`](../upeftguard/orchestration/slurm/experiment.py)
   to see the high-level submission boundary.
5. [`features/registry.py`](../upeftguard/features/registry.py) and
   [`artifacts/extraction.py`](../upeftguard/artifacts/extraction.py) to see the
   computation/persistence split.
6. [`artifacts/aggregation`](../upeftguard/artifacts/aggregation) to understand
   the supervised input representation.
7. [`supervised/lifecycle/pipeline.py`](../upeftguard/supervised/lifecycle/pipeline.py)
   for the prepare/worker/finalize dispatch.
8. [`supervised/models/registry.py`](../upeftguard/supervised/models/registry.py)
   and [`supervised/contracts.py`](../upeftguard/supervised/contracts.py) for
   model creation and shared contracts.
9. [`supervised/lifecycle/finalization.py`](../upeftguard/supervised/lifecycle/finalization.py)
   and [`reporting`](../upeftguard/reporting) for final outputs.

When navigating unfamiliar code, keep this compact mental model nearby:

```text
commands translate
workflows coordinate
orchestration schedules
features compute
artifacts persist and transform
supervised trains and evaluates
reporting presents
utilities support
```
