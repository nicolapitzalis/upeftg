# Slurm Orchestration

This guide describes how `upeftguard` submits and coordinates Slurm jobs. The
active implementation is Python-only: workflow-specific `sbatch` shell scripts
are not required.

## Quick Start

Slurm is the default backend for public experiment commands:

```bash
python -m upeftguard.cli experiment full \
  --manifest-json manifests/single_datasets/llama2_7b_toxic_backdoors_hard.json \
  --train-split 80 \
  --model logistic_regression \
  --run-id example_run
```

Inspect the submissions without launching jobs:

```bash
python -m upeftguard.cli experiment full \
  --manifest-json manifests/single_datasets/llama2_7b_toxic_backdoors_hard.json \
  --train-split 80 \
  --model logistic_regression \
  --run-id example_run \
  --dry-run
```

Run only feature extraction:

```bash
python -m upeftguard.cli experiment extract \
  --manifest-json <MANIFEST_JSON> \
  --run-id <RUN_ID> \
  --partition extra \
  --cpus-per-worker 4 \
  --nodes auto \
  --workers-per-node auto
```

## Package Layout

Slurm code is contained in `upeftguard/orchestration/slurm`:

| File | Responsibility |
| --- | --- |
| `client.py` | Builds `sbatch --wrap` commands, submits them, forwards environment values, parses job IDs, and reads dependency IDs from job indexes. |
| `resources.py` | Discovers homogeneous CPU capacity across non-DOWN partition nodes and resolves packed task indexes. |
| `extraction.py` | Prepares schema-aware extraction shards and submits the extraction worker and merge graph. |
| `supervised.py` | Prepares supervised tuning tasks and submits the tuning worker and finalization graph. |

Domain workers remain outside the orchestration package. In particular,
`upeftguard.orchestration.slurm.shard_worker` performs one extraction
shard, while the supervised lifecycle implements preparation, tuning, and
finalization.

## Submission Model

Every job is submitted as an argument vector through `sbatch --wrap`. For
example, a worker command is represented internally as:

```text
[python, -m, upeftguard.cli, run, supervised, --stage, worker, ...]
```

The client converts it to a safely quoted command similar to:

```bash
sbatch --parsable \
  --partition extra \
  --nodes 2 \
  --ntasks 16 \
  --ntasks-per-node 8 \
  --cpus-per-task 4 \
  --job-name upeftguard_supervised_worker_b0_example_run \
  --chdir '<REPOSITORY_ROOT>' \
  --wrap '<PYTHON> -m upeftguard.cli run supervised ...'
```

`--parsable` makes Slurm print a machine-readable job ID. If Slurm returns
`12345;cluster-name`, the client records `12345`.

The submitted command uses the absolute `sys.executable` of the process that
launched the CLI. Activate the desired Python environment before submission,
and ensure that the same absolute environment and repository paths are visible
from compute nodes.

## Common Slurm Flags And Defaults

All public `experiment` commands accept the following backend flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--backend` | `slurm` | Use `slurm` to submit jobs or `local` to execute in the current process. |
| `--partition` | `extra` | Slurm partition used for every job in the workflow. |
| `--cpus-per-worker` | `4` | CPUs assigned to each worker. |
| `--dry-run` | disabled | Build configs and submission metadata without executing `sbatch`. |
| `--output-root` | `runs` | Root for submission metadata and scientific artifacts. |
| `--run-id` | UTC timestamp | Stable name shared by related stages. The generated form is `YYYYMMDDTHHMMSSZ`. |

### Automatic Resource Resolution

Packed extraction and training discover partition capacity with:

```bash
sinfo -N -h -p <PARTITION> -o '%N|%c|%T'
```

Nodes in state `DOWN` are excluded. All other partition nodes count toward the
default `nodes`; allocated or busy nodes remain eligible and Slurm queues until
they can be assigned. The partition is required to be CPU-homogeneous. Defaults
are `cpus_per_worker=4` and
`workers_per_node=cpus_per_node//cpus_per_worker`.

## Full Experiment Dependency Graph

For CNN, DANN, and Transformer models, `experiment full` creates this graph:

```text
extraction prepare controller
            |
            +--> packed extraction jobs by rank/schema
                         |
                         +-- afterok --> extraction merge
                                              |
                                              +-- afterok --> aggregation
                                                                  |
                                                                  +-- afterok --> supervised prepare controller
                                                                                       |
                                                                                       +--> packed tuning batches
                                                                                                  |
                                                                                                  +-- afterok --> finalization
                                                                                                                       |
                                                                                                                       +-- afterok --> checkpoint inference
```

Classical models skip aggregation. Their supervised prepare controller depends
directly on the extraction merge, and finalization still feeds checkpoint
inference.

The extraction controller is submitted with `sbatch --wait` during a real full
run. This waits only for the controller to prepare shards, submit its child
jobs, and write `extraction/.work/job_index.json`; it does not wait for extraction workers
or merging to finish. The full workflow reads the merge job ID from that index
and uses it as the aggregation dependency for sequence models or the training
dependency for classical models.

All downstream dependencies use `afterok`. A failed worker batch therefore
prevents its merge or finalization job from running. A failed merge prevents
the next selected stage from starting, and a failed finalization prevents
checkpoint inference.

The current requests made by each job type are:

| Job | CPUs | Memory | Scheduling |
| --- | ---: | --- | --- |
| Extraction prepare controller | 2 | 8GB | No |
| Extraction worker | `--cpus-per-worker` | Cluster default | Packed |
| Extraction merge | 4 | Cluster default | No |
| Aggregation | `--cpus-per-worker` | 16GB | Single job |
| Supervised prepare controller | 4 | 16GB | No |
| Supervised tuning worker | `--cpus-per-worker` | Cluster default | Packed batches |
| Supervised finalization | 4 | Cluster default | No |
| Checkpoint inference | `--cpus-per-worker` | 16GB | Single job |

## Feature Extraction Orchestration

### Controller Job

The public extraction workflow first writes a JSON controller config
under the submission directory and submits:

```text
python -m upeftguard.cli run slurm-extraction-controller <CONFIG_PATH>
```

The prepare controller requests:

| Resource | Value |
| --- | --- |
| CPUs | `2` |
| Memory | `8GB` |
| Partition | `--partition` |

The controller resolves worker resources, inspects adapter schemas, writes
shard manifests, submits workers, submits the dependent merge, and atomically
writes the job index.

### Why Schema Groups Exist

A manifest can contain adapters with different tensor layouts, base-model
architectures, target modules, or adapter variants. Such adapters cannot always
be merged into one uniform spectral feature table without first handling their
schemas separately.

Preparation samples one adapter schema per inferred dataset source and creates
a schema signature. Most adapters use an exact signature containing block
names, LoRA A/B shapes, and optional E tensor information. AdaLoRA schemas use
a rank-tolerant signature based on block names, input/output dimensions, and E
presence so variable effective ranks can remain compatible.

Sources with the same signature are placed in the same schema group. Group IDs
have the form:

```text
group_000_<12-character-schema-digest>
```

The complete decision is recorded in:

```text
<OUTPUT_ROOT>/<RUN_ID>/extraction/.work/schema_partition_report.json
```

Useful report fields include:

- `group_count`, `n_items`, and `n_manifest_sources`;
- each group's `schema_digest` and `schema_signature_mode`;
- item and label counts;
- number of blocks and whether LoRA rank varies;
- requested and effective q+v mode;
- per-group manifest, shard, and merge paths;
- warnings about heterogeneous schemas or q+v fallbacks.

### How Shards Are Created

For each rank/schema group, worker capacity and shard count are:

```text
capacity = nodes * workers_per_node
n_shards = min(number_of_adapters, capacity)
Slurm ntasks = n_shards
```

Items are divided into contiguous balanced slices, and every worker computes
exactly one shard. Empty shards are never created. Each group receives its own
packed job, and `SLURM_PROCID` is the shard index inside that job.

### Extraction Worker Resources

Every extraction worker job submits:

```text
--nodes <nodes>
--ntasks <n_shards>
--ntasks-per-node <workers_per_node>
--cpus-per-task <cpus_per_worker>
```

The worker CPU count is also exported to common numerical libraries through:

```text
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
NUMEXPR_NUM_THREADS
VECLIB_MAXIMUM_THREADS
```

No explicit worker memory request is currently added, so the partition or
cluster default applies.

### Default And Rank-Specific Settings

The default is discovered from every non-DOWN node in the partition:

```text
nodes = number of non-DOWN partition nodes
cpus_per_worker = 4
workers_per_node = cpus_per_node // 4
```

It can be overridden globally:

```bash
python -m upeftguard.cli experiment extract \
  --manifest-json <MANIFEST_JSON> \
  --nodes 2 \
  --cpus-per-worker 8 \
  --workers-per-node 3
```

Rank overrides always contain four mandatory fields:

```bash
python -m upeftguard.cli experiment extract \
  --manifest-json <MANIFEST_JSON> \
  --parallelization-settings 8:2:4:8 256:2:8:4 2048:4:16:2
```

Each value has the form:

```text
RANK:NODES:CPUS_PER_WORKER:WORKERS_PER_NODE
```

Unlisted ranks use the default and emit a warning because optimal CPU and memory
allocations can differ by rank. The controller rejects settings that exceed the
partition's node or per-node CPU capacity. Lowering `workers_per_node` is the
intended memory-control mechanism for large ranks.

For `experiment full`, the equivalent flag is
`--extract-parallelization-settings`.

### q+v Behavior Across Schemas

`--spectral-qv-sum-mode` can be:

- `none`: extract original blocks only;
- `append`: preserve original q/v blocks and append supported q+v blocks;
- `only`: retain supported q+v blocks only.

Compatibility is evaluated separately for each schema group. If `append` is
requested but a group has no supported q/v pairs, that group falls back to
`none` and records a warning. If `only` is requested and a group has no
supported pairs, preparation fails because a meaningful output cannot be
created for that group.

### Merge Job

After the worker job succeeds, a merge job runs with:

| Resource | Value |
| --- | --- |
| CPUs | `4` |
| Memory | cluster default |
| Dependency | `afterok:<worker_job_id>` |

For one schema group, its shard outputs are merged directly into the final
directory. For multiple groups, every group is first merged into its own
directory and the group outputs are then finalized into one feature bundle.

The final Slurm extraction artifact is:

```text
<OUTPUT_ROOT>/<RUN_ID>/extraction/features/spectral_features.npy
```

### Extraction Defaults

| Flag | Default | Notes |
| --- | --- | --- |
| `--nodes` | `auto` | All non-DOWN nodes in the partition. |
| `--cpus-per-worker` | `4` | CPUs assigned to each extraction worker. |
| `--workers-per-node` | `auto` | `cpus_per_node // cpus_per_worker`. |
| `--features` | `energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank` | Spectral feature groups. |
| `--spectral-sv-top-k` | `8` | Number of leading singular values emitted by `sv_topk`. |
| `--spectral-moment-source` | `sv` | One of `entrywise`, `sv`, or `both`. |
| `--spectral-qv-sum-mode` | `none` | Default for `full` and `extract`; aggregation and standalone training default to `append`. |
| `--spectral-entrywise-delta-mode` | `auto` | One of `auto`, `dense`, or `stream`. |
| `--spectral-attention-granularity` | `module` | One of `module` or `head`. |
| `--stream-block-size` | `131072` | Block size for streaming entrywise calculations. |
| `--dtype` | `float32` | One of `float32` or `float64`. |

## Supervised Orchestration

### Prepare Controller

Training first writes `training/.work/slurm.json` and submits:

```text
python -m upeftguard.cli run slurm-supervised-controller <CONFIG_PATH>
```

The prepare controller requests `4` CPUs and `16GB` of memory. It loads data,
constructs splits, expands the selected model's hyperparameter candidates, and
writes the supervised tuning manifest.

One packed worker task is created per model/hyperparameter candidate. A task
can perform multiple CV fits internally across configured CV seeds and folds.
Therefore `n_tasks` is the number of candidates, not necessarily the total
number of model fits.

If `--cv-seeds` is omitted for a Slurm submission, the workflow uses
`42 43 44`. `--model` is required. The model-neutral `--hyperparams` JSON is
required for CNN/DANN/Transformer and optionally overrides a classical model's
registered grid. `--cv-folds` defaults to `5`, and `--cv-strategy` defaults to
`stratified`.

Important supervised defaults are:

| Flag | Default | Scheduling relevance |
| --- | --- | --- |
| `--model` | required | Chooses which hyperparameter candidate grid is expanded. |
| `--cv-folds` | `5` | Affects fits performed inside each candidate task when CV is active. |
| `--cv-strategy` | `stratified` | Leave-one-out strategies derive their own fold counts. |
| `--cv-seeds` | `42 43 44` on Slurm | Repeats CV inside candidate tasks; duplicates are removed. |
| `--input-normalization` | `none` | Scientific preprocessing; does not change Slurm resources. |
| `--train-split` | unset | Standalone `train` treats the manifest as training-only; `full` requires a value in `[1, 99]` with `--manifest-json`. |
| `--calibration-split` | unset | No calibration partition by default. |
| `--task-mode` | `binary` | Multiclass mode changes labels and metrics, not the job graph. |

Hyperparameter files determine the number of candidate tasks. Inspect
`<OUTPUT_ROOT>/<RUN_ID>/training/inputs/model_grid.json` and the tuning
manifest before launching a particularly large search.

A single candidate with ordinary stratified settings can use
`singleton_no_cv`, in which tuning does not need CV fits and the model is fit
during finalization. Explicit leave-one-out strategies or
`--cv-derived-refit-epochs` still require cross-validation for a single
candidate. `--no-refit` also forces CV because it must promote a persisted
validation-fold model.

### Packed Tuning Batches

Training uses the same canonical resources as extraction:

```text
capacity = nodes * workers_per_node
number_of_batches = ceil(n_tasks / capacity)
```

Each batch is a packed job using `--nodes`, `--ntasks-per-node`, and
`--cpus-per-task`. The final partial batch uses only its remaining task count.
Every worker runs:

```text
python -m upeftguard.cli run supervised \
  --stage worker \
  --run-dir <SUPERVISED_RUN_DIR> \
  --n-jobs <cpus_per_worker> \
  --task-index-offset <batch_start>
```

The candidate index is `task_index_offset + SLURM_PROCID`. Each task writes a
result under:

```text
<OUTPUT_ROOT>/<RUN_ID>/training/reports/tuning_tasks/task_NNNN.json
```

For four workers per node:

```bash
python -m upeftguard.cli experiment full \
  ... \
  --train-nodes 1 \
  --train-cpus-per-worker 4 \
  --train-workers-per-node 4
```

When candidate count exceeds capacity, additional packed batches are submitted;
no candidate is discarded. No explicit tuning-worker memory request is added.

### Finalization

Finalization requests `4` CPUs and depends on successful completion of the
all packed tuning batches:

```text
python -m upeftguard.cli run supervised \
  --stage finalize \
  --run-dir <SUPERVISED_RUN_DIR>
```

It validates that expected task results exist, selects the winning candidate,
refits as required, calibrates when requested, and writes the training reports
and self-contained model artifact. Its `final_dependency_job_id` becomes the
dependency for the separate inference job in a full experiment.

## Aggregation And Inference Jobs

Aggregation and checkpoint inference do not require controller graphs. Their
workflow functions submit one wrapped public CLI command with `--backend local`
so the scientific implementation remains the same on local and Slurm backends.

Both jobs request:

- CPUs set by `--cpus-per-worker`;
- `16GB` memory;
- the selected partition;
- workflow-specific output and error logs.

In a full sequence-model run, aggregation depends on the extraction merge and
the supervised prepare controller depends on aggregation. For a classical
model, supervised preparation depends directly on the extraction merge.
Standalone aggregation or inference has no dependency unless called
programmatically with one.

## Output Directories And Metadata

Submission metadata and scientific outputs share one authoritative run root:

```text
<OUTPUT_ROOT>/
└── <RUN_ID>/
    ├── experiment.json
    ├── extraction/
    │   ├── .work/slurm.json
    │   ├── .work/job_index.json
    │   ├── logs/
    │   └── features/spectral_features.npy
    ├── aggregation/
    │   ├── features/spectral_features.npy
    │   └── logs/
    ├── training/
    │   ├── .work/slurm.json
    │   ├── .work/job_index.json
    │   ├── .work/tuning.json
    │   ├── .work/prepared_arrays.npz
    │   ├── logs/
    │   ├── models/
    │   └── reports/
    └── inference/
        ├── .work/prepared_arrays.npz
        ├── logs/
        └── reports/
```

Standalone commands create only their corresponding stage. `.work/` contains
operational controller state, not scientific results. Extraction schema groups
and shards temporarily live there and are removed after a successful merge.
The `aggregation/` directory is absent from classical-model full runs because
that stage is skipped.

### Submission Configs

The stage-local `.work/slurm.json` files are controller inputs. They
record resolved paths, workflow arguments, requested Slurm resources, log
locations, and the repository working directory. They are also useful when
auditing exactly what a controller received.

### Job Indexes

Controllers write `.work/job_index.json` atomically, first after worker
submission and again after submitting the dependent merge or finalizer. Common
fields include:

- `controller_job_id`;
- worker, merge, or finalize job IDs;
- `final_dependency_job_id`;
- partition and resolved resource values;
- number of tasks or shards;
- compact resource and task counts needed to monitor the job graph.

Extraction additionally records schema-group counts and packed-mode
information. `final_dependency_job_id` is the value downstream
workflows should depend on.

### Log Name Substitutions

The generated paths use normal Slurm substitutions:

| Token | Meaning |
| --- | --- |
| `%j` | Job ID. |
| `%A` | Array job ID. |
| `%a` | Array task index. |
| `%t` | `srun` task rank in packed mode. |

## Dry Runs

`--dry-run` writes submission configs, run metadata, and fully assembled
`sbatch` argument lists, but does not:

- execute `sbatch`;
- inspect adapter schemas inside the extraction controller;
- create real child job IDs;
- run workers, merges, aggregation, finalization, or checkpoint inference.

The full workflow uses readable placeholder dependencies such as
`DRYRUN_FEATURE_FINALIZE`, `DRYRUN_AGGREGATE`, and `DRYRUN_TRAIN_FINALIZE` so
the intended graph remains visible in `experiment.json` and the stage-local
`.work/` files. Classical-model dry runs omit `DRYRUN_AGGREGATE` because that
stage is skipped.

## Environment And Filesystem Requirements

- Submit from the Python environment intended for compute jobs.
- The absolute Python interpreter path must exist on compute nodes.
- The repository, dataset root, output root, and any checkpoint directories
  must be visible at the same absolute paths from submit and compute nodes.
- Additional worker environment variables are passed to the `sbatch` process
  and rely on normal Slurm environment export behavior.
- Log directories are created before submission.
- Job names sanitize characters outside letters, digits, `_`, `.`, and `-` in
  controller-generated worker graphs.

## Monitoring And Troubleshooting

Start with the submission metadata and job index:

The following examples assume the default `--output-root runs`:

```bash
cat runs/<RUN_ID>/experiment.json
cat runs/<RUN_ID>/extraction/.work/job_index.json
cat runs/<RUN_ID>/training/.work/job_index.json
```

Then inspect Slurm state and accounting:

```bash
squeue -j <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Common failure patterns:

| Symptom | Likely cause or next check |
| --- | --- |
| `sinfo` is unavailable during automatic capacity discovery | Restore `sinfo` access on the controller node; discovery is also used to validate explicit settings. |
| Controller cannot import `upeftguard` | The recorded Python environment or repository path is not visible on the compute node. |
| Extraction worker processes the wrong shard | Confirm `SLURM_PROCID` in its packed-worker log. |
| Merge remains pending after a worker failure | Expected with `afterok`; inspect the failed packed worker first. |
| `only` q+v preparation fails | At least one schema group has no supported q/v pairing. Use `append` or `none`, or restrict the manifest. |
| Worker capacity validation fails | Ensure `workers_per_node * cpus_per_worker` fits the homogeneous node CPU count. |
| Finalization reports missing tuning tasks | Inspect the corresponding `task_NNNN.json` worker and packed-batch log. |
| Output paths contain older partial results | Use a new `--run-id` or deliberately review the existing run before reusing it. |

After a controller exits, canceling only its job ID does not cancel child jobs
that it already submitted. Use the worker, merge, aggregation, or finalization
IDs recorded in the job indexes when cancellation is required.

## Direct Controller Invocation

The controller modules can be invoked directly when debugging an already
written config:

```bash
python -m upeftguard.cli run slurm-extraction-controller \
  runs/<RUN_ID>/extraction/.work/slurm.json

python -m upeftguard.cli run slurm-supervised-controller \
  runs/<RUN_ID>/training/.work/slurm.json
```

These commands perform real child submissions. Prefer `experiment ...
--dry-run` when the goal is only to inspect commands; controller configs do not
carry a general dry-run switch.

## Related Documentation

- [Architecture](architecture.md)
- [Spectral features](spectral-features.md)
- [Main CLI and flag reference](../README.md)
