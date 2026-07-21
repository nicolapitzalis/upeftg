# Experiment CLI Runbook

This is the command-oriented companion to
[section 7 of the pipeline overview](pipeline-overview.md#7-experiment-variants-use-the-same-pipeline).
It shows how to launch the five public workflows and how to express the main
experiment variants. For pipeline concepts and output meanings, use the other
documents; this file is deliberately about invocations.

Run commands from the repository root after activating the environment:

```bash
conda activate upeftg
```

The public entry point is always:

```text
python -m upeftguard.cli experiment <extract|aggregate|train|infer|full>
```

## Command conventions

The generic commands below spell out every default that the CLI can express as
an argument. Replace angle-bracket placeholders before running them.

Some defaults cannot be written on the command line: `argparse` represents
them by the *absence* of a flag. They are listed after each command as
“intentionally omitted,” so that unset, `false`, and automatic values remain
visible without changing the command's behavior.

The default dataset root is resolved in this order:

1. `$UPEFTGUARD_DATA_ROOT`, when set;
2. `$UPEFTGUARD_STORAGE_ROOT/data`, when set; or
3. `/models/$USER/unsupervised-peftguard/data`.

The examples use the third form explicitly. Substitute the appropriate path
when either environment override is set.

Every dataset manifest in `manifests/` follows the public workflow schema:

```json
{
  "path": []
}
```

A training manifest may additionally define `cv_always_train` using the same
entry schema. Those rows participate in extraction and final refitting and are
pinned into every CV training fold, but they are never validation candidates.
Use this only with explicit train/inference manifests, not `--train-split`.

Experiments with fixed outer partitions use adjacent `_train.json` and
`_infer.json` files. Pass the training file to `experiment train`, the
inference file to `experiment infer`, or pass both to `experiment full` with
`--train-manifest-json` and `--infer-manifest-json`. The IMDb dataset holdout
uses the shorter paired names
`manifests/leave_one_dataset_out/non_imdb_train.json` and
`manifests/leave_one_dataset_out/imdb_infer.json`.
The few-shot variant uses
`manifests/leave_one_dataset_out/non_imdb_plus_10_imdb_train.json` and
`manifests/leave_one_dataset_out/imdb_minus_10_infer.json`; its ten backdoored
IMDb rows are in `cv_always_train`, so dataset LOO still creates only the four
non-IMDb validation folds.

The default spectral feature list used below is:

```text
energy kurtosis l1_norm l2_norm linf_norm mean_abs
concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank
```

Both `train` and `full` require `--model`; there is no implicit model default.
Pass the selected model's candidate-grid JSON through `--hyperparams`. The flag
is required for CNN, DANN, and Transformer models and optional for classical
models.

`--backend slurm` is the default. Change it to `local` to execute in the current
process. `--partition` and worker allocation flags are accepted by local runs
but only affect Slurm scheduling.

## Generic workflows

### Extract

```bash
python -m upeftguard.cli experiment extract \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --manifest-json <MANIFEST_JSON> \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --output-root runs \
  --run-id <EXTRACTION_RUN_ID> \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode none \
  --spectral-entrywise-delta-mode auto \
  --spectral-attention-granularity module \
  --stream-block-size 131072 \
  --dtype float32 \
  --nodes auto \
  --workers-per-node auto
```

Intentionally omitted defaults:

| Flag | Default | Why absent |
| --- | --- | --- |
| `--dry-run` | off | Adding it would plan instead of launch. |
| `--parallelization-settings` | unset | Every rank uses the discovered default allocation. |

The extracted feature matrix is normally written to
`runs/<EXTRACTION_RUN_ID>/extraction/features/spectral_features.npy`.

### Aggregate

`--spectral-attention-granularity` is intentionally left unset here so that
aggregation infers and validates it from extraction metadata. Unlike the other
spectral workflows, this command's q/v default is `append`.

```bash
python -m upeftguard.cli experiment aggregate \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --feature-file runs/<EXTRACTION_RUN_ID>/extraction/features/spectral_features.npy \
  --output-root runs \
  --run-id <AGGREGATION_RUN_ID> \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-qv-sum-mode append
```

Intentionally omitted defaults:

| Flag | Default | Why absent |
| --- | --- | --- |
| `--dry-run` | off | Adding it would plan instead of launch. |
| `--output-filename` | unset | The workflow chooses `aggregation/features/spectral_features.npy`. |
| `--feature-root` | unset | The workflow uses the aggregation stage's feature directory. |
| `--spectral-attention-granularity` | unset | It is inferred from the source artifact. |

The aggregated feature matrix is normally written to
`runs/<AGGREGATION_RUN_ID>/aggregation/features/spectral_features.npy`.

### Train

This template treats the manifest as training-only and selects `cnn_1d`
explicitly. Replace `--model cnn_1d` with any other registered model; the
model-specific hyperparameter flags below apply only to their corresponding
models.

```bash
python -m upeftguard.cli experiment train \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --manifest-json <TRAIN_MANIFEST_JSON> \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --output-root runs \
  --run-id <TRAINING_RUN_ID> \
  --feature-file runs/<AGGREGATION_RUN_ID>/aggregation/features/spectral_features.npy \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode append \
  --spectral-entrywise-delta-mode auto \
  --spectral-attention-granularity module \
  --stream-block-size 131072 \
  --dtype float32 \
  --model cnn_1d \
  --hyperparams manifests/cnn_hyperparams/cnn_1d_default.json \
  --cv-folds 5 \
  --cv-strategy stratified \
  --input-normalization none \
  --random-state 42 \
  --n-jobs -1 \
  --task-mode binary \
  --selection-metric task_default \
  --nodes auto \
  --workers-per-node auto
```

Intentionally omitted defaults:

| Flag | Default | Why absent |
| --- | --- | --- |
| `--dry-run` | off | Adding it would plan instead of launch. |
| `--cv-derived-refit-epochs` | off | Adding it enables Transformer-specific refit behavior. |
| `--no-refit` | off | Adding it promotes the best validation-fold estimator from the CV-winning candidate. |
| `--train-split` | unset | The supplied manifest is entirely training data. |
| `--calibration-split`, `--accepted-fpr` | unset | Calibration is disabled; these must be supplied together. |
| `--split-by-folder` / `--no-split-by-folder` | automatic | With no calibration it resolves to off; with calibration it resolves to on. |
| `--class-weight-loss`, `--rank-label-weight-loss` | off | Adding either enables that loss weighting. |
| `--cv-seeds` | unset | The workflow resolves backend-specific seed execution; set it explicitly for strict reproducibility. |
| `--checkpoint-interval-hours`, `--resume-checkpoint` | unset | Periodic Transformer checkpointing and explicit resume are disabled. |
| `--multiclass-attack-names` | unset | Binary mode does not define attack classes. |

Registered `--model` values are `adaboost`, `cnn_1d`, `cnn_1d_dann`,
`kernel_svm`, `linear_svm`, `logistic_regression`, `random_forest`,
`ridge_classifier`, and `transformer`.

#### Model hyperparameters

Every model accepts the same `--hyperparams <JSON>` flag. The selected
`--model` determines which keys are expected and how values are normalized.
Every JSON value must be a non-empty list; the Cartesian product of those lists
is the candidate set.

CNN, DANN, and Transformer models require `--hyperparams`; they have no
implicit grid. Classical models use their registered grid when the flag is
omitted, or replace it with the supplied grid when the flag is present.

Classical models (`adaboost`, `kernel_svm`, `linear_svm`,
`logistic_regression`, `random_forest`, and `ridge_classifier`) expand the
built-in candidate grids registered in `upeftguard/supervised/models/registry.py`.
The resolved grid and its source are recorded in the tuning manifest for
reproducibility.

For example, this produces one logistic-regression candidate:

```json
{
  "C": [1.0],
  "class_weight": [null]
}
```

When the Cartesian product contains one candidate and `--cv-strategy` is the
ordinary `stratified` mode, the run uses `singleton_no_cv`: candidate CV is
skipped and the model is fitted once on the full training partition during
finalization. Attack-family or dataset leave-one-out, `--no-refit`, and
`--cv-derived-refit-epochs` still force cross-validation for a single
candidate.

### Infer

Use `.pt` checkpoints for torch sequence models (CNN, DANN, and Transformer)
and `.joblib` checkpoints for tabular scikit-learn models. The feature artifact
must match the checkpoint contract: sequence models use aggregated features,
while classical models can use extracted tabular features directly. The
example below uses a sequence-model checkpoint.

```bash
python -m upeftguard.cli experiment infer \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --checkpoint runs/<TRAINING_RUN_ID>/training/models/best_model.pt \
  --manifest-json <INFERENCE_MANIFEST_JSON> \
  --feature-file runs/<AGGREGATION_RUN_ID>/aggregation/features/spectral_features.npy \
  --output-root runs \
  --run-id <INFERENCE_RUN_ID>
```

The only intentionally omitted default is `--dry-run` (off). The inference
manifest is always inference-only; this command has no split flags.

### Full

This template uses one complete, labelled manifest and makes an 80/20 outer
train/inference split. The value `80` is required by this manifest form; it is
an experimental choice, not a CLI default.

```bash
python -m upeftguard.cli experiment full \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --extract-nodes auto \
  --extract-workers-per-node auto \
  --train-nodes auto \
  --train-workers-per-node auto \
  --manifest-json <COMPLETE_MANIFEST_JSON> \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --output-root runs \
  --run-id <FULL_RUN_ID> \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode none \
  --spectral-entrywise-delta-mode auto \
  --spectral-attention-granularity module \
  --stream-block-size 131072 \
  --dtype float32 \
  --model cnn_1d \
  --hyperparams manifests/cnn_hyperparams/cnn_1d_default.json \
  --cv-folds 5 \
  --cv-strategy stratified \
  --input-normalization none \
  --random-state 42 \
  --train-split 80 \
  --n-jobs -1 \
  --task-mode binary \
  --selection-metric task_default
```

Intentionally omitted defaults:

| Flag | Default | Why absent |
| --- | --- | --- |
| `--dry-run` | off | Adding it would plan instead of launch. |
| `--extract-cpus-per-worker`, `--aggregate-cpus-per-worker`, `--train-cpus-per-worker` | unset | Each inherits `--cpus-per-worker 4`. |
| `--extract-parallelization-settings` | unset | Unlisted ranks use discovered extraction resources. |
| `--train-manifest-json`, `--infer-manifest-json` | unset | This template uses the mutually exclusive complete-manifest form. |
| `--cv-derived-refit-epochs` | off | Adding it enables Transformer-specific refit behavior. |
| `--no-refit` | off | Adding it promotes the best validation-fold estimator from the CV-winning candidate. |
| `--calibration-split`, `--accepted-fpr` | unset | Calibration is disabled. |
| `--split-by-folder` / `--no-split-by-folder` | automatic | With no calibration it resolves to off. |
| `--class-weight-loss`, `--rank-label-weight-loss` | off | Loss weighting is disabled. |
| `--cv-seeds` | unset | Set explicit seeds when repetitions must be fixed across backends. |
| `--checkpoint-interval-hours`, `--resume-checkpoint` | unset | Periodic Transformer checkpointing and explicit resume are disabled. |
| `--multiclass-attack-names` | unset | Binary mode does not define attack classes. |

To use explicit outer partitions instead, replace
`--manifest-json ... --train-split 80` with both:

```bash
--train-manifest-json <TRAIN_MANIFEST_JSON> \
--infer-manifest-json <INFERENCE_MANIFEST_JSON>
```

## Experiment variants

The commands in this section emphasize the flags that define the experiment.
All unspecified execution, spectral, and training values retain the defaults
shown exhaustively above.

### Outer leave-one-out / zero-shot evaluation

This example holds RIPPLE out of an insertsent-trained attack-wise experiment.
The repository provides the training and inference populations as separate
single-`path` manifests, so they can be passed directly to the full pipeline:

```bash
python -m upeftguard.cli experiment full \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --extract-nodes auto \
  --extract-workers-per-node auto \
  --train-nodes auto \
  --train-workers-per-node auto \
  --train-manifest-json manifests/zero_shots/attack_wise/llama2_7b_ag_news_imdb_zero_shot_insertsent_to_ripple_train.json \
  --infer-manifest-json manifests/zero_shots/attack_wise/llama2_7b_ag_news_imdb_zero_shot_insertsent_to_ripple_infer.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --output-root runs \
  --run-id cnn_insertsent_to_ripple \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode none \
  --spectral-entrywise-delta-mode auto \
  --spectral-attention-granularity module \
  --stream-block-size 131072 \
  --dtype float32 \
  --model cnn_1d \
  --hyperparams manifests/cnn_hyperparams/cnn_1d_default.json \
  --cv-folds 5 \
  --cv-strategy stratified \
  --input-normalization none \
  --random-state 42 \
  --n-jobs -1 \
  --task-mode binary \
  --selection-metric task_default
```

This is an outer holdout: RIPPLE is not used in training or CV. For an inner
binary attack-family leave-one-out CV experiment on a training manifest that
contains multiple positive attack families, use:

```bash
python -m upeftguard.cli experiment train \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --manifest-json manifests/others/list2.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --feature-file <LIST2_AGGREGATED_FEATURE_FILE> \
  --output-root runs \
  --run-id transformer_attack_family_loo_cv \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode append \
  --spectral-entrywise-delta-mode auto \
  --spectral-attention-granularity module \
  --stream-block-size 131072 \
  --dtype float32 \
  --model transformer \
  --hyperparams manifests/transformer_hyperparams/transformer_default.json \
  --cv-folds 5 \
  --cv-strategy attack_family_leave_one_out \
  --cv-seeds 42 \
  --cv-derived-refit-epochs \
  --input-normalization none \
  --random-state 42 \
  --n-jobs -1 \
  --task-mode binary \
  --selection-metric task_default \
  --nodes auto \
  --workers-per-node auto
```

`--cv-folds 5` is included to make the configured default explicit, but this
CV strategy derives one fold per attack family and ignores that value.
`attack_family_leave_one_out` cannot be combined with multiclass task mode.

### Attack-wise multiclass classification

The complete manifest below contains AG News and IMDB clean samples plus the
`insertsent`, `RIPPLE`, `stybkd`, and `syntactic` attack families. The class
names must match the attack names inferred from the manifest.

```bash
python -m upeftguard.cli experiment full \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --extract-nodes auto \
  --extract-workers-per-node auto \
  --train-nodes auto \
  --train-workers-per-node auto \
  --manifest-json manifests/single_datasets/llama2_7b_single_dataset_cnn_union.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --output-root runs \
  --run-id cnn_attack_family_multiclass \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode none \
  --spectral-entrywise-delta-mode auto \
  --spectral-attention-granularity module \
  --stream-block-size 131072 \
  --dtype float32 \
  --model cnn_1d \
  --hyperparams manifests/cnn_hyperparams/cnn_1d_ag_news_imdb_attack_family_multiclass_tuning.json \
  --cv-folds 5 \
  --cv-strategy stratified \
  --input-normalization none \
  --random-state 42 \
  --train-split 80 \
  --n-jobs -1 \
  --task-mode attack_family_multiclass \
  --multiclass-attack-names insertsent RIPPLE stybkd syntactic \
  --selection-metric task_default
```

The resulting model has the classes `clean`, `insertsent`, `RIPPLE`, `stybkd`,
and `syntactic`. Its clean-vs-backdoor projection is still available for the
binary AUROC used by `task_default` model selection.

### Calibration with dataset leave-one-out CV

This example combines an 80/20 outer inference split, dataset-wise inner CV,
and a 20% calibration partition taken from the training population. It selects
operating thresholds for accepted FPRs of 1% and 5%.

```bash
python -m upeftguard.cli experiment full \
  --backend slurm \
  --partition extra \
  --cpus-per-worker 4 \
  --extract-nodes auto \
  --extract-workers-per-node auto \
  --train-nodes auto \
  --train-workers-per-node auto \
  --manifest-json manifests/single_datasets/llama2_7b_single_dataset_cnn_union.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --output-root runs \
  --run-id cnn_dataset_loo_calibrated \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode none \
  --spectral-entrywise-delta-mode auto \
  --spectral-attention-granularity module \
  --stream-block-size 131072 \
  --dtype float32 \
  --model cnn_1d \
  --hyperparams manifests/cnn_hyperparams/cnn_1d_default.json \
  --cv-folds 5 \
  --cv-strategy dataset_leave_one_out \
  --input-normalization dataset_feature_standard \
  --random-state 42 \
  --train-split 80 \
  --calibration-split 20 \
  --accepted-fpr 0.01 0.05 \
  --split-by-folder \
  --cv-seeds 42 \
  --n-jobs -1 \
  --task-mode binary \
  --selection-metric task_default
```

`dataset_leave_one_out` derives one fold per inferred dataset and ignores
`--cv-folds 5`. `--split-by-folder` is written explicitly even though it is the
effective automatic behavior when calibration is enabled. Use
`--no-split-by-folder` when a plain label-stratified calibration split is
required instead.

## Flag availability and defaults

This matrix is a compact completeness check against the public parsers.
Required values are marked “required”; `unset` and `off` mean the option must
be omitted to retain the default.

| Flag | Commands | Default / requirement |
| --- | --- | --- |
| `--backend` | all | `slurm` (`slurm`, `local`) |
| `--partition` | all | `extra` |
| `--cpus-per-worker` | all | `4` |
| `--dry-run` | all | off |
| `--manifest-json` | extract, train, infer; full complete-pool form | required except conditionally in full |
| `--dataset-root` | extract, train, full | environment-derived path described above |
| `--output-root` | all | `runs` |
| `--run-id` | all | unset; generated UTC identifier |
| `--feature-file` | aggregate, train, infer | required |
| `--checkpoint` | infer | required |
| `--output-filename` | aggregate | unset; stage default path |
| `--feature-root` | aggregate | unset; stage feature directory |
| `--train-manifest-json`, `--infer-manifest-json` | full | unset; both required in explicit-partition form |
| `--features` | extract, aggregate, train, full | 11-feature list shown above |
| `--spectral-sv-top-k` | extract, train, full | `8` |
| `--spectral-moment-source` | extract, train, full | `sv` (`entrywise`, `sv`, `both`) |
| `--spectral-qv-sum-mode` | extract, full | `none` (`none`, `append`, `only`) |
| `--spectral-qv-sum-mode` | aggregate, train | `append` (`none`, `append`, `only`) |
| `--spectral-entrywise-delta-mode` | extract, train, full | `auto` (`auto`, `dense`, `stream`) |
| `--spectral-attention-granularity` | extract, train, full | `module` (`module`, `head`) |
| `--spectral-attention-granularity` | aggregate | unset; inferred (`module`, `head`) |
| `--stream-block-size` | extract, train, full | `131072` |
| `--dtype` | extract, train, full | `float32` (`float32`, `float64`) |
| `--nodes`, `--workers-per-node` | extract, train | `auto` |
| `--parallelization-settings` | extract | unset; values are `RANK:NODES:CPUS_PER_WORKER:WORKERS_PER_NODE` |
| `--extract-nodes`, `--extract-workers-per-node` | full | `auto` |
| `--train-nodes`, `--train-workers-per-node` | full | `auto` |
| `--extract-cpus-per-worker`, `--aggregate-cpus-per-worker`, `--train-cpus-per-worker` | full | unset; inherit `--cpus-per-worker` |
| `--extract-parallelization-settings` | full | unset; same rank-setting format as extract |
| `--model` | train, full | required; choices listed above |
| `--cv-folds` | train, full | `5` |
| `--cv-strategy` | train, full | `stratified` (`stratified`, `attack_family_leave_one_out`, `dataset_leave_one_out`) |
| `--input-normalization` | train, full | `none` (`none`, `dataset_feature_standard`) |
| `--cv-derived-refit-epochs` | train, full | off |
| `--no-refit` | train, full | off; forces CV and cannot be combined with refit-epoch or interval-checkpoint options |
| `--random-state` | train, full | `42` |
| `--train-split` / `--train_split` | train, full | unset; full requires it with `--manifest-json` |
| `--calibration-split` / `--calibration_split` | train, full | unset |
| `--accepted-fpr` / `--accepted_fpr` | train, full | unset; one or more floats in `[0, 1]` |
| `--split-by-folder` / `--no-split-by-folder` | train, full | unset/automatic |
| `--class-weight-loss` | train, full | off |
| `--rank-label-weight-loss` | train, full | off |
| `--cv-seeds` | train, full | unset |
| `--n-jobs` | train, full | `-1` |
| `--hyperparams` | train, full | required for CNN/DANN/Transformer; optional classical-grid override |
| `--checkpoint-interval-hours` | train, full | unset |
| `--resume-checkpoint` | train, full | unset |
| `--task-mode` | train, full | `binary` (`binary`, `attack_family_multiclass`) |
| `--multiclass-attack-names` | train, full | unset |
| `--selection-metric` | train, full | `task_default` (`task_default`, `roc_auc`, `binary_auroc`) |

Use the live parser to confirm the installed checkout at any time:

```bash
python -m upeftguard.cli experiment <COMMAND> --help
```
