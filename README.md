# upeftguard

Modular repository for feature extraction, clustering, and unsupervised PEFT adapter analysis.

This codebase uses a **single interface only**:

```bash
python -m upeftguard.cli ...
```

Legacy top-level scripts (`prepare_data.py`, `cluster_z_space.py`, `gmm_train_inference.py`, etc.) were removed.

## Environment

Use the `upeftg` conda environment:

```bash
source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg
```

Required packages include:
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `safetensors`
- `huggingface_hub`

## Package Layout

- `upeftguard/features`: feature extractors (`svd`, `spectral`) and shared extraction registry
- `upeftguard/clustering`: unsupervised clustering algorithms, metrics, reporting, and pipeline
- `upeftguard/unsupervised`: `gmm_train_inference` pipeline
- `upeftguard/supervised`: supervised spectral-feature pipeline with local/Slurm-array tuning stages
- `upeftguard/utilities/core`: manifest parsing, path defaults, run context, and JSON serialization helpers
- `upeftguard/utilities/artifacts`: dataset-reference reports, spectral metadata, and export helpers
- `upeftguard/utilities/merge`: shard preparation plus feature/schema merge workflows
- `upeftguard/utilities/data`: dataset download helpers
- `upeftguard/utilities/maintenance`: backfill utilities for older artifacts
- `upeftguard/utilities`: compatibility shims for legacy flat utility imports
- `upeftguard/cli.py`: unified CLI entrypoint
- `sbatch/`: Slurm submission scripts

## Manifest Layout

Experiment manifests are grouped under `manifests/` by study:
- `zero_shots/attack_wise`
- `zero_shots/rank_wise`
- `zero_shots/adapter_wise`
- `rank_exploration`
- `adapter_exploration`
- `architecture_exploration`
- `single_datasets`
- `others`

## Run Artifacts

All commands write run artifacts under:

```text
runs/<pipeline>/<run_id>/
```

Each run contains:
- `features/`
- `models/`
- `reports/`
- `plots/`
- `logs/`
- `run_config.json`
- `artifact_index.json`
- `timings.json`

## CLI Commands

Default dataset root: `/models/$USER/unsupervised-peftguard/data`

Override it with `UPEFTGUARD_DATA_ROOT` or `UPEFTGUARD_STORAGE_ROOT` if needed.

### 1. Clustering pipeline

```bash
python -m upeftguard.cli run clustering \
  --manifest-json manifests/others/prepare_manifest_v0.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --extractor svd \
  --svd-n-components 30 \
  --output-root runs \
  --run-id clustering_demo \
  --algorithms kmeans hierarchical dbscan gmm mahalanobis isolation_forest lof \
  --selection-metric silhouette \
  --use-offline-label-metrics
```

### 2. GMM train/inference pipeline (canonical)

```bash
python -m upeftguard.cli run gmm-train-inference \
  --manifest-json manifests/others/gmm_manifest.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --output-root runs \
  --run-id gmm_demo \
  --svd-components-grid 35 40 45 \
  --gmm-components 1 2 3 4 5 \
  --gmm-covariance-types diag full tied spherical
```

Notes:
- Training may include clean and backdoored adapters.
- Selection remains unsupervised (BIC).
- Threshold percentiles are computed from **all train scores**.
- Canonical report: `gmm_train_inference_report.json`.

### 3. Feature extraction

```bash
python -m upeftguard.cli feature extract \
  --manifest-json manifests/others/prepare_manifest_v0.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --extractor svd \
  --output-root runs \
  --run-id feature_svd_demo
```

Supported extractors:
- `svd`
- `spectral`

For multi-node spectral extraction with Slurm arrays:

```bash
MANIFEST_JSON=manifests/others/prepare_manifest_v0.json \
RUN_ID=feature_spectral_demo \
FEATURES="energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank" \
sbatch ./sbatch/feature_extract_array.sh
```

Useful spectral options:
- `--spectral-qv-sum-mode none|append|only`
- `none`: baseline per-block features (default)
- `append`: baseline per-block + additional per-layer `q+v` blocks
- `only`: only per-layer `q+v` blocks
- `--spectral-entrywise-delta-mode auto|dense|stream`
- `auto`: materialize dense block deltas when the estimated working set fits comfortably in RAM
- `dense`: always materialize each block delta before computing entry-wise moments
- `stream`: always compute entry-wise moments from streamed delta chunks

Each feature extraction run is standalone. The array launcher always writes one final feature bundle under:
- `runs/feature_extract/<RUN_ID>/merged/spectral_features.npy`
- `runs/feature_extract/<RUN_ID>/merged/spectral_model_names.json`
- `runs/feature_extract/<RUN_ID>/merged/spectral_labels.npy`
- `runs/feature_extract/<RUN_ID>/merged/spectral_metadata.json`

Feature extraction does not append into an existing run. If you want to combine two completed feature runs, use `util merge-features` as a separate step.

### 4. Unsupervised t-SNE over features

```bash
python -m upeftguard.cli run unsupervised-tsne \
  --feature-file <RUN_ID> \
  --output-root runs \
  --run-id tsne_demo \
  --over rank \
  --view full \
  --perplexity 30 \
  --learning-rate auto \
  --max-iter 1000
```

Useful options:
- `--view per_layer` to run t-SNE separately for each layer slice instead of the full feature bundle.
- `--perplexities`, `--learning-rates`, `--max-iters-grid`, `--metrics`, `--inits`, and `--random-states` to sweep multiple t-SNE settings in one run.
- Bare `--feature-file` names resolve under `runs/feature_extract/<RUN_ID>/merged/` and automatically pick up sibling `spectral_model_names.json`, optional `spectral_labels.npy`, and optional `spectral_metadata.json`.

### 5. Layer scatter plots over features

```bash
python -m upeftguard.cli run unsupervised-layer-scatter \
  --feature-file <RUN_ID> \
  --output-root runs \
  --run-id layer_scatter_demo \
  --point-size 6 \
  --alpha 0.18
```

Notes:
- This command requires labels; it resolves sibling `spectral_labels.npy` automatically when present, or can use `--dataset-reference-report` for label provenance.
- Each emitted feature now produces two images: a scatter figure with side-by-side `clean` and `backdoor` panels, plus a paired boxplot figure with one `clean` box and one `backdoor` box for each layer.
- The run also saves `all_features_boxplots.png`, a single multi-panel summary figure that tiles the boxplots for all emitted features together.

### 6. Supervised pipeline

```bash
python -m upeftguard.cli run supervised \
  --manifest-json manifests/others/gmm_manifest.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --feature-file <RUN_ID> \
  --model logistic_regression \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-entrywise-delta-mode auto \
  --tuning-executor local
```

Useful options:
- `--model all` to tune across all registered classifiers.
- `--feature-file` is required for `stage=prepare` and `stage=all`.
- `--feature-file` can be a feature run name, a feature output directory, or an explicit `spectral_features.npy` path.
- separate `--feature-model-names-file` / `--feature-metadata-file` flags are not supported; sibling companion files are resolved automatically from `--feature-file`
- `--features` is required for `stage=prepare` and `stage=all`.
- sibling `model_names` and `metadata` files are resolved automatically from the same feature bundle.
- `--train_split 80` to create a deterministic 80/20 train/inference split from a single manifest.
- `--split-by-folder` to apply folder/label-aware splitting instead of global label stratification. For single manifests this affects both the outer train/inference split and, when enabled, the train/calibration split; for joint manifests it applies to calibration only.
- `--calibration-split 20 --accepted-fpr 0.01 0.05` to carve a calibration set out of the training partition, then choose one final threshold per requested FPR target by maximizing recall subject to each `false_positive_rate` constraint on calibration.
- `--cv-seeds 42 43 44` to repeat CV with multiple random seeds.
- `--tuning-executor local` runs tuning in the current process; `--tuning-executor slurm_array` prepares the run for distributed Slurm array workers.
- Joint manifests already define train/infer partitions, so `--train_split` should stay at `100` for those.

The supervised pipeline no longer launches feature extraction internally. The supported sequence is:
- feature extraction
- supervised tuning / evaluation

With calibration enabled, the supervised protocol becomes:
- tune model family and hyperparameters by CV on the fit-train subset only
- refit the winning model on the fit-train subset
- score the calibration subset and select the threshold under the requested FPR constraint
- evaluate the frozen model and threshold on the inference/test split

For Slurm-array tuning:

```bash
MANIFEST_JSON=manifests/others/gmm_manifest.json \
FEATURE_FILE=<RUN_ID> \
FEATURES="energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank" \
sbatch ./sbatch/supervised_array.sh
```

For post-hoc distributed winner feature importance on an existing supervised run:

```bash
RUN_DIR=runs/supervised/<RUN_ID> sbatch ./sbatch/supervised_feature_importance_array.sh
```

The same contract applies to direct CLI runs and Slurm-array runs: supervised always consumes a precomputed feature bundle.

```bash
python -m upeftguard.cli run supervised \
  --manifest-json manifests/others/gmm_manifest.json \
  --dataset-root /models/$USER/unsupervised-peftguard/data \
  --feature-file <RUN_ID> \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --tuning-executor slurm_array
```

### 7. Dataset utility

```bash
python -m upeftguard.cli util download-dataset --show-list
python -m upeftguard.cli util download-dataset \
  --dataset llama2_7b_toxic_backdoors_alpaca_rank256_qv \
  --show-list
python -m upeftguard.cli util download-dataset \
  --dataset llama2_7b_imdb_insertsent_rank256_qv \
  --all
python -m upeftguard.cli util download-dataset \
  --dataset llama2_7b_imdb_insertsent_rank256_qv \
  --clean 100 --backdoored 60
python -m upeftguard.cli util download-dataset \
  --dataset llama2_7b_imdb_syntactic_rank256_qv \
  --backdoored 150 152
```

Use `--show-list` without `--dataset` to discover folder names, or with `--dataset` to inspect the clean/backdoor indices available inside a specific folder. Use `--all` to download the full clean+backdoor contents of the selected folder. `--dataset` is required for any download.

Downloads land under `/models/$USER/unsupervised-peftguard/data` by default.

### 8. Feature merge utility

```bash
python -m upeftguard.cli util merge-features \
  --merge <RUN_A> <RUN_B> \
  --output-filename <RUN_OUT>
```

Bare names resolve under `runs/feature_extract/<NAME>/merged/spectral_features.npy` by default. The command resolves sibling `model_names`, optional `labels`, and optional `metadata` files automatically, then writes the output feature artifacts under `runs/feature_extract/<RUN_OUT>/merged/`. Pass `--feature-root` to change that base directory, or pass explicit `.npy` paths if you need to bypass name-based resolution.

When the two inputs have disjoint model rows, the utility zero-fills cross-schema cells so heterogeneous runs can still be combined into one matrix. If the same model row appears in both inputs, the merge remains strict and will still reject conflicting or incomplete overlap.

### 9. Feature subset utility

```bash
python -m upeftguard.cli util export-feature-subset \
  --feature-file <RUN_ID_OR_PATH> \
  --output-filename <RUN_OUT> \
  --dataset-name flan_t5_xl_toxic_backdoors_hard_rank256_qv
```

This command is provenance-backed rather than zero-pattern-backed. It:
- selects rows from the input bundle using dataset-reference metadata such as `--dataset-name`, `--subset-name`, `--model-family`, `--attack-name`, or `--model-name`
- walks `merge_source_feature_files` recursively down to the leaf source feature bundles that originally produced those rows
- keeps the exact columns owned by those leaf sources

By default the utility keeps all provenance-backed columns for the selected rows. To further narrow the output, pass spectral feature groups via `--features`:

```bash
python -m upeftguard.cli util export-feature-subset \
  --feature-file <RUN_ID_OR_PATH> \
  --output-filename <RUN_OUT> \
  --dataset-name flan_t5_xl_toxic_backdoors_hard_rank256_qv \
  --features energy stable_rank
```

`--features energy stable_rank` keeps every provenance-backed `energy` and `stable_rank` column across the selected blocks/layers. `--columns` remains accepted as an alias for the same feature-family selector.

Bare run names resolve under `runs/feature_extract/<NAME>/merged/spectral_features.npy` by default. The exported bundle includes filtered `model_names`, optional `labels`, filtered metadata, and a filtered dataset-reference report.

The older manifest-plus-zero-pruning subset path has been removed. Use the provenance-backed subset utility instead.

## Maintenance Scripts

- `scripts/generate_backdoor_detection_summaries.py` is a post-processing helper for completed supervised runs. It expects run ids to follow the current manifest naming conventions under `manifests/zero_shots/`, `manifests/rank_exploration/`, `manifests/adapter_exploration/`, and `manifests/architecture_exploration/`.
- `python -m upeftguard.utilities.maintenance.backfill_dataset_reference_reports --root <feature_extract_root>` can recreate missing dataset-reference reports for older feature and merge outputs.
- `python -m upeftguard.utilities.maintenance.backfill_spectral_metadata --root <feature_extract_root>` rewrites older spectral metadata files into the current public/internal sidecar layout.
