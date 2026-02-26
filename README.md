# Unsupervised PEFTGuard Baselines

This repository contains an analysis pipeline for **unsupervised detection of suspicious LoRA adapters**.

The workflow is organized into phases:

1. **Prepare embeddings** from adapter parameters with truncated SVD (`prepare_data.py`)
2. **Run unsupervised baselines** in embedding space (`cluster_z_space.py`)
3. **Train clean-only SVD+GMM and score mixed inference set** (`gmm_clean_inference.py`)
4. **Extract Delta-based features** (`extract_delta_features.py`)
5. **Compare representations and decide go/no-go** for a flow model (`compare_representation_reports.py`)

The current target regime is:
- Mixed unlabeled pool
- Majority-clean prior
- Labels allowed only for offline evaluation

## Repository Structure

- `prepare_data.py`: data loading, consistency checks, truncated SVD, representativeness audit, artifact export
- `cluster_z_space.py`: unsupervised clustering/anomaly baselines + standardized report
- `gmm_clean_inference.py`: clean-only SVD+GMM tuning by BIC, then mixed-set inference scoring
- `extract_delta_features.py`: Delta feature extraction (`B @ A`) with low-rank spectral/Frobenius features
- `compare_representation_reports.py`: aggregate multiple report files and produce flow go/no-go summary
- `download_dataset.py`: helper to fetch PADBench adapters
- `run_prepare_data.sh`, `run_clustering.sh`, `run_gmm_clean_inference.sh`: SLURM-oriented runners
- `tests/test_cli_and_pipeline.py`: unit + smoke tests

## Requirements

Python packages used by the pipeline:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `safetensors`
- `huggingface_hub` (only for dataset download helper)

Example (conda + pip):

```bash
conda create -n upeftg python=3.12 -y
conda activate upeftg
pip install numpy scipy scikit-learn matplotlib safetensors huggingface_hub
```

## Data Layout

Expected adapter layout under `--data-dir`:

```text
data/llama3_8b_toxic_backdoors_hard_rank256_qv/
  llama3_8b_toxic_backdoors_hard_rank256_qv_label0_0/
    adapter_model.safetensors
  ...
  llama3_8b_toxic_backdoors_hard_rank256_qv_label1_19/
    adapter_model.safetensors
```

## Phase 0: (Optional) Download Dataset

```bash
python download_dataset.py --clean 100 --backdoored 30
```

Selection semantics:

- `--clean <count>`: download `<count>` clean adapters.
- `--clean <start> <end>`: download clean adapters with indices in `[start, end]` (inclusive).
- `--backdoored <count>`: download `<count>` backdoored adapters using round-robin allocation across attacks.
- `--backdoored <start> <end>`: download backdoored adapters with indices in `[start, end]` (inclusive) for each backdoored source (not round-robin).

Examples:

```bash
# Download clean indices 0..100
python download_dataset.py --clean 0 100

# Download 60 backdoored adapters (round-robin across attacks)
python download_dataset.py --backdoored 60

# Download backdoored indices 0..50 from each backdoored source
python download_dataset.py --backdoored 0 50
```

## Phase 1: Prepare SVD Embeddings + Representativeness Audit

```bash
python prepare_data.py \
  --data-dir data/llama3_8b_toxic_backdoors_hard_rank256_qv \
  --output-dir processed_data_seq \
  --n-per-label 20 \
  --sample-mode first \
  --sample-seed 42 \
  --trunc-svds-components 20 25 30
```

Useful options:

- `--save-x-raw`: save `X_raw.npy` and `X_mean.npy`
- `--dtype {float32,float64}`
- `--disable-offline-label-diagnostics`
- `--acceptance-spearman-threshold`, `--acceptance-variance-threshold`

Main outputs in `processed_data_seq/`:

- `Z_<k>.npy`, `Vt_<k>.npy`
- `svd_info_<k>.json`
- `labels.npy`, `model_names.json`, `metadata.json`
- `representativeness_summary.json`
- `run_config.json`

## Phase 2: Run Unsupervised Baselines on SVD Features

```bash
python cluster_z_space.py \
  --data-dir processed_data_seq \
  --output-dir clustering_results \
  --n-components 20 \
  --algorithms kmeans hierarchical dbscan gmm mahalanobis isolation_forest lof \
  --k-list 2 3 4 5 \
  --eps-list 0.5 1.0 1.5 2.0 \
  --min-samples 2 \
  --selection-metric silhouette \
  --use-offline-label-metrics
```

Selection metrics (unsupervised only):

- `silhouette`
- `bic`
- `stability`

Main outputs in `clustering_results/`:

- `clustering_report.json`
- `run_config.json`
- `best_score_model_by_auroc_2d.png` (generated when best-AUROC model has partition labels)
- `best_partition_model_by_ari_2d.png`
- `best_partition_model_by_silhouette_2d.png` (winner of unsupervised silhouette selection, when partition-capable)
- `metrics_comparison_partition.png` (partition-capable models, including score/partition models like GMM)
- `metrics_comparison_scores.png` (score models)
- `model_suspicion_scores.csv` (for score-based methods)

## Phase 2B: Clean-Only SVD + GMM Inference

Create one JSON manifest:

- `gmm_manifest.json`: includes both `train` and `infer` source sets

Manifest format:

- top-level keys: `train`, `infer`
- each key maps to a list of source objects
- each source object has `{ "path": "...", "indices": [a, b] }`
- `[a,b]` means indices `a..b`
- relative paths are resolved from `--dataset-root`
- `path` should end right before the index (for example `..._label0_`)

Example:

```json
{
  "train": [
    {
      "path": "llama3_8b_toxic_backdoors_hard_rank256_qv/llama3_8b_toxic_backdoors_hard_rank256_qv_label0_",
      "indices": [0, 9]
    }
  ],
  "infer": [
    {
      "path": "llama3_8b_toxic_backdoors_hard_rank256_qv/llama3_8b_toxic_backdoors_hard_rank256_qv_label0_",
      "indices": [10, 19]
    },
    {
      "path": "llama3_8b_toxic_backdoors_hard_rank256_qv/llama3_8b_toxic_backdoors_hard_rank256_qv_label1_",
      "indices": [0, 19]
    }
  ]
}
```

Run:

```bash
python gmm_clean_inference.py \
  --dataset-root data \
  --manifest-json gmm_manifest.json \
  --output-dir gmm_clean_inference_results \
  --svd-components-grid 20 25 30 \
  --gmm-components 1 2 3 4 5 \
  --gmm-covariance-types diag full tied spherical \
  --stability-seeds 42 43 44 \
  --score-percentiles 90 95 97 99
```

Main outputs in `gmm_clean_inference_results/`:

- `gmm_clean_inference_report.json`
- `inference_scores.csv`
- `train_clean_scores.csv`
- `run_config.json`

## Phase 3/4: Delta Feature Baseline Track

Extract Delta features:

```bash
python extract_delta_features.py \
  --data-dir data/llama3_8b_toxic_backdoors_hard_rank256_qv \
  --output-dir delta_features \
  --n-per-label 20 \
  --sample-mode first \
  --top-k-singular-values 8
```

Evaluate Delta singular-value features:

```bash
python cluster_z_space.py \
  --data-dir delta_features \
  --feature-file delta_features/delta_singular_values.npy \
  --output-dir clustering_results_delta_sv \
  --algorithms gmm mahalanobis isolation_forest lof \
  --selection-metric stability \
  --use-offline-label-metrics
```

Evaluate Delta Frobenius features:

```bash
python cluster_z_space.py \
  --data-dir delta_features \
  --feature-file delta_features/delta_frobenius.npy \
  --output-dir clustering_results_delta_fro \
  --algorithms gmm mahalanobis isolation_forest lof \
  --selection-metric stability \
  --use-offline-label-metrics
```

## Phase 5: Compare Representations and Decide Flow Go/No-Go

```bash
python compare_representation_reports.py \
  --reports \
    raw=clustering_results/clustering_report.json \
    delta_sv=clustering_results_delta_sv/clustering_report.json \
    delta_fro=clustering_results_delta_fro/clustering_report.json \
  --output-file representation_comparison.json \
  --target-auroc 0.80 \
  --target-stability 0.80
```

The output includes:

- per-representation summary
- best AUROC and best stability models
- `go_to_flow` decision flag

## Tests

Run test suite:

```bash
python -m unittest discover -s tests -v
```

Covers:

- CLI/path failure handling
- zero-metric formatting (`0.0` is not shown as N/A)
- bounds/clipping behavior
- end-to-end smoke test for SVD path
- end-to-end smoke test for Delta feature path

## SLURM Runners

- `run_prepare_data.sh`: launches Phase 1
- `run_clustering.sh`: launches Phase 2

Adjust SBATCH resources and CLI flags as needed for your cluster.

## Notes

- Labels are intended for **offline benchmarking only** (`--use-offline-label-metrics`).
- Unsupervised winner selection is always based on unsupervised criteria (`--selection-metric`).
- For reproducibility, each phase writes a `run_config.json` with resolved arguments and metadata.
