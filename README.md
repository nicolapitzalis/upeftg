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

- `upeftguard/features`: feature extractors (`svd`, `spectral`) and cache-backed registry
- `upeftguard/clustering`: unsupervised clustering algorithms, metrics, reporting, and pipeline
- `upeftguard/unsupervised`: `gmm_train_inference` pipeline
- `upeftguard/supervised`: supervised spectral-feature pipeline with local/Slurm-array tuning stages
- `upeftguard/utilities`: manifest parsing, run context, hashing, dataset download, report comparison
- `upeftguard/cli.py`: unified CLI entrypoint

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

Feature caching is automatic under:

```text
runs/cache/features/<cache_key>/
```

Cache key:
`sha256(dataset_signature + extractor + params + version + dtype)`

## CLI Commands

### 1. Clustering pipeline

```bash
python -m upeftguard.cli run clustering \
  --manifest-json prepare_manifest.json \
  --dataset-root data \
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
  --manifest-json gmm_manifest.json \
  --dataset-root data \
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
  --manifest-json prepare_manifest.json \
  --dataset-root data \
  --extractor svd \
  --output-root runs \
  --run-id feature_svd_demo
```

Supported extractors:
- `svd`
- `spectral`

For multi-node spectral extraction with Slurm array + merge:

```bash
./run_extract_delta_features_array.sh
```

Merged outputs are written under:
- `runs/feature_extract/<RUN_ID>/merged/spectral_features.npy`
- `runs/feature_extract/<RUN_ID>/merged/spectral_model_names.json`
- `runs/feature_extract/<RUN_ID>/merged/spectral_metadata.json`

### 4. Supervised pipeline

```bash
python -m upeftguard.cli run supervised \
  --manifest-json gmm_manifest.json \
  --dataset-root data \
  --model logistic_regression \
  --features frobenius energy kurtosis l1_norm linf_norm sv_topk \
  --spectral-sv-top-k 8 \
  --tuning-executor local
```

Useful options:
- `--model all` to tune across all registered classifiers.
- `--cv-seeds 42 43 44` to repeat CV with multiple random seeds.

For Slurm-array tuning:

```bash
./run_supervised_tuning_array.sh
```

To skip extraction and use precomputed merged spectral features:

```bash
python -m upeftguard.cli run supervised \
  --manifest-json gmm_manifest.json \
  --dataset-root data \
  --feature-file runs/feature_extract/<RUN_ID>/merged/spectral_features.npy \
  --feature-model-names-file runs/feature_extract/<RUN_ID>/merged/spectral_model_names.json \
  --feature-metadata-file runs/feature_extract/<RUN_ID>/merged/spectral_metadata.json \
  --tuning-executor slurm_array
```

### 5. Dataset utility

```bash
python -m upeftguard.cli util download-dataset --clean 100 --backdoored 60
```

## Tests

Run tests with the unified CLI only:

```bash
python -m unittest discover -s tests -v
```
