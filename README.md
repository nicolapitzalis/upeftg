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

- `upeftguard/features`: feature extractors (`svd`, `delta`, `norms`) and cache-backed registry
- `upeftguard/clustering`: unsupervised clustering algorithms, metrics, reporting, and pipeline
- `upeftguard/unsupervised`: `gmm_train_inference` pipeline
- `upeftguard/supervised`: supervised scaffold (interfaces/registry/pipeline placeholder)
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
- `delta_singular_values`
- `delta_frobenius`
- `norms`

### 4. Dataset utility

```bash
python -m upeftguard.cli util download-dataset --clean 100 --backdoored 60
```

### 5. Representation report comparison

```bash
python -m upeftguard.cli report compare-representations \
  --reports \
    raw=runs/clustering/raw_run/reports/clustering_report.json \
    delta_sv=runs/clustering/delta_sv_run/reports/clustering_report.json \
    delta_fro=runs/clustering/delta_fro_run/reports/clustering_report.json \
  --output-file runs/report/compare_representations/representation_comparison.json \
  --target-auroc 0.80 \
  --target-stability 0.80
```

## Tests

Run tests with the unified CLI only:

```bash
python -m unittest discover -s tests -v
```
