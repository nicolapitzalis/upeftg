#!/bin/bash
#SBATCH --job-name=upeftguard_clustering
#SBATCH --output=logs/clustering_%j.log
#SBATCH --error=logs/clustering_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

MANIFEST_JSON=${MANIFEST_JSON:-prepare_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}
RUN_ID=${RUN_ID:-clustering_${SLURM_JOB_ID:-manual}}

python -m upeftguard.cli run clustering \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --extractor svd \
  --svd-n-components 30 \
  --output-root runs \
  --run-id "${RUN_ID}" \
  --algorithms kmeans hierarchical dbscan gmm mahalanobis isolation_forest lof \
  --k-list 2 3 4 5 \
  --eps-list 0.5 1.0 1.5 2.0 \
  --min-samples 2 \
  --selection-metric silhouette \
  --use-offline-label-metrics
