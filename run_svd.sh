#!/bin/bash
#SBATCH --job-name=upeftguard_feature_svd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/feature_svd_%j.log
#SBATCH --error=logs/feature_svd_%j.err

set -euo pipefail

mkdir -p logs

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

MANIFEST_JSON=${MANIFEST_JSON:-prepare_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}
RUN_ID=${RUN_ID:-feature_svd_${SLURM_JOB_ID:-manual}}

python -m upeftguard.cli feature extract \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --extractor svd \
  --output-root runs \
  --run-id "${RUN_ID}" \
  --dtype float64 \
  --svd-components-grid 30 35 40 45 \
  --stream-block-size 131072
