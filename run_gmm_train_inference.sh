#!/bin/bash
#SBATCH --job-name=upeftguard_gmm_train_infer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/gmm_train_inference_%j.log
#SBATCH --error=logs/gmm_train_inference_%j.err

set -euo pipefail

mkdir -p logs

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

MANIFEST_JSON=${MANIFEST_JSON:-gmm_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}
RUN_ID=${RUN_ID:-gmm_train_inference_${SLURM_JOB_ID:-manual}}

python -m upeftguard.cli run gmm-train-inference \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root runs \
  --run-id "${RUN_ID}" \
  --svd-components-grid 35 40 45 \
  --gmm-components 1 2 3 4 5 \
  --gmm-covariance-types diag full tied spherical \
  --stability-seeds 42 43 44 \
  --score-percentiles 65 70 75 80 85 90 95 99 \
  --dtype float64 \
  --stream-block-size 131072
