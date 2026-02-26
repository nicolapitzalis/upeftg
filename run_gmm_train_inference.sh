#!/bin/bash
#SBATCH --job-name=gmm_clean_infer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/gmm_clean_inference_%j.log
#SBATCH --error=logs/gmm_clean_inference_%j.err

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Starting clean-only SVD + GMM inference"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

# Activate conda environment
source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

# Update this manifest before running.
MANIFEST_JSON=${MANIFEST_JSON:-gmm_manifest.json}

python gmm_train_inference.py \
  --dataset-root data \
  --manifest-json "${MANIFEST_JSON}" \
  --output-dir gmm_train_inference_results150+50-40 \
  --svd-components-grid 35 40 45 \
  --gmm-components 1 2 3 4 5 \
  --gmm-covariance-types diag full tied spherical \
  --stability-seeds 42 43 44 \
  --score-percentiles 65 70 75 80 85 90 95 99 \
  --dtype float64 \
  --stream-block-size 131072

echo ""
echo "=========================================="
echo "Clean-only SVD + GMM inference completed"
echo "Finished at: $(date)"
echo "=========================================="
echo "Saved artifacts:"
ls -lh gmm_clean_inference_results
