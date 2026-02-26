#!/bin/bash
#SBATCH --job-name=prepare_data_svd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_data_%j.log
#SBATCH --error=logs/prepare_data_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Starting data preparation + SVD + representativeness audit"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

# Activate conda environment
source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

# Update these before running, or pass via environment.
MANIFEST_JSON=${MANIFEST_JSON:-prepare_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}

# Run preprocessing
python prepare_data.py \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir processed_data_120+120 \
  --dtype float64 \
  --svd-backend dual \
  --trunc-svds-components 30 35 40 45 \
  --stream-block-size 131072

echo ""
echo "=========================================="
echo "Data preparation completed!"
echo "Job finished at: $(date)"
echo "=========================================="
