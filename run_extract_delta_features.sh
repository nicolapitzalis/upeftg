#!/bin/bash
#SBATCH --job-name=extract_delta_features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/extract_delta_features_%j.log
#SBATCH --error=logs/extract_delta_features_%j.err

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Starting Delta feature extraction"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

# Activate conda environment
source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

python extract_delta_features.py \
  --data-dir data/llama3_8b_toxic_backdoors_hard_rank256_qv \
  --output-dir delta_features \
  --n-per-label 20 \
  --sample-mode first \
  --sample-seed 42 \
  --top-k-singular-values 8 \
  --dtype float64

echo ""
echo "=========================================="
echo "Delta feature extraction completed"
echo "Finished at: $(date)"
echo "=========================================="
echo "Saved artifacts:"
ls -lh delta_features
