#!/bin/bash
#SBATCH --job-name=upeftguard_compare_repr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/compare_representation_%j.log
#SBATCH --error=logs/compare_representation_%j.err

set -euo pipefail

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

python -m upeftguard.cli report compare-representations \
  --reports \
  raw=runs/clustering/RAW_RUN_ID/reports/clustering_report.json \
  delta_sv=runs/clustering/DELTA_SV_RUN_ID/reports/clustering_report.json \
  delta_fro=runs/clustering/DELTA_FRO_RUN_ID/reports/clustering_report.json \
  --output-file runs/report/compare_representations/representation_comparison.json \
  --target-auroc 0.80 \
  --target-stability 0.80
