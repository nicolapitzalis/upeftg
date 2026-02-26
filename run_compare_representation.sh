#!/bin/bash
#SBATCH --job-name=compare_repr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/compare_representation_%j.log
#SBATCH --error=logs/compare_representation_%j.err

set -euo pipefail

mkdir -p logs

RAW_REPORT="${RAW_REPORT:-clustering_results/clustering_report.json}"
DELTA_SV_REPORT="${DELTA_SV_REPORT:-clustering_results_delta_sv/clustering_report.json}"
DELTA_FRO_REPORT="${DELTA_FRO_REPORT:-clustering_results_delta_fro/clustering_report.json}"
OUTPUT_FILE="${OUTPUT_FILE:-representation_comparison.json}"
TARGET_AUROC="${TARGET_AUROC:-0.80}"
TARGET_STABILITY="${TARGET_STABILITY:-0.80}"

echo "=========================================="
echo "Starting representation comparison"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Raw report: $RAW_REPORT"
echo "Delta-SV report: $DELTA_SV_REPORT"
echo "Delta-Fro report: $DELTA_FRO_REPORT"
echo "Output: $OUTPUT_FILE"
echo ""

for report in "$RAW_REPORT" "$DELTA_SV_REPORT" "$DELTA_FRO_REPORT"; do
  if [[ ! -f "$report" ]]; then
    echo "Missing required report: $report" >&2
    exit 1
  fi
done

# Activate conda environment
source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

python compare_representation_reports.py \
  --reports \
    raw="$RAW_REPORT" \
    delta_sv="$DELTA_SV_REPORT" \
    delta_fro="$DELTA_FRO_REPORT" \
  --output-file "$OUTPUT_FILE" \
  --target-auroc "$TARGET_AUROC" \
  --target-stability "$TARGET_STABILITY"

echo ""
echo "=========================================="
echo "Representation comparison completed"
echo "Finished at: $(date)"
echo "=========================================="
ls -lh "$OUTPUT_FILE"
