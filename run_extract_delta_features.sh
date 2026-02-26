#!/bin/bash
#SBATCH --job-name=upeftguard_feature_delta
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/feature_delta_%j.log
#SBATCH --error=logs/feature_delta_%j.err

set -euo pipefail

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

MANIFEST_JSON=${MANIFEST_JSON:-prepare_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}
RUN_ID_SV=${RUN_ID_SV:-feature_delta_sv_${SLURM_JOB_ID:-manual}}
RUN_ID_FRO=${RUN_ID_FRO:-feature_delta_fro_${SLURM_JOB_ID:-manual}}

python -m upeftguard.cli feature extract \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --extractor delta_singular_values \
  --top-k-singular-values 8 \
  --output-root runs \
  --run-id "${RUN_ID_SV}"

python -m upeftguard.cli feature extract \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --extractor delta_frobenius \
  --top-k-singular-values 8 \
  --output-root runs \
  --run-id "${RUN_ID_FRO}"
