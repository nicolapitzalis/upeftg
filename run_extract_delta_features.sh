#!/bin/bash
#SBATCH --job-name=upeftguard_feature_delta
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/feature_delta_%j.log
#SBATCH --error=logs/feature_delta_%j.err
#SBATCH --partition=extra

set -euo pipefail

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

MANIFEST_JSON=${MANIFEST_JSON:-prepare_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}
RUN_ID=${RUN_ID:-feature_spectral_${SLURM_JOB_ID:-manual}}

python -m upeftguard.cli feature extract \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --extractor spectral \
  --spectral-features frobenius energy kurtosis l1_norm linf_norm sv_topk \
  --spectral-sv-top-k 8 \
  --output-root runs \
  --run-id "${RUN_ID}"
