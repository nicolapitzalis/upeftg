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
CURRENT_USER=${USER:-$(id -un)}
PROJECT_STORAGE_ROOT=${UPEFTGUARD_STORAGE_ROOT:-/models/${CURRENT_USER}/unsupervised-peftguard}
DATASET_ROOT=${DATASET_ROOT:-${UPEFTGUARD_DATA_ROOT:-${PROJECT_STORAGE_ROOT}/data}}
RUN_ID=${RUN_ID:-feature_spectral_${SLURM_JOB_ID:-manual}}
SPECTRAL_QV_SUM_MODE=${SPECTRAL_QV_SUM_MODE:-none}
SPECTRAL_MOMENT_SOURCE=${SPECTRAL_MOMENT_SOURCE:-sv}

python -m upeftguard.cli feature extract \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --extractor spectral \
  --spectral-features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source "${SPECTRAL_MOMENT_SOURCE}" \
  --spectral-qv-sum-mode "${SPECTRAL_QV_SUM_MODE}" \
  --output-root runs \
  --run-id "${RUN_ID}"
