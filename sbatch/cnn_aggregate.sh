#!/bin/bash
#SBATCH --job-name=upeftguard_cnn_aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/cnn_aggregate_%j.out
#SBATCH --error=logs/cnn_aggregate_%j.err
#SBATCH --partition=extra

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/upeftguard/cli.py" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi
if [[ ! -f "${REPO_ROOT}/upeftguard/cli.py" ]]; then
  echo "Could not resolve repository root: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-upeftg}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

FEATURE_FILE=${FEATURE_FILE:-}
OUTPUT_FILENAME=${OUTPUT_FILENAME:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-cnn_aggregate_${SLURM_JOB_ID:-manual}}
FEATURE_ROOT=${FEATURE_ROOT:-runs/feature_extract}
FEATURES=${FEATURES:-"energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank"}
SPECTRAL_QV_SUM_MODE=${SPECTRAL_QV_SUM_MODE:-append}

if [[ -z "${FEATURE_FILE}" ]]; then
  echo "FEATURE_FILE is required." >&2
  exit 1
fi
if [[ -z "${OUTPUT_FILENAME}" ]]; then
  echo "OUTPUT_FILENAME is required." >&2
  exit 1
fi

read -r -a FEATURE_VALUES <<< "${FEATURES}"

ARGS=(
  python -m upeftguard.cli cnn aggregate
  --backend local
  --feature-file "${FEATURE_FILE}"
  --output-filename "${OUTPUT_FILENAME}"
  --output-root "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
  --feature-root "${FEATURE_ROOT}"
  --spectral-qv-sum-mode "${SPECTRAL_QV_SUM_MODE}"
)
if [[ "${#FEATURE_VALUES[@]}" -gt 0 ]]; then
  ARGS+=(--features "${FEATURE_VALUES[@]}")
fi

"${ARGS[@]}"
