#!/bin/bash
#SBATCH --job-name=upeftguard_cnn_infer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/cnn_infer_%j.out
#SBATCH --error=logs/cnn_infer_%j.err
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

CHECKPOINT=${CHECKPOINT:-}
RUN_DIR=${RUN_DIR:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-cnn_infer_${SLURM_JOB_ID:-manual}}

if [[ -z "${CHECKPOINT}" && -z "${RUN_DIR}" ]]; then
  echo "Either CHECKPOINT or RUN_DIR is required." >&2
  exit 1
fi

ARGS=(
  python -m upeftguard.cli cnn infer
  --backend local
  --output-root "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${RUN_DIR}" ]]; then
  ARGS+=(--run-dir "${RUN_DIR}")
fi

"${ARGS[@]}"
