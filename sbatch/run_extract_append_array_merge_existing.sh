#!/bin/bash
#SBATCH --job-name=upeftguard_feature_full_append_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/feature_full_append_prepare_%j.out
#SBATCH --error=logs/feature_full_append_prepare_%j.err
#SBATCH --partition=extra
#SBATCH --chdir=/home/n.pitzalis/unsupervised-peftguard

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/n.pitzalis/unsupervised-peftguard}
cd "${REPO_ROOT}"

# Full spectral extraction (baseline blocks + qv_sum blocks), then merge into an
# existing consolidated feature table.
export SPECTRAL_QV_SUM_MODE="${SPECTRAL_QV_SUM_MODE:-append}"
export MERGE_WITH_EXISTING="${MERGE_WITH_EXISTING:-1}"
export RUN_ID="${RUN_ID:-feature_spectral_full_append_${SLURM_JOB_ID:-manual}}"
export EXISTING_MERGED_DIR="${EXISTING_MERGED_DIR:-/home/n.pitzalis/unsupervised-peftguard/runs/feature_extract/feature_spectral_array_34017/merged}"

# Alias for convenience: where the pre-existing merged files already live.
# Example: EXISTING_MERGED_DIR=runs/feature_extract/feature_spectral_array_123456/merged
if [[ -n "${EXISTING_MERGED_DIR:-}" ]]; then
  export MERGED_OUTPUT_DIR="${EXISTING_MERGED_DIR}"
fi

if [[ ! -d "${MERGED_OUTPUT_DIR}" ]]; then
  echo "ERROR: MERGED_OUTPUT_DIR does not exist: ${MERGED_OUTPUT_DIR}" >&2
  exit 1
fi
for name in spectral_features.npy spectral_model_names.json spectral_metadata.json; do
  if [[ ! -f "${MERGED_OUTPUT_DIR}/${name}" ]]; then
    echo "ERROR: Missing required existing merged artifact: ${MERGED_OUTPUT_DIR}/${name}" >&2
    exit 1
  fi
done

DELEGATE_SCRIPT="${REPO_ROOT}/sbatch/run_extract_delta_features_array.sh"
if [[ ! -x "${DELEGATE_SCRIPT}" ]]; then
  echo "ERROR: Delegate script not found/executable: ${DELEGATE_SCRIPT}" >&2
  exit 1
fi

echo "Delegating to: ${DELEGATE_SCRIPT}"
echo "SPECTRAL_QV_SUM_MODE=${SPECTRAL_QV_SUM_MODE}"
echo "MERGE_WITH_EXISTING=${MERGE_WITH_EXISTING}"
if [[ -n "${MERGED_OUTPUT_DIR:-}" ]]; then
  echo "MERGED_OUTPUT_DIR=${MERGED_OUTPUT_DIR}"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Top-level logs: ${REPO_ROOT}/logs/feature_full_append_prepare_${SLURM_JOB_ID}.out/.err"
fi

"${DELEGATE_SCRIPT}"
