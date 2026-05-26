#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi
cd "${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-upeftg}
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

CURRENT_USER=${USER:-$(id -un)}
PROJECT_STORAGE_ROOT=${UPEFTGUARD_STORAGE_ROOT:-/models/${CURRENT_USER}/unsupervised-peftguard}
DATASET_ROOT=${DATASET_ROOT:-${UPEFTGUARD_DATA_ROOT:-${PROJECT_STORAGE_ROOT}/data}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/runs}
PARTITION=${PARTITION:-extra}
WORKER_CPUS=${WORKER_CPUS:-auto}
MAX_CONCURRENT=${MAX_CONCURRENT:-auto}
DRY_RUN=${DRY_RUN:-0}
CLASS_WEIGHT_LOSS=${CLASS_WEIGHT_LOSS:-0}
RANK_LABEL_WEIGHT_LOSS=${RANK_LABEL_WEIGHT_LOSS:-0}

RUN_PREFIX=${RUN_PREFIX:-cnn_tbh_tba_zero_shot_r256_to}
MANIFEST_DIR=${MANIFEST_DIR:-${REPO_ROOT}/manifests/zero_shots/tbh+tba_rank_wise}
FEATURE_FILE=${FEATURE_FILE:-${REPO_ROOT}/runs/feature_extract/list2_features-merged-cnn/merged/spectral_features.npy}
CNN_HYPERPARAMS=${CNN_HYPERPARAMS:-${REPO_ROOT}/manifests/cnn_hyperparams/cnn_1d_single_dataset_small_grid.json}

FEATURES=(
  energy
  kurtosis
  l1_norm
  l2_norm
  linf_norm
  mean_abs
  concentration_of_energy
  sv_topk
  stable_rank
  spectral_entropy
  effective_rank
)

COMMON_ARGS=(
  python -m upeftguard.cli experiment supervised-slurm
  --feature-file "${FEATURE_FILE}"
  --features "${FEATURES[@]}"
  --model cnn_1d
  --cnn-hyperparams "${CNN_HYPERPARAMS}"
  --dataset-root "${DATASET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --partition "${PARTITION}"
  --train-split 100
  --calibration-split 20
  --accepted-fpr 0.01 0.05 0.1
  --split-by-folder
  --cv-folds 5
  --cv-seeds 42
  --random-state 42
  --spectral-sv-top-k 8
  --spectral-moment-source both
  --spectral-qv-sum-mode append
  --spectral-entrywise-delta-mode dense
  --skip-feature-importance
)

if [[ "${WORKER_CPUS,,}" != "auto" ]]; then
  COMMON_ARGS+=(--worker-cpus "${WORKER_CPUS}")
fi
if [[ "${MAX_CONCURRENT,,}" != "auto" ]]; then
  COMMON_ARGS+=(--max-concurrent "${MAX_CONCURRENT}")
fi

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi
if [[ "${CLASS_WEIGHT_LOSS}" == "1" || "${CLASS_WEIGHT_LOSS,,}" == "true" ]]; then
  COMMON_ARGS+=(--class-weight-loss)
fi
if [[ "${RANK_LABEL_WEIGHT_LOSS}" == "1" || "${RANK_LABEL_WEIGHT_LOSS,,}" == "true" ]]; then
  COMMON_ARGS+=(--rank-label-weight-loss)
fi

if [[ -n "${MANIFEST_JSON:-}" ]]; then
  MANIFEST_PATHS=("${MANIFEST_JSON}")
else
  mapfile -t MANIFEST_PATHS < <(
    find "${MANIFEST_DIR}" -maxdepth 1 -type f \
      -name "llama2_7b_tbh+tba_zero_shot_r256_to_rank*.json" \
      | sort -V
  )
fi

if [[ "${#MANIFEST_PATHS[@]}" -eq 0 ]]; then
  echo "No tbh+tba rank-wise manifests found under ${MANIFEST_DIR}" >&2
  exit 1
fi

echo "Submitting ${#MANIFEST_PATHS[@]} tbh+tba rank-wise zero-shot CNN runs from ${MANIFEST_DIR}"
for manifest_json in "${MANIFEST_PATHS[@]}"; do
  manifest_json="$(cd "$(dirname "${manifest_json}")" && pwd -P)/$(basename "${manifest_json}")"
  manifest_name="$(basename "${manifest_json}")"
  if [[ "${manifest_name}" =~ (rank[0-9]+)\.json$ ]]; then
    run_id="${RUN_PREFIX}_${BASH_REMATCH[1]}"
  else
    run_id="${RUN_PREFIX}_${manifest_name%.json}"
  fi

  echo "Submitting ${run_id} from ${manifest_json}"
  "${COMMON_ARGS[@]}" \
    --manifest-json "${manifest_json}" \
    --run-id "${run_id}"
done
