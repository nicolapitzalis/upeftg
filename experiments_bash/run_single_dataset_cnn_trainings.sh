#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi
cd "${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-upeftg}
if [[ -f "${CONDA_SH}" ]]; then
  # Needed for the launcher process; the submitted Slurm jobs source this again.
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
RUN_PREFIX=${RUN_PREFIX:-single_dataset_cnn}
DRY_RUN=${DRY_RUN:-0}
CLASS_WEIGHT_LOSS=${CLASS_WEIGHT_LOSS:-0}
RANK_LABEL_WEIGHT_LOSS=${RANK_LABEL_WEIGHT_LOSS:-0}

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

RUN_SPECS=(
  # Already computed base Llama2-7B task/attack datasets.
  # "squad_insertsent|manifests/single_datasets/llama2_7b_squad_insertsent.json"
  # "tba|manifests/single_datasets/llama2_7b_toxic_backdoors_alpaca.json"
  # "tbh|manifests/single_datasets/llama2_7b_toxic_backdoors_hard.json"
  # "imdb_insertsent|manifests/single_datasets/llama2_7b_imdb_insertsent.json"
  # "imdb_ripple|manifests/single_datasets/llama2_7b_imdb_ripple.json"
  # "imdb_stybkd|manifests/single_datasets/llama2_7b_imdb_stybkd.json"
  # "imdb_syntactic|manifests/single_datasets/llama2_7b_imdb_syntactic.json"
  # "ag_news_insertsent|manifests/single_datasets/llama2_7b_ag_news_insertsent.json"
  # "ag_news_ripple|manifests/single_datasets/llama2_7b_ag_news_ripple.json"
  # "ag_news_stybkd|manifests/single_datasets/llama2_7b_ag_news_stybkd.json"
  # "ag_news_syntactic|manifests/single_datasets/llama2_7b_ag_news_syntactic.json"
  # Architecture variants available in list2_features-merged-cnn.
  "tbh_flan_t5_xl|manifests/architecture_exploration/flan_t5_xl_architecture_tbh.json"
  "tbh_llama2_13b|manifests/architecture_exploration/llama2_13b_architecture_tbh.json"
  "tbh_qwen1_5_7b|manifests/architecture_exploration/qwen1.5_7b_architecture_tbh.json"
  "roberta_base_imdb_insertsent|manifests/architecture_exploration/roberta_base_architecture.json"
  # Adapter variants available in list2_features-merged-cnn.
  "tbh_adalora|manifests/adapter_exploration/llama2_7b_adalora_tbh.json"
  "tbh_dora|manifests/adapter_exploration/llama2_7b_dora_tbh.json"
  "tbh_lora_plus|manifests/adapter_exploration/llama2_7b_lora_plus_tbh.json"
  "tbh_qlora|manifests/adapter_exploration/llama2_7b_qlora_tbh.json"
  # Rank variants available in list2_features-merged-cnn; rank256 is covered by tbh above.
  "tbh_rank8|manifests/rank_exploration/llama2_7b_tbh_rank8.json"
  "tbh_rank16|manifests/rank_exploration/llama2_7b_tbh_rank16.json"
  "tbh_rank32|manifests/rank_exploration/llama2_7b_tbh_rank32.json"
  "tbh_rank64|manifests/rank_exploration/llama2_7b_tbh_rank64.json"
  "tbh_rank128|manifests/rank_exploration/llama2_7b_tbh_rank128.json"
  "tbh_rank512|manifests/rank_exploration/llama2_7b_tbh_rank512.json"
  "tbh_rank1024|manifests/rank_exploration/llama2_7b_tbh_rank1024.json"
  "tbh_rank2048|manifests/rank_exploration/llama2_7b_tbh_rank2048.json"
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
  --train-split 80
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

for spec in "${RUN_SPECS[@]}"; do
  IFS="|" read -r run_suffix manifest_json <<< "${spec}"
  run_id="${RUN_PREFIX}_${run_suffix}"
  echo "Submitting ${run_id} from ${manifest_json}"
  "${COMMON_ARGS[@]}" \
    --manifest-json "${REPO_ROOT}/${manifest_json}" \
    --run-id "${run_id}"
done
