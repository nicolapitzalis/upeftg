#!/bin/bash
#SBATCH --job-name=upeftguard_supervised_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/supervised_prepare_%j.out
#SBATCH --error=logs/supervised_prepare_%j.err
#SBATCH --partition=extra

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-upeftg}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

MANIFEST_JSON=${MANIFEST_JSON:-}
FEATURE_FILE=${FEATURE_FILE:-}
FEATURES=${FEATURES:-}
if [[ -z "${MANIFEST_JSON}" ]]; then
  echo "MANIFEST_JSON is required." >&2
  exit 1
fi
if [[ -z "${FEATURE_FILE}" ]]; then
  echo "FEATURE_FILE is required. Pass a feature run name, merged dir, or spectral_features.npy path." >&2
  exit 1
fi
if [[ -z "${FEATURES}" ]]; then
  echo "FEATURES is required. Specify the feature groups to select from the extracted bundle." >&2
  exit 1
fi

CURRENT_USER=${USER:-$(id -un)}
PROJECT_STORAGE_ROOT=${UPEFTGUARD_STORAGE_ROOT:-/models/${CURRENT_USER}/unsupervised-peftguard}
DATASET_ROOT=${DATASET_ROOT:-${UPEFTGUARD_DATA_ROOT:-${PROJECT_STORAGE_ROOT}/data}}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-supervised_${SLURM_JOB_ID:-manual}}
PIPELINE_MODE=${PIPELINE_MODE:-full}
MODEL=${MODEL:-all}
SV_TOP_K=${SV_TOP_K:-8}
SPECTRAL_MOMENT_SOURCE=${SPECTRAL_MOMENT_SOURCE:-sv}
SPECTRAL_QV_SUM_MODE=${SPECTRAL_QV_SUM_MODE:-none}
SPECTRAL_ENTRYWISE_DELTA_MODE=${SPECTRAL_ENTRYWISE_DELTA_MODE:-auto}
CV_FOLDS=${CV_FOLDS:-5}
CV_SEEDS=${CV_SEEDS:-"42 43 44"}
TRAIN_SPLIT=${TRAIN_SPLIT:-100}
CALIBRATION_SPLIT=${CALIBRATION_SPLIT:-}
ACCEPTED_FPR=${ACCEPTED_FPR:-}
SPLIT_BY_FOLDER=${SPLIT_BY_FOLDER:-0}
SCORE_PERCENTILES=${SCORE_PERCENTILES:-}
SLURM_PARTITION=${SLURM_PARTITION:-extra}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-logs}
SKIP_FEATURE_IMPORTANCE=${SKIP_FEATURE_IMPORTANCE:-0}

mkdir -p "${SLURM_LOG_DIR}"

read -r -a FEATURE_VALUES <<< "${FEATURES}"
if [[ "${#FEATURE_VALUES[@]}" -eq 0 ]]; then
  echo "FEATURES must include at least one feature group." >&2
  exit 1
fi

SPLIT_BY_FOLDER_FLAG=""
if [[ "${SPLIT_BY_FOLDER}" == "1" || "${SPLIT_BY_FOLDER,,}" == "true" ]]; then
  SPLIT_BY_FOLDER_FLAG="--split-by-folder"
fi

CALIBRATION_ARGS=()
if [[ -n "${CALIBRATION_SPLIT}" || -n "${ACCEPTED_FPR}" ]]; then
  if [[ -z "${CALIBRATION_SPLIT}" || -z "${ACCEPTED_FPR}" ]]; then
    echo "CALIBRATION_SPLIT and ACCEPTED_FPR must either both be set or both be empty." >&2
    exit 1
  fi
  read -r -a ACCEPTED_FPR_VALUES <<< "${ACCEPTED_FPR}"
  if [[ "${#ACCEPTED_FPR_VALUES[@]}" -eq 0 ]]; then
    echo "ACCEPTED_FPR must include at least one value when CALIBRATION_SPLIT is set." >&2
    exit 1
  fi
  CALIBRATION_ARGS+=(--calibration-split "${CALIBRATION_SPLIT}")
  CALIBRATION_ARGS+=(--accepted-fpr "${ACCEPTED_FPR_VALUES[@]}")
fi

read -r -a CV_SEED_VALUES <<< "${CV_SEEDS}"
SCORE_PERCENTILE_VALUES=()
if [[ -n "${SCORE_PERCENTILES}" ]]; then
  read -r -a SCORE_PERCENTILE_VALUES <<< "${SCORE_PERCENTILES}"
fi

PREPARE_ARGS=(
  python -m upeftguard.cli run supervised
  --stage prepare
  --manifest-json "${MANIFEST_JSON}"
  --dataset-root "${DATASET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
  --model "${MODEL}"
  --features "${FEATURE_VALUES[@]}"
  --spectral-sv-top-k "${SV_TOP_K}"
  --spectral-moment-source "${SPECTRAL_MOMENT_SOURCE}"
  --spectral-qv-sum-mode "${SPECTRAL_QV_SUM_MODE}"
  --spectral-entrywise-delta-mode "${SPECTRAL_ENTRYWISE_DELTA_MODE}"
  --cv-folds "${CV_FOLDS}"
  --train-split "${TRAIN_SPLIT}"
  --feature-file "${FEATURE_FILE}"
  --tuning-executor slurm_array
)
if [[ -n "${SPLIT_BY_FOLDER_FLAG}" ]]; then
  PREPARE_ARGS+=("${SPLIT_BY_FOLDER_FLAG}")
fi
if [[ "${#CALIBRATION_ARGS[@]}" -gt 0 ]]; then
  PREPARE_ARGS+=("${CALIBRATION_ARGS[@]}")
fi
if [[ "${#CV_SEED_VALUES[@]}" -gt 0 ]]; then
  PREPARE_ARGS+=(--cv-seeds "${CV_SEED_VALUES[@]}")
fi
if [[ "${#SCORE_PERCENTILE_VALUES[@]}" -gt 0 ]]; then
  PREPARE_ARGS+=(--score-percentiles "${SCORE_PERCENTILE_VALUES[@]}")
fi

"${PREPARE_ARGS[@]}"

RUN_DIR=$(python - <<PY
from pathlib import Path
print((Path("${OUTPUT_ROOT}").expanduser().resolve() / "supervised" / "${RUN_ID}").as_posix())
PY
)

TUNING_MANIFEST="${RUN_DIR}/reports/tuning_manifest.json"
if [[ ! -f "${TUNING_MANIFEST}" ]]; then
  echo "Missing tuning manifest: ${TUNING_MANIFEST}" >&2
  exit 1
fi

N_TASKS=$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${TUNING_MANIFEST}").read_text(encoding="utf-8"))
print(len(payload["tuning"]["tasks"]))
PY
)

if [[ "${N_TASKS}" -le 0 ]]; then
  echo "No tuning tasks found in ${TUNING_MANIFEST}" >&2
  exit 1
fi

DEFAULT_SLURM_MAX_CONCURRENT=$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${TUNING_MANIFEST}").read_text(encoding="utf-8"))
print(int(payload["runtime"]["slurm_max_concurrent"]))
PY
)
DEFAULT_SLURM_CPUS_PER_TASK=$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${TUNING_MANIFEST}").read_text(encoding="utf-8"))
print(int(payload["runtime"]["slurm_cpus_per_task"]))
PY
)

SLURM_MAX_CONCURRENT=${SLURM_MAX_CONCURRENT_OVERRIDE:-${DEFAULT_SLURM_MAX_CONCURRENT}}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK_OVERRIDE:-${DEFAULT_SLURM_CPUS_PER_TASK}}
ARRAY_MAX=$((N_TASKS - 1))
FINALIZE_SKIP_FLAG=""
if [[ "${SKIP_FEATURE_IMPORTANCE}" == "1" || "${SKIP_FEATURE_IMPORTANCE,,}" == "true" ]]; then
  FINALIZE_SKIP_FLAG=" --skip-feature-importance"
fi

FINALIZE_SCORE_PERCENTILES_ARGS=""
if [[ -n "${SCORE_PERCENTILES}" ]]; then
  FINALIZE_SCORE_PERCENTILES_ARGS=" --score-percentiles ${SCORE_PERCENTILES}"
fi

if [[ -z "${FINALIZE_SKIP_FLAG}" ]]; then
FEATURE_DIM=$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${TUNING_MANIFEST}").read_text(encoding="utf-8"))
extractor = payload.get("extractor", {})
metadata = extractor.get("metadata", {}) if isinstance(extractor, dict) else {}
print(int(metadata.get("feature_dim", 0)))
PY
)

if [[ "${FEATURE_DIM}" -le 0 ]]; then
  echo "Unable to resolve feature_dim from ${TUNING_MANIFEST}" >&2
  exit 1
fi

PARTITION_NODE_COUNT=$(sinfo -h -p "${SLURM_PARTITION}" -o "%D" | head -n 1)
PARTITION_CPUS_PER_NODE=$(sinfo -h -p "${SLURM_PARTITION}" -o "%c" | sort -nr | head -n 1)
PARTITION_NODE_COUNT=${PARTITION_NODE_COUNT:-1}
PARTITION_CPUS_PER_NODE=${PARTITION_CPUS_PER_NODE:-${SLURM_CPUS_PER_TASK}}

FINALIZE_EXPORT_MAX_CONCURRENT=${FINALIZE_EXPORT_MAX_CONCURRENT_OVERRIDE:-${PARTITION_NODE_COUNT}}
FINALIZE_EXPORT_CPUS_PER_TASK=${FINALIZE_EXPORT_CPUS_PER_TASK_OVERRIDE:-${PARTITION_CPUS_PER_NODE}}
FINALIZE_EXPORT_TASKS_PER_NODE=${FINALIZE_EXPORT_TASKS_PER_NODE:-4}
FINALIZE_EXPORT_TASKS=${FINALIZE_EXPORT_TASKS_OVERRIDE:-$(python - <<PY
feature_dim = int("${FEATURE_DIM}")
max_concurrent = max(1, int("${FINALIZE_EXPORT_MAX_CONCURRENT}"))
tasks_per_node = max(1, int("${FINALIZE_EXPORT_TASKS_PER_NODE}"))
print(max(1, min(feature_dim, max_concurrent * tasks_per_node)))
PY
)}
FINALIZE_EXPORT_ARRAY_MAX=$((FINALIZE_EXPORT_TASKS - 1))
fi

echo "Run dir: ${RUN_DIR}"
echo "Feature file: ${FEATURE_FILE}"
echo "Feature companions: auto-resolved from FEATURE_FILE"
echo "Features: ${FEATURES}"
echo "Train split: ${TRAIN_SPLIT}"
echo "Calibration split: ${CALIBRATION_SPLIT:-disabled}"
echo "Accepted FPR: ${ACCEPTED_FPR:-disabled}"
echo "Split by folder: ${SPLIT_BY_FOLDER}"
if [[ -n "${SCORE_PERCENTILES}" ]]; then
  echo "Score percentiles: ${SCORE_PERCENTILES}"
else
  echo "Score percentiles: <manifest default>"
fi
echo "Tuning tasks: ${N_TASKS}"
echo "Pipeline mode: ${PIPELINE_MODE}"

if [[ "${PIPELINE_MODE}" == "prepare_only" ]]; then
  echo "Preparation completed; skipping worker/finalize submission."
  exit 0
fi

WORKER_JOB_ID=$(sbatch \
  --parsable \
  --partition "${SLURM_PARTITION}" \
  --cpus-per-task "${SLURM_CPUS_PER_TASK}" \
  --array "0-${ARRAY_MAX}%${SLURM_MAX_CONCURRENT}" \
  --job-name "upeftguard_supervised_worker_${RUN_ID}" \
  --output "${SLURM_LOG_DIR}/supervised_worker_${RUN_ID}_%A_%a.out" \
  --error "${SLURM_LOG_DIR}/supervised_worker_${RUN_ID}_%A_%a.err" \
  --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli run supervised --stage worker --run-dir ${RUN_DIR} --task-index \${SLURM_ARRAY_TASK_ID} --n-jobs ${SLURM_CPUS_PER_TASK}")

if [[ -n "${FINALIZE_SKIP_FLAG}" ]]; then
  FINALIZE_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task 4 \
    --dependency "afterok:${WORKER_JOB_ID}" \
    --job-name "upeftguard_supervised_finalize_${RUN_ID}" \
    --output "${SLURM_LOG_DIR}/supervised_finalize_${RUN_ID}_%j.out" \
    --error "${SLURM_LOG_DIR}/supervised_finalize_${RUN_ID}_%j.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli run supervised --stage finalize --run-dir ${RUN_DIR}${FINALIZE_SCORE_PERCENTILES_ARGS}${FINALIZE_SKIP_FLAG}")
else
  FINALIZE_PREPARE_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task 4 \
    --dependency "afterok:${WORKER_JOB_ID}" \
    --job-name "upeftguard_supervised_finalize_prepare_${RUN_ID}" \
    --output "${SLURM_LOG_DIR}/supervised_finalize_prepare_${RUN_ID}_%j.out" \
    --error "${SLURM_LOG_DIR}/supervised_finalize_prepare_${RUN_ID}_%j.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli run supervised --stage finalize_prepare --run-dir ${RUN_DIR}${FINALIZE_SCORE_PERCENTILES_ARGS} --finalize-export-shards ${FINALIZE_EXPORT_TASKS}")

  FINALIZE_WORKER_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task "${FINALIZE_EXPORT_CPUS_PER_TASK}" \
    --array "0-${FINALIZE_EXPORT_ARRAY_MAX}%${FINALIZE_EXPORT_MAX_CONCURRENT}" \
    --dependency "afterok:${FINALIZE_PREPARE_JOB_ID}" \
    --job-name "upeftguard_supervised_finalize_worker_${RUN_ID}" \
    --output "${SLURM_LOG_DIR}/supervised_finalize_worker_${RUN_ID}_%A_%a.out" \
    --error "${SLURM_LOG_DIR}/supervised_finalize_worker_${RUN_ID}_%A_%a.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli run supervised --stage finalize_worker --run-dir ${RUN_DIR} --task-index \${SLURM_ARRAY_TASK_ID} --n-jobs ${FINALIZE_EXPORT_CPUS_PER_TASK}")

  FINALIZE_MERGE_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task 4 \
    --dependency "afterok:${FINALIZE_WORKER_JOB_ID}" \
    --job-name "upeftguard_supervised_finalize_merge_${RUN_ID}" \
    --output "${SLURM_LOG_DIR}/supervised_finalize_merge_${RUN_ID}_%j.out" \
    --error "${SLURM_LOG_DIR}/supervised_finalize_merge_${RUN_ID}_%j.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli run supervised --stage finalize_merge --run-dir ${RUN_DIR}")
fi

echo "Worker job id: ${WORKER_JOB_ID}"
echo "Skip feature importance: ${SKIP_FEATURE_IMPORTANCE}"
if [[ -n "${FINALIZE_SKIP_FLAG}" ]]; then
  echo "Finalize job id: ${FINALIZE_JOB_ID}"
else
  echo "Finalize export shards: ${FINALIZE_EXPORT_TASKS}"
  echo "Finalize export max concurrent: ${FINALIZE_EXPORT_MAX_CONCURRENT}"
  echo "Finalize export cpus per task: ${FINALIZE_EXPORT_CPUS_PER_TASK}"
  echo "Finalize prepare job id: ${FINALIZE_PREPARE_JOB_ID}"
  echo "Finalize worker job id: ${FINALIZE_WORKER_JOB_ID}"
  echo "Finalize merge job id: ${FINALIZE_MERGE_JOB_ID}"
fi
