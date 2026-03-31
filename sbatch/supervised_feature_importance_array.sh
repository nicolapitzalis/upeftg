#!/bin/bash
#SBATCH --job-name=upeftguard_feature_importance_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --output=logs/supervised_feature_importance_prepare_%j.out
#SBATCH --error=logs/supervised_feature_importance_prepare_%j.err
#SBATCH --partition=extra

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-upeftg}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

RUN_DIR=${RUN_DIR:-}
if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR is required, e.g. RUN_DIR=runs/supervised/<RUN_ID>" >&2
  exit 1
fi

RUN_DIR=$(python - <<PY
from pathlib import Path
print(Path("${RUN_DIR}").expanduser().resolve().as_posix())
PY
)
RUN_ID=${RUN_ID:-$(basename "${RUN_DIR}")}
SLURM_PARTITION=${SLURM_PARTITION:-extra}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-logs}
SCORE_PERCENTILES=${SCORE_PERCENTILES:-}

mkdir -p "${SLURM_LOG_DIR}"

TUNING_MANIFEST="${RUN_DIR}/reports/tuning_manifest.json"
ARTIFACT_INDEX="${RUN_DIR}/artifact_index.json"
if [[ ! -f "${TUNING_MANIFEST}" ]]; then
  echo "Missing tuning manifest: ${TUNING_MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${ARTIFACT_INDEX}" ]]; then
  echo "Missing artifact index: ${ARTIFACT_INDEX}" >&2
  exit 1
fi

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
PARTITION_CPUS_PER_NODE=${PARTITION_CPUS_PER_NODE:-1}

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
SCORE_PERCENTILES_ARGS=()
if [[ -n "${SCORE_PERCENTILES}" ]]; then
  read -r -a SCORE_PERCENTILE_VALUES <<< "${SCORE_PERCENTILES}"
  SCORE_PERCENTILES_ARGS=(--score-percentiles "${SCORE_PERCENTILE_VALUES[@]}")
fi

python -m upeftguard.cli run supervised \
  --stage finalize_prepare \
  --run-dir "${RUN_DIR}" \
  "${SCORE_PERCENTILES_ARGS[@]}" \
  --finalize-export-shards "${FINALIZE_EXPORT_TASKS}"

WINNER_EXPORT_MANIFEST="${RUN_DIR}/reports/winner_feature_weights_manifest.json"
if [[ ! -f "${WINNER_EXPORT_MANIFEST}" ]]; then
  echo "Missing winner feature export manifest: ${WINNER_EXPORT_MANIFEST}" >&2
  exit 1
fi

WINNER_EXPORT_MODE=$(python - <<PY
import json
from pathlib import Path

payload = json.loads(Path("${WINNER_EXPORT_MANIFEST}").read_text(encoding="utf-8"))
print(str(payload.get("mode", "")))
PY
)
WINNER_EXPORT_TASKS=$(python - <<PY
import json
from pathlib import Path

payload = json.loads(Path("${WINNER_EXPORT_MANIFEST}").read_text(encoding="utf-8"))
print(int(payload.get("n_tasks", 0)))
PY
)

FINALIZE_PREPARE_JOB_ID="${SLURM_JOB_ID:-manual}"
FINALIZE_WORKER_JOB_ID=""
FINALIZE_MERGE_JOB_ID=""

if [[ "${WINNER_EXPORT_MODE}" == "completed" || "${WINNER_EXPORT_TASKS}" -le 0 ]]; then
  python -m upeftguard.cli run supervised \
    --stage finalize_merge \
    --run-dir "${RUN_DIR}"
else
  FINALIZE_EXPORT_ARRAY_MAX=$((WINNER_EXPORT_TASKS - 1))

  FINALIZE_WORKER_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task "${FINALIZE_EXPORT_CPUS_PER_TASK}" \
    --array "0-${FINALIZE_EXPORT_ARRAY_MAX}%${FINALIZE_EXPORT_MAX_CONCURRENT}" \
    --job-name "upeftguard_feature_importance_worker_${RUN_ID}" \
    --output "${SLURM_LOG_DIR}/feature_importance_worker_${RUN_ID}_%A_%a.out" \
    --error "${SLURM_LOG_DIR}/feature_importance_worker_${RUN_ID}_%A_%a.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli run supervised --stage finalize_worker --run-dir ${RUN_DIR} --task-index \${SLURM_ARRAY_TASK_ID} --n-jobs ${FINALIZE_EXPORT_CPUS_PER_TASK}")

  FINALIZE_MERGE_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task 4 \
    --dependency "afterok:${FINALIZE_WORKER_JOB_ID}" \
    --job-name "upeftguard_feature_importance_merge_${RUN_ID}" \
    --output "${SLURM_LOG_DIR}/feature_importance_merge_${RUN_ID}_%j.out" \
    --error "${SLURM_LOG_DIR}/feature_importance_merge_${RUN_ID}_%j.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli run supervised --stage finalize_merge --run-dir ${RUN_DIR}")
fi

echo "Run dir: ${RUN_DIR}"
echo "Run id: ${RUN_ID}"
if [[ -n "${SCORE_PERCENTILES}" ]]; then
  echo "Score percentiles override: ${SCORE_PERCENTILES}"
else
  echo "Score percentiles override: <manifest default>"
fi
echo "Feature dim: ${FEATURE_DIM}"
echo "Winner export mode: ${WINNER_EXPORT_MODE}"
echo "Winner export tasks: ${WINNER_EXPORT_TASKS}"
echo "Finalize export shards: ${FINALIZE_EXPORT_TASKS}"
echo "Finalize export max concurrent: ${FINALIZE_EXPORT_MAX_CONCURRENT}"
echo "Finalize export cpus per task: ${FINALIZE_EXPORT_CPUS_PER_TASK}"
echo "Finalize prepare job id: ${FINALIZE_PREPARE_JOB_ID}"
if [[ -n "${FINALIZE_WORKER_JOB_ID}" ]]; then
  echo "Finalize worker job id: ${FINALIZE_WORKER_JOB_ID}"
  echo "Finalize merge job id: ${FINALIZE_MERGE_JOB_ID}"
else
  echo "Finalize merge: completed in launcher job"
fi
