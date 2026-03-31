#!/bin/bash
#SBATCH --job-name=upeftguard_feature_extract_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/feature_extract_prepare_%j.out
#SBATCH --error=logs/feature_extract_prepare_%j.err
#SBATCH --partition=extra

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-upeftg}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

MANIFEST_JSON=${MANIFEST_JSON:-}
if [[ -z "${MANIFEST_JSON}" ]]; then
  echo "MANIFEST_JSON is required." >&2
  exit 1
fi

CURRENT_USER=${USER:-$(id -un)}
PROJECT_STORAGE_ROOT=${UPEFTGUARD_STORAGE_ROOT:-/models/${CURRENT_USER}/unsupervised-peftguard}
DATASET_ROOT=${DATASET_ROOT:-${UPEFTGUARD_DATA_ROOT:-${PROJECT_STORAGE_ROOT}/data}}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-feature_extract_${SLURM_JOB_ID:-manual}}
FEATURES=${FEATURES:-"energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank"}
SV_TOP_K=${SV_TOP_K:-8}
SPECTRAL_MOMENT_SOURCE=${SPECTRAL_MOMENT_SOURCE:-both}
SPECTRAL_QV_SUM_MODE=${SPECTRAL_QV_SUM_MODE:-none}
SPECTRAL_ENTRYWISE_DELTA_MODE=${SPECTRAL_ENTRYWISE_DELTA_MODE:-dense}
STREAM_BLOCK_SIZE=${STREAM_BLOCK_SIZE:-131072}
DTYPE=${DTYPE:-float32}
N_SHARDS=${N_SHARDS:-8}
PIPELINE_START_EPOCH_SECONDS=${PIPELINE_START_EPOCH_SECONDS:-$(date -u +%s)}
SLURM_PARTITION=${SLURM_PARTITION:-extra}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
SLURM_MAX_CONCURRENT=${SLURM_MAX_CONCURRENT:-8}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-logs}

read -r -a FEATURE_VALUES <<< "${FEATURES}"
if [[ "${#FEATURE_VALUES[@]}" -eq 0 ]]; then
  echo "FEATURES must include at least one feature group." >&2
  exit 1
fi

mkdir -p "${SLURM_LOG_DIR}"

RUN_ROOT=$(python - <<PY
from pathlib import Path
print((Path("${OUTPUT_ROOT}").expanduser().resolve() / "feature_extract" / "${RUN_ID}").as_posix())
PY
)
SCHEMA_GROUP_ROOT="${RUN_ROOT}/schema_groups"
MERGED_DIR="${RUN_ROOT}/merged"
SCHEMA_REPORT_PATH="${RUN_ROOT}/schema_partition_report.json"

mkdir -p "${SCHEMA_GROUP_ROOT}" "${MERGED_DIR}"

python -m upeftguard.utilities.prepare_spectral_shards \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${SCHEMA_GROUP_ROOT}" \
  --n-shards "${N_SHARDS}" \
  --spectral-qv-sum-mode "${SPECTRAL_QV_SUM_MODE}" \
  --report-path "${SCHEMA_REPORT_PATH}"

GROUP_COUNT=$(python - <<PY
import json
from pathlib import Path

payload = json.loads(Path("${SCHEMA_REPORT_PATH}").read_text(encoding="utf-8"))
print(int(payload.get("group_count", 0)))
PY
)

if [[ "${GROUP_COUNT}" -le 0 ]]; then
  echo "No schema groups were prepared." >&2
  exit 1
fi

python - <<PY
import json
from pathlib import Path

payload = json.loads(Path("${SCHEMA_REPORT_PATH}").read_text(encoding="utf-8"))
for warning in payload.get("warnings", []):
    print(f"Schema warning: {warning}")
PY

WORKER_JOB_IDS=()
MERGE_JOB_IDS=()
MERGE_DEPENDENCY_JOB_IDS=()
FINALIZE_JOB_ID=""
while IFS=$'\t' read -r GROUP_ID GROUP_MANIFEST_JSON GROUP_SHARD_MANIFEST_DIR GROUP_SHARD_OUTPUT_ROOT GROUP_MERGED_OUTPUT_DIR GROUP_QV_MODE GROUP_N_SHARDS GROUP_N_ITEMS; do
  if [[ -z "${GROUP_ID}" ]]; then
    continue
  fi

  ARRAY_MAX=$((GROUP_N_SHARDS - 1))
  EFFECTIVE_MERGED_OUTPUT_DIR="${GROUP_MERGED_OUTPUT_DIR}"
  if [[ "${GROUP_COUNT}" -eq 1 ]]; then
    EFFECTIVE_MERGED_OUTPUT_DIR="${MERGED_DIR}"
  fi

  WORKER_CMD="source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.cli feature extract --manifest-json ${GROUP_SHARD_MANIFEST_DIR}/shard_\${SLURM_ARRAY_TASK_ID}.json --dataset-root ${DATASET_ROOT} --extractor spectral --spectral-features ${FEATURES} --spectral-sv-top-k ${SV_TOP_K} --spectral-moment-source ${SPECTRAL_MOMENT_SOURCE} --spectral-qv-sum-mode ${GROUP_QV_MODE} --spectral-entrywise-delta-mode ${SPECTRAL_ENTRYWISE_DELTA_MODE} --stream-block-size ${STREAM_BLOCK_SIZE} --dtype ${DTYPE} --output-root ${GROUP_SHARD_OUTPUT_ROOT} --run-id shard_\${SLURM_ARRAY_TASK_ID}"

  MERGE_CMD="source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.utilities.merge_spectral_shards --manifest-json ${GROUP_MANIFEST_JSON} --dataset-root ${DATASET_ROOT} --output-dir ${EFFECTIVE_MERGED_OUTPUT_DIR} --pipeline-start-epoch-seconds ${PIPELINE_START_EPOCH_SECONDS} --shard-run-dir-glob '${GROUP_SHARD_OUTPUT_ROOT}/feature_extract/shard_*'"

  WORKER_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task "${SLURM_CPUS_PER_TASK}" \
    --array "0-${ARRAY_MAX}%${SLURM_MAX_CONCURRENT}" \
    --job-name "upeftguard_feature_extract_worker_${RUN_ID}_${GROUP_ID}" \
    --output "${SLURM_LOG_DIR}/feature_extract_worker_${RUN_ID}_${GROUP_ID}_%A_%a.out" \
    --error "${SLURM_LOG_DIR}/feature_extract_worker_${RUN_ID}_${GROUP_ID}_%A_%a.err" \
    --wrap "${WORKER_CMD}")

  MERGE_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task 4 \
    --dependency "afterok:${WORKER_JOB_ID}" \
    --job-name "upeftguard_feature_extract_merge_${RUN_ID}_${GROUP_ID}" \
    --output "${SLURM_LOG_DIR}/feature_extract_merge_${RUN_ID}_${GROUP_ID}_%j.out" \
    --error "${SLURM_LOG_DIR}/feature_extract_merge_${RUN_ID}_${GROUP_ID}_%j.err" \
    --wrap "${MERGE_CMD}")

  WORKER_JOB_IDS+=("${GROUP_ID}:${WORKER_JOB_ID}")
  MERGE_JOB_IDS+=("${GROUP_ID}:${MERGE_JOB_ID}")
  MERGE_DEPENDENCY_JOB_IDS+=("${MERGE_JOB_ID}")
  echo "Prepared ${GROUP_ID}: items=${GROUP_N_ITEMS}, shards=${GROUP_N_SHARDS}, q+v mode=${GROUP_QV_MODE}, feature output=${EFFECTIVE_MERGED_OUTPUT_DIR}"
done < <(
  python - <<PY
import json
from pathlib import Path

payload = json.loads(Path("${SCHEMA_REPORT_PATH}").read_text(encoding="utf-8"))
for group in payload.get("groups", []):
    print(
        "\t".join(
            [
                str(group["group_id"]),
                str(group["manifest_path"]),
                str(group["shard_manifest_dir"]),
                str(group["shard_output_root"]),
                str(group["merged_output_dir"]),
                str(group["effective_spectral_qv_sum_mode"]),
                str(int(group["n_shards"])),
                str(int(group["n_items"])),
            ]
        )
    )
PY
)

if [[ "${#WORKER_JOB_IDS[@]}" -eq 0 ]]; then
  echo "No schema groups were submitted." >&2
  exit 1
fi

if [[ "${GROUP_COUNT}" -gt 1 ]]; then
  DEPENDENCY=$(IFS=:; echo "${MERGE_DEPENDENCY_JOB_IDS[*]}")
  FINALIZE_CMD="source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -m upeftguard.utilities.finalize_schema_group_merge --schema-report-path ${SCHEMA_REPORT_PATH} --output-dir ${MERGED_DIR}"

  FINALIZE_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task 4 \
    --dependency "afterok:${DEPENDENCY}" \
    --job-name "upeftguard_feature_extract_finalize_${RUN_ID}" \
    --output "${SLURM_LOG_DIR}/feature_extract_finalize_${RUN_ID}_%j.out" \
    --error "${SLURM_LOG_DIR}/feature_extract_finalize_${RUN_ID}_%j.err" \
    --wrap "${FINALIZE_CMD}")
fi

echo "Run root: ${RUN_ROOT}"
echo "Schema partition report: ${SCHEMA_REPORT_PATH}"
echo "Schema groups: ${GROUP_COUNT}"
echo "Final feature output dir: ${MERGED_DIR}"
if [[ "${GROUP_COUNT}" -gt 1 ]]; then
  echo "Intermediate schema-group feature outputs: ${SCHEMA_GROUP_ROOT}/group_*/merged"
fi
echo "Requested spectral q+v mode: ${SPECTRAL_QV_SUM_MODE}"
echo "Pipeline start epoch seconds: ${PIPELINE_START_EPOCH_SECONDS}"
for job in "${WORKER_JOB_IDS[@]}"; do
  echo "Worker job id (${job%%:*}): ${job#*:}"
done
for job in "${MERGE_JOB_IDS[@]}"; do
  echo "Merge job id (${job%%:*}): ${job#*:}"
done
if [[ -n "${FINALIZE_JOB_ID}" ]]; then
  echo "Finalize job id: ${FINALIZE_JOB_ID}"
fi
