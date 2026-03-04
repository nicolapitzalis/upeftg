#!/bin/bash
#SBATCH --job-name=upeftguard_supervised_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/supervised_prepare_%j.out
#SBATCH --error=logs/supervised_prepare_%j.err
#SBATCH --partition=extra

set -euo pipefail

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

MANIFEST_JSON=${MANIFEST_JSON:-gmm_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-supervised_array_merged_${SLURM_JOB_ID:-manual}}
MODEL=${MODEL:-all}
FEATURES=${FEATURES:-"frobenius energy kurtosis l1_norm linf_norm sv_topk"}
SV_TOP_K=${SV_TOP_K:-8}
CV_FOLDS=${CV_FOLDS:-5}
CV_SEEDS=${CV_SEEDS:-"42 43 44"}
SCORE_PERCENTILES=${SCORE_PERCENTILES:-"50 60 70 80 90 95 99"}
MERGED_FEATURE_RUN_ID=${MERGED_FEATURE_RUN_ID:-feature_spectral_array_34017}
MERGED_FEATURE_DIR=${MERGED_FEATURE_DIR:-${OUTPUT_ROOT}/feature_extract/${MERGED_FEATURE_RUN_ID}/merged}
FEATURE_FILE=${FEATURE_FILE:-${MERGED_FEATURE_DIR}/spectral_features.npy}
FEATURE_MODEL_NAMES_FILE=${FEATURE_MODEL_NAMES_FILE:-${MERGED_FEATURE_DIR}/spectral_model_names.json}
FEATURE_METADATA_FILE=${FEATURE_METADATA_FILE:-${MERGED_FEATURE_DIR}/spectral_metadata.json}
SLURM_PARTITION=${SLURM_PARTITION:-extra}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-logs}

mkdir -p "${SLURM_LOG_DIR}"

for required_file in "${FEATURE_FILE}" "${FEATURE_MODEL_NAMES_FILE}" "${FEATURE_METADATA_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required merged feature artifact: ${required_file}" >&2
    exit 1
  fi
done

python -m upeftguard.cli run supervised \
  --stage prepare \
  --manifest-json "${MANIFEST_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --model "${MODEL}" \
  --features ${FEATURES} \
  --spectral-sv-top-k "${SV_TOP_K}" \
  --cv-folds "${CV_FOLDS}" \
  --cv-seeds ${CV_SEEDS} \
  --score-percentiles ${SCORE_PERCENTILES} \
  --feature-file "${FEATURE_FILE}" \
  --feature-model-names-file "${FEATURE_MODEL_NAMES_FILE}" \
  --feature-metadata-file "${FEATURE_METADATA_FILE}" \
  --tuning-executor slurm_array

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

WORKER_JOB_ID=$(sbatch \
  --parsable \
  --partition "${SLURM_PARTITION}" \
  --cpus-per-task "${SLURM_CPUS_PER_TASK}" \
  --array "0-${ARRAY_MAX}%${SLURM_MAX_CONCURRENT}" \
  --job-name "upeftguard_supervised_worker_${RUN_ID}" \
  --output "${SLURM_LOG_DIR}/supervised_worker_${RUN_ID}_%A_%a.out" \
  --error "${SLURM_LOG_DIR}/supervised_worker_${RUN_ID}_%A_%a.err" \
  --wrap "source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh && conda activate upeftg && python -m upeftguard.cli run supervised --stage worker --run-dir ${RUN_DIR} --task-index \${SLURM_ARRAY_TASK_ID} --n-jobs ${SLURM_CPUS_PER_TASK}")

FINALIZE_JOB_ID=$(sbatch \
  --parsable \
  --partition "${SLURM_PARTITION}" \
  --cpus-per-task 4 \
  --dependency "afterok:${WORKER_JOB_ID}" \
  --job-name "upeftguard_supervised_finalize_${RUN_ID}" \
  --output "${SLURM_LOG_DIR}/supervised_finalize_${RUN_ID}_%j.out" \
  --error "${SLURM_LOG_DIR}/supervised_finalize_${RUN_ID}_%j.err" \
  --wrap "source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh && conda activate upeftg && python -m upeftguard.cli run supervised --stage finalize --run-dir ${RUN_DIR} --score-percentiles ${SCORE_PERCENTILES}")

echo "Run dir: ${RUN_DIR}"
echo "Merged feature run id: ${MERGED_FEATURE_RUN_ID}"
echo "Merged feature dir: ${MERGED_FEATURE_DIR}"
echo "Feature file: ${FEATURE_FILE}"
echo "Model names file: ${FEATURE_MODEL_NAMES_FILE}"
echo "Metadata file: ${FEATURE_METADATA_FILE}"
echo "Score percentiles: ${SCORE_PERCENTILES}"
echo "Tuning tasks: ${N_TASKS}"
echo "Worker job id: ${WORKER_JOB_ID}"
echo "Finalize job id: ${FINALIZE_JOB_ID}"
