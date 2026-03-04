#!/bin/bash
#SBATCH --job-name=upeftguard_feature_spectral_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/feature_spectral_prepare_%j.out
#SBATCH --error=logs/feature_spectral_prepare_%j.err
#SBATCH --partition=extra

set -euo pipefail

source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh
conda activate upeftg

MANIFEST_JSON=${MANIFEST_JSON:-gmm_manifest.json}
DATASET_ROOT=${DATASET_ROOT:-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
RUN_ID=${RUN_ID:-feature_spectral_array_${SLURM_JOB_ID:-manual}}
FEATURES=${FEATURES:-"frobenius energy kurtosis l1_norm l2_norm linf_norm mean_abs sv_topk stable_rank spectral_entropy effective_rank"}
SV_TOP_K=${SV_TOP_K:-8}
STREAM_BLOCK_SIZE=${STREAM_BLOCK_SIZE:-131072}
DTYPE=${DTYPE:-float32}
N_SHARDS=${N_SHARDS:-8}
SLURM_PARTITION=${SLURM_PARTITION:-extra}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
DEFAULT_SLURM_MAX_CONCURRENT=$(python - <<PY
import os

for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
    value = os.getenv(key)
    if value and value.isdigit():
        print(max(1, int(value) - 1))
        break
else:
    print(1)
PY
)
SLURM_MAX_CONCURRENT=${SLURM_MAX_CONCURRENT:-${DEFAULT_SLURM_MAX_CONCURRENT}}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-logs}

mkdir -p "${SLURM_LOG_DIR}"

RUN_ROOT=$(python - <<PY
from pathlib import Path
print((Path("${OUTPUT_ROOT}").expanduser().resolve() / "feature_extract" / "${RUN_ID}").as_posix())
PY
)
SHARD_MANIFEST_DIR="${RUN_ROOT}/shard_manifests"
SHARD_OUTPUT_ROOT="${RUN_ROOT}/shards"
MERGED_DIR="${RUN_ROOT}/merged"

mkdir -p "${SHARD_MANIFEST_DIR}" "${SHARD_OUTPUT_ROOT}" "${MERGED_DIR}"

N_SHARDS_RESOLVED=$(python - <<PY
import json
from pathlib import Path

from upeftguard.utilities.manifest import parse_single_manifest_json

manifest_json = Path("${MANIFEST_JSON}").expanduser().resolve()
dataset_root = Path("${DATASET_ROOT}").expanduser().resolve()
out_dir = Path("${SHARD_MANIFEST_DIR}").expanduser().resolve()
n_shards = int("${N_SHARDS}")
if n_shards <= 0:
    raise ValueError(f"N_SHARDS must be positive, got {n_shards}")

items = parse_single_manifest_json(
    manifest_path=manifest_json,
    dataset_root=dataset_root,
    section_key="path",
)
if not items:
    raise ValueError("Manifest resolved to zero items")

n_items = len(items)
n_resolved = min(n_shards, n_items)
chunk = (n_items + n_resolved - 1) // n_resolved

for shard in range(n_resolved):
    start = shard * chunk
    end = min(start + chunk, n_items)
    entries = [str(item.adapter_path.resolve()) for item in items[start:end]]
    if not entries:
        continue
    payload = {"path": entries}
    path = out_dir / f"shard_{shard}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(n_resolved)
PY
)

if [[ "${N_SHARDS_RESOLVED}" -le 0 ]]; then
  echo "No shard manifests were created" >&2
  exit 1
fi

ARRAY_MAX=$((N_SHARDS_RESOLVED - 1))

WORKER_JOB_ID=$(sbatch \
  --parsable \
  --partition "${SLURM_PARTITION}" \
  --cpus-per-task "${SLURM_CPUS_PER_TASK}" \
  --array "0-${ARRAY_MAX}%${SLURM_MAX_CONCURRENT}" \
  --job-name "upeftguard_feature_spectral_worker_${RUN_ID}" \
  --output "${SLURM_LOG_DIR}/feature_spectral_worker_${RUN_ID}_%A_%a.out" \
  --error "${SLURM_LOG_DIR}/feature_spectral_worker_${RUN_ID}_%A_%a.err" \
  --wrap "source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh && conda activate upeftg && python -m upeftguard.cli feature extract --manifest-json ${SHARD_MANIFEST_DIR}/shard_\${SLURM_ARRAY_TASK_ID}.json --dataset-root ${DATASET_ROOT} --extractor spectral --spectral-features ${FEATURES} --spectral-sv-top-k ${SV_TOP_K} --stream-block-size ${STREAM_BLOCK_SIZE} --dtype ${DTYPE} --output-root ${SHARD_OUTPUT_ROOT} --run-id shard_\${SLURM_ARRAY_TASK_ID}")

MERGE_JOB_ID=$(sbatch \
  --parsable \
  --partition "${SLURM_PARTITION}" \
  --cpus-per-task 4 \
  --dependency "afterok:${WORKER_JOB_ID}" \
  --job-name "upeftguard_feature_spectral_merge_${RUN_ID}" \
  --output "${SLURM_LOG_DIR}/feature_spectral_merge_${RUN_ID}_%j.out" \
  --error "${SLURM_LOG_DIR}/feature_spectral_merge_${RUN_ID}_%j.err" \
  --wrap "source /home/n.pitzalis/miniconda3/etc/profile.d/conda.sh && conda activate upeftg && python -m upeftguard.utilities.merge_spectral_shards --manifest-json ${MANIFEST_JSON} --dataset-root ${DATASET_ROOT} --output-dir ${MERGED_DIR} --shard-run-dir-glob '${SHARD_OUTPUT_ROOT}/feature_extract/shard_*'")

echo "Run root: ${RUN_ROOT}"
echo "Shard manifests: ${SHARD_MANIFEST_DIR}"
echo "Shard output root: ${SHARD_OUTPUT_ROOT}"
echo "Merged output dir: ${MERGED_DIR}"
echo "Shards: ${N_SHARDS_RESOLVED}"
echo "Worker job id: ${WORKER_JOB_ID}"
echo "Merge job id: ${MERGE_JOB_ID}"
