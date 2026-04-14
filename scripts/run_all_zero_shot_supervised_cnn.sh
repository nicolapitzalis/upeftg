#!/bin/bash

set -euo pipefail

usage() {
  cat <<EOF
Usage:
  bash scripts/run_all_zero_shot_supervised_cnn.sh --hyperparam_config <RUN_ID_OR_RUN_DIR> [options]

Required:
  --hyperparam_config, --hyperparam-config
      Supervised reference run used to load tuning.winner params.
      Accepts either a run id under runs/supervised/ or an explicit run directory path.

Optional:
  --manifest_filter, --manifest-filter <TEXT>
      Only process ${SUITE_LABEL_LOWER} manifests whose path contains TEXT.
  --dry_run, --dry-run
      Print what would run without submitting jobs.
  --help, -h
      Show this help text.

Environment overrides such as OUTPUT_ROOT, DATASET_ROOT, ZERO_SHOT_MANIFEST_ROOT,
RUN_ID_PREFIX, SLURM_PARTITION, CONDA_SH, CONDA_ENV, and SKIP_FEATURE_IMPORTANCE
are still supported.
EOF
}

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi

if [[ ! -f "${REPO_ROOT}/upeftguard/cli.py" ]]; then
  echo "Could not resolve repository root: ${REPO_ROOT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

ZERO_SHOT_MANIFEST_ROOT=${ZERO_SHOT_MANIFEST_ROOT:-${REPO_ROOT}/manifests/zero_shots}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs}
TRAIN_SPLIT=${TRAIN_SPLIT:-100}
DRY_RUN=${DRY_RUN:-0}
MANIFEST_FILTER=${MANIFEST_FILTER:-}
HYPERPARAM_CONFIG=${HYPERPARAM_CONFIG:-}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-}
SUITE_LABEL=${SUITE_LABEL:-Zero-shot}
SUITE_LABEL_LOWER=${SUITE_LABEL_LOWER:-zero-shot}

CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-upeftg}
SLURM_PARTITION=${SLURM_PARTITION:-extra}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-${REPO_ROOT}/logs}
SKIP_FEATURE_IMPORTANCE=${SKIP_FEATURE_IMPORTANCE:-0}
SCORE_PERCENTILES=${SCORE_PERCENTILES:-}

CURRENT_USER=${USER:-$(id -un)}
PROJECT_STORAGE_ROOT=${UPEFTGUARD_STORAGE_ROOT:-/models/${CURRENT_USER}/unsupervised-peftguard}
DATASET_ROOT=${DATASET_ROOT:-${UPEFTGUARD_DATA_ROOT:-${PROJECT_STORAGE_ROOT}/data}}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --hyperparam_config|--hyperparam-config)
      if [[ "$#" -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      HYPERPARAM_CONFIG="$2"
      shift 2
      ;;
    --manifest_filter|--manifest-filter)
      if [[ "$#" -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      MANIFEST_FILTER="$2"
      shift 2
      ;;
    --dry_run|--dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${HYPERPARAM_CONFIG}" ]]; then
  echo "--hyperparam_config is required." >&2
  usage >&2
  exit 1
fi

if [[ "${HYPERPARAM_CONFIG}" == /* || "${HYPERPARAM_CONFIG}" == *"/"* ]]; then
  REFERENCE_RUN_DIR="$(
    HYPERPARAM_CONFIG_VALUE="${HYPERPARAM_CONFIG}" python - <<'PY'
from pathlib import Path
import os
spec = Path(os.environ["HYPERPARAM_CONFIG_VALUE"]).expanduser()
resolved = spec if spec.is_absolute() else (Path.cwd().resolve() / spec)
print(resolved.resolve())
PY
  )"
  if [[ ! -d "${REFERENCE_RUN_DIR}" ]]; then
    echo "Reference run directory not found: ${REFERENCE_RUN_DIR}" >&2
    exit 1
  fi
  REFERENCE_RUN_ID="$(basename "${REFERENCE_RUN_DIR}")"
else
  REFERENCE_RUN_ID="${HYPERPARAM_CONFIG}"
  REFERENCE_RUN_DIR="${REPO_ROOT}/runs/supervised/${REFERENCE_RUN_ID}"
fi

REFERENCE_RUN_CONFIG="${REFERENCE_RUN_DIR}/run_config.json"
REFERENCE_REPORT="${REFERENCE_RUN_DIR}/reports/supervised_report.json"
REFERENCE_TUNING_MANIFEST="${REFERENCE_RUN_DIR}/reports/tuning_manifest.json"

if [[ ! -d "${ZERO_SHOT_MANIFEST_ROOT}" ]]; then
  echo "${SUITE_LABEL} manifest root not found: ${ZERO_SHOT_MANIFEST_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_RUN_CONFIG}" ]]; then
  echo "Reference run_config not found: ${REFERENCE_RUN_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_REPORT}" ]]; then
  echo "Reference supervised_report not found: ${REFERENCE_REPORT}" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_TUNING_MANIFEST}" ]]; then
  echo "Reference tuning_manifest not found: ${REFERENCE_TUNING_MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

eval "$(
  REFERENCE_RUN_CONFIG="${REFERENCE_RUN_CONFIG}" \
  REFERENCE_REPORT="${REFERENCE_REPORT}" \
  REFERENCE_TUNING_MANIFEST="${REFERENCE_TUNING_MANIFEST}" \
  python - <<'PY'
import json
import os
import shlex
from pathlib import Path

run_config = json.loads(Path(os.environ["REFERENCE_RUN_CONFIG"]).read_text(encoding="utf-8"))
report = json.loads(Path(os.environ["REFERENCE_REPORT"]).read_text(encoding="utf-8"))
tuning_manifest = json.loads(Path(os.environ["REFERENCE_TUNING_MANIFEST"]).read_text(encoding="utf-8"))

winner = report.get("tuning", {}).get("winner")
if not isinstance(winner, dict):
    raise SystemExit("Reference supervised_report.json is missing tuning.winner")
winner_params = winner.get("params")
if not isinstance(winner_params, dict):
    raise SystemExit("Reference winner is missing params")

extractor = tuning_manifest.get("extractor", {})
extractor_params = extractor.get("params", {})
extractor_metadata = extractor.get("metadata", {})
threshold_selection = tuning_manifest.get("threshold_selection", {})
tuning = tuning_manifest.get("tuning", {})

feature_file = extractor_metadata.get("external_feature_source")
if not isinstance(feature_file, str) or not feature_file:
    raise SystemExit("Reference tuning manifest is missing extractor.metadata.external_feature_source")

spectral_features = extractor_params.get("spectral_features")
if not isinstance(spectral_features, list) or not spectral_features:
    raise SystemExit("Reference tuning manifest is missing extractor.params.spectral_features")

accepted_fprs = threshold_selection.get("accepted_fprs")
if not isinstance(accepted_fprs, list):
    accepted_fprs = []

assignments = {
    "REFERENCE_MODEL_DEFAULT": str(winner.get("model_name", "cnn_1d")),
    "REFERENCE_WINNER_TASK_INDEX_DEFAULT": str(int(winner.get("task_index", 0))),
    "REFERENCE_WINNER_PARAMS_JSON_DEFAULT": json.dumps(winner_params, separators=(",", ":")),
    "FEATURE_FILE_DEFAULT": feature_file,
    "FEATURES_DEFAULT": " ".join(str(x) for x in spectral_features),
    "SV_TOP_K_DEFAULT": str(int(extractor_params.get("spectral_sv_top_k", 8))),
    "SPECTRAL_MOMENT_SOURCE_DEFAULT": str(extractor_params.get("spectral_moment_source", "both")),
    "SPECTRAL_QV_SUM_MODE_DEFAULT": str(extractor_params.get("spectral_qv_sum_mode", "append")),
    "SPECTRAL_ENTRYWISE_DELTA_MODE_DEFAULT": str(extractor_params.get("spectral_entrywise_delta_mode", "dense")),
    "CV_FOLDS_DEFAULT": str(int(tuning.get("cv_folds_requested", 5))),
    "CV_SEEDS_DEFAULT": " ".join(str(int(x)) for x in tuning.get("cv_random_states", [42])),
    "CALIBRATION_SPLIT_DEFAULT": str(int(threshold_selection.get("calibration_split_percent", 20))),
    "ACCEPTED_FPR_DEFAULT": " ".join(str(float(x)) for x in accepted_fprs),
    "SPLIT_BY_FOLDER_DEFAULT": "1" if bool(threshold_selection.get("split_by_folder", False)) else "0",
}

for key, value in assignments.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

FEATURE_FILE=${FEATURE_FILE:-${FEATURE_FILE_DEFAULT}}
FEATURES=${FEATURES:-${FEATURES_DEFAULT}}
MODEL=${MODEL:-${REFERENCE_MODEL_DEFAULT}}
SV_TOP_K=${SV_TOP_K:-${SV_TOP_K_DEFAULT}}
SPECTRAL_MOMENT_SOURCE=${SPECTRAL_MOMENT_SOURCE:-${SPECTRAL_MOMENT_SOURCE_DEFAULT}}
SPECTRAL_QV_SUM_MODE=${SPECTRAL_QV_SUM_MODE:-${SPECTRAL_QV_SUM_MODE_DEFAULT}}
SPECTRAL_ENTRYWISE_DELTA_MODE=${SPECTRAL_ENTRYWISE_DELTA_MODE:-${SPECTRAL_ENTRYWISE_DELTA_MODE_DEFAULT}}
CV_FOLDS=${CV_FOLDS:-${CV_FOLDS_DEFAULT}}
CV_SEEDS=${CV_SEEDS:-${CV_SEEDS_DEFAULT}}
CALIBRATION_SPLIT=${CALIBRATION_SPLIT:-${CALIBRATION_SPLIT_DEFAULT}}
ACCEPTED_FPR=${ACCEPTED_FPR:-${ACCEPTED_FPR_DEFAULT}}
SPLIT_BY_FOLDER=${SPLIT_BY_FOLDER:-${SPLIT_BY_FOLDER_DEFAULT}}
REFERENCE_WINNER_TASK_INDEX=${REFERENCE_WINNER_TASK_INDEX:-${REFERENCE_WINNER_TASK_INDEX_DEFAULT}}
REFERENCE_WINNER_PARAMS_JSON=${REFERENCE_WINNER_PARAMS_JSON:-${REFERENCE_WINNER_PARAMS_JSON_DEFAULT}}

mapfile -t MANIFESTS < <(find "${ZERO_SHOT_MANIFEST_ROOT}" -type f -name "*.json" | sort)
if [[ -n "${MANIFEST_FILTER}" ]]; then
  FILTERED_MANIFESTS=()
  for manifest in "${MANIFESTS[@]}"; do
    if [[ "${manifest}" == *"${MANIFEST_FILTER}"* ]]; then
      FILTERED_MANIFESTS+=("${manifest}")
    fi
  done
  MANIFESTS=("${FILTERED_MANIFESTS[@]}")
fi

if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
  echo "No ${SUITE_LABEL_LOWER} manifests found under ${ZERO_SHOT_MANIFEST_ROOT}" >&2
  exit 1
fi

mkdir -p "${SLURM_LOG_DIR}"

read -r -a FEATURE_VALUES <<< "${FEATURES}"
if [[ "${#FEATURE_VALUES[@]}" -eq 0 ]]; then
  echo "FEATURES must include at least one feature group." >&2
  exit 1
fi

read -r -a CV_SEED_VALUES <<< "${CV_SEEDS}"
if [[ "${#CV_SEED_VALUES[@]}" -eq 0 ]]; then
  echo "CV_SEEDS must include at least one value." >&2
  exit 1
fi

read -r -a ACCEPTED_FPR_VALUES <<< "${ACCEPTED_FPR}"
if [[ "${#ACCEPTED_FPR_VALUES[@]}" -eq 0 ]]; then
  echo "ACCEPTED_FPR must include at least one value." >&2
  exit 1
fi

SPLIT_BY_FOLDER_ARGS=()
if [[ "${SPLIT_BY_FOLDER}" == "1" || "${SPLIT_BY_FOLDER,,}" == "true" ]]; then
  SPLIT_BY_FOLDER_ARGS+=(--split-by-folder)
fi

FINALIZE_EXTRA_ARGS=()
if [[ -n "${SCORE_PERCENTILES}" ]]; then
  read -r -a SCORE_PERCENTILE_VALUES <<< "${SCORE_PERCENTILES}"
  FINALIZE_EXTRA_ARGS+=(--score-percentiles "${SCORE_PERCENTILE_VALUES[@]}")
fi
if [[ "${SKIP_FEATURE_IMPORTANCE}" == "1" || "${SKIP_FEATURE_IMPORTANCE,,}" == "true" ]]; then
  FINALIZE_EXTRA_ARGS+=(--skip-feature-importance)
fi

echo "Repository root: ${REPO_ROOT}"
echo "${SUITE_LABEL} manifest root: ${ZERO_SHOT_MANIFEST_ROOT}"
echo "Reference run: ${REFERENCE_RUN_ID}"
echo "Reference feature file: ${FEATURE_FILE}"
echo "Reference winner task_index: ${REFERENCE_WINNER_TASK_INDEX}"
echo "Model: ${MODEL}"
echo "Features: ${FEATURES}"
echo "SV_TOP_K: ${SV_TOP_K}"
echo "SPECTRAL_MOMENT_SOURCE: ${SPECTRAL_MOMENT_SOURCE}"
echo "SPECTRAL_QV_SUM_MODE: ${SPECTRAL_QV_SUM_MODE}"
echo "SPECTRAL_ENTRYWISE_DELTA_MODE: ${SPECTRAL_ENTRYWISE_DELTA_MODE}"
echo "CV_FOLDS: ${CV_FOLDS}"
echo "CV_SEEDS: ${CV_SEEDS}"
echo "TRAIN_SPLIT: ${TRAIN_SPLIT}"
echo "CALIBRATION_SPLIT: ${CALIBRATION_SPLIT}"
echo "ACCEPTED_FPR: ${ACCEPTED_FPR}"
echo "SPLIT_BY_FOLDER: ${SPLIT_BY_FOLDER}"
echo "Manifest count: ${#MANIFESTS[@]}"
if [[ -n "${MANIFEST_FILTER}" ]]; then
  echo "Manifest filter: ${MANIFEST_FILTER}"
fi
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "DRY_RUN: enabled"
else
  echo "DRY_RUN: disabled"
fi

submitted=0
for idx in "${!MANIFESTS[@]}"; do
  manifest="${MANIFESTS[$idx]}"
  base_run_id="$(basename "${manifest}" .json)"
  run_id="${base_run_id}"
  if [[ -n "${RUN_ID_PREFIX}" ]]; then
    run_id="${RUN_ID_PREFIX}/${base_run_id}"
  fi
  slurm_safe_run_id="${run_id//\//__}"
  rel_manifest="${manifest#${REPO_ROOT}/}"

  printf '[%d/%d] %s <- %s\n' "$((idx + 1))" "${#MANIFESTS[@]}" "${base_run_id}" "${rel_manifest}"

  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
    echo "  prepare: python -m upeftguard.cli run supervised --stage prepare ..."
    echo "  freeze:  reference winner from ${REFERENCE_RUN_ID}"
    echo "  worker:  sbatch single-task worker"
    echo "  finalize: sbatch dependent finalize"
    continue
  fi

  PREPARE_ARGS=(
    python -m upeftguard.cli run supervised
    --stage prepare
    --manifest-json "${manifest}"
    --dataset-root "${DATASET_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --run-id "${run_id}"
    --model "${MODEL}"
    --features "${FEATURE_VALUES[@]}"
    --spectral-sv-top-k "${SV_TOP_K}"
    --spectral-moment-source "${SPECTRAL_MOMENT_SOURCE}"
    --spectral-qv-sum-mode "${SPECTRAL_QV_SUM_MODE}"
    --spectral-entrywise-delta-mode "${SPECTRAL_ENTRYWISE_DELTA_MODE}"
    --cv-folds "${CV_FOLDS}"
    --cv-seeds "${CV_SEED_VALUES[@]}"
    --train-split "${TRAIN_SPLIT}"
    --calibration-split "${CALIBRATION_SPLIT}"
    --accepted-fpr "${ACCEPTED_FPR_VALUES[@]}"
    --feature-file "${FEATURE_FILE}"
    --tuning-executor slurm_array
  )
  if [[ "${#SPLIT_BY_FOLDER_ARGS[@]}" -gt 0 ]]; then
    PREPARE_ARGS+=("${SPLIT_BY_FOLDER_ARGS[@]}")
  fi

  "${PREPARE_ARGS[@]}"

  RUN_DIR="$(
    OUTPUT_ROOT_VALUE="${OUTPUT_ROOT}" RUN_ID_VALUE="${run_id}" python - <<'PY'
from pathlib import Path
import os
print((Path(os.environ["OUTPUT_ROOT_VALUE"]).expanduser().resolve() / "supervised" / os.environ["RUN_ID_VALUE"]).as_posix())
PY
  )"
  TUNING_MANIFEST_PATH="${RUN_DIR}/reports/tuning_manifest.json"

  TARGET_TUNING_MANIFEST="${TUNING_MANIFEST_PATH}" \
  REFERENCE_RUN_ID_VALUE="${REFERENCE_RUN_ID}" \
  REFERENCE_WINNER_TASK_INDEX_VALUE="${REFERENCE_WINNER_TASK_INDEX}" \
  REFERENCE_WINNER_PARAMS_JSON_VALUE="${REFERENCE_WINNER_PARAMS_JSON}" \
  python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["TARGET_TUNING_MANIFEST"])
payload = json.loads(path.read_text(encoding="utf-8"))
winner_params = json.loads(os.environ["REFERENCE_WINNER_PARAMS_JSON_VALUE"])
reference_run_id = os.environ["REFERENCE_RUN_ID_VALUE"]
reference_task_index = int(os.environ["REFERENCE_WINNER_TASK_INDEX_VALUE"])

tasks = payload.get("tuning", {}).get("tasks", [])
matches = [
    task for task in tasks
    if isinstance(task, dict)
    and str(task.get("model_name")) == "cnn_1d"
    and task.get("params") == winner_params
]
if len(matches) != 1:
    raise SystemExit(
        f"Expected exactly one matching cnn_1d task for reference winner params in {path}, found {len(matches)}"
    )

selected = dict(matches[0])
selected["task_index"] = 0
payload["tuning"]["tasks"] = [selected]

cv_split_groups = payload.get("tuning", {}).get("cv_split_groups")
if isinstance(cv_split_groups, list):
    total_splits = sum(
        len(group.get("cv_splits", []))
        for group in cv_split_groups
        if isinstance(group, dict)
    )
else:
    total_splits = len(payload.get("tuning", {}).get("cv_splits", []))
payload["tuning"]["estimated_total_fits"] = int(max(1, total_splits))
payload["tuning"]["fixed_reference_winner"] = {
    "source_run_id": reference_run_id,
    "source_task_index": reference_task_index,
    "model_name": "cnn_1d",
    "params": winner_params,
}

warnings = [
    str(row)
    for row in payload.get("warnings", [])
    if not str(row).startswith("Large supervised grid search:")
]
warnings.append(
    "Fixed supervised tuning to the cnn_1d winner from "
    f"{reference_run_id} (source task_index={reference_task_index})"
)
payload["warnings"] = warnings

path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(
    "Locked tuning manifest to reference winner:",
    json.dumps(payload["tuning"]["tasks"][0], sort_keys=True),
)
PY

  read -r DEFAULT_SLURM_CPUS_PER_TASK DEFAULT_SCORE_PERCENTILES <<< "$(
    TARGET_TUNING_MANIFEST="${TUNING_MANIFEST_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["TARGET_TUNING_MANIFEST"]).read_text(encoding="utf-8"))
runtime = payload.get("runtime", {})
score_percentiles = runtime.get("score_percentiles", [])
joined = ",".join(str(float(x)) for x in score_percentiles)
print(int(runtime.get("slurm_cpus_per_task", 4)), joined)
PY
  )"
  DEFAULT_SLURM_CPUS_PER_TASK=${DEFAULT_SLURM_CPUS_PER_TASK:-4}
  WORKER_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK_OVERRIDE:-${DEFAULT_SLURM_CPUS_PER_TASK}}

  FINALIZE_ARGS_STRING=""
  if [[ "${#FINALIZE_EXTRA_ARGS[@]}" -gt 0 ]]; then
    FINALIZE_ARGS_STRING=" ${FINALIZE_EXTRA_ARGS[*]}"
  elif [[ -n "${DEFAULT_SCORE_PERCENTILES}" ]]; then
    FINALIZE_ARGS_STRING=" --score-percentiles ${DEFAULT_SCORE_PERCENTILES//,/ }"
  fi

  WORKER_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task "${WORKER_CPUS_PER_TASK}" \
    --job-name "upeftguard_supervised_worker_${slurm_safe_run_id}" \
    --output "${SLURM_LOG_DIR}/supervised_worker_${slurm_safe_run_id}_%j.out" \
    --error "${SLURM_LOG_DIR}/supervised_worker_${slurm_safe_run_id}_%j.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && cd ${REPO_ROOT} && python -m upeftguard.cli run supervised --stage worker --run-dir ${RUN_DIR} --task-index 0 --n-jobs ${WORKER_CPUS_PER_TASK}")

  FINALIZE_JOB_ID=$(sbatch \
    --parsable \
    --partition "${SLURM_PARTITION}" \
    --cpus-per-task 4 \
    --dependency "afterok:${WORKER_JOB_ID}" \
    --job-name "upeftguard_supervised_finalize_${slurm_safe_run_id}" \
    --output "${SLURM_LOG_DIR}/supervised_finalize_${slurm_safe_run_id}_%j.out" \
    --error "${SLURM_LOG_DIR}/supervised_finalize_${slurm_safe_run_id}_%j.err" \
    --wrap "source ${CONDA_SH} && conda activate ${CONDA_ENV} && cd ${REPO_ROOT} && python -m upeftguard.cli run supervised --stage finalize --run-dir ${RUN_DIR}${FINALIZE_ARGS_STRING}")

  echo "  prepared run_dir ${RUN_DIR}"
  echo "  locked to reference winner from ${REFERENCE_RUN_ID}"
  echo "  worker job id: ${WORKER_JOB_ID}"
  echo "  finalize job id: ${FINALIZE_JOB_ID}"
  submitted=$((submitted + 1))
done

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "Dry run complete: ${#MANIFESTS[@]} manifest(s) enumerated."
else
  echo "Submission complete: ${submitted} ${SUITE_LABEL_LOWER} job(s) submitted."
fi
