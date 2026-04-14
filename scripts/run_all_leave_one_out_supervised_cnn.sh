#!/bin/bash

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi

export ZERO_SHOT_MANIFEST_ROOT="${ZERO_SHOT_MANIFEST_ROOT:-${REPO_ROOT}/manifests/leave_one_out}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-leave_one_out_cnn}"
export SUITE_LABEL="${SUITE_LABEL:-Leave-one-out}"
export SUITE_LABEL_LOWER="${SUITE_LABEL_LOWER:-leave-one-out}"

exec bash "${REPO_ROOT}/scripts/run_all_zero_shot_supervised_cnn.sh" "$@"
