#!/usr/bin/env bash

set -euo pipefail

MANIFEST_PATH="${1:?manifest path required}"
PYTHON_BIN="${OPENSR_HPC_PYTHON:-python}"

if [[ -n "${OPENSR_HPC_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  IFS=',' read -r -a MODULE_LIST <<< "${OPENSR_HPC_MODULES}"
  for module_name in "${MODULE_LIST[@]}"; do
    module load "${module_name}"
  done
fi

if [[ -n "${OPENSR_HPC_CONDA_ENV:-}" ]]; then
  source activate "${OPENSR_HPC_CONDA_ENV}"
fi

exec "${PYTHON_BIN}" -m deployment.opensr_hpc.cli run task --manifest "${MANIFEST_PATH}"
