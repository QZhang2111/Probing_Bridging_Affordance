#!/usr/bin/env bash

# Run siglip2 linear probing once per layer listed in the base model config,
# overriding the hooks/output dir to isolate each feature map.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
DEFAULT_CFG="${PROJECT_ROOT}/configs/default.yaml"
BASE_LOCAL_CFG="${PROJECT_ROOT}/configs/local_siglip2.yaml"
MODEL_CFG="${PROJECT_ROOT}/configs/models/siglip2.yaml"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  log "Unable to locate training entrypoint at ${TRAIN_SCRIPT}"
  exit 1
fi
if [[ ! -f "${DEFAULT_CFG}" ]]; then
  log "Unable to locate default config at ${DEFAULT_CFG}"
  exit 1
fi
if [[ ! -f "${BASE_LOCAL_CFG}" ]]; then
  log "Unable to locate siglip2 local config at ${BASE_LOCAL_CFG}"
  exit 1
fi
if [[ ! -f "${MODEL_CFG}" ]]; then
  log "Unable to locate siglip2 model config at ${MODEL_CFG}"
  exit 1
fi

DATASET_ROOT="$("${PYTHON_BIN}" - "${BASE_LOCAL_CFG}" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
dataset = data.get("dataset") or {}
print(dataset.get("root", ""), end="")
PY
)"

if [[ -z "${DATASET_ROOT}" ]]; then
  log "Dataset root is not specified in ${BASE_LOCAL_CFG}"
  exit 1
fi

layers_raw="$("${PYTHON_BIN}" - "${MODEL_CFG}" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
layers = data.get("layers_to_hook") or []
print(" ".join(str(layer) for layer in layers), end="")
PY
)"

if [[ -z "${layers_raw}" ]]; then
  log "No layers_to_hook entries found in ${MODEL_CFG}"
  exit 1
fi

read -r -a LAYERS <<<"${layers_raw}"

tmp_configs=()
cleanup() {
  local path
  for path in "${tmp_configs[@]:-}"; do
    [[ -f "${path}" ]] && rm -f "${path}"
  done
}
trap cleanup EXIT

log "Preparing to run ${#LAYERS[@]} siglip2 probes (one per layer)."

for layer in "${LAYERS[@]}"; do
  if [[ -z "${layer}" ]]; then
    continue
  fi
  if ! [[ "${layer}" =~ ^[0-9]+$ ]]; then
    log "Skipping non-integer layer value '${layer}'."
    continue
  fi

  layer_label=$(printf "%02d" "${layer}")
  output_dir_name="siglip2_layer${layer_label}"
  tmp_config="$(mktemp "${TMPDIR:-/tmp}/siglip2_layer_${layer_label}_XXXX.yaml")"
  tmp_configs+=("${tmp_config}")

  cat >"${tmp_config}" <<EOF
dataset:
  root: ${DATASET_ROOT}

model:
  target: siglip2
  config_path: configs/models/siglip2.yaml
  overrides:
    layers_to_hook: [${layer}]
    output_dir_name: ${output_dir_name}
    head:
      feature_keys: [${layer}]
      primary_key: ${layer}
EOF

  log "Running siglip2 layer ${layer} -> output ${output_dir_name}"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --defaults "${DEFAULT_CFG}" \
    --local "${tmp_config}"
done

log "Completed siglip2 single-layer sweep."
