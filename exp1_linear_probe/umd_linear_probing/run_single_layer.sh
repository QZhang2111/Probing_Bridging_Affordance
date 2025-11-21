#!/usr/bin/env bash

# Minimal sequential runner for single_layer experiments (clip, dino, sam)
# Usage: bash run_single_layer.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python
TRAIN_SCRIPT="$ROOT_DIR/scripts/train.py"

declare -A BASE_CFG=(
  [clip]="$ROOT_DIR/configs/openclip.yaml"
  [dino]="$ROOT_DIR/configs/dino.yaml"
  [sam]="$ROOT_DIR/configs/sam.yaml"
)

MODELS=(dino)

echo "Starting single_layer sequential runs..."

for MODEL in "${MODELS[@]}"; do
  # Guard against unset keys under `set -u`
  if [[ -z "${BASE_CFG[$MODEL]+x}" ]]; then
    echo "[WARN] No base config mapping for model: $MODEL (skipping)" >&2
    continue
  fi
  DEFAULTS_CFG="${BASE_CFG[$MODEL]}"
  LOCAL_DIR="$ROOT_DIR/configs/single_layer/$MODEL"

  if [[ ! -f "$DEFAULTS_CFG" ]]; then
    echo "[WARN] Base config not found for $MODEL: $DEFAULTS_CFG (skipping)" >&2
    continue
  fi
  if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "[WARN] Local override dir not found for $MODEL: $LOCAL_DIR (skipping)" >&2
    continue
  fi

  echo "=== Model: $MODEL"; echo "Defaults: $DEFAULTS_CFG"; echo "Local dir: $LOCAL_DIR"; echo

  # Preferred explicit order
  ORDER=(
    2_both 2_depth 2_normal
    5_both 5_depth 5_normal
    8_both 8_depth 8_normal
    11_both 11_depth 11_normal
  )

  declare -A SEEN=()

  for NAME in "${ORDER[@]}"; do
    FILE="$LOCAL_DIR/$NAME.yaml"
    if [[ -f "$FILE" ]]; then
      echo ">>> Running $MODEL :: $NAME"
      "$PY" "$TRAIN_SCRIPT" --defaults "$DEFAULTS_CFG" --local "$FILE"
      SEEN["$FILE"]=1
    fi
  done

  # Run any other YAMLs present in the folder (sorted), if not already run
  shopt -s nullglob
  mapfile -t OTHERS < <(ls -1 "$LOCAL_DIR"/*.yaml | sort)
  shopt -u nullglob
  for FILE in "${OTHERS[@]:-}"; do
    if [[ -n "${SEEN[$FILE]:-}" ]]; then
      continue
    fi
    echo ">>> Running $MODEL :: $(basename "$FILE")"
    "$PY" "$TRAIN_SCRIPT" --defaults "$DEFAULTS_CFG" --local "$FILE"
  done
done

echo "All single_layer runs completed."
