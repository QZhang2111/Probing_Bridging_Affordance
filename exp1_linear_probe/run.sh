#!/usr/bin/env bash
# Convenience wrapper to run linear probing without changing working tree.

set -e
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$EXP_ROOT/umd_linear_probing/scripts/train.py" "$@"
