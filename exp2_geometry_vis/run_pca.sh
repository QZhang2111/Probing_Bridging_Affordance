#!/usr/bin/env bash
# Convenience wrapper for geometry visualization scripts.

set -e
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_ROOT/Section2_exp"

python scripts/extract_all.py "$@"
