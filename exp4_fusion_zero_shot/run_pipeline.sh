#!/usr/bin/env bash
# Wrapper to run the zero-shot fusion pipeline.

set -e
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_ROOT/zero_shot"

python run_knife_affordance_pipeline.py "$@"
