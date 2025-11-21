#!/usr/bin/env bash
# Wrapper to run Flux/SD cross-attention probe.

set -e
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_ROOT/section4_probing"

python cross_attention_probe.py "$@"
