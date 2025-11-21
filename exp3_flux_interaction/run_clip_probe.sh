#!/usr/bin/env bash
# Wrapper to run CLIP patch probing.

set -e
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_ROOT/section4_probing"

python clip_patch_probe.py "$@"
