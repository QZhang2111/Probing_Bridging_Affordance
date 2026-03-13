#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/3] exp1 launcher"
python run.py geometry-train -- --help >/dev/null

echo "[2/3] exp3 launcher"
python run.py interaction-probe -- --help >/dev/null

echo "[3/3] exp4 launcher"
python run.py fusion-eval -- --help >/dev/null

echo "Smoke test passed: all three main experiment launchers are callable."
