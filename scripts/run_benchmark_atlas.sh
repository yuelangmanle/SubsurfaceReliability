#!/usr/bin/env bash
# Run verification gates and execute the 24-regime benchmark scan.
set -eu

cd "$(dirname "$0")/.."
export PYTHONPATH=src

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo "=== Running Solver and Field Verification Suite ==="
"$PY" -m subrel.verify

echo "=== Running 24-Regime Benchmark Scan ==="
"$PY" -m subrel.scan
