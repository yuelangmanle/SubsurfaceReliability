#!/usr/bin/env bash
# Verify benchmark integrity, run unit tests, print summary tables,
# and regenerate all seven publication figures from archived evaluation records.
set -eu

cd "$(dirname "$0")/.."
export PYTHONPATH=src

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo "=== 1. Verification Suite & Unit Tests ==="
"$PY" -m subrel.verify
"$PY" src/subrel/tests/test_blind_descriptors.py

echo "=== 2. Extended Validation Summary ==="
"$PY" scripts/summarize_extended_validation.py

echo "=== 3. Benchmark Summary Tables & Screening AUCs ==="
"$PY" -m subrel.analysis

echo "=== 4. Regenerating Publication Figures (Figures 1-7) ==="
"$PY" -m subrel.fig1
"$PY" -m subrel.fig2
"$PY" -m subrel.fig3
"$PY" -m subrel.fig4
"$PY" -m subrel.fig5
"$PY" -m subrel.fig6
"$PY" -m subrel.fig7
echo "All verification checks, summary tables, and figures completed."
