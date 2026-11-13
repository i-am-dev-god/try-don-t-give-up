#!/usr/bin/env bash
# Create the Python environment for the from-scratch ML experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet numpy matplotlib
echo "done. run: .venv/bin/python from_scratch/check_grad.py"
