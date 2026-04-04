#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name a10-py311 --display-name "Python (A10)"

echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
echo "Open the notebook and select kernel: Python (A10)"
