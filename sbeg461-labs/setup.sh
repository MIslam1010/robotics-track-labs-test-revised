#!/bin/bash
# macOS / Linux: create the course Python environment in .venv and install everything
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.11 || command -v python3)
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e library
.venv/bin/python check_env.py
echo; echo "Done. Activate the environment with:  source .venv/bin/activate"
