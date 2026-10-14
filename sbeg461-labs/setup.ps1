# Windows (PowerShell): create the course Python environment in .venv and install everything
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.11 -m venv .venv 2>$null; if (-not $?) { python -m venv .venv }
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e library
.venv\Scripts\python check_env.py
Write-Host "`nDone. Activate the environment with:  .venv\Scripts\Activate.ps1"
