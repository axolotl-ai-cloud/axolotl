<#
.SYNOPSIS
    Create and initialize a Python 3.11 virtual environment for Axolotl on Windows.
.DESCRIPTION
    - Requires Python 3.11 installed and discoverable via `py -3.11`.
    - Creates .venv311 at repo root if missing.
    - Upgrades essential build tooling.
    - Installs project in editable mode with dev + test dependencies.
    - Retries with a conservative NumPy pin if first install fails.
.NOTES
    Optional heavy deps (bitsandbytes, xformers, autoawq, etc.) are skipped on Windows per requirements.txt comments.
#>

param(
    [switch]$ForceRecreate,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$venvPath = Join-Path -Path (Get-Location) -ChildPath '.venv311'

function Write-Status($msg) { Write-Host "[setup_py311] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[setup_py311][WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[setup_py311][ERROR] $msg" -ForegroundColor Red }

Write-Status 'Checking Python 3.11 availability'
try {
    $pyVersion = & py -3.11 -c "import sys;print(sys.version)" 2>$null
} catch {
    Write-Err 'Python 3.11 not found. Install it from https://www.python.org/downloads/'
    exit 1
}
Write-Status "Found Python 3.11: $pyVersion"

if ((Test-Path $venvPath) -and $ForceRecreate) {
    Write-Warn 'Removing existing .venv311 due to -ForceRecreate'
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Status 'Creating virtual environment (.venv311)'
    & py -3.11 -m venv .venv311
}

$activate = ".venv311\\Scripts\\Activate.ps1"
if (-not (Test-Path $activate)) { Write-Err 'Activation script missing; venv creation failed.'; exit 1 }

Write-Status 'Activating virtual environment'
. $activate

Write-Status 'Upgrading pip/setuptools/wheel'
python -m pip install --upgrade pip setuptools wheel

Write-Status 'Installing base + dev dependencies (editable)'
$extras = 'dev'
if (-not $SkipTests) { $extras = 'dev,test' }

$cmd = "pip install -e .[$extras]"
Write-Status $cmd
if (-not (Invoke-Expression $cmd)) {
    Write-Warn 'Initial install failed; retry with conservative NumPy pin (1.26.4)'
    pip install "numpy==1.26.4"
    Invoke-Expression $cmd
}

Write-Status 'Verifying axolotl import'
try {
    python -c "import sys,importlib; m=importlib.import_module('axolotl'); print('axolotl version:', getattr(m,'__version__','unknown')); print('python:', sys.version)"
} catch {
    Write-Warn 'Axolotl import failed'
}

Write-Status 'Done. Activate later with:  . .venv311\Scripts\Activate.ps1'
