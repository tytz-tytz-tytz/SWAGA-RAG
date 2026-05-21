[CmdletBinding()]
param(
    [string]$EnvDir = "venv",
    [switch]$Recreate,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path $RepoRoot $EnvDir
$EnvPython = Join-Path $EnvPath "Scripts\python.exe"
$WorkspaceTmp = Join-Path $RepoRoot ".tmp"

New-Item -ItemType Directory -Force -Path $WorkspaceTmp | Out-Null
$env:TEMP = $WorkspaceTmp
$env:TMP = $WorkspaceTmp

function Invoke-Checked {
    param(
        [ScriptBlock]$Command,
        [string]$Description
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-PreferredPython {
    $directPython = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
    if (Test-Path $directPython) {
        return $directPython
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -notmatch "WindowsApps") {
        $version = & $pythonCmd.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version -eq "3.12") {
            return $pythonCmd.Source
        }
    }

    throw @"
Python 3.12 not found in a usable location.
Install CPython 3.12 and re-run this script.
Avoid the Windows Store alias from WindowsApps: it creates broken venv launchers on this machine.
"@
}

$BasePython = Get-PreferredPython
Write-Host "Base Python: $BasePython"

if ($Recreate -and (Test-Path $EnvPath)) {
    Remove-Item -Recurse -Force $EnvPath
}

if (-not (Test-Path $EnvPython)) {
    Invoke-Checked -Description "Creating virtual environment" -Command { & $BasePython -m venv $EnvPath }
}

if (-not (Test-Path $EnvPython)) {
    throw "Virtual environment was not created: $EnvPython"
}

Invoke-Checked -Description "Checking Python in virtual environment" -Command { & $EnvPython --version }
Invoke-Checked -Description "Checking pip in virtual environment" -Command { & $EnvPython -m pip --version }

if (-not $SkipInstall) {
    Invoke-Checked -Description "Installing project dependencies" -Command { & $EnvPython -m pip install -e ".[dev]" }
}

Write-Host ""
Write-Host "Use the environment via:"
Write-Host "  $EnvPython"
Write-Host "Example:"
Write-Host "  $EnvPython scripts/run_queries_swaga.py --config configs/ablations/stable_baseline.json"

