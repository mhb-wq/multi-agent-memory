param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("mem0", "amem")]
    [string]$Baseline,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $ProjectRoot
try {
    $Venv = Join-Path $ProjectRoot ".venvs/$Baseline"
    $PythonExe = Join-Path $Venv "Scripts/python.exe"
    if (!(Test-Path -LiteralPath $PythonExe)) {
        & $Python -m venv $Venv
        if ($LASTEXITCODE) { throw "Virtual environment creation failed" }
    }
    if ($Baseline -eq "mem0") {
        & $PythonExe -m pip install -e '.[mem0]' numpy sentence-transformers
        if ($LASTEXITCODE) { throw "Mem0 dependency installation failed" }
    } else {
        $Repo = Join-Path $ProjectRoot "vendor/AgenticMemory"
        $Revision = "b31a1dddf56f87ec762cce3539418602d7e2ffb1"
        if (!(Test-Path -LiteralPath (Join-Path $Repo ".git"))) {
            if (Test-Path -LiteralPath $Repo) { throw "Existing non-Git directory: $Repo" }
            git clone https://github.com/WujiangXu/AgenticMemory.git $Repo
            if ($LASTEXITCODE) { throw "A-MEM clone failed" }
            git -C $Repo checkout --detach $Revision
            if ($LASTEXITCODE) { throw "A-MEM checkout failed" }
        }
        $Current = git -C $Repo rev-parse HEAD
        if ($LASTEXITCODE -or $Current.Trim() -ne $Revision) {
            throw "A-MEM revision mismatch; preserve local edits and inspect manually"
        }
        & $PythonExe -m pip install -e '.[baselines,amem]'
        if ($LASTEXITCODE) { throw "A-MEM dependency installation failed" }
    }
} finally {
    Pop-Location
}
