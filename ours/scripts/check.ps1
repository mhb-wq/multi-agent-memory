$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $ProjectRoot
try {
    $env:NO_PROXY = "127.0.0.1,localhost"
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    & .venv/Scripts/python.exe -X utf8 -m ruff check src tests scripts
    if ($LASTEXITCODE) { throw "Lint failed" }
    & .venv/Scripts/python.exe -X utf8 -m ruff format --check src tests scripts
    if ($LASTEXITCODE) { throw "Format check failed" }
    & .venv/Scripts/python.exe -X utf8 -m pytest tests -q --junitxml=reports/tests.xml
    if ($LASTEXITCODE) { throw "Tests failed" }
} finally {
    Pop-Location
}
