param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $ProjectRoot
try {
    if (!(Test-Path .venv/Scripts/python.exe)) {
        & $Python -m venv .venv
        if ($LASTEXITCODE) { throw "venv creation failed" }
    }
    $Pin = (Get-Content upstream.lock.json -Raw | ConvertFrom-Json).commit
    if (!(Test-Path vendor/MemOS/.git) -and !(Test-Path vendor/MemOS/pyproject.toml)) {
        git clone https://github.com/MemTensor/MemOS.git vendor/MemOS
        if ($LASTEXITCODE) { throw "MemOS clone failed" }
        git -C vendor/MemOS checkout --detach $Pin
        if ($LASTEXITCODE) { throw "MemOS checkout failed" }
    }
    if (Test-Path vendor/MemOS/.git) {
        $Current = git -C vendor/MemOS rev-parse HEAD
        if ($Current.Trim() -ne $Pin) { throw "MemOS revision mismatch; preserve local edits and inspect manually" }
    } else {
        $Hashes = (Get-Content upstream.lock.json -Raw | ConvertFrom-Json).source_sha256
        foreach ($entry in $Hashes.PSObject.Properties) {
            # The lock's hashes were recorded with Windows CRLF checkout. Normalize
            # before hashing so an LF checkout verifies the same source correctly.
            $text = [IO.File]::ReadAllText((Join-Path (Get-Location) (Join-Path 'vendor/MemOS' $entry.Name)))
            $normalized = $text.Replace("`r`n", "`n").Replace("`n", "`r`n")
            $sha = [Security.Cryptography.SHA256]::Create()
            try { $actual = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($normalized))).Replace('-', '').ToLowerInvariant() }
            finally { $sha.Dispose() }
            if ($actual -ne $entry.Value) { throw "Packaged MemOS source checksum mismatch: $($entry.Name)" }
        }
    }
    & .venv/Scripts/python.exe -X utf8 -m pip install -e ./vendor/MemOS 'qdrant-client==1.20.0' 'sentence-transformers>=4.1,<5' -e '.[test,server]'
    if ($LASTEXITCODE) { throw "Dependency installation failed" }
    & .venv/Scripts/python.exe -X utf8 -m ours_memory.cli doctor
} finally {
    Pop-Location
}
