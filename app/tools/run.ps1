param(
    [switch]$Install,
    [switch]$BuildNative,
    [switch]$WithRust,
    [switch]$NoLaunch,
    [string]$Python = $env:MEGAGLITCH_PYTHON
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

if ([string]::IsNullOrWhiteSpace($Python)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { $Python = $PythonCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path $Python)) {
    throw 'Python was not found. Pass -Python or set MEGAGLITCH_PYTHON.'
}

Push-Location $ProjectRoot
try {
    & $Python -V
    if ($Install) {
        Write-Host 'Installing dependencies without replacing the external OpenCV build...'
        & $Python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    }

    if ($BuildNative) {
        $BuildArgs = @('-ExecutionPolicy', 'Bypass', '-File', 'tools\build_and_run.ps1', '-Python', $Python, '-NoLaunch')
        if ($WithRust) { $BuildArgs += '-WithRust' }
        & powershell @BuildArgs
        if ($LASTEXITCODE -ne 0) { throw 'Native build/diagnostics failed.' }
    }

    if (-not $NoLaunch) {
        & $Python main.py
        if ($LASTEXITCODE -ne 0) { throw "MegaGlitch exited with code $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}
