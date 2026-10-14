param(
    [switch]$Force,
    [string]$Python = $env:MEGAGLITCH_PYTHON
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
if ([string]::IsNullOrWhiteSpace($Python)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { $Python = $PythonCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path $Python)) {
    throw "Python was not found. Pass -Python or set MEGAGLITCH_PYTHON."
}
$BuildTemp = Join-Path $env:TEMP 'megaglitch-temporal-build'
$Arguments = @(
    (Join-Path $PSScriptRoot 'setup.py'),
    'build_ext', '--inplace', '--build-temp', $BuildTemp
)
if ($Force) { $Arguments += '--force' }

Push-Location $Root
try {
    & $Python -c "import struct,sys; assert sys.version_info[:2] == (3,12), 'MegaGlitch native builds require CPython 3.12'; assert struct.calcsize('P') == 8, '64-bit Python is required'; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) { throw "Python compatibility validation failed" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Native temporal build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
