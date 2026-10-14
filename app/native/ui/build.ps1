param(
    [ValidateSet('debug', 'release')]
    [string]$Profile = 'debug',
    [string]$Python = $env:MEGAGLITCH_PYTHON,
    [switch]$Run
)

$ErrorActionPreference = 'Stop'
$Cargo = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
if (-not (Test-Path $Cargo)) { throw "Cargo not found at $Cargo" }

$Arguments = @('build', '--locked', '--manifest-path', (Join-Path $PSScriptRoot 'Cargo.toml'))
if ($Profile -eq 'release') { $Arguments += '--release' }
& $Cargo @Arguments
if ($LASTEXITCODE -ne 0) { throw 'MegaGlitch Rust UI build failed.' }

if ($Run) {
    if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path -LiteralPath $Python)) {
        throw 'Pass -Python with the dependency-capable MegaGlitch Python executable.'
    }
    $env:MEGAGLITCH_PYTHON = (Resolve-Path -LiteralPath $Python).Path
    $RunArguments = @('run', '--locked', '--manifest-path', (Join-Path $PSScriptRoot 'Cargo.toml'))
    if ($Profile -eq 'release') { $RunArguments += '--release' }
    & $Cargo @RunArguments
}
