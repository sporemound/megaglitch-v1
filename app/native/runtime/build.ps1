param(
    [switch]$Debug,
    [string]$Cargo = $env:MEGAGLITCH_CARGO
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
if ([string]::IsNullOrWhiteSpace($Cargo)) {
    $CargoCommand = Get-Command cargo -ErrorAction SilentlyContinue
    if ($CargoCommand) { $Cargo = $CargoCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($Cargo)) {
    $Cargo = Join-Path $HOME '.cargo\bin\cargo.exe'
}
$Target = Join-Path $env:TEMP 'megaglitch-rust-target'
$Destination = Join-Path $Root 'megaglitch\_native'
if (-not (Test-Path $Cargo)) { throw "Cargo was not found at $Cargo" }

Push-Location $PSScriptRoot
try {
    $env:CARGO_TARGET_DIR = $Target
    if ($Debug) { & $Cargo build } else { & $Cargo build --release }
    if ($LASTEXITCODE -ne 0) { throw "Rust runtime build failed with exit code $LASTEXITCODE" }
    $Profile = if ($Debug) { 'debug' } else { 'release' }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Force (Join-Path $Target "$Profile\megaglitch_runtime.dll") (Join-Path $Destination 'megaglitch_runtime.dll')
} finally {
    Pop-Location
}
