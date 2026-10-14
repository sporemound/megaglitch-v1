param(
    [string]$Python = $env:MEGAGLITCH_PYTHON,
    [switch]$WithRust,
    [switch]$SkipNativeBuild,
    [switch]$ReleaseGate,
    [string]$ArtifactRoot = "",
    [string]$DistRoot = "",
    [string]$WorkRoot = ""
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')

function Resolve-BuildPath {
    param(
        [string]$Configured,
        [Parameter(Mandatory = $true)]
        [string]$Fallback
    )
    $candidate = if ([string]::IsNullOrWhiteSpace($Configured)) { $Fallback } else { $Configured }
    if (-not [IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $Root $candidate
    }
    return [IO.Path]::GetFullPath($candidate)
}

$resolvedArtifactRoot = ""
if (-not [string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $resolvedArtifactRoot = Resolve-BuildPath -Configured $ArtifactRoot -Fallback $ArtifactRoot
}
$defaultDistRoot = if ($resolvedArtifactRoot) { Join-Path $resolvedArtifactRoot 'dist' } else { Join-Path $Root 'dist' }
$defaultWorkRoot = if ($resolvedArtifactRoot) { Join-Path $resolvedArtifactRoot 'pyinstaller-work' } else { Join-Path $Root 'build' }
$resolvedDistRoot = Resolve-BuildPath -Configured $DistRoot -Fallback $defaultDistRoot
$resolvedWorkRoot = Resolve-BuildPath -Configured $WorkRoot -Fallback $defaultWorkRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { $Python = $PythonCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path $Python)) {
    throw 'Python was not found. Pass -Python or set MEGAGLITCH_PYTHON.'
}

Push-Location $Root
try {
    if ($ReleaseGate -and -not $WithRust) {
        throw '-ReleaseGate requires -WithRust.'
    }
    if ($ReleaseGate -and $SkipNativeBuild) {
        throw '-ReleaseGate cannot be combined with -SkipNativeBuild.'
    }
    & $Python -c "import PyInstaller; print(PyInstaller.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller is not installed. Run: python -m pip install pyinstaller'
    }
    if (-not $SkipNativeBuild) {
        $BuildArgs = @('-ExecutionPolicy', 'Bypass', '-File', 'tools\build_and_run.ps1', '-Python', $Python, '-NoLaunch')
        if ($WithRust) { $BuildArgs += '-WithRust' }
        if ($ReleaseGate) { $BuildArgs += '-ReleaseGate' }
        if ($resolvedArtifactRoot) { $BuildArgs += @('-ArtifactRoot', $resolvedArtifactRoot) }
        & powershell @BuildArgs
        if ($LASTEXITCODE -ne 0) { throw 'Native build/diagnostics failed.' }
    }
    New-Item -ItemType Directory -Path $resolvedDistRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $resolvedWorkRoot -Force | Out-Null
    & $Python -m PyInstaller --clean --noconfirm `
        --distpath $resolvedDistRoot --workpath $resolvedWorkRoot `
        'packaging\pyinstaller\megaglitch.spec'
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    $bundleName = if ([string]::IsNullOrWhiteSpace($env:MEGAGLITCH_BUNDLE_NAME)) {
        'MegaGlitchNode'
    } else {
        $env:MEGAGLITCH_BUNDLE_NAME
    }
    $Executable = Join-Path (Join-Path $resolvedDistRoot $bundleName) 'MegaGlitchNode.exe'
    if ($ReleaseGate) {
        if (-not (Test-Path $Executable)) { throw "Frozen executable not found: $Executable" }
        $env:MEGAGLITCH_RUNTIME_BACKEND = 'python'
        $env:MEGAGLITCH_FRAME_TRANSPORT = 'shared'
        & $Executable --diagnose-only
        if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime diagnostics failed.' }
        & $Executable --diagnose-worker
        if ($LASTEXITCODE -ne 0) { throw 'Frozen worker diagnostics failed.' }
    }
    Write-Host "Bundle created at $Executable"
} finally {
    Pop-Location
}
