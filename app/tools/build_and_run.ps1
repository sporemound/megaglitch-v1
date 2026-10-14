param(
    [string]$Python = $env:MEGAGLITCH_PYTHON,
    [switch]$WithRust,
    [switch]$Force,
    [switch]$NoLaunch,
    [switch]$SkipDiagnostics,
    [switch]$ReleaseGate,
    [string]$ArtifactRoot = ""
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$resolvedArtifactRoot = ""
if (-not [string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $candidateArtifactRoot = $ArtifactRoot
    if (-not [IO.Path]::IsPathRooted($candidateArtifactRoot)) {
        $candidateArtifactRoot = Join-Path $ProjectRoot $candidateArtifactRoot
    }
    $resolvedArtifactRoot = [IO.Path]::GetFullPath($candidateArtifactRoot)
    New-Item -ItemType Directory -Path $resolvedArtifactRoot -Force | Out-Null
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { $Python = $PythonCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path $Python)) {
    throw 'Python was not found. Pass -Python or set MEGAGLITCH_PYTHON.'
}

Push-Location $ProjectRoot
try {
    if ($ReleaseGate -and -not $WithRust) {
        throw '-ReleaseGate requires -WithRust.'
    }
    if ($ReleaseGate -and $SkipDiagnostics) {
        throw '-ReleaseGate cannot be combined with -SkipDiagnostics.'
    }
    Write-Host '=== MegaGlitch environment ==='
    & $Python -c "import platform,struct,sys; print(sys.executable); print(sys.version); print(platform.platform()); assert sys.version_info[:2] == (3,12); assert struct.calcsize('P') == 8"
    if ($LASTEXITCODE -ne 0) { throw 'MegaGlitch requires 64-bit CPython 3.12.' }

    Write-Host '=== Building native temporal/P-frame extension ==='
    $TemporalArgs = @('-ExecutionPolicy', 'Bypass', '-File', 'native\temporal_ring\build.ps1', '-Python', $Python)
    if ($Force) { $TemporalArgs += '-Force' }
    & powershell @TemporalArgs
    if ($LASTEXITCODE -ne 0) { throw 'Native temporal build failed.' }

    if ($WithRust) {
        Write-Host '=== Building Rust graph IR runtime ==='
        & powershell -ExecutionPolicy Bypass -File 'native\runtime\build.ps1'
        if ($LASTEXITCODE -ne 0) { throw 'Rust runtime build failed.' }
    }

    if (-not $SkipDiagnostics) {
        Write-Host '=== Running runtime diagnostics ==='
        $diagnosticOutput = if ($resolvedArtifactRoot) {
            $diagnosticDirectory = Join-Path $resolvedArtifactRoot 'diagnostics'
            New-Item -ItemType Directory -Path $diagnosticDirectory -Force | Out-Null
            Join-Path $diagnosticDirectory 'build.json'
        } else {
            'tools\diagnostics\build.json'
        }
        $DiagnosticArgs = @('tools\diagnose_runtime.py', '--json', $diagnosticOutput, '--require-temporal')
        if ($WithRust) { $DiagnosticArgs += '--require-rust' }
        if ($ReleaseGate) { $DiagnosticArgs += @('--require-cuda', '--cuda-selftest') }
        & $Python @DiagnosticArgs
        if ($LASTEXITCODE -ne 0) { throw 'Runtime diagnostics failed.' }
    }

    if ($ReleaseGate) {
        Write-Host '=== Running release regression gate ==='
        $releaseArtifactBase = if ($resolvedArtifactRoot) {
            $resolvedArtifactRoot
        } else {
            Join-Path $ProjectRoot 'build'
        }
        $ReleaseTestTemp = Join-Path $releaseArtifactBase ('pytest-release-gate-' + [Guid]::NewGuid().ToString('N'))
        $ReleaseResults = Join-Path $releaseArtifactBase ('release-benchmarks-' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $ReleaseTestTemp | Out-Null
        New-Item -ItemType Directory -Path $ReleaseResults | Out-Null
        & $Python -m pytest -q --basetemp $ReleaseTestTemp
        if ($LASTEXITCODE -ne 0) { throw 'Automated test suite failed.' }

        Write-Host '=== Measuring current release sources ==='
        & $Python 'tools\benchmark\run_pframe_hotstage.py' --label 'pframe-native-segment' --warmup 3 --iterations 20 --output-dir $ReleaseResults
        if ($LASTEXITCODE -ne 0) { throw 'Current P-frame hot-stage benchmark failed.' }
        & $Python 'tools\benchmark\run_shared_transport.py' --label 'shared-transport-pframe-segment' --warmup 5 --iterations 30 --output-dir $ReleaseResults
        if ($LASTEXITCODE -ne 0) { throw 'Current shared-transport benchmark failed.' }
        & $Python 'tools\benchmark\run_worker_lifecycle.py' --label 'worker-lifecycle' --cycles 3 --output-dir $ReleaseResults
        if ($LASTEXITCODE -ne 0) { throw 'Current worker-lifecycle benchmark failed.' }
        & $Python 'tools\benchmark\run_native_runtime.py' --label 'native-runtime-graph-ir-v1' --warmup 3 --iterations 15 --output-dir $ReleaseResults
        if ($LASTEXITCODE -ne 0) { throw 'Current Rust graph-IR benchmark failed.' }
        & $Python 'tools\benchmark\check_regressions.py' --results-dir $ReleaseResults --rust-performance-advisory --json (Join-Path $ReleaseResults 'regression-check.json')
        if ($LASTEXITCODE -ne 0) { throw 'Fresh benchmark regression gate failed.' }
        & $Python 'tools\benchmark\run_render_quality_path.py' `
            --resolution '640x360' `
            --frame-count 12 `
            --source-fps '30000/1001' `
            --output-fps 55 `
            --semantics 'true_accelerated' `
            --seed 994 `
            --json-output (Join-Path $ReleaseResults 'render-quality-master-path.json') `
            --csv-output (Join-Path $ReleaseResults 'render-quality-master-path.csv')
        if ($LASTEXITCODE -ne 0) { throw 'Render-quality master-path benchmark failed.' }
        & $Python 'test_pipeline.py'
        if ($LASTEXITCODE -ne 0) { throw 'Pipeline parity smoke test failed.' }
        & $Python 'scripts\genesis_vdp_smoke_test.py'
        if ($LASTEXITCODE -ne 0) { throw 'Genesis VDP smoke test failed.' }
    }

    if (-not $NoLaunch) {
        Write-Host '=== Launching MegaGlitch ==='
        & $Python 'main.py'
        if ($LASTEXITCODE -ne 0) { throw "MegaGlitch exited with code $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}
