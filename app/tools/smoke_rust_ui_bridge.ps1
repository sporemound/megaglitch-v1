param(
    [string]$Python = $env:MEGAGLITCH_PYTHON,
    [ValidateRange(1, 120)]
    [int]$Seconds = 6,
    [ValidateRange(1, 50)]
    [int]$Cycles = 1
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Executable = (Resolve-Path (Join-Path $Root 'native\ui\target\release\megaglitch-ui.exe')).Path
if (-not $Executable.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Release executable resolved outside the repository.'
}
if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path -LiteralPath $Python)) {
    throw 'Pass -Python with the dependency-capable MegaGlitch Python executable.'
}

$env:MEGAGLITCH_PYTHON = (Resolve-Path -LiteralPath $Python).Path
for ($Cycle = 1; $Cycle -le $Cycles; $Cycle++) {
    $Process = Start-Process -FilePath $Executable -WorkingDirectory $Root -PassThru -WindowStyle Hidden
    $BridgePid = $null
    try {
        Start-Sleep -Seconds $Seconds
        if ($Process.HasExited) {
            throw "Rust UI exited early in cycle $Cycle with $($Process.ExitCode)."
        }
        $Children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $Process.Id }
        $Bridge = $Children | Where-Object {
            $_.Name -match '^python' -and $_.CommandLine -match 'rust_ui_bridge.py'
        } | Select-Object -First 1
        if (-not $Bridge) {
            throw "Rust UI did not own a Python bridge process in cycle $Cycle."
        }
        $BridgePid = $Bridge.ProcessId
        Write-Output "Cycle $Cycle/$Cycles Rust UI PID=$($Process.Id) bridge PID=$BridgePid"

        [void]$Process.CloseMainWindow()
        if (-not $Process.WaitForExit(5000)) {
            throw "Rust UI did not close cleanly within 5 seconds in cycle $Cycle."
        }
        Start-Sleep -Milliseconds 500
        if (Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $BridgePid }) {
            throw "Python bridge remained after Rust UI shutdown in cycle $Cycle."
        }
    }
    finally {
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force
        }
    }
}
Write-Output "Rust-owned bridge startup and orderly shutdown passed $Cycles cycle(s)."
