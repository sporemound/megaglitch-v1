[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$PackageRoot,

    [ValidateRange(1, 300)]
    [int]$TimeoutSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedProtocolVersion = 2
$process = $null
$processStarted = $false
$stdoutTask = $null
$stderrTask = $null
$reportPath = $null
$reportOwned = $false

function Get-RequiredProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Context is missing required property '$Name'."
    }
    return $property.Value
}

function Get-RequiredInteger {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Context,
        [long]$Minimum = 0
    )

    $value = Get-RequiredProperty -Object $Object -Name $Name -Context $Context
    $isInteger = (
        $value -is [byte] -or
        $value -is [sbyte] -or
        $value -is [int16] -or
        $value -is [uint16] -or
        $value -is [int32] -or
        $value -is [uint32] -or
        $value -is [int64]
    )
    if (-not $isInteger) {
        throw "$Context property '$Name' must be an integer."
    }
    $converted = [Convert]::ToInt64($value, [Globalization.CultureInfo]::InvariantCulture)
    if ($converted -lt $Minimum) {
        throw "$Context property '$Name' must be at least $Minimum; received $converted."
    }
    return $converted
}

function Get-RequiredBoolean {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $value = Get-RequiredProperty -Object $Object -Name $Name -Context $Context
    if ($value -isnot [bool]) {
        throw "$Context property '$Name' must be a boolean."
    }
    return $value
}

function Get-RequiredString {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $value = Get-RequiredProperty -Object $Object -Name $Name -Context $Context
    if ($value -isnot [string] -or [string]::IsNullOrWhiteSpace($value)) {
        throw "$Context property '$Name' must be a non-empty string."
    }
    return $value
}

function Multiply-Checked {
    param(
        [long]$Left,
        [long]$Right,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    if ($Left -lt 0 -or $Right -lt 0) {
        throw "$Context cannot multiply negative integers."
    }
    if ($Left -ne 0 -and $Right -gt ([long]::MaxValue / $Left)) {
        throw "$Context exceeds the supported 64-bit range."
    }
    return [long]($Left * $Right)
}

try {
    $resolvedPackageRoot = (Resolve-Path -LiteralPath $PackageRoot -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $resolvedPackageRoot -PathType Container)) {
        throw "PackageRoot must be a directory: $resolvedPackageRoot"
    }

    $packagedUi = Join-Path $resolvedPackageRoot "MegaGlitchNode-Mycotroph-Functional-Preview.exe"
    $unrenamedUi = Join-Path $resolvedPackageRoot "megaglitch-ui.exe"
    if (Test-Path -LiteralPath $packagedUi -PathType Leaf) {
        $uiExecutable = $packagedUi
    }
    elseif (Test-Path -LiteralPath $unrenamedUi -PathType Leaf) {
        $uiExecutable = $unrenamedUi
    }
    else {
        throw "PackageRoot has no packaged MegaGlitch Rust host executable: $resolvedPackageRoot"
    }

    $engineBundle = Join-Path $resolvedPackageRoot "engine-v2"
    $engineExecutable = Join-Path $engineBundle "megaglitch-engine-v2.exe"
    $engineInternal = Join-Path $engineBundle "_internal"
    if (-not (Test-Path -LiteralPath $engineExecutable -PathType Leaf)) {
        throw "Packaged Rust host has no adjacent Engine V2 executable: $engineExecutable"
    }
    if (-not (Test-Path -LiteralPath $engineInternal -PathType Container)) {
        throw "Packaged Rust host has an incomplete Engine V2 onedir: $engineInternal"
    }

    $systemDirectory = [Environment]::SystemDirectory
    $windowsDirectory = Split-Path -Path $systemDirectory -Parent
    $systemPathCandidates = @(
        $systemDirectory,
        $windowsDirectory,
        (Join-Path $systemDirectory "Wbem"),
        (Join-Path $systemDirectory "WindowsPowerShell\v1.0")
    )
    $sanitizedPath = (($systemPathCandidates | Where-Object {
        Test-Path -LiteralPath $_ -PathType Container
    } | Select-Object -Unique) -join ";")
    if ([string]::IsNullOrWhiteSpace($sanitizedPath)) {
        throw "Could not construct a Windows-system-only PATH for the packaged-host gate."
    }

    $reportDirectory = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $reportLeaf = "megaglitch-host-gate-$PID-$([Guid]::NewGuid().ToString('N')).json"
    $reportPath = Join-Path $reportDirectory $reportLeaf
    $reportStream = [IO.File]::Open(
        $reportPath,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    $reportStream.Dispose()
    $reportOwned = $true

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $uiExecutable
    $quote = [char]34
    $startInfo.Arguments = "--verify-engine-v2 " + $quote + $reportPath + $quote
    $startInfo.WorkingDirectory = $resolvedPackageRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [Text.Encoding]::UTF8
    $startInfo.EnvironmentVariables["PATH"] = $sanitizedPath

    $inheritedEnvironmentNames = @(
        $startInfo.EnvironmentVariables.Keys |
            ForEach-Object { [string]$_ }
    )
    foreach ($name in $inheritedEnvironmentNames) {
        if (
            $name -match "^(PYTHON|CONDA)" -or
            $name -match "^VIRTUAL_ENV$" -or
            $name -match "^MEGAGLITCH_(PYTHON|UI_BRIDGE|OPENCV|CUDA)" -or
            $name -match "^OPENCV" -or
            $name -match "^CUDA_PATH"
        ) {
            [void]$startInfo.EnvironmentVariables.Remove($name)
        }
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Windows declined to start the packaged MegaGlitch Rust host."
    }
    $processStarted = $true
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    $timeoutMilliseconds = [int]($TimeoutSeconds * 1000)
    if (-not $process.WaitForExit($timeoutMilliseconds)) {
        try {
            $process.Kill()
            $process.WaitForExit()
        }
        catch {
            # The finally block retries a bounded, exact-process cleanup.
        }
        throw "Packaged Rust host did not finish its Engine V2 smoke within $TimeoutSeconds seconds."
    }
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $exitCode = $process.ExitCode

    if (-not [string]::IsNullOrEmpty($stdout)) {
        throw "Packaged Rust host wrote to stdout during its headless smoke; the host contract requires silence."
    }
    if (-not [string]::IsNullOrEmpty($stderr)) {
        throw "Packaged Rust host wrote to stderr during its headless smoke; the host contract requires silence."
    }
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "Packaged Rust host did not leave its requested smoke report."
    }
    $reportText = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($reportText)) {
        throw "Packaged Rust host left an empty smoke report."
    }
    try {
        $report = $reportText | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Packaged Rust host left invalid smoke-report JSON: $($_.Exception.Message)"
    }
    if (
        $null -eq $report -or
        $report -is [System.Array] -or
        $report -is [string] -or
        $report -is [ValueType]
    ) {
        throw "Packaged Rust host smoke report must be one JSON object."
    }

    $passed = Get-RequiredBoolean -Object $report -Name "passed" -Context "Host smoke report"
    if ($exitCode -ne 0) {
        $failure = $report.PSObject.Properties["error"]
        if ($null -ne $failure -and $failure.Value -is [string] -and -not [string]::IsNullOrWhiteSpace($failure.Value)) {
            throw ("Packaged Rust host smoke failed with exit code {0}: {1}" -f $exitCode, $failure.Value)
        }
        throw "Packaged Rust host smoke failed with exit code $exitCode."
    }
    if (-not $passed) {
        throw "Packaged Rust host returned exit code zero but did not mark its smoke report passed."
    }

    $protocolVersion = Get-RequiredInteger -Object $report -Name "protocol_version" -Context "Host smoke report"
    if ($protocolVersion -ne $expectedProtocolVersion) {
        throw "Packaged Rust host used Engine protocol $protocolVersion; expected $expectedProtocolVersion."
    }
    $backend = Get-RequiredString -Object $report -Name "backend" -Context "Host smoke report"
    if ($backend.Trim().ToLowerInvariant() -eq "offline") {
        throw "Packaged Rust host reported an offline backend after a passing smoke."
    }
    $graphAccepted = Get-RequiredBoolean -Object $report -Name "graph_accepted" -Context "Host smoke report"
    if (-not $graphAccepted) {
        throw "Packaged Rust host did not report an accepted graph."
    }

    $stateRevision = Get-RequiredInteger -Object $report -Name "state_revision" -Context "Host smoke report" -Minimum 1
    $graphCommandId = Get-RequiredInteger -Object $report -Name "graph_command_id" -Context "Host smoke report" -Minimum 1
    $lastAckCommandId = Get-RequiredInteger -Object $report -Name "last_ack_command_id" -Context "Host smoke report" -Minimum 1
    if ($lastAckCommandId -ne $graphCommandId) {
        throw "Packaged Rust host accepted command $lastAckCommandId instead of submitted graph command $graphCommandId."
    }
    $lastEffectiveSequence = Get-RequiredInteger -Object $report -Name "last_effective_frame_sequence" -Context "Host smoke report" -Minimum 1
    $graphGeneration = Get-RequiredInteger -Object $report -Name "graph_generation" -Context "Host smoke report" -Minimum 1
    $graphSequence = Get-RequiredInteger -Object $report -Name "graph_sequence" -Context "Host smoke report" -Minimum 1
    $initialPreviewSequence = Get-RequiredInteger -Object $report -Name "initial_preview_sequence" -Context "Host smoke report" -Minimum 1
    $previewSequence = Get-RequiredInteger -Object $report -Name "preview_sequence" -Context "Host smoke report" -Minimum 1
    if ($previewSequence -le $initialPreviewSequence) {
        throw "Packaged Rust host preview did not advance after graph submission."
    }
    if ($previewSequence -le $lastEffectiveSequence) {
        throw "Packaged Rust host did not observe a frame produced after graph acknowledgement."
    }

    $previewWidth = Get-RequiredInteger -Object $report -Name "preview_width" -Context "Host smoke report" -Minimum 1
    $previewHeight = Get-RequiredInteger -Object $report -Name "preview_height" -Context "Host smoke report" -Minimum 1
    $previewBytes = Get-RequiredInteger -Object $report -Name "preview_bytes" -Context "Host smoke report" -Minimum 1
    $expectedPixels = Multiply-Checked -Left $previewWidth -Right $previewHeight -Context "Host RGB preview pixel count"
    $expectedBytes = Multiply-Checked -Left $expectedPixels -Right 3 -Context "Host RGB preview byte count"
    if ($previewBytes -ne $expectedBytes) {
        throw "Packaged Rust host reported $previewBytes RGB bytes for $($previewWidth)x$($previewHeight); expected $expectedBytes."
    }

    [pscustomobject]@{
        Passed = $true
        PackageRoot = $resolvedPackageRoot
        UiExecutable = $uiExecutable
        EngineExecutable = $engineExecutable
        ProtocolVersion = $protocolVersion
        Backend = $backend
        StateRevision = $stateRevision
        GraphGeneration = $graphGeneration
        GraphSequence = $graphSequence
        PreviewSequence = $previewSequence
        PreviewDimensions = ("{0}x{1}" -f $previewWidth, $previewHeight)
        PreviewBytes = $previewBytes
        ChildOutputSilent = $true
    }
}
finally {
    if ($processStarted -and $null -ne $process) {
        try {
            if (-not $process.HasExited) {
                $process.Kill()
                $process.WaitForExit()
            }
        }
        catch {
            # Never enumerate or terminate by image name; cleanup is limited to
            # the exact Process instance started by this gate.
        }
        $process.Dispose()
    }
    if ($reportOwned -and -not [string]::IsNullOrWhiteSpace($reportPath)) {
        $expectedDirectory = [IO.Path]::GetDirectoryName(
            (Join-Path ([IO.Path]::GetTempPath()) ".megaglitch-host-gate-directory-sentinel")
        )
        $resolvedReportPath = [IO.Path]::GetFullPath($reportPath)
        if (
            [IO.Path]::GetDirectoryName($resolvedReportPath) -eq $expectedDirectory -and
            [IO.Path]::GetFileName($resolvedReportPath) -match ("^megaglitch-host-gate-" + [regex]::Escape([string]$PID) + "-[0-9a-f]{32}\.json$")
        ) {
            if (Test-Path -LiteralPath $resolvedReportPath -PathType Leaf) {
                Remove-Item -LiteralPath $resolvedReportPath -Force
            }
        }
        else {
            throw "Refusing to clean an unexpected host-smoke report path: $resolvedReportPath"
        }
    }
}
