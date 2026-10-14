[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$EngineBundle,

    [ValidateRange(1, 300)]
    [int]$StartupTimeoutSeconds = 45,

    [ValidateRange(1, 120)]
    [int]$FrameTimeoutSeconds = 15,

    [ValidateRange(1, 120)]
    [int]$ShutdownTimeoutSeconds = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$protocolVersion = 2
$graphCommandId = 1
$stopCommandId = 2
$shutdownCommandId = 3
$process = $null
$processStarted = $false
$stderrTask = $null
$surfacePath = $null
$surfaceFileStream = $null
$mappedFile = $null
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

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

function ConvertTo-ProtocolInteger {
    param(
        [AllowNull()]
        [object]$Value,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Context,
        [long]$Minimum = 0
    )

    $isInteger = (
        $Value -is [byte] -or
        $Value -is [sbyte] -or
        $Value -is [int16] -or
        $Value -is [uint16] -or
        $Value -is [int32] -or
        $Value -is [uint32] -or
        $Value -is [int64]
    )
    if (-not $isInteger) {
        throw "$Context property '$Name' must be an integer."
    }
    $converted = [Convert]::ToInt64($Value, [Globalization.CultureInfo]::InvariantCulture)
    if ($converted -lt $Minimum) {
        throw "$Context property '$Name' must be at least $Minimum; received $converted."
    }
    return $converted
}

function Get-ProtocolInteger {
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
    return ConvertTo-ProtocolInteger -Value $value -Name $Name -Context $Context -Minimum $Minimum
}

function Get-ProtocolString {
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

function Multiply-ProtocolInteger {
    param(
        [long]$Left,
        [long]$Right,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    if ($Left -lt 0 -or $Right -lt 0) {
        throw "$Context cannot multiply negative protocol integers."
    }
    if ($Left -ne 0 -and $Right -gt ([long]::MaxValue / $Left)) {
        throw "$Context exceeds the supported 64-bit range."
    }
    return [long]($Left * $Right)
}

function Add-ProtocolInteger {
    param(
        [long]$Left,
        [long]$Right,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    if ($Left -lt 0 -or $Right -lt 0) {
        throw "$Context cannot add negative protocol integers."
    }
    if ($Right -gt ([long]::MaxValue - $Left)) {
        throw "$Context exceeds the supported 64-bit range."
    }
    return [long]($Left + $Right)
}

function Assert-ProtocolHeader {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Event,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $version = Get-ProtocolInteger -Object $Event -Name "version" -Context $Context
    if ($version -ne $protocolVersion) {
        throw "$Context uses protocol version $version; expected $protocolVersion."
    }
    [void](Get-ProtocolString -Object $Event -Name "type" -Context $Context)
}

function Read-ProtocolEvent {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.StreamReader]$Reader,
        [Parameter(Mandatory = $true)]
        [datetime]$DeadlineUtc,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $remainingMilliseconds = [int][Math]::Ceiling(($DeadlineUtc - [datetime]::UtcNow).TotalMilliseconds)
    if ($remainingMilliseconds -le 0) {
        throw "Timed out while waiting for $Context."
    }

    $readTask = $Reader.ReadLineAsync()
    if (-not $readTask.Wait($remainingMilliseconds)) {
        throw "Timed out while waiting for $Context."
    }
    $line = $readTask.GetAwaiter().GetResult()
    if ($null -eq $line) {
        throw "Engine stdout closed while waiting for $Context."
    }
    if ([string]::IsNullOrWhiteSpace($line)) {
        throw "Engine emitted a non-JSON stdout line while waiting for $Context."
    }

    try {
        $event = $line | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        $displayLine = if ($line.Length -gt 240) { $line.Substring(0, 240) + "..." } else { $line }
        throw "Engine emitted non-JSON stdout while waiting for ${Context}: $displayLine"
    }
    if ($null -eq $event -or $event -is [System.Array] -or $event -is [string] -or $event -is [ValueType]) {
        throw "Engine emitted JSON that is not a protocol object while waiting for $Context."
    }
    Assert-ProtocolHeader -Event $event -Context "Engine stdout event"
    return $event
}

function Assert-RemainingJsonOutput {
    param(
        [AllowEmptyString()]
        [string]$Text
    )

    if ([string]::IsNullOrEmpty($Text)) {
        return
    }
    $reader = [System.IO.StringReader]::new($Text)
    try {
        while ($true) {
            $line = $reader.ReadLine()
            if ($null -eq $line) {
                break
            }
            if ([string]::IsNullOrWhiteSpace($line)) {
                throw "Engine emitted a non-JSON stdout line after shutdown acknowledgement."
            }
            try {
                $event = $line | ConvertFrom-Json -ErrorAction Stop
            }
            catch {
                $displayLine = if ($line.Length -gt 240) { $line.Substring(0, 240) + "..." } else { $line }
                throw "Engine emitted non-JSON stdout after shutdown acknowledgement: $displayLine"
            }
            if ($null -eq $event -or $event -is [System.Array] -or $event -is [string] -or $event -is [ValueType]) {
                throw "Engine emitted JSON that is not a protocol object after shutdown acknowledgement."
            }
            Assert-ProtocolHeader -Event $event -Context "Trailing engine stdout event"
            $eventType = Get-ProtocolString -Object $event -Name "type" -Context "Trailing engine stdout event"
            if ($eventType -eq "runtime_error") {
                $errorObject = Get-RequiredProperty -Object $event -Name "error" -Context "Trailing runtime_error event"
                $errorMessage = Get-ProtocolString -Object $errorObject -Name "message" -Context "Trailing runtime_error.error"
                throw "Engine reported a runtime error after shutdown acknowledgement: $errorMessage"
            }
            throw "Engine emitted an unexpected trailing '$eventType' event after shutdown acknowledgement."
        }
    }
    finally {
        $reader.Dispose()
    }
}

function Send-ProtocolCommand {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.StreamWriter]$Writer,
        [Parameter(Mandatory = $true)]
        [hashtable]$Command
    )

    $json = $Command | ConvertTo-Json -Compress -Depth 12
    $Writer.WriteLine($json)
    $Writer.Flush()
}

function Get-ProtocolBoolean {
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
    return [bool]$value
}

function Get-ValidatedEventType {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Event,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $eventType = Get-ProtocolString -Object $Event -Name "type" -Context $Context
    if ($eventType -eq "runtime_error") {
        $errorObject = Get-RequiredProperty -Object $Event -Name "error" -Context "runtime_error event"
        $errorMessage = Get-ProtocolString -Object $errorObject -Name "message" -Context "runtime_error.error"
        throw "Engine reported a runtime error while waiting for ${Context}: $errorMessage"
    }
    return $eventType
}

function Get-BytesSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes
    )

    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return -join ($sha256.ComputeHash($Bytes) | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $sha256.Dispose()
    }
}

function New-ExpectedDeterministicRgb8 {
    param(
        [Parameter(Mandatory = $true)]
        [long]$Sequence,
        [Parameter(Mandatory = $true)]
        [int]$Width,
        [Parameter(Mandatory = $true)]
        [int]$Height,
        [Parameter(Mandatory = $true)]
        [bool]$Invert
    )

    $pixelCount = Multiply-ProtocolInteger -Left $Width -Right $Height -Context "Expected RGB8 pixel count"
    $byteCount = Multiply-ProtocolInteger -Left $pixelCount -Right 3 -Context "Expected RGB8 byte count"
    if ($byteCount -gt [int]::MaxValue) {
        throw "Expected deterministic frame is too large to construct safely: $byteCount bytes."
    }

    $bytes = New-Object byte[] ([int]$byteCount)
    $sequenceModulo = [long]($Sequence % 256)
    $redBase = [int](($sequenceModulo * 3) % 256)
    $greenBase = [int](($sequenceModulo * 5) % 256)
    $blueBase = [int](($sequenceModulo * 7) % 256)
    $index = 0
    for ($y = 0; $y -lt $Height; $y++) {
        $green = [int](($y * 2 + $greenBase) % 256)
        for ($x = 0; $x -lt $Width; $x++) {
            $red = [int](($x + $redBase) % 256)
            $blue = [int]((($x -bxor $y) + $blueBase) % 256)
            if ($Invert) {
                $bytes[$index] = [byte](255 - $red)
                $bytes[$index + 1] = [byte](255 - $green)
                $bytes[$index + 2] = [byte](255 - $blue)
            }
            else {
                $bytes[$index] = [byte]$red
                $bytes[$index + 1] = [byte]$green
                $bytes[$index + 2] = [byte]$blue
            }
            $index += 3
        }
    }
    return ,$bytes
}

function Read-ValidatedPreviewFrame {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Event,
        [Parameter(Mandatory = $true)]
        [System.IO.MemoryMappedFiles.MemoryMappedFile]$Mapping,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceGeneration,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceSlots,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceSlotBytes,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceMaxWidth,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceMaxHeight,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceFileLength,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedWidth,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedHeight,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $frameGeneration = Get-ProtocolInteger -Object $Event -Name "generation" -Context $Context
    if ($frameGeneration -ne $SurfaceGeneration) {
        throw "$Context generation $frameGeneration does not match surface generation $SurfaceGeneration."
    }
    $slot = Get-ProtocolInteger -Object $Event -Name "slot" -Context $Context
    if ($slot -ge $SurfaceSlots) {
        throw "$Context slot $slot is outside the advertised $SurfaceSlots slots."
    }
    $sequence = Get-ProtocolInteger -Object $Event -Name "sequence" -Context $Context -Minimum 1
    $width = Get-ProtocolInteger -Object $Event -Name "width" -Context $Context -Minimum 1
    $height = Get-ProtocolInteger -Object $Event -Name "height" -Context $Context -Minimum 1
    $stride = Get-ProtocolInteger -Object $Event -Name "stride" -Context $Context -Minimum 1
    $nbytes = Get-ProtocolInteger -Object $Event -Name "nbytes" -Context $Context -Minimum 1
    if ($width -gt $SurfaceMaxWidth -or $height -gt $SurfaceMaxHeight) {
        throw "$Context dimensions ${width}x${height} exceed surface maximum ${SurfaceMaxWidth}x${SurfaceMaxHeight}."
    }
    if ($width -ne $ExpectedWidth -or $height -ne $ExpectedHeight) {
        throw "$Context dimensions ${width}x${height} do not match the deterministic source ${ExpectedWidth}x${ExpectedHeight}."
    }

    $rowBytes = Multiply-ProtocolInteger -Left $width -Right 3 -Context "$Context RGB8 row bytes"
    if ($stride -lt $rowBytes) {
        throw "$Context stride $stride is smaller than its $rowBytes-byte RGB8 row."
    }
    $precedingRows = Multiply-ProtocolInteger -Left ($height - 1) -Right $stride -Context "$Context preceding rows"
    $requiredFrameBytes = Add-ProtocolInteger -Left $precedingRows -Right $rowBytes -Context "$Context required byte range"
    if ($nbytes -lt $requiredFrameBytes -or $nbytes -gt $SurfaceSlotBytes) {
        throw "$Context nbytes $nbytes is inconsistent with its dimensions, stride, and $SurfaceSlotBytes-byte slot."
    }
    $offset = Multiply-ProtocolInteger -Left $slot -Right $SurfaceSlotBytes -Context "$Context slot offset"
    $end = Add-ProtocolInteger -Left $offset -Right $nbytes -Context "$Context mapped range"
    if ($end -gt $SurfaceFileLength) {
        throw "$Context mapped range [$offset, $end) exceeds the $SurfaceFileLength-byte surface."
    }
    if ($nbytes -gt [int]::MaxValue -or $requiredFrameBytes -gt [int]::MaxValue) {
        throw "$Context is too large for the release gate to read safely."
    }

    $view = $null
    try {
        $view = $Mapping.CreateViewStream(
            $offset,
            $nbytes,
            [System.IO.MemoryMappedFiles.MemoryMappedFileAccess]::Read
        )
        $rawBytes = New-Object byte[] ([int]$nbytes)
        $totalRead = 0
        while ($totalRead -lt $rawBytes.Length) {
            $read = $view.Read($rawBytes, $totalRead, $rawBytes.Length - $totalRead)
            if ($read -le 0) {
                throw "$Context mmap reached EOF after $totalRead of $($rawBytes.Length) bytes."
            }
            $totalRead += $read
        }
    }
    finally {
        if ($null -ne $view) {
            $view.Dispose()
        }
    }

    $compactByteCount = Multiply-ProtocolInteger -Left $height -Right $rowBytes -Context "$Context compact RGB8 bytes"
    $rgbBytes = New-Object byte[] ([int]$compactByteCount)
    if ($stride -eq $rowBytes) {
        [Array]::Copy($rawBytes, 0, $rgbBytes, 0, [int]$compactByteCount)
    }
    else {
        for ($row = 0; $row -lt $height; $row++) {
            $sourceOffset = [int]($row * $stride)
            $targetOffset = [int]($row * $rowBytes)
            [Array]::Copy($rawBytes, $sourceOffset, $rgbBytes, $targetOffset, [int]$rowBytes)
        }
    }

    return [pscustomobject]@{
        Slot = $slot
        Generation = $frameGeneration
        Sequence = $sequence
        Width = $width
        Height = $height
        Stride = $stride
        NBytes = $nbytes
        Bytes = $rgbBytes
        Sha256 = Get-BytesSha256 -Bytes $rgbBytes
    }
}

function Assert-ExactDeterministicPreview {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Event,
        [Parameter(Mandatory = $true)]
        [System.IO.MemoryMappedFiles.MemoryMappedFile]$Mapping,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceGeneration,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceSlots,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceSlotBytes,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceMaxWidth,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceMaxHeight,
        [Parameter(Mandatory = $true)]
        [long]$SurfaceFileLength,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedWidth,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedHeight,
        [Parameter(Mandatory = $true)]
        [bool]$Invert,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $frame = Read-ValidatedPreviewFrame `
        -Event $Event `
        -Mapping $Mapping `
        -SurfaceGeneration $SurfaceGeneration `
        -SurfaceSlots $SurfaceSlots `
        -SurfaceSlotBytes $SurfaceSlotBytes `
        -SurfaceMaxWidth $SurfaceMaxWidth `
        -SurfaceMaxHeight $SurfaceMaxHeight `
        -SurfaceFileLength $SurfaceFileLength `
        -ExpectedWidth $ExpectedWidth `
        -ExpectedHeight $ExpectedHeight `
        -Context $Context
    $expectedBytes = New-ExpectedDeterministicRgb8 `
        -Sequence $frame.Sequence `
        -Width $ExpectedWidth `
        -Height $ExpectedHeight `
        -Invert $Invert
    if ($frame.Bytes.Length -ne $expectedBytes.Length) {
        throw "$Context compact RGB8 length is $($frame.Bytes.Length); expected $($expectedBytes.Length)."
    }
    for ($index = 0; $index -lt $expectedBytes.Length; $index++) {
        if ($frame.Bytes[$index] -ne $expectedBytes[$index]) {
            $mode = if ($Invert) { "Invert" } else { "pass-through" }
            $expectedHash = Get-BytesSha256 -Bytes $expectedBytes
            throw "$Context failed exact $mode parity at byte $index (expected $($expectedBytes[$index]), actual $($frame.Bytes[$index])); expected SHA-256 $expectedHash, actual $($frame.Sha256)."
        }
    }
    return $frame
}

function Send-PreviewSlotRelease {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.StreamWriter]$Writer,
        [Parameter(Mandatory = $true)]
        [object]$Frame
    )

    Send-ProtocolCommand -Writer $Writer -Command @{
        version = $protocolVersion
        type = "release_preview_slot"
        slot = $Frame.Slot
        generation = $Frame.Generation
        sequence = $Frame.Sequence
    }
}

function Assert-SnapshotState {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Event,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedRevision,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedGeneration,
        [Parameter(Mandatory = $true)]
        [bool]$ExpectedTransportRunning,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedWidth,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedHeight,
        [Parameter(Mandatory = $true)]
        [bool]$Invert,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $revision = Get-ProtocolInteger -Object $Event -Name "state_revision" -Context $Context
    $generation = Get-ProtocolInteger -Object $Event -Name "graph_generation" -Context $Context
    $running = Get-ProtocolBoolean -Object $Event -Name "transport_running" -Context $Context
    if ($revision -ne $ExpectedRevision) {
        throw "$Context state revision is $revision; expected $ExpectedRevision."
    }
    if ($generation -ne $ExpectedGeneration) {
        throw "$Context graph generation is $generation; expected $ExpectedGeneration."
    }
    if ($running -ne $ExpectedTransportRunning) {
        throw "$Context transport_running is $running; expected $ExpectedTransportRunning."
    }
    $sourceReady = Get-ProtocolBoolean -Object $Event -Name "source_ready" -Context $Context
    $sourceKind = Get-ProtocolString -Object $Event -Name "source_kind" -Context $Context
    if (-not $sourceReady -or $sourceKind -ne "test_pattern") {
        throw "$Context no longer reports the deterministic test-pattern source as ready."
    }
    $width = Get-ProtocolInteger -Object $Event -Name "width" -Context $Context -Minimum 1
    $height = Get-ProtocolInteger -Object $Event -Name "height" -Context $Context -Minimum 1
    if ($width -ne $ExpectedWidth -or $height -ne $ExpectedHeight) {
        throw "$Context output dimensions ${width}x${height} do not match ${ExpectedWidth}x${ExpectedHeight}."
    }
    $sequence = Get-ProtocolInteger -Object $Event -Name "graph_sequence" -Context $Context
    $authoritativeGeneration = Get-ProtocolInteger -Object $Event -Name "authoritative_graph_generation" -Context $Context
    if ($authoritativeGeneration -gt $ExpectedGeneration) {
        throw "$Context authoritative graph generation $authoritativeGeneration is newer than canonical generation $ExpectedGeneration."
    }
    $authoritativeMatchesCanonical = $authoritativeGeneration -eq $ExpectedGeneration
    if ($sequence -gt 0 -and $authoritativeMatchesCanonical) {
        $actualHash = Get-ProtocolString -Object $Event -Name "output_sha256" -Context $Context
        $expectedBytes = New-ExpectedDeterministicRgb8 `
            -Sequence $sequence `
            -Width $ExpectedWidth `
            -Height $ExpectedHeight `
            -Invert $Invert
        $expectedHash = Get-BytesSha256 -Bytes $expectedBytes
        if ($actualHash -ne $expectedHash) {
            $mode = if ($Invert) { "Invert" } else { "pass-through" }
            throw "$Context output SHA-256 failed exact $mode parity for sequence $sequence (expected $expectedHash, actual $actualHash)."
        }
    }
    if (-not $authoritativeMatchesCanonical) {
        # A decoupled authoritative frame from the previous graph may still be
        # draining after replace_graph is acknowledged. Its sequence/hash remain
        # truthful for that older generation but cannot be parity-checked against
        # the newly accepted graph yet.
        return [long]-1
    }
    return $sequence
}

function Assert-AcceptedAcknowledgement {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Event,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedId,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedCommandType,
        [Parameter(Mandatory = $true)]
        [long]$PreviousRevision,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedGeneration,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $ackId = Get-ProtocolInteger -Object $Event -Name "id" -Context $Context
    if ($ackId -ne $ExpectedId) {
        throw "$Context id is $ackId; expected $ExpectedId."
    }
    $commandType = Get-ProtocolString -Object $Event -Name "command_type" -Context $Context
    if ($commandType -ne $ExpectedCommandType) {
        throw "$Context names command '$commandType'; expected '$ExpectedCommandType'."
    }
    $accepted = Get-ProtocolBoolean -Object $Event -Name "accepted" -Context $Context
    if (-not $accepted) {
        $errorProperty = $Event.PSObject.Properties["error"]
        $detail = if ($null -eq $errorProperty -or $null -eq $errorProperty.Value) {
            "no structured error"
        }
        else {
            $errorProperty.Value | ConvertTo-Json -Compress -Depth 8
        }
        throw "Engine rejected $($ExpectedCommandType): $detail"
    }
    $errorProperty = $Event.PSObject.Properties["error"]
    if ($null -eq $errorProperty -or $null -ne $errorProperty.Value) {
        throw "Accepted $ExpectedCommandType acknowledgement must contain an explicit null error property."
    }
    $revision = Get-ProtocolInteger -Object $Event -Name "state_revision" -Context $Context
    if ($revision -ne ($PreviousRevision + 1)) {
        throw "$ExpectedCommandType advanced state revision from $PreviousRevision to $revision; expected $($PreviousRevision + 1)."
    }
    $generation = Get-ProtocolInteger -Object $Event -Name "graph_generation" -Context $Context
    if ($generation -ne $ExpectedGeneration) {
        throw "$ExpectedCommandType acknowledgement graph generation is $generation; expected $ExpectedGeneration."
    }
    $effectiveSequence = Get-ProtocolInteger -Object $Event -Name "effective_frame_sequence" -Context $Context
    return [pscustomobject]@{
        Revision = $revision
        Generation = $generation
        EffectiveSequence = $effectiveSequence
    }
}

try {
    $resolvedEngineBundle = (Resolve-Path -LiteralPath $EngineBundle -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $resolvedEngineBundle -PathType Container)) {
        throw "EngineBundle must be a PyInstaller onedir directory: $resolvedEngineBundle"
    }
    $engineExecutable = Join-Path $resolvedEngineBundle "megaglitch-engine-v2.exe"
    $engineInternal = Join-Path $resolvedEngineBundle "_internal"
    if (-not (Test-Path -LiteralPath $engineExecutable -PathType Leaf)) {
        throw "EngineBundle is missing megaglitch-engine-v2.exe: $engineExecutable"
    }
    if (-not (Test-Path -LiteralPath $engineInternal -PathType Container)) {
        throw "EngineBundle is not a complete PyInstaller onedir; _internal is missing: $engineInternal"
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
        throw "Could not construct a Windows-system-only PATH for the frozen-engine test."
    }

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $engineExecutable
    $startInfo.Arguments = "--hash-mode sha256"
    $startInfo.WorkingDirectory = $resolvedEngineBundle
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [Text.Encoding]::UTF8
    $startInfo.EnvironmentVariables["PATH"] = $sanitizedPath

    $environmentOverrides = @(
        "MEGAGLITCH_PYTHON",
        "MEGAGLITCH_UI_BRIDGE_SCRIPT",
        "MEGAGLITCH_UI_BRIDGE_EXECUTABLE",
        "MEGAGLITCH_UI_BRIDGE_STARTUP_TIMEOUT_MS",
        "MEGAGLITCH_OPENCV_BIN",
        "MEGAGLITCH_OPENCV_PYTHON",
        "MEGAGLITCH_CUDA_BIN",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONEXECUTABLE",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
        "CONDA_SHLVL",
        "VIRTUAL_ENV"
    )
    foreach ($name in $environmentOverrides) {
        [void]$startInfo.EnvironmentVariables.Remove($name)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Windows declined to start the frozen Engine V2 executable."
    }
    $processStarted = $true
    $stderrTask = $process.StandardError.ReadToEndAsync()

    $startupDeadline = [datetime]::UtcNow.AddSeconds($StartupTimeoutSeconds)

    $readyEvent = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $startupDeadline -Context "the ready event"
    $readyEventType = Get-ValidatedEventType -Event $readyEvent -Context "the ready event"
    if ($readyEventType -ne "ready") {
        throw "Engine's first protocol event was '$readyEventType'; expected 'ready'."
    }

    $surfaceEvent = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $startupDeadline -Context "the preview_surface event"
    $surfaceEventType = Get-ValidatedEventType -Event $surfaceEvent -Context "the preview_surface event"
    if ($surfaceEventType -ne "preview_surface") {
        throw "Engine's second protocol event was '$surfaceEventType'; expected 'preview_surface'."
    }

    $backend = Get-ProtocolString -Object $readyEvent -Name "backend" -Context "ready event"
    $build = Get-ProtocolString -Object $readyEvent -Name "build" -Context "ready event"
    $initialStateRevision = Get-ProtocolInteger -Object $readyEvent -Name "state_revision" -Context "ready event"
    $initialGraphGeneration = Get-ProtocolInteger -Object $readyEvent -Name "graph_generation" -Context "ready event"
    $readyTransportRunning = Get-ProtocolBoolean -Object $readyEvent -Name "transport_running" -Context "ready event"
    $sourceReady = Get-ProtocolBoolean -Object $readyEvent -Name "source_ready" -Context "ready event"
    $sourceKind = Get-ProtocolString -Object $readyEvent -Name "source_kind" -Context "ready event"
    [void](Get-ProtocolString -Object $readyEvent -Name "source_label" -Context "ready event")
    if (-not $readyTransportRunning) {
        throw "Engine ready event did not start with transport running."
    }
    if (-not $sourceReady -or $sourceKind -ne "test_pattern") {
        throw "Engine ready event did not advertise a usable deterministic test-pattern source."
    }
    $readyWidthValue = Get-ProtocolInteger -Object $readyEvent -Name "width" -Context "ready event" -Minimum 1
    $readyHeightValue = Get-ProtocolInteger -Object $readyEvent -Name "height" -Context "ready event" -Minimum 1
    if ($readyWidthValue -gt [int]::MaxValue -or $readyHeightValue -gt [int]::MaxValue) {
        throw "Engine default source dimensions are too large for the release gate."
    }
    $readyWidth = [int]$readyWidthValue
    $readyHeight = [int]$readyHeightValue

    $capabilities = Get-RequiredProperty -Object $readyEvent -Name "capabilities" -Context "ready event"
    $transactionalValue = Get-RequiredProperty -Object $capabilities -Name "transactional_commands" -Context "ready capabilities"
    $transactionalCommands = @($transactionalValue)
    foreach ($requiredCommand in @("replace_graph", "set_transport", "shutdown")) {
        if ($transactionalCommands -notcontains $requiredCommand) {
            throw "Engine ready capabilities do not include transactional '$requiredCommand'."
        }
    }
    $fireAndForgetValue = Get-RequiredProperty -Object $capabilities -Name "fire_and_forget_commands" -Context "ready capabilities"
    if (@($fireAndForgetValue) -notcontains "release_preview_slot") {
        throw "Engine ready capabilities do not include fire-and-forget preview-slot release."
    }
    $sourceKindsValue = Get-RequiredProperty -Object $capabilities -Name "source_kinds" -Context "ready capabilities"
    if (@($sourceKindsValue) -notcontains "test_pattern") {
        throw "Engine ready capabilities do not include the deterministic test-pattern source."
    }

    $previewCapabilities = Get-RequiredProperty -Object $capabilities -Name "preview" -Context "ready capabilities"
    $previewTransport = Get-ProtocolString -Object $previewCapabilities -Name "transport" -Context "ready preview capabilities"
    $readyPixelFormat = Get-ProtocolString -Object $previewCapabilities -Name "pixel_format" -Context "ready preview capabilities"
    if ($previewTransport -ne "mmap_fixed_slots") {
        throw "Engine preview transport is '$previewTransport'; expected 'mmap_fixed_slots'."
    }
    if ($readyPixelFormat -ne "RGB8") {
        throw "Engine ready preview pixel format is '$readyPixelFormat'; expected 'RGB8'."
    }
    $capabilitySlots = Get-ProtocolInteger -Object $previewCapabilities -Name "slots" -Context "ready preview capabilities" -Minimum 1
    $capabilitySlotBytes = Get-ProtocolInteger -Object $previewCapabilities -Name "slot_bytes" -Context "ready preview capabilities" -Minimum 1
    $capabilityMaxWidth = Get-ProtocolInteger -Object $previewCapabilities -Name "max_width" -Context "ready preview capabilities" -Minimum 1
    $capabilityMaxHeight = Get-ProtocolInteger -Object $previewCapabilities -Name "max_height" -Context "ready preview capabilities" -Minimum 1

    $surfacePixelFormat = Get-ProtocolString -Object $surfaceEvent -Name "pixel_format" -Context "preview_surface event"
    if ($surfacePixelFormat -ne "RGB8") {
        throw "Preview surface pixel format is '$surfacePixelFormat'; expected 'RGB8'."
    }
    $surfaceGeneration = Get-ProtocolInteger -Object $surfaceEvent -Name "generation" -Context "preview_surface event"
    $surfaceSlots = Get-ProtocolInteger -Object $surfaceEvent -Name "slots" -Context "preview_surface event" -Minimum 1
    $surfaceSlotBytes = Get-ProtocolInteger -Object $surfaceEvent -Name "slot_bytes" -Context "preview_surface event" -Minimum 1
    $surfaceMaxWidth = Get-ProtocolInteger -Object $surfaceEvent -Name "max_width" -Context "preview_surface event" -Minimum 1
    $surfaceMaxHeight = Get-ProtocolInteger -Object $surfaceEvent -Name "max_height" -Context "preview_surface event" -Minimum 1
    if (
        $surfaceSlots -ne $capabilitySlots -or
        $surfaceSlotBytes -ne $capabilitySlotBytes -or
        $surfaceMaxWidth -ne $capabilityMaxWidth -or
        $surfaceMaxHeight -ne $capabilityMaxHeight
    ) {
        throw "Preview surface descriptor does not match the negotiated ready capabilities."
    }
    if ($readyWidth -gt $surfaceMaxWidth -or $readyHeight -gt $surfaceMaxHeight) {
        throw "Default deterministic source $($readyWidth)x$($readyHeight) exceeds preview surface maximum $($surfaceMaxWidth)x$($surfaceMaxHeight)."
    }

    $surfacePath = Get-ProtocolString -Object $surfaceEvent -Name "path" -Context "preview_surface event"
    if (-not [IO.Path]::IsPathRooted($surfacePath)) {
        throw "Preview surface path is not absolute: $surfacePath"
    }
    if (-not (Test-Path -LiteralPath $surfacePath -PathType Leaf)) {
        throw "Engine advertised a preview surface that does not exist: $surfacePath"
    }
    $minimumSurfaceBytes = Multiply-ProtocolInteger -Left $surfaceSlots -Right $surfaceSlotBytes -Context "Preview surface capacity"
    $surfaceFileLength = (Get-Item -LiteralPath $surfacePath -ErrorAction Stop).Length
    if ($surfaceFileLength -lt $minimumSurfaceBytes) {
        throw "Preview surface is $surfaceFileLength bytes; descriptor requires at least $minimumSurfaceBytes bytes."
    }

    # Windows PowerShell 5.1 requires a non-empty mapping name for this
    # overload. Keep the producer-owned file open read-only and retain the map
    # only until set_transport(false) has been accepted and observed.
    $gateMapName = "megaglitch-engine-v2-gate-$PID-$([Guid]::NewGuid().ToString('N'))"
    $surfaceFileStream = [System.IO.FileStream]::new(
        $surfacePath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
    )
    $mappedFile = [System.IO.MemoryMappedFiles.MemoryMappedFile]::CreateFromFile(
        $surfaceFileStream,
        $gateMapName,
        $surfaceFileLength,
        [System.IO.MemoryMappedFiles.MemoryMappedFileAccess]::Read,
        [System.IO.HandleInheritability]::None,
        $true
    )

    $previewArguments = @{
        Mapping = $mappedFile
        SurfaceGeneration = $surfaceGeneration
        SurfaceSlots = $surfaceSlots
        SurfaceSlotBytes = $surfaceSlotBytes
        SurfaceMaxWidth = $surfaceMaxWidth
        SurfaceMaxHeight = $surfaceMaxHeight
        SurfaceFileLength = $surfaceFileLength
        ExpectedWidth = $readyWidth
        ExpectedHeight = $readyHeight
    }

    $previewReleases = 0
    $defaultFrame = $null
    $initialFrameDeadline = [datetime]::UtcNow.AddSeconds($FrameTimeoutSeconds)
    while ($null -eq $defaultFrame) {
        $event = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $initialFrameDeadline -Context "the deterministic pass-through preview"
        $eventType = Get-ValidatedEventType -Event $event -Context "the deterministic pass-through preview"
        if ($eventType -eq "preview_frame") {
            $defaultFrame = Assert-ExactDeterministicPreview -Event $event -Invert $false -Context "Default preview_frame" @previewArguments
            Send-PreviewSlotRelease -Writer $process.StandardInput -Frame $defaultFrame
            $previewReleases += 1
        }
        elseif ($eventType -eq "snapshot") {
            [void](Assert-SnapshotState -Event $event -ExpectedRevision $initialStateRevision -ExpectedGeneration $initialGraphGeneration -ExpectedTransportRunning $true -ExpectedWidth $readyWidth -ExpectedHeight $readyHeight -Invert $false -Context "Default snapshot")
        }
        else {
            throw "Engine emitted invalid '$eventType' event while waiting for the default preview."
        }
    }
    $lastPreGraphSequence = [long]$defaultFrame.Sequence

    $invertGraph = @{
        version = "1.0"
        nodes = @(
            @{
                id = "InvertColorNode:1"
                type = "InvertColorNode"
                label = "Invert Colors"
                position = @(200.0, 100.0)
                params = @{
                    enabled = $true
                    invert_chance = 100
                }
            }
        )
        links = @(
            @{
                from_node = "SOURCE"
                to_node = "Invert Colors"
                to_node_id = "InvertColorNode:1"
                from_attr = "image_out"
                to_attr = "image_in"
            },
            @{
                from_node = "Invert Colors"
                to_node = "DISPLAY"
                from_node_id = "InvertColorNode:1"
                from_attr = "image_out"
                to_attr = "image_in"
            }
        )
    }
    Send-ProtocolCommand -Writer $process.StandardInput -Command @{
        version = $protocolVersion
        id = $graphCommandId
        expected_state_revision = $initialStateRevision
        type = "replace_graph"
        graph = $invertGraph
    }

    $graphAck = $null
    $graphAckDeadline = [datetime]::UtcNow.AddSeconds($FrameTimeoutSeconds)
    while ($null -eq $graphAck) {
        $event = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $graphAckDeadline -Context "the replace_graph acknowledgement"
        $eventType = Get-ValidatedEventType -Event $event -Context "the replace_graph acknowledgement"
        if ($eventType -eq "ack") {
            $ackId = Get-ProtocolInteger -Object $event -Name "id" -Context "replace_graph ack candidate"
            if ($ackId -ne $graphCommandId) {
                throw "Engine emitted an acknowledgement for unexpected command id $ackId while waiting for replace_graph."
            }
            $graphAck = $event
        }
        elseif ($eventType -eq "preview_frame") {
            $preGraphFrame = Assert-ExactDeterministicPreview -Event $event -Invert $false -Context "Pre-graph preview_frame" @previewArguments
            if ($preGraphFrame.Sequence -le $lastPreGraphSequence) {
                throw "Pre-graph preview sequence $($preGraphFrame.Sequence) did not advance beyond $lastPreGraphSequence."
            }
            $lastPreGraphSequence = [long]$preGraphFrame.Sequence
            Send-PreviewSlotRelease -Writer $process.StandardInput -Frame $preGraphFrame
            $previewReleases += 1
        }
        elseif ($eventType -eq "snapshot") {
            [void](Assert-SnapshotState -Event $event -ExpectedRevision $initialStateRevision -ExpectedGeneration $initialGraphGeneration -ExpectedTransportRunning $true -ExpectedWidth $readyWidth -ExpectedHeight $readyHeight -Invert $false -Context "Pre-graph snapshot")
        }
        else {
            throw "Engine emitted invalid '$eventType' event before replace_graph acknowledgement."
        }
    }

    $graphGeneration = $initialGraphGeneration + 1
    $graphState = Assert-AcceptedAcknowledgement -Event $graphAck -ExpectedId $graphCommandId -ExpectedCommandType "replace_graph" -PreviousRevision $initialStateRevision -ExpectedGeneration $graphGeneration -Context "replace_graph ack"
    if ($graphState.EffectiveSequence -le $lastPreGraphSequence) {
        throw "replace_graph effective sequence $($graphState.EffectiveSequence) did not follow the last pass-through sequence $lastPreGraphSequence."
    }

    $requiredInvertFrames = [int]$surfaceSlots + 1
    $invertFramesValidated = 0
    $seenInvertSequences = [System.Collections.Generic.HashSet[long]]::new()
    $slotLastSequence = @{}
    $slotReused = $false
    $firstInvertSequence = [long]0
    $lastInvertSequence = [long]0
    $lastInvertHash = ""
    $invertDeadline = [datetime]::UtcNow.AddSeconds($FrameTimeoutSeconds)
    while ($invertFramesValidated -lt $requiredInvertFrames) {
        $event = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $invertDeadline -Context "$requiredInvertFrames exact Invert preview frames"
        $eventType = Get-ValidatedEventType -Event $event -Context "exact Invert preview sampling"
        if ($eventType -eq "preview_frame") {
            $invertFrame = Assert-ExactDeterministicPreview -Event $event -Invert $true -Context "Invert preview_frame" @previewArguments
            if ($invertFrame.Sequence -lt $graphState.EffectiveSequence) {
                throw "Invert preview sequence $($invertFrame.Sequence) precedes graph effective sequence $($graphState.EffectiveSequence)."
            }
            if (-not $seenInvertSequences.Add([long]$invertFrame.Sequence)) {
                throw "Engine repeated Invert preview sequence $($invertFrame.Sequence)."
            }
            if ($lastInvertSequence -gt 0 -and $invertFrame.Sequence -le $lastInvertSequence) {
                throw "Invert preview sequence $($invertFrame.Sequence) did not advance beyond $lastInvertSequence."
            }
            $slotKey = [string]$invertFrame.Slot
            if ($slotLastSequence.ContainsKey($slotKey)) {
                if ([long]$slotLastSequence[$slotKey] -ne [long]$invertFrame.Sequence) {
                    $slotReused = $true
                }
            }
            $slotLastSequence[$slotKey] = [long]$invertFrame.Sequence
            if ($firstInvertSequence -eq 0) {
                $firstInvertSequence = [long]$invertFrame.Sequence
            }
            $lastInvertSequence = [long]$invertFrame.Sequence
            $lastInvertHash = [string]$invertFrame.Sha256
            Send-PreviewSlotRelease -Writer $process.StandardInput -Frame $invertFrame
            $previewReleases += 1
            $invertFramesValidated += 1
        }
        elseif ($eventType -eq "snapshot") {
            $snapshotSequence = Assert-SnapshotState -Event $event -ExpectedRevision $graphState.Revision -ExpectedGeneration $graphGeneration -ExpectedTransportRunning $true -ExpectedWidth $readyWidth -ExpectedHeight $readyHeight -Invert $true -Context "Invert snapshot"
            if ($snapshotSequence -ge 0 -and $snapshotSequence -lt $graphState.EffectiveSequence) {
                throw "Invert snapshot sequence $snapshotSequence precedes graph effective sequence $($graphState.EffectiveSequence)."
            }
        }
        else {
            throw "Engine emitted invalid '$eventType' event during Invert preview sampling."
        }
    }

    Send-ProtocolCommand -Writer $process.StandardInput -Command @{
        version = $protocolVersion
        id = $stopCommandId
        expected_state_revision = $graphState.Revision
        type = "set_transport"
        running = $false
    }

    $stopAck = $null
    $stopAckDeadline = [datetime]::UtcNow.AddSeconds($ShutdownTimeoutSeconds)
    while ($null -eq $stopAck) {
        $event = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $stopAckDeadline -Context "the set_transport(false) acknowledgement"
        $eventType = Get-ValidatedEventType -Event $event -Context "the set_transport(false) acknowledgement"
        if ($eventType -eq "ack") {
            $ackId = Get-ProtocolInteger -Object $event -Name "id" -Context "set_transport ack candidate"
            if ($ackId -ne $stopCommandId) {
                throw "Engine emitted an acknowledgement for unexpected command id $ackId while waiting for set_transport."
            }
            $stopAck = $event
        }
        elseif ($eventType -eq "preview_frame") {
            $invertFrame = Assert-ExactDeterministicPreview -Event $event -Invert $true -Context "Pre-quiescence Invert preview_frame" @previewArguments
            if ($invertFrame.Sequence -lt $graphState.EffectiveSequence) {
                throw "Pre-quiescence preview sequence $($invertFrame.Sequence) precedes graph effective sequence $($graphState.EffectiveSequence)."
            }
            if (-not $seenInvertSequences.Add([long]$invertFrame.Sequence)) {
                throw "Engine repeated pre-quiescence preview sequence $($invertFrame.Sequence)."
            }
            $slotKey = [string]$invertFrame.Slot
            if ($slotLastSequence.ContainsKey($slotKey) -and [long]$slotLastSequence[$slotKey] -ne [long]$invertFrame.Sequence) {
                $slotReused = $true
            }
            $slotLastSequence[$slotKey] = [long]$invertFrame.Sequence
            $lastInvertSequence = [long]$invertFrame.Sequence
            $lastInvertHash = [string]$invertFrame.Sha256
            Send-PreviewSlotRelease -Writer $process.StandardInput -Frame $invertFrame
            $previewReleases += 1
            $invertFramesValidated += 1
        }
        elseif ($eventType -eq "snapshot") {
            [void](Assert-SnapshotState -Event $event -ExpectedRevision $graphState.Revision -ExpectedGeneration $graphGeneration -ExpectedTransportRunning $true -ExpectedWidth $readyWidth -ExpectedHeight $readyHeight -Invert $true -Context "Pre-quiescence snapshot")
        }
        else {
            throw "Engine emitted invalid '$eventType' event before set_transport acknowledgement."
        }
    }

    $stopState = Assert-AcceptedAcknowledgement -Event $stopAck -ExpectedId $stopCommandId -ExpectedCommandType "set_transport" -PreviousRevision $graphState.Revision -ExpectedGeneration $graphGeneration -Context "set_transport ack"
    if ($stopState.EffectiveSequence -le $lastInvertSequence) {
        throw "set_transport effective sequence $($stopState.EffectiveSequence) did not follow the final running sequence $lastInvertSequence."
    }
    if ($invertFramesValidated -lt $requiredInvertFrames) {
        throw "Validated only $invertFramesValidated Invert frames; required at least $requiredInvertFrames."
    }
    if (-not $slotReused) {
        throw "Observed $invertFramesValidated released frames across $surfaceSlots slots without proving slot reuse."
    }

    $pausedSnapshot = $null
    $pausedDeadline = [datetime]::UtcNow.AddSeconds($ShutdownTimeoutSeconds)
    while ($null -eq $pausedSnapshot) {
        $event = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $pausedDeadline -Context "a quiesced transport snapshot"
        $eventType = Get-ValidatedEventType -Event $event -Context "a quiesced transport snapshot"
        if ($eventType -eq "snapshot") {
            $pausedSnapshot = $event
        }
        elseif ($eventType -eq "preview_frame") {
            throw "Engine published preview sequence $(Get-ProtocolInteger -Object $event -Name 'sequence' -Context 'post-quiescence preview') after accepting set_transport(false)."
        }
        elseif ($eventType -eq "ack") {
            throw "Engine emitted an unexpected acknowledgement after accepting set_transport(false)."
        }
        else {
            throw "Engine emitted invalid '$eventType' event after accepting set_transport(false)."
        }
    }
    $pausedSequence = Assert-SnapshotState -Event $pausedSnapshot -ExpectedRevision $stopState.Revision -ExpectedGeneration $graphGeneration -ExpectedTransportRunning $false -ExpectedWidth $readyWidth -ExpectedHeight $readyHeight -Invert $true -Context "Quiesced snapshot"
    if ($pausedSequence -ne ($stopState.EffectiveSequence - 1)) {
        throw "Quiesced snapshot sequence is $pausedSequence; set_transport effective sequence predicts $($stopState.EffectiveSequence - 1)."
    }

    # The consumer must release its view before the producer shuts down and
    # unlinks the backing file on Windows.
    $mappedFile.Dispose()
    $mappedFile = $null
    $surfaceFileStream.Dispose()
    $surfaceFileStream = $null

    Send-ProtocolCommand -Writer $process.StandardInput -Command @{
        version = $protocolVersion
        id = $shutdownCommandId
        type = "shutdown"
        expected_state_revision = $stopState.Revision
    }

    $shutdownAck = $null
    $shutdownAckDeadline = [datetime]::UtcNow.AddSeconds($ShutdownTimeoutSeconds)
    while ($null -eq $shutdownAck) {
        $event = Read-ProtocolEvent -Reader $process.StandardOutput -DeadlineUtc $shutdownAckDeadline -Context "the shutdown acknowledgement"
        $eventType = Get-ValidatedEventType -Event $event -Context "the shutdown acknowledgement"
        if ($eventType -eq "ack") {
            $ackId = Get-ProtocolInteger -Object $event -Name "id" -Context "shutdown ack candidate"
            if ($ackId -ne $shutdownCommandId) {
                throw "Engine emitted an acknowledgement for unexpected command id $ackId during shutdown."
            }
            $shutdownAck = $event
        }
        elseif ($eventType -eq "snapshot") {
            $shutdownWaitSequence = Assert-SnapshotState -Event $event -ExpectedRevision $stopState.Revision -ExpectedGeneration $graphGeneration -ExpectedTransportRunning $false -ExpectedWidth $readyWidth -ExpectedHeight $readyHeight -Invert $true -Context "Pre-shutdown snapshot"
            if ($shutdownWaitSequence -ne $pausedSequence) {
                throw "Graph sequence advanced from $pausedSequence to $shutdownWaitSequence while transport was quiesced."
            }
        }
        elseif ($eventType -eq "preview_frame") {
            throw "Engine published a preview frame while transport was quiesced and shutdown was pending."
        }
        else {
            throw "Engine emitted invalid '$eventType' event while shutdown was pending."
        }
    }

    $shutdownState = Assert-AcceptedAcknowledgement -Event $shutdownAck -ExpectedId $shutdownCommandId -ExpectedCommandType "shutdown" -PreviousRevision $stopState.Revision -ExpectedGeneration $graphGeneration -Context "shutdown ack"
    if ($shutdownState.EffectiveSequence -ne $pausedSequence) {
        throw "Shutdown effective sequence is $($shutdownState.EffectiveSequence); expected stable quiesced sequence $pausedSequence."
    }

    $process.StandardInput.Close()
    $remainingStdoutTask = $process.StandardOutput.ReadToEndAsync()
    if (-not $process.WaitForExit($ShutdownTimeoutSeconds * 1000)) {
        throw "Frozen Engine V2 did not exit within $ShutdownTimeoutSeconds seconds after accepting shutdown."
    }
    if (-not $remainingStdoutTask.Wait(2000)) {
        throw "Frozen Engine V2 stdout did not close after process exit."
    }
    if (-not $stderrTask.Wait(2000)) {
        throw "Frozen Engine V2 stderr did not close after process exit."
    }

    $remainingStdout = $remainingStdoutTask.GetAwaiter().GetResult()
    Assert-RemainingJsonOutput -Text $remainingStdout
    $stderrText = $stderrTask.GetAwaiter().GetResult()
    if ($stderrText.Length -ne 0) {
        $displayError = if ($stderrText.Length -gt 1000) { $stderrText.Substring(0, 1000) + "..." } else { $stderrText }
        throw "Frozen Engine V2 wrote to stderr:$([Environment]::NewLine)$displayError"
    }
    if ($process.ExitCode -ne 0) {
        throw "Frozen Engine V2 exited with code $($process.ExitCode); expected 0."
    }

    $surfaceCleanupDeadline = [datetime]::UtcNow.AddSeconds($ShutdownTimeoutSeconds)
    while (Test-Path -LiteralPath $surfacePath) {
        if ([datetime]::UtcNow -ge $surfaceCleanupDeadline) {
            throw "Engine exited successfully but did not remove its preview surface: $surfacePath"
        }
        Start-Sleep -Milliseconds 50
    }

    $stopwatch.Stop()
    Write-Output ([pscustomobject]@{
        Passed = $true
        EngineBundle = $resolvedEngineBundle
        ProtocolVersion = $protocolVersion
        Backend = $backend
        Build = $build
        PixelFormat = "RGB8"
        DefaultFrame = "$($readyWidth)x$($readyHeight)"
        DefaultFrameSequence = $defaultFrame.Sequence
        DefaultFrameBytes = $defaultFrame.Bytes.Length
        DefaultFrameSha256 = $defaultFrame.Sha256
        InitialStateRevision = $initialStateRevision
        InitialGraphGeneration = $initialGraphGeneration
        GraphRevision = $graphState.Revision
        GraphGeneration = $graphGeneration
        GraphEffectiveSequence = $graphState.EffectiveSequence
        InvertFramesRequired = $requiredInvertFrames
        InvertFramesValidated = $invertFramesValidated
        FirstInvertSequence = $firstInvertSequence
        LastInvertSequence = $lastInvertSequence
        LastInvertFrameSha256 = $lastInvertHash
        PreviewSlots = $surfaceSlots
        PreviewSlotReleases = $previewReleases
        SlotReuseProven = $slotReused
        TransportRevision = $stopState.Revision
        TransportEffectiveSequence = $stopState.EffectiveSequence
        QuiescedSequence = $pausedSequence
        ShutdownRevision = $shutdownState.Revision
        ExitCode = $process.ExitCode
        SurfaceRemoved = $true
        ElapsedMilliseconds = [Math]::Round($stopwatch.Elapsed.TotalMilliseconds, 1)
        SanitizedPath = $sanitizedPath
    })
}
catch {
    $failure = $_.Exception
    $diagnostics = [System.Collections.Generic.List[string]]::new()
    $diagnostics.Add("Frozen Engine V2 release gate failed: $($failure.Message)")
    if ($processStarted -and $null -ne $process) {
        try {
            if ($process.HasExited) {
                $failureExitCode = [int]$process.ExitCode
                $failureExitCodeUnsigned = [BitConverter]::ToUInt32(
                    [BitConverter]::GetBytes($failureExitCode),
                    0
                )
                $diagnostics.Add((
                    "Engine exit code at failure: {0} (0x{1:X8})" -f
                    $failureExitCode,
                    $failureExitCodeUnsigned
                ))
            }
            else {
                $diagnostics.Add("Engine was still running at failure and will be terminated by bounded cleanup.")
            }
        }
        catch {
            $diagnostics.Add("Engine process status was unavailable at failure.")
        }
    }
    if ($null -ne $stderrTask) {
        try {
            if (-not $stderrTask.IsCompleted -and $processStarted -and $process.HasExited) {
                [void]$stderrTask.Wait(1000)
            }
            if ($stderrTask.IsCompleted) {
                $failureStderr = $stderrTask.GetAwaiter().GetResult()
                if ($failureStderr.Length -gt 0) {
                    $displayFailureStderr = if ($failureStderr.Length -gt 2000) {
                        $failureStderr.Substring(0, 2000) + "..."
                    }
                    else {
                        $failureStderr
                    }
                    $diagnostics.Add("Engine stderr:`n$displayFailureStderr")
                }
            }
        }
        catch {
            $diagnostics.Add("Engine stderr could not be collected after failure.")
        }
    }
    throw [InvalidOperationException]::new(($diagnostics -join "`n"), $failure)
}
finally {
    if ($null -ne $mappedFile) {
        $mappedFile.Dispose()
    }
    if ($null -ne $surfaceFileStream) {
        $surfaceFileStream.Dispose()
    }
    if ($null -ne $process) {
        if ($processStarted) {
            try {
                if (-not $process.StandardInput.BaseStream.CanWrite) {
                    # The stream is already closed after an orderly shutdown.
                }
                else {
                    $process.StandardInput.Close()
                }
            }
            catch {
                # Best effort only: the owned process may already have closed its pipe.
            }
            try {
                if (-not $process.HasExited) {
                    $process.Kill()
                    [void]$process.WaitForExit(2000)
                }
            }
            catch {
                # Do not hide the release-gate failure with cleanup of the owned process.
            }
        }
        $process.Dispose()
    }
}
