param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,

    [ValidateSet("source", "extracted")]
    [string]$VerificationContext = "source",

    [ValidatePattern("^(?:|[0-9a-fA-F]{64})$")]
    [string]$ExpectedExecutableSha256 = "",

    [ValidatePattern("^(?:|[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")]
    [string]$ExpectedSourceCommit = "",

    [ValidatePattern("^(?:|[0-9a-fA-F]{64})$")]
    [string]$ExpectedSpecSha256 = "",

    [ValidatePattern("^(?:|[0-9a-fA-F]{64})$")]
    [string]$ExpectedRuntimeHookSha256 = "",

    [ValidatePattern("^(?:|[0-9a-fA-F]{64})$")]
    [string]$ExpectedExternalToolsSha256 = "",

    [ValidatePattern("^(?:|[0-9a-fA-F]{64})$")]
    [string]$ExpectedOpenCvManifestSha256 = "",

    [ValidatePattern("^(?:|[0-9a-fA-F]{64})$")]
    [string]$ExpectedCudaManifestSha256 = "",

    [switch]$RequireBundledFfmpeg,

    [ValidateRange(10, 600)]
    [int]$DiagnosticTimeoutSeconds = 90
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-StrictChildPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Parent,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $pathFull = [IO.Path]::GetFullPath($Path)
    $prefix = $parentFull + [IO.Path]::DirectorySeparatorChar
    if (-not $pathFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Context escapes package root '$parentFull': $pathFull"
    }
    return $pathFull
}

function Get-RequiredMetadataProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Metadata,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$MetadataPath
    )

    $property = $Metadata.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "Python-host build metadata is missing '$Name': $MetadataPath"
    }
    return $property.Value
}

function Test-MetadataInteger {
    param([AllowNull()][object]$Value)

    return (
        $Value -is [byte] -or
        $Value -is [sbyte] -or
        $Value -is [int16] -or
        $Value -is [uint16] -or
        $Value -is [int32] -or
        $Value -is [uint32] -or
        $Value -is [int64] -or
        $Value -is [uint64]
    )
}

function Get-MetadataSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Metadata,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$MetadataPath,
        [string]$ExpectedSha256 = ""
    )

    $value = Get-RequiredMetadataProperty -Metadata $Metadata -Name $Name -MetadataPath $MetadataPath
    if ($value -isnot [string] -or $value -notmatch "^[0-9a-fA-F]{64}$") {
        throw "Python-host build metadata '$Name' is not a SHA-256 identity: $MetadataPath"
    }
    $normalized = $value.ToLowerInvariant()
    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedSha256) -and
        $normalized -ne $ExpectedSha256.ToLowerInvariant()
    ) {
        throw "Python-host build metadata '$Name' does not match the current clean source."
    }
    return $normalized
}

function Get-ValidatedRuntimeManifestNames {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$NamePattern
    )

    try {
        $document = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Bundled $Label manifest is not valid JSON: $ManifestPath"
    }
    $names = @($document)
    if ($names.Count -eq 0) {
        throw "Bundled $Label manifest contains no runtime names: $ManifestPath"
    }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($nameValue in $names) {
        if (
            $nameValue -isnot [string] -or
            [IO.Path]::GetFileName($nameValue) -ne $nameValue -or
            $nameValue -notmatch $NamePattern -or
            -not $seen.Add($nameValue)
        ) {
            throw "Bundled $Label manifest contains an invalid or duplicate runtime name: $nameValue"
        }
    }
    return $names
}

function Assert-RuntimeDirectoryIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InternalRoot,
        [Parameter(Mandatory = $true)]
        [string]$Directory,
        [Parameter(Mandatory = $true)]
        [string[]]$ExpectedNames,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedCount,
        [Parameter(Mandatory = $true)]
        [string]$Filter,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [switch]$ManifestIsPreloadSubset
    )

    $files = @(Get-ChildItem -LiteralPath $Directory -Filter $Filter -File)
    if ($files.Count -ne $ExpectedCount) {
        throw "Bundled $Label count mismatch: metadata=$ExpectedCount, manifest=$($ExpectedNames.Count), files=$($files.Count)."
    }
    if (
        ($ManifestIsPreloadSubset -and $ExpectedNames.Count -gt $ExpectedCount) -or
        (-not $ManifestIsPreloadSubset -and $ExpectedNames.Count -ne $ExpectedCount)
    ) {
        throw "Bundled $Label manifest coverage mismatch: metadata=$ExpectedCount, manifest=$($ExpectedNames.Count)."
    }
    $actualNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $files) {
        [void]$actualNames.Add($file.Name)
    }
    foreach ($expectedName in $ExpectedNames) {
        if (-not $actualNames.Contains($expectedName)) {
            throw "Bundled $Label identity is missing manifest runtime: $expectedName"
        }
    }
    if ($ManifestIsPreloadSubset) {
        # The OpenCV manifest is the dependency-first direct-import preload
        # order for cv2, while the isolated directory contains the complete
        # modular OpenCV closure. Audit every matching runtime in that case.
        $allMatchingRuntimeFiles = @(
            Get-ChildItem -LiteralPath $InternalRoot -Filter $Filter -File -Recurse
        )
    }
    else {
        $expectedNameSet = [Collections.Generic.HashSet[string]]::new(
            $ExpectedNames,
            [StringComparer]::OrdinalIgnoreCase
        )
        $allMatchingRuntimeFiles = @(Get-ChildItem -LiteralPath $InternalRoot -Filter $Filter -File -Recurse | Where-Object {
            $expectedNameSet.Contains($_.Name)
        })
    }
    if ($allMatchingRuntimeFiles.Count -ne $ExpectedCount) {
        throw "Bundled $Label identities are duplicated outside their isolated runtime directory."
    }
    foreach ($runtimeFile in $allMatchingRuntimeFiles) {
        if (-not $runtimeFile.DirectoryName.Equals($Directory, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Bundled $Label identity escaped its isolated runtime directory: $($runtimeFile.FullName)"
        }
    }
}

function Assert-NoUnrelatedQtRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InternalRoot
    )

    $bindingComponentPattern = "(?i)(?:^|[\\/])(?:PyQt5|PyQt6|PySide2|PySide6)(?:[\\/]|$)"
    $qtRuntimeDllPattern = "(?i)^Qt[56].*\.dll$"
    $violations = @(
        Get-ChildItem -LiteralPath $InternalRoot -Recurse -Force | Where-Object {
            $_.FullName -match $bindingComponentPattern -or
            (-not $_.PSIsContainer -and $_.Name -match $qtRuntimeDllPattern)
        }
    )
    if ($violations.Count -ne 0) {
        throw "Package contains an unrelated Qt runtime excluded from the Dear PyGui host: $($violations[0].FullName)"
    }
}

function Assert-PythonHostBuildMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$InternalRoot,
        [string]$ExpectedCommit = "",
        [string]$ExpectedSpec = "",
        [string]$ExpectedRuntimeHook = "",
        [string]$ExpectedExternalTools = "",
        [string]$ExpectedOpenCvManifest = "",
        [string]$ExpectedCudaManifest = ""
    )

    $metadataPath = Join-Path $InternalRoot "python-host-build.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        throw "PackageRoot is missing exact Python-host build metadata: $metadataPath"
    }
    try {
        $metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Python-host build metadata is not valid JSON: $metadataPath"
    }
    if ($null -eq $metadata -or $metadata -is [Array] -or $metadata -is [ValueType]) {
        throw "Python-host build metadata must be one JSON object: $metadataPath"
    }

    $schema = Get-RequiredMetadataProperty -Metadata $metadata -Name "schema" -MetadataPath $metadataPath
    $artifact = Get-RequiredMetadataProperty -Metadata $metadata -Name "artifact" -MetadataPath $metadataPath
    $bundleName = Get-RequiredMetadataProperty -Metadata $metadata -Name "bundle_name" -MetadataPath $metadataPath
    $sourceCommit = Get-RequiredMetadataProperty -Metadata $metadata -Name "source_commit" -MetadataPath $metadataPath
    $sourceDirty = Get-RequiredMetadataProperty -Metadata $metadata -Name "source_dirty" -MetadataPath $metadataPath
    if ($schema -ne "megaglitch.python-host-build/v1" -or $artifact -ne "megaglitch-python-host") {
        throw "PackageRoot carries unsupported Python-host build metadata identity: $metadataPath"
    }
    if (
        $bundleName -isnot [string] -or
        $bundleName -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$" -or
        -not $bundleName.Equals([IO.Path]::GetFileName($Root), [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Python-host metadata bundle_name does not match the onedir root."
    }
    if ($sourceCommit -isnot [string] -or $sourceCommit -notmatch "^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$") {
        throw "Python-host build metadata has no valid full source commit."
    }
    $sourceCommit = $sourceCommit.ToLowerInvariant()
    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedCommit) -and
        $sourceCommit -ne $ExpectedCommit.ToLowerInvariant()
    ) {
        throw "Python-host artifact was built from commit '$sourceCommit'; expected current clean commit '$ExpectedCommit'."
    }
    if ($sourceDirty -isnot [bool] -or $sourceDirty) {
        throw "Python-host artifact metadata records a dirty source build."
    }
    foreach ($identityName in @("python_cache_tag", "python_version", "pyinstaller_version")) {
        $identity = Get-RequiredMetadataProperty -Metadata $metadata -Name $identityName -MetadataPath $metadataPath
        if ($identity -isnot [string] -or [string]::IsNullOrWhiteSpace($identity)) {
            throw "Python-host build metadata '$identityName' is empty."
        }
    }
    $metadataFfmpegVersion = Get-RequiredMetadataProperty -Metadata $metadata -Name "ffmpeg_version" -MetadataPath $metadataPath
    $metadataFfprobeVersion = Get-RequiredMetadataProperty -Metadata $metadata -Name "ffprobe_version" -MetadataPath $metadataPath
    if ($metadataFfmpegVersion -isnot [string] -or -not $metadataFfmpegVersion.StartsWith("ffmpeg version ", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Python-host build metadata has no recognizable ffmpeg_version identity."
    }
    if ($metadataFfprobeVersion -isnot [string] -or -not $metadataFfprobeVersion.StartsWith("ffprobe version ", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Python-host build metadata has no recognizable ffprobe_version identity."
    }
    $ffmpegDistribution = Get-RequiredMetadataProperty -Metadata $metadata -Name "ffmpeg_distribution" -MetadataPath $metadataPath
    if (
        $ffmpegDistribution -isnot [string] -or
        $ffmpegDistribution -notmatch "^[A-Za-z0-9][A-Za-z0-9._+ -]{0,255}$"
    ) {
        throw "Python-host build metadata has no safe FFmpeg distribution identity."
    }

    $specHash = Get-MetadataSha256 -Metadata $metadata -Name "spec_sha256" -MetadataPath $metadataPath -ExpectedSha256 $ExpectedSpec
    $runtimeHookHash = Get-MetadataSha256 -Metadata $metadata -Name "runtime_hook_sha256" -MetadataPath $metadataPath -ExpectedSha256 $ExpectedRuntimeHook
    $externalToolsHash = Get-MetadataSha256 -Metadata $metadata -Name "external_tools_sha256" -MetadataPath $metadataPath -ExpectedSha256 $ExpectedExternalTools
    $cv2Hash = Get-MetadataSha256 -Metadata $metadata -Name "cv2_sha256" -MetadataPath $metadataPath
    $opencvManifestHash = Get-MetadataSha256 -Metadata $metadata -Name "opencv_preload_manifest_sha256" -MetadataPath $metadataPath -ExpectedSha256 $ExpectedOpenCvManifest
    $cudaManifestHash = Get-MetadataSha256 -Metadata $metadata -Name "cuda_runtime_manifest_sha256" -MetadataPath $metadataPath -ExpectedSha256 $ExpectedCudaManifest
    $ffmpegHash = Get-MetadataSha256 -Metadata $metadata -Name "ffmpeg_sha256" -MetadataPath $metadataPath
    $ffprobeHash = Get-MetadataSha256 -Metadata $metadata -Name "ffprobe_sha256" -MetadataPath $metadataPath
    $ffmpegLicenseHash = Get-MetadataSha256 -Metadata $metadata -Name "ffmpeg_license_sha256" -MetadataPath $metadataPath
    $ffmpegReadmeHash = Get-MetadataSha256 -Metadata $metadata -Name "ffmpeg_readme_sha256" -MetadataPath $metadataPath
    $ffmpegSourceArchiveHash = Get-MetadataSha256 -Metadata $metadata -Name "ffmpeg_source_archive_sha256" -MetadataPath $metadataPath

    $cv2Extension = Get-RequiredMetadataProperty -Metadata $metadata -Name "cv2_extension" -MetadataPath $metadataPath
    if (
        $cv2Extension -isnot [string] -or
        [IO.Path]::GetFileName($cv2Extension) -ne $cv2Extension -or
        $cv2Extension -notmatch "^cv2[A-Za-z0-9._-]*\.pyd$"
    ) {
        throw "Python-host build metadata has an unsafe cv2_extension."
    }
    $expectedCv2 = Join-Path $InternalRoot $cv2Extension
    $allCv2 = @(Get-ChildItem -LiteralPath $InternalRoot -Filter "cv2*.pyd" -File -Recurse)
    if (
        $allCv2.Count -ne 1 -or
        -not (Test-Path -LiteralPath $expectedCv2 -PathType Leaf) -or
        -not $allCv2[0].FullName.Equals($expectedCv2, [StringComparison]::OrdinalIgnoreCase) -or
        (Get-FileHash -LiteralPath $expectedCv2 -Algorithm SHA256).Hash.ToLowerInvariant() -ne $cv2Hash
    ) {
        throw "Bundled cv2 identity does not match Python-host build metadata."
    }

    $opencvCount = Get-RequiredMetadataProperty -Metadata $metadata -Name "opencv_dll_count" -MetadataPath $metadataPath
    $cudaCount = Get-RequiredMetadataProperty -Metadata $metadata -Name "cuda_runtime_dll_count" -MetadataPath $metadataPath
    if (
        -not (Test-MetadataInteger -Value $opencvCount) -or [long]$opencvCount -lt 1 -or
        -not (Test-MetadataInteger -Value $cudaCount) -or [long]$cudaCount -lt 1
    ) {
        throw "Python-host build metadata has invalid OpenCV/CUDA runtime counts."
    }
    $opencvRoot = Join-Path $InternalRoot "bin\opencv"
    $cudaRoot = Join-Path $InternalRoot "bin\cuda"
    $bundledOpenCvManifest = Join-Path $opencvRoot "opencv-preload-manifest.json"
    $bundledCudaManifest = Join-Path $cudaRoot "cuda-runtime-manifest.json"
    foreach ($requiredDirectory in @($opencvRoot, $cudaRoot)) {
        if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
            throw "Python-host bundle is missing an isolated native runtime directory: $requiredDirectory"
        }
    }
    $manifestIdentities = @(
        [pscustomobject]@{ Path = $bundledOpenCvManifest; Sha256 = $opencvManifestHash; Label = "OpenCV" }
        [pscustomobject]@{ Path = $bundledCudaManifest; Sha256 = $cudaManifestHash; Label = "CUDA" }
    )
    foreach ($manifestIdentity in $manifestIdentities) {
        if (-not (Test-Path -LiteralPath $manifestIdentity.Path -PathType Leaf)) {
            throw "Python-host bundle is missing its $($manifestIdentity.Label) runtime manifest."
        }
        $actualManifestHash = (Get-FileHash -LiteralPath $manifestIdentity.Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualManifestHash -ne $manifestIdentity.Sha256) {
            throw "Bundled $($manifestIdentity.Label) manifest does not match Python-host build metadata."
        }
    }
    $opencvNames = @(Get-ValidatedRuntimeManifestNames -ManifestPath $bundledOpenCvManifest -Label "OpenCV" -NamePattern "(?i)^opencv_.+\.dll$")
    $cudaNames = @(Get-ValidatedRuntimeManifestNames -ManifestPath $bundledCudaManifest -Label "CUDA" -NamePattern "(?i)^.+\.dll$")
    Assert-RuntimeDirectoryIdentity -InternalRoot $InternalRoot -Directory $opencvRoot -ExpectedNames $opencvNames -ExpectedCount ([long]$opencvCount) -Filter "opencv_*.dll" -Label "OpenCV runtime" -ManifestIsPreloadSubset
    Assert-RuntimeDirectoryIdentity -InternalRoot $InternalRoot -Directory $cudaRoot -ExpectedNames $cudaNames -ExpectedCount ([long]$cudaCount) -Filter "*.dll" -Label "CUDA runtime"

    $ffmpegPath = Join-Path $InternalRoot "bin\ffmpeg\ffmpeg.exe"
    $ffprobePath = Join-Path $InternalRoot "bin\ffmpeg\ffprobe.exe"
    $toolIdentities = @(
        [pscustomobject]@{ Path = $ffmpegPath; Sha256 = $ffmpegHash; Label = "ffmpeg.exe" }
        [pscustomobject]@{ Path = $ffprobePath; Sha256 = $ffprobeHash; Label = "ffprobe.exe" }
    )
    foreach ($toolIdentity in $toolIdentities) {
        if (
            -not (Test-Path -LiteralPath $toolIdentity.Path -PathType Leaf) -or
            (Get-FileHash -LiteralPath $toolIdentity.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $toolIdentity.Sha256
        ) {
            throw "Bundled $($toolIdentity.Label) identity does not match Python-host build metadata."
        }
    }
    $ffmpegLicensePath = Join-Path $InternalRoot "licenses\ffmpeg\LICENSE"
    $ffmpegReadmePath = Join-Path $InternalRoot "licenses\ffmpeg\README.txt"
    $documentIdentities = @(
        [pscustomobject]@{ Path = $ffmpegLicensePath; Sha256 = $ffmpegLicenseHash; Label = "FFmpeg LICENSE" }
        [pscustomobject]@{ Path = $ffmpegReadmePath; Sha256 = $ffmpegReadmeHash; Label = "FFmpeg README.txt" }
    )
    foreach ($documentIdentity in $documentIdentities) {
        if (
            -not (Test-Path -LiteralPath $documentIdentity.Path -PathType Leaf) -or
            (Get-FileHash -LiteralPath $documentIdentity.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $documentIdentity.Sha256
        ) {
            throw "Bundled $($documentIdentity.Label) does not match Python-host build metadata."
        }
    }

    Add-Member -InputObject $metadata -NotePropertyName "_metadata_path" -NotePropertyValue $metadataPath -Force
    Add-Member -InputObject $metadata -NotePropertyName "_metadata_sha256" -NotePropertyValue ((Get-FileHash -LiteralPath $metadataPath -Algorithm SHA256).Hash.ToLowerInvariant()) -Force
    Add-Member -InputObject $metadata -NotePropertyName "_validated_spec_sha256" -NotePropertyValue $specHash -Force
    Add-Member -InputObject $metadata -NotePropertyName "_validated_runtime_hook_sha256" -NotePropertyValue $runtimeHookHash -Force
    Add-Member -InputObject $metadata -NotePropertyName "_validated_external_tools_sha256" -NotePropertyValue $externalToolsHash -Force
    Add-Member -InputObject $metadata -NotePropertyName "_validated_ffmpeg_source_archive_sha256" -NotePropertyValue $ffmpegSourceArchiveHash -Force
    return $metadata
}

function Assert-BundleManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    $expected = [Collections.Generic.Dictionary[string, string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    $lineNumber = 0
    foreach ($line in @(Get-Content -LiteralPath $ManifestPath -Encoding UTF8)) {
        $lineNumber += 1
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line -notmatch "^(?<hash>[0-9a-fA-F]{64})  (?<path>.+)$") {
            throw "Invalid BUNDLE-SHA256.txt entry at line $lineNumber."
        }

        $relative = [string]$Matches["path"]
        $segments = @($relative -split "[\\/]")
        if (
            [IO.Path]::IsPathRooted($relative) -or
            $relative.Contains(":") -or
            $segments.Count -eq 0 -or
            @($segments | Where-Object { $_ -eq "" -or $_ -eq "." -or $_ -eq ".." }).Count -ne 0
        ) {
            throw "Unsafe BUNDLE-SHA256.txt path at line ${lineNumber}: $relative"
        }
        if ($expected.ContainsKey($relative)) {
            throw "Duplicate BUNDLE-SHA256.txt path at line ${lineNumber}: $relative"
        }

        $candidate = Join-Path $Root ($relative.Replace("/", [IO.Path]::DirectorySeparatorChar))
        $candidate = Assert-StrictChildPath -Path $candidate -Parent $Root -Context "Manifest entry"
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Bundle manifest refers to a missing file: $relative"
        }
        $actualHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
        $expectedHash = ([string]$Matches["hash"]).ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw "Bundle manifest hash mismatch: $relative"
        }
        $expected.Add($relative, $expectedHash)
    }
    if ($expected.Count -eq 0) {
        throw "BUNDLE-SHA256.txt contains no artifact files."
    }

    $actual = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($file in @(Get-ChildItem -LiteralPath $Root -Recurse -Force -File)) {
        $relative = $file.FullName.Substring($Root.Length).TrimStart("\", "/").Replace("\", "/")
        if ($relative -in @("BUILD-INFO.txt", "BUNDLE-SHA256.txt")) {
            continue
        }
        [void]$actual.Add($relative)
    }
    if ($actual.Count -ne $expected.Count) {
        throw "Bundle contents do not match BUNDLE-SHA256.txt: expected $($expected.Count) artifact files, found $($actual.Count)."
    }
    foreach ($relative in $actual) {
        if (-not $expected.ContainsKey($relative)) {
            throw "Bundle contains an artifact file absent from BUNDLE-SHA256.txt: $relative"
        }
    }
}

function Get-DiagnosticJson {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,
        [Parameter(Mandatory = $true)]
        [string]$Argument
    )

    $lines = @($Text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    for ($index = $lines.Count - 1; $index -ge 0; $index -= 1) {
        try {
            $candidate = $lines[$index] | ConvertFrom-Json -ErrorAction Stop
            if ($null -ne $candidate -and $null -ne $candidate.PSObject.Properties["ok"]) {
                return $candidate
            }
        }
        catch {
            # Bootstrap status lines may precede the final diagnostic JSON.
        }
    }
    throw "$Argument emitted no diagnostic JSON object.`n$Text"
}

function Set-StandaloneProcessEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [Diagnostics.ProcessStartInfo]$StartInfo,
        [switch]$MegaGlitchRuntime
    )

    # Do not let a frozen gate borrow the development Python/Conda environment,
    # custom OpenCV/CUDA paths, or a checkout-side UI bridge. Copy the key list
    # before removal because ProcessStartInfo exposes a mutable dictionary.
    foreach ($nameObject in @($StartInfo.EnvironmentVariables.Keys)) {
        $name = [string]$nameObject
        if (
            $name -match "^(?i:PYTHON)" -or
            $name -match "^(?i:CONDA)" -or
            $name -match "^(?i:VIRTUAL_ENV)" -or
            $name -match "^(?i:MEGAGLITCH_(?:PYTHON|OPENCV|CUDA|UI_BRIDGE|NATIVE_))"
        ) {
            [void]$StartInfo.EnvironmentVariables.Remove($name)
        }
    }
    $systemRoot = [Environment]::GetEnvironmentVariable("SystemRoot")
    if ([string]::IsNullOrWhiteSpace($systemRoot)) {
        throw "SystemRoot is unavailable; cannot create a standalone verification environment."
    }
    $StartInfo.EnvironmentVariables["PATH"] = @(
        (Join-Path $systemRoot "System32")
        $systemRoot
        (Join-Path $systemRoot "System32\Wbem")
    ) -join ";"
    if ($MegaGlitchRuntime) {
        $StartInfo.EnvironmentVariables["MEGAGLITCH_RUNTIME_BACKEND"] = "python"
        $StartInfo.EnvironmentVariables["MEGAGLITCH_FRAME_TRANSPORT"] = "shared"
    }
}

function Get-BoundedProcessText {
    param(
        [AllowEmptyString()]
        [string]$Text,
        [int]$MaximumCharacters = 4000
    )

    if ($Text.Length -le $MaximumCharacters) {
        return $Text
    }
    return $Text.Substring(0, $MaximumCharacters) + "... [truncated]"
}

function Invoke-BundledTool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds
    )

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Executable
    $startInfo.Arguments = $Arguments
    $startInfo.WorkingDirectory = $Root
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    Set-StandaloneProcessEnvironment -StartInfo $startInfo

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $processStarted = $false
    try {
        if (-not $process.Start()) {
            throw "Could not start bundled $Label."
        }
        $processStarted = $true
        # Read both pipes asynchronously before waiting so a verbose FFmpeg
        # build cannot deadlock the release gate by filling one pipe.
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try {
                $process.Kill()
                [void]$process.WaitForExit(5000)
            }
            catch {
                # The bounded timeout remains the primary failure.
            }
            throw "Bundled $Label exceeded its $TimeoutSeconds second timeout."
        }
        if (-not $stdoutTask.Wait(5000) -or -not $stderrTask.Wait(5000)) {
            throw "Bundled $Label output pipes did not close after process exit."
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) {
            $safeStdout = Get-BoundedProcessText -Text $stdout
            $safeStderr = Get-BoundedProcessText -Text $stderr
            throw "Bundled $Label exited with code $($process.ExitCode).`nSTDOUT:`n$safeStdout`nSTDERR:`n$safeStderr"
        }
        return [pscustomobject]@{
            Stdout = $stdout
            Stderr = $stderr
            ExitCode = $process.ExitCode
        }
    }
    finally {
        if ($processStarted -and -not $process.HasExited) {
            try {
                $process.Kill()
                [void]$process.WaitForExit(5000)
            }
            catch {
                # Do not mask the release-gate failure during bounded cleanup.
            }
        }
        $process.Dispose()
    }
}

function Invoke-FrozenDiagnostic {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [ValidateSet("--diagnose-only", "--diagnose-worker")]
        [string]$Argument,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds
    )

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Executable
    $startInfo.Arguments = $Argument
    $startInfo.WorkingDirectory = $Root
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    Set-StandaloneProcessEnvironment -StartInfo $startInfo -MegaGlitchRuntime

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $processStarted = $false
    try {
        if (-not $process.Start()) {
            throw "Could not start $Argument."
        }
        $processStarted = $true
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try {
                $process.Kill()
                [void]$process.WaitForExit(5000)
            }
            catch {
                # The timeout remains the primary failure.
            }
            throw "$Argument exceeded its $TimeoutSeconds second timeout."
        }
        if (-not $stdoutTask.Wait(5000) -or -not $stderrTask.Wait(5000)) {
            throw "$Argument output pipes did not close after process exit."
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) {
            throw "$Argument exited with code $($process.ExitCode).`nSTDOUT:`n$stdout`nSTDERR:`n$stderr"
        }
        $diagnostic = Get-DiagnosticJson -Text $stdout -Argument $Argument
        if ($diagnostic.ok -isnot [bool] -or -not $diagnostic.ok) {
            throw "$Argument did not report ok=true."
        }
        if ($diagnostic.frozen -isnot [bool] -or -not $diagnostic.frozen) {
            throw "$Argument did not run inside the frozen Python host."
        }
        return $diagnostic
    }
    finally {
        if ($processStarted -and -not $process.HasExited) {
            try {
                $process.Kill()
                [void]$process.WaitForExit(5000)
            }
            catch {
                # Do not mask the verification failure during bounded cleanup.
            }
        }
        $process.Dispose()
    }
}

$resolvedPackageRoot = (Resolve-Path -LiteralPath $PackageRoot -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $resolvedPackageRoot -PathType Container)) {
    throw "PackageRoot must be a Python PyInstaller onedir: $resolvedPackageRoot"
}
$executable = Join-Path $resolvedPackageRoot "MegaGlitchNode.exe"
$internalRoot = Join-Path $resolvedPackageRoot "_internal"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "PackageRoot is missing MegaGlitchNode.exe: $executable"
}
if (-not (Test-Path -LiteralPath $internalRoot -PathType Container)) {
    throw "PackageRoot is not a complete PyInstaller onedir; _internal is missing."
}
if (-not @(Get-ChildItem -LiteralPath $internalRoot -Filter "python3*.dll" -File).Count) {
    throw "PackageRoot has no bundled CPython runtime under _internal."
}
Assert-NoUnrelatedQtRuntime -InternalRoot $internalRoot
$buildMetadata = Assert-PythonHostBuildMetadata `
    -Root $resolvedPackageRoot `
    -InternalRoot $internalRoot `
    -ExpectedCommit $ExpectedSourceCommit `
    -ExpectedSpec $ExpectedSpecSha256 `
    -ExpectedRuntimeHook $ExpectedRuntimeHookSha256 `
    -ExpectedExternalTools $ExpectedExternalToolsSha256 `
    -ExpectedOpenCvManifest $ExpectedOpenCvManifestSha256 `
    -ExpectedCudaManifest $ExpectedCudaManifestSha256
$ffmpegRoot = Join-Path $internalRoot "bin\ffmpeg"

$actualExecutableHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
if (
    -not [string]::IsNullOrWhiteSpace($ExpectedExecutableSha256) -and
    $actualExecutableHash -ne $ExpectedExecutableSha256.ToLowerInvariant()
) {
    throw "MegaGlitchNode.exe does not match the expected source artifact hash."
}

$manifestPath = Join-Path $resolvedPackageRoot "BUNDLE-SHA256.txt"
$buildInfoPath = Join-Path $resolvedPackageRoot "BUILD-INFO.txt"
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    Assert-BundleManifest -Root $resolvedPackageRoot -ManifestPath $manifestPath
}
elseif ($VerificationContext -eq "extracted") {
    throw "Freshly extracted package is missing BUNDLE-SHA256.txt."
}
if (
    $VerificationContext -eq "extracted" -and
    -not (Test-Path -LiteralPath $buildInfoPath -PathType Leaf)
) {
    throw "Freshly extracted package is missing BUILD-INFO.txt."
}

$mediaToolEncoders = @()
$ffprobeVersionLine = ""
if ($RequireBundledFfmpeg) {
    $ffmpegPath = Join-Path $ffmpegRoot "ffmpeg.exe"
    $ffprobePath = Join-Path $ffmpegRoot "ffprobe.exe"
    $encoderResult = Invoke-BundledTool `
        -Executable $ffmpegPath `
        -Root $resolvedPackageRoot `
        -Arguments "-hide_banner -encoders" `
        -Label "ffmpeg.exe -hide_banner -encoders" `
        -TimeoutSeconds $DiagnosticTimeoutSeconds
    $encoderText = $encoderResult.Stdout + "`n" + $encoderResult.Stderr
    $requiredEncoders = @("libx264", "libx265", "mpeg4", "ffv1")
    foreach ($encoder in $requiredEncoders) {
        $pattern = "(?im)^\s*V\S*\s+" + [regex]::Escape($encoder) + "\b"
        if ($encoderText -notmatch $pattern) {
            throw "Bundled ffmpeg.exe does not advertise required video encoder '$encoder'."
        }
    }
    $mediaToolEncoders = $requiredEncoders

    $probeResult = Invoke-BundledTool `
        -Executable $ffprobePath `
        -Root $resolvedPackageRoot `
        -Arguments "-version" `
        -Label "ffprobe.exe -version" `
        -TimeoutSeconds $DiagnosticTimeoutSeconds
    $probeText = $probeResult.Stdout + "`n" + $probeResult.Stderr
    if ($probeText -notmatch "(?im)^ffprobe version\s+\S+") {
        throw "Bundled ffprobe.exe -version returned no recognizable version banner."
    }
    $probeLines = @($probeResult.Stdout -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($probeLines.Count -eq 0) {
        throw "Bundled ffprobe.exe -version returned no stdout version line."
    }
    $ffprobeVersionLine = ([string]$probeLines[0]).Trim()
    if ($ffprobeVersionLine -ne [string]$buildMetadata.ffprobe_version) {
        throw "Bundled ffprobe.exe runtime version does not match Python-host build metadata."
    }
}

$runtimeDiagnostic = Invoke-FrozenDiagnostic `
    -Executable $executable `
    -Root $resolvedPackageRoot `
    -Argument "--diagnose-only" `
    -TimeoutSeconds $DiagnosticTimeoutSeconds
$workerDiagnostic = Invoke-FrozenDiagnostic `
    -Executable $executable `
    -Root $resolvedPackageRoot `
    -Argument "--diagnose-worker" `
    -TimeoutSeconds $DiagnosticTimeoutSeconds

if (
    $workerDiagnostic.reset_round_trip -isnot [bool] -or
    -not $workerDiagnostic.reset_round_trip
) {
    throw "--diagnose-worker did not complete the bounded RESET queue round trip."
}
$verifiedFrameTransport = [string]$workerDiagnostic.frame_transport
if ($verifiedFrameTransport -ne "shared_v1") {
    throw "--diagnose-worker did not attach the required shared_v1 frame transport."
}
if (
    $null -eq $workerDiagnostic.PSObject.Properties["frame_round_trip"] -or
    $null -eq $workerDiagnostic.frame_round_trip -or
    $workerDiagnostic.frame_round_trip.ok -isnot [bool] -or
    -not $workerDiagnostic.frame_round_trip.ok
) {
    throw "--diagnose-worker did not complete the deterministic shared-frame round trip."
}
$frameRoundTrip = $workerDiagnostic.frame_round_trip
foreach ($requiredFrameField in @(
    "ok", "input_transport", "output_transport", "shape", "sha256"
)) {
    if ($null -eq $frameRoundTrip.PSObject.Properties[$requiredFrameField]) {
        throw "--diagnose-worker shared-frame result is missing '$requiredFrameField'."
    }
}
$expectedFrameSha256 = "266d05fcd17ddd9f96f5cb7eae3c9f486fd5697a2b0198586a82af2cdc7c0558"
$frameShape = @($frameRoundTrip.shape)
if (
    [string]$frameRoundTrip.input_transport -ne "shared_v1" -or
    [string]$frameRoundTrip.output_transport -ne "shared_v1" -or
    [string]$frameRoundTrip.sha256 -ne $expectedFrameSha256 -or
    $frameShape.Count -ne 3 -or
    [int]$frameShape[0] -ne 12 -or
    [int]$frameShape[1] -ne 16 -or
    [int]$frameShape[2] -ne 3
) {
    throw "--diagnose-worker returned an invalid deterministic shared-frame result."
}
if (
    $null -eq $workerDiagnostic.PSObject.Properties["lifecycle"] -or
    $null -eq $workerDiagnostic.lifecycle
) {
    throw "--diagnose-worker emitted no final lifecycle snapshot."
}
$workerLifecycle = $workerDiagnostic.lifecycle
foreach ($requiredLifecycleField in @(
    "state", "exitcode", "forced_termination", "error", "status_channel_error"
)) {
    if ($null -eq $workerLifecycle.PSObject.Properties[$requiredLifecycleField]) {
        throw "--diagnose-worker lifecycle is missing '$requiredLifecycleField'."
    }
}
if ([string]$workerLifecycle.state -ne "stopped") {
    throw "--diagnose-worker did not reach the stopped lifecycle state."
}
if (
    $null -eq $workerLifecycle.PSObject.Properties["exitcode"] -or
    $null -eq $workerLifecycle.exitcode -or
    [int]$workerLifecycle.exitcode -ne 0
) {
    throw "--diagnose-worker did not report worker exit code 0."
}
if (
    $workerLifecycle.forced_termination -isnot [bool] -or
    $workerLifecycle.forced_termination
) {
    throw "--diagnose-worker required forced worker termination."
}
if ($null -ne $workerLifecycle.error) {
    throw "--diagnose-worker reported a lifecycle error."
}
if ($null -ne $workerLifecycle.status_channel_error) {
    throw "--diagnose-worker reported a lifecycle status-channel error."
}
if (
    $null -eq $workerDiagnostic.PSObject.Properties["shared_transport"] -or
    $null -eq $workerDiagnostic.shared_transport
) {
    throw "--diagnose-worker emitted no shared-transport statistics."
}
$sharedTransportStats = $workerDiagnostic.shared_transport
foreach ($requiredTransportField in @(
    "input_frames_written", "output_frames_read",
    "input_slots_in_use", "output_slots_in_use"
)) {
    if (
        $null -eq $sharedTransportStats.PSObject.Properties[$requiredTransportField] -or
        $null -eq $sharedTransportStats.$requiredTransportField
    ) {
        throw "--diagnose-worker shared statistics are missing '$requiredTransportField'."
    }
}
if (
    [int64]$sharedTransportStats.input_frames_written -ne 1 -or
    [int64]$sharedTransportStats.output_frames_read -ne 1 -or
    [int64]$sharedTransportStats.input_slots_in_use -ne 0 -or
    [int64]$sharedTransportStats.output_slots_in_use -ne 0
) {
    throw "--diagnose-worker retained shared-memory transport slots."
}

if ($null -ne $runtimeDiagnostic.PSObject.Properties["cv2_path"]) {
    $cv2Path = [string]$runtimeDiagnostic.cv2_path
    if ([string]::IsNullOrWhiteSpace($cv2Path)) {
        throw "--diagnose-only reported an empty cv2 path."
    }
    [void](Assert-StrictChildPath -Path $cv2Path -Parent $resolvedPackageRoot -Context "Frozen cv2")
}

Write-Output ([pscustomobject]@{
    Passed = $true
    Context = $VerificationContext
    PackageRoot = $resolvedPackageRoot
    ExecutableSha256 = $actualExecutableHash
    RuntimeBackend = "python"
    FrameTransport = $verifiedFrameTransport
    DiagnoseOnly = $true
    DiagnoseWorker = $true
    BundleManifestVerified = (Test-Path -LiteralPath $manifestPath -PathType Leaf)
    BundledFfmpegVerified = [bool]$RequireBundledFfmpeg
    BundledVideoEncoders = @($mediaToolEncoders)
    FfprobeVersion = $ffprobeVersionLine
    BuildMetadataSha256 = $buildMetadata._metadata_sha256
    BuildSourceCommit = $buildMetadata.source_commit
    BuildBundleName = $buildMetadata.bundle_name
    Cv2Sha256 = $buildMetadata.cv2_sha256
    FfmpegSha256 = $buildMetadata.ffmpeg_sha256
    FfprobeSha256 = $buildMetadata.ffprobe_sha256
})
