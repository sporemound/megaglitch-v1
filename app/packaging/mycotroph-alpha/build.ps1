param(
    [ValidateSet("debug", "release")]
    [string]$Configuration = "release",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    [string]$PackageName = "MegaGlitchNode-00.9.4-Mycotroph-Functional-Preview",
    [Parameter(Mandatory = $true)]
    [string]$EngineBundle,
    [string]$PreviewImage = "",
    [string]$VerificationRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-CleanSourceCommit {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepositoryRoot,
        [string]$ExpectedCommit = ""
    )

    $commitLines = @(& git -C $RepositoryRoot rev-parse --verify HEAD)
    if ($LASTEXITCODE -ne 0 -or $commitLines.Count -ne 1) {
        throw "Could not resolve the release source commit."
    }
    $resolvedCommit = ([string]$commitLines[0]).Trim().ToLowerInvariant()
    if ($resolvedCommit -notmatch "^(?:[0-9a-f]{40}|[0-9a-f]{64})$") {
        throw "Git returned an invalid full release commit: $resolvedCommit"
    }

    $sourceStatus = @(& git -C $RepositoryRoot status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the source-tree state."
    }
    if ($sourceStatus.Count -ne 0) {
        throw "Refusing to package a dirty source tree. Commit the exact release sources first."
    }
    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedCommit) -and
        $resolvedCommit -ne $ExpectedCommit.ToLowerInvariant()
    ) {
        throw "Release source commit changed during packaging: expected $ExpectedCommit, found $resolvedCommit."
    }
    return $resolvedCommit
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
        throw "Engine build metadata is missing '$Name': $MetadataPath"
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
        $Value -is [int64]
    )
}

function Assert-EngineBuildMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EngineInternal,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedCommit,
        [Parameter(Mandatory = $true)]
        [hashtable]$ExpectedSourceHashes
    )

    $metadataPath = Join-Path $EngineInternal "engine-v2-build.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        throw "EngineBundle has no Engine V2 source metadata: $metadataPath"
    }
    try {
        $metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Engine V2 source metadata is not valid JSON: $metadataPath`n$($_.Exception.Message)"
    }
    if ($null -eq $metadata -or $metadata -is [System.Array] -or $metadata -is [ValueType]) {
        throw "Engine V2 source metadata must be one JSON object: $metadataPath"
    }

    $schema = Get-RequiredMetadataProperty -Metadata $metadata -Name "schema" -MetadataPath $metadataPath
    $artifact = Get-RequiredMetadataProperty -Metadata $metadata -Name "artifact" -MetadataPath $metadataPath
    $sourceCommit = Get-RequiredMetadataProperty -Metadata $metadata -Name "source_commit" -MetadataPath $metadataPath
    $sourceDirty = Get-RequiredMetadataProperty -Metadata $metadata -Name "source_dirty" -MetadataPath $metadataPath
    $bridgeBuildId = Get-RequiredMetadataProperty -Metadata $metadata -Name "bridge_build_id" -MetadataPath $metadataPath
    $protocolVersion = Get-RequiredMetadataProperty -Metadata $metadata -Name "protocol_version" -MetadataPath $metadataPath
    $pythonCacheTag = Get-RequiredMetadataProperty -Metadata $metadata -Name "python_cache_tag" -MetadataPath $metadataPath
    $pythonVersion = Get-RequiredMetadataProperty -Metadata $metadata -Name "python_version" -MetadataPath $metadataPath
    $pyinstallerVersion = Get-RequiredMetadataProperty -Metadata $metadata -Name "pyinstaller_version" -MetadataPath $metadataPath
    $cv2Extension = Get-RequiredMetadataProperty -Metadata $metadata -Name "cv2_extension" -MetadataPath $metadataPath
    $cv2Hash = Get-RequiredMetadataProperty -Metadata $metadata -Name "cv2_sha256" -MetadataPath $metadataPath
    $engineExecutableHash = Get-RequiredMetadataProperty -Metadata $metadata -Name "engine_executable_sha256" -MetadataPath $metadataPath
    $opencvDllCount = Get-RequiredMetadataProperty -Metadata $metadata -Name "opencv_dll_count" -MetadataPath $metadataPath
    $cudaRuntimeDllCount = Get-RequiredMetadataProperty -Metadata $metadata -Name "cuda_runtime_dll_count" -MetadataPath $metadataPath

    if ($schema -ne "megaglitch.engine-v2-build/v1" -or $artifact -ne "megaglitch-engine-v2") {
        throw "EngineBundle carries unsupported source metadata identity: $metadataPath"
    }
    if ($sourceCommit -isnot [string] -or $sourceCommit.ToLowerInvariant() -ne $ExpectedCommit.ToLowerInvariant()) {
        throw "EngineBundle was built from commit '$sourceCommit'; current clean release commit is '$ExpectedCommit'."
    }
    if ($sourceDirty -isnot [bool] -or $sourceDirty) {
        throw "EngineBundle source metadata is dirty; rebuild it from the current clean commit."
    }
    $protocolIsInteger = Test-MetadataInteger -Value $protocolVersion
    if ($bridgeBuildId -ne "rust-ui-bridge-v2" -or -not $protocolIsInteger -or [long]$protocolVersion -ne 2) {
        throw "EngineBundle does not identify the required Engine V2 bridge/protocol."
    }
    if (
        $pythonCacheTag -isnot [string] -or [string]::IsNullOrWhiteSpace($pythonCacheTag) -or
        $pythonVersion -isnot [string] -or [string]::IsNullOrWhiteSpace($pythonVersion) -or
        $pyinstallerVersion -isnot [string] -or [string]::IsNullOrWhiteSpace($pyinstallerVersion)
    ) {
        throw "EngineBundle source metadata has no usable Python/PyInstaller build identity."
    }
    if (
        $cv2Extension -isnot [string] -or
        [IO.Path]::GetFileName($cv2Extension) -ne $cv2Extension -or
        $cv2Extension -notmatch "^cv2[A-Za-z0-9._-]*\.pyd$" -or
        $cv2Hash -isnot [string] -or $cv2Hash -notmatch "^[0-9a-fA-F]{64}$" -or
        $engineExecutableHash -isnot [string] -or $engineExecutableHash -notmatch "^[0-9a-fA-F]{64}$" -or
        -not (Test-MetadataInteger -Value $opencvDllCount) -or [long]$opencvDllCount -lt 1 -or
        -not (Test-MetadataInteger -Value $cudaRuntimeDllCount) -or [long]$cudaRuntimeDllCount -lt 1
    ) {
        throw "EngineBundle source metadata has no usable cv2/OpenCV/CUDA build identity."
    }

    $bundledCv2 = Join-Path $EngineInternal $cv2Extension
    $bundledExecutable = Join-Path (Split-Path -Path $EngineInternal -Parent) "megaglitch-engine-v2.exe"
    if (
        -not (Test-Path -LiteralPath $bundledCv2 -PathType Leaf) -or
        -not (Test-Path -LiteralPath $bundledExecutable -PathType Leaf)
    ) {
        throw "EngineBundle source metadata refers to a missing cv2 or Engine V2 executable."
    }
    $actualCv2Hash = (Get-FileHash -LiteralPath $bundledCv2 -Algorithm SHA256).Hash.ToLowerInvariant()
    $actualExecutableHash = (Get-FileHash -LiteralPath $bundledExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        $actualCv2Hash -ne $cv2Hash.ToLowerInvariant() -or
        $actualExecutableHash -ne $engineExecutableHash.ToLowerInvariant()
    ) {
        throw "EngineBundle executable identity does not match its build metadata."
    }

    $bundledOpenCvDirectory = Join-Path $EngineInternal "bin\opencv"
    $bundledCudaDirectory = Join-Path $EngineInternal "bin\cuda"
    $bundledOpenCvManifest = Join-Path $bundledOpenCvDirectory "opencv-preload-manifest.json"
    $bundledCudaManifest = Join-Path $bundledCudaDirectory "cuda-runtime-manifest.json"
    if (
        -not (Test-Path -LiteralPath $bundledOpenCvManifest -PathType Leaf) -or
        -not (Test-Path -LiteralPath $bundledCudaManifest -PathType Leaf)
    ) {
        throw "EngineBundle is missing an isolated OpenCV or CUDA runtime manifest."
    }
    $bundledOpenCvDlls = @(Get-ChildItem -LiteralPath $bundledOpenCvDirectory -Filter "opencv_*.dll" -File)
    $bundledCudaDlls = @(Get-ChildItem -LiteralPath $bundledCudaDirectory -Filter "*.dll" -File)
    if (
        $bundledOpenCvDlls.Count -ne [long]$opencvDllCount -or
        $bundledCudaDlls.Count -ne [long]$cudaRuntimeDllCount
    ) {
        throw "EngineBundle native DLL counts do not match its build metadata."
    }
    $bundledOpenCvManifestHash = (Get-FileHash -LiteralPath $bundledOpenCvManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    $bundledCudaManifestHash = (Get-FileHash -LiteralPath $bundledCudaManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        $bundledOpenCvManifestHash -ne $ExpectedSourceHashes["opencv_preload_manifest_sha256"] -or
        $bundledCudaManifestHash -ne $ExpectedSourceHashes["cuda_runtime_manifest_sha256"]
    ) {
        throw "EngineBundle native runtime manifests do not match the current clean source."
    }

    foreach ($name in $ExpectedSourceHashes.Keys) {
        $actualHash = Get-RequiredMetadataProperty -Metadata $metadata -Name $name -MetadataPath $metadataPath
        if ($actualHash -isnot [string] -or $actualHash.ToLowerInvariant() -ne $ExpectedSourceHashes[$name]) {
            throw "EngineBundle source metadata '$name' does not match the current clean source."
        }
    }
    return $metadata
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$packageName = $PackageName
$distRoot = Join-Path $repoRoot "dist"
$stagingRoot = Join-Path $distRoot ".package-staging"
$stageRoot = Join-Path $stagingRoot $packageName
$archivePath = Join-Path $distRoot ($packageName + ".zip")
$checksumPath = $archivePath + ".sha256"
$candidateArchivePath = Join-Path $distRoot ($packageName + ".candidate.zip")
$candidateChecksumPath = $candidateArchivePath + ".sha256"
$executable = Join-Path $repoRoot "native\ui\target\$Configuration\megaglitch-ui.exe"
$readme = Join-Path $PSScriptRoot "README.txt"
$resolvedEngineBundle = (Resolve-Path -LiteralPath $EngineBundle -ErrorAction Stop).Path
$engineExecutable = Join-Path $resolvedEngineBundle "megaglitch-engine-v2.exe"
$engineInternal = Join-Path $resolvedEngineBundle "_internal"
$engineGate = Join-Path $repoRoot "tools\verify_engine_v2_bundle.ps1"
$hostGate = Join-Path $repoRoot "tools\verify_mycotroph_host.ps1"
$bridgeSource = Join-Path $repoRoot "tools\rust_ui_bridge.py"
$previewExecutorSource = Join-Path $repoRoot "megaglitch\preview_executor.py"
$opencvPreloadManifest = Join-Path $repoRoot "packaging\pyinstaller\opencv-preload-manifest.json"
$cudaRuntimeManifest = Join-Path $repoRoot "packaging\pyinstaller\cuda-runtime-manifest.json"

if (-not (Test-Path -LiteralPath $readme -PathType Leaf)) {
    throw "Missing package README: $readme"
}
if (-not (Test-Path -LiteralPath $resolvedEngineBundle -PathType Container)) {
    throw "EngineBundle must be a completed PyInstaller onedir: $resolvedEngineBundle"
}
if (-not (Test-Path -LiteralPath $engineExecutable -PathType Leaf)) {
    throw "EngineBundle is missing megaglitch-engine-v2.exe: $engineExecutable"
}
if (-not (Test-Path -LiteralPath $engineInternal -PathType Container)) {
    throw "EngineBundle is EXE-only or incomplete; expected PyInstaller _internal directory: $engineInternal"
}
$pythonRuntime = Get-ChildItem -LiteralPath $engineInternal -Filter "python3*.dll" -File -ErrorAction SilentlyContinue
$cv2Runtime = Get-ChildItem -LiteralPath $engineInternal -Filter "cv2*.pyd" -File -ErrorAction SilentlyContinue
$opencvRuntime = Get-ChildItem -LiteralPath (Join-Path $engineInternal "bin\opencv") -Filter "opencv_*.dll" -File -ErrorAction SilentlyContinue
$mklRuntime = Get-ChildItem -LiteralPath $engineInternal -Filter "mkl_rt*.dll" -File -ErrorAction SilentlyContinue
if (-not $pythonRuntime) {
    throw "EngineBundle has no bundled CPython runtime in $engineInternal"
}
if (-not $cv2Runtime -or @($cv2Runtime).Count -ne 1) {
    throw "EngineBundle must contain exactly one explicit custom cv2 extension in $engineInternal"
}
if (-not $opencvRuntime) {
    throw "EngineBundle has no isolated custom OpenCV DLLs under _internal\bin\opencv"
}
if (-not $mklRuntime) {
    throw "EngineBundle has no MKL dispatcher required by the bundled Conda NumPy runtime"
}
if (-not (Test-Path -LiteralPath $engineGate -PathType Leaf)) {
    throw "Missing frozen Engine V2 release gate: $engineGate"
}
if (-not (Test-Path -LiteralPath $hostGate -PathType Leaf)) {
    throw "Missing packaged Rust-host release gate: $hostGate"
}
foreach ($sourceIdentityFile in @($bridgeSource, $previewExecutorSource, $opencvPreloadManifest, $cudaRuntimeManifest)) {
    if (-not (Test-Path -LiteralPath $sourceIdentityFile -PathType Leaf)) {
        throw "Missing Engine V2 source-identity input: $sourceIdentityFile"
    }
}

if (
    (Test-Path -LiteralPath $stageRoot) -or
    (Test-Path -LiteralPath $archivePath) -or
    (Test-Path -LiteralPath $checksumPath) -or
    (Test-Path -LiteralPath $candidateArchivePath) -or
    (Test-Path -LiteralPath $candidateChecksumPath)
) {
    throw "Refusing to overwrite an existing Mycotroph package. Move or rename the existing package first."
}

$commit = Get-CleanSourceCommit -RepositoryRoot $repoRoot
$expectedEngineSourceHashes = @{
    bridge_source_sha256 = (Get-FileHash -LiteralPath $bridgeSource -Algorithm SHA256).Hash.ToLowerInvariant()
    preview_executor_source_sha256 = (Get-FileHash -LiteralPath $previewExecutorSource -Algorithm SHA256).Hash.ToLowerInvariant()
    opencv_preload_manifest_sha256 = (Get-FileHash -LiteralPath $opencvPreloadManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    cuda_runtime_manifest_sha256 = (Get-FileHash -LiteralPath $cudaRuntimeManifest -Algorithm SHA256).Hash.ToLowerInvariant()
}
$engineMetadata = Assert-EngineBuildMetadata `
    -EngineInternal $engineInternal `
    -ExpectedCommit $commit `
    -ExpectedSourceHashes $expectedEngineSourceHashes
$engineMetadataPath = Join-Path $engineInternal "engine-v2-build.json"
$engineMetadataHash = (Get-FileHash -LiteralPath $engineMetadataPath -Algorithm SHA256).Hash.ToLowerInvariant()

# Always compile the checked-in Rust host immediately before staging. Packaging
# a pre-existing target executable is how a new engine can be paired with an old
# interface while BUILD-INFO incorrectly names the current source revision.
$uiBuildScript = Join-Path $repoRoot "native\ui\build.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $uiBuildScript -Profile $Configuration
if ($LASTEXITCODE -ne 0) {
    throw "Rust UI release build failed."
}
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Rust UI build did not produce the expected executable: $executable"
}
[void](Get-CleanSourceCommit -RepositoryRoot $repoRoot -ExpectedCommit $commit)

# Validate the frozen sidecar before creating any staging directory. This direct
# protocol/mmap gate runs with a Windows-system-only PATH and no development
# Python or bridge overrides, so an archive cannot pass by borrowing the checkout.
& powershell -NoProfile -ExecutionPolicy Bypass -File $engineGate -EngineBundle $resolvedEngineBundle | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Source Engine V2 bundle failed its standalone release gate."
}

New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null
New-Item -ItemType Directory -Path $stageRoot | Out-Null
Copy-Item -LiteralPath $executable -Destination (Join-Path $stageRoot "MegaGlitchNode-Mycotroph-Functional-Preview.exe")
Copy-Item -LiteralPath $readme -Destination (Join-Path $stageRoot "README.txt")
$stagedEngine = Join-Path $stageRoot "engine-v2"
Copy-Item -LiteralPath $resolvedEngineBundle -Destination $stagedEngine -Recurse
if (-not (Test-Path -LiteralPath (Join-Path $stagedEngine "megaglitch-engine-v2.exe") -PathType Leaf)) {
    throw "Staged package lost its Engine V2 executable; refusing to create an EXE-only archive."
}
if (-not (Test-Path -LiteralPath (Join-Path $stagedEngine "_internal") -PathType Container)) {
    throw "Staged package lost its Engine V2 runtime; refusing to create an EXE-only archive."
}
$stagedEngineInternal = Join-Path $stagedEngine "_internal"
$stagedMetadata = Assert-EngineBuildMetadata `
    -EngineInternal $stagedEngineInternal `
    -ExpectedCommit $commit `
    -ExpectedSourceHashes $expectedEngineSourceHashes
$stagedMetadataPath = Join-Path $stagedEngineInternal "engine-v2-build.json"
$stagedMetadataHash = (Get-FileHash -LiteralPath $stagedMetadataPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($stagedMetadataHash -ne $engineMetadataHash) {
    throw "Staging changed the Engine V2 source metadata."
}
& powershell -NoProfile -ExecutionPolicy Bypass -File $engineGate -EngineBundle $stagedEngine | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Staged Engine V2 failed its standalone release gate."
}
& powershell -NoProfile -ExecutionPolicy Bypass -File $hostGate -PackageRoot $stageRoot | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Staged Rust host failed to own and exercise its adjacent Engine V2 bundle."
}
if (-not [string]::IsNullOrWhiteSpace($PreviewImage)) {
    $resolvedPreview = (Resolve-Path -LiteralPath $PreviewImage).Path
    if (-not (Test-Path -LiteralPath $resolvedPreview -PathType Leaf)) {
        throw "PreviewImage must resolve to one file: $resolvedPreview"
    }
    Copy-Item -LiteralPath $resolvedPreview -Destination (Join-Path $stageRoot "EXPECTED-LAYOUT.png")
}

$uiHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
$engineHash = (Get-FileHash -LiteralPath $engineExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
$buildTime = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$buildInfo = @(
    "Product: MegaGlitchNode"
    "Release: 00.9.4 Mycotroph Functional Preview"
    "Platform: Windows x64"
    "Frontend: Rust / eframe / egui / wgpu"
    "Source commit: $commit"
    "Source tree: clean"
    "Build time (UTC): $buildTime"
    "Rust UI SHA-256: $uiHash"
    "Engine V2 SHA-256: $engineHash"
    "Engine metadata SHA-256: $engineMetadataHash"
    "Engine build identity: $($engineMetadata.bridge_build_id) / protocol $($engineMetadata.protocol_version) / $($engineMetadata.python_cache_tag) / PyInstaller $($engineMetadata.pyinstaller_version)"
    "Node catalog entries: 115"
    "Engine bridge: protocol v2 bundled at engine-v2\megaglitch-engine-v2.exe"
    "Preview transport: memory-mapped fixed-slot RGB8 frames"
    "Python runtime: bundled PyInstaller onedir"
    "OpenCV runtime: explicit custom build isolated under engine-v2\_internal\bin\opencv"
    "Standalone behavior: bundled functional preview engine; no external Python or source tree required"
)
Set-Content -LiteralPath (Join-Path $stageRoot "BUILD-INFO.txt") -Value $buildInfo -Encoding UTF8

$archivePublished = $false
try {
    Compress-Archive -LiteralPath $stageRoot -DestinationPath $candidateArchivePath -CompressionLevel Optimal

    if ([string]::IsNullOrWhiteSpace($VerificationRoot)) {
        $VerificationRoot = [IO.Path]::GetTempPath()
    }
    try {
        $verificationRootFullPath = [IO.Path]::GetFullPath($VerificationRoot)
        [void][IO.Directory]::CreateDirectory($verificationRootFullPath)
    }
    catch {
        throw "Could not create VerificationRoot '$VerificationRoot': $($_.Exception.Message)"
    }
    $resolvedVerificationRoot = (Resolve-Path -LiteralPath $verificationRootFullPath -ErrorAction Stop).Path
    $verificationDirectory = Join-Path $resolvedVerificationRoot ("megaglitch-package-verify-" + [Guid]::NewGuid().ToString("N"))
    $verificationMarker = Join-Path $verificationDirectory ".megaglitch-package-verification"
    $verificationToken = [Guid]::NewGuid().ToString("N")
    try {
        New-Item -ItemType Directory -Path $verificationDirectory | Out-Null
        $resolvedVerificationDirectory = (Resolve-Path -LiteralPath $verificationDirectory).Path
        if ([IO.Path]::GetDirectoryName($resolvedVerificationDirectory) -ne $resolvedVerificationRoot) {
            throw "Verification directory escaped its requested root: $resolvedVerificationDirectory"
        }
        Set-Content -LiteralPath $verificationMarker -Value $verificationToken -Encoding ASCII
        Expand-Archive -LiteralPath $candidateArchivePath -DestinationPath $resolvedVerificationDirectory
        $extractedRoot = Join-Path $resolvedVerificationDirectory $packageName
        $extractedEngine = Join-Path $extractedRoot "engine-v2"
        $extractedEngineInternal = Join-Path $extractedEngine "_internal"
        [void](Assert-EngineBuildMetadata `
            -EngineInternal $extractedEngineInternal `
            -ExpectedCommit $commit `
            -ExpectedSourceHashes $expectedEngineSourceHashes)
        & powershell -NoProfile -ExecutionPolicy Bypass -File $engineGate -EngineBundle $extractedEngine | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Freshly extracted Engine V2 failed its standalone release gate."
        }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $hostGate -PackageRoot $extractedRoot | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Freshly extracted Rust host failed its Engine V2 ownership gate."
        }
        $extractedUi = Join-Path $extractedRoot "MegaGlitchNode-Mycotroph-Functional-Preview.exe"
        $extractedEngineExecutable = Join-Path $extractedEngine "megaglitch-engine-v2.exe"
        $extractedMetadataPath = Join-Path $extractedEngineInternal "engine-v2-build.json"
        $extractedUiHash = (Get-FileHash -LiteralPath $extractedUi -Algorithm SHA256).Hash.ToLowerInvariant()
        $extractedEngineHash = (Get-FileHash -LiteralPath $extractedEngineExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
        $extractedMetadataHash = (Get-FileHash -LiteralPath $extractedMetadataPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if (
            $extractedUiHash -ne $uiHash -or
            $extractedEngineHash -ne $engineHash -or
            $extractedMetadataHash -ne $engineMetadataHash
        ) {
            throw "Fresh extraction changed a packaged executable or Engine V2 metadata hash."
        }
    }
    finally {
        if (Test-Path -LiteralPath $verificationDirectory -PathType Container) {
            $resolvedCleanupDirectory = (Resolve-Path -LiteralPath $verificationDirectory).Path
            $cleanupMarker = Join-Path $resolvedCleanupDirectory ".megaglitch-package-verification"
            $cleanupMarkerMatches = (
                (Test-Path -LiteralPath $cleanupMarker -PathType Leaf) -and
                ((Get-Content -LiteralPath $cleanupMarker -Raw -Encoding ASCII).Trim() -eq $verificationToken)
            )
            if (
                [IO.Path]::GetDirectoryName($resolvedCleanupDirectory) -eq $resolvedVerificationRoot -and
                $cleanupMarkerMatches
            ) {
                Remove-Item -LiteralPath $resolvedCleanupDirectory -Recurse -Force
            }
            else {
                throw "Refusing to clean an unverified package-test directory: $resolvedCleanupDirectory"
            }
        }
    }

    [void](Get-CleanSourceCommit -RepositoryRoot $repoRoot -ExpectedCommit $commit)
    $hash = (Get-FileHash -LiteralPath $candidateArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath $candidateChecksumPath -Value ($hash + "  " + [IO.Path]::GetFileName($archivePath)) -Encoding ASCII
    Move-Item -LiteralPath $candidateArchivePath -Destination $archivePath
    $archivePublished = $true
    Move-Item -LiteralPath $candidateChecksumPath -Destination $checksumPath
}
catch {
    if (Test-Path -LiteralPath $candidateArchivePath -PathType Leaf) {
        Remove-Item -LiteralPath $candidateArchivePath -Force
    }
    if (Test-Path -LiteralPath $candidateChecksumPath -PathType Leaf) {
        Remove-Item -LiteralPath $candidateChecksumPath -Force
    }
    if ($archivePublished -and (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        Remove-Item -LiteralPath $archivePath -Force
    }
    throw
}

Write-Host "Created $archivePath"
Write-Host "SHA-256 $hash"
