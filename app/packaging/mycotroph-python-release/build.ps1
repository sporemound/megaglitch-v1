param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$WorkRoot,

    [Parameter(Mandatory = $true)]
    [string]$VerificationRoot,

    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    [string]$PackageName = "MegaGlitchNode-00.9.4-Mycotroph-Render-Quality-Python",

    [string]$SevenZip = "",

    [ValidateRange(10, 600)]
    [int]$DiagnosticTimeoutSeconds = 90
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$BasePath
    )

    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

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
        throw "$Context must be a strict child of '$parentFull': $pathFull"
    }
    return $pathFull
}

function Assert-OutsideArtifact {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Artifact,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $artifactFull = [IO.Path]::GetFullPath($Artifact).TrimEnd("\", "/")
    $pathFull = [IO.Path]::GetFullPath($Path).TrimEnd("\", "/")
    if (
        $pathFull.Equals($artifactFull, [StringComparison]::OrdinalIgnoreCase) -or
        $pathFull.StartsWith(
            $artifactFull + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "$Name cannot be the artifact or live inside it: $pathFull"
    }
}

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
    $commit = ([string]$commitLines[0]).Trim().ToLowerInvariant()
    if ($commit -notmatch "^(?:[0-9a-f]{40}|[0-9a-f]{64})$") {
        throw "Git returned an invalid full release commit: $commit"
    }
    $status = @(& git -C $RepositoryRoot status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the source-tree state."
    }
    if ($status.Count -ne 0) {
        throw "Refusing to package a dirty source tree. Commit the exact release sources first."
    }
    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedCommit) -and
        $commit -ne $ExpectedCommit.ToLowerInvariant()
    ) {
        throw "Release source commit changed during packaging: expected $ExpectedCommit, found $commit."
    }
    return $commit
}

function Resolve-SevenZip {
    param([string]$RequestedPath)

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $resolved = (Resolve-Path -LiteralPath $RequestedPath -ErrorAction Stop).Path
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "SevenZip must resolve to one executable: $resolved"
        }
        return $resolved
    }
    foreach ($commandName in @("7z.exe", "7zz.exe", "7z")) {
        $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command.Source
        }
    }
    $commonCandidates = [Collections.Generic.List[string]]::new()
    foreach ($programFilesRoot in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not [string]::IsNullOrWhiteSpace($programFilesRoot)) {
            $commonCandidates.Add((Join-Path $programFilesRoot "7-Zip\7z.exe"))
        }
    }
    foreach ($candidate in $commonCandidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "7-Zip was not found. Pass -SevenZip with 7z.exe or 7zz.exe."
}

function New-OwnedDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Parent,
        [Parameter(Mandatory = $true)]
        [string]$Leaf,
        [Parameter(Mandatory = $true)]
        [string]$Token,
        [Parameter(Mandatory = $true)]
        [string]$Purpose
    )

    $path = Assert-StrictChildPath -Path (Join-Path $Parent $Leaf) -Parent $Parent -Context $Purpose
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to reuse an existing $Purpose directory: $path"
    }
    [void][IO.Directory]::CreateDirectory($path)
    $marker = @{
        schema = "megaglitch.release-owner/v1"
        token = $Token
        purpose = $Purpose
        path = $path
    } | ConvertTo-Json -Compress
    Set-Content -LiteralPath (Join-Path $path ".megaglitch-release-owner.json") -Value $marker -Encoding UTF8
    return $path
}

function Remove-OwnedDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Parent,
        [Parameter(Mandatory = $true)]
        [string]$Token,
        [Parameter(Mandatory = $true)]
        [string]$Purpose
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedParent = (Resolve-Path -LiteralPath $Parent -ErrorAction Stop).Path
    $resolvedPath = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    [void](Assert-StrictChildPath -Path $resolvedPath -Parent $resolvedParent -Context "$Purpose cleanup")
    $directory = Get-Item -LiteralPath $resolvedPath -Force
    if (($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to recursively remove reparse-point $Purpose directory: $resolvedPath"
    }
    $nestedReparsePoints = @(Get-ChildItem -LiteralPath $resolvedPath -Recurse -Force | Where-Object {
        ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    })
    if ($nestedReparsePoints.Count -ne 0) {
        throw "Refusing to recursively remove $Purpose directory containing a reparse point: $($nestedReparsePoints[0].FullName)"
    }
    $markerPath = Join-Path $resolvedPath ".megaglitch-release-owner.json"
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        throw "Refusing to remove unowned $Purpose directory without its marker: $resolvedPath"
    }
    try {
        $marker = Get-Content -LiteralPath $markerPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Refusing to remove $Purpose directory with an invalid ownership marker: $resolvedPath"
    }
    if (
        $marker.schema -ne "megaglitch.release-owner/v1" -or
        $marker.token -ne $Token -or
        $marker.purpose -ne $Purpose -or
        -not ([IO.Path]::GetFullPath([string]$marker.path).Equals($resolvedPath, [StringComparison]::OrdinalIgnoreCase))
    ) {
        throw "Refusing to remove $Purpose directory whose ownership marker does not match: $resolvedPath"
    }
    Remove-Item -LiteralPath $resolvedPath -Recurse -Force
}

function Remove-OwnedOutputFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$OutputDirectory,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedLeaf
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedOutput = (Resolve-Path -LiteralPath $OutputDirectory -ErrorAction Stop).Path
    $resolvedPath = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    [void](Assert-StrictChildPath -Path $resolvedPath -Parent $resolvedOutput -Context "Candidate output cleanup")
    if ([IO.Path]::GetFileName($resolvedPath) -ne $ExpectedLeaf) {
        throw "Refusing to remove an unexpected output file: $resolvedPath"
    }
    Remove-Item -LiteralPath $resolvedPath -Force
}

function Write-BundleManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BundleRoot,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    $reparsePoints = @(Get-ChildItem -LiteralPath $BundleRoot -Recurse -Force | Where-Object {
        ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    })
    if ($reparsePoints.Count -ne 0) {
        throw "Refusing to archive an artifact containing a reparse point: $($reparsePoints[0].FullName)"
    }
    $files = @(Get-ChildItem -LiteralPath $BundleRoot -Recurse -Force -File | Sort-Object FullName)
    if ($files.Count -eq 0) {
        throw "ArtifactRoot contains no files."
    }
    $lines = [Collections.Generic.List[string]]::new()
    [long]$totalBytes = 0
    foreach ($file in $files) {
        if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing to archive a reparse-point file: $($file.FullName)"
        }
        $relative = $file.FullName.Substring($BundleRoot.Length).TrimStart("\", "/").Replace("\", "/")
        if ($relative.Contains("`r") -or $relative.Contains("`n")) {
            throw "Artifact contains a filename that cannot be represented safely in the checksum manifest."
        }
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $lines.Add("$hash  $relative")
        $totalBytes += [long]$file.Length
    }
    Set-Content -LiteralPath $Destination -Value $lines -Encoding UTF8
    return [pscustomobject]@{
        FileCount = $files.Count
        TotalBytes = $totalBytes
        ManifestSha256 = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Assert-BundleMatchesManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BundleRoot,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    $expected = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($line in @(Get-Content -LiteralPath $ManifestPath -Encoding UTF8)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line -notmatch "^(?<hash>[0-9a-f]{64})  (?<path>.+)$") {
            throw "Generated bundle checksum manifest is invalid."
        }
        $relative = [string]$Matches["path"]
        $path = Join-Path $BundleRoot $relative.Replace("/", [IO.Path]::DirectorySeparatorChar)
        [void](Assert-StrictChildPath -Path $path -Parent $BundleRoot -Context "Source artifact manifest")
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Source artifact changed during packaging; missing $relative."
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$Matches["hash"]).ToLowerInvariant()) {
            throw "Source artifact changed during packaging: $relative"
        }
        if (-not $expected.Add($relative)) {
            throw "Generated bundle checksum manifest contains a duplicate path: $relative"
        }
    }

    $actualPaths = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($file in @(Get-ChildItem -LiteralPath $BundleRoot -Recurse -Force -File)) {
        $relative = $file.FullName.Substring($BundleRoot.Length).TrimStart("\", "/").Replace("\", "/")
        [void]$actualPaths.Add($relative)
    }
    if ($actualPaths.Count -ne $expected.Count) {
        throw "Source artifact file set changed during packaging: expected $($expected.Count) files, found $($actualPaths.Count)."
    }
    foreach ($relative in $actualPaths) {
        if (-not $expected.Contains($relative)) {
            throw "Source artifact gained a file during packaging: $relative"
        }
    }
}

function Invoke-SevenZip {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $Executable @Arguments | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed during $Context with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$artifact = (Resolve-Path -LiteralPath $ArtifactRoot -ErrorAction Stop).Path.TrimEnd("\", "/")
if (-not (Test-Path -LiteralPath $artifact -PathType Container)) {
    throw "ArtifactRoot must be an already-built Python PyInstaller onedir: $artifact"
}
$artifactLeaf = [IO.Path]::GetFileName($artifact)
if ($artifactLeaf -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$") {
    throw "ArtifactRoot leaf must be archive-safe; rename the onedir before packaging: $artifactLeaf"
}
$artifactExecutable = Join-Path $artifact "MegaGlitchNode.exe"
$artifactInternal = Join-Path $artifact "_internal"
if (-not (Test-Path -LiteralPath $artifactExecutable -PathType Leaf)) {
    throw "ArtifactRoot is missing MegaGlitchNode.exe."
}
if (-not (Test-Path -LiteralPath $artifactInternal -PathType Container)) {
    throw "ArtifactRoot is incomplete; expected a PyInstaller _internal directory."
}
foreach ($reservedMetadata in @("BUILD-INFO.txt", "BUNDLE-SHA256.txt")) {
    if (Test-Path -LiteralPath (Join-Path $artifact $reservedMetadata)) {
        throw "ArtifactRoot already contains release-owned metadata '$reservedMetadata'; rebuild a clean onedir."
    }
}

$output = Get-FullPath -Path $OutputRoot -BasePath $repositoryRoot
$work = Get-FullPath -Path $WorkRoot -BasePath $repositoryRoot
$verification = Get-FullPath -Path $VerificationRoot -BasePath $repositoryRoot
Assert-OutsideArtifact -Path $output -Artifact $artifact -Name "OutputRoot"
Assert-OutsideArtifact -Path $work -Artifact $artifact -Name "WorkRoot"
Assert-OutsideArtifact -Path $verification -Artifact $artifact -Name "VerificationRoot"

$sevenZipExecutable = Resolve-SevenZip -RequestedPath $SevenZip
$commit = Get-CleanSourceCommit -RepositoryRoot $repositoryRoot
$sourceIdentityPaths = @{
    spec = Join-Path $repositoryRoot "packaging\pyinstaller\megaglitch.spec"
    runtime_hook = Join-Path $repositoryRoot "packaging\pyinstaller\runtime_hook.py"
    external_tools = Join-Path $repositoryRoot "megaglitch\external_tools.py"
    opencv_manifest = Join-Path $repositoryRoot "packaging\pyinstaller\opencv-preload-manifest.json"
    cuda_manifest = Join-Path $repositoryRoot "packaging\pyinstaller\cuda-runtime-manifest.json"
}
$sourceIdentityHashes = @{}
foreach ($identityName in $sourceIdentityPaths.Keys) {
    $identityPath = $sourceIdentityPaths[$identityName]
    if (-not (Test-Path -LiteralPath $identityPath -PathType Leaf)) {
        throw "Missing Python-host release identity source '$identityName': $identityPath"
    }
    $sourceIdentityHashes[$identityName] = (Get-FileHash -LiteralPath $identityPath -Algorithm SHA256).Hash.ToLowerInvariant()
}

[void][IO.Directory]::CreateDirectory($output)
[void][IO.Directory]::CreateDirectory($work)
[void][IO.Directory]::CreateDirectory($verification)
$output = (Resolve-Path -LiteralPath $output).Path
$work = (Resolve-Path -LiteralPath $work).Path
$verification = (Resolve-Path -LiteralPath $verification).Path

$archivePath = Join-Path $output ($PackageName + ".zip")
$checksumPath = $archivePath + ".sha256"
if ((Test-Path -LiteralPath $archivePath) -or (Test-Path -LiteralPath $checksumPath)) {
    throw "Refusing to overwrite an existing release archive or checksum. Move or rename the existing output first."
}

$token = [Guid]::NewGuid().ToString("N")
$workDirectory = $null
$verificationDirectory = $null
$candidateOutputLeaf = ".$PackageName.$token.candidate.zip"
$candidateChecksumLeaf = ".$PackageName.$token.candidate.zip.sha256"
$candidateOutputPath = Join-Path $output $candidateOutputLeaf
$candidateChecksumPath = Join-Path $output $candidateChecksumLeaf
$publishedArchiveCreated = $false
$publishedChecksumCreated = $false
$releasePublished = $false

try {
    $workDirectory = New-OwnedDirectory `
        -Parent $work `
        -Leaf ("megaglitch-python-release-" + $token) `
        -Token $token `
        -Purpose "release-work"
    $verificationDirectory = New-OwnedDirectory `
        -Parent $verification `
        -Leaf ("megaglitch-python-verify-" + $token) `
        -Token $token `
        -Purpose "release-verification"

    $metadataParent = Join-Path $workDirectory "metadata"
    $metadataRoot = Join-Path $metadataParent $artifactLeaf
    [void][IO.Directory]::CreateDirectory($metadataRoot)
    $manifestPath = Join-Path $metadataRoot "BUNDLE-SHA256.txt"
    $manifest = Write-BundleManifest -BundleRoot $artifact -Destination $manifestPath
    $executableHash = (Get-FileHash -LiteralPath $artifactExecutable -Algorithm SHA256).Hash.ToLowerInvariant()

    $verifier = Join-Path $PSScriptRoot "verify.ps1"
    if (-not (Test-Path -LiteralPath $verifier -PathType Leaf)) {
        throw "Missing Python-host release verifier: $verifier"
    }
    & $verifier `
        -PackageRoot $artifact `
        -VerificationContext source `
        -ExpectedExecutableSha256 $executableHash `
        -ExpectedSourceCommit $commit `
        -ExpectedSpecSha256 $sourceIdentityHashes["spec"] `
        -ExpectedRuntimeHookSha256 $sourceIdentityHashes["runtime_hook"] `
        -ExpectedExternalToolsSha256 $sourceIdentityHashes["external_tools"] `
        -ExpectedOpenCvManifestSha256 $sourceIdentityHashes["opencv_manifest"] `
        -ExpectedCudaManifestSha256 $sourceIdentityHashes["cuda_manifest"] `
        -RequireBundledFfmpeg `
        -DiagnosticTimeoutSeconds $DiagnosticTimeoutSeconds | Out-Host

    # Diagnostics are forbidden from writing into the immutable onedir. Catch
    # added files as well as modifications before any candidate archive exists.
    Assert-BundleMatchesManifest -BundleRoot $artifact -ManifestPath $manifestPath

    $buildMetadataPath = Join-Path $artifactInternal "python-host-build.json"
    try {
        $buildMetadata = Get-Content -LiteralPath $buildMetadataPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Verified Python-host metadata could not be read for BUILD-INFO: $buildMetadataPath"
    }
    $buildMetadataHash = (Get-FileHash -LiteralPath $buildMetadataPath -Algorithm SHA256).Hash.ToLowerInvariant()

    $buildTime = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $sevenZipVersion = (Get-Item -LiteralPath $sevenZipExecutable).VersionInfo.ProductVersion
    if ([string]::IsNullOrWhiteSpace($sevenZipVersion)) {
        $sevenZipVersion = "unknown ($sevenZipExecutable)"
    }
    $buildInfo = @(
        "Product: MegaGlitchNode"
        "Release: 00.9.4 Mycotroph Render Quality"
        "Artifact: Full Python / Dear PyGui host"
        "Platform: Windows x64"
        "Packaging source commit: $commit"
        "Packaging source tree: clean"
        "Frozen build source commit: $($buildMetadata.source_commit)"
        "Frozen build source tree: clean"
        "Python-host metadata SHA-256: $buildMetadataHash"
        "Python host identity: $($buildMetadata.python_cache_tag) / Python $($buildMetadata.python_version) / PyInstaller $($buildMetadata.pyinstaller_version)"
        "PyInstaller spec SHA-256: $($buildMetadata.spec_sha256)"
        "Runtime hook SHA-256: $($buildMetadata.runtime_hook_sha256)"
        "External tool resolver SHA-256: $($buildMetadata.external_tools_sha256)"
        "cv2 identity: $($buildMetadata.cv2_extension) / $($buildMetadata.cv2_sha256)"
        "OpenCV runtime count: $($buildMetadata.opencv_dll_count)"
        "OpenCV manifest SHA-256: $($buildMetadata.opencv_preload_manifest_sha256)"
        "CUDA runtime count: $($buildMetadata.cuda_runtime_dll_count)"
        "CUDA manifest SHA-256: $($buildMetadata.cuda_runtime_manifest_sha256)"
        "FFmpeg distribution: $($buildMetadata.ffmpeg_distribution)"
        "FFmpeg source archive SHA-256: $($buildMetadata.ffmpeg_source_archive_sha256)"
        "FFmpeg version: $($buildMetadata.ffmpeg_version)"
        "FFmpeg SHA-256: $($buildMetadata.ffmpeg_sha256)"
        "FFprobe version: $($buildMetadata.ffprobe_version)"
        "FFprobe SHA-256: $($buildMetadata.ffprobe_sha256)"
        "FFmpeg license SHA-256: $($buildMetadata.ffmpeg_license_sha256)"
        "FFmpeg README SHA-256: $($buildMetadata.ffmpeg_readme_sha256)"
        "Package time (UTC): $buildTime"
        "Archive root: $artifactLeaf"
        "MegaGlitchNode.exe SHA-256: $executableHash"
        "Bundle manifest SHA-256: $($manifest.ManifestSha256)"
        "Bundle file count: $($manifest.FileCount)"
        "Bundle bytes before release metadata: $($manifest.TotalBytes)"
        "Runtime backend verified: python"
        "Frame transport verified: shared"
        "Source diagnostics: --diagnose-only, --diagnose-worker"
        "Fresh extraction diagnostics: --diagnose-only, --diagnose-worker"
        "Offline master tools verified: executed ffmpeg.exe -hide_banner -encoders and ffprobe.exe -version"
        "Required bundled encoders verified: libx264, libx265, mpeg4, ffv1"
        "Archive writer: $sevenZipVersion"
        "ZIP64: automatic 7-Zip ZIP64 extensions when classic ZIP limits are crossed"
        "Artifact provenance: exact file identities are recorded in BUNDLE-SHA256.txt"
    )
    Set-Content -LiteralPath (Join-Path $metadataRoot "BUILD-INFO.txt") -Value $buildInfo -Encoding UTF8

    $candidateArchive = Join-Path $workDirectory ($PackageName + ".candidate.zip")
    $sevenZipWork = Join-Path $workDirectory "7z-work"
    [void][IO.Directory]::CreateDirectory($sevenZipWork)

    # Archive the completed PyInstaller onedir in place. No second bundle-sized
    # staging copy is made; only the small release metadata overlay is staged.
    Invoke-SevenZip `
        -Executable $sevenZipExecutable `
        -WorkingDirectory (Split-Path -Path $artifact -Parent) `
        -Arguments @(
            "a", "-tzip", "-mx=9", "-mmt=on", "-mcu=on", "-sse", "-bd", "-y",
            ("-w" + $sevenZipWork), "--", $candidateArchive, $artifactLeaf
        ) `
        -Context "direct Python onedir archival (ZIP64-capable)"
    Invoke-SevenZip `
        -Executable $sevenZipExecutable `
        -WorkingDirectory $metadataParent `
        -Arguments @(
            "a", "-tzip", "-mx=9", "-mmt=on", "-mcu=on", "-sse", "-bd", "-y",
            ("-w" + $sevenZipWork), "--", $candidateArchive, $artifactLeaf
        ) `
        -Context "release metadata overlay"
    Invoke-SevenZip `
        -Executable $sevenZipExecutable `
        -WorkingDirectory $workDirectory `
        -Arguments @("t", "-tzip", "-bd", "--", $candidateArchive) `
        -Context "candidate archive integrity test"

    # Re-hash the original bundle after 7-Zip has read it. A concurrently
    # modified build is never published under the earlier manifest.
    Assert-BundleMatchesManifest -BundleRoot $artifact -ManifestPath $manifestPath
    [void](Get-CleanSourceCommit -RepositoryRoot $repositoryRoot -ExpectedCommit $commit)

    Invoke-SevenZip `
        -Executable $sevenZipExecutable `
        -WorkingDirectory $verificationDirectory `
        -Arguments @("x", "-tzip", "-aoa", "-bd", "-y", ("-o" + $verificationDirectory), "--", $candidateArchive) `
        -Context "fresh verification extraction"
    $extractedEntries = @(Get-ChildItem -LiteralPath $verificationDirectory -Force | Where-Object {
        $_.Name -ne ".megaglitch-release-owner.json"
    })
    if (
        $extractedEntries.Count -ne 1 -or
        -not $extractedEntries[0].PSIsContainer -or
        $extractedEntries[0].Name -ne $artifactLeaf
    ) {
        throw "Archive must extract to exactly one '$artifactLeaf' onedir."
    }
    $extractedPackage = $extractedEntries[0].FullName
    & $verifier `
        -PackageRoot $extractedPackage `
        -VerificationContext extracted `
        -ExpectedExecutableSha256 $executableHash `
        -ExpectedSourceCommit $commit `
        -ExpectedSpecSha256 $sourceIdentityHashes["spec"] `
        -ExpectedRuntimeHookSha256 $sourceIdentityHashes["runtime_hook"] `
        -ExpectedExternalToolsSha256 $sourceIdentityHashes["external_tools"] `
        -ExpectedOpenCvManifestSha256 $sourceIdentityHashes["opencv_manifest"] `
        -ExpectedCudaManifestSha256 $sourceIdentityHashes["cuda_manifest"] `
        -RequireBundledFfmpeg `
        -DiagnosticTimeoutSeconds $DiagnosticTimeoutSeconds | Out-Host

    # Complete all recursive cleanup before publication. A cleanup ownership or
    # reparse-point failure therefore cannot leave a release that looks valid.
    Remove-OwnedDirectory `
        -Path $verificationDirectory `
        -Parent $verification `
        -Token $token `
        -Purpose "release-verification"
    $verificationDirectory = $null

    if (
        (Test-Path -LiteralPath $archivePath) -or
        (Test-Path -LiteralPath $checksumPath) -or
        (Test-Path -LiteralPath $candidateOutputPath) -or
        (Test-Path -LiteralPath $candidateChecksumPath)
    ) {
        throw "Output paths changed during verification; refusing to overwrite them."
    }

    [IO.File]::Copy($candidateArchive, $candidateOutputPath, $false)
    $workArchiveHash = (Get-FileHash -LiteralPath $candidateArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    $outputArchiveHash = (Get-FileHash -LiteralPath $candidateOutputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($outputArchiveHash -ne $workArchiveHash) {
        throw "Candidate archive changed while being copied to OutputRoot."
    }
    Set-Content `
        -LiteralPath $candidateChecksumPath `
        -Value ("$outputArchiveHash  " + [IO.Path]::GetFileName($archivePath)) `
        -Encoding ASCII

    Remove-OwnedDirectory `
        -Path $workDirectory `
        -Parent $work `
        -Token $token `
        -Purpose "release-work"
    $workDirectory = $null
    [void](Get-CleanSourceCommit -RepositoryRoot $repositoryRoot -ExpectedCommit $commit)
    if ((Test-Path -LiteralPath $archivePath) -or (Test-Path -LiteralPath $checksumPath)) {
        throw "Final output paths appeared before publication; refusing to overwrite them."
    }

    [IO.File]::Move($candidateOutputPath, $archivePath)
    $publishedArchiveCreated = $true
    [IO.File]::Move($candidateChecksumPath, $checksumPath)
    $publishedChecksumCreated = $true
    $releasePublished = $true

    Write-Output ([pscustomobject]@{
        Passed = $true
        Archive = $archivePath
        Checksum = $checksumPath
        ArchiveSha256 = $outputArchiveHash
        SourceCommit = $commit
        ArtifactRoot = $artifact
        ArtifactExecutableSha256 = $executableHash
        ArtifactManifestSha256 = $manifest.ManifestSha256
        ArtifactFileCount = $manifest.FileCount
        BuildMetadataSha256 = $buildMetadataHash
        BuildSourceCommit = $buildMetadata.source_commit
        Cv2Sha256 = $buildMetadata.cv2_sha256
        FfmpegSha256 = $buildMetadata.ffmpeg_sha256
        FfprobeSha256 = $buildMetadata.ffprobe_sha256
        SevenZip = $sevenZipExecutable
        Zip64 = "automatic"
        SourceVerified = $true
        FreshExtractionVerified = $true
        RuntimeBackend = "python"
    })
}
finally {
    if (-not $releasePublished) {
        if ($publishedChecksumCreated) {
            Remove-OwnedOutputFile -Path $checksumPath -OutputDirectory $output -ExpectedLeaf ([IO.Path]::GetFileName($checksumPath))
        }
        if ($publishedArchiveCreated) {
            Remove-OwnedOutputFile -Path $archivePath -OutputDirectory $output -ExpectedLeaf ([IO.Path]::GetFileName($archivePath))
        }
    }
    Remove-OwnedOutputFile -Path $candidateChecksumPath -OutputDirectory $output -ExpectedLeaf $candidateChecksumLeaf
    Remove-OwnedOutputFile -Path $candidateOutputPath -OutputDirectory $output -ExpectedLeaf $candidateOutputLeaf
    if ($null -ne $verificationDirectory) {
        Remove-OwnedDirectory `
            -Path $verificationDirectory `
            -Parent $verification `
            -Token $token `
            -Purpose "release-verification"
    }
    if ($null -ne $workDirectory) {
        Remove-OwnedDirectory `
            -Path $workDirectory `
            -Parent $work `
            -Token $token `
            -Purpose "release-work"
    }
}
