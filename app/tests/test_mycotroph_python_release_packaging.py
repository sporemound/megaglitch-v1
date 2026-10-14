from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPO_ROOT / "packaging" / "mycotroph-python-release"
BUILD_SCRIPT = RELEASE_ROOT / "build.ps1"
VERIFY_SCRIPT = RELEASE_ROOT / "verify.ps1"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [BUILD_SCRIPT, VERIFY_SCRIPT])
def test_release_powershell_is_syntactically_valid(script: Path) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    command = (
        "$tokens=$null; $errors=$null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{script}',"
        "[ref]$tokens,[ref]$errors); "
        "if ($errors.Count -ne 0) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_wrapper_has_explicit_external_roots_and_refuses_unsafe_publication() -> None:
    script = _text(BUILD_SCRIPT)
    for parameter in ("ArtifactRoot", "OutputRoot", "WorkRoot", "VerificationRoot"):
        assert f"[string]${parameter}" in script
    assert "status --porcelain=v1 --untracked-files=all" in script
    assert "Refusing to package a dirty source tree" in script
    assert "Refusing to overwrite an existing release archive or checksum" in script
    assert "Assert-OutsideArtifact" in script
    assert "Get-CleanSourceCommit -RepositoryRoot $repositoryRoot -ExpectedCommit $commit" in script


def test_wrapper_archives_onedir_directly_with_7zip_zip64_capability() -> None:
    script = _text(BUILD_SCRIPT)
    assert "Compress-Archive" not in script
    assert "Expand-Archive" not in script
    assert "Copy-Item" not in script
    assert "direct Python onedir archival (ZIP64-capable)" in script
    assert '"a", "-tzip"' in script
    assert '"t", "-tzip"' in script
    assert '"x", "-tzip"' in script
    assert "automatic 7-Zip ZIP64 extensions" in script
    assert "[IO.File]::Copy($candidateArchive, $candidateOutputPath, $false)" in script


def test_wrapper_writes_provenance_manifest_and_archive_checksum() -> None:
    script = _text(BUILD_SCRIPT)
    for required in (
        "BUILD-INFO.txt",
        "BUNDLE-SHA256.txt",
        "MegaGlitchNode.exe SHA-256",
        "Bundle manifest SHA-256",
        "Python-host metadata SHA-256",
        "Frozen build source commit",
        "FFmpeg distribution: $($buildMetadata.ffmpeg_distribution)",
        "FFmpeg source archive SHA-256: $($buildMetadata.ffmpeg_source_archive_sha256)",
        "FFmpeg version: $($buildMetadata.ffmpeg_version)",
        "FFmpeg SHA-256: $($buildMetadata.ffmpeg_sha256)",
        "FFprobe version: $($buildMetadata.ffprobe_version)",
        "FFprobe SHA-256: $($buildMetadata.ffprobe_sha256)",
        "FFmpeg license SHA-256: $($buildMetadata.ffmpeg_license_sha256)",
        "FFmpeg README SHA-256: $($buildMetadata.ffmpeg_readme_sha256)",
        '.zip.sha256',
        "Get-FileHash -LiteralPath $candidateArchive -Algorithm SHA256",
    ):
        assert required in script


def test_wrapper_rejects_files_added_by_source_diagnostics_before_archiving() -> None:
    script = _text(BUILD_SCRIPT)
    assert "Source artifact file set changed during packaging" in script
    assert "Source artifact gained a file during packaging" in script

    source_diagnostic = script.index("-VerificationContext source")
    post_diagnostic_guard = script.index(
        "Assert-BundleMatchesManifest -BundleRoot $artifact", source_diagnostic
    )
    archive = script.index("direct Python onedir archival", source_diagnostic)
    assert source_diagnostic < post_diagnostic_guard < archive


def test_wrapper_verifies_source_and_fresh_extraction_with_python_backend() -> None:
    build = _text(BUILD_SCRIPT)
    verify = _text(VERIFY_SCRIPT)
    assert "-VerificationContext source" in build
    assert "-VerificationContext extracted" in build
    assert '"--diagnose-only"' in verify
    assert '"--diagnose-worker"' in verify
    diagnostic_function = verify[verify.index("function Invoke-FrozenDiagnostic") :]
    assert "$startInfo.CreateNoWindow = $true" in diagnostic_function
    assert 'EnvironmentVariables["MEGAGLITCH_RUNTIME_BACKEND"] = "python"' in verify
    assert 'EnvironmentVariables["MEGAGLITCH_FRAME_TRANSPORT"] = "shared"' in verify
    assert 'reset_round_trip' in verify
    assert '$verifiedFrameTransport -ne "shared_v1"' in verify
    assert 'frame_round_trip' in verify
    assert 'input_transport' in verify
    assert 'output_transport' in verify
    assert '266d05fcd17ddd9f96f5cb7eae3c9f486fd5697a2b0198586a82af2cdc7c0558' in verify
    assert '[string]$workerLifecycle.state -ne "stopped"' in verify
    assert '$null -eq $workerLifecycle.exitcode' in verify
    assert '[int]$workerLifecycle.exitcode -ne 0' in verify
    assert '$workerLifecycle.forced_termination' in verify
    assert '$workerLifecycle.status_channel_error' in verify
    assert '[int64]$sharedTransportStats.input_frames_written -ne 1' in verify
    assert '[int64]$sharedTransportStats.output_frames_read -ne 1' in verify
    assert '[int64]$sharedTransportStats.input_slots_in_use -ne 0' in verify
    assert '[int64]$sharedTransportStats.output_slots_in_use -ne 0' in verify
    assert 'FrameTransport = $verifiedFrameTransport' in verify
    assert "Assert-BundleManifest" in verify
    assert "Freshly extracted package is missing BUILD-INFO.txt" in verify
    assert 'Join-Path $internalRoot "bin\\ffmpeg"' in verify


def test_wrapper_binds_frozen_metadata_to_current_clean_sources() -> None:
    build = _text(BUILD_SCRIPT)
    verify = _text(VERIFY_SCRIPT)
    for source_path in (
        "packaging\\pyinstaller\\megaglitch.spec",
        "packaging\\pyinstaller\\runtime_hook.py",
        "megaglitch\\external_tools.py",
        "packaging\\pyinstaller\\opencv-preload-manifest.json",
        "packaging\\pyinstaller\\cuda-runtime-manifest.json",
    ):
        assert source_path in build
    for expected_parameter in (
        "ExpectedSourceCommit",
        "ExpectedSpecSha256",
        "ExpectedRuntimeHookSha256",
        "ExpectedExternalToolsSha256",
        "ExpectedOpenCvManifestSha256",
        "ExpectedCudaManifestSha256",
    ):
        assert build.count(f"-{expected_parameter}") == 2
        assert f"[string]${expected_parameter}" in verify
    assert "source_dirty" in verify
    assert "Python-host artifact metadata records a dirty source build" in verify
    assert "source_commit" in verify


def test_verifier_checks_metadata_schema_and_frozen_runtime_identities() -> None:
    verify = _text(VERIFY_SCRIPT)
    for required in (
        "megaglitch.python-host-build/v1",
        "megaglitch-python-host",
        "python-host-build.json",
        "cv2_extension",
        "cv2_sha256",
        "opencv_dll_count",
        "opencv_preload_manifest_sha256",
        "cuda_runtime_dll_count",
        "cuda_runtime_manifest_sha256",
        "ffmpeg_sha256",
        "ffprobe_sha256",
        "spec_sha256",
        "runtime_hook_sha256",
        "external_tools_sha256",
    ):
        assert required in verify
    assert "Assert-RuntimeDirectoryIdentity" in verify
    assert "Bundled cv2 identity does not match" in verify
    assert 'Label = "ffmpeg.exe"' in verify
    assert 'Label = "ffprobe.exe"' in verify
    assert "Bundled $($toolIdentity.Label) identity does not match" in verify


def test_verifier_distinguishes_preload_subset_from_complete_runtime_manifest() -> None:
    verify = _text(VERIFY_SCRIPT)
    assert "[switch]$ManifestIsPreloadSubset" in verify
    opencv_call = next(
        line for line in verify.splitlines() if 'Label "OpenCV runtime"' in line
    )
    cuda_call = next(
        line for line in verify.splitlines() if 'Label "CUDA runtime"' in line
    )
    assert "-ManifestIsPreloadSubset" in opencv_call
    assert "-ManifestIsPreloadSubset" not in cuda_call
    assert "$ExpectedNames.Count -gt $ExpectedCount" in verify


def test_verifier_rejects_unrelated_qt_before_metadata_and_diagnostics() -> None:
    verify = _text(VERIFY_SCRIPT)
    assert "function Assert-NoUnrelatedQtRuntime" in verify
    assert '"(?i)^Qt[56].*\\.dll$"' in verify
    for binding in ("PyQt5", "PyQt6", "PySide2", "PySide6"):
        assert binding in verify

    guard_call = verify.index(
        "Assert-NoUnrelatedQtRuntime -InternalRoot $internalRoot"
    )
    metadata_call = verify.index(
        "$buildMetadata = Assert-PythonHostBuildMetadata", guard_call
    )
    diagnostic_call = verify.index("$runtimeDiagnostic = Invoke-FrozenDiagnostic")
    assert guard_call < metadata_call < diagnostic_call


def test_no_unrelated_qt_guard_is_executable_and_rejects_orphans(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")

    verify = _text(VERIFY_SCRIPT)
    function_start = verify.index("function Assert-NoUnrelatedQtRuntime")
    function_end = verify.index(
        "\nfunction Assert-PythonHostBuildMetadata", function_start
    )
    fixture = tmp_path / "no-unrelated-qt-fixture.ps1"
    fixture.write_text(
        "param([Parameter(Mandatory = $true)][string]$InternalRoot)\n"
        "$ErrorActionPreference = 'Stop'\n"
        + verify[function_start:function_end]
        + "\nAssert-NoUnrelatedQtRuntime -InternalRoot $InternalRoot\n",
        encoding="utf-8",
    )

    def run(internal_root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(fixture),
                "-InternalRoot",
                str(internal_root),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    clean = tmp_path / "clean-internal"
    clean.mkdir()
    (clean / "dearpygui.pyd").write_bytes(b"fixture")
    clean_result = run(clean)
    assert clean_result.returncode == 0, clean_result.stdout + clean_result.stderr

    orphan = tmp_path / "orphan-internal"
    orphan.mkdir()
    (orphan / "Qt6Core.dll").write_bytes(b"fixture")
    orphan_result = run(orphan)
    assert orphan_result.returncode != 0
    assert "unrelated Qt runtime" in orphan_result.stdout + orphan_result.stderr

    binding = tmp_path / "binding-internal" / "PySide6" / "plugins"
    binding.mkdir(parents=True)
    (binding / "qwindows.dll").write_bytes(b"fixture")
    binding_result = run(binding.parents[1])
    assert binding_result.returncode != 0
    assert "unrelated Qt runtime" in binding_result.stdout + binding_result.stderr


def test_runtime_identity_subset_is_executable_and_rejects_escape(tmp_path: Path) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")

    verify = _text(VERIFY_SCRIPT)
    function_start = verify.index("function Assert-RuntimeDirectoryIdentity")
    function_end = verify.index("\nfunction Assert-PythonHostBuildMetadata", function_start)
    fixture = tmp_path / "runtime-identity-fixture.ps1"
    fixture.write_text(
        "param([Parameter(Mandatory = $true)][string]$InternalRoot, [switch]$Subset)\n"
        "$ErrorActionPreference = 'Stop'\n"
        + verify[function_start:function_end]
        + "\n$directory = Join-Path $InternalRoot 'bin\\opencv'\n"
        "$arguments = @{ InternalRoot = $InternalRoot; Directory = $directory; "
        "ExpectedNames = @('opencv_a.dll', 'opencv_b.dll'); ExpectedCount = 3; "
        "Filter = 'opencv_*.dll'; Label = 'OpenCV runtime' }\n"
        "if ($Subset) { Assert-RuntimeDirectoryIdentity @arguments -ManifestIsPreloadSubset }\n"
        "else { Assert-RuntimeDirectoryIdentity @arguments }\n",
        encoding="utf-8",
    )

    internal = tmp_path / "internal"
    opencv = internal / "bin" / "opencv"
    opencv.mkdir(parents=True)
    for name in ("opencv_a.dll", "opencv_b.dll", "opencv_c.dll"):
        (opencv / name).write_bytes(b"fixture")

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(fixture),
                "-InternalRoot",
                str(internal),
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    subset = run("-Subset")
    assert subset.returncode == 0, subset.stdout + subset.stderr

    exact = run()
    assert exact.returncode != 0
    assert "manifest coverage mismatch" in exact.stdout + exact.stderr

    escaped = internal / "escaped"
    escaped.mkdir()
    (escaped / "opencv_a.dll").write_bytes(b"duplicate")
    duplicate = run("-Subset")
    assert duplicate.returncode != 0
    assert "duplicated outside" in duplicate.stdout + duplicate.stderr


def test_verifier_executes_bundled_media_tools_and_requires_mastering_codecs() -> None:
    verify = _text(VERIFY_SCRIPT)
    assert '-Arguments "-hide_banner -encoders"' in verify
    assert '-Arguments "-version"' in verify
    assert '@("libx264", "libx265", "mpeg4", "ffv1")' in verify
    assert "ReadToEndAsync" in verify
    assert "Get-BoundedProcessText" in verify
    assert 'EnvironmentVariables["PATH"]' in verify
    assert "BundledVideoEncoders" in verify


def test_verifier_scrubs_development_environment_before_processes() -> None:
    verify = _text(VERIFY_SCRIPT)
    for prefix in ("PYTHON", "CONDA", "VIRTUAL_ENV", "OPENCV", "CUDA", "UI_BRIDGE"):
        assert prefix in verify
    assert "Set-StandaloneProcessEnvironment -StartInfo $startInfo -MegaGlitchRuntime" in verify
    assert 'EnvironmentVariables["MEGAGLITCH_RUNTIME_BACKEND"] = "python"' in verify
    assert 'EnvironmentVariables["MEGAGLITCH_FRAME_TRANSPORT"] = "shared"' in verify


def test_recursive_cleanup_is_marker_and_boundary_guarded() -> None:
    script = _text(BUILD_SCRIPT)
    assert "Remove-OwnedDirectory" in script
    assert ".megaglitch-release-owner.json" in script
    assert "Assert-StrictChildPath" in script
    assert "ReparsePoint" in script
    assert "marker.token -ne $Token" in script
    recursive_deletes = [
        line.strip()
        for line in script.splitlines()
        if "Remove-Item" in line and "-Recurse" in line
    ]
    assert recursive_deletes == ["Remove-Item -LiteralPath $resolvedPath -Recurse -Force"]
