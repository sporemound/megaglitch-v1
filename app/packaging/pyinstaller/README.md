# MegaGlitch PyInstaller bundle

The production spec is `packaging/pyinstaller/megaglitch.spec`. It is environment-configurable and validates the custom OpenCV and native artifacts before analysis.

```powershell
$env:MEGAGLITCH_PYTHON='C:\path\to\python.exe'
$env:MEGAGLITCH_OPENCV_BIN='C:\src\opencv-cuda\build_py312\bin\Release'
$env:MEGAGLITCH_OPENCV_PYTHON='C:\src\opencv-cuda\build_py312\lib\python3\Release'
$env:MEGAGLITCH_CUDA_BIN='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
$env:MEGAGLITCH_FFMPEG_BIN='V:\MegaGlitch-Render-Quality-20260827\ffmpeg\ffmpeg-9.0.1-essentials_build\bin'
$env:MEGAGLITCH_FFMPEG_SOURCE_ARCHIVE_SHA256='FEC81AE03971D9DD4BE3EBE02E263BD2EC1D789483F931BDBA5F5715E65DA2E9'
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File packaging\pyinstaller\build.ps1 `
  -Python $env:MEGAGLITCH_PYTHON -WithRust `
  -ArtifactRoot 'V:\MegaGlitch-build'
```

With `-ArtifactRoot`, the one-directory bundle is written to
`V:\MegaGlitch-build\dist\MegaGlitchNode\`; omitting it preserves the legacy
`dist\MegaGlitchNode\` location. `-DistRoot` and `-WorkRoot` may override the
two roots independently. This is important for multi-gigabyte builds on a
workspace drive with limited free space.

The bundle includes the exact dependency-first OpenCV and CUDA Toolkit closure,
the custom `cv2` extension, the native temporal/P-frame extension, optional
experimental Rust graph runtime, audited FFmpeg/FFprobe executables, the
Quantum Gestalt model, presets, shaders, Lua assets, and SVG templates. UPX is
disabled. `MEGAGLITCH_FFMPEG_BIN` is mandatory and must identify an audited
x64 build providing `libx264`, `libx265`, `mpeg4`, and `ffv1`. Prefer a static
distribution whose parent retains its root `LICENSE` and `README.txt`; copying
only two executables is not a complete runtime. The source-archive SHA-256 is
also mandatory. The current audited input is Gyan FFmpeg 9.0.1 essentials,
source ZIP SHA-256
`FEC81AE03971D9DD4BE3EBE02E263BD2EC1D789483F931BDBA5F5715E65DA2E9`.

Analysis writes `_internal\python-host-build.json` with the exact Git commit,
dirty flag, Python/PyInstaller identity, FFmpeg/FFprobe version banners, and
hashes for the selected cv2, media tools, media license/README, source archive,
manifests, spec, runtime hook, and frozen tool resolver. The release wrapper
rejects a dirty build, stale commit, or mismatched bundled identity.

Validate the import-sensitive frozen runtime without opening the UI:

```powershell
V:\MegaGlitch-build\dist\MegaGlitchNode\MegaGlitchNode.exe --diagnose-only
V:\MegaGlitch-build\dist\MegaGlitchNode\MegaGlitchNode.exe --diagnose-worker
```

The first command checks packaged native imports and CUDA visibility. The
second starts the real Python processing worker, runs its CUDA self-tests,
waits for lifecycle readiness, and shuts it down without creating the UI.
Neither replaces a normal-launch smoke that confirms the
`MegaGlitchFM Node Editor` window and worker remain responsive.

The spec deliberately excludes any installed `cv2` package, isolates the
custom OpenCV DLLs under `_internal/bin/opencv`, and validates the checked-in
preload manifest against the selected extension's direct PE imports. That
dependency-first preload manifest is an ordered subset of the complete modular
OpenCV runtime closure; the frozen metadata records and verifies the full
directory count separately. The spec recursively validates and stages the
checked-in CUDA closure under `_internal/bin/cuda`.
Because Dear PyGui does not use Qt, analysis also removes every `Qt5*.dll` or
`Qt6*.dll` binary and every PyQt/PySide destination subtree, even when an
unrelated development package causes PyInstaller to collect a DLL without a
bundled importer. Release verification repeats this invariant recursively and
rejects an orphan such as `_internal\Qt6Core.dll` before metadata checks or
frozen diagnostics run.
The runtime hook also registers sounddevice's bundled PortAudio directory so
audio/MIDI startup remains standalone under a system-only `PATH`. It preloads
and retains the selected normal or ASIO PortAudio DLL before application
imports; this also covers frozen workers created with `CREATE_NO_WINDOW`, where
CFFI's first load can fail with `ERROR_INVALID_HANDLE`. The same detached-worker
launch mode can reject SciPy `_fblas` while resolving its uniquely versioned
OpenBLAS DLL, so the hook validates, preloads, and retains that bundled runtime
before `scipy.signal` or `scipy.linalg` can initialize.
Driver-owned APIs such as `nvcuda.dll` remain NVIDIA display-driver
prerequisites. The frozen runtime never borrows a CUDA Toolkit directory from
the development machine.

Default GUI values `ffmpeg` and `ffprobe` resolve to absolute executables under
`_internal/bin/ffmpeg` only when frozen. Explicit paths entered in Offline
Master settings remain authoritative. The traditional datamosh/P-frame
corruption command is deliberately separate and remains a GUI-parameterized
external CLI rather than being invented or silently substituted at packaging
time.

The Rust Mycotroph functional-preview sidecar uses `mycotroph_engine.spec`. It validates the
AMD64 OpenCV/CUDA PE dependency closure against checked-in dependency-first
manifests, copies the required CUDA Toolkit redistributable DLLs under
`_internal/bin/cuda`, and preloads only that isolated directory. Set
`MEGAGLITCH_CUDA_BIN` while building to select the matching Toolkit `bin`
directory; the frozen runtime never consults it. Driver-owned APIs such as
`nvcuda.dll` are not bundled and require a compatible NVIDIA display driver.
The sidecar records its full source commit, dirty flag, bridge/protocol identity,
and source-manifest hashes in `_internal/engine-v2-build.json`.

That artifact is not the full MegaGlitch machine: its own README records the
missing Deck B, automation/modulation, audio/MIDI, recording/export, and virtual
camera surfaces. Do not use `packaging\mycotroph-alpha\build.ps1` for the full
Python-host render-quality release. Use the dedicated Python-host wrapper
documented alongside that package after building `megaglitch.spec`.

For the Rust functional-preview artifact only, package its sidecar with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File packaging\mycotroph-alpha\build.ps1 `
  -EngineBundle build\mycotroph-engine-dist\megaglitch-engine-v2 `
  -VerificationRoot 'V:\MegaGlitch-package-verification'
```

`VerificationRoot` is the parent under which the builder creates one uniquely
named extraction directory. The builder gates the source bundle, staged copy,
and fresh extraction; it removes only its marked extraction directory. Omit the
parameter to use the current user's temporary directory. Packaging is rejected
unless the repository is clean and the Engine V2 metadata names that exact
commit.

All Lupa Lua runtime extensions are collected explicitly because Lupa selects
its implementation dynamically by inspecting its on-disk package directory.
Frozen multiprocessing startup also installs safe null streams when a Windows
child has no inherited console and records otherwise-unhandled frozen
bootstrap failures as `%TEMP%\megaglitch-frozen-<pid>.log`.

Review NVIDIA and FFmpeg redistribution terms for every release profile and
test each full Python bundle on a clean driver-only Windows machine with a
Windows-system-only `PATH`. Do not distribute until the exact native DLL set,
Offline Master encode/readback, camera/audio integrations, virtual camera, and
CUDA behavior have been verified outside the development environment.

Full prerequisites, diagnostics, regression commands, troubleshooting, and release risks are in [the Windows production guide](../../docs/production/windows-build-and-release.md).

`megaglitch.spec.example` is retained only as historical reference; use `megaglitch.spec` for current builds.
