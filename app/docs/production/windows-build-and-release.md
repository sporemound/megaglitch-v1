# Windows build, diagnostics, and release guide

## Supported development target

- Windows 11 x64
- 64-bit CPython 3.12 (native extension artifacts are ABI-specific)
- Visual Studio 2022 Build Tools with MSVC and a Windows SDK
- NumPy, Cython, and setuptools for the C++/Cython extension
- Rust/Cargo only for the optional experimental runtime DLL
- Custom OpenCV 4.12 CUDA Python extension and modular DLL directory
- Current NVIDIA driver; CUDA Toolkit 12.6 for the development configuration

The custom OpenCV paths are configurable:

```powershell
$env:MEGAGLITCH_PYTHON='C:\path\to\python.exe'
$env:MEGAGLITCH_OPENCV_BIN='C:\src\opencv-cuda\build_py312\bin\Release'
$env:MEGAGLITCH_OPENCV_PYTHON='C:\src\opencv-cuda\build_py312\lib\python3\Release'
$env:MEGAGLITCH_CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:MEGAGLITCH_CUDA_BIN="$env:MEGAGLITCH_CUDA_PATH\bin"
```

## Dependencies

`requirements.txt` intentionally does not install PyPI OpenCV. This prevents `pip` from replacing or shadowing the custom CUDA extension.

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
```

For a CPU-only environment, explicitly use:

```powershell
python -m pip install -r requirements-cpu-opencv.txt
```

Do not install both custom OpenCV and `opencv-python` into the same release environment. The diagnostic reports the exact imported module path.

## Build, diagnose, and launch

Build the production native C++/Cython segment, validate dependencies, and launch:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_and_run.ps1 `
  -Python $env:MEGAGLITCH_PYTHON
```

Build the experimental Rust runtime too:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_and_run.ps1 `
  -Python $env:MEGAGLITCH_PYTHON -WithRust
```

Use `-Force` to force native recompilation and `-NoLaunch` for CI/build validation. `tools/run.ps1 -BuildNative` is a shorter development wrapper.

Individual builds remain available:

```powershell
powershell -ExecutionPolicy Bypass -File native\temporal_ring\build.ps1 -Python $env:MEGAGLITCH_PYTHON
powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1
```

The temporal build rejects non-3.12 or 32-bit Python before invoking the compiler. Cargo intermediates and C++ intermediates are placed under `%TEMP%`; final artifacts are copied into `megaglitch/`.

## Diagnostics

Run all current hardware checks:

```powershell
python tools\diagnose_runtime.py --json tools\diagnostics\latest.json `
  --require-temporal --require-rust --require-cuda --cuda-selftest
```

The report includes Python ABI/architecture, required packages, MSVC runtime DLLs, shared-memory round trip, native extension exports/path, Rust DLL loadability, exact OpenCV extension path, OpenCV DLL count, CUDA device count, and CUDA operation results.

Importing `megaglitchnode` no longer imports GPU modules eagerly. This is necessary: eager import previously loaded pip's CPU `cv2` before the bootstrap could select the custom CUDA extension. GPU symbols remain available through lazy package attributes.

## Regression checks

```powershell
python tools\benchmark\check_regressions.py `
  --json tools\diagnostics\regression-check.json
python -m unittest tools.benchmark.test_benchmark `
  tests.test_visual_regressions tests.test_runtime_hardening
python -m unittest tests.test_worker_lifecycle tests.test_shared_frame_transport `
  tests.test_native_temporal tests.test_native_runtime tests.test_deck_crossfader
python test_pipeline.py
python scripts\genesis_vdp_smoke_test.py
```

`regression_thresholds.json` is intentionally tied to the recorded i9-13950HX/RTX 4060/CPython 3.12 profile. It gates exact native output, shared-worker mean/p99, lifecycle readiness/first-frame/shutdown time, forced termination, deterministic hashes, and retained slots. Establish a separate threshold profile for materially different hardware rather than weakening this one.

The checker is strict by default. Its direct invocation remains the dedicated
Rust qualification gate, so Graph IR absolute and Python-relative timing limits
still fail the command. The full Python-host `-ReleaseGate` invokes the same
threshold profile with `--rust-performance-advisory`: Python/shared production
metrics and Rust pixel equality, output hash, backend identity, and no-drop
behavior remain hard failures, while only experimental Rust mean/p99 and ratio
misses are emitted as advisories. The thresholds are not relaxed or copied into
a softer profile.

This severity split follows the supported host decision: production remains
Python/shared and Rust remains explicitly selected. A Rust timing advisory
blocks performance/promotion claims for the optional backend, not a
pixel-identical Python-host release. Missing or malformed Rust evidence is
still a release-gate failure when the runtime is included.

The release gate also records the deterministic master-resolution quality graph
with true-acceleration FPS semantics into its selected artifact root. This is a
correctness/telemetry gate; it does not claim a speed improvement and is kept
separate from the hardware-specific regression thresholds.

The spawned-worker compatibility soak also samples steady-state RSS and Windows handle count when `psutil` is available. Sixty graph-switching/reset frames must remain below 64 MiB RSS growth and eight additional handles while all shared slots return to the pool. This is a focused leak sentinel, not a substitute for an overnight hardware soak.

## PyInstaller bundle

Install PyInstaller through `requirements-build.txt`, then run:

```powershell
$env:MEGAGLITCH_FFMPEG_BIN='V:\MegaGlitch-Render-Quality-20260827\ffmpeg\ffmpeg-9.0.1-essentials_build\bin'
$env:MEGAGLITCH_FFMPEG_SOURCE_ARCHIVE_SHA256='FEC81AE03971D9DD4BE3EBE02E263BD2EC1D789483F931BDBA5F5715E65DA2E9'
powershell -ExecutionPolicy Bypass -File packaging\pyinstaller\build.ps1 `
  -Python $env:MEGAGLITCH_PYTHON -WithRust `
  -ArtifactRoot 'V:\MegaGlitch-build'
```

`-ArtifactRoot` moves PyInstaller distribution/work files, diagnostics, pytest
temporary files, and release benchmark results off the repository volume.
`-DistRoot` and `-WorkRoot` override the PyInstaller roots independently. The
checked-in `megaglitch.spec` fails early if OpenCV DLLs, the custom `cv2`
extension, the exact checked-in CUDA closure, native temporal extension, or
build-configured FFmpeg/FFprobe tools are missing. It collects the Quantum
Gestalt model and established lazy effect modules in addition to presets, node
presets, shaders, Lua assets, and SVG templates. UPX is disabled to avoid
native/CUDA startup and antivirus variability.

Every onedir carries `_internal\python-host-build.json`, recording the exact
source commit and dirty flag plus hashes for the spec, runtime hook, tool
resolver, cv2, manifests, FFmpeg tools, and media license/README. It records
both tool version banners and the verified source-archive SHA-256. The current
audited Gyan FFmpeg 9.0.1 essentials ZIP SHA-256 is
`FEC81AE03971D9DD4BE3EBE02E263BD2EC1D789483F931BDBA5F5715E65DA2E9`.

Run the noninteractive frozen-runtime gate after every bundle build:

```powershell
V:\MegaGlitch-build\dist\MegaGlitchNode\MegaGlitchNode.exe --diagnose-only
V:\MegaGlitch-build\dist\MegaGlitchNode\MegaGlitchNode.exe --diagnose-worker
```

On July 14, 2026, the development-machine bundle passed this gate with the
bundled custom OpenCV 4.12 extension, one CUDA device, the native temporal
extension, and the Rust runtime. The audited one-directory output contained
4,591 files and was 3.04 GiB: one root `cv2` extension, 65 OpenCV DLLs only in
`_internal/bin/opencv`, no nested CPU `cv2`, one native temporal extension,
and one Rust runtime DLL. The size is inflated by optional legacy
Quantum Gestalt/Torch dependencies and is a release-size optimization target,
not evidence for changing the frame-runtime architecture.

The later full-launch audit found and corrected two issues that the import-only
gate could not detect: Lupa's dynamically selected `.pyd` files were absent,
and frozen multiprocessing children could start with `sys.stdout` or
`sys.stderr` set to `None`. The corrected development bundle passes both
diagnostic commands. A controlled normal launch remained alive with a
responsive `MegaGlitchFM Node Editor` window and a separate worker process;
the worker passed CUDA self-tests and reported lifecycle readiness. The splash
screen remains optional and is skipped with a warning when this Miniforge
environment's Tk installation is unavailable.

PyInstaller must not auto-collect a second installed `cv2` package or place
alternate OpenCV/CUDA dependency copies in `_internal`. The spec verifies the
dependency-first OpenCV and CUDA manifests against the selected AMD64 PE files,
then stages only that explicit closure under `_internal\bin`. The custom runtime
hook preloads it before importing `cv2`; the bootstrap retains Windows
DLL-directory and module handles for process lifetime. Driver-owned APIs remain
NVIDIA display-driver prerequisites, but the frozen app does not require the
development CUDA Toolkit on `PATH`.

The strict release verifier launches the frozen diagnostics with Windows
`CREATE_NO_WINDOW` and a system-only `PATH`. Under that detached launch mode,
normal first-load paths can fail with `ERROR_INVALID_HANDLE` even though their
wheel-owned binaries are valid. The runtime hook therefore validates and
retains early handles for SciPy's unique `libscipy_openblas*.dll` and the
selected sounddevice PortAudio variant before importing application modules.
Do not weaken this verifier to a console-attached launch; both source-bundle and
fresh-extraction diagnostics must exercise the detached worker path.

The full bundle also contains audited FFmpeg and FFprobe executables under
`_internal\bin\ffmpeg`. Frozen sentinel settings resolve there by absolute path;
user-entered GUI paths remain unchanged. The FFmpeg build must provide
`libx264`, `libx265`, `mpeg4`, and `ffv1`. The distribution `LICENSE` and
`README.txt` are retained under `_internal\licenses\ffmpeg`. The
codec-corruption executable remains a separate GUI-parameterized external tool.

The development-machine pass does not close the clean-machine gate. Before
release, test a freshly extracted archive under a Windows-system-only `PATH`,
run both diagnostic commands, perform a direct Offline Master encode/readback,
then exercise camera/audio input, recording, virtual camera, representative
CPU/CUDA/GLSL/Lua graphs, project and preset loading, worker restart, and normal
shutdown. Use a ZIP64-capable archiver such as 7-Zip for the multi-gigabyte
onedir; PowerShell `Compress-Archive` is not the release path.

After those gates pass from a clean, committed tree, create the full
Python-host Mycotroph archive without a second bundle-sized staging copy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File packaging\mycotroph-python-release\build.ps1 `
  -ArtifactRoot 'V:\MegaGlitch-build\dist\MegaGlitchNode' `
  -OutputRoot 'V:\MegaGlitch-release' `
  -WorkRoot 'V:\MegaGlitch-release-work' `
  -VerificationRoot 'E:\MegaGlitch-release-verify'
```

The wrapper uses 7-Zip's ZIP64-capable writer, refuses existing outputs, records
the exact source commit and bundle hashes, and repeats the Python-host frozen
diagnostics after a fresh extraction. It is separate from
`packaging\mycotroph-alpha`, which remains the limited Rust functional-preview
release path and must not be used for this full-machine artifact.

## Backend selection and rollback

Production defaults remain:

```powershell
$env:MEGAGLITCH_RUNTIME_BACKEND='python'
$env:MEGAGLITCH_FRAME_TRANSPORT='shared'
```

Rollback shared transport with `MEGAGLITCH_FRAME_TRANSPORT=legacy`. The Rust runtime remains explicit with `MEGAGLITCH_RUNTIME_BACKEND=native`; unsupported graphs fall back to the Python worker. A missing explicitly requested Rust DLL is a structured startup failure.

## Troubleshooting

- **Wrong `cv2` path:** uninstall the PyPI OpenCV wheel from the CUDA environment and run diagnostics in a fresh process. The expected path contains `opencv-cuda/build_py312`.
- **DLL load failure:** confirm all 65 modular `opencv_*.dll` files are available and the MSVC 2015–2022 x64 runtime is installed.
- **CUDA namespace but zero devices:** run with `--cuda-selftest`; update the NVIDIA driver and confirm `nvcuda.dll` is loadable. CPU fallback remains supported.
- **Native extension import failure:** confirm CPython 3.12 x64, rebuild with `-Force`, and inspect `_native_temporal.cp312-win_amd64.pyd` dependencies.
- **Worker readiness timeout:** inspect the structured lifecycle traceback and increase `MEGAGLITCH_WORKER_STARTUP_TIMEOUT` only after identifying slow antivirus/DLL initialization.
- **Performance regression:** rerun the named artifact on an idle machine, compare raw JSON, then change a threshold only with a recorded replacement baseline.
- **Native crash:** preserve console output, `tools/diagnostics/latest.json`, benchmark artifacts, Windows Event Viewer application error, Python version, OpenCV path, GPU driver version, and the active preset.
