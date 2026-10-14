# MegaGlitch Node 0.9.4

## Opt-in Rust UI functional preview

The incremental eframe/egui/wgpu workstation lives in `native/ui`. Its Engine V2 sidecar accepts
transactional graph, source, transport, and seek commands, executes the existing Python frame path,
and publishes real RGB program pixels through fixed leased memory-mapped preview slots. Offline,
pending, accepted, and rejected engine state remains explicit; production UI code does not replace
a failed bridge with demo telemetry.

```powershell
powershell -ExecutionPolicy Bypass -File native\ui\build.ps1 -Run
```

This remains a Gate-1 functional preview, not a replacement for the complete Python/Dear PyGui
machine. See `docs/architecture/rust-ui-beta-shell.md` and
`docs/architecture/mycotroph-functional-parity.md` for its boundary, rollback, and remaining gates.

MegaGlitch is a modular real-time visual signal-corruption system built with Python and Dear PyGui. It combines CPU and OpenCV CUDA effects, ModernGL/GLSL/Lua effects, temporal processing, audio and MIDI modulation, a node graph, recording, and virtual-camera output.

This repository is a developer beta. The Python engine is the supported fallback; the native temporal segment and Rust runtime remain incremental, optional components.

## Platform status

The primary development and release target is Windows 11 x64 with CPython 3.12. CPU fallback is supported, while CUDA, camera, audio/MIDI, virtual-camera, and native-runtime features depend on local hardware and drivers. Linux has not yet completed the same hardware-in-the-loop release gate.

## Quick start

Create and activate a Python 3.12 virtual environment, then choose one OpenCV path.

For a normal CPU-only installation:

```powershell
python -m pip install -r requirements-cpu-opencv.txt
python main.py
```

For the custom CUDA-enabled OpenCV build, install the core dependencies without a PyPI OpenCV wheel:

```powershell
python -m pip install -r requirements.txt
$env:MEGAGLITCH_OPENCV_BIN='C:\path\to\opencv\bin\Release'
$env:MEGAGLITCH_OPENCV_PYTHON='C:\path\to\opencv\python\Release'
python main.py
```

Do not install `opencv-python` into the custom CUDA environment; it can shadow or replace the custom extension.

The optional legacy Quantum Gestalt effects have a separate, large dependency set:

```powershell
python -m pip install -r requirements-quantum.txt
```

## Optional integrations

- Set `MEGAGLITCH_PYTHON` or pass `-Python` to the PowerShell build scripts when Python is not on `PATH`.
- Set `MEGAGLITCH_CDP_PROGRAMS` to a CDP `_cdprogs` directory to enable the external CDP worker integration.
- Set `MEGAGLITCH_SPLASH_IMAGE` to an image path to show a startup splash. Without it, startup continues normally.
- Build the native temporal extension with `powershell -ExecutionPolicy Bypass -File native\temporal_ring\build.ps1`.
- Build the experimental Rust runtime with `powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1`.

Detailed production and PyInstaller instructions are in [docs/production/windows-build-and-release.md](docs/production/windows-build-and-release.md).

## Tests

Install the development dependencies and run the automated suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Focused smoke checks:

```powershell
python scripts/genesis_vdp_smoke_test.py
python test_pipeline.py
python test_cuda.py
```

The CUDA and native tests require their corresponding local runtimes. See [AGENTS.md](AGENTS.md) and the documents under `docs/` for the benchmark and migration gates.

## Release status and known limits

- This is a beta source release, not a clean-machine production certification.
- The supported public entry point is `python main.py`.
- Native components are coarse-grained accelerators; they do not replace the Python UI and project-management layer.
- Some features require external software or devices and degrade or remain unavailable when those dependencies are missing.
- This imported source snapshot is distributed under the GPLv3 license selected at the repository root. Preserve separately credited third-party media and dependency notices.

Please report security-sensitive issues privately using the repository host's private vulnerability-reporting feature rather than a public issue.
