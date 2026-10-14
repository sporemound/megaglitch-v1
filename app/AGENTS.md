# MegaGlitch repository guide

## 0.0.9.4 release-convergence override

MegaGlitchNode 0.0.9.4 is in **release-convergence mode**. Before changing production code, read:

1. `PROJECT_SCOPE.md`
2. `ARCHITECTURE_FREEZE.md`
3. `RELEASE_GATES.md`
4. `KNOWN_FAILURES.md`
5. `MEGAGLITCH_0.0.9.4_RELEASE_STATUS.md`

These files override open-ended feature development for the 0.0.9.4 release.

Until the release scorecard is current-head certified:

- do not add new feature families, node families, speculative UI/workspace systems, AI/cognitive mutation features, cloud/commercial features, or new architectural abstractions;
- do not perform opportunistic refactors, renames, relocations, dependency migrations, or adjacent cleanup;
- identify the failing release gate or verified defect before editing production code;
- make the smallest change that closes that gate or defect;
- separate audit from implementation for nontrivial work;
- preserve rollback paths and supported fallbacks;
- report exact files, commands, tests, results, regressions, and the next bounded action;
- treat historical passing benchmarks as evidence, not automatic proof that current HEAD passes after later code changes.

Existing mixed feature commits must not be blindly reverted when they also contain fixes or parity work. Validate them against the release gates and remove/disable only what is demonstrably out of scope or destabilizing.

MegaGlitch is a modular, real-time visual signal-corruption system. The current application is hosted by Python and Dear PyGui, with CPU effects, OpenCV CUDA effects, ModernGL/GLSL/Lua effects, temporal state, audio/MIDI modulation, a node graph, recording, and virtual-camera output.

## Build, run, and test

- Install Python dependencies with `python -m pip install -r requirements.txt`. On the primary development machine, preserve the custom CUDA-enabled OpenCV build: do not let pip replace it with `opencv-python`. `tools/run.ps1 -Install` creates a temporary OpenCV-free requirements file for this purpose.
- Launch with `python main.py`. The development launcher is `powershell -ExecutionPolicy Bypass -File tools/run.ps1`; pass `-Python <python.exe>` or set `MEGAGLITCH_PYTHON` when Python is not on `PATH`.
- Run the automated suite with `python -m pytest -q`. `pytest` is not currently listed in `requirements.txt`, so the development environment must provide it separately.
- Run focused smoke checks with `python scripts/genesis_vdp_smoke_test.py`, `python test_pipeline.py`, and `python test_cuda.py`. The latter two require the configured custom OpenCV CUDA environment for meaningful GPU coverage.
- Reproduce the native-migration frame baseline with `python tools\benchmark\run_baseline.py --label baseline --warmup 5 --iterations 30 --output-dir tools\benchmark\results`.
- Build and benchmark the opt-in temporal prototype with `powershell -ExecutionPolicy Bypass -File native\temporal_ring\build.ps1` and `python tools\benchmark\run_temporal_prototype.py`.
- Build the Rust runtime vertical slice with `powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1`; benchmark it with `python tools\benchmark\run_native_runtime.py`.
- Build, diagnose, and launch the supported Python-hosted stack with `powershell -ExecutionPolicy Bypass -File tools\build_and_run.ps1 -Python <python.exe>`; add `-WithRust` only for the experimental runtime.
- Benchmark shared worker transport with `python tools\benchmark\run_shared_transport.py`; measure the active P-frame hot stage with `python tools\benchmark\run_pframe_hotstage.py`.
- Benchmark managed worker readiness and shutdown with `python tools\benchmark\run_worker_lifecycle.py --cycles 3`.
- PyInstaller guidance and an example specification live under `packaging/pyinstaller/`.

## Architecture rules

- Preserve visual behavior, project/preset compatibility, UI controls, shader effects, audio/MIDI behavior, recording, and virtual-camera behavior unless a documented bug requires a change.
- Measure before migrating. Distinguish Python dispatch, allocation/copying, synchronization, transfer, CPU work, GPU work, and memory-capacity limits with reproducible benchmarks.
- Keep Python/Dear PyGui responsible for UI, project and preset management, scripting, configuration, graph editing, and low-frequency controls. Remove Python from a per-node/per-frame path only when measurement demonstrates that path is a bottleneck.
- Keep GPU frames within one clearly owned backend across a graph where practical. Avoid per-node Python/Rust/C++ frame crossings, unnecessary `GpuMat` clones, host/device transfers, global synchronization, and per-frame filter or full-frame buffer allocation.
- Prefer persistent resource pools and fixed temporal rings over shifting or reallocating frame history. Handle resolution and format changes explicitly.
- Any native API must be coarse-grained, lifecycle-safe, and explicit about resource/thread ownership, errors, command queues, backpressure, and shutdown.
- Migration must be incremental, benchmark-gated, reversible, and buildable at every stage. Do not perform a wholesale rewrite.
- Preserve the existing Python engine as a selectable fallback until native behavior, reliability, and performance are proven. Keep the user-facing `python main.py` launch path where practical.
- Do not remove features silently. Record compatibility differences and provide a rollback path for every migration stage.

MegaGlitch is not an AI-product redesign. Do not add prompt-based generation, an AI creative director, a language model in the runtime experience, or generated controls/presets in place of deliberate visual controls. Existing experimental AI-related code is legacy scope and must not drive the native-runtime architecture.

## Testing expectations

Run the most relevant unit tests and a focused smoke test for each change. For frame-path changes, add deterministic output checks where possible and report end-to-end frame time, high-percentile frame time, FPS, latency, copies/transfers, allocations, RAM/VRAM, and the exact benchmark configuration. State which tests could not run and why. Never claim a performance improvement without measurements against the recorded baseline.

The migration hypothesis, gates, risks, and rollback plan are maintained in [docs/architecture/native-runtime-migration.md](docs/architecture/native-runtime-migration.md). The evidence-based live-path audit is [docs/architecture/current-frame-pipeline.md](docs/architecture/current-frame-pipeline.md). The reproducible pre-native performance baseline is [docs/profiling/framebuffer-baseline.md](docs/profiling/framebuffer-baseline.md); the temporal prototype result is [docs/profiling/native-temporal-prototype.md](docs/profiling/native-temporal-prototype.md), the Python-hosted Rust slice is [docs/architecture/python-hosted-native-runtime.md](docs/architecture/python-hosted-native-runtime.md), the shared transport result is [docs/profiling/shared-transport-and-pframe-hotstage.md](docs/profiling/shared-transport-and-pframe-hotstage.md), the accepted native P-frame segment is [docs/profiling/native-pframe-segment.md](docs/profiling/native-pframe-segment.md), and deterministic startup/shutdown is [docs/architecture/deterministic-worker-lifecycle.md](docs/architecture/deterministic-worker-lifecycle.md). Production commands are in [docs/production/windows-build-and-release.md](docs/production/windows-build-and-release.md), and the host decision is [docs/architecture/rust-host-decision.md](docs/architecture/rust-host-decision.md). Future profiling findings belong in focused documents under `docs/profiling/`, not in this file.
