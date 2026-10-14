# Host architecture decision after production hardening

## Decision

The next stable MegaGlitch release should remain a Python/Dear PyGui application hosting a spawned Python processing worker, shared/preallocated frame transport, and coarse native C++/Cython hot-stage segments. The experimental Rust runtime remains opt-in. Do not build a Rust-hosted executable now, and do not reopen Prompt 6 without new profiling evidence.

This follows measured results rather than target-architecture momentum:

- disabled-node Python dispatch was below 0.1 ms for 32 nodes;
- the Rust minimal worker improved the old serialized boundary by only 2.85 ms;
- shared transport removed 14–20 ms from representative P-frame worker paths;
- native algorithm segments improved the actual P-frame hot work by roughly 4x over the already split native path;
- the current shared/fused representative P-frame worker reaches 20.83 ms mean and 22.22 ms p99 at 1200x800;
- deterministic readiness and shutdown are already achieved without changing hosts.

## Option comparison

### Continue with Python as host — selected

Python already owns Dear PyGui, graph editing, project/preset serialization, scripting, audio/MIDI, capture, recording, virtual camera, and development tooling. Keeping it avoids embedding complexity and preserves the existing `python main.py` entry point. Frame transport is bounded and shared; demonstrated hot algorithms can move behind coarse native calls without moving UI or scheduler code.

Crash isolation is adequate at the processing-process boundary, though a UI crash still ends integrations. Headless operation would require a deliberate service interface, not merely a host-language change.

### Rust executable embedding Python/Dear PyGui — rejected

Embedding retains Python, the GIL, Python packages, and Dear PyGui while adding interpreter initialization, module-path, multiprocessing/freeze, packaging, stack-trace, and shutdown complexity. Native crashes would still share the host process. It offers no measured frame-path benefit because UI-host work is not the current bottleneck.

### Separate Rust engine and Python UI — defer as an operational product option

A separate engine could provide true UI restart, headless operation, and stronger crash isolation while preserving temporal state. It would also require versioned IPC, authentication/local access rules, project synchronization, capture/output ownership decisions, shared GPU/CPU surface transport, reconnection semantics, and another installable process. The current shared-memory worker already captures part of this benefit with much lower complexity.

Reconsider this option only if headless operation, UI restart without output interruption, or multi-client control becomes a concrete product requirement. Do not justify it with current frame performance.

## Packaging and ownership implications

The selected release still has a substantial Windows dependency surface: CPython 3.12 ABI, custom OpenCV CUDA extension, 65 modular OpenCV DLLs, MSVC runtime, CUDA/driver runtime, Cython extension, optional Rust DLL, Dear PyGui, ModernGL, Lua, audio/MIDI, camera, and virtual-camera dependencies. Moving the host to Rust would not remove most of these while Python effects/UI remain.

The checked-in PyInstaller spec and diagnostics now make this surface explicit. A clean-machine bundle test is still required before calling packaging production-ready.

## Prioritized native migration backlog

1. **Ownership/copy profiling after the fused P-frame segment.** Quantify initial node copy, history copy, shared output copy, and UI consumer copy before changing aliases or lifetimes.
2. **Device-resident CUDA segment gate.** Identify a representative sequence of at least several adjacent CUDA-compatible nodes that can retain `GpuMat` ownership across one upload/download. Benchmark it end to end before implementation is enabled.
3. **Additional measured CPU hot stages.** Migrate only active effects whose algorithms dominate representative graphs; use coarse calls and preallocated outputs.
4. **Format-aware shared transport.** Add explicitly versioned RGBA or higher-precision pools only when a real graph requires them; do not make the current RGB8 pool dynamically unbounded.
5. **Recording/output ownership.** Separate low-latency latest-frame policy from no-drop recording policy with explicit bounded capacity and backpressure metrics.
6. **Clean-machine packaging and hardware-in-the-loop tests.** Exercise camera disconnect, virtual-camera failure, CUDA driver-only installation, resolution transitions, and long-running memory/handle stability.
7. **Optional engine-service feasibility prototype.** Only if headless/UI-restart requirements emerge; measure IPC and recovery semantics before choosing Rust.

General graph-scheduler migration is not on this backlog until profiling shows scheduling/resource routing consuming a material share of representative frame time.
