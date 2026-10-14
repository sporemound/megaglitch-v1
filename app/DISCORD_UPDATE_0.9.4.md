# MegaGlitch Node 0.9.4 - Performance & Stability Beta

MegaGlitch Node 0.9.4 is a major under-the-hood update focused on faster frame processing, smoother P-frame effects, and more dependable startup and shutdown - without replacing the node editor or changing the creative workflow.

## What feels better

### Faster P-frame effects

The active P-frame motion and prediction stages now run as one optimized native segment. It preserves the original seeded visual output while avoiding large temporary frame allocations.

On the development test system:

- The fused P-frame segment was approximately **3.9x faster** than the previous split motion/native-prediction path.
- The optimized prediction hot stage was approximately **15.4x faster** than its original NumPy implementation.
- The representative 1200x800 P-frame graph reached **48 FPS average** with a **45 FPS 1% low** using the new shared transport.

These figures are measurements from one test system and preset, not guaranteed FPS for every graph or PC.

### Less frame-copy overhead

Frames now cross the processing-worker boundary through a fixed shared-memory pool instead of repeatedly serializing full images through queues.

- A representative frame message dropped from roughly **2.88 MB to 820 bytes** of descriptor and metadata data.
- On the final fused P-frame graph, shared transport reduced average frame time from **33.05 ms to 20.83 ms**.
- The program output now has one canonical frame, reducing redundant Deck A/Deck B output copies.

The result should be lower transport latency and steadier frame pacing, especially in heavier graphs.

### More reliable processing startup

The processing worker now has an explicit lifecycle:

`initializing -> warming up -> ready -> running -> stopped`

Before accepting frames, it verifies OpenCV, checks the CUDA device, runs GPU self-tests, and attaches its shared-memory frame pool. Startup failures now produce structured diagnostics instead of leaving a half-started worker behind.

### Safer shutdown and graph changes

This update improves cleanup during normal exit, failed startup, graph replacement, reset, and forced worker termination.

- Shared frame slots are reclaimed instead of being left locked.
- The worker is stopped before shared memory is released.
- Repeated start/stop cycles complete without retained slots or forced termination in the current automated tests.
- Deterministic output checks protect the optimized native path from silent visual changes.

### Improved Windows beta bundle

The Windows build now includes the custom CUDA-enabled OpenCV extension, native P-frame extension, Lua/Lupa runtimes, presets, shaders, and the optional Rust runtime DLL.

The corrected beta bundle has passed:

- Native import and CUDA diagnostics
- Real processing-worker startup and CUDA self-tests
- A normal launch with a responsive **MegaGlitchFM Node Editor** window
- Focused runtime-hardening and worker-lifecycle tests

## What has not changed

- The Python/Dear PyGui node editor remains the main application.
- Existing project, preset, shader, Lua, audio/MIDI, recording, and virtual-camera workflows remain the intended compatibility target.
- The experimental Rust runtime is still opt-in and is not the default engine.
- There has been no wholesale scheduler rewrite; optimizations target measured bottlenecks while keeping fallback paths available.

## Installing and launching the beta

1. Extract the complete `MegaGlitchNode` folder.
2. Keep `MegaGlitchNode.exe` beside its `_internal` directory. The EXE is not standalone and will not work if copied out by itself.
3. Launch `MegaGlitchNode.exe`.

The current development bundle expects:

- Windows 11 x64
- A current NVIDIA driver
- CUDA Toolkit 12.6 at its standard location

The splash screen may be skipped on systems without a working Tk installation; this does not prevent the main node editor from opening.

## Optional diagnostics

If the application does not open, run these from PowerShell inside the extracted folder:

```powershell
.\MegaGlitchNode.exe --diagnose-only
.\MegaGlitchNode.exe --diagnose-worker
```

The first checks OpenCV, CUDA, and native components. The second starts the real processing worker, runs its CUDA tests, waits for readiness, and shuts it down without opening the UI.

Frozen bootstrap crashes may also leave a log named `megaglitch-frozen-<pid>.log` in `%TEMP%`.

## Beta status and feedback

This is a **developer/limited-beta candidate**, not a final production release. Clean-machine testing and hardware-in-the-loop checks for cameras, audio devices, recording, virtual-camera drivers, CUDA installations, and long-running sessions are still in progress.

When reporting a problem on Discord, please include:

- GPU model and NVIDIA driver version
- Windows version
- The preset or graph being used
- Whether both diagnostic commands pass
- Console output or the frozen bootstrap log
- Steps that reproduce the issue

Thanks for testing MegaGlitch Node and helping harden the next release.
