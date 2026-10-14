# Native runtime migration

## Status and problem statement

This document records an investigation plan, not approval for a rewrite. MegaGlitch currently hosts the UI, capture and scheduling in Python and executes the visual graph in a separate Python `multiprocessing.Process`. The project needs evidence showing whether its practical real-time limits come from Python orchestration, per-node dispatch, full-frame copies, CPU/GPU conversion, allocation, synchronization, host/device transfer, system RAM, VRAM, preview/output, or another subsystem.

The migration goal is stable real-time processing with preserved visual behavior and an unchanged `python main.py` launch path where practical. The Python engine remains the fallback until a native path is demonstrably compatible and faster or more reliable.

## Current orientation

- `main.py` is the supported bootstrap. It configures the custom OpenCV CUDA environment, optionally shows a splash, then calls `megaglitchfm_app_new.main()`.
- `megaglitchfm_app_new.py` owns the Dear PyGui application, capture/media selection, the main UI/render loop, two bounded multiprocessing queues, recording, and virtual-camera submission. `MegaGlitchFMApp.run()` starts `glitch_process_worker()` in a child process.
- `node_editor.py` compiles each Dear PyGui graph into a flat Python list. Current parameter dictionaries are copied into that list on each retrieval.
- `glitch_process_worker()` converts source frames to NumPy, executes Deck A and Deck B through `node_pipeline_runtime.execute_flat_node_pipeline()`, blends them, and returns NumPy `uint8` frames through a multiprocessing queue.
- `node_pipeline_runtime.py` iterates the flattened graph in Python and calls `effects_registry.apply_node_effect()` per effect. CPU, OpenCV CUDA, and ModernGL/Lua effects can be mixed, but final pipeline output is explicitly converted/downloaded to a CPU RGB NumPy array.
- `frame_state.py` owns per-deck CPU temporal state. History is a Python list of NumPy arrays; each output is copied into it and excess entries are removed from index zero.
- `megaglitchnode/gpu/` contains an experimental Python `Frame`/`Context`/`Pipeline` abstraction around `cv2.cuda_GpuMat` and a small GPU-node set. Other CUDA implementations also exist in `cuda_effects.py`, `gpu_ops.py`, `effect_nodes_cuda.py`, and `opencv_backend.py`.
- `lua_glsl_demo/`, `effects/`, and `shaders/` contain ModernGL, Lua, and GLSL effects. `megaglitch/` contains newer audio and generator subsystems. Presets/projects are JSON under `presets/`, `node_presets/`, and `shows/`.

The observed live path is approximately:

```text
media/image/GIF/video/webcam/window capture (main Python process)
    -> PIL image and/or CPU NumPy frame
    -> graph flattening, timeline and modulation updates in the Dear PyGui loop
    -> bounded multiprocessing input queue (serialized process boundary)
    -> Python worker: per-deck flat-list iteration and per-node dispatch
    -> mixed CPU NumPy / OpenCV CUDA / ModernGL-Lua effect implementations
    -> forced CPU RGB uint8 NumPy output
    -> bounded multiprocessing output queue (serialized process boundary)
    -> Dear PyGui dynamic-texture preview
    -> optional recording copies and virtual-camera sender thread
```

The detailed copy, synchronization, representation, and ownership audit is recorded in [current-frame-pipeline.md](current-frame-pipeline.md).

## Tentative architecture

```text
Python / Dear PyGui
    UI, presets/projects, graph editing, scripting,
    low-frequency commands and parameter updates
                    |
                    v
Rust native runtime (only if benchmark evidence supports it)
    frame loop, graph scheduling, temporal state,
    resource ownership, worker threads and command queues
                    |
                    v
Narrow C++ / CUDA backend (only if needed)
    OpenCV CUDA objects, GpuMat pools, streams/events,
    reusable filters, kernels and device-resident buffers
```

Rust is not predetermined. Before a native prototype, compare direct Rust OpenCV access, a narrow C++ bridge, a C++-first prototype, and retention of existing Python/native components. A C++ layer is justified only if it materially improves OpenCV CUDA access and ownership clarity.

## Constraints

- Target environment is Windows 11, Python/Dear PyGui, an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM, and a custom CUDA-enabled modular OpenCV build.
- The current development launcher contains machine-specific Miniforge and OpenCV paths; packaging must account for modular OpenCV DLL discovery.
- This workspace has no usable repository-local Git metadata (`.git/` is empty), so rollback currently also depends on restoring proper version-control metadata or keeping explicit patch/backup artifacts.
- Existing graphs can combine CPU NumPy, OpenCV CUDA, and ModernGL/Lua implementations, so every boundary and fallback can introduce a copy or synchronization point.
- Output currently returns to host memory for Dear PyGui preview, recording, and `pyvirtualcam` output.
- Temporal behavior, project/preset formats, parameter ranges, shader behavior, audio/MIDI, recording, and virtual camera must remain compatible.
- The live queues intentionally drop stale frames for latency; recording uses a different no-drop intent. Any replacement must specify equivalent backpressure policies.
- Every stage must build and run independently, retain a feature-flagged Python fallback, and preserve explicit cleanup on resolution changes, exceptions, and shutdown.
- Native work must not turn MegaGlitch into an AI-driven product. Existing Quantum Gestalt/PyTorch code is outside the migration goal.

## Initial-audit observations (historical)

- At the initial audit, no native source or build definition was present. The repository now contains the accepted Cython/C++ temporal segment and the optional Rust runtime vertical slice.
- The repository does contain an explicitly AI-driven Quantum Gestalt effect, PyTorch dependencies, a model, and ML-oriented UI labels. The feature flag in `config.py` defaults off, but the node/effect code is present. The product direction for this migration prohibits expanding that approach; removing legacy behavior is not part of this stage.
- OBS is not integrated through an OBS API in the inspected code. The application provides `pyvirtualcam` output and UI guidance for consuming it from OBS/Unity Capture.
- The initial requirements file installed PyPI OpenCV even in the custom CUDA environment. Core and CPU-OpenCV dependency files are now separated so that installation does not silently replace the custom extension.
- The initial tree had no root README or native build definition. Public setup guidance now lives in `README.md`, with native and production details under `native/`, `packaging/`, and `docs/production/`.
- The current frame history is CPU NumPy storage, not a device-resident OpenCV CUDA history. Some individual effects maintain other persistent GPU or GL state, which still requires a detailed audit.

## Open questions

1. What fraction of frame time is capture, graph/parameter preparation, inter-process transfer, Python per-node dispatch, CPU effects, GPU effects, synchronization, final download, preview upload, virtual camera, and recording?
2. How many full-frame copies and process-serialization copies occur per frame for representative graphs?
3. Which CUDA nodes keep frames resident across adjacent nodes, and which silently download or re-upload?
4. Are CUDA operations on the default stream, where do implicit synchronizations occur, and are filters/maps/output `GpuMat`s recreated per frame?
5. What owns ModernGL contexts and textures, and can they interoperate with CUDA without a host round trip?
6. How much time and memory are consumed by the Python list history, `copy()`, and `pop(0)` at realistic history lengths and resolutions?
7. Are separate temporal nodes duplicating equivalent history? Which histories truly require RGBA16F/32F rather than compact RGBA8, luma-only, or reduced-resolution storage?
8. Is the limiting capacity system RAM, multiprocessing serialization bandwidth, pinned memory, VRAM, or GPU compute?
9. Does disabling preview, virtual camera, recording, audio/MIDI, or final GPU download materially change throughput or latency?
10. Can Dear PyGui consume a device-resident surface through an existing repository-compatible interop path, or is a final host copy unavoidable?
11. Does direct Rust access provide complete, maintainable OpenCV CUDA bindings in this Windows/custom-build environment, or is a narrow C++ ABI safer?
12. Can the accepted native toolchains and packaging boundary be reproduced on a clean build machine without relying on the development workstation?

## Incremental stages and gates

### Stage 0: orientation and audit

Document the exact frame path, representations, thread/process ownership, copies, transfers, allocations, and synchronization. No processing changes.

Gate: every important source-to-output boundary is tied to exact files/functions, and uncertain behavior is labeled for measurement.

### Stage 1: reproducible baseline

Add low-overhead CPU timing, CUDA-event timing where supported, transfer/copy/allocation counters, memory telemetry, and machine-readable benchmark output. Cover supported resolutions, formats, history lengths, representative graphs, preview/output toggles, warm-up, average FPS, high-percentile frame time, and latency.

Gate: repeated runs are stable enough to distinguish candidate bottlenecks, and a baseline plus hardware/software configuration is committed.

### Stage 2: native temporal-ring prototype

Prototype only persistent temporal storage, fixed-ring indexing, freeze/clear/resize, a basic mix, statistics, and explicit shutdown. Select Rust direct bindings, Rust plus narrow C++/CUDA, C++-first, or an existing boundary based on Stage 1 evidence.

Gate: deterministic/compatible output, stable allocation after warm-up, safe lifecycle and resolution changes, and a measured material improvement or reliability gain over Python history. Otherwise stop and retain Python.

### Stage 3: Python-hosted native frame runtime

If Stage 2 passes, move the demonstrated hot path behind a coarse engine API with an explicit state machine, command queue, parameter snapshots, bounded input, stated drop/backpressure policy, runtime statistics, structured errors, and explicit shutdown. Release the GIL around long native work where applicable.

Gate: existing UI and files work, the Python fallback remains selectable, cross-language calls are coarse, failure/start-stop tests pass, and end-to-end results beat the Stage 1 baseline.

### Stage 4: native graph-scheduler vertical slice

Only if Python graph dispatch remains measured as significant, migrate a small slice: input, history, one feedback node, one remap/displacement node, one blend node, and output. Reuse existing effect backends where practical.

Gate: scheduling, resource pooling, parameter swaps, graph replacement, error isolation, per-node profiling, and visual compatibility are proven. Python fallback nodes are measured synchronization boundaries.

Current progress: graph-v1 recognizes a real primary Source -> Invert Colors -> Display graph,
executes the effect in the Rust-owned worker, preserves exact output, and falls back for secondary
sources, active temporal macros, and unsupported nodes. This proves the existing preset graph can
cross the native selection boundary, but it is not yet the versioned multi-node graph IR required
to complete Stage 4. See [rust-graph-v1-invert.md](../profiling/rust-graph-v1-invert.md).

Graph IR v1 now replaces mode-per-effect selection in the application worker. It validates and
executes bounded Input, Invert, Brightness/Contrast, Temporal Mix, history Blend, and Output
programs, and queues immutable graph snapshots for next-frame parameter changes. Legacy runtime
modes are compatibility adapters only. The current gate remains single-input/single-deck; see
[rust-graph-ir-v1.md](rust-graph-ir-v1.md) and the
[measurement report](../profiling/rust-graph-ir-v1.md).

### Stage 5: lifecycle and production hardening

Formalize startup/warm-up/readiness/failure/shutdown; make Windows builds reproducible; document OpenCV/CUDA DLL discovery; add regression, leak, repeated-start/stop, format/resolution-transition, and visual tests.

Gate: development build-and-launch needs no manual DLL copying, fallback recovery is safe, benchmark regressions are automated, and native failures produce actionable diagnostics.

### Stage 6: host decision

Compare the Python host, Rust host with embedded Python/Dear PyGui, and a separate Python UI process. Recommend a Rust-hosted executable only for a measured benefit unavailable to the Python-hosted extension.

## Benchmark gates

Each migration decision must use the same named graph/preset, source, resolution, format, history length, warm-up, sample count, preview/output state, CUDA state, and application build. Record at least:

- average FPS and frame time;
- 1% low FPS or p99 frame time;
- end-to-end latency and dropped/skipped frames;
- per-phase CPU time and CUDA-event GPU time;
- full-frame copy, upload, download, allocation, and synchronization counts/bytes;
- process/thread CPU utilization, GPU utilization, system RAM, and VRAM;
- deterministic output hashes or image-difference tolerances where meaningful.

A stage passes only with repeatable measurements, compatible behavior, no new unbounded resource growth, and a working rollback switch. Report raw numbers and uncertainty; do not substitute qualitative claims.

## Risks and rollback

- **Visual drift:** mixed color order, alpha, precision, interpolation, border modes, random seeds, and floating-point differences can change effects. Preserve golden inputs/outputs and parameter contracts.
- **Interop overhead:** an elegant native scheduler may lose its benefit at Python, OpenCV CUDA, or ModernGL boundaries. Measure whole graphs, not isolated kernels only.
- **VRAM exhaustion:** persistent high-precision history can exceed 8 GB. Budget by resolution/format/capacity, fail configuration safely, and support compact/tiered histories.
- **Driver/DLL fragility:** custom OpenCV CUDA and Windows packaging can fail before runtime. Keep the existing bootstrap and Python-only path until native discovery and diagnostics are proven.
- **Lifecycle failures:** leaked threads, CUDA work, GL contexts, or cameras can make restart unsafe. Require explicit ownership, ordered shutdown, and repeated-cycle tests.
- **Queue semantic changes:** native queues can alter latency, frame dropping, recording completeness, or temporal ordering. Specify and test live and recording policies separately.
- **Scope creep:** rewriting every node or making AI features part of the runtime would obscure the measured goal. Use narrow vertical slices and stop at every gate.

Rollback is stage-local: keep existing serialization and graph formats, add new paths behind explicit configuration/feature flags, avoid destructive data migrations, and retain `python main.py` plus the current Python worker. If a benchmark gate, compatibility check, resource bound, or lifecycle test fails, disable the new path and continue profiling the existing engine rather than expanding the port.
