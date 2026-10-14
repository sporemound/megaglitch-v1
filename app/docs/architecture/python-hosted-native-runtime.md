# Python-hosted Rust runtime vertical slice

## Scope and outcome

MegaGlitch still launches through `python main.py` and Python/Dear PyGui remains the application host. An explicitly selected Rust runtime can now own a bounded processing thread, frame pool, temporal storage, parameter commands, and output queue for pass-through, basic temporal mix, and the first existing graph effect: `Invert Colors`. Existing node graphs remain compatible by falling back permanently to the existing Python worker when the first unsupported graph is observed.

This is the beginning of the native graph-scheduler stage, not a complete scheduler. It does not yet migrate the broader node library, OpenCV CUDA resources, shaders, capture, preview, recording, virtual-camera submission, audio, MIDI, project files, or presets.

## Build and selection

```powershell
powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1
$env:MEGAGLITCH_RUNTIME_BACKEND='native'
python main.py
```

Rollback requires no rebuild:

```powershell
$env:MEGAGLITCH_RUNTIME_BACKEND='python'
python main.py
```

The application default is Python. `auto` is available through the standalone runtime facade, but `main.py` activates Rust only for an explicit `native` setting. If the DLL is missing, explicit native initialization reports the build command. If a live graph contains any active node other than `Display` or the prototype-only `Native Temporal Mix`, the native worker replays the current item into and permanently enters `glitch_process_worker`; no frame or graph change is silently discarded.

## Responsibility split

For a supported native frame, Rust owns:

- the processing thread and its deterministic stop/join;
- a fixed pool of queue-capacity RGB8 frame buffers;
- a fixed temporal ring;
- reject-newest bounded input behavior;
- the bounded output queue and slot recycling;
- immutable per-frame parameter snapshots consumed from a command queue;
- freeze, clear, delay, mix, sequence numbers, counters, and state;
- temporal mixing and persistent allocation lifetime.

Python still performs every frame:

- capture/source selection and graph/preset parameter construction;
- multiprocessing serialization into the worker;
- graph compatibility selection;
- conversion/validation of the NumPy input;
- one coarse `submit_frame` call and one blocking `wait_for_frame` call;
- multiprocessing serialization back to the application;
- preview, recording, virtual camera, OBS-facing behavior, and UI work.

Arbitrary effects, P-frame processing, CUDA nodes, GLSL/Lua nodes, deck blending, generators, and all current temporal nodes still execute through the Python worker. A primary Source -> Invert Colors -> Display graph with no active global temporal macros is compiled to the Rust `invert` mode. The Rust call made by `ctypes.CDLL` releases the GIL; processing itself runs on the Rust-owned thread.

## Lifecycle and errors

The FFI exposes an explicit engine handle. Its states are initializing, warming up, ready, running, stopping, stopped, and failed; a null/destroyed Python handle is uninitialized. Creation validates dimensions, graph mode, integer overflow, history capacity, and queue capacity, preallocates resources, performs allocation warm-up without inserting synthetic history, then becomes ready. Start is allowed only from ready. Shutdown rejects no new work once stopping begins, signals the command queue, joins the worker, and is idempotent through the Python facade.

FFI entry points contain unwind guards and return structured status codes plus error text. The Python layer converts these into `NativeRuntimeUnavailable`, `NativeRuntimeFailure`, `UnsupportedNativeGraph`, `ValueError`, queue rejection, or timeout. Rust allocator exhaustion may abort before an error can be translated; impossible size arithmetic is rejected before allocation, but deterministic real OOM testing was intentionally not attempted on the development machine.

The queue policy is reject-newest when all pooled slots are in input, processing, or output ownership. Polling an output returns the slot to the pool. Output saturation is also bounded; no queue or history grows dynamically.

## Performance

Raw data is in [native-runtime.json](../../tools/benchmark/results/native-runtime.json) and [native-runtime.csv](../../tools/benchmark/results/native-runtime.csv). The final run uses five warm-up and thirty measured frames.

Direct Python-host/runtime round trips, including one input copy, Rust/Python processing, one output copy, thread handoff, and exactly two coarse calls per frame:

| Mode | Backend | 1280x720 mean / p99 | 1920x1080 mean / p99 | 1200x800 mean / p99 |
| --- | --- | ---: | ---: | ---: |
| Pass-through | Python thread | 1.230 / 1.874 ms | 2.472 / 2.968 ms | 2.805 / 4.023 ms |
| Pass-through | Rust | 0.558 / 0.896 ms | 0.979 / 1.573 ms | 1.794 / 2.991 ms |
| Temporal mix | Python thread | 21.293 / 24.093 ms | 42.975 / 47.921 ms | 49.301 / 54.432 ms |
| Temporal mix | Rust | 1.855 / 2.710 ms | 11.427 / 14.133 ms | 5.786 / 6.567 ms |

The Python temporal fallback is a semantic reference based on NumPy float expressions, not the optimized OpenCV baseline. The Rust mix must not replace `cv2.addWeighted` based on this table; Prompt 4 measured OpenCV substantially faster for a simple blend.

The paired 1200x800 application worker measurement is the important result:

| Worker | Mean | p99 | FPS | 1% low FPS |
| --- | ---: | ---: | ---: | ---: |
| Python minimal graph | 19.894 ms | 24.292 ms | 50.27 | 41.17 |
| Rust minimal graph | 17.044 ms | 20.119 ms | 58.67 | 49.70 |

Rust reduces this minimal boundary by 2.850 ms mean (14.3%) and 4.173 ms p99 (17.2%). The direct 1200x800 Rust pass-through is about 1.8 ms, leaving roughly 15.2 ms in multiprocessing transport, Python wrapper work, and scheduling. The migration therefore confirms that coarse native calls work, but also confirms the earlier conclusion that the process/frame-transport boundary remains dominant.

At 1200x800 with history 8 and queue capacity 2, Rust reports 28,800,000 allocated frame bytes: 23,040,000 for history and 5,760,000 for two pooled queue frames. These allocations are persistent until shutdown.

## Compatibility and visual behavior

- Empty/Display-only graphs are byte-identical pass-through in tests.
- The basic temporal mode uses deterministic fixed-point RGB8 blending. It is a prototype mode and is not substituted for any existing named effect.
- Existing graphs, projects, presets, UI controls, audio/MIDI, shader effects, OBS behavior, virtual camera, recording, and command-line arguments stay on the Python worker.
- Resolution changes reconstruct the runtime at the frame boundary. Existing Python graphs follow their previous behavior.
- The minimal native result aliases `frame`, `frame_a`, and `frame_b` before multiprocessing serialization because all three are identical for the supported graph. Unsupported deck graphs fall back.

## Validation and remaining gaps

Tests cover native and fallback initialization, pass-through parity, parameter changes on the next frame, freeze, clear, queue saturation and recovery, rapid parameter changes, invalid state/input/graph errors, repeated start/stop, resolution reconstruction, shutdown with queued work, worker integration, and unsupported-graph fallback.

Actual Dear PyGui closure is represented by deterministic runtime shutdown while work is queued; the automated suite does not launch an interactive UI. CUDA-active shutdown, CUDA initialization failure, virtual-camera failure, and native CUDA exceptions cannot be tested because this vertical slice owns no CUDA resources. Rust panic containment exists at the FFI boundary, but deliberately triggering undefined behavior or allocator abort is not a safe test.

## Gate for further migration

The runtime structure is stable enough to retain behind the feature flag, but it does not justify moving the full graph scheduler yet. Python node dispatch was already measured below 0.1 ms for 32 disabled nodes. The next justified work is eliminating multiprocessing frame copies or building a device-resident multi-node CUDA slice; graph scheduling should move only after a representative active graph shows dispatch/resource-routing cost rather than algorithm cost.

The subsequent graph-v1 result migrates the existing deterministic `Invert Colors` effect and
is documented in [rust-graph-v1-invert.md](../profiling/rust-graph-v1-invert.md). It is an
architecture/compatibility gate toward the requested complete Rust runtime; broader promotion
still requires a versioned graph IR, dynamic parameter snapshots, dual-deck execution, and
coarse CUDA/GL backend ownership rather than one FFI mode per effect.

That follow-up is now implemented as [Rust graph IR v1](rust-graph-ir-v1.md). The application
worker no longer selects a Rust mode per effect: it compiles supported flat pipelines into an
immutable versioned program. The old mode API remains for backward compatibility and benchmarks,
but its constructors immediately translate to the same IR interpreter.
