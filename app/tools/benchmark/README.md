# MegaGlitch benchmark harness

Run from the repository root with the same Python environment used for MegaGlitch:

```powershell
python tools\benchmark\run_baseline.py --quick --label smoke
python tools\benchmark\run_baseline.py --label baseline --warmup 5 --iterations 30 --output-dir tools\benchmark\results
```

The harness bootstraps the configured OpenCV build before importing `cv2`, records capability failures explicitly, and writes JSON plus flattened CSV under `tools/benchmark/results/` by default. Override the destination with `--output-dir`.

The baseline covers supported RGB8 CPU pipeline/history cases, the saved `node_presets/6-20-26.json` P-frame preset, Python dispatch scaling, multiprocessing round trips, preallocated CPU-ring storage, and preview CPU preparation. The preview measurement excludes Dear PyGui's internal texture upload/presentation. CUDA, high-precision formats, GPU history, and virtual-camera submission are measured only when the current runtime exposes a valid implementation; otherwise they appear as `unsupported` records.

The interpreted baseline and measurement limitations are documented in `docs/profiling/framebuffer-baseline.md`.

Measure the current worker-boundary transport and the saved-preset P-frame hot stage with:

```powershell
python tools\benchmark\run_shared_transport.py --label shared-transport --warmup 5 --iterations 30
python tools\benchmark\run_pframe_hotstage.py --warmup 3 --iterations 20
```

Their contract, ownership rules, rollback flag, and results are documented in `docs/profiling/shared-transport-and-pframe-hotstage.md`.

The same transport harness now emits per-frame explicit-copy counts and byte totals for shared
input, initial source-node ownership, history retention, shared output, and application/UI
consumer ownership. The committed ownership baseline is reproduced with:

```powershell
python tools\benchmark\run_shared_transport.py --label frame-copy-ownership --warmup 5 --iterations 30
```

Its results and measurement limits are documented in
`docs/profiling/frame-copy-ownership-telemetry.md`.

Compare the compatibility source-node copy with the opt-in unique-source borrow policy using
a deterministic representative P-frame sequence:

```powershell
python tools\benchmark\run_source_copy_policy.py --warmup 3 --iterations 15
```

The policy gate, rollback environment variable, and measured result are documented in
`docs/profiling/source-node-borrow-policy.md`.

Build and measure the Rust runtime, including the first existing graph node, with:

```powershell
powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1
python tools\benchmark\run_native_runtime.py --warmup 3 --iterations 15 --label native-runtime-graph-v1
```

The graph-v1 parity and performance gate is documented in
`docs/profiling/rust-graph-v1-invert.md`.

Measure the versioned multi-node graph IR gate with:

```powershell
python tools\benchmark\run_native_runtime.py --warmup 3 --iterations 15 --label native-runtime-graph-ir-v1
```

This includes a pixel-exact Input -> Brightness/Contrast -> Invert -> Output comparison. See
`docs/profiling/rust-graph-ir-v1.md`.

The hot-stage harness also measures the fused spiral-motion/uniform-prediction segment. Its accepted result and full-worker follow-up are documented in `docs/profiling/native-pframe-segment.md`.

Measure repeated managed-worker readiness, first-frame delivery, and shutdown with:

```powershell
python tools\benchmark\run_worker_lifecycle.py --cycles 3
```

The lifecycle protocol and failure behavior are documented in `docs/architecture/deterministic-worker-lifecycle.md`.

Check committed native/transport/lifecycle results against the recorded hardware profile:

```powershell
python tools\benchmark\check_regressions.py
```

That default is strict, including the experimental Rust Graph IR absolute and
Python-relative timing limits. The Python-host production release gate uses:

```powershell
python tools\benchmark\check_regressions.py --rust-performance-advisory
```

Advisory mode does not remove the Rust gate. Pixel equality, output hashes,
backend identity, dropped frames, missing evidence, and every Python/shared
production metric remain hard failures; only Rust mean/p99 and ratio misses
move to the JSON `advisories` list. The process exits successfully only when
the hard `failures` list is empty. Thresholds live in
`tools/benchmark/regression_thresholds.json` and remain identical in both
modes. Use a separate profile for materially different hardware rather than
changing these limits or using advisory severity as a substitute profile.

Measure the Rust UI's versioned Python sidecar against the same direct flat-pipeline call:

```powershell
python tools\benchmark\run_rust_ui_bridge.py --warmup 5 --iterations 30
```

This negotiates protocol v2, replaces the graph transactionally, verifies exact Invert parity from
the mapped RGB8 preview slots, releases every consumed slot, and validates revisioned shutdown and
surface cleanup. It records direct and sidecar-reported mean/p99 processing time plus consumer mmap
read time. It does not include camera capture or egui upload/presentation. See
`docs/profiling/rust-ui-python-bridge.md`.

Exercise the deterministic quality-first graph through the full-resolution
`MasterGraphExecutor` and `OfflineMasterRenderer` contracts with:

```powershell
python tools\benchmark\run_render_quality_path.py --resolution 1920x1080 --frame-count 30 --source-fps 30 --output-fps 55 --semantics true_accelerated --seed 917 --json-output tools\benchmark\results\render-quality-1080p.json --csv-output tools\benchmark\results\render-quality-1080p.csv
```

This harness checks master/history resolution, exact rational FPS scheduling, cross-process output
hashes, graph/provider metadata, copies, and observed CPU/GPU transfers. The reference Optical Flow
Remap is a CPU node and its full-resolution analysis state is inventoried. Encoder egress is an
in-memory hash sink, so bitrate is intentionally null and its encode timing must not be presented as
FFmpeg performance. Use Offline Master export for codec measurements. The gates and remaining
provider-residency limits are in `docs/profiling/render-quality-overhaul.md`.
