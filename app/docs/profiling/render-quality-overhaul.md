# Render-quality master-path benchmark

## Decision and scope

The quality-first overhaul is gated on render-path correctness before throughput. MegaGlitch now
has a serial, deterministic master path for offline work:

```text
deterministic master-size RGB8 source
    -> OfflineMasterRenderer (one callback per source frame)
    -> MasterGraphExecutor (fixed quality-first graph and master-size temporal state)
    -> immutable master frame
    -> encoder egress schedule
    -> direct FFmpeg in the application, or a hash-only sink in this benchmark
```

Preview textures, UI scale, OBS frames, and virtual-camera frames are not inputs to this path. FPS
placement happens after graph processing. The benchmark does not replace the existing framebuffer,
worker-transport, native-runtime, or encoder benchmarks, and this document makes no speed-improvement
claim.

## Reproduction

Run from the repository root with the same Python/OpenCV environment used by MegaGlitch:

```powershell
python tools\benchmark\run_render_quality_path.py `
  --resolution 1920x1080 `
  --frame-count 30 `
  --source-fps 30 `
  --output-fps 55 `
  --semantics true_accelerated `
  --seed 917 `
  --json-output tools\benchmark\results\render-quality-1080p.json `
  --csv-output tools\benchmark\results\render-quality-1080p.csv
```

Use `--semantics duration_preserving` to measure zero-order-hold frame placement that retains the
source duration. `true_accelerated` processes and emits each source frame once on the requested
output timebase, so 30 source frames emitted at 55 fps last `30 / 55` seconds. Neither mode advances
the temporal graph more than once per source frame.

The source is deterministic RGB8 with gradients, checkerboard energy, fixed seeded texture, and
moving hard-edged bars. It is created at the requested master resolution; the harness has no ingress
resize. The built-in quality-first graph is hashed in its fixed preset form and compiled into the
same flat Python pipeline used by the master executor.

For the default CPU correctness run, the harness pins OpenCV to one worker and disables OpenCL for
the duration of the render, then restores the caller's settings. This removes scheduler/provider
variance from the hash gate; it is a reproducibility control, not a throughput optimization. The
OpenCV version and these controls are recorded under `configuration.opencv_determinism`.

The benchmark encoder is intentionally a memory/hash sink. It measures egress write/hash cost,
frame count, byte count, and an output SHA-256, but does not run a codec. Consequently `bitrate` is
`null`; encode timing from this harness is not FFmpeg throughput. Use the GUI Offline Master path or
a separate direct-FFmpeg run when bitrate and codec time are required.

## Recorded telemetry

JSON preserves the nested record and CSV emits one flattened row. The record includes:

- master resolution, source FPS, requested output FPS, FPS semantics, source/processed/encoded frame
  counts, and source/output duration. `fps_exact` and `duration_exact` retain the rational values
  used by the scheduler instead of relying on floating-point display values;
- inferred node-provider assignment plus centrally observed CPU-to-GPU and GPU-to-CPU event counts,
  bytes, direction, and source index;
- measured master-history resolution, reachable temporal-buffer inventory, per-deck history depth,
  and explicit frame-copy counts/bytes;
- dropped frames, duplicated encoder-egress frames, and source frames intentionally omitted by a
  lower duration-preserving output rate;
- render time per source frame and memory-sink write time per encoded frame;
- bitrate (`null` for the benchmark sink), graph SHA-256, random seed, and master-output SHA-256.

Provider assignment is a graph declaration/inference, while transfer events are observations from
the central `gpu_ops` boundary. Keeping both prevents a CUDA-labelled graph from being interpreted as
GPU-resident without transfer evidence. In the quality-first reference graph, Optical Flow Remap is
the provider-independent CPU node and is reported as `cpu`; its two-channel flow vector and the
analysis masks appear in the measured nested temporal inventory at master resolution.

## Correctness gates

A result is eligible for later performance comparison only when all of these conditions hold:

1. Every ingress, graph output, core temporal buffer, and inventoried image/history buffer has the
   requested master width and height. A mismatch is an error, never a benchmark-side resize.
2. `master_processed` equals the source frame count and `dropped_frames` is zero.
3. `true_accelerated` has no duplicates or omitted source frames and its output duration equals
   `frame_count / output_fps`. Duration-preserving conversion may duplicate or omit only at encoder
   egress, with those counts reported.
4. Repeating the same graph, source configuration, environment, and render seed in separate Python
   processes produces the same graph and output hashes even when `PYTHONHASHSEED` differs. A
   different render seed is allowed to change the output hash.
5. Preview, UI, OBS, and virtual-camera products never re-enter this benchmarked graph. Their egress
   cost belongs to live-path measurements.
6. The quality-first graph retains its controlled-destruction order and leaves sharpening/local
   contrast to final structural restoration.
7. Any optimization that adds hidden scaling, runtime frame dropping, history-resolution drift, or
   output-branch re-entry fails even if reported frame time improves.

No throughput threshold is committed here. Hardware, OpenCV provider, CUDA availability, frame
count, resolution, graph hash, and seed must match before comparing timing records.

## Remaining measurement and residency limits

- The Python flat graph still dispatches nodes serially and the deck blend/final normalized result is
  CPU RGB8. This harness measures that supported path; it does not imply a native graph migration.
- The cross-process hash gate covers the default CPU quality graph. CUDA kernels, driver versions,
  and heterogeneous hardware need their own provider-specific reproducibility gate.
- Legacy CUDA and ModernGL/Lua nodes can introduce host/device transfers and synchronization.
  Transfers routed through `gpu_ops` are counted, but backend-internal copies, implicit pixel-format
  conversions, driver synchronization, and encoder-side RGB-to-YUV conversion need Nsight, GPU
  events, GL tooling, or FFmpeg instrumentation.
- Temporal inventory covers core `FrameState` buffers and reachable nested NumPy/GpuMat-like state.
  Opaque storage owned inside a third-party backend may require a backend-specific inventory hook.
- The memory sink has no compression, muxing, disk I/O, OBS, virtual camera, Dear PyGui upload, or
  display presentation. Do not extrapolate its encode timing or null bitrate to production output.
- A synthetic sequence makes correctness reproducible but does not replace representative Uniflora
  footage, audio modulation, scene cuts, long feedback runs, VRAM-capacity tests, or codec-domain
  post-processing measurements.

## Rollback

The overhaul is incremental. Existing project and node-preset loading remains available, so a user
can load a legacy graph instead of the quality-first preset. The legacy batch renderer remains
separate from Offline Master, and the Python-hosted engine remains the supported fallback while
native stages are opt-in and benchmark-gated. Reverting this benchmark requires no runtime change;
it is a read-only harness over the master/offline contracts.

## Recorded CPU correctness run (2026-08-27)

This run is correctness evidence for the all-CPU quality graph, not a speedup claim or an accepted
throughput threshold.

| Field | Recorded value |
|---|---:|
| Master resolution | 1920x1080 |
| Source / processed / encoded frames | 30 / 30 / 30 |
| FPS semantics | 30 -> 55, `true_accelerated` |
| Source / output duration | 1.0 s / 0.5454545 s |
| Dropped / duplicated / omitted frames | 0 / 0 / 0 |
| Render mean / p50 | 8481.471 ms / 8432.967 ms |
| Render p95 / p99 / max | 9250.568 ms / 9495.939 ms / 9577.464 ms |
| CPU/GPU transfers | 0 events / 0 bytes |
| Graph hash | `ae10a638840fead4aa115698f06e7ed5a978ae7fbf0b02a6994797b1a4c97c75` |
| Random seed | 917 |
| Master-output SHA-256 | `e7fd39939a2b2140fc58e67fdd2be26c8b77c100a188cfe5e8309f6bcfc98df6` |

All recorded correctness gates were `true`. The JSON and CSV artifacts are stored outside the
repository at `V:\MegaGlitch-Render-Quality-20260827\render-quality-1080p.json` and
`V:\MegaGlitch-Render-Quality-20260827\render-quality-1080p.csv`.
