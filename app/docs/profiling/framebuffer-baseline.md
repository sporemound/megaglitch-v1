# Framebuffer baseline

This is the pre-native baseline for the live frame path. The raw, machine-readable results are [baseline.json](../../tools/benchmark/results/baseline.json) and [baseline.csv](../../tools/benchmark/results/baseline.csv). No processing behavior or native runtime was added to obtain these measurements.

## Reproduction and scope

Run from the repository root in the MegaGlitch Miniforge environment:

```powershell
python tools\benchmark\run_baseline.py --label baseline --warmup 5 --iterations 30 --output-dir tools\benchmark\results
```

The run was captured on 2026-07-14 with Python 3.12.11, Windows 11, 32 logical CPUs, and an NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB). It uses 5 warm-up and 30 measured frames except the expensive saved-preset and real-worker cases, which intentionally use fewer samples. Results are synthetic steady-state frame timings, not a camera-paced session; `dropped_frames` therefore means harness deadline misses and is zero, not capture-driver loss.

The matrix covers 1280x720, 1920x1080, and the current 1200x800 working size; history lengths 1/8/32/64/120; the minimal graph and saved `node_presets/6-20-26.json` graph; Python-list and preallocated CPU history; actual worker IPC; isolated CUDA transfer/kernel work; and preview CPU preparation. Timing uses wall-clock frame latency, percentiles, FPS and 1% low FPS. CUDA cases additionally use CUDA events. The harness records process/system RAM snapshots, allocation deltas, serialized/copy bytes, output hashes, and NVIDIA utilization/VRAM snapshots.

Important limitations: preview excludes Dear PyGui's texture upload and presentation; capture and virtual-camera backends require a live external device/application and were not exercised; NVIDIA snapshots are not peak telemetry; CPU process time is coarse on Windows; and the one-frame `tracemalloc` samples perturb neither the reported steady-state timings nor claim native allocation sizes.

## End-to-end baseline

All values below are milliseconds unless noted.

| 1200x800 path | Mean | p99 | FPS | 1% low FPS |
| --- | ---: | ---: | ---: | ---: |
| In-process minimal graph | 0.993 | 1.134 | 1006.70 | 881.92 |
| Real worker minimal round trip | 19.903 | 21.214 | 50.24 | 47.14 |
| Real worker, two CUDA nodes | 30.485 | 31.950 | 32.80 | 31.30 |
| In-process saved P-frame preset | 63.787 | 68.399 | 15.68 | 14.62 |
| Real worker saved P-frame preset | 79.109 | 90.917 | 12.64 | 11.00 |

The real-worker interval begins before the input multiprocessing queue write and ends after reading the real worker's output queue. It includes serialization, graph processing, history update, deck blend, and returning the worker's result, but excludes capture, UI preview, recording, and virtual-camera submission. This is the closest reproducible end-to-end runtime boundary available without external devices.

Resolution scaling makes the representative algorithmic bottleneck unambiguous:

| Graph | 1280x720 mean / p99 | 1920x1080 mean / p99 | 1200x800 mean / p99 |
| --- | ---: | ---: | ---: |
| Minimal | 0.912 / 1.000 | 2.061 / 2.355 | 0.993 / 1.134 |
| Saved P-frame preset | 60.492 / 64.575 | 133.515 / 144.124 | 63.787 / 68.399 |

At 1200x800 the P-Frame Glitches node accounts for 62.468 ms of the 63.787 ms in-process mean; at 1080p it accounts for 130.873 ms of 133.515 ms. A native scheduler around this unchanged CPU algorithm cannot by itself reach real-time rates.

## Copies, transport, preview, and dispatch

- A multiprocessing echo of one 1200x800 RGB8 frame (2,880,000 payload bytes, plus serialization metadata) costs 7.778 ms mean / 8.993 ms p99. Three frames cost 29.368 / 31.867 ms. The real minimal worker is 18.909 ms slower than the in-process minimal path. Full-frame process transport and the multi-frame result contract are confirmed bottlenecks.
- CPU preparation for one 640x360 preview from the 1200x800 frame costs 2.315 / 2.558 ms; two previews cost 4.951 / 5.874 ms. Dear PyGui upload/presentation remains unmeasured, so these are lower bounds.
- Dispatching 1, 8, and 32 disabled nodes at 64x64 costs 0.008, 0.026, and 0.087 ms mean. Python per-node dispatch is rejected as a dominant bottleneck for this workload.
- A one-frame allocation sample retains roughly 5.76 MiB / 14 allocations for the current-size minimal case and 8.64 MiB / 23 allocations for the saved preset. At 1080p those deltas are 12.44 MiB / 14 and 18.66 MiB / 23. Steady full-frame allocation/copying exists and should be removed with resource reuse, though these samples are not a complete allocator trace.

## CUDA residency

The parent process initially imports the pip OpenCV package and reports zero CUDA devices. The spawned benchmark/processing worker explicitly loads the configured custom OpenCV module at `C:\src\opencv-cuda\build_py312\lib\python3\Release\cv2.cp312-win_amd64.pyd`, reports one device, and supports CUDA events. This environment split is itself a reproducibility risk.

| CUDA operation | 1280x720 | 1920x1080 | 1200x800 |
| --- | ---: | ---: | ---: |
| Upload | 0.537 | 0.930 | 0.533 |
| Resident Gaussian kernel | 0.355 | 0.575 | 0.319 |
| Download | 0.517 | 1.064 | 0.579 |
| Upload + kernel + download | 1.702 | 2.985 | 1.740 |
| Resident kernel, no download | 0.409 | 0.562 | 0.370 |

Host timings and CUDA-event timings are close in these isolated cases. Requiring upload and download makes the current-size operation 4.70x slower than the resident/no-download path and the 1080p operation 5.31x slower. Transfer and synchronization overhead is confirmed, but it is only a few milliseconds per crossing and is not the main cause of the CPU saved-preset result. It becomes important when a graph crosses the host/device boundary repeatedly.

VRAM was 751 MiB used before and 709 MiB after the run; utilization snapshots were 20% and 37%. Because these are point samples after child processes exit, they neither demonstrate a peak nor rule out VRAM pressure. No pinned-memory path was found or measured.

## Temporal storage

For the 1080p Row Stutter case, Python history writes take about 1.09-1.25 ms per frame and sampled lookup rises only from 0.0045 ms at length 1 to 0.0115 ms at length 120. Mean total time is 5.773 ms at length 1 and 87-99 FPS for lengths 8-120; after the effect's working delay range is populated, deeper configured history primarily consumes memory rather than causing progressive frame-time collapse.

A preallocated CPU ring writes a 1080p RGB8 frame in 0.169-0.196 ms across lengths 1-64, versus roughly 1.1 ms for the existing copied Python history. This is a useful 5-7x storage-write improvement but less than one millisecond absolute per frame. A 64-frame 1080p RGB8 ring occupies 398,131,200 bytes. A theoretical 120-frame ring would occupy 746,496,000 bytes; RGBA16F and RGBA32F equivalents would be about 1.85 GiB and 3.71 GiB.

Length 120 for the preallocated ring is marked unsupported because the existing CUDA Temporal Slit-Scan contract clamps capacity to 90. GPU history is unsupported because the current graph has no general device-resident history backend.

## Unsupported cases

Seven records are deliberately `unsupported`, not silently omitted:

- RGBA16F and RGBA32F: the live pipeline normalizes to RGB8 and has no end-to-end high-precision contract.
- Three preallocated length-120 cases: current temporal capacity is 90.
- GPU-history matrix: no general implementation exists.
- Virtual-camera submission: requires an installed backend and consuming application for a meaningful latency/drop test.

Capture on/off and download on/off are represented by synthetic input and by resident-versus-download CUDA cases, respectively. They are not claims about a particular camera driver. Recording was outside the requested matrix and remains a separate external-I/O benchmark.

## Decisions for migration

The hypotheses resolve as follows:

1. **CPU effect work dominates the representative graph — confirmed.** The P-frame node alone consumes 62.5 ms at the current size and 130.9 ms at 1080p. Its algorithm/data movement must change; moving only orchestration to native code will not fix it.
2. **Multiprocessing serialization and returning multiple full frames are material — confirmed.** The minimal real worker costs 19.9 ms, and an isolated three-frame payload costs 29.4 ms.
3. **Repeated CPU/GPU crossings are material — confirmed conditionally.** A hybrid isolated CUDA operation is 4.7-5.3x slower than a resident operation, though its absolute cost is below the representative CPU effect.
4. **Python dispatch is the main cost — rejected.** Thirty-two disabled nodes dispatch in under 0.1 ms.
5. **History lookup scales into a frame-time bottleneck — rejected for tested depths.** Lookup remains in microseconds. History copying and capacity are the relevant concerns.
6. **Preview preparation is free — rejected.** Two preview preparations consume about 5 ms before UI upload/render.
7. **VRAM exhaustion or pinned-memory behavior explains current stalls — not established.** The current instrumentation does not capture reliable peaks and no pinned path exists.

The smallest high-value subsystem to migrate first is the **frame transport and ownership boundary**, not the entire scheduler and not a temporal ring in isolation: use coarse shared/preallocated buffers, return only the master frame when auxiliary frames are not consumed, and make ownership/backpressure explicit. It attacks the measured ~19 ms minimal-worker gap and the 29 ms three-frame payload without changing effect semantics.

For the first GPU-native vertical slice, keep a small two-node CUDA chain device-resident with one upload and one final download. The isolated 4.7-5.3x residency advantage provides a clear gate. A temporal ring remains valuable for predictable memory and sub-millisecond copy reduction, but the evidence does not justify choosing it ahead of frame transport or the dominant P-frame algorithm.

Future changes should rerun this exact harness and compare JSON/CSV results. Do not combine cases captured with a different OpenCV runtime, resolution, graph, warm-up, sample count, or preview scope without labeling the difference.
