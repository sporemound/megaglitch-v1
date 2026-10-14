# Native temporal-ring prototype results

The C++/Cython temporal prototype tests fixed native ownership, constant-time delay indexing, and coarse native blending without migrating the graph engine. Raw results are [temporal-prototype.json](../../tools/benchmark/results/temporal-prototype.json) and [temporal-prototype.csv](../../tools/benchmark/results/temporal-prototype.csv). The boundary decision is recorded in [native-backend-decision.md](../architecture/native-backend-decision.md).

## Build and reproduce

```powershell
powershell -ExecutionPolicy Bypass -File native\temporal_ring\build.ps1 -Force
python -m unittest tests.test_native_temporal
python tools\benchmark\run_temporal_prototype.py --label temporal-prototype --warmup 5 --iterations 30 --output-dir tools\benchmark\results
```

The final run contains 90 measured cases and no unsupported cases: three resolutions, five history lengths, list writes, preallocated NumPy writes/mixes, native writes/mixes, and the existing OpenCV optimized CPU blend reference. As with the framebuffer baseline, allocator tracing is performed for one separate steady-state operation and is excluded from timing.

Select the implementation with `MEGAGLITCH_HISTORY_BACKEND=native`, `python`, or `auto`. The application continues to use its old list history unless the opt-in `FrameState.get_temporal_history(...)` boundary is requested, so the effective production rollback is immediate and preserves current visuals.

## Correctness and lifecycle

The Python and native implementations produce byte-identical basic mix output in the parity test. Tests cover ring wrap and delay order, freeze/unfreeze, clear, resize/reconfigure, invalid delay/mix/dimensions, impossible allocation-size rejection, five repeated initialize/shutdown cycles, cleanup after an exception, and `FrameState.reset()` cleanup. `shutdown` is idempotent and a shut-down object can be configured again.

`read_frame` copies one frame into Python deliberately; this prevents a NumPy view from outliving native storage after resize or shutdown. `process_temporal_mix` avoids that readback and directly consumes native history. Supplying an output array avoids output allocation after warm-up.

## Measured comparison

Representative 64-frame RGB8 results:

| Resolution and operation | Existing list | Python preallocated | Native C++ | Existing OpenCV blend reference |
| --- | ---: | ---: | ---: | ---: |
| 1200x800 write mean / p99 | 0.439 / 0.456 ms | 0.088 / 0.097 ms | 0.102 / 0.145 ms | n/a |
| 1200x800 mix mean / p99 | n/a | 16.635 / 19.674 ms | 2.985 / 3.122 ms | 0.461 / 0.480 ms |
| 1920x1080 write mean / p99 | 1.009 / 1.098 ms | 0.179 / 0.278 ms | 0.180 / 0.196 ms | n/a |
| 1920x1080 mix mean / p99 | n/a | 37.409 / 40.087 ms | 6.075 / 6.710 ms | 1.001 / 1.092 ms |

Across history lengths 1-120, native write means remain 0.081-0.102 ms at 1200x800 and 0.180-0.219 ms at 1080p. Native mix means remain 2.705-3.024 ms and 5.976-6.109 ms, respectively. History length therefore changes capacity, not the cost of write, lookup, or one delayed-frame mix. The ring uses the required modulo index and never shifts storage.

One sampled steady-state native operation retains only about 152 bytes of Python tracing metadata, not a full frame. The list write sample retains approximately one full new frame (2,880,248 bytes at 1200x800 and 6,221,048 bytes at 1080p). Both preallocated backends avoid per-frame history allocation after configuration.

The native ring improves the current list write path by roughly 4-6x, but that gain is not uniquely native: NumPy assignment to a preallocated ring is effectively tied. The native scalar mix is roughly 5-6x faster than the reference facade's NumPy float expression, but it is roughly 6x slower than the existing SIMD-optimized OpenCV CPU blend. This prototype should not replace `cv2.addWeighted`.

## Memory requirements

The implemented storage is RGB8 (3 bytes/pixel):

| Resolution | 1 | 8 | 32 | 64 | 120 frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1280x720 | 2.64 MiB | 21.09 MiB | 84.38 MiB | 168.75 MiB | 316.41 MiB |
| 1200x800 | 2.75 MiB | 21.97 MiB | 87.89 MiB | 175.78 MiB | 329.59 MiB |
| 1920x1080 | 5.93 MiB | 47.46 MiB | 189.84 MiB | 379.69 MiB | 711.91 MiB |

Potential 1080p tiers, not implemented:

| Format/tier | Bytes/pixel | 1 frame | 120 frames |
| --- | ---: | ---: | ---: |
| Luma8 | 1 | 1.98 MiB | 237.30 MiB |
| RGB8 implemented | 3 | 5.93 MiB | 711.91 MiB |
| RGBA8 | 4 | 7.91 MiB | 949.22 MiB |
| RGBA16F | 8 | 15.82 MiB | 1.85 GiB |
| RGBA32F | 16 | 31.64 MiB | 3.71 GiB |

A half-width/half-height tier uses one quarter of these values. A short RGBA16F feedback ring plus a longer compact RGB8, luma, or reduced-resolution ring remains feasible in the API, but there is no benchmark evidence to implement it yet.

## Remaining copies and synchronization

- `write_frame` performs one host-to-host full-frame copy into native storage.
- `read_frame` performs one full-frame copy back to a new NumPy array.
- `process_temporal_mix` reads native history and writes one output surface; it does not export the historical frame.
- Calls still originate in Python once per coarse write or mix. There are no per-pixel callbacks.
- This CPU prototype performs no upload, download, CUDA synchronization, pinned allocation, or VRAM allocation.
- The live graph/list adapter has not been replaced, so existing per-frame list history remains in the production hot path until a separately gated integration is warranted.

## Gate decision

The prototype validates deterministic native ownership and a safe coarse API, but it **does not justify migrating more of the engine on temporal-ring performance alone**. Preallocation, not language choice, explains the write improvement, and existing OpenCV is substantially faster for the tested blend. The previous baseline's larger targets remain process/frame transport and the P-frame algorithm; for GPU graphs, a device-resident vertical slice remains the stronger native experiment.

Keep this prototype opt-in as an ownership/lifecycle reference. Do not proceed to broad native-runtime integration until a native boundary eliminates measured process copies, keeps multiple GPU operations resident, or implements a demonstrably faster temporal algorithm rather than duplicating optimized OpenCV work.
