# Shared frame transport and P-frame hot stage

## Decision

Prompt 6 remains closed: its scheduler-migration performance premise failed. The Python graph scheduler and UI stay intact.

The current Python worker boundary now has a fixed shared-memory transport, enabled by default and reversible with `MEGAGLITCH_FRAME_TRANSPORT=legacy`. Queue messages contain small descriptors and metadata. The worker result has one authoritative program frame; it no longer serializes redundant deck A/deck B frame arrays. The measured P-frame hot stage has also been replaced by an exact native CPU segment. CUDA is not used for this segment because the preceding motion-vector operation and following graph/output work are host-resident today; an isolated upload/kernel/download would violate the resident-ownership gate.

## Transport contract and ownership

`SharedFrameTransportOwner` creates four input slots and two output slots by default for the application. At the configured 1920x1080 RGB8 ceiling this is 37,324,800 bytes. The 1200x800 benchmark uses 17,280,000 bytes. A shared bitmap plus process-safe bounded semaphore controls ownership; a slot has exactly one owner at a time.

The lifecycle is:

1. The UI process claims an input slot and copies the source frame into it.
2. The queue carries a descriptor, graph metadata, and optional secondary-frame descriptor.
3. The worker reads the shared input view without another queue-unpickle frame allocation, processes it, and releases the input slot in `finally`.
4. The worker claims an output slot and copies only the canonical program output into it.
5. The UI process copies that output into its local canonical frame and immediately releases the slot.

Stale queue items, queue-put failures, reset, normal shutdown, forced worker termination, and repeated close are covered by explicit reclamation paths. Frames larger than the configured capacity use a safe legacy input payload and a canonical legacy output; they do not resize live shared regions. The application capacity can be changed with `MEGAGLITCH_SHARED_FRAME_MAX_WIDTH` and `MEGAGLITCH_SHARED_FRAME_MAX_HEIGHT`.

The legacy result included `frame`, `frame_a`, and `frame_b`. Shared mode deliberately makes the worker's `frame` authoritative and returns `frame_a=None` and `frame_b=None`. Consequently the UI no longer reapplies its post-dequeue crossfader endpoint correction to deck copies. The worker's graph/crossfader result is the canonical output. Set `MEGAGLITCH_FRAME_TRANSPORT=legacy` for exact legacy boundary behavior. The separate experimental Rust worker remains incompatible with this shared Python-worker transport and is not selected implicitly.

## Worker-boundary measurements

Command:

```powershell
python tools\benchmark\run_shared_transport.py --label shared-transport --warmup 5 --iterations 30
python tools\benchmark\run_shared_transport.py --label shared-transport-pframe-optimized --warmup 5 --iterations 30
```

Configuration: Windows, CPython 3.12, spawned worker, RGB8 1200x800, sequential round trips, saved `node_presets/6-20-26.json`, CPU graph execution (`use_cuda=False`). Timing includes parent encode, queue round trip, graph execution, and parent decode. P-frame cases use 15 measured frames and two warmups. No frames were dropped.

| Graph / implementation | Transport | Mean ms | p99 ms | FPS | 1% low FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Minimal | legacy | 12.512 | 13.984 | 79.92 | 71.51 |
| Minimal | shared | 5.226 | 5.855 | 191.35 | 170.79 |
| P-frame, original NumPy hot stage | legacy | 91.429 | 108.779 | 10.94 | 9.19 |
| P-frame, original NumPy hot stage | shared | 77.126 | 83.001 | 12.97 | 12.05 |
| P-frame, native hot stage | legacy | 53.347 | 57.151 | 18.75 | 17.50 |
| P-frame, native hot stage | shared | 33.596 | 36.800 | 29.77 | 27.17 |

On the pre-optimization P-frame graph, shared transport removes 14.303 ms mean (15.6%) and 25.778 ms p99 (23.7%). After the native hot-stage change, shared transport removes 19.751 ms mean (37.0%) and 20.351 ms p99 (35.6%). The minimal-graph paired run removes 7.286 ms mean (58.2%); its absolute values should not be compared across separate benchmark invocations because worker startup and machine load vary.

At 1200x800 the representative serialized queue sample falls from 2,880,926 bytes to an 820-byte descriptor/metadata payload. The optimized run ends with zero input and zero output slots in use for both shared cases. Output hashes match between legacy and shared for the deterministic minimal graph. The P-frame graph contains unseeded random behavior, so separate spawned runs are not expected to hash-match.

Raw results are in `tools/benchmark/results/shared-transport.json` and `tools/benchmark/results/shared-transport-pframe-optimized.json` (plus CSV equivalents).

## P-frame hot-stage measurements

Profiling the active saved preset identified uniform prediction-error amplification—not scheduler dispatch—as the dominant stage. The replacement is a native C++ loop in the existing Cython extension. It avoids the float residual, mask, amplification, `where`, and result-sized temporary arrays and accepts a preallocated output. It matches the prior NumPy result byte-for-byte; unsupported layouts still take the Python implementation.

Command:

```powershell
python tools\benchmark\run_pframe_hotstage.py --warmup 3 --iterations 20
```

| Resolution | Stage | Mean ms | p99 ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1200x800 | original NumPy uniform prediction | 50.707 | 53.008 | — |
| 1200x800 | native, preallocated | 3.277 | 3.609 | 15.47x |
| 1920x1080 | original NumPy uniform prediction | 104.968 | 110.427 | — |
| 1920x1080 | native, preallocated | 6.810 | 7.338 | 15.41x |

Both native cases report `pixel_exact=true` and `max_channel_delta=0`. The adjacent spiral motion-vector stage remains host-side at 13.747 ms (1200x800) and 32.945 ms (1920x1080), so it is now the next measured P-frame target.

## Next gate

Do not migrate the general scheduler. The next optimization candidate is a coarse native P-frame segment only if it can combine the spiral motion work, prediction work, and adjacent compatible nodes under one ownership boundary. A CUDA variant is justified only when that whole segment retains a `GpuMat` across its operations and avoids per-frame host/device round trips. Re-run the full worker benchmark before enabling such a segment by default.

That CPU-native segment has now passed its gate. See [native-pframe-segment.md](native-pframe-segment.md) for compatibility tests, current measurements, the CUDA deferral, and adapted Prompt 7 readiness.
