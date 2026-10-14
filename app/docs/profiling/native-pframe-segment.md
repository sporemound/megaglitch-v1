# Native P-frame segment and shared-boundary acceptance

## Outcome

The general Python graph scheduler remains intact. The saved P-frame preset's adjacent spiral-motion and uniform prediction stages now execute through one coarse native C++/Cython call. The call reuses the node's already allocated RGB8 output, consumes a small array of Python-compatible random decisions, copies selected reference macroblocks, and performs prediction amplification in place. It allocates no full-frame intermediate or coordinate maps.

The segment preserves the existing Python `random` decision sequence. With the same seed its output is byte-identical to the original two Python stages. If the extension is absent, inputs are unsupported, or the native call fails, the existing stages run normally; failure restores the random state before fallback.

CUDA was deliberately not added. The preceding source/history ownership and following P-frame operations, graph routing, preview, recording, and virtual-camera consumers are currently host-resident. Uploading and downloading around only these two operations would not satisfy the resident-segment gate. A future CUDA segment must retain a `GpuMat` across several compatible adjacent nodes and demonstrate an end-to-end worker improvement.

## Shared-boundary acceptance

The spawned-worker acceptance test processes 60 dual-deck frames while alternating hard crossfader endpoints, replacing Display-only graphs with deterministic invert graphs, resetting worker state halfway through, and reusing shared slots. It verifies:

- the worker's canonical frame owns both crossfader endpoints;
- decoded frames remain stable after their output slot is reused, as required by preview, recording, and virtual-camera consumers;
- graph replacement and reset preserve the canonical contract;
- no input or output slots remain owned after the soak;
- secondary-input pool saturation rolls back an already claimed primary slot;
- three repeated owner/worker start-process-stop-close cycles exit without forced termination or retained slots.

Oversize canonical fallback, stale-output reclamation, input release on failure, and idempotent shutdown remain covered by the lower-level transport tests. Actual camera drivers, virtual-camera presentation, interactive Dear PyGui closure, and file encoding are not activated by this automated test and remain manual acceptance items.

## Isolated measurements

Command:

```powershell
python tools\benchmark\run_pframe_hotstage.py --label pframe-native-segment --warmup 3 --iterations 20
```

Configuration: RGB8, saved-preset values (`motion=0.451`, spiral, block size 16; `prediction=0.978`, uniform, threshold 30), seed 1984, three warmups, twenty measurements.

| Resolution | Implementation | Mean ms | p99 ms | Relative to split path |
| --- | --- | ---: | ---: | ---: |
| 1200x800 | Original motion + NumPy prediction | 63.086 | 68.456 | — |
| 1200x800 | Python motion + native prediction | 18.233 | 19.402 | baseline |
| 1200x800 | Fused native segment | 4.656 | 4.777 | 3.92x faster |
| 1920x1080 | Original motion + NumPy prediction | 143.637 | 154.486 | — |
| 1920x1080 | Python motion + native prediction | 40.394 | 41.624 | baseline |
| 1920x1080 | Fused native segment | 10.499 | 11.623 | 3.85x faster |

Both fused cases report `pixel_exact=true` and `max_channel_delta=0`. The only segment scratch allocation is the random-decision array: 29,008 bytes at 1200x800 and 63,784 bytes at 1920x1080. The output is the existing node output allocation.

## Full worker measurements

Command:

```powershell
python tools\benchmark\run_shared_transport.py --label shared-transport-pframe-segment --warmup 5 --iterations 30
```

The spawned 1200x800 representative P-frame case uses two warmups and fifteen measured frames. Timing includes parent encoding, queue transport, graph execution, and parent decoding.

| Transport | Mean ms | p99 ms | FPS | 1% low FPS |
| --- | ---: | ---: | ---: | ---: |
| Legacy serialization | 33.050 | 42.611 | 30.26 | 23.47 |
| Shared canonical frame | 20.834 | 22.217 | 48.00 | 45.01 |

Shared transport removes 12.216 ms mean (37.0%) and 20.394 ms p99 (47.9%) from the current fused graph. Both shared cases finish with zero owned input/output slots and no dropped frames. Compared with the earlier shared path containing Python motion plus native prediction (33.596 ms mean), the fused shared worker is 38.0% faster. Separate benchmark invocations contain normal machine-load variation, so paired rows within one artifact are the transport comparison.

Raw results are `tools/benchmark/results/pframe-native-segment.json` and `tools/benchmark/results/shared-transport-pframe-segment.json`, with CSV equivalents.

## Remaining copies and next gate

The P-frame registry still creates its initial output copy and returns copies for output/history ownership. Shared transport copies the program frame into its output slot and the UI copies it out for independent consumer lifetime. These are now more important than scheduler dispatch, but changing them requires an explicit ownership redesign rather than aliasing mutable history.

The project is ready for an adapted Prompt 7 focused on the current Python-hosted worker lifecycle: owner allocation, worker readiness handshake, processing readiness, integration startup, command rejection, ordered worker/integration shutdown, and partial-initialization cleanup. Prompt 7 must not assume that the experimental Rust worker or a CUDA graph owns arbitrary production graphs. Prompt 6 remains closed.
