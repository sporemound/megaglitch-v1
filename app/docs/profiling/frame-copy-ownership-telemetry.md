# Frame-copy ownership telemetry

## Purpose

The first prioritized item after the accepted fused P-frame segment was to quantify explicit
full-frame copies before changing aliases or lifetimes. The shared-transport benchmark now carries
copy metadata with each result and records these ownership boundaries:

- `shared_input`: application frame copied into an input shared-memory slot;
- `initial_node`: source node takes its own frame copy;
- `history`: processed deck output retained in temporal history;
- `shared_output`: canonical worker output copied into an output shared-memory slot;
- `ui_consumer`: application decodes the output slot into a stable consumer-owned array.

Each category reports a count and byte total. Transport-owner statistics also retain cumulative
input and consumer copy totals for leak and soak checks. The counters are metadata only and do not
change pixel data, queue policy, slot ownership, fallback behavior, or graph execution.

## Reproduction

Run from the repository root in the configured CUDA-enabled CPython 3.12 environment:

```powershell
python tools\benchmark\run_shared_transport.py `
  --label frame-copy-ownership `
  --warmup 5 `
  --iterations 30 `
  --output-dir tools\benchmark\results
```

The run used the custom OpenCV 4.12 CUDA extension at
`C:\src\opencv-cuda\build_py312\lib\python3\Release\cv2.cp312-win_amd64.pyd`, one CUDA device,
1200x800 RGB8 frames, `use_cuda=False` for graph execution, five warm-up and thirty measured
minimal-graph frames, and two warm-up and fifteen measured representative P-frame frames. The
representative cap is part of the existing harness.

Raw results are in `tools/benchmark/results/frame-copy-ownership.json` and `.csv`.

## Results

| Graph | Transport | Mean / p99 | FPS / 1% low | Explicit copies per result | Explicit bytes per result |
| --- | --- | ---: | ---: | ---: | ---: |
| Minimal | legacy | 12.129 / 13.275 ms | 82.44 / 75.33 | 0 observed | 0 observed |
| Representative P-frame | legacy | 87.739 / 94.867 ms | 11.40 / 10.54 | 2 observed | 5.493 MiB |
| Minimal | shared | 6.235 / 8.474 ms | 160.40 / 118.01 | 3 | 8.240 MiB |
| Representative P-frame | shared | 72.715 / 76.730 ms | 13.75 / 13.03 | 5 | 13.733 MiB |

The shared minimal path copies one 2,880,000-byte frame into the input slot, one into the output
slot, and one into application-owned memory. The representative graph adds one source-node copy
and one history-retention copy of the same size.

These timings are a new instrumentation run, not evidence that telemetry improved performance.
The deterministic unit tests verify that source-copy accounting preserves output and that shared
output decoding reports stable consumer ownership.

## Limits and next gate

The legacy counts are lower bounds: Python multiprocessing serialization copies are not visible to
the explicit NumPy/shared-slot instrumentation. The `ui_consumer` boundary is the stable array
returned by shared-result decoding; Dear PyGui's internal texture upload/presentation, recording,
and virtual-camera consumer copies remain outside this measurement. Contiguity normalization can
also allocate before a slot copy when an input is not already C-contiguous; representative inputs
in this run were contiguous.

The next ownership experiment should remove or alias only one boundary at a time and rerun this
exact benchmark plus deterministic output tests. Initial candidates are the source-node copy and
history copy. The input/output/UI copies express cross-process or consumer lifetime boundaries and
must not be removed without replacing their ownership guarantees. This evidence does not justify
general graph-scheduler migration or a new GPU backend by itself.
