# Rust graph IR v1 benchmark

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1
python tools\benchmark\run_native_runtime.py --warmup 3 --iterations 15 --label native-runtime-graph-ir-v1 --output-dir tools\benchmark\results
```

Configuration: Windows, RGB8 1200x800, deterministic gradient input, three warmups, fifteen
measured frames, legacy multiprocessing worker transport, capture and preview excluded. The graph
is Input -> Brightness(+17)/Contrast(-25) -> Invert -> Output. The custom CUDA-enabled OpenCV
worker initialized successfully, although these operations are CPU-only.

## Result

| Worker | Mean | p99 | FPS | 1% low |
| --- | ---: | ---: | ---: | ---: |
| Python graph worker | 30.934 ms | 32.288 ms | 32.33 | 30.97 |
| Rust graph IR v1 | 10.245 ms | 10.748 ms | 97.61 | 93.04 |

Both outputs hash to
`32f9f24967841a63f3d9f746a3cba8f51dbf5c2ea82623e821b1e62508c1d948`.
In this run Rust reduced mean time by 20.688 ms (66.9%) and p99 by 21.540 ms (66.7%). This is a
narrow supported CPU graph over the legacy worker transport; it is not a performance claim for
mixed CUDA, GLSL/Lua, dual-deck, recording, or preview paths.

## 2026-08-28 AC release-gate observation

The full Python-host gate for source `b8f4776` passed 384 automated tests,
runtime/CUDA diagnostics, and the production P-frame, shared-worker, and
lifecycle thresholds. The strict combined checker stopped only on experimental
Rust Graph IR timing. Its gate sample recorded 16.875 ms mean and 25.174 ms p99
against the unchanged 14/16 ms limits.

A follow-up run in the same artifact directory measured:

| Worker | Mean | p99 | FPS | 1% low |
| --- | ---: | ---: | ---: | ---: |
| Python graph worker | 41.071 ms | 51.725 ms | 24.35 | 19.33 |
| Rust graph IR v1 | 17.276 ms | 23.021 ms | 57.88 | 43.44 |

The Rust/Python ratios were 0.421 mean and 0.445 p99, below the retained 0.75
relative limits. Both workers produced
`32f9f24967841a63f3d9f746a3cba8f51dbf5c2ea82623e821b1e62508c1d948`,
and Graph IR pixel equality remained true. The raw follow-up artifact is
`V:\MegaGlitch-Render-Quality-20260827\source-build-b8f4776-ac\release-benchmarks-49c6aa4318eb4b159bd4e19636d5d9a8\native-runtime-graph-ir-v1.json`.

The operator observed AC power with Windows exposing only the Balanced scheme
and an NVIDIA 30 W power cap. That context is incomplete: this graph is CPU-only,
and the harness does not record CPU clocks, package power, thermal state, or
power-plan identity. The 30 W GPU cap therefore does not explain the timing
miss, and this run cannot replace the recorded threshold profile or justify
loosening it. It is retained as evidence that Rust performance needs a
profile-controlled rerun; it is not evidence of a Python/shared release-path
regression.

## Gate

IR v1 passes as the new opt-in Rust graph-selection architecture. Production default remains
Python until coverage and repeated end-to-end testing expand. The next scheduler gate is explicit
dual-input/value-slot execution and master blend. The next performance gate is a coarse
device-resident CUDA segment represented as one graph operation, not per-node FFI calls.

The dedicated checker remains strict for Rust promotion. The Python-host
release gate treats only Rust timing as advisory while retaining hard Rust
correctness checks and all production-path thresholds.

Raw results are `tools/benchmark/results/native-runtime-graph-ir-v1.json` and
`tools/benchmark/results/native-runtime-graph-ir-v1.csv`.
