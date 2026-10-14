# Rust graph-v1: Invert Colors

## Outcome

The opt-in Rust runtime now executes the existing `Invert Colors` graph node. A primary
Source -> Invert Colors -> Display pipeline is recognized without changing the project/preset
format. The Python selector compiles that supported graph to the Rust `invert` mode; the Rust
worker performs the full RGB8 operation in its pooled frame buffer.

Compatibility remains fail-closed. Secondary sources, Deck B, Genesis debug requests, active
global temporal macros, multiple active effects, unknown nodes, and unsupported input formats
permanently replay the current item into the existing Python worker. `invert_chance <= 0` compiles
to pass-through, and a parameter-level disabled node remains inactive.

Select the candidate with `MEGAGLITCH_RUNTIME_BACKEND=native`. Set it to `python` to roll back.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1
python tools\benchmark\run_native_runtime.py --warmup 3 --iterations 15 --label native-runtime-graph-v1 --output-dir tools\benchmark\results
```

Configuration: Windows, RGB8 1200x800, three warmups, fifteen measured frames, deterministic
gradient input, existing multiprocessing worker boundary, capture and preview excluded. The
Python worker initialized the custom CUDA-enabled OpenCV build successfully, although this node
is CPU-only in both implementations.

## Result

| Scope | Backend | Mean | p99 | FPS | 1% low |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct runtime round trip | Python thread | 2.375 ms | 3.559 ms | 421.11 | 280.99 |
| Direct runtime round trip | Rust | 0.844 ms | 1.274 ms | 1185.09 | 785.23 |
| Source -> Invert -> Display worker | Python process | 19.739 ms | 20.562 ms | 50.66 | 48.63 |
| Source -> Invert -> Display worker | Rust | 8.481 ms | 8.894 ms | 117.90 | 112.43 |

The worker outputs have the same SHA-256 hash
`0ce2c4e11be4ad5fd7257985195825255e2ea14316c048fc699816109cedd12c`.
In this run the Rust worker reduced mean time by 11.257 ms (57.0%) and p99 by 11.668 ms
(56.7%). This result is specific to the supported graph and legacy multiprocessing boundary; it
does not imply the representative mixed P-frame graph is 57% faster.

## Gate decision and next slice

Graph-v1 passes as an opt-in native compatibility/performance slice. It does not become the
default backend because coverage is deliberately narrow and this is one short benchmark run.

The mode-per-effect follow-up is now implemented as versioned
[Rust graph IR v1](../architecture/rust-graph-ir-v1.md), with its benchmark recorded in
[rust-graph-ir-v1.md](rust-graph-ir-v1.md). Dual-deck execution and coarse CUDA/GL backend
segments remain future gates.

Raw results are in `tools/benchmark/results/native-runtime-graph-v1.json` and
`tools/benchmark/results/native-runtime-graph-v1.csv`.
