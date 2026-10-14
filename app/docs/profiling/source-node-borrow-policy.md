# Source-node borrow policy

## Decision

MegaGlitch now has a reversible `borrow_unique` source-copy policy for the Python-hosted
worker. The compatibility policy remains `copy`. Set
`MEGAGLITCH_SOURCE_COPY_POLICY=borrow_unique` before launch to exercise the candidate; unset
the variable (or set it to `copy`) to roll back without changing projects or presets. Unknown
values fail closed to `copy` at the application/worker boundary.

The policy borrows a source array only when that source key occurs exactly once in a flattened
pipeline. Repeated references retain independent copies so graph branches cannot alias the same
writable array. This is a coarse-grained ownership change inside the existing Python-hosted
runtime; it does not change the selected host architecture or make Rust the default scheduler.

## Reproduction

```powershell
python tools\benchmark\run_source_copy_policy.py --warmup 3 --iterations 15 --label source-copy-policy --output-dir tools\benchmark\results
```

Configuration: Windows, custom CUDA-enabled OpenCV worker, shared transport, RGB8 1200x800,
saved representative P-frame preset `node_presets/6-20-26.json`, three warmups, fifteen measured
frames, deterministic per-frame seed base `406013950`.

## Result

| Policy | Explicit copies/frame | Bytes copied/frame | Mean | p99 | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| `copy` | 5 | 14.40 MB | 71.939 ms | 77.725 ms | reference |
| `borrow_unique` | 4 | 11.52 MB | 71.514 ms | 73.223 ms | pixel-exact |

The candidate eliminates the measured initial source-node copy: one 2.88 MB copy per frame at
this resolution. The single short run shows only a 0.6% mean-time difference, which is not enough
to claim a performance improvement or change the production default. The useful result is the
verified ownership/copy reduction. Promotion requires repeated end-to-end measurements across
representative single- and dual-deck graphs, including temporal, Lua/GLSL, CUDA, recording, and
virtual-camera paths.

Raw results are committed as `tools/benchmark/results/source-copy-policy.json` and
`tools/benchmark/results/source-copy-policy.csv`.
