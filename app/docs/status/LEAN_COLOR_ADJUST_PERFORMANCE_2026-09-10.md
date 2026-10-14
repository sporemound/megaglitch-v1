# LEAN-003: measured Color Adjust RGB8 optimization

## Value and baseline

R3: approximately 60 new preview frames per second with effect coverage disclosed. Preserve R1/R4 Color Adjust behavior and LEAN-001/002 fixes. Scope is the isolated MegaGlitch-Node0.9.4 checkout based on `ff8b53c`; V: remains read-only reference.

LEAN-002 measured 26.30 new frames/s at 960x540 with Color Adjust enabled. A fresh 60-frame synchronous profile attributes 0.917 s to gamma and 0.763 s to HSV out of 1.986 s total. Repeated full-frame floating-point work accounts for about 85% of sampled time. Python dispatch and color conversion are not the main measured costs. Traced peak allocation footprint over a separate ten-frame window is 48,216,928 bytes (not complete process/native/GPU memory).

## Bounded change

Use bounded cached lookup tables for the existing Color Adjust RGB8 path. Retain the existing floating-point functions for non-RGB8 inputs and GPU mode. Preserve byte-level output by building tables with the original float32/float64 arithmetic, including the different gamma arithmetic when HSV was skipped. No language migration, pipeline restructuring, new effect, or parameter change.

## Acceptance

- [x] Exact differential pixels across control combinations and all byte values.
- [x] Existing graph, preview, program, surface and temporal checks pass.
- [x] Same-workload before/after throughput and p95 measured.
- [x] Profile and traced-memory comparison recorded separately from timing runs.
- [x] Record limitations, compatibility, and rollback.

Baseline index tree after LEAN-002: `8ac37abc6b58b4a45d22a265fbe152a92323f750`. Completed locally on `codex/lean-preview-coverage`; staged, not committed, pushed, merged, or packaged. No V: changes.

## Implementation and compatibility

`glitch_effects.py` adds a CPU RGB8 HSV/gamma helper. Its 32-entry LRU cache holds immutable tables, with at most 32 KiB of table data plus Python/cache metadata. HSV component transforms use the same float32 arithmetic as the retained implementation. Gamma tables distinguish float32 input after HSV from the float64 promotion of untouched byte input. Tables are regenerated when controls change and can be safely shared across frame streams; they contain no frame/history data.

`effects_registry.py` selects this path only for three-channel uint8 arrays in CPU mode with finite parameters. Other cases retain the previous functions. The optimized helper returns final RGB8 values directly, removing full-frame floating-point intermediates and final conversion. The existing frame pipeline and multimodal selection normalize effect results to RGB8 before subsequent processing, preserving their byte contract. Original helper functions remain unchanged.

This optimization applies to Color Adjust through the shared registry, including program processing, not just preview. It changes computation cost without changing intended pixels, parameters, graph ownership, or enabled-node coverage. No GPU residency path or cross-language API changed.

## Verification results

- OpenCV headless 5.0.0.93: **108 tests passed in 0.54 seconds** for optimized/reference equivalence and Color Adjust workflow behavior.
- Installed custom OpenCV 4.12.0: **159 tests and four subtests passed in 2.59 seconds**, including preview, master isolation, shared surface, graph compilation, and both earlier LEAN fixes.
- Differential coverage: 96 hue/saturation/gamma combinations, every grayscale byte value plus seeded color samples, more than 32 parameter revisions to exercise cache eviction, noncontiguous frames, defaults, disabled controls, and source preservation.
- Before/after profiled frame SHA-256 is identical: `742b0916b9d86c009e2a6023382e164c7e81b3bcaa46cdbe2333bdd9f8f7241b`.
- `git diff --check` passed. No production edits followed verification.

Environment: Windows 11, Intel64 Family 6 Model 183 Stepping 1, isolated Python 3.12.14, NumPy 2.5.3, pytest 9.1.1. Throughput and profile runs use the installed custom OpenCV 4.12.0 selected by the existing bootstrap. That build detects one CUDA device; the measured Color Adjust path has GPU execution disabled. No dependency installations or native builds occurred.

## Same-workload performance

Both runs enable hue 30, saturation 0.5, gamma 1.2. Input is a 1920x1080 synthetic gradient with a changing sequence marker; preview is fixed at 960x540. Each run uses 30 warmup frames and 300 measured new renders, one-at-a-time requests to the asynchronous executor, with adaptation disabled. Neither run skips Color Adjust.

| Measurement | Before LEAN-003 | After LEAN-003 |
|---|---:|---:|
| Mean render time | 36.73 ms | 3.02 ms |
| p95 render time | 41.72 ms | 3.42 ms |
| New render throughput, including submit/wait | 25.99 FPS | 217.91 FPS |
| Dropped / superseded | 0 / 0 | 0 / 0 |
| Separate traced peak, ten warmed frames | 48,216,928 bytes | 27,999,240 bytes |

Observed throughput improved about 8.4× on this workload. Profiling separately confirms the hot operations moved from full-frame floating-point gamma/HSV calculations to color conversions and byte lookup operations. Per-frame source/history copying is retained; the synchronous profile still records one request source-array copy and one history copy per frame. The former five full-frame `astype` calls per frame disappear from the measured hot path.

Tracemalloc reports traced Python/NumPy allocation footprint, not full process RAM, all native allocations, allocation count, or VRAM. Its window is independent of throughput measurement. No CPU/GPU transfers are introduced by the CPU lookup path; transfer counters were not instrumented. End-to-end UI latency, decoding, GUI presentation/scanout, devices, packaging, and mixed temporal-graph performance remain unmeasured. Single before/after runs establish a bounded improvement, not whole-application 60 FPS certification.

## Reproduction

Run from `work/MegaGlitch-Node0.9.4`:

```powershell
../lean-test-env/Scripts/python.exe -m pytest -q tests/test_color_adjust_rgb8.py tests/test_color_adjust_workflow.py --tb=short
../lean-test-env/Scripts/python.exe -c "from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults; bootstrap_for_dev_defaults(); import cv2, pytest; print('OpenCV', cv2.__version__, cv2.__file__); raise SystemExit(pytest.main(['-q','tests/test_color_adjust_rgb8.py','tests/test_color_adjust_workflow.py','tests/test_preview_executor.py','tests/test_preview_master_isolation.py','tests/test_rust_ui_bridge.py','tests/test_session_graph_compiler.py','--tb=short']))"
../lean-test-env/Scripts/python.exe tools/benchmark/run_color_adjust_preview.py --label after-lean-003 --output ../../outputs/lean-003-preview-after.json
../lean-test-env/Scripts/python.exe tools/benchmark/profile_color_adjust.py --label after --output ../../outputs/lean-003-profile-repro-check.json
git diff --check
```

The before throughput used the same benchmark command with `before-lean-003` and `lean-003-preview-before.json` before production edits. Initial before/after profiles used `../profile_color_adjust.py before` and `../profile_color_adjust.py after`; that scratch harness was then retained as `tools/benchmark/profile_color_adjust.py` with explicit label/output arguments. Its reproduction smoke produced the same frame hash and traced peak. Re-running a command labeled "before" on fixed code does not recreate the baseline; use the recorded baseline tree in an isolated checkout.

## Changed files and next pull

- `glitch_effects.py`: bounded lookup tables and RGB8 helper.
- `effects_registry.py`: guarded Color Adjust routing.
- `tests/test_color_adjust_rgb8.py`: 98 reference-equivalence/cache/input-layout cases.
- `tools/benchmark/profile_color_adjust.py`: reusable profile and traced-memory harness.
- This document: requirements, evidence, measurements, and rollback.

The standalone LEAN-003 patch is relative to the LEAN-002 index tree recorded above. Roll back only this patch if needed, keeping the earlier parity/reporting fixes. The original reference math remains available for comparison and fallback.

Next pull: validate Color Adjust alongside an existing temporal effect in a representative graph, checking combined throughput and history isolation before selecting another optimization. Do not extrapolate the isolated node speedup to all effects or the desired headless-render aesthetic.
