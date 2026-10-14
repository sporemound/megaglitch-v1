# LEAN-002: Color Adjust controls reach preview and program output

## Value and requirements

- R1: existing Python capabilities work through the Rust UI.
- R3: preview executes the disclosed effects; new-frame performance is measured separately.
- R4: graph parameter changes reach processing and preview.
- Release checks: core image processing (PROJECT_SCOPE.md), Gates C and I regression coverage.

## Reference reconciliation

V: is an investigation/reference source only. `V:/MegaGlitch-Node0.9.4` is clean at `d4b62cc78ec9153c166953aafe081660840bea91`, branch `release/megaglitch-stabilization`: eight commits beyond successor `26ce7791bf6d8781aa8fe8f186a5c6c9fa7e8760`, twelve beyond scoped recovery `ff8b53c9376c271c1b7ce8ca95f38065f3073603`. `V:/megaglitch` is empty, not another checkout.

The eight later commits add audio/modulation, route controls, capability checks, a test-pattern overflow fix, and packaging investigations. Historical `V:/MegaGlitch-build/gui-comparison-20260905/RESULTS.md` still reports partial parity, failed packaged audio, and uncertified isolated render dependencies. Those are historical results, not tests rerun here.

Color Adjust preview support and saturation-zero corrections already exist in V:, predating those eight commits. The existing scoped Rust node catalog exposes hue, saturation, and gamma. Scoped preview excludes Color Adjust, and both scoped `effects_registry.py` and `glitch_effects.py` incorrectly treat zero saturation as unchanged color.

## Smallest change

Reuse the V: saturation-zero fixes and allow this one existing Color Adjust node in the scoped preview plan. Do not copy V:'s blanket acceptance of any backend identity. Keep LEAN-001's truthful fallback reporting; V: still lacks it. No new node, language boundary, runtime abstraction, or UI control is required.

## Acceptance

- [x] Reproduce the missing preview and saturation-zero behavior.
- [x] Verify hue, saturation, gamma, defaults, and disabled behavior on real pixels.
- [x] Verify revisioned GUI graph commands reach the mapped preview surface and match program output at equal resolution.
- [x] Verify reduced-resolution preview leaves the source unchanged and respects existing isolation checks.
- [x] Record focused regression results and a bounded preview measurement with coverage disclosed.

Status: completed locally on `codex/lean-preview-coverage`. The scoped checkout remains based on `ff8b53c`, with LEAN-001 retained. No V: files were modified. No commits, remote pushes, merges, or packaged releases were made.

## Verification

The new workflow tests reproduced nine failures and one passing disabled-control case before the production change. Afterward, the focused command below passed **61 tests and four subtests in 10.34 seconds**. No further production changes followed that run.

```powershell
# Run from work/MegaGlitch-Node0.9.4.
../lean-test-env/Scripts/python.exe -m pytest -q tests/test_color_adjust_workflow.py --tb=short
../lean-test-env/Scripts/python.exe -m pytest -q tests/test_color_adjust_workflow.py tests/test_preview_executor.py tests/test_preview_master_isolation.py tests/test_rust_ui_bridge.py tests/test_session_graph_compiler.py --tb=short
git diff --check
```

Tests include formula-derived HSV saturation-zero grayscale, omitted/None defaults, hue and gamma changes, disabled-node behavior, equal-resolution preview/program pixels, four graph revisions published to the UI's shared frame surface, and 1280x720 source preservation with 640x360 preview. Existing temporal isolation and LEAN-001 fallback regressions remain passing. Independent code review found no actionable issue; it did not rerun the suite.

The test environment uses isolated Python 3.12.14, NumPy 2.5.3, pytest 9.1.1, and OpenCV headless 5.0.0.93. No dependency installation was needed for this pull. The benchmark runner uses the repository's existing bootstrap, which selects the installed custom OpenCV 4.12.0 at `C:/src/opencv-cuda/build_py312/lib/python3/Release/cv2.cp312-win_amd64.pyd`; this differs from test import order. That build detects one CUDA device, but this benchmark's `gpu_ops` execution mode is CPU. No existing OpenCV installation was changed.

## Measured preview behavior

Each run used a 1920x1080 synthetic RGB gradient with a changing sequence marker, 30 warmup renders and 300 measured renders, fixed preview resolution, and one-at-a-time requests to the asynchronous executor. Enabled settings: hue 30, saturation 0.5, gamma 1.2. Host: Windows 11, Intel64 Family 6 Model 183 Stepping 1. These are single-run observations, not a sustained packaged-application performance certificate.

| Run | Coverage | Preview size | Mean render ms | p95 render ms | New render FPS | Dropped / superseded |
|---|---|---|---:|---:|---:|---|
| Before fix | Color Adjust unsupported and skipped | 960x540 | 0.82 | 0.92 | 473.96 | 0 / 0 |
| After fix | Color Adjust enabled | 960x540 | 36.39 | 44.76 | 26.30 | 0 / 0 |
| After fix, lower tier | Color Adjust enabled | 640x360 | 15.19 | 16.87 | 59.88 | 0 / 0 |

The before result is a pass-through measurement, not a valid effect-performance baseline. The fix restores behavior; it is not a speedup. At 640x360, throughput is approximately 60 new frames per second on this narrow workload, with p95 render time slightly above 16.67 ms. There is little frame-time margin. Automatic tier switching was disabled for measurement; actual GUI presentation and responsiveness were not exercised. The 960x540 CPU configuration remains below target.

The reused harness measures render duration and submit/wait wall throughput. UI latency, RAM/VRAM, allocation counts, copies/transfers, media decoding, devices, GPU execution, and packaged operation were not measured. No claims are made for them.

```powershell
# The before command was run before production edits; rerunning it now measures the fixed code.
../lean-test-env/Scripts/python.exe tools/benchmark/run_color_adjust_preview.py --label before-lean-002 --output ../../outputs/lean-002-preview-before.json
../lean-test-env/Scripts/python.exe tools/benchmark/run_color_adjust_preview.py --label after-lean-002 --output ../../outputs/lean-002-preview-after.json
../lean-test-env/Scripts/python.exe tools/benchmark/run_color_adjust_preview.py --label after-lean-002-360p --tier 640x360 --output ../../outputs/lean-002-preview-360p.json
```

## Changes and rollback

- `megaglitch/preview_executor.py`: enable the existing Color Adjust node; preserve all other coverage policy.
- `effects_registry.py`: preserve numeric zero saturation instead of substituting the default.
- `glitch_effects.py`: apply zero saturation rather than taking the unchanged-image shortcut.
- `tests/test_color_adjust_workflow.py`: ten workflow cases using actual processing and shared surface pixels.
- `tools/benchmark/run_color_adjust_preview.py`: reproducible bounded measurement using the existing preview harness.
- This document: requirement trace, V: provenance, results, and limitations.

The two saturation corrections match the V: reference implementation. Existing presets explicitly saved with saturation zero now desaturate as their control value indicates; defaults and disabled nodes retain their previous behavior. The shared HSV helper also corrects other callers requesting zero saturation. This is an intentional observable bug fix.

The standalone LEAN-002 patch is relative to the staged LEAN-001 tree `8126e1b8245e3e0e1c2e959253cf0276ecbbcc6e`. Roll back only LEAN-002 if necessary, preserving LEAN-001 and unrelated work. No full-repository reset is needed.

## Next pull

Profile the enabled Color Adjust path at 960x540 before selecting an optimization: its measured 36.39 ms mean establishes a concrete bottleneck, while the approved smaller tier already approaches the target. Keep output parity checks and the 640x360 result as constraints. Audio/modulation work should be reconciled against the recovered V: commits before implementation there; the original successful headless media remains an unresolved reference gap.
