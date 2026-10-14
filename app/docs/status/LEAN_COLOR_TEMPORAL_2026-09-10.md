# LEAN-004: Color Adjust with Feedback / Trail

R3/R4 and release Gates D/E/I: verify mixed-graph throughput, faithful node order, and stream-owned history after the Color Adjust optimization. Scope remains the isolated recovery-based checkout; V: is unchanged reference material.

Baseline index tree after LEAN-003: `b0dadc41da5f6a13ce46f168423b987f877e2303`.

This pull validates existing behavior before selecting more optimization. Use the existing Feedback / Trail parameters (feedback 0.25, trail 0.5) and Color Adjust (hue 30, saturation 0.5, gamma 1.2), in both graph orders. Compare 14-frame RGB sequences and final temporal buffers against the retained floating-point color reference, then exercise reduced preview tiers and preview-only parameter revisions against an unchanged full-resolution master stream.

- [x] Mixed output/history is byte-identical to the reference implementation.
- [x] Temporal effect and graph order produce observable output changes.
- [x] Preview resize/control changes preserve master pixels and history, with no shared buffers.
- [x] Fixed 960x540 combined throughput measured with enabled-node coverage disclosed.
- [x] Record results and choose next work from evidence; do not change production code without a defect or measured need.

## Result

Completed locally on `codex/lean-preview-coverage`. This was a validation pull: no production-code changes were necessary. Existing LEAN-001/002/003 changes are retained. No V: changes, commits, pushes, merges, or package builds occurred.

Five new mixed-graph tests passed in 7.41 seconds using the installed custom OpenCV 4.12.0, CPU processing, Python 3.12.14, NumPy 2.5.3, and pytest 9.1.1. The tests verify:

- Both graph orders across 14 changing RGB frames, with identical optimized/reference output and final feedback/trail/history buffers.
- Temporal processing actually changes the color-only sequence; reordering the two effects also changes output. Graph order is an artistic control, not a suggested performance workaround.
- 1280x720 master processing is identical with and without an independent preview, including preview changes from 960x540 to 640x360 and saturation revision to zero.
- Source arrays remain unchanged; master and preview temporal buffers share no memory and retain their respective resolutions.
- Preview history resets for the tier change, but not for a parameter-only graph revision.

The reference comparison uses the retained floating-point HSV/gamma functions, converted to the frame pipeline's RGB8 contract. The previous 159-test LEAN-003 result remains historical evidence; this pull ran only the relevant new checks and measurements because production code did not change.

## Combined measurements

Each asynchronous-executor run used a synthetic 1920x1080 RGB gradient with a changing sequence marker, fixed 960x540 preview, 30 warmup frames, and 300 measured renders. All nodes were enabled and reported implemented. Adaptive resolution was disabled; requests were submitted one at a time without producer overload. Runs were sequential on the same Windows 11 / Intel64 Family 6 Model 183 Stepping 1 host. GPU execution was disabled even though the installed custom OpenCV build detects a CUDA device.

| Graph order | Mean render time | p95 render time | New render FPS | Dropped / superseded |
|---|---:|---:|---:|---|
| Color Adjust → Feedback / Trail | 11.88 ms | 17.26 ms | 73.72 | 0 / 0 |
| Feedback / Trail → Color Adjust | 10.63 ms | 14.77 ms | 81.92 | 0 / 0 |

Both narrow workloads averaged above 60 new renders/s. The first order's p95 exceeds a 16.67 ms frame budget. These are single-run observations and do not establish an inherent speed advantage for either order, sustained no-stall 60 FPS, actual presentation rate, or full application performance. The gradient benchmark changes a sequence marker rather than the entire scene; the correctness tests use changing full RGB frames.

Compared with the isolated Color Adjust result, the mixed graph has less frame-time margin. No additional speedup is claimed here. GUI/scanout latency, video decoding, RAM/VRAM, allocation/copy/transfer counts, actual devices, full-quality export, and packaged operation were not measured. No regression requiring a production fix was found in this bounded validation.

## Exact commands and artifacts

Run from `work/MegaGlitch-Node0.9.4`:

```powershell
../lean-test-env/Scripts/python.exe -c "from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults; bootstrap_for_dev_defaults(); import pytest; raise SystemExit(pytest.main(['-q','tests/test_color_adjust_temporal.py','--tb=short']))"
../lean-test-env/Scripts/python.exe tools/benchmark/run_color_adjust_preview.py --label lean-004-color-then-feedback --feedback --output ../../outputs/lean-004-color-then-feedback.json
../lean-test-env/Scripts/python.exe tools/benchmark/run_color_adjust_preview.py --label lean-004-feedback-then-color --feedback --temporal-first --output ../../outputs/lean-004-feedback-then-color.json
git diff --check
```

Changes: `tests/test_color_adjust_temporal.py` adds the five checks; `tools/benchmark/run_color_adjust_preview.py` adds optional feedback/order selections while retaining the original default workload; this document records evidence. No new runtime abstraction, dependency, or feature family was added.

The standalone LEAN-004 patch is relative to the LEAN-003 tree recorded above. Reversing only that patch removes this validation tooling and record without undoing the earlier production fixes.

## Next pull

Validate this known mixed graph through the running Rust UI/bridge using a changing video source and new-frame presentation measurements. That advances end-to-end acceptance beyond executor timing. Investigate any measured UI/transport or frame-time failure before adding another optimization; preserve user-selected node order.
