# LEAN-007: surface queue-depth comparison

## Value and scope

R3 requires approximately 60 genuinely new preview frames/s with disclosed approximations. LEAN-006 measured a roughly 12 ms interval between texture preparation and WGPU drawing. This pull tests whether reducing the surface queue depth helps. Baseline staged tree: `7284ba68c3ba1182ed27589d2f529dcca574b7d6`. Recovery checkout and earlier staged work are preserved; V: was not modified.

## Checklist

- [x] Inspect the installed renderer's actual default before selecting the experiment.
- [x] Change only desired surface queue depth; preserve AutoVsync.
- [x] Build the candidate and run three trials per setting in alternating order.
- [x] Compare unique-frame throughput, latency, functional behavior and preview geometry.
- [x] Reject unsupported production changes and retain reproducible evidence.
- [ ] R3 acceptance: approximately 60 new preview frames/s remains unverified.

## Method

Local `egui-wgpu-0.35.0/src/lib.rs` sets `WgpuConfiguration::default()` to `SurfaceConfig::HIGH_THROUGHPUT`, with `AutoVsync` and `desired_maximum_frame_latency: Some(2)`. The candidate changes only `native/ui/src/main.rs` to set that latency to `Some(1)`. This is a requested queue bound, not a measurement of the actual queue depth.

Build command: `cargo build --release --locked --offline --manifest-path native/ui/Cargo.toml`. It succeeded in 1m05s with three existing dead-code warnings. No dependency changes. The baseline executable comes from LEAN-006; its source differs from the staged baseline only in formatting/comment changes documented there. Candidate executable SHA256: `688e80841f2217d792047bd05f28dc35f800c99cbd04afff17c0469d8a002daf`. Baseline SHA256: `fb6fa113d3c9446be5bad7f6a47c34a7cf69312464597c01c7aa59604dca3c2d`.

Added `tools/benchmark/compare_ui_surface_pacing.py`, which launches the existing smoke harness sequentially in order baseline/candidate/candidate/baseline/baseline/candidate and records raw reports, provenance, logs and medians. It requires a new output directory and stops on a failed functional run. Reproduction from the checkout:

```powershell
../lean-test-env/Scripts/python.exe tools/benchmark/compare_ui_surface_pacing.py --baseline ../lean-007-baseline-ui.exe --candidate ../lean-007-candidate-ui.exe --source ../lean-005-motion-1080p60.mp4 --output ../../outputs/lean-007-surface-comparison
```

Use a different output directory when repeating. All runs use Color Adjust → Feedback/Trail, the same 1920x1080 60 FPS moving test video, 300 unique WGPU markers, and an existing parameter change/restore probe. The source hash and executable hashes are recorded per run. No builds or tests ran concurrently with the comparison.

## Results

| Trial | Queue request | Unique marker FPS | Final preview |
|---|---:|---:|---|
| 1 | 2 | 51.776 | 640x360 |
| 2 | 1 | 51.440 | 720x480 |
| 3 | 1 | 52.130 | 640x360 |
| 4 | 2 | 51.987 | 640x360 |
| 5 | 2 | 53.115 | 640x360 |
| 6 | 1 | 55.611 | 720x480 |

| Median across three runs | Queue 2 baseline | Queue 1 candidate |
|---|---:|---:|
| Unique marker FPS | 51.987 | 52.130 |
| Source submission → marker mean, ms | 42.363 | 41.862 |
| Parameter change → marker, ms | 82.409 | 76.167 |
| Texture preparation → marker mean, ms | 11.968 | 11.929 |

All six functional runs passed, produced 300 unique markers each, and exited with code zero without timeout. Each had 299 matched stage samples, as expected from the initial smoke reset. The new harness itself was exercised by the complete six-run comparison. No new unit test was added for the temporary configuration assignment, which is not retained.

## Decision and next action

Reject the queue-depth change. Median drawing throughput differs by only 0.143 FPS (about 0.27%), within the observed spread; the measured preparation-to-marker interval is effectively unchanged. Adaptive preview geometry differed, so this is an application-level comparison rather than an isolated fixed-workload microbenchmark. It does not establish causation or equivalence. No trial reached 60 new markers/s, and candidate parameter response ranged up to 107.18 ms. The source assignment and default executable were restored; the candidate binary and patch remain explicitly labeled experiment artifacts.

The next bounded investigation is whether frames are replaced in the bridge's latest-frame state before UI texture preparation. Existing zero prepared-frame replacements does not cover that interval. First trace the existing receive/snapshot path; add only the accounting needed to distinguish that loss from repaint pacing. Do not repeat queue-depth tuning without new evidence.

Raw evidence: workspace `outputs/lean-007-surface-comparison/summary.json` and its six reports with provenance/logs. `outputs/lean-007-candidate.patch` is the rejected experiment, not part of the retained development patch. WGPU markers measure render-pass encoding, not GPU completion or physical scanout. These short synthetic trials do not certify original artistic media, fixed-resolution 60 FPS, or full application parity. Performance from September 10 is not a controlled baseline for today's machine conditions.

Retained changes are this record and the comparison harness, staged locally. No commit, push, merge or release.
