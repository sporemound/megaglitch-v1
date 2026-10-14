# LEAN-006: locate the live presentation-rate gap

R3/R4: the prior real-UI run rendered about 60 preview frames/s but drew only 56.50–57.31 new WGPU markers/s. Identify timing between receipt, texture preparation, and drawing before changing pacing or graph semantics.

Baseline index tree after LEAN-005: `0e5ae7c6dcb8f0c7981fdf072e522f94eac0a6da`. Scope remains the same recovery-based checkout; earlier LEAN work is retained and V: remains read-only reference.

Bounded instrumentation: timestamp a frame after its shared-memory bytes are copied into Rust ownership; measure waiting until CPU texture preparation, texture preparation duration, and time from preparation to its matching WGPU marker. Track prepared frames replaced before their matching marker. Use monotonic clocks and frame identity; do not count redraws or GPU completion as new observations. All summaries have constant storage and accompany existing smoke reports.

- [x] Timing aggregation passes deterministic matching/duplicate/reset checks.
- [x] Build and run the instrumented real UI on the same moving video and both mixed graph orders.
- [x] Locate the dominant measured interval without conflating asynchronous frame counts.
- [x] Assess whether evidence establishes a bounded fix: it does not yet isolate a cause; retain instrumentation without changing pacing.
- [x] Record performance acceptance and limitations candidly.

## Experiment and evidence

The release UI uses the same 15-second 1920x1080, 60 FPS moving `testsrc2` video as LEAN-005. Each run draws 300 unique preview markers, changes and restores an existing graph parameter, then exits normally. Both previews adapt to 720x480. No engine error or timeout occurred.

| Measurement | Color Adjust → Feedback/Trail | Feedback/Trail → Color Adjust |
|---|---:|---:|
| Unique WGPU marker FPS, full window | 57.24 | 55.05 |
| Preview renderer FPS, rolling | 59.99 | 59.99 |
| Matched timing samples | 299 | 299 |
| Copied frame → CPU texture preparation, mean ms | 6.95 | 6.51 |
| CPU texture preparation/enqueue, mean ms | 0.389 | 0.388 |
| Preparation → WGPU marker, mean ms | 12.24 | 12.19 |
| Prepared frames replaced before drawing | 0 | 0 |
| Source submission → marker, mean ms | 35.73 | 35.54 |
| Parameter change → marker, ms | 62.60 | 55.50 |

Evidence in the workspace `outputs` directory:

- `lean-006-ui-timing-color-feedback.json` and `lean-006-ui-timing-feedback-color.json`: raw measurements, means and maxima, asynchronous sequences, functional results.
- Corresponding `.provenance.json`, `.preset.json`, `.stdout.log` and `.stderr.log`: launcher inputs, executable/source hashes and exit evidence.
- Release executable SHA256: `fb6fa113d3c9446be5bad7f6a47c34a7cf69312464597c01c7aa59604dca3c2d`.
- Source video SHA256: `73be44644961c44a7d3d2c9af9bcae453b535c2b45b6fd594b6b7c0055397a45`.

Build: `cargo build --release --locked --offline --manifest-path native/ui/Cargo.toml` succeeded with three existing dead-code warnings. Validation: `cargo test --release --locked --offline --manifest-path native/ui/Cargo.toml presentation` passed all five selected tests (40 filtered out), including the new deterministic timing test. Two existing test-build warnings remain. Final source formatting and a timestamp comment were applied after these runs; executable behavior is unchanged.

## Learning and next pull

CPU image conversion and texture enqueue occupy about 0.39 ms; the larger measured interval is between enqueue and the matching WGPU render-pass callback. That interval includes downstream UI/render work and possible surface acquisition waiting. It does not isolate GPU upload, event-loop scheduling, or display synchronization. Local `egui-wgpu-0.35.0/src/winit.rs` acquires the current surface texture in `paint_and_update_textures`; `src/lib.rs` defines AutoVsync surface defaults. This makes surface pacing a concrete investigation target, not a proven defect.

R3 remains open: neither run demonstrates 60 newly drawn frames/s. The prior 56.50–57.31 FPS measurements and these 55.05–57.24 FPS runs are short observations, not a statistically controlled regression comparison. Next pull: compare one controlled surface-latency setting with the current default on repeated runs of the same graph, retaining a change only if it improves unique marker throughput without unacceptable latency or functional loss.

The first marker after the smoke reset has no matching preparation timestamp, hence 299 samples for 300 markers. Zero prepared-frame replacements does not count frames replaced in the bridge's latest-frame state before texture preparation. Renderer FPS and marker FPS use different windows; raw final sequence differences cannot be interpreted as exact losses. `passed: true` means functional smoke success, not R3 acceptance. WGPU callbacks observe render-pass encoding, not completed GPU execution or physical display scanout. The legacy `authoritative_dropped_frames` field is not evidence of offline output loss. Original artistic media and full application parity remain unverified.

Only `native/ui/src/app.rs`, `native/ui/src/state.rs`, and this record belong to LEAN-006. Work stays staged locally; no commit, push, merge or packaged release. V: remains a read-only reference.
