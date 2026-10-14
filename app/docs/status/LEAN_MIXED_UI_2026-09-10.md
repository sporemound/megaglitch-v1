# LEAN-005: mixed graph through the running Rust UI

## Value and scope

R1/R3/R4: exercise an existing graph and parameter controls through the real Rust UI, video decoding, Python bridge, shared RGB surface, upload, and WGPU render pass. This advances validation beyond the executor-only measurements in LEAN-004.

Scope remains the isolated `MegaGlitch-Node0.9.4` checkout on `codex/lean-preview-coverage`, based on `ff8b53c9376c271c1b7ce8ca95f38065f3073603`. Baseline index tree after LEAN-004: `3045621081384d1a6c6c04703e8cc18a5b718dfe`. V: remains an unchanged reference. No production source changes were needed for these runs.

## Acceptance and result

- [x] Build the scoped Rust UI release from its locked dependencies.
- [x] Verify a changing 1080p60 video input and load the mixed preset.
- [x] Observe 300 new WGPU-frame markers in each graph order.
- [x] Observe a parameter revision reach a newly drawn frame and restore the original value.
- [x] Record rendering separately from presentation, resolution, and latency.
- [x] Verify normal UI process exit and collect source/executable provenance.
- [ ] Establish 60 newly presented frames per second: **not met in these runs**.

Functional smoke checks passed in both orders, each with exit code 0 and no reported engine error. The UI's existing `passed: true` means the target frame count and parameter probe completed; it is not a 60 FPS performance gate. This validation pull is complete; performance acceptance remains partial.

## Build and input

The release build completed offline in 2 minutes 25 seconds, using the existing Cargo lock. Three existing dead-code warnings remain (`deck_b_selected`, `CanonicalSession::as_value`, and unused PreviewFrame fields). No dependency migration or Rust source edit occurred.

Executable SHA-256: `66d7868f71ec9f537191c5fee6aaff4ab276a39261ead32eefce92e6e3e4d213`.

The FFmpeg-generated `testsrc2` clip is H.264, 1920x1080, 60/1 FPS, 15 seconds, 900 frames, verified with ffprobe. This contains moving image content rather than only a changing sequence marker. Input SHA-256: `73be44644961c44a7d3d2c9af9bcae453b535c2b45b6fd594b6b7c0055397a45`. It is a synthetic functional/performance source, not the missing artistic headless reference.

Both presets use Color Adjust (hue 30, saturation 0.5, gamma 1.2) and Feedback / Trail (feedback 0.25, trail 0.5). The existing UI smoke temporarily perturbs and then restores a floating-point control during measurement. Graphs run in both orders, preserving their different visual semantics.

## Live results

| Measurement | Color → Feedback | Feedback → Color |
|---|---:|---:|
| New WGPU-frame markers | 300 | 300 |
| Measurement duration | 5.29 s | 5.22 s |
| New-frame presentation rate | 56.50 FPS | 57.31 FPS |
| Final rolling preview render rate | 59.88 FPS | 60.32 FPS |
| Final preview size after adaptation | 720x480 | 720x480 |
| Final rolling p95 preview render time | 15.19 ms | 15.24 ms |
| Preview dropped / superseded | 0 / 1 | 0 / 2 |
| Mean source-submit-to-WGPU-marker latency | 36.89 ms | 38.20 ms |
| Parameter-to-new-marker latency | 62.49 ms | 62.59 ms |
| Final rolling authoritative rate | 18.10 FPS | 15.83 FPS |

Color-first probed saturation 0.5 → 0.503; feedback-first probed feedback intensity 0.25 → 0.251. Both probes resolved and graph revisions advanced to 3 after restoration. Final lifecycle is reported RUNNING because the report is written before requesting clean shutdown; the launcher independently observed exit code 0 afterward. Shared temporary surfaces were cleaned up. A process check found no matching remaining process under the scoped workspace; a broader WMI query was unavailable due to local access restrictions, so exhaustive system-wide orphan certification is not claimed.

The rendering rate is a rolling sample; presentation rate spans the 300-marker window. Their difference indicates a presentation-rate shortfall but does not by itself identify its cause or an exact number of lost frames. Last received/uploaded/drawn sequence values are asynchronous samples. The raw report's `authoritative_dropped_frames` field comes from the existing runtime dropped-frame field; it must not be reinterpreted as an independently measured offline-export frame-loss count.

These results do not establish fixed 960x540 UI performance: adaptation changed resolution to 720x480. They also do not establish physical scanout completion; `swapchain_completion_observed` is explicitly false. The marker is encoded in the normal eframe/WGPU render pass. Video decoding and the simultaneous authoritative worker are included, unlike the isolated executor benchmarks.

## Reproduction and artifacts

From `work/MegaGlitch-Node0.9.4`:

```powershell
cargo build --release --locked --offline --manifest-path native/ui/Cargo.toml
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=1920x1080:rate=60 -t 15 -c:v libx264 -threads 2 -preset ultrafast -crf 20 -pix_fmt yuv420p ../lean-005-motion-1080p60.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration -of json ../lean-005-motion-1080p60.mp4
../lean-test-env/Scripts/python.exe tools/benchmark/run_mixed_ui_smoke.py --source ../lean-005-motion-1080p60.mp4 --python ../lean-test-env/Scripts/python.exe --report ../../outputs/lean-005-ui-color-feedback.json
../lean-test-env/Scripts/python.exe tools/benchmark/run_mixed_ui_smoke.py --source ../lean-005-motion-1080p60.mp4 --python ../lean-test-env/Scripts/python.exe --temporal-first --report ../../outputs/lean-005-ui-feedback-color.json
git diff --check
```

Use fresh report filenames when repeating; the launcher rejects an existing report. Source generation likewise does not overwrite an existing clip without FFmpeg confirmation. The launcher records executable/video/preset hashes, selected Python and repository paths, four processing-source hashes, exit code, timeout status, and elapsed time. It saves the presets and stdout/stderr beside each raw report. Both stderr logs were empty. Temporary surfaces are directed to this checkout's `.tmp`, not V:.

Changes are limited to `tools/benchmark/run_mixed_ui_smoke.py` and this evidence record. No new unit tests were required for unchanged production code; the two real UI runs are this pull's acceptance checks. Earlier test results remain attributed to their respective pulls.

## Limits and next pull

No installable release package, OBS/audio device test, subjective visual acceptance, GPU scanout measurement, RAM/VRAM/allocation/transfer profile, or offline export certification was performed. The development UI uses the explicitly selected Python environment and current scoped bridge. No commits, pushes, merges, or V: modifications occurred.

Next pull: measure frame timing between bridge receipt, upload, and WGPU drawing to locate the roughly 3–4 FPS presentation shortfall. Separate UI pacing, frame replacement, and concurrent processing pressure before making a production change. Preserve node order and parameter semantics.

The LEAN-005 patch is relative to the baseline index tree above; rolling it back only removes the new launcher/evidence record and preserves LEAN-001–004.
