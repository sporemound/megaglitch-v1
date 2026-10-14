# Verification

## Source import and portable launchers

Checks ran on Windows x64 with Python 3.12.11 against this checkout's `app/` engine, with the CPU fallback and explicit bundled dependency paths. Application module identities were checked to ensure the tests used the imported source.

- Desktop editor construction, both decks, three rendered UI frames and a real shared-memory worker passed. Worker frame/reset round trips and normal shutdown passed; no forced termination or leaked shared slots were reported.
- **Four launcher/export regression tests passed**, covering source FFmpeg discovery, portable FFmpeg selection, the full-size NVIDIA encoder probe and CPU encoder fallback.
- **68 focused tests passed**, with zero failures, errors or skips, covering session graph compilation/persistence, production graph contracts, render contracts, master-frame behavior, temporal buffer dimensions, offline controls/audio/master output and worker lifecycle.
- A bounded export passed using the unchanged 720 × 1280 processing canvas, 2160 × 3840 Lanczos output, H.264 NVENC at 24 fps and stereo 48 kHz AAC. FFprobe confirmed six frames and 0.250 seconds. The contact sheet showed changing full-frame effects without protected lettering or borders.
- An earlier 24-frame check reached 21 frames before Windows memory commitment prevented another allocation. No orphan test/render processes remained. This is recorded as a host resource limit; the bounded pass does not certify full-duration performance.

The focused suite can be reproduced from the checkout root after installing its CPU test dependencies:

```powershell
$env:PYTHONPATH = (Resolve-Path .\app).Path
python -m pytest -q app/tests/test_session_graph_compiler.py app/tests/test_production_graph_contract.py app/tests/test_render_contract.py app/tests/test_master_graph.py app/tests/test_temporal_no_resize.py app/tests/test_offline_controls.py app/tests/test_offline_audio.py app/tests/test_offline_master.py app/tests/test_worker_lifecycle.py
```

## Preset preservation

The headless adapter's effect implementation and finishing stages match the tested working preset. Its default control timeline was checked against the original: 47 gesture records, 14 history flushes and sampled interpolation values matched. Encoder auto-selection probes the requested output dimensions, so a valid 4K NVENC path is not rejected by an unsupported tiny probe.

The completed local reference video was previously decoded and checked for 60 seconds, 1,440 frames, 2160 × 3840 output at 24 fps and stereo audio. The earlier working Windows package passed dependency checks, desktop/worker startup and a two-second 4K export. Those artifacts used their recorded runtime profiles and are separate evidence from this checkout's bounded source test.

## Declared-input packaging

The CPU portable folder was assembled from the declared Python 3.12.11 input, pinned dependency inventories, standalone NumPy 2.2.6, noise 1.2.2 and CPU OpenCV 4.12.0.88 wheels, and the supplied static FFmpeg archive. Native dependency closure and notices for 17 native/runtime owners passed. The standard-library copy excludes unrelated top-level CUDA DLLs; package-local native variants are inspected independently.

The bundled-runtime dependency check passed with a search path limited to the package and Windows. A real two-second export passed: **48 frames, 2160 × 3840, 24 fps, H.264 NVENC and stereo 48 kHz AAC**, using the unchanged 720 × 1280 effect canvas. FFprobe checked the complete media contract. These checks used only the newly assembled runtime and tools. Verification callbacks resumed after correcting Windows environment-key casing; assembly was preserved.

The builder's syntax and input validation passed. Normal, mixed-case and absent Windows environment-key cases were checked against the real Windows-directory API. The builder requires its dependency/render checks, ZIP CRC and complete archive inventory to succeed before reporting a completed download. Output and input hashes are retained in the package and checksum sidecar.

No clean-machine, optional native/Rust, hardware integration or complete upstream release-gate certification is claimed here. The imported upstream scorecard and known limits remain in `app/`.
