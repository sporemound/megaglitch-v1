# Rust UI Engine V2 bridge gate

## Scope

The opt-in Rust/egui frontend owns an Engine V2 Python child over JSON Lines protocol version 2.
Low-frequency mutations are optimistic transactions: each command carries an ID and expected state
revision, and acceptance reports the authoritative revision, graph generation, and effective frame
sequence. Preview pixels do not travel through JSON. Python owns a fixed three-slot RGB8 mmap
surface, Rust or the benchmark consumer copies a leased slot, and the consumer releases that exact
generation/slot/sequence without blocking graph processing.

The graph boundary is no longer the old linear-only prototype. The shared UI-independent compiler
accepts general primary-source preset-v1 graphs, including branches, generators and selected
outputs, and the supported mixer, Genesis multi-input, and Lua socket contracts. It rejects cycles,
ambiguous or invalid sockets, unreachable nodes, and Source B while preserving the last accepted
graph. Engine V2 currently owns deterministic test-pattern, image, and video sources plus run/stop
and video seeking. Deck B/master mix, webcam/window capture, audio/MIDI, automation, recording,
virtual camera, and file-backed whole sessions remain outside this gate and stay available through
`python main.py`.

## Reproduction

Run from the repository root with the dependency-capable MegaGlitch CPython 3.12 interpreter:

```powershell
python -m pytest -q --basetemp build\pytest-rust-ui-bridge tests\test_rust_ui_bridge.py
python tools\benchmark\run_rust_ui_bridge.py `
  --width 320 --height 180 --warmup 5 --iterations 30 `
  --output-dir tools\benchmark\results
cargo test --manifest-path native/ui/Cargo.toml
cargo clippy --manifest-path native/ui/Cargo.toml --all-targets -- -D warnings
cargo build --release --locked --manifest-path native/ui/Cargo.toml
powershell -ExecutionPolicy Bypass -File tools\smoke_rust_ui_bridge.ps1 -Python <python.exe>
```

The benchmark is a protocol-v2 consumer, not a direct call disguised as IPC coverage. It requires:

- a `ready` event advertising transactional graph/shutdown commands and RGB8 mmap capabilities;
- a matching `preview_surface` descriptor and existing fixed-capacity backing file;
- accepted `replace_graph` at the expected revision;
- exact SHA-256 parity for both authoritative snapshots and bytes copied from every measured mmap
  frame after Invert becomes effective;
- a release command for every consumed preview lease;
- zero processing/preview drops during the recorded sample;
- quiesced transport, consumer unmap, accepted revisioned shutdown, exit code zero, JSON-only
  stdout, empty stderr, and removal of the producer-owned surface.

The focused tests also fix six consecutive `Feedback / Trail` output hashes at 32x18, retaining
temporal state across frames. A real media test writes a short temporary video with the first
working OpenCV codec/container from MJPG, XVID, or mp4v, opens it through `VideoFrameSource`, reads
successive frames, rewinds, seeks to 0.5 seconds, and verifies the decoded frame moves to the target
region. It skips only when none of those writers can initialize and produce a decodable container.

## Recorded result

Configuration: Windows x64, CPython 3.12.11, 320x180 RGB8 deterministic test pattern, Invert
Colors at 100%, five warm-up frames, and 30 measured frames. Interpreter metadata in the raw JSON
is deliberately portable and contains no developer-machine path.

| Path | Measured scope | Mean | p99 | FPS from mean |
| --- | --- | ---: | ---: | ---: |
| Direct | Test-pattern generation plus `execute_flat_node_pipeline` | 0.615 ms | 1.055 ms | 1,624.77 |
| Engine V2 sidecar | Bridge-reported source read plus graph processing | 0.707 ms | 0.982 ms | 1,414.05 |

All 35 authoritative snapshot hashes and all 35 mapped frames (five warm-up plus 30 measured)
matched their exact deterministic Invert output. The consumer issued 36 slot releases, including
one startup frame. Mean/p99 mmap read-and-copy time was 0.023/0.052 ms. No processing or
preview frames were dropped.

Readiness was 330.441 ms, graph replacement acknowledgement was 2.082 ms, transport-stop
acknowledgement was 0.820 ms, and shutdown acknowledgement was 2.299 ms. Shutdown advanced the
state revision to 3, exited with code zero, emitted no stderr or trailing events, and removed the
8,294,400-byte preview surface. The focused Python bridge suite passed 12/12, including the fixed
Feedback/Trail sequence, real video read/seek, and a small subprocess form of this complete v2
gate.

The performance gate permits the sidecar's reported frame work up to the larger of twice direct
or direct plus 1 ms; this run passed against 1.615 ms. These rates exclude capture-device latency,
JSON/event delivery, mmap publication, egui texture upload, repaint, and display presentation, so
they are not an end-to-end product FPS claim.

Raw results are in `tools/benchmark/results/rust-ui-bridge.json` and `.csv`.

## Risks and rollback

- Python import/startup failures leave the frontend visibly offline with captured diagnostics;
  production does not substitute animated demo telemetry.
- Invalid graph replacements are rejected without replacing the last accepted pipeline.
- Preview backpressure drops preview publication only; the graph clock does not wait for egui.
- The Rust shell remains opt-in. `python main.py` is the full-machine rollback without project or
  preset conversion.
