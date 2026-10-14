# Rust UI functional-preview shell

## Decision and scope

The 00.9.4 UX direction uses a bespoke eframe/egui/wgpu workstation. It remains an opt-in
frontend under `native/ui`; it does not replace `python main.py`. The Python machine is still the
behavioral specification, full-featured application, and immediate rollback.

Engine V2 is the first honest visual vertical slice. It accepts transactional graph/source/
transport commands, compiles general primary-source graphs through the shared UI-independent
Python graph contract, executes the existing Python frame path, and transfers real RGB program
pixels into egui through a fixed, leased memory-mapped preview surface. It does not claim Source B,
two-deck, modulation, audio/MIDI, recording, virtual-camera, or whole-machine parity.

The complete ownership, workspace, compatibility, and release gates are defined in
[mycotroph-functional-parity.md](mycotroph-functional-parity.md).

## Functional Gate 1

The frontend currently provides:

- an Engine V2 Python child owned and shut down by Rust;
- optimistic-concurrency commands with command IDs, authoritative state revisions, structured
  acknowledgements/errors, and effective frame sequences;
- one mutation in flight, with graph edits coalesced behind the accepted revision;
- test-pattern, still-image, and video sources, including run/stop and video seek-to-start;
- drag-and-drop or explicit-path media selection with authoritative source identity;
- an accepted Deck-A-shaped primary-source graph, including branches, generators, selected
  outputs, and supported multi-input/video-mixer/Lua socket semantics;
- existing `execute_flat_node_pipeline` behavior and temporal frame state;
- a headerless, fixed three-slot RGB8 mmap surface, immediate Rust-owned frame copies, explicit
  slot releases, and preview-only drop behavior;
- real aspect-fit program pixels in the output pane;
- measured graph/preview rates, processing time and drop counts;
- captured Python stderr and visible offline/rejected/runtime errors;
- bounded shutdown with Windows consumer-unmap-before-producer-unlink ordering;
- a generated NodeSpec catalog and editable typed parameter controls;
- graph creation/deletion/bypass/wiring plus in-memory legacy preset-v1 import/export.

The interface no longer substitutes a procedural preview, simulated audio/MIDI/GPU state, fake
per-node cost, or animated node scopes. Unsupported workspaces and controls are visibly reserved.
Process liveness alone is not labelled as an accepted graph.

## Protocol and preview ownership

Control is JSON Lines protocol version 2. Mutating commands contain:

```text
version = 2
id
expected_state_revision
type
command payload
```

Acknowledgements return `accepted`, the authoritative `state_revision`, `graph_generation`,
`effective_frame_sequence`, and an optional top-level structured error with `code`, `path`,
`message`, and `recoverable`.

Preview pixels do not travel through JSON. Python preallocates three 1280x720 RGB8 slots in one
temporary mmap file and announces the surface generation. It leases a free slot, writes a packed
or strided frame, and emits only its descriptor. Rust validates the generation, slot, dimensions,
stride and byte range, copies rows into an `Arc<[u8]>`, then releases the lease. If every slot is
leased, Python drops that preview publication without stalling processing.

Shutdown reverses ownership deterministically: Rust stops accepting work, drops the consumer
mapping, asks Python to shut down, waits up to the bounded deadline, and performs best-effort path
cleanup after the child exits. A ten-cycle Windows stress test verified exit code zero, no stderr,
twenty slot releases, and no remaining mmap files.

## Graph boundary

The protocol version is 2. The graph compiler is now shared behind a Dear-PyGui-independent Python
contract and accepts general primary-source graphs: stable node identity, branches, generators,
selected outputs, and supported multi-input/video-mixer/Lua socket semantics are preserved. It
still rejects cycles, invalid or ambiguous sockets, unreachable graph elements, and `SOURCE_B`
without replacing the last accepted pipeline.

That Source-B rejection is an explicit ownership boundary, not an implicit simplification. Deck B,
the master transition/crossfader, and lossless dual-deck session persistence remain subsequent
gates; effect algorithms continue to live in the established Python engine rather than Rust UI
callbacks.

## Verification

Current focused gates:

```powershell
$Python = 'C:\path\to\dependency-capable\python.exe'
& $Python -m pytest -q tests\test_rust_ui_bridge.py tests\test_session_graph_compiler.py
$env:MEGAGLITCH_PYTHON = $Python
$env:MEGAGLITCH_BRIDGE_INTEGRATION_TEST='1'
cargo test --manifest-path native\ui\Cargo.toml
```

At the current checkpoint, the combined Python bridge/compiler tests pass 28/28 and Rust tests pass
14/14, including a live child/mmap integration test. The ten-cycle lifecycle harness also passes.
These prove the current control/frame and primary-source graph boundaries; they do not establish
whole-machine parity or a performance win.

Do not migrate processing algorithms into egui callbacks. The UI presents and controls engine
state; capture, processing, media, automation and output clocks remain outside the repaint loop.
