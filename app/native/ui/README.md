# MegaGlitch Rust functional preview

This opt-in eframe/egui/wgpu frontend is an incremental control surface for the existing Python
MegaGlitch engine. It does not replace `python main.py`; the Python/Dear PyGui machine remains the
full-featured application and rollback.

Engine V2 currently has a real visual path:

- transactional graph, source, transport and seek commands;
- structured acceptance/rejection and authoritative state revisions;
- test-pattern, image and video sources;
- general primary-source graphs, including branches, generators, selected outputs, and supported
  multi-input/video-mixer/Lua socket semantics, compiled through the shared Python contract;
- fixed leased mmap preview slots and real RGB output rendered in egui;
- measured frame/preview telemetry and explicit lifecycle errors;
- generated NodeSpec catalog, named video pins and typed inspector controls;
- editable graph state and in-memory legacy preset-v1 import/export.

Source B, Deck B/master transitions, file-backed sessions, modulation/timeline, audio/MIDI,
recording/export, virtual camera, and broader whole-machine parity are not connected at this gate.
Their UI is disabled or labelled reserved; no simulated activity stands in for them.

Build:

```powershell
powershell -ExecutionPolicy Bypass -File native\ui\build.ps1 -Profile release
```

Run from the repository with a dependency-capable MegaGlitch Python:

```powershell
$Python = 'C:\path\to\dependency-capable\python.exe'
powershell -ExecutionPolicy Bypass -File native\ui\build.ps1 -Run `
  -Python $Python
```

Verify:

```powershell
& $Python -m pytest -q tests\test_rust_ui_bridge.py tests\test_session_graph_compiler.py
$env:MEGAGLITCH_PYTHON = $Python
$env:MEGAGLITCH_BRIDGE_INTEGRATION_TEST='1'
cargo test --manifest-path native\ui\Cargo.toml
```

Regenerate NodeSpec after Python node/socket/parameter changes:

```powershell
python tools\export_rust_node_catalog.py
```

See [the functional parity contract](../../docs/architecture/mycotroph-functional-parity.md) for
the Deck A/B, source, automation, audio/MIDI, output, compatibility and release gates.
