# Rust graph IR v1

## Boundary and compatibility

The explicitly selected Rust application worker now compiles supported MegaGlitch pipelines to
an immutable graph program. Mode-per-effect selection is no longer used by the application
worker. Legacy `passthrough`, `temporal_mix`, and `invert` facade modes remain ABI-compatible,
but each is translated immediately to the same graph interpreter.

The serialized FFI operation is a fixed-layout record:

```text
opcode:u32 flags:u32 arg0:f32 arg1:f32 arg2:u32 arg3:u32
```

The graph carries an explicit version and is copied into Rust-owned memory during creation or
replacement. IR v1 requires 2-64 operations, exactly one primary Input at index zero, exactly one
Output at the end, finite numeric parameters, mix in `[0,1]`, and temporal delays within the
configured fixed history capacity. Validation failure never starts or mutates an engine.

## Operations

| Opcode | v1 semantics |
| --- | --- |
| Input | Select primary RGB8 input; `arg2=0`. The field reserves future source indices. |
| Invert | Replace each channel with `255-channel`. |
| Brightness/Contrast | `arg0=brightness`, `arg1=contrast`; matches the Python float/clip/uint8 contract. |
| Temporal Mix | Fixed-point blend with history; `arg0=mix`, `arg2=delay`, flag bit 0 freezes the ring. |
| Blend | v1 history blend with the same bounded parameters; reserved operands allow later graph/deck blending. |
| Output | Publish the current RGB8 frame. |

Temporal Mix and Blend share the runtime's persistent fixed history ring. A graph replacement at
the same dimensions/capacity preserves history; RESET clears it. If any temporal operation is
frozen, the shared ring does not advance for that frame.

## Immutable updates

Python represents programs as frozen `GraphSnapshot` and `GraphOperation` dataclasses. A changed
pipeline produces a new snapshot. `mg_engine_set_graph` validates and copies the program, then
queues an `Arc<GraphProgram>` command. Commands received while the Rust worker waits for input
are applied before the next submitted frame. No operation list or parameter buffer is mutated in
place across threads.

## Compiler and fallback

IR v1 compilation currently accepts primary Source/Display routing and ordered combinations of:

- `Invert Colors`;
- `Brightness / Contrast`;
- prototype `Native Temporal Mix`;
- prototype `Native History Blend`.

Disabled operations compile away. Malformed parameters, secondary sources, Deck B, Genesis debug
requests, active global temporal macros, unknown nodes, or unsupported formats return the current
item to the unchanged Python worker. Project and preset formats are not changed.

## Dual-deck path

The record already reserves source indices and additional integer/float operands, but v1 rejects
non-primary Input. Dual-deck scheduling is a later versioned gate: add secondary frame submission,
multiple value slots, explicit Blend operands, Deck A/B temporal ownership, and master-transition
output. Until all of those semantics and backpressure rules are tested, Deck B remains a single
coarse Python fallback boundary.
