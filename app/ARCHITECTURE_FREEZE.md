# MegaGlitchNode 0.0.9.4 — Architecture Freeze

This document freezes the release architecture for 0.0.9.4. It is intentionally conservative.

## Authoritative render path

```text
SOURCE
  ↓
MASTER FRAME BUFFER
  ↓
MULTIMODAL ANALYSIS
  ↓
ACTIVITY / CONTROL MASK
  ↓
TEMPORAL + SPATIAL PROCESSING
  ↓
STRUCTURAL RESTORATION
  ↓
MASTER OUTPUT
  ├────────→ PREVIEW COPY
  │
  └────────→ OUTPUT
                 ↓
          OPTIONAL CLI CODEC GLITCH
                 ↓
            FINAL ENCODE
```

## Frozen invariants

### Master path

- One authoritative master-resolution processing path.
- Master input, graph output, and core temporal/history buffers must remain at the configured master resolution unless an explicit graph node performs a deliberate resize.
- Hidden repair-resizes are errors, not convenience behavior.

### Preview

Preview is a terminal consumer.

Preview/UI/OBS/virtual-camera products must never become inputs to:

- graph processing;
- recursive feedback;
- held-frame history;
- P-frame simulation;
- optical-flow history;
- future master frames.

### Temporal state

Temporal state is explicit and bounded. Core history stays at master resolution.

Do not introduce repeated resize → feedback → resize cycles.

### Offline Master

Offline Master:

- advances the graph once per source frame;
- preserves deterministic temporal ordering;
- performs FPS placement at egress;
- bypasses preview, UI presentation, OBS capture, and virtual-camera capture;
- outputs directly to the encoder path.

### FPS semantics

FPS and timeline behavior are separate concepts.

Supported semantics include:

- duration-preserving conversion;
- `true_accelerated` / source-frames-at-target-rate, in which each source frame is processed once and emitted once on the requested output timebase.

Do not silently reinterpret one as the other.

### Structural restoration

Sharpening/local-contrast restoration belongs after destructive temporal/spatial operations in the quality-first path. Do not move it earlier merely to improve preview appearance.

### Codec-domain glitch

Traditional MPEG-4/P-frame/packet corruption remains a downstream, optional post-process. It must not become part of core graph scheduling or temporal-state ownership.

### Native migration

Native/Rust/CUDA migration remains incremental, benchmark-gated, reversible, and buildable at every stage. The supported Python engine remains a fallback until a native path proves behavior, reliability, compatibility, and performance.

## Architecture-change exception

No architecture change is allowed for release work unless a failing release gate cannot reasonably be satisfied inside this frozen design.

Before such a change, document:

1. failing gate;
2. current behavior;
3. why the frozen architecture cannot satisfy it;
4. minimal proposed change;
5. affected modules;
6. compatibility and migration risk;
7. verification tests;
8. rollback method.
