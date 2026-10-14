# MegaGlitchNode 0.0.9.4 — Release Gates

0.0.9.4 is complete only when the following gates are backed by current-head evidence.

## Gate A — Build and dependency health

- Release build succeeds.
- Required Python/native dependencies resolve.
- FFmpeg availability is detected.
- CUDA availability is detected without breaking CPU fallback.
- Feature flags and launch scripts select the intended runtime.

## Gate B — Launch lifecycle

- Cold launch succeeds.
- Project open succeeds.
- Clean exit succeeds.
- A second cold launch succeeds without manual cleanup.

## Gate C — Graph persistence

A reference graph must load, validate, execute, save, and reload with equivalent topology and parameters.

## Gate D — Master-buffer integrity

- Master input, output, and core temporal buffers match configured master resolution.
- Preview buffers cannot enter temporal or processing state.
- No hidden repair-resize is accepted.

## Gate E — Temporal correctness

For a fixed source sequence:

- every requested source frame advances temporal state exactly once;
- frame order is stable;
- unexplained drops/duplicates are zero;
- feedback/held-frame state does not reset unexpectedly.

## Gate F — FPS semantics

Verify independently:

1. duration-preserving FPS conversion;
2. `true_accelerated` / source-frames-at-target-rate.

For `true_accelerated`, each source frame is processed once and emitted once on the requested output timebase.

## Gate G — Offline Master

- Runtime → master frame stream → direct FFmpeg output.
- No preview, OBS, virtual-camera, or display-capture dependency.
- Output file is validated for frame count, dimensions, FPS semantics, and duration.

## Gate H — Provider behavior

For each supported accelerated provider used in release scope:

- execution succeeds on supported hardware;
- provider failure is reported clearly;
- temporal state is not corrupted by provider failure;
- supported fallback is used when valid;
- absence of a valid fallback aborts cleanly.

## Gate I — Regression corpus

Maintain representative fixtures for at least:

- passthrough;
- RGB tear;
- feedback;
- optical flow;
- feedback + flow;
- P-frame/temporal simulation;
- multimodal modulation;
- heavy temporal graph;
- true accelerated 55 FPS;
- Offline Master / codec post path.

Validate properties appropriate to each fixture: dimensions, frame count, FPS, duration, completion, output existence, no unexplained black frames/decoder failure, and deterministic hashes where determinism is expected.

## Gate J — Packaging

- Supported Windows release package is reproducible.
- Clean-machine dependencies are documented.
- Packaged application launches and completes a reference render.
- Packaged runtime does not accidentally depend on unrelated Qt/Python installations.

## Evidence rule

A historical passing benchmark is useful evidence but does not automatically mark current HEAD PASS after later production-code changes.

Each gate status must identify the commit/test evidence used to support it.
