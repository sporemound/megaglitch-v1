# MegaGlitchNode 0.0.9.4 — Release Scope

## Purpose

MegaGlitchNode 0.0.9.4 is now in **release-convergence mode**. The goal is to finish a dependable, reproducible workstation build from the existing architecture rather than expand the feature surface.

Every change must close a documented release gate, fix a verified defect, prevent a regression, or improve measured performance without weakening correctness.

## Required for 0.0.9.4

- Reliable Windows 11 x64 launch and clean exit.
- Video/image source loading.
- Graph construction, validation, save, load, and reload-equivalent topology.
- Core image-processing nodes.
- Core temporal nodes with master-resolution state.
- Preview that is downstream-only and cannot contaminate processing history.
- Deterministic Offline Master execution.
- Explicit FPS semantics, including `true_accelerated` / source-frames-at-target-rate behavior.
- Direct FFmpeg master output.
- Clear provider/node failure reporting.
- Safe fallback when an accelerated provider is unavailable and a supported fallback exists.
- A reproducible regression suite sufficient to certify the release.
- Reproducible release packaging.

## Desirable but not release-blocking

- CUDA acceleration where already proven stable.
- Reduced provider transfers and improved residency.
- Responsive live preview and MIDI/audio control.
- Performance telemetry that does not alter render behavior.

## Out of scope until 0.0.9.4 is frozen

Do not add or expand:

- new node families;
- speculative UI/workspace features;
- new dual-deck behavior beyond existing compatibility requirements;
- cognitive/AI mutation or AI creative-control features;
- new sequencer systems beyond existing supported behavior;
- new retro/display feature families;
- cloud rendering, accounts, billing, or distributed execution;
- plugin marketplaces or new plugin architectures;
- broad dependency/framework migrations;
- repository-wide refactors or reorganizations;
- new native/runtime abstractions that are not required by a failing release gate.

Existing experimental code is not automatically release scope merely because it is present at HEAD.

## Change rule

Before modifying production code, identify the release gate or verified defect the change addresses. If neither can be named, do not make the change.

Architecture changes require a written exception containing: problem, current architecture, why the gate cannot be met within it, minimal proposed change, affected modules, risk, tests, and rollback procedure.
