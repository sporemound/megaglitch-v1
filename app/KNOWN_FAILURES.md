# MegaGlitchNode 0.0.9.4 — Known Failures and Convergence Risks

This file records verified defects and high-confidence release risks. It is not a feature backlog.

## 1. Feature-expansion drift at recent HEAD

**Status:** verified repository-history risk.

Recent production commits after the render-quality overhaul resumed broad UI/feature expansion, including dual-deck workspace additions, a cognitive mutation interface, sequencer functionality, retro/export controls, drag/drop interaction work, and parity expansion.

This is inconsistent with release-convergence mode and with the existing repository rule that MegaGlitch is not an AI-product redesign.

**Course correction:** no further feature expansion until release gates are closed. Existing mixed commits are not blindly reverted because they also contain bug fixes/parity work; they must be assessed by the regression gates.

## 2. Current HEAD has not been re-certified after recent feature commits

**Status:** verified evidence gap.

The quality-first master-path correctness benchmark predates the latest UI/runtime-facing commits. Historical passing evidence must not be treated as proof that current HEAD still passes every gate.

**Required action:** rerun focused build, launch, graph, master-buffer, temporal, FPS, Offline Master, and packaging checks on current HEAD before marking PASS.

## 3. Rust UI remains a functional preview, not a certified replacement

**Status:** documented repository state.

The README identifies the eframe/egui/wgpu workstation as an opt-in functional preview and the Python engine as the supported fallback.

**Release implication:** do not make Rust UI replacement a prerequisite for 0.0.9.4 unless an explicit release decision changes this boundary.

## 4. Provider residency still has known historical breaks

**Status:** documented architecture risk requiring current measurement.

The frame-pipeline audit identifies paths where CUDA-labelled processing can force CPU residency, including previous-frame conversion and hybrid helpers. ModernGL/Lua nodes also cross host/GPU boundaries per node.

**Required action:** keep these as measurable optimization targets, not reasons for speculative rewrite. Correctness gates precede transfer optimization.

## 5. Temporal storage ownership remains heterogeneous in the supported Python path

**Status:** documented architecture risk.

Historical audit shows temporal state split among `FrameState`, Python history lists, CUDA/global effect state, and backend-specific state.

The quality-first Offline Master path has stronger master-resolution contracts, but live-path ownership still requires current-head verification.

## 6. Quality-first CPU correctness path is not a throughput target

**Status:** verified benchmark limitation.

The recorded 1920×1080 all-CPU quality reference took roughly 8.5 seconds per source frame. It is correctness evidence, not acceptable interactive performance.

**Required action:** preserve its correctness properties while measuring provider-specific acceleration separately.

## 7. Packaged executable in repository is not equivalent to current-head certification

**Status:** release risk.

A packaged workstation executable exists in the repository, but later commits can make it stale relative to source HEAD.

**Required action:** packaging Gate J must record the source commit used to build the release artifact and verify a reference render from that package.

## Reporting rule

Add entries only when supported by code, reproducible tests, benchmark evidence, or repository history. Move resolved items to a dated resolved section rather than deleting the evidence trail.
