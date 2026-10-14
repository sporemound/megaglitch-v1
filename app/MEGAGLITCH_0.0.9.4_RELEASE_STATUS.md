# MegaGlitchNode 0.0.9.4 — Release Status

## Audit basis

Repository inspected on current `main` after the recent render-quality overhaul and subsequent UI/parity commits.

Important distinction:

- historical benchmark/documentation evidence exists for several subsystems;
- the latest production-code HEAD has **not** been re-run in this audit environment;
- therefore historical PASS evidence is not promoted to current-head PASS without a fresh run.

Status values: `PASS`, `PARTIAL`, `FAIL`, `UNKNOWN`.

## Scorecard

| Gate | Status | Current evidence / next action |
|---|---|---|
| Build and dependency health | PARTIAL | Release builds and packaging have historical evidence; recent commits include a packaged workstation executable and release-build fixes. Re-run release build and dependency diagnostics on current HEAD. |
| Launch lifecycle | UNKNOWN | Deterministic worker lifecycle is documented historically, but current full-workstation cold-launch/open/exit/relaunch evidence is not recorded here. |
| Graph persistence | PASS | Certified 28-preset canonical corpus (including intentional quality-first `render_quality_master.json`). 28/28 presets pass parsing, canonical validation, wire serialization, idempotency, and dual-deck round-trip. `tests/test_session_document.py` (14/14 tests) PASS on current HEAD. |
| Master-buffer integrity | PARTIAL | Quality-first Offline Master has explicit master-resolution gates and historical passing evidence. Re-run after recent commits and separately verify live path. |
| Temporal correctness | PARTIAL | Historical quality reference reports zero drops/duplicates and master-resolution temporal inventory. Re-run current HEAD and heavy temporal fixtures. |
| FPS semantics | PARTIAL | `true_accelerated` and duration-preserving semantics are documented and benchmarked historically. Re-run current HEAD and verify production output metadata/duration. |
| Offline Master | PARTIAL | Direct master-path and FFmpeg contract are documented. Run an actual current-head FFmpeg output test; the benchmark memory sink alone does not certify encoding. |
| Provider behavior | PARTIAL | CPU fallback and CUDA/native tooling exist; current provider-specific failure/fallback and residency tests are incomplete for release certification. |
| Regression corpus | PARTIAL | Multiple smoke/benchmark scripts exist. Consolidate them into a release corpus with a single recorded current-head result. |
| Packaging | PARTIAL | Windows packaging and standalone verification have extensive historical work. Rebuild from current HEAD and record commit → artifact provenance plus reference render. |

## Repository observations driving the course correction

### Strong foundations already present

- `docs/architecture/current-frame-pipeline.md` contains a detailed evidence-based live frame-path audit.
- `docs/profiling/render-quality-overhaul.md` establishes master-resolution Offline Master correctness, explicit FPS semantics, deterministic graph/hash controls, and downstream codec separation.
- Native migration documents explicitly require incremental, reversible, benchmark-gated work.
- Production Windows build/release documentation already exists.
- The repository has focused benchmark tooling for framebuffer, worker lifecycle, shared transport, native runtime, temporal prototype, P-frame hot stage, and render-quality path.

### Trajectory problem observed

The latest production commit history resumed broad feature/UI expansion immediately after the quality-first overhaul. The latest inspected commit includes cognitive mutation, sequencer, dual-deck and additional workspace state alongside useful parity/interaction changes.

Because these changes are mixed, this course correction does **not** blindly revert them. Instead, it freezes further feature expansion and makes current-head regression evidence the deciding factor.

## Immediate bounded task queue

Execute in order. Do not combine stages unless the earlier gate passes.

1. **Current-head build audit**
   - run release build(s) and Python dependency diagnostics;
   - record exact commands and failures;
   - no feature edits.

2. **Launch lifecycle**
   - cold launch → project open → clean exit → second cold launch;
   - record any orphan worker/process state.

3. **Reference graph round trip**
   - load → validate → execute → save → reload;
   - compare topology and parameters.

4. **Master/temporal integrity**
   - rerun `tools/benchmark/run_render_quality_path.py` at the documented reference configuration;
   - verify every correctness gate, frame count, history resolution, graph hash and output hash behavior.

5. **Production Offline Master**
   - run direct FFmpeg output from current HEAD;
   - verify frame count, dimensions, requested semantics, duration and decodeability.

6. **Provider fallback audit**
   - CPU-only reference;
   - CUDA-enabled reference where hardware exists;
   - deliberately unavailable provider path to verify clean fallback/error behavior.

7. **Package from current HEAD**
   - produce artifact with source commit recorded;
   - clean-machine or isolated-environment launch and reference render.

## Agent operating rule

Until this scorecard is current-head certified, do not add new features. For each code change, name the failing gate first, make the smallest change that addresses it, run the relevant focused tests, and report exact evidence.

## Definition of convergence

0.0.9.4 is ready to freeze when all release gates in `RELEASE_GATES.md` are PASS on a known source commit and the packaged artifact can complete the reference render without developer intervention.
