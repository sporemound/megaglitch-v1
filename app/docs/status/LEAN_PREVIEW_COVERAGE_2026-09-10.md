# LEAN-001: truthful preview coverage during structural fallback

## Value and cross-reference

- R3: approximately 60 new preview frames per second with effect coverage and approximations disclosed.
- R4: effective graph controls and trustworthy processing-state reporting.
- Release checks: clear node/provider failure reporting (PROJECT_SCOPE.md), Gate H reporting, and Gate I regression evidence.

## Baseline and evidence

Scope remains `sporemound/MegaGlitch-Node0.9.4`. This isolated development branch starts at recovery revision `ff8b53c9376c271c1b7ce8ca95f38065f3073603`, 17 commits beyond scoped main. The successor repository was inspected in the preceding audit but is not being merged or used as this checkout's remote. Later unavailable V: work remains unreconciled; no global baseline/convergence claim is made.

`build_preview_plan` replaces a graph containing an unsupported structural node with a source/display-only plan. It retains statuses from the original graph, so supported effects removed by that fallback still claim `implemented`. `tools/rust_ui_bridge.py` and the Rust UI use those statuses to report unsupported nodes. A source-only preview can therefore overstate active effect coverage.

## Smallest change

Keep the existing fallback and rendered pixels. Mark active nodes omitted by the fallback as `unsupported`, with provider `source_display_fallback`. Keep the original unsupported node's status, disabled statuses, the retained source, and terminal display statuses. Reuse the existing bridge/UI unsupported reporting without a protocol or architectural change.

## Acceptance

- [x] Reproduce false implemented status with a supported effect plus unsupported structural node.
- [x] Verify source-only output and truthful omitted-effect/source statuses.
- [x] Verify disabled nodes retain disabled status and ordinary supported graphs still execute.
- [x] Run focused preview and bridge regressions plus a synchronous real-executor smoke check.
- [x] Record environment, results, limitations, and rollback.

Status: completed locally on `codex/lean-preview-coverage`; not committed, pushed, merged, or packaged. This does not implement missing preview effects or certify throughput, visual parity, CUDA, devices, or packaging.

## Verification record

Environment: isolated Windows Python 3.12.14 environment, NumPy 2.5.3, OpenCV headless 5.0.0.93, Pillow 12.3.0, pytest 9.1.1. Dependencies were installed only in the task's `work/lean-test-env`; the user's existing CUDA/OpenCV installation was not replaced.

Commands below were run from the scoped checkout in `work/MegaGlitch-Node0.9.4`:

```powershell
# Reproduction before the production fix: 4 failed, 1 passed.
../lean-test-env/Scripts/python.exe -m pytest -q tests/test_preview_executor.py -k 'structural_fallback_reports or supported_preview_effect_executes' --tb=line

# After the production fix: 34 passed in 14.04 seconds.
../lean-test-env/Scripts/python.exe -m pytest -q tests/test_preview_executor.py tests/test_preview_master_isolation.py tests/test_rust_ui_bridge.py --tb=short

# Added UI-facing bridge acceptance check: 1 passed in 0.36 seconds.
../lean-test-env/Scripts/python.exe -m pytest -q tests/test_rust_ui_bridge.py -k reports_effects_omitted --tb=short

# No whitespace errors.
git diff --check
```

Total: 35 passing focused tests across the two post-fix runs. The real-executor tests render a patterned 64×36 RGB frame: fallback preserves source pixels, while a supported inversion-only graph produces exactly `255 - source`. Generator/mixer fallback cases cover enabled and disabled effects, omitted secondary sources, and unchanged input snapshots. The bridge test submits a real compiled branching-mixer preset, checks that omitted inversion appears in `preview_unsupported_nodes`, then replaces the graph with supported inversion and verifies that the warning clears.

No Rust code changed. The existing Rust UI filters the same `unsupported` status surfaced by the bridge. A live GUI/build/package check was not run, so the bridge assertion is not presented as visual UI certification. No performance improvement is claimed: rendering, history ownership, and fallback pixel behavior remain unchanged.

## Changed files and rollback

- `megaglitch/preview_executor.py`: reconcile statuses with the source/display fallback plan.
- `tests/test_preview_executor.py`: five real-executor regression cases.
- `tests/test_rust_ui_bridge.py`: one graph-replacement/status propagation regression.
- This document: requirement trace, baseline, acceptance, and verification record.

Rollback only this work item's patch if necessary; do not reset or revert the recovery branch as a whole. The change preserves existing status values and adds only the descriptive provider string `source_display_fallback` for omitted active nodes.

## Next pull item

Use the existing supported-effect inventory to select one missing user workflow, reproduce its failure, and close its smallest end-to-end gap. Retain the broader R1–R7 requirements; this reporting fix closes neither full preview coverage nor overall release convergence. Recovering the original visual reference and later local work remains necessary before claiming product-wide parity.
