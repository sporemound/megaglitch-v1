"""Validate committed benchmark artifacts against hardware-profile gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case(cases, **match):
    for case in cases:
        if all(case.get(key) == value for key, value in match.items()):
            return case
    raise KeyError(f"benchmark case not found: {match}")


def evaluate_report(
    results_dir: Path,
    thresholds: dict,
    *,
    rust_performance_advisory: bool = False,
) -> tuple[list[str], list[str]]:
    failures = []
    advisories = []

    segment_gate = thresholds["pframe_segment"]
    segment = _load(results_dir / segment_gate["artifact"])["cases"]
    for width, height, prefix in ((1200, 800, "1200x800"), (1920, 1080, "1920x1080")):
        case = _case(
            segment, name="pframe_chain_fused_native", width=width, height=height
        )
        if not case.get("pixel_exact") or case.get("max_channel_delta") != 0:
            failures.append(f"{prefix} fused P-frame output is not pixel-exact")
        for field, suffix in (("mean_frame_ms", "mean_ms_max"), ("p99_frame_ms", "p99_ms_max")):
            limit = float(segment_gate[f"{prefix}_{suffix}"])
            if float(case[field]) > limit:
                failures.append(f"{prefix} {field} {case[field]:.3f}ms exceeds {limit:.3f}ms")

    shared_gate = thresholds["shared_pframe_worker"]
    shared_cases = _load(results_dir / shared_gate["artifact"])["cases"]
    shared = _case(
        shared_cases, name="worker_representative_pframe", transport="shared"
    )
    for field, key in (("mean_frame_ms", "mean_ms_max"), ("p99_frame_ms", "p99_ms_max")):
        limit = float(shared_gate[key])
        if float(shared[field]) > limit:
            failures.append(f"shared P-frame {field} {shared[field]:.3f}ms exceeds {limit:.3f}ms")
    for field in ("shared_input_slots_in_use_after_run", "shared_output_slots_in_use_after_run"):
        if shared.get(field) != 0:
            failures.append(f"shared P-frame retained slots: {field}={shared.get(field)}")

    lifecycle_gate = thresholds["worker_lifecycle"]
    lifecycle_cases = _load(results_dir / lifecycle_gate["artifact"])["cases"]
    hashes = {case.get("output_sha256") for case in lifecycle_cases}
    if len(hashes) != 1:
        failures.append("worker lifecycle output hashes differ across cycles")
    for case in lifecycle_cases:
        label = f"lifecycle cycle {case.get('cycle')}"
        for field, key in (
            ("readiness_ms", "readiness_ms_max"),
            ("first_frame_ms", "first_frame_ms_max"),
            ("shutdown_ms", "shutdown_ms_max"),
        ):
            limit = float(lifecycle_gate[key])
            if float(case[field]) > limit:
                failures.append(f"{label} {field} {case[field]:.3f}ms exceeds {limit:.3f}ms")
        if case.get("forced_termination"):
            failures.append(f"{label} required forced termination")
        if case.get("exitcode") != 0 or case.get("final_state") != "stopped":
            failures.append(f"{label} did not stop cleanly")
        if case.get("input_slots_in_use") != 0 or case.get("output_slots_in_use") != 0:
            failures.append(f"{label} retained shared slots")

    rust_gate = thresholds.get("rust_graph_ir")
    if rust_gate:
        rust_result = _load(results_dir / rust_gate["artifact"])
        rust_cases = rust_result["cases"]
        python_case = _case(rust_cases, name=rust_gate["python_case"])
        native_case = _case(rust_cases, name=rust_gate["native_case"])
        if not rust_result.get("graph_ir_worker_pixel_exact"):
            failures.append("Rust graph IR worker output is not pixel-exact")
        if python_case.get("output_sha256") != native_case.get("output_sha256"):
            failures.append("Rust graph IR worker hash differs from Python")
        if native_case.get("backend") != "native_rust":
            failures.append(
                f"Rust graph IR benchmark used unexpected backend {native_case.get('backend')}"
            )
        for field, key in (
            ("mean_frame_ms", "native_mean_ms_max"),
            ("p99_frame_ms", "native_p99_ms_max"),
        ):
            limit = float(rust_gate[key])
            if float(native_case[field]) > limit:
                finding = (
                    f"Rust graph IR {field} {native_case[field]:.3f}ms exceeds {limit:.3f}ms"
                )
                (advisories if rust_performance_advisory else failures).append(finding)
        for field, key in (
            ("mean_frame_ms", "max_mean_ratio_to_python"),
            ("p99_frame_ms", "max_p99_ratio_to_python"),
        ):
            ratio = float(native_case[field]) / max(float(python_case[field]), 1e-9)
            limit = float(rust_gate[key])
            if ratio > limit:
                finding = (
                    f"Rust graph IR {field} ratio {ratio:.3f} exceeds {limit:.3f} of Python"
                )
                (advisories if rust_performance_advisory else failures).append(finding)
        if native_case.get("dropped_frames", 0) != 0:
            failures.append("Rust graph IR benchmark dropped frames")
    return failures, advisories


def evaluate(results_dir: Path, thresholds: dict) -> list[str]:
    """Evaluate every configured regression as a strict hard gate."""

    failures, _ = evaluate_report(results_dir, thresholds)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--thresholds", type=Path, default=HERE / "regression_thresholds.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--rust-performance-advisory",
        action="store_true",
        help=(
            "Report Rust graph-IR timing/ratio regressions as advisories while "
            "keeping Rust correctness and all production Python-host gates strict."
        ),
    )
    args = parser.parse_args()
    thresholds = _load(args.thresholds)
    try:
        failures, advisories = evaluate_report(
            args.results_dir,
            thresholds,
            rust_performance_advisory=args.rust_performance_advisory,
        )
    except Exception as exc:
        failures = [f"regression evaluation failed: {type(exc).__name__}: {exc}"]
        advisories = []
    result = {
        "hardware_profile": thresholds.get("hardware_profile"),
        "passed": not failures,
        "failures": failures,
        "advisories": advisories,
        "rust_performance_policy": (
            "advisory" if args.rust_performance_advisory else "strict"
        ),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
