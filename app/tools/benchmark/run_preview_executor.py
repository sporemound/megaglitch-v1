"""Measure the Preview Engine V1 executor at each fixed preview tier.

The qualifying invocation renders both Source -> Display and the repository's
representative saved Deck-A preset at every Preview Engine V1 tier.  Each cell
uses an independent executor and temporal state.  Adaptive resolution is
disabled so the reported tier remains fixed for the entire measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults

bootstrap_for_dev_defaults()

from megaglitch.preview_executor import (  # noqa: E402
    PREVIEW_TIERS,
    PreviewExecutor,
    PreviewGraphSnapshot,
)
from megaglitch.session_graph_compiler import (  # noqa: E402
    compile_preset_graph,
)


DEFAULT_PRESET = ROOT / "node_presets" / "6-20-26.json"
DEFAULT_OUTPUT = ROOT / "tools" / "benchmark" / "results" / "preview-executor.json"
MINIMUM_MEASURED_RENDERS = 300
DEFAULT_WARMUP = 30
DEFAULT_INPUT_RESOLUTION = (1920, 1080)
SOURCE_NAMESPACE = {
    "SOURCE": "primary",
    "SOURCE_B": "secondary",
}
WORKLOADS = ("source_to_display", "saved_preset_deck_a")


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample set."""

    if not samples:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(float(sample) for sample in samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def deterministic_source(width: int, height: int) -> np.ndarray:
    """Create one stable RGB8 2-D gradient without source-decoder overhead."""

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y[:, None]
    frame[..., 2] = (
        (x.astype(np.uint16)[None, :] + y.astype(np.uint16)[:, None]) // 2
    ).astype(np.uint8)
    return frame


def source_to_display_pipeline() -> list[dict[str, Any]]:
    return [
        {
            "effect_type": "Source",
            "params": {},
            "node_type": "source",
            "source_key": "primary",
        },
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]


def load_saved_deck_a(preset_path: Path) -> tuple[list[dict[str, Any]], str]:
    document = json.loads(preset_path.read_text(encoding="utf-8"))
    decks = document.get("decks")
    if not isinstance(decks, Mapping) or not isinstance(decks.get("a"), Mapping):
        raise ValueError(f"saved preset has no Deck-A graph: {preset_path}")
    pipeline = compile_preset_graph(decks["a"], SOURCE_NAMESPACE)
    if not pipeline:
        raise ValueError(f"saved preset Deck A compiled to an empty graph: {preset_path}")
    try:
        identity = str(preset_path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        identity = str(preset_path.resolve())
    return pipeline, identity


def load_workloads(
    selected: str,
    preset_path: Path,
) -> list[tuple[str, list[dict[str, Any]], str | None]]:
    names = WORKLOADS if selected == "all" else (selected,)
    workloads: list[tuple[str, list[dict[str, Any]], str | None]] = []
    for name in names:
        if name == "source_to_display":
            workloads.append((name, source_to_display_pipeline(), None))
        elif name == "saved_preset_deck_a":
            pipeline, identity = load_saved_deck_a(preset_path)
            workloads.append((name, pipeline, identity))
        else:  # argparse constrains this, but keep the library entry point safe.
            raise ValueError(f"unknown preview benchmark workload {name!r}")
    return workloads


def selected_tiers(selected: str) -> tuple[tuple[int, int], ...]:
    if selected == "all":
        return PREVIEW_TIERS
    width_text, height_text = selected.lower().split("x", 1)
    tier = (int(width_text), int(height_text))
    if tier not in PREVIEW_TIERS:
        raise ValueError(f"unsupported preview tier {selected!r}")
    return (tier,)


def _node_statuses(executor: PreviewExecutor) -> list[dict[str, str]]:
    return [status.as_dict() for status in executor.telemetry().node_statuses]


def _counter_delta(after: Any, before: Any, field: str) -> int:
    return int(getattr(after, field)) - int(getattr(before, field))


def run_cell(
    *,
    workload: str,
    pipeline: Sequence[Mapping[str, Any]],
    preset_identity: str | None,
    tier: tuple[int, int],
    source_frame: np.ndarray,
    warmup: int,
    iterations: int,
    timeout: float,
) -> dict[str, Any]:
    snapshot = PreviewGraphSnapshot.from_pipeline(pipeline, graph_revision=1)
    executor = PreviewExecutor(
        snapshot,
        preferred_tier=tier,
        adaptive=False,
        autostart=True,
    )
    render_ms: list[float] = []
    measured_sequences: set[int] = set()
    final_telemetry = None
    warmup_telemetry = None
    wall_seconds = 0.0
    try:
        for sequence in range(1, warmup + 1):
            source_frame[0, 0, 0] = sequence % 256
            executor.submit_latest(
                source_frame,
                source_timestamp=sequence / 60.0,
                graph_revision=1,
                sequence=sequence,
                source_position=sequence / 60.0,
            )
            result = executor.wait_for_result(
                after_sequence=sequence - 1,
                timeout=timeout,
            )
            if result.sequence != sequence:
                raise RuntimeError(
                    f"warmup sequence mismatch: expected {sequence}, got {result.sequence}"
                )

        warmup_telemetry = executor.telemetry()
        wall_started = time.perf_counter()
        for offset in range(iterations):
            sequence = warmup + offset + 1
            source_frame[0, 0, 0] = sequence % 256
            executor.submit_latest(
                source_frame,
                source_timestamp=sequence / 60.0,
                graph_revision=1,
                sequence=sequence,
                source_position=sequence / 60.0,
            )
            result = executor.wait_for_result(
                after_sequence=sequence - 1,
                timeout=timeout,
            )
            if result.sequence != sequence:
                raise RuntimeError(
                    f"measurement sequence mismatch: expected {sequence}, got {result.sequence}"
                )
            if result.tier != tier:
                raise RuntimeError(f"fixed tier changed from {tier!r} to {result.tier!r}")
            expected_shape = (tier[1], tier[0], 3)
            if result.frame.shape != expected_shape:
                raise RuntimeError(
                    f"preview shape mismatch: expected {expected_shape!r}, "
                    f"got {result.frame.shape!r}"
                )
            render_ms.append(float(result.render_ms))
            measured_sequences.add(result.sequence)
        wall_seconds = time.perf_counter() - wall_started
        final_telemetry = executor.telemetry()
    finally:
        executor.stop(timeout=max(5.0, timeout))

    if warmup_telemetry is None or final_telemetry is None:
        raise RuntimeError("preview executor benchmark did not complete")
    if len(measured_sequences) != iterations:
        raise RuntimeError(
            f"expected {iterations} unique renders, measured {len(measured_sequences)}"
        )

    measured_requests = _counter_delta(
        final_telemetry, warmup_telemetry, "preview_requests"
    )
    measured_rendered = _counter_delta(
        final_telemetry, warmup_telemetry, "preview_rendered"
    )
    measured_superseded = _counter_delta(
        final_telemetry, warmup_telemetry, "preview_superseded"
    )
    measured_dropped = _counter_delta(
        final_telemetry, warmup_telemetry, "preview_dropped"
    )
    return {
        "workload": workload,
        "preset": preset_identity,
        "tier": [tier[0], tier[1]],
        "adaptive": False,
        "warmup_renders": warmup,
        "measured_renders": len(measured_sequences),
        "mean_ms": statistics.fmean(render_ms),
        "p50_ms": percentile(render_ms, 0.50),
        "p95_ms": percentile(render_ms, 0.95),
        "unique_render_fps": len(measured_sequences) / wall_seconds,
        "measurement_wall_seconds": wall_seconds,
        "requests": measured_requests,
        "rendered": measured_rendered,
        "superseded": measured_superseded,
        "dropped": measured_dropped,
        "final_resolution": [
            int(final_telemetry.preview_resolution[0]),
            int(final_telemetry.preview_resolution[1]),
        ],
        "executor_totals": {
            "requests": int(final_telemetry.preview_requests),
            "rendered": int(final_telemetry.preview_rendered),
            "superseded": int(final_telemetry.preview_superseded),
            "dropped": int(final_telemetry.preview_dropped),
        },
        "graph_revision": int(final_telemetry.graph_revision),
        "resolution_generation": int(final_telemetry.resolution_generation),
        "node_statuses": _node_statuses(executor),
        "last_error": final_telemetry.last_error,
    }


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _opencv_metadata() -> dict[str, Any]:
    try:
        import cv2
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    cuda_device_count: int | None = None
    cuda_error: str | None = None
    try:
        if hasattr(cv2, "cuda"):
            cuda_device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as error:
        cuda_error = f"{type(error).__name__}: {error}"
    return {
        "available": True,
        "version": str(getattr(cv2, "__version__", "unknown")),
        "path": str(getattr(cv2, "__file__", "")),
        "cuda_device_count": cuda_device_count,
        "cuda_error": cuda_error,
    }


def run_benchmark(
    *,
    workload: str,
    tier: str,
    preset_path: Path,
    input_width: int,
    input_height: int,
    warmup: int,
    iterations: int,
    timeout: float,
) -> dict[str, Any]:
    tiers = selected_tiers(tier)
    if any(input_width < width or input_height < height for width, height in tiers):
        raise ValueError(
            "input resolution must be at least as large as every selected preview tier"
        )
    source_frame = deterministic_source(input_width, input_height)
    workloads = load_workloads(workload, preset_path)
    results: list[dict[str, Any]] = []
    for workload_name, pipeline, preset_identity in workloads:
        for fixed_tier in tiers:
            results.append(
                run_cell(
                    workload=workload_name,
                    pipeline=pipeline,
                    preset_identity=preset_identity,
                    tier=fixed_tier,
                    source_frame=source_frame,
                    warmup=warmup,
                    iterations=iterations,
                    timeout=timeout,
                )
            )

    return {
        "schema_version": 1,
        "benchmark": "preview-executor-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "logical_cpu_count": os.cpu_count(),
            "opencv": _opencv_metadata(),
        },
        "configuration": {
            "workload": workload,
            "tier": tier,
            "input_resolution": [input_width, input_height],
            "warmup": warmup,
            "iterations": iterations,
            "minimum_qualifying_iterations": MINIMUM_MEASURED_RENDERS,
            "qualifying_measurement": iterations >= MINIMUM_MEASURED_RENDERS,
            "timeout_seconds_per_render": timeout,
            "source": "deterministic_rgb8_gradient",
        },
        "runs": results,
    }


def write_json(result: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    tier_choices = ("all",) + tuple(f"{width}x{height}" for width, height in PREVIEW_TIERS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=("all",) + WORKLOADS,
        default="all",
    )
    parser.add_argument("--tier", choices=tier_choices, default="all")
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_RESOLUTION[0])
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_RESOLUTION[1])
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=MINIMUM_MEASURED_RENDERS)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-short-run",
        action="store_true",
        help="permit fewer than 300 measurements for non-qualifying smoke checks",
    )
    args = parser.parse_args()

    if min(args.input_width, args.input_height, args.iterations) <= 0:
        parser.error("input dimensions and iterations must be positive")
    if args.warmup < 0:
        parser.error("warmup cannot be negative")
    if args.timeout <= 0.0:
        parser.error("timeout must be positive")
    if args.iterations < MINIMUM_MEASURED_RENDERS and not args.allow_short_run:
        parser.error(
            f"qualifying runs require at least {MINIMUM_MEASURED_RENDERS} iterations; "
            "use --allow-short-run only for a non-qualifying smoke check"
        )

    result = run_benchmark(
        workload=args.workload,
        tier=args.tier,
        preset_path=args.preset,
        input_width=args.input_width,
        input_height=args.input_height,
        warmup=args.warmup,
        iterations=args.iterations,
        timeout=args.timeout,
    )
    write_json(result, args.output)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
