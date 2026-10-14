# ruff: noqa: E402
"""Benchmark B: Headless Quality Reference Harness.

Measures the raw, UI-independent rendering throughput and aesthetic fidelity
of the MegaGlitch master pipeline without UI presentation, preview generation,
IPC, or live SHA calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
except Exception:
    pass

import cv2
import numpy as np

from megaglitch.master_graph import MasterGraphExecutor, describe_provider_assignments
from megaglitch.offline_master import (
    FpsSemantics,
    OfflineMasterConfig,
    OfflineMasterRenderer,
    build_frame_schedule,
    stable_graph_hash,
)
from megaglitch.production_graph import (
    PRODUCTION_GRAPH_PROFILE_ID,
    REFERENCE_QA_PROFILE_ID,
    get_graph_by_profile,
)
from megaglitch.render_contract import MasterResolution
from tools.benchmark.run_render_quality_path import (
    HashMemorySink,
    quality_graph_pipeline,
    synthetic_master_frames,
)


def _fraction_text(value: Fraction | str | int | float) -> str:
    fraction = value if isinstance(value, Fraction) else Fraction(str(value))
    return f"{fraction.numerator}/{fraction.denominator}"


def behavioral_graph_hash(profile_id: str, graph: dict[str, Any]) -> str:
    """Hash graph behavior without reseeding the frozen QA oracle on a label change."""

    graph_for_hash = dict(graph)
    if profile_id == REFERENCE_QA_PROFILE_ID:
        graph_for_hash.pop("profile_id", None)
    return stable_graph_hash(graph_for_hash)


def run_headless_benchmark(
    *,
    graph_profile: str = PRODUCTION_GRAPH_PROFILE_ID,
    width: int = 1920,
    height: int = 1080,
    frame_count: int = 30,
    source_fps: str | int | float = "30",
    output_fps: str | int | float = "55",
    semantics: FpsSemantics | str = FpsSemantics.TRUE_ACCELERATED,
    seed: int = 917,
    use_cuda: bool = True,
    threads: int = 2,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Execute headless master render and record pure compute telemetry."""
    cv2.setNumThreads(int(threads))
    cv2.ocl.setUseOpenCL(False)

    resolved_profile, graph = get_graph_by_profile(graph_profile)
    fps_mode = FpsSemantics(semantics)
    schedule = build_frame_schedule(frame_count, source_fps, output_fps, fps_mode)
    pipeline = quality_graph_pipeline(graph)
    graph_hash = behavioral_graph_hash(resolved_profile, graph)
    resolution = MasterResolution(width=width, height=height)
    executor = MasterGraphExecutor(resolution, use_cuda=use_cuda)
    providers = describe_provider_assignments(pipeline, None)

    config = OfflineMasterConfig(
        width=width,
        height=height,
        source_fps=source_fps,
        output_fps=output_fps,
        fps_semantics=fps_mode,
        graph_hash=graph_hash,
        random_seed=int(seed),
        node_provider_assignment=providers,
    )

    sink = HashMemorySink()

    # Source frame iterator
    if source_path and Path(source_path).is_file():
        cap = cv2.VideoCapture(str(source_path))
        def read_source_frames():
            for _ in range(frame_count):
                ret, f = cap.read()
                if not ret:
                    break
                if f.shape[:2] != (height, width):
                    f = cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
                yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            cap.release()
        frames_iter = read_source_frames()
    else:
        frames_iter = synthetic_master_frames(width, height, frame_count, int(seed))

    frame_render_times: list[float] = []

    def process(frame: np.ndarray, context: Any) -> np.ndarray:
        t0 = time.perf_counter()
        result = executor.process(
            frame,
            pipeline_a=pipeline,
            master_mix=0.0,
            source_index=context.source_index,
        )
        t1 = time.perf_counter()
        frame_render_times.append((t1 - t0) * 1000.0)
        return result.frame

    t_wall_start = time.perf_counter()
    render_summary = OfflineMasterRenderer(config).render(
        frames_iter,
        source_frame_count=frame_count,
        processor=process,
        encoder=sink,
    )
    t_wall_end = time.perf_counter()
    wall_duration_seconds = t_wall_end - t_wall_start

    processed_count = len(frame_render_times)
    total_compute_ms = sum(frame_render_times)
    mean_render_ms = float(np.mean(frame_render_times)) if frame_render_times else 0.0
    p50_render_ms = float(np.percentile(frame_render_times, 50)) if frame_render_times else 0.0
    p95_render_ms = float(np.percentile(frame_render_times, 95)) if frame_render_times else 0.0

    processing_fps_unique = float(processed_count / (total_compute_ms / 1000.0)) if total_compute_ms > 0 else 0.0
    source_duration = float(schedule.source_duration_seconds)
    speed_multiplier = float(source_duration / wall_duration_seconds) if wall_duration_seconds > 0 else 0.0

    telemetry = {
        "benchmark": "megaglitch-headless-quality-reference",
        "mode": "headless_quality_reference",
        "graph_profile": resolved_profile,
        "resolution": [width, height],
        "source_fps": float(schedule.source_fps),
        "output_fps": float(schedule.output_fps),
        "fps_semantics": str(fps_mode.value),
        "frame_counts": {
            "source": frame_count,
            "unique_master_processed": processed_count,
            "encoded": schedule.output_frame_count,
        },
        "durations": {
            "source_seconds": source_duration,
            "output_seconds": float(schedule.output_duration_seconds),
            "wall_time_seconds": wall_duration_seconds,
        },
        "throughput": {
            "processing_fps_unique": processing_fps_unique,
            "speed_multiplier": speed_multiplier,
            "render_time_per_frame": {
                "mean_ms": mean_render_ms,
                "p50_ms": p50_render_ms,
                "p95_ms": p95_render_ms,
            },
        },
        "environment": {
            "use_cuda": use_cuda,
            "threads": threads,
            "seed": seed,
            "source": "video_file" if source_path else "deterministic_synthetic",
        },
        "master_output_sha256": sink.output_sha256.hexdigest(),
    }
    return telemetry


def main():
    parser = argparse.ArgumentParser(description="Benchmark B: Headless Quality Reference Harness")
    parser.add_argument("--graph-profile", default=PRODUCTION_GRAPH_PROFILE_ID, help="Graph profile (production-quality-v1, reference-qa-v1)")
    parser.add_argument("--resolution", default="1920x1080", help="Master resolution WxH")
    parser.add_argument("--frame-count", type=int, default=30, help="Number of frames")
    parser.add_argument("--source-fps", default="30", help="Source frame rate")
    parser.add_argument("--output-fps", default="55", help="Output timebase frame rate")
    parser.add_argument("--semantics", default="true_accelerated", choices=["true_accelerated", "duration_preserving"])
    parser.add_argument("--seed", type=int, default=917, help="Random seed")
    parser.add_argument("--use-cuda", action="store_true", default=True, help="Enable CUDA operations")
    parser.add_argument("--threads", type=int, default=2, help="OpenCV worker thread count")
    parser.add_argument("--source-file", default=None, help="Optional source video path")
    parser.add_argument("--json-output", default=None, help="Path for output JSON")
    parser.add_argument("--csv-output", default=None, help="Path for output CSV")
    args = parser.parse_args()

    parts = args.resolution.lower().split("x")
    width, height = int(parts[0]), int(parts[1])

    telemetry = run_headless_benchmark(
        graph_profile=args.graph_profile,
        width=width,
        height=height,
        frame_count=args.frame_count,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        semantics=args.semantics,
        seed=args.seed,
        use_cuda=args.use_cuda,
        threads=args.threads,
        source_path=args.source_file,
    )

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        print(f"Wrote JSON telemetry to {out_path}")

    print(json.dumps({
        "graph_profile": telemetry["graph_profile"],
        "throughput": telemetry["throughput"],
        "frame_counts": telemetry["frame_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
