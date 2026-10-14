# ruff: noqa: E402
"""Exercise the deterministic quality-first master render path.

This harness is a render-path correctness instrument first.  It runs sharp,
moving synthetic RGB8 frames through ``MasterGraphExecutor`` under
``OfflineMasterRenderer`` and uses an in-memory hash sink at encoder egress.
The sink deliberately reports no bitrate: use the application's direct
FFmpeg offline export when encoder throughput or bitrate is the subject of a
measurement.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
except Exception:
    pass

import numpy as np
import cv2

from megaglitch.master_graph import MasterGraphExecutor, describe_provider_assignments
from megaglitch.offline_master import (
    FFmpegEncodeResult,
    FpsSemantics,
    OfflineMasterConfig,
    OfflineMasterRenderer,
    build_frame_schedule,
    stable_graph_hash,
)
from megaglitch.quality_graph import build_quality_first_graph
from megaglitch.render_contract import MasterResolution


SCHEMA_VERSION = 1
BENCHMARK_NAME = "megaglitch-render-quality-path"


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@contextmanager
def _deterministic_opencv_runtime() -> Iterator[None]:
    """Pin the CPU benchmark's OpenCV scheduler and restore caller settings."""

    previous_threads = int(cv2.getNumThreads())
    previous_opencl = bool(cv2.ocl.useOpenCL())
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    try:
        yield
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse ``WIDTHxHEIGHT`` without permitting an implicit resize."""

    normalized = str(value).strip().lower().replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def quality_graph_pipeline(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile the pure quality preset into the flat Python executor shape."""

    pipeline: list[dict[str, Any]] = [
        {
            "node_type": "source",
            "effect_type": "Source",
            "node_id": "quality_master_source",
            "source_key": "primary",
        }
    ]
    for node in graph.get("nodes", ()):  # graph order is the serial stage order
        pipeline.append(
            {
                "node_type": "effect",
                "effect_type": str(node["label"]),
                "node_id": str(node["id"]),
                "params": dict(node.get("params") or {}),
            }
        )
    pipeline.append(
        {
            "node_type": "display",
            "effect_type": "Display",
            "node_id": "quality_master_display",
            "params": {},
        }
    )
    return pipeline


def synthetic_master_frames(
    width: int,
    height: int,
    frame_count: int,
    seed: int,
) -> Iterable[np.ndarray]:
    """Yield deterministic sharp/moving frames at exactly master resolution."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    y, x = np.indices((height, width), dtype=np.int32)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    fixed_texture = rng.integers(0, 16, size=(height, width), dtype=np.uint8)
    checker = (((x // 4) + (y // 4)) & 1).astype(np.uint8) * 37

    for index in range(frame_count):
        red = (x * 7 + y * 3 + index * 17) & 255
        green = (x * 2 + y * 11 + checker + index * 9) & 255
        blue = ((x ^ (y * 3)) + fixed_texture + index * 13) & 255
        frame = np.stack((red, green, blue), axis=2).astype(np.uint8)

        # Moving hard-edged bars exercise flow, difference, edge, feedback,
        # held-frame, and reconstruction stages without source-side scaling.
        bar_x = (index * 5) % max(1, width)
        bar_y = (index * 3) % max(1, height)
        frame[:, bar_x : min(width, bar_x + max(1, width // 12)), :] = (
            245,
            38,
            196,
        )
        frame[bar_y : min(height, bar_y + max(1, height // 14)), :, :] = (
            24,
            238,
            92,
        )
        yield np.ascontiguousarray(frame)


class HashMemorySink:
    """Encoder-egress sink that retains only count, bytes, and a frame hash."""

    def __init__(self) -> None:
        self.frames_written = 0
        self.bytes_written = 0
        self.output_sha256 = hashlib.sha256()
        self.write_seconds = 0.0
        self.aborted = False

    def write_frame(self, frame: np.ndarray) -> None:
        started = time.perf_counter()
        array = np.asarray(frame)
        self.output_sha256.update(memoryview(array).cast("B"))
        self.frames_written += 1
        self.bytes_written += int(array.nbytes)
        self.write_seconds += time.perf_counter() - started

    def close(self) -> FFmpegEncodeResult:
        return FFmpegEncodeResult(
            command=("memory-hash-sink",),
            frames_written=self.frames_written,
            bytes_written=self.bytes_written,
            returncode=0,
            stderr="",
            elapsed_seconds=self.write_seconds,
            readback={
                "available": True,
                "sink": "memory_hash",
                "bit_rate": None,
                "output_sha256": self.output_sha256.hexdigest(),
            },
        )

    def abort(self) -> None:
        self.aborted = True


def _merge_copy_stats(
    totals: dict[str, dict[str, int]],
    current: Mapping[str, Mapping[str, int]],
) -> None:
    for boundary, values in current.items():
        target = totals.setdefault(str(boundary), {"count": 0, "bytes": 0})
        target["count"] += int(values.get("count", 0))
        target["bytes"] += int(values.get("bytes", 0))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_benchmark_impl(
    *,
    width: int = 640,
    height: int = 360,
    frame_count: int = 30,
    source_fps: str | int | float = "30",
    output_fps: str | int | float = "55",
    semantics: FpsSemantics | str = FpsSemantics.TRUE_ACCELERATED,
    seed: int = 917,
    use_cuda: bool = False,
) -> dict[str, Any]:
    """Run one fixed-state offline master sequence and return telemetry."""

    width = int(width)
    height = int(height)
    frame_count = int(frame_count)
    if width <= 0 or height <= 0:
        raise ValueError("master resolution must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    fps_mode = FpsSemantics(semantics)
    schedule = build_frame_schedule(frame_count, source_fps, output_fps, fps_mode)
    graph = build_quality_first_graph()
    pipeline = quality_graph_pipeline(graph)
    graph_hash = stable_graph_hash(graph)
    resolution = MasterResolution(width=width, height=height)
    executor = MasterGraphExecutor(resolution, use_cuda=use_cuda)
    providers = describe_provider_assignments(pipeline, None)
    initial_history = executor.history_buffer_resolution
    config = OfflineMasterConfig(
        width=width,
        height=height,
        source_fps=source_fps,
        output_fps=output_fps,
        fps_semantics=fps_mode,
        graph_hash=graph_hash,
        random_seed=int(seed),
        history_buffer_resolution=initial_history,
        node_provider_assignment=providers,
    )
    sink = HashMemorySink()
    last_graph_result = None
    copy_totals: dict[str, dict[str, int]] = {}

    def process(frame: np.ndarray, context: Any) -> np.ndarray:
        nonlocal last_graph_result
        last_graph_result = executor.process(
            frame,
            pipeline_a=pipeline,
            master_mix=0.0,
            instrumentation=context.instrumentation,
            source_index=context.source_index,
        )
        _merge_copy_stats(copy_totals, last_graph_result.frame_copy_stats)
        return last_graph_result.frame

    result = OfflineMasterRenderer(config).render(
        synthetic_master_frames(width, height, frame_count, int(seed)),
        source_frame_count=frame_count,
        processor=process,
        encoder=sink,
    )
    if last_graph_result is None:  # guarded by positive frame_count, kept explicit
        raise RuntimeError("quality benchmark produced no graph result")

    measured_history = {
        str(name): [int(size[0]), int(size[1])]
        for name, size in sorted(executor.history_buffer_resolution.items())
    }
    expected_resolution = [width, height]
    history_is_master = all(
        value == expected_resolution for value in measured_history.values()
    )
    telemetry = dict(result.telemetry)
    schedule_matches = (
        telemetry["frame_count"]["source"] == schedule.source_frame_count
        and telemetry["frame_count"]["master_processed"] == schedule.source_frame_count
        and telemetry["frame_count"]["encoded"] == schedule.output_frame_count
        and telemetry["duplicated_frames"] == schedule.duplicated_frames
        and telemetry["source_frames_omitted_at_egress"]
        == schedule.source_frames_omitted_at_egress
        and telemetry["fps_semantics"] == schedule.semantics.value
    )
    telemetry.update(
        {
            "fps_exact": {
                "source": _fraction_text(schedule.source_fps),
                "output": _fraction_text(schedule.output_fps),
            },
            "duration_exact": {
                "source": _fraction_text(schedule.source_duration_seconds),
                "output": _fraction_text(schedule.output_duration_seconds),
            },
            "history_buffer_resolution": measured_history,
            "history_buffer_inventory": {
                "deck_a": _json_ready(last_graph_result.temporal_inventory_a),
                "deck_b": _json_ready(last_graph_result.temporal_inventory_b),
            },
            "history_depth": {
                "deck_a": int(last_graph_result.history_depth_a),
                "deck_b": int(last_graph_result.history_depth_b),
            },
            "frame_copy_stats": copy_totals,
            "master_output_sha256": sink.output_sha256.hexdigest(),
            "benchmark_sink": {
                "kind": "memory_hash",
                "bitrate_measured": False,
                "note": "No codec is run; bitrate is intentionally null.",
            },
            "correctness_gates": {
                "all_source_frames_processed_once": (
                    telemetry["frame_count"]["master_processed"] == frame_count
                ),
                "no_runtime_dropped_frames": telemetry["dropped_frames"] == 0,
                "all_measured_history_at_master_resolution": history_is_master,
                "fps_schedule_matches_exact_rational_model": schedule_matches,
                "master_output_shape": expected_resolution,
                "preview_obs_vcam_reentry": False,
            },
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "configuration": {
            "resolution": expected_resolution,
            "frame_count": frame_count,
            "source_fps": str(source_fps),
            "output_fps": str(output_fps),
            "fps_semantics": fps_mode.value,
            "random_seed": int(seed),
            "use_cuda": bool(use_cuda),
            "source": "deterministic_synthetic_rgb8",
            "egress_sink": "memory_hash",
            "opencv_determinism": {
                "version": str(cv2.__version__),
                "threads": 1,
                "opencl": False,
            },
        },
        "telemetry": _json_ready(telemetry),
    }


def run_benchmark(
    *,
    width: int = 640,
    height: int = 360,
    frame_count: int = 30,
    source_fps: str | int | float = "30",
    output_fps: str | int | float = "55",
    semantics: FpsSemantics | str = FpsSemantics.TRUE_ACCELERATED,
    seed: int = 917,
    use_cuda: bool = False,
    profile_mode: str = "deterministic",
) -> dict[str, Any]:
    """Run benchmark under either deterministic single-threaded or performance mode."""
    if str(profile_mode).lower() == "performance":
        return _run_benchmark_impl(
            width=width,
            height=height,
            frame_count=frame_count,
            source_fps=source_fps,
            output_fps=output_fps,
            semantics=semantics,
            seed=seed,
            use_cuda=use_cuda,
        )

    with _deterministic_opencv_runtime():
        return _run_benchmark_impl(
            width=width,
            height=height,
            frame_count=frame_count,
            source_fps=source_fps,
            output_fps=output_fps,
            semantics=semantics,
            seed=seed,
            use_cuda=use_cuda,
        )


def _flatten_for_csv(
    value: Any,
    *,
    prefix: str = "",
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = output if output is not None else {}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten_for_csv(nested, prefix=name, output=target)
    elif isinstance(value, (list, tuple)):
        target[prefix] = json.dumps(_json_ready(value), separators=(",", ":"))
    elif value is None:
        target[prefix] = ""
    else:
        target[prefix] = value
    return target


def write_results(
    result: Mapping[str, Any],
    *,
    json_output: Path,
    csv_output: Path,
) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(_json_ready(result), indent=2) + "\n", encoding="utf-8")
    row = _flatten_for_csv(result)
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MegaGlitch's deterministic master-resolution quality path."
    )
    parser.add_argument("--resolution", type=parse_resolution, default=(640, 360))
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--source-fps", default="30")
    parser.add_argument("--output-fps", default="55")
    parser.add_argument(
        "--semantics",
        choices=[mode.value for mode in FpsSemantics],
        default=FpsSemantics.TRUE_ACCELERATED.value,
    )
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument(
        "--profile-mode",
        choices=["deterministic", "performance"],
        default="deterministic",
        help="Deterministic mode locks 1-thread CPU/OpenCV; performance mode allows production scheduler.",
    )
    parser.add_argument(
        "--json-output",
        "--output-json",
        dest="json_output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--csv-output",
        "--output-csv",
        dest="csv_output",
        type=Path,
        default=None,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    width, height = args.resolution

    if args.json_output is None:
        if args.profile_mode == "performance":
            json_output = Path("tools/benchmark/results/render-quality-performance.json")
        else:
            json_output = Path("tools/benchmark/results/render-quality-path.json")
    else:
        json_output = args.json_output

    if args.csv_output is None:
        if args.profile_mode == "performance":
            csv_output = Path("tools/benchmark/results/render-quality-performance.csv")
        else:
            csv_output = Path("tools/benchmark/results/render-quality-path.csv")
    else:
        csv_output = args.csv_output

    result = run_benchmark(
        width=width,
        height=height,
        frame_count=args.frame_count,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        semantics=args.semantics,
        seed=args.seed,
        use_cuda=args.use_cuda,
        profile_mode=args.profile_mode,
    )
    write_results(
        result,
        json_output=json_output,
        csv_output=csv_output,
    )
    telemetry = result["telemetry"]
    mean_ms = telemetry["render_time_per_frame"]["mean_ms"]
    print(
        f"Wrote {json_output} and {csv_output}\n"
        f"Master {width}x{height}; {args.frame_count} frames; "
        f"mean render {mean_ms:.3f} ms; drops {telemetry['dropped_frames']}; "
        f"bitrate {telemetry['bitrate'] or 'not measured'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
