"""Compare the native temporal prototype with Python history implementations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from megaglitch.native_temporal import TemporalFrameRing, native_available
from tools.benchmark.run_baseline import _process_memory, _summarize_times


RESOLUTIONS = ((1280, 720), (1920, 1080), (1200, 800))
CAPACITIES = (1, 8, 32, 64, 120)


def _frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    return np.stack(((x + seed) % 256 + y * 0, (y + seed * 3) % 256 + x * 0,
                     (x + y + seed * 7) % 256), axis=2).astype(np.uint8)


def _measure(callable_, warmup: int, iterations: int):
    for _ in range(warmup):
        callable_()
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        callable_()
        times.append((time.perf_counter() - started) * 1000.0)
    # Keep allocator tracing out of the timing loop; sample one steady-state
    # operation independently, as in the main framebuffer baseline.
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    callable_()
    after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    allocation = sum(stat.size_diff for stat in after.compare_to(before, "lineno"))
    allocation_count = sum(stat.count_diff for stat in after.compare_to(before, "lineno"))
    return {
        **_summarize_times(times),
        "steady_allocation_bytes_delta": allocation,
        "steady_allocation_count_delta": allocation_count,
        "allocation_sample_operations": 1,
        "tracemalloc_current_bytes": current,
        "tracemalloc_peak_bytes": peak,
    }


def _list_case(width, height, capacity, warmup, iterations):
    frame = _frame(width, height)
    history = [frame.copy() for _ in range(capacity)]

    def write():
        history.append(frame.copy())
        history.pop(0)

    memory_before = _process_memory()
    result = _measure(write, warmup, iterations)
    return {
        "name": "temporal_write", "status": "measured", "backend": "python_list_copy_pop",
        "width": width, "height": height, "format": "rgb8", "history_length": capacity,
        "allocated_bytes": capacity * frame.nbytes, "frame_bytes": frame.nbytes,
        "rss_before_bytes": memory_before["rss_bytes"], "rss_after_bytes": _process_memory()["rss_bytes"],
        **result,
    }


def _opencv_mix_case(width, height, capacity, warmup, iterations):
    """Existing optimized CPU blend primitive, excluding history storage."""
    current = _frame(width, height, 17)
    historical = _frame(width, height, 3)
    output = np.empty_like(current)
    result = _measure(
        lambda: cv2.addWeighted(current, 0.65, historical, 0.35, 0.0, dst=output),
        warmup, iterations,
    )
    return {
        "name": "temporal_mix", "status": "measured", "backend": "opencv_cpu_reference",
        "width": width, "height": height, "format": "rgb8", "history_length": capacity,
        "allocated_bytes": 0, "frame_bytes": current.nbytes,
        "scope": "existing optimized blend primitive; history storage excluded", **result,
    }


def _ring_cases(backend, width, height, capacity, warmup, iterations):
    frame = _frame(width, height)
    current = _frame(width, height, 17)
    ring = TemporalFrameRing(backend)
    ring.configure(width, height, capacity)
    for index in range(capacity):
        ring.write_frame(_frame(width, height, index))
    output = np.empty_like(frame)
    memory_before = _process_memory()
    write_result = _measure(lambda: ring.write_frame(frame), warmup, iterations)
    mix_result = _measure(
        lambda: ring.process_temporal_mix(current, min(capacity - 1, 7), 0.35, output),
        warmup, iterations,
    )
    stats = ring.get_statistics()
    ring.shutdown()
    common = {
        "status": "measured", "backend": stats["backend"], "width": width, "height": height,
        "format": "rgb8", "history_length": capacity, "allocated_bytes": stats["allocated_bytes"],
        "frame_bytes": frame.nbytes, "rss_before_bytes": memory_before["rss_bytes"],
        "rss_after_bytes": _process_memory()["rss_bytes"],
    }
    return [
        {"name": "temporal_write", **common, **write_result},
        {"name": "temporal_mix", **common, **mix_result},
    ]


def run(warmup: int, iterations: int, quick: bool):
    resolutions = ((320, 180), (1200, 800)) if quick else RESOLUTIONS
    capacities = (1, 8, 64) if quick else CAPACITIES
    cases = []
    for width, height in resolutions:
        for capacity in capacities:
            cases.append(_list_case(width, height, capacity, warmup, iterations))
            cases.append(_opencv_mix_case(width, height, capacity, warmup, iterations))
            for backend in ("python", "native"):
                if backend == "native" and not native_available():
                    cases.append({
                        "name": "temporal_prototype", "status": "unsupported", "backend": "native_cpp",
                        "width": width, "height": height, "format": "rgb8", "history_length": capacity,
                        "reason": "native extension is not built",
                    })
                    continue
                cases.extend(_ring_cases(backend, width, height, capacity, warmup, iterations))
    return {
        "schema_version": 1,
        "benchmark": "megaglitch-native-temporal-prototype",
        "configuration": {"resolutions": resolutions, "history_lengths": capacities,
                          "warmup_frames": warmup, "measurement_frames": iterations},
        "native_available": native_available(),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--label", default="temporal-prototype")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/benchmark/results"))
    args = parser.parse_args()
    result = run(args.warmup, args.iterations, args.quick)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.label}.json"
    csv_path = args.output_dir / f"{args.label}.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = sorted({key for case in result["cases"] for key in case})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["cases"])
    print(f"Wrote {json_path} and {csv_path}")
    print(f"Cases: {sum(c['status'] == 'measured' for c in result['cases'])} measured, "
          f"{sum(c['status'] == 'unsupported' for c in result['cases'])} unsupported")


if __name__ == "__main__":
    main()
