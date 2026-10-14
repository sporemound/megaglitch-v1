"""Benchmark the Python-hosted Rust runtime vertical slice."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from megaglitch.native_runtime import RuntimeConfig, create_runtime, native_runtime_available
from tools.benchmark.run_baseline import (
    _base_frame,
    _process_memory,
    _run_worker_roundtrip_case,
    _summarize_times,
)


RESOLUTIONS = ((1280, 720), (1920, 1080), (1200, 800))


def _case(backend, mode, width, height, warmup, iterations):
    frame = _base_frame(np, width, height)
    output = np.empty_like(frame)
    engine = create_runtime(RuntimeConfig(width, height, 3, 8, 2, mode), backend)
    memory_before = _process_memory()
    engine.start()
    for _ in range(warmup):
        if engine.submit_frame(frame):
            engine.wait_for_frame(output)
    times = []
    poll_calls = []
    for _ in range(iterations):
        started = time.perf_counter()
        accepted = engine.submit_frame(frame)
        polls = 0
        if accepted:
            if backend == "native":
                engine.wait_for_frame(output)
                polls = 1
            else:
                while True:
                    polls += 1
                    result = engine.poll_frame(output)
                    if result is not None:
                        break
                    time.sleep(0.0001)
        poll_calls.append(polls)
        times.append((time.perf_counter() - started) * 1000.0)
    stats = engine.get_statistics()
    engine.close()
    memory_after = _process_memory()
    return {
        "name": "runtime_roundtrip",
        "status": "measured",
        "backend": stats["backend"],
        "mode": mode,
        "width": width,
        "height": height,
        "format": "rgb8",
        "history_length": 8,
        "queue_capacity": 2,
        "cross_language_submit_calls_per_frame": 1 if backend == "native" else 0,
        "mean_poll_calls_per_frame": sum(poll_calls) / len(poll_calls),
        "allocated_bytes": stats["allocated_bytes"],
        "submitted": stats["submitted"],
        "processed": stats["processed"],
        "delivered": stats["delivered"],
        "input_dropped": stats["input_dropped"],
        "output_dropped": stats["output_dropped"],
        "rss_before_bytes": memory_before["rss_bytes"],
        "rss_after_bytes": memory_after["rss_bytes"],
        **_summarize_times(times),
    }


def _native_worker_case(width, height, warmup, iterations, pipeline=None, label="minimal_graph"):
    """Measure the user-facing mp.Queue boundary with the Rust worker selected."""
    from megaglitch.native_runtime_worker import native_runtime_process_worker
    from megaglitchfm_app_new import glitch_process_worker

    frame = _base_frame(np, width, height)
    context = mp.get_context("spawn")
    input_queue = context.Queue(maxsize=2)
    output_queue = context.Queue(maxsize=2)
    stop_event = context.Event()
    process = context.Process(
        target=native_runtime_process_worker,
        args=(input_queue, output_queue, stop_event, None, glitch_process_worker),
    )
    process.start()

    def payload(index):
        return {
            "image": frame,
            "secondary_image": None,
            "pipeline": list(pipeline or []),
            "pipeline_b": [],
            "master_mix": 0.0,
            "master_transition": "Normal Mix",
            "extra_params": {"time_accum": index / 60.0, "frame_num": index},
            "use_cuda": False,
            "fallback_width": width,
            "fallback_height": height,
            "timestamp": time.perf_counter(),
        }

    times = []
    last = None
    try:
        for index in range(warmup):
            input_queue.put(payload(index), timeout=30)
            last = output_queue.get(timeout=60)
        for index in range(iterations):
            started = time.perf_counter()
            input_queue.put(payload(warmup + index), timeout=30)
            last = output_queue.get(timeout=60)
            times.append((time.perf_counter() - started) * 1000.0)
    finally:
        stop_event.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        input_queue.close()
        output_queue.close()
    return {
        "name": f"worker_roundtrip_native_{label}",
        "status": "measured",
        "backend": last.get("runtime_backend") if isinstance(last, dict) else "unknown",
        "mode": label if pipeline else "passthrough",
        "width": width,
        "height": height,
        "format": "rgb8",
        "scope": "mp.Queue input, Rust pooled runtime, mp.Queue output; capture/preview excluded",
        "output_contract": "frame/frame_a/frame_b alias the same native result before serialization",
        "output_sha256": hashlib.sha256(np.asarray(last["frame"]).tobytes()).hexdigest(),
        **_summarize_times(times),
    }


def run(warmup, iterations, quick):
    resolutions = ((320, 180), (1200, 800)) if quick else RESOLUTIONS
    cases = []
    for width, height in resolutions:
        for mode in ("passthrough", "temporal_mix", "invert"):
            for backend in ("python", "native"):
                if backend == "native" and not native_runtime_available():
                    cases.append({
                        "name": "runtime_roundtrip", "status": "unsupported",
                        "backend": "native_rust", "mode": mode, "width": width, "height": height,
                        "reason": "Rust runtime DLL is not built",
                    })
                else:
                    cases.append(_case(backend, mode, width, height, warmup, iterations))
    invert_pixel_exact = None
    graph_ir_pixel_exact = None
    if not quick and native_runtime_available():
        python_worker = _run_worker_roundtrip_case(
            np, [], 1200, 800, min(warmup, 5), min(iterations, 30),
            "worker_roundtrip_python_minimal_graph",
        )
        python_worker["backend"] = "python_process"
        cases.append(python_worker)
        cases.append(_native_worker_case(1200, 800, min(warmup, 5), min(iterations, 30)))
        invert_pipeline = [
            {"effect_type": "Source", "node_type": "source", "source_key": "primary"},
            {"effect_type": "Invert Colors", "params": {"invert_chance": 100}},
            {"effect_type": "Display", "node_type": "display", "params": {}},
        ]
        python_invert = _run_worker_roundtrip_case(
            np, invert_pipeline, 1200, 800, min(warmup, 5), min(iterations, 30),
            "worker_roundtrip_python_invert",
        )
        python_invert["backend"] = "python_process"
        python_invert["mode"] = "invert"
        cases.append(python_invert)
        cases.append(_native_worker_case(
            1200, 800, min(warmup, 5), min(iterations, 30),
            pipeline=invert_pipeline, label="invert",
        ))
        invert_cases = [case for case in cases if case["name"] in {
            "worker_roundtrip_python_invert", "worker_roundtrip_native_invert"
        }]
        invert_pixel_exact = (
            len(invert_cases) == 2
            and len({case.get("output_sha256") for case in invert_cases}) == 1
        )
        graph_pipeline = [
            {"effect_type": "Source", "node_type": "source", "source_key": "primary"},
            {"effect_type": "Brightness / Contrast", "params": {
                "brightness": 17, "contrast": -25,
            }},
            {"effect_type": "Invert Colors", "params": {"invert_chance": 100}},
            {"effect_type": "Display", "node_type": "display", "params": {}},
        ]
        python_graph = _run_worker_roundtrip_case(
            np, graph_pipeline, 1200, 800, min(warmup, 5), min(iterations, 30),
            "worker_roundtrip_python_graph_ir_v1",
        )
        python_graph["backend"] = "python_process"
        python_graph["mode"] = "graph_ir_v1"
        cases.append(python_graph)
        cases.append(_native_worker_case(
            1200, 800, min(warmup, 5), min(iterations, 30),
            pipeline=graph_pipeline, label="graph_ir_v1",
        ))
        graph_cases = [case for case in cases if case["name"] in {
            "worker_roundtrip_python_graph_ir_v1",
            "worker_roundtrip_native_graph_ir_v1",
        }]
        graph_ir_pixel_exact = (
            len(graph_cases) == 2
            and len({case.get("output_sha256") for case in graph_cases}) == 1
        )
    return {
        "schema_version": 1,
        "benchmark": "megaglitch-python-hosted-native-runtime",
        "configuration": {
            "resolutions": resolutions, "modes": ["passthrough", "temporal_mix", "invert"],
            "warmup_frames": warmup, "measurement_frames": iterations,
        },
        "native_available": native_runtime_available(),
        "invert_worker_pixel_exact": invert_pixel_exact,
        "graph_ir_worker_pixel_exact": graph_ir_pixel_exact,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--label", default="native-runtime")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/benchmark/results"))
    args = parser.parse_args()
    result = run(args.warmup, args.iterations, args.quick)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.label}.json"
    csv_path = args.output_dir / f"{args.label}.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = sorted({key for case in result["cases"] for key in case})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["cases"])
    print(f"Wrote {json_path} and {csv_path}")
    print(f"Cases: {sum(c['status'] == 'measured' for c in result['cases'])} measured, "
          f"{sum(c['status'] == 'unsupported' for c in result['cases'])} unsupported")


if __name__ == "__main__":
    main()
