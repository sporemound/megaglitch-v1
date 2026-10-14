"""Compare legacy pickled frames with fixed shared-memory worker transport."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from megaglitch.shared_frame_transport import SharedFrameTransportOwner
from tools.benchmark.run_baseline import (
    _base_frame,
    _representative_pipeline,
    _summarize_times,
)


def _case(transport, pipeline, graph_name, width, height, warmup, iterations,
          source_copy_policy="copy", deterministic_seed=None):
    from megaglitchfm_app_new import glitch_process_worker

    frame = _base_frame(np, width, height)
    context = mp.get_context("spawn")
    input_queue = context.Queue(maxsize=2)
    output_queue = context.Queue(maxsize=2)
    stop_event = context.Event()
    owner = None
    spec = None
    if transport == "shared":
        owner = SharedFrameTransportOwner(
            context, width, height, channels=3, input_slots=4, output_slots=2
        )
        spec = owner.spec()
    process = context.Process(
        target=glitch_process_worker,
        args=(input_queue, output_queue, stop_event, None, spec),
    )
    process.start()

    def payload(index):
        item = {
            "image": frame,
            "secondary_image": None,
            "pipeline": pipeline,
            "pipeline_b": [],
            "master_mix": 0.0,
            "master_transition": "Normal Mix",
            "extra_params": {"time_accum": index / 60.0, "frame_num": index},
            "use_cuda": False,
            "fallback_width": width,
            "fallback_height": height,
            "timestamp": time.perf_counter(),
            "source_copy_policy": source_copy_policy,
        }
        if deterministic_seed is not None:
            item["_benchmark_seed"] = int(deterministic_seed) + int(index)
        if owner is not None:
            encoded = owner.encode_payload(item, block=True, timeout=30.0)
            if encoded is None:
                raise RuntimeError("shared input pool exhausted during sequential benchmark")
            return encoded
        return item

    def roundtrip(index):
        item = payload(index)
        input_queue.put(item, timeout=30)
        result = output_queue.get(timeout=120)
        return owner.decode_result(result) if owner is not None else result

    output = None
    times = []
    try:
        for index in range(warmup):
            output = roundtrip(index)
        for index in range(iterations):
            started = time.perf_counter()
            output = roundtrip(warmup + index)
            times.append((time.perf_counter() - started) * 1000.0)
    finally:
        stop_event.set()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        input_queue.close()
        output_queue.close()

    output_frame = output.get("frame") if isinstance(output, dict) else output
    sample_payload = {
        "image": frame if transport == "legacy" else {"shared_descriptor": True},
        "pipeline": pipeline,
    }
    transport_stats = owner.get_statistics() if owner is not None else {}
    frame_copy_stats = output.get("frame_copy_stats", {}) if isinstance(output, dict) else {}
    flattened_copy_stats = {}
    for category, values in sorted(frame_copy_stats.items()):
        if not isinstance(values, dict):
            continue
        flattened_copy_stats[f"copy_{category}_count"] = int(values.get("count", 0))
        flattened_copy_stats[f"copy_{category}_bytes"] = int(values.get("bytes", 0))
    explicit_copy_count = sum(
        int(values.get("count", 0)) for values in frame_copy_stats.values()
        if isinstance(values, dict)
    )
    explicit_copy_bytes = sum(
        int(values.get("bytes", 0)) for values in frame_copy_stats.values()
        if isinstance(values, dict)
    )
    if owner is not None:
        owner.close()
    return {
        "name": f"worker_{graph_name}",
        "status": "measured",
        "transport": transport,
        "result_contract": "canonical_frame" if transport == "shared" else "frame_plus_deck_a_plus_deck_b",
        "source_copy_policy": source_copy_policy,
        "width": width,
        "height": height,
        "format": "rgb8",
        "warmup_frames": warmup,
        "measurement_frames": len(times),
        "serialized_sample_bytes": len(pickle.dumps(sample_payload, protocol=pickle.HIGHEST_PROTOCOL)),
        "shared_allocated_bytes": transport_stats.get("allocated_bytes", 0),
        "shared_input_frames_written": transport_stats.get("input_frames_written", 0),
        "shared_output_frames_read": transport_stats.get("output_frames_read", 0),
        "shared_input_slots_in_use_after_run": transport_stats.get("input_slots_in_use", 0),
        "shared_output_slots_in_use_after_run": transport_stats.get("output_slots_in_use", 0),
        "shared_input_copy_count_total": transport_stats.get("input_copy_count", 0),
        "shared_input_copy_bytes_total": transport_stats.get("input_copy_bytes", 0),
        "ui_consumer_copy_count_total": transport_stats.get("ui_consumer_copy_count", 0),
        "ui_consumer_copy_bytes_total": transport_stats.get("ui_consumer_copy_bytes", 0),
        "frame_copy_stats": frame_copy_stats,
        "explicit_copy_count_per_result": explicit_copy_count,
        "explicit_copy_bytes_per_result": explicit_copy_bytes,
        "output_has_frame_a": isinstance(output, dict) and output.get("frame_a") is not None,
        "output_has_frame_b": isinstance(output, dict) and output.get("frame_b") is not None,
        "output_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(output_frame))).hexdigest(),
        "dropped_frames": 0,
        **flattened_copy_stats,
        **_summarize_times(times),
    }


def run(warmup, iterations, quick):
    representative, preset = _representative_pipeline()
    cases = []
    for transport in ("legacy", "shared"):
        cases.append(_case(
            transport, [], "minimal_graph", 1200, 800,
            min(warmup, 3 if quick else warmup), min(iterations, 5 if quick else iterations),
        ))
        cases.append(_case(
            transport, representative, "representative_pframe", 1200, 800,
            min(warmup, 2), min(iterations, 3 if quick else 15),
        ))
    return {
        "schema_version": 2,
        "benchmark": "megaglitch-shared-frame-transport",
        "representative_preset": preset,
        "configuration": {
            "resolution": [1200, 800], "warmup_frames": warmup,
            "measurement_frames": iterations,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--label", default="shared-transport")
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
    print(f"Cases: {len(result['cases'])} measured")


if __name__ == "__main__":
    main()
