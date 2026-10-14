"""Measure processing-worker readiness, first frame, and ordered shutdown cycles."""

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

from megaglitch.shared_frame_transport import SharedFrameTransportOwner
from megaglitch.worker_lifecycle import WorkerProcessLifecycle
from tools.benchmark.run_baseline import _base_frame


def _cycle(index, width, height):
    from megaglitchfm_app_new import glitch_process_worker

    context = mp.get_context("spawn")
    stop_event = context.Event()
    raw_queue = context.Queue(maxsize=2)
    output_queue = context.Queue(maxsize=2)
    owner = SharedFrameTransportOwner(
        context, width, height, input_slots=4, output_slots=2
    )
    lifecycle = WorkerProcessLifecycle(
        context, stop_event, startup_timeout=30.0,
        graceful_timeout=5.0, terminate_timeout=2.0,
    )
    output = None
    try:
        started = time.perf_counter()
        lifecycle.start(
            glitch_process_worker,
            (raw_queue, output_queue, stop_event, None, owner.spec()),
        )
        readiness_ms = (time.perf_counter() - started) * 1000.0
        frame = _base_frame(np, width, height)
        payload = owner.encode_payload({
            "image": frame,
            "secondary_image": None,
            "pipeline": [],
            "pipeline_b": [],
            "master_mix": 0.0,
            "extra_params": {},
            "use_cuda": False,
            "fallback_width": width,
            "fallback_height": height,
            "timestamp": 0.0,
        }, block=True, timeout=10.0)
        frame_started = time.perf_counter()
        raw_queue.put(payload, timeout=10.0)
        output = owner.decode_result(output_queue.get(timeout=30.0))["frame"]
        first_frame_ms = (time.perf_counter() - frame_started) * 1000.0
        shutdown_started = time.perf_counter()
        forced = lifecycle.stop()
        shutdown_ms = (time.perf_counter() - shutdown_started) * 1000.0
        stats = owner.get_statistics()
        return {
            "cycle": index,
            "width": width,
            "height": height,
            "readiness_ms": readiness_ms,
            "first_frame_ms": first_frame_ms,
            "shutdown_ms": shutdown_ms,
            "forced_termination": forced,
            "exitcode": lifecycle.process.exitcode,
            "final_state": lifecycle.state.value,
            "input_slots_in_use": stats["input_slots_in_use"],
            "output_slots_in_use": stats["output_slots_in_use"],
            "output_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(output))).hexdigest(),
            "transitions": ">".join(item["state"] for item in lifecycle.transitions),
        }
    finally:
        if lifecycle.state.value != "stopped":
            lifecycle.stop()
        raw_queue.close()
        output_queue.close()
        owner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--label", default="worker-lifecycle")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/benchmark/results"))
    args = parser.parse_args()
    cases = [_cycle(index + 1, args.width, args.height) for index in range(args.cycles)]
    result = {
        "schema_version": 1,
        "benchmark": "megaglitch-worker-lifecycle",
        "configuration": {
            "cycles": args.cycles,
            "resolution": [args.width, args.height],
            "transport": "shared_v1",
        },
        "cases": cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.label}.json"
    csv_path = args.output_dir / f"{args.label}.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(f"Wrote {json_path} and {csv_path}")
    for case in cases:
        print(
            f"cycle {case['cycle']}: ready={case['readiness_ms']:.1f}ms "
            f"first={case['first_frame_ms']:.1f}ms stop={case['shutdown_ms']:.1f}ms "
            f"forced={case['forced_termination']}"
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
