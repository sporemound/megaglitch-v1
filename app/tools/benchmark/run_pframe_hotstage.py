"""Measure the real saved-preset P-frame hot stage and its exact native replacement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import glitch_effects as ge
from megaglitch.native_temporal import amplify_prediction_errors_uniform, native_available
from tools.benchmark.run_baseline import _base_frame, _summarize_times


INTENSITY = 0.978
ERROR_THRESHOLD = 30
MOTION_INTENSITY = 0.451
RANDOM_SEED = 1984


def _python_uniform(current, reference):
    """Frozen copy of the pre-optimization uniform implementation."""
    current_f = np.array(current, dtype=np.float32)
    reference_f = np.array(reference, dtype=np.float32)
    residual = current_f - reference_f
    error_mask = np.abs(residual) > ERROR_THRESHOLD
    amplification_mask = np.ones(current.shape[:2])
    amplification_factor = 1 + INTENSITY * 3 * amplification_mask[..., np.newaxis]
    result = reference_f + np.where(
        error_mask, residual * amplification_factor, residual
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def _measure(name, function, warmup, iterations, expected=None):
    output = None
    for _ in range(warmup):
        output = function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = function()
        samples.append((time.perf_counter() - started) * 1000.0)
    result = {
        "name": name,
        "status": "measured",
        "output_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(output))).hexdigest(),
        **_summarize_times(samples),
    }
    if expected is not None:
        delta = np.abs(output.astype(np.int16) - expected.astype(np.int16))
        result["pixel_exact"] = bool(np.array_equal(output, expected))
        result["max_channel_delta"] = int(delta.max(initial=0))
    return result


def _resolution(width, height, warmup, iterations):
    current = _base_frame(np, width, height)
    reference = np.roll(current, shift=(7, -11), axis=(0, 1)).copy()
    expected = _python_uniform(current, reference)
    native_output = np.empty_like(current)

    def original_chain():
        random.seed(RANDOM_SEED)
        motion = ge.corrupt_motion_vectors(
            current, reference, MOTION_INTENSITY, block_size=16, direction="spiral"
        )
        return _python_uniform(motion, reference)

    def split_native_chain():
        random.seed(RANDOM_SEED)
        motion = ge.corrupt_motion_vectors(
            current, reference, MOTION_INTENSITY, block_size=16, direction="spiral"
        )
        return amplify_prediction_errors_uniform(
            motion, reference, INTENSITY, ERROR_THRESHOLD, native_output
        )

    def fused_native_chain():
        random.seed(RANDOM_SEED)
        output = current.copy()
        return ge.process_pframe_spiral_uniform(
            output, reference, MOTION_INTENSITY, 16, INTENSITY,
            ERROR_THRESHOLD, output=output,
        )

    original_chain_output = original_chain()
    cases = [
        _measure("prediction_uniform_python", lambda: _python_uniform(current, reference), warmup, iterations),
    ]
    if native_available():
        cases.append(_measure(
            "prediction_uniform_native_preallocated",
            lambda: amplify_prediction_errors_uniform(
                current, reference, INTENSITY, ERROR_THRESHOLD, native_output
            ),
            warmup,
            iterations,
            expected,
        ))
    else:
        cases.append({"name": "prediction_uniform_native_preallocated", "status": "unsupported"})
    cases.append(_measure(
        "prediction_uniform_dispatched",
        lambda: ge.amplify_prediction_errors(
            current, reference, INTENSITY, ERROR_THRESHOLD, "uniform"
        ),
        warmup,
        iterations,
        expected,
    ))
    cases.append(_measure(
        "motion_spiral",
        lambda: ge.corrupt_motion_vectors(
            current, reference, MOTION_INTENSITY, block_size=16, direction="spiral"
        ),
        warmup,
        iterations,
    ))
    cases.append(_measure(
        "pframe_chain_original_numpy",
        original_chain,
        warmup,
        iterations,
    ))
    if native_available():
        cases.append(_measure(
            "pframe_chain_split_native_prediction",
            split_native_chain,
            warmup,
            iterations,
            original_chain_output,
        ))
        cases.append(_measure(
            "pframe_chain_fused_native",
            fused_native_chain,
            warmup,
            iterations,
            original_chain_output,
        ))
    for case in cases:
        case.update({"width": width, "height": height, "format": "rgb8"})
    return cases


def run(warmup, iterations):
    cases = []
    for width, height in ((1200, 800), (1920, 1080)):
        cases.extend(_resolution(width, height, warmup, iterations))
    return {
        "schema_version": 1,
        "benchmark": "megaglitch-pframe-hot-stage",
        "saved_preset_parameters": {
            "motion_vector_corruption": MOTION_INTENSITY,
            "motion_vector_direction": "spiral",
            "prediction_error_amplify": INTENSITY,
            "prediction_error_direction": "uniform",
            "error_threshold": ERROR_THRESHOLD,
            "random_seed": RANDOM_SEED,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--label", default="pframe-hotstage")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/benchmark/results"))
    args = parser.parse_args()
    result = run(args.warmup, args.iterations)
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
    print(f"Cases: {len(result['cases'])}")


if __name__ == "__main__":
    main()
