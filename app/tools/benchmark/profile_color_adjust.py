"""Profile Color Adjust separately from throughput; trace Python/NumPy memory."""

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults

bootstrap_for_dev_defaults()
from megaglitch.preview_executor import PreviewExecutor, PreviewGraphSnapshot
from tools.benchmark.run_preview_executor import deterministic_source, _opencv_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pipeline = [
        {"effect_type": "Source", "node_type": "source", "source_key": "primary", "params": {}},
        {
            "effect_type": "Color Adjust", "node_type": "effect",
            "params": {"enabled": True, "hue_shift": 30, "saturation": 0.5, "gamma": 1.2},
        },
        {"effect_type": "Display", "node_type": "display", "params": {}},
    ]
    frame = deterministic_source(1920, 1080)
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(pipeline, 1),
        preferred_tier=(960, 540), adaptive=False, autostart=False,
    )

    def render(sequence):
        frame[0, 0, 0] = sequence % 256
        executor.submit_latest(frame, sequence / 60, 1, sequence=sequence)
        return executor.render_latest()

    for sequence in range(30):
        render(sequence)
    profile = cProfile.Profile()
    profile.enable()
    for sequence in range(30, 90):
        result = render(sequence)
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
    tracemalloc.start()
    for sequence in range(90, 100):
        render(sequence)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload = {
        "label": args.label,
        "opencv": _opencv_metadata(),
        "profile_60_frames": stream.getvalue(),
        "tracemalloc_10_frames_current_bytes": current,
        "tracemalloc_10_frames_peak_bytes": peak,
        "frame_89_sha256": hashlib.sha256(result.frame).hexdigest(),
        "limitations": "Synchronous executor; traced memory excludes some native RAM and VRAM; profiling overhead is not throughput.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
