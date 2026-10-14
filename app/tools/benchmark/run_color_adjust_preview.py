"""Bounded CPU preview measurement for the existing Color Adjust workflow.

This measures executor throughput, not GUI presentation or full application FPS.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark.run_preview_executor import (
    _opencv_metadata,
    deterministic_source,
    run_cell,
)
import gpu_ops


def color_adjust_pipeline(*, feedback=False, temporal_first=False):
    pipeline = [
        {"effect_type": "Source", "node_type": "source", "source_key": "primary", "params": {}},
        {
            "effect_type": "Color Adjust", "node_type": "effect",
            "node_id": "ColorAdjustNode:benchmark",
            "params": {"enabled": True, "hue_shift": 30, "saturation": 0.5, "gamma": 1.2},
        },
        {"effect_type": "Display", "node_type": "display", "params": {}},
    ]
    if feedback:
        pipeline.insert(1 if temporal_first else 2, {
            "effect_type": "Feedback / Trail", "node_type": "effect",
            "node_id": "FeedbackTrailNode:benchmark",
            "params": {"enabled": True, "feedback_intensity": 0.25, "trailing_intensity": 0.5},
        })
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=("960x540", "640x360"), default="960x540")
    parser.add_argument("--feedback", action="store_true", help="Include existing Feedback / Trail")
    parser.add_argument("--temporal-first", action="store_true", help="Place Feedback / Trail before Color Adjust")
    args = parser.parse_args()
    if args.temporal_first and not args.feedback:
        parser.error("--temporal-first requires --feedback")
    tier = tuple(int(part) for part in args.tier.split("x"))
    if gpu_ops.get_use_cuda():
        raise RuntimeError("this CPU measurement requires gpu_ops CUDA execution disabled")
    pipeline = color_adjust_pipeline(feedback=args.feedback, temporal_first=args.temporal_first)
    result = {
        "label": args.label,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_changes": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "opencv": _opencv_metadata(),
        "gpu_ops_cuda_enabled": gpu_ops.get_use_cuda(),
        "source": "1920x1080 synthetic RGB gradient, changing sequence marker; no media decoder",
        "pipeline": pipeline,
        "limitations": [
            "Executor throughput only; no GUI presentation, scanout, devices, or package certification.",
            "Before-fix coverage may bypass Color Adjust: inspect node_statuses before comparing timing.",
            "No RAM/VRAM, allocations, copies/transfers, or end-to-end UI latency instrumentation.",
            f"Fixed {args.tier} preview, adaptive control disabled, serial requests without producer overload.",
        ],
        "measurement": run_cell(
            workload="color_adjust_feedback" if args.feedback else "color_adjust",
            pipeline=pipeline, preset_identity=None,
            tier=tier, source_frame=deterministic_source(1920, 1080),
            warmup=30, iterations=300, timeout=5.0,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["measurement"], indent=2))


if __name__ == "__main__":
    main()
