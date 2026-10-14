"""Compare two prebuilt UI executables with alternating, identical smoke runs."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    launcher = Path(__file__).with_name("run_mixed_ui_smoke.py")
    rows = []
    # Reverse the middle pair to reduce a simple warm-up/order bias.
    for index, variant in enumerate(("baseline", "candidate", "candidate", "baseline", "baseline", "candidate"), 1):
        report = output / f"{index:02d}-{variant}.json"
        command = [sys.executable, str(launcher), "--source", str(args.source.resolve()),
                   "--python", sys.executable, "--exe", str(getattr(args, variant).resolve()),
                   "--report", str(report)]
        with report.with_suffix(".launcher.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(report.read_text(encoding="utf-8"))
        row = {"variant": variant, "report": report.name}
        for key in ("passed", "preview_present_fps", "preview_render_fps",
                    "source_submit_to_present_ms_mean", "parameter_to_preview_ms",
                    "texture_prepare_to_wgpu_marker_ms_mean", "preview_width", "preview_height",
                    "presented_new_frames", "stage_timing_samples"):
            row[key] = result[key]
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"run_order": rows, "medians": {}}
    for variant in ("baseline", "candidate"):
        selected = [row for row in rows if row["variant"] == variant]
        summary["medians"][variant] = {
            key: statistics.median(row[key] for row in selected)
            for key in ("preview_present_fps", "source_submit_to_present_ms_mean",
                        "parameter_to_preview_ms", "texture_prepare_to_wgpu_marker_ms_mean")
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["medians"], indent=2))


if __name__ == "__main__":
    main()
