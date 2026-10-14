"""Run the existing Rust WGPU smoke mode with a mixed graph and real video."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=ROOT / "native/ui/target/release/megaglitch-ui.exe")
    parser.add_argument("--temporal-first", action="store_true")
    args = parser.parse_args()
    source, report, python, exe = (p.resolve() for p in (args.source, args.report, args.python, args.exe))
    for path in (source, python, exe):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    if report.exists():
        parser.error(f"report already exists: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    nodes = [
        {"id": "ColorAdjustNode:ui", "type": "ColorAdjustNode", "label": "Color Adjust",
         "position": [220, 150], "params": {"enabled": True, "hue_shift": 30, "saturation": 0.5, "gamma": 1.2}},
        {"id": "FeedbackTrailNode:ui", "type": "FeedbackTrailNode", "label": "Feedback / Trail",
         "position": [500, 150], "params": {"enabled": True, "feedback_intensity": 0.25, "trailing_intensity": 0.5}},
    ]
    if args.temporal_first:
        nodes.reverse()
    ends = [{"label": "SOURCE"}, *nodes, {"label": "DISPLAY"}]
    links = []
    for left, right in zip(ends, ends[1:]):
        link = {"from_node": left["label"], "to_node": right["label"], "from_attr": "image_out", "to_attr": "image_in"}
        if "id" in left:
            link["from_node_id"] = left["id"]
        if "id" in right:
            link["to_node_id"] = right["id"]
        links.append(link)
    preset = report.with_suffix(".preset.json")
    preset.write_text(json.dumps({"version": "1.0", "nodes": nodes, "links": links}, indent=2), encoding="utf-8")
    temp = ROOT / ".tmp"
    temp.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.pop("MEGAGLITCH_UI_BRIDGE_EXECUTABLE", None)
    environment.update({
        "MEGAGLITCH_REPOSITORY_ROOT": str(ROOT), "MEGAGLITCH_PYTHON": str(python),
        "MEGAGLITCH_UI_BRIDGE_SCRIPT": str(ROOT / "tools/rust_ui_bridge.py"),
        "MEGAGLITCH_PREVIEW_SMOKE_REPORT": str(report),
        "MEGAGLITCH_PREVIEW_SMOKE_TARGET": "300",
        "MEGAGLITCH_PREVIEW_SMOKE_SOURCE": str(source),
        "MEGAGLITCH_PREVIEW_SMOKE_PRESET": str(preset),
        "MEGAGLITCH_PREVIEW_SMOKE_DECK": "a",
        "MEGAGLITCH_PREVIEW_SMOKE_CASE": report.stem,
        "TEMP": str(temp), "TMP": str(temp),
    })
    started = time.monotonic()
    timed_out = False
    with report.with_suffix(".stdout.log").open("w", encoding="utf-8") as out, report.with_suffix(".stderr.log").open("w", encoding="utf-8") as err:
        process = subprocess.Popen([str(exe)], cwd=ROOT, env=environment, stdout=out, stderr=err)
        try:
            exit_code = process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            exit_code = process.wait(timeout=10)
    metadata = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "executable": str(exe), "executable_sha256": digest(exe),
        "source": str(source), "source_sha256": digest(source),
        "preset_sha256": digest(preset), "python": str(python),
        "production_sha256": {p: digest(ROOT / p) for p in (
            "tools/rust_ui_bridge.py", "megaglitch/preview_executor.py", "effects_registry.py", "glitch_effects.py",
        )},
        "exit_code": exit_code, "timed_out": timed_out,
        "process_wall_seconds": time.monotonic() - started,
    }
    report.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if report.exists():
        result = json.loads(report.read_text(encoding="utf-8"))
        print(json.dumps(result, indent=2))
        passed = result.get("passed") is True
    else:
        passed = False
    raise SystemExit(0 if passed and exit_code == 0 and not timed_out else 1)


if __name__ == "__main__":
    main()
