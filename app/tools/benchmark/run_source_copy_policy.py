"""Compare copied versus uniquely borrowed source-node ownership."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark.run_baseline import _representative_pipeline
from tools.benchmark.run_shared_transport import _case


def run(warmup: int, iterations: int, seed: int):
    pipeline, preset = _representative_pipeline()
    cases = [
        _case(
            "shared", pipeline, "representative_pframe", 1200, 800,
            warmup, iterations, source_copy_policy=policy, deterministic_seed=seed,
        )
        for policy in ("copy", "borrow_unique")
    ]
    hashes = {case["output_sha256"] for case in cases}
    return {
        "schema_version": 1,
        "benchmark": "megaglitch-source-copy-policy",
        "representative_preset": preset,
        "configuration": {
            "resolution": [1200, 800],
            "format": "rgb8",
            "warmup_frames": warmup,
            "measurement_frames": iterations,
            "deterministic_seed_base": seed,
            "transport": "shared",
        },
        "pixel_exact": len(hashes) == 1,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=406013950)
    parser.add_argument("--label", default="source-copy-policy")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/benchmark/results"))
    args = parser.parse_args()
    result = run(args.warmup, args.iterations, args.seed)
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
    print(f"Pixel exact: {result['pixel_exact']}")
    if not result["pixel_exact"]:
        raise SystemExit("source ownership policies produced different output hashes")


if __name__ == "__main__":
    main()
