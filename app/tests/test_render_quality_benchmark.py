from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from megaglitch.offline_master import FpsSemantics
from tools.benchmark.run_render_quality_path import (
    parse_resolution,
    run_benchmark,
    write_results,
)


def test_resolution_parser_requires_explicit_positive_dimensions():
    assert parse_resolution("64x48") == (64, 48)
    assert parse_resolution(" 80 X 60 ") == (80, 60)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_resolution("640")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_resolution("0x48")


def test_quality_benchmark_exercises_master_graph_and_offline_scheduler():
    result = run_benchmark(
        width=48,
        height=32,
        frame_count=3,
        source_fps="30",
        output_fps="55",
        semantics=FpsSemantics.TRUE_ACCELERATED,
        seed=41,
    )

    assert result["schema_version"] == 1
    assert result["benchmark"] == "megaglitch-render-quality-path"
    assert result["configuration"]["opencv_determinism"]["threads"] == 1
    assert result["configuration"]["opencv_determinism"]["opencl"] is False
    telemetry = result["telemetry"]
    assert telemetry["resolution"] == [48, 32]
    assert telemetry["source_fps"] == 30.0
    assert telemetry["output_fps"] == 55.0
    assert telemetry["frame_count"] == {
        "source": 3,
        "master_processed": 3,
        "encoded": 3,
    }
    assert telemetry["duration"]["output_seconds"] == pytest.approx(3 / 55)
    assert telemetry["fps_exact"] == {"source": "30/1", "output": "55/1"}
    assert telemetry["duration_exact"] == {
        "source": "1/10",
        "output": "3/55",
    }
    assert telemetry["dropped_frames"] == 0
    assert telemetry["duplicated_frames"] == 0
    assert telemetry["bitrate"] is None
    assert telemetry["benchmark_sink"]["bitrate_measured"] is False
    assert len(telemetry["graph_hash"]) == 64
    assert telemetry["random_seed"] == 41
    assert telemetry["render_time_per_frame"]["samples"] == 3
    assert telemetry["encode_time_per_frame"]["stdin_write"]["samples"] == 3
    assert telemetry["node_provider_assignment"]
    assert "master_deck_blend" in telemetry["node_provider_assignment"]
    assert (
        telemetry["node_provider_assignment"]["OpticalFlowRemapNode:1"] == "cpu"
    )
    assert "count" in telemetry["cpu_gpu_transfers"]
    assert "bytes" in telemetry["cpu_gpu_transfers"]
    assert telemetry["history_buffer_inventory"]["deck_a"]
    optical_flow_records = [
        record
        for path, record in telemetry["history_buffer_inventory"]["deck_a"].items()
        if path.endswith("analysis.optical_flow_vector")
    ]
    assert optical_flow_records == [
        {
            "shape": [32, 48, 2],
            "resolution": [48, 32],
            "dtype": "float32",
            "provider": "cpu",
            "master_resolution": True,
        }
    ]
    assert telemetry["correctness_gates"] == {
        "all_source_frames_processed_once": True,
        "no_runtime_dropped_frames": True,
        "all_measured_history_at_master_resolution": True,
        "fps_schedule_matches_exact_rational_model": True,
        "master_output_shape": [48, 32],
        "preview_obs_vcam_reentry": False,
    }
    assert all(
        resolution == [48, 32]
        for resolution in telemetry["history_buffer_resolution"].values()
    )


def test_quality_benchmark_json_and_csv_keep_required_fields(tmp_path):
    result = run_benchmark(
        width=32,
        height=24,
        frame_count=2,
        source_fps="24",
        output_fps="48",
        semantics=FpsSemantics.DURATION_PRESERVING,
        seed=9,
    )
    json_path = tmp_path / "quality.json"
    csv_path = tmp_path / "quality.csv"
    write_results(result, json_output=json_path, csv_output=csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["telemetry"]["frame_count"]["encoded"] == 4
    assert payload["telemetry"]["frame_count"]["master_processed"] == 2
    assert payload["telemetry"]["duplicated_frames"] == 2
    assert payload["telemetry"]["fps_semantics"] == "duration_preserving"
    assert payload["telemetry"]["fps_exact"] == {
        "source": "24/1",
        "output": "48/1",
    }
    assert payload["telemetry"]["duration_exact"] == {
        "source": "1/12",
        "output": "1/12",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["telemetry.resolution"] == "[32,24]"
    assert rows[0]["telemetry.frame_count.master_processed"] == "2"
    assert rows[0]["telemetry.bitrate"] == ""
    assert rows[0]["telemetry.graph_hash"] == payload["telemetry"]["graph_hash"]


def test_quality_benchmark_repeats_master_output_for_same_seed():
    options = {
        "width": 32,
        "height": 24,
        "frame_count": 2,
        "source_fps": "30",
        "output_fps": "55",
        "semantics": FpsSemantics.TRUE_ACCELERATED,
        "seed": 73,
    }
    first = run_benchmark(**options)
    second = run_benchmark(**options)

    assert (
        first["telemetry"]["master_output_sha256"]
        == second["telemetry"]["master_output_sha256"]
    )
    assert first["telemetry"]["graph_hash"] == second["telemetry"]["graph_hash"]


def _quality_hash_from_subprocess(python_hash_seed: int) -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[1]
    program = "\n".join(
        (
            "import json",
            "from tools.benchmark.run_render_quality_path import run_benchmark",
            "result = run_benchmark(width=32, height=24, frame_count=2, "
            "source_fps='30000/1001', output_fps='55', "
            "semantics='true_accelerated', seed=703)",
            "print(json.dumps({",
            "    'graph_hash': result['telemetry']['graph_hash'],",
            "    'master_output_sha256': result['telemetry']['master_output_sha256'],",
            "    'source_fps_exact': result['telemetry']['fps_exact']['source'],",
            "    'output_duration_exact': result['telemetry']['duration_exact']['output'],",
            "}, sort_keys=True))",
        )
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(python_hash_seed)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_quality_benchmark_is_cross_process_deterministic_across_hash_seeds():
    first = _quality_hash_from_subprocess(1)
    same_seed_repeat = _quality_hash_from_subprocess(1)
    second = _quality_hash_from_subprocess(8675309)

    assert first == same_seed_repeat
    assert first == second
    assert first["source_fps_exact"] == "30000/1001"
    assert first["output_duration_exact"] == "2/55"
