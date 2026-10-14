import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
except Exception:
    pass

import cv2
from frame_state import FrameState
from megaglitch.master_graph import MasterGraphExecutor
from megaglitch.quality_graph import build_quality_first_graph
from megaglitch.render_contract import MasterResolution
from tools.benchmark.run_render_quality_path import (
    quality_graph_pipeline,
    run_benchmark,
    synthetic_master_frames,
)
from tools.rust_ui_bridge import (
    BridgeEngine,
    PreviewSurface,
    compile_preset_graph,
    downscale_preview_rgb,
    frame_to_rgb8,
    process_pipeline,
)


def invert_preset():
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "InvertColorNode:1",
                "type": "InvertColorNode",
                "label": "Invert Colors",
                "position": [200, 100],
                "params": {"enabled": True, "invert_chance": 100},
            }
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Invert Colors",
                "to_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Invert Colors",
                "to_node": "DISPLAY",
                "from_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def test_decoded_source_establishes_master_dimensions():
    """Gate 4: Source decode dimensions strictly establish master frame dimensions."""
    preset = invert_preset()
    pipeline = compile_preset_graph(preset)

    for width, height in [(1920, 1080), (1280, 720), (800, 600), (320, 240)]:
        src_frame = np.zeros((height, width, 3), dtype=np.uint8)
        src_frame[:, :, 0] = 120
        src_frame[:, :, 1] = 200

        state = FrameState()
        state.bind_master_shape((height, width, 3), allow_resolution_change=True)

        output, state = process_pipeline(pipeline, src_frame, state)
        assert output.shape == (height, width, 3), f"Master output shape must be ({height}, {width}, 3)"
        assert state.master_shape == (height, width, 3)


def test_temporal_buffers_are_strictly_master_resolution():
    """Gate 4: All core and nested temporal buffers match configured master resolution."""
    width, height = 1920, 1080
    resolution = MasterResolution(width=width, height=height)
    executor = MasterGraphExecutor(resolution, use_cuda=False)

    graph = build_quality_first_graph()
    pipeline = quality_graph_pipeline(graph)

    frames = list(synthetic_master_frames(width, height, 5, seed=917))
    for i, frame in enumerate(frames):
        res = executor.process(
            frame,
            pipeline_a=pipeline,
            master_mix=0.0,
            source_index=i,
        )
        assert res.frame.shape == (height, width, 3)

    # Inspect all registered temporal buffers
    history_res = executor.history_buffer_resolution
    assert len(history_res) > 0
    for name, size in history_res.items():
        assert size == (width, height), f"Temporal buffer {name} size {size} != ({width}, {height})"


def test_preview_size_changes_and_preview_disabled_do_not_alter_master_hashes():
    """Gate 4: Preview downscaling is strictly downstream and does NOT affect master output."""
    width, height = 1920, 1080
    frame_count = 30
    seed = 917

    def run_with_preview_mode(preview_mode: str) -> list[str]:
        frames = list(synthetic_master_frames(width, height, frame_count, seed=seed))
        preset = invert_preset()
        pipeline = compile_preset_graph(preset)
        state = FrameState()
        state.bind_master_shape((height, width, 3), allow_resolution_change=True)

        master_hashes = []
        for seq, frame in enumerate(frames, start=1):
            # 1. Master Graph Processing
            output, state = process_pipeline(pipeline, frame, state)
            assert output.shape == (height, width, 3)

            # Master Hash
            master_hash = hashlib.sha256(memoryview(np.ascontiguousarray(output))).hexdigest()
            master_hashes.append(master_hash)

            # 2. Downstream Preview Processing
            if preview_mode == "preview_640x360":
                preview = downscale_preview_rgb(frame_to_rgb8(output))
                assert preview.shape[1] <= 1280 and preview.shape[0] <= 720
            elif preview_mode == "preview_320x180":
                # Smaller custom preview
                preview = cv2.resize(frame_to_rgb8(output), (320, 180), interpolation=cv2.INTER_NEAREST)
                assert preview.shape == (180, 320, 3)
            elif preview_mode == "preview_disabled":
                # No preview generated
                pass

        return master_hashes

    hashes_640 = run_with_preview_mode("preview_640x360")
    hashes_320 = run_with_preview_mode("preview_320x180")
    hashes_off = run_with_preview_mode("preview_disabled")

    assert len(hashes_640) == frame_count
    assert hashes_640 == hashes_320, "Master hashes must be identical regardless of preview resolution"
    assert hashes_640 == hashes_off, "Master hashes must be identical when preview is disabled"


def test_repeated_seed_917_reference_runs_are_deterministic():
    """Gate 4: Repeated reference quality runs produce 100% bit-for-bit identical master output."""
    result1 = run_benchmark(
        width=1920,
        height=1080,
        frame_count=5,
        source_fps="30",
        output_fps="55",
        semantics="true_accelerated",
        seed=917,
        use_cuda=False,
    )
    result2 = run_benchmark(
        width=1920,
        height=1080,
        frame_count=5,
        source_fps="30",
        output_fps="55",
        semantics="true_accelerated",
        seed=917,
        use_cuda=False,
    )

    hash1 = result1["telemetry"]["master_output_sha256"]
    hash2 = result2["telemetry"]["master_output_sha256"]

    assert hash1 != "", "Master output hash must not be empty"
    assert hash1 == hash2, f"Reference runs must produce identical master output hash: {hash1} vs {hash2}"
    assert result1["telemetry"]["correctness_gates"]["all_measured_history_at_master_resolution"] is True


def test_preview_surface_retains_no_upstream_reference():
    """Gate 4: PreviewSurface does not hold or mutate master frame buffers."""
    with PreviewSurface(generation=1) as surface:
        test_master = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_master[100:200, 100:200] = 255
        test_master_copy = test_master.copy()

        event = surface.write_frame(test_master, sequence=1, timestamp=0.0)
        assert event is not None
        assert event["type"] == "preview_frame"
        assert event["width"] <= 1280
        assert event["height"] <= 720

        # Verify input master buffer was not mutated
        np.testing.assert_array_equal(test_master, test_master_copy)
