"""Color controls must affect program output and the Rust UI preview surface."""

import numpy as np
import pytest

from megaglitch.preview_executor import PreviewExecutor, PreviewGraphSnapshot
from tools.rust_ui_bridge import (
    BridgeEngine,
    PreviewSurface,
    compile_preset_graph,
    deterministic_frame,
    process_pipeline,
)


def color_adjust_preset(**params):
    return {
        "version": "1.0",
        "nodes": [{
            "id": "ColorAdjustNode:controls",
            "type": "ColorAdjustNode",
            "label": "Color Adjust",
            "params": {"enabled": True, **params},
        }],
        "links": [
            {
                "from_node": "SOURCE", "from_attr": "image_out",
                "to_node_id": "ColorAdjustNode:controls", "to_attr": "image_in",
            },
            {
                "from_node_id": "ColorAdjustNode:controls", "from_attr": "image_out",
                "to_node": "DISPLAY", "to_attr": "image_in",
            },
        ],
    }


def test_zero_saturation_removes_color_from_program_output():
    frame = deterministic_frame(11, 64, 36)
    original = frame.copy()
    pipeline = compile_preset_graph(color_adjust_preset(saturation=0.0))
    output, _ = process_pipeline(pipeline, frame)

    # HSV saturation zero preserves value and makes every RGB channel equal.
    expected = np.repeat(frame.max(axis=2, keepdims=True), 3, axis=2)
    np.testing.assert_array_equal(output, expected)
    assert not np.array_equal(output, frame)
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("params,changed", [
    ({}, False),
    ({"saturation": None}, False),
    ({"saturation": 0.0}, True),
    ({"saturation": 0.5}, True),
    ({"hue_shift": 60}, True),
    ({"gamma": 2.0}, True),
    ({"enabled": False, "saturation": 0.0, "gamma": 2.0}, False),
])
def test_color_control_preview_matches_program_at_equal_resolution(params, changed):
    frame = deterministic_frame(11, 64, 36)
    pipeline = compile_preset_graph(color_adjust_preset(**params))
    expected, _ = process_pipeline(pipeline, frame)
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(pipeline, 1), adaptive=False, autostart=False
    )
    executor.submit_latest(frame, 1.0, 1, sequence=1)
    result = executor.render_latest()

    assert result is not None
    np.testing.assert_array_equal(result.frame, expected)
    assert (not np.array_equal(result.frame, frame)) == changed
    status = next(s for s in result.node_statuses if s.node_id == "ColorAdjustNode:controls")
    assert status.status == ("disabled" if params.get("enabled") is False else "implemented")


def test_color_control_revisions_reach_ui_preview_surface():
    frame = deterministic_frame(11, 64, 36)
    with PreviewSurface(generation=43) as surface:
        engine = BridgeEngine(64, 36, surface=surface)
        try:
            for revision, saturation in enumerate((1.0, 0.0, 0.5, 1.0), start=1):
                ack, shutdown = engine.handle_command({
                    "version": 2, "id": f"color-{revision}",
                    "expected_state_revision": revision - 1,
                    "type": "replace_graph",
                    "graph": color_adjust_preset(saturation=saturation),
                })
                assert ack["accepted"] and not shutdown
                assert ack["graph_generation"] == revision
                engine.preview_executor.submit_latest(frame, float(revision), revision, sequence=revision)
                assert engine.preview_executor.render_latest() is not None
                events = engine.poll_decoupled_events()
                preview = next(event for event in events if event["type"] == "preview_frame")
                assert preview["graph_revision"] == revision
                pixels = np.frombuffer(
                    surface.read_slot_bytes(preview["slot"], preview["nbytes"]), dtype=np.uint8
                ).reshape(frame.shape)
                expected, _ = process_pipeline(engine.pipeline, frame)
                np.testing.assert_array_equal(pixels, expected)
                if saturation == 0.0:
                    np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
                    np.testing.assert_array_equal(pixels[..., 1], pixels[..., 2])
                assert engine.snapshot()["preview_unsupported_nodes"] == []
                assert surface.release(preview["slot"], preview["generation"], preview["sequence"])
        finally:
            engine.close()


def test_color_adjust_scaled_preview_preserves_source():
    frame = deterministic_frame(11, 1280, 720)
    original = frame.copy()
    pipeline = compile_preset_graph(color_adjust_preset(saturation=0.0))
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(pipeline, 1),
        preferred_tier=(640, 360), adaptive=False, autostart=False,
    )
    executor.submit_latest(frame, 1.0, 1, sequence=1)
    result = executor.render_latest()

    assert result is not None and result.frame.shape == (360, 640, 3)
    np.testing.assert_array_equal(result.frame[..., 0], result.frame[..., 1])
    np.testing.assert_array_equal(result.frame[..., 1], result.frame[..., 2])
    np.testing.assert_array_equal(frame, original)
    assert executor.temporal_state.frame_buffer[-1].shape == result.frame.shape
