import numpy as np
import pytest

from megaglitch.master_graph import (
    MasterGraphExecutor,
    describe_provider_assignments,
)
from megaglitch.render_contract import MasterResolution


def _source_display_pipeline():
    return [
        {"node_type": "source", "effect_type": "Source", "node_id": "source"},
        {"node_type": "display", "effect_type": "Display", "node_id": "display"},
    ]


def test_master_graph_keeps_output_and_all_core_history_at_master_resolution():
    resolution = MasterResolution(width=8, height=6)
    executor = MasterGraphExecutor(resolution)
    frame = np.arange(np.prod(resolution.shape), dtype=np.uint8).reshape(
        resolution.shape
    )

    first = executor.process(frame, pipeline_a=_source_display_pipeline())
    second = executor.process(255 - frame, pipeline_a=_source_display_pipeline())

    assert first.frame.shape == resolution.shape
    assert second.frame.shape == resolution.shape
    assert executor.state_a.previous_frame.shape == resolution.shape
    assert executor.state_a.feedback_image.shape == resolution.shape
    assert executor.state_a.trail_image.shape == resolution.shape
    assert all(item.shape == resolution.shape for item in executor.state_a.frame_buffer)
    assert second.history_depth_a == 2


def test_master_graph_rejects_preview_sized_ingress():
    executor = MasterGraphExecutor(MasterResolution(width=8, height=6))
    with pytest.raises(ValueError, match="does not match master shape"):
        executor.process(
            np.zeros((3, 4, 3), np.uint8),
            pipeline_a=_source_display_pipeline(),
        )


def test_provider_assignment_is_explicit_for_each_deck_and_blend():
    assignments = describe_provider_assignments(
        [
            {"effect_type": "CUDA Flow Warp", "node_id": "flow"},
            {"effect_type": "Structural Restoration", "node_id": "restore"},
        ],
        [{"effect_type": "Lua Shader", "node_id": "shader"}],
    )

    assert assignments == {
        "flow": "cuda",
        "master_deck_blend": "cpu",
        "restore": "cpu",
        "shader": "opengl",
    }


def test_nested_image_history_must_match_master_resolution():
    executor = MasterGraphExecutor(MasterResolution(width=8, height=6))
    executor.state_a.effect_states["node"] = {
        "previous_image": np.zeros((3, 4, 3), dtype=np.uint8)
    }
    with pytest.raises(ValueError, match="nested temporal buffers"):
        executor.state_a.assert_master_integrity()
