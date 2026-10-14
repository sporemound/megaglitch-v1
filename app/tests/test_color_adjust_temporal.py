"""Mixed color/feedback output equivalence and stream-owned temporal history."""

import copy

import numpy as np
import pytest

import glitch_effects as ge
from megaglitch.master_graph import MasterGraphExecutor
from megaglitch.preview_executor import PreviewExecutor, PreviewGraphSnapshot
from megaglitch.render_contract import MasterResolution
from tools.benchmark.run_color_adjust_preview import color_adjust_pipeline
from tools.rust_ui_bridge import deterministic_frame


def _reference_color(frame, hue_shift=0, saturation=1, gamma=1):
    return np.clip(
        ge.apply_gamma(ge.apply_hsv_adjust(frame, hue_shift, saturation), gamma), 0, 255
    ).astype(np.uint8)


def _arrays(state):
    return [value for value in (
        state.previous_frame, state.feedback_image, state.trail_image, *state.frame_buffer
    ) if isinstance(value, np.ndarray)]


def _sequence(pipeline):
    master = MasterGraphExecutor(MasterResolution(96, 64))
    outputs = []
    for sequence in range(1, 15):
        source = deterministic_frame(sequence, 96, 64)
        outputs.append(master.process(source, pipeline_a=pipeline).frame.copy())
    return outputs, master.state_a


@pytest.mark.parametrize("temporal_first", [False, True])
def test_mixed_sequence_matches_reference_pixels_and_history(monkeypatch, temporal_first):
    pipeline = color_adjust_pipeline(feedback=True, temporal_first=temporal_first)
    actual, actual_state = _sequence(pipeline)
    with monkeypatch.context() as reference:
        reference.setattr(ge, "apply_color_adjust_rgb8", _reference_color)
        expected, expected_state = _sequence(pipeline)
    for output, reference_output in zip(actual, expected):
        np.testing.assert_array_equal(output, reference_output)
    for history, reference_history in zip(_arrays(actual_state), _arrays(expected_state)):
        np.testing.assert_array_equal(history, reference_history)
    assert len(_arrays(actual_state)) == len(_arrays(expected_state))
    assert len(actual_state.frame_buffer) <= 10
    assert actual_state.feedback_image is not None
    assert actual_state.trail_image is not None
    # The temporal stage must actually affect the sequence, rather than merely
    # appearing in reported coverage while passing the color result through.
    color_only, _ = _sequence(color_adjust_pipeline())
    assert any(not np.array_equal(a, b) for a, b in zip(actual[1:], color_only[1:]))


@pytest.mark.parametrize("temporal_first", [False, True])
def test_preview_tier_and_color_changes_do_not_contaminate_mixed_master(temporal_first):
    pipeline = color_adjust_pipeline(feedback=True, temporal_first=temporal_first)
    master = MasterGraphExecutor(MasterResolution(1280, 720))
    baseline = MasterGraphExecutor(MasterResolution(1280, 720))
    preview = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(pipeline, 1),
        preferred_tier=(960, 540), adaptive=False, autostart=False,
    )
    revision = 1
    for sequence in range(1, 5):
        source = deterministic_frame(sequence, 1280, 720)
        original = source.copy()
        state_before = [value.copy() for value in _arrays(master.state_a)]
        if sequence == 2:
            preview.set_tier((640, 360))
        if sequence == 3:
            changed = copy.deepcopy(pipeline)
            next(n for n in changed if n["effect_type"] == "Color Adjust")["params"]["saturation"] = 0.0
            revision = 2
            preview.update_graph(PreviewGraphSnapshot.from_pipeline(changed, revision))
        preview.submit_latest(source, sequence / 60, revision, sequence=sequence)
        result = preview.render_latest()
        assert result is not None and result.graph_revision == revision
        assert result.frame.shape == ((540, 960, 3) if sequence == 1 else (360, 640, 3))
        assert all(status.status == "implemented" for status in result.node_statuses)
        for before, after in zip(state_before, _arrays(master.state_a)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(source, original)
        actual = master.process(source, pipeline_a=pipeline)
        expected = baseline.process(source, pipeline_a=pipeline)
        np.testing.assert_array_equal(actual.frame, expected.frame)
        for a, b in zip(_arrays(master.state_a), _arrays(baseline.state_a)):
            np.testing.assert_array_equal(a, b)
        master.state_a.assert_master_integrity()
        assert all(a.shape == (720, 1280, 3) for a in _arrays(master.state_a))
        assert all(a.shape == result.frame.shape for a in _arrays(preview.temporal_state))
        assert not any(
            np.shares_memory(a, b) for a in _arrays(master.state_a)
            for b in _arrays(preview.temporal_state)
        )
    # Initial setup and tier change reset preview; parameter-only change does not.
    assert preview.temporal_state.temporal_reset_count == 2


def test_mixed_graph_order_is_observable():
    color_first, _ = _sequence(color_adjust_pipeline(feedback=True))
    temporal_first, _ = _sequence(color_adjust_pipeline(feedback=True, temporal_first=True))
    assert any(not np.array_equal(a, b) for a, b in zip(color_first, temporal_first))
