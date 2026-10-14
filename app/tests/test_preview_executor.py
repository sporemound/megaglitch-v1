import hashlib
import random
import threading
import time

import numpy as np
import pytest

from frame_state import FrameState
import megaglitch.preview_executor as preview_executor_module
from megaglitch.preview_executor import (
    AdaptivePreviewController,
    PreviewExecutor,
    PreviewGraphSnapshot,
    PreviewTemporalState,
)
import tools.rust_ui_bridge as rust_ui_bridge


def source_display_pipeline():
    return [
        {
            "effect_type": "Source",
            "params": {},
            "node_type": "source",
            "source_key": "primary",
        },
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]


def unsupported_pipeline():
    return [
        source_display_pipeline()[0],
        {
            "effect_type": "Multimodal Analysis",
            "node_id": "MultimodalAnalysisNode:1",
            "node_type": "effect",
            "params": {"enabled": True},
        },
        source_display_pipeline()[1],
    ]


def temporal_slicing_pipeline():
    return [
        source_display_pipeline()[0],
        {
            "effect_type": "Temporal Slicing",
            "node_id": "TemporalSlicingNode:python-rng",
            "node_type": "effect",
            "params": {
                "enabled": True,
                "slice_width": 7,
                "slice_direction": "vertical",
                "time_offset_range": 5,
            },
        },
        {
            "effect_type": "Temporal Slicing",
            "node_id": "TemporalSlicingNode:numpy-rng",
            "node_type": "effect",
            "params": {
                "enabled": True,
                "slice_width": 9,
                "slice_direction": "diagonal",
                "time_offset_range": 4,
            },
        },
        source_display_pipeline()[1],
    ]


def pframe_random_branches_pipeline():
    return [
        source_display_pipeline()[0],
        {
            "effect_type": "P-Frame Glitches",
            "node_id": "PFrameGlitchesNode:rng-isolation",
            "node_type": "effect",
            "params": {
                "enabled": True,
                "motion_vector_corruption": 0.8,
                "motion_vector_direction": "random",
                "prediction_error_amplify": 0.0,
                "macroblock_size": "16",
                "temporal_bleeding": 0.7,
                "bleed_distance": 3,
                "temporal_bleeding_direction": "random",
                "macroblock_scramble": 0.8,
                "scrambling_pattern": "random",
                "residual_injection": 0.6,
                "residual_noise_type": "block_artifacts",
                "residual_direction": "uniform",
                "reference_corruption": 0.6,
                "reference_corruption_pattern": "scattered",
            },
        },
        source_display_pipeline()[1],
    ]


def patterned_frame(width=1920, height=1080):
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    return np.stack(
        [
            np.broadcast_to((x % 256).astype(np.uint8), (height, width)),
            np.broadcast_to((y % 256).astype(np.uint8), (height, width)),
            ((x + y) % 256).astype(np.uint8),
        ],
        axis=2,
    )


def test_preview_pregraph_scaling_and_master_state_isolation():
    master_state = FrameState()
    master_state.previous_frame = np.full((8, 12, 3), 91, dtype=np.uint8)
    master_before = hashlib.sha256(master_state.previous_frame.tobytes()).hexdigest()
    snapshot = PreviewGraphSnapshot.from_pipeline(source_display_pipeline(), 4)
    executor = PreviewExecutor(snapshot, preferred_tier=(960, 540), adaptive=False, autostart=False)

    executor.submit_latest(patterned_frame(), time.time(), 4, sequence=10)
    result = executor.render_latest()

    assert result is not None
    assert result.frame.shape == (540, 960, 3)
    assert result.frame.flags.c_contiguous
    assert isinstance(executor.temporal_state, PreviewTemporalState)
    assert executor.temporal_state is not master_state
    assert hashlib.sha256(master_state.previous_frame.tobytes()).hexdigest() == master_before


def test_all_preview_tiers_are_applied_before_source_display_graph():
    frame = patterned_frame()
    for width, height in ((1280, 720), (960, 540), (720, 480), (640, 360)):
        executor = PreviewExecutor(
            PreviewGraphSnapshot.from_pipeline(source_display_pipeline(), 1),
            preferred_tier=(width, height),
            adaptive=False,
            autostart=False,
        )
        executor.submit_latest(frame, 1.0, 1, sequence=1)
        result = executor.render_latest()
        assert result is not None
        assert result.frame.shape == (height, width, 3)


def test_latest_frame_mailbox_has_capacity_one_and_counts_superseded_requests():
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(source_display_pipeline(), 1),
        adaptive=False,
        autostart=False,
    )
    frame = np.zeros((36, 64, 3), dtype=np.uint8)

    executor.submit_latest(frame, 1.0, 1, sequence=11)
    executor.submit_latest(frame + 1, 2.0, 1, sequence=12)
    executor.submit_latest(frame + 2, 3.0, 1, sequence=13)
    before = executor.telemetry()
    result = executor.render_latest()
    after = executor.telemetry()

    assert executor.mailbox_capacity == 1
    assert before.preview_requests == 3
    assert before.preview_superseded == 2
    assert result is not None and result.sequence == 13
    assert after.preview_rendered == 1


def test_graph_snapshot_is_detached_and_new_revision_applies_to_next_render():
    mutable_pipeline = source_display_pipeline()
    first = PreviewGraphSnapshot.from_pipeline(mutable_pipeline, 1)
    mutable_pipeline.insert(1, unsupported_pipeline()[1])
    executor = PreviewExecutor(first, adaptive=False, autostart=False)

    executor.submit_latest(np.zeros((90, 160, 3), dtype=np.uint8), 1.0, 1, sequence=1)
    first_result = executor.render_latest()
    assert first_result is not None
    assert first_result.graph_revision == 1
    assert all(status.label != "Multimodal Analysis" for status in first_result.node_statuses)

    second = PreviewGraphSnapshot.from_pipeline(mutable_pipeline, 2)
    executor.update_graph(second)
    executor.submit_latest(np.zeros((90, 160, 3), dtype=np.uint8), 2.0, 2, sequence=2)
    second_result = executor.render_latest()
    assert second_result is not None
    assert second_result.graph_revision == 2
    multimodal = next(status for status in second_result.node_statuses if status.label == "Multimodal Analysis")
    assert multimodal.status == "unsupported"
    assert multimodal.provider == "none"


def test_parameter_only_graph_update_preserves_preview_temporal_state():
    pipeline = source_display_pipeline()
    pipeline.insert(
        1,
        {
            "effect_type": "Feedback / Trail",
            "node_id": "FeedbackTrailNode:1",
            "node_type": "effect",
            "params": {"enabled": True, "feedback_intensity": 0.1},
        },
    )
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(pipeline, 1),
        preferred_tier=(640, 360),
        adaptive=False,
        autostart=False,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    executor.submit_latest(frame, 1.0, 1, sequence=1)
    executor.render_latest()
    resets_before = executor.temporal_state.temporal_reset_count
    history_before = list(executor.temporal_state.frame_buffer)

    pipeline[1]["params"]["feedback_intensity"] = 0.2
    executor.update_graph(PreviewGraphSnapshot.from_pipeline(pipeline, 2))
    assert executor.temporal_state.temporal_reset_count == resets_before
    assert executor.temporal_state.frame_buffer == history_before


def test_adaptive_controller_uses_sustained_p95_hysteresis_and_cooldown():
    controller = AdaptivePreviewController(
        preferred_tier=(960, 540),
        min_samples=3,
        window_size=3,
        cooldown_seconds=2.0,
    )

    assert controller.observe(20.0, now=0.0) is None
    assert controller.observe(19.0, now=0.0) is None
    assert controller.observe(18.0, now=0.0) == (720, 480)

    assert controller.observe(5.0, now=1.0) is None
    assert controller.observe(5.0, now=1.0) is None
    assert controller.observe(5.0, now=1.0) is None
    assert controller.current_tier == (720, 480)
    assert controller.observe(5.0, now=2.1) == (960, 540)


def test_preview_tier_change_resets_only_preview_temporal_state():
    master_state = FrameState()
    master_state.frame_buffer.append(np.ones((4, 4, 3), dtype=np.uint8))
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(source_display_pipeline(), 1),
        preferred_tier=(960, 540),
        adaptive=False,
        autostart=False,
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    executor.submit_latest(frame, 1.0, 1, sequence=1)
    executor.render_latest()
    initial_generation = executor.temporal_state.resolution_generation
    executor.temporal_state.frame_buffer.append(np.ones((540, 960, 3), dtype=np.uint8))

    executor.set_tier((720, 480))
    executor.submit_latest(frame, 2.0, 1, sequence=2)
    result = executor.render_latest()

    assert result is not None and result.frame.shape == (480, 720, 3)
    assert executor.temporal_state.resolution_generation == initial_generation + 1
    assert len(executor.temporal_state.frame_buffer) == 1
    assert executor.temporal_state.frame_buffer[0].shape == (480, 720, 3)
    assert len(master_state.frame_buffer) == 1
    assert master_state.frame_buffer[0].shape == (4, 4, 3)


def test_async_worker_applies_graph_change_without_waiting_for_authoritative_state():
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(source_display_pipeline(), 1),
        preferred_tier=(640, 360),
        adaptive=False,
    )
    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        executor.submit_latest(frame, time.time(), 1, sequence=1)
        first = executor.wait_for_result(timeout=5.0)
        assert first.graph_revision == 1

        executor.update_graph(PreviewGraphSnapshot.from_pipeline(unsupported_pipeline(), 2))
        executor.submit_latest(frame, time.time(), 2, sequence=2)
        second = executor.wait_for_result(after_sequence=1, timeout=5.0)
        assert second.graph_revision == 2
        assert any(status.status == "unsupported" for status in second.node_statuses)
    finally:
        executor.stop()


def test_async_worker_failure_is_visible_in_structured_telemetry(monkeypatch):
    def fail_render(*_args, **_kwargs):
        raise ModuleNotFoundError("preview-only dependency is missing")

    monkeypatch.setattr(
        preview_executor_module,
        "execute_flat_node_pipeline",
        fail_render,
    )
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(source_display_pipeline(), 1),
        preferred_tier=(640, 360),
        adaptive=False,
    )
    try:
        executor.submit_latest(
            np.zeros((36, 64, 3), dtype=np.uint8),
            time.time(),
            1,
            sequence=17,
        )
        deadline = time.perf_counter() + 2.0
        telemetry = executor.telemetry()
        while telemetry.last_error is None and time.perf_counter() < deadline:
            time.sleep(0.005)
            telemetry = executor.telemetry()

        assert telemetry.last_error == {
            "code": "preview_render_failed",
            "message": "preview-only dependency is missing",
            "exception": "ModuleNotFoundError",
        }
        assert telemetry.preview_rendered == 0
        assert telemetry.preview_dropped == 1
    finally:
        executor.stop()


def test_unsupported_structural_node_is_reported_and_never_reaches_executor(monkeypatch):
    observed_pipelines = []

    def capture_pipeline(pipeline, primary_frame, **kwargs):
        observed_pipelines.append(pipeline)
        return (
            primary_frame,
            kwargs.get("previous_frame"),
            kwargs.get("feedback_image"),
            kwargs.get("trail_image"),
        )

    monkeypatch.setattr(
        preview_executor_module,
        "execute_flat_node_pipeline",
        capture_pipeline,
    )
    structural_pipeline = [
        source_display_pipeline()[0],
        {
            "effect_type": "Unimplemented Procedural Generator",
            "node_id": "GeneratorNode:unsafe",
            "node_type": "generator",
            "is_generator": True,
            "params": {"enabled": True},
        },
        source_display_pipeline()[1],
    ]
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(structural_pipeline, 1),
        adaptive=False,
        autostart=False,
    )
    frame = patterned_frame(64, 36)
    executor.submit_latest(frame, 1.0, 1, sequence=1)
    result = executor.render_latest()

    assert result is not None
    np.testing.assert_array_equal(result.frame, frame)
    assert len(observed_pipelines) == 1
    assert [step["effect_type"] for step in observed_pipelines[0]] == ["Source", "Display"]
    structural_status = next(
        status for status in result.node_statuses if status.node_id == "GeneratorNode:unsafe"
    )
    assert structural_status.status == "unsupported"
    assert structural_status.provider == "none"


@pytest.mark.parametrize("node_type", ["generator", "mixer"])
@pytest.mark.parametrize("effect_enabled", [True, False])
def test_structural_fallback_reports_omitted_effects_and_sources(node_type, effect_enabled):
    pipeline = [
        {**source_display_pipeline()[0], "node_id": "primary-source"},
        {
            "effect_type": "Invert Colors",
            "node_id": "invert",
            "params": {"enabled": effect_enabled, "invert_chance": 100},
        },
        {
            **source_display_pipeline()[0],
            "node_id": "secondary-source",
            "source_key": "secondary",
        },
        {
            "effect_type": "Unsupported Structural Node",
            "node_id": "unsupported-structural",
            "node_type": node_type,
            "is_generator": node_type == "generator",
            "params": {"enabled": True},
        },
        source_display_pipeline()[1],
    ]
    snapshot = PreviewGraphSnapshot.from_pipeline(pipeline, 1)
    executor = PreviewExecutor(snapshot, adaptive=False, autostart=False)
    frame = patterned_frame(64, 36)
    executor.submit_latest(frame, 1.0, 1, sequence=1)
    result = executor.render_latest()

    assert result is not None
    np.testing.assert_array_equal(result.frame, frame)
    statuses = {status.node_id: status for status in result.node_statuses}
    assert statuses["invert"].status == ("unsupported" if effect_enabled else "disabled")
    assert statuses["invert"].provider == (
        "source_display_fallback" if effect_enabled else "disabled"
    )
    assert statuses["secondary-source"].status == "unsupported"
    assert statuses["secondary-source"].provider == "source_display_fallback"
    assert statuses["primary-source"].status == "implemented"
    assert statuses["unsupported-structural"].provider == "none"
    assert result.node_statuses[-1].status == "implemented"
    assert snapshot.materialize_pipeline() == pipeline


def test_supported_preview_effect_executes_without_structural_fallback():
    pipeline = source_display_pipeline()
    pipeline.insert(
        1, {"effect_type": "Invert Colors", "params": {"enabled": True, "invert_chance": 100}}
    )
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(pipeline, 1), adaptive=False, autostart=False
    )
    frame = patterned_frame(64, 36)
    executor.submit_latest(frame, 1.0, 1, sequence=1)
    result = executor.render_latest()

    assert result is not None
    np.testing.assert_array_equal(result.frame, 255 - frame)
    assert result.node_statuses[1].status == "implemented"
    assert result.node_statuses[1].provider == "preview_scaled_existing"


def test_preview_rng_is_local_and_does_not_advance_process_global_rng_state():
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    try:
        random.seed(8137)
        np.random.seed(9127)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()

        executor = PreviewExecutor(
            PreviewGraphSnapshot.from_pipeline(temporal_slicing_pipeline(), 1),
            adaptive=False,
            autostart=False,
        )
        executor.submit_latest(patterned_frame(160, 90), 1.0, 1, sequence=1)
        result = executor.render_latest()

        assert result is not None
        assert random.getstate() == expected_python_state
        actual_numpy_state = np.random.get_state()
        assert actual_numpy_state[0] == expected_numpy_state[0]
        np.testing.assert_array_equal(actual_numpy_state[1], expected_numpy_state[1])
        assert actual_numpy_state[2:] == expected_numpy_state[2:]
    finally:
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)


def test_pframe_preview_random_branches_do_not_advance_process_global_rng_state():
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    try:
        random.seed(1776)
        np.random.seed(2026)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()

        executor = PreviewExecutor(
            PreviewGraphSnapshot.from_pipeline(pframe_random_branches_pipeline(), 1),
            preferred_tier=(640, 360),
            adaptive=False,
            autostart=False,
        )
        for sequence in range(1, 4):
            executor.submit_latest(
                patterned_frame(160, 90),
                float(sequence),
                1,
                sequence=sequence,
            )
            assert executor.render_latest() is not None

        assert random.getstate() == expected_python_state
        actual_numpy_state = np.random.get_state()
        assert actual_numpy_state[0] == expected_numpy_state[0]
        np.testing.assert_array_equal(actual_numpy_state[1], expected_numpy_state[1])
        assert actual_numpy_state[2:] == expected_numpy_state[2:]
    finally:
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)


def test_preview_history_is_bounded_and_retains_only_newest_frames():
    executor = PreviewExecutor(
        PreviewGraphSnapshot.from_pipeline(source_display_pipeline(), 1),
        adaptive=False,
        autostart=False,
    )
    for sequence in range(15):
        frame = np.full((18, 32, 3), sequence, dtype=np.uint8)
        executor.submit_latest(frame, float(sequence), 1, sequence=sequence + 1)
        assert executor.render_latest() is not None

    assert len(executor.temporal_state.frame_buffer) == 10
    assert int(executor.temporal_state.frame_buffer[0][0, 0, 0]) == 5
    assert int(executor.temporal_state.frame_buffer[-1][0, 0, 0]) == 14


def test_bridge_preview_continues_while_authoritative_processing_is_blocked(monkeypatch):
    authoritative_started = threading.Event()
    release_authoritative = threading.Event()
    original_authoritative_process = rust_ui_bridge._process_authoritative_frame

    def slow_authoritative_process(*args, **kwargs):
        authoritative_started.set()
        if not release_authoritative.wait(timeout=5.0):
            raise TimeoutError("test did not release authoritative render")
        return original_authoritative_process(*args, **kwargs)

    monkeypatch.setattr(
        rust_ui_bridge,
        "_process_authoritative_frame",
        slow_authoritative_process,
    )

    with rust_ui_bridge.PreviewSurface(generation=707) as surface:
        engine = rust_ui_bridge.BridgeEngine(64, 36, surface=surface)
        engine.start_decoupled()
        try:
            engine.submit_decoupled_frame()
            assert authoritative_started.wait(timeout=2.0)

            preview_sequences = []
            for preview_number in range(3):
                if preview_number:
                    engine.submit_decoupled_frame()
                deadline = time.perf_counter() + 2.0
                while time.perf_counter() < deadline:
                    preview_events = [
                        event
                        for event in engine.poll_decoupled_events()
                        if event["type"] == "preview_frame"
                    ]
                    if preview_events:
                        event = preview_events[-1]
                        preview_sequences.append(event["source_sequence"])
                        assert surface.release(
                            event["slot"],
                            event["generation"],
                            event["sequence"],
                        )
                        break
                    time.sleep(0.005)
                else:
                    pytest.fail("preview stopped while authoritative processing was blocked")

            snapshot = engine.snapshot()
            assert preview_sequences == sorted(set(preview_sequences))
            assert len(preview_sequences) == 3
            assert snapshot["preview_rendered"] >= 3
            assert snapshot["preview_sequence"] == preview_sequences[-1]
            assert snapshot["authoritative_sequence"] == 0
            assert snapshot["authoritative_dropped"] >= 2

            release_authoritative.set()
            deadline = time.perf_counter() + 2.0
            while engine.authoritative_sequence == 0 and time.perf_counter() < deadline:
                engine.poll_decoupled_events()
                time.sleep(0.005)
            assert engine.authoritative_sequence == 1
        finally:
            release_authoritative.set()
            engine.close()
