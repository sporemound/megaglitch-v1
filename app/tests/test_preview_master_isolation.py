import copy
import hashlib

import numpy as np

from megaglitch.master_graph import MasterGraphExecutor
from megaglitch.preview_executor import PreviewExecutor, PreviewGraphSnapshot
from megaglitch.render_contract import MasterResolution
from tools.rust_ui_bridge import compile_preset_graph, deterministic_frame


MASTER_WIDTH = 1280
MASTER_HEIGHT = 720
MASTER_SHAPE = (MASTER_HEIGHT, MASTER_WIDTH, 3)


def _feedback_trail_preset() -> dict:
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "FeedbackTrailNode:master-isolation",
                "type": "FeedbackTrailNode",
                "label": "Feedback / Trail",
                "params": {
                    "enabled": True,
                    "feedback_intensity": 0.25,
                    "trailing_intensity": 0.5,
                },
            }
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Feedback / Trail",
                "to_node_id": "FeedbackTrailNode:master-isolation",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Feedback / Trail",
                "from_node_id": "FeedbackTrailNode:master-isolation",
                "to_node": "DISPLAY",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def _array_fingerprint(value: np.ndarray) -> tuple:
    return (
        id(value),
        tuple(int(part) for part in value.shape),
        str(value.dtype),
        hashlib.sha256(value.tobytes()).hexdigest(),
    )


def _master_state_fingerprint(state) -> tuple:
    named_buffers = tuple(
        (name, None if value is None else _array_fingerprint(value))
        for name in ("previous_frame", "feedback_image", "trail_image")
        for value in (getattr(state, name),)
    )
    history = tuple(_array_fingerprint(frame) for frame in state.frame_buffer)
    inventory = tuple(
        (
            path,
            tuple(record.get("shape", ())),
            bool(record.get("master_resolution", False)),
        )
        for path, record in sorted(state.temporal_buffer_inventory().items())
    )
    return (
        id(state),
        id(state.frame_buffer),
        state.master_shape,
        state.master_generation,
        state.resolution_change_count,
        state.frame_count,
        named_buffers,
        history,
        inventory,
    )


def _assert_master_state_contract(state, expected_identity: int) -> None:
    assert id(state) == expected_identity
    assert state.master_shape == MASTER_SHAPE
    assert state.master_generation == 0
    assert state.resolution_change_count == 0
    state.assert_master_integrity()
    assert all(frame.shape == MASTER_SHAPE for frame in state.frame_buffer)
    assert all(
        record["master_resolution"]
        for record in state.temporal_buffer_inventory().values()
    )


def _assert_no_shared_temporal_arrays(master_state, preview_state) -> None:
    master_arrays = [
        value
        for value in (
            master_state.previous_frame,
            master_state.feedback_image,
            master_state.trail_image,
            *master_state.frame_buffer,
        )
        if isinstance(value, np.ndarray)
    ]
    preview_arrays = [
        value
        for value in (
            preview_state.previous_frame,
            preview_state.feedback_image,
            preview_state.trail_image,
            *preview_state.frame_buffer,
        )
        if isinstance(value, np.ndarray)
    ]
    assert all(master is not preview for master in master_arrays for preview in preview_arrays)


def _run_authoritative_sequence(preview_mode: str) -> tuple[list[np.ndarray], dict]:
    master_pipeline = compile_preset_graph(_feedback_trail_preset())
    master = MasterGraphExecutor(MasterResolution(MASTER_WIDTH, MASTER_HEIGHT))
    master_state_identity = id(master.state_a)
    outputs: list[np.ndarray] = []
    preview_shapes: list[tuple[int, int, int]] = []
    preview_revisions: list[int] = []
    executor = None

    if preview_mode != "off":
        preview_pipeline = copy.deepcopy(master_pipeline)
        executor = PreviewExecutor(
            PreviewGraphSnapshot.from_pipeline(preview_pipeline, 1),
            preferred_tier=(960, 540),
            adaptive=False,
        )

    try:
        for sequence in range(1, 5):
            source = deterministic_frame(sequence, MASTER_WIDTH, MASTER_HEIGHT)
            source_digest = hashlib.sha256(source.tobytes()).hexdigest()
            state_before_preview = _master_state_fingerprint(master.state_a)

            if executor is not None:
                if preview_mode == "tier_changes" and sequence == 2:
                    executor.set_tier((720, 480))
                elif preview_mode == "tier_changes" and sequence == 3:
                    executor.set_tier((640, 360))

                revision = 1
                if preview_mode == "parameter_changes" and sequence == 3:
                    changed_pipeline = copy.deepcopy(master_pipeline)
                    feedback_node = next(
                        step
                        for step in changed_pipeline
                        if step.get("effect_type") == "Feedback / Trail"
                    )
                    feedback_node["params"]["feedback_intensity"] = 0.75
                    feedback_node["params"]["trailing_intensity"] = 0.1
                    executor.update_graph(
                        PreviewGraphSnapshot.from_pipeline(changed_pipeline, 2)
                    )
                if preview_mode == "parameter_changes" and sequence >= 3:
                    revision = 2

                executor.submit_latest(
                    source,
                    source_timestamp=float(sequence) / 60.0,
                    graph_revision=revision,
                    sequence=sequence,
                )
                preview_result = executor.wait_for_result(
                    after_sequence=sequence - 1,
                    timeout=5.0,
                )
                preview_shapes.append(preview_result.frame.shape)
                preview_revisions.append(preview_result.graph_revision)

                # Preview owns its source copy and every piece of temporal state.
                assert hashlib.sha256(source.tobytes()).hexdigest() == source_digest
                assert executor.temporal_state is not master.state_a
                assert executor.temporal_state.master_shape is None
                assert _master_state_fingerprint(master.state_a) == state_before_preview
                _assert_no_shared_temporal_arrays(master.state_a, executor.temporal_state)

            result = master.process(source, pipeline_a=master_pipeline)
            outputs.append(result.frame.copy())
            _assert_master_state_contract(master.state_a, master_state_identity)
            assert result.frame.shape == MASTER_SHAPE

        preview_diagnostics = None
        if executor is not None:
            preview_diagnostics = {
                "shapes": tuple(preview_shapes),
                "revisions": tuple(preview_revisions),
                "temporal_reset_count": executor.temporal_state.temporal_reset_count,
                "resolution_generation": executor.temporal_state.resolution_generation,
            }
            state_before_stop = _master_state_fingerprint(master.state_a)
            executor.stop()
            executor = None
            assert _master_state_fingerprint(master.state_a) == state_before_stop

        return outputs, {
            "master_state_identity": master_state_identity,
            "master_shape": master.state_a.master_shape,
            "master_generation": master.state_a.master_generation,
            "resolution_change_count": master.state_a.resolution_change_count,
            "history_shapes": tuple(frame.shape for frame in master.state_a.frame_buffer),
            "preview": preview_diagnostics,
        }
    finally:
        if executor is not None:
            executor.stop()


def test_preview_never_changes_authoritative_sequence_or_master_temporal_state():
    baseline, baseline_state = _run_authoritative_sequence("off")
    preview_on, preview_on_state = _run_authoritative_sequence("on")
    tier_changes, tier_state = _run_authoritative_sequence("tier_changes")
    parameter_changes, parameter_state = _run_authoritative_sequence("parameter_changes")

    for scenario in (preview_on, tier_changes, parameter_changes):
        assert len(scenario) == len(baseline)
        for expected, actual in zip(baseline, scenario):
            assert actual.shape == expected.shape == MASTER_SHAPE
            assert actual.tobytes() == expected.tobytes()

    for state in (baseline_state, preview_on_state, tier_state, parameter_state):
        assert state["master_shape"] == MASTER_SHAPE
        assert state["master_generation"] == 0
        assert state["resolution_change_count"] == 0
        assert state["history_shapes"] == (MASTER_SHAPE,) * 4

    assert preview_on_state["preview"]["shapes"] == ((540, 960, 3),) * 4
    assert tier_state["preview"]["shapes"] == (
        (540, 960, 3),
        (480, 720, 3),
        (360, 640, 3),
        (360, 640, 3),
    )
    assert tier_state["preview"]["resolution_generation"] == 2
    assert parameter_state["preview"]["revisions"] == (1, 1, 2, 2)
    assert parameter_state["preview"]["temporal_reset_count"] == 1
