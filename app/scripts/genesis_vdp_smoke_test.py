import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults

    bootstrap_for_dev_defaults()
except Exception:
    pass

from frame_state import FrameState
from node_pipeline_runtime import (
    _validate_genesis_outputs,
    execute_flat_node_pipeline,
    generate_node_outputs,
)


WIDTH = 320
HEIGHT = 224


def make_genesis_node(node_id, selected_output="image_out", params=None, input_order=None):
    return {
        "effect_type": "Genesis VDP Gen",
        "node_type": "generator",
        "is_generator": True,
        "node_id": node_id,
        "selected_output": selected_output,
        "params": dict(params or {}),
        "input_order": list(input_order or []),
    }


def run_pipeline(pipeline, state, time_value=0.0, primary_frame=None):
    frame, previous_frame, feedback_image, trail_image = execute_flat_node_pipeline(
        pipeline,
        primary_frame=primary_frame,
        secondary_frame=None,
        extra_params={"time_accum": float(time_value)},
        previous_frame=state.previous_frame,
        feedback_image=state.feedback_image,
        trail_image=state.trail_image,
        frame_buffer=state.frame_buffer,
        worker_state=state,
        fallback_size=(WIDTH, HEIGHT),
    )
    state.previous_frame = previous_frame
    state.feedback_image = feedback_image
    state.trail_image = trail_image
    state.frame_count += 1
    return frame


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_output_contract():
    state = FrameState()
    outputs = generate_node_outputs(
        "Genesis VDP Gen",
        WIDTH,
        HEIGHT,
        0.0,
        {},
        worker_state=state,
        node_key="contract",
        input_frames={},
    )
    _validate_genesis_outputs(outputs)
    assert_true(np.array_equal(outputs["image_out"], outputs["final_frame"]), "image_out must mirror final_frame")
    assert_true(outputs["final_frame"].shape == (HEIGHT, WIDTH, 3), "Genesis final frame has unexpected shape")
    print("PASS output_contract")


def test_generator_without_optional_inputs():
    state = FrameState()
    pipeline = [
        make_genesis_node("standalone", selected_output="image_out"),
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]
    frame = run_pipeline(pipeline, state, time_value=0.0)
    assert_true(frame.shape == (HEIGHT, WIDTH, 3), "Standalone Genesis pipeline did not render correctly")
    assert_true("genesis_vdp::standalone" in state.generator_states, "Stable Genesis machine-state cache was not created")
    assert_true(
        f"genesis_outputs::standalone" in state.effect_states,
        "Genesis auxiliary outputs were not cached for downstream nodes",
    )
    print("PASS standalone_pipeline_without_optional_inputs")


def test_priority_composite_semantics():
    state = FrameState()
    node_id = "priority-stack"
    pipeline = [
        make_genesis_node(node_id, selected_output="plane_a"),
        make_genesis_node(node_id, selected_output="sprite_layer"),
        make_genesis_node(node_id, selected_output="priority_mask"),
        {
            "effect_type": "Genesis Priority Composite",
            "node_type": "genesis_priority_composite",
            "node_id": "priority-compose",
            "params": {
                "mode": "strict_priority",
                "threshold": 0.45,
                "softness": 0.12,
                "mix": 1.0,
            },
            "input_order": ["base_image", "overlay_image", "priority_mask"],
        },
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]
    composite = run_pipeline(pipeline, state, time_value=0.25)
    cached = state.effect_states[f"genesis_outputs::{node_id}"]
    base = cached["plane_a"]
    priority = cached["priority_mask"]
    diff = np.mean(np.abs(composite.astype(np.int16) - base.astype(np.int16)))
    assert_true(priority.max() > 0, "Priority mask should contain routing information")
    assert_true(diff > 0.1, "Priority composite did not materially change the base image")
    print("PASS priority_composite_semantics")


def test_mask_key_semantics():
    state = FrameState()
    node_id = "mask-key"
    glitch_params = {
        "vram_scramble": 0.35,
        "tile_index_corruption": 0.45,
        "palette_corruption": 0.3,
        "nametable_tear": 0.25,
    }
    pipeline = [
        make_genesis_node(node_id, selected_output="final_frame", params=glitch_params),
        make_genesis_node(node_id, selected_output="corruption_mask", params=glitch_params),
        {
            "effect_type": "Genesis Mask Key",
            "node_type": "genesis_mask_key",
            "node_id": "mask-key-node",
            "params": {
                "mode": "corruption_gate",
                "threshold": 0.15,
                "softness": 0.12,
                "strength": 1.0,
            },
            "input_order": ["image_in", "corruption_mask"],
        },
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]
    keyed = run_pipeline(pipeline, state, time_value=0.5)
    cached = state.effect_states[f"genesis_outputs::{node_id}"]
    base = cached["final_frame"]
    corruption = cached["corruption_mask"]
    diff = np.mean(np.abs(keyed.astype(np.int16) - base.astype(np.int16)))
    assert_true(corruption.max() > 0, "Corruption mask should contain gated regions")
    assert_true(diff > 0.1, "Genesis Mask Key did not change the incoming image")
    print("PASS mask_key_semantics")


def test_machine_state_cache_stability():
    state = FrameState()
    pipeline = [
        make_genesis_node("stable-cache", selected_output="image_out"),
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]
    run_pipeline(pipeline, state, time_value=0.0)
    state_ref = state.generator_states.get("genesis_vdp::stable-cache")
    assert_true(state_ref is not None, "Stable cache state was not created")
    run_pipeline(pipeline, state, time_value=1.0)
    state_ref_after = state.generator_states.get("genesis_vdp::stable-cache")
    assert_true(state_ref_after is state_ref, "Stable Genesis machine-state cache should persist across frames")
    print("PASS machine_state_cache_stability")


def main():
    test_output_contract()
    test_generator_without_optional_inputs()
    test_priority_composite_semantics()
    test_mask_key_semantics()
    test_machine_state_cache_stability()
    print("Genesis VDP smoke test complete")


if __name__ == "__main__":
    main()
