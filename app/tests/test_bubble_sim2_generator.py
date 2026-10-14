from types import SimpleNamespace

import numpy as np

import node_pipeline_runtime as runtime
from frame_state import FrameState
from megaglitch.bubble_sim2_cpu import generate_bubble_sim2_cpu_outputs


BUFFER_NAMES = (
    "bubble_mask",
    "bubble_normal",
    "bubble_depth",
    "bubble_refraction",
    "bubble_highlight",
)


def test_cpu_outputs_are_distinct_and_well_formed():
    outputs = generate_bubble_sim2_cpu_outputs(96, 64, 1.25, {}, None)

    for name in ("image_out", "final_frame", "bubble_debug", *BUFFER_NAMES):
        assert outputs[name].shape == (64, 96, 3)
        assert outputs[name].dtype == np.uint8

    assert len({outputs[name].tobytes() for name in BUFFER_NAMES}) == len(BUFFER_NAMES)
    assert np.array_equal(outputs["bubble_mask"][..., 0], outputs["bubble_mask"][..., 1])
    assert np.array_equal(outputs["bubble_mask"][..., 1], outputs["bubble_mask"][..., 2])


def test_cpu_fallback_is_deterministic_and_composites_input():
    source = np.full((48, 80, 3), (35, 65, 95), dtype=np.uint8)
    first = generate_bubble_sim2_cpu_outputs(80, 48, 2.5, {}, source)
    second = generate_bubble_sim2_cpu_outputs(80, 48, 2.5, {}, source)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert not np.array_equal(first["image_out"], source)


def test_unified_parameters_affect_cpu_output():
    baseline = generate_bubble_sim2_cpu_outputs(96, 64, 1.75, {}, None)

    for name, value in (
        ("drag", 0.1),
        ("depth_motion", 1.8),
        ("orbit_strength", 1.8),
        ("perspective", 0.1),
    ):
        changed = generate_bubble_sim2_cpu_outputs(96, 64, 1.75, {name: value}, None)
        assert not np.array_equal(changed["bubble_mask"], baseline["bubble_mask"])


def test_runtime_dispatch_uses_cpu_without_worker_state():
    outputs = runtime.generate_node_outputs(
        "Bubble Sim Gen (2)",
        80,
        48,
        1.0,
        {},
        worker_state=None,
        node_key="test-node",
        input_frames={},
    )

    assert set(BUFFER_NAMES).issubset(outputs)
    assert outputs["image_out"].shape == (48, 80, 3)


def test_default_beauty_has_visible_motion_over_one_second():
    first = generate_bubble_sim2_cpu_outputs(160, 90, 0.0, {}, None)["image_out"]
    second = generate_bubble_sim2_cpu_outputs(160, 90, 1.0, {}, None)["image_out"]

    mean_difference = np.abs(first.astype(np.int16) - second.astype(np.int16)).mean()
    assert mean_difference > 1.0


def test_primary_simulation_controls_have_visible_effect():
    baseline = generate_bubble_sim2_cpu_outputs(160, 90, 2.0, {}, None)
    controls = {
        "buoyancy": 1.5,
        "drag": 0.1,
        "turbulence": 1.5,
        "flow_speed": 2.5,
        "depth_motion": 1.8,
        "orbit_strength": 1.8,
        "show_chaos": 1.0,
    }

    for name, value in controls.items():
        changed = generate_bubble_sim2_cpu_outputs(160, 90, 2.0, {name: value}, None)
        mask_difference = np.abs(
            changed["bubble_mask"].astype(np.int16)
            - baseline["bubble_mask"].astype(np.int16)
        ).mean()
        assert mask_difference > 2.0, name


def test_motion_styles_are_visually_distinct():
    outputs = {
        style: generate_bubble_sim2_cpu_outputs(
            160,
            90,
            2.0,
            {"motion_style": style},
            None,
        )["bubble_mask"]
        for style in ("Fluid Drift", "Rising", "Vortex", "Burst")
    }

    assert len({frame.tobytes() for frame in outputs.values()}) == 4


def test_abyssal_multi_input_passes_named_aux_frames(monkeypatch):
    captured = {}

    def fake_apply(
        label,
        image,
        params,
        previous,
        feedback,
        trail,
        frame_buffer,
        worker_state=None,
    ):
        captured["label"] = label
        captured["params"] = params
        return image, previous, feedback, trail

    monkeypatch.setattr(runtime, "apply_node_effect", fake_apply)
    image = np.full((8, 8, 3), 20, dtype=np.uint8)
    mask = np.full((8, 8, 3), 200, dtype=np.uint8)
    pipeline = [
        {"node_type": "source", "source_key": "primary", "effect_type": "Source"},
        {"node_type": "source", "source_key": "secondary", "effect_type": "Secondary Source"},
        {
            "node_type": "abyssal_multi_input",
            "effect_type": "Abyssal Signal Feedback",
            "params": {},
            "node_id": "abyssal",
            "input_order": ["image_in", "bubble_mask"],
        },
    ]

    output, *_ = runtime.execute_flat_node_pipeline(
        pipeline,
        image,
        secondary_frame=mask,
        worker_state=SimpleNamespace(effect_states={}),
    )

    assert np.array_equal(output, image)
    assert captured["label"] == "Abyssal Signal Feedback"
    assert np.array_equal(captured["params"]["_aux_input_frames"]["bubble_mask"], mask)


def test_frame_state_reset_destroys_persistent_runtime():
    class Resource:
        destroyed = False

        def destroy(self):
            self.destroyed = True

    resource = Resource()
    state = FrameState(effect_states={"bubble_sim2::node": {"runtime": resource}})
    state.reset()

    assert resource.destroyed
    assert state.effect_states == {}
