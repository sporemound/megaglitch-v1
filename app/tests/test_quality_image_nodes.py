from types import SimpleNamespace

import numpy as np
import pytest

import effects_registry
from effects_registry import (
    apply_edge_reconstruction,
    apply_node_effect,
    apply_quantization_solarization,
)
from node_pipeline_runtime import execute_flat_node_pipeline
from tools.export_rust_node_catalog import build_specs


def _dispatch(label, frame, params, *, worker_state=None):
    return apply_node_effect(
        label,
        frame,
        params,
        previous_frame=None,
        feedback_image=None,
        trail_image=None,
        frame_buffer=[],
        worker_state=worker_state,
    )[0]


@pytest.mark.parametrize(
    ("label", "params"),
    [
        (
            "Quantization / Solarization",
            {
                "quantization_levels": 5,
                "solarization_threshold": 150,
                "solarization_blend": 0.4,
            },
        ),
        (
            "Edge Reconstruction",
            {"edge_luma_reinforcement": 0.12, "edge_sensitivity": 0.04},
        ),
    ],
)
def test_quality_nodes_are_deterministic_rgb8_master_resolution_operations(label, params):
    rng = np.random.default_rng(20260827)
    backing = rng.integers(0, 256, size=(47, 118, 3), dtype=np.uint8)
    frame = backing[:, ::2]
    assert not frame.flags.c_contiguous

    first = _dispatch(label, frame, {"enabled": True, **params})
    second = _dispatch(label, frame, {"enabled": True, **params})

    assert first.shape == frame.shape
    assert first.dtype == np.uint8
    assert first.flags.c_contiguous
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "label",
    ["Quantization / Solarization", "Edge Reconstruction"],
)
def test_disabled_quality_nodes_are_exact_bypasses(label):
    frame = np.arange(19 * 31 * 3, dtype=np.uint8).reshape((19, 31, 3))

    output = _dispatch(label, frame, {"enabled": False})

    assert output is frame
    assert np.array_equal(output, frame)


def test_quantization_uses_requested_palette_and_luminance_solar_threshold():
    ramp = np.arange(256, dtype=np.uint8)[None, :, None]
    frame = np.repeat(ramp, 3, axis=2)

    quantized = apply_quantization_solarization(
        frame,
        quantization_levels=4,
        solarization_blend=0.0,
    )
    assert np.array_equal(np.unique(quantized), np.array([0, 85, 170, 255]))

    color = np.array([[[240, 220, 80], [80, 160, 240]]], dtype=np.uint8)
    solarized = apply_quantization_solarization(
        color,
        quantization_levels=256,
        solarization_threshold=180,
        solarization_blend=1.0,
    )
    assert np.array_equal(solarized[0, 0], 255 - color[0, 0])
    assert np.array_equal(solarized[0, 1], color[0, 1])


def test_identity_settings_preserve_pixels_without_resizing():
    rng = np.random.default_rng(17)
    frame = rng.integers(0, 256, size=(23, 37, 3), dtype=np.uint8)

    quantized = apply_quantization_solarization(
        frame,
        quantization_levels=256,
        solarization_blend=0.0,
    )
    reconstructed = apply_edge_reconstruction(
        frame,
        edge_luma_reinforcement=0.0,
    )

    assert np.array_equal(quantized, frame)
    assert np.array_equal(reconstructed, frame)
    assert quantized.shape == reconstructed.shape == frame.shape


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 7, 3), (9, 1, 3)])
def test_quality_nodes_keep_tiny_and_single_axis_master_shapes(shape):
    frame = np.zeros(shape, dtype=np.uint8)

    quantized = apply_quantization_solarization(frame)
    reconstructed = apply_edge_reconstruction(frame)

    assert quantized.shape == reconstructed.shape == shape


def test_edge_reconstruction_changes_only_edge_derived_luminance():
    luma = np.full((32, 48), 48, dtype=np.uint8)
    luma[:, 24:] = 144
    frame = np.stack([luma, luma, luma], axis=2)

    reconstructed = apply_edge_reconstruction(
        frame,
        edge_luma_reinforcement=0.20,
        edge_sensitivity=0.01,
    )

    assert np.array_equal(reconstructed[:, :22], frame[:, :22])
    assert np.array_equal(reconstructed[:, 26:], frame[:, 26:])
    assert np.any(reconstructed[:, 22:26] != frame[:, 22:26])
    assert np.array_equal(reconstructed[..., 0], reconstructed[..., 1])
    assert np.array_equal(reconstructed[..., 1], reconstructed[..., 2])


def test_quality_nodes_ignore_temporal_macro_mutation_and_keep_no_state():
    frame = np.tile(np.arange(40, dtype=np.uint8), (24, 1))
    frame = np.stack([frame, frame, frame], axis=2)
    worker_state = SimpleNamespace(effect_states={})
    temporal_macros = {
        "history_offset": 12.0,
        "memory_decay": 0.6,
        "event_hazard": 0.9,
        "clock_quantization": 0.2,
        "sync_desync": 0.8,
        "phase_reset_chance": 0.4,
        "local_time_warp": 1.5,
        "rupture_threshold": 0.1,
        "hysteresis_speed": 0.8,
        "modulator_coupling": 0.9,
    }
    cases = [
        (
            "Quantization / Solarization",
            {"quantization_levels": 8, "solarization_blend": 0.0},
        ),
        (
            "Edge Reconstruction",
            {"edge_luma_reinforcement": 0.1, "edge_sensitivity": 0.02},
        ),
    ]

    for label, params in cases:
        baseline = _dispatch(label, frame, {"enabled": True, **params})
        mutated = _dispatch(
            label,
            frame,
            {"enabled": True, **params, **temporal_macros},
            worker_state=worker_state,
        )
        assert np.array_equal(mutated, baseline)

    assert worker_state.effect_states == {}
    assert {
        "Quantization / Solarization",
        "Edge Reconstruction",
    } <= effects_registry._TEMPORAL_MACRO_EXCLUDED_LABELS


def test_quality_nodes_execute_as_one_strict_resolution_pipeline():
    rng = np.random.default_rng(84)
    frame = rng.integers(0, 256, size=(35, 57, 3), dtype=np.uint8)
    pipeline = [
        {
            "effect_type": "Source",
            "node_type": "source",
            "source_key": "primary",
        },
        {
            "effect_type": "Quantization / Solarization",
            "node_id": "quantize",
            "params": {
                "enabled": True,
                "quantization_levels": 12,
                "solarization_threshold": 196,
                "solarization_blend": 0.15,
            },
        },
        {
            "effect_type": "Edge Reconstruction",
            "node_id": "edges",
            "params": {
                "enabled": True,
                "edge_luma_reinforcement": 0.04,
                "edge_sensitivity": 0.05,
            },
        },
        {"effect_type": "Display", "node_type": "display", "params": {}},
    ]

    first, *_ = execute_flat_node_pipeline(pipeline, frame)
    second, *_ = execute_flat_node_pipeline(pipeline, frame)

    assert first.shape == frame.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first, frame)


@pytest.mark.parametrize(
    "operation",
    [apply_quantization_solarization, apply_edge_reconstruction],
)
def test_quality_implementations_reject_non_rgb8_domains(operation):
    with pytest.raises(TypeError, match="uint8"):
        operation(np.zeros((4, 6, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="HxWx3"):
        operation(np.zeros((4, 6, 4), dtype=np.uint8))


def test_exported_quality_node_specs_have_explicit_restrained_controls():
    specs = {spec.backend_id: spec for spec in build_specs()}
    quantization = specs["QuantizationSolarizationNode"]
    edge = specs["EdgeReconstructionNode"]

    assert quantization.label == "Quantization / Solarization"
    assert edge.label == "Edge Reconstruction"
    assert quantization.category == edge.category == "Analysis & Quality"
    assert {param.name for param in quantization.parameters} >= {
        "enabled",
        "quantization_levels",
        "solarization_threshold",
        "solarization_blend",
    }
    edge_params = {param.name for param in edge.parameters}
    assert edge_params >= {"enabled", "edge_luma_reinforcement", "edge_sensitivity"}
    assert "unsharp_amount" not in edge_params
    assert "local_contrast" not in edge_params
