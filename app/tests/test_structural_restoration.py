import cv2
import numpy as np

from effects_registry import apply_node_effect
from megaglitch.structural_restoration import apply_structural_restoration
from tools.export_rust_node_catalog import build_specs


ZERO_RESTORATION = {
    "unsharp_amount": 0.0,
    "local_contrast": 0.0,
    "edge_luma_reinforcement": 0.0,
    "grain_normalization": 0.0,
    "final_clamp_low": 0,
    "final_clamp_high": 255,
}


def _dispatch(frame, params):
    return apply_node_effect(
        "Structural Restoration",
        frame,
        params,
        previous_frame=None,
        feedback_image=None,
        trail_image=None,
        frame_buffer=[],
    )[0]


def _laplacian_variance(frame):
    luma = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(luma, cv2.CV_32F).var())


def test_zero_settings_are_exact_master_resolution_identity_and_contiguous():
    rng = np.random.default_rng(41)
    backing = rng.integers(0, 256, size=(53, 142, 3), dtype=np.uint8)
    frame = backing[:, ::2]
    assert not frame.flags.c_contiguous

    restored = apply_structural_restoration(frame, **ZERO_RESTORATION)

    assert restored.shape == frame.shape
    assert restored.dtype == frame.dtype
    assert restored.flags.c_contiguous
    assert np.array_equal(restored, frame)


def test_disabled_registry_node_is_an_exact_bypass():
    frame = np.arange(24 * 32 * 3, dtype=np.uint8).reshape((24, 32, 3))

    restored = _dispatch(frame, {"enabled": False})

    assert restored is frame
    assert np.array_equal(restored, frame)


def test_extreme_restoration_is_bounded_and_never_resizes():
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, size=(61, 97, 3), dtype=np.uint8)

    restored = apply_structural_restoration(
        frame,
        unsharp_amount=10.0,
        unsharp_radius=10.0,
        local_contrast=10.0,
        local_contrast_radius=100.0,
        edge_luma_reinforcement=10.0,
        edge_sensitivity=0.0,
        grain_normalization=10.0,
        grain_target=1.0,
        final_clamp_low=18,
        final_clamp_high=238,
    )

    assert restored.shape == frame.shape
    assert restored.dtype == np.uint8
    assert restored.flags.c_contiguous
    assert int(restored.min()) >= 18
    assert int(restored.max()) <= 238


def test_restrained_unsharp_and_local_contrast_restore_soft_structure():
    source = np.full((96, 128, 3), 116, dtype=np.uint8)
    source[:, 40:88] = 140
    softened = cv2.GaussianBlur(source, (0, 0), 2.4)

    restored = apply_structural_restoration(
        softened,
        unsharp_amount=0.55,
        unsharp_radius=1.2,
        local_contrast=0.35,
        local_contrast_radius=9.0,
        edge_luma_reinforcement=0.12,
        grain_normalization=0.0,
    )

    assert _laplacian_variance(restored) > _laplacian_variance(softened)
    assert float(restored.std()) > float(softened.std())


def test_grain_normalization_is_repeatable_without_random_synthesis():
    rng = np.random.default_rng(2026)
    base = np.full((72, 104, 3), 128, dtype=np.int16)
    grain = rng.integers(-18, 19, size=base.shape, dtype=np.int16)
    frame = np.clip(base + grain, 0, 255).astype(np.uint8)
    params = dict(
        unsharp_amount=0.2,
        local_contrast=0.1,
        edge_luma_reinforcement=0.08,
        grain_normalization=0.75,
        grain_target=0.015,
    )

    first = apply_structural_restoration(frame, **params)
    second = apply_structural_restoration(frame, **params)

    assert np.array_equal(first, second)


def test_registry_dispatch_matches_the_pure_cpu_implementation():
    rng = np.random.default_rng(9)
    frame = rng.integers(0, 256, size=(48, 80, 3), dtype=np.uint8)
    params = {
        "enabled": True,
        "unsharp_amount": 0.27,
        "unsharp_radius": 1.3,
        "local_contrast": 0.18,
        "local_contrast_radius": 7.0,
        "edge_luma_reinforcement": 0.09,
        "edge_sensitivity": 0.04,
        "grain_normalization": 0.25,
        "grain_target": 0.01,
        "final_clamp_low": 4,
        "final_clamp_high": 250,
    }

    expected = apply_structural_restoration(
        frame,
        **{key: value for key, value in params.items() if key != "enabled"},
    )
    actual = _dispatch(frame, params)

    assert np.array_equal(actual, expected)


def test_exported_node_spec_is_truthfully_cpu_and_final_stage_capable():
    spec = next(
        item for item in build_specs() if item.backend_id == "StructuralRestorationNode"
    )
    params = {param.name: param for param in spec.parameters}

    assert spec.label == "Structural Restoration"
    assert spec.backend == "cpu"
    assert [socket.name for socket in spec.inputs] == ["image_in"]
    assert [socket.name for socket in spec.outputs] == ["image_out"]
    assert {
        "unsharp_amount",
        "local_contrast",
        "edge_luma_reinforcement",
        "grain_normalization",
        "final_clamp_low",
        "final_clamp_high",
    } <= params.keys()
