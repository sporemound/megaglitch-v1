"""Compare the optimized CPU path to the retained full-precision reference."""

import numpy as np
import pytest

import glitch_effects as ge


@pytest.mark.parametrize("hue", [0.0, 30.0, -60.0, 180.0])
@pytest.mark.parametrize("saturation", [0.0, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("gamma", [0.1, 0.7, 1.0, 1.2, 2.0, 3.0])
def test_rgb8_color_adjust_matches_reference_bytes(hue, saturation, gamma):
    rng = np.random.default_rng(481)
    frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    # Include every byte value even when HSV is bypassed, plus color samples.
    frame[:4, :64] = np.arange(256, dtype=np.uint8).reshape(4, 64, 1)
    original = frame.copy()
    expected = ge.apply_gamma(
        ge.apply_hsv_adjust(frame, hue_shift=hue, saturation=saturation), gamma=gamma
    )
    expected = np.clip(expected, 0, 255).astype(np.uint8)
    actual = ge.apply_color_adjust_rgb8(frame, hue, saturation, gamma)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frame, original)


def test_color_adjust_parameter_changes_do_not_reuse_stale_tables():
    frame = np.random.default_rng(54).integers(0, 256, (36, 64, 3), dtype=np.uint8)
    for index in list(range(40)) + [0, 39, 10]:
        hue, saturation, gamma = index - 20.0, index / 20.0, 0.1 + index / 15.0
        expected = ge.apply_gamma(ge.apply_hsv_adjust(frame, hue, saturation), gamma)
        actual = ge.apply_color_adjust_rgb8(frame, hue, saturation, gamma)
        np.testing.assert_array_equal(actual, np.clip(expected, 0, 255).astype(np.uint8))


def test_noncontiguous_rgb8_color_adjust_matches_reference():
    frame = np.random.default_rng(55).integers(0, 256, (72, 128, 3), dtype=np.uint8)[::2, ::2]
    expected = ge.apply_gamma(ge.apply_hsv_adjust(frame, 30, 0.5), 1.2)
    actual = ge.apply_color_adjust_rgb8(frame, 30, 0.5, 1.2)
    np.testing.assert_array_equal(actual, np.clip(expected, 0, 255).astype(np.uint8))
