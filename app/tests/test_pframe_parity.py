import random
import numpy as np
import pytest
import glitch_effects as ge


def np_state_equal(s1, s2):
    return s1[0] == s2[0] and np.array_equal(s1[1], s2[1]) and s1[2:] == s2[2:]


def test_corrupt_motion_vectors_preserves_rng_and_output():
    h, w = 1080, 1920
    current = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
    reference = np.random.RandomState(43).randint(0, 256, (h, w, 3), dtype=np.uint8)

    random.seed(917)
    py_state_before = random.getstate()
    np_state_before = np.random.get_state()

    out = ge.corrupt_motion_vectors(current, reference, 0.07, block_size=16, direction="wave")

    py_state_after = random.getstate()
    np_state_after = np.random.get_state()

    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert py_state_before != py_state_after  # RNG was consumed deterministically
    assert np_state_equal(np_state_before, np_state_after)  # NumPy RNG untouched


def test_temporal_bleeding_preserves_rng_and_output():
    h, w = 1080, 1920
    buf = [np.random.RandomState(i).randint(0, 256, (h, w, 3), dtype=np.uint8) for i in range(5)]

    np.random.seed(917)
    py_state_before = random.getstate()
    np_state_before = np.random.get_state()

    out = ge.temporal_bleeding_effect(buf, 0.06, bleed_distance=2, direction="forward")

    py_state_after = random.getstate()
    np_state_after = np.random.get_state()

    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert py_state_before == py_state_after  # Python random untouched
    assert not np_state_equal(np_state_before, np_state_after)  # NumPy RNG consumed deterministically


def test_macroblock_scrambling_preserves_rng_and_output():
    h, w = 1080, 1920
    current = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
    reference = np.random.RandomState(43).randint(0, 256, (h, w, 3), dtype=np.uint8)

    random.seed(917)
    py_state_before = random.getstate()
    np_state_before = np.random.get_state()

    out = ge.macroblock_scrambling(current, reference, 0.025, block_size=16, pattern="horizontal_shift")

    py_state_after = random.getstate()
    np_state_after = np.random.get_state()

    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert py_state_before != py_state_after
    assert np_state_equal(np_state_before, np_state_after)


def test_residual_injection_preserves_rng_and_output():
    h, w = 1080, 1920
    current = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)

    np.random.seed(917)
    py_state_before = random.getstate()
    np_state_before = np.random.get_state()

    out = ge.residual_injection(current, 0.025, noise_type="gaussian", direction="uniform")

    py_state_after = random.getstate()
    np_state_after = np.random.get_state()

    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert py_state_before == py_state_after
    assert not np_state_equal(np_state_before, np_state_after)
