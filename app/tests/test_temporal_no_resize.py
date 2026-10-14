import numpy as np

import effects_registry
from glitch_effects import (
    amplify_prediction_errors,
    corrupt_motion_vectors,
    macroblock_scrambling,
    temporal_bleeding_effect,
)


def test_temporal_macro_history_discards_non_master_frames():
    small = np.full((3, 4, 3), 200, np.uint8)
    master = np.full((6, 8, 3), 40, np.uint8)

    assert effects_registry._sample_temporal_history(
        [small], None, 0.0, (6, 8)
    ) is None
    sampled = effects_registry._sample_temporal_history(
        [small, master], None, 0.0, (6, 8)
    )
    assert np.array_equal(sampled, master)


def test_pframe_operations_bypass_or_reseed_mismatched_references():
    master = np.full((8, 12, 3), 80, np.uint8)
    preview = np.full((4, 6, 3), 220, np.uint8)

    assert corrupt_motion_vectors(master, preview, 1.0) is master
    assert amplify_prediction_errors(master, preview, 1.0) is master
    scrambled = macroblock_scrambling(master, preview, 1.0, block_size=4)
    assert scrambled.shape == master.shape


def test_temporal_bleed_skips_mismatched_history_without_scaling_it():
    preview = np.full((4, 6, 3), 255, np.uint8)
    master = np.full((8, 12, 3), 25, np.uint8)

    result = temporal_bleeding_effect(
        [preview, master], intensity=1.0, bleed_distance=2
    )

    assert result.shape == master.shape
    assert np.array_equal(result, master)
