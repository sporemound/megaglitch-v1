import numpy as np
import pytest

from node_pipeline_runtime import blend_decks, execute_flat_node_pipeline


def test_left_endpoint_excludes_deck_b():
    deck_a = np.full((2, 3, 3), 17, dtype=np.uint8)
    deck_b = np.full((2, 3, 3), 231, dtype=np.uint8)

    output = blend_decks(deck_a, deck_b, mix_ratio=0.0)

    assert np.array_equal(output, deck_a)


def test_left_endpoint_is_black_when_deck_a_is_missing():
    deck_b = np.full((2, 3, 3), 231, dtype=np.uint8)

    output = blend_decks(None, deck_b, mix_ratio=0.0)

    assert np.array_equal(output, np.zeros_like(deck_b))


def test_right_endpoint_is_black_when_deck_b_is_missing():
    deck_a = np.full((2, 3, 3), 17, dtype=np.uint8)

    output = blend_decks(deck_a, None, mix_ratio=1.0)

    assert np.array_equal(output, np.zeros_like(deck_a))


def test_near_left_endpoint_snaps_to_deck_a():
    deck_a = np.full((2, 3, 3), 17, dtype=np.uint8)
    deck_b = np.full((2, 3, 3), 231, dtype=np.uint8)

    output = blend_decks(deck_a, deck_b, mix_ratio=0.00001)

    assert np.array_equal(output, deck_a)


def test_deck_a_primary_source_does_not_receive_deck_b():
    deck_a = np.full((2, 3, 3), 17, dtype=np.uint8)
    deck_b = np.full((2, 3, 3), 231, dtype=np.uint8)
    pipeline = [
        {
            "effect_type": "Source",
            "params": {},
            "node_type": "source",
            "source_key": "primary",
        },
        {"effect_type": "Display", "params": {}},
    ]

    output, *_ = execute_flat_node_pipeline(
        pipeline,
        deck_a,
        secondary_frame=deck_b,
    )

    assert np.array_equal(output, deck_a)


def test_luma_wipe_reveals_more_deck_b_as_mix_increases():
    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    deck_a = np.repeat(gradient[None, :, None], 3, axis=2)
    deck_b = np.full_like(deck_a, 255)

    early = blend_decks(deck_a, deck_b, mix_ratio=0.25, mode="Luma Wipe")
    late = blend_decks(deck_a, deck_b, mix_ratio=0.75, mode="Luma Wipe")

    assert late.mean() > early.mean()


def test_all_crossfader_modes_return_valid_frames():
    deck_a = np.full((24, 32, 3), 40, dtype=np.uint8)
    deck_b = np.full((24, 32, 3), 210, dtype=np.uint8)
    modes = [
        "Normal Mix",
        "Luma Wipe",
        "Glitch Dissolve",
        "Horizontal Wipe",
        "Vertical Wipe",
        "Radial Wipe",
        "Checker Wipe",
        "Hard Cut",
        "Additive",
        "Multiply",
        "Screen",
        "Difference",
    ]

    for mode in modes:
        output = blend_decks(deck_a, deck_b, mix_ratio=0.5, mode=mode)
        assert output.shape == deck_a.shape
        assert output.dtype == np.uint8


def test_crossfader_rejects_mismatched_deck_resolution_instead_of_scaling():
    deck_a = np.zeros((24, 32, 3), dtype=np.uint8)
    preview_sized_deck_b = np.zeros((12, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="must share master shape"):
        blend_decks(deck_a, preview_sized_deck_b, mix_ratio=0.5)
