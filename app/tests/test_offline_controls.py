import random

from megaglitch.offline_controls import (
    FrozenModulation,
    apply_frozen_timeline,
    evaluate_timeline_track,
)


def test_timeline_evaluation_is_pure_and_interpolated() -> None:
    track = {
        "type": "float",
        "keyframes": [
            {"time": 0.0, "value": 0.0, "interp": "linear"},
            {"time": 2.0, "value": 1.0, "interp": "linear"},
        ],
    }
    assert evaluate_timeline_track(track, 1.0) == 0.5


def test_frozen_timeline_targets_serialized_node_id() -> None:
    pipeline = [{"node_id": "node", "params": {"amount": 0.0}}]
    tracks = {
        "node:amount": {
            "node_id": "node",
            "param": "amount",
            "type": "float",
            "keyframes": [{"time": 0, "value": 0.75}],
        }
    }
    apply_frozen_timeline(pipeline, tracks, 1.0, enabled=True)
    assert pipeline[0]["params"]["amount"] == 0.75


def test_frozen_random_modulation_repeats_for_same_seed() -> None:
    modulation = FrozenModulation(
        enabled=True,
        connections=("Random -> amount",),
        parameter_ranges={"amount": (0.0, 1.0)},
        parameter_types={"amount": "float"},
        parameter_items={},
        parameter_modes={"amount": "replace"},
        parameter_spans={},
        source_values={},
        target_deck="a",
    )
    first = modulation.apply(
        {"amount": 0.0}, seconds=0.0, time_scale=1.0, rng=random.Random(9)
    )
    second = modulation.apply(
        {"amount": 0.0}, seconds=0.0, time_scale=1.0, rng=random.Random(9)
    )
    assert first == second
