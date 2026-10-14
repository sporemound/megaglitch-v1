from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from megaglitch.mycotroph_session_runtime import (
    MASTER_TRANSITION_MODES,
    MycotrophSessionError,
    MycotrophSessionRuntime,
)
from megaglitch.session_document import normalize_session_document
from node_pipeline_runtime import blend_decks


def _node_graph(label: str | None = None, params: dict | None = None) -> dict:
    if label is None:
        nodes = []
        links = []
    else:
        backend = {
            "Invert Colors": "InvertColorNode",
            "Feedback / Trail": "FeedbackTrailNode",
        }[label]
        node_id = f"{backend}:1"
        nodes = [
            {
                "id": node_id,
                "type": backend,
                "label": label,
                "params": {"enabled": True, **(params or {})},
            }
        ]
        links = [
            {
                "from_node": "SOURCE",
                "to_node": label,
                "to_node_id": node_id,
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": label,
                "from_node_id": node_id,
                "to_node": "DISPLAY",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ]
    return {"version": "1.0", "nodes": nodes, "links": links}


def _direct_graph() -> dict:
    return {
        "version": "1.0",
        "nodes": [],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "DISPLAY",
                "from_attr": "image_out",
                "to_attr": "image_in",
            }
        ],
    }


def _legacy_session(
    deck_a: dict | None = None,
    deck_b: dict | None = None,
    *,
    mix: float = 0.5,
    transition: str = "Normal Mix",
    **extra,
) -> dict:
    session = {
        "version": "2.0",
        "preset_type": "dual_deck_session",
        "decks": {
            "a": deepcopy(deck_a if deck_a is not None else _direct_graph()),
            "b": deepcopy(deck_b if deck_b is not None else _direct_graph()),
        },
        "master_mix": mix,
        "master_transition": transition,
    }
    session.update(extra)
    return session


def _plain(value):
    return json.loads(json.dumps(value))


def test_pass_through_uses_explicit_primary_and_secondary_namespaces():
    runtime = MycotrophSessionRuntime(_legacy_session())
    primary = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    secondary = np.flip(primary, axis=1).copy()

    result = runtime.render(primary, secondary, 17, 1.25, {"event_hazard": 0.2})

    np.testing.assert_array_equal(result.deck_a, primary)
    np.testing.assert_array_equal(result.deck_b, secondary)
    np.testing.assert_array_equal(
        result.program,
        blend_decks(primary, secondary, 0.5, "Normal Mix"),
    )
    assert result.program.flags.c_contiguous
    assert result.program.dtype == np.uint8
    assert result.telemetry["sequence"] == 17
    assert result.telemetry["time_seconds"] == 1.25
    assert result.telemetry["program"]["pixel_format"] == "RGB8"


def test_invert_is_executed_independently_on_each_deck():
    invert = _node_graph("Invert Colors", {"invert_chance": 100})
    runtime = MycotrophSessionRuntime(_legacy_session(invert, invert, mix=0.37))
    primary = np.array([[[0, 11, 255], [42, 99, 201]]], dtype=np.uint8)
    secondary = np.array([[[1, 2, 3], [210, 220, 230]]], dtype=np.uint8)

    result = runtime.render(primary, secondary, 1, 0.0)

    np.testing.assert_array_equal(result.deck_a, 255 - primary)
    np.testing.assert_array_equal(result.deck_b, 255 - secondary)
    np.testing.assert_array_equal(
        result.program,
        blend_decks(255 - primary, 255 - secondary, 0.37, "Normal Mix"),
    )


@pytest.mark.parametrize("mode", MASTER_TRANSITION_MODES)
@pytest.mark.parametrize("mix", (0.0, 0.5, 1.0))
def test_every_master_transition_matches_existing_blender_at_endpoints_and_midpoint(
    mode, mix
):
    primary = np.arange(15 * 19 * 3, dtype=np.uint8).reshape(15, 19, 3)
    secondary = np.roll(255 - primary, 4, axis=1)
    runtime = MycotrophSessionRuntime(
        _legacy_session(mix=mix, transition=mode)
    )

    result = runtime.render(primary, secondary, 4, 0.125)
    expected = blend_decks(primary, secondary, mix, mode)

    np.testing.assert_array_equal(result.program, expected)
    if mix == 0.0:
        np.testing.assert_array_equal(result.program, result.deck_a)
    elif mix == 1.0:
        np.testing.assert_array_equal(result.program, result.deck_b)


def test_rejected_candidate_preserves_last_session_generations_and_temporal_state():
    feedback = _node_graph(
        "Feedback / Trail",
        {"feedback_intensity": 0.5, "trailing_intensity": 0.0},
    )
    accepted = _legacy_session(feedback, feedback)
    runtime = MycotrophSessionRuntime(accepted)
    runtime.render(
        np.full((3, 5, 3), 20, np.uint8),
        np.full((3, 5, 3), 180, np.uint8),
        1,
        0.0,
    )
    before_document = runtime.session_document
    before_snapshot = runtime.snapshot()

    rejected = _legacy_session(feedback, feedback)
    rejected["decks"]["a"]["links"][-1]["to_node"] = "MISSING"
    with pytest.raises(MycotrophSessionError) as caught:
        runtime.replace_session(rejected)

    assert caught.value.code == "deck_graph_invalid"
    assert caught.value.details["deck"] == "a"
    assert caught.value.details["graph_code"] == "missing_endpoint"
    assert runtime.session_document == before_document
    assert runtime.snapshot() == before_snapshot


def test_decks_keep_separate_temporal_state_and_only_changed_deck_is_reset():
    feedback_a = _node_graph(
        "Feedback / Trail",
        {"feedback_intensity": 0.5, "trailing_intensity": 0.0},
    )
    feedback_b = deepcopy(feedback_a)
    runtime = MycotrophSessionRuntime(_legacy_session(feedback_a, feedback_b))

    runtime.render(
        np.full((2, 4, 3), 20, np.uint8),
        np.full((2, 4, 3), 180, np.uint8),
        1,
        0.0,
    )
    second = runtime.render(
        np.full((2, 4, 3), 80, np.uint8),
        np.full((2, 4, 3), 220, np.uint8),
        2,
        1 / 60,
    )
    assert int(second.deck_a[0, 0, 0]) == 50
    assert int(second.deck_b[0, 0, 0]) == 200

    changed_a = _node_graph(
        "Feedback / Trail",
        {"feedback_intensity": 0.25, "trailing_intensity": 0.0},
    )
    runtime.replace_session(_legacy_session(changed_a, feedback_b))
    after_commit = runtime.snapshot()

    assert after_commit["decks"]["a"]["graph_generation"] == 2
    assert after_commit["decks"]["b"]["graph_generation"] == 1
    assert after_commit["decks"]["a"]["temporal"]["frame_count"] == 0
    assert after_commit["decks"]["b"]["temporal"]["frame_count"] == 2

    third = runtime.render(
        np.full((2, 4, 3), 100, np.uint8),
        np.full((2, 4, 3), 100, np.uint8),
        3,
        2 / 60,
    )
    # Deck A has no retained feedback after its graph change.  Deck B retains
    # its independent 200-valued feedback frame from the preceding render.
    assert int(third.deck_a[0, 0, 0]) == 100
    assert int(third.deck_b[0, 0, 0]) == 150


def test_master_only_replacement_keeps_graph_generations_and_temporal_owners():
    feedback = _node_graph(
        "Feedback / Trail",
        {"feedback_intensity": 0.25, "trailing_intensity": 0.0},
    )
    runtime = MycotrophSessionRuntime(_legacy_session(feedback, feedback))
    runtime.render(
        np.full((2, 2, 3), 10, np.uint8),
        np.full((2, 2, 3), 240, np.uint8),
        1,
        0.0,
    )

    runtime.replace_session(
        _legacy_session(feedback, feedback, mix=0.75, transition="Screen")
    )
    snapshot = runtime.snapshot()

    assert runtime.session_revision == 2
    assert runtime.deck_generations == {"a": 1, "b": 1}
    assert snapshot["decks"]["a"]["temporal"]["frame_count"] == 1
    assert snapshot["decks"]["b"]["temporal"]["frame_count"] == 1


def test_legacy_dual_deck_normalizes_and_canonical_unknown_fields_are_preserved():
    legacy = _legacy_session(
        _node_graph("Invert Colors", {"invert_chance": 100}),
        _direct_graph(),
        vendor_legacy={"token": [1, 2, 3]},
    )
    runtime = MycotrophSessionRuntime(legacy)
    canonical = runtime.session_document

    assert canonical["format"] == "megaglitch.session"
    assert canonical["format_version"] == 2
    assert canonical["decks"]["b"]["source_bindings"]["SOURCE"] == "secondary"
    assert canonical["extensions"]["legacy"]["unmapped_root"]["vendor_legacy"] == {
        "token": [1, 2, 3]
    }

    canonical["vendor_canonical"] = {"future": True}
    canonical["decks"]["a"]["vendor_deck"] = "keep"
    canonical_before = deepcopy(canonical)
    runtime.replace_session(canonical)

    assert canonical == canonical_before
    assert runtime.session_document["vendor_canonical"] == {"future": True}
    assert runtime.session_document["decks"]["a"]["vendor_deck"] == "keep"


def test_prepared_session_and_returned_documents_cannot_mutate_accepted_state():
    source = _legacy_session()
    source_before = deepcopy(source)
    prepared = MycotrophSessionRuntime.prepare_session(source)
    with pytest.raises(TypeError):
        prepared.canonical["master"] = {"mix": 1.0}

    runtime = MycotrophSessionRuntime()
    runtime.commit_session(prepared)
    assert source == source_before
    detached = runtime.session_document
    detached["master"]["mix"] = 1.0

    assert runtime.session_document["master"]["mix"] == 0.5


def test_telemetry_is_deterministic_and_contains_no_wall_clock_measurement():
    document = normalize_session_document(_legacy_session())
    a = MycotrophSessionRuntime(document)
    b = MycotrophSessionRuntime(document)
    primary = np.full((3, 7, 3), 33, np.uint8)
    secondary = np.full((3, 7, 3), 177, np.uint8)

    left = a.render(primary, secondary, 9, 0.75, {"history_offset": 2.0})
    right = b.render(primary, secondary, 9, 0.75, {"history_offset": 2.0})

    np.testing.assert_array_equal(left.program, right.program)
    assert _plain(left.telemetry) == _plain(right.telemetry)
    assert "frame_ms" not in left.telemetry
    assert left.telemetry["program"]["sha256"]
