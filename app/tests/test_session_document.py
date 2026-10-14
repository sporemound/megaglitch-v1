import json
import unittest
from copy import deepcopy
from pathlib import Path

from megaglitch.session_document import (
    SESSION_FORMAT,
    SESSION_FORMAT_VERSION,
    export_legacy_dual_deck,
    load_session_document,
    normalize_session_document,
)


ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "node_presets"


def graph(nodes=None, links=None, **extra):
    value = {
        "version": "1.0",
        "nodes": list(nodes or []),
        "links": list(links or []),
    }
    value.update(extra)
    return value


def node(backend_id, label, params=None, **extra):
    value = {
        "type": backend_id,
        "label": label,
        "params": dict(params or {}),
    }
    value.update(extra)
    return value


def link(source, target, **extra):
    value = {
        "from_node": source,
        "to_node": target,
        "from_attr": "image_out",
        "to_attr": "image_in",
    }
    value.update(extra)
    return value


def dual(deck_a=None, deck_b=None, **extra):
    value = {
        "version": "2.0",
        "preset_type": "dual_deck_session",
        "decks": {"a": deck_a or graph(), "b": deck_b or graph()},
    }
    value.update(extra)
    return value


class SessionDocumentTests(unittest.TestCase):
    def assert_canonical_shape(self, session):
        self.assertEqual(session["format"], SESSION_FORMAT)
        self.assertEqual(session["format_version"], SESSION_FORMAT_VERSION)
        self.assertTrue(
            {
                "metadata",
                "decks",
                "sources",
                "source_compositor",
                "master",
                "macros",
                "modulation",
                "timeline",
                "audio",
                "transport",
                "render",
                "outputs",
                "ui",
                "extensions",
            }.issubset(session)
        )

    def test_real_graph_v1_fixture_migrates_with_stable_ids_and_links(self):
        session = load_session_document(PRESET_DIR / "5-13-26.json")

        self.assert_canonical_shape(session)
        self.assertEqual(
            session["decks"]["a"]["source_bindings"],
            {"SOURCE": "primary", "SOURCE_B": "secondary"},
        )
        self.assertEqual(
            session["decks"]["b"]["source_bindings"],
            {"SOURCE": "secondary", "SOURCE_B": "secondary"},
        )
        graph_a = session["decks"]["a"]["graph"]
        self.assertEqual(
            [item["id"] for item in graph_a["nodes"]],
            [
                "CUDAChannelDriftNode:1",
                "CUDAPersistentFeedbackNode:1",
                "CUDATemporalMedianNode:1",
            ],
        )
        self.assertEqual(graph_a["links"][0]["to_node_id"], "CUDAChannelDriftNode:1")
        self.assertEqual(
            graph_a["links"][1]["from_node_id"], "CUDAChannelDriftNode:1"
        )
        self.assertNotIn("from_node_id", graph_a["links"][0])
        self.assertEqual(session["extensions"]["legacy"]["source_format"], "graph_v1")

    def test_real_dual_deck_fixture_uses_secondary_source_for_deck_b(self):
        session = load_session_document(PRESET_DIR / "6-17-26.json")

        self.assert_canonical_shape(session)
        self.assertEqual(session["decks"]["a"]["source_bindings"]["SOURCE"], "primary")
        self.assertEqual(session["decks"]["b"]["source_bindings"]["SOURCE"], "secondary")
        self.assertEqual(session["decks"]["b"]["source_bindings"]["SOURCE_B"], "secondary")
        self.assertEqual(session["master"]["transition"], "Additive")
        self.assertAlmostEqual(session["master"]["mix"], 0.5329999923706055)
        self.assertEqual(session["decks"]["b"]["graph"]["nodes"][0]["id"], "LuaShaderNode:1")

    def test_full_28_json_corpus_normalizes_serializes_and_is_idempotent(self):
        files = sorted(PRESET_DIR.glob("*.json"))
        self.assertEqual(len(files), 28)

        for path in files:
            with self.subTest(path=path.name):
                source = json.loads(path.read_text(encoding="utf-8"))
                before = deepcopy(source)
                session = normalize_session_document(source)
                self.assertEqual(source, before)
                self.assert_canonical_shape(session)
                wire_copy = json.loads(json.dumps(session, allow_nan=False))
                self.assertEqual(wire_copy, session)
                self.assertEqual(normalize_session_document(session), session)
                exported = export_legacy_dual_deck(session)
                self.assertEqual(exported["preset_type"], "dual_deck_session")
                self.assertEqual(exported["version"], "2.0")
                self.assertIn("a", exported["decks"])
                self.assertIn("b", exported["decks"])
                json.dumps(exported, allow_nan=False)
                round_trip = normalize_session_document(exported)
                self.assertEqual(round_trip["decks"], session["decks"])
                self.assertEqual(round_trip["master"], session["master"])
                self.assertEqual(round_trip["macros"], session["macros"])
                self.assertEqual(round_trip["audio"], session["audio"])

    def test_missing_and_null_service_payloads_remain_distinct(self):
        missing = dual(future_root={"keep": [1, 2, 3]})
        explicit_null = dual(
            media_audio=None,
            mod_matrix_state=None,
            parameter_timeline_state=None,
        )

        missing_session = normalize_session_document(missing)
        null_session = normalize_session_document(explicit_null)

        self.assertEqual(missing_session["audio"], {})
        self.assertIsNone(null_session["audio"])
        self.assertIsNone(null_session["modulation"])
        self.assertIsNone(null_session["timeline"])
        self.assertEqual(
            missing_session["extensions"]["legacy"]["unmapped_root"]["future_root"],
            {"keep": [1, 2, 3]},
        )
        missing_export = export_legacy_dual_deck(missing_session)
        null_export = export_legacy_dual_deck(null_session)
        self.assertNotIn("media_audio", missing_export)
        self.assertNotIn("mod_matrix_state", missing_export)
        self.assertNotIn("parameter_timeline_state", missing_export)
        self.assertIn("media_audio", null_export)
        self.assertIsNone(null_export["media_audio"])
        self.assertIsNone(null_export["mod_matrix_state"])
        self.assertIsNone(null_export["parameter_timeline_state"])

    def test_canonical_unknown_fields_and_nulls_survive_repeated_normalization(self):
        canonical = {
            "format": SESSION_FORMAT,
            "format_version": 2,
            "metadata": {"name": "Unknown-field test", "future": 7},
            "decks": {
                "a": {
                    "source_bindings": {"SOURCE": "camera-9"},
                    "graph": graph(future_graph={"mode": "x"}),
                    "future_deck": True,
                }
            },
            "audio": None,
            "future_root": {"do_not_drop": None},
            "extensions": {"vendor.example": {"revision": 3}},
        }

        normalized = normalize_session_document(canonical)
        normalized_again = normalize_session_document(normalized)

        self.assertEqual(normalized_again, normalized)
        self.assertIsNone(normalized["audio"])
        self.assertEqual(normalized["future_root"], {"do_not_drop": None})
        self.assertEqual(normalized["extensions"]["vendor.example"]["revision"], 3)
        self.assertTrue(normalized["decks"]["a"]["future_deck"])
        self.assertEqual(
            normalized["decks"]["a"]["graph"]["future_graph"], {"mode": "x"}
        )
        self.assertEqual(normalized["decks"]["a"]["source_bindings"]["SOURCE"], "camera-9")
        self.assertEqual(normalized["decks"]["a"]["source_bindings"]["SOURCE_B"], "secondary")

    def test_ambiguous_link_labels_are_preserved_instead_of_retargeted(self):
        source = graph(
            nodes=[node("LuaShaderNode", "Lua Shader"), node("LuaShaderNode", "Lua Shader")],
            links=[link("SOURCE", "Lua Shader"), link("Lua Shader", "DISPLAY")],
        )

        session = normalize_session_document(source)
        migrated = session["decks"]["a"]["graph"]

        self.assertEqual(
            [item["id"] for item in migrated["nodes"]],
            ["LuaShaderNode:1", "LuaShaderNode:2"],
        )
        self.assertNotIn("to_node_id", migrated["links"][0])
        self.assertNotIn("from_node_id", migrated["links"][1])
        self.assertEqual(migrated["links"], source["links"])

    def test_existing_ambiguous_ids_and_dangling_explicit_references_are_not_rewritten(self):
        source = graph(
            nodes=[
                node("BlurNode", "One", id="duplicate"),
                node("BlurNode", "Two", id="duplicate"),
            ],
            links=[
                link("SOURCE", "One", to_node_id="stale-runtime-id"),
                link("One", "DISPLAY", from_node_id="duplicate"),
            ],
        )

        migrated = normalize_session_document(source)["decks"]["a"]["graph"]

        self.assertEqual([item["id"] for item in migrated["nodes"]], ["duplicate", "duplicate"])
        self.assertEqual(migrated["links"][0]["to_node_id"], "stale-runtime-id")
        self.assertEqual(migrated["links"][1]["from_node_id"], "duplicate")

    def test_timeline_transient_ids_migrate_only_with_unique_node_label_and_parameter(self):
        timeline = {
            "enabled": True,
            "future_timeline": "preserved",
            "tracks": {
                "399:amount": {
                    "target_id": "399:amount",
                    "node_id": "399",
                    "node_label": "Blur",
                    "param": "amount",
                    "type": "float",
                    "label": "Blur [399] / Amount",
                    "future_track": 11,
                    "keyframes": [{"time": 2.5, "value": 0.75, "interp": "ease"}],
                }
            },
        }
        source = graph(
            nodes=[node("BlurNode", "Blur", {"amount": 0.2})],
            links=[link("SOURCE", "Blur"), link("Blur", "DISPLAY")],
            parameter_timeline_state=timeline,
        )

        session = normalize_session_document(source)
        migrated = session["timeline"]

        self.assertEqual(migrated["future_timeline"], "preserved")
        self.assertEqual(migrated["unresolved_tracks"], [])
        self.assertEqual(len(migrated["tracks"]), 1)
        track = migrated["tracks"][0]
        self.assertEqual(
            track["target"],
            {"deck": "a", "node_id": "BlurNode:1", "parameter": "amount"},
        )
        self.assertEqual(track["keyframes"][0]["interpolation"], "ease")
        self.assertEqual(track["legacy"]["track"]["future_track"], 11)

    def test_timeline_ambiguous_across_decks_remains_unresolved(self):
        timeline = {
            "enabled": True,
            "tracks": {
                "812:amount": {
                    "node_id": "812",
                    "node_label": "Blur",
                    "param": "amount",
                    "type": "float",
                    "keyframes": [{"time": 1.0, "value": 0.4, "interp": "linear"}],
                }
            },
        }
        source = dual(
            graph(nodes=[node("BlurNode", "Blur", {"amount": 0.1})]),
            graph(nodes=[node("BlurNode", "Blur", {"amount": 0.9})]),
            parameter_timeline_state=timeline,
        )

        migrated = normalize_session_document(source)["timeline"]

        self.assertEqual(migrated["tracks"], [])
        self.assertEqual(len(migrated["unresolved_tracks"]), 1)
        self.assertEqual(migrated["unresolved_tracks"][0]["node_id"], "812")
        self.assertEqual(migrated["unresolved_tracks"][0]["legacy_key"], "812:amount")

    def test_modulation_routes_macros_and_unique_deck_parameters_without_guessing(self):
        source = graph(
            nodes=[node("BlurNode", "Blur", {"amount": 0.1})],
            mod_matrix_state={
                "enabled": True,
                "future_modulation": {"x": 1},
                "connections": [
                    "LFO -> amount",
                    "Time -> history_offset",
                    "Random -> absent_parameter",
                ],
            },
        )

        modulation = normalize_session_document(source)["modulation"]

        self.assertEqual(modulation["future_modulation"], {"x": 1})
        self.assertEqual(
            modulation["routes"][0]["target"],
            {"scope": "deck_parameter_name", "deck": "a", "parameter": "amount"},
        )
        self.assertEqual(
            modulation["routes"][1]["target"],
            {"scope": "global_macro", "parameter": "history_offset"},
        )
        self.assertEqual(
            modulation["unresolved_legacy_connections"],
            ["Random -> absent_parameter"],
        )
        self.assertEqual(modulation["routes"][0]["legacy_connection"], "LFO -> amount")

    def test_modulation_parameter_owned_by_both_decks_is_unresolved(self):
        source = dual(
            graph(nodes=[node("BlurNode", "Blur A", {"amount": 0.1})]),
            graph(nodes=[node("BlurNode", "Blur B", {"amount": 0.9})]),
            mod_matrix_state={"enabled": True, "connections": ["LFO -> amount"]},
        )

        modulation = normalize_session_document(source)["modulation"]

        self.assertEqual(modulation["routes"], [])
        self.assertEqual(modulation["unresolved_legacy_connections"], ["LFO -> amount"])

    def test_real_timeline_fixture_is_retained_and_targets_stabilized(self):
        session = load_session_document(PRESET_DIR / "5-30-26.json")

        self.assertEqual(len(session["timeline"]["tracks"]), 3)
        self.assertEqual(session["timeline"]["unresolved_tracks"], [])
        self.assertEqual(
            {track["target"]["node_id"] for track in session["timeline"]["tracks"]},
            {"CUDATemporalSlitScanNode:1"},
        )

    def test_legacy_export_has_application_shape_and_preserves_unknown_root(self):
        source = dual(
            graph(nodes=[node("BlurNode", "Blur", {"amount": 0.2})]),
            graph(),
            master_mix=0.25,
            master_transition="Additive",
            display_macros={"history_offset": 2.0},
            media_audio=None,
            mod_matrix_state={"enabled": True, "connections": ["LFO -> amount"]},
            parameter_timeline_state={"enabled": False, "tracks": {}},
            vendor_future={"still_here": True},
        )

        exported = export_legacy_dual_deck(normalize_session_document(source))

        self.assertEqual(exported["version"], "2.0")
        self.assertEqual(exported["preset_type"], "dual_deck_session")
        self.assertEqual(exported["master_mix"], 0.25)
        self.assertEqual(exported["master_transition"], "Additive")
        self.assertEqual(exported["display_macros"], {"history_offset": 2.0})
        self.assertIsNone(exported["media_audio"])
        self.assertEqual(exported["mod_matrix_state"]["connections"], ["LFO -> amount"])
        self.assertEqual(exported["parameter_timeline_state"]["tracks"], {})
        self.assertEqual(exported["vendor_future"], {"still_here": True})
        self.assertEqual(exported["decks"]["a"]["nodes"][0]["id"], "BlurNode:1")

    def test_invalid_roots_and_versions_are_rejected(self):
        cases = [
            [],
            {"format": SESSION_FORMAT, "format_version": 99},
            {"preset_type": "dual_deck_session", "version": "3.0", "decks": {}},
            {"version": "9.0", "nodes": [], "links": []},
        ]
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    normalize_session_document(value)


if __name__ == "__main__":
    unittest.main()
