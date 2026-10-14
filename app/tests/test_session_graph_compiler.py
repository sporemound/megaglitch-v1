import unittest
from copy import deepcopy

from megaglitch.session_graph_compiler import GraphCompileError, compile_preset_graph


DECK_A_SOURCES = {"SOURCE": "primary", "SOURCE_B": "secondary"}


def node(node_id, backend_id, label, params=None, **extra):
    value = {
        "id": node_id,
        "type": backend_id,
        "label": label,
        "params": dict(params or {}),
    }
    value.update(extra)
    return value


def link(source, target, from_attr="image_out", to_attr="image_in", **ids):
    value = {
        "from_node": source,
        "to_node": target,
        "from_attr": from_attr,
        "to_attr": to_attr,
    }
    value.update(ids)
    return value


def preset(nodes, links):
    return {"version": "1.0", "nodes": nodes, "links": links}


class SessionGraphCompilerTests(unittest.TestCase):
    def test_linear_v1_matches_legacy_flat_shape_and_does_not_mutate_input(self):
        graph = preset(
            [
                node(
                    "InvertColorNode:1",
                    "InvertColorNode",
                    "Invert Colors",
                    {"enabled": True, "invert_chance": 100},
                )
            ],
            [
                link(
                    "SOURCE",
                    "Invert Colors",
                    to_node_id="InvertColorNode:1",
                ),
                link(
                    "Invert Colors",
                    "DISPLAY",
                    from_node_id="InvertColorNode:1",
                ),
            ],
        )
        before = deepcopy(graph)

        pipeline = compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(graph, before)
        self.assertEqual(
            pipeline,
            [
                {
                    "effect_type": "Source",
                    "params": {},
                    "node_type": "source",
                    "source_key": "primary",
                },
                {
                    "effect_type": "Invert Colors",
                    "params": {"enabled": True, "invert_chance": 100},
                    "node_id": "InvertColorNode:1",
                    "node_type": "effect",
                    "enabled": True,
                },
                {"effect_type": "Display", "params": {}, "node_type": "display"},
            ],
        )

    def test_missing_ids_receive_saved_preset_type_ordinals(self):
        graph = preset(
            [node(None, "InvertColorNode", "First"), node(None, "InvertColorNode", "Second")],
            [
                link("SOURCE", "First"),
                link("First", "Second"),
                link("Second", "DISPLAY"),
            ],
        )

        pipeline = compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(
            [step.get("node_id") for step in pipeline],
            [None, "InvertColorNode:1", "InvertColorNode:2", None],
        )

    def test_explicit_source_namespace_preserves_both_sources(self):
        graph = preset(
            [node("VideoMixerNode:1", "VideoMixerNode", "Video Mixer", {"mix": 0.4})],
            [
                link("SOURCE", "Video Mixer", to_attr="image_a", to_node_id="VideoMixerNode:1"),
                link("SOURCE_B", "Video Mixer", to_attr="image_b", to_node_id="VideoMixerNode:1"),
                link("Video Mixer", "DISPLAY", from_node_id="VideoMixerNode:1"),
            ],
        )

        pipeline = compile_preset_graph(
            graph,
            {
                "SOURCE": {"source_key": "secondary", "label": "Deck B"},
                "SOURCE_B": {"source_key": "primary", "label": "Deck A"},
            },
        )

        self.assertEqual(
            [step.get("source_key") for step in pipeline if step["node_type"] == "source"],
            ["secondary", "primary"],
        )
        self.assertEqual(pipeline[2]["node_type"], "mixer")
        self.assertEqual(pipeline[2]["node_id"], "VideoMixerNode:1")

    def test_generator_selected_outputs_and_multi_input_order_are_preserved(self):
        graph = preset(
            [
                node("BubbleSimGenerator2Node:1", "BubbleSimGenerator2Node", "Bubble Sim Gen (2)"),
                node("GenesisMaskKeyNode:1", "GenesisMaskKeyNode", "Genesis Mask Key"),
            ],
            [
                link("SOURCE", "Bubble Sim Gen (2)", to_node_id="BubbleSimGenerator2Node:1"),
                link(
                    "Bubble Sim Gen (2)",
                    "Genesis Mask Key",
                    from_attr="image_out",
                    to_attr="image_in",
                    from_node_id="BubbleSimGenerator2Node:1",
                    to_node_id="GenesisMaskKeyNode:1",
                ),
                link(
                    "Bubble Sim Gen (2)",
                    "Genesis Mask Key",
                    from_attr="bubble_mask",
                    to_attr="mask_in",
                    from_node_id="BubbleSimGenerator2Node:1",
                    to_node_id="GenesisMaskKeyNode:1",
                ),
                link("Genesis Mask Key", "DISPLAY", from_node_id="GenesisMaskKeyNode:1"),
            ],
        )

        pipeline = compile_preset_graph(graph, DECK_A_SOURCES)

        generators = [step for step in pipeline if step.get("is_generator")]
        self.assertEqual(
            [step["selected_output"] for step in generators], ["image_out", "bubble_mask"]
        )
        self.assertTrue(all(step["input_order"] == ["image_in"] for step in generators))
        composite = pipeline[-2]
        self.assertEqual(composite["node_type"], "genesis_mask_key")
        self.assertEqual(composite["input_order"], ["image_in", "mask_in"])

    def test_genesis_priority_uses_contract_order_not_link_order(self):
        graph = preset(
            [node("GenesisPriorityCompositeNode:1", "GenesisPriorityCompositeNode", "Priority")],
            [
                link("SOURCE_B", "Priority", to_attr="overlay_image", to_node_id="GenesisPriorityCompositeNode:1"),
                link("SOURCE", "Priority", to_attr="base_image", to_node_id="GenesisPriorityCompositeNode:1"),
                link("Priority", "DISPLAY", from_node_id="GenesisPriorityCompositeNode:1"),
            ],
        )

        pipeline = compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual([step["source_key"] for step in pipeline[:2]], ["primary", "secondary"])
        self.assertEqual(pipeline[2]["input_order"], ["base_image", "overlay_image"])
        self.assertEqual(pipeline[2]["node_type"], "genesis_priority_composite")

    def test_lua_multi_input_keeps_metadata_and_auxiliary_order(self):
        lua_params = {
            "enabled": True,
            "shader_preset": "kaleido_feedback",
            "shader_path": "lua_glsl_demo/shaders/kaleido_feedback.frag",
            "lua_path": "lua_glsl_demo/scripts/kaleido_feedback.lua",
            "metadata_path": "lua_glsl_demo/scripts/kaleido_feedback.effect",
        }
        graph = preset(
            [node("LuaShaderNode:1", "LuaShaderNode", "Lua Shader", lua_params)],
            [
                link("SOURCE_B", "Lua Shader", to_attr="feedback_in", to_node_id="LuaShaderNode:1"),
                link("SOURCE", "Lua Shader", to_node_id="LuaShaderNode:1"),
                link("Lua Shader", "DISPLAY", from_node_id="LuaShaderNode:1"),
            ],
        )

        pipeline = compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual([step["source_key"] for step in pipeline[:2]], ["primary", "secondary"])
        shader = pipeline[2]
        self.assertEqual(shader["node_type"], "lua_multi_input")
        self.assertEqual(shader["input_order"], ["image_in", "feedback_in"])
        self.assertEqual(shader["params"], lua_params)

    def test_top_level_active_overrides_stale_enabled_param(self):
        graph = preset(
            [node("ColorAdjustNode:1", "ColorAdjustNode", "Color", {"enabled": True}, active=False)],
            [
                link("SOURCE", "Color", to_node_id="ColorAdjustNode:1"),
                link("Color", "DISPLAY", from_node_id="ColorAdjustNode:1"),
            ],
        )

        effect = compile_preset_graph(graph, DECK_A_SOURCES)[1]

        self.assertIs(effect["enabled"], False)
        self.assertIs(effect["params"]["enabled"], False)

    def test_duplicate_ids_and_missing_endpoints_or_sockets_fail(self):
        cases = [
            (
                lambda graph: graph["nodes"].append(
                    node("InvertColorNode:1", "ColorAdjustNode", "Other")
                ),
                "duplicate_node_id",
            ),
            (lambda graph: graph["links"][0].update(from_node="DOES_NOT_EXIST"), "missing_endpoint"),
            (lambda graph: graph["links"][0].update(from_attr="not_an_output"), "missing_socket"),
            (lambda graph: graph["links"][0].update(to_attr="not_an_input"), "missing_socket"),
        ]
        for mutation, code in cases:
            with self.subTest(code=code, mutation=mutation):
                graph = preset(
                    [node("InvertColorNode:1", "InvertColorNode", "Invert")],
                    [
                        link("SOURCE", "Invert", to_node_id="InvertColorNode:1"),
                        link("Invert", "DISPLAY", from_node_id="InvertColorNode:1"),
                    ],
                )
                mutation(graph)
                with self.assertRaises(GraphCompileError) as caught:
                    compile_preset_graph(graph, DECK_A_SOURCES)
                self.assertEqual(caught.exception.code, code)

    def test_ambiguous_legacy_labels_require_stable_link_ids(self):
        graph = preset(
            [
                node("LuaShaderNode:1", "LuaShaderNode", "Lua Shader"),
                node("LuaShaderNode:2", "LuaShaderNode", "Lua Shader"),
            ],
            [link("SOURCE", "Lua Shader"), link("Lua Shader", "DISPLAY")],
        )

        with self.assertRaises(GraphCompileError) as caught:
            compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(caught.exception.code, "ambiguous_node_label")
        self.assertEqual(
            set(caught.exception.details["node_ids"]), {"LuaShaderNode:1", "LuaShaderNode:2"}
        )

    def test_duplicate_input_links_are_rejected(self):
        graph = preset(
            [node("InvertColorNode:1", "InvertColorNode", "Invert")],
            [
                link("SOURCE", "Invert", to_node_id="InvertColorNode:1"),
                link("SOURCE_B", "Invert", to_node_id="InvertColorNode:1"),
                link("Invert", "DISPLAY", from_node_id="InvertColorNode:1"),
            ],
        )

        with self.assertRaises(GraphCompileError) as caught:
            compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(caught.exception.code, "duplicate_input")

    def test_cycles_are_reported_even_off_the_display_path(self):
        graph = preset(
            [
                node("ColorAdjustNode:1", "ColorAdjustNode", "A"),
                node("InvertColorNode:1", "InvertColorNode", "B"),
            ],
            [
                link("A", "B", from_node_id="ColorAdjustNode:1", to_node_id="InvertColorNode:1"),
                link("B", "A", from_node_id="InvertColorNode:1", to_node_id="ColorAdjustNode:1"),
            ],
        )

        with self.assertRaises(GraphCompileError) as caught:
            compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(caught.exception.code, "cycle")
        cycle = caught.exception.details["cycle"]
        self.assertEqual(cycle[0], cycle[-1])

    def test_disconnected_nodes_and_dead_branch_links_are_not_dropped(self):
        graph = preset(
            [
                node("InvertColorNode:1", "InvertColorNode", "Live"),
                node("ColorAdjustNode:1", "ColorAdjustNode", "Dead"),
            ],
            [
                link("SOURCE", "Live", to_node_id="InvertColorNode:1"),
                link("Live", "DISPLAY", from_node_id="InvertColorNode:1"),
                link("SOURCE", "Dead", to_node_id="ColorAdjustNode:1"),
            ],
        )

        with self.assertRaises(GraphCompileError) as caught:
            compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(caught.exception.code, "unreachable_nodes")
        self.assertEqual(caught.exception.details["node_ids"], ["ColorAdjustNode:1"])

    def test_empty_disconnected_editor_matches_legacy_empty_pipeline(self):
        self.assertEqual(compile_preset_graph(preset([], []), DECK_A_SOURCES), [])

    def test_generator_only_legacy_graph_does_not_require_source(self):
        graph = preset(
            [node(None, "FractalGeneratorNode", "Fractal Gen", {"enabled": False})],
            [link("Fractal Gen", "DISPLAY")],
        )

        pipeline = compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(pipeline[0]["node_type"], "generator")
        self.assertEqual(pipeline[0]["selected_output"], "image_out")
        self.assertEqual(pipeline[0]["input_order"], [])
        self.assertIs(pipeline[0]["params"]["enabled"], False)

    def test_unsupported_visual_contract_is_explicit_when_wired(self):
        graph = preset(
            [node("FrameDifferenceNode:1", "FrameDifferenceNode", "Frame Difference")],
            [
                link(
                    "Frame Difference",
                    "DISPLAY",
                    from_node_id="FrameDifferenceNode:1",
                    from_attr="result",
                )
            ],
        )

        with self.assertRaises(GraphCompileError) as caught:
            compile_preset_graph(graph, DECK_A_SOURCES)

        self.assertEqual(caught.exception.code, "unsupported_node")

    def test_dual_deck_wrapper_requires_selecting_one_v1_graph(self):
        with self.assertRaises(GraphCompileError) as caught:
            compile_preset_graph(
                {"version": "2.0", "decks": {"a": preset([], []), "b": preset([], [])}},
                DECK_A_SOURCES,
            )

        self.assertEqual(caught.exception.code, "unsupported_version")


if __name__ == "__main__":
    unittest.main()
