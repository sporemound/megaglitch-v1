from copy import deepcopy

import dearpygui.dearpygui as dpg
import effect_nodes
import effect_nodes_cuda
from node_editor import NodeEditor

from megaglitch.quality_graph import (
    DESTRUCTIVE_STAGE_LABELS,
    QUALITY_FIRST_STAGE_LABELS,
    build_quality_first_graph,
    build_quality_first_preset,
)
from megaglitch.session_graph_compiler import compile_preset_graph


EXPECTED_TYPES = (
    "MultimodalAnalysisNode",
    "OpticalFlowRemapNode",
    "ControlledRecursiveFeedbackNode",
    "PFrameGlitchesNode",
    "TemporalSlicingNode",
    "MeshTearNode",
    "ChannelFlowOffsetNode",
    "QuantizationSolarizationNode",
    "EdgeReconstructionNode",
    "StructuralRestorationNode",
)


def test_quality_first_graph_is_an_exact_serial_chain():
    graph = build_quality_first_graph()
    nodes = graph["nodes"]
    links = graph["links"]

    assert graph["version"] == "1.0"
    assert tuple(node["label"] for node in nodes) == QUALITY_FIRST_STAGE_LABELS
    assert tuple(node["type"] for node in nodes) == EXPECTED_TYPES
    assert len(links) == len(nodes) + 1

    endpoint_labels = ("SOURCE", *QUALITY_FIRST_STAGE_LABELS, "DISPLAY")
    endpoint_ids = (None, *(node["id"] for node in nodes), None)
    for index, link in enumerate(links):
        assert link == {
            "from_node": endpoint_labels[index],
            "to_node": endpoint_labels[index + 1],
            "from_node_id": endpoint_ids[index],
            "to_node_id": endpoint_ids[index + 1],
            "from_attr": "image_out",
            "to_attr": "image_in",
        }


def test_quality_first_graph_uses_node_editor_loader_types_and_compiles():
    graph = build_quality_first_graph()
    before = deepcopy(graph)

    for node in graph["nodes"]:
        node_class = getattr(effect_nodes, node["type"], None)
        if node_class is None:
            node_class = getattr(effect_nodes_cuda, node["type"], None)
        assert node_class is not None, node["type"]
        assert node["id"] == f"{node['type']}:1"
        assert set(node) == {"id", "type", "label", "position", "params"}

    pipeline = compile_preset_graph(graph, {"SOURCE": "primary"})

    assert graph == before
    assert [step["effect_type"] for step in pipeline] == [
        "Source",
        *QUALITY_FIRST_STAGE_LABELS,
        "Display",
    ]
    assert all(step["node_type"] == "effect" for step in pipeline[1:-1])


def test_node_editor_loads_quality_first_graph_without_schema_adaptation():
    dpg.create_context()
    try:
        with dpg.window() as parent:
            editor = NodeEditor()
            editor.create_node_editor(parent)

        assert editor.load_node_preset(
            "quality-first-test", preset_data=build_quality_first_graph()
        )
        assert tuple(
            node.label
            for node in editor.nodes
            if isinstance(node, effect_nodes.GlitchEffectNode)
        ) == QUALITY_FIRST_STAGE_LABELS
        loaded_params = {
            node.label: dict(node.param_values)
            for node in editor.nodes
            if isinstance(node, effect_nodes.GlitchEffectNode)
        }
        assert loaded_params["Optical Flow Remap"]["flow_remap_strength"] == 0.65
        assert loaded_params["Quantization / Solarization"]["solarization_blend"] == 0.08
        assert loaded_params["Structural Restoration"]["local_contrast"] == 0.08
        assert len(editor.links) == len(QUALITY_FIRST_STAGE_LABELS) + 1
    finally:
        dpg.destroy_context()


def test_destructive_stages_are_fully_multimodal_selected():
    graph = build_quality_first_graph()
    params_by_label = {node["label"]: node["params"] for node in graph["nodes"]}

    assert DESTRUCTIVE_STAGE_LABELS < set(params_by_label)
    for label in DESTRUCTIVE_STAGE_LABELS:
        assert params_by_label[label]["enabled"] is True
        assert params_by_label[label]["multimodal_amount"] == 1.0

    assert params_by_label["Multimodal Analysis"]["multimodal_amount"] == 0.0
    assert params_by_label["Edge Reconstruction"]["multimodal_amount"] == 0.0
    assert params_by_label["Structural Restoration"]["multimodal_amount"] == 0.0


def test_sharpening_and_local_contrast_exist_only_at_final_restoration():
    graph = build_quality_first_graph()
    restoration_keys = {
        "sharpen_amount",
        "unsharp_amount",
        "unsharp_radius",
        "local_contrast",
        "local_contrast_radius",
    }

    for node in graph["nodes"][:-1]:
        assert restoration_keys.isdisjoint(node["params"]), node["label"]

    final = graph["nodes"][-1]
    assert final["label"] == "Structural Restoration"
    assert final["params"]["unsharp_amount"] > 0.0
    assert final["params"]["local_contrast"] > 0.0


def test_quality_defaults_are_subtle_and_preserve_a_clean_source_floor():
    params = {node["label"]: node["params"] for node in build_quality_first_graph()["nodes"]}

    assert params["Optical Flow Remap"]["flow_remap_strength"] < 1.0
    assert params["Controlled Recursive Feedback"]["feedback_opacity"] <= 0.20
    assert params["Controlled Recursive Feedback"]["clean_source_floor"] >= 0.70
    for name in (
        "motion_vector_corruption",
        "prediction_error_amplify",
        "temporal_bleeding",
        "macroblock_scramble",
        "residual_injection",
        "reference_corruption",
    ):
        assert params["P-Frame Glitches"][name] <= 0.08
    assert params["Temporal Slicing"]["time_offset_range"] <= 2
    assert params["Mesh Tear"]["mesh_tear"] <= 0.05
    assert params["Channel Flow Offset"]["channel_flow"] <= 0.05
    assert params["Quantization / Solarization"]["solarization_blend"] <= 0.10
    assert params["Structural Restoration"]["unsharp_amount"] <= 0.20
    assert params["Structural Restoration"]["local_contrast"] <= 0.10


def test_builders_return_independent_nested_preset_objects():
    first = build_quality_first_graph()
    second = build_quality_first_graph()
    alias_result = build_quality_first_preset()

    first["source_pos"][0] = -1
    first["nodes"][0]["position"][0] = -1
    first["nodes"][0]["params"]["motion_low"] = -1
    first["links"][0]["to_node"] = "changed"

    assert second == alias_result
    assert second["source_pos"][0] == 40
    assert second["nodes"][0]["position"][0] == 280
    assert second["nodes"][0]["params"]["motion_low"] == 0.03
    assert second["links"][0]["to_node"] == "Multimodal Analysis"
