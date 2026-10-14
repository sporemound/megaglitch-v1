from tools.export_rust_node_catalog import (
    NODE_SPEC_VERSION,
    OUTPUT,
    ROOT,
    build_specs,
    derives_from,
    python_classes,
    render,
)


def _spec(specs, backend_id):
    return next(spec for spec in specs if spec.backend_id == backend_id)


def test_node_spec_export_covers_representative_python_node_families():
    specs = build_specs()

    pixel_sort = _spec(specs, "PixelSortingNode")
    assert [socket.name for socket in pixel_sort.inputs] == ["image_in"]
    assert [socket.name for socket in pixel_sort.outputs] == ["image_out"]
    params = {param.name: param for param in pixel_sort.parameters}
    assert params["enabled"].kind == "bool"
    assert params["pixel_sorting_direction"].choices == ("Horizontal", "Vertical")
    assert params["sort_chance"].kind == "int"
    assert (params["sort_chance"].minimum, params["sort_chance"].maximum) == (
        0.0,
        100.0,
    )
    assert params["pixel_sort_evolution"].kind == "float"

    composite = _spec(specs, "GenesisPriorityCompositeNode")
    assert [socket.name for socket in composite.inputs] == [
        "base_image",
        "overlay_image",
        "priority_mask",
        "luma_mask",
        "corruption_mask",
    ]

    cuda = _spec(specs, "CUDAFlowWarpNode")
    assert cuda.backend == "cuda"
    assert {param.kind for param in cuda.parameters} >= {"bool", "float"}

    generator = _spec(specs, "GradientGeneratorNode")
    assert generator.inputs == ()
    assert generator.outputs[0].name == "image_out"
    assert {param.name for param in generator.parameters} >= {
        "mode",
        "mutation_strength",
        "mutation_echo",
    }


def test_rendered_contract_is_versioned_and_stable():
    specs = build_specs()
    assert len({spec.backend_id for spec in specs}) == len(specs)
    generated = render(specs)
    assert "version: NODE_SPEC_VERSION" in generated
    assert "pub const NODES: &[NodeSpec]" in generated
    assert NODE_SPEC_VERSION == 1


def test_every_glitch_effect_inherits_multimodal_amount():
    classes = {}
    for filename in ("effect_nodes.py", "effect_nodes_cuda.py"):
        classes.update(python_classes(ROOT / filename))
    glitch_effect_ids = {
        name
        for name in classes
        if derives_from(name, "GlitchEffectNode", classes)
    }
    specs = {spec.backend_id: spec for spec in build_specs()}

    assert glitch_effect_ids <= specs.keys()
    for backend_id in glitch_effect_ids:
        params = {param.name: param for param in specs[backend_id].parameters}
        multimodal = params["multimodal_amount"]
        assert multimodal.kind == "float", backend_id
        assert multimodal.default == 0.0, backend_id
        assert (multimodal.minimum, multimodal.maximum) == (0.0, 1.0), backend_id


def test_quality_first_node_specs_include_current_python_contracts():
    specs = build_specs()

    multimodal = _spec(specs, "MultimodalAnalysisNode")
    assert multimodal.category == "Analysis & Quality"
    assert {param.name for param in multimodal.parameters} >= {
        "multimodal_amount",
        "entropy_window",
        "flow_scale_pixels",
        "modulation_gamma",
    }

    optical_flow = _spec(specs, "OpticalFlowRemapNode")
    optical_params = {param.name: param for param in optical_flow.parameters}
    assert optical_flow.label == "Optical Flow Remap"
    assert optical_flow.category == "Analysis & Quality"
    assert optical_params["flow_remap_strength"].default == 0.65
    assert optical_params["flow_direction"].choices == ("backward", "forward")

    feedback = _spec(specs, "ControlledRecursiveFeedbackNode")
    feedback_params = {param.name: param for param in feedback.parameters}
    assert feedback.category == "Analysis & Quality"
    assert feedback_params["clean_source_floor"].default == 0.32
    assert feedback_params["opacity_cap"].maximum == 1.0

    quantization = _spec(specs, "QuantizationSolarizationNode")
    quantization_params = {param.name: param for param in quantization.parameters}
    assert quantization.label == "Quantization / Solarization"
    assert quantization_params["quantization_levels"].default == 192
    assert quantization_params["solarization_threshold"].maximum == 255.0

    edge = _spec(specs, "EdgeReconstructionNode")
    edge_params = {param.name: param for param in edge.parameters}
    assert edge.label == "Edge Reconstruction"
    assert edge_params["edge_luma_reinforcement"].default == 0.035
    assert edge_params["edge_sensitivity"].minimum == 0.001

    restoration = _spec(specs, "StructuralRestorationNode")
    restoration_params = {param.name: param for param in restoration.parameters}
    assert restoration.category == "Analysis & Quality"
    assert restoration_params["unsharp_amount"].default == 0.2
    assert restoration_params["local_contrast"].default == 0.12
    assert restoration_params["final_clamp_high"].default == 255


def test_checked_in_rust_catalog_matches_the_exporter():
    assert OUTPUT.read_text(encoding="utf-8") == render(build_specs())
