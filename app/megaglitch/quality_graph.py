"""Pure builder for MegaGlitch's quality-first default node graph.

The returned object is the version 1.0 single-editor preset shape consumed by
``NodeEditor.load_node_preset``.  This module deliberately has no Dear PyGui or
runtime imports, so callers can construct, inspect, or serialize the default
without creating application state.
"""

from __future__ import annotations

from typing import Any


QUALITY_FIRST_STAGE_LABELS = (
    "Multimodal Analysis",
    "Optical Flow Remap",
    "Controlled Recursive Feedback",
    "P-Frame Glitches",
    "Temporal Slicing",
    "Mesh Tear",
    "Channel Flow Offset",
    "Quantization / Solarization",
    "Edge Reconstruction",
    "Structural Restoration",
)

DESTRUCTIVE_STAGE_LABELS = frozenset(
    {
        "Optical Flow Remap",
        "Controlled Recursive Feedback",
        "P-Frame Glitches",
        "Temporal Slicing",
        "Mesh Tear",
        "Channel Flow Offset",
        "Quantization / Solarization",
    }
)


def _stage_specs() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Return new parameter mappings for the fixed quality-first stages."""

    return (
        (
            "MultimodalAnalysisNode",
            "Multimodal Analysis",
            {
                "enabled": True,
                "multimodal_amount": 0.0,
                "entropy_window": 9,
                "flow_scale_pixels": 14.0,
                "motion_low": 0.03,
                "motion_high": 0.38,
                "edge_low": 0.02,
                "edge_high": 0.28,
                "scene_change_threshold": 0.30,
                "modulation_gamma": 1.25,
            },
        ),
        (
            "OpticalFlowRemapNode",
            "Optical Flow Remap",
            {
                "enabled": True,
                "multimodal_amount": 1.0,
                "flow_remap_strength": 0.65,
                "flow_displacement_cap": 24.0,
                "flow_direction": "backward",
                "flow_border_mode": "reflect",
            },
        ),
        (
            "ControlledRecursiveFeedbackNode",
            "Controlled Recursive Feedback",
            {
                "enabled": True,
                "multimodal_amount": 1.0,
                "feedback_opacity": 0.16,
                "opacity_cap": 0.24,
                "feedback_decay": 0.88,
                "clean_source_floor": 0.76,
                "motion_suppression": 0.85,
                "edge_protection": 0.55,
                "scene_change_threshold": 0.30,
            },
        ),
        (
            "PFrameGlitchesNode",
            "P-Frame Glitches",
            {
                "enabled": True,
                "multimodal_amount": 1.0,
                "motion_vector_corruption": 0.07,
                "motion_vector_direction": "wave",
                "prediction_error_amplify": 0.04,
                "prediction_error_direction": "uniform",
                "temporal_bleeding": 0.06,
                "temporal_bleeding_direction": "forward",
                "macroblock_scramble": 0.025,
                "scrambling_pattern": "horizontal_shift",
                "residual_injection": 0.025,
                "residual_direction": "uniform",
                "reference_corruption": 0.02,
                "reference_corruption_pattern": "horizontal_bands",
                "macroblock_size": "16",
                "error_threshold": 42,
                "bleed_distance": 2,
                "residual_noise_type": "gaussian",
                "reference_corruption_type": "temporal_artifacts",
                "buffer_size": 12,
            },
        ),
        (
            "TemporalSlicingNode",
            "Temporal Slicing",
            {
                "enabled": True,
                "multimodal_amount": 1.0,
                "slice_width": 24,
                "time_offset_range": 2,
                "slice_direction": "vertical",
            },
        ),
        (
            "MeshTearNode",
            "Mesh Tear",
            {
                "enabled": True,
                "multimodal_amount": 1.0,
                "mesh_tear": 0.045,
                "mesh_grid": 12,
                "mesh_tear_evolution": 0.02,
                "mesh_tear_evolution_speed": 0.35,
            },
        ),
        (
            "ChannelFlowOffsetNode",
            "Channel Flow Offset",
            {
                "enabled": True,
                "multimodal_amount": 1.0,
                "channel_flow": 0.035,
                "channel_flow_max": 4,
                "channel_flow_evolution": 0.02,
                "channel_flow_evolution_speed": 0.30,
            },
        ),
        (
            "QuantizationSolarizationNode",
            "Quantization / Solarization",
            {
                "enabled": True,
                "multimodal_amount": 1.0,
                "quantization_levels": 192,
                "solarization_threshold": 208,
                "solarization_blend": 0.08,
            },
        ),
        (
            "EdgeReconstructionNode",
            "Edge Reconstruction",
            {
                "enabled": True,
                "multimodal_amount": 0.0,
                "edge_luma_reinforcement": 0.03,
                "edge_sensitivity": 0.05,
            },
        ),
        (
            "StructuralRestorationNode",
            "Structural Restoration",
            {
                "enabled": True,
                "multimodal_amount": 0.0,
                "unsharp_amount": 0.14,
                "unsharp_radius": 0.85,
                "local_contrast": 0.08,
                "local_contrast_radius": 12.0,
                "edge_luma_reinforcement": 0.035,
                "edge_sensitivity": 0.055,
                "grain_normalization": 0.10,
                "grain_target": 0.008,
                "final_clamp_low": 0,
                "final_clamp_high": 255,
            },
        ),
    )


def build_quality_first_graph() -> dict[str, Any]:
    """Build a fresh, serial, quality-first NodeEditor preset.

    Destructive stages are spatially selected by the upstream multimodal
    analysis mask.  Edge reconstruction remains neutral, and all sharpening
    and local-contrast work is reserved for the final restoration node.
    """

    specs = _stage_specs()
    nodes: list[dict[str, Any]] = []
    for index, (node_type, label, params) in enumerate(specs):
        nodes.append(
            {
                "id": f"{node_type}:1",
                "type": node_type,
                "label": label,
                "position": [280 + index * 260, 80],
                "params": dict(params),
            }
        )

    links: list[dict[str, Any]] = []
    previous_label = "SOURCE"
    previous_id: str | None = None
    for node in nodes:
        links.append(
            {
                "from_node": previous_label,
                "to_node": node["label"],
                "from_node_id": previous_id,
                "to_node_id": node["id"],
                "from_attr": "image_out",
                "to_attr": "image_in",
            }
        )
        previous_label = node["label"]
        previous_id = node["id"]

    links.append(
        {
            "from_node": previous_label,
            "to_node": "DISPLAY",
            "from_node_id": previous_id,
            "to_node_id": None,
            "from_attr": "image_out",
            "to_attr": "image_in",
        }
    )

    return {
        "nodes": nodes,
        "links": links,
        "version": "1.0",
        "source_pos": [40, 80],
        "display_pos": [280 + len(nodes) * 260, 80],
    }


def build_quality_first_preset() -> dict[str, Any]:
    """Readable alias for :func:`build_quality_first_graph`."""

    return build_quality_first_graph()


__all__ = [
    "DESTRUCTIVE_STAGE_LABELS",
    "QUALITY_FIRST_STAGE_LABELS",
    "build_quality_first_graph",
    "build_quality_first_preset",
]
