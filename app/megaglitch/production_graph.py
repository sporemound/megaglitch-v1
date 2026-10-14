"""Production Quality Graph V1 Definition & Profile Contract.

This module defines the canonical Production Quality Graph V1 for MegaGlitch
workstation and headless rendering. It is explicitly separated from the
Reference QA graph (megaglitch.quality_graph.build_quality_first_graph) to
prevent QA stress coverage from being conflated with production requirements.

Stage Pipeline Order:
    SOURCE
      ↓
    analysis / modulation (Multimodal Analysis)
      ↓
    destructive glitch stages (Optical Flow Remap, Controlled Feedback,
                               P-Frame, Temporal Slicing, Mesh Tear, Channel Flow)
      ↓
    color / quantization (Quantization / Solarization)
      ↓
    edge reconstruction (Edge Reconstruction)
      ↓
    structural restoration (Structural Restoration)
      ↓
    DISPLAY / OUTPUT
"""

from __future__ import annotations

from typing import Any

PRODUCTION_GRAPH_PROFILE_ID: str = "production-quality-v1"
HEADLESS_QUALITY_PROFILE_ID: str = "headless-quality-v1"
REFERENCE_QA_PROFILE_ID: str = "reference-qa-v1"

CORE_PRODUCTION_STAGE_LABELS: tuple[str, ...] = (
    "Controlled Recursive Feedback",
    "P-Frame Glitches",
    "Temporal Slicing",
    "Mesh Tear",
    "Channel Flow Offset",
    "Quantization / Solarization",
    "Edge Reconstruction",
    "Structural Restoration",
)

OPTIONAL_MODULATION_STAGE_LABELS: tuple[str, ...] = (
    "Multimodal Analysis",
    "Optical Flow Remap",
)

PRODUCTION_STAGE_ORDER: tuple[str, ...] = (
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


def _production_stage_specs(
    *,
    enable_multimodal: bool = True,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Return stage parameter specifications for Production Quality Graph V1."""

    return (
        (
            "MultimodalAnalysisNode",
            "Multimodal Analysis",
            {
                "enabled": bool(enable_multimodal),
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
                "enabled": bool(enable_multimodal),
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
                "multimodal_amount": 1.0 if enable_multimodal else 0.0,
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
                "multimodal_amount": 1.0 if enable_multimodal else 0.0,
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
                "multimodal_amount": 1.0 if enable_multimodal else 0.0,
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
                "multimodal_amount": 1.0 if enable_multimodal else 0.0,
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
                "multimodal_amount": 1.0 if enable_multimodal else 0.0,
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
                "multimodal_amount": 1.0 if enable_multimodal else 0.0,
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


def build_production_quality_graph_v1(
    *,
    enable_multimodal: bool = True,
) -> dict[str, Any]:
    """Build the canonical Production Quality Graph V1 dictionary.

    This preset shape is consumable by NodeEditor.load_node_preset and
    the master graph compilation pipeline. It has explicit profile identity
    'production-quality-v1'.
    """

    specs = _production_stage_specs(enable_multimodal=enable_multimodal)
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
        "profile_id": PRODUCTION_GRAPH_PROFILE_ID,
        "nodes": nodes,
        "links": links,
        "version": "1.0",
        "source_pos": [40, 80],
        "display_pos": [280 + len(nodes) * 260, 80],
    }


def build_production_quality_preset_v1(
    *,
    enable_multimodal: bool = True,
) -> dict[str, Any]:
    """Readable alias for :func:`build_production_quality_graph_v1`."""

    return build_production_quality_graph_v1(enable_multimodal=enable_multimodal)


def get_graph_by_profile(
    profile_id: str = PRODUCTION_GRAPH_PROFILE_ID,
) -> tuple[str, dict[str, Any]]:
    """Resolve graph builder and return (resolved_profile_id, graph_dict)."""
    normalized = str(profile_id).strip().lower()

    if normalized in {PRODUCTION_GRAPH_PROFILE_ID, HEADLESS_QUALITY_PROFILE_ID, "production", "headless"}:
        return (PRODUCTION_GRAPH_PROFILE_ID, build_production_quality_graph_v1())

    if normalized in {REFERENCE_QA_PROFILE_ID, "reference-qa", "qa"}:
        from megaglitch.quality_graph import build_quality_first_graph
        graph = build_quality_first_graph()
        graph["profile_id"] = REFERENCE_QA_PROFILE_ID
        return (REFERENCE_QA_PROFILE_ID, graph)

    raise ValueError(f"Unknown graph profile: {profile_id!r}. Supported: {PRODUCTION_GRAPH_PROFILE_ID!r}, {REFERENCE_QA_PROFILE_ID!r}")


__all__ = [
    "CORE_PRODUCTION_STAGE_LABELS",
    "HEADLESS_QUALITY_PROFILE_ID",
    "OPTIONAL_MODULATION_STAGE_LABELS",
    "PRODUCTION_GRAPH_PROFILE_ID",
    "PRODUCTION_STAGE_ORDER",
    "REFERENCE_QA_PROFILE_ID",
    "build_production_quality_graph_v1",
    "build_production_quality_preset_v1",
    "get_graph_by_profile",
]
