"""Tests for Production Quality Graph V1 and Benchmark Contract Separation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from megaglitch.production_graph import (
    CORE_PRODUCTION_STAGE_LABELS,
    HEADLESS_QUALITY_PROFILE_ID,
    OPTIONAL_MODULATION_STAGE_LABELS,
    PRODUCTION_GRAPH_PROFILE_ID,
    PRODUCTION_STAGE_ORDER,
    REFERENCE_QA_PROFILE_ID,
    build_production_quality_graph_v1,
    build_production_quality_preset_v1,
    get_graph_by_profile,
)
from megaglitch.quality_graph import (
    DESTRUCTIVE_STAGE_LABELS,
    QUALITY_FIRST_STAGE_LABELS,
    build_quality_first_graph,
    build_quality_first_preset,
)
from megaglitch.session_graph_compiler import compile_preset_graph


def test_reference_qa_builder_exists_and_retains_canonical_structure():
    """Assertion 1: Reference QA builder exists unchanged and retains 10 stages."""
    qa_graph = build_quality_first_graph()
    assert isinstance(qa_graph, dict)
    assert len(qa_graph["nodes"]) == 10
    labels = tuple(node["label"] for node in qa_graph["nodes"])
    assert labels == QUALITY_FIRST_STAGE_LABELS
    assert build_quality_first_preset() == qa_graph


def test_production_builder_has_explicit_profile_identity():
    """Assertion 2: Production builder has explicit profile identity 'production-quality-v1'."""
    prod_graph = build_production_quality_graph_v1()
    assert prod_graph.get("profile_id") == PRODUCTION_GRAPH_PROFILE_ID
    assert prod_graph.get("version") == "1.0"
    assert len(prod_graph["nodes"]) == 10


def test_production_graph_ordering_places_destructive_stages_before_restoration():
    """Assertion 3: Destructive processing precedes Edge Reconstruction and Structural Restoration."""
    prod_graph = build_production_quality_graph_v1()
    labels = [node["label"] for node in prod_graph["nodes"]]

    # Edge Reconstruction and Structural Restoration must be the last two stages
    assert labels[-2] == "Edge Reconstruction"
    assert labels[-1] == "Structural Restoration"

    # All destructive stages must appear before Edge Reconstruction
    edge_idx = labels.index("Edge Reconstruction")
    for destr in DESTRUCTIVE_STAGE_LABELS:
        if destr in labels:
            assert labels.index(destr) < edge_idx


def test_edge_reconstruction_precedes_structural_restoration():
    """Assertion 4: Edge Reconstruction strictly precedes Structural Restoration."""
    prod_graph = build_production_quality_graph_v1()
    labels = [node["label"] for node in prod_graph["nodes"]]
    assert labels.index("Edge Reconstruction") < labels.index("Structural Restoration")


def test_preview_ui_nodes_absent_from_master_processing_graph():
    """Assertion 5: Preview, OBS, and UI nodes are absent from master processing graph."""
    prod_graph = build_production_quality_graph_v1()
    for node in prod_graph["nodes"]:
        node_type = str(node.get("type", "")).lower()
        node_label = str(node.get("label", "")).lower()
        assert "preview" not in node_type
        assert "preview" not in node_label
        assert "obs" not in node_type
        assert "virtual_camera" not in node_type


def test_no_production_graph_definition_contains_benchmark_sha_behavior():
    """Assertion 6: Production graph definitions do not encode live SHA hashing parameters."""
    prod_graph = build_production_quality_graph_v1()
    for node in prod_graph["nodes"]:
        params = node.get("params", {})
        assert "sha256" not in params
        assert "live_hash" not in params
        assert "calculate_sha" not in params


def test_benchmark_b_profile_resolution_defaults_to_production():
    """Assertion 7: Benchmark B resolution defaults to production-quality-v1."""
    profile_id, graph = get_graph_by_profile()
    assert profile_id == PRODUCTION_GRAPH_PROFILE_ID
    assert graph["profile_id"] == PRODUCTION_GRAPH_PROFILE_ID

    # Headless alias also resolves to production
    h_profile, h_graph = get_graph_by_profile(HEADLESS_QUALITY_PROFILE_ID)
    assert h_profile == PRODUCTION_GRAPH_PROFILE_ID
    assert h_graph["profile_id"] == PRODUCTION_GRAPH_PROFILE_ID


def test_reference_qa_and_production_identities_are_distinct():
    """Assertion 8: Reference QA and Production graph identities and builders are distinct."""
    assert PRODUCTION_GRAPH_PROFILE_ID != REFERENCE_QA_PROFILE_ID

    qa_profile, qa_graph = get_graph_by_profile("reference-qa-v1")
    prod_profile, prod_graph = get_graph_by_profile("production-quality-v1")

    assert qa_profile == REFERENCE_QA_PROFILE_ID
    assert prod_profile == PRODUCTION_GRAPH_PROFILE_ID
    assert qa_graph.get("profile_id") == REFERENCE_QA_PROFILE_ID
    assert prod_graph.get("profile_id") == PRODUCTION_GRAPH_PROFILE_ID


def test_reference_qa_profile_label_does_not_reseed_frozen_behavior():
    from megaglitch.offline_master import stable_graph_hash
    from tools.benchmark.run_headless_quality_reference import behavioral_graph_hash

    profile_id, graph = get_graph_by_profile(REFERENCE_QA_PROFILE_ID)
    graph_without_profile_label = dict(graph)
    graph_without_profile_label.pop("profile_id")

    assert behavioral_graph_hash(profile_id, graph) == stable_graph_hash(
        graph_without_profile_label
    )


def test_serializing_and_loading_production_preset_preserves_definition(tmp_path: Path):
    """Assertion 9: Serializing and loading production preset round-trips losslessly."""
    preset = build_production_quality_preset_v1()
    out_file = tmp_path / "prod_preset.json"
    out_file.write_text(json.dumps(preset, indent=2), encoding="utf-8")

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["profile_id"] == PRODUCTION_GRAPH_PROFILE_ID
    assert len(loaded["nodes"]) == len(preset["nodes"])
    assert loaded["links"] == preset["links"]


def test_production_graph_is_resolution_agnostic_and_compiles():
    """Assertion 10: Ingress/egress has no hardcoded preview dimensions and compiles to pipeline."""
    from tools.benchmark.run_render_quality_path import quality_graph_pipeline

    prod_graph = build_production_quality_graph_v1()
    pipeline = quality_graph_pipeline(prod_graph)
    # Includes 1 source, 10 effect nodes, and 1 display node = 12 steps
    assert len(pipeline) == len(prod_graph["nodes"]) + 2

    for step in pipeline:
        params = step.get("params", {})
        # None of the steps should hardcode 640x360 or preview shapes
        for k, v in params.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                assert v != (640, 360)
                assert v != (360, 640)
