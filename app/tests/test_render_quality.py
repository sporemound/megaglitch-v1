"""
Automated Test Suite for MegaGlitch Render-Quality-First Optimization Overhaul.
"""

from fractions import Fraction
import json
import os
import sys
import numpy as np
import pytest

import glitch_effects as ge
from megaglitch.mastering import create_fps_plan, DirectEncoderConfig, DirectFFmpegMasterEncoder
from megaglitch.codec_domain import CodecDomainConfig, CodecDomainPipeline
from node_pipeline_runtime import execute_flat_node_pipeline


def test_multimodal_analysis_maps():
    """Verify multimodal analysis produces correct maps and master-resolution shapes."""
    h, w = 1080, 1920
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame1[200:400, 200:400] = 255
    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2[220:420, 220:420] = 255

    out_img, maps = ge.multimodal_analysis_np(frame2, previous_frame=frame1)

    assert out_img.shape == (h, w, 3)
    assert out_img.dtype == np.uint8

    expected_keys = [
        "optical_flow_vector",
        "optical_flow_magnitude",
        "frame_difference",
        "edge_strength",
        "luminance",
        "local_entropy",
        "spectrum_energy_map",
        "motion_mask",
        "edge_mask",
        "modulation_mask",
    ]
    for key in expected_keys:
        assert key in maps, f"Missing map: {key}"
        arr = maps[key]
        assert arr.dtype == np.float32, f"Map {key} has dtype {arr.dtype} instead of float32"
        if key == "optical_flow_vector":
            assert arr.shape == (h, w, 2)
        elif key == "optical_flow_magnitude":
            assert arr.shape == (h, w)
            assert np.min(arr) >= 0.0
        else:
            assert arr.shape == (h, w)
            assert np.min(arr) >= 0.0 and np.max(arr) <= 1.0


def test_optical_flow_remap():
    """Verify optical flow remap warps coordinates without altering frame dimensions."""
    h, w = 1080, 1920
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    flow = np.ones((h, w, 2), dtype=np.float32) * 5.0
    mod_mask = np.ones((h, w), dtype=np.float32)

    remapped = ge.optical_flow_remap_np(frame, flow_vector=flow, modulation_mask=mod_mask)
    assert remapped.shape == (h, w, 3)
    assert remapped.dtype == np.uint8
    assert not np.array_equal(frame, remapped)


def test_controlled_recursive_feedback():
    """Verify bounded recursive feedback respects decay, clean mix, and scene cuts."""
    h, w = 1080, 1920
    frame1 = np.full((h, w, 3), 100, dtype=np.uint8)
    frame2 = np.full((h, w, 3), 120, dtype=np.uint8)
    state = {}

    out1, fb1 = ge.controlled_recursive_feedback_np(
        frame1, previous_feedback=None, clean_source=frame1, state=state
    )
    assert out1.shape == (h, w, 3)
    assert fb1.shape == (h, w, 3)

    out2, fb2 = ge.controlled_recursive_feedback_np(
        frame2, previous_feedback=fb1, clean_source=frame2, state=state
    )
    assert out2.shape == (h, w, 3)

    # Test scene change cut reset: high difference should trigger reset
    cut_frame = np.full((h, w, 3), 255, dtype=np.uint8)
    out_cut, fb_cut = ge.controlled_recursive_feedback_np(
        cut_frame, previous_feedback=fb2, clean_source=cut_frame,
        params={"scene_change_threshold": 0.2}, state=state
    )
    # Feedback should have been reset on cut
    assert np.all(out_cut == 255)


def test_quantization_solarization():
    """Verify quantization and solarization tonal operations."""
    h, w = 1080, 1920
    frame = np.linspace(0, 255, h * w * 3, dtype=np.uint8).reshape((h, w, 3))
    out = ge.quantization_solarization_np(frame, params={"levels": 8, "solarize_mix": 0.5})
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8


def test_structural_restoration():
    """Verify structural restoration performs unsharp masking, CLAHE, and clamping."""
    h, w = 1080, 1920
    frame = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    out = ge.structural_restoration_np(
        frame,
        params={
            "unsharp_amount": 0.8,
            "local_contrast_clip": 2.0,
            "edge_reinforce": 0.5,
            "grain_amount": 0.02,
        },
    )
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert np.min(out) >= 0 and np.max(out) <= 255


def test_fps_semantics_rational_timing():
    """Verify true_accelerated vs duration_preserving FPS planning."""
    # 30 frames at 30 fps accelerated to 55 fps
    plan_acc = create_fps_plan(
        source_frames=30,
        source_fps=30,
        output_fps=55,
        fps_semantics="true_accelerated",
    )
    assert plan_acc.output_frames == 30
    assert plan_acc.source_duration == Fraction(1, 1)
    assert plan_acc.output_duration == Fraction(6, 11)
    assert len(plan_acc.source_frame_indices) == 30
    assert plan_acc.source_frame_indices == list(range(30))

    # 30 frames at 30 fps duration-preserving to 55 fps
    plan_dur = create_fps_plan(
        source_frames=30,
        source_fps=30,
        output_fps=55,
        fps_semantics="duration_preserving",
    )
    assert plan_dur.output_frames == 55
    assert plan_dur.source_duration == Fraction(1, 1)
    assert plan_dur.output_duration == Fraction(1, 1)
    assert len(plan_dur.source_frame_indices) == 55


def test_direct_master_encoder_memory_hash():
    """Verify DirectFFmpegMasterEncoder computes telemetry in memory_hash sink mode."""
    config = DirectEncoderConfig(
        width=1920,
        height=1080,
        output_fps=55,
        sink="memory_hash",
    )
    encoder = DirectFFmpegMasterEncoder(config)
    for _ in range(5):
        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        encoder.write_frame(frame)

    telemetry = encoder.close()
    assert telemetry["frames"] == 5
    assert telemetry["bytes_written"] == 5 * (1920 * 1080 * 3)
    assert telemetry["returncode"] == 0
    assert telemetry["output_sha256"] is not None
    assert len(telemetry["output_sha256"]) == 64
