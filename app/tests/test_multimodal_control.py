from dataclasses import replace

import numpy as np
import pytest

from megaglitch.multimodal_control import (
    ControlledFeedbackConfig,
    ControlledRecursiveFeedback,
    MasterMultimodalAnalyzer,
    MasterMultimodalFeedbackStage,
    MasterResolutionError,
    MultimodalAnalysis,
    apply_master_optical_flow_remap,
    MultimodalAnalysisConfig,
)


def test_master_optical_flow_remap_is_strict_full_resolution() -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[:, 4:8] = 255
    analysis = _synthetic_analysis((12, 16, 3), modulation=1.0, motion=1.0)
    vector = analysis.optical_flow_vector.copy()
    vector[..., 0] = 2.0
    analysis = MultimodalAnalysis(
        **{
            **analysis.__dict__,
            "optical_flow_vector": vector,
        }
    )
    output = apply_master_optical_flow_remap(
        frame, analysis, strength=1.0, displacement_cap_pixels=4.0
    )
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, frame)


def _moving_pair(height=72, width=96):
    previous = np.zeros((height, width, 3), dtype=np.uint8)
    current = np.zeros_like(previous)
    previous[24:48, 18:42] = (236, 154, 72)
    current[24:48, 30:54] = (236, 154, 72)
    return previous, current


def _synthetic_analysis(shape, *, modulation=1.0, motion=0.0, edge=0.0, scene=0.0):
    height, width = shape[:2]
    zeros = np.zeros((height, width), dtype=np.float32)

    def full(value):
        return np.full((height, width), value, dtype=np.float32)

    return MultimodalAnalysis(
        master_shape=shape,
        optical_flow_vector=np.zeros(shape[:2] + (2,), dtype=np.float32),
        optical_flow_magnitude=zeros.copy(),
        frame_difference=zeros.copy(),
        edge_strength=full(edge),
        luminance=zeros.copy(),
        local_entropy=zeros.copy(),
        spectrum_energy_map=zeros.copy(),
        motion_mask=full(motion),
        edge_mask=full(edge),
        modulation_mask=full(modulation),
        telemetry={"scene_change_score": scene},
    )


def test_all_feature_and_control_masks_remain_at_exact_master_resolution():
    previous, current = _moving_pair(61, 103)
    analyzer = MasterMultimodalAnalyzer()

    analysis = analyzer.analyze(
        current,
        previous,
        audio_fft_bands={"sub": 0.7, "mid": 0.35, "air": 0.8},
        transient=0.65,
        beat_state=True,
    )

    assert analysis.master_shape == current.shape
    analysis.assert_master_integrity()
    for value in (
        analysis.optical_flow_magnitude,
        analysis.frame_difference,
        analysis.edge_strength,
        analysis.luminance,
        analysis.local_entropy,
        analysis.spectrum_energy_map,
        analysis.motion_mask,
        analysis.edge_mask,
        analysis.modulation_mask,
    ):
        assert value.shape == current.shape[:2]
        assert value.dtype == np.float32
        assert value.flags.c_contiguous


def test_analysis_is_repeatable_for_identical_master_inputs_and_audio_state():
    previous, current = _moving_pair()
    args = dict(
        audio_fft_bands={"bass": 0.82, "voice": 0.27, "treble": 0.61},
        transient=0.49,
        beat_state=0.75,
    )

    first = MasterMultimodalAnalyzer().analyze(current, previous, **args)
    second = MasterMultimodalAnalyzer().analyze(current, previous, **args)

    for name in (
        "optical_flow_magnitude",
        "frame_difference",
        "edge_strength",
        "local_entropy",
        "spectrum_energy_map",
        "motion_mask",
        "edge_mask",
        "modulation_mask",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.telemetry == second.telemetry


def test_canonical_optical_flow_vector_uses_deterministic_subpixel_grid():
    previous, current = _moving_pair()

    analysis = MasterMultimodalAnalyzer().analyze(current, previous)
    scaled = analysis.optical_flow_vector.astype(np.float64) * 256.0

    assert np.array_equal(scaled, np.rint(scaled))
    assert analysis.optical_flow_vector.dtype == np.float32
    assert analysis.optical_flow_vector.flags.c_contiguous


def test_motion_and_edge_masks_localize_a_moving_structure():
    previous, current = _moving_pair(96, 128)
    analysis = MasterMultimodalAnalyzer().analyze(current, previous)

    active_region = analysis.motion_mask[18:55, 12:62]
    static_region = analysis.motion_mask[70:90, 90:120]
    contour_region = analysis.edge_mask[22:50, 27:57]
    static_edges = analysis.edge_mask[70:90, 90:120]

    assert float(np.mean(active_region)) > float(np.mean(static_region)) + 0.2
    assert float(np.mean(contour_region)) > float(np.mean(static_edges)) + 0.03
    assert analysis.telemetry["motion_coverage"] > 0.0


def test_fft_transient_and_beat_drive_spatial_selection_not_a_uniform_gain():
    previous, current = _moving_pair(80, 120)
    quiet = MasterMultimodalAnalyzer().analyze(current, previous)
    excited = MasterMultimodalAnalyzer().analyze(
        current,
        previous,
        audio_fft_bands={"bass": 1.0, "mid": 0.8, "high": 0.9},
        transient=1.0,
        beat_state=1.0,
    )

    assert float(np.mean(excited.modulation_mask)) > float(
        np.mean(quiet.modulation_mask)
    )
    assert float(np.std(excited.modulation_mask)) > 0.01
    assert not np.allclose(
        excited.modulation_mask - quiet.modulation_mask,
        float(np.mean(excited.modulation_mask - quiet.modulation_mask)),
    )
    assert excited.telemetry["audio_fft_low"] == pytest.approx(1.0)
    assert excited.telemetry["transient"] == pytest.approx(1.0)
    assert excited.telemetry["beat_state"] == pytest.approx(1.0)


def test_spectrum_energy_distinguishes_flat_and_high_frequency_frames():
    flat = np.full((64, 96, 3), 128, dtype=np.uint8)
    yy, xx = np.indices(flat.shape[:2])
    checker = (((xx + yy) % 2) * 255).astype(np.uint8)
    high_frequency = np.repeat(checker[..., None], 3, axis=2)

    flat_result = MasterMultimodalAnalyzer().analyze(flat)
    high_result = MasterMultimodalAnalyzer().analyze(high_frequency)

    assert flat_result.telemetry["image_spectrum_energy"] == pytest.approx(0.0)
    assert high_result.telemetry["image_spectrum_energy"] > 0.95
    assert float(np.mean(high_result.spectrum_energy_map)) > 0.5


def test_master_contract_rejects_hidden_copies_and_resolution_reconciliation():
    previous, current = _moving_pair()
    analyzer = MasterMultimodalAnalyzer()
    analyzer.analyze(current, previous)

    with pytest.raises(MasterResolutionError, match="does not match master shape"):
        analyzer.analyze(np.zeros((48, 64, 3), dtype=np.uint8))
    with pytest.raises(MasterResolutionError, match="contiguous"):
        MasterMultimodalAnalyzer().analyze(current[:, ::2])
    with pytest.raises(TypeError, match="uint8"):
        MasterMultimodalAnalyzer().analyze(current.astype(np.float32))


def test_explicit_reset_is_required_before_a_new_master_resolution():
    analyzer = MasterMultimodalAnalyzer()
    first = np.zeros((40, 64, 3), dtype=np.uint8)
    second = np.zeros((52, 78, 3), dtype=np.uint8)
    analyzer.analyze(first)

    with pytest.raises(MasterResolutionError):
        analyzer.analyze(second)
    analyzer.reset()

    result = analyzer.analyze(second)
    assert result.master_shape == second.shape


def test_feedback_initialization_disabled_and_scene_reset_are_exact_bypasses():
    shape = (24, 36, 3)
    initial = np.full(shape, 210, dtype=np.uint8)
    next_frame = np.full(shape, 30, dtype=np.uint8)
    feedback = ControlledRecursiveFeedback()

    initialized = feedback.process(initial, _synthetic_analysis(shape))
    disabled = feedback.process(
        next_frame,
        _synthetic_analysis(shape),
        enabled=False,
    )
    reset = feedback.process(
        initial,
        _synthetic_analysis(shape, scene=1.0),
    )

    assert initialized.frame is initial
    assert disabled.frame is next_frame
    assert reset.frame is initial
    assert np.array_equal(disabled.frame, next_frame)
    assert reset.telemetry.scene_reset is True


def test_feedback_opacity_cap_decay_and_clean_floor_bound_recursion():
    shape = (20, 28, 3)
    white = np.full(shape, 255, dtype=np.uint8)
    black = np.zeros(shape, dtype=np.uint8)
    config = ControlledFeedbackConfig(
        opacity=1.0,
        opacity_cap=0.8,
        decay=1.0,
        clean_source_floor=0.45,
        motion_suppression=0.0,
        edge_protection=0.0,
        scene_change_threshold=1.0,
    )
    feedback = ControlledRecursiveFeedback(config)
    analysis = _synthetic_analysis(shape, modulation=1.0)
    feedback.process(white, analysis)

    result = feedback.process(black, analysis)

    assert result.frame.shape == shape
    assert result.frame.flags.c_contiguous
    assert int(result.frame.max()) <= 141  # 255 * (1 - clean floor) rounded
    assert result.telemetry.bounded_opacity == pytest.approx(0.55)
    assert result.telemetry.peak_effective_opacity <= 0.55 + 1e-7


def test_motion_sensitive_feedback_protects_new_motion_from_old_history():
    shape = (32, 48, 3)
    old = np.full(shape, 220, dtype=np.uint8)
    current = np.full(shape, 20, dtype=np.uint8)
    config = ControlledFeedbackConfig(
        opacity=0.6,
        opacity_cap=0.6,
        decay=1.0,
        clean_source_floor=0.3,
        motion_suppression=1.0,
        edge_protection=0.0,
        scene_change_threshold=1.0,
    )
    feedback = ControlledRecursiveFeedback(config)
    feedback.process(old, _synthetic_analysis(shape))
    motion = np.zeros(shape[:2], dtype=np.float32)
    motion[:, : shape[1] // 2] = 1.0
    analysis = replace(_synthetic_analysis(shape), motion_mask=motion)

    result = feedback.process(current, analysis)

    assert np.array_equal(result.frame[:, : shape[1] // 2], current[:, : shape[1] // 2])
    assert float(np.mean(result.frame[:, shape[1] // 2 :])) > 20.0


def test_feedback_refuses_analysis_or_history_from_another_resolution():
    shape = (24, 36, 3)
    feedback = ControlledRecursiveFeedback()
    frame = np.zeros(shape, dtype=np.uint8)
    feedback.process(frame, _synthetic_analysis(shape))

    other_shape = (12, 18, 3)
    with pytest.raises(MasterResolutionError):
        feedback.process(
            np.zeros(other_shape, dtype=np.uint8),
            _synthetic_analysis(other_shape),
        )


def test_combined_stage_is_reproducible_and_reset_rebinds_every_temporal_owner():
    previous, current = _moving_pair()
    params = dict(
        audio_fft_bands=[0.2, 0.7, 0.9, 0.4, 0.1, 0.8],
        transient=0.4,
        beat_state=0.75,
    )
    first_stage = MasterMultimodalFeedbackStage()
    second_stage = MasterMultimodalFeedbackStage()
    first_stage.process(previous, **params)
    second_stage.process(previous, **params)

    first = first_stage.process(current, **params)
    second = second_stage.process(current, **params)

    assert np.array_equal(first.frame, second.frame)
    assert np.array_equal(
        first.analysis.modulation_mask, second.analysis.modulation_mask
    )

    first_stage.reset()
    replacement = np.zeros((48, 70, 3), dtype=np.uint8)
    rebound = first_stage.process(replacement)
    assert rebound.analysis.master_shape == replacement.shape
    assert rebound.frame is replacement


def test_config_rejects_ambiguous_or_unbounded_controls():
    with pytest.raises(ValueError, match="odd"):
        MultimodalAnalysisConfig(entropy_window=8)
    with pytest.raises(ValueError, match="opacity_cap"):
        ControlledFeedbackConfig(opacity_cap=1.2)
