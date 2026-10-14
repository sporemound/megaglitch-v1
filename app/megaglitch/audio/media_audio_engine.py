"""Live MP4/GIF analysis-to-audio engine used by the global Media Audio page."""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageSequence
from scipy.io import wavfile
from scipy.signal import lfilter, resample_poly

from .audio_device import AudioDevice
from .cdp_graph import CDPGraphProcessor
from .media_audio_sequencer import MediaAudioSequencer
from .scales import midi_to_frequency
from .tx802 import (
    TX802_ALGORITHM_TARGETS,
    TX802_FEEDBACK_OPERATORS,
    parse_tx802_sysex,
    tx802_patch_to_media_params,
)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

TX802_ALGORITHM_NAMES = [
    f"TX802 {index:02d}" for index in range(1, 33)
]

IDM_RATIO_SNAP_TABLES = {
    "harmonic": np.asarray((0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0), dtype=np.float64),
    "metallic": np.asarray((1.414, 1.618, 2.414, 3.141, 5.333, 7.071, 9.14, 13.7), dtype=np.float64),
    "broken": np.asarray((0.73, 1.01, 1.99, 2.01, 4.03, 8.07, 15.97, 21.3), dtype=np.float64),
}

IDM_ALGORITHM_MORPH_GROUPS = (
    (3, 4, 5, 8, 12),
    (18, 22, 24, 26, 27),
    (21, 23, 25, 29, 32),
)

# Six-operator TX802/DX7-style routing maps. Each tuple entry is the target
# operator index for that operator, or -1 when the operator is an audible carrier.
# Operators only modulate lower-numbered operators, which keeps block rendering
# vectorized and callback-safe.
MEDIA_AUDIO_MOD_SOURCES = (
    "None",
    "brightness",
    "contrast",
    "motion",
    "entropy",
    "edge_density",
    "red_energy",
    "green_energy",
    "blue_energy",
    "glitch_spike",
    "center_mass_x",
    "center_mass_y",
)

SACRED_MICROTONAL_PRESETS = {
    "174 Hz - Golden Foundation": {
        "carrier_freq": 174.0,
        "modulator_ratio": 1.61803398875,
        "modulation_index": 2.4,
        "algorithm": 5,
        "voice_count": 5,
        "poly_spread": 0.32,
        "chord_depth": 0.42,
        "timbre_spread": 0.36,
        "visual_drive": 0.78,
        "feedback_color": 0.18,
        "motion_gating": 0.18,
        "motion_response": 1.1,
        "brightness_response": 0.72,
        "color_stereo": 0.55,
    },
    "285 Hz - Undecimal Bloom 11:8": {
        "carrier_freq": 285.0,
        "modulator_ratio": 11.0 / 8.0,
        "modulation_index": 3.2,
        "algorithm": 12,
        "voice_count": 6,
        "poly_spread": 0.46,
        "chord_depth": 0.58,
        "timbre_spread": 0.5,
        "visual_drive": 0.95,
        "feedback_color": 0.26,
        "motion_gating": 0.22,
        "motion_response": 1.3,
        "brightness_response": 0.68,
        "color_stereo": 0.62,
    },
    "396 Hz - Harmonic Seventh 7:4": {
        "carrier_freq": 396.0,
        "modulator_ratio": 7.0 / 4.0,
        "modulation_index": 2.8,
        "algorithm": 18,
        "voice_count": 7,
        "poly_spread": 0.52,
        "chord_depth": 0.66,
        "timbre_spread": 0.54,
        "visual_drive": 1.05,
        "feedback_color": 0.34,
        "motion_gating": 0.26,
        "motion_response": 1.5,
        "brightness_response": 0.62,
        "color_stereo": 0.68,
    },
    "417 Hz - Tridecimal Gate 13:8": {
        "carrier_freq": 417.0,
        "modulator_ratio": 13.0 / 8.0,
        "modulation_index": 4.1,
        "algorithm": 24,
        "voice_count": 7,
        "poly_spread": 0.6,
        "chord_depth": 0.72,
        "timbre_spread": 0.66,
        "visual_drive": 1.2,
        "feedback_color": 0.42,
        "motion_gating": 0.34,
        "motion_response": 1.65,
        "brightness_response": 0.58,
        "color_stereo": 0.74,
    },
    "432 Hz - Golden Spiral": {
        "carrier_freq": 432.0,
        "modulator_ratio": 1.61803398875,
        "modulation_index": 2.2,
        "algorithm": 8,
        "voice_count": 6,
        "poly_spread": 0.48,
        "chord_depth": 0.62,
        "timbre_spread": 0.48,
        "visual_drive": 0.92,
        "feedback_color": 0.28,
        "motion_gating": 0.2,
        "motion_response": 0.9,
        "brightness_response": 0.76,
        "color_stereo": 0.5,
    },
    "528 Hz - Undecimal Heart 11:8": {
        "carrier_freq": 528.0,
        "modulator_ratio": 11.0 / 8.0,
        "modulation_index": 3.6,
        "algorithm": 29,
        "voice_count": 8,
        "poly_spread": 0.56,
        "chord_depth": 0.74,
        "timbre_spread": 0.62,
        "visual_drive": 1.08,
        "feedback_color": 0.36,
        "motion_gating": 0.24,
        "motion_response": 1.2,
        "brightness_response": 0.7,
        "color_stereo": 0.7,
    },
    "963 Hz - Harmonic Crown 7:6": {
        "carrier_freq": 963.0,
        "modulator_ratio": 7.0 / 6.0,
        "modulation_index": 1.8,
        "algorithm": 32,
        "voice_count": 5,
        "poly_spread": 0.38,
        "chord_depth": 0.5,
        "timbre_spread": 0.42,
        "visual_drive": 0.72,
        "feedback_color": 0.2,
        "motion_gating": 0.16,
        "motion_response": 0.75,
        "brightness_response": 0.64,
        "color_stereo": 0.82,
    },
    "TX802 Drone - Brass Air Stack": {
        "carrier_freq": 82.4069,
        "modulator_ratio": 1.0,
        "modulation_index": 2.8,
        "feedback": 0.12,
        "algorithm": 1,
        "voice_count": 8,
        "poly_spread": 0.18,
        "chord_depth": 0.22,
        "timbre_spread": 0.28,
        "visual_drive": 0.55,
        "feedback_color": 0.2,
        "motion_gating": 0.08,
        "motion_response": 0.45,
        "brightness_response": 0.58,
        "color_stereo": 0.72,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 1.0,
        "operator_ratio_3": 2.0,
        "operator_ratio_4": 3.0,
        "operator_ratio_5": 4.0,
        "operator_ratio_6": 0.5,
        "operator_level_1": 0.9,
        "operator_level_2": 0.62,
        "operator_level_3": 0.46,
        "operator_level_4": 0.34,
        "operator_level_5": 0.24,
        "operator_level_6": 0.16,
    },
    "TX802 Drone - Glass Cathedral": {
        "carrier_freq": 110.0,
        "modulator_ratio": 1.41421356237,
        "modulation_index": 4.6,
        "feedback": 0.05,
        "algorithm": 8,
        "voice_count": 10,
        "poly_spread": 0.34,
        "chord_depth": 0.48,
        "timbre_spread": 0.62,
        "visual_drive": 0.72,
        "feedback_color": 0.14,
        "motion_gating": 0.05,
        "motion_response": 0.32,
        "brightness_response": 0.7,
        "color_stereo": 0.92,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 2.01,
        "operator_ratio_3": 3.997,
        "operator_ratio_4": 5.005,
        "operator_ratio_5": 7.003,
        "operator_ratio_6": 8.0,
        "operator_level_1": 0.68,
        "operator_level_2": 0.58,
        "operator_level_3": 0.52,
        "operator_level_4": 0.3,
        "operator_level_5": 0.28,
        "operator_level_6": 0.18,
        "operator_detune_2": 3.0,
        "operator_detune_3": -5.0,
        "operator_detune_5": 8.0,
    },
    "TX802 Drone - Dark Metal Bed": {
        "carrier_freq": 55.0,
        "modulator_ratio": 1.25,
        "modulation_index": 7.2,
        "feedback": 0.28,
        "algorithm": 24,
        "voice_count": 6,
        "poly_spread": 0.24,
        "chord_depth": 0.18,
        "timbre_spread": 0.78,
        "visual_drive": 0.82,
        "feedback_color": 0.4,
        "motion_gating": 0.12,
        "motion_response": 0.7,
        "brightness_response": 0.45,
        "color_stereo": 0.64,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 0.5,
        "operator_ratio_3": 1.5,
        "operator_ratio_4": 2.5,
        "operator_ratio_5": 6.0,
        "operator_ratio_6": 9.0,
        "operator_level_1": 0.78,
        "operator_level_2": 0.7,
        "operator_level_3": 0.54,
        "operator_level_4": 0.46,
        "operator_level_5": 0.36,
        "operator_level_6": 0.28,
        "operator_detune_4": -11.0,
        "operator_detune_6": 17.0,
    },
    "TX802 Drone - Slow Organ Swell": {
        "carrier_freq": 65.4064,
        "modulator_ratio": 2.0,
        "modulation_index": 1.4,
        "feedback": 0.02,
        "algorithm": 32,
        "voice_count": 5,
        "poly_spread": 0.2,
        "chord_depth": 0.7,
        "timbre_spread": 0.24,
        "visual_drive": 0.48,
        "feedback_color": 0.08,
        "motion_gating": 0.03,
        "motion_response": 0.2,
        "brightness_response": 0.82,
        "color_stereo": 0.58,
        "operator_ratio_1": 0.5,
        "operator_ratio_2": 1.0,
        "operator_ratio_3": 2.0,
        "operator_ratio_4": 3.0,
        "operator_ratio_5": 4.0,
        "operator_ratio_6": 6.0,
        "operator_level_1": 0.72,
        "operator_level_2": 0.64,
        "operator_level_3": 0.46,
        "operator_level_4": 0.36,
        "operator_level_5": 0.28,
        "operator_level_6": 0.2,
    },
    "TX802 Drone - Ice Feedback Halo": {
        "carrier_freq": 146.8324,
        "modulator_ratio": 1.61803398875,
        "modulation_index": 5.4,
        "feedback": 0.18,
        "algorithm": 17,
        "voice_count": 9,
        "poly_spread": 0.42,
        "chord_depth": 0.38,
        "timbre_spread": 0.72,
        "visual_drive": 0.9,
        "feedback_color": 0.24,
        "motion_gating": 0.06,
        "motion_response": 0.42,
        "brightness_response": 0.76,
        "color_stereo": 1.0,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 1.997,
        "operator_ratio_3": 2.414,
        "operator_ratio_4": 3.0,
        "operator_ratio_5": 5.0,
        "operator_ratio_6": 11.0,
        "operator_level_1": 0.64,
        "operator_level_2": 0.52,
        "operator_level_3": 0.46,
        "operator_level_4": 0.38,
        "operator_level_5": 0.3,
        "operator_level_6": 0.22,
        "operator_detune_2": -4.0,
        "operator_detune_3": 6.0,
        "operator_detune_6": 13.0,
    },
    "TX802 Drone - Fixed Bell Undercurrent": {
        "carrier_freq": 73.4162,
        "modulator_ratio": 1.0,
        "modulation_index": 3.8,
        "feedback": 0.1,
        "algorithm": 30,
        "voice_count": 7,
        "poly_spread": 0.28,
        "chord_depth": 0.34,
        "timbre_spread": 0.46,
        "visual_drive": 0.64,
        "feedback_color": 0.18,
        "motion_gating": 0.04,
        "motion_response": 0.36,
        "brightness_response": 0.68,
        "color_stereo": 0.82,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 2.0,
        "operator_ratio_3": 3.0,
        "operator_ratio_4": 4.0,
        "operator_ratio_5": 5.0,
        "operator_ratio_6": 6.0,
        "operator_level_1": 0.76,
        "operator_level_2": 0.48,
        "operator_level_3": 0.42,
        "operator_level_4": 0.34,
        "operator_level_5": 0.28,
        "operator_level_6": 0.2,
        "operator_fixed_5": True,
        "operator_fixed_freq_5": 293.6648,
        "operator_fixed_6": True,
        "operator_fixed_freq_6": 587.3295,
        "operator_pan_5": -0.8,
        "operator_pan_6": 0.8,
    },
}

IDM_MEDIA_AUDIO_PRESETS = {
    "IDM - Amber Lattice Drift": {
        "master_gain": 0.24,
        "carrier_freq": 110.0,
        "modulator_ratio": 1.5,
        "modulation_index": 2.6,
        "feedback": 0.06,
        "algorithm": 8,
        "voice_count": 8,
        "poly_spread": 0.28,
        "chord_depth": 0.46,
        "timbre_spread": 0.48,
        "visual_drive": 1.08,
        "visual_influence": 1.45,
        "visual_evolution": 0.78,
        "visual_pitch_motion": 0.22,
        "visual_ir_motion": 0.38,
        "feedback_color": 0.18,
        "motion_gating": 0.08,
        "motion_response": 0.72,
        "brightness_response": 0.74,
        "color_stereo": 0.78,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 1.5,
        "operator_ratio_3": 2.0,
        "operator_ratio_4": 3.0,
        "operator_ratio_5": 5.0,
        "operator_ratio_6": 0.5,
        "operator_level_1": 0.74,
        "operator_level_2": 0.5,
        "operator_level_3": 0.38,
        "operator_level_4": 0.28,
        "operator_level_5": 0.24,
        "operator_level_6": 0.18,
        "operator_detune_2": -3.0,
        "operator_detune_5": 5.0,
        "operator_pan_5": -0.45,
        "operator_pan_6": 0.45,
    },
    "IDM - Glassine Tape Halo": {
        "master_gain": 0.22,
        "carrier_freq": 146.832,
        "modulator_ratio": 1.41421356237,
        "modulation_index": 3.4,
        "feedback": 0.08,
        "algorithm": 17,
        "voice_count": 9,
        "poly_spread": 0.36,
        "chord_depth": 0.42,
        "timbre_spread": 0.62,
        "visual_drive": 1.02,
        "visual_influence": 1.55,
        "visual_evolution": 0.82,
        "visual_pitch_motion": 0.18,
        "visual_ir_motion": 0.44,
        "feedback_color": 0.2,
        "motion_gating": 0.06,
        "motion_response": 0.58,
        "brightness_response": 0.78,
        "color_stereo": 0.96,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 1.997,
        "operator_ratio_3": 2.414,
        "operator_ratio_4": 3.005,
        "operator_ratio_5": 5.0,
        "operator_ratio_6": 7.0,
        "operator_level_1": 0.66,
        "operator_level_2": 0.52,
        "operator_level_3": 0.45,
        "operator_level_4": 0.34,
        "operator_level_5": 0.26,
        "operator_level_6": 0.2,
        "operator_detune_2": -4.0,
        "operator_detune_3": 6.0,
        "operator_detune_6": 9.0,
        "operator_pan_3": -0.35,
        "operator_pan_6": 0.72,
    },
    "IDM - Soft Packet Rain": {
        "master_gain": 0.2,
        "carrier_freq": 220.0,
        "modulator_ratio": 1.25,
        "modulation_index": 4.2,
        "feedback": 0.1,
        "algorithm": 30,
        "voice_count": 4,
        "poly_spread": 0.22,
        "chord_depth": 0.28,
        "timbre_spread": 0.7,
        "visual_drive": 1.18,
        "visual_influence": 1.6,
        "visual_evolution": 0.72,
        "visual_pitch_motion": 0.2,
        "visual_ir_motion": 0.55,
        "feedback_color": 0.24,
        "motion_gating": 0.18,
        "motion_response": 1.05,
        "brightness_response": 0.66,
        "color_stereo": 0.9,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 2.0,
        "operator_ratio_3": 3.0,
        "operator_ratio_4": 4.0,
        "operator_ratio_5": 5.0,
        "operator_ratio_6": 6.0,
        "operator_level_1": 0.58,
        "operator_level_2": 0.48,
        "operator_level_3": 0.4,
        "operator_level_4": 0.28,
        "operator_level_5": 0.3,
        "operator_level_6": 0.24,
        "operator_fixed_5": True,
        "operator_fixed_freq_5": 880.0,
        "operator_fixed_6": True,
        "operator_fixed_freq_6": 1320.0,
        "operator_pan_5": -0.82,
        "operator_pan_6": 0.82,
    },
    "IDM - Boards Melt Organ": {
        "master_gain": 0.26,
        "carrier_freq": 65.406,
        "modulator_ratio": 2.0,
        "modulation_index": 1.6,
        "feedback": 0.03,
        "algorithm": 32,
        "voice_count": 12,
        "poly_spread": 0.24,
        "chord_depth": 0.72,
        "timbre_spread": 0.3,
        "visual_drive": 0.86,
        "visual_influence": 1.25,
        "visual_evolution": 0.58,
        "visual_pitch_motion": 0.14,
        "visual_ir_motion": 0.36,
        "feedback_color": 0.08,
        "motion_gating": 0.04,
        "motion_response": 0.32,
        "brightness_response": 0.84,
        "color_stereo": 0.64,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 0.5,
        "operator_ratio_2": 1.0,
        "operator_ratio_3": 2.0,
        "operator_ratio_4": 3.0,
        "operator_ratio_5": 4.0,
        "operator_ratio_6": 6.0,
        "operator_level_1": 0.78,
        "operator_level_2": 0.66,
        "operator_level_3": 0.48,
        "operator_level_4": 0.36,
        "operator_level_5": 0.24,
        "operator_level_6": 0.18,
        "operator_detune_2": -7.0,
        "operator_detune_4": 5.0,
    },
    "IDM - Oval Dust Harmonics": {
        "master_gain": 0.23,
        "carrier_freq": 174.0,
        "modulator_ratio": 11.0 / 8.0,
        "modulation_index": 3.1,
        "feedback": 0.07,
        "algorithm": 12,
        "voice_count": 7,
        "poly_spread": 0.3,
        "chord_depth": 0.5,
        "timbre_spread": 0.68,
        "visual_drive": 1.28,
        "visual_influence": 1.7,
        "visual_evolution": 0.86,
        "visual_pitch_motion": 0.26,
        "visual_ir_motion": 0.48,
        "feedback_color": 0.2,
        "motion_gating": 0.12,
        "motion_response": 0.9,
        "brightness_response": 0.68,
        "color_stereo": 0.84,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 1.375,
        "operator_ratio_3": 2.75,
        "operator_ratio_4": 3.0,
        "operator_ratio_5": 5.5,
        "operator_ratio_6": 8.0,
        "operator_level_1": 0.64,
        "operator_level_2": 0.56,
        "operator_level_3": 0.42,
        "operator_level_4": 0.32,
        "operator_level_5": 0.3,
        "operator_level_6": 0.22,
        "operator_detune_3": -6.0,
        "operator_detune_5": 11.0,
        "operator_pan_2": -0.28,
        "operator_pan_6": 0.62,
    },
    "IDM - Lichen Chord Memory": {
        "master_gain": 0.25,
        "carrier_freq": 82.406,
        "modulator_ratio": 1.61803398875,
        "modulation_index": 2.1,
        "feedback": 0.05,
        "algorithm": 5,
        "voice_count": 5,
        "poly_spread": 0.42,
        "chord_depth": 0.82,
        "timbre_spread": 0.44,
        "visual_drive": 0.98,
        "visual_influence": 1.38,
        "visual_evolution": 0.74,
        "visual_pitch_motion": 0.2,
        "visual_ir_motion": 0.34,
        "feedback_color": 0.14,
        "motion_gating": 0.03,
        "motion_response": 0.42,
        "brightness_response": 0.8,
        "color_stereo": 0.72,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 0.5,
        "operator_ratio_2": 1.0,
        "operator_ratio_3": 1.5,
        "operator_ratio_4": 2.0,
        "operator_ratio_5": 3.0,
        "operator_ratio_6": 4.0,
        "operator_level_1": 0.76,
        "operator_level_2": 0.58,
        "operator_level_3": 0.48,
        "operator_level_4": 0.36,
        "operator_level_5": 0.28,
        "operator_level_6": 0.2,
        "operator_detune_2": -5.0,
        "operator_detune_5": 7.0,
    },
    "IDM - Frosted Microbells": {
        "master_gain": 0.18,
        "carrier_freq": 293.665,
        "modulator_ratio": 1.5,
        "modulation_index": 5.0,
        "feedback": 0.12,
        "algorithm": 29,
        "voice_count": 7,
        "poly_spread": 0.2,
        "chord_depth": 0.32,
        "timbre_spread": 0.74,
        "visual_drive": 1.18,
        "visual_influence": 1.65,
        "visual_evolution": 0.9,
        "visual_pitch_motion": 0.16,
        "visual_ir_motion": 0.58,
        "feedback_color": 0.22,
        "motion_gating": 0.1,
        "motion_response": 0.84,
        "brightness_response": 0.7,
        "color_stereo": 1.0,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 2.0,
        "operator_ratio_3": 3.997,
        "operator_ratio_4": 5.0,
        "operator_ratio_5": 7.0,
        "operator_ratio_6": 9.0,
        "operator_level_1": 0.52,
        "operator_level_2": 0.48,
        "operator_level_3": 0.46,
        "operator_level_4": 0.34,
        "operator_level_5": 0.28,
        "operator_level_6": 0.24,
        "operator_fixed_5": True,
        "operator_fixed_freq_5": 1174.659,
        "operator_fixed_6": True,
        "operator_fixed_freq_6": 1760.0,
        "operator_detune_3": -4.0,
        "operator_pan_5": -0.9,
        "operator_pan_6": 0.9,
    },
    "IDM - Warm Error Bed": {
        "master_gain": 0.24,
        "carrier_freq": 55.0,
        "modulator_ratio": 1.25,
        "modulation_index": 4.8,
        "feedback": 0.22,
        "algorithm": 24,
        "voice_count": 8,
        "poly_spread": 0.26,
        "chord_depth": 0.3,
        "timbre_spread": 0.8,
        "visual_drive": 1.35,
        "visual_influence": 1.8,
        "visual_evolution": 0.82,
        "visual_pitch_motion": 0.24,
        "visual_ir_motion": 0.42,
        "feedback_color": 0.3,
        "motion_gating": 0.09,
        "motion_response": 0.76,
        "brightness_response": 0.55,
        "color_stereo": 0.7,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 0.5,
        "operator_ratio_3": 1.25,
        "operator_ratio_4": 2.5,
        "operator_ratio_5": 6.0,
        "operator_ratio_6": 9.0,
        "operator_level_1": 0.72,
        "operator_level_2": 0.64,
        "operator_level_3": 0.52,
        "operator_level_4": 0.42,
        "operator_level_5": 0.32,
        "operator_level_6": 0.24,
        "operator_detune_4": -9.0,
        "operator_detune_5": 6.0,
        "operator_detune_6": 14.0,
    },
    "IDM - Slow Raster Choir": {
        "master_gain": 0.26,
        "carrier_freq": 98.0,
        "modulator_ratio": 1.0,
        "modulation_index": 1.9,
        "feedback": 0.04,
        "algorithm": 1,
        "voice_count": 11,
        "poly_spread": 0.3,
        "chord_depth": 0.7,
        "timbre_spread": 0.36,
        "visual_drive": 0.92,
        "visual_influence": 1.32,
        "visual_evolution": 0.68,
        "visual_pitch_motion": 0.2,
        "visual_ir_motion": 0.3,
        "feedback_color": 0.1,
        "motion_gating": 0.05,
        "motion_response": 0.38,
        "brightness_response": 0.88,
        "color_stereo": 0.68,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 0.5,
        "operator_ratio_2": 1.0,
        "operator_ratio_3": 2.0,
        "operator_ratio_4": 2.995,
        "operator_ratio_5": 4.0,
        "operator_ratio_6": 5.0,
        "operator_level_1": 0.82,
        "operator_level_2": 0.58,
        "operator_level_3": 0.5,
        "operator_level_4": 0.36,
        "operator_level_5": 0.22,
        "operator_level_6": 0.16,
        "operator_detune_2": -8.0,
        "operator_detune_4": 4.0,
        "operator_pan_3": -0.25,
        "operator_pan_4": 0.25,
    },
    "IDM - Distant Modem Aurora": {
        "master_gain": 0.21,
        "carrier_freq": 123.471,
        "modulator_ratio": 1.61803398875,
        "modulation_index": 5.4,
        "feedback": 0.3,
        "algorithm": 18,
        "voice_count": 7,
        "poly_spread": 0.34,
        "chord_depth": 0.4,
        "timbre_spread": 0.76,
        "visual_drive": 1.22,
        "visual_influence": 1.72,
        "visual_evolution": 0.94,
        "visual_pitch_motion": 0.28,
        "visual_ir_motion": 0.62,
        "feedback_color": 0.28,
        "motion_gating": 0.12,
        "motion_response": 0.96,
        "brightness_response": 0.62,
        "color_stereo": 0.86,
        "midi_style": "Musical",
        "midi_note_focus": 0.15,
        "midi_pitch_slew_ms": 0.0,
        "midi_timbre_depth": 0.12,
        "midi_note_punch": 0.08,
        "operator_ratio_1": 1.0,
        "operator_ratio_2": 1.618,
        "operator_ratio_3": 2.0,
        "operator_ratio_4": 3.236,
        "operator_ratio_5": 5.0,
        "operator_ratio_6": 8.0,
        "operator_level_1": 0.6,
        "operator_level_2": 0.56,
        "operator_level_3": 0.48,
        "operator_level_4": 0.4,
        "operator_level_5": 0.3,
        "operator_level_6": 0.24,
        "operator_detune_2": -5.0,
        "operator_detune_4": 7.0,
        "operator_detune_6": 12.0,
        "operator_pan_5": -0.64,
        "operator_pan_6": 0.64,
    },
}

TX802_ACCURATE_MEDIA_AUDIO_PRESETS = {
    "TX802 EP - Tine Vector": {
        "tx802_patch_name": "TINE VECTOR",
        "master_gain": 0.24,
        "carrier_freq": 110.0,
        "modulator_ratio": 1.0,
        "modulation_index": 5.8,
        "feedback": 0.08,
        "algorithm": 5,
        "voice_count": 4,
        "poly_spread": 0.18,
        "chord_depth": 0.18,
        "timbre_spread": 0.34,
        "visual_drive": 1.0,
        "visual_influence": 1.25,
        "visual_evolution": 0.56,
        "visual_pitch_motion": 0.12,
        "visual_ir_motion": 0.28,
        "feedback_color": 0.12,
        "motion_gating": 0.08,
        "motion_response": 0.55,
        "brightness_response": 0.72,
        "color_stereo": 0.58,
        "visual_morph_depth": 0.5,
        "lfo_rate": 0.42,
        "lfo_depth": 0.035,
        "lfo_delay": 0.15,
        "lfo_wave": "sine",
        "operator_ratio_1": 1.0, "operator_ratio_2": 14.0, "operator_ratio_3": 1.0, "operator_ratio_4": 2.0, "operator_ratio_5": 3.0, "operator_ratio_6": 6.0,
        "operator_output_level_1": 0.72, "operator_output_level_2": 0.68, "operator_output_level_3": 0.42, "operator_output_level_4": 0.36, "operator_output_level_5": 0.24, "operator_output_level_6": 0.16,
        "operator_velocity_1": 0.4, "operator_velocity_2": 0.85, "operator_velocity_3": 0.25, "operator_velocity_4": 0.55, "operator_velocity_5": 0.35, "operator_velocity_6": 0.4,
    },
    "TX802 Bell - Glass Ratio": {
        "tx802_patch_name": "GLASS RATIO",
        "master_gain": 0.2,
        "carrier_freq": 220.0,
        "modulator_ratio": 1.0,
        "modulation_index": 8.8,
        "feedback": 0.18,
        "algorithm": 8,
        "voice_count": 4,
        "poly_spread": 0.26,
        "chord_depth": 0.28,
        "timbre_spread": 0.72,
        "visual_drive": 1.3,
        "visual_influence": 1.55,
        "visual_evolution": 0.78,
        "visual_pitch_motion": 0.18,
        "visual_ir_motion": 0.62,
        "feedback_color": 0.22,
        "motion_gating": 0.05,
        "motion_response": 0.45,
        "brightness_response": 0.78,
        "color_stereo": 1.0,
        "visual_morph_depth": 0.62,
        "lfo_rate": 0.18,
        "lfo_depth": 0.018,
        "lfo_delay": 0.0,
        "lfo_wave": "triangle",
        "operator_ratio_1": 1.0, "operator_ratio_2": 2.414, "operator_ratio_3": 3.997, "operator_ratio_4": 6.003, "operator_ratio_5": 8.0, "operator_ratio_6": 11.0,
        "operator_output_level_1": 0.48, "operator_output_level_2": 0.7, "operator_output_level_3": 0.52, "operator_output_level_4": 0.42, "operator_output_level_5": 0.34, "operator_output_level_6": 0.28,
        "operator_detune_2": -4.0, "operator_detune_3": 5.0, "operator_detune_6": 11.0,
        "operator_fixed_5": True, "operator_fixed_freq_5": 1174.659, "operator_fixed_6": True, "operator_fixed_freq_6": 1760.0,
    },
    "TX802 Brass - Stack Pressure": {
        "tx802_patch_name": "STACK BRASS",
        "master_gain": 0.28,
        "carrier_freq": 82.4069,
        "modulator_ratio": 1.0,
        "modulation_index": 3.2,
        "feedback": 0.1,
        "algorithm": 1,
        "voice_count": 4,
        "poly_spread": 0.22,
        "chord_depth": 0.26,
        "timbre_spread": 0.28,
        "visual_drive": 0.82,
        "visual_influence": 1.0,
        "visual_evolution": 0.42,
        "visual_pitch_motion": 0.1,
        "visual_ir_motion": 0.22,
        "feedback_color": 0.15,
        "motion_gating": 0.16,
        "motion_response": 1.0,
        "brightness_response": 0.64,
        "color_stereo": 0.62,
        "visual_morph_depth": 0.34,
        "lfo_rate": 4.8,
        "lfo_depth": 0.012,
        "lfo_delay": 0.35,
        "lfo_wave": "sine",
        "operator_ratio_1": 1.0, "operator_ratio_2": 1.0, "operator_ratio_3": 2.0, "operator_ratio_4": 2.0, "operator_ratio_5": 3.0, "operator_ratio_6": 4.0,
        "operator_output_level_1": 0.82, "operator_output_level_2": 0.6, "operator_output_level_3": 0.48, "operator_output_level_4": 0.34, "operator_output_level_5": 0.22, "operator_output_level_6": 0.16,
    },
    "TX802 Bass - Rubber Feedback": {
        "tx802_patch_name": "RUBBER BASS",
        "master_gain": 0.3,
        "carrier_freq": 55.0,
        "modulator_ratio": 0.5,
        "modulation_index": 6.6,
        "feedback": 0.32,
        "algorithm": 24,
        "voice_count": 3,
        "poly_spread": 0.08,
        "chord_depth": 0.0,
        "timbre_spread": 0.55,
        "visual_drive": 1.08,
        "visual_influence": 1.2,
        "visual_evolution": 0.38,
        "visual_pitch_motion": 0.04,
        "visual_ir_motion": 0.08,
        "feedback_color": 0.36,
        "motion_gating": 0.22,
        "motion_response": 1.35,
        "brightness_response": 0.58,
        "color_stereo": 0.22,
        "visual_morph_depth": 0.46,
        "lfo_rate": 0.0,
        "lfo_depth": 0.0,
        "lfo_delay": 0.0,
        "lfo_wave": "sine",
        "operator_ratio_1": 0.5, "operator_ratio_2": 1.0, "operator_ratio_3": 1.0, "operator_ratio_4": 1.5, "operator_ratio_5": 3.0, "operator_ratio_6": 7.0,
        "operator_output_level_1": 0.86, "operator_output_level_2": 0.76, "operator_output_level_3": 0.62, "operator_output_level_4": 0.44, "operator_output_level_5": 0.28, "operator_output_level_6": 0.2,
    },
    "TX802 Perc - Metallic Skin": {
        "tx802_patch_name": "METAL SKIN",
        "master_gain": 0.22,
        "carrier_freq": 146.8324,
        "modulator_ratio": 1.0,
        "modulation_index": 10.5,
        "feedback": 0.44,
        "algorithm": 18,
        "voice_count": 4,
        "poly_spread": 0.16,
        "chord_depth": 0.12,
        "timbre_spread": 0.9,
        "visual_drive": 1.45,
        "visual_influence": 1.8,
        "visual_evolution": 0.86,
        "visual_pitch_motion": 0.2,
        "visual_ir_motion": 0.48,
        "feedback_color": 0.45,
        "motion_gating": 0.35,
        "motion_response": 1.8,
        "brightness_response": 0.5,
        "color_stereo": 0.85,
        "visual_morph_depth": 0.78,
        "lfo_rate": 7.0,
        "lfo_depth": 0.02,
        "lfo_delay": 0.0,
        "lfo_wave": "sample_hold",
        "operator_ratio_1": 1.0, "operator_ratio_2": 3.0, "operator_ratio_3": 5.0, "operator_ratio_4": 7.0, "operator_ratio_5": 9.0, "operator_ratio_6": 13.0,
        "operator_output_level_1": 0.52, "operator_output_level_2": 0.72, "operator_output_level_3": 0.56, "operator_output_level_4": 0.5, "operator_output_level_5": 0.42, "operator_output_level_6": 0.34,
        "operator_fixed_4": True, "operator_fixed_freq_4": 880.0, "operator_fixed_6": True, "operator_fixed_freq_6": 2637.02,
    },
    "TX802 Organ - Choir Parallel": {
        "tx802_patch_name": "CHOIR PARA",
        "master_gain": 0.25,
        "carrier_freq": 65.4064,
        "modulator_ratio": 1.0,
        "modulation_index": 1.2,
        "feedback": 0.02,
        "algorithm": 32,
        "voice_count": 4,
        "poly_spread": 0.28,
        "chord_depth": 0.72,
        "timbre_spread": 0.2,
        "visual_drive": 0.62,
        "visual_influence": 0.9,
        "visual_evolution": 0.52,
        "visual_pitch_motion": 0.16,
        "visual_ir_motion": 0.36,
        "feedback_color": 0.06,
        "motion_gating": 0.04,
        "motion_response": 0.32,
        "brightness_response": 0.86,
        "color_stereo": 0.7,
        "visual_morph_depth": 0.28,
        "lfo_rate": 0.28,
        "lfo_depth": 0.028,
        "lfo_delay": 0.2,
        "lfo_wave": "sine",
        "operator_ratio_1": 0.5, "operator_ratio_2": 1.0, "operator_ratio_3": 2.0, "operator_ratio_4": 3.0, "operator_ratio_5": 4.0, "operator_ratio_6": 6.0,
        "operator_output_level_1": 0.76, "operator_output_level_2": 0.66, "operator_output_level_3": 0.54, "operator_output_level_4": 0.42, "operator_output_level_5": 0.32, "operator_output_level_6": 0.22,
    },
    "TX802 Noise - Feedback Glass": {
        "tx802_patch_name": "FDBK GLASS",
        "master_gain": 0.18,
        "carrier_freq": 123.471,
        "modulator_ratio": 1.618,
        "modulation_index": 12.0,
        "feedback": 0.62,
        "algorithm": 27,
        "voice_count": 4,
        "poly_spread": 0.36,
        "chord_depth": 0.22,
        "timbre_spread": 1.0,
        "visual_drive": 1.8,
        "visual_influence": 2.0,
        "visual_evolution": 1.0,
        "visual_pitch_motion": 0.34,
        "visual_ir_motion": 0.58,
        "feedback_color": 0.72,
        "motion_gating": 0.12,
        "motion_response": 1.1,
        "brightness_response": 0.48,
        "color_stereo": 1.0,
        "visual_morph_depth": 0.9,
        "lfo_rate": 2.6,
        "lfo_depth": 0.04,
        "lfo_delay": 0.0,
        "lfo_wave": "sample_hold",
        "operator_ratio_1": 1.0, "operator_ratio_2": 1.414, "operator_ratio_3": 2.718, "operator_ratio_4": 4.0, "operator_ratio_5": 5.657, "operator_ratio_6": 8.0,
        "operator_output_level_1": 0.46, "operator_output_level_2": 0.82, "operator_output_level_3": 0.74, "operator_output_level_4": 0.58, "operator_output_level_5": 0.5, "operator_output_level_6": 0.44,
    },
    "TX802 Pad - Evolving Matrix": {
        "tx802_patch_name": "EVO MATRIX",
        "master_gain": 0.23,
        "carrier_freq": 98.0,
        "modulator_ratio": 1.0,
        "modulation_index": 4.0,
        "feedback": 0.16,
        "algorithm": 16,
        "voice_count": 4,
        "poly_spread": 0.42,
        "chord_depth": 0.58,
        "timbre_spread": 0.6,
        "visual_drive": 1.1,
        "visual_influence": 1.45,
        "visual_evolution": 0.92,
        "visual_pitch_motion": 0.24,
        "visual_ir_motion": 0.5,
        "feedback_color": 0.2,
        "motion_gating": 0.06,
        "motion_response": 0.62,
        "brightness_response": 0.76,
        "color_stereo": 0.9,
        "visual_morph_depth": 0.64,
        "lfo_rate": 0.12,
        "lfo_depth": 0.045,
        "lfo_delay": 0.8,
        "lfo_wave": "triangle",
        "operator_ratio_1": 0.5, "operator_ratio_2": 1.0, "operator_ratio_3": 1.5, "operator_ratio_4": 2.0, "operator_ratio_5": 3.0, "operator_ratio_6": 5.0,
        "operator_output_level_1": 0.68, "operator_output_level_2": 0.58, "operator_output_level_3": 0.52, "operator_output_level_4": 0.44, "operator_output_level_5": 0.34, "operator_output_level_6": 0.26,
    },
}

IDM_MEDIA_AUDIO_PRESETS = {
    "IDM: Windowlicker Glass": {
        "tx802_patch_name": "WLKR GLASS",
        "master_gain": 0.2,
        "carrier_freq": 220.0,
        "algorithm": 8,
        "voice_count": 2,
        "poly_spread": 0.08,
        "chord_depth": 0.05,
        "timbre_spread": 0.78,
        "visual_drive": 1.35,
        "visual_influence": 1.2,
        "visual_evolution": 0.82,
        "visual_pitch_motion": 0.04,
        "feedback": 0.22,
        "feedback_color": 0.16,
        "motion_gating": 0.24,
        "motion_response": 1.6,
        "brightness_response": 0.68,
        "color_stereo": 0.92,
        "visual_morph_depth": 0.28,
        "idm_mode": True,
        "idm_intensity": 0.72,
        "idm_bpm": 172.0,
        "idm_grid_rate": 16.0,
        "idm_probability": 0.78,
        "idm_microtiming": 0.18,
        "idm_env_slice_depth": 0.82,
        "idm_modulator_chop_depth": 0.9,
        "idm_carrier_chop_depth": 0.42,
        "idm_ratio_snap_depth": 0.55,
        "idm_algorithm_morph_depth": 0.28,
        "idm_bitcrush_depth": 0.12,
        "idm_sample_rate_fold": 0.1,
        "idm_wavefold_depth": 0.18,
        "operator_ratio_1": 1.0, "operator_ratio_2": 2.414, "operator_ratio_3": 3.997, "operator_ratio_4": 6.003, "operator_ratio_5": 8.0, "operator_ratio_6": 11.0,
        "operator_output_level_1": 0.5, "operator_output_level_2": 0.74, "operator_output_level_3": 0.62, "operator_output_level_4": 0.5, "operator_output_level_5": 0.32, "operator_output_level_6": 0.24,
    },
    "IDM: Drill EP Fragments": {
        "tx802_patch_name": "DRILL EP",
        "master_gain": 0.24,
        "carrier_freq": 110.0,
        "algorithm": 5,
        "voice_count": 1,
        "poly_spread": 0.0,
        "chord_depth": 0.0,
        "timbre_spread": 0.42,
        "visual_drive": 1.15,
        "visual_influence": 0.9,
        "visual_evolution": 0.58,
        "visual_pitch_motion": 0.02,
        "feedback": 0.1,
        "feedback_color": 0.12,
        "motion_gating": 0.3,
        "motion_response": 2.2,
        "brightness_response": 0.62,
        "color_stereo": 0.45,
        "visual_morph_depth": 0.2,
        "idm_mode": True,
        "idm_intensity": 0.86,
        "idm_bpm": 185.0,
        "idm_grid_rate": 32.0,
        "idm_probability": 0.68,
        "idm_microtiming": 0.28,
        "idm_stutter_depth": 0.62,
        "idm_retrigger_depth": 0.78,
        "idm_env_slice_depth": 0.9,
        "idm_modulator_chop_depth": 1.0,
        "idm_carrier_chop_depth": 0.58,
        "idm_ratio_snap_depth": 0.36,
        "idm_bitcrush_depth": 0.08,
        "idm_sample_rate_fold": 0.18,
        "operator_ratio_1": 1.0, "operator_ratio_2": 14.0, "operator_ratio_3": 1.0, "operator_ratio_4": 2.0, "operator_ratio_5": 3.0, "operator_ratio_6": 6.0,
        "operator_output_level_1": 0.68, "operator_output_level_2": 0.76, "operator_output_level_3": 0.44, "operator_output_level_4": 0.4, "operator_output_level_5": 0.26, "operator_output_level_6": 0.18,
    },
    "IDM: Rubber FM Bass": {
        "tx802_patch_name": "IDM RUBBER",
        "master_gain": 0.3,
        "carrier_freq": 55.0,
        "algorithm": 24,
        "voice_count": 1,
        "poly_spread": 0.0,
        "chord_depth": 0.0,
        "timbre_spread": 0.5,
        "visual_drive": 1.05,
        "visual_influence": 0.8,
        "visual_evolution": 0.35,
        "visual_pitch_motion": 0.0,
        "feedback": 0.42,
        "feedback_color": 0.28,
        "motion_gating": 0.22,
        "motion_response": 1.4,
        "brightness_response": 0.7,
        "color_stereo": 0.18,
        "visual_morph_depth": 0.18,
        "idm_mode": True,
        "idm_intensity": 0.62,
        "idm_bpm": 138.0,
        "idm_grid_rate": 16.0,
        "idm_probability": 0.86,
        "idm_env_slice_depth": 0.48,
        "idm_modulator_chop_depth": 0.68,
        "idm_carrier_chop_depth": 0.24,
        "idm_ratio_snap_depth": 0.22,
        "idm_wavefold_depth": 0.26,
        "operator_ratio_1": 0.5, "operator_ratio_2": 1.0, "operator_ratio_3": 1.0, "operator_ratio_4": 1.5, "operator_ratio_5": 3.0, "operator_ratio_6": 7.0,
        "operator_output_level_1": 0.9, "operator_output_level_2": 0.8, "operator_output_level_3": 0.64, "operator_output_level_4": 0.44, "operator_output_level_5": 0.28, "operator_output_level_6": 0.2,
    },
    "IDM: Metallic Ratchet": {
        "tx802_patch_name": "RATCHET",
        "master_gain": 0.18,
        "carrier_freq": 146.8324,
        "algorithm": 18,
        "voice_count": 2,
        "poly_spread": 0.08,
        "chord_depth": 0.08,
        "timbre_spread": 1.0,
        "visual_drive": 1.75,
        "visual_influence": 1.4,
        "visual_evolution": 0.9,
        "visual_pitch_motion": 0.08,
        "feedback": 0.58,
        "feedback_color": 0.7,
        "motion_gating": 0.38,
        "motion_response": 2.6,
        "brightness_response": 0.5,
        "color_stereo": 1.0,
        "visual_morph_depth": 0.44,
        "idm_mode": True,
        "idm_intensity": 0.96,
        "idm_bpm": 196.0,
        "idm_grid_rate": 32.0,
        "idm_probability": 0.74,
        "idm_microtiming": 0.16,
        "idm_stutter_depth": 0.8,
        "idm_env_slice_depth": 1.0,
        "idm_modulator_chop_depth": 1.0,
        "idm_carrier_chop_depth": 0.72,
        "idm_ratio_snap_depth": 0.82,
        "idm_algorithm_morph_depth": 0.36,
        "idm_bitcrush_depth": 0.28,
        "idm_sample_rate_fold": 0.36,
        "idm_wavefold_depth": 0.32,
        "operator_ratio_1": 1.0, "operator_ratio_2": 3.0, "operator_ratio_3": 5.0, "operator_ratio_4": 7.0, "operator_ratio_5": 9.0, "operator_ratio_6": 13.0,
        "operator_output_level_1": 0.48, "operator_output_level_2": 0.82, "operator_output_level_3": 0.62, "operator_output_level_4": 0.56, "operator_output_level_5": 0.48, "operator_output_level_6": 0.38,
    },
    "IDM: Feedback Insect": {
        "tx802_patch_name": "FDBK INSECT",
        "master_gain": 0.16,
        "carrier_freq": 123.471,
        "algorithm": 27,
        "voice_count": 2,
        "poly_spread": 0.22,
        "chord_depth": 0.06,
        "timbre_spread": 1.0,
        "visual_drive": 1.9,
        "visual_influence": 1.7,
        "visual_evolution": 1.0,
        "visual_pitch_motion": 0.12,
        "feedback": 0.76,
        "feedback_color": 0.9,
        "motion_gating": 0.18,
        "motion_response": 1.8,
        "brightness_response": 0.48,
        "color_stereo": 1.0,
        "visual_morph_depth": 0.58,
        "idm_mode": True,
        "idm_intensity": 0.88,
        "idm_bpm": 161.0,
        "idm_grid_rate": 16.0,
        "idm_probability": 0.72,
        "idm_microtiming": 0.34,
        "idm_env_slice_depth": 0.72,
        "idm_modulator_chop_depth": 0.92,
        "idm_carrier_chop_depth": 0.36,
        "idm_ratio_snap_depth": 0.7,
        "idm_algorithm_morph_depth": 0.48,
        "idm_bitcrush_depth": 0.18,
        "idm_wavefold_depth": 0.42,
        "operator_ratio_1": 1.0, "operator_ratio_2": 1.414, "operator_ratio_3": 2.718, "operator_ratio_4": 4.0, "operator_ratio_5": 5.657, "operator_ratio_6": 8.0,
        "operator_output_level_1": 0.42, "operator_output_level_2": 0.86, "operator_output_level_3": 0.78, "operator_output_level_4": 0.62, "operator_output_level_5": 0.54, "operator_output_level_6": 0.48,
    },
    "IDM: Chopped Choir": {
        "tx802_patch_name": "CHOP CHOIR",
        "master_gain": 0.22,
        "carrier_freq": 65.4064,
        "algorithm": 32,
        "voice_count": 3,
        "poly_spread": 0.18,
        "chord_depth": 0.38,
        "timbre_spread": 0.24,
        "visual_drive": 0.8,
        "visual_influence": 0.85,
        "visual_evolution": 0.46,
        "visual_pitch_motion": 0.06,
        "feedback": 0.06,
        "feedback_color": 0.1,
        "motion_gating": 0.34,
        "motion_response": 1.55,
        "brightness_response": 0.82,
        "color_stereo": 0.82,
        "visual_morph_depth": 0.22,
        "idm_mode": True,
        "idm_intensity": 0.58,
        "idm_bpm": 151.0,
        "idm_grid_rate": 16.0,
        "idm_probability": 0.62,
        "idm_env_slice_depth": 0.76,
        "idm_modulator_chop_depth": 0.48,
        "idm_carrier_chop_depth": 0.86,
        "idm_ratio_snap_depth": 0.18,
        "idm_sample_rate_fold": 0.08,
        "operator_ratio_1": 0.5, "operator_ratio_2": 1.0, "operator_ratio_3": 2.0, "operator_ratio_4": 3.0, "operator_ratio_5": 4.0, "operator_ratio_6": 6.0,
        "operator_output_level_1": 0.76, "operator_output_level_2": 0.68, "operator_output_level_3": 0.56, "operator_output_level_4": 0.42, "operator_output_level_5": 0.32, "operator_output_level_6": 0.22,
    },
}

LUSH_PAD_DRONE_PRESETS = {
    "Lush Pad - Velvet Operator Cloud": {
        "tx802_patch_name": "LUSH CLOUD",
        "master_gain": 0.24,
        "carrier_freq": 65.4064,
        "algorithm": 32,
        "voice_count": 12,
        "poly_spread": 0.34,
        "chord_depth": 0.78,
        "timbre_spread": 0.34,
        "visual_drive": 0.72,
        "visual_influence": 0.82,
        "visual_evolution": 0.64,
        "visual_pitch_motion": 0.08,
        "visual_ir_motion": 0.42,
        "feedback": 0.03,
        "feedback_color": 0.08,
        "motion_gating": 0.02,
        "motion_response": 0.24,
        "brightness_response": 0.86,
        "color_stereo": 0.86,
        "visual_morph_depth": 0.24,
        "lfo_rate": 0.08,
        "lfo_depth": 0.012,
        "lfo_delay": 1.6,
        "operator_ratio_1": 0.5, "operator_ratio_2": 1.0, "operator_ratio_3": 2.0, "operator_ratio_4": 3.0, "operator_ratio_5": 4.0, "operator_ratio_6": 6.0,
        "operator_output_level_1": 0.78, "operator_output_level_2": 0.66, "operator_output_level_3": 0.5, "operator_output_level_4": 0.38, "operator_output_level_5": 0.26, "operator_output_level_6": 0.18,
        "filter_cutoff": 5200.0, "filter_resonance": 0.08, "amp_attack_ms": 3600.0, "amp_release_ms": 8500.0,
    },
    "Lush Pad - Glass Choir Halo": {
        "tx802_patch_name": "GLASS HALO",
        "master_gain": 0.2,
        "carrier_freq": 98.0,
        "algorithm": 29,
        "voice_count": 10,
        "poly_spread": 0.46,
        "chord_depth": 0.62,
        "timbre_spread": 0.68,
        "visual_drive": 0.9,
        "visual_influence": 1.05,
        "visual_evolution": 0.82,
        "visual_pitch_motion": 0.1,
        "visual_ir_motion": 0.58,
        "feedback": 0.11,
        "feedback_color": 0.18,
        "motion_gating": 0.03,
        "motion_response": 0.38,
        "brightness_response": 0.78,
        "color_stereo": 1.0,
        "visual_morph_depth": 0.36,
        "lfo_rate": 0.05,
        "lfo_depth": 0.018,
        "lfo_delay": 2.4,
        "operator_ratio_1": 1.0, "operator_ratio_2": 2.414, "operator_ratio_3": 3.997, "operator_ratio_4": 5.005, "operator_ratio_5": 8.0, "operator_ratio_6": 11.0,
        "operator_output_level_1": 0.5, "operator_output_level_2": 0.62, "operator_output_level_3": 0.5, "operator_output_level_4": 0.36, "operator_output_level_5": 0.28, "operator_output_level_6": 0.2,
        "operator_detune_2": -4.0, "operator_detune_3": 5.0, "operator_pan_5": -0.74, "operator_pan_6": 0.74,
        "filter_cutoff": 7800.0, "filter_resonance": 0.05, "amp_attack_ms": 2400.0, "amp_release_ms": 9200.0,
    },
    "Lush Drone - Subharmonic Bloom": {
        "tx802_patch_name": "SUB BLOOM",
        "master_gain": 0.26,
        "carrier_freq": 41.2034,
        "algorithm": 24,
        "voice_count": 8,
        "poly_spread": 0.18,
        "chord_depth": 0.26,
        "timbre_spread": 0.72,
        "visual_drive": 0.68,
        "visual_influence": 0.72,
        "visual_evolution": 0.48,
        "visual_pitch_motion": 0.03,
        "visual_ir_motion": 0.22,
        "feedback": 0.24,
        "feedback_color": 0.24,
        "motion_gating": 0.04,
        "motion_response": 0.44,
        "brightness_response": 0.54,
        "color_stereo": 0.58,
        "visual_morph_depth": 0.28,
        "lfo_rate": 0.03,
        "lfo_depth": 0.006,
        "operator_ratio_1": 0.5, "operator_ratio_2": 1.0, "operator_ratio_3": 1.25, "operator_ratio_4": 2.5, "operator_ratio_5": 5.0, "operator_ratio_6": 9.0,
        "operator_output_level_1": 0.86, "operator_output_level_2": 0.72, "operator_output_level_3": 0.5, "operator_output_level_4": 0.4, "operator_output_level_5": 0.3, "operator_output_level_6": 0.22,
        "operator_detune_4": -8.0, "operator_detune_6": 13.0,
        "filter_cutoff": 2400.0, "filter_resonance": 0.16, "amp_attack_ms": 5200.0, "amp_release_ms": 12000.0,
    },
    "Lush Pad - Tape Aurora": {
        "tx802_patch_name": "TAPE AURORA",
        "master_gain": 0.22,
        "carrier_freq": 123.471,
        "algorithm": 16,
        "voice_count": 11,
        "poly_spread": 0.5,
        "chord_depth": 0.54,
        "timbre_spread": 0.62,
        "visual_drive": 0.95,
        "visual_influence": 1.18,
        "visual_evolution": 0.9,
        "visual_pitch_motion": 0.14,
        "visual_ir_motion": 0.5,
        "feedback": 0.13,
        "feedback_color": 0.22,
        "motion_gating": 0.05,
        "motion_response": 0.5,
        "brightness_response": 0.72,
        "color_stereo": 0.92,
        "visual_morph_depth": 0.42,
        "lfo_rate": 0.11,
        "lfo_depth": 0.02,
        "lfo_wave": "triangle",
        "operator_ratio_1": 1.0, "operator_ratio_2": 1.5, "operator_ratio_3": 2.0, "operator_ratio_4": 3.0, "operator_ratio_5": 5.0, "operator_ratio_6": 7.0,
        "operator_output_level_1": 0.66, "operator_output_level_2": 0.54, "operator_output_level_3": 0.48, "operator_output_level_4": 0.36, "operator_output_level_5": 0.28, "operator_output_level_6": 0.22,
        "operator_detune_2": -7.0, "operator_detune_5": 9.0, "operator_pan_2": -0.36, "operator_pan_5": 0.48,
        "filter_cutoff": 4300.0, "filter_resonance": 0.1, "amp_attack_ms": 4200.0, "amp_release_ms": 10000.0,
    },
    "Lush Drone - Fixed Starfield": {
        "tx802_patch_name": "FIXED STARS",
        "master_gain": 0.19,
        "carrier_freq": 73.4162,
        "algorithm": 30,
        "voice_count": 9,
        "poly_spread": 0.32,
        "chord_depth": 0.38,
        "timbre_spread": 0.58,
        "visual_drive": 0.8,
        "visual_influence": 0.95,
        "visual_evolution": 0.86,
        "visual_pitch_motion": 0.06,
        "visual_ir_motion": 0.62,
        "feedback": 0.08,
        "feedback_color": 0.16,
        "motion_gating": 0.03,
        "motion_response": 0.32,
        "brightness_response": 0.76,
        "color_stereo": 1.0,
        "visual_morph_depth": 0.34,
        "lfo_rate": 0.07,
        "lfo_depth": 0.01,
        "operator_ratio_1": 1.0, "operator_ratio_2": 2.0, "operator_ratio_3": 3.0, "operator_ratio_4": 4.0, "operator_ratio_5": 5.0, "operator_ratio_6": 6.0,
        "operator_output_level_1": 0.68, "operator_output_level_2": 0.46, "operator_output_level_3": 0.4, "operator_output_level_4": 0.32, "operator_output_level_5": 0.24, "operator_output_level_6": 0.2,
        "operator_fixed_5": True, "operator_fixed_freq_5": 587.3295, "operator_fixed_6": True, "operator_fixed_freq_6": 1760.0,
        "operator_pan_5": -0.88, "operator_pan_6": 0.88,
        "filter_cutoff": 6800.0, "filter_resonance": 0.06, "amp_attack_ms": 3000.0, "amp_release_ms": 11000.0,
    },
    "Lush Pad - Warm Digital Reeds": {
        "tx802_patch_name": "WARM REEDS",
        "master_gain": 0.25,
        "carrier_freq": 82.4069,
        "algorithm": 1,
        "voice_count": 10,
        "poly_spread": 0.22,
        "chord_depth": 0.42,
        "timbre_spread": 0.28,
        "visual_drive": 0.62,
        "visual_influence": 0.75,
        "visual_evolution": 0.5,
        "visual_pitch_motion": 0.07,
        "visual_ir_motion": 0.28,
        "feedback": 0.09,
        "feedback_color": 0.12,
        "motion_gating": 0.04,
        "motion_response": 0.36,
        "brightness_response": 0.7,
        "color_stereo": 0.7,
        "visual_morph_depth": 0.22,
        "lfo_rate": 0.16,
        "lfo_depth": 0.014,
        "lfo_delay": 1.0,
        "operator_ratio_1": 1.0, "operator_ratio_2": 1.0, "operator_ratio_3": 2.0, "operator_ratio_4": 2.0, "operator_ratio_5": 3.0, "operator_ratio_6": 4.0,
        "operator_output_level_1": 0.8, "operator_output_level_2": 0.58, "operator_output_level_3": 0.48, "operator_output_level_4": 0.34, "operator_output_level_5": 0.24, "operator_output_level_6": 0.16,
        "filter_cutoff": 3600.0, "filter_resonance": 0.12, "amp_attack_ms": 2600.0, "amp_release_ms": 7600.0,
    },
    "Lush Drone - Blue Feedback Sea": {
        "tx802_patch_name": "BLUE SEA",
        "master_gain": 0.18,
        "carrier_freq": 55.0,
        "algorithm": 27,
        "voice_count": 8,
        "poly_spread": 0.4,
        "chord_depth": 0.2,
        "timbre_spread": 0.88,
        "visual_drive": 1.0,
        "visual_influence": 1.12,
        "visual_evolution": 1.0,
        "visual_pitch_motion": 0.09,
        "visual_ir_motion": 0.7,
        "feedback": 0.48,
        "feedback_color": 0.52,
        "motion_gating": 0.06,
        "motion_response": 0.6,
        "brightness_response": 0.52,
        "color_stereo": 0.95,
        "visual_morph_depth": 0.52,
        "lfo_rate": 0.025,
        "lfo_depth": 0.008,
        "operator_ratio_1": 1.0, "operator_ratio_2": 1.414, "operator_ratio_3": 2.718, "operator_ratio_4": 4.0, "operator_ratio_5": 5.657, "operator_ratio_6": 8.0,
        "operator_output_level_1": 0.48, "operator_output_level_2": 0.76, "operator_output_level_3": 0.66, "operator_output_level_4": 0.52, "operator_output_level_5": 0.42, "operator_output_level_6": 0.34,
        "filter_cutoff": 3100.0, "filter_resonance": 0.2, "amp_attack_ms": 6800.0, "amp_release_ms": 14000.0,
    },
    "Lush Pad - Celestial Tines": {
        "tx802_patch_name": "CELEST TINE",
        "master_gain": 0.21,
        "carrier_freq": 146.8324,
        "algorithm": 8,
        "voice_count": 9,
        "poly_spread": 0.3,
        "chord_depth": 0.46,
        "timbre_spread": 0.52,
        "visual_drive": 0.84,
        "visual_influence": 0.98,
        "visual_evolution": 0.74,
        "visual_pitch_motion": 0.08,
        "visual_ir_motion": 0.48,
        "feedback": 0.07,
        "feedback_color": 0.14,
        "motion_gating": 0.02,
        "motion_response": 0.3,
        "brightness_response": 0.8,
        "color_stereo": 0.9,
        "visual_morph_depth": 0.3,
        "lfo_rate": 0.09,
        "lfo_depth": 0.012,
        "operator_ratio_1": 1.0, "operator_ratio_2": 7.0, "operator_ratio_3": 1.0, "operator_ratio_4": 2.0, "operator_ratio_5": 3.0, "operator_ratio_6": 6.0,
        "operator_output_level_1": 0.62, "operator_output_level_2": 0.42, "operator_output_level_3": 0.46, "operator_output_level_4": 0.34, "operator_output_level_5": 0.24, "operator_output_level_6": 0.18,
        "operator_detune_2": -3.0, "operator_detune_4": 6.0,
        "filter_cutoff": 6200.0, "filter_resonance": 0.07, "amp_attack_ms": 2200.0, "amp_release_ms": 8400.0,
    },
}

for _preset in {**TX802_ACCURATE_MEDIA_AUDIO_PRESETS, **IDM_MEDIA_AUDIO_PRESETS, **LUSH_PAD_DRONE_PRESETS}.values():
    _preset.setdefault("idm_mode", False)
    _preset.setdefault("idm_intensity", 0.0)
    _preset.setdefault("idm_bpm", 172.0)
    _preset.setdefault("idm_grid_rate", 16.0)
    _preset.setdefault("idm_probability", 0.75)
    _preset.setdefault("idm_microtiming", 0.0)
    _preset.setdefault("idm_chaos", 0.35 if _preset.get("idm_mode") else 0.0)
    _preset.setdefault("idm_phrase_depth", 0.55 if _preset.get("idm_mode") else 0.0)
    _preset.setdefault("idm_polyrhythm_depth", 0.62 if _preset.get("idm_mode") else 0.0)
    _preset.setdefault("idm_scene_reactivity", 0.7 if _preset.get("idm_mode") else 0.0)
    _preset.setdefault("idm_density_drift", 0.5 if _preset.get("idm_mode") else 0.0)
    _preset.setdefault("idm_retrigger_depth", 0.0)
    _preset.setdefault("idm_stutter_depth", 0.0)
    _preset.setdefault("idm_env_slice_depth", 0.0)
    _preset.setdefault("idm_carrier_chop_depth", 0.0)
    _preset.setdefault("idm_modulator_chop_depth", 0.0)
    _preset.setdefault("idm_ratio_snap_depth", 0.0)
    _preset.setdefault("idm_algorithm_morph_depth", 0.0)
    _preset.setdefault("idm_bitcrush_depth", 0.0)
    _preset.setdefault("idm_sample_rate_fold", 0.0)
    _preset.setdefault("idm_wavefold_depth", 0.0)
    _preset.setdefault("idm_seed", 19.0)
    for _op in range(1, 7):
        _preset.setdefault(f"operator_level_{_op}", _preset.get(f"operator_output_level_{_op}", 0.0))
        _preset.setdefault(f"operator_velocity_{_op}", 0.35 if _op > 1 else 0.1)
        _preset.setdefault(f"operator_key_level_{_op}", 0.15)
        _preset.setdefault(f"operator_key_rate_{_op}", 0.0)
        _preset.setdefault(f"operator_pan_{_op}", 0.0)
        _preset.setdefault(f"operator_detune_{_op}", 0.0)
        _preset.setdefault(f"operator_fixed_{_op}", False)
        _preset.setdefault(f"operator_fixed_freq_{_op}", 440.0)
        _preset.setdefault(f"operator_enabled_{_op}", True)
        _preset.setdefault(f"operator_glitch_depth_{_op}", 0.72 if _op > 1 and _preset.get("idm_mode") else 0.0)
        _preset.setdefault(f"operator_burst_probability_{_op}", 0.35 if _op > 1 and _preset.get("idm_mode") else 0.0)
        _preset.setdefault(f"operator_level_jitter_{_op}", 0.42 if _op > 1 and _preset.get("idm_mode") else 0.0)
        _preset.setdefault(f"operator_ratio_snap_{_op}", 0.68 if _op > 1 and _preset.get("idm_mode") else 0.0)
        is_carrier = TX802_ALGORITHM_TARGETS[int(_preset["algorithm"]) - 1][_op - 1] < 0
        rates = (82.0, 58.0, 40.0, 42.0) if is_carrier else (94.0, 72.0, 28.0, 70.0)
        levels = (1.0, 0.72, 0.58, 0.0) if is_carrier else (1.0, 0.36, 0.0, 0.0)
        if "Perc" in _preset["tx802_patch_name"] or "METAL" in _preset["tx802_patch_name"]:
            rates = (99.0, 86.0, 52.0, 82.0)
            levels = (1.0, 0.18, 0.0, 0.0)
        if (
            ("CHOIR" in _preset["tx802_patch_name"] and not _preset.get("idm_mode", False))
            or "EVO" in _preset["tx802_patch_name"]
            or any(_token in _preset["tx802_patch_name"] for _token in ("LUSH", "CLOUD", "HALO", "BLOOM", "AURORA", "STARS", "REEDS", "SEA", "CELEST"))
        ):
            rates = (52.0, 34.0, 20.0, 38.0)
            levels = (1.0, 0.86, 0.74, 0.0)
        for _stage, _value in enumerate(rates, 1):
            _preset.setdefault(f"operator_eg_rate_{_op}_{_stage}", _value)
        for _stage, _value in enumerate(levels, 1):
            _preset.setdefault(f"operator_eg_level_{_op}_{_stage}", _value)
    for _stage, _value in enumerate((50.0, 50.0, 50.0, 50.0), 1):
        _preset.setdefault(f"pitch_eg_rate_{_stage}", _value)
    for _stage, _value in enumerate((0.5, 0.5, 0.5, 0.5), 1):
        _preset.setdefault(f"pitch_eg_level_{_stage}", _value)

MEDIA_AUDIO_PRESETS = {
    **TX802_ACCURATE_MEDIA_AUDIO_PRESETS,
    **IDM_MEDIA_AUDIO_PRESETS,
    **LUSH_PAD_DRONE_PRESETS,
}


@dataclass(frozen=True)
class MediaAnalysis:
    timestamps: np.ndarray
    brightness: np.ndarray
    colors: np.ndarray
    motion: np.ndarray
    duration: float
    source_path: str


class PartitionedStereoConvolver:
    """Uniform partitioned overlap-add convolution for fixed audio blocks."""

    def __init__(self, impulse_response, block_size):
        impulse = np.asarray(impulse_response, dtype=np.float32)
        if impulse.ndim == 1:
            impulse = np.column_stack((impulse, impulse))
        elif impulse.ndim == 2 and impulse.shape[1] == 1:
            impulse = np.repeat(impulse, 2, axis=1)
        elif impulse.ndim != 2:
            raise ValueError("Impulse response must be mono or multichannel audio.")
        impulse = impulse[:, :2]
        if impulse.shape[0] < 1:
            raise ValueError("Impulse response is empty.")

        self.block_size = max(1, int(block_size))
        self.fft_size = self.block_size * 2
        self.partition_count = int(
            math.ceil(impulse.shape[0] / self.block_size)
        )
        frequency_bins = self.fft_size // 2 + 1
        self.ir_fft = np.zeros(
            (2, self.partition_count, frequency_bins), dtype=np.complex64
        )
        for partition_index in range(self.partition_count):
            start = partition_index * self.block_size
            end = min(start + self.block_size, impulse.shape[0])
            padded = np.zeros((self.fft_size, 2), dtype=np.float32)
            padded[: end - start] = impulse[start:end]
            transformed = np.fft.rfft(padded, axis=0)
            self.ir_fft[:, partition_index] = transformed.T.astype(
                np.complex64
            )

        self.input_history = np.zeros(
            (2, self.partition_count, frequency_bins), dtype=np.complex64
        )
        self.overlap = np.zeros((self.block_size, 2), dtype=np.float32)
        self.cursor = 0

    def process(self, stereo_block):
        block = np.asarray(stereo_block, dtype=np.float32)
        if block.shape != (self.block_size, 2):
            raise ValueError(
                f"IR block must be ({self.block_size}, 2), got {block.shape}."
            )
        padded = np.zeros((self.fft_size, 2), dtype=np.float32)
        padded[: self.block_size] = block
        transformed = np.fft.rfft(padded, axis=0).T.astype(np.complex64)
        self.input_history[:, self.cursor] = transformed

        first_count = self.cursor + 1
        accumulated = np.einsum(
            "ckf,ckf->cf",
            self.ir_fft[:, :first_count],
            self.input_history[:, :first_count][:, ::-1],
            optimize=False,
        ).astype(np.complex64, copy=False)
        remaining = self.partition_count - first_count
        if remaining:
            accumulated += np.einsum(
                "ckf,ckf->cf",
                self.ir_fft[:, first_count:],
                self.input_history[:, first_count:][:, ::-1],
                optimize=False,
            ).astype(np.complex64, copy=False)

        convolved = np.fft.irfft(
            accumulated, n=self.fft_size, axis=1
        ).T.astype(np.float32)
        output = convolved[: self.block_size] + self.overlap
        self.overlap[:] = convolved[self.block_size :]
        self.cursor = (self.cursor + 1) % self.partition_count
        return output


def load_impulse_response(
    path,
    target_sample_rate,
    max_seconds=4.0,
    normalize=True,
    normalize_mode="room",
):
    """Load PCM/float WAV, resample, trim, and return stereo float32 audio."""
    if not path or not os.path.isfile(path):
        raise ValueError(f"IR file does not exist: {path}")
    try:
        sample_rate, impulse = wavfile.read(path)
    except Exception as exc:
        raise ValueError(f"Could not read IR WAV: {exc}") from exc

    if impulse.size == 0:
        raise ValueError("IR WAV is empty.")
    if impulse.ndim == 1:
        impulse = impulse[:, None]
    elif impulse.ndim != 2:
        raise ValueError("IR WAV has an unsupported channel layout.")

    if np.issubdtype(impulse.dtype, np.integer):
        info = np.iinfo(impulse.dtype)
        divisor = float(max(abs(info.min), info.max))
        impulse = impulse.astype(np.float32) / divisor
    else:
        impulse = impulse.astype(np.float32)
    if impulse.shape[1] > 2:
        odd_channels = impulse[:, 1::2]
        impulse = np.column_stack(
            (
                impulse[:, ::2].mean(axis=1),
                (
                    odd_channels.mean(axis=1)
                    if odd_channels.shape[1]
                    else impulse[:, ::2].mean(axis=1)
                ),
            )
        )
    impulse = np.nan_to_num(impulse, nan=0.0, posinf=0.0, neginf=0.0)

    target_sample_rate = int(target_sample_rate)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("IR WAV has an invalid sample rate.")
    if sample_rate != target_sample_rate:
        common = math.gcd(sample_rate, target_sample_rate)
        impulse = resample_poly(
            impulse,
            target_sample_rate // common,
            sample_rate // common,
            axis=0,
        ).astype(np.float32)

    max_samples = max(1, int(float(max_seconds) * target_sample_rate))
    impulse = impulse[:max_samples]
    impulse -= impulse.mean(axis=0, keepdims=True)
    if impulse.shape[0] > 32:
        fade_samples = min(256, impulse.shape[0])
        impulse[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, None]
        impulse[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
    peak = float(np.max(np.abs(impulse)))
    if peak <= 1e-9:
        raise ValueError("IR WAV contains silence.")
    if normalize:
        mode = str(normalize_mode or "room").lower()
        if mode == "peak":
            impulse = impulse / peak
        else:
            # Reverb IRs with one loud direct spike should not be peak-maxed;
            # use energy normalization so the tail remains spacious and sane.
            energy = float(np.sqrt(np.mean(np.sum(impulse * impulse, axis=1))))
            target = 0.16
            if energy > 1e-9:
                impulse = impulse * min(8.0, target / energy)
            post_peak = float(np.max(np.abs(impulse)))
            if post_peak > 0.98:
                impulse = impulse * (0.98 / post_peak)
    return np.ascontiguousarray(impulse, dtype=np.float32)


def _small_rgb(frame_rgb, max_dimension=320):
    height, width = frame_rgb.shape[:2]
    scale = min(1.0, float(max_dimension) / max(width, height, 1))
    if scale < 1.0:
        frame_rgb = cv2.resize(
            frame_rgb,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return frame_rgb


def _decode_analysis_frames(path, max_analysis_fps=12.0, max_dimension=320):
    extension = os.path.splitext(path)[1].lower()
    frames = []
    timestamps = []

    if extension == ".gif":
        elapsed = 0.0
        with Image.open(path) as gif:
            for frame in ImageSequence.Iterator(gif):
                frames.append(
                    _small_rgb(np.asarray(frame.convert("RGB")), max_dimension)
                )
                timestamps.append(elapsed)
                elapsed += max(
                    float(frame.info.get("duration", 100) or 100) / 1000.0,
                    0.001,
                )
        if not frames:
            raise ValueError("The GIF contains no decodable frames.")
        return frames, np.asarray(timestamps, dtype=np.float64), max(elapsed, 0.001)

    if extension not in VIDEO_EXTENSIONS:
        raise ValueError(
            "Unsupported media type. Select GIF, MP4, MOV, AVI, MKV, M4V, or WEBM."
        )

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open the video: {path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / source_fps if frame_count > 0 else 0.0
    sample_step = max(
        1, int(round(source_fps / max(float(max_analysis_fps), 0.1)))
    )

    try:
        frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            frames.append(
                _small_rgb(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    max_dimension,
                )
            )
            timestamps.append(frame_index / source_fps)
            skipped = 0
            for _ in range(sample_step - 1):
                if not capture.grab():
                    break
                skipped += 1
            frame_index += 1 + skipped
            if skipped < sample_step - 1:
                break
    finally:
        capture.release()

    if not frames:
        raise ValueError(
            "The video opened, but its codec produced no decodable frames."
        )
    if duration <= 0.0:
        duration = max(timestamps[-1] + sample_step / source_fps, 0.001)
    return frames, np.asarray(timestamps, dtype=np.float64), duration


def analyze_media(path, max_analysis_fps=12.0, max_dimension=320):
    """Decode a compact brightness/color/motion timeline for callback playback."""
    if not path or not os.path.isfile(path):
        raise ValueError(f"Media file does not exist: {path}")

    frames, timestamps, duration = _decode_analysis_frames(
        path,
        max_analysis_fps=max_analysis_fps,
        max_dimension=max_dimension,
    )
    brightness = np.empty(len(frames), dtype=np.float32)
    colors = np.empty((len(frames), 3), dtype=np.float32)
    gray_frames = []

    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)
        brightness[index] = float(gray.mean()) / 255.0
        colors[index] = frame.reshape(-1, 3).mean(axis=0) / 255.0

    motion = np.zeros(len(frames), dtype=np.float32)
    for index in range(1, len(gray_frames)):
        previous = gray_frames[index - 1]
        current = gray_frames[index]
        if current.shape != previous.shape:
            current = cv2.resize(
                current, (previous.shape[1], previous.shape[0])
            )
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        motion[index] = float(np.percentile(magnitude, 75))

    scale = float(np.percentile(motion, 95)) if motion.size else 0.0
    if scale > 1e-6:
        motion = np.clip(motion / scale, 0.0, 1.0)

    return MediaAnalysis(
        timestamps=np.ascontiguousarray(timestamps, dtype=np.float64),
        brightness=np.ascontiguousarray(np.clip(brightness, 0.0, 1.0)),
        colors=np.ascontiguousarray(np.clip(colors, 0.0, 1.0)),
        motion=np.ascontiguousarray(motion),
        duration=float(max(duration, 0.001)),
        source_path=os.path.abspath(path),
    )


class MediaAudioEngine:
    """Callback-safe polyphonic, multitimbral FM engine driven by visuals."""

    PARAM_LIMITS = {
        "master_gain": (0.0, 1.0),
        "carrier_freq": (20.0, 4000.0),
        "modulator_ratio": (0.125, 8.0),
        "modulation_index": (0.0, 24.0),
        "feedback": (0.0, 0.95),
        "algorithm": (1, 32),
        "voice_count": (1, 16),
        "poly_spread": (0.0, 1.0),
        "chord_depth": (0.0, 1.0),
        "timbre_spread": (0.0, 1.0),
        "visual_drive": (0.0, 2.0),
        "visual_influence": (0.0, 2.0),
        "visual_evolution": (0.0, 1.0),
        "visual_pitch_motion": (0.0, 1.0),
        "visual_ir_motion": (0.0, 1.0),
        "feedback_color": (0.0, 1.0),
        "motion_gating": (0.0, 1.0),
        "motion_response": (0.0, 4.0),
        "brightness_response": (0.0, 1.0),
        "color_stereo": (0.0, 1.0),
        "playback_rate": (0.05, 4.0),
        "hybrid_mix": (0.0, 1.0),
        "ir_wet": (0.0, 1.0),
        "ir_gain": (0.0, 1.25),
        "amp_attack_ms": (0.0, 10000.0),
        "amp_decay_ms": (0.0, 10000.0),
        "amp_sustain": (0.0, 1.0),
        "amp_release_ms": (0.0, 15000.0),
        "filter_cutoff": (40.0, 18000.0),
        "filter_resonance": (0.0, 0.95),
        "filter_env_amount": (-1.0, 1.0),
        "filter_visual_amount": (0.0, 1.0),
        "filter_drive": (0.0, 2.0),
        "midi_gate": (0.0, 1.0),
        "midi_velocity": (0.0, 1.0),
        "midi_transpose": (-48.0, 48.0),
        "midi_velocity_depth": (0.0, 1.0),
        "midi_timbre_depth": (0.0, 2.0),
        "midi_note_punch": (0.0, 1.0),
        "midi_pitch_slew_ms": (0.0, 500.0),
        "midi_note_focus": (0.0, 1.0),
        "lfo_rate": (0.0, 30.0),
        "lfo_depth": (0.0, 1.0),
        "lfo_delay": (0.0, 10.0),
        "visual_morph_depth": (0.0, 1.0),
        "idm_intensity": (0.0, 1.0),
        "idm_bpm": (40.0, 260.0),
        "idm_grid_rate": (4.0, 64.0),
        "idm_probability": (0.0, 1.0),
        "idm_microtiming": (0.0, 1.0),
        "idm_chaos": (0.0, 1.0),
        "idm_phrase_depth": (0.0, 1.0),
        "idm_polyrhythm_depth": (0.0, 1.0),
        "idm_scene_reactivity": (0.0, 1.0),
        "idm_density_drift": (0.0, 1.0),
        "idm_retrigger_depth": (0.0, 1.0),
        "idm_stutter_depth": (0.0, 1.0),
        "idm_env_slice_depth": (0.0, 1.0),
        "idm_carrier_chop_depth": (0.0, 1.0),
        "idm_modulator_chop_depth": (0.0, 1.0),
        "idm_ratio_snap_depth": (0.0, 1.0),
        "idm_algorithm_morph_depth": (0.0, 1.0),
        "idm_bitcrush_depth": (0.0, 1.0),
        "idm_sample_rate_fold": (0.0, 1.0),
        "idm_wavefold_depth": (0.0, 1.0),
        "idm_seed": (0.0, 999.0),
    }

    def __init__(self, sample_rate=48000, block_size=256):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.params = {
            "enabled": True,
            "loop": True,
            "master_gain": 0.25,
            "carrier_freq": 432.0,
            "modulator_ratio": 1.61803398875,
            "modulation_index": 2.2,
            "feedback": 0.05,
            "algorithm": 1,
            "voice_count": 6,
            "poly_spread": 0.56,
            "chord_depth": 0.68,
            "timbre_spread": 0.52,
            "visual_drive": 1.0,
            "visual_influence": 1.25,
            "visual_evolution": 0.65,
            "visual_pitch_motion": 0.55,
            "visual_ir_motion": 0.35,
            "feedback_color": 0.32,
            "motion_gating": 0.24,
            "motion_response": 0.9,
            "brightness_response": 0.76,
            "color_stereo": 0.5,
            "playback_rate": 1.0,
            "hybrid_mix": 0.5,
            "ir_wet": 0.22,
            "ir_gain": 0.45,
            "amp_env_enabled": True,
            "amp_attack_ms": 15.0,
            "amp_decay_ms": 450.0,
            "amp_sustain": 1.0,
            "amp_release_ms": 900.0,
            "filter_enabled": True,
            "filter_cutoff": 3600.0,
            "filter_resonance": 0.12,
            "filter_env_amount": 0.0,
            "filter_visual_amount": 0.22,
            "filter_drive": 0.05,
            "midi_input_enabled": True,
            "midi_note_follow": True,
            "midi_gate_enabled": False,
            "midi_gate": 1.0,
            "midi_velocity": 1.0,
            "midi_transpose": 0.0,
            "midi_velocity_depth": 0.45,
            "midi_timbre_depth": 0.18,
            "midi_note_punch": 0.16,
            "midi_pitch_slew_ms": 0.0,
            "midi_note_focus": 0.55,
            "midi_retrigger_phase": True,
            "midi_style": "Musical",
            "tx802_patch_name": "INIT",
            "lfo_rate": 0.0,
            "lfo_depth": 0.0,
            "lfo_delay": 0.0,
            "lfo_wave": "sine",
            "visual_morph_depth": 0.5,
            "idm_mode": False,
            "idm_intensity": 0.0,
            "idm_bpm": 172.0,
            "idm_grid_rate": 16.0,
            "idm_probability": 0.75,
            "idm_microtiming": 0.0,
            "idm_chaos": 0.0,
            "idm_phrase_depth": 0.0,
            "idm_polyrhythm_depth": 0.0,
            "idm_scene_reactivity": 0.0,
            "idm_density_drift": 0.0,
            "idm_retrigger_depth": 0.0,
            "idm_stutter_depth": 0.0,
            "idm_env_slice_depth": 0.0,
            "idm_carrier_chop_depth": 0.0,
            "idm_modulator_chop_depth": 0.0,
            "idm_ratio_snap_depth": 0.0,
            "idm_algorithm_morph_depth": 0.0,
            "idm_bitcrush_depth": 0.0,
            "idm_sample_rate_fold": 0.0,
            "idm_wavefold_depth": 0.0,
            "idm_seed": 19.0,
        }
        ratio_defaults = (1.0, 1.61803398875, 2.0, 3.0, 0.5, 4.0)
        level_defaults = (1.0, 0.45, 0.32, 0.22, 0.24, 0.2)
        pan_defaults = (-0.18, 0.18, -0.45, 0.45, -0.65, 0.65)
        for index in range(1, 7):
            self.PARAM_LIMITS.setdefault(f"operator_ratio_{index}", (0.125, 32.0))
            self.PARAM_LIMITS.setdefault(f"operator_output_level_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_level_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_detune_{index}", (-100.0, 100.0))
            self.PARAM_LIMITS.setdefault(f"operator_pan_{index}", (-1.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_fixed_freq_{index}", (0.1, 12000.0))
            self.PARAM_LIMITS.setdefault(f"operator_velocity_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_key_level_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_key_rate_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_glitch_depth_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_burst_probability_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_level_jitter_{index}", (0.0, 1.0))
            self.PARAM_LIMITS.setdefault(f"operator_ratio_snap_{index}", (0.0, 1.0))
            self.params[f"operator_ratio_{index}"] = ratio_defaults[index - 1]
            self.params[f"operator_output_level_{index}"] = level_defaults[index - 1]
            self.params[f"operator_level_{index}"] = level_defaults[index - 1]
            self.params[f"operator_detune_{index}"] = 0.0
            self.params[f"operator_pan_{index}"] = pan_defaults[index - 1]
            self.params[f"operator_fixed_{index}"] = False
            self.params[f"operator_fixed_freq_{index}"] = 440.0
            self.params[f"operator_velocity_{index}"] = 0.35 if index > 1 else 0.1
            self.params[f"operator_key_level_{index}"] = 0.15
            self.params[f"operator_key_rate_{index}"] = 0.0
            self.params[f"operator_glitch_depth_{index}"] = 0.0
            self.params[f"operator_burst_probability_{index}"] = 0.0
            self.params[f"operator_level_jitter_{index}"] = 0.0
            self.params[f"operator_ratio_snap_{index}"] = 0.0
            self.params[f"operator_enabled_{index}"] = True
            for stage, default in enumerate((86.0, 62.0, 36.0, 50.0), 1):
                self.PARAM_LIMITS.setdefault(f"operator_eg_rate_{index}_{stage}", (0.0, 99.0))
                self.params[f"operator_eg_rate_{index}_{stage}"] = default
            for stage, default in enumerate((1.0, 0.65, 0.5, 0.0), 1):
                self.PARAM_LIMITS.setdefault(f"operator_eg_level_{index}_{stage}", (0.0, 1.0))
                self.params[f"operator_eg_level_{index}_{stage}"] = default
        for stage, default in enumerate((50.0, 50.0, 50.0, 50.0), 1):
            self.PARAM_LIMITS.setdefault(f"pitch_eg_rate_{stage}", (0.0, 99.0))
            self.params[f"pitch_eg_rate_{stage}"] = default
        for stage, default in enumerate((0.5, 0.5, 0.5, 0.5), 1):
            self.PARAM_LIMITS.setdefault(f"pitch_eg_level_{stage}", (0.0, 1.0))
            self.params[f"pitch_eg_level_{stage}"] = default
        self.source_mode = "Live Display"
        self._analysis = None
        self._live_metrics = self._default_metrics()
        self._param_mod_sources = {
            name: "None"
            for name in self.PARAM_LIMITS
            if name not in {"playback_rate", "hybrid_mix", "ir_wet", "ir_gain"}
        }
        self._effective_params = dict(self.params)
        self._live_metrics_received = False
        self._convolver = None
        self.cdp_graph = CDPGraphProcessor()
        self._ir_path = ""
        self._position = 0.0
        self._render_time = 0.0
        self._max_voices = 16
        self._operator_count = 6
        self._voice_phase = np.zeros(
            (self._max_voices, self._operator_count), dtype=np.float64
        )
        self._voice_feedback = np.zeros(
            (self._max_voices, self._operator_count), dtype=np.float64
        )
        self._voice_lfo_phase = np.linspace(
            0.0, math.pi, self._max_voices, dtype=np.float64
        )
        self._smoothed_metrics = self._default_metrics()
        self._playing = False
        self._midi_active_notes: list[tuple[int, float]] = []
        self._midi_target_frequency = float(self.params["carrier_freq"])
        self._midi_smoothed_frequency = float(self.params["carrier_freq"])
        self._midi_note_pulse = 0.0
        self._amp_env_level = 1.0
        self._amp_env_stage = "sustain"
        self._filter_low = np.zeros(2, dtype=np.float64)
        self._filter_band = np.zeros(2, dtype=np.float64)
        self._operator_env_level = np.zeros(
            (self._max_voices, self._operator_count), dtype=np.float64
        )
        self._operator_env_stage = np.full(
            (self._max_voices, self._operator_count), 5, dtype=np.int32
        )
        self._lock = threading.RLock()

    @staticmethod
    def _default_metrics():
        return {
            "brightness": 0.5,
            "contrast": 0.0,
            "motion": 0.0,
            "entropy": 0.0,
            "edge_density": 0.0,
            "red_energy": 0.5,
            "green_energy": 0.5,
            "blue_energy": 0.5,
            "glitch_spike": 0.0,
            "center_mass_x": 0.5,
            "center_mass_y": 0.5,
        }

    def set_analysis(self, analysis):
        with self._lock:
            self._analysis = analysis
            self._position = 0.0

    def set_source_mode(self, mode):
        mode = str(mode)
        if mode not in {"Live Display", "Loaded Media", "Hybrid"}:
            raise ValueError(f"Unknown Media Audio source mode: {mode}")
        with self._lock:
            self.source_mode = mode

    def set_live_metrics(self, metrics):
        if hasattr(metrics, "as_dict"):
            metrics = metrics.as_dict()
        if not isinstance(metrics, dict):
            return
        with self._lock:
            for name in self._live_metrics:
                try:
                    self._live_metrics[name] = min(
                        1.0,
                        max(0.0, float(metrics.get(name, self._live_metrics[name]))),
                    )
                except (TypeError, ValueError):
                    continue
            self._effective_params = self._apply_param_modulation_from_metrics(
                self.params,
                self._live_metrics,
            )
            self._live_metrics_received = True

    def set_convolver(self, convolver, path=""):
        with self._lock:
            self._convolver = convolver
            self._ir_path = str(path or "")

    def clear_convolver(self):
        self.set_convolver(None, "")

    def set_param(self, name, value):
        name = str(name)
        alias_target = None
        if name.startswith("operator_level_"):
            alias_target = name.replace("operator_level_", "operator_output_level_", 1)
        if name not in self.params:
            raise KeyError(f"Unknown media-audio parameter: {name}")
        if name in self.PARAM_LIMITS:
            low, high = self.PARAM_LIMITS[name]
            if name in {"voice_count", "algorithm"}:
                value = min(int(high), max(int(low), int(round(float(value)))))
            else:
                value = min(high, max(low, float(value)))
        elif name in {"enabled", "loop", "amp_env_enabled", "filter_enabled", "midi_input_enabled", "midi_note_follow", "midi_gate_enabled", "midi_retrigger_phase", "idm_mode"} or name.startswith(("operator_fixed_", "operator_enabled_")):
            value = bool(value)
        elif name == "midi_style":
            value = str(value or "Musical")
            if value not in {"Musical", "Drone"}:
                value = "Musical"
        elif name == "lfo_wave":
            value = str(value or "sine")
            if value not in {"sine", "triangle", "saw_up", "saw_down", "square", "sample_hold"}:
                value = "sine"
        elif name == "tx802_patch_name":
            value = str(value or "INIT")[:32]
        with self._lock:
            self.params[name] = value
            self._effective_params[name] = value
            if alias_target and alias_target in self.params:
                self.params[alias_target] = value
                self._effective_params[alias_target] = value
            elif name.startswith("operator_output_level_"):
                alias = name.replace("operator_output_level_", "operator_level_", 1)
                if alias in self.params:
                    self.params[alias] = value
                    self._effective_params[alias] = value

    def handle_midi_performance(self, msg) -> bool:
        if not bool(self.params.get("midi_input_enabled", True)):
            return False
        msg_type = getattr(msg, "type", "")
        if msg_type not in {"note_on", "note_off"}:
            return False
        note = int(getattr(msg, "note", 0))
        velocity = int(getattr(msg, "velocity", 0))
        if msg_type == "note_on" and velocity > 0:
            normalized_velocity = max(0.0, min(1.0, velocity / 127.0))
            transpose = int(round(float(self.params.get("midi_transpose", 0.0))))
            frequency = midi_to_frequency(max(0, min(127, note + transpose)))
            with self._lock:
                self._midi_active_notes = [
                    (active_note, active_velocity)
                    for active_note, active_velocity in self._midi_active_notes
                    if active_note != note
                ]
                self._midi_active_notes.append((note, normalized_velocity))
                self.params["midi_velocity"] = normalized_velocity
                self.params["midi_gate"] = normalized_velocity
                voice_index = min(self._max_voices - 1, max(0, len(self._midi_active_notes) - 1))
                if self.params.get("midi_note_follow", True):
                    self._midi_target_frequency = frequency
                    if not self._midi_active_notes[:-1]:
                        self._midi_smoothed_frequency = frequency
                self._midi_note_pulse = 1.0
                if self.params.get("amp_env_enabled", True):
                    self._amp_env_stage = "attack"
                if str(self.params.get("midi_style", "Musical")) == "Musical":
                    self._trigger_operator_envelopes_locked(voice_index=voice_index)
                else:
                    self._trigger_operator_envelopes_locked()
                if self.params.get("midi_retrigger_phase", True):
                    if str(self.params.get("midi_style", "Musical")) == "Musical":
                        self._voice_phase[voice_index, :].fill(0.0)
                        self._voice_feedback[voice_index, :].fill(0.0)
                    else:
                        self._voice_phase.fill(0.0)
                        self._voice_feedback.fill(0.0)
                self._effective_params.update(
                    midi_velocity=normalized_velocity,
                    midi_gate=normalized_velocity,
                )
            return True
        if msg_type == "note_off" or (msg_type == "note_on" and velocity <= 0):
            with self._lock:
                self._midi_active_notes = [
                    (active_note, active_velocity)
                    for active_note, active_velocity in self._midi_active_notes
                    if active_note != note
                ]
                if self._midi_active_notes:
                    active_note, active_velocity = self._midi_active_notes[-1]
                    transpose = int(round(float(self.params.get("midi_transpose", 0.0))))
                    if self.params.get("midi_note_follow", True):
                        frequency = midi_to_frequency(max(0, min(127, active_note + transpose)))
                        self._midi_target_frequency = frequency
                    self.params["midi_velocity"] = active_velocity
                    self.params["midi_gate"] = active_velocity
                else:
                    self.params["midi_gate"] = 0.0 if self.params.get("midi_gate_enabled", False) else 1.0
                    self.params["midi_velocity"] = 1.0
                    if self.params.get("midi_gate_enabled", False) and self.params.get("amp_env_enabled", True):
                        self._amp_env_stage = "release"
                    if self.params.get("midi_gate_enabled", False):
                        self._release_operator_envelopes_locked()
                self._effective_params.update(
                    midi_velocity=self.params["midi_velocity"],
                    midi_gate=self.params["midi_gate"],
                )
            return True
        return False

    def set_param_mod_source(self, name, source):
        if name not in self._param_mod_sources:
            raise KeyError(f"Parameter cannot be modulated: {name}")
        source = str(source or "None")
        if source not in MEDIA_AUDIO_MOD_SOURCES:
            source = "None"
        with self._lock:
            self._param_mod_sources[name] = source
            self._effective_params = self._apply_param_modulation_from_metrics(
                self.params,
                self._live_metrics,
            )

    def apply_preset(self, name):
        preset = MEDIA_AUDIO_PRESETS.get(name)
        if preset is None:
            raise KeyError(f"Unknown media-audio preset: {name}")
        for param_name, value in preset.items():
            self.set_param(param_name, value)
        with self._lock:
            self._trigger_operator_envelopes_locked()
        return dict(preset)

    def add_imported_patches(self, patches):
        added = {}
        for patch in patches:
            values = tx802_patch_to_media_params(patch)
            base_name = patch.name or "Imported TX802"
            name = base_name
            suffix = 2
            while name in MEDIA_AUDIO_PRESETS or name in added:
                name = f"{base_name} {suffix}"
                suffix += 1
            added[name] = values
        MEDIA_AUDIO_PRESETS.update(added)
        return added

    def _trigger_operator_envelopes_locked(self, voice_index=None):
        if voice_index is None:
            voice_slice = slice(None)
        else:
            voice_slice = max(0, min(self._max_voices - 1, int(voice_index)))
        for op in range(self._operator_count):
            level4 = float(self.params.get(f"operator_eg_level_{op + 1}_4", 0.0))
            self._operator_env_level[voice_slice, op] = min(1.0, max(0.0, level4))
            self._operator_env_stage[voice_slice, op] = 0

    def _release_operator_envelopes_locked(self):
        self._operator_env_stage[:, :] = 4

    def play(self, restart=False):
        with self._lock:
            if restart:
                self._position = 0.0
                self._render_time = 0.0
                self._voice_phase.fill(0.0)
                self._voice_feedback.fill(0.0)
                self._voice_lfo_phase[:] = np.linspace(
                    0.0, math.pi, self._max_voices, dtype=np.float64
                )
                self._filter_low.fill(0.0)
                self._filter_band.fill(0.0)
                self._trigger_operator_envelopes_locked()
                self._amp_env_level = 0.0 if self.params.get("amp_env_enabled", True) else 1.0
                self._amp_env_stage = "attack" if self.params.get("amp_env_enabled", True) else "sustain"
            can_play = (
                self.source_mode != "Loaded Media"
                or self._analysis is not None
            )
            self._playing = can_play
            return self._playing

    def stop(self):
        with self._lock:
            self._playing = False

    def seek(self, seconds):
        with self._lock:
            duration = self._analysis.duration if self._analysis else 0.0
            self._position = min(max(0.0, float(seconds)), duration)

    def snapshot(self):
        with self._lock:
            return {
                "loaded": self._analysis is not None,
                "playing": self._playing,
                "position": float(self._position),
                "duration": (
                    float(self._analysis.duration) if self._analysis else 0.0
                ),
                "source_path": (
                    self._analysis.source_path if self._analysis else ""
                ),
                "source_mode": self.source_mode,
                "live_metrics_received": self._live_metrics_received,
                "ir_path": self._ir_path,
                "ir_loaded": self._convolver is not None,
                "params": dict(self.params),
                "effective_params": dict(self._effective_params),
                "param_mod_sources": dict(self._param_mod_sources),
                "cdp_graph": self.cdp_graph.snapshot(),
            }

    def export_preset_state(self):
        with self._lock:
            return {
                "source_mode": self.source_mode,
                "source_path": self._analysis.source_path if self._analysis else "",
                "ir_path": self._ir_path,
                "params": dict(self.params),
                "param_mod_sources": dict(self._param_mod_sources),
                "cdp_graph": self.cdp_graph.snapshot(),
            }

    def import_preset_state(self, state):
        if not isinstance(state, dict):
            return
        params = state.get("params")
        if isinstance(params, dict):
            for name, value in params.items():
                if name in self.params:
                    try:
                        self.set_param(name, value)
                    except Exception:
                        logging.debug("Skipping invalid Media Audio preset parameter %s", name, exc_info=True)
        mod_sources = state.get("param_mod_sources")
        if isinstance(mod_sources, dict):
            for name, source in mod_sources.items():
                if name in self._param_mod_sources:
                    try:
                        self.set_param_mod_source(name, source)
                    except Exception:
                        logging.debug("Skipping invalid Media Audio modulation source %s", name, exc_info=True)
        source_mode = state.get("source_mode")
        if source_mode:
            try:
                self.set_source_mode(source_mode)
            except Exception:
                logging.debug("Skipping invalid Media Audio source mode %s", source_mode, exc_info=True)
        self.cdp_graph.apply_snapshot(state.get("cdp_graph"))
        with self._lock:
            self._trigger_operator_envelopes_locked()
            self._effective_params = self._apply_param_modulation_from_metrics(
                self.params,
                self._live_metrics,
            )

    def render(self, frames=None, out=None):
        frames = self.block_size if frames is None else max(1, int(frames))
        if out is None:
            out = np.zeros((frames, 2), dtype=np.float32)
        else:
            out[:frames].fill(0.0)

        with self._lock:
            analysis = self._analysis
            params = dict(self.params)
            playing = self._playing
            source_mode = self.source_mode
            live_metrics = dict(self._live_metrics)
            smoothed_metrics = dict(self._smoothed_metrics)
            convolver = self._convolver
            position = float(self._position)
            render_time = float(self._render_time)
            voice_phase = self._voice_phase.copy()
            voice_feedback = self._voice_feedback.copy()
            voice_lfo_phase = self._voice_lfo_phase.copy()
            midi_active_notes = list(self._midi_active_notes)
            midi_target_frequency = float(self._midi_target_frequency)
            midi_smoothed_frequency = float(self._midi_smoothed_frequency)
            midi_note_pulse = float(self._midi_note_pulse)
            amp_env_level = float(self._amp_env_level)
            amp_env_stage = str(self._amp_env_stage)
            filter_low = self._filter_low.copy()
            filter_band = self._filter_band.copy()
            operator_env_level = self._operator_env_level.copy()
            operator_env_stage = self._operator_env_stage.copy()

        if not playing or not params["enabled"]:
            return out

        rate = float(params["playback_rate"])
        sample_offsets = np.arange(frames, dtype=np.float64) / self.sample_rate
        visual = {
            name: np.full(frames, float(live_metrics[name]), dtype=np.float64)
            for name in live_metrics
        }
        loop = bool(params["loop"])

        has_media = analysis is not None and analysis.duration > 0.0
        if source_mode in {"Loaded Media", "Hybrid"} and has_media:
            media_times = position + sample_offsets * rate
            if loop:
                lookup_times = np.mod(media_times, analysis.duration)
            else:
                lookup_times = np.minimum(media_times, analysis.duration)
            media_brightness = np.interp(
                lookup_times, analysis.timestamps, analysis.brightness
            )
            media_motion = np.interp(
                lookup_times, analysis.timestamps, analysis.motion
            )
            media_red = np.interp(
                lookup_times, analysis.timestamps, analysis.colors[:, 0]
            )
            media_blue = np.interp(
                lookup_times, analysis.timestamps, analysis.colors[:, 2]
            )
            media_green = np.interp(
                lookup_times, analysis.timestamps, analysis.colors[:, 1]
            )
            media_metrics = {
                "brightness": media_brightness,
                "motion": media_motion,
                "red_energy": media_red,
                "green_energy": media_green,
                "blue_energy": media_blue,
                "contrast": np.clip(np.abs(media_brightness - 0.5) * 2.0, 0.0, 1.0),
                "entropy": np.clip(media_brightness * (1.0 - media_brightness) * 4.0, 0.0, 1.0),
                "edge_density": np.clip(media_motion * 0.7 + np.abs(media_red - media_blue) * 0.3, 0.0, 1.0),
                "glitch_spike": np.clip(media_motion * np.maximum(media_brightness, 0.2), 0.0, 1.0),
                "center_mass_x": np.clip(0.5 + (media_red - media_blue) * 0.35, 0.0, 1.0),
                "center_mass_y": np.clip(0.5 + (media_green - media_brightness) * 0.35, 0.0, 1.0),
            }
            if source_mode == "Loaded Media":
                visual = media_metrics
            else:
                hybrid_mix = float(params["hybrid_mix"])
                for name, media_values in media_metrics.items():
                    visual[name] = visual[name] * (1.0 - hybrid_mix) + media_values * hybrid_mix
        elif source_mode == "Loaded Media" and not has_media:
            return out

        params = self._apply_param_modulation(params, visual)
        visual_ir_motion = max(0.0, min(1.0, float(params.get("visual_ir_motion", 0.35))))
        if visual_ir_motion > 0.0:
            visual_room = (
                float(np.mean(np.clip(visual["entropy"], 0.0, 1.0))) * 0.45
                + float(np.mean(np.clip(visual["edge_density"], 0.0, 1.0))) * 0.25
                + float(np.mean(np.clip(visual["motion"], 0.0, 1.0))) * 0.3
            )
            params["ir_wet"] = min(
                1.0,
                max(
                    0.0,
                    float(params.get("ir_wet", 0.0))
                    * (1.0 + visual_ir_motion * (visual_room - 0.35) * 0.9),
                ),
            )
            params["ir_gain"] = min(
                1.25,
                max(
                    0.0,
                    float(params.get("ir_gain", 0.0))
                    * (1.0 + visual_ir_motion * (visual_room - 0.5) * 0.35),
                ),
            )
        if params.get("midi_note_follow", True):
            midi_style = str(params.get("midi_style", "Musical"))
            musical_midi = midi_style == "Musical"
            midi_poly_notes = []
            if musical_midi and midi_active_notes:
                transpose = int(round(float(params.get("midi_transpose", 0.0))))
                for note, velocity in midi_active_notes[-self._max_voices:]:
                    frequency = midi_to_frequency(max(0, min(127, int(note) + transpose)))
                    midi_poly_notes.append((frequency, max(0.0, min(1.0, float(velocity)))))
            slew_ms = max(0.0, float(params.get("midi_pitch_slew_ms", 45.0)))
            if slew_ms <= 0.0:
                midi_smoothed_frequency = midi_target_frequency
            else:
                slew_samples = max(1.0, self.sample_rate * slew_ms * 0.001)
                slew_alpha = 1.0 - math.exp(-frames / slew_samples)
                midi_smoothed_frequency += (
                    midi_target_frequency - midi_smoothed_frequency
                ) * slew_alpha
            params["carrier_freq"] = max(5.0, min(20000.0, midi_smoothed_frequency))

            midi_velocity = max(0.0, min(1.0, float(params.get("midi_velocity", 1.0))))
            timbre_depth = max(0.0, min(2.0, float(params.get("midi_timbre_depth", 0.45))))
            note_punch = max(0.0, min(1.0, float(params.get("midi_note_punch", 0.28))))
            note_focus = max(0.0, min(1.0, float(params.get("midi_note_focus", 0.55))))
            pulse = max(0.0, min(1.0, midi_note_pulse))
            if musical_midi:
                def blend(raw_value, focused_value):
                    return raw_value + (focused_value - raw_value) * note_focus

                voice_count = int(params.get("voice_count", 1))
                if midi_poly_notes:
                    params["_midi_poly_notes"] = tuple(midi_poly_notes)
                    if note_focus > 0.0:
                        params["voice_count"] = max(1, len(midi_poly_notes))
                else:
                    params["voice_count"] = max(
                        1,
                        int(round(blend(float(voice_count), float(min(3, voice_count))))),
                    )
                poly_spread = float(params.get("poly_spread", 0.0))
                chord_depth = float(params.get("chord_depth", 0.0))
                timbre_spread = float(params.get("timbre_spread", 0.0))
                motion_response = float(params.get("motion_response", 0.0))
                feedback_color = float(params.get("feedback_color", 0.0))
                visual_drive = float(params.get("visual_drive", 1.0))
                raw_modulation_index = float(params["modulation_index"])
                raw_feedback = float(params.get("feedback", 0.0))
                params["poly_spread"] = blend(poly_spread, poly_spread * (0.46 if midi_poly_notes else 0.22))
                params["chord_depth"] = (
                    blend(chord_depth, 0.0)
                    if midi_poly_notes
                    else blend(chord_depth, chord_depth * 0.24)
                )
                params["timbre_spread"] = blend(timbre_spread, timbre_spread * 0.38)
                params["motion_response"] = blend(motion_response, motion_response * 0.32)
                params["feedback_color"] = blend(feedback_color, feedback_color * 0.18)
                params["visual_drive"] = blend(visual_drive, min(1.0, visual_drive))
                expressive_index = raw_modulation_index * (
                    1.0 + midi_velocity * timbre_depth * 0.3 + pulse * note_punch * 0.2
                )
                focused_index = min(
                    4.5,
                    raw_modulation_index
                    * (1.0 + midi_velocity * timbre_depth * 0.55 + pulse * note_punch * 0.35),
                )
                params["modulation_index"] = blend(expressive_index, focused_index)
                focused_feedback = min(
                    0.35,
                    raw_feedback * 0.35 + pulse * note_punch * 0.035,
                )
                params["feedback"] = blend(raw_feedback, focused_feedback)
                params["_midi_musical_articulation"] = 0.58 + pulse * 0.42
                params["_midi_reduce_visual_bend"] = blend(1.0, 0.25)
            else:
                params["modulation_index"] = float(params["modulation_index"]) * (
                    1.0 + midi_velocity * timbre_depth + pulse * note_punch
                )
                params["feedback"] = min(
                    0.95,
                    float(params.get("feedback", 0.0)) + pulse * note_punch * 0.18,
                )
                params["visual_drive"] = min(
                    2.0,
                    float(params.get("visual_drive", 1.0)) + pulse * note_punch * 0.35,
                )
            pulse_decay = math.exp(-frames / max(1.0, self.sample_rate * 0.18))
            midi_note_pulse *= pulse_decay

        out[:frames] = self._render_fm_texture(
            frames,
            params,
            visual,
            smoothed_metrics,
            voice_phase,
            voice_feedback,
            voice_lfo_phase,
            operator_env_level,
            operator_env_stage,
            render_time,
        )
        if params.get("idm_mode", False):
            out[:frames] = self._apply_idm_dirt(out[:frames], params)
        if params.get("amp_env_enabled", True):
            amp_gate = bool(float(params.get("midi_gate", 1.0)) > 0.0) if params.get("midi_gate_enabled", False) else True
            envelope, amp_env_level, amp_env_stage = self._render_amp_envelope(
                frames,
                params,
                amp_gate,
                amp_env_level,
                amp_env_stage,
            )
            out[:frames] *= envelope[:, None].astype(np.float32)
            params["_amp_env_level"] = float(amp_env_level)
            params["_amp_env_stage"] = amp_env_stage
        if params.get("filter_enabled", True):
            out[:frames], filter_low, filter_band = self._apply_filter(
                out[:frames],
                params,
                visual,
                amp_env_level,
                filter_low,
                filter_band,
            )
        out[:frames] = self.cdp_graph.process(
            out[:frames],
            self.sample_rate,
            visual={name: float(np.mean(values)) for name, values in visual.items()},
        )
        dry = out[:frames].copy()
        if convolver is not None and frames == convolver.block_size:
            wet = convolver.process(dry)
            wet_amount = min(1.0, max(0.0, float(params["ir_wet"])))
            wet_gain = min(1.25, max(0.0, float(params["ir_gain"])))
            wet *= np.float32(wet_gain)
            wet_peak = float(np.max(np.abs(wet))) if wet.size else 0.0
            if wet_peak > 1.25:
                wet *= np.float32(1.25 / wet_peak)
            out[:frames] = (
                dry * (1.0 - wet_amount * 0.45)
                + wet * wet_amount
            )
        np.tanh(out[:frames] * 1.35, out=out[:frames])
        out[:frames] /= np.float32(math.tanh(1.35))
        np.clip(out[:frames], -1.0, 1.0, out=out[:frames])
        np.nan_to_num(out[:frames], copy=False)

        new_position = position
        still_playing = True
        if has_media:
            new_position += frames / self.sample_rate * rate
            if loop:
                new_position %= analysis.duration
            elif new_position >= analysis.duration:
                new_position = analysis.duration
                still_playing = False

        with self._lock:
            self._position = new_position
            self._render_time = render_time + frames / self.sample_rate
            self._playing = still_playing
            self._voice_phase[:] = voice_phase
            self._voice_feedback[:] = voice_feedback
            self._voice_lfo_phase[:] = voice_lfo_phase
            self._midi_smoothed_frequency = midi_smoothed_frequency
            self._midi_note_pulse = midi_note_pulse
            self._amp_env_level = amp_env_level
            self._amp_env_stage = amp_env_stage
            self._filter_low[:] = filter_low
            self._filter_band[:] = filter_band
            self._operator_env_level[:] = operator_env_level
            self._operator_env_stage[:] = operator_env_stage
            self._smoothed_metrics = smoothed_metrics
            self._effective_params = dict(params)
        return out

    def _render_amp_envelope(self, frames, params, gate_on, level, stage):
        attack_ms = max(0.0, float(params.get("amp_attack_ms", 15.0)))
        decay_ms = max(0.0, float(params.get("amp_decay_ms", 450.0)))
        sustain = max(0.0, min(1.0, float(params.get("amp_sustain", 1.0))))
        release_ms = max(0.0, float(params.get("amp_release_ms", 900.0)))
        envelope = np.empty(frames, dtype=np.float64)
        level = max(0.0, min(1.0, float(level)))
        stage = str(stage or "idle")
        if gate_on and stage in {"idle", "release"}:
            stage = "attack"
        elif not gate_on and stage != "idle":
            stage = "release"

        attack_step = 1.0 if attack_ms <= 0.0 else 1.0 / max(1.0, self.sample_rate * attack_ms * 0.001)
        decay_step = 1.0 if decay_ms <= 0.0 else 1.0 / max(1.0, self.sample_rate * decay_ms * 0.001)
        release_step = 1.0 if release_ms <= 0.0 else 1.0 / max(1.0, self.sample_rate * release_ms * 0.001)

        for index in range(frames):
            if stage == "attack":
                level += attack_step
                if level >= 1.0:
                    level = 1.0
                    stage = "decay"
            elif stage == "decay":
                if level > sustain:
                    level -= decay_step * max(0.05, 1.0 - sustain)
                    if level <= sustain:
                        level = sustain
                        stage = "sustain"
                else:
                    level = sustain
                    stage = "sustain"
            elif stage == "sustain":
                level = sustain if gate_on else level
            elif stage == "release":
                level -= release_step * max(0.05, level)
                if level <= 1e-4:
                    level = 0.0
                    stage = "idle"
            else:
                level = 0.0 if not gate_on else level
                stage = "idle" if not gate_on else "attack"
            envelope[index] = level
        return envelope, level, stage

    def _apply_filter(self, audio, params, visual, amp_env_level, low_state, band_state):
        output = np.asarray(audio, dtype=np.float32).copy()
        if output.size == 0:
            return output, low_state, band_state
        cutoff = float(params.get("filter_cutoff", 6200.0))
        resonance = max(0.0, min(0.95, float(params.get("filter_resonance", 0.18))))
        env_amount = max(-1.0, min(1.0, float(params.get("filter_env_amount", 0.0))))
        visual_amount = max(0.0, min(1.0, float(params.get("filter_visual_amount", 0.35))))
        drive = max(0.0, min(2.0, float(params.get("filter_drive", 0.15))))
        brightness = np.clip(visual.get("brightness", np.zeros(output.shape[0])), 0.0, 1.0)
        motion = np.clip(visual.get("motion", np.zeros(output.shape[0])), 0.0, 1.0)
        edge = np.clip(visual.get("edge_density", np.zeros(output.shape[0])), 0.0, 1.0)
        visual_curve = brightness * 0.55 + motion * 0.25 + edge * 0.2
        env_octaves = env_amount * (float(amp_env_level) - 0.5) * 4.0
        visual_octaves = (float(np.mean(visual_curve)) - 0.5) * visual_amount * 3.0
        cutoff_value = float(np.clip(
            cutoff * math.pow(2.0, env_octaves + visual_octaves),
            40.0,
            min(18000.0, self.sample_rate * 0.24),
        ))
        coefficient = 1.0 - math.exp(-2.0 * math.pi * cutoff_value / self.sample_rate)
        coefficient = max(0.0, min(0.98, coefficient))
        if drive > 0.0:
            output = np.tanh(output * np.float32(1.0 + drive * 1.8)).astype(np.float32)
        pole = 1.0 - coefficient
        numerator = np.asarray([coefficient], dtype=np.float64)
        denominator = np.asarray([1.0, -pole], dtype=np.float64)

        stage1, zf1 = lfilter(
            numerator,
            denominator,
            output.astype(np.float64, copy=False),
            axis=0,
            zi=(pole * low_state.astype(np.float64, copy=False))[None, :],
        )
        stage2, zf2 = lfilter(
            numerator,
            denominator,
            stage1,
            axis=0,
            zi=(pole * band_state.astype(np.float64, copy=False))[None, :],
        )
        filtered = stage2 * (1.0 - resonance * 0.28) + stage1 * (resonance * 0.28)
        np.clip(filtered, -4.0, 4.0, out=filtered)
        np.nan_to_num(filtered, copy=False)
        output[:] = filtered.astype(np.float32, copy=False)
        if pole > 1e-9:
            stage1_state = zf1[0] / pole
            stage2_state = zf2[0] / pole
        else:
            stage1_state = stage1[-1]
            stage2_state = stage2[-1]
        stage1_state = np.clip(np.nan_to_num(stage1_state, nan=0.0, posinf=0.0, neginf=0.0), -4.0, 4.0)
        stage2_state = np.clip(np.nan_to_num(stage2_state, nan=0.0, posinf=0.0, neginf=0.0), -4.0, 4.0)
        return output, stage1_state, stage2_state

    def _apply_param_modulation(self, params, visual):
        effective = dict(params)
        sources = dict(self._param_mod_sources)
        for name, source in sources.items():
            if source == "None" or source not in visual or name not in self.PARAM_LIMITS:
                continue
            low, high = self.PARAM_LIMITS[name]
            value = float(np.mean(np.clip(visual[source], 0.0, 1.0)))
            mapped = low + value * (high - low)
            if name in {"voice_count", "algorithm"}:
                mapped = int(round(mapped))
            effective[name] = mapped
            if name.startswith("operator_level_"):
                effective[name.replace("operator_level_", "operator_output_level_", 1)] = mapped
            elif name.startswith("operator_output_level_"):
                effective[name.replace("operator_output_level_", "operator_level_", 1)] = mapped
        return effective

    def _apply_param_modulation_from_metrics(self, params, metrics):
        effective = dict(params)
        for name, source in self._param_mod_sources.items():
            if source == "None" or source not in metrics or name not in self.PARAM_LIMITS:
                continue
            low, high = self.PARAM_LIMITS[name]
            value = min(1.0, max(0.0, float(metrics.get(source, 0.0))))
            mapped = low + value * (high - low)
            if name in {"voice_count", "algorithm"}:
                mapped = int(round(mapped))
            effective[name] = mapped
            if name.startswith("operator_level_"):
                effective[name.replace("operator_level_", "operator_output_level_", 1)] = mapped
            elif name.startswith("operator_output_level_"):
                effective[name.replace("operator_output_level_", "operator_level_", 1)] = mapped
        return effective

    def _render_fm_texture(
        self,
        frames,
        params,
        visual,
        smoothed_metrics,
        voice_phase,
        voice_feedback,
        voice_lfo_phase,
        operator_env_level,
        operator_env_stage,
        render_time,
    ):
        block = np.zeros((frames, 2), dtype=np.float32)
        alpha = 1.0 - math.exp(-frames / max(1.0, self.sample_rate * 0.035))
        block_metrics = {}
        for name, values in visual.items():
            target = float(np.mean(values))
            previous = float(smoothed_metrics.get(name, target))
            smoothed_metrics[name] = previous + (target - previous) * alpha
            block_metrics[name] = smoothed_metrics[name]

        midi_poly_notes = tuple(params.get("_midi_poly_notes", ()) or ())
        if midi_poly_notes:
            voices = min(self._max_voices, len(midi_poly_notes))
            midi_poly_frequencies = np.asarray(
                [float(note[0]) for note in midi_poly_notes[:voices]],
                dtype=np.float64,
            )
            midi_poly_velocities = np.asarray(
                [float(note[1]) for note in midi_poly_notes[:voices]],
                dtype=np.float64,
            )
        else:
            voices = min(self._max_voices, int(params["voice_count"]))
            midi_poly_frequencies = None
            midi_poly_velocities = None
        base_frequency = float(params["carrier_freq"])
        mod_ratio = float(params["modulator_ratio"])
        modulation_index = math.tanh(float(params["modulation_index"]) / 6.0) * 6.0
        manual_feedback = float(params.get("feedback", 0.0))
        master_gain = float(params["master_gain"])
        midi_gate = float(params.get("midi_gate", 1.0)) if params.get("midi_gate_enabled", False) else 1.0
        midi_velocity = float(params.get("midi_velocity", 1.0))
        midi_velocity_depth = float(params.get("midi_velocity_depth", 0.6))
        midi_amp = midi_gate * ((1.0 - midi_velocity_depth) + midi_velocity * midi_velocity_depth)
        brightness_mix = float(params["brightness_response"])
        visual_influence = float(params.get("visual_influence", 1.0))
        visual_evolution = float(params.get("visual_evolution", 0.5))
        visual_pitch_motion = float(params.get("visual_pitch_motion", 0.5))
        motion_response = float(params["motion_response"]) * (0.55 + visual_influence * 0.8)
        visual_drive = float(params["visual_drive"]) * (0.35 + visual_influence)
        poly_spread = float(params["poly_spread"])
        chord_depth = float(params["chord_depth"])
        timbre_spread = float(params["timbre_spread"]) * (0.5 + visual_influence * 0.75)
        feedback_color = float(params["feedback_color"]) * (0.45 + visual_influence * 0.7)
        motion_gating = float(params["motion_gating"])
        stereo_amount = float(params["color_stereo"])
        midi_articulation = float(params.get("_midi_musical_articulation", 1.0))
        visual_bend_amount = float(params.get("_midi_reduce_visual_bend", 1.0))
        visual_morph_depth = max(0.0, min(1.0, float(params.get("visual_morph_depth", 0.5))))
        idm_enabled = bool(params.get("idm_mode", False))
        idm_intensity = max(0.0, min(1.0, float(params.get("idm_intensity", 0.0)))) if idm_enabled else 0.0
        algorithm_index = min(31, max(0, int(params.get("algorithm", 1)) - 1))
        idm_curves = self._idm_event_curves(frames, render_time, params, visual) if idm_intensity > 0.0 else None
        if idm_curves is not None:
            morph_depth = idm_intensity * max(
                0.0,
                min(1.0, float(params.get("idm_algorithm_morph_depth", 0.0))),
            )
            if morph_depth > 0.0:
                group = min(
                    IDM_ALGORITHM_MORPH_GROUPS,
                    key=lambda values: min(abs((value - 1) - algorithm_index) for value in values),
                )
                target = group[int(idm_curves["block_hash"] * len(group)) % len(group)] - 1
                if idm_curves["event_strength"] * morph_depth > 0.18:
                    algorithm_index = min(31, max(0, target))
        algorithm_targets = TX802_ALGORITHM_TARGETS[algorithm_index]
        feedback_operator = TX802_FEEDBACK_OPERATORS[algorithm_index]
        operator_ratios = np.asarray(
            [float(params.get(f"operator_ratio_{index}", 1.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_levels = np.asarray(
            [float(params.get(f"operator_output_level_{index}", params.get(f"operator_level_{index}", 1.0 if index == 1 else 0.0))) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_detunes = np.asarray(
            [float(params.get(f"operator_detune_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_pans = np.asarray(
            [float(params.get(f"operator_pan_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_fixed = np.asarray(
            [bool(params.get(f"operator_fixed_{index}", False)) for index in range(1, 7)],
            dtype=bool,
        )
        operator_fixed_freqs = np.asarray(
            [float(params.get(f"operator_fixed_freq_{index}", 440.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_enabled = np.asarray(
            [bool(params.get(f"operator_enabled_{index}", True)) for index in range(1, 7)],
            dtype=bool,
        )
        operator_velocity = np.asarray(
            [float(params.get(f"operator_velocity_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_key_level = np.asarray(
            [float(params.get(f"operator_key_level_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_key_rate = np.asarray(
            [float(params.get(f"operator_key_rate_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_glitch = np.asarray(
            [float(params.get(f"operator_glitch_depth_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_burst = np.asarray(
            [float(params.get(f"operator_burst_probability_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_level_jitter = np.asarray(
            [float(params.get(f"operator_level_jitter_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )
        operator_ratio_snap = np.asarray(
            [float(params.get(f"operator_ratio_snap_{index}", 0.0)) for index in range(1, 7)],
            dtype=np.float64,
        )

        brightness = np.clip(visual["brightness"], 0.0, 1.0)
        motion = np.clip(visual["motion"], 0.0, 1.0)
        red = np.clip(visual["red_energy"], 0.0, 1.0)
        green = np.clip(visual["green_energy"], 0.0, 1.0)
        blue = np.clip(visual["blue_energy"], 0.0, 1.0)
        contrast = np.clip(visual["contrast"], 0.0, 1.0)
        entropy = np.clip(visual["entropy"], 0.0, 1.0)
        edge = np.clip(visual["edge_density"], 0.0, 1.0)
        spike = np.clip(visual["glitch_spike"], 0.0, 1.0)
        center_x = np.clip(visual["center_mass_x"], 0.0, 1.0)
        center_y = np.clip(visual["center_mass_y"], 0.0, 1.0)

        brightness_curve = (1.0 - brightness_mix) + brightness * brightness_mix
        gate_curve = (1.0 - motion_gating) + motion_gating * np.clip(
            0.18 + motion * 1.1 + spike * 0.7, 0.0, 1.0
        )
        if idm_curves is not None:
            gate_curve *= idm_curves["amp_gate"]
        frame_gain = brightness_curve * gate_curve * midi_amp * midi_articulation
        index_curve = modulation_index * (
            1.0
            + motion * motion_response
            + edge * 0.85 * visual_drive
            + spike * 1.4 * visual_drive
        )
        feedback_curve = np.clip(
            manual_feedback
            + visual_morph_depth * feedback_color * (0.15 + red * 0.35 + blue * 0.25 + spike * 0.5),
            0.0,
            0.95,
        )
        base_bend = np.power(
            2.0,
            (
                ((brightness - 0.5) * 7.0 * visual_drive + (center_y - 0.5) * 3.0)
                * visual_bend_amount
                * visual_pitch_motion
            )
            / 12.0,
        )

        intervals = np.array([0.0, 7.0, 12.0, 19.0, 24.0, 28.0, 31.0, 36.0, 43.0, 48.0, 55.0, 60.0, 67.0, 72.0, 79.0, 84.0])
        dark_intervals = np.array([0.0, 3.0, 7.0, 10.0, 14.0, 19.0, 22.0, 27.0, 31.0, 34.0, 39.0, 43.0, 46.0, 51.0, 55.0, 58.0])
        color_balance = block_metrics["red_energy"] - block_metrics["blue_energy"]
        palette = np.clip(block_metrics["green_energy"] * 0.55 + block_metrics["entropy"] * 0.45, 0.0, 1.0)
        evolution_amount = visual_evolution * (
            0.35
            + block_metrics["entropy"] * 0.3
            + block_metrics["motion"] * 0.22
            + block_metrics["edge_density"] * 0.13
        )
        voice_norm = max(1.0, math.sqrt(float(voices)))

        for voice in range(voices):
            position = 0.0 if voices == 1 else voice / float(voices - 1)
            if midi_poly_frequencies is not None:
                interval = 0.0
                spread_cents = (position - 0.5) * 26.0 * poly_spread
                voice_velocity = float(midi_poly_velocities[voice])
                voice_base_frequency = float(midi_poly_frequencies[voice])
            else:
                interval = (
                    dark_intervals[voice] * (1.0 - palette)
                    + intervals[voice] * palette
                ) * chord_depth * (1.0 + evolution_amount * (0.6 + position * 0.8))
                spread_cents = (position - 0.5) * 90.0 * poly_spread
                voice_velocity = midi_velocity
                voice_base_frequency = base_frequency
            spread_cents += color_balance * 32.0 * feedback_color
            spread_cents += (
                block_metrics["center_mass_y"] - 0.5
            ) * 42.0 * visual_pitch_motion * visual_evolution
            voice_frequency = voice_base_frequency * np.power(
                2.0, (interval + spread_cents / 100.0) / 12.0
            )
            voice_pan = np.clip(
                (position * 2.0 - 1.0) * stereo_amount
                + (block_metrics["center_mass_x"] - 0.5) * 0.75,
                -1.0,
                1.0,
            )
            left_gain, right_gain = self._pan_gains(voice_pan)
            voice_tone = position * 2.0 - 1.0
            visual_ratios = np.vstack(
                (
                    np.full(frames, 1.0, dtype=np.float64),
                    1.0 + (green - 0.5) * timbre_spread * visual_morph_depth,
                    1.0 + (blue - 0.5) * timbre_spread * visual_morph_depth + entropy * 0.12 * evolution_amount,
                    1.0 + (red - 0.5) * timbre_spread * visual_morph_depth + entropy * 0.16 * evolution_amount,
                    1.0 + (contrast - 0.5) * timbre_spread * visual_morph_depth + center_x * 0.06 * evolution_amount,
                    1.0 + (edge - 0.5) * timbre_spread * visual_morph_depth + spike * 0.18 * evolution_amount,
                )
            )
            ratios = visual_ratios * operator_ratios[:, None] * np.asarray(
                (
                    1.0,
                    max(0.125, mod_ratio),
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                dtype=np.float64,
            )[:, None]
            if idm_curves is not None:
                ratios = self._apply_idm_ratio_snaps(
                    ratios,
                    params,
                    idm_curves,
                    operator_ratio_snap,
                    idm_intensity,
                )
            env_curves = np.vstack(
                [
                    self._render_operator_envelope_block(
                        frames,
                        params,
                        voice,
                        op,
                        operator_env_level,
                        operator_env_stage,
                        gate_on=midi_gate > 0.0 or not params.get("midi_gate_enabled", False),
                        key_rate=operator_key_rate[op],
                    )
                    for op in range(self._operator_count)
                ]
            )
            if idm_curves is not None:
                carrier_flags = np.asarray(
                    [target < 0 for target in algorithm_targets],
                    dtype=bool,
                )
                env_curves = self._apply_idm_envelope_slicing(
                    env_curves,
                    params,
                    idm_curves,
                    carrier_flags,
                    idm_intensity,
                )
            visual_level = np.vstack(
                (
                    1.0 + (brightness - 0.5) * 0.16 * visual_morph_depth,
                    1.0 + contrast * 0.4 * timbre_spread * visual_morph_depth,
                    1.0 + edge * 0.36 * timbre_spread * visual_morph_depth,
                    1.0 + spike * 0.52 * visual_drive * visual_morph_depth,
                    1.0 + red * 0.34 * timbre_spread * visual_morph_depth,
                    1.0 + blue * 0.38 * timbre_spread * visual_morph_depth,
                )
            )
            midi_op_response = (1.0 - operator_velocity[:, None]) + operator_velocity[:, None] * voice_velocity
            key_level_response = 1.0 - operator_key_level[:, None] * np.clip((voice_frequency / 880.0) - 0.5, 0.0, 0.65)
            audible = (
                operator_levels[:, None]
                * operator_enabled[:, None]
                * env_curves
                * visual_level
                * midi_op_response
                * np.clip(key_level_response, 0.35, 1.25)
            )
            if idm_curves is not None:
                audible = self._apply_idm_operator_bursts(
                    audible,
                    idm_curves,
                    operator_glitch,
                    operator_burst,
                    operator_level_jitter,
                    idm_intensity,
                )

            lfo_rate = float(params.get("lfo_rate", 0.0)) + visual_morph_depth * (motion * 0.55 + position * 0.025)
            lfo_steps = 2.0 * math.pi * lfo_rate / self.sample_rate
            lfo_phase = voice_lfo_phase[voice] + np.cumsum(lfo_steps) - lfo_steps
            lfo = self._lfo_wave(lfo_phase, params.get("lfo_wave", "sine"))
            voice_lfo_phase[voice] = float(
                (lfo_phase[-1] + lfo_steps[-1]) % (2.0 * math.pi)
            )
            lfo_pitch_cents = (
                max(0.0, min(1.0, float(params.get("lfo_depth", 0.0)))) * 50.0
                + visual_morph_depth * visual_pitch_motion * 6.0
            )
            freq = voice_frequency * base_bend * np.power(
                2.0,
                (lfo * lfo_pitch_cents * (1.0 + voice_tone * 0.25)) / 1200.0,
            )
            op_freqs = freq[None, :] * ratios * np.power(2.0, operator_detunes[:, None] / 1200.0)
            if np.any(operator_fixed):
                fixed_motion = 1.0 + visual_morph_depth * (edge - 0.5) * 0.08
                op_freqs[operator_fixed] = operator_fixed_freqs[operator_fixed][:, None] * fixed_motion
            op_freqs = np.minimum(op_freqs, self.sample_rate * 0.22)
            phase_steps = 2.0 * math.pi * op_freqs / self.sample_rate
            phases = voice_phase[voice, :, None] + np.cumsum(phase_steps, axis=1) - phase_steps
            carrier_amplitude = self._tx802_carrier_amplitude(audible)
            mod_depths = self._tx802_modulation_depth(audible)
            if idm_curves is not None:
                lane_trigger = idm_curves.get("lane_trigger", idm_curves["trigger"][None, :])
                mod_depths *= 1.0 + lane_trigger * idm_intensity * (
                    operator_glitch[:, None] * 0.9
                )
            visual_mod_scale = 1.0 + visual_morph_depth * (
                motion * motion_response * 0.12
                + edge * visual_drive * 0.12
                + spike * visual_drive * 0.22
                + entropy * visual_evolution * 0.1
            )
            mod_depths *= np.clip(visual_mod_scale, 0.35, 2.5)
            feedback = feedback_curve

            operators = [None] * self._operator_count
            for op_index in range(self._operator_count - 1, -1, -1):
                modulation = np.zeros(frames, dtype=np.float64)
                for mod_index, target in enumerate(algorithm_targets):
                    if target == op_index and operators[mod_index] is not None:
                        modulation += operators[mod_index] * mod_depths[mod_index]
                if op_index == feedback_operator:
                    feedback_depth = feedback * (2.0 + 10.0 * feedback)
                    modulation += voice_feedback[voice, op_index] * feedback_depth * (
                        1.0 + spike * visual_morph_depth
                    )
                operators[op_index] = np.sin(phases[op_index] + modulation)
            operators = np.vstack(operators)

            carrier_mask = np.asarray(
                [1.0 if target < 0 else 0.0 for target in algorithm_targets],
                dtype=np.float64,
            )[:, None]
            carrier_gains = carrier_amplitude * carrier_mask
            carrier_weights = carrier_mask[:, 0] * operator_enabled * np.mean(
                carrier_amplitude,
                axis=1,
            )
            carrier_weight_sum = max(1e-9, float(np.sum(np.abs(carrier_weights))))
            carrier_pan = float(np.sum(operator_pans * np.abs(carrier_weights)) / carrier_weight_sum)
            left_gain, right_gain = self._pan_gains(
                np.clip(voice_pan + carrier_pan * 0.5, -1.0, 1.0)
            )
            sample = np.sum(operators * carrier_gains, axis=0)
            normalization = np.maximum(
                1.0,
                np.sqrt(np.sum(carrier_gains * carrier_gains, axis=0)),
            )
            sample = (
                sample
                / normalization
                * frame_gain
                * ((1.0 - midi_velocity_depth * 0.35) + voice_velocity * midi_velocity_depth * 0.35)
                * master_gain
                / voice_norm
            )
            block[:, 0] += (sample * left_gain).astype(np.float32)
            block[:, 1] += (sample * right_gain).astype(np.float32)
            voice_feedback[voice] = operators[:, -1]
            voice_phase[voice] = np.mod(phases[:, -1] + phase_steps[:, -1], 2.0 * math.pi)

        return block

    def _render_operator_envelope_block(
        self,
        frames,
        params,
        voice,
        op_index,
        env_levels,
        env_stages,
        gate_on=True,
        key_rate=0.0,
    ):
        op = op_index + 1
        rates = [
            max(0.0, min(99.0, float(params.get(f"operator_eg_rate_{op}_{stage}", 50.0))))
            for stage in range(1, 5)
        ]
        targets = [
            max(0.0, min(1.0, float(params.get(f"operator_eg_level_{op}_{stage}", 0.0))))
            for stage in range(1, 5)
        ]
        level = max(0.0, min(1.0, float(env_levels[voice, op_index])))
        stage = int(env_stages[voice, op_index])
        if gate_on and stage == 5:
            stage = 0
            level = targets[3]
        elif not gate_on and stage != 5:
            stage = 4

        if gate_on and stage == 3:
            env_levels[voice, op_index] = targets[2]
            env_stages[voice, op_index] = 3
            return np.full(frames, targets[2], dtype=np.float64)
        if not gate_on and stage == 5:
            env_levels[voice, op_index] = 0.0
            env_stages[voice, op_index] = 5
            return np.zeros(frames, dtype=np.float64)

        curve = np.empty(frames, dtype=np.float64)
        rate_boost = 1.0 + max(0.0, min(1.0, float(key_rate))) * 1.8
        for index in range(frames):
            if stage == 0:
                level, done = self._tx802_step_toward(level, targets[0], rates[0] * rate_boost)
                if done:
                    stage = 1
            elif stage == 1:
                level, done = self._tx802_step_toward(level, targets[1], rates[1] * rate_boost)
                if done:
                    stage = 2
            elif stage == 2:
                level, done = self._tx802_step_toward(level, targets[2], rates[2] * rate_boost)
                if done:
                    stage = 3
            elif stage == 3:
                level = targets[2] if gate_on else level
                if not gate_on:
                    stage = 4
            elif stage == 4:
                level, done = self._tx802_step_toward(level, targets[3], rates[3] * rate_boost)
                if done and level <= 1e-5:
                    stage = 5
                    level = 0.0
            else:
                level = 0.0 if not gate_on else targets[3]
                stage = 5 if not gate_on else 0
            curve[index] = level
        env_levels[voice, op_index] = level
        env_stages[voice, op_index] = stage
        return curve

    def _tx802_step_toward(self, value, target, rate):
        rate = max(0.0, min(99.0, float(rate)))
        seconds = 0.003 + (1.0 - rate / 99.0) ** 2.35 * 4.5
        step = 1.0 / max(1.0, seconds * self.sample_rate)
        if value < target:
            value = min(target, value + step)
        else:
            value = max(target, value - step)
        return value, abs(value - target) <= 1e-6

    @staticmethod
    def _tx802_carrier_amplitude(level):
        level = np.clip(level, 0.0, 1.5)
        return np.power(level, 1.55)

    @staticmethod
    def _tx802_modulation_depth(level):
        level = np.clip(level, 0.0, 1.5)
        return 18.0 * np.power(level, 2.15)

    @staticmethod
    def _hash01(value):
        return np.mod(np.sin(value * 12.9898 + 78.233) * 43758.5453, 1.0)

    def _idm_event_curves(self, frames, render_time, params, visual):
        sample_times = float(render_time) + np.arange(frames, dtype=np.float64) / self.sample_rate
        bpm = max(40.0, min(260.0, float(params.get("idm_bpm", 172.0))))
        grid_rate = max(4.0, min(64.0, float(params.get("idm_grid_rate", 16.0))))
        cells_per_second = max(0.1, bpm / 60.0 * (grid_rate / 4.0))
        motion = np.clip(visual.get("motion", np.zeros(frames)), 0.0, 1.0)
        brightness = np.clip(visual.get("brightness", np.full(frames, 0.5)), 0.0, 1.0)
        edge = np.clip(visual.get("edge_density", np.zeros(frames)), 0.0, 1.0)
        spike = np.clip(visual.get("glitch_spike", np.zeros(frames)), 0.0, 1.0)
        entropy = np.clip(visual.get("entropy", np.zeros(frames)), 0.0, 1.0)
        red = np.clip(visual.get("red_energy", np.full(frames, 0.5)), 0.0, 1.0)
        blue = np.clip(visual.get("blue_energy", np.full(frames, 0.5)), 0.0, 1.0)
        micro = max(0.0, min(1.0, float(params.get("idm_microtiming", 0.0))))
        chaos = max(0.0, min(1.0, float(params.get("idm_chaos", 0.0))))
        phrase_depth = max(0.0, min(1.0, float(params.get("idm_phrase_depth", 0.0))))
        poly_depth = max(0.0, min(1.0, float(params.get("idm_polyrhythm_depth", 0.0))))
        scene_depth = max(0.0, min(1.0, float(params.get("idm_scene_reactivity", 0.0))))
        density_drift = max(0.0, min(1.0, float(params.get("idm_density_drift", 0.0))))
        beats = sample_times * bpm / 60.0
        phrase = np.floor(beats / 16.0)
        bar = np.floor(beats / 4.0)
        phrase_hash = self._hash01(phrase * 0.731 + 41.0)
        bar_hash = self._hash01(bar * 1.917 + 13.0)
        phrase_swing = (phrase_hash - 0.5) * phrase_depth
        density_curve = 1.0 + density_drift * (
            np.sin(2.0 * math.pi * beats / 11.0 + phrase_hash * 6.28318) * 0.22
            + (bar_hash - 0.5) * 0.45
        )
        scene_delta = np.concatenate(([0.0], np.abs(np.diff(brightness)) + np.abs(np.diff(motion)) + np.abs(np.diff(red - blue))))
        scene_accent = np.clip(scene_delta * (4.0 + scene_depth * 10.0) + spike * scene_depth, 0.0, 1.0)
        phase = (
            sample_times * cells_per_second * np.clip(density_curve, 0.35, 1.85)
            + (motion - 0.5) * micro * (0.22 + chaos * 0.42)
            + phrase_swing * 0.18
        )
        cell = np.floor(phase)
        cell_pos = np.mod(phase, 1.0)
        seed = float(params.get("idm_seed", 19.0))
        cell_hash = self._hash01(cell + seed + phrase_hash * 17.0)
        event_energy = np.clip(
            motion * 0.34
            + edge * 0.25
            + spike * 0.38
            + entropy * 0.24
            + np.abs(brightness - 0.5) * 0.18
            + scene_accent * 0.55,
            0.0,
            1.0,
        )
        probability = max(0.0, min(1.0, float(params.get("idm_probability", 0.75))))
        probability_curve = np.clip(
            probability * (0.68 + density_curve * 0.32)
            + event_energy * (0.2 + chaos * 0.24)
            + scene_accent * scene_depth * 0.4,
            0.0,
            1.0,
        )
        active = (cell_hash < probability_curve).astype(np.float64)
        trigger = np.exp(-cell_pos * (18.0 + event_energy * 48.0 + chaos * 18.0)) * active
        hold = 0.18 + 0.22 * (1.0 - event_energy)
        slice_env = np.where(
            cell_pos < hold,
            1.0,
            np.exp(-(cell_pos - hold) * (5.0 + event_energy * 18.0)),
        )
        stutter_depth = max(0.0, min(1.0, float(params.get("idm_stutter_depth", 0.0))))
        stutter_rate = 2.0 + np.floor(event_energy * (6.0 + chaos * 10.0) + bar_hash * phrase_depth * 5.0)
        stutter_phase = np.mod(cell_pos * stutter_rate, 1.0)
        stutter_gate = np.where(stutter_phase < 0.52, 1.0, 0.14 + 0.18 * (1.0 - event_energy))
        retrigger = max(0.0, min(1.0, float(params.get("idm_retrigger_depth", 0.0))))
        amp_gate = (1.0 - retrigger) + retrigger * slice_env
        amp_gate *= (1.0 - stutter_depth) + stutter_depth * stutter_gate
        lane_trigger = np.empty((self._operator_count, frames), dtype=np.float64)
        lane_slice = np.empty((self._operator_count, frames), dtype=np.float64)
        lane_rates = np.asarray((1.0, 1.5, 2.0, 2.5, 3.0, 4.0), dtype=np.float64)
        for op in range(self._operator_count):
            lane_phase = (
                phase
                * (1.0 + poly_depth * (lane_rates[op] - 1.0) * 0.38)
                + op * (0.173 + phrase_depth * 0.037)
                + phrase_hash * (op + 1) * phrase_depth * 0.23
            )
            lane_cell = np.floor(lane_phase)
            lane_pos = np.mod(lane_phase, 1.0)
            lane_hash = self._hash01(lane_cell + seed + op * 23.0 + bar_hash * 11.0)
            euclid = np.mod(lane_cell.astype(np.int64) * (op * 2 + 3) + int(seed), 8 + op)
            euclid_gate = (euclid < (2 + (op % 4))).astype(np.float64)
            lane_prob = np.clip(probability_curve * (0.62 + 0.08 * op) + chaos * lane_hash * 0.18, 0.0, 1.0)
            lane_active = ((lane_hash < lane_prob) | (euclid_gate > 0.0) | (scene_accent > 0.72)).astype(np.float64)
            lane_trigger[op] = np.exp(-lane_pos * (20.0 + event_energy * 36.0 + op * 2.0)) * lane_active
            lane_hold = np.clip(hold * (0.72 + 0.1 * op + lane_hash * chaos * 0.4), 0.035, 0.65)
            lane_slice[op] = np.where(
                lane_pos < lane_hold,
                1.0,
                np.exp(-(lane_pos - lane_hold) * (6.0 + event_energy * 20.0 + op)),
            )
        return {
            "cell": cell,
            "cell_pos": cell_pos,
            "cell_hash": cell_hash,
            "trigger": trigger,
            "slice": slice_env,
            "lane_trigger": lane_trigger,
            "lane_slice": lane_slice,
            "amp_gate": np.clip(amp_gate, 0.0, 1.2),
            "event_energy": event_energy,
            "scene_accent": scene_accent,
            "event_strength": float(np.mean((event_energy + scene_accent) * active)),
            "block_hash": float(self._hash01(np.floor(sample_times[0] * cells_per_second) + seed * 3.17 + phrase_hash[0] * 19.0)),
        }

    def _apply_idm_ratio_snaps(self, ratios, params, idm_curves, operator_ratio_snap, intensity):
        depth = intensity * max(0.0, min(1.0, float(params.get("idm_ratio_snap_depth", 0.0))))
        if depth <= 0.0:
            return ratios
        table_name = "metallic" if float(np.mean(idm_curves["event_energy"])) > 0.45 else "broken"
        table = IDM_RATIO_SNAP_TABLES[table_name]
        snapped = ratios.copy()
        cells = idm_curves["cell"]
        trigger = np.clip(idm_curves["trigger"] * 1.8 + idm_curves["event_energy"] * 0.35, 0.0, 1.0)
        for op in range(self._operator_count):
            op_depth = depth * max(0.0, min(1.0, float(operator_ratio_snap[op])))
            if op_depth <= 0.0:
                continue
            indices = np.mod(cells.astype(np.int64) + op * 3, len(table))
            target = table[indices]
            lane_trigger = idm_curves.get("lane_trigger", np.zeros_like(ratios))[op]
            mix = np.clip(op_depth * (trigger * 0.35 + lane_trigger * 1.15), 0.0, 1.0)
            snapped[op] = snapped[op] * (1.0 - mix) + target * mix
        return np.clip(snapped, 0.125, 32.0)

    def _apply_idm_envelope_slicing(self, env_curves, params, idm_curves, carrier_flags, intensity):
        depth = intensity * max(0.0, min(1.0, float(params.get("idm_env_slice_depth", 0.0))))
        if depth <= 0.0:
            return env_curves
        carrier_depth = max(0.0, min(1.0, float(params.get("idm_carrier_chop_depth", 0.0))))
        mod_depth = max(0.0, min(1.0, float(params.get("idm_modulator_chop_depth", 0.0))))
        sliced = env_curves.copy()
        for op in range(self._operator_count):
            op_depth = depth * (carrier_depth if carrier_flags[op] else mod_depth)
            if op_depth <= 0.0:
                continue
            lane_slice = idm_curves.get("lane_slice", np.tile(idm_curves["slice"], (self._operator_count, 1)))[op]
            gate = (1.0 - op_depth) + op_depth * lane_slice
            sliced[op] *= np.clip(gate, 0.0, 1.15)
        return sliced

    def _apply_idm_operator_bursts(
        self,
        audible,
        idm_curves,
        operator_glitch,
        operator_burst,
        operator_level_jitter,
        intensity,
    ):
        shaped = audible.copy()
        cells = idm_curves["cell"]
        trigger = idm_curves["trigger"]
        event_energy = idm_curves["event_energy"]
        lane_trigger = idm_curves.get("lane_trigger")
        for op in range(self._operator_count):
            glitch = intensity * max(0.0, min(1.0, float(operator_glitch[op])))
            jitter = intensity * max(0.0, min(1.0, float(operator_level_jitter[op])))
            burst = intensity * max(0.0, min(1.0, float(operator_burst[op])))
            if glitch <= 0.0 and jitter <= 0.0 and burst <= 0.0:
                continue
            noise = self._hash01(cells + 11.0 + op * 17.0) * 2.0 - 1.0
            level_scale = 1.0 + noise * jitter * 0.55
            burst_gate = (self._hash01(cells + 29.0 + op * 7.0) < (burst * (0.25 + event_energy * 0.65))).astype(np.float64)
            op_trigger = trigger if lane_trigger is None else np.maximum(trigger * 0.35, lane_trigger[op])
            level_scale += burst_gate * op_trigger * glitch * (1.4 + event_energy * 1.8)
            shaped[op] *= np.clip(level_scale, 0.0, 3.0)
        return shaped

    def _apply_idm_dirt(self, audio, params):
        intensity = max(0.0, min(1.0, float(params.get("idm_intensity", 0.0))))
        if intensity <= 0.0 or audio.size == 0:
            return audio
        output = np.asarray(audio, dtype=np.float32).copy()
        fold = intensity * max(0.0, min(1.0, float(params.get("idm_wavefold_depth", 0.0))))
        if fold > 0.0:
            driven = output * np.float32(1.0 + fold * 9.0)
            folded = np.sin(driven).astype(np.float32)
            output = output * np.float32(1.0 - fold * 0.7) + folded * np.float32(fold * 0.7)
        rate_fold = intensity * max(0.0, min(1.0, float(params.get("idm_sample_rate_fold", 0.0))))
        if rate_fold > 0.0:
            hold = int(1 + round(rate_fold * 36.0))
            if hold > 1:
                indices = (np.arange(output.shape[0]) // hold) * hold
                output = output[indices]
        bit_depth = intensity * max(0.0, min(1.0, float(params.get("idm_bitcrush_depth", 0.0))))
        if bit_depth > 0.0:
            levels = max(8.0, 32768.0 / (1.0 + bit_depth * 96.0))
            output = np.round(output * levels) / levels
        np.tanh(output * np.float32(1.0 + intensity * 0.35), out=output)
        return np.clip(output, -1.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _lfo_wave(phase, wave):
        wave = str(wave or "sine")
        wrapped = np.mod(phase, 2.0 * math.pi)
        norm = wrapped / (2.0 * math.pi)
        if wave == "triangle":
            return 4.0 * np.abs(norm - 0.5) - 1.0
        if wave == "saw_up":
            return norm * 2.0 - 1.0
        if wave == "saw_down":
            return 1.0 - norm * 2.0
        if wave == "square":
            return np.where(norm < 0.5, 1.0, -1.0)
        if wave == "sample_hold":
            held = np.floor(norm * 16.0)
            return np.sin(held * 12.9898 + 78.233)
        return np.sin(phase)

    @staticmethod
    def _pan_gains(pan):
        angle = (min(1.0, max(-1.0, float(pan))) + 1.0) * math.pi * 0.25
        return math.cos(angle), math.sin(angle)


class MediaAudioController:
    """Owns asynchronous analysis and the live sounddevice output stream."""

    def __init__(self, sample_rate=48000, block_size=512):
        self.engine = MediaAudioEngine(sample_rate, block_size)
        self.audio_device = AudioDevice(
            self.engine.render, sample_rate, block_size, latency="high"
        )
        self._status = "No media loaded."
        self._error = ""
        self._loading = False
        self._ir_loading = False
        self._load_generation = 0
        self._ir_load_generation = 0
        self._ir_status = "No IR loaded."
        self._sysex_status = "No TX802/DX7 SysEx imported."
        self._lock = threading.RLock()
        self._thread = None
        self.sequencer = MediaAudioSequencer(
            self.engine.PARAM_LIMITS,
            self.engine.set_param,
        )

    def load_async(self, path):
        path = os.path.abspath(path)
        with self._lock:
            self._load_generation += 1
            generation = self._load_generation
            self._loading = True
            self._error = ""
            self._status = f"Analyzing {os.path.basename(path)}..."

        def worker():
            try:
                analysis = analyze_media(path)
                with self._lock:
                    if generation != self._load_generation:
                        return
                    self.engine.set_analysis(analysis)
                    self._status = (
                        f"Ready: {os.path.basename(path)} "
                        f"({analysis.duration:.2f}s, "
                        f"{len(analysis.timestamps)} analysis frames)"
                    )
                    self._error = ""
            except Exception as exc:
                logging.exception("Media Audio analysis failed")
                with self._lock:
                    if generation != self._load_generation:
                        return
                    self._error = str(exc)
                    self._status = f"Load failed: {exc}"
            finally:
                with self._lock:
                    if generation == self._load_generation:
                        self._loading = False

        self._thread = threading.Thread(
            target=worker,
            name="MegaGlitchMediaAudioAnalysis",
            daemon=True,
        )
        self._thread.start()

    def start(self, restart=False):
        if not self.engine.play(restart=restart):
            with self._lock:
                self._status = "Load a media file before starting audio."
            return False
        if not self.audio_device.start():
            self.engine.stop()
            with self._lock:
                error = self.audio_device.last_error or "unknown output error"
                self._error = error
                self._status = f"Audio output failed: {error}"
            return False
        with self._lock:
            self._status = (
                f"Media Audio playing: {self.engine.snapshot()['source_mode']}."
            )
            self._error = ""
        return True

    def stop(self):
        self.engine.stop()
        self.audio_device.stop()
        with self._lock:
            self._status = "Live media audio stopped."

    def set_param(self, name, value):
        self.engine.set_param(name, value)

    def set_param_mod_source(self, name, source):
        self.engine.set_param_mod_source(name, source)

    def set_source_mode(self, mode):
        self.engine.set_source_mode(mode)
        with self._lock:
            self._status = f"Source mode: {mode}."

    def set_live_metrics(self, metrics):
        self.engine.set_live_metrics(metrics)

    def handle_midi_performance(self, msg):
        return self.engine.handle_midi_performance(msg)

    def tick_sequencer(self, metrics=None, now=None):
        return self.sequencer.tick(metrics or {}, now=now)

    def mutate_sequence_from_visual(self, metrics=None, strength=None):
        self.sequencer.mutate_from_visual(metrics or self.engine._live_metrics, strength=strength)

    def handle_midi(self, msg, metrics=None):
        if self.handle_midi_performance(msg):
            return True
        return self.sequencer.handle_midi(msg, metrics=metrics)

    def apply_preset(self, name):
        return self.engine.apply_preset(name)

    def load_tx802_sysex_async(self, path):
        path = os.path.abspath(path)
        with self._lock:
            self._sysex_status = f"Importing SysEx: {os.path.basename(path)}..."

        def worker():
            try:
                patches = parse_tx802_sysex(path)
                added = self.engine.add_imported_patches(patches)
                first_name = next(iter(added), "")
                if first_name:
                    self.engine.apply_preset(first_name)
                with self._lock:
                    self._sysex_status = (
                        f"Imported {len(added)} TX802/DX7 patch"
                        f"{'' if len(added) == 1 else 'es'} from {os.path.basename(path)}."
                    )
                    self._error = ""
            except Exception as exc:
                logging.exception("Media Audio TX802/DX7 SysEx import failed")
                with self._lock:
                    self._error = str(exc)
                    self._sysex_status = f"SysEx import failed: {exc}"

        thread = threading.Thread(
            target=worker,
            name="MegaGlitchMediaAudioSysexImport",
            daemon=True,
        )
        thread.start()
        return thread

    def load_ir_async(
        self,
        path,
        max_seconds=4.0,
        normalize=True,
    ):
        path = os.path.abspath(path)
        with self._lock:
            self._ir_load_generation += 1
            generation = self._ir_load_generation
            self._ir_loading = True
            self._ir_status = f"Loading IR: {os.path.basename(path)}..."

        def worker():
            try:
                impulse = load_impulse_response(
                    path,
                    self.engine.sample_rate,
                    max_seconds=max_seconds,
                    normalize=normalize,
                )
                convolver = PartitionedStereoConvolver(
                    impulse, self.engine.block_size
                )
                with self._lock:
                    if generation != self._ir_load_generation:
                        return
                    self.engine.set_convolver(convolver, path)
                    self._ir_status = (
                        f"IR active: {os.path.basename(path)} "
                        f"({len(impulse) / self.engine.sample_rate:.2f}s, "
                        f"{convolver.partition_count} partitions)"
                    )
                    self._error = ""
            except Exception as exc:
                logging.exception("Media Audio IR loading failed")
                with self._lock:
                    if generation != self._ir_load_generation:
                        return
                    self._error = str(exc)
                    self._ir_status = f"IR load failed: {exc}"
            finally:
                with self._lock:
                    if generation == self._ir_load_generation:
                        self._ir_loading = False

        thread = threading.Thread(
            target=worker,
            name="MegaGlitchMediaAudioIRLoad",
            daemon=True,
        )
        thread.start()
        return thread

    def clear_ir(self):
        with self._lock:
            self._ir_load_generation += 1
            self._ir_loading = False
            self._ir_status = "No IR loaded."
        self.engine.clear_convolver()

    def configure_cdp_graph(self, **kwargs):
        self.engine.cdp_graph.configure(**kwargs)

    def add_cdp_graph_node(self, effect):
        return self.engine.cdp_graph.add_node(effect)

    def remove_cdp_graph_node(self, index):
        self.engine.cdp_graph.remove_node(index)

    def move_cdp_graph_node(self, index, delta):
        self.engine.cdp_graph.move_node(index, delta)

    def update_cdp_graph_node(self, index, **updates):
        self.engine.cdp_graph.update_node(index, **updates)

    def export_preset_state(self):
        state = self.engine.export_preset_state()
        with self._lock:
            state.update(
                status=self._status,
                ir_status=self._ir_status,
                sysex_status=self._sysex_status,
            )
        return state

    def import_preset_state(self, state, reload_media=True, reload_ir=False):
        if not isinstance(state, dict):
            return
        self.engine.import_preset_state(state)
        source_path = str(state.get("source_path") or "")
        ir_path = str(state.get("ir_path") or "")
        if reload_media and source_path and os.path.exists(source_path):
            self.load_async(source_path)
        if reload_ir and ir_path and os.path.exists(ir_path):
            self.load_ir_async(ir_path)
        with self._lock:
            self._status = "Media Audio state restored from preset."
            if ir_path and not os.path.exists(ir_path):
                self._ir_status = f"Saved IR not found: {os.path.basename(ir_path)}"

    def snapshot(self):
        engine_state = self.engine.snapshot()
        with self._lock:
            engine_state.update(
                status=self._status,
                error=self._error,
                loading=self._loading,
                ir_loading=self._ir_loading,
                ir_status=self._ir_status,
                sysex_status=self._sysex_status,
                output_running=self.audio_device.running,
                sequencer=self.sequencer.snapshot(),
            )
        return engine_state

    def destroy(self):
        with self._lock:
            self._load_generation += 1
            self._ir_load_generation += 1
        self.engine.stop()
        self.engine.cdp_graph.shutdown()
        self.audio_device.destroy()
