"""Pitch, tuning, and scale helpers."""

from __future__ import annotations

import math


SCALES = {
    "chromatic": tuple(range(12)),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "diminished": (0, 2, 3, 5, 6, 8, 9, 11),
}


def midi_to_frequency(note: float, tuning_a4: float = 440.0) -> float:
    return float(tuning_a4) * (2.0 ** ((float(note) - 69.0) / 12.0))


def frequency_to_midi(frequency: float, tuning_a4: float = 440.0) -> float:
    frequency = max(1e-6, float(frequency))
    return 69.0 + 12.0 * math.log2(frequency / float(tuning_a4))


def quantize_midi(note: float, root: int = 36, mode: str = "minor_pentatonic", custom=None) -> int:
    intervals = tuple(custom or ()) if mode == "custom" else SCALES.get(mode, SCALES["chromatic"])
    if not intervals:
        intervals = SCALES["chromatic"]
    root = int(root)
    note = float(note)
    octave_guess = math.floor((note - root) / 12.0)
    candidates = []
    for octave in range(octave_guess - 1, octave_guess + 2):
        candidates.extend(root + octave * 12 + interval for interval in intervals)
    return int(min(candidates, key=lambda candidate: (abs(candidate - note), candidate)))


def quantize_frequency(frequency: float, root: int = 36, mode: str = "minor_pentatonic", custom=None) -> float:
    return midi_to_frequency(quantize_midi(frequency_to_midi(frequency), root, mode, custom))
