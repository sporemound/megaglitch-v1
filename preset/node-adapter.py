"""Headless full-canvas adapter around the original MegaGlitchNode CPU stages.

No GUI is started, no source code or environment is changed, and every output
pixel comes from the same destructive stage chain.  Source RGB8 dimensions
are preserved.  The caller owns export size, frame rate and knob timeline.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import deque
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image

NODE_ROOT = Path(os.environ.get("MEGAGLITCH_NODE_ROOT", str(Path(__file__).resolve().parent.parent / "app"))).expanduser().resolve()
if not (NODE_ROOT / "effects_registry.py").is_file():
    raise FileNotFoundError(f"MegaGlitch source was not found at {NODE_ROOT}; set --node-root or MEGAGLITCH_NODE_ROOT.")
if str(NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NODE_ROOT))

import effects_registry as registry
import glitch_effects as ge
import gpu_ops
from megaglitch.determinism import use_numpy_rng

gpu_ops.set_use_cuda(False)
# Pure seeded noise fields repeat between control gestures. A bounded cache
# preserves the original stage's values while avoiding repeated large blurs.
ge._smooth_noise = lru_cache(maxsize=6)(ge._smooth_noise)

CONTROL_RANGES = {
    "edge_mix": (0.90, 1.0),
    "contour_gain": (1.2, 3.1),
    "warp_amplitude": (12.0, 88.0),
    "wave_frequency": (1.2, 7.2),
    "wave_speed": (0.20, 2.4),
    "noise_warp": (0.03, 0.65),
    "noise_scale": (80.0, 220.0),
    "unfolding_speed": (0.30, 5.0),
    "euclidean_parameter": (1.0, 9.0),
    "rgb_split": (8.0, 58.0),
    "feedback": (0.68, 0.91),
    "feedback_gain": (1.04, 1.18),
    "feedback_warp": (4.0, 44.0),
    "feedback_rotation": (-0.70, 0.70),
    "trailing": (0.02, 0.35),
    "strip_shift": (0.0, 105.0),
    "strip_width": (8.0, 58.0),
    "tile_strength": (0.0, 0.18),
    "tile_size": (16.0, 68.0),
    "line_shift": (0.0, 65.0),
    "hue_shift": (-70.0, 70.0),
}

default_controls = {
    "edge_mix": 1.0,
    "contour_gain": 2.25,
    "warp_amplitude": 34.0,
    "wave_frequency": 3.1,
    "wave_speed": 0.78,
    "noise_warp": 0.17,
    "noise_scale": 160.0,
    "unfolding_speed": 3.0,
    "euclidean_parameter": 4.5,
    "rgb_split": 24.0,
    "feedback": 0.84,
    "feedback_gain": 1.105,
    "feedback_warp": 19.0,
    "feedback_rotation": 0.20,
    "trailing": 0.08,
    "strip_shift": 16.0,
    "strip_width": 32.0,
    "tile_strength": 0.025,
    "tile_size": 40.0,
    "line_shift": 8.0,
    "hue_shift": 0.0,
}

STAGE_PROVENANCE = {
    "node_source": str(NODE_ROOT),
    "registry": str(Path(registry.__file__).resolve()),
    "registry_sha256": hashlib.sha256(Path(registry.__file__).read_bytes()).hexdigest(),
    "glitch_effects_sha256": hashlib.sha256(Path(ge.__file__).read_bytes()).hexdigest(),
    "method": "Original Node individual CPU stages, driven headlessly by a recorded timeline",
    "source_files_modified": False,
    "protected_regions": False,
    "stages": [
        "glitch_effects.apply_sharpen + effects_registry.apply_node_effect('Edges / Emboss')",
        "effects_registry.apply_node_effect('Unfold / Euclid / Random')",
        "effects_registry.apply_node_effect('Ripple')",
        "glitch_effects.noise_field_warp_np",
        "effects_registry.apply_node_effect('FrameDrop / RGB Split')",
        "glitch_effects.ripple_np + glitch_effects.rotate_in_frame_np on recursive history",
        "effects_registry.apply_node_effect('Feedback / Trail')",
        "effects_registry.apply_node_effect('Temporal Slicing')",
        "effects_registry.apply_node_effect('Data Moshing')",
        "effects_registry.apply_node_effect('Line Glitch')",
        "effects_registry.apply_node_effect('Color Adjust')",
    ],
}


class NodeAdapter:
    """One independent destructive stream with resettable feedback history."""
    CONTROL_RANGES = CONTROL_RANGES
    default_controls = default_controls
    provenance = STAGE_PROVENANCE

    def __init__(self, source_rgb: np.ndarray, seed: int = 1961):
        if not isinstance(source_rgb, np.ndarray) or source_rgb.dtype != np.uint8:
            raise TypeError("source_rgb must be a uint8 RGB numpy array")
        if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
            raise ValueError("source_rgb must have shape H x W x 3")
        self.source = np.ascontiguousarray(source_rgb.copy())
        self.seed = int(seed)
        # Native Node edge extraction keeps the entire poster, including all text.
        edge_source = cv2.GaussianBlur(self.source, (0, 0), 0.85)
        self.contours = self._node("Edges / Emboss", ge.apply_sharpen(edge_source.astype(np.float32), 0.85, 1),
                                   {"edge_mix": 1.0, "emboss_strength": 0.0})
        self.reset()

    def reset(self):
        self.previous = None
        self.feedback_image = None
        self.trail_image = None
        self.history = deque(maxlen=12)
        self.frame_index = 0

    @staticmethod
    def _rgb(arr):
        if isinstance(arr, Image.Image):
            arr = np.array(arr.convert("RGB"))
        return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _node(self, label, arr, params, *, feedback=None, trail=None, rng=None, python_rng=None):
        settings = dict(params)
        if rng is not None:
            settings["_preview_numpy_rng"] = rng
        if python_rng is not None:
            settings["_preview_python_rng"] = python_rng
        return self._rgb(registry.apply_node_effect(
            label, arr, settings, getattr(self, "previous", None), feedback, trail,
            list(getattr(self, "history", ())), None)[0])

    def process(self, t: float, controls: dict | None = None, audio_level: float = 0.0) -> np.ndarray:
        c = dict(default_controls)
        c.update(controls or {})
        for name, limits in CONTROL_RANGES.items():
            c[name] = float(np.clip(c[name], *limits))
        audio = float(np.clip(audio_level, 0.0, 1.0))
        phase = float(t) * c["wave_speed"]
        # Random strip/tile arrangements hold for a few frames, like held controls.
        epoch = int(max(0.0, float(t)) * 5.0)
        rng = np.random.default_rng(self.seed + epoch * 917)
        py_rng = random.Random(self.seed + epoch * 739)
        with use_numpy_rng(rng):
            contour = np.clip(self.contours.astype(np.float32) * c["contour_gain"], 0, 255).astype(np.uint8)
            src = cv2.addWeighted(self.source, 1.0 - c["edge_mix"], contour, c["edge_mix"], 0)
            out = self._node("Unfold / Euclid / Random", src, {
                "unfolding_speed": c["unfolding_speed"],
                "euclidean_parameter": c["euclidean_parameter"],
                "randomize_intensity": 0.0,
            })
            out = self._node("Ripple", out, {
                "ripple_amp": c["warp_amplitude"] * (0.92 + 0.16 * audio),
                "ripple_freq": c["wave_frequency"], "noise_warp_time": phase,
            })
            out = self._rgb(ge.noise_field_warp_np(out, c["noise_warp"], c["noise_scale"], phase * 0.14))
            out = self._node("FrameDrop / RGB Split", out, {
                "rgb_split_distance": int(round(c["rgb_split"] * (0.92 + 0.22 * audio))),
                "frame_drop_chance": 0,
            })
            # The recursion is moved using the same original Node warp primitives.
            fb = self.feedback_image
            if fb is not None:
                fb = self._rgb(ge.ripple_np(fb, c["feedback_warp"], 1.5 + c["wave_frequency"] * 0.18, phase * 0.71))
                fb = self._rgb(ge.rotate_in_frame_np(fb, c["feedback_rotation"], overscan=0.0))
            out = self._node("Feedback / Trail", out, {
                "feedback_intensity": c["feedback"], "trailing_intensity": c["trailing"],
            }, feedback=fb, trail=self.trail_image)
            # Native stage history is followed by the user's destructive strip/tile controls.
            if c["strip_shift"] > 0.5:
                out = self._node("Temporal Slicing", out, {
                    "slice_width": int(round(c["strip_width"])),
                    "time_offset_range": int(round(c["strip_shift"] * (0.70 + 0.60 * audio))),
                    "slice_direction": "vertical" if int(t / 3.0) % 2 == 0 else "horizontal",
                }, rng=rng, python_rng=py_rng)
            if c["tile_strength"] > 0.002:
                # Legacy block helpers use Python's random module; seed and restore it.
                prior_random = random.getstate()
                random.seed(self.seed + epoch * 127)
                try:
                    out = self._node("Data Moshing", out, {
                        "block_size": int(round(c["tile_size"])),
                        "block_displacement": c["tile_strength"],
                        "displacement_direction": "random",
                        "block_duplication": c["tile_strength"] * 0.25,
                        "duplication_pattern": "horizontal",
                    }, rng=rng, python_rng=py_rng)
                finally:
                    random.setstate(prior_random)
            if c["line_shift"] > 0.5:
                out = self._node("Line Glitch", out, {
                    "line_count": 46, "max_offset": int(round(c["line_shift"])),
                    "glitch_frequency": 80.0, "glitch_intensity": 100.0,
                }, rng=rng, python_rng=py_rng)
            # Gain offsets the native convex feedback decay while retaining deep blacks.
            out = np.clip(out.astype(np.float32) * c["feedback_gain"], 0, 255).astype(np.uint8)
            if abs(c["hue_shift"]) > 0.02:
                out = self._rgb(ge.apply_hsv_adjust(out, hue_shift=c["hue_shift"], saturation=1.55))
        self.previous = out.copy()
        self.feedback_image = out.copy()
        self.trail_image = out.copy()
        self.history.append(out.copy())
        self.frame_index += 1
        if out.shape != self.source.shape:
            raise RuntimeError("Node stage changed the source dimensions")
        return out
