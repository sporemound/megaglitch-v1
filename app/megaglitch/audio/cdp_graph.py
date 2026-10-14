"""Nonblocking CDP-style node graph processing for Media Audio."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import queue
import shlex
import subprocess
import tempfile
import threading
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter


DEFAULT_CDP_PROGRAMS_PATH = os.environ.get("MEGAGLITCH_CDP_PROGRAMS", "")


CDP_EFFECT_SPECS = {
    "Blur Average": {"category": "blur", "mode": "avrg", "program": "blur.cmd", "worker": False, "params": {"amount": 4.0, "mix": 0.78}},
    "Combine Crossfade": {"category": "combine", "mode": "cross", "program": "combine.cmd", "worker": False, "params": {"delay_ms": 38.0, "feedback": 0.42, "mix": 0.72}},
    "Distort Average": {"category": "distort", "mode": "average", "program": "distort.cmd", "worker": False, "params": {"cycles": 4.0, "drive": 0.72, "mix": 0.82}},
    "Env Tremolo": {"category": "env", "mode": "tremolo", "program": "env.cmd", "worker": False, "params": {"rate": 11.0, "depth": 0.78, "mix": 0.82}},
    "Filter Fixed": {"category": "filter", "mode": "fixed", "program": "filter.cmd", "worker": False, "params": {"freq": 1200.0, "q": 1.4, "gain": 0.0, "mix": 0.8}},
    "Formants Shift": {"category": "formants", "mode": "shift", "program": "formants.cmd", "worker": False, "params": {"freq": 900.0, "q": 2.0, "gain": 0.72, "mix": 0.78}},
    "Grain Timewarp": {"category": "grain", "mode": "timewarp", "program": "grain.cmd", "worker": False, "params": {"grain_ms": 22.0, "jitter": 0.72, "mix": 0.82}},
    "HFPerm Shuffle": {"category": "hfperm", "mode": "perm", "program": "hfperm.cmd", "worker": False, "params": {"amount": 0.82, "mix": 0.78}},
    "Hilite Filter": {"category": "hilite", "mode": "filter", "program": "hilite.cmd", "worker": False, "params": {"freq": 2400.0, "q": 1.5, "gain": 0.82, "mix": 0.78}},
    "Housekeep Clean": {"category": "houskeep", "mode": "clean", "program": "houskeep.cmd", "worker": False, "params": {"gain": 1.0, "mix": 0.0}},
    "Mix Balance": {"category": "mix", "mode": "balance", "program": "mix.cmd", "worker": False, "params": {"gain": 1.1, "delay_ms": 24.0, "feedback": 0.22, "mix": 0.62}},
    "Modify Loudness": {"category": "modify", "mode": "loudness", "program": "modify.cmd", "worker": False, "params": {"gain": 1.8, "shape": 0.58, "mix": 0.82}},
    "Morph Spectral": {"category": "morph", "mode": "spectral", "program": "morph.cmd", "params": {"amount": 0.48, "mix": 0.55}},
    "Pitch Shift": {"category": "pitch", "mode": "shift", "program": "pitch.cmd", "params": {"semitones": 7.0, "mix": 0.55}},
    "PitchInfo Probe": {"category": "pitchinfo", "mode": "info", "program": "pitchinfo.cmd", "params": {"mix": 0.0}},
    "PView Probe": {"category": "pview", "mode": "view", "program": "pview.cmd", "params": {"mix": 0.0}},
    "PVOC Freeze": {"category": "pvoc", "mode": "freeze", "program": "pvoc.cmd", "params": {"freeze": 0.6, "mix": 0.55}},
    "Repitch Warp": {"category": "repitch", "mode": "warp", "program": "repitch.cmd", "params": {"semitones": -5.0, "mix": 0.55}},
    "SFConvert Format": {"category": "sfconvert", "mode": "format", "program": "sfconvert.cmd", "params": {"mix": 0.0}},
    "SFEdit Trim": {"category": "sfedit", "mode": "trim", "program": "sfedit.cmd", "params": {"amount": 0.12, "mix": 0.2}},
    "SFProps Probe": {"category": "sfprops", "mode": "props", "program": "sfprops.cmd", "params": {"mix": 0.0}},
    "SndInfo Probe": {"category": "sndinfo", "mode": "info", "program": "sndinfo.cmd", "params": {"mix": 0.0}},
    "Spec Blur": {"category": "spec", "mode": "blur", "program": "spec.cmd", "params": {"amount": 0.42, "mix": 0.6}},
    "SpecInfo Probe": {"category": "specinfo", "mode": "info", "program": "specinfo.cmd", "params": {"mix": 0.0}},
    "Stretch Time": {"category": "stretch", "mode": "time", "program": "stretch.cmd", "params": {"factor": 1.6, "mix": 0.55}},
    "Submix Fold": {"category": "submix", "mode": "fold", "program": "submix.cmd", "params": {"delay_ms": 63.0, "feedback": 0.45, "mix": 0.5}},
    "Visual Glitch - Motion Slicer": {"category": "visual_glitch", "mode": "motion_slicer", "program": "", "worker": False, "params": {"rate": 18.0, "depth": 0.78, "jitter": 0.42, "mix": 0.82}},
    "Visual Glitch - Edge Comb Shred": {"category": "visual_glitch", "mode": "edge_comb", "program": "", "worker": False, "params": {"delay_ms": 7.0, "feedback": 0.68, "tone": 0.55, "mix": 0.78}},
    "Visual Glitch - Color Bitcrush": {"category": "visual_glitch", "mode": "color_bitcrush", "program": "", "worker": False, "params": {"bits": 5.0, "fold": 0.36, "mix": 0.76}},
    "Visual Glitch - Brightness Gate": {"category": "visual_glitch", "mode": "brightness_gate", "program": "", "worker": False, "params": {"threshold": 0.48, "softness": 0.18, "mix": 0.72}},
    "Visual Glitch - Spike Buffer Repeat": {"category": "visual_glitch", "mode": "spike_repeat", "program": "", "worker": False, "params": {"slice_ms": 38.0, "repeats": 5.0, "mix": 0.84}},
    "Visual Glitch - Entropy Spectral Tear": {"category": "visual_glitch", "mode": "entropy_tear", "program": "", "worker": False, "params": {"amount": 0.62, "tilt": 0.48, "mix": 0.74}},
    "Visual Glitch - Center Pan Scanner": {"category": "visual_glitch", "mode": "center_pan", "program": "", "worker": False, "params": {"width": 0.9, "rate": 4.0, "mix": 0.68}},
    "Visual Glitch - RGB Ring Mod": {"category": "visual_glitch", "mode": "rgb_ringmod", "program": "", "worker": False, "params": {"rate": 36.0, "depth": 0.72, "mix": 0.8}},
    "Visual Glitch - Block Dropout Matrix": {"category": "visual_glitch", "mode": "block_dropout", "program": "", "worker": False, "params": {"blocks": 16.0, "drop": 0.42, "mix": 0.8}},
}

CDP_WRAPPER_PROGRAMS = tuple(sorted({spec["program"] for spec in CDP_EFFECT_SPECS.values() if spec.get("program")}))


@dataclass
class CDPGraphNode:
    id: str
    effect: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "effect": self.effect,
            "enabled": bool(self.enabled),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=str(data.get("id", "")),
            effect=str(data.get("effect", "Blur Average")),
            enabled=bool(data.get("enabled", True)),
            params=dict(data.get("params", {})),
        )


class CDPNodeGraph:
    """A deliberately linear graph with node metadata for UI editing."""

    def __init__(self, nodes=None):
        self.nodes: list[CDPGraphNode] = list(nodes or [])

    def add_node(self, effect):
        effect = str(effect or "Blur Average")
        if effect not in CDP_EFFECT_SPECS:
            effect = "Blur Average"
        node = CDPGraphNode(
            id=f"cdp_{len(self.nodes) + 1:02d}",
            effect=effect,
            enabled=True,
            params=dict(CDP_EFFECT_SPECS[effect]["params"]),
        )
        self.nodes.append(node)
        self._renumber()
        return node

    def remove_node(self, index):
        index = int(index)
        if 0 <= index < len(self.nodes):
            del self.nodes[index]
            self._renumber()

    def move_node(self, index, delta):
        index = int(index)
        new_index = index + int(delta)
        if 0 <= index < len(self.nodes) and 0 <= new_index < len(self.nodes):
            self.nodes[index], self.nodes[new_index] = self.nodes[new_index], self.nodes[index]
            self._renumber()

    def update_node(self, index, **updates):
        index = int(index)
        if not 0 <= index < len(self.nodes):
            return
        node = self.nodes[index]
        if "effect" in updates:
            effect = str(updates["effect"])
            if effect in CDP_EFFECT_SPECS and effect != node.effect:
                node.effect = effect
                node.params = dict(CDP_EFFECT_SPECS[effect]["params"])
        if "enabled" in updates:
            node.enabled = bool(updates["enabled"])
        for key, value in updates.get("params", {}).items():
            try:
                node.params[str(key)] = float(value)
            except (TypeError, ValueError):
                node.params[str(key)] = value

    def to_dict(self):
        return {"nodes": [node.to_dict() for node in self.nodes]}

    @classmethod
    def from_dict(cls, data):
        return cls(CDPGraphNode.from_dict(item) for item in data.get("nodes", []))

    def _renumber(self):
        for index, node in enumerate(self.nodes, 1):
            node.id = f"cdp_{index:02d}"


class CDPGraphProcessor:
    """Callback-safe graph processor with optional asynchronous CDP rendering."""

    def __init__(self, cdp_programs_path=DEFAULT_CDP_PROGRAMS_PATH):
        self.graph = CDPNodeGraph()
        self.enabled = False
        self.use_cdp = False
        self.wet_mix = 0.65
        self.cpu_preview = True
        self.cdp_programs_path = str(cdp_programs_path)
        self.status = "CDP graph bypassed."
        self.last_preview_delta = 0.0
        self.last_worker_error = ""
        self._lock = threading.RLock()
        self._request_queue: queue.Queue[tuple[np.ndarray, int, dict[str, float]]] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._last_cdp = None
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, name="MegaGlitchCDPGraph", daemon=True)
        self._worker.start()

    def shutdown(self):
        self._running = False

    def snapshot(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "use_cdp": self.use_cdp,
                "wet_mix": self.wet_mix,
                "cpu_preview": self.cpu_preview,
                "cdp_programs_path": self.cdp_programs_path,
                "status": self.status,
                "last_preview_delta": self.last_preview_delta,
                "last_worker_error": self.last_worker_error,
                "graph": self.graph.to_dict(),
                "available_effects": list(CDP_EFFECT_SPECS),
            }

    def apply_snapshot(self, state):
        if not isinstance(state, dict):
            return
        with self._lock:
            self.enabled = bool(state.get("enabled", self.enabled))
            self.use_cdp = bool(state.get("use_cdp", self.use_cdp))
            self.cpu_preview = bool(state.get("cpu_preview", self.cpu_preview))
            try:
                self.wet_mix = max(0.0, min(1.0, float(state.get("wet_mix", self.wet_mix))))
            except (TypeError, ValueError):
                pass
            if state.get("cdp_programs_path"):
                self.cdp_programs_path = str(state.get("cdp_programs_path"))
            graph_state = state.get("graph")
            if isinstance(graph_state, dict):
                self.graph = CDPNodeGraph.from_dict(graph_state)
            self.status = "CDP graph restored from preset." if self.enabled else "CDP graph bypassed."
            self.last_preview_delta = 0.0
            self.last_worker_error = ""

    def configure(self, **kwargs):
        with self._lock:
            if "enabled" in kwargs:
                self.enabled = bool(kwargs["enabled"])
                if self.enabled and self.wet_mix <= 0.0:
                    self.wet_mix = 0.65
            if "use_cdp" in kwargs:
                self.use_cdp = bool(kwargs["use_cdp"])
            if "cpu_preview" in kwargs:
                self.cpu_preview = bool(kwargs["cpu_preview"])
            if "wet_mix" in kwargs:
                self.wet_mix = max(0.0, min(1.0, float(kwargs["wet_mix"])))
            if "cdp_programs_path" in kwargs:
                self.cdp_programs_path = str(kwargs["cdp_programs_path"] or self.cdp_programs_path)

    def add_node(self, effect):
        with self._lock:
            node = self.graph.add_node(effect)
            self.status = f"Added CDP node: {node.effect}."
            return node.to_dict()

    def remove_node(self, index):
        with self._lock:
            self.graph.remove_node(index)
            self.status = "Removed CDP graph node."

    def move_node(self, index, delta):
        with self._lock:
            self.graph.move_node(index, delta)
            self.status = "Moved CDP graph node."

    def update_node(self, index, **updates):
        with self._lock:
            self.graph.update_node(index, **updates)
            self.status = "Updated CDP graph node."

    def process(self, audio, sample_rate, visual=None):
        with self._lock:
            enabled = self.enabled
            wet_mix = self.wet_mix
            cpu_preview = self.cpu_preview
            use_cdp = self.use_cdp
            graph = CDPNodeGraph.from_dict(self.graph.to_dict())
        if not enabled or wet_mix <= 0.0 or audio.size == 0 or not graph.nodes:
            if enabled and wet_mix <= 0.0:
                with self._lock:
                    self.status = "CDP graph enabled but Wet Mix is 0. Raise Wet Mix to hear it."
            return audio
        dry = np.asarray(audio, dtype=np.float32)
        wet = dry
        if cpu_preview:
            wet = self._apply_cpu_graph(dry, int(sample_rate), visual or {}, graph)
            delta = float(np.mean(np.abs(wet - dry))) if wet.shape == dry.shape else 0.0
            with self._lock:
                self.last_preview_delta = delta
                if not use_cdp:
                    self.status = f"CDP CPU preview active: {len(graph.nodes)} node(s), delta {delta:.4f}."
        if use_cdp:
            self._submit_cdp(dry, int(sample_rate), visual or {})
            cdp = self._poll_cdp(dry.shape)
            if cdp is not None:
                wet = cdp if not cpu_preview else wet * 0.35 + cdp * 0.65
        out = dry * np.float32(1.0 - wet_mix) + wet.astype(np.float32, copy=False) * np.float32(wet_mix)
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    def _apply_cpu_graph(self, audio, sample_rate, visual, graph):
        y = np.asarray(audio, dtype=np.float32)
        for node in graph.nodes:
            if not node.enabled:
                continue
            y = self._apply_cpu_effect(y, sample_rate, node.effect, node.params, visual)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

    def _apply_cpu_effect(self, audio, sample_rate, effect, params, visual):
        mix = max(0.0, min(1.0, float(params.get("mix", 0.6))))
        dry = audio
        category = str(CDP_EFFECT_SPECS.get(effect, {}).get("category", "")).lower()
        if category in {"pitchinfo", "pview", "sfconvert", "sfprops", "sndinfo", "specinfo", "houskeep"}:
            wet = dry
        elif category == "blur":
            amount = int(max(1, min(256, round(float(params.get("amount", 4.0)) * 4.0))))
            kernel = np.ones(amount, dtype=np.float64) / amount
            wet = np.column_stack([np.convolve(dry[:, channel], kernel, mode="same") for channel in range(dry.shape[1])])
        elif category in {"combine", "mix", "submix"}:
            delay = max(1, int(sample_rate * float(params.get("delay_ms", 45.0)) * 0.001))
            feedback = max(0.0, min(0.95, float(params.get("feedback", 0.35))))
            wet = dry.copy()
            if delay < dry.shape[0]:
                wet[delay:] += dry[:-delay] * feedback
                wet[:delay] *= 1.0 - feedback * 0.25
            wet = np.tanh(wet * float(params.get("gain", 1.0)))
        elif category == "distort":
            drive = 1.0 + float(params.get("drive", 0.55)) * 24.0
            wet = np.tanh(dry * drive)
            wet = np.round(wet * 96.0) / 96.0
            wet = (wet + np.roll(wet, 1, axis=0) + np.roll(wet, 2, axis=0)) / 3.0
        elif category == "env":
            rate = max(0.01, float(params.get("rate", 11.0)))
            depth = max(0.0, min(1.0, float(params.get("depth", 0.65))))
            t = np.arange(dry.shape[0], dtype=np.float64) / float(sample_rate)
            mod = (1.0 - depth) + depth * (0.5 + 0.5 * np.sign(np.sin(2.0 * np.pi * rate * t)))
            wet = dry * mod[:, None]
        elif category in {"filter", "formants"}:
            freq = max(30.0, min(sample_rate * 0.45, float(params.get("freq", 1200.0))))
            q = max(0.1, min(8.0, float(params.get("q", 0.7))))
            coeff = 1.0 - np.exp(-2.0 * np.pi * freq / sample_rate)
            low = lfilter([coeff], [1.0, -(1.0 - coeff)], dry, axis=0)
            if category == "formants":
                high = dry - low
                band_freqs = (freq * 0.65, freq, min(sample_rate * 0.45, freq * 1.7))
                formant = np.zeros_like(dry)
                for band_freq in band_freqs:
                    band_coeff = 1.0 - np.exp(-2.0 * np.pi * band_freq / sample_rate)
                    band_low = lfilter([band_coeff], [1.0, -(1.0 - band_coeff)], dry, axis=0)
                    formant += dry - band_low
                wet = np.tanh(low * 0.35 + formant * float(params.get("gain", 0.5)) * (1.5 + q))
            else:
                high = dry - low
                wet = np.tanh(low * 1.2 + high * (q * 0.45))
        elif category == "grain":
            grain = max(4, int(sample_rate * float(params.get("grain_ms", 34.0)) * 0.001))
            jitter = max(0.0, min(1.0, float(params.get("jitter", 0.45))))
            wet = dry.copy()
            for start in range(0, dry.shape[0], grain):
                end = min(dry.shape[0], start + grain)
                offset = int((self._hash01(start + grain * 3.0) - 0.5) * jitter * grain * 2.0)
                src = min(max(0, start + offset), max(0, dry.shape[0] - (end - start)))
                segment = dry[src:src + (end - start)]
                if self._hash01(start + 17.0) < jitter * 0.35:
                    segment = segment[::-1]
                if self._hash01(start + 31.0) < jitter * 0.45:
                    segment = np.roll(segment, int(self._hash01(start + 47.0) * max(1, len(segment))), axis=0)
                wet[start:end] = segment
        elif category == "hfperm":
            wet = dry.copy()
            frame = max(16, int(256 * (1.0 + float(params.get("amount", 0.55)))))
            for start in range(0, dry.shape[0], frame):
                end = min(dry.shape[0], start + frame)
                block = dry[start:end]
                spectrum = np.fft.rfft(block, axis=0)
                if spectrum.shape[0] > 4:
                    split = spectrum.shape[0] // 2
                    order = np.arange(split, spectrum.shape[0])
                    order = np.roll(order[::-1], int(self._hash01(start + 5.0) * max(1, len(order))))
                    spectrum[split:] = spectrum[order]
                restored = np.fft.irfft(spectrum, n=end - start, axis=0).astype(np.float32)
                wet[start:end] = restored
        elif category == "hilite":
            freq = max(30.0, min(sample_rate * 0.45, float(params.get("freq", 2400.0))))
            gain = float(params.get("gain", 0.55))
            coeff = 1.0 - np.exp(-2.0 * np.pi * freq / sample_rate)
            low = lfilter([coeff], [1.0, -(1.0 - coeff)], dry, axis=0)
            high = dry - low
            wet = np.tanh(dry * 0.4 + high * gain * 7.0)
        elif category == "modify":
            gain = max(0.0, min(4.0, float(params.get("gain", 1.2))))
            shape = max(0.0, min(1.0, float(params.get("shape", 0.35))))
            wet = np.tanh(dry * gain * (1.0 + shape * 3.0))
        elif category == "morph":
            amount = max(0.0, min(1.0, float(params.get("amount", 0.48))))
            shifted = np.roll(dry[:, ::-1], max(1, int(19 + amount * 179)), axis=0)
            wet = np.tanh(dry * (1.0 - amount * 0.35) + shifted * amount * 2.5)
        elif category in {"pitch", "repitch"}:
            semitones = float(params.get("semitones", 7.0 if category == "pitch" else -5.0))
            rate = max(0.25, min(4.0, 2.0 ** (semitones / 12.0)))
            src = np.arange(dry.shape[0], dtype=np.float64) * rate
            src = np.mod(src, max(1, dry.shape[0] - 1))
            wet = np.empty_like(dry)
            base = np.arange(dry.shape[0], dtype=np.float64)
            for channel in range(dry.shape[1]):
                wet[:, channel] = np.interp(src, base, dry[:, channel]).astype(np.float32)
        elif category == "pvoc":
            freeze = max(0.0, min(1.0, float(params.get("freeze", 0.6))))
            frame = min(dry.shape[0], 512)
            if frame <= 1:
                wet = dry
            else:
                spectrum = np.fft.rfft(dry[:frame], axis=0)
                frozen = np.fft.irfft(spectrum, n=frame, axis=0).astype(np.float32)
                repeats = int(np.ceil(dry.shape[0] / frame))
                wet = np.tile(frozen, (repeats, 1))[: dry.shape[0]]
                wet = dry * (1.0 - freeze) + wet * freeze
                wet = np.tanh(wet * (1.0 + freeze * 1.5))
        elif category == "sfedit":
            amount = max(0.0, min(0.95, float(params.get("amount", 0.12))))
            cut = int(dry.shape[0] * amount)
            wet = np.zeros_like(dry)
            if cut < dry.shape[0]:
                wet[: dry.shape[0] - cut] = dry[cut:]
        elif category == "spec":
            amount = max(0.0, min(1.0, float(params.get("amount", 0.42))))
            spectrum = np.fft.rfft(dry, axis=0)
            kernel_width = max(1, int(1 + amount * 12))
            kernel = np.ones(kernel_width, dtype=np.float64) / kernel_width
            for channel in range(spectrum.shape[1]):
                real = np.convolve(spectrum[:, channel].real, kernel, mode="same")
                imag = np.convolve(spectrum[:, channel].imag, kernel, mode="same")
                spectrum[:, channel] = real + 1j * imag
            wet = np.fft.irfft(spectrum, n=dry.shape[0], axis=0).astype(np.float32)
            wet = np.tanh(wet * (1.0 + amount * 2.0))
        elif category == "stretch":
            factor = max(0.25, min(4.0, float(params.get("factor", 1.6))))
            src = np.linspace(0.0, max(0.0, dry.shape[0] - 1.0), max(2, int(dry.shape[0] / factor)))
            base = np.arange(dry.shape[0], dtype=np.float64)
            stretched = np.empty((len(src), dry.shape[1]), dtype=np.float32)
            for channel in range(dry.shape[1]):
                stretched[:, channel] = np.interp(src, base, dry[:, channel]).astype(np.float32)
            target = np.linspace(0.0, max(0.0, stretched.shape[0] - 1.0), dry.shape[0])
            wet = np.empty_like(dry)
            stretch_base = np.arange(stretched.shape[0], dtype=np.float64)
            for channel in range(dry.shape[1]):
                wet[:, channel] = np.interp(target, stretch_base, stretched[:, channel]).astype(np.float32)
        elif category == "visual_glitch":
            wet = self._apply_visual_glitch_effect(dry, sample_rate, effect, params, visual)
        else:
            wet = dry
        return np.clip(dry * (1.0 - mix) + wet * mix, -1.0, 1.0).astype(np.float32, copy=False)

    def _visual_value(self, visual, name, default=0.0):
        try:
            value = visual.get(name, default) if isinstance(visual, dict) else default
            if isinstance(value, np.ndarray):
                value = float(np.mean(value)) if value.size else default
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return float(default)

    def _apply_visual_glitch_effect(self, dry, sample_rate, effect, params, visual):
        mode = str(CDP_EFFECT_SPECS.get(effect, {}).get("mode", ""))
        frames = dry.shape[0]
        if frames <= 1:
            return dry
        t = np.arange(frames, dtype=np.float64) / max(1.0, float(sample_rate))
        motion = self._visual_value(visual, "motion")
        brightness = self._visual_value(visual, "brightness", 0.5)
        edge = self._visual_value(visual, "edge_density")
        entropy = self._visual_value(visual, "entropy")
        spike = self._visual_value(visual, "glitch_spike")
        red = self._visual_value(visual, "red_energy", 0.33)
        green = self._visual_value(visual, "green_energy", 0.33)
        blue = self._visual_value(visual, "blue_energy", 0.33)
        center_x = self._visual_value(visual, "center_mass_x", 0.5)
        center_y = self._visual_value(visual, "center_mass_y", 0.5)

        if mode == "motion_slicer":
            rate = max(1.0, float(params.get("rate", 18.0)) * (0.5 + motion * 2.8))
            depth = max(0.0, min(1.0, float(params.get("depth", 0.78)))) * (0.35 + motion + spike * 0.8)
            jitter = max(0.0, min(1.0, float(params.get("jitter", 0.42)))) * (0.25 + entropy)
            phase = np.mod(t * rate + jitter * np.sin(t * rate * 0.73 + edge * 8.0), 1.0)
            gate = np.where(phase < 0.5 + (brightness - 0.5) * 0.28, 1.0, 1.0 - depth)
            wet = dry * gate[:, None]
            shift = int((self._hash01(rate + frames) - 0.5) * jitter * frames * 0.12)
            if shift:
                wet = np.roll(wet, shift, axis=0)
        elif mode == "edge_comb":
            delay = max(1, int(sample_rate * float(params.get("delay_ms", 7.0)) * 0.001 * (0.35 + edge * 2.5)))
            feedback = max(0.0, min(0.95, float(params.get("feedback", 0.68)) * (0.3 + edge + spike * 0.5)))
            tone = max(0.0, min(1.0, float(params.get("tone", 0.55))))
            wet = dry.copy()
            if delay < frames:
                for index in range(delay, frames):
                    wet[index] += wet[index - delay] * feedback
            coeff = 1.0 - np.exp(-2.0 * np.pi * (600.0 + tone * 7600.0 * (0.4 + brightness)) / sample_rate)
            low = lfilter([coeff], [1.0, -(1.0 - coeff)], wet, axis=0)
            wet = np.tanh(low * (0.55 + tone) + (wet - low) * (1.0 + edge * 3.0))
        elif mode == "color_bitcrush":
            color_drive = max(red, green, blue) - min(red, green, blue)
            bits = max(2.0, min(12.0, float(params.get("bits", 5.0)) - color_drive * 3.0 - spike * 2.0))
            levels = max(3.0, 2.0 ** bits)
            fold = max(0.0, min(1.0, float(params.get("fold", 0.36)) * (0.5 + red + blue + spike)))
            wet = np.round(np.tanh(dry * (1.0 + fold * 6.0)) * levels) / levels
            stride = max(1, int(1 + (1.0 - green) * 14.0 + spike * 18.0))
            wet = np.repeat(wet[::stride], stride, axis=0)[:frames]
            if wet.shape[0] < frames:
                wet = np.pad(wet, ((0, frames - wet.shape[0]), (0, 0)), mode="edge")
        elif mode == "brightness_gate":
            threshold = max(0.0, min(1.0, float(params.get("threshold", 0.48))))
            softness = max(0.001, min(1.0, float(params.get("softness", 0.18))))
            gate = np.clip((brightness - threshold + softness) / (softness * 2.0), 0.0, 1.0)
            trem_rate = 4.0 + motion * 38.0 + spike * 70.0
            trem = 0.5 + 0.5 * np.sign(np.sin(2.0 * np.pi * trem_rate * t))
            wet = dry * (gate * (0.35 + 0.65 * trem))[:, None]
        elif mode == "spike_repeat":
            slice_ms = max(2.0, float(params.get("slice_ms", 38.0)) * (1.4 - spike * 0.9))
            size = max(2, int(sample_rate * slice_ms * 0.001))
            repeats = max(1, int(round(float(params.get("repeats", 5.0)) * (0.4 + spike + motion * 0.5))))
            wet = dry.copy()
            for start in range(0, frames, size):
                end = min(frames, start + size)
                segment = dry[start:end]
                if segment.size == 0:
                    continue
                if self._hash01(start + spike * 101.0) < 0.25 + spike * 0.7:
                    tiny = max(1, len(segment) // repeats)
                    repeated = np.tile(segment[:tiny], (int(np.ceil(len(segment) / tiny)), 1))[:len(segment)]
                    if self._hash01(start + 17.0) < motion:
                        repeated = repeated[::-1]
                    wet[start:end] = repeated
        elif mode == "entropy_tear":
            amount = max(0.0, min(1.0, float(params.get("amount", 0.62)) * (0.35 + entropy + spike * 0.7)))
            tilt = max(0.0, min(1.0, float(params.get("tilt", 0.48))))
            spectrum = np.fft.rfft(dry, axis=0)
            bins = spectrum.shape[0]
            if bins > 4:
                phase = np.exp(1j * np.sin(np.arange(bins) * (0.11 + entropy * 0.37) + center_y * 9.0) * amount * math.pi)
                tilt_curve = np.linspace(1.0 - tilt * 0.5, 1.0 + tilt * (0.8 + edge), bins)
                spectrum *= phase[:, None] * tilt_curve[:, None]
                cut = int(bins * (0.18 + brightness * 0.45))
                roll = int((center_x - 0.5) * amount * bins * 0.35)
                spectrum[cut:] = np.roll(spectrum[cut:], roll, axis=0)
            wet = np.fft.irfft(spectrum, n=frames, axis=0).astype(np.float32)
            wet = np.tanh(wet * (1.0 + amount * 2.5))
        elif mode == "center_pan":
            width = max(0.0, min(1.0, float(params.get("width", 0.9))))
            rate = max(0.01, float(params.get("rate", 4.0)) * (0.25 + motion + edge))
            pan = np.clip((center_x - 0.5) * 2.0 + np.sin(2.0 * np.pi * rate * t + center_y * math.tau) * width * 0.65, -1.0, 1.0)
            mono = np.mean(dry, axis=1)
            wet = np.empty_like(dry)
            wet[:, 0] = mono * np.sqrt((1.0 - pan) * 0.5)
            wet[:, 1] = mono * np.sqrt((1.0 + pan) * 0.5)
        elif mode == "rgb_ringmod":
            rate = max(0.1, float(params.get("rate", 36.0)) * (0.25 + red * 0.9 + blue * 1.4 + spike))
            depth = max(0.0, min(1.0, float(params.get("depth", 0.72)) * (0.35 + green + motion)))
            left_mod = np.sin(2.0 * np.pi * rate * t + red * math.tau)
            right_mod = np.sin(2.0 * np.pi * rate * (0.74 + blue) * t + blue * math.tau)
            mod = np.column_stack((left_mod, right_mod))
            wet = dry * ((1.0 - depth) + depth * mod)
            wet = np.tanh(wet * (1.0 + depth * 2.4))
        elif mode == "block_dropout":
            blocks = max(2, int(round(float(params.get("blocks", 16.0)) * (0.5 + edge + entropy))))
            drop = max(0.0, min(1.0, float(params.get("drop", 0.42)) * (0.3 + motion + spike + (1.0 - brightness) * 0.5)))
            wet = dry.copy()
            block_size = max(1, int(math.ceil(frames / blocks)))
            for block_index, start in enumerate(range(0, frames, block_size)):
                end = min(frames, start + block_size)
                seed = block_index * 37.0 + frames * 0.013 + center_x * 17.0
                if self._hash01(seed) < drop:
                    wet[start:end] *= 0.0 if self._hash01(seed + 4.0) < 0.55 else -0.65
                elif self._hash01(seed + 9.0) < drop * 0.45:
                    wet[start:end] = wet[start:end][::-1]
        else:
            wet = dry
        return np.clip(wet, -1.0, 1.0).astype(np.float32, copy=False)

    def _submit_cdp(self, audio, sample_rate, visual):
        if self._request_queue.full():
            return
        try:
            self._request_queue.put_nowait((audio.copy(), int(sample_rate), dict(visual)))
        except queue.Full:
            return

    def _poll_cdp(self, shape):
        try:
            while True:
                self._last_cdp = self._result_queue.get_nowait()
        except queue.Empty:
            pass
        if self._last_cdp is None:
            return None
        result = self._last_cdp
        if result.shape != shape:
            fixed = np.zeros(shape, dtype=np.float32)
            frames = min(shape[0], result.shape[0])
            channels = min(shape[1], result.shape[1])
            fixed[:frames, :channels] = result[:frames, :channels]
            result = fixed
        return result

    def _worker_loop(self):
        while self._running:
            try:
                audio, sample_rate, visual = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    graph = CDPNodeGraph.from_dict(self.graph.to_dict())
                    cdp_programs_path = self.cdp_programs_path
                result = self._run_cdp_graph(audio, sample_rate, graph, cdp_programs_path)
                try:
                    self._result_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._result_queue.put_nowait(result)
                with self._lock:
                    self.status = "CDP graph processed latest audio chunk."
                    self.last_worker_error = ""
            except Exception as exc:  # pragma: no cover - depends on local CDP install
                with self._lock:
                    self.last_worker_error = str(exc)
                    self.status = f"CDP worker unavailable for this graph: {exc}"
            finally:
                self._request_queue.task_done()

    def _run_cdp_graph(self, audio, sample_rate, graph, cdp_programs_path):
        current = np.asarray(audio, dtype=np.float32)
        ran = 0
        for node in graph.nodes:
            if not node.enabled:
                continue
            spec = CDP_EFFECT_SPECS.get(node.effect, {})
            if not bool(spec.get("worker", False)):
                continue
            current = self._run_cdp_node(current, sample_rate, node, cdp_programs_path)
            ran += 1
        if ran == 0:
            raise RuntimeError("No graph nodes are marked CDP-worker compatible for raw live WAV chunks; CPU Preview is the active live path.")
        return current.astype(np.float32, copy=False)

    def _run_cdp_node(self, audio, sample_rate, node, cdp_programs_path):
        spec = CDP_EFFECT_SPECS.get(node.effect)
        if not spec:
            return audio
        program_path = os.path.join(cdp_programs_path, str(spec.get("program", "")))
        if not os.path.isfile(program_path):
            raise FileNotFoundError(program_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "in.wav")
            out_path = os.path.join(tmpdir, "out.wav")
            wavfile.write(in_path, int(sample_rate), np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype=np.int16))
            args = self._cdp_args(node.effect, node.params)
            category = str(spec.get("category", ""))
            command_text = " ".join(
                shlex.quote(str(part))
                for part in (
                    "cdpwrap",
                    category,
                    *args[:1],
                    self._windows_path_to_wsl(in_path),
                    self._windows_path_to_wsl(out_path),
                    *args[1:],
                )
                if str(part)
            )
            result = subprocess.run(
                ["wsl", "bash", "-lc", command_text],
                cwd=cdp_programs_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8.0,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
                raise RuntimeError(f"CDP failed for {node.effect}: {detail}")
            if not os.path.isfile(out_path):
                raise RuntimeError(f"CDP did not produce output for {node.effect}.")
            out_rate, data = wavfile.read(out_path)
            if out_rate != sample_rate:
                raise RuntimeError(f"CDP output sample rate changed: {out_rate}")
            if data.ndim == 1:
                data = data[:, None]
            return np.asarray(data[:, : audio.shape[1]], dtype=np.float32) / 32768.0

    @staticmethod
    def _cdp_args(effect, params):
        spec = CDP_EFFECT_SPECS.get(effect, {})
        category = str(spec.get("category", "")).lower()
        mode = str(spec.get("mode", ""))
        if category in {"pitchinfo", "pview", "sfconvert", "sfprops", "sndinfo", "specinfo", "houskeep"}:
            return [mode]
        if effect == "Blur Average":
            return ["avrg", str(max(1, int(round(float(params.get("amount", 4.0))))))]
        if effect == "Distort Average":
            return [
                "average",
                str(max(1, int(round(float(params.get("cycles", 4.0)))))),
                "0.5",
                "0",
            ]
        if effect == "Filter Fixed":
            return ["fixed", "1", str(float(params.get("freq", 1200.0))), str(float(params.get("gain", 0.0)))]
        if effect == "Grain Timewarp":
            return ["timewarp", "1.0"]
        if effect == "Hilite Filter":
            return ["filter", "1", str(float(params.get("freq", 2400.0))), str(float(params.get("q", 1.5)))]
        if effect == "Modify Loudness":
            return ["loudness", "1", str(float(params.get("gain", 1.2)))]
        return [mode]

    @staticmethod
    def _windows_path_to_wsl(path):
        path = os.path.abspath(str(path))
        drive, rest = os.path.splitdrive(path)
        rest = rest.replace("\\", "/")
        if drive:
            return f"/mnt/{drive[0].lower()}{rest}"
        return rest

    @staticmethod
    def _hash01(value):
        return float(np.mod(np.sin(float(value) * 12.9898 + 78.233) * 43758.5453, 1.0))
