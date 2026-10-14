"""TX802-inspired six-operator real-time FM synthesis engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading

import numpy as np

from .envelopes import ADSR
from .scales import midi_to_frequency, quantize_frequency, quantize_midi


ALGORITHMS = ("stack", "parallel", "pair", "feedback_bell")
ALGORITHM_ALIASES = {
    "stack": 1,
    "parallel": 32,
    "pair": 30,
    "feedback_bell": 8,
}

# Each tuple entry is the target operator index for that operator, or -1 when
# the operator is an audible carrier. This table mirrors the existing richer
# media audio engine routing so FM nodes share one musical interpretation.
ALGORITHMS_6OP = (
    (-1, 0, 1, 2, 3, 4),
    (-1, 0, 1, 2, -1, 4),
    (-1, 0, 1, -1, 3, 4),
    (-1, 0, -1, 2, 3, 4),
    (-1, 0, 1, -1, 3, -1),
    (-1, 0, -1, 2, -1, 4),
    (-1, 0, -1, -1, 3, 4),
    (-1, 0, 1, -1, -1, 4),
    (-1, 0, -1, 2, 3, -1),
    (-1, 0, -1, 2, -1, -1),
    (-1, 0, -1, -1, 3, -1),
    (-1, 0, 1, -1, -1, -1),
    (-1, 0, -1, 0, 3, 4),
    (-1, 0, -1, 0, -1, 4),
    (-1, 0, -1, 0, -1, -1),
    (-1, 0, 1, 0, 3, -1),
    (-1, 0, 1, 0, -1, -1),
    (-1, 0, -1, 2, 2, -1),
    (-1, 0, 0, -1, 3, 3),
    (-1, 0, 0, -1, -1, 4),
    (-1, 0, 0, -1, -1, -1),
    (-1, 0, 0, 2, 2, -1),
    (-1, 0, 0, 2, -1, -1),
    (-1, 0, 0, 0, 3, -1),
    (-1, 0, 0, 0, -1, -1),
    (-1, 0, -1, 2, 2, 2),
    (-1, 0, 0, 0, 0, -1),
    (-1, 0, 0, 0, 0, 0),
    (-1, -1, 1, 1, 3, 3),
    (-1, -1, 1, -1, 3, -1),
    (-1, -1, -1, 2, -1, 4),
    (-1, -1, -1, -1, -1, -1),
)

TX802_FEEDBACK_OPERATORS = (
    5, 5, 5, 5, 2, 5, 5, 2,
    5, 3, 5, 2, 5, 5, 3, 2,
    2, 5, 5, 5, 3, 5, 3, 4,
    4, 5, 5, 5, 5, 3, 5, 0,
)

OPERATOR_COUNT = 6
PART_COUNT = 8
MAX_POLYPHONY = 16


@dataclass
class TX802Envelope:
    """Four-rate/four-level operator envelope.

    Rates are normalized 0..99 values. Higher rates move faster, which matches
    the editing feel of Yamaha FM without trying to recreate exact chip timing.
    """

    sample_rate: float = 48000.0
    rates: tuple[float, float, float, float] = (86.0, 62.0, 36.0, 50.0)
    levels: tuple[float, float, float, float] = (1.0, 0.65, 0.5, 0.0)

    def __post_init__(self):
        self.value = 0.0
        self.stage = "idle"

    @property
    def active(self) -> bool:
        return self.stage != "idle"

    def configure(self, rates=None, levels=None):
        if rates is not None:
            self.rates = tuple(min(99.0, max(0.0, float(v))) for v in rates)
        if levels is not None:
            self.levels = tuple(min(1.0, max(0.0, float(v))) for v in levels)

    def trigger(self):
        self.value = self.levels[3]
        self.stage = "rate1"

    def release_gate(self):
        if self.stage != "idle":
            self.stage = "release"

    def reset(self):
        self.value = 0.0
        self.stage = "idle"

    def _step_toward(self, target, rate):
        # 99 reaches the next level in roughly 5 ms; 1 takes roughly 2 s.
        seconds = 0.005 + (1.0 - rate / 99.0) ** 2 * 2.0
        step = 1.0 / max(1.0, seconds * self.sample_rate)
        if self.value < target:
            self.value = min(target, self.value + step)
        else:
            self.value = max(target, self.value - step)
        return abs(self.value - target) <= 1e-6

    def next_value(self) -> float:
        if self.stage == "rate1":
            if self._step_toward(self.levels[0], self.rates[0]):
                self.stage = "rate2"
        elif self.stage == "rate2":
            if self._step_toward(self.levels[1], self.rates[1]):
                self.stage = "rate3"
        elif self.stage == "rate3":
            if self._step_toward(self.levels[2], self.rates[2]):
                self.stage = "sustain"
        elif self.stage == "sustain":
            self.value = self.levels[2]
        elif self.stage == "release":
            if self._step_toward(self.levels[3], self.rates[3]):
                self.reset()
        return min(1.0, max(0.0, self.value))


class FMEngine:
    """Polyphonic six-operator FM voice engine with callback-safe rendering."""

    PARAM_LIMITS = {
        "master_gain": (0.0, 1.0),
        "carrier_freq": (20.0, 4000.0),
        "modulation_index": (0.0, 24.0),
        "feedback": (0.0, 0.95),
        "detune": (-50.0, 50.0),
        "attack": (0.001, 10.0),
        "decay": (0.001, 20.0),
        "sustain": (0.0, 1.0),
        "release": (0.001, 30.0),
        "lfo_rate": (0.0, 30.0),
        "lfo_depth": (0.0, 1.0),
        "tempo_bpm": (20.0, 300.0),
        "velocity_sensitivity": (0.0, 1.0),
        "polyphony": (1, MAX_POLYPHONY),
    }

    def __init__(self, sample_rate=48000, block_size=256, seed=1977):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.rng = np.random.default_rng(seed)
        self.params = self._default_params()
        self._smoothed = {
            name: float(value)
            for name, value in self.params.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self.smoothing_seconds = 0.025
        self.envelope = ADSR(self.sample_rate)
        self._voice_phase = np.zeros((MAX_POLYPHONY, OPERATOR_COUNT), dtype=np.float64)
        self._voice_feedback = np.zeros((MAX_POLYPHONY, OPERATOR_COUNT), dtype=np.float64)
        self._voice_note = np.full(MAX_POLYPHONY, np.nan, dtype=np.float64)
        self._voice_velocity = np.ones(MAX_POLYPHONY, dtype=np.float64)
        self._voice_part = np.zeros(MAX_POLYPHONY, dtype=np.int32)
        self._voice_age = np.zeros(MAX_POLYPHONY, dtype=np.int64)
        self._voice_envelopes = [
            [TX802Envelope(self.sample_rate) for _ in range(OPERATOR_COUNT)]
            for _ in range(MAX_POLYPHONY)
        ]
        self._operator_energy = np.zeros(OPERATOR_COUNT, dtype=np.float64)
        self._part_rms = np.zeros(PART_COUNT, dtype=np.float64)
        self._lfo_phase = 0.0
        self._velocity = 1.0
        self._note = None
        self._trigger_counter = 0
        self._panic = False
        self._lock = threading.RLock()
        self._analysis = self._default_analysis()
        self._previous_rms = 0.0
        self._audio_clock_samples = 0

    @staticmethod
    def _default_params():
        params = {
            "enabled": True,
            "voice_mode": "tx802",
            "master_gain": 0.35,
            "carrier_freq": 110.0,
            "modulation_index": 1.5,
            "feedback": 0.05,
            "detune": 0.0,
            "attack": 0.02,
            "decay": 0.4,
            "sustain": 0.6,
            "release": 1.2,
            "lfo_rate": 0.1,
            "lfo_depth": 0.1,
            "lfo_sync": False,
            "lfo_division": 1.0,
            "tempo_bpm": 120.0,
            "algorithm": 1,
            "polyphony": MAX_POLYPHONY,
            "scale_root": 36,
            "scale_mode": "minor_pentatonic",
            "quantize_pitch": True,
            "velocity_sensitivity": 0.8,
            "show_energy": 0.0,
            "show_chaos": 0.0,
        }
        ratio_defaults = (1.0, 2.0, 3.0, 5.0, 0.5, 7.0)
        level_defaults = (1.0, 0.65, 0.45, 0.3, 0.22, 0.18)
        pan_defaults = (-0.2, 0.2, -0.55, 0.55, -0.35, 0.35)
        for index in range(1, OPERATOR_COUNT + 1):
            params[f"operator_ratio_{index}"] = ratio_defaults[index - 1]
            params[f"operator_detune_{index}"] = 0.0
            params[f"operator_mix_{index}"] = level_defaults[index - 1]
            params[f"operator_level_{index}"] = level_defaults[index - 1]
            params[f"operator_pan_{index}"] = pan_defaults[index - 1]
            params[f"operator_fixed_{index}"] = False
            params[f"operator_fixed_freq_{index}"] = 440.0
            params[f"operator_velocity_{index}"] = 0.0 if index == 1 else 0.35
            params[f"operator_key_scale_{index}"] = 0.0
            params[f"operator_enabled_{index}"] = True
            for stage, default in enumerate((86.0, 62.0, 36.0, 50.0), 1):
                params[f"operator_eg_rate_{index}_{stage}"] = default
            for stage, default in enumerate((1.0, 0.65, 0.5, 0.0), 1):
                params[f"operator_eg_level_{index}_{stage}"] = default
        for part in range(1, PART_COUNT + 1):
            params[f"part_{part}_level"] = 1.0 if part == 1 else 0.0
            params[f"part_{part}_pan"] = 0.0
            params[f"part_{part}_detune"] = 0.0
            params[f"part_{part}_note_low"] = 0
            params[f"part_{part}_note_high"] = 127
            params[f"part_{part}_polyphony_reserve"] = 0
            params[f"part_{part}_channel"] = part
        return params

    @staticmethod
    def _default_analysis():
        analysis = {
            "rms": 0.0,
            "peak": 0.0,
            "spectral_brightness": 0.0,
            "transient": 0.0,
            "beat_phase": 0.0,
            "modulation_activity": 0.0,
            "voice_count": 0.0,
            "feedback_activity": 0.0,
        }
        for index in range(1, OPERATOR_COUNT + 1):
            analysis[f"operator_{index}_energy"] = 0.0
        for part in range(1, PART_COUNT + 1):
            analysis[f"part_{part}_rms"] = 0.0
        return analysis

    @staticmethod
    def _coerce_algorithm(value):
        if isinstance(value, str):
            if value in ALGORITHM_ALIASES:
                return ALGORITHM_ALIASES[value]
            digits = "".join(ch for ch in value if ch.isdigit())
            return min(32, max(1, int(digits))) if digits else 1
        numeric = int(round(float(value)))
        if numeric in range(0, 4):
            return (1, 32, 30, 8)[numeric]
        return min(32, max(1, numeric))

    def set_param(self, name, value):
        if name not in self.params:
            raise KeyError(f"Unknown FM parameter: {name}")
        if name == "algorithm":
            value = self._coerce_algorithm(value)
        elif name in self.PARAM_LIMITS:
            lo, hi = self.PARAM_LIMITS[name]
            if name == "polyphony":
                value = min(int(hi), max(int(lo), int(round(float(value)))))
            else:
                value = min(hi, max(lo, float(value)))
        elif name == "voice_mode":
            value = "legacy" if str(value) == "legacy" else "tx802"
        elif name.startswith("operator_ratio_"):
            value = min(32.0, max(0.125, float(value)))
        elif name.startswith(("operator_mix_", "operator_level_", "operator_velocity_", "operator_key_scale_")):
            value = min(1.0, max(0.0, float(value)))
        elif name.startswith("operator_detune_"):
            value = min(100.0, max(-100.0, float(value)))
        elif name.startswith("operator_pan_") or (name.startswith("part_") and name.endswith("_pan")):
            value = min(1.0, max(-1.0, float(value)))
        elif name.startswith("operator_fixed_freq_"):
            value = min(12000.0, max(0.1, float(value)))
        elif name.startswith("operator_eg_rate_"):
            value = min(99.0, max(0.0, float(value)))
        elif name.startswith("operator_eg_level_") or (name.startswith("part_") and name.endswith("_level")):
            value = min(1.0, max(0.0, float(value)))
        elif name.startswith("part_") and name.endswith("_detune"):
            value = min(1200.0, max(-1200.0, float(value)))
        elif name.startswith("part_") and name.endswith(("_note_low", "_note_high")):
            value = min(127, max(0, int(round(float(value)))))
        elif name.startswith("part_") and name.endswith(("_polyphony_reserve", "_channel")):
            value = max(0, int(round(float(value))))
        elif name == "scale_root":
            value = min(127, max(0, int(value)))
        elif name in {"enabled", "lfo_sync", "quantize_pitch"} or name.startswith(("operator_fixed_", "operator_enabled_")):
            value = bool(value)
        elif name in {"show_energy", "show_chaos"}:
            value = min(1.0, max(0.0, float(value)))

        with self._lock:
            self.params[name] = value
            if name.startswith("operator_mix_"):
                index = name.rsplit("_", 1)[1]
                self.params[f"operator_level_{index}"] = value
                self._smoothed[name] = float(value)
                self._smoothed[f"operator_level_{index}"] = float(value)
            elif name.startswith("operator_level_"):
                index = name.rsplit("_", 1)[1]
                self.params[f"operator_mix_{index}"] = value
                self._smoothed[name] = float(value)
                self._smoothed[f"operator_mix_{index}"] = float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._smoothed.setdefault(name, float(value))
            if name in {"attack", "decay", "sustain", "release"}:
                self.envelope.configure(**{name: value})
                self._apply_adsr_compatibility_locked()

    def get_param(self, name):
        with self._lock:
            if name not in self.params:
                raise KeyError(f"Unknown FM parameter: {name}")
            return self.params[name]

    def trigger_note(self, note=None, velocity=1.0):
        with self._lock:
            if note is not None:
                note = float(note)
                if self.params["quantize_pitch"]:
                    note = quantize_midi(note, self.params["scale_root"], self.params["scale_mode"])
                self._note = note
                self.params["carrier_freq"] = midi_to_frequency(note)
            elif self.params["quantize_pitch"]:
                self.params["carrier_freq"] = quantize_frequency(
                    self.params["carrier_freq"],
                    self.params["scale_root"],
                    self.params["scale_mode"],
                )
                note = 69.0 + 12.0 * math.log2(self.params["carrier_freq"] / 440.0)
                self._note = note
            else:
                note = 69.0 + 12.0 * math.log2(self.params["carrier_freq"] / 440.0)
                self._note = note
            self._velocity = min(1.0, max(0.0, float(velocity)))
            self._panic = False
            self.envelope.trigger()
            self._allocate_voice(float(note), self._velocity, self._part_for_note(float(note)))

    def release_note(self):
        with self._lock:
            self.envelope.release_gate()
            for envelopes in self._voice_envelopes:
                for envelope in envelopes:
                    envelope.release_gate()

    def panic(self):
        with self._lock:
            self._panic = True
            self.envelope.reset()
            self._voice_phase.fill(0.0)
            self._voice_feedback.fill(0.0)
            self._voice_note[:] = np.nan
            self._operator_energy.fill(0.0)
            self._part_rms.fill(0.0)
            for envelopes in self._voice_envelopes:
                for envelope in envelopes:
                    envelope.reset()
            for name in self._analysis:
                if name != "beat_phase":
                    self._analysis[name] = 0.0

    def _apply_adsr_compatibility_locked(self):
        attack = float(self.params["attack"])
        decay = float(self.params["decay"])
        sustain = float(self.params["sustain"])
        release = float(self.params["release"])
        rate1 = self._seconds_to_rate(attack)
        rate2 = self._seconds_to_rate(decay)
        rate4 = self._seconds_to_rate(release)
        for op in range(1, OPERATOR_COUNT + 1):
            if op == 1:
                self.params[f"operator_eg_rate_{op}_1"] = rate1
                self.params[f"operator_eg_rate_{op}_2"] = rate2
                self.params[f"operator_eg_rate_{op}_3"] = 0.0
                self.params[f"operator_eg_rate_{op}_4"] = rate4
                self.params[f"operator_eg_level_{op}_1"] = 1.0
                self.params[f"operator_eg_level_{op}_2"] = sustain
                self.params[f"operator_eg_level_{op}_3"] = sustain
                self.params[f"operator_eg_level_{op}_4"] = 0.0

    @staticmethod
    def _seconds_to_rate(seconds):
        seconds = max(0.001, float(seconds))
        normalized = 1.0 - math.sqrt(max(0.0, min(1.0, (seconds - 0.005) / 2.0)))
        return min(99.0, max(1.0, normalized * 99.0))

    def _part_for_note(self, note):
        for part in range(1, PART_COUNT + 1):
            level = float(self.params[f"part_{part}_level"])
            low = int(self.params[f"part_{part}_note_low"])
            high = int(self.params[f"part_{part}_note_high"])
            if level > 0.0 and low <= note <= high:
                return part - 1
        return 0

    def _allocate_voice(self, note, velocity, part):
        active = self._active_voice_indices()
        limit = int(self.params["polyphony"])
        if active.size < limit:
            inactive = np.where(~self._voice_active_mask())[0]
            voice = int(inactive[0]) if inactive.size else int(np.argmin(self._voice_age))
        else:
            active_ages = self._voice_age[active]
            voice = int(active[int(np.argmin(active_ages))])
        self._trigger_counter += 1
        self._voice_note[voice] = note
        self._voice_velocity[voice] = velocity
        self._voice_part[voice] = int(part)
        self._voice_age[voice] = self._trigger_counter
        self._voice_phase[voice].fill(0.0)
        self._voice_feedback[voice].fill(0.0)
        self._configure_voice_envelopes(voice)
        for envelope in self._voice_envelopes[voice]:
            envelope.trigger()

    def _configure_voice_envelopes(self, voice):
        for op in range(1, OPERATOR_COUNT + 1):
            rates = tuple(self.params[f"operator_eg_rate_{op}_{stage}"] for stage in range(1, 5))
            levels = tuple(self.params[f"operator_eg_level_{op}_{stage}"] for stage in range(1, 5))
            self._voice_envelopes[voice][op - 1].configure(rates=rates, levels=levels)

    def _voice_active_mask(self):
        return np.array(
            [any(envelope.active for envelope in envelopes) for envelopes in self._voice_envelopes],
            dtype=bool,
        )

    def _active_voice_indices(self):
        active = self._voice_active_mask()
        return np.where(active)[0]

    def _snapshot_params(self):
        with self._lock:
            return dict(self.params), self._panic

    def _update_smoothed(self, params, frames):
        alpha = 1.0 - math.exp(-max(1, frames) / max(1.0, self.smoothing_seconds * self.sample_rate))
        for name, target in params.items():
            if name not in self._smoothed or isinstance(target, bool):
                continue
            self._smoothed[name] += (float(target) - self._smoothed[name]) * alpha

    @staticmethod
    def _pan_gains(pan):
        angle = (min(1.0, max(-1.0, float(pan))) + 1.0) * math.pi * 0.25
        return math.cos(angle), math.sin(angle)

    def render(self, frames=None, out=None):
        frames = self.block_size if frames is None else max(1, int(frames))
        if out is None:
            out = np.zeros((frames, 2), dtype=np.float32)
        else:
            out[:frames].fill(0.0)

        params, panic = self._snapshot_params()
        active_voices = self._active_voice_indices()
        if panic or not params["enabled"] or active_voices.size == 0:
            self._audio_clock_samples += frames
            self._update_clock_analysis(params)
            return out

        self._update_smoothed(params, frames)
        algorithm = self._coerce_algorithm(params["algorithm"])
        targets = ALGORITHMS_6OP[algorithm - 1]
        feedback_operator = TX802_FEEDBACK_OPERATORS[algorithm - 1]
        base_frequency = float(self._smoothed["carrier_freq"])
        global_detune = float(self._smoothed["detune"])
        modulation_index = float(self._smoothed["modulation_index"])
        feedback = float(self._smoothed["feedback"])
        master_gain = min(0.85, float(self._smoothed["master_gain"]))
        lfo_rate = (
            float(self._smoothed["tempo_bpm"]) / 60.0 / max(0.125, float(self._smoothed["lfo_division"]))
            if params["lfo_sync"]
            else float(self._smoothed["lfo_rate"])
        )
        operator_energy = np.zeros(OPERATOR_COUNT, dtype=np.float64)
        part_energy = np.zeros(PART_COUNT, dtype=np.float64)
        part_samples = np.zeros(PART_COUNT, dtype=np.float64)
        feedback_sum = 0.0
        dynamic_index = modulation_index

        ratios = np.array([self._smoothed[f"operator_ratio_{i}"] for i in range(1, OPERATOR_COUNT + 1)])
        detunes = np.array([self._smoothed[f"operator_detune_{i}"] for i in range(1, OPERATOR_COUNT + 1)])
        levels = np.array([self._smoothed[f"operator_level_{i}"] for i in range(1, OPERATOR_COUNT + 1)])
        pans = np.array([self._smoothed[f"operator_pan_{i}"] for i in range(1, OPERATOR_COUNT + 1)])
        velocity_depth = np.array([self._smoothed[f"operator_velocity_{i}"] for i in range(1, OPERATOR_COUNT + 1)])
        fixed_freqs = np.array([self._smoothed[f"operator_fixed_freq_{i}"] for i in range(1, OPERATOR_COUNT + 1)])
        fixed = np.array([bool(params[f"operator_fixed_{i}"]) for i in range(1, OPERATOR_COUNT + 1)])
        enabled = np.array([bool(params[f"operator_enabled_{i}"]) for i in range(1, OPERATOR_COUNT + 1)])
        carrier_mask = np.array([1.0 if target < 0 else 0.0 for target in targets], dtype=np.float64)
        voice_norm = max(1.0, math.sqrt(float(active_voices.size)))

        for frame in range(frames):
            lfo = math.sin(self._lfo_phase)
            self._lfo_phase = (self._lfo_phase + 2.0 * math.pi * lfo_rate / self.sample_rate) % (2.0 * math.pi)
            dynamic_index = modulation_index * (1.0 + lfo * float(self._smoothed["lfo_depth"]))

            for voice in active_voices:
                envelopes = self._voice_envelopes[int(voice)]
                if not any(envelope.active for envelope in envelopes):
                    self._voice_note[int(voice)] = np.nan
                    continue
                note = self._voice_note[int(voice)]
                if not np.isfinite(note):
                    continue
                part = int(self._voice_part[int(voice)])
                part_level = float(params.get(f"part_{part + 1}_level", 1.0))
                if part_level <= 0.0:
                    continue
                part_detune = float(params.get(f"part_{part + 1}_detune", 0.0))
                part_pan = float(params.get(f"part_{part + 1}_pan", 0.0))
                velocity = float(self._voice_velocity[int(voice)])
                velocity_gain = (1.0 - float(self._smoothed["velocity_sensitivity"])) + (
                    float(self._smoothed["velocity_sensitivity"]) * velocity
                )
                note_frequency = midi_to_frequency(note)
                if note == self._note:
                    note_frequency = base_frequency
                frequencies = np.where(
                    fixed,
                    fixed_freqs,
                    note_frequency * ratios * np.power(2.0, (global_detune + part_detune + detunes) / 1200.0),
                )
                frequencies = np.minimum(frequencies, self.sample_rate * 0.45)
                phase_steps = 2.0 * math.pi * frequencies / self.sample_rate
                op_values = np.zeros(OPERATOR_COUNT, dtype=np.float64)
                op_env = np.array([envelope.next_value() for envelope in envelopes], dtype=np.float64)
                op_env *= enabled.astype(np.float64)
                op_levels = levels * op_env * ((1.0 - velocity_depth) + velocity_depth * velocity)

                for op in range(OPERATOR_COUNT - 1, -1, -1):
                    modulation = 0.0
                    for source, target in enumerate(targets):
                        if target == op:
                            modulation += op_values[source]
                    if op == feedback_operator:
                        modulation += self._voice_feedback[int(voice), op] * feedback
                    depth = dynamic_index * (0.28 + 0.1 * op)
                    op_values[op] = math.sin(self._voice_phase[int(voice), op] + modulation * depth) * op_levels[op]

                left = 0.0
                right = 0.0
                audible_weight = 0.0
                for op in range(OPERATOR_COUNT):
                    if carrier_mask[op] <= 0.0:
                        continue
                    gain_l, gain_r = self._pan_gains(min(1.0, max(-1.0, pans[op] + part_pan)))
                    left += op_values[op] * gain_l
                    right += op_values[op] * gain_r
                    audible_weight += abs(op_levels[op])
                normalization = max(1.0, audible_weight)
                amplitude = master_gain * part_level * velocity_gain / voice_norm / normalization
                sample_l = left * amplitude
                sample_r = right * amplitude
                out[frame, 0] += sample_l
                out[frame, 1] += sample_r
                part_energy[part] += (sample_l + sample_r) * (sample_l + sample_r) * 0.25
                part_samples[part] += 1.0
                operator_energy += op_values * op_values
                feedback_sum += abs(float(self._voice_feedback[int(voice), feedback_operator]))
                self._voice_feedback[int(voice)] = op_values
                self._voice_phase[int(voice)] = np.mod(
                    self._voice_phase[int(voice)] + phase_steps,
                    2.0 * math.pi,
                )
                if not any(envelope.active for envelope in envelopes):
                    self._voice_note[int(voice)] = np.nan

        np.tanh(out[:frames] * 1.35, out=out[:frames])
        out[:frames] /= np.float32(math.tanh(1.35))
        np.clip(out[:frames], -1.0, 1.0, out=out[:frames])
        np.nan_to_num(out[:frames], copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        self._operator_energy[:] = operator_energy / max(1, frames * max(1, active_voices.size))
        self._part_rms[:] = np.sqrt(part_energy / np.maximum(1.0, part_samples))
        self._audio_clock_samples += frames
        self._update_analysis(out[:frames], dynamic_index, feedback_sum, params)
        return out

    def _update_clock_analysis(self, params):
        bpm = float(params.get("tempo_bpm", 120.0))
        seconds = self._audio_clock_samples / float(self.sample_rate)
        self._analysis["beat_phase"] = (seconds * bpm / 60.0) % 1.0

    def _update_analysis(self, audio, dynamic_index, feedback_sum, params):
        mono = audio.mean(axis=1, dtype=np.float64)
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        spectrum = np.abs(np.fft.rfft(mono))
        if spectrum.size > 1 and float(spectrum.sum()) > 1e-12:
            frequencies = np.fft.rfftfreq(mono.size, 1.0 / self.sample_rate)
            brightness = float(np.sum(spectrum * frequencies) / np.sum(spectrum) / (self.sample_rate * 0.5))
        else:
            brightness = 0.0
        transient = min(1.0, max(0.0, (rms - self._previous_rms) * 12.0))
        self._previous_rms = rms
        active_count = int(self._active_voice_indices().size)
        self._analysis.update(
            rms=min(1.0, rms),
            peak=min(1.0, peak),
            spectral_brightness=min(1.0, max(0.0, brightness)),
            transient=transient,
            modulation_activity=min(1.0, abs(float(dynamic_index)) / 24.0),
            voice_count=min(1.0, active_count / float(MAX_POLYPHONY)),
            feedback_activity=min(1.0, feedback_sum / max(1.0, active_count * audio.shape[0])),
        )
        for index, energy in enumerate(self._operator_energy, 1):
            self._analysis[f"operator_{index}_energy"] = min(1.0, max(0.0, float(energy)))
        for index, rms_value in enumerate(self._part_rms, 1):
            self._analysis[f"part_{index}_rms"] = min(1.0, max(0.0, float(rms_value)))
        self._update_clock_analysis(params)

    def get_output(self, name):
        if name not in self._analysis:
            raise KeyError(f"Unknown FM analysis output: {name}")
        return self._analysis[name]

    def get_outputs(self):
        return dict(self._analysis)

    def reset(self):
        with self._lock:
            self._voice_phase.fill(0.0)
            self._voice_feedback.fill(0.0)
            self._voice_note[:] = np.nan
            self._lfo_phase = 0.0
            self._audio_clock_samples = 0
            self._previous_rms = 0.0
            self._panic = False
            self.envelope.reset()
            for envelopes in self._voice_envelopes:
                for envelope in envelopes:
                    envelope.reset()
