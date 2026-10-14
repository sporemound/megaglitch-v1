"""Media-to-Audio step sequencing and visual randomization."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Callable


STEP_COUNT = 16
LANE_COUNT = 8


TARGET_DEFAULTS = (
    "carrier_freq",
    "modulation_index",
    "feedback",
    "algorithm",
    "poly_spread",
    "chord_depth",
    "timbre_spread",
    "ir_wet",
)


VISUAL_TARGET_HINTS = {
    "brightness": ("carrier_freq", "brightness_response", "ir_wet"),
    "motion": ("modulation_index", "motion_gating", "motion_response"),
    "entropy": ("algorithm", "timbre_spread", "poly_spread"),
    "edge_density": ("feedback", "operator_level_5", "operator_detune_6"),
    "glitch_spike": ("feedback", "modulation_index", "visual_drive"),
    "red_energy": ("operator_pan_5", "color_stereo"),
    "blue_energy": ("operator_pan_6", "color_stereo"),
    "center_mass_x": ("color_stereo", "operator_pan_1", "operator_pan_6"),
}


@dataclass
class MediaStep:
    enabled: bool = False
    value: float = 0.5
    probability: float = 1.0
    random_amount: float = 0.0


@dataclass
class MediaStepLane:
    enabled: bool = False
    target: str = "modulation_index"
    min_value: float = 0.0
    max_value: float = 1.0
    visual_source: str = "brightness"
    visual_amount: float = 0.0
    random_amount: float = 0.0
    smoothing_ms: float = 80.0
    probability: float = 1.0
    curve: str = "linear"
    steps: list[MediaStep] = field(default_factory=lambda: [MediaStep() for _ in range(STEP_COUNT)])
    current_value: float | None = None


class MediaAudioSequencer:
    """Main-thread parameter step sequencer for MediaAudioEngine."""

    def __init__(self, param_limits: dict[str, tuple[float, float]], set_param: Callable[[str, object], None]):
        self.param_limits = dict(param_limits)
        self.set_param = set_param
        self.enabled = False
        self.playing = False
        self.bpm = 90.0
        self.step_count = STEP_COUNT
        self.step_index = 0
        self.random_strength = 0.2
        self.chaos_ceiling = 0.45
        self.visual_mode = "assist"
        self.freeze_randomization = False
        self.clock_source = "internal"
        self.last_midi_status = ""
        self._last_step_time = time.monotonic()
        self._midi_clock_counter = 0
        self._last_algorithm_step_time = 0.0
        self.lanes = [self._default_lane(index) for index in range(LANE_COUNT)]

    def _default_lane(self, index: int) -> MediaStepLane:
        target = TARGET_DEFAULTS[index % len(TARGET_DEFAULTS)]
        low, high = self.param_limits.get(target, (0.0, 1.0))
        lane = MediaStepLane(
            enabled=index < 4,
            target=target,
            min_value=float(low),
            max_value=float(high),
            visual_source=("brightness", "motion", "entropy", "edge_density")[index % 4],
            visual_amount=0.25 if index < 4 else 0.0,
            random_amount=0.08 if index < 4 else 0.0,
            probability=0.85,
        )
        for step_index, step in enumerate(lane.steps):
            step.enabled = step_index % (4 if index != 1 else 2) == 0
            step.value = step_index / max(1, STEP_COUNT - 1)
        return lane

    def target_bounds(self, target: str) -> tuple[float, float]:
        return tuple(map(float, self.param_limits.get(target, (0.0, 1.0))))

    def set_lane_target(self, lane_index: int, target: str) -> None:
        lane = self.lanes[int(lane_index)]
        lane.target = str(target)
        lane.min_value, lane.max_value = self.target_bounds(lane.target)
        lane.current_value = None

    def update_lane(self, lane_index: int, field_name: str, value) -> None:
        lane = self.lanes[int(lane_index)]
        if field_name == "target":
            self.set_lane_target(lane_index, str(value))
            return
        if field_name == "enabled":
            lane.enabled = bool(value)
        elif field_name in {"min_value", "max_value", "visual_amount", "random_amount", "smoothing_ms", "probability"}:
            setattr(lane, field_name, float(value))
        elif field_name in {"visual_source", "curve"}:
            setattr(lane, field_name, str(value))
        self._normalize_lane(lane)

    def update_step(self, lane_index: int, step_index: int, field_name: str, value) -> None:
        step = self.lanes[int(lane_index)].steps[int(step_index) % self.step_count]
        if field_name == "enabled":
            step.enabled = bool(value)
        elif field_name in {"value", "probability", "random_amount"}:
            setattr(step, field_name, max(0.0, min(1.0, float(value))))

    def toggle_step(self, lane_index: int, step_index: int) -> bool:
        step = self.lanes[int(lane_index)].steps[int(step_index) % self.step_count]
        step.enabled = not step.enabled
        return step.enabled

    def play(self) -> None:
        self.enabled = True
        self.playing = True
        self._last_step_time = time.monotonic()

    def stop(self) -> None:
        self.playing = False

    def reset(self) -> None:
        self.step_index = 0
        self._last_step_time = time.monotonic()

    def tick(self, metrics: dict[str, float] | None = None, now: float | None = None) -> bool:
        if not self.enabled or not self.playing:
            return False
        now = time.monotonic() if now is None else float(now)
        if self.clock_source == "beatstep":
            return False
        seconds = 60.0 / max(20.0, min(300.0, self.bpm)) / 4.0
        if now - self._last_step_time < seconds:
            self.apply_current_step(metrics or {}, now, advance=False)
            return False
        self._last_step_time = now
        self.advance(metrics or {}, now)
        return True

    def advance(self, metrics: dict[str, float] | None = None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.apply_current_step(metrics or {}, now, advance=True)
        self.step_index = (self.step_index + 1) % self.step_count

    def apply_current_step(self, metrics: dict[str, float], now: float, advance: bool) -> dict[str, float]:
        applied = {}
        for lane in self.lanes:
            if not lane.enabled or lane.target not in self.param_limits:
                continue
            step = lane.steps[self.step_index]
            if not step.enabled:
                continue
            if advance and random.random() > max(0.0, min(1.0, lane.probability * step.probability)):
                continue
            normalized = self._curve(step.value, lane.curve)
            visual = max(0.0, min(1.0, float(metrics.get(lane.visual_source, 0.0))))
            normalized = normalized * (1.0 - lane.visual_amount) + visual * lane.visual_amount
            random_amount = 0.0 if self.freeze_randomization else min(self.chaos_ceiling, lane.random_amount + step.random_amount)
            if random_amount > 0.0:
                normalized += random.uniform(-random_amount, random_amount) * self.random_strength
            value = self._lane_value(lane, normalized, now)
            self.set_param(lane.target, value)
            applied[lane.target] = float(value)
        return applied

    def mutate_from_visual(self, metrics: dict[str, float], strength: float | None = None) -> None:
        amount = self.random_strength if strength is None else max(0.0, min(1.0, float(strength)))
        if self.freeze_randomization:
            return
        dominant = sorted(metrics.items(), key=lambda item: float(item[1]), reverse=True)
        hinted_targets = []
        for name, _ in dominant[:3]:
            hinted_targets.extend(VISUAL_TARGET_HINTS.get(name, ()))
        for lane_index, lane in enumerate(self.lanes):
            if hinted_targets and lane_index < len(hinted_targets):
                target = hinted_targets[lane_index]
                if target in self.param_limits:
                    self.set_lane_target(lane_index, target)
            lane.enabled = True
            lane.visual_source = dominant[lane_index % len(dominant)][0] if dominant else "brightness"
            lane.visual_amount = min(1.0, 0.15 + amount * 0.65)
            lane.random_amount = min(self.chaos_ceiling, 0.04 + amount * 0.3)
            for step_index, step in enumerate(lane.steps):
                pulse = float(metrics.get(lane.visual_source, 0.5))
                step.enabled = random.random() < (0.25 + pulse * 0.55)
                step.value = max(0.0, min(1.0, pulse * 0.6 + random.random() * 0.4))
                step.random_amount = min(self.chaos_ceiling, amount * 0.2)

    def handle_midi(self, msg, metrics: dict[str, float] | None = None) -> bool:
        msg_type = getattr(msg, "type", "")
        self.last_midi_status = str(msg)
        if msg_type == "start":
            self.clock_source = "beatstep"
            self.reset()
            self.play()
            return True
        if msg_type == "stop":
            self.stop()
            return True
        if msg_type == "continue":
            self.play()
            return True
        if msg_type == "clock" and self.clock_source == "beatstep":
            self._midi_clock_counter = (self._midi_clock_counter + 1) % 6
            if self._midi_clock_counter == 0:
                self.advance(metrics or {}, time.monotonic())
            return True
        if msg_type in {"note_on", "note_off"}:
            note = int(getattr(msg, "note", 0))
            velocity = int(getattr(msg, "velocity", 0))
            if msg_type == "note_on" and velocity > 0 and 36 <= note < 36 + self.step_count:
                self.toggle_step(0, note - 36)
                return True
            if msg_type == "note_on" and velocity > 0 and 52 <= note < 52 + LANE_COUNT:
                lane = self.lanes[note - 52]
                lane.enabled = not lane.enabled
                return True
        if msg_type == "control_change":
            control = int(getattr(msg, "control", -1))
            value = int(getattr(msg, "value", 0)) / 127.0
            return self._handle_cc(control, value)
        return False

    def _handle_cc(self, control: int, value: float) -> bool:
        if control == 10:
            self.random_strength = value
            return True
        if control == 11:
            self.chaos_ceiling = value
            return True
        if control == 12:
            self.bpm = 40.0 + value * 220.0
            return True
        if 20 <= control < 20 + LANE_COUNT:
            lane = self.lanes[control - 20]
            lane.visual_amount = value
            return True
        if 30 <= control < 30 + LANE_COUNT:
            lane = self.lanes[control - 30]
            lane.random_amount = min(self.chaos_ceiling, value)
            return True
        if 40 <= control < 40 + LANE_COUNT:
            lane = self.lanes[control - 40]
            step = lane.steps[self.step_index]
            step.value = value
            return True
        return False

    def _lane_value(self, lane: MediaStepLane, normalized: float, now: float):
        normalized = max(0.0, min(1.0, normalized))
        low, high = sorted((float(lane.min_value), float(lane.max_value)))
        value = low + normalized * (high - low)
        if lane.target in {"algorithm", "voice_count"}:
            if lane.target == "algorithm" and now - self._last_algorithm_step_time < 0.75:
                return int(round(lane.current_value if lane.current_value is not None else value))
            if lane.target == "algorithm":
                self._last_algorithm_step_time = now
            lane.current_value = int(round(value))
            return int(round(value))
        if lane.current_value is None or lane.smoothing_ms <= 0:
            lane.current_value = value
        else:
            alpha = 1.0 - math.exp(-0.016 / max(0.001, lane.smoothing_ms * 0.001))
            lane.current_value += (value - lane.current_value) * alpha
        return float(lane.current_value)

    @staticmethod
    def _curve(value: float, curve: str) -> float:
        value = max(0.0, min(1.0, float(value)))
        if curve in {"smoothstep", "ease_in_out"}:
            return value * value * (3.0 - 2.0 * value)
        if curve == "exponential":
            return value * value
        return value

    def _normalize_lane(self, lane: MediaStepLane) -> None:
        lane.visual_amount = max(0.0, min(1.0, lane.visual_amount))
        lane.random_amount = max(0.0, min(1.0, lane.random_amount))
        lane.probability = max(0.0, min(1.0, lane.probability))
        lane.smoothing_ms = max(0.0, min(5000.0, lane.smoothing_ms))
        if lane.target in self.param_limits:
            low, high = self.target_bounds(lane.target)
            lane.min_value = max(min(low, high), min(max(low, high), lane.min_value))
            lane.max_value = max(min(low, high), min(max(low, high), lane.max_value))

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "playing": self.playing,
            "bpm": self.bpm,
            "step_index": self.step_index,
            "random_strength": self.random_strength,
            "chaos_ceiling": self.chaos_ceiling,
            "freeze_randomization": self.freeze_randomization,
            "clock_source": self.clock_source,
            "last_midi_status": self.last_midi_status,
        }
