from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from megaglitch_midi_router import MidiDeviceManager, make_message
from megaglitch_visual_bus import VISUAL_METRIC_NAMES, VisualBus


MAX_LANES = 8
MAX_NOTES_PER_TICK = 8
MAX_ACTIVE_NOTES_PER_LANE = 8
MIN_NOTE_INTERVAL_MS = 35
MAX_CC_SLOTS = 16

SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "pentatonic": (0, 2, 4, 7, 9),
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "chromatic": tuple(range(12)),
}

STEP_DIVISIONS = {
    "1/1": 4.0,
    "1/2": 2.0,
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,
}

LANE_MODES = ("mono", "poly", "chord", "drone", "percussion")


@dataclass
class CCSlotConfig:
    enabled: bool = False
    number: int = 74
    source: str = "contrast"


@dataclass
class LaneConfig:
    enabled: bool = False
    output_port: str = ""
    midi_channel: int = 1
    mode: str = "mono"
    root_note: int = 48
    scale: str = "minor"
    min_note: int = 36
    max_note: int = 72
    step_division: str = "1/8"
    gate_length: float = 0.45
    probability: float = 0.55
    velocity_min: int = 42
    velocity_max: int = 110
    pitch_source: str = "brightness"
    velocity_source: str = "motion"
    probability_source: str = "entropy"
    visual_amount: float = 0.65
    mutation_amount: float = 0.08
    cc_enabled: bool = False
    cc_number: int = 74
    cc_source: str = "contrast"
    cc_slots: List[CCSlotConfig] = field(default_factory=lambda: [CCSlotConfig() for _ in range(MAX_CC_SLOTS)])


@dataclass
class SequencerConfig:
    enabled: bool = False
    playing: bool = False
    bpm: float = 120.0
    swing: float = 0.0
    humanize_ms: float = 0.0
    global_density: float = 1.0
    global_mutation: float = 0.0
    lanes: List[LaneConfig] = field(default_factory=lambda: [LaneConfig() for _ in range(MAX_LANES)])


class LiveSequencer:
    def __init__(self, visual_bus: VisualBus, midi_manager: MidiDeviceManager):
        self.visual_bus = visual_bus
        self.midi_manager = midi_manager
        self.config = SequencerConfig()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active_notes: List[Set[Tuple[str, int, int]]] = [set() for _ in range(MAX_LANES)]
        self._note_off_deadlines: List[List[Tuple[float, str, int, int]]] = [[] for _ in range(MAX_LANES)]
        self._last_note_time: List[float] = [0.0 for _ in range(MAX_LANES)]
        self._lane_next_tick: List[float] = [0.0 for _ in range(MAX_LANES)]
        self._tick_index = 0
        self.status_message = "Stopped"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        now = time.monotonic()
        with self._lock:
            self._lane_next_tick = [now for _ in range(MAX_LANES)]
        self._thread = threading.Thread(target=self._run, name="LiveSequencerClock", daemon=True)
        self._thread.start()
        logging.info("Live sequencer clock started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self.panic_all()
        self.status_message = "Stopped"

    def play(self) -> None:
        with self._lock:
            self.config.enabled = True
            self.config.playing = True
            now = time.monotonic()
            self._lane_next_tick = [now for _ in range(MAX_LANES)]
        self.status_message = "Playing"

    def pause(self) -> None:
        with self._lock:
            self.config.playing = False
        self.panic_all()
        self.status_message = "Paused"

    def reset(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tick_index = 0
            self._lane_next_tick = [now for _ in range(MAX_LANES)]
        self.panic_all()
        logging.info("Live sequencer reset")

    def test_lane_note(self, lane_index: int, duration: float = 1.0) -> bool:
        if not 0 <= lane_index < MAX_LANES:
            logging.warning("Live sequencer test note ignored for invalid lane %s", lane_index)
            return False
        lane = self.get_lane_config(lane_index)
        logging.info(
            "Live sequencer test note requested: lane=%d output=%r channel=%d",
            lane_index + 1,
            lane.output_port,
            lane.midi_channel,
        )
        if not lane.output_port:
            self.status_message = "Select a MIDI output before testing"
            logging.warning("Live sequencer test note skipped: no MIDI output selected")
            return False

        note = max(0, min(127, int(lane.root_note)))
        velocity = max(1, min(127, int(lane.velocity_max)))
        channel = midi_channel(lane.midi_channel)
        note_on = make_message("note_on", note=note, velocity=velocity, channel=channel)
        if note_on is None:
            self.status_message = "MIDI backend unavailable"
            logging.warning("Live sequencer test note failed: MIDI backend unavailable")
            return False
        if not self.midi_manager.send(lane.output_port, note_on):
            self.status_message = f"Test note failed: {lane.output_port}"
            logging.warning("Live sequencer test note failed on %s", lane.output_port)
            return False

        deadline = time.monotonic() + max(0.03, min(2.0, float(duration)))
        with self._lock:
            self._active_notes[lane_index].add((lane.output_port, lane.midi_channel, note))
            self._note_off_deadlines[lane_index].append((deadline, lane.output_port, lane.midi_channel, note))
        self.status_message = f"Sent test note to {lane.output_port}"
        logging.info("Live sequencer sent test note to %s on channel %d", lane.output_port, lane.midi_channel)
        return True

    def test_output_port(self, port_name: str, channel: int = 1, note: int = 60, velocity: int = 120, duration: float = 1.0) -> bool:
        port_name = str(port_name or "")
        logging.info("Live sequencer direct output test requested: output=%r channel=%d", port_name, channel)
        if not port_name or port_name == "No MIDI outputs":
            self.status_message = "Select a MIDI output before testing"
            logging.warning("Live sequencer direct output test skipped: no MIDI output selected")
            return False
        note_on = make_message(
            "note_on",
            note=max(0, min(127, int(note))),
            velocity=max(1, min(127, int(velocity))),
            channel=midi_channel(channel),
        )
        note_off = make_message(
            "note_off",
            note=max(0, min(127, int(note))),
            velocity=0,
            channel=midi_channel(channel),
        )
        if note_on is None or note_off is None:
            self.status_message = "MIDI backend unavailable"
            logging.warning("Live sequencer direct output test failed: MIDI backend unavailable")
            return False
        if not self.midi_manager.send(port_name, note_on):
            self.status_message = f"Test note failed: {port_name}"
            logging.warning("Live sequencer direct output test failed on %s", port_name)
            return False

        def send_off() -> None:
            self.midi_manager.send(port_name, note_off)

        timer = threading.Timer(max(0.03, min(2.0, float(duration))), send_off)
        timer.daemon = True
        timer.start()
        self.status_message = f"Sent test note to {port_name}"
        logging.info("Live sequencer sent direct test note to %s on channel %d", port_name, channel)
        return True

    def update_global(self, name: str, value) -> None:
        should_panic = False
        with self._lock:
            if hasattr(self.config, name):
                if name in ("enabled", "playing"):
                    if name in ("enabled", "playing") and getattr(self.config, name) and not bool(value):
                        should_panic = True
                    setattr(self.config, name, bool(value))
                elif name == "bpm":
                    setattr(self.config, name, max(20.0, min(300.0, float(value))))
                else:
                    setattr(self.config, name, float(value))
        if should_panic:
            self.panic_all()

    def update_lane(self, lane_index: int, name: str, value) -> None:
        if not 0 <= lane_index < MAX_LANES:
            return
        with self._lock:
            lane = self.config.lanes[lane_index]
            old_port = lane.output_port
            old_enabled = lane.enabled
            if hasattr(lane, name):
                current = getattr(lane, name)
                if isinstance(current, bool):
                    setattr(lane, name, bool(value))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(lane, name, int(value))
                elif isinstance(current, float):
                    setattr(lane, name, float(value))
                else:
                    setattr(lane, name, str(value))
            self._normalize_lane(lane)
            port_changed = name == "output_port" and old_port != lane.output_port
            disabled = name == "enabled" and old_enabled and not lane.enabled
        if port_changed or disabled or name in ("midi_channel", "mode"):
            self.panic_lane(lane_index)

    def apply_lane_preset(self, lane_index: int, preset: Dict[str, object]) -> None:
        if not 0 <= lane_index < MAX_LANES:
            return
        with self._lock:
            lane = self.config.lanes[lane_index]
            for key, value in preset.items():
                if hasattr(lane, key):
                    setattr(lane, key, value)
            self._normalize_lane(lane)
        self.panic_lane(lane_index)

    def update_lane_cc_slot(self, lane_index: int, slot_index: int, name: str, value) -> None:
        if not 0 <= lane_index < MAX_LANES or not 0 <= slot_index < MAX_CC_SLOTS:
            return
        with self._lock:
            lane = self.config.lanes[lane_index]
            self._ensure_cc_slots(lane)
            slot = lane.cc_slots[slot_index]
            if hasattr(slot, name):
                current = getattr(slot, name)
                if isinstance(current, bool):
                    setattr(slot, name, bool(value))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(slot, name, int(value))
                else:
                    setattr(slot, name, str(value))
            self._normalize_cc_slot(slot)
            if slot_index == 0:
                lane.cc_enabled = slot.enabled
                lane.cc_number = slot.number
                lane.cc_source = slot.source

    def send_cc_slot(self, lane_index: int, slot_index: int, value: int = 127) -> bool:
        if not 0 <= lane_index < MAX_LANES or not 0 <= slot_index < MAX_CC_SLOTS:
            logging.warning("Live sequencer CC slot send ignored: lane=%s slot=%s", lane_index, slot_index)
            return False
        lane = self.get_lane_config(lane_index)
        if not lane.output_port:
            self.status_message = "Select a MIDI output before sending CC"
            logging.warning("Live sequencer CC slot send skipped: no MIDI output selected")
            return False
        slot = lane.cc_slots[slot_index]
        self._normalize_cc_slot(slot)
        msg = make_message(
            "control_change",
            control=max(0, min(127, int(slot.number))),
            value=max(0, min(127, int(value))),
            channel=midi_channel(lane.midi_channel),
        )
        if msg is None:
            self.status_message = "MIDI backend unavailable"
            logging.warning("Live sequencer CC slot send failed: MIDI backend unavailable")
            return False
        if not self.midi_manager.send(lane.output_port, msg):
            self.status_message = f"CC send failed: {lane.output_port}"
            logging.warning("Live sequencer CC slot send failed on %s", lane.output_port)
            return False
        self.status_message = f"Sent CC {slot.number} from lane {lane_index + 1} slot {slot_index + 1}"
        logging.info(
            "Live sequencer sent CC slot: lane=%d slot=%d output=%s cc=%d value=%d channel=%d",
            lane_index + 1,
            slot_index + 1,
            lane.output_port,
            slot.number,
            value,
            lane.midi_channel,
        )
        return True

    def get_lane_config(self, lane_index: int) -> LaneConfig:
        with self._lock:
            src = self.config.lanes[lane_index]
            return clone_lane_config(src)

    def get_config_snapshot(self) -> SequencerConfig:
        with self._lock:
            return SequencerConfig(
                enabled=self.config.enabled,
                playing=self.config.playing,
                bpm=self.config.bpm,
                swing=self.config.swing,
                humanize_ms=self.config.humanize_ms,
                global_density=self.config.global_density,
                global_mutation=self.config.global_mutation,
                lanes=[clone_lane_config(lane) for lane in self.config.lanes],
            )

    def panic_lane(self, lane_index: int, log_event: bool = True) -> None:
        if not 0 <= lane_index < MAX_LANES:
            return
        with self._lock:
            active = list(self._active_notes[lane_index])
            self._active_notes[lane_index].clear()
            self._note_off_deadlines[lane_index].clear()
        for port_name, channel, note in active:
            self._send_note_off(port_name, channel, note)
        lane = self.get_lane_config(lane_index)
        if lane.output_port:
            self._send_all_notes_off(lane.output_port, lane.midi_channel)
        if log_event:
            logging.info("Live sequencer lane %d panic", lane_index + 1)

    def panic_all(self) -> None:
        for lane_index in range(MAX_LANES):
            self.panic_lane(lane_index, log_event=False)
        logging.info("Live sequencer MIDI panic")

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            metrics = self.visual_bus.get_latest()
            notes_this_tick = 0
            with self._lock:
                config = self.get_config_snapshot()
            if config.enabled and config.playing:
                for lane_index, lane in enumerate(config.lanes):
                    self._expire_notes(lane_index, now)
                    if not lane.enabled:
                        continue
                    if now + 0.0005 < self._lane_next_tick[lane_index]:
                        continue
                    if notes_this_tick >= MAX_NOTES_PER_TICK:
                        break
                    if self._trigger_lane(lane_index, lane, config, metrics, now):
                        notes_this_tick += 1
                    self._lane_next_tick[lane_index] = now + self._step_seconds(lane, config, lane_index)
                self._tick_index += 1
            else:
                for lane_index in range(MAX_LANES):
                    self._expire_notes(lane_index, now)
            time.sleep(0.004)

    def _trigger_lane(
        self,
        lane_index: int,
        lane: LaneConfig,
        config: SequencerConfig,
        metrics: Dict[str, float],
        now: float,
    ) -> bool:
        if not lane.output_port:
            self.status_message = "Select a MIDI output for enabled lanes"
            return False
        if (now - self._last_note_time[lane_index]) * 1000.0 < MIN_NOTE_INTERVAL_MS:
            return False

        probability_metric = metric_value(metrics, lane.probability_source)
        visual_probability = blend(lane.probability, probability_metric, lane.visual_amount)
        final_probability = clamp01(visual_probability * clamp01(config.global_density))
        if random.random() > final_probability:
            self._send_cc_if_needed(lane, metrics)
            return False

        mode = lane.mode if lane.mode in LANE_MODES else "mono"
        if mode != "mono":
            # Poly/chord/drone/percussion are selectable now; this pass uses the stable mono engine.
            mode = "mono"

        note = self._map_note(lane, metrics, config)
        velocity = self._map_velocity(lane, metrics)
        gate_seconds = max(0.01, min(self._step_seconds(lane, config, lane_index) * max(0.05, lane.gate_length), 8.0))
        channel = midi_channel(lane.midi_channel)

        if mode == "mono":
            self._stop_tracked_lane_notes(lane_index)

        with self._lock:
            if len(self._active_notes[lane_index]) >= MAX_ACTIVE_NOTES_PER_LANE:
                return False

        note_on = make_message("note_on", note=note, velocity=velocity, channel=channel)
        if note_on is None or not self.midi_manager.send(lane.output_port, note_on):
            return False
        deadline = time.monotonic() + gate_seconds
        with self._lock:
            self._active_notes[lane_index].add((lane.output_port, lane.midi_channel, note))
            self._note_off_deadlines[lane_index].append((deadline, lane.output_port, lane.midi_channel, note))
            self._last_note_time[lane_index] = now
        self._send_cc_if_needed(lane, metrics)
        self.status_message = f"Playing: lane {lane_index + 1} note {note}"
        logging.info(
            "Live sequencer sent generated note: lane=%d output=%s note=%d velocity=%d channel=%d probability=%.3f",
            lane_index + 1,
            lane.output_port,
            note,
            velocity,
            lane.midi_channel,
            final_probability,
        )
        return True

    def _map_note(self, lane: LaneConfig, metrics: Dict[str, float], config: SequencerConfig) -> int:
        source_value = metric_value(metrics, lane.pitch_source)
        base_value = 0.5
        amount = clamp01(lane.visual_amount)
        mapped_value = blend(base_value, source_value, amount)
        candidates = scale_notes(lane.root_note, lane.scale, lane.min_note, lane.max_note)
        if not candidates:
            return int(max(0, min(127, lane.root_note)))
        index = int(round(mapped_value * (len(candidates) - 1)))
        index = max(0, min(len(candidates) - 1, index))
        mutation = clamp01(lane.mutation_amount + config.global_mutation)
        if mutation > 0 and random.random() < mutation:
            jump = random.choice((-4, -3, -1, 1, 3, 4, 5, 7))
            index = max(0, min(len(candidates) - 1, index + jump))
        return int(candidates[index])

    def _map_velocity(self, lane: LaneConfig, metrics: Dict[str, float]) -> int:
        low = max(1, min(127, int(lane.velocity_min)))
        high = max(low, min(127, int(lane.velocity_max)))
        source_value = metric_value(metrics, lane.velocity_source)
        base_value = 0.65
        mapped_value = blend(base_value, source_value, lane.visual_amount)
        return int(max(1, min(127, round(low + mapped_value * (high - low)))))

    def _send_cc_if_needed(self, lane: LaneConfig, metrics: Dict[str, float]) -> None:
        if not lane.output_port:
            return
        for slot in self._normalized_cc_slots_for_send(lane):
            if not slot.enabled:
                continue
            value = int(round(metric_value(metrics, slot.source) * 127.0))
            msg = make_message(
                "control_change",
                control=max(0, min(127, int(slot.number))),
                value=max(0, min(127, value)),
                channel=midi_channel(lane.midi_channel),
            )
            if msg is not None:
                self.midi_manager.send(lane.output_port, msg)

    def _expire_notes(self, lane_index: int, now: float) -> None:
        due: List[Tuple[float, str, int, int]] = []
        with self._lock:
            pending = self._note_off_deadlines[lane_index]
            keep = []
            for item in pending:
                if item[0] <= now:
                    due.append(item)
                else:
                    keep.append(item)
            self._note_off_deadlines[lane_index] = keep
            for _, port_name, channel, note in due:
                self._active_notes[lane_index].discard((port_name, channel, note))
        for _, port_name, channel, note in due:
            self._send_note_off(port_name, channel, note)

    def _stop_tracked_lane_notes(self, lane_index: int) -> None:
        with self._lock:
            active = list(self._active_notes[lane_index])
            self._active_notes[lane_index].clear()
            self._note_off_deadlines[lane_index].clear()
        for port_name, channel, note in active:
            self._send_note_off(port_name, channel, note)

    def _send_note_off(self, port_name: str, channel: int, note: int) -> None:
        msg = make_message("note_off", note=int(note), velocity=0, channel=midi_channel(channel))
        if msg is not None:
            self.midi_manager.send(port_name, msg)

    def _send_all_notes_off(self, port_name: str, channel: int) -> None:
        msg = make_message("control_change", control=123, value=0, channel=midi_channel(channel))
        if msg is not None:
            self.midi_manager.send(port_name, msg)

    def _step_seconds(self, lane: LaneConfig, config: SequencerConfig, lane_index: int) -> float:
        beat_seconds = 60.0 / max(20.0, min(300.0, float(config.bpm)))
        beats = STEP_DIVISIONS.get(lane.step_division, 0.5)
        seconds = beat_seconds * beats
        swing = clamp01(config.swing)
        if swing and self._tick_index % 2 == 1:
            seconds *= 1.0 + swing * 0.5
        humanize = max(0.0, float(config.humanize_ms)) / 1000.0
        if humanize > 0:
            seconds += random.uniform(-humanize, humanize)
        return max(0.02, seconds)

    def _normalize_lane(self, lane: LaneConfig) -> None:
        lane.midi_channel = max(1, min(16, int(lane.midi_channel)))
        lane.root_note = max(0, min(127, int(lane.root_note)))
        lane.min_note = max(0, min(127, int(lane.min_note)))
        lane.max_note = max(0, min(127, int(lane.max_note)))
        if lane.min_note > lane.max_note:
            lane.min_note, lane.max_note = lane.max_note, lane.min_note
        lane.velocity_min = max(1, min(127, int(lane.velocity_min)))
        lane.velocity_max = max(1, min(127, int(lane.velocity_max)))
        if lane.velocity_min > lane.velocity_max:
            lane.velocity_min, lane.velocity_max = lane.velocity_max, lane.velocity_min
        lane.gate_length = clamp01(lane.gate_length)
        lane.probability = clamp01(lane.probability)
        lane.visual_amount = clamp01(lane.visual_amount)
        lane.mutation_amount = clamp01(lane.mutation_amount)
        if lane.mode not in LANE_MODES:
            lane.mode = "mono"
        if lane.scale not in SCALES:
            lane.scale = "minor"
        if lane.step_division not in STEP_DIVISIONS:
            lane.step_division = "1/8"
        for attr in ("pitch_source", "velocity_source", "probability_source", "cc_source"):
            if getattr(lane, attr) not in VISUAL_METRIC_NAMES:
                setattr(lane, attr, "brightness")
        lane.cc_number = max(0, min(127, int(lane.cc_number)))
        self._ensure_cc_slots(lane)
        for slot in lane.cc_slots:
            self._normalize_cc_slot(slot)
        if lane.cc_enabled or lane.cc_number != 74 or lane.cc_source != "contrast":
            lane.cc_slots[0].enabled = bool(lane.cc_enabled)
            lane.cc_slots[0].number = lane.cc_number
            lane.cc_slots[0].source = lane.cc_source

    def _ensure_cc_slots(self, lane: LaneConfig) -> None:
        slots = list(getattr(lane, "cc_slots", []) or [])
        normalized_slots = []
        for slot in slots[:MAX_CC_SLOTS]:
            if isinstance(slot, CCSlotConfig):
                normalized_slots.append(slot)
            elif isinstance(slot, dict):
                normalized_slots.append(CCSlotConfig(**{key: slot[key] for key in ("enabled", "number", "source") if key in slot}))
        while len(normalized_slots) < MAX_CC_SLOTS:
            normalized_slots.append(CCSlotConfig())
        lane.cc_slots = normalized_slots

    def _normalize_cc_slot(self, slot: CCSlotConfig) -> None:
        slot.enabled = bool(slot.enabled)
        slot.number = max(0, min(127, int(slot.number)))
        if slot.source not in VISUAL_METRIC_NAMES:
            slot.source = "brightness"

    def _normalized_cc_slots_for_send(self, lane: LaneConfig) -> List[CCSlotConfig]:
        slots = list(getattr(lane, "cc_slots", []) or [])
        if not slots:
            slots = [CCSlotConfig(enabled=lane.cc_enabled, number=lane.cc_number, source=lane.cc_source)]
        result: List[CCSlotConfig] = []
        for slot in slots[:MAX_CC_SLOTS]:
            if isinstance(slot, CCSlotConfig):
                copy = CCSlotConfig(slot.enabled, slot.number, slot.source)
            elif isinstance(slot, dict):
                copy = CCSlotConfig(**{key: slot[key] for key in ("enabled", "number", "source") if key in slot})
            else:
                continue
            self._normalize_cc_slot(copy)
            result.append(copy)
        return result


def scale_notes(root_note: int, scale_name: str, min_note: int, max_note: int) -> List[int]:
    intervals = SCALES.get(scale_name, SCALES["minor"])
    notes = []
    root_pc = int(root_note) % 12
    for note in range(max(0, min_note), min(127, max_note) + 1):
        if (note - root_pc) % 12 in intervals:
            notes.append(note)
    return notes


def clone_lane_config(lane: LaneConfig) -> LaneConfig:
    slots = []
    for slot in list(getattr(lane, "cc_slots", []) or [])[:MAX_CC_SLOTS]:
        if isinstance(slot, CCSlotConfig):
            slots.append(CCSlotConfig(slot.enabled, slot.number, slot.source))
        elif isinstance(slot, dict):
            slots.append(CCSlotConfig(**{key: slot[key] for key in ("enabled", "number", "source") if key in slot}))
    while len(slots) < MAX_CC_SLOTS:
        slots.append(CCSlotConfig())
    return LaneConfig(
        enabled=lane.enabled,
        output_port=lane.output_port,
        midi_channel=lane.midi_channel,
        mode=lane.mode,
        root_note=lane.root_note,
        scale=lane.scale,
        min_note=lane.min_note,
        max_note=lane.max_note,
        step_division=lane.step_division,
        gate_length=lane.gate_length,
        probability=lane.probability,
        velocity_min=lane.velocity_min,
        velocity_max=lane.velocity_max,
        pitch_source=lane.pitch_source,
        velocity_source=lane.velocity_source,
        probability_source=lane.probability_source,
        visual_amount=lane.visual_amount,
        mutation_amount=lane.mutation_amount,
        cc_enabled=lane.cc_enabled,
        cc_number=lane.cc_number,
        cc_source=lane.cc_source,
        cc_slots=slots,
    )


def metric_value(metrics: Dict[str, float], name: str) -> float:
    return clamp01(metrics.get(name, 0.0))


def blend(base: float, visual: float, amount: float) -> float:
    amount = clamp01(amount)
    return clamp01(float(base) * (1.0 - amount) + float(visual) * amount)


def clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def midi_channel(channel_1_based: int) -> int:
    return max(0, min(15, int(channel_1_based) - 1))
