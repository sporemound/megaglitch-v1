from __future__ import annotations

import time
from typing import Dict, Optional

from megaglitch_live_sequencer import LiveSequencer


def _norm(value: int) -> float:
    return max(0.0, min(1.0, float(value) / 127.0))


def _scale(value: int, low: float, high: float) -> float:
    return low + _norm(value) * (high - low)


class AkaiMPD26Controls:
    """Akai MPD26 performance mappings for the live visual MIDI sequencer.

    The profile targets the common factory-style MPD pad notes 36-51 and CC
    ranges used by the six faders/knobs. MPD26 presets can be edited on the
    hardware, so the mapping intentionally accepts a few common CC aliases.
    """

    DEVICE_HINTS = ("mpd26", "mpd 26", "akai mpd")

    # Pads 1-8 toggle lanes. Pads 9-16 focus lanes for the shared knobs.
    LANE_TOGGLE_NOTES: Dict[int, int] = {36 + i: i for i in range(8)}
    LANE_FOCUS_NOTES: Dict[int, int] = {44 + i: i for i in range(8)}

    # Six faders control the first six lane probabilities. Include common
    # factory and user-preset CC ranges seen on MPD-style controllers.
    FADER_CC_ALIASES = (
        (7, 8, 9, 10, 11, 12),
        (48, 49, 50, 51, 52, 53),
    )

    # Six knobs: global performance controls plus two focused-lane controls.
    KNOB_CC_ALIASES = (
        (16, 17, 18, 19, 20, 21),
        (22, 23, 24, 25, 26, 27),
    )

    TRANSPORT_CC = {
        115: "play",
        116: "pause",
        117: "reset",
        118: "panic",
    }

    def __init__(self, sequencer: LiveSequencer):
        self.sequencer = sequencer
        self.enabled = True
        self.detected_input_name = ""
        self.focus_lane_index = 0
        self.status_message = "MPD26 controls waiting for MIDI input"
        self._last_tap = 0.0
        self._last_note_actions: Dict[int, float] = {}

    def choose_input(self, input_names) -> str:
        for name in input_names or []:
            lowered = str(name).lower()
            if any(hint in lowered for hint in self.DEVICE_HINTS):
                self.detected_input_name = str(name)
                self.status_message = f"MPD26 controls armed: {name}"
                return str(name)
        self.detected_input_name = ""
        self.status_message = "MPD26 controls armed on default MIDI input"
        return ""

    def handle_midi(self, msg) -> None:
        if not self.enabled:
            return
        msg_type = getattr(msg, "type", "")
        if msg_type in ("start", "continue"):
            self.sequencer.play()
            self.status_message = "MPD26 transport: play"
            return
        if msg_type == "stop":
            self.sequencer.pause()
            self.status_message = "MPD26 transport: stop"
            return
        if msg_type == "note_on":
            note = int(getattr(msg, "note", 0))
            velocity = int(getattr(msg, "velocity", 0))
            if velocity <= 0:
                return
            self._handle_note(note, velocity)
            return
        if msg_type == "control_change":
            self._handle_cc(int(getattr(msg, "control", 0)), int(getattr(msg, "value", 0)))

    def _handle_note(self, note: int, velocity: int) -> None:
        now = time.monotonic()
        if now - self._last_note_actions.get(note, 0.0) < 0.08:
            return
        self._last_note_actions[note] = now

        if note in self.LANE_TOGGLE_NOTES:
            lane_index = self.LANE_TOGGLE_NOTES[note]
            lane = self.sequencer.get_lane_config(lane_index)
            self.sequencer.update_lane(lane_index, "enabled", not lane.enabled)
            self.focus_lane_index = lane_index
            self.status_message = f"MPD26 pad {note}: lane {lane_index + 1} {'on' if not lane.enabled else 'off'}"
            return

        if note in self.LANE_FOCUS_NOTES:
            lane_index = self.LANE_FOCUS_NOTES[note]
            self.focus_lane_index = lane_index
            lane_probability = max(0.05, _norm(velocity))
            self.sequencer.update_lane(lane_index, "probability", lane_probability)
            self.status_message = f"MPD26 pad {note}: focus lane {lane_index + 1}"

    def _handle_cc(self, cc: int, value: int) -> None:
        fader_index = self._index_in_aliases(cc, self.FADER_CC_ALIASES)
        if fader_index is not None:
            self.sequencer.update_lane(fader_index, "probability", _norm(value))
            self.status_message = f"MPD26 fader {fader_index + 1}: lane probability"
            return

        knob_index = self._index_in_aliases(cc, self.KNOB_CC_ALIASES)
        if knob_index is not None:
            self._handle_knob(knob_index, value)
            return

        action = self.TRANSPORT_CC.get(cc)
        if action and value > 0:
            self._handle_transport_action(action)

    def _handle_knob(self, knob_index: int, value: int) -> None:
        lane_index = self.focus_lane_index
        if knob_index == 0:
            self.sequencer.update_global("global_density", _norm(value))
            label = "global density"
        elif knob_index == 1:
            self.sequencer.update_global("global_mutation", _norm(value))
            label = "global mutation"
        elif knob_index == 2:
            self.sequencer.update_global("swing", _scale(value, 0.0, 0.55))
            label = "swing"
        elif knob_index == 3:
            self.sequencer.update_global("humanize_ms", _scale(value, 0.0, 80.0))
            label = "humanize"
        elif knob_index == 4:
            self.sequencer.update_lane(lane_index, "gate_length", _scale(value, 0.05, 1.0))
            label = f"lane {lane_index + 1} gate"
        else:
            self.sequencer.update_lane(lane_index, "visual_amount", _norm(value))
            label = f"lane {lane_index + 1} visual amount"
        self.status_message = f"MPD26 knob {knob_index + 1}: {label}"

    def _handle_transport_action(self, action: str) -> None:
        if action == "play":
            self.sequencer.play()
        elif action == "pause":
            self.sequencer.pause()
        elif action == "reset":
            self.sequencer.reset()
        elif action == "panic":
            self.sequencer.panic_all()
        self.status_message = f"MPD26 transport: {action}"

    @staticmethod
    def _index_in_aliases(cc: int, aliases) -> Optional[int]:
        for group in aliases:
            if cc in group:
                return group.index(cc)
        return None
