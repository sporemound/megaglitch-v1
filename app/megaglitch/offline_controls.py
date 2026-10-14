"""Pure, frozen control evaluation for deterministic Offline Master renders."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


def evaluate_timeline_track(track: Mapping[str, Any], seconds: float) -> Any:
    keyframes = sorted(
        (copy.deepcopy(value) for value in track.get("keyframes", ())),
        key=lambda value: float(value.get("time", 0.0)),
    )
    if not keyframes:
        return None
    seconds = float(seconds)
    if seconds <= float(keyframes[0].get("time", 0.0)):
        return keyframes[0].get("value")
    if seconds >= float(keyframes[-1].get("time", 0.0)):
        return keyframes[-1].get("value")

    previous = keyframes[0]
    following = keyframes[-1]
    for index in range(1, len(keyframes)):
        candidate = keyframes[index]
        if seconds <= float(candidate.get("time", 0.0)):
            previous = keyframes[index - 1]
            following = candidate
            break
    interpolation = str(
        following.get("interp", previous.get("interp", "linear")) or "linear"
    )
    if interpolation == "step" or track.get("type") == "checkbox":
        return previous.get("value")
    t0 = float(previous.get("time", 0.0))
    t1 = float(following.get("time", t0))
    amount = 0.0 if t1 <= t0 else (seconds - t0) / (t1 - t0)
    amount = max(0.0, min(1.0, amount))
    if interpolation == "ease":
        amount = amount * amount * (3.0 - 2.0 * amount)
    try:
        value = float(previous.get("value", 0.0)) * (1.0 - amount) + float(
            following.get("value", 0.0)
        ) * amount
    except (TypeError, ValueError):
        return previous.get("value")
    return int(round(value)) if track.get("type") == "int" else value


def apply_frozen_timeline(
    pipeline: list[dict[str, Any]],
    tracks: Mapping[str, Mapping[str, Any]],
    seconds: float,
    *,
    enabled: bool,
) -> None:
    if not enabled or not tracks:
        return
    changes: dict[str, dict[str, Any]] = {}
    for track in tracks.values():
        value = evaluate_timeline_track(track, seconds)
        if value is None:
            continue
        node_id = str(track.get("node_id", ""))
        parameter = str(track.get("param", ""))
        if node_id and parameter:
            changes.setdefault(node_id, {})[parameter] = value
    for node in pipeline:
        node_changes = changes.get(str(node.get("node_id", "")))
        if not node_changes:
            continue
        params = dict(node.get("params") or {})
        for name, value in node_changes.items():
            if name in params:
                params[name] = value
        node["params"] = params


@dataclass
class FrozenModulation:
    enabled: bool
    connections: tuple[str, ...]
    parameter_ranges: Mapping[str, tuple[float, float]]
    parameter_types: Mapping[str, str]
    parameter_items: Mapping[str, tuple[Any, ...]]
    parameter_modes: Mapping[str, str]
    parameter_spans: Mapping[str, float]
    source_values: Mapping[str, Any]
    target_deck: str | None
    _discrete_event_state: MutableMapping[str, tuple[int, int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _source_norm(
        self,
        source: str,
        seconds: float,
        time_scale: float,
        rng: random.Random,
        source_values: Mapping[str, Any] | None = None,
    ) -> float:
        values = source_values or self.source_values
        bands = values.get("fft_bands") or {}

        def band(name: str) -> float:
            try:
                return float(bands.get(name, 0.0))
            except (AttributeError, TypeError, ValueError):
                return 0.0

        if source == "Time":
            value = (math.sin(float(seconds) * float(time_scale)) + 1.0) * 0.5
        elif source == "Random":
            value = rng.random()
        elif source in {"Beat Pulse", "Beat Trigger"}:
            value = float(values.get("bpm_pulse", 0.0))
        elif source == "Beat Phase":
            value = float(values.get("bpm_phase", 0.0))
        elif source == "BPM (Normalized)":
            value = float(values.get("bpm_normalized", 0.0))
        elif source in {"Onset Energy", "Onset Trigger"}:
            value = float(values.get("onset_value", 0.0))
        elif source in {
            "Sub", "Bass", "Low Mid", "Mid", "High Mid", "Presence",
            "Brilliance", "Air",
        }:
            value = band(source)
        elif source.startswith("FFT Band "):
            value = band(source.replace("FFT ", ""))
        elif source == "Bass Amplitude":
            value = band("Bass")
        elif source == "Treble Amplitude":
            value = max(band("Presence"), band("Brilliance"), band("Air"))
        elif source == "MIDI Note":
            value = float(values.get("midi_note", 0.0)) / 127.0
        elif source == "MIDI CC":
            value = float(values.get("midi_cc", 0.0)) / 127.0
        else:
            value = 0.0
        return float(max(0.0, min(1.0, value)))

    def _event_id(
        self, source: str, source_values: Mapping[str, Any] | None = None
    ) -> int:
        values = source_values or self.source_values
        if source == "Onset Trigger":
            return int(values.get("onset_event_id", 0))
        if source == "Beat Trigger":
            return int(values.get("beat_event_id", 0))
        return 0

    def apply(
        self,
        params: Mapping[str, Any],
        *,
        seconds: float,
        time_scale: float,
        rng: random.Random,
        source_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        output = dict(params)
        if not self.enabled:
            return output
        for connection in self.connections:
            try:
                source, target = [part.strip() for part in connection.split("->")]
            except ValueError:
                continue
            if target not in output:
                continue
            minimum, maximum = self.parameter_ranges.get(target, (0.0, 1.0))
            normalized = self._source_norm(
                source, seconds, time_scale, rng, source_values
            )
            if self.parameter_types.get(target) == "combo":
                choices = self.parameter_items.get(target, ())
                if not choices:
                    continue
                event_id = self._event_id(source, source_values)
                key = f"{source}->{target}"
                if event_id > 0:
                    last_event, last_index = self._discrete_event_state.get(
                        key, (0, -1)
                    )
                    if event_id != last_event:
                        last_index = (last_index + 1) % len(choices)
                        self._discrete_event_state[key] = (event_id, last_index)
                    output[target] = choices[last_index]
                else:
                    index = int(round(normalized * (len(choices) - 1)))
                    output[target] = choices[max(0, min(len(choices) - 1, index))]
                continue
            mode = self.parameter_modes.get(target, "replace")
            if mode == "offset":
                span = float(
                    self.parameter_spans.get(
                        target, max(0.0, (maximum - minimum) * 0.5)
                    )
                )
                try:
                    base = float(output[target])
                except (TypeError, ValueError):
                    base = float(minimum)
                value = base + ((normalized * 2.0) - 1.0) * span
            else:
                value = float(minimum) + normalized * (
                    float(maximum) - float(minimum)
                )
            value = max(float(minimum), min(float(maximum), value))
            output[target] = (
                int(round(value))
                if self.parameter_types.get(target) == "int"
                else value
            )
        return output

    def document(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "connections": list(self.connections),
            "parameter_ranges": {
                key: list(value) for key, value in self.parameter_ranges.items()
            },
            "parameter_types": dict(self.parameter_types),
            "parameter_items": {
                key: list(value) for key, value in self.parameter_items.items()
            },
            "parameter_modes": dict(self.parameter_modes),
            "parameter_spans": dict(self.parameter_spans),
            "source_values": copy.deepcopy(dict(self.source_values)),
            "target_deck": self.target_deck,
        }
