"""Sample-accurate utility envelopes for the FM engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ADSR:
    sample_rate: float = 48000.0
    attack: float = 0.02
    decay: float = 0.4
    sustain: float = 0.6
    release: float = 1.2

    def __post_init__(self):
        self.value = 0.0
        self.stage = "idle"
        self._release_step = 0.0

    @property
    def active(self) -> bool:
        return self.stage != "idle"

    def configure(self, attack=None, decay=None, sustain=None, release=None):
        if attack is not None:
            self.attack = max(0.001, float(attack))
        if decay is not None:
            self.decay = max(0.001, float(decay))
        if sustain is not None:
            self.sustain = min(1.0, max(0.0, float(sustain)))
        if release is not None:
            self.release = max(0.001, float(release))

    def trigger(self):
        self.stage = "attack"

    def release_gate(self):
        if self.stage != "idle":
            self.stage = "release"
            self._release_step = self.value / max(1.0, self.release * self.sample_rate)

    def reset(self):
        self.value = 0.0
        self.stage = "idle"
        self._release_step = 0.0

    def next_value(self) -> float:
        if self.stage == "attack":
            self.value += 1.0 / max(1.0, self.attack * self.sample_rate)
            if self.value >= 1.0:
                self.value = 1.0
                self.stage = "decay"
        elif self.stage == "decay":
            self.value -= (1.0 - self.sustain) / max(1.0, self.decay * self.sample_rate)
            if self.value <= self.sustain:
                self.value = self.sustain
                self.stage = "sustain"
        elif self.stage == "sustain":
            self.value = self.sustain
        elif self.stage == "release":
            self.value -= self._release_step
            if self.value <= 0.0:
                self.reset()
        return min(1.0, max(0.0, self.value))
