"""Node-facing audiovisual FM synth adapter."""

from __future__ import annotations

from megaglitch.audio.audio_device import AudioDevice
from megaglitch.audio.fm_engine import FMEngine
from megaglitch.audio.modulation_bus import ModulationBus


class FMSynthNode:
    OUTPUT_NAMES = (
        "rms",
        "peak",
        "spectral_brightness",
        "transient",
        "beat_phase",
        "operator_1_energy",
        "operator_2_energy",
        "operator_3_energy",
        "operator_4_energy",
        "operator_5_energy",
        "operator_6_energy",
        "modulation_activity",
        "voice_count",
        "feedback_activity",
        "part_1_rms",
        "part_2_rms",
        "part_3_rms",
        "part_4_rms",
        "part_5_rms",
        "part_6_rms",
        "part_7_rms",
        "part_8_rms",
    )

    def __init__(self, sample_rate=48000, block_size=256, modulation_bus=None):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.engine = FMEngine(self.sample_rate, self.block_size)
        self.modulation_bus = modulation_bus or ModulationBus()
        self.audio_device = AudioDevice(self.engine.render, self.sample_rate, self.block_size)
        self._last_update_time = None
        self._trigger_high = False

    def start(self):
        started = self.audio_device.start()
        if not self.engine.envelope.active:
            self.trigger_note()
        return started

    def stop(self):
        self.audio_device.stop()

    def set_param(self, name, value):
        if name == "trigger":
            high = float(value) >= 0.5
            if high and not self._trigger_high:
                self.trigger_note()
            self._trigger_high = high
            return
        self.engine.set_param(name, value)

    def get_param(self, name):
        if name == "trigger":
            return float(self._trigger_high)
        return self.engine.get_param(name)

    def get_output(self, name):
        return self.engine.get_output(name)

    def get_outputs(self):
        return self.engine.get_outputs()

    def update(self, time, dt, slot_time):
        outputs = self.engine.get_outputs()
        for name, value in outputs.items():
            self.modulation_bus.publish(f"fm_synth_node.{name}", value)
        for name, value in self.engine.params.items():
            if isinstance(value, (int, float, bool)):
                self.modulation_bus.publish(f"fm_synth_node.{name}", float(value))
        self._last_update_time = float(time)
        return outputs

    def trigger_note(self, note=None, velocity=1.0):
        self.engine.trigger_note(note=note, velocity=velocity)

    def release_note(self):
        self.engine.release_note()

    def panic(self):
        self.engine.panic()

    def destroy(self):
        self.panic()
        self.audio_device.destroy()
