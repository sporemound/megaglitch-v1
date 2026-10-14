import json
from pathlib import Path

import numpy as np

from megaglitch.audio.envelopes import ADSR
from megaglitch.audio.fm_engine import FMEngine, TX802Envelope
from megaglitch.audio.scales import quantize_midi
from megaglitch.nodes.fm_synth_node import FMSynthNode


def render_triggered(engine, blocks=8):
    engine.trigger_note(45, velocity=0.8)
    return np.concatenate([engine.render() for _ in range(blocks)], axis=0)


def test_engine_generates_finite_limited_stereo_audio():
    engine = FMEngine(sample_rate=48000, block_size=256)
    audio = render_triggered(engine)
    assert audio.shape == (2048, 2)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 1.0
    assert float(np.max(np.abs(audio))) > 0.0


def test_engine_is_deterministic_from_reset_state():
    first = FMEngine(sample_rate=48000, block_size=128, seed=12)
    second = FMEngine(sample_rate=48000, block_size=128, seed=12)
    assert np.allclose(render_triggered(first), render_triggered(second))


def test_parameter_clamping_and_smoothing():
    engine = FMEngine(block_size=128)
    engine.set_param("master_gain", 9.0)
    engine.set_param("feedback", -2.0)
    assert engine.get_param("master_gain") == 1.0
    assert engine.get_param("feedback") == 0.0
    engine.trigger_note(48)
    engine.render()
    assert 0.35 < engine._smoothed["master_gain"] < 1.0


def test_panic_immediately_silences_voice():
    engine = FMEngine(block_size=128)
    engine.trigger_note(48)
    assert np.any(engine.render())
    engine.panic()
    assert not np.any(engine.render())
    assert engine.get_output("rms") == 0.0


def test_analysis_outputs_are_normalized():
    engine = FMEngine(block_size=512)
    render_triggered(engine, blocks=4)
    outputs = engine.get_outputs()
    assert set(outputs) == set(FMSynthNode.OUTPUT_NAMES)
    assert all(0.0 <= value <= 1.0 for value in outputs.values())


def test_adsr_reaches_release_and_idle():
    envelope = ADSR(sample_rate=1000, attack=0.01, decay=0.01, sustain=0.5, release=0.01)
    envelope.trigger()
    for _ in range(40):
        envelope.next_value()
    assert envelope.active
    envelope.release_gate()
    for _ in range(20):
        envelope.next_value()
    assert not envelope.active
    assert envelope.value == 0.0


def test_pitch_quantization_uses_requested_scale():
    assert quantize_midi(38, root=36, mode="minor_pentatonic") == 39
    assert quantize_midi(40, root=36, mode="major") == 40


def test_one_hour_show_targets_are_accepted():
    show_path = Path(__file__).parents[1] / "shows" / "one_hour_fm_synth_show.json"
    show = json.loads(show_path.read_text(encoding="utf-8"))
    engine = FMEngine()
    for event in show["events"]:
        name = event["target"].split(".", 1)[1]
        values = []
        for key in ("value", "from", "to", "base"):
            if key in event:
                values.append(event[key])
        if "values" in event:
            values.extend(event["values"])
        for value in values:
            engine.set_param(name, value)


def test_tx802_algorithm_clamping_and_aliases():
    engine = FMEngine()
    engine.set_param("algorithm", "feedback_bell")
    assert engine.get_param("algorithm") == 8
    engine.set_param("algorithm", "TX802 32")
    assert engine.get_param("algorithm") == 32
    engine.set_param("algorithm", 99)
    assert engine.get_param("algorithm") == 32
    engine.set_param("algorithm", 0)
    assert engine.get_param("algorithm") == 1


def test_tx802_outputs_include_new_operator_and_part_metrics():
    engine = FMEngine(block_size=256)
    render_triggered(engine, blocks=4)
    outputs = engine.get_outputs()
    assert "operator_5_energy" in outputs
    assert "operator_6_energy" in outputs
    assert "voice_count" in outputs
    assert "feedback_activity" in outputs
    for index in range(1, 9):
        assert f"part_{index}_rms" in outputs
    assert all(0.0 <= value <= 1.0 for value in outputs.values())


def test_tx802_envelope_releases_to_idle():
    envelope = TX802Envelope(
        sample_rate=1000,
        rates=(99.0, 99.0, 99.0, 99.0),
        levels=(1.0, 0.7, 0.5, 0.0),
    )
    envelope.trigger()
    for _ in range(30):
        envelope.next_value()
    assert envelope.active
    envelope.release_gate()
    for _ in range(30):
        envelope.next_value()
    assert not envelope.active
    assert envelope.value == 0.0


def test_tx802_algorithms_have_distinct_spectra():
    spectra = []
    for algorithm in (1, 8, 16, 32):
        engine = FMEngine(block_size=512)
        engine.set_param("algorithm", algorithm)
        engine.set_param("modulation_index", 6.0)
        audio = render_triggered(engine, blocks=3).mean(axis=1)
        spectra.append(np.abs(np.fft.rfft(audio)))
    assert not np.allclose(spectra[0], spectra[1])
    assert not np.allclose(spectra[1], spectra[2])
    assert not np.allclose(spectra[2], spectra[3])


def test_tx802_polyphony_mixes_multiple_notes_without_clipping():
    engine = FMEngine(block_size=256)
    engine.set_param("polyphony", 4)
    for note in (45, 52, 57, 64):
        engine.trigger_note(note, velocity=0.8)
    audio = np.concatenate([engine.render() for _ in range(4)], axis=0)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 1.0
    assert engine.get_output("voice_count") > 0.0


def test_zero_level_carriers_are_silent():
    engine = FMEngine(block_size=128)
    engine.set_param("algorithm", 32)
    for index in range(1, 7):
        engine.set_param(f"operator_level_{index}", 0.0)
    engine.trigger_note(48)
    assert not np.any(engine.render())
