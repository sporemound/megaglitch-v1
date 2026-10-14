import time
from types import SimpleNamespace

import cv2
import numpy as np

from megaglitch.audio.media_audio_engine import (
    MEDIA_AUDIO_PRESETS,
    MediaAnalysis,
    MediaAudioController,
    MediaAudioEngine,
    PartitionedStereoConvolver,
    analyze_media,
    load_impulse_response,
)
from megaglitch.audio.cdp_graph import CDP_EFFECT_SPECS, CDPGraphProcessor, CDP_WRAPPER_PROGRAMS
from megaglitch.audio.tx802 import dx7_checksum, parse_tx802_sysex, tx802_patch_to_media_params


def synthetic_analysis(duration=1.0):
    timestamps = np.linspace(0.0, duration, 12, endpoint=False)
    return MediaAnalysis(
        timestamps=timestamps,
        brightness=np.linspace(0.2, 1.0, 12, dtype=np.float32),
        colors=np.column_stack(
            (
                np.linspace(0.0, 1.0, 12),
                np.full(12, 0.4),
                np.linspace(1.0, 0.0, 12),
            )
        ).astype(np.float32),
        motion=np.linspace(0.0, 1.0, 12, dtype=np.float32),
        duration=duration,
        source_path="synthetic.mp4",
    )


def assert_param_value(actual, expected):
    if isinstance(expected, str):
        assert actual == expected
    elif isinstance(expected, bool):
        assert actual is expected
    else:
        assert np.isclose(actual, expected)


def test_engine_renders_finite_stereo_and_advances():
    engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    engine.set_analysis(synthetic_analysis())
    assert engine.play(restart=True)
    audio = engine.render(512)
    state = engine.snapshot()

    assert audio.shape == (512, 2)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert 0.0 < float(np.max(np.abs(audio))) <= 1.0
    assert state["position"] > 0.0
    assert not np.allclose(audio[:, 0], audio[:, 1])


def test_non_looping_engine_stops_at_media_end():
    engine = MediaAudioEngine(sample_rate=1000, block_size=256)
    engine.set_analysis(synthetic_analysis(duration=0.1))
    engine.set_param("loop", False)
    engine.play(restart=True)
    engine.render(256)
    state = engine.snapshot()

    assert not state["playing"]
    assert state["position"] == state["duration"]


def test_live_display_mode_works_without_loaded_media():
    engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics(
        {
            "brightness": 0.8,
            "motion": 0.65,
            "red_energy": 0.9,
            "blue_energy": 0.1,
        }
    )
    assert engine.play(restart=True)
    audio = engine.render(256)
    state = engine.snapshot()

    assert state["live_metrics_received"]
    assert state["source_mode"] == "Live Display"
    assert float(np.max(np.abs(audio))) > 0.0
    assert not np.allclose(audio[:, 0], audio[:, 1])


def test_visual_influence_controls_autonomous_drone_evolution():
    muted = MediaAudioEngine(sample_rate=8000, block_size=256)
    vivid = MediaAudioEngine(sample_rate=8000, block_size=256)
    metrics = {
        "brightness": 0.82,
        "motion": 0.7,
        "entropy": 0.75,
        "edge_density": 0.65,
        "red_energy": 0.85,
        "green_energy": 0.3,
        "blue_energy": 0.1,
        "center_mass_x": 0.78,
        "center_mass_y": 0.22,
    }
    for engine in (muted, vivid):
        engine.set_source_mode("Live Display")
        engine.set_live_metrics(metrics)
        engine.set_param("midi_note_follow", False)
        engine.set_param("motion_response", 1.0)
        engine.set_param("feedback_color", 0.25)
        engine.set_param("timbre_spread", 0.5)
        assert engine.play(restart=True)

    muted.set_param("visual_influence", 0.0)
    muted.set_param("visual_evolution", 0.0)
    muted.set_param("visual_pitch_motion", 0.0)
    vivid.set_param("visual_influence", 1.8)
    vivid.set_param("visual_evolution", 1.0)
    vivid.set_param("visual_pitch_motion", 1.0)

    dry = muted.render(256)
    animated = vivid.render(256)

    assert np.isfinite(animated).all()
    assert 0.0 < float(np.max(np.abs(animated))) <= 1.0
    assert not np.allclose(dry, animated, atol=1e-4)


def test_tx802_operator_params_are_exposed_and_affect_render():
    base = MediaAudioEngine(sample_rate=8000, block_size=256)
    altered = MediaAudioEngine(sample_rate=8000, block_size=256)
    for engine in (base, altered):
        engine.set_source_mode("Live Display")
        engine.set_live_metrics({"brightness": 0.7, "motion": 0.3})
        assert engine.play(restart=True)

    altered.set_param("voice_count", 16)
    altered.set_param("feedback", 0.4)
    altered.set_param("operator_ratio_5", 9.0)
    altered.set_param("operator_level_5", 1.0)
    altered.set_param("operator_detune_6", 37.0)
    altered.set_param("operator_pan_6", -1.0)
    altered.set_param("operator_fixed_6", True)
    altered.set_param("operator_fixed_freq_6", 777.0)

    dry = base.render(256)
    wet = altered.render(256)
    state = altered.snapshot()["params"]

    assert state["voice_count"] == 16
    assert state["operator_fixed_6"] is True
    assert state["operator_fixed_freq_6"] == 777.0
    assert np.isfinite(wet).all()
    assert not np.allclose(dry, wet, atol=1e-4)


def test_disabling_all_media_audio_operators_silences_engine():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    for index in range(1, 7):
        engine.set_param(f"operator_enabled_{index}", False)
    assert engine.play(restart=True)
    audio = engine.render(128)
    assert np.allclose(audio, 0.0)


def test_midi_note_input_drives_media_audio_pitch_and_gate():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_param("midi_gate_enabled", True)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.7, "motion": 0.2})
    assert engine.play(restart=True)

    assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=45, velocity=100))
    state = engine.snapshot()["params"]
    assert np.isclose(state["carrier_freq"], 432.0, atol=0.01)
    assert 0.78 < state["midi_gate"] < 0.79
    assert 0.78 < state["midi_velocity"] < 0.79
    audio = engine.render(128)
    assert float(np.max(np.abs(audio))) > 0.0
    effective = engine.snapshot()["effective_params"]
    assert np.isclose(effective["carrier_freq"], 110.0, atol=0.01)

    assert engine.handle_midi_performance(SimpleNamespace(type="note_off", note=45, velocity=0))
    assert engine.snapshot()["params"]["midi_gate"] == 0.0


def test_midi_chords_render_as_true_polyphonic_voices():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.6, "motion": 0.15})
    engine.set_param("midi_gate_enabled", True)
    engine.set_param("midi_style", "Musical")
    engine.set_param("midi_note_focus", 0.8)
    engine.set_param("voice_count", 12)
    assert engine.play(restart=True)

    for note, velocity in ((48, 80), (52, 96), (55, 112)):
        assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=note, velocity=velocity))
    audio = engine.render(128)
    effective = engine.snapshot()["effective_params"]

    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 0.0
    assert effective["voice_count"] == 3
    assert len(effective["_midi_poly_notes"]) == 3
    frequencies = [round(pair[0], 3) for pair in effective["_midi_poly_notes"]]
    assert frequencies == [130.813, 164.814, 195.998]


def test_amp_adsr_releases_to_silence_when_midi_gated():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.7, "motion": 0.2})
    engine.set_param("midi_gate_enabled", True)
    engine.set_param("amp_env_enabled", True)
    engine.set_param("amp_attack_ms", 0.0)
    engine.set_param("amp_decay_ms", 0.0)
    engine.set_param("amp_sustain", 1.0)
    engine.set_param("amp_release_ms", 0.0)
    engine.set_param("filter_enabled", False)
    assert engine.play(restart=True)

    assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=45, velocity=100))
    opened = engine.render(128)
    assert float(np.max(np.abs(opened))) > 0.0

    assert engine.handle_midi_performance(SimpleNamespace(type="note_off", note=45, velocity=0))
    released = engine.render(128)
    assert np.allclose(released, 0.0, atol=1e-5)
    assert engine.snapshot()["effective_params"]["_amp_env_stage"] == "idle"


def test_filter_controls_shape_media_audio_output():
    dry_engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    filtered_engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    for engine in (dry_engine, filtered_engine):
        engine.set_source_mode("Live Display")
        engine.set_live_metrics({"brightness": 0.75, "motion": 0.4, "edge_density": 0.5})
        engine.set_param("midi_note_follow", False)
        assert engine.play(restart=True)

    dry_engine.set_param("filter_enabled", False)
    filtered_engine.set_param("filter_enabled", True)
    filtered_engine.set_param("filter_cutoff", 240.0)
    filtered_engine.set_param("filter_resonance", 0.35)
    filtered_engine.set_param("filter_visual_amount", 0.0)
    filtered_engine.set_param("filter_drive", 0.0)

    dry = dry_engine.render(256)
    filtered = filtered_engine.render(256)

    assert np.isfinite(filtered).all()
    assert 0.0 < float(np.max(np.abs(filtered))) <= 1.0
    assert not np.allclose(dry, filtered, atol=1e-4)


def test_midi_velocity_adds_timbre_punch_without_clipping():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.7, "motion": 0.2})
    engine.set_param("midi_timbre_depth", 1.0)
    engine.set_param("midi_note_punch", 0.6)
    engine.set_param("midi_pitch_slew_ms", 20.0)
    assert engine.play(restart=True)

    before = engine.render(128)
    assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=45, velocity=120))
    after = engine.render(128)
    state = engine.snapshot()

    assert np.isfinite(after).all()
    assert 0.0 < float(np.max(np.abs(after))) <= 1.0
    assert state["effective_params"]["modulation_index"] > state["params"]["modulation_index"]
    assert not np.allclose(before, after, atol=1e-4)


def test_midi_musical_style_restrains_drone_smear():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.8, "motion": 0.6, "edge_density": 0.7})
    engine.set_param("voice_count", 12)
    engine.set_param("poly_spread", 0.9)
    engine.set_param("chord_depth", 0.9)
    engine.set_param("feedback", 0.8)
    engine.set_param("midi_timbre_depth", 1.0)
    engine.set_param("midi_note_punch", 0.8)
    assert engine.play(restart=True)

    assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=48, velocity=100))
    audio = engine.render(128)
    state = engine.snapshot()
    effective = state["effective_params"]

    assert state["params"]["midi_style"] == "Musical"
    assert np.isfinite(audio).all()
    assert effective["voice_count"] < state["params"]["voice_count"]
    assert effective["feedback"] < state["params"]["feedback"]
    assert effective["poly_spread"] < state["params"]["poly_spread"]
    assert effective["chord_depth"] < state["params"]["chord_depth"]


def test_midi_note_focus_zero_preserves_visual_gui_parameters():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.8, "motion": 0.6, "edge_density": 0.7})
    engine.set_param("voice_count", 10)
    engine.set_param("poly_spread", 0.82)
    engine.set_param("chord_depth", 0.73)
    engine.set_param("feedback", 0.42)
    engine.set_param("midi_note_focus", 0.0)
    assert engine.play(restart=True)

    assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=48, velocity=100))
    audio = engine.render(128)
    effective = engine.snapshot()["effective_params"]

    assert np.isfinite(audio).all()
    assert effective["voice_count"] == 10
    assert np.isclose(effective["poly_spread"], 0.82)
    assert np.isclose(effective["chord_depth"], 0.73)
    assert np.isclose(effective["feedback"], 0.42)


def test_midi_note_focus_keeps_base_gui_parameters_stable():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.8, "motion": 0.6, "edge_density": 0.7})
    stable_values = {
        "carrier_freq": 432.0,
        "voice_count": 10,
        "modulation_index": 2.7,
        "feedback": 0.42,
        "motion_response": 1.7,
    }
    for name, value in stable_values.items():
        engine.set_param(name, value)
    assert engine.play(restart=True)

    assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=48, velocity=100))
    engine.render(128)
    params = engine.snapshot()["params"]

    for name, value in stable_values.items():
        assert np.isclose(params[name], value)


def test_media_audio_controller_routes_notes_to_synth_before_step_grid():
    controller = MediaAudioController(sample_rate=8000, block_size=128)
    try:
        controller.engine.set_param("midi_gate_enabled", True)
        before = controller.sequencer.lanes[0].steps[0].enabled
        assert controller.handle_midi(SimpleNamespace(type="note_on", note=36, velocity=100), metrics={})
        assert controller.engine.snapshot()["params"]["carrier_freq"] > 0.0
        assert controller.sequencer.lanes[0].steps[0].enabled is before
    finally:
        controller.destroy()


def test_ir_convolution_measurably_changes_audio():
    analysis = synthetic_analysis()
    dry_engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    dry_engine.set_analysis(analysis)
    dry_engine.set_source_mode("Loaded Media")
    dry_engine.play(restart=True)
    dry = np.vstack([dry_engine.render(128) for _ in range(6)])

    wet_engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    wet_engine.set_analysis(analysis)
    wet_engine.set_source_mode("Loaded Media")
    impulse = np.zeros((800, 2), dtype=np.float32)
    impulse[0] = 0.7
    impulse[160] = 0.5
    impulse[400] = -0.25
    wet_engine.set_convolver(
        PartitionedStereoConvolver(impulse, wet_engine.block_size),
        "test-ir.wav",
    )
    wet_engine.set_param("ir_wet", 0.8)
    wet_engine.set_param("ir_gain", 1.0)
    wet_engine.play(restart=True)
    wet = np.vstack([wet_engine.render(128) for _ in range(6)])

    assert wet_engine.snapshot()["ir_loaded"]
    assert not np.allclose(dry, wet, atol=1e-4)
    assert float(np.mean(np.abs(dry - wet))) > 0.005


def test_cdp_graph_cpu_preview_changes_audio_without_blocking():
    processor = CDPGraphProcessor()
    try:
        assert processor.snapshot()["wet_mix"] > 0.0
        processor.add_node("Distort Average")
        processor.configure(enabled=True, cpu_preview=True, use_cdp=False)
        t = np.linspace(0.0, 1.0, 512, endpoint=False)
        audio = np.column_stack((np.sin(2 * np.pi * 220.0 * t), np.sin(2 * np.pi * 330.0 * t))).astype(np.float32) * 0.2
        wet = processor.process(audio, 8000, visual={"motion": 0.8})

        assert wet.shape == audio.shape
        assert np.isfinite(wet).all()
        assert not np.allclose(audio, wet, atol=1e-5)
        assert processor.snapshot()["last_preview_delta"] > 0.01
    finally:
        processor.shutdown()


def test_cdp_graph_exposes_complete_cdp_wrapper_set():
    expected = {
        "blur.cmd",
        "combine.cmd",
        "distort.cmd",
        "env.cmd",
        "filter.cmd",
        "formants.cmd",
        "grain.cmd",
        "hfperm.cmd",
        "hilite.cmd",
        "houskeep.cmd",
        "mix.cmd",
        "modify.cmd",
        "morph.cmd",
        "pitch.cmd",
        "pitchinfo.cmd",
        "pview.cmd",
        "pvoc.cmd",
        "repitch.cmd",
        "sfconvert.cmd",
        "sfedit.cmd",
        "sfprops.cmd",
        "sndinfo.cmd",
        "spec.cmd",
        "specinfo.cmd",
        "stretch.cmd",
        "submix.cmd",
    }
    assert set(CDP_WRAPPER_PROGRAMS) == expected


def test_visual_glitch_effects_are_live_cpu_preview_nodes():
    names = [
        name
        for name, spec in CDP_EFFECT_SPECS.items()
        if spec.get("category") == "visual_glitch"
    ]
    expected = {
        "Visual Glitch - Motion Slicer",
        "Visual Glitch - Edge Comb Shred",
        "Visual Glitch - Color Bitcrush",
        "Visual Glitch - Brightness Gate",
        "Visual Glitch - Spike Buffer Repeat",
        "Visual Glitch - Entropy Spectral Tear",
        "Visual Glitch - Center Pan Scanner",
        "Visual Glitch - RGB Ring Mod",
        "Visual Glitch - Block Dropout Matrix",
    }
    assert expected.issubset(set(names))

    t = np.linspace(0.0, 1.0, 512, endpoint=False)
    audio = np.column_stack((
        np.sin(2.0 * np.pi * 110.0 * t),
        np.sin(2.0 * np.pi * 147.0 * t),
    )).astype(np.float32) * 0.25
    visual = {
        "brightness": 0.82,
        "motion": 0.9,
        "edge_density": 0.78,
        "entropy": 0.86,
        "glitch_spike": 0.72,
        "red_energy": 0.9,
        "green_energy": 0.25,
        "blue_energy": 0.7,
        "center_mass_x": 0.76,
        "center_mass_y": 0.3,
    }
    processor = CDPGraphProcessor()
    try:
        for name in names:
            processor.graph.nodes.clear()
            processor.add_node(name)
            processor.configure(enabled=True, wet_mix=1.0, cpu_preview=True, use_cdp=False)
            wet = processor.process(audio, 8000, visual=visual)
            assert wet.shape == audio.shape
            assert np.isfinite(wet).all()
            assert not np.allclose(audio, wet, atol=1e-5), name
    finally:
        processor.shutdown()


def test_media_audio_cdp_graph_affects_live_render():
    dry_engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    wet_engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    metrics = {"brightness": 0.75, "motion": 0.6, "edge_density": 0.55, "glitch_spike": 0.3}
    for engine in (dry_engine, wet_engine):
        engine.set_source_mode("Live Display")
        engine.set_live_metrics(metrics)
        assert engine.play(restart=True)
    wet_engine.cdp_graph.add_node("Distort Average")
    wet_engine.cdp_graph.configure(enabled=True, wet_mix=0.8, cpu_preview=True, use_cdp=False)

    dry = dry_engine.render(256)
    wet = wet_engine.render(256)

    assert np.isfinite(wet).all()
    assert 0.0 < float(np.max(np.abs(wet))) <= 1.0
    assert not np.allclose(dry, wet, atol=1e-5)


def test_ir_normalization_keeps_room_return_below_clipping(tmp_path):
    path = tmp_path / "room.wav"
    sample_rate = 8000
    samples = np.arange(sample_rate, dtype=np.float32)
    impulse = np.zeros((sample_rate, 2), dtype=np.float32)
    impulse[0] = 1.0
    tail = np.exp(-samples / (sample_rate * 0.45)) * 0.25
    impulse[:, 0] += tail
    impulse[:, 1] += np.roll(tail, 11)
    from scipy.io import wavfile

    wavfile.write(str(path), sample_rate, impulse)
    loaded = load_impulse_response(str(path), sample_rate, normalize=True)

    assert float(np.max(np.abs(loaded))) <= 0.98
    assert float(np.sqrt(np.mean(loaded * loaded))) < 0.25


def test_all_media_audio_presets_apply_within_limits():
    engine = MediaAudioEngine()
    for name, expected in MEDIA_AUDIO_PRESETS.items():
        applied = engine.apply_preset(name)
        assert applied == expected
        state = engine.snapshot()["params"]
        for param_name, value in expected.items():
            assert param_name in state
            if param_name in engine.PARAM_LIMITS:
                low, high = engine.PARAM_LIMITS[param_name]
                assert low <= float(value) <= high
            assert_param_value(state[param_name], value)


def test_tx802_preset_changes_media_audio_output():
    default_engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    idm_engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    metrics = {
        "brightness": 0.76,
        "motion": 0.42,
        "entropy": 0.68,
        "edge_density": 0.36,
        "red_energy": 0.64,
        "green_energy": 0.38,
        "blue_energy": 0.22,
        "center_mass_x": 0.62,
        "center_mass_y": 0.4,
    }
    for engine in (default_engine, idm_engine):
        engine.set_source_mode("Live Display")
        engine.set_live_metrics(metrics)
        assert engine.play(restart=True)

    idm_engine.apply_preset("TX802 EP - Tine Vector")
    default_audio = default_engine.render(256)
    idm_audio = idm_engine.render(256)

    assert np.isfinite(idm_audio).all()
    assert 0.0 < float(np.max(np.abs(idm_audio))) <= 1.0
    assert not np.allclose(default_audio, idm_audio, atol=1e-4)


def test_tx802_preset_keeps_base_params_stable_under_midi():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics({"brightness": 0.7, "motion": 0.3})
    preset = engine.apply_preset("TX802 Bell - Glass Ratio")
    stable_names = (
        "carrier_freq",
        "voice_count",
        "modulation_index",
        "feedback",
        "motion_response",
    )
    assert engine.play(restart=True)

    assert engine.handle_midi_performance(SimpleNamespace(type="note_on", note=48, velocity=96))
    engine.render(128)
    params = engine.snapshot()["params"]

    for name in stable_names:
        assert_param_value(params[name], preset[name])


def test_tx802_preset_bank_contains_planned_names():
    expected_names = {
        "TX802 EP - Tine Vector",
        "TX802 Bell - Glass Ratio",
        "TX802 Brass - Stack Pressure",
        "TX802 Bass - Rubber Feedback",
        "TX802 Perc - Metallic Skin",
        "TX802 Organ - Choir Parallel",
        "TX802 Noise - Feedback Glass",
        "TX802 Pad - Evolving Matrix",
    }
    assert expected_names.issubset(set(MEDIA_AUDIO_PRESETS))
    assert {
        "IDM: Windowlicker Glass",
        "IDM: Drill EP Fragments",
        "IDM: Rubber FM Bass",
        "IDM: Metallic Ratchet",
        "IDM: Feedback Insect",
        "IDM: Chopped Choir",
    }.issubset(set(MEDIA_AUDIO_PRESETS))
    assert {
        "Lush Pad - Velvet Operator Cloud",
        "Lush Pad - Glass Choir Halo",
        "Lush Drone - Subharmonic Bloom",
        "Lush Pad - Tape Aurora",
        "Lush Drone - Fixed Starfield",
        "Lush Pad - Warm Digital Reeds",
        "Lush Drone - Blue Feedback Sea",
        "Lush Pad - Celestial Tines",
    }.issubset(set(MEDIA_AUDIO_PRESETS))


def test_media_audio_preset_state_round_trips_params_and_cdp_graph():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    engine.apply_preset("Lush Pad - Glass Choir Halo")
    engine.set_param("visual_morph_depth", 0.17)
    engine.set_param_mod_source("feedback", "motion")
    engine.cdp_graph.configure(enabled=True, wet_mix=0.42, cpu_preview=True)
    engine.cdp_graph.add_node("Filter Fixed")

    state = engine.export_preset_state()
    restored = MediaAudioEngine(sample_rate=8000, block_size=128)
    restored.import_preset_state(state)
    snapshot = restored.snapshot()

    assert snapshot["params"]["tx802_patch_name"] == "GLASS HALO"
    assert np.isclose(snapshot["params"]["visual_morph_depth"], 0.17)
    assert snapshot["param_mod_sources"]["feedback"] == "motion"
    assert snapshot["cdp_graph"]["enabled"] is True
    assert np.isclose(snapshot["cdp_graph"]["wet_mix"], 0.42)
    assert snapshot["cdp_graph"]["graph"]["nodes"][0]["effect"] == "Filter Fixed"


def test_idm_mode_changes_tx802_render_deterministically():
    metrics = {
        "brightness": 0.82,
        "motion": 0.7,
        "entropy": 0.72,
        "edge_density": 0.68,
        "red_energy": 0.2,
        "green_energy": 0.8,
        "blue_energy": 0.9,
        "glitch_spike": 0.75,
    }
    dry = MediaAudioEngine(sample_rate=8000, block_size=256)
    wet_a = MediaAudioEngine(sample_rate=8000, block_size=256)
    wet_b = MediaAudioEngine(sample_rate=8000, block_size=256)
    for engine in (dry, wet_a, wet_b):
        engine.set_source_mode("Live Display")
        engine.set_live_metrics(metrics)
        engine.apply_preset("TX802 Bell - Glass Ratio")
        assert engine.play(restart=True)
    for engine in (wet_a, wet_b):
        engine.set_param("idm_mode", True)
        engine.set_param("idm_intensity", 0.85)
        engine.set_param("idm_bpm", 181.0)
        engine.set_param("idm_grid_rate", 32.0)
        engine.set_param("idm_env_slice_depth", 0.9)
        engine.set_param("idm_modulator_chop_depth", 1.0)
        engine.set_param("idm_ratio_snap_depth", 0.7)
        engine.set_param("idm_bitcrush_depth", 0.25)
        for op in range(1, 7):
            engine.set_param(f"operator_glitch_depth_{op}", 0.7)
            engine.set_param(f"operator_burst_probability_{op}", 0.45)
            engine.set_param(f"operator_level_jitter_{op}", 0.4)
            engine.set_param(f"operator_ratio_snap_{op}", 0.8)

    dry_audio = np.vstack([dry.render(256) for _ in range(4)])
    wet_audio_a = np.vstack([wet_a.render(256) for _ in range(4)])
    wet_audio_b = np.vstack([wet_b.render(256) for _ in range(4)])

    assert np.isfinite(wet_audio_a).all()
    assert 0.0 < float(np.max(np.abs(wet_audio_a))) <= 1.0
    assert np.allclose(wet_audio_a, wet_audio_b)
    assert not np.allclose(dry_audio, wet_audio_a, atol=1e-4)


def test_idm_mode_generates_phrase_level_variation():
    engine = MediaAudioEngine(sample_rate=8000, block_size=256)
    engine.set_source_mode("Live Display")
    engine.set_live_metrics(
        {
            "brightness": 0.82,
            "motion": 0.74,
            "entropy": 0.68,
            "edge_density": 0.72,
            "red_energy": 0.25,
            "green_energy": 0.78,
            "blue_energy": 0.9,
            "glitch_spike": 0.7,
        }
    )
    engine.apply_preset("IDM: Metallic Ratchet")
    engine.set_param("idm_chaos", 0.85)
    engine.set_param("idm_phrase_depth", 0.9)
    engine.set_param("idm_polyrhythm_depth", 0.85)
    engine.set_param("idm_scene_reactivity", 0.9)
    engine.set_param("idm_density_drift", 0.8)
    assert engine.play(restart=True)

    blocks = [engine.render(512).mean(axis=1) for _ in range(10)]
    spectra = []
    for block in blocks:
        spectrum = np.abs(np.fft.rfft(block))
        spectra.append(spectrum / max(1e-9, float(np.linalg.norm(spectrum))))
    distances = [
        float(np.linalg.norm(first - second))
        for first, second in zip(spectra, spectra[1:])
    ]

    assert np.isfinite(np.asarray(blocks)).all()
    assert max(distances) > 0.08
    assert np.std(distances) > 0.01


def test_tx802_replacement_presets_have_distinct_spectra():
    metrics = {
        "brightness": 0.76,
        "motion": 0.42,
        "entropy": 0.68,
        "edge_density": 0.36,
        "red_energy": 0.64,
        "green_energy": 0.38,
        "blue_energy": 0.22,
        "center_mass_x": 0.62,
        "center_mass_y": 0.4,
    }
    spectra = []
    for name in MEDIA_AUDIO_PRESETS:
        engine = MediaAudioEngine(sample_rate=8000, block_size=256)
        engine.set_source_mode("Live Display")
        engine.set_live_metrics(metrics)
        engine.apply_preset(name)
        assert engine.play(restart=True)
        audio = np.vstack([engine.render(256) for _ in range(6)])
        spectrum = np.abs(np.fft.rfft(audio.mean(axis=1)))
        spectra.append(spectrum / max(1e-9, float(np.linalg.norm(spectrum))))

    for first, second in zip(spectra, spectra[1:]):
        assert float(np.linalg.norm(first - second)) > 0.05


def test_tx802_sysex_bank_import_preserves_patch_fields(tmp_path):
    def voice(name, algorithm):
        data = [0] * 128
        for op in range(6):
            base = op * 17
            data[base:base + 4] = [99, 70, 40, 55]
            data[base + 4:base + 8] = [99, 60, 30, 0]
            data[base + 12] = 7
            data[base + 13] = 3
            data[base + 14] = 80 - op * 5
            data[base + 15] = 1 + op
            data[base + 16] = 0
        data[102:106] = [50, 50, 50, 50]
        data[106:110] = [50, 50, 50, 50]
        data[110] = algorithm - 1
        data[111] = 4
        data[112] = 10
        data[113] = 0
        data[114] = 8
        data[115] = 4
        data[117] = 24
        data[118:128] = list(name.encode("ascii")[:10].ljust(10, b" "))
        return data

    bank = []
    for index in range(32):
        bank.extend(voice(f"TEST{index:02d}", 32 if index % 2 else 1))
    message = bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00] + bank + [dx7_checksum(bank), 0xF7])
    path = tmp_path / "bank.syx"
    path.write_bytes(message)

    patches = parse_tx802_sysex(str(path))
    params = tx802_patch_to_media_params(patches[0])

    assert len(patches) == 32
    assert patches[0].name == "TEST00"
    assert patches[0].algorithm == 1
    assert params["operator_output_level_1"] < params["operator_output_level_6"]
    assert params["voice_count"] == 1
    assert params["visual_morph_depth"] == 0.0
    assert params["visual_influence"] == 0.0
    assert params["chord_depth"] == 0.0
    assert params["modulation_index"] == 1.0
    assert params["lfo_depth"] == 0.0


def test_tx802_sysex_ratio_fine_scales_operator_ratio(tmp_path):
    data = [0] * 128
    for op in range(6):
        base = op * 17
        data[base:base + 4] = [99, 70, 40, 55]
        data[base + 4:base + 8] = [99, 60, 30, 0]
        data[base + 12] = 7
        data[base + 13] = 3
        data[base + 14] = 80
        data[base + 15] = 1
        data[base + 16] = 0
    data[15] = 8
    data[16] = 50
    data[5 * 17 + 15] = 0
    data[5 * 17 + 16] = 50
    data[102:106] = [50, 50, 50, 50]
    data[106:110] = [50, 50, 50, 50]
    data[110] = 31
    data[111] = 0
    data[117] = 24
    data[118:128] = list(b"RATIOFINE ")

    bank = data * 32
    message = bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00] + bank + [dx7_checksum(bank), 0xF7])
    path = tmp_path / "ratio_fine.syx"
    path.write_bytes(message)

    params = tx802_patch_to_media_params(parse_tx802_sysex(str(path))[0])

    assert np.isclose(params["operator_ratio_6"], 12.0)
    assert np.isclose(params["operator_ratio_1"], 0.75)
    assert np.isclose(params["carrier_freq"], 261.6255653005986)


def test_tx802_sysex_rejects_bad_checksum(tmp_path):
    bank = [0] * (128 * 32)
    message = bytearray([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00] + bank + [dx7_checksum(bank), 0xF7])
    message[-2] = (message[-2] + 1) & 0x7F
    path = tmp_path / "bad.syx"
    path.write_bytes(bytes(message))

    try:
        parse_tx802_sysex(str(path))
    except ValueError:
        return
    raise AssertionError("Bad TX802/DX7 checksum was accepted.")


def test_analyze_mp4_extracts_distinct_envelopes(tmp_path):
    path = tmp_path / "analysis.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (96, 64),
    )
    assert writer.isOpened()
    for index in range(24):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[..., 0] = index * 8
        frame[..., 1] = 180 - index * 4
        cv2.circle(frame, (8 + index * 3, 32), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    analysis = analyze_media(str(path))
    assert analysis.duration > 0.0
    assert len(analysis.timestamps) >= 2
    assert float(np.ptp(analysis.brightness)) > 0.0
    assert float(np.max(analysis.motion)) > 0.0
    assert analysis.colors.shape == (len(analysis.timestamps), 3)


def test_controller_async_load_without_starting_device(tmp_path):
    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (32, 32),
    )
    assert writer.isOpened()
    for value in (0, 80, 160, 240):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()

    controller = MediaAudioController(sample_rate=8000, block_size=128)
    try:
        controller.load_async(str(path))
        deadline = time.time() + 5.0
        while controller.snapshot()["loading"] and time.time() < deadline:
            time.sleep(0.01)
        state = controller.snapshot()
        assert state["loaded"]
        assert not state["error"]
        assert "Ready:" in state["status"]
    finally:
        controller.destroy()
