from types import SimpleNamespace

from megaglitch.audio.media_audio_engine import MediaAudioEngine
from megaglitch.audio.media_audio_sequencer import MediaAudioSequencer


def make_seq():
    engine = MediaAudioEngine(sample_rate=8000, block_size=128)
    applied = []

    def set_param(name, value):
        applied.append((name, value))
        engine.set_param(name, value)

    return MediaAudioSequencer(engine.PARAM_LIMITS, set_param), engine, applied


def test_step_sequencer_applies_clamped_parameter_values():
    seq, engine, applied = make_seq()
    for other_lane in seq.lanes[1:]:
        other_lane.enabled = False
    lane = seq.lanes[0]
    seq.set_lane_target(0, "modulation_index")
    lane.enabled = True
    lane.min_value = 1.0
    lane.max_value = 3.0
    lane.visual_amount = 0.0
    lane.random_amount = 0.0
    lane.probability = 1.0
    lane.steps[0].enabled = True
    lane.steps[0].value = 0.5
    lane.steps[0].probability = 1.0

    seq.apply_current_step({}, now=10.0, advance=True)

    applied_values = [value for name, value in applied if name == "modulation_index"]
    assert applied_values
    assert 1.0 <= applied_values[-1] <= 3.0
    assert engine.snapshot()["params"]["modulation_index"] == applied_values[-1]


def test_visual_mutation_stays_within_lane_bounds():
    seq, _, _ = make_seq()
    metrics = {"brightness": 0.9, "motion": 0.7, "entropy": 0.6, "edge_density": 0.5}
    seq.mutate_from_visual(metrics, strength=0.75)

    assert any(lane.enabled for lane in seq.lanes)
    for lane in seq.lanes:
        low, high = seq.target_bounds(lane.target)
        assert low <= lane.min_value <= high
        assert low <= lane.max_value <= high
        for step in lane.steps:
            assert 0.0 <= step.value <= 1.0
            assert 0.0 <= step.random_amount <= seq.chaos_ceiling


def test_beatstep_transport_and_clock_advance_steps():
    seq, _, _ = make_seq()
    seq.handle_midi(SimpleNamespace(type="start"), metrics={})
    assert seq.playing
    assert seq.clock_source == "beatstep"
    start_step = seq.step_index
    for _ in range(6):
        seq.handle_midi(SimpleNamespace(type="clock"), metrics={})
    assert seq.step_index == (start_step + 1) % seq.step_count
    seq.handle_midi(SimpleNamespace(type="stop"), metrics={})
    assert not seq.playing


def test_beatstep_notes_toggle_steps_and_lanes():
    seq, _, _ = make_seq()
    before = seq.lanes[0].steps[0].enabled
    assert seq.handle_midi(SimpleNamespace(type="note_on", note=36, velocity=100), metrics={})
    assert seq.lanes[0].steps[0].enabled is (not before)
    lane_before = seq.lanes[0].enabled
    assert seq.handle_midi(SimpleNamespace(type="note_on", note=52, velocity=100), metrics={})
    assert seq.lanes[0].enabled is (not lane_before)


def test_beatstep_cc_maps_performance_controls():
    seq, _, _ = make_seq()
    assert seq.handle_midi(SimpleNamespace(type="control_change", control=10, value=64), metrics={})
    assert 0.49 < seq.random_strength < 0.51
    assert seq.handle_midi(SimpleNamespace(type="control_change", control=20, value=127), metrics={})
    assert seq.lanes[0].visual_amount == 1.0
    assert seq.handle_midi(SimpleNamespace(type="control_change", control=40, value=0), metrics={})
    assert seq.lanes[0].steps[seq.step_index].value == 0.0
