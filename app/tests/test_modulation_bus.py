import math

from megaglitch.audio.modulation_bus import ModulationBus, ModulationMapping


def test_linear_source_to_target_mapping():
    bus = ModulationBus()
    bus.add_mapping(
        source="visual.energy",
        target="fm_synth_node.master_gain",
        source_range=(0.0, 1.0),
        target_range=(0.1, 0.5),
        smoothing_ms=0,
    )
    bus.publish("visual.energy", 0.5)
    values = bus.update(1.0 / 60.0)
    assert math.isclose(values["fm_synth_node.master_gain"], 0.3)


def test_curves_clamping_and_stepping():
    bus = ModulationBus()
    bus.add_mapping(
        source="x",
        target="smooth",
        source_range=(0.0, 1.0),
        target_range=(0.0, 1.0),
        curve="smoothstep",
        smoothing_ms=0,
    )
    bus.add_mapping(
        source="x",
        target="step",
        source_range=(0.0, 1.0),
        target_range=(0.0, 1.0),
        curve="stepped",
        steps=3,
        smoothing_ms=0,
    )
    bus.publish("x", 2.0)
    values = bus.update(0.1)
    assert values["smooth"] == 1.0
    assert values["step"] == 1.0


def test_many_sources_blend_into_one_target():
    bus = ModulationBus()
    bus.add_mapping(source="a", target="target", target_range=(0.0, 1.0), smoothing_ms=0)
    bus.add_mapping(source="b", target="target", target_range=(0.0, 1.0), smoothing_ms=0, blend_mode="add")
    bus.publish("a", 0.25)
    bus.publish("b", 0.5)
    assert bus.update(0.1)["target"] == 0.75


def test_manual_override_has_priority():
    bus = ModulationBus()
    bus.add_mapping(source="a", target="target", smoothing_ms=0)
    bus.publish("a", 0.1)
    bus.set_manual_override("target", 0.9)
    assert bus.update(0.1)["target"] == 0.9
    bus.clear_manual_override("target")
    assert bus.update(0.1)["target"] == 0.1


def test_smoothing_and_slew_limit_are_deterministic():
    def run():
        bus = ModulationBus()
        bus.add_mapping(ModulationMapping(source="a", target="target", smoothing_ms=100, slew_limit=2.0))
        bus.publish("a", 0.0)
        bus.update(0.05)
        bus.publish("a", 1.0)
        return [bus.update(0.05)["target"] for _ in range(4)]

    first = run()
    assert first == run()
    assert first[0] <= 0.1
    assert first == sorted(first)


def test_bidirectional_named_routes_and_debug_inspection():
    bus = ModulationBus()
    mapping = bus.add_mapping(source="fm_synth_node.rms", target="bubble.sparkle", smoothing_ms=0)
    bus.publish("fm_synth_node.rms", 0.7)
    bus.update(0.1)
    snapshot = bus.inspect()
    assert snapshot["values"]["bubble.sparkle"] == 0.7
    assert snapshot["mappings"][0]["source"] == mapping.source
