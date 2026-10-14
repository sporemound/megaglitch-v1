import pytest


moderngl = pytest.importorskip("moderngl")


def make_context():
    try:
        return moderngl.create_standalone_context(require=330)
    except Exception as exc:
        pytest.skip(f"standalone ModernGL context unavailable: {exc}")


def make_node(width=64, height=48, particle_texture_size=16):
    from megaglitch.nodes.bubble_sim_generator_node import BubbleSimGeneratorNode

    ctx = make_context()
    node = BubbleSimGeneratorNode(ctx, width, height, particle_texture_size=particle_texture_size)
    return ctx, node


def test_lifecycle_allocates_outputs():
    ctx, node = make_node()
    try:
        outputs = node.get_output_textures()
        assert set(outputs) == {
            "bubble_mask",
            "bubble_normal",
            "bubble_depth",
            "bubble_refraction",
            "bubble_highlight",
            "bubble_debug",
        }
        assert outputs["bubble_mask"].size == (64, 48)
        assert node.particle_count == 16 * 16
    finally:
        node.destroy()
        ctx.release()


def test_param_clamping_and_unknown_param():
    ctx, node = make_node()
    try:
        node.set_param("density", 3.0)
        node.set_param("drag", -1.0)
        node.set_param("particle_texture_size", 8)
        assert node.get_param("density") == 1.0
        assert node.get_param("drag") == 0.0
        assert node.get_param("particle_texture_size") == 16
        with pytest.raises(KeyError):
            node.set_param("missing", 1.0)
    finally:
        node.destroy()
        ctx.release()


def test_update_ping_pong_and_disabled_skip():
    ctx, node = make_node()
    try:
        start = node._state_index
        node.update(time=1.0, dt=1.0 / 60.0, slot_time=1.0)
        assert node._state_index == 1 - start
        after_update = node._state_index
        node.set_param("enabled", False)
        node.update(time=2.0, dt=1.0 / 60.0, slot_time=2.0)
        assert node._state_index == after_update
    finally:
        node.destroy()
        ctx.release()


def test_reset_same_seed_is_deterministic():
    ctx, node = make_node()
    try:
        node.reset(seed=1234)
        first = node._state_textures[node._state_index].read()
        node.update(time=1.0, dt=1.0 / 60.0, slot_time=1.0)
        node.reset(seed=1234)
        second = node._state_textures[node._state_index].read()
        assert first == second
    finally:
        node.destroy()
        ctx.release()


def test_resize_recreates_output_textures():
    ctx, node = make_node()
    try:
        old_mask = node.get_output_texture("bubble_mask")
        node.resize(80, 60)
        new_mask = node.get_output_texture("bubble_mask")
        assert new_mask.size == (80, 60)
        assert new_mask is not old_mask
    finally:
        node.destroy()
        ctx.release()


def test_v2_runtime_produces_delayed_distinct_outputs():
    from megaglitch.nodes import BubbleSimGenerator2Runtime

    ctx = make_context()
    node = BubbleSimGenerator2Runtime(ctx, 64, 48, particle_texture_size=16)
    try:
        ready = None
        for index in range(4):
            node.update(index / 60.0, 1.0 / 60.0, index / 60.0)
            ready = node.render_and_readback()
        assert ready is not None
        assert set(ready) == {
            "bubble_mask",
            "bubble_normal",
            "bubble_depth",
            "bubble_refraction",
            "bubble_highlight",
        }
        assert len({ready[name].tobytes() for name in ready}) > 3
    finally:
        node.destroy()
        ctx.release()
