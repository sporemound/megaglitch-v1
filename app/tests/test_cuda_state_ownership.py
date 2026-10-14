from __future__ import annotations

import numpy as np

import cuda_effects
import gpu_ops
from frame_state import FrameState


def _frame(value: int) -> np.ndarray:
    return np.full((8, 10, 3), value, dtype=np.uint8)


def test_cuda_temporal_state_is_isolated_by_stream_and_node() -> None:
    gpu_ops.set_use_cuda(False)
    store_a: dict = {}
    store_b: dict = {}
    params = {"window": 3, "mode": "exact", "mix": 1.0}

    for value in (0, 10):
        cuda_effects.apply_cuda_effect(
            "CUDA Temporal Median",
            _frame(value),
            params,
            state_store=store_a,
            state_namespace="same-node-id",
        )

    # A distinct deck/stream with the same serialized node id starts clean.
    isolated = cuda_effects.apply_cuda_effect(
        "CUDA Temporal Median",
        _frame(200),
        params,
        state_store=store_b,
        state_namespace="same-node-id",
    )[0]
    assert np.array_equal(isolated, _frame(200))

    # A duplicate node in the same stream also owns a separate ring.
    duplicate = cuda_effects.apply_cuda_effect(
        "CUDA Temporal Median",
        _frame(150),
        params,
        state_store=store_a,
        state_namespace="duplicate-node-id",
    )[0]
    assert np.array_equal(duplicate, _frame(150))

    temporal_keys_a = [key for key in store_a if isinstance(key, tuple)]
    temporal_keys_b = [key for key in store_b if isinstance(key, tuple)]
    assert {key[0] for key in temporal_keys_a} == {
        "same-node-id",
        "duplicate-node-id",
    }
    assert {key[0] for key in temporal_keys_b} == {"same-node-id"}


def test_frame_state_reset_discards_owned_cuda_temporal_state() -> None:
    state = FrameState()
    state.bind_master_shape((8, 10, 3))
    cuda_store = state.effect_states.setdefault("cuda_temporal_states", {})
    cuda_effects.apply_cuda_effect(
        "CUDA Temporal Median",
        _frame(25),
        {"window": 3, "mix": 1.0},
        state_store=cuda_store,
        state_namespace="node-1",
    )
    assert any(isinstance(key, tuple) for key in cuda_store)

    state.reset()
    state.bind_master_shape((8, 10, 3))
    assert "cuda_temporal_states" not in state.effect_states


def test_luma_fluid_temporal_arrays_are_master_resolution(monkeypatch) -> None:
    store: dict = {}
    monkeypatch.setattr(gpu_ops, "get_use_cuda", lambda: True)
    frame = _frame(60)
    cuda_effects.apply_cuda_effect(
        "CUDA Luma-Keyed Fluid Dynamics",
        frame,
        {"mix": 1.0, "grid_resolution": 32},
        state_store=store,
        state_namespace="fluid",
    )
    states = [value for key, value in store.items() if isinstance(key, tuple)]
    assert len(states) == 1
    temporal = states[0]
    for name in ("vel_x", "vel_y", "dye", "prev_gray"):
        assert temporal[name].shape[:2] == frame.shape[:2]
