import numpy as np
import pytest

from frame_state import FrameState
from megaglitch.render_contract import (
    EgressFrame,
    FrameRole,
    MasterResolution,
    RenderContractError,
    make_egress_frame,
    prepare_master_ingress,
    validate_master_frame,
)


def test_primary_source_defines_master_without_spatial_resize():
    primary = np.arange(18 * 32 * 3, dtype=np.uint8).reshape(18, 32, 3)
    ingress = prepare_master_ingress(primary)

    assert ingress.resolution == MasterResolution(width=32, height=18)
    assert ingress.primary.shape == primary.shape
    assert np.array_equal(ingress.primary, primary)
    assert ingress.ingress_resize_count == 0


def test_secondary_source_is_resized_exactly_once_at_ingress():
    primary = np.zeros((18, 32, 3), dtype=np.uint8)
    secondary = np.zeros((9, 16, 3), dtype=np.uint8)

    ingress = prepare_master_ingress(primary, secondary)

    assert ingress.primary.shape == (18, 32, 3)
    assert ingress.secondary.shape == (18, 32, 3)
    assert ingress.ingress_resize_count == 1
    assert ingress.ingress_resize_bytes == secondary.nbytes


def test_terminal_preview_frame_cannot_reenter_master_path():
    resolution = MasterResolution(width=32, height=18)
    master = np.zeros(resolution.shape, dtype=np.uint8)
    preview = make_egress_frame(
        master,
        resolution,
        role=FrameRole.PREVIEW,
        scale=0.5,
    )

    assert isinstance(preview, EgressFrame)
    assert preview.data.shape == (9, 16, 3)
    with pytest.raises(RenderContractError, match="cannot re-enter"):
        prepare_master_ingress(preview)


def test_master_validation_rejects_shape_mismatch_instead_of_resizing():
    with pytest.raises(RenderContractError, match="does not match master shape"):
        validate_master_frame(
            np.zeros((9, 16, 3), dtype=np.uint8),
            MasterResolution(width=32, height=18),
            role=FrameRole.MASTER_INGRESS,
        )


def test_frame_state_resolution_change_resets_history_instead_of_scaling():
    state = FrameState()
    state.bind_master_shape((18, 32, 3))
    state.frame_buffer.append(np.full((18, 32, 3), 77, dtype=np.uint8))
    state.effect_states["feedback"] = {"frame": np.ones((18, 32, 3), dtype=np.uint8)}

    changed = state.bind_master_shape((36, 64, 3), allow_resolution_change=True)

    assert changed is True
    assert state.master_shape == (36, 64, 3)
    assert state.previous_frame.shape == (36, 64, 3)
    assert state.feedback_image.shape == (36, 64, 3)
    assert state.trail_image.shape == (36, 64, 3)
    assert state.frame_buffer == []
    assert state.effect_states == {}
    assert state.resolution_change_count == 1


def test_frame_state_rejects_mixed_resolution_history():
    state = FrameState()
    state.bind_master_shape((18, 32, 3))
    state.frame_buffer.append(np.zeros((9, 16, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="non-master shapes"):
        state.bind_master_shape((18, 32, 3))
