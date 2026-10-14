"""Render-domain contracts for the quality-first MegaGlitch frame path.

The processing graph accepts only ``MASTER_INGRESS`` frames and produces only
``MASTER_PROGRAM`` frames.  Preview, virtual-camera, OBS, and UI frames are
terminal branches: callers may resize them, but they must never submit them to
the graph again.

Resolution changes are explicit ingress events.  This module never reconciles
temporal state by resizing it; the worker resets temporal state when the bound
master resolution changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np


class RenderContractError(ValueError):
    """Raised when a frame crosses a render-domain boundary incorrectly."""


class RenderMode(str, Enum):
    LIVE = "live"
    OFFLINE_MASTER = "offline_master"


class FrameRole(str, Enum):
    MASTER_INGRESS = "master_ingress"
    MASTER_PROGRAM = "master_program"
    PREVIEW = "preview"
    UI = "ui"
    OBS = "obs"
    VIRTUAL_CAMERA = "virtual_camera"
    ENCODER = "encoder"


_TERMINAL_ROLES = {
    FrameRole.PREVIEW,
    FrameRole.UI,
    FrameRole.OBS,
    FrameRole.VIRTUAL_CAMERA,
    FrameRole.ENCODER,
}


@dataclass(frozen=True)
class MasterResolution:
    width: int
    height: int
    channels: int = 3

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise RenderContractError("master resolution must be positive")
        if self.channels != 3:
            raise RenderContractError("the master path is RGB with exactly 3 channels")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def to_dict(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "MasterResolution":
        if not isinstance(value, dict):
            raise RenderContractError("master_resolution metadata is required")
        return cls(
            width=int(value.get("width", 0)),
            height=int(value.get("height", 0)),
            channels=int(value.get("channels", 3)),
        )


@dataclass(frozen=True)
class MasterIngress:
    primary: Optional[np.ndarray]
    secondary: Optional[np.ndarray]
    resolution: MasterResolution
    ingress_resize_count: int = 0
    ingress_resize_bytes: int = 0


@dataclass(frozen=True)
class EgressFrame:
    """A terminal branch frame which is intentionally ineligible for graph input."""

    data: np.ndarray
    role: FrameRole
    master_resolution: MasterResolution

    def __post_init__(self) -> None:
        if self.role not in _TERMINAL_ROLES:
            raise RenderContractError(f"{self.role.value} is not a terminal egress role")


def _frame_to_rgb_u8(frame: Any, *, name: str) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if isinstance(frame, EgressFrame):
        raise RenderContractError(
            f"{name} is a terminal {frame.role.value} frame and cannot re-enter the graph"
        )

    try:
        from PIL import Image

        if isinstance(frame, Image.Image):
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        else:
            array = np.asarray(frame)
    except ImportError:
        array = np.asarray(frame)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[2] != 3:
        raise RenderContractError(f"{name} must be HxWx3 RGB, got {array.shape}")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def frame_resolution(frame: Any, *, name: str = "frame") -> MasterResolution:
    array = _frame_to_rgb_u8(frame, name=name)
    if array is None:
        raise RenderContractError(f"{name} is missing")
    return MasterResolution(width=int(array.shape[1]), height=int(array.shape[0]))


def validate_master_frame(
    frame: Any,
    resolution: MasterResolution,
    *,
    role: FrameRole,
    name: str = "frame",
) -> Optional[np.ndarray]:
    if role not in {FrameRole.MASTER_INGRESS, FrameRole.MASTER_PROGRAM}:
        raise RenderContractError(
            f"{name} has role {role.value}; terminal branches cannot enter the master path"
        )
    array = _frame_to_rgb_u8(frame, name=name)
    if array is not None and array.shape != resolution.shape:
        raise RenderContractError(
            f"{name} shape {array.shape} does not match master shape {resolution.shape}"
        )
    return array


def _resize_at_ingress(frame: np.ndarray, resolution: MasterResolution) -> np.ndarray:
    """Perform the one deliberate resize allowed when normalizing a source deck."""
    import cv2

    shrinking = frame.shape[0] > resolution.height or frame.shape[1] > resolution.width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    resized = cv2.resize(
        frame,
        (resolution.width, resolution.height),
        interpolation=interpolation,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8)


def prepare_master_ingress(
    primary: Any,
    secondary: Any = None,
    *,
    fallback_size: Tuple[int, int] = (1280, 720),
) -> MasterIngress:
    """Normalize source decks once and bind the full-resolution master frame.

    The primary source defines master resolution.  When only the secondary deck
    has media, it defines the resolution.  Generator-only graphs use the
    explicit fallback size.  The defining source is never spatially resized.
    """
    primary_array = _frame_to_rgb_u8(primary, name="primary source")
    secondary_array = _frame_to_rgb_u8(secondary, name="secondary source")

    reference = primary_array if primary_array is not None else secondary_array
    if reference is None:
        width, height = int(fallback_size[0]), int(fallback_size[1])
        resolution = MasterResolution(width=width, height=height)
    else:
        resolution = MasterResolution(width=reference.shape[1], height=reference.shape[0])

    resize_count = 0
    resize_bytes = 0
    if primary_array is not None and primary_array.shape != resolution.shape:
        resize_bytes += int(primary_array.nbytes)
        primary_array = _resize_at_ingress(primary_array, resolution)
        resize_count += 1
    if secondary_array is not None and secondary_array.shape != resolution.shape:
        resize_bytes += int(secondary_array.nbytes)
        secondary_array = _resize_at_ingress(secondary_array, resolution)
        resize_count += 1

    validate_master_frame(
        primary_array,
        resolution,
        role=FrameRole.MASTER_INGRESS,
        name="primary source",
    )
    validate_master_frame(
        secondary_array,
        resolution,
        role=FrameRole.MASTER_INGRESS,
        name="secondary source",
    )
    return MasterIngress(
        primary=primary_array,
        secondary=secondary_array,
        resolution=resolution,
        ingress_resize_count=resize_count,
        ingress_resize_bytes=resize_bytes,
    )


def make_egress_frame(
    master_frame: Any,
    resolution: MasterResolution,
    *,
    role: FrameRole,
    scale: float = 1.0,
    target_size: Optional[Tuple[int, int]] = None,
) -> EgressFrame:
    """Create a one-way preview/output branch from a validated master frame."""
    master = validate_master_frame(
        master_frame,
        resolution,
        role=FrameRole.MASTER_PROGRAM,
        name="master program frame",
    )
    if master is None:
        raise RenderContractError("master program frame is missing")
    if role not in _TERMINAL_ROLES:
        raise RenderContractError(f"{role.value} is not a valid egress role")

    if target_size is None:
        bounded_scale = float(np.clip(float(scale), 0.05, 1.0))
        target_size = (
            max(1, int(round(resolution.width * bounded_scale))),
            max(1, int(round(resolution.height * bounded_scale))),
        )
    width, height = int(target_size[0]), int(target_size[1])
    if (width, height) == (resolution.width, resolution.height):
        branch = master
    else:
        import cv2

        shrinking = width < resolution.width or height < resolution.height
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
        branch = cv2.resize(master, (width, height), interpolation=interpolation)
    return EgressFrame(
        data=np.ascontiguousarray(branch, dtype=np.uint8),
        role=role,
        master_resolution=resolution,
    )


def parse_render_mode(value: Any) -> RenderMode:
    try:
        return RenderMode(str(value or RenderMode.LIVE.value))
    except ValueError as exc:
        raise RenderContractError(f"unsupported render mode: {value!r}") from exc
