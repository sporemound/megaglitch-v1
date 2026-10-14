"""UI-independent media sources for the canonical Mycotroph session runtime.

The service deliberately owns only source lifecycle and transport.  It has no
Dear PyGui dependency and performs no camera, window, or device enumeration at
import time.  Every returned frame is a uniquely owned, C-contiguous RGB
``numpy.uint8`` array.

Static and animated images use Pillow, loaded only when one of those sources is
opened.  OpenCV is loaded only when a video source is opened.  Webcam and
window capture are optional adapter points: unless an application explicitly
registers an adapter, their capability probes report an unavailable reason and
opening them raises a structured error.  They never substitute generated
frames for an unavailable live source.
"""

from __future__ import annotations

import bisect
import math
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


SNAPSHOT_VERSION = 1
PIXEL_FORMAT = "RGB8"
BUILTIN_SOURCE_KINDS = ("test_pattern", "image", "gif", "video")
OPTIONAL_SOURCE_KINDS = ("webcam", "window_capture")

# This is copied into image/GIF telemetry so consumers do not have to infer how
# a four-channel input became the canonical three-channel frame.
ALPHA_HANDLING_POLICY: dict[str, Any] = {
    "input": "straight RGBA",
    "output": PIXEL_FORMAT,
    "policy": "composite_over_solid_background",
    "rounding": "nearest_integer",
    "default_background_rgb": [0, 0, 0],
}


class SourceError(RuntimeError):
    """Stable, serialisable source-service failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        recoverable: bool = True,
        path: str | os.PathLike[str] | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.recoverable = bool(recoverable)
        self.path = str(path) if path is not None else None
        self.details = dict(details)

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.path is not None:
            error["path"] = self.path
        if self.details:
            error["details"] = dict(self.details)
        return error


def _load_pillow():
    try:
        from PIL import Image, ImageOps
    except Exception as error:  # pragma: no cover - installation dependent
        raise SourceError(
            "pillow_unavailable",
            "Pillow is required for image and animated GIF sources",
            exception=type(error).__name__,
        ) from error
    return Image, ImageOps


def _load_cv2():
    """Load OpenCV on first video connection, never on module import."""

    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - installation dependent
        raise SourceError(
            "opencv_unavailable",
            "OpenCV is required for video sources",
            exception=type(error).__name__,
        ) from error
    return cv2


def _checked_path(path_value: Any, kind: str) -> Path:
    try:
        raw_path = os.fspath(path_value)
    except TypeError as error:
        raise SourceError(
            "source_path_required", f"{kind} source requires a filesystem path"
        ) from error
    if not isinstance(raw_path, (str, bytes)) or not raw_path:
        raise SourceError("source_path_required", f"{kind} source requires a filesystem path")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise SourceError(
            "source_not_found",
            f"source file does not exist: {path}",
            path=path,
            kind=kind,
        )
    return path


def _dimension(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise SourceError("invalid_dimensions", f"{name} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceError("invalid_dimensions", f"{name} must be a positive integer") from error
    if converted <= 0 or converted != value:
        raise SourceError("invalid_dimensions", f"{name} must be a positive integer")
    return converted


def _seconds(value: Any, *, code: str = "invalid_seek", name: str = "seconds") -> float:
    if isinstance(value, bool):
        raise SourceError(code, f"{name} must be finite and non-negative")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceError(code, f"{name} must be finite and non-negative") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise SourceError(code, f"{name} must be finite and non-negative")
    return converted


def _background_rgb(value: Sequence[int] | None) -> tuple[int, int, int]:
    if value is None:
        return (0, 0, 0)
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise SourceError(
            "invalid_alpha_background", "alpha background must contain exactly three RGB values"
        )
    converted: list[int] = []
    for channel in value:
        if isinstance(channel, bool):
            raise SourceError(
                "invalid_alpha_background", "alpha background channels must be integers from 0 to 255"
            )
        try:
            number = int(channel)
        except (TypeError, ValueError, OverflowError) as error:
            raise SourceError(
                "invalid_alpha_background", "alpha background channels must be integers from 0 to 255"
            ) from error
        if number != channel or not 0 <= number <= 255:
            raise SourceError(
                "invalid_alpha_background", "alpha background channels must be integers from 0 to 255"
            )
        converted.append(number)
    return tuple(converted)  # type: ignore[return-value]


def _alpha_policy(background: tuple[int, int, int]) -> dict[str, Any]:
    policy = dict(ALPHA_HANDLING_POLICY)
    policy["background_rgb"] = list(background)
    return policy


def _rgba_to_rgb8(rgba: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise SourceError("image_decode_failed", f"decoded RGBA image has invalid shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    rgb = array[:, :, :3].astype(np.uint16)
    alpha = array[:, :, 3:4].astype(np.uint16)
    matte = np.asarray(background, dtype=np.uint16).reshape((1, 1, 3))
    # Straight-alpha composition with integer nearest rounding.
    composed = (rgb * alpha + matte * (255 - alpha) + 127) // 255
    return np.ascontiguousarray(composed.astype(np.uint8))


def _pil_frame_to_rgb8(image: Any, background: tuple[int, int, int]) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return _rgba_to_rgb8(rgba, background)


def _opencv_frame_to_rgb8(frame: Any, cv2: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 1:
        rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 3:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    else:
        raise SourceError("video_decode_failed", f"decoded video frame has invalid shape {array.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _frame_data(width: int | None, height: int | None) -> dict[str, Any]:
    return {
        "pixel_format": PIXEL_FORMAT,
        "dtype": "uint8",
        "channels": 3,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "contiguous": True,
        "ownership": "caller_owned_copy",
    }


class FrameSource:
    """Base class for a connected source.

    ``read`` returns a new array on every call because the frame pipeline may
    mutate a uniquely owned input.  ``close`` is deterministic and idempotent.
    """

    kind = "unknown"

    def __init__(self, label: str) -> None:
        self.label = str(label)
        self._lock = threading.RLock()
        self._closed = False
        self._frames_read = 0
        self._width: int | None = None
        self._height: int | None = None

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def ready(self) -> bool:
        with self._lock:
            return not self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise SourceError(
                "source_closed",
                f"{self.kind} source is closed",
                recoverable=False,
                kind=self.kind,
            )

    def read(
        self,
        sequence: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def advance(self, elapsed_seconds: float) -> None:
        del elapsed_seconds
        raise SourceError(
            "transport_unsupported",
            f"{self.kind} source has no timed transport",
            kind=self.kind,
        )

    def play(self) -> None:
        raise SourceError(
            "transport_unsupported", f"{self.kind} source cannot play", kind=self.kind
        )

    def pause(self) -> None:
        raise SourceError(
            "transport_unsupported", f"{self.kind} source cannot pause", kind=self.kind
        )

    def stop(self) -> None:
        raise SourceError(
            "transport_unsupported", f"{self.kind} source cannot stop", kind=self.kind
        )

    def seek(self, seconds: float) -> None:
        del seconds
        raise SourceError(
            "seek_unsupported", f"{self.kind} source cannot seek", kind=self.kind
        )

    def set_loop(self, enabled: bool) -> None:
        del enabled
        raise SourceError(
            "loop_unsupported", f"{self.kind} source cannot loop", kind=self.kind
        )

    def _capabilities(self) -> dict[str, bool]:
        return {
            "read": not self._closed,
            "play": False,
            "pause": False,
            "stop": False,
            "seek": False,
            "loop": False,
            "timed_advance": False,
            "per_frame_durations": False,
        }

    def _transport_snapshot(self) -> dict[str, Any]:
        return {
            "supported": False,
            "state": "not_applicable",
            "position_seconds": None,
            "duration_seconds": None,
            "fps": None,
            "frame_index": None,
            "frame_count": None,
            "loop": None,
            "loops_completed": 0,
        }

    def _metadata(self) -> dict[str, Any]:
        return {}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": SNAPSHOT_VERSION,
                "kind": self.kind,
                "label": self.label,
                "connected": not self._closed,
                "ready": not self._closed,
                "state": "closed" if self._closed else "ready",
                "frames_read": self._frames_read,
                "frame": _frame_data(self._width, self._height),
                "capabilities": self._capabilities(),
                "transport": self._transport_snapshot(),
                "metadata": self._metadata(),
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def deterministic_test_pattern(sequence: int, width: int, height: int) -> np.ndarray:
    """Return the bridge-compatible deterministic RGB8 test pattern."""

    checked_width = _dimension(width, "width")
    checked_height = _dimension(height, "height")
    if isinstance(sequence, bool):
        raise SourceError("invalid_sequence", "sequence must be an integer")
    try:
        checked_sequence = int(sequence)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceError("invalid_sequence", "sequence must be an integer") from error
    if checked_sequence != sequence:
        raise SourceError("invalid_sequence", "sequence must be an integer")
    yy, xx = np.indices((checked_height, checked_width), dtype=np.uint16)
    return np.ascontiguousarray(
        np.stack(
            (
                (xx + checked_sequence * 3) % 256,
                (yy * 2 + checked_sequence * 5) % 256,
                ((xx ^ yy) + checked_sequence * 7) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
    )


class TestPatternSource(FrameSource):
    kind = "test_pattern"

    def __init__(self, *, width: int = 160, height: int = 90) -> None:
        super().__init__("TEST PATTERN")
        self._width = _dimension(width, "width")
        self._height = _dimension(height, "height")
        self._next_sequence = 0

    def read(
        self,
        sequence: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> np.ndarray:
        del elapsed_seconds
        with self._lock:
            self._require_open()
            output_width = self._width if width is None else _dimension(width, "width")
            output_height = self._height if height is None else _dimension(height, "height")
            if sequence is None:
                output_sequence = self._next_sequence
                self._next_sequence += 1
            else:
                output_sequence = sequence
            frame = deterministic_test_pattern(output_sequence, output_width, output_height)
            self._width, self._height = output_width, output_height
            self._frames_read += 1
            return frame

    def _metadata(self) -> dict[str, Any]:
        return {"generator": "megaglitch_deterministic_rgb_v1"}


class ImageSource(FrameSource):
    kind = "image"

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        alpha_background: Sequence[int] | None = None,
    ) -> None:
        checked_path = _checked_path(path, self.kind)
        super().__init__(str(checked_path))
        self.path = checked_path
        self._background = _background_rgb(alpha_background)
        Image, ImageOps = _load_pillow()
        try:
            with Image.open(self.path) as opened:
                original_mode = opened.mode
                alpha_present = "A" in opened.getbands() or "transparency" in opened.info
                oriented = ImageOps.exif_transpose(opened)
                self._image = _pil_frame_to_rgb8(oriented, self._background)
        except SourceError:
            raise
        except Exception as error:
            self._closed = True
            raise SourceError(
                "image_decode_failed",
                f"Pillow could not decode image: {self.path}",
                path=self.path,
                exception=type(error).__name__,
            ) from error
        self._height, self._width = self._image.shape[:2]
        self._original_mode = str(original_mode)
        self._alpha_present = bool(alpha_present)

    def read(
        self,
        sequence: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> np.ndarray:
        del sequence, width, height, elapsed_seconds
        with self._lock:
            self._require_open()
            self._frames_read += 1
            return self._image.copy(order="C")

    def _metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "original_mode": self._original_mode,
            "alpha_present": self._alpha_present,
            "alpha_handling": _alpha_policy(self._background),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._image = np.empty((0, 0, 3), dtype=np.uint8)
            self._closed = True


class AnimatedGifSource(FrameSource):
    kind = "gif"

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        loop: bool = True,
        alpha_background: Sequence[int] | None = None,
        default_frame_duration_seconds: float = 0.1,
    ) -> None:
        checked_path = _checked_path(path, self.kind)
        super().__init__(str(checked_path))
        self.path = checked_path
        self._background = _background_rgb(alpha_background)
        fallback_duration = _seconds(
            default_frame_duration_seconds,
            code="invalid_frame_duration",
            name="default frame duration",
        )
        if fallback_duration <= 0.0:
            raise SourceError(
                "invalid_frame_duration", "default frame duration must be greater than zero"
            )
        Image, _ImageOps = _load_pillow()
        frames: list[np.ndarray] = []
        durations: list[float] = []
        source_modes: list[str] = []
        alpha_present = False
        try:
            with Image.open(self.path) as opened:
                frame_count = int(getattr(opened, "n_frames", 1) or 1)
                file_loop_count = opened.info.get("loop")
                for index in range(frame_count):
                    opened.seek(index)
                    source_modes.append(str(opened.mode))
                    alpha_present = alpha_present or "A" in opened.getbands() or (
                        "transparency" in opened.info
                    )
                    frame = _pil_frame_to_rgb8(opened, self._background)
                    frames.append(frame)
                    raw_duration = opened.info.get("duration", fallback_duration * 1000.0)
                    try:
                        duration = float(raw_duration) / 1000.0
                    except (TypeError, ValueError, OverflowError):
                        duration = fallback_duration
                    if not math.isfinite(duration) or duration <= 0.0:
                        duration = fallback_duration
                    durations.append(duration)
        except SourceError:
            raise
        except Exception as error:
            self._closed = True
            raise SourceError(
                "gif_decode_failed",
                f"Pillow could not decode animated GIF: {self.path}",
                path=self.path,
                exception=type(error).__name__,
            ) from error
        if not frames:
            self._closed = True
            raise SourceError(
                "gif_decode_failed", f"animated GIF contains no decodable frames: {self.path}", path=self.path
            )
        self._frames = frames
        self._durations = tuple(durations)
        self._starts: tuple[float, ...] = tuple(
            float(sum(durations[:index])) for index in range(len(durations))
        )
        self._duration = float(sum(durations))
        self._frame_index = 0
        self._offset_seconds = 0.0
        self._transport_state = "playing"
        self._loop = bool(loop)
        self._loops_completed = 0
        self._file_loop_count = file_loop_count
        self._source_modes = tuple(source_modes)
        self._alpha_present = bool(alpha_present)
        self._height, self._width = frames[0].shape[:2]

    @property
    def frame_durations(self) -> tuple[float, ...]:
        return self._durations

    def _current_position(self) -> float:
        if self._transport_state == "ended":
            return self._duration
        return min(
            self._duration,
            self._starts[self._frame_index] + self._offset_seconds,
        )

    def _seek_locked(self, target: float) -> None:
        clamped = min(target, self._duration)
        if clamped >= self._duration:
            self._frame_index = len(self._frames) - 1
            self._offset_seconds = self._durations[-1]
            return
        index = max(0, bisect.bisect_right(self._starts, clamped) - 1)
        self._frame_index = min(index, len(self._frames) - 1)
        self._offset_seconds = clamped - self._starts[self._frame_index]

    def _advance_locked(self, elapsed: float) -> None:
        if self._transport_state != "playing" or elapsed <= 0.0:
            return
        target = self._current_position() + elapsed
        if self._loop and self._duration > 0.0:
            completed = int(target // self._duration)
            if completed:
                self._loops_completed += completed
                target = target % self._duration
            self._seek_locked(target)
            return
        if target >= self._duration:
            self._seek_locked(self._duration)
            self._transport_state = "ended"
            return
        self._seek_locked(target)

    def read(
        self,
        sequence: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> np.ndarray:
        del sequence, width, height
        with self._lock:
            self._require_open()
            if elapsed_seconds is not None:
                self._advance_locked(
                    _seconds(elapsed_seconds, code="invalid_elapsed", name="elapsed seconds")
                )
            frame = self._frames[self._frame_index]
            self._height, self._width = frame.shape[:2]
            self._frames_read += 1
            return frame.copy(order="C")

    def advance(self, elapsed_seconds: float) -> None:
        with self._lock:
            self._require_open()
            self._advance_locked(
                _seconds(elapsed_seconds, code="invalid_elapsed", name="elapsed seconds")
            )

    def play(self) -> None:
        with self._lock:
            self._require_open()
            if self._transport_state == "ended":
                self._seek_locked(0.0)
            self._transport_state = "playing"

    def pause(self) -> None:
        with self._lock:
            self._require_open()
            if self._transport_state != "ended":
                self._transport_state = "paused"

    def stop(self) -> None:
        with self._lock:
            self._require_open()
            self._seek_locked(0.0)
            self._transport_state = "stopped"

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._require_open()
            self._seek_locked(_seconds(seconds))

    def set_loop(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise SourceError("invalid_loop", "loop must be a boolean")
        with self._lock:
            self._require_open()
            self._loop = enabled

    def _capabilities(self) -> dict[str, bool]:
        return {
            "read": not self._closed,
            "play": not self._closed,
            "pause": not self._closed,
            "stop": not self._closed,
            "seek": not self._closed,
            "loop": not self._closed,
            "timed_advance": not self._closed,
            "per_frame_durations": True,
        }

    def _transport_snapshot(self) -> dict[str, Any]:
        return {
            "supported": True,
            "state": "closed" if self._closed else self._transport_state,
            "position_seconds": self._current_position(),
            "duration_seconds": self._duration,
            "fps": None,
            "frame_index": self._frame_index,
            "frame_count": len(self._frames),
            "frame_durations_seconds": list(self._durations),
            "loop": self._loop,
            "loops_completed": self._loops_completed,
        }

    def _metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_modes": list(self._source_modes),
            "file_loop_count": self._file_loop_count,
            "alpha_present": self._alpha_present,
            "alpha_handling": _alpha_policy(self._background),
        }

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot["state"] = "closed" if self._closed else self._transport_state
        return snapshot

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._frames.clear()
            self._closed = True
            self._transport_state = "closed"


class VideoSource(FrameSource):
    kind = "video"

    def __init__(self, path: str | os.PathLike[str], *, loop: bool = True) -> None:
        checked_path = _checked_path(path, self.kind)
        super().__init__(str(checked_path))
        self.path = checked_path
        cv2 = _load_cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            self._closed = True
            raise SourceError(
                "video_open_failed",
                f"OpenCV could not open video: {self.path}",
                path=self.path,
            )
        ok, frame = capture.read()
        if not ok or frame is None or np.asarray(frame).size == 0:
            capture.release()
            self._closed = True
            raise SourceError(
                "video_decode_failed",
                f"OpenCV could not decode the first video frame: {self.path}",
                path=self.path,
            )
        try:
            first = _opencv_frame_to_rgb8(frame, cv2)
        except Exception:
            capture.release()
            self._closed = True
            raise
        self._cv2 = cv2
        self._capture = capture
        self._current_frame = first
        self._height, self._width = first.shape[:2]
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0.0 else None
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self._frame_count = (
            max(1, int(round(raw_count))) if math.isfinite(raw_count) and raw_count > 0.0 else None
        )
        self._duration = (
            float(self._frame_count / self._fps)
            if self._frame_count is not None and self._fps is not None
            else None
        )
        self._frame_index = 0
        self._position_seconds = 0.0
        self._return_current_once = True
        self._transport_state = "playing"
        self._loop = bool(loop)
        self._loops_completed = 0

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def duration_seconds(self) -> float | None:
        return self._duration

    def _update_position_from_backend(self) -> None:
        raw_index = float(self._capture.get(self._cv2.CAP_PROP_POS_FRAMES) or 0.0) - 1.0
        if math.isfinite(raw_index) and raw_index >= 0.0:
            self._frame_index = int(round(raw_index))
        if self._fps is not None:
            self._position_seconds = self._frame_index / self._fps
            return
        raw_msec = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC) or 0.0)
        if math.isfinite(raw_msec) and raw_msec >= 0.0:
            self._position_seconds = raw_msec / 1000.0

    def _decode_next_locked(self) -> bool:
        ok, frame = self._capture.read()
        if not ok or frame is None or np.asarray(frame).size == 0:
            return False
        self._current_frame = _opencv_frame_to_rgb8(frame, self._cv2)
        self._height, self._width = self._current_frame.shape[:2]
        self._update_position_from_backend()
        return True

    def _seek_locked(self, target_seconds: float) -> None:
        if self._frame_count is not None and self._fps is not None:
            frame_index = int(math.floor(target_seconds * self._fps + 1e-9))
            frame_index = max(0, min(frame_index, self._frame_count - 1))
            accepted = self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        else:
            frame_index = None
            accepted = self._capture.set(
                self._cv2.CAP_PROP_POS_MSEC, float(target_seconds) * 1000.0
            )
        if not accepted:
            raise SourceError(
                "video_seek_failed",
                f"video backend rejected seek to {target_seconds:.6f}s",
                path=self.path,
                seconds=target_seconds,
            )
        if not self._decode_next_locked():
            raise SourceError(
                "video_seek_failed",
                f"video backend could not decode at {target_seconds:.6f}s",
                path=self.path,
                seconds=target_seconds,
            )
        if frame_index is not None:
            self._frame_index = frame_index
            self._position_seconds = frame_index / self._fps  # type: ignore[operator]
        self._return_current_once = True

    def _advance_locked(self, elapsed: float) -> None:
        if self._transport_state != "playing" or elapsed <= 0.0:
            return
        target = self._position_seconds + elapsed
        if self._duration is not None and target >= self._duration:
            if self._loop and self._duration > 0.0:
                completed = int(target // self._duration)
                self._loops_completed += max(1, completed)
                target %= self._duration
            else:
                self._seek_locked(self._duration)
                self._transport_state = "ended"
                return
        self._seek_locked(target)

    def read(
        self,
        sequence: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> np.ndarray:
        del sequence, width, height
        with self._lock:
            self._require_open()
            if elapsed_seconds is not None:
                self._advance_locked(
                    _seconds(elapsed_seconds, code="invalid_elapsed", name="elapsed seconds")
                )
            elif self._transport_state == "playing":
                if self._return_current_once:
                    self._return_current_once = False
                elif not self._decode_next_locked():
                    if self._loop:
                        self._loops_completed += 1
                        self._seek_locked(0.0)
                        self._return_current_once = False
                    else:
                        self._transport_state = "ended"
            self._frames_read += 1
            return self._current_frame.copy(order="C")

    def advance(self, elapsed_seconds: float) -> None:
        with self._lock:
            self._require_open()
            self._advance_locked(
                _seconds(elapsed_seconds, code="invalid_elapsed", name="elapsed seconds")
            )

    def play(self) -> None:
        with self._lock:
            self._require_open()
            if self._transport_state == "ended":
                self._seek_locked(0.0)
            self._transport_state = "playing"

    def pause(self) -> None:
        with self._lock:
            self._require_open()
            if self._transport_state != "ended":
                self._transport_state = "paused"

    def stop(self) -> None:
        with self._lock:
            self._require_open()
            self._seek_locked(0.0)
            self._transport_state = "stopped"

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._require_open()
            target = _seconds(seconds)
            if self._duration is not None:
                target = min(target, self._duration)
            self._seek_locked(target)

    def set_loop(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise SourceError("invalid_loop", "loop must be a boolean")
        with self._lock:
            self._require_open()
            self._loop = enabled

    def _capabilities(self) -> dict[str, bool]:
        return {
            "read": not self._closed,
            "play": not self._closed,
            "pause": not self._closed,
            "stop": not self._closed,
            "seek": not self._closed,
            "loop": not self._closed,
            "timed_advance": not self._closed,
            "per_frame_durations": False,
        }

    def _transport_snapshot(self) -> dict[str, Any]:
        return {
            "supported": True,
            "state": "closed" if self._closed else self._transport_state,
            "position_seconds": self._position_seconds,
            "duration_seconds": self._duration,
            "fps": self._fps,
            "frame_index": self._frame_index,
            "frame_count": self._frame_count,
            "loop": self._loop,
            "loops_completed": self._loops_completed,
            "position_basis": (
                "decoded_frame_index" if self._fps is not None else "backend_timestamp"
            ),
        }

    def _metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "backend": "opencv_video_capture",
            "backend_open": bool(
                not self._closed
                and getattr(self, "_capture", None) is not None
                and self._capture.isOpened()
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot["state"] = "closed" if self._closed else self._transport_state
        return snapshot

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            capture = getattr(self, "_capture", None)
            if capture is not None:
                capture.release()
            self._current_frame = np.empty((0, 0, 3), dtype=np.uint8)
            self._closed = True
            self._transport_state = "closed"


@runtime_checkable
class OptionalAdapterFactory(Protocol):
    def __call__(self, **options: Any) -> FrameSource: ...


_OPTIONAL_ADAPTERS: dict[
    str, tuple[OptionalAdapterFactory, Callable[[], Mapping[str, Any]]]
] = {}
_OPTIONAL_ADAPTERS_LOCK = threading.RLock()


def register_optional_adapter(
    kind: str,
    factory: OptionalAdapterFactory,
    probe: Callable[[], Mapping[str, Any]],
) -> None:
    """Register an explicit webcam/window adapter without probing it.

    Registration itself must not enumerate devices.  ``probe`` is invoked only
    by an explicit probe or connection request.
    """

    if kind not in OPTIONAL_SOURCE_KINDS:
        raise SourceError(
            "invalid_adapter_kind",
            f"optional adapter kind must be one of {OPTIONAL_SOURCE_KINDS!r}",
            kind=kind,
        )
    if not callable(factory) or not callable(probe):
        raise SourceError("invalid_adapter", "adapter factory and probe must be callable")
    with _OPTIONAL_ADAPTERS_LOCK:
        _OPTIONAL_ADAPTERS[kind] = (factory, probe)


def unregister_optional_adapter(kind: str) -> None:
    if kind not in OPTIONAL_SOURCE_KINDS:
        raise SourceError("invalid_adapter_kind", f"unknown optional adapter kind: {kind!r}")
    with _OPTIONAL_ADAPTERS_LOCK:
        _OPTIONAL_ADAPTERS.pop(kind, None)


def probe_optional_adapter(kind: str) -> dict[str, Any]:
    """Truthfully probe one optional source; no fake availability is inferred."""

    if kind not in OPTIONAL_SOURCE_KINDS:
        raise SourceError("invalid_adapter_kind", f"unknown optional adapter kind: {kind!r}")
    with _OPTIONAL_ADAPTERS_LOCK:
        registration = _OPTIONAL_ADAPTERS.get(kind)
    if registration is None:
        return {
            "version": SNAPSHOT_VERSION,
            "kind": kind,
            "available": False,
            "reason_code": "adapter_not_registered",
            "reason": f"no {kind} adapter is registered in this runtime",
            "enumerated": False,
        }
    _factory, probe = registration
    try:
        result = dict(probe())
    except Exception as error:
        return {
            "version": SNAPSHOT_VERSION,
            "kind": kind,
            "available": False,
            "reason_code": "adapter_probe_failed",
            "reason": f"{kind} adapter probe failed: {type(error).__name__}",
            "enumerated": True,
        }
    available = result.get("available") is True
    reason = result.get("reason")
    if not available and not isinstance(reason, str):
        reason = f"{kind} adapter reported no available capture target"
    return {
        "version": SNAPSHOT_VERSION,
        "kind": kind,
        "available": available,
        "reason_code": result.get("reason_code") if not available else None,
        "reason": reason if not available else None,
        "enumerated": True,
        "details": {key: value for key, value in result.items() if key != "available"},
    }


def probe_optional_adapters() -> dict[str, dict[str, Any]]:
    return {kind: probe_optional_adapter(kind) for kind in OPTIONAL_SOURCE_KINDS}


def create_source(kind: Any, path: Any = None, **options: Any) -> FrameSource:
    """Construct a source without changing any existing slot."""

    if not isinstance(kind, str) or not kind.strip():
        raise SourceError("unsupported_source_kind", f"unsupported source kind: {kind!r}")
    normalized = kind.strip().lower()
    if normalized == "animated_gif":
        normalized = "gif"
    if normalized == "test_pattern":
        if path is not None:
            raise SourceError("unexpected_source_path", "test_pattern source does not use a path")
        return TestPatternSource(**options)
    if normalized == "image":
        return ImageSource(path, **options)
    if normalized == "gif":
        return AnimatedGifSource(path, **options)
    if normalized == "video":
        return VideoSource(path, **options)
    if normalized in OPTIONAL_SOURCE_KINDS:
        probe = probe_optional_adapter(normalized)
        if not probe["available"]:
            raise SourceError(
                "source_adapter_unavailable",
                str(probe["reason"]),
                kind=normalized,
                probe=probe,
            )
        with _OPTIONAL_ADAPTERS_LOCK:
            registration = _OPTIONAL_ADAPTERS.get(normalized)
        if registration is None:  # A concurrent unregister after the probe.
            raise SourceError(
                "source_adapter_unavailable",
                f"{normalized} adapter was unregistered during connection",
                kind=normalized,
            )
        factory, _probe = registration
        if path is not None:
            options.setdefault("path", path)
        source = factory(**options)
        if not isinstance(source, FrameSource):
            try:
                source.close()
            except Exception:
                pass
            raise SourceError(
                "invalid_adapter_source",
                f"{normalized} adapter did not return a FrameSource",
                recoverable=False,
                kind=normalized,
            )
        return source
    raise SourceError(
        "unsupported_source_kind",
        f"unsupported source kind: {kind!r}",
        supported=list(BUILTIN_SOURCE_KINDS + OPTIONAL_SOURCE_KINDS),
    )


class SourceSlot:
    """Atomic, reconnectable owner for one canonical session source slot."""

    def __init__(self, slot_id: str) -> None:
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise SourceError("invalid_slot", "slot id must be a non-empty string")
        self.slot_id = slot_id.strip()
        self._lock = threading.RLock()
        self._source: FrameSource | None = None
        self._generation = 0
        self._closed = False
        self._last_error: dict[str, Any] | None = None

    def _require_open(self) -> None:
        if self._closed:
            raise SourceError(
                "slot_closed",
                f"source slot {self.slot_id!r} is closed",
                recoverable=False,
                slot_id=self.slot_id,
            )

    def connect(self, kind: Any, path: Any = None, **options: Any) -> dict[str, Any]:
        try:
            replacement = create_source(kind, path, **options)
        except SourceError as error:
            with self._lock:
                self._last_error = error.to_dict()
            raise
        with self._lock:
            try:
                self._require_open()
            except Exception:
                replacement.close()
                raise
            previous = self._source
            self._source = replacement
            self._generation += 1
            self._last_error = None
            if previous is not None:
                previous.close()
            return self.snapshot()

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            previous = self._source
            if previous is not None:
                self._source = None
                self._generation += 1
                previous.close()
            self._last_error = None
            return self.snapshot()

    def _connected_source(self) -> FrameSource:
        self._require_open()
        if self._source is None:
            raise SourceError(
                "source_disconnected",
                f"source slot {self.slot_id!r} is disconnected",
                slot_id=self.slot_id,
            )
        return self._source

    def read(
        self,
        sequence: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> np.ndarray:
        with self._lock:
            return self._connected_source().read(
                sequence,
                width,
                height,
                elapsed_seconds=elapsed_seconds,
            )

    def play(self) -> None:
        with self._lock:
            self._connected_source().play()

    def pause(self) -> None:
        with self._lock:
            self._connected_source().pause()

    def stop(self) -> None:
        with self._lock:
            self._connected_source().stop()

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._connected_source().seek(seconds)

    def advance(self, elapsed_seconds: float) -> None:
        with self._lock:
            self._connected_source().advance(elapsed_seconds)

    def set_loop(self, enabled: bool) -> None:
        with self._lock:
            self._connected_source().set_loop(enabled)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            source_snapshot = self._source.snapshot() if self._source is not None else None
            return {
                "version": SNAPSHOT_VERSION,
                "slot_id": self.slot_id,
                "generation": self._generation,
                "closed": self._closed,
                "connected": source_snapshot is not None and bool(source_snapshot["connected"]),
                "source": source_snapshot,
                "last_error": dict(self._last_error) if self._last_error is not None else None,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._source is not None:
                self._source.close()
                self._source = None
                self._generation += 1
            self._closed = True
            self._last_error = None

    def __enter__(self) -> "SourceSlot":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class SourceService:
    """Owner for the named source slots referenced by canonical sessions."""

    def __init__(self, slot_ids: Iterable[str] = ("primary", "secondary")) -> None:
        identifiers = tuple(slot_ids)
        if not identifiers:
            raise SourceError("invalid_slot", "source service requires at least one slot")
        if any(not isinstance(item, str) or not item.strip() for item in identifiers):
            raise SourceError("invalid_slot", "slot ids must be non-empty strings")
        normalized = tuple(item.strip() for item in identifiers)
        if len(set(normalized)) != len(normalized):
            raise SourceError("duplicate_slot", "source slot ids must be unique")
        self._lock = threading.RLock()
        self._slots = {identifier: SourceSlot(identifier) for identifier in normalized}
        self._closed = False

    def slot(self, slot_id: str) -> SourceSlot:
        with self._lock:
            if self._closed:
                raise SourceError(
                    "service_closed", "source service is closed", recoverable=False
                )
            try:
                return self._slots[slot_id]
            except (KeyError, TypeError) as error:
                raise SourceError(
                    "unknown_slot",
                    f"unknown source slot: {slot_id!r}",
                    slot_id=slot_id,
                    available=list(self._slots),
                ) from error

    def connect(self, slot_id: str, kind: Any, path: Any = None, **options: Any) -> dict[str, Any]:
        return self.slot(slot_id).connect(kind, path, **options)

    def disconnect(self, slot_id: str) -> dict[str, Any]:
        return self.slot(slot_id).disconnect()

    def read(
        self,
        slot_id: str,
        sequence: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> np.ndarray:
        return self.slot(slot_id).read(
            sequence, width, height, elapsed_seconds=elapsed_seconds
        )

    def play(self, slot_id: str) -> None:
        self.slot(slot_id).play()

    def pause(self, slot_id: str) -> None:
        self.slot(slot_id).pause()

    def stop(self, slot_id: str) -> None:
        self.slot(slot_id).stop()

    def seek(self, slot_id: str, seconds: float) -> None:
        self.slot(slot_id).seek(seconds)

    def advance(self, slot_id: str, elapsed_seconds: float) -> None:
        self.slot(slot_id).advance(elapsed_seconds)

    def set_loop(self, slot_id: str, enabled: bool) -> None:
        self.slot(slot_id).set_loop(enabled)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": SNAPSHOT_VERSION,
                "closed": self._closed,
                "slots": {slot_id: slot.snapshot() for slot_id, slot in self._slots.items()},
                "optional_adapters": probe_optional_adapters(),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for slot in self._slots.values():
                slot.close()
            self._closed = True

    def __enter__(self) -> "SourceService":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "ALPHA_HANDLING_POLICY",
    "AnimatedGifSource",
    "BUILTIN_SOURCE_KINDS",
    "FrameSource",
    "ImageSource",
    "OPTIONAL_SOURCE_KINDS",
    "PIXEL_FORMAT",
    "SNAPSHOT_VERSION",
    "SourceError",
    "SourceService",
    "SourceSlot",
    "TestPatternSource",
    "VideoSource",
    "create_source",
    "deterministic_test_pattern",
    "probe_optional_adapter",
    "probe_optional_adapters",
    "register_optional_adapter",
    "unregister_optional_adapter",
]
