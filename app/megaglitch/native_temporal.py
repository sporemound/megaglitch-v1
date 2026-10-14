"""Backend-neutral temporal ring facade with an always-available Python fallback."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    from ._native_temporal import (
        NativeTemporalRingCore,
        amplify_prediction_errors_uniform as _native_prediction_errors_uniform,
        process_pframe_spiral_uniform as _native_pframe_spiral_uniform,
    )
except ImportError as exc:  # The source tree must remain runnable before a native build.
    NativeTemporalRingCore = None
    _native_prediction_errors_uniform = None
    _native_pframe_spiral_uniform = None
    _NATIVE_IMPORT_ERROR = exc
else:
    _NATIVE_IMPORT_ERROR = None


class PythonTemporalRingCore:
    """Reference implementation matching the native prototype's semantics."""

    def __init__(self) -> None:
        self._storage: np.ndarray | None = None
        self._width = self._height = self._channels = self._capacity = 0
        self._count = self._write_index = self._delay = 0
        self._mix = 0.5
        self._frozen = False
        self._writes = self._reads = self._mixes = self._frozen_writes = 0
        self._bytes_copied = 0

    def configure(self, width: int, height: int, channels: int, capacity: int) -> None:
        if min(width, height, channels, capacity) <= 0:
            raise ValueError("width, height, channels, and capacity must be positive")
        self._width, self._height = int(width), int(height)
        self._channels, self._capacity = int(channels), int(capacity)
        self._storage = np.zeros((capacity, height, width, channels), dtype=np.uint8)
        self._count = self._write_index = self._delay = 0
        self._frozen = False
        self._writes = self._reads = self._mixes = self._frozen_writes = 0
        self._bytes_copied = 0

    @property
    def _frame_bytes(self) -> int:
        return self._width * self._height * self._channels

    def _require_configured(self) -> np.ndarray:
        if self._storage is None:
            raise RuntimeError("temporal ring is not configured")
        return self._storage

    def _frame(self, frame: Any) -> np.ndarray:
        array = np.asarray(frame)
        expected = (self._height, self._width, self._channels)
        if array.dtype != np.uint8 or array.ndim != 3:
            raise TypeError("frame must be a three-dimensional uint8 array")
        if array.shape != expected:
            raise ValueError(f"expected frame shape {expected}, got {array.shape}")
        if not array.flags.c_contiguous:
            raise ValueError("frame must be C-contiguous")
        return array

    def _slot(self, delay: int) -> int:
        self._require_configured()
        if delay < 0 or delay >= self._count:
            raise ValueError(f"delay {delay} is outside written history [0, {self._count - 1}]")
        return (self._write_index + self._capacity - 1 - delay) % self._capacity

    def write_frame(self, frame: Any) -> bool:
        storage = self._require_configured()
        if self._frozen:
            self._frozen_writes += 1
            return False
        storage[self._write_index] = self._frame(frame)
        self._write_index = (self._write_index + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)
        self._writes += 1
        self._bytes_copied += self._frame_bytes
        return True

    def read_frame(self, delay: int | None = None) -> np.ndarray:
        requested = self._delay if delay is None else int(delay)
        result = self._require_configured()[self._slot(requested)].copy()
        self._reads += 1
        self._bytes_copied += self._frame_bytes
        return result

    def process_temporal_mix(self, current: Any, delay: int | None = None,
                             mix: float | None = None, output: Any = None) -> np.ndarray:
        requested = self._delay if delay is None else int(delay)
        amount = self._mix if mix is None else float(mix)
        if not 0.0 <= amount <= 1.0:
            raise ValueError("mix must be between 0 and 1")
        current_array = self._frame(current)
        historical = self._require_configured()[self._slot(requested)]
        target = np.empty_like(current_array) if output is None else self._frame(output)
        # Match the native prototype's round-to-nearest RGB8 behavior.
        amount32 = np.float32(amount)
        target[...] = (
            current_array.astype(np.float32) * (np.float32(1.0) - amount32)
            + historical.astype(np.float32) * amount32
            + np.float32(0.5)
        ).astype(np.uint8)
        self._mixes += 1
        self._bytes_copied += self._frame_bytes
        return target

    def set_delay(self, delay: int) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self._delay = int(delay)

    def set_mix(self, mix: float) -> None:
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be between 0 and 1")
        self._mix = float(mix)

    def freeze(self, enabled: bool = True) -> None:
        self._require_configured()
        self._frozen = bool(enabled)

    def clear(self) -> None:
        self._require_configured().fill(0)
        self._count = self._write_index = 0

    def get_statistics(self) -> dict[str, Any]:
        return {
            "backend": "python_numpy",
            "configured": self._storage is not None,
            "width": self._width,
            "height": self._height,
            "channels": self._channels,
            "capacity": self._capacity,
            "count": self._count,
            "write_index": self._write_index,
            "allocated_bytes": 0 if self._storage is None else self._storage.nbytes,
            "writes": self._writes,
            "reads": self._reads,
            "mixes": self._mixes,
            "frozen_writes": self._frozen_writes,
            "bytes_copied": self._bytes_copied,
            "frozen": self._frozen,
        }

    def shutdown(self) -> None:
        self._storage = None
        self._count = self._write_index = 0
        self._frozen = False


class TemporalFrameRing:
    """Coarse public API shared by native and rollback implementations."""

    def __init__(self, backend: str | None = None) -> None:
        requested = (backend or os.getenv("MEGAGLITCH_HISTORY_BACKEND", "auto")).strip().lower()
        if requested not in {"auto", "native", "python"}:
            raise ValueError("history backend must be 'auto', 'native', or 'python'")
        if requested == "native" and NativeTemporalRingCore is None:
            raise RuntimeError(f"native temporal extension is unavailable: {_NATIVE_IMPORT_ERROR}")
        use_native = NativeTemporalRingCore is not None and requested in {"auto", "native"}
        self._core = NativeTemporalRingCore() if use_native else PythonTemporalRingCore()

    @property
    def backend(self) -> str:
        return self.get_statistics()["backend"]

    def initialize(self, width: int, height: int, capacity: int, channels: int = 3) -> None:
        self.configure(width, height, capacity, channels)

    def configure(self, width: int, height: int, capacity: int, channels: int = 3) -> None:
        self._core.configure(width, height, channels, capacity)

    def write_frame(self, frame: Any) -> bool:
        return self._core.write_frame(frame)

    def read_frame(self, delay: int | None = None) -> np.ndarray:
        return self._core.read_frame(delay)

    def process_temporal_mix(self, current: Any, delay: int | None = None,
                             mix: float | None = None, output: Any = None) -> np.ndarray:
        return self._core.process_temporal_mix(current, delay, mix, output)

    def set_delay(self, delay: int) -> None:
        self._core.set_delay(delay)

    def set_mix(self, mix: float) -> None:
        self._core.set_mix(mix)

    def freeze(self, enabled: bool = True) -> None:
        self._core.freeze(enabled)

    def clear(self) -> None:
        self._core.clear()

    def get_statistics(self) -> dict[str, Any]:
        return self._core.get_statistics()

    def shutdown(self) -> None:
        self._core.shutdown()

    def __enter__(self) -> "TemporalFrameRing":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()


def native_available() -> bool:
    return NativeTemporalRingCore is not None


def amplify_prediction_errors_uniform(current, reference, intensity, error_threshold, output=None):
    """Run the optional exact native kernel, returning None when it is not built."""
    if _native_prediction_errors_uniform is None:
        return None
    return _native_prediction_errors_uniform(
        current, reference, float(intensity), int(error_threshold), output
    )


def process_pframe_spiral_uniform(current, reference, motion_intensity, block_size,
                                  random_values, prediction_intensity,
                                  error_threshold, output=None):
    """Run the optional fused native P-frame segment."""
    if _native_pframe_spiral_uniform is None:
        return None
    return _native_pframe_spiral_uniform(
        current, reference, float(motion_intensity), int(block_size), random_values,
        float(prediction_intensity), int(error_threshold), output,
    )
