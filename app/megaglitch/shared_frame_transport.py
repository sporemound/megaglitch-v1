"""Fixed shared-memory frame slots for the existing multiprocessing worker."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import queue
from typing import Any

import numpy as np


_FRAME_MARKER = "megaglitch_shared_frame_v1"
_RESULT_MARKER = "megaglitch_shared_result_v1"


class SharedSlotPool:
    """Atomic fixed-slot ownership shared across spawned processes."""

    def __init__(self, mp_context, capacity: int):
        self.capacity = int(capacity)
        self._used = mp_context.Array("b", capacity, lock=False)
        self._lock = mp_context.Lock()
        self._available = mp_context.BoundedSemaphore(capacity)

    def get(self, block: bool = True, timeout: float | None = None) -> int:
        acquired = self._available.acquire(block, timeout)
        if not acquired:
            raise queue.Empty
        with self._lock:
            for slot in range(self.capacity):
                if not self._used[slot]:
                    self._used[slot] = 1
                    return slot
        self._available.release()
        raise RuntimeError("slot semaphore and ownership bitmap diverged")

    def get_nowait(self) -> int:
        return self.get(False)

    def put_nowait(self, slot: int) -> None:
        slot = int(slot)
        if slot < 0 or slot >= self.capacity:
            raise ValueError(f"slot {slot} is outside pool capacity {self.capacity}")
        with self._lock:
            if not self._used[slot]:
                raise ValueError(f"slot {slot} is already free")
            self._used[slot] = 0
        self._available.release()

    def in_use(self) -> int:
        with self._lock:
            return sum(bool(value) for value in self._used)


@dataclass(frozen=True)
class SharedTransportSpec:
    input_name: str
    output_name: str
    slot_bytes: int
    input_slots: int
    output_slots: int
    max_width: int
    max_height: int
    channels: int
    input_free: Any
    output_free: Any


def _descriptor(pool: str, slot: int, array: np.ndarray) -> dict[str, Any]:
    return {
        "marker": _FRAME_MARKER,
        "pool": pool,
        "slot": int(slot),
        "shape": tuple(int(value) for value in array.shape),
        "dtype": "uint8",
        "nbytes": int(array.nbytes),
    }


def is_shared_frame(value: Any, pool: str | None = None) -> bool:
    return (
        isinstance(value, dict)
        and value.get("marker") == _FRAME_MARKER
        and (pool is None or value.get("pool") == pool)
    )


def _copy_stats_with(stats, category: str, nbytes: int) -> dict[str, dict[str, int]]:
    result = {
        str(name): {
            "count": int(values.get("count", 0)),
            "bytes": int(values.get("bytes", 0)),
        }
        for name, values in (stats or {}).items()
        if isinstance(values, dict)
    }
    bucket = result.setdefault(category, {"count": 0, "bytes": 0})
    bucket["count"] += 1
    bucket["bytes"] += int(nbytes)
    return result


class SharedFrameTransportOwner:
    """Creates pools and owns unlinking; used only by the application process."""

    def __init__(self, mp_context, max_width: int, max_height: int, channels: int = 3,
                 input_slots: int = 4, output_slots: int = 2):
        if min(max_width, max_height, channels, input_slots, output_slots) <= 0:
            raise ValueError("shared transport dimensions and slot counts must be positive")
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.channels = int(channels)
        self.slot_bytes = self.max_width * self.max_height * self.channels
        self.input_slots = int(input_slots)
        self.output_slots = int(output_slots)
        self.input_shm = shared_memory.SharedMemory(create=True, size=self.slot_bytes * input_slots)
        self.output_shm = shared_memory.SharedMemory(create=True, size=self.slot_bytes * output_slots)
        self.input_free = SharedSlotPool(mp_context, input_slots)
        self.output_free = SharedSlotPool(mp_context, output_slots)
        self.closed = False
        self.stats = {
            "input_frames_written": 0,
            "output_frames_read": 0,
            "input_pool_exhausted": 0,
            "oversize_fallbacks": 0,
            "input_slots_released": 0,
            "output_slots_released": 0,
            "input_copy_count": 0,
            "input_copy_bytes": 0,
            "ui_consumer_copy_count": 0,
            "ui_consumer_copy_bytes": 0,
        }

    def spec(self) -> SharedTransportSpec:
        return SharedTransportSpec(
            self.input_shm.name, self.output_shm.name, self.slot_bytes,
            self.input_slots, self.output_slots, self.max_width, self.max_height,
            self.channels, self.input_free, self.output_free,
        )

    def _slot_array(self, shm, slot: int, shape: tuple[int, int, int]) -> np.ndarray:
        offset = int(slot) * self.slot_bytes
        return np.ndarray(shape, dtype=np.uint8, buffer=shm.buf, offset=offset)

    def encode_frame(self, frame: Any, *, block: bool = False, timeout: float | None = None):
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != self.channels:
            return None
        height, width, _ = array.shape
        if width > self.max_width or height > self.max_height:
            self.stats["oversize_fallbacks"] += 1
            return None
        array = np.ascontiguousarray(array)
        try:
            slot = self.input_free.get(block=block, timeout=timeout if block else None)
        except queue.Empty:
            self.stats["input_pool_exhausted"] += 1
            return False
        self._slot_array(self.input_shm, slot, array.shape)[...] = array
        self.stats["input_frames_written"] += 1
        self.stats["input_copy_count"] += 1
        self.stats["input_copy_bytes"] += int(array.nbytes)
        return _descriptor("input", slot, array)

    def encode_payload(self, payload: dict[str, Any], *, block: bool = False,
                       timeout: float | None = None):
        encoded = dict(payload)
        acquired = []
        frame_copy_stats = encoded.get("frame_copy_stats")
        for key in ("image", "secondary_image"):
            value = payload.get(key)
            if value is None:
                continue
            descriptor = self.encode_frame(value, block=block, timeout=timeout)
            if descriptor is False:
                for item in acquired:
                    self.release_input_descriptor(item)
                return None
            if descriptor is None:
                # Unsupported shape/type remains a legacy payload for compatibility.
                continue
            encoded[key] = descriptor
            acquired.append(descriptor)
            frame_copy_stats = _copy_stats_with(
                frame_copy_stats, "shared_input", int(descriptor["nbytes"])
            )
        if frame_copy_stats:
            encoded["frame_copy_stats"] = frame_copy_stats
        encoded["frame_transport"] = "shared_v1"
        return encoded

    def release_input_descriptor(self, descriptor):
        if not is_shared_frame(descriptor, "input"):
            return
        self.input_free.put_nowait(int(descriptor["slot"]))
        self.stats["input_slots_released"] += 1

    def release_input_payload(self, payload):
        if not isinstance(payload, dict):
            return
        for key in ("image", "secondary_image"):
            self.release_input_descriptor(payload.get(key))

    def release_output_result(self, result):
        if not isinstance(result, dict) or result.get("marker") != _RESULT_MARKER:
            return
        descriptor = result.get("frame")
        if is_shared_frame(descriptor, "output"):
            self.output_free.put_nowait(int(descriptor["slot"]))
            self.stats["output_slots_released"] += 1

    def decode_result(self, result):
        if not isinstance(result, dict) or result.get("marker") != _RESULT_MARKER:
            return result
        descriptor = result.get("frame")
        if not is_shared_frame(descriptor, "output"):
            return result
        try:
            shape = tuple(int(value) for value in descriptor["shape"])
            frame = self._slot_array(self.output_shm, int(descriptor["slot"]), shape).copy()
            decoded = {key: value for key, value in result.items() if key != "marker"}
            decoded["frame"] = frame
            decoded["frame_a"] = None
            decoded["frame_b"] = None
            decoded["frame_transport"] = "shared_v1"
            decoded["frame_copy_stats"] = _copy_stats_with(
                decoded.get("frame_copy_stats"), "ui_consumer", frame.nbytes
            )
            self.stats["output_frames_read"] += 1
            self.stats["ui_consumer_copy_count"] += 1
            self.stats["ui_consumer_copy_bytes"] += int(frame.nbytes)
            return decoded
        finally:
            self.release_output_result(result)

    def get_statistics(self):
        return dict(self.stats) | {
            "slot_bytes": self.slot_bytes,
            "input_slots": self.input_slots,
            "output_slots": self.output_slots,
            "allocated_bytes": self.slot_bytes * (self.input_slots + self.output_slots),
            "input_slots_in_use": self.input_free.in_use(),
            "output_slots_in_use": self.output_free.in_use(),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        for shm in (self.input_shm, self.output_shm):
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass


class SharedFrameTransportEndpoint:
    """Worker-side attachment. It never unlinks owner-managed memory."""

    def __init__(self, spec: SharedTransportSpec):
        self.spec = spec
        self.input_shm = shared_memory.SharedMemory(name=spec.input_name)
        self.output_shm = shared_memory.SharedMemory(name=spec.output_name)
        self.closed = False

    def _slot_array(self, shm, slot: int, shape: tuple[int, int, int]) -> np.ndarray:
        offset = int(slot) * self.spec.slot_bytes
        return np.ndarray(shape, dtype=np.uint8, buffer=shm.buf, offset=offset)

    def resolve_input(self, value):
        if not is_shared_frame(value, "input"):
            return value, None
        shape = tuple(int(item) for item in value["shape"])
        if len(shape) != 3 or shape[2] != self.spec.channels:
            raise ValueError(f"invalid shared input shape {shape}")
        if int(value["nbytes"]) > self.spec.slot_bytes:
            raise ValueError("shared input descriptor exceeds slot capacity")
        return self._slot_array(self.input_shm, int(value["slot"]), shape), int(value["slot"])

    def release_input_slot(self, slot):
        if slot is not None:
            self.spec.input_free.put_nowait(int(slot))

    def release_output_result(self, result):
        if not isinstance(result, dict) or result.get("marker") != _RESULT_MARKER:
            return
        descriptor = result.get("frame")
        if is_shared_frame(descriptor, "output"):
            self.spec.output_free.put_nowait(int(descriptor["slot"]))

    def _acquire_output_slot(self, target_queue):
        try:
            return self.spec.output_free.get_nowait()
        except queue.Empty:
            try:
                stale = target_queue.get_nowait()
            except queue.Empty:
                return None
            self.release_output_result(stale)
            try:
                return self.spec.output_free.get_nowait()
            except queue.Empty:
                return None

    def publish_result(self, target_queue, frame: Any, metadata: dict[str, Any]) -> bool:
        array = np.asarray(frame)
        if (
            array.dtype != np.uint8 or array.ndim != 3
            or array.shape[2] != self.spec.channels
            or array.shape[1] > self.spec.max_width
            or array.shape[0] > self.spec.max_height
        ):
            # Canonical legacy fallback: still return only the master frame.
            result = dict(metadata)
            result["frame"] = np.ascontiguousarray(array)
            result["frame_transport"] = "legacy_oversize"
            try:
                target_queue.put_nowait(result)
                return True
            except queue.Full:
                return False
        array = np.ascontiguousarray(array)
        slot = self._acquire_output_slot(target_queue)
        if slot is None:
            return False
        self._slot_array(self.output_shm, slot, array.shape)[...] = array
        result = dict(metadata)
        result["frame_copy_stats"] = _copy_stats_with(
            result.get("frame_copy_stats"), "shared_output", array.nbytes
        )
        result["marker"] = _RESULT_MARKER
        result["frame"] = _descriptor("output", slot, array)
        try:
            target_queue.put_nowait(result)
            return True
        except queue.Full:
            try:
                stale = target_queue.get_nowait()
            except queue.Empty:
                stale = None
            self.release_output_result(stale)
            try:
                target_queue.put_nowait(result)
                return True
            except queue.Full:
                self.spec.output_free.put_nowait(slot)
                return False

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.input_shm.close()
        self.output_shm.close()
