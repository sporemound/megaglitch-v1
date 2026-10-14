"""Python-hosted facade for the bounded Rust frame runtime."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np


class EngineState(IntEnum):
    UNINITIALIZED = 0
    INITIALIZING = 1
    WARMING_UP = 2
    READY = 3
    RUNNING = 4
    STOPPING = 5
    STOPPED = 6
    FAILED = 7


class RuntimeErrorBase(RuntimeError):
    pass


class NativeRuntimeUnavailable(RuntimeErrorBase):
    pass


class NativeRuntimeFailure(RuntimeErrorBase):
    pass


class UnsupportedNativeGraph(RuntimeErrorBase):
    pass


GRAPH_IR_VERSION = 1


class GraphOpcode(IntEnum):
    INPUT = 1
    INVERT = 2
    BRIGHTNESS_CONTRAST = 3
    TEMPORAL_MIX = 4
    BLEND = 5
    OUTPUT = 6


@dataclass(frozen=True)
class GraphOperation:
    opcode: GraphOpcode
    flags: int = 0
    arg0: float = 0.0
    arg1: float = 0.0
    arg2: int = 0
    arg3: int = 0


@dataclass(frozen=True)
class GraphSnapshot:
    operations: tuple[GraphOperation, ...]
    version: int = GRAPH_IR_VERSION


@dataclass(frozen=True)
class RuntimeConfig:
    width: int
    height: int
    channels: int = 3
    history_capacity: int = 8
    queue_capacity: int = 2
    mode: str = "passthrough"
    graph: GraphSnapshot | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * self.channels


@dataclass(frozen=True)
class ParameterSnapshot:
    delay: int = 0
    mix: float = 0.5
    frozen: bool = False


class _CStats(ctypes.Structure):
    _fields_ = [
        ("submitted", ctypes.c_uint64),
        ("processed", ctypes.c_uint64),
        ("delivered", ctypes.c_uint64),
        ("input_dropped", ctypes.c_uint64),
        ("output_dropped", ctypes.c_uint64),
        ("commands_applied", ctypes.c_uint64),
        ("frames_in_history", ctypes.c_uint64),
        ("allocated_bytes", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("state", ctypes.c_uint32),
    ]


class _CGraphOp(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("arg0", ctypes.c_float),
        ("arg1", ctypes.c_float),
        ("arg2", ctypes.c_uint32),
        ("arg3", ctypes.c_uint32),
    ]


_DLL_PATH = Path(__file__).with_name("_native") / "megaglitch_runtime.dll"
_MODE = {"passthrough": 0, "temporal_mix": 1, "invert": 2}
_ERROR_SIZE = 1024
_MG_OK, _MG_EMPTY, _MG_QUEUE_FULL = 0, 1, 2
_TEMPORAL_MACRO_NAMES = {
    "history_offset", "memory_decay", "event_hazard", "clock_quantization",
    "sync_desync", "phase_reset_chance", "local_time_warp", "rupture_threshold",
    "hysteresis_speed", "modulator_coupling",
}


def has_active_temporal_macros(params) -> bool:
    values = params or {}
    for name in _TEMPORAL_MACRO_NAMES:
        try:
            if abs(float(values.get(name, 0.0) or 0.0)) > 1e-8:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _load_library():
    if not _DLL_PATH.exists():
        raise NativeRuntimeUnavailable(
            f"Rust runtime DLL is not built at {_DLL_PATH}; run native/runtime/build.ps1"
        )
    library = ctypes.CDLL(str(_DLL_PATH))
    library.mg_engine_create.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.mg_engine_create.restype = ctypes.c_void_p
    library.mg_engine_create_graph.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(_CGraphOp), ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.mg_engine_create_graph.restype = ctypes.c_void_p
    library.mg_engine_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    library.mg_engine_start.restype = ctypes.c_int32
    library.mg_engine_state.argtypes = [ctypes.c_void_p]
    library.mg_engine_state.restype = ctypes.c_uint32
    library.mg_engine_frame_bytes.argtypes = [ctypes.c_void_p]
    library.mg_engine_frame_bytes.restype = ctypes.c_size_t
    library.mg_engine_submit.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.mg_engine_submit.restype = ctypes.c_int32
    library.mg_engine_poll.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.mg_engine_poll.restype = ctypes.c_int32
    library.mg_engine_wait.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.mg_engine_wait.restype = ctypes.c_int32
    library.mg_engine_set_parameters.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float, ctypes.c_bool,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.mg_engine_set_parameters.restype = ctypes.c_int32
    library.mg_engine_set_graph.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_CGraphOp), ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.mg_engine_set_graph.restype = ctypes.c_int32
    library.mg_engine_clear.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    library.mg_engine_clear.restype = ctypes.c_int32
    library.mg_engine_get_statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CStats)]
    library.mg_engine_get_statistics.restype = ctypes.c_int32
    library.mg_engine_shutdown.argtypes = [ctypes.c_void_p]
    library.mg_engine_shutdown.restype = ctypes.c_int32
    library.mg_engine_destroy.argtypes = [ctypes.c_void_p]
    library.mg_engine_destroy.restype = None
    return library


def native_runtime_available() -> bool:
    try:
        _load_library()
    except (OSError, NativeRuntimeUnavailable):
        return False
    return True


def _error_buffer():
    return ctypes.create_string_buffer(_ERROR_SIZE)


def _message(buffer) -> str:
    return buffer.value.decode("utf-8", errors="replace") or "native runtime operation failed"


def _graph_array(snapshot: GraphSnapshot):
    operations = tuple(snapshot.operations)
    array_type = _CGraphOp * len(operations)
    return array_type(*(
        _CGraphOp(
            int(operation.opcode), int(operation.flags), float(operation.arg0),
            float(operation.arg1), int(operation.arg2), int(operation.arg3),
        )
        for operation in operations
    ))


class NativeRuntime:
    """Explicit-handle Rust runtime; calls into the DLL release the Python GIL."""

    backend = "native_rust"

    def __init__(self, config: RuntimeConfig):
        if config.graph is None and config.mode not in _MODE:
            raise UnsupportedNativeGraph(f"unsupported native runtime mode: {config.mode}")
        self.config = config
        self._library = _load_library()
        error = _error_buffer()
        if config.graph is None:
            self._handle = self._library.mg_engine_create(
                config.width, config.height, config.channels, config.history_capacity,
                config.queue_capacity, _MODE[config.mode], error, len(error),
            )
        else:
            validate_graph_snapshot(config.graph, config.history_capacity)
            operations = _graph_array(config.graph)
            self._handle = self._library.mg_engine_create_graph(
                config.width, config.height, config.channels, config.history_capacity,
                config.queue_capacity, config.graph.version, operations, len(operations),
                error, len(error),
            )
        if not self._handle:
            raise NativeRuntimeFailure(_message(error))
        self._parameters = ParameterSnapshot()
        self._graph = config.graph

    @property
    def state(self) -> EngineState:
        if not self._handle:
            return EngineState.UNINITIALIZED
        return EngineState(self._library.mg_engine_state(self._handle))

    def start(self) -> None:
        error = _error_buffer()
        code = self._library.mg_engine_start(self._handle, error, len(error))
        if code != _MG_OK:
            raise NativeRuntimeFailure(_message(error))

    def _frame(self, frame: Any) -> np.ndarray:
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.shape != self.config.shape:
            raise ValueError(f"expected uint8 frame shape {self.config.shape}, got {array.dtype} {array.shape}")
        if not array.flags.c_contiguous:
            raise ValueError("frame must be C-contiguous")
        return array

    def submit_frame(self, frame: Any) -> bool:
        array = self._frame(frame)
        error = _error_buffer()
        pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        code = self._library.mg_engine_submit(self._handle, pointer, array.nbytes, error, len(error))
        if code == _MG_QUEUE_FULL:
            return False
        if code != _MG_OK:
            raise NativeRuntimeFailure(_message(error))
        return True

    def poll_frame(self, output: Any = None):
        target = np.empty(self.config.shape, dtype=np.uint8) if output is None else self._frame(output)
        error = _error_buffer()
        sequence = ctypes.c_uint64()
        pointer = target.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        code = self._library.mg_engine_poll(
            self._handle, pointer, target.nbytes, ctypes.byref(sequence), error, len(error)
        )
        if code == _MG_EMPTY:
            return None
        if code != _MG_OK:
            raise NativeRuntimeFailure(_message(error))
        return target, int(sequence.value)

    def wait_for_frame(self, output: Any = None, timeout: float = 1.0):
        target = np.empty(self.config.shape, dtype=np.uint8) if output is None else self._frame(output)
        error = _error_buffer()
        sequence = ctypes.c_uint64()
        pointer = target.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        timeout_ms = max(0, int(round(timeout * 1000.0)))
        code = self._library.mg_engine_wait(
            self._handle, pointer, target.nbytes, timeout_ms,
            ctypes.byref(sequence), error, len(error),
        )
        if code == _MG_EMPTY:
            raise TimeoutError("timed out waiting for native runtime output")
        if code != _MG_OK:
            raise NativeRuntimeFailure(_message(error))
        return target, int(sequence.value)

    def update_parameters(self, snapshot: ParameterSnapshot) -> None:
        error = _error_buffer()
        code = self._library.mg_engine_set_parameters(
            self._handle, snapshot.delay, snapshot.mix, snapshot.frozen, error, len(error)
        )
        if code != _MG_OK:
            raise NativeRuntimeFailure(_message(error))
        self._parameters = snapshot

    def update_graph(self, snapshot: GraphSnapshot) -> None:
        validate_graph_snapshot(snapshot, self.config.history_capacity)
        operations = _graph_array(snapshot)
        error = _error_buffer()
        code = self._library.mg_engine_set_graph(
            self._handle, snapshot.version, operations, len(operations), error, len(error)
        )
        if code != _MG_OK:
            raise NativeRuntimeFailure(_message(error))
        self._graph = snapshot

    def clear(self) -> None:
        error = _error_buffer()
        code = self._library.mg_engine_clear(self._handle, error, len(error))
        if code != _MG_OK:
            raise NativeRuntimeFailure(_message(error))

    def get_statistics(self) -> dict[str, int | str]:
        result = _CStats()
        code = self._library.mg_engine_get_statistics(self._handle, ctypes.byref(result))
        if code != _MG_OK:
            raise NativeRuntimeFailure("failed to retrieve native runtime statistics")
        return {
            name: (EngineState(value).name.lower() if name == "state" else int(value))
            for name, value in ((field[0], getattr(result, field[0])) for field in result._fields_)
        } | {"backend": self.backend}

    def shutdown(self) -> None:
        if self._handle:
            self._library.mg_engine_shutdown(self._handle)

    def close(self) -> None:
        if self._handle:
            self.shutdown()
            self._library.mg_engine_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _legacy_graph_for_mode(mode: str) -> GraphSnapshot:
    operations = [GraphOperation(GraphOpcode.INPUT)]
    if mode == "invert":
        operations.append(GraphOperation(GraphOpcode.INVERT))
    elif mode == "temporal_mix":
        operations.append(GraphOperation(GraphOpcode.TEMPORAL_MIX, flags=2, arg0=0.5))
    elif mode != "passthrough":
        raise UnsupportedNativeGraph(f"unsupported fallback mode: {mode}")
    operations.append(GraphOperation(GraphOpcode.OUTPUT))
    return GraphSnapshot(tuple(operations))


def validate_graph_snapshot(snapshot: GraphSnapshot, history_capacity: int) -> None:
    if snapshot.version != GRAPH_IR_VERSION:
        raise NativeRuntimeFailure(f"unsupported graph IR version {snapshot.version}")
    operations = tuple(snapshot.operations)
    if not 2 <= len(operations) <= 64:
        raise NativeRuntimeFailure("graph must contain between 2 and 64 operations")
    if operations[0].opcode != GraphOpcode.INPUT or operations[-1].opcode != GraphOpcode.OUTPUT:
        raise NativeRuntimeFailure("graph must begin with Input and end with Output")
    for index, operation in enumerate(operations):
        try:
            opcode = GraphOpcode(operation.opcode)
        except ValueError as exc:
            raise NativeRuntimeFailure(f"unsupported graph opcode {operation.opcode}") from exc
        if opcode == GraphOpcode.INPUT and (index != 0 or operation.arg2 != 0):
            raise NativeRuntimeFailure("graph v1 supports one primary Input at operation 0")
        if opcode == GraphOpcode.OUTPUT and index != len(operations) - 1:
            raise NativeRuntimeFailure("Output must be the final operation")
        if opcode in {GraphOpcode.TEMPORAL_MIX, GraphOpcode.BLEND}:
            if not math.isfinite(operation.arg0) or not 0.0 <= operation.arg0 <= 1.0:
                raise NativeRuntimeFailure("temporal/blend mix must be between 0 and 1")
            if not 0 <= operation.arg2 < history_capacity:
                raise NativeRuntimeFailure(
                    "temporal/blend delay must be smaller than history capacity"
                )
        if opcode == GraphOpcode.BRIGHTNESS_CONTRAST and (
            not math.isfinite(operation.arg0) or not math.isfinite(operation.arg1)
        ):
            raise NativeRuntimeFailure("brightness/contrast parameters must be finite")


class PythonRuntime:
    """Threaded compatibility implementation of the same bounded contract."""

    backend = "python_thread"

    def __init__(self, config: RuntimeConfig):
        self._graph = config.graph or _legacy_graph_for_mode(config.mode)
        validate_graph_snapshot(self._graph, config.history_capacity)
        self.config = config
        self._state = EngineState.READY
        self._input = queue.Queue(maxsize=config.queue_capacity)
        self._output = queue.Queue(maxsize=config.queue_capacity)
        self._commands = queue.Queue(maxsize=32)
        self._slots = threading.BoundedSemaphore(config.queue_capacity)
        self._stop = threading.Event()
        self._thread = None
        self._parameters = ParameterSnapshot()
        self._history = np.zeros((config.history_capacity,) + config.shape, np.uint8)
        self._count = self._write_index = self._sequence = 0
        self._stats = {name: 0 for name in (
            "submitted", "processed", "delivered", "input_dropped", "output_dropped", "commands_applied"
        )}

    @property
    def state(self):
        return self._state

    def start(self):
        if self._state != EngineState.READY:
            raise NativeRuntimeFailure(f"engine cannot start from {self._state.name.lower()}")
        self._state = EngineState.RUNNING
        self._thread = threading.Thread(target=self._run, name="megaglitch-python-runtime", daemon=False)
        self._thread.start()

    def _frame(self, frame):
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.shape != self.config.shape or not array.flags.c_contiguous:
            raise ValueError(f"expected contiguous uint8 frame shape {self.config.shape}")
        return array

    def submit_frame(self, frame):
        if self._state != EngineState.RUNNING:
            raise NativeRuntimeFailure("engine is not running")
        if not self._slots.acquire(blocking=False):
            self._stats["input_dropped"] += 1
            return False
        try:
            self._input.put_nowait(self._frame(frame).copy())
        except queue.Full:
            self._slots.release()
            self._stats["input_dropped"] += 1
            return False
        except Exception:
            self._slots.release()
            raise
        self._stats["submitted"] += 1
        return True

    def _run(self):
        try:
            while not self._stop.is_set():
                while True:
                    try:
                        command, value = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == "parameters":
                        self._parameters = value
                    elif command == "graph":
                        self._graph = value
                    elif command == "clear":
                        self._history.fill(0)
                        self._count = self._write_index = 0
                    self._stats["commands_applied"] += 1
                try:
                    frame = self._input.get(timeout=0.01)
                except queue.Empty:
                    continue
                # Match Rust command ordering: snapshots queued while the worker
                # waited for input apply to this frame.
                while True:
                    try:
                        command, value = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == "parameters":
                        self._parameters = value
                    elif command == "graph":
                        self._graph = value
                    elif command == "clear":
                        self._history.fill(0)
                        self._count = self._write_index = 0
                    self._stats["commands_applied"] += 1
                params = self._parameters
                temporal_ops = [operation for operation in self._graph.operations if operation.opcode in {
                    GraphOpcode.TEMPORAL_MIX, GraphOpcode.BLEND,
                }]
                history_frozen = any(
                    params.frozen if operation.flags & 2 else bool(operation.flags & 1)
                    for operation in temporal_ops
                )
                if temporal_ops and not history_frozen:
                    self._history[self._write_index] = frame
                    self._write_index = (self._write_index + 1) % self.config.history_capacity
                    self._count = min(self._count + 1, self.config.history_capacity)
                for operation in self._graph.operations:
                    if operation.opcode in {GraphOpcode.INPUT, GraphOpcode.OUTPUT}:
                        continue
                    if operation.opcode == GraphOpcode.INVERT:
                        np.subtract(255, frame, out=frame)
                    elif operation.opcode == GraphOpcode.BRIGHTNESS_CONTRAST:
                        adjusted = frame.astype(np.float32) * (1.0 + operation.arg1 / 100.0)
                        adjusted += operation.arg0
                        frame[...] = np.clip(adjusted, 0, 255).astype(np.uint8)
                    elif operation.opcode in {GraphOpcode.TEMPORAL_MIX, GraphOpcode.BLEND} and self._count:
                        delay = params.delay if operation.flags & 2 else operation.arg2
                        mix = params.mix if operation.flags & 2 else operation.arg0
                        delay = min(int(delay), self._count - 1)
                        index = (self._write_index + self.config.history_capacity - 1 - delay) % self.config.history_capacity
                        old = self._history[index]
                        alpha = int(float(mix) * 65536.0 + 0.5)
                        current_u32 = frame.astype(np.uint32)
                        old_u32 = old.astype(np.uint32)
                        frame[...] = (
                            (current_u32 * (65536 - alpha) + old_u32 * alpha + 32768) >> 16
                        ).astype(np.uint8)
                self._sequence += 1
                self._stats["processed"] += 1
                try:
                    self._output.put_nowait((frame, self._sequence))
                except queue.Full:
                    self._stats["output_dropped"] += 1
                    self._slots.release()
        except Exception:
            self._state = EngineState.FAILED
        finally:
            if self._state != EngineState.FAILED:
                self._state = EngineState.STOPPED

    def poll_frame(self, output=None):
        try:
            frame, sequence = self._output.get_nowait()
        except queue.Empty:
            return None
        self._stats["delivered"] += 1
        self._slots.release()
        if output is not None:
            target = self._frame(output)
            target[...] = frame
            frame = target
        return frame, sequence

    def wait_for_frame(self, output=None, timeout=1.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            result = self.poll_frame(output)
            if result is not None:
                return result
            time.sleep(0.0005)
        raise TimeoutError("timed out waiting for Python runtime output")

    def update_parameters(self, snapshot):
        if snapshot.delay < 0 or snapshot.delay >= self.config.history_capacity or not 0 <= snapshot.mix <= 1:
            raise NativeRuntimeFailure("invalid parameter snapshot")
        self._commands.put_nowait(("parameters", snapshot))

    def update_graph(self, snapshot: GraphSnapshot):
        validate_graph_snapshot(snapshot, self.config.history_capacity)
        if self._state != EngineState.RUNNING:
            raise NativeRuntimeFailure("engine is not running")
        self._commands.put_nowait(("graph", snapshot))

    def clear(self):
        self._commands.put_nowait(("clear", None))

    def get_statistics(self):
        return dict(self._stats) | {
            "backend": self.backend,
            "frames_in_history": self._count,
            "allocated_bytes": self._history.nbytes,
            "sequence": self._sequence,
            "state": self._state.name.lower(),
        }

    def shutdown(self):
        if self._state in {EngineState.STOPPED, EngineState.FAILED}:
            return
        self._state = EngineState.STOPPING
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._state = EngineState.FAILED
                raise NativeRuntimeFailure("Python fallback thread did not stop")
        self._state = EngineState.STOPPED

    close = shutdown


def create_runtime(config: RuntimeConfig, backend: str | None = None):
    requested = (backend or os.getenv("MEGAGLITCH_RUNTIME_BACKEND", "auto")).strip().lower()
    if requested not in {"auto", "native", "python"}:
        raise ValueError("runtime backend must be 'auto', 'native', or 'python'")
    if requested == "python":
        return PythonRuntime(config)
    if requested == "native":
        return NativeRuntime(config)
    try:
        return NativeRuntime(config)
    except NativeRuntimeUnavailable:
        return PythonRuntime(config)


def native_graph_for_pipeline(pipeline) -> GraphSnapshot | None:
    """Compile a supported flat MegaGlitch pipeline into graph IR v1."""
    nodes = [node for node in (pipeline or []) if node.get("enabled", True)]
    if any(
        node.get("node_type") == "source"
        and node.get("source_key", "primary") != "primary"
        for node in nodes
    ):
        return None
    active = [
        node for node in nodes
        if node.get("node_type") not in {"source", "display"}
        and node.get("effect_type") != "Display"
        and str((node.get("params") or {}).get("enabled", True)).lower() != "false"
    ]
    operations = [GraphOperation(GraphOpcode.INPUT)]
    for node in active:
        label = node.get("effect_type")
        params = node.get("params") or {}
        if has_active_temporal_macros(params):
            return None
        try:
            if label == "Invert Colors":
                if int(params.get("invert_chance", 0) or 0) > 0:
                    operations.append(GraphOperation(GraphOpcode.INVERT))
            elif label == "Brightness / Contrast":
                brightness = float(params.get("brightness", 0) or 0)
                contrast = float(params.get("contrast", 0) or 0)
                if not math.isfinite(brightness) or not math.isfinite(contrast):
                    return None
                if brightness != 0.0 or contrast != 0.0:
                    operations.append(GraphOperation(
                        GraphOpcode.BRIGHTNESS_CONTRAST,
                        arg0=brightness,
                        arg1=contrast,
                    ))
            elif label in {"Native Temporal Mix", "Native History Blend"}:
                mix = float(params.get("mix", 0.5) or 0.0)
                delay = max(0, int(params.get("delay", 0) or 0))
                if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
                    return None
                opcode = (
                    GraphOpcode.TEMPORAL_MIX
                    if label == "Native Temporal Mix"
                    else GraphOpcode.BLEND
                )
                operations.append(GraphOperation(
                    opcode,
                    flags=1 if bool(params.get("freeze", False)) else 0,
                    arg0=mix,
                    arg2=delay,
                ))
            else:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    operations.append(GraphOperation(GraphOpcode.OUTPUT))
    return GraphSnapshot(tuple(operations))


def native_mode_for_pipeline(pipeline) -> str | None:
    """Backward-compatible classifier; execution uses `native_graph_for_pipeline`."""
    graph = native_graph_for_pipeline(pipeline)
    if graph is None:
        return None
    opcodes = [operation.opcode for operation in graph.operations[1:-1]]
    if not opcodes:
        return "passthrough"
    if opcodes == [GraphOpcode.INVERT]:
        return "invert"
    if opcodes == [GraphOpcode.TEMPORAL_MIX]:
        return "temporal_mix"
    return "graph_v1"
