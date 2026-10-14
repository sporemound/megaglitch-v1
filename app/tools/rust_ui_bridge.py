"""Versioned JSON-lines bridge from the Rust UI to MegaGlitch's Python frame path."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
except Exception:
    pass

from node_pipeline_runtime import execute_flat_node_pipeline
from megaglitch.preview_executor import PreviewExecutor, PreviewGraphSnapshot
from megaglitch.session_graph_compiler import (
    GraphCompileError,
    compile_preset_graph as compile_session_graph,
)


PROTOCOL_VERSION = 2
BUILD_ID = "rust-ui-bridge-v2"
DEFAULT_WIDTH = 160
DEFAULT_HEIGHT = 90
PREVIEW_SLOTS = 3
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720
PREVIEW_CHANNELS = 3
PREVIEW_SLOT_BYTES = PREVIEW_MAX_WIDTH * PREVIEW_MAX_HEIGHT * PREVIEW_CHANNELS


class BridgeProtocolError(ValueError):
    """A bridge validation failure with an optional stable protocol code."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)


class SourceError(BridgeProtocolError):
    """A source failure with a stable protocol error code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message, code=code, **details)


def structured_error(
    code: str,
    message: str,
    *,
    path: str | None = None,
    recoverable: bool = True,
    **details: Any,
) -> dict[str, Any]:
    """Return the protocol-v2 error envelope consumed directly by the Rust UI."""

    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "recoverable": bool(recoverable),
    }
    if path:
        error["path"] = str(path)
    if details:
        error["details"] = details
    return error


def _emit(message: dict[str, Any], stream=sys.stdout) -> None:
    stream.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def deterministic_frame(sequence: int, width: int, height: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            (xx + sequence * 3) % 256,
            (yy * 2 + sequence * 5) % 256,
            ((xx ^ yy) + sequence * 7) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _load_cv2():
    """Import OpenCV only when a file-backed source is selected."""

    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local optional runtime
        raise SourceError(
            "opencv_unavailable",
            "OpenCV is required for image and video sources",
            exception=type(error).__name__,
        ) from error
    return cv2


def _checked_source_path(path_value: Any, kind: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise SourceError("source_path_required", f"{kind} source requires a path")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise SourceError("source_not_found", f"source file does not exist: {path}", path=str(path))
    return path


class FrameSource:
    kind = "unknown"
    label = "UNKNOWN SOURCE"

    def __init__(self) -> None:
        self.ready = True

    def read(self, sequence: int, width: int, height: int) -> np.ndarray:
        raise NotImplementedError

    def seek(self, seconds: float) -> None:
        raise SourceError("seek_unsupported", f"{self.kind} source does not support seeking")

    def close(self) -> None:
        self.ready = False


class TestPatternSource(FrameSource):
    kind = "test_pattern"
    label = "TEST PATTERN"

    def read(self, sequence: int, width: int, height: int) -> np.ndarray:
        return deterministic_frame(sequence, width, height)


class ImageFrameSource(FrameSource):
    kind = "image"

    def __init__(self, path_value: Any) -> None:
        super().__init__()
        self.path = _checked_source_path(path_value, self.kind)
        self.label = str(self.path)
        cv2 = _load_cv2()
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            self.ready = False
            raise SourceError(
                "image_decode_failed",
                f"OpenCV could not decode image: {self.path}",
                path=str(self.path),
            )
        self._image = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def read(self, sequence: int, width: int, height: int) -> np.ndarray:
        del sequence, width, height
        if not self.ready:
            raise SourceError("source_not_ready", "image source is not ready", path=str(self.path))
        # The frame path is allowed to mutate a uniquely owned input.
        return self._image.copy()


class VideoFrameSource(FrameSource):
    kind = "video"

    def __init__(self, path_value: Any) -> None:
        super().__init__()
        self.path = _checked_source_path(path_value, self.kind)
        self.label = str(self.path)
        cv2 = _load_cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            self.ready = False
            raise SourceError(
                "video_open_failed",
                f"OpenCV could not open video: {self.path}",
                path=str(self.path),
            )
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            capture.release()
            self.ready = False
            raise SourceError(
                "video_decode_failed",
                f"OpenCV could not decode the first video frame: {self.path}",
                path=str(self.path),
            )
        self._cv2 = cv2
        self._capture = capture
        self._prefetched_frame: np.ndarray | None = np.ascontiguousarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = min(120.0, max(1.0, fps)) if np.isfinite(fps) and fps > 0.0 else 30.0

    def read(self, sequence: int, width: int, height: int) -> np.ndarray:
        del sequence, width, height
        if not self.ready:
            raise SourceError("source_not_ready", "video source is not ready", path=str(self.path))
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            return frame
        ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            # File sources loop. Failure after a rewind is a real decoder error.
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0.0)
            ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            self.ready = False
            raise SourceError(
                "video_decode_failed",
                f"OpenCV failed while decoding video: {self.path}",
                path=str(self.path),
            )
        return np.ascontiguousarray(self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB))

    def seek(self, seconds: float) -> None:
        if not self.ready:
            raise SourceError("source_not_ready", "video source is not ready", path=str(self.path))
        if not np.isfinite(seconds) or seconds < 0.0:
            raise SourceError("invalid_seek", "seek seconds must be finite and non-negative")
        self._prefetched_frame = None
        if not self._capture.set(self._cv2.CAP_PROP_POS_MSEC, float(seconds) * 1000.0):
            raise SourceError(
                "video_seek_failed",
                f"video backend rejected seek to {seconds:.3f}s",
                path=str(self.path),
                seconds=float(seconds),
            )

    def close(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()
        super().close()


def create_source(kind: Any, path: Any = None) -> FrameSource:
    if kind == "test_pattern":
        return TestPatternSource()
    if kind == "image":
        return ImageFrameSource(path)
    if kind == "video":
        return VideoFrameSource(path)
    raise SourceError(
        "unsupported_source_kind",
        f"unsupported source kind: {kind!r}",
        supported=["test_pattern", "image", "video"],
    )


def frame_to_rgb8(frame: Any) -> np.ndarray:
    """Normalize the canonical worker RGB/RGBA/grayscale output to packed RGB8."""

    if hasattr(frame, "download"):
        frame = frame.download()
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], PREVIEW_CHANNELS, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, PREVIEW_CHANNELS, axis=2)
    elif array.ndim == 3 and array.shape[2] >= PREVIEW_CHANNELS:
        array = array[:, :, :PREVIEW_CHANNELS]
    else:
        raise BridgeProtocolError(f"preview frame has unsupported shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def downscale_preview_rgb(frame: np.ndarray) -> np.ndarray:
    """Fit RGB8 into the fixed surface without adding a required image dependency."""

    height, width = frame.shape[:2]
    scale = min(1.0, PREVIEW_MAX_WIDTH / width, PREVIEW_MAX_HEIGHT / height)
    if scale >= 1.0:
        return np.ascontiguousarray(frame)
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    x_indices = np.linspace(0, width - 1, target_width).astype(np.intp)
    y_indices = np.linspace(0, height - 1, target_height).astype(np.intp)
    return np.ascontiguousarray(frame[y_indices][:, x_indices])


def _resolve_temp_dir() -> str | None:
    candidates = [
        os.environ.get("MEGAGLITCH_TEMP_DIR"),
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
        str(ROOT / ".tmp"),
        r"v:\.tmp",
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            try:
                test_path = os.path.join(candidate, f".mg_test_{os.getpid()}_{time.time_ns()}")
                with open(test_path, "wb") as f:
                    f.write(b"1")
                os.unlink(test_path)
                return candidate
            except Exception:
                continue
    return None


class PreviewSurface:
    """Headerless, fixed-size RGB8 slots shared with the Rust UI via mmap."""

    def __init__(self, generation: int | None = None) -> None:
        temp_dir = _resolve_temp_dir()
        handle, raw_path = tempfile.mkstemp(prefix="megaglitch-preview-", suffix=".rgb8", dir=temp_dir)
        os.close(handle)
        self.path = Path(raw_path).resolve()
        self.generation = int(generation if generation is not None else time.time_ns() & 0x7FFF_FFFF_FFFF_FFFF)
        self.slots = PREVIEW_SLOTS
        self.slot_bytes = PREVIEW_SLOT_BYTES
        self._file = self.path.open("r+b")
        self._file.truncate(self.slots * self.slot_bytes)
        self._file.flush()
        self._map = mmap.mmap(self._file.fileno(), self.slots * self.slot_bytes, access=mmap.ACCESS_WRITE)
        self._leases: dict[int, int] = {}
        self._closed = False

    def descriptor(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "type": "preview_surface",
            "path": str(self.path),
            "generation": self.generation,
            "slots": self.slots,
            "slot_bytes": self.slot_bytes,
            "pixel_format": "RGB8",
            "max_width": PREVIEW_MAX_WIDTH,
            "max_height": PREVIEW_MAX_HEIGHT,
        }

    def write_frame(
        self,
        rgb_frame: np.ndarray,
        sequence: int,
        timestamp: float,
    ) -> dict[str, Any] | None:
        if self._closed:
            raise BridgeProtocolError("preview surface is closed")
        frame = downscale_preview_rgb(rgb_frame)
        height, width = frame.shape[:2]
        stride = width * PREVIEW_CHANNELS
        nbytes = height * stride
        if nbytes > self.slot_bytes:
            raise BridgeProtocolError("preview frame exceeds fixed slot capacity")
        slot = next((candidate for candidate in range(self.slots) if candidate not in self._leases), None)
        if slot is None:
            return None
        offset = slot * self.slot_bytes
        view = memoryview(np.ascontiguousarray(frame)).cast("B")
        self._map[offset : offset + nbytes] = view[:nbytes]
        self._leases[slot] = int(sequence)
        return {
            "version": PROTOCOL_VERSION,
            "type": "preview_frame",
            "slot": slot,
            "generation": self.generation,
            "sequence": int(sequence),
            "width": int(width),
            "height": int(height),
            "stride": int(stride),
            "nbytes": int(nbytes),
            "timestamp": float(timestamp),
        }

    def release(self, slot: Any, generation: Any, sequence: Any = None) -> bool:
        if generation != self.generation or not isinstance(slot, int) or isinstance(slot, bool):
            return False
        leased_sequence = self._leases.get(slot)
        if leased_sequence is None:
            return False
        if sequence is not None and sequence != leased_sequence:
            return False
        del self._leases[slot]
        return True

    @property
    def leased_slots(self) -> dict[int, int]:
        return dict(self._leases)

    def read_slot_bytes(self, slot: int, nbytes: int) -> bytes:
        if not 0 <= slot < self.slots or not 0 <= nbytes <= self.slot_bytes:
            raise ValueError("invalid preview slot read")
        offset = slot * self.slot_bytes
        return bytes(self._map[offset : offset + nbytes])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._map.close()
        self._file.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            # A consumer that violates the shutdown contract may still have the Windows mapping
            # open. Do not turn an otherwise orderly engine shutdown into a process failure; the
            # Rust owner unmaps before shutdown and performs a final best-effort path cleanup.
            pass

    def __enter__(self) -> "PreviewSurface":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


_DECK_A_SOURCE_NAMESPACE = {
    "SOURCE": "primary",
    "SOURCE_B": "secondary",
}


def compile_preset_graph(preset: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile one Deck-A v1 graph while preserving the bridge's one-argument API.

    Engine V2 Gate 1 owns only the primary frame source.  The shared compiler
    still receives the complete Deck-A namespace so it can validate legacy
    graphs faithfully, then this compatibility boundary rejects a pipeline
    that would otherwise let the flat executor substitute black for Source B.
    """

    try:
        pipeline = compile_session_graph(preset, _DECK_A_SOURCE_NAMESPACE)
    except GraphCompileError as error:
        raise BridgeProtocolError(
            str(error),
            code=error.code,
            **error.details,
        ) from error

    secondary_steps = [
        step
        for step in pipeline
        if step.get("node_type") == "source" and step.get("source_key") == "secondary"
    ]
    if secondary_steps:
        raise BridgeProtocolError(
            "graph requires Source B, but Engine V2 Gate 1 has no secondary source owner",
            code="secondary_source_unavailable",
            source_key="secondary",
            source_labels=[str(step.get("effect_type") or "Source B") for step in secondary_steps],
        )
    return pipeline


@dataclass
class FrameState:
    previous_frame: Any = None
    feedback_image: Any = None
    trail_image: Any = None
    frame_buffer: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.frame_buffer is None:
            self.frame_buffer = []


def process_pipeline(
    pipeline: list[dict[str, Any]],
    frame: np.ndarray,
    state: FrameState | None = None,
) -> tuple[np.ndarray, FrameState]:
    state = state or FrameState()
    output, previous, feedback, trail = execute_flat_node_pipeline(
        pipeline,
        frame,
        previous_frame=state.previous_frame,
        feedback_image=state.feedback_image,
        trail_image=state.trail_image,
        frame_buffer=state.frame_buffer,
        fallback_size=(frame.shape[1], frame.shape[0]),
        source_copy_policy="borrow_unique",
    )
    state.previous_frame = previous
    state.feedback_image = feedback
    state.trail_image = trail
    return np.ascontiguousarray(output), state


@dataclass(frozen=True)
class AuthoritativeFrameResult:
    source_sequence: int
    epoch: int
    graph_generation: int
    output: np.ndarray
    state: FrameState
    frame_ms: float


def _process_authoritative_frame(
    pipeline: list[dict[str, Any]],
    frame: np.ndarray,
    state: FrameState,
    source_sequence: int,
    epoch: int,
    graph_generation: int,
) -> AuthoritativeFrameResult:
    started = time.perf_counter()
    output, next_state = process_pipeline(pipeline, frame, state)
    return AuthoritativeFrameResult(
        source_sequence=int(source_sequence),
        epoch=int(epoch),
        graph_generation=int(graph_generation),
        output=np.ascontiguousarray(output),
        state=next_state,
        frame_ms=(time.perf_counter() - started) * 1000.0,
    )


class RateMeter:
    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def mark(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self._timestamps.append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def rate(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        self._trim(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0.0 else 0.0


class BridgeEngine:
    """Protocol state machine and one coarse-grained Python frame owner."""

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        surface: PreviewSurface | None = None,
        hash_mode: str = "off",
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.hash_mode = str(hash_mode).lower()
        self.surface = surface or PreviewSurface()
        self._owns_surface = surface is None
        self.pipeline = [
            {
                "effect_type": "Source",
                "params": {},
                "node_type": "source",
                "source_key": "primary",
            },
            {"effect_type": "Display", "params": {}, "node_type": "display"},
        ]
        self.frame_state = FrameState()
        self.source: FrameSource = TestPatternSource()
        self.transport_running = True
        self.state_revision = 0
        self.graph_generation = 0
        self.sequence = 0
        self.preview_sequence = 0
        self.processing_drops = 0
        self.preview_drops = 0
        self.frame_ms = 0.0
        self.output_width = self.width
        self.output_height = self.height
        self.last_output_sha256: str | None = None
        self.last_error: dict[str, Any] | None = None
        self.graph_rate = RateMeter()
        self.preview_rate = RateMeter()
        self.preview_executor = PreviewExecutor(
            PreviewGraphSnapshot.from_pipeline(self.pipeline, self.graph_generation),
            preferred_tier=(960, 540),
            adaptive=True,
            autostart=False,
        )
        self.authoritative_sequence = 0
        self.authoritative_graph_generation = self.graph_generation
        self.authoritative_drops = 0
        self._authoritative_epoch = 0
        self._authoritative_pool: ThreadPoolExecutor | None = None
        self._authoritative_future: Future[AuthoritativeFrameResult] | None = None
        self._decoupled_started = False
        self._last_preview_error: dict[str, Any] | None = None
        self._minimum_preview_sequence = 0
        self._closed = False

    def start_decoupled(self) -> None:
        if self._decoupled_started:
            return
        self._authoritative_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mg-authoritative-current",
        )
        self.preview_executor.start()
        self._decoupled_started = True

    @property
    def source_kind(self) -> str:
        return self.source.kind

    @property
    def source_label(self) -> str:
        return self.source.label

    @property
    def source_ready(self) -> bool:
        return self.source.ready

    def ready_event(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "type": "ready",
            "backend": "python_flat_pipeline",
            "build": BUILD_ID,
            "capabilities": {
                "transactional_commands": [
                    "replace_graph",
                    "set_transport",
                    "set_source",
                    "seek",
                    "shutdown",
                ],
                "fire_and_forget_commands": ["release_preview_slot"],
                "source_kinds": ["test_pattern", "image", "video"],
                "preview": {
                    "transport": "mmap_fixed_slots",
                    "pixel_format": "RGB8",
                    "executor": "preview-v1-decoupled",
                    "mailbox_capacity": self.preview_executor.mailbox_capacity,
                    "adaptive_tiers": [[1280, 720], [960, 540], [720, 480], [640, 360]],
                    "slots": self.surface.slots,
                    "slot_bytes": self.surface.slot_bytes,
                    "max_width": PREVIEW_MAX_WIDTH,
                    "max_height": PREVIEW_MAX_HEIGHT,
                },
            },
            "state_revision": self.state_revision,
            "graph_generation": self.graph_generation,
            "transport_running": self.transport_running,
            "source_kind": self.source_kind,
            "source_label": self.source_label,
            "source_ready": self.source_ready,
            "width": self.width,
            "height": self.height,
        }

    def _ack(
        self,
        command: dict[str, Any],
        accepted: bool,
        error: dict[str, Any] | None = None,
        effective_frame_sequence: int | None = None,
    ) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "type": "ack",
            "id": command.get("id"),
            "command_type": command.get("type"),
            "accepted": bool(accepted),
            "state_revision": self.state_revision,
            "effective_frame_sequence": int(
                self.sequence if effective_frame_sequence is None else effective_frame_sequence
            ),
            "graph_generation": self.graph_generation,
            "error": error,
        }

    def _reject(
        self,
        command: dict[str, Any],
        code: str,
        message: str,
        **details: Any,
    ) -> tuple[dict[str, Any], bool]:
        return self._ack(command, False, structured_error(code, message, **details)), False

    def handle_command(self, command: Any) -> tuple[dict[str, Any] | None, bool]:
        """Apply one command. Returns (optional ack, should_shutdown)."""

        if not isinstance(command, dict):
            synthetic = {"id": None, "type": None}
            return self._reject(
                synthetic,
                "invalid_command",
                "command must be a JSON object",
                received_type=type(command).__name__,
            )
        command_type = command.get("type")
        if command_type == "_eof":
            return None, True
        if command_type == "_invalid":
            return self._reject(
                command,
                "invalid_json",
                str(command.get("error") or "input was not valid JSON"),
            )
        # Slot release is intentionally not part of the revisioned command stream.
        # A sequence match prevents a delayed release from freeing a reused slot.
        if command_type == "release_preview_slot":
            if command.get("version") == PROTOCOL_VERSION:
                self.surface.release(
                    command.get("slot"),
                    command.get("generation"),
                    command.get("sequence"),
                )
            return None, False
        if command.get("version") != PROTOCOL_VERSION:
            return self._reject(
                command,
                "protocol_version_mismatch",
                f"expected protocol version {PROTOCOL_VERSION}",
                received=command.get("version"),
            )
        if command.get("id") is None:
            return self._reject(command, "missing_command_id", "transactional command requires id")
        expected_revision = command.get("expected_state_revision")
        if not isinstance(expected_revision, int) or isinstance(expected_revision, bool):
            return self._reject(
                command,
                "invalid_state_revision",
                "transactional command requires integer expected_state_revision",
            )
        if expected_revision != self.state_revision:
            return self._reject(
                command,
                "state_revision_conflict",
                "command was based on a stale state revision",
                expected=expected_revision,
                actual=self.state_revision,
            )

        try:
            if command_type == "replace_graph":
                graph = command.get("graph")
                if not isinstance(graph, dict):
                    raise BridgeProtocolError("replace_graph requires a graph object")
                # Compile first: assignment is the commit point and preserves the
                # last accepted pipeline on every validation/compile failure.
                candidate_pipeline = compile_preset_graph(graph)
                next_generation = self.graph_generation + 1
                preview_snapshot = PreviewGraphSnapshot.from_pipeline(
                    candidate_pipeline,
                    next_generation,
                )
                self.preview_executor.update_graph(preview_snapshot)
                self.pipeline = candidate_pipeline
                self.frame_state = FrameState()
                self.graph_generation = next_generation
                self._authoritative_epoch += 1
            elif command_type == "set_transport":
                requested_running = command.get("running")
                if not isinstance(requested_running, bool):
                    raise BridgeProtocolError("set_transport requires boolean running")
                self.transport_running = requested_running
                if not requested_running:
                    self._minimum_preview_sequence = self.sequence + 1
            elif command_type == "set_source":
                candidate_source = create_source(command.get("kind"), command.get("path"))
                previous_source = self.source
                self.source = candidate_source
                previous_source.close()
                self.frame_state = FrameState()
                self.preview_executor.reset_temporal_state()
                self._authoritative_epoch += 1
                self._minimum_preview_sequence = self.sequence + 1
                self.output_width = self.width
                self.output_height = self.height
            elif command_type == "seek":
                seconds = command.get("seconds")
                if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
                    raise SourceError("invalid_seek", "seek requires numeric seconds")
                self.source.seek(float(seconds))
                self.frame_state = FrameState()
                self.preview_executor.reset_temporal_state()
                self._authoritative_epoch += 1
                self._minimum_preview_sequence = self.sequence + 1
            elif command_type == "shutdown":
                self.transport_running = False
            else:
                return self._reject(
                    command,
                    "unknown_command",
                    f"unknown command {command_type!r}",
                )
        except SourceError as error:
            return self._reject(command, error.code, str(error), **error.details)
        except BridgeProtocolError as error:
            fallback_code = "invalid_graph" if command_type == "replace_graph" else "invalid_command"
            return self._reject(
                command,
                error.code or fallback_code,
                str(error),
                **error.details,
            )
        except Exception as error:
            return self._reject(
                command,
                "command_failed",
                str(error),
                exception=type(error).__name__,
            )

        self.state_revision += 1
        self.last_error = None
        effective_sequence = self.sequence if command_type == "shutdown" else self.sequence + 1
        return self._ack(command, True, None, effective_sequence), command_type == "shutdown"

    def _poll_authoritative(self, now: float | None = None) -> list[dict[str, Any]]:
        future = self._authoritative_future
        if future is None or not future.done():
            return []
        self._authoritative_future = None
        now = time.perf_counter() if now is None else now
        try:
            result = future.result()
        except Exception as error:
            self.processing_drops += 1
            self.last_error = structured_error(
                "authoritative_frame_failed",
                str(error),
                exception=type(error).__name__,
            )
            return [
                {
                    "version": PROTOCOL_VERSION,
                    "type": "runtime_error",
                    "state_revision": self.state_revision,
                    "graph_generation": self.graph_generation,
                    "effective_frame_sequence": self.sequence,
                    "error": self.last_error,
                }
            ]
        if (
            result.epoch != self._authoritative_epoch
            or result.graph_generation != self.graph_generation
        ):
            return []
        self.frame_state = result.state
        self.frame_ms = result.frame_ms
        self.output_height, self.output_width = result.output.shape[:2]
        self.authoritative_sequence = result.source_sequence
        self.authoritative_graph_generation = result.graph_generation
        if self.hash_mode == "sha256":
            self.last_output_sha256 = hashlib.sha256(
                memoryview(result.output)
            ).hexdigest()
        else:
            self.last_output_sha256 = None
        self.last_error = None
        self.graph_rate.mark(now)
        return []

    def submit_decoupled_frame(self, now: float | None = None) -> list[dict[str, Any]]:
        """Read one source frame and fan it out without waiting for either graph."""

        if not self._decoupled_started:
            raise RuntimeError("decoupled bridge has not been started")
        now = time.perf_counter() if now is None else now
        events = self._poll_authoritative(now)
        self.sequence += 1
        try:
            frame = self.source.read(self.sequence, self.width, self.height)
            source_timestamp = time.time()
            self.preview_executor.submit_latest(
                frame,
                source_timestamp,
                self.graph_generation,
                sequence=self.sequence,
            )
            if self._authoritative_future is None:
                assert self._authoritative_pool is not None
                self._authoritative_future = self._authoritative_pool.submit(
                    _process_authoritative_frame,
                    self.pipeline,
                    frame,
                    self.frame_state,
                    self.sequence,
                    self._authoritative_epoch,
                    self.graph_generation,
                )
            else:
                self.authoritative_drops += 1
                self.processing_drops += 1
        except Exception as error:
            self.processing_drops += 1
            if isinstance(error, SourceError):
                self.last_error = structured_error(error.code, str(error), **error.details)
            else:
                self.last_error = structured_error(
                    "source_fanout_failed",
                    str(error),
                    exception=type(error).__name__,
                )
            events.append(
                {
                    "version": PROTOCOL_VERSION,
                    "type": "runtime_error",
                    "state_revision": self.state_revision,
                    "graph_generation": self.graph_generation,
                    "effective_frame_sequence": self.sequence,
                    "error": self.last_error,
                }
            )
        return events

    def poll_decoupled_events(self, now: float | None = None) -> list[dict[str, Any]]:
        now = time.perf_counter() if now is None else now
        events = self._poll_authoritative(now)
        result = self.preview_executor.consume_latest()
        if result is not None and (
            not self.transport_running
            or result.sequence < self._minimum_preview_sequence
            or result.graph_revision != self.graph_generation
        ):
            result = None
        if result is not None:
            preview = self.surface.write_frame(
                result.frame,
                sequence=result.sequence,
                timestamp=result.source_timestamp,
            )
            if preview is None:
                self.preview_drops += 1
            else:
                preview.update(
                    {
                        "source_sequence": result.sequence,
                        "graph_generation": result.graph_revision,
                        "graph_revision": result.graph_revision,
                        "resolution_generation": result.resolution_generation,
                        "source_timestamp": result.source_timestamp,
                        "submitted_at": result.submitted_at,
                        "rendered_at": result.rendered_at,
                        "submitted_at_unix_ns": int(result.submitted_at * 1_000_000_000),
                        "rendered_at_unix_ns": int(result.rendered_at * 1_000_000_000),
                        "render_ms": result.render_ms,
                        "source_position": result.source_position,
                    }
                )
                self.preview_sequence = result.sequence
                self.preview_rate.mark(now)
                events.append(preview)

        preview_telemetry = self.preview_executor.telemetry()
        if preview_telemetry.last_error != self._last_preview_error:
            self._last_preview_error = preview_telemetry.last_error
            if self._last_preview_error is not None:
                events.append(
                    {
                        "version": PROTOCOL_VERSION,
                        "type": "runtime_error",
                        "state_revision": self.state_revision,
                        "graph_generation": self.graph_generation,
                        "effective_frame_sequence": self.sequence,
                        "error": {
                            **self._last_preview_error,
                            "recoverable": True,
                        },
                    }
                )
        return events

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.perf_counter() if now is None else now
        preview = self.preview_executor.telemetry()
        preview_drop_total = self.preview_drops + preview.preview_dropped
        node_statuses = [status.as_dict() for status in preview.node_statuses]
        unsupported_nodes = [
            status for status in node_statuses if status["status"] == "unsupported"
        ]
        return {
            "version": PROTOCOL_VERSION,
            "type": "snapshot",
            "backend": "python_flat_pipeline",
            "build": BUILD_ID,
            "state_revision": self.state_revision,
            "graph_generation": self.graph_generation,
            "transport_running": self.transport_running,
            "source_kind": self.source_kind,
            "source_label": self.source_label,
            "source_ready": self.source_ready,
            "graph_sequence": self.authoritative_sequence,
            "preview_sequence": self.preview_sequence,
            "graph_fps": self.graph_rate.rate(now),
            "preview_fps": preview.preview_render_fps,
            "frame_ms": self.frame_ms,
            "processing_drops": self.processing_drops,
            "preview_drops": preview_drop_total,
            "dropped_frames": self.processing_drops + preview_drop_total,
            "authoritative_fps": self.graph_rate.rate(now),
            "authoritative_frame_ms": self.frame_ms,
            "authoritative_sequence": self.authoritative_sequence,
            "authoritative_graph_generation": self.authoritative_graph_generation,
            "authoritative_dropped": self.authoritative_drops,
            "preview_render_fps": preview.preview_render_fps,
            "preview_frame_ms": preview.preview_frame_ms,
            "preview_p95_ms": preview.preview_p95_ms,
            "preview_width": preview.preview_resolution[0],
            "preview_height": preview.preview_resolution[1],
            "preview_tier_width": preview.configured_tier[0],
            "preview_tier_height": preview.configured_tier[1],
            "preview_requests": preview.preview_requests,
            "preview_rendered": preview.preview_rendered,
            "preview_superseded": preview.preview_superseded,
            "preview_dropped": preview_drop_total,
            "preview_rendered_sequence": preview.preview_sequence,
            "preview_resolution_generation": preview.resolution_generation,
            "preview_graph_revision": preview.last_rendered_graph_revision,
            "preview_node_statuses": node_statuses,
            "preview_unsupported_nodes": unsupported_nodes,
            "preview_last_error": preview.last_error,
            "output_sha256": self.last_output_sha256,
            "width": self.output_width,
            "height": self.output_height,
            "last_error": self.last_error,
        }

    def render_frame(self, now: float | None = None) -> list[dict[str, Any]]:
        """Read, process, and publish exactly one graph frame."""

        now = time.perf_counter() if now is None else now
        self.sequence += 1
        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        try:
            frame = self.source.read(self.sequence, self.width, self.height)
            output, self.frame_state = process_pipeline(self.pipeline, frame, self.frame_state)
            self.frame_ms = (time.perf_counter() - started) * 1000.0
            self.output_height, self.output_width = output.shape[:2]
            if self.hash_mode == "sha256":
                self.last_output_sha256 = hashlib.sha256(
                    memoryview(np.ascontiguousarray(output))
                ).hexdigest()
            else:
                self.last_output_sha256 = None
            self.last_error = None
            self.graph_rate.mark(now)
            preview = self.surface.write_frame(
                frame_to_rgb8(output),
                sequence=self.sequence,
                timestamp=time.time(),
            )
            if preview is None:
                self.preview_drops += 1
            else:
                self.preview_sequence = self.sequence
                self.preview_rate.mark(now)
                events.append(preview)
        except Exception as error:
            self.processing_drops += 1
            if isinstance(error, SourceError):
                self.last_error = structured_error(error.code, str(error), **error.details)
            else:
                self.last_error = structured_error(
                    "frame_processing_failed",
                    str(error),
                    exception=type(error).__name__,
                )
            events.append(
                {
                    "version": PROTOCOL_VERSION,
                    "type": "runtime_error",
                    "state_revision": self.state_revision,
                    "graph_generation": self.graph_generation,
                    "effective_frame_sequence": self.sequence,
                    "error": self.last_error,
                }
            )
        return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.source.close()
        if self._decoupled_started:
            self.preview_executor.stop()
            if self._authoritative_pool is not None:
                self._authoritative_pool.shutdown(wait=True, cancel_futures=True)
            self._authoritative_pool = None
            self._authoritative_future = None
            self._decoupled_started = False
        if self._owns_surface:
            self.surface.close()


def _reader(commands: queue.Queue[Any]) -> None:
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                commands.put(json.loads(line))
            except json.JSONDecodeError as error:
                commands.put({"version": PROTOCOL_VERSION, "type": "_invalid", "error": str(error)})
    finally:
        commands.put({"version": PROTOCOL_VERSION, "type": "_eof"})


def run_bridge(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    hash_mode: str = "off",
) -> int:
    commands: queue.Queue[Any] = queue.Queue()
    threading.Thread(target=_reader, args=(commands,), name="mg-bridge-stdin", daemon=True).start()
    engine = BridgeEngine(width, height, hash_mode=hash_mode)
    engine.start_decoupled()
    next_frame = time.perf_counter()
    last_snapshot_time = 0.0
    _emit(engine.ready_event())
    _emit(engine.surface.descriptor())
    should_shutdown = False
    try:
        while not should_shutdown:
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                was_running = engine.transport_running
                previous_source = engine.source
                ack, should_shutdown = engine.handle_command(command)
                if ack is not None:
                    _emit(ack)
                if (not was_running and engine.transport_running) or previous_source is not engine.source:
                    next_frame = time.perf_counter()
                if should_shutdown:
                    break

            now = time.perf_counter()
            if should_shutdown:
                break
            for event in engine.poll_decoupled_events(now):
                _emit(event)
            if not engine.transport_running:
                if now - last_snapshot_time >= 0.25:
                    _emit(engine.snapshot(now))
                    last_snapshot_time = now
                time.sleep(0.002)
                continue
            if now < next_frame:
                time.sleep(min(0.001, next_frame - now))
                continue
            source_fps = float(getattr(engine.source, "fps", 60.0))
            period = 1.0 / min(120.0, max(1.0, source_fps))
            if now - next_frame > period:
                skipped = int((now - next_frame) / period)
                engine.processing_drops += skipped
                next_frame += skipped * period
            next_frame += period
            for event in engine.submit_decoupled_frame(now):
                _emit(event)
            for event in engine.poll_decoupled_events(time.perf_counter()):
                _emit(event)
            if now - last_snapshot_time >= 0.05:
                _emit(engine.snapshot(now))
                last_snapshot_time = now
    finally:
        engine.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--hash-mode", choices=["off", "sha256"], default="off")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    return run_bridge(args.width, args.height, hash_mode=args.hash_mode)


if __name__ == "__main__":
    raise SystemExit(main())
