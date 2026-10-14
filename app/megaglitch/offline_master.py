"""Deterministic, master-resolution offline rendering and direct FFmpeg output.

This module is deliberately UI independent.  It knows nothing about preview
textures, OBS, virtual cameras, or window scaling.  Callers provide master-size
source frames and an existing MegaGlitch processing callback; the callback is
advanced exactly once for every source frame.  FPS conversion is an egress
operation and therefore cannot feed duplicated or resampled frames back into
temporal state.

The two FPS semantics are intentionally different:

``duration_preserving``
    Keeps the source duration.  Master frames are processed in source order and
    deterministically sampled/duplicated only as they are sent to the encoder.

``true_accelerated``
    Sends every processed source frame exactly once on the requested output
    timebase.  For example, 30 source frames at 30 fps written at 55 fps last
    30/55 seconds; the original one-second duration is *not* preserved.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Protocol

import numpy as np

from .determinism import use_numpy_rng
from .external_tools import resolve_ffmpeg_executable, resolve_ffprobe_executable


class FpsSemantics(str, Enum):
    """How source frames are placed on the output timebase."""

    DURATION_PRESERVING = "duration_preserving"
    TRUE_ACCELERATED = "true_accelerated"


class MasterCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"


@dataclass(frozen=True)
class FrameScheduleItem:
    output_index: int
    source_index: int
    output_time_seconds: Fraction


@dataclass(frozen=True)
class FrameSchedule:
    semantics: FpsSemantics
    source_frame_count: int
    source_fps: Fraction
    output_fps: Fraction
    items: tuple[FrameScheduleItem, ...]
    source_duration_seconds: Fraction
    output_duration_seconds: Fraction
    duplicated_frames: int
    source_frames_omitted_at_egress: int

    @property
    def output_frame_count(self) -> int:
        return len(self.items)

    def output_indices_by_source(self) -> tuple[tuple[int, ...], ...]:
        grouped: list[list[int]] = [[] for _ in range(self.source_frame_count)]
        for item in self.items:
            grouped[item.source_index].append(item.output_index)
        return tuple(tuple(indices) for indices in grouped)


def _positive_rate(value: int | float | str | Fraction, name: str) -> Fraction:
    try:
        rate = value if isinstance(value, Fraction) else Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a positive finite frame rate") from error
    if rate <= 0:
        raise ValueError(f"{name} must be a positive finite frame rate")
    return rate


def _round_fraction_half_up(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)


def build_frame_schedule(
    source_frame_count: int,
    source_fps: int | float | str | Fraction,
    output_fps: int | float | str | Fraction,
    semantics: FpsSemantics | str,
) -> FrameSchedule:
    """Build a deterministic egress schedule without touching graph state.

    Duration-preserving conversion uses zero-order hold (frame duplication) at
    the encoder boundary.  It never runs the processing callback twice for a
    duplicated frame.  When the requested rate is lower than the source rate,
    every source frame is still processed; some master outputs are simply not
    selected for the output timebase and are reported separately from runtime
    drops.
    """

    try:
        frame_count = int(source_frame_count)
    except (TypeError, ValueError) as error:
        raise ValueError("source_frame_count must be a non-negative integer") from error
    if frame_count != source_frame_count or frame_count < 0:
        raise ValueError("source_frame_count must be a non-negative integer")

    source_rate = _positive_rate(source_fps, "source_fps")
    output_rate = _positive_rate(output_fps, "output_fps")
    mode = FpsSemantics(semantics)
    source_duration = Fraction(frame_count, 1) / source_rate

    if mode is FpsSemantics.TRUE_ACCELERATED:
        output_count = frame_count
        source_indices = range(frame_count)
    else:
        output_count = _round_fraction_half_up(source_duration * output_rate)
        if frame_count and output_count == 0:
            output_count = 1
        # Exact rational arithmetic avoids platform-dependent float rounding.
        source_indices = (
            min(frame_count - 1, (output_index * source_rate) // output_rate)
            for output_index in range(output_count)
        )

    items = tuple(
        FrameScheduleItem(
            output_index=output_index,
            source_index=int(source_index),
            output_time_seconds=Fraction(output_index, 1) / output_rate,
        )
        for output_index, source_index in enumerate(source_indices)
    )
    counts = [0] * frame_count
    for item in items:
        counts[item.source_index] += 1
    duplicated = sum(max(0, count - 1) for count in counts)
    omitted = sum(count == 0 for count in counts)
    output_duration = Fraction(len(items), 1) / output_rate

    return FrameSchedule(
        semantics=mode,
        source_frame_count=frame_count,
        source_fps=source_rate,
        output_fps=output_rate,
        items=items,
        source_duration_seconds=source_duration,
        output_duration_seconds=output_duration,
        duplicated_frames=duplicated,
        source_frames_omitted_at_egress=omitted,
    )


def stable_graph_hash(graph: Mapping[str, Any] | Sequence[Any]) -> str:
    payload = json.dumps(
        graph,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CpuGpuTransfer:
    direction: str
    byte_count: int
    source_index: int


class OfflineInstrumentation:
    """Mutable per-render collector exposed to existing processing callbacks."""

    def __init__(self, provider_assignments: Mapping[str, str] | None = None) -> None:
        self._providers = {
            str(node): str(provider)
            for node, provider in (provider_assignments or {}).items()
        }
        self._transfers: list[CpuGpuTransfer] = []

    def assign_provider(self, node_id: str, provider: str) -> None:
        node = str(node_id)
        value = str(provider)
        previous = self._providers.get(node)
        if previous is not None and previous != value:
            raise ValueError(
                f"provider assignment for {node!r} changed from {previous!r} to {value!r}"
            )
        self._providers[node] = value

    def record_cpu_gpu_transfer(
        self, direction: str, byte_count: int, source_index: int
    ) -> None:
        normalized = str(direction).strip().lower()
        if normalized not in {"cpu_to_gpu", "gpu_to_cpu"}:
            raise ValueError("direction must be cpu_to_gpu or gpu_to_cpu")
        size = int(byte_count)
        if size < 0:
            raise ValueError("byte_count cannot be negative")
        self._transfers.append(CpuGpuTransfer(normalized, size, int(source_index)))

    @property
    def provider_assignments(self) -> dict[str, str]:
        return dict(sorted(self._providers.items()))

    @property
    def transfers(self) -> tuple[CpuGpuTransfer, ...]:
        return tuple(self._transfers)


@dataclass(frozen=True)
class OfflineFrameContext:
    source_index: int
    source_time_seconds: Fraction
    output_indices: tuple[int, ...]
    frame_seed: int
    graph_hash: str
    master_resolution: tuple[int, int]
    instrumentation: OfflineInstrumentation
    numpy_rng: np.random.Generator = field(repr=False)
    python_rng: random.Random = field(repr=False)


class MasterFrameProcessor(Protocol):
    def __call__(
        self, frame: np.ndarray, context: OfflineFrameContext
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FFmpegEncodeResult:
    command: tuple[str, ...]
    frames_written: int
    bytes_written: int
    returncode: int
    stderr: str
    elapsed_seconds: float
    readback: Mapping[str, Any]


class MasterFrameSink(Protocol):
    def write_frame(self, frame: np.ndarray) -> None: ...

    def close(self) -> FFmpegEncodeResult: ...

    def abort(self) -> None: ...


@dataclass(frozen=True)
class OfflineMasterConfig:
    width: int
    height: int
    source_fps: int | float | str | Fraction
    output_fps: int | float | str | Fraction
    fps_semantics: FpsSemantics
    graph_hash: str
    random_seed: int = 0
    history_buffer_resolution: Mapping[str, tuple[int, int]] = field(
        default_factory=dict
    )
    node_provider_assignment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width = int(self.width)
        height = int(self.height)
        if width <= 0 or height <= 0:
            raise ValueError("master resolution must be positive")
        _positive_rate(self.source_fps, "source_fps")
        _positive_rate(self.output_fps, "output_fps")
        FpsSemantics(self.fps_semantics)
        graph_hash = str(self.graph_hash).strip().lower()
        if len(graph_hash) != 64 or any(
            c not in "0123456789abcdef" for c in graph_hash
        ):
            raise ValueError("graph_hash must be a 64-character hexadecimal SHA-256")
        history: dict[str, tuple[int, int]] = {}
        for name, resolution in self.history_buffer_resolution.items():
            try:
                actual = (int(resolution[0]), int(resolution[1]))
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"invalid history resolution for {name!r}") from error
            if actual != (width, height):
                raise ValueError(
                    f"history buffer {name!r} is {actual}, expected master resolution "
                    f"{(width, height)}"
                )
            history[str(name)] = actual
        providers = {
            str(node): str(provider)
            for node, provider in self.node_provider_assignment.items()
        }
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "fps_semantics", FpsSemantics(self.fps_semantics))
        object.__setattr__(self, "graph_hash", graph_hash)
        object.__setattr__(
            self,
            "history_buffer_resolution",
            MappingProxyType(dict(sorted(history.items()))),
        )
        object.__setattr__(
            self,
            "node_provider_assignment",
            MappingProxyType(dict(sorted(providers.items()))),
        )


@dataclass(frozen=True)
class OfflineMasterResult:
    telemetry: Mapping[str, Any]
    encoder: FFmpegEncodeResult


class OfflineMasterError(RuntimeError):
    pass


_RNG_SCOPE_LOCK = threading.RLock()
_OPENCV_SCOPE_LOCK = threading.RLock()


class _DeterministicRngScope:
    """Make legacy ``random``/``numpy.random`` callbacks repeatable and isolated."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed) & 0xFFFFFFFF

    def __enter__(self) -> None:
        _RNG_SCOPE_LOCK.acquire()
        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        random.seed(self._seed)
        np.random.seed(self._seed)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        np.random.set_state(self._numpy_state)
        random.setstate(self._python_state)
        _RNG_SCOPE_LOCK.release()


class _DeterministicOpenCvScope:
    """Pin process-global OpenCV scheduling for a reproducible master render."""

    def __enter__(self) -> Mapping[str, Any]:
        _OPENCV_SCOPE_LOCK.acquire()
        self._cv2 = None
        self._previous_threads = None
        self._previous_opencl = None
        try:
            import cv2

            self._cv2 = cv2
            self._previous_threads = int(cv2.getNumThreads())
            self._previous_opencl = bool(
                hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
            )
            cv2.setNumThreads(1)
            if hasattr(cv2, "ocl"):
                cv2.ocl.setUseOpenCL(False)
            self.info = {
                "available": True,
                "num_threads": 1,
                "opencl": False,
                "previous_num_threads": self._previous_threads,
                "previous_opencl": self._previous_opencl,
            }
        except Exception as error:
            self.info = {
                "available": False,
                "num_threads": None,
                "opencl": None,
                "error": str(error),
            }
        return MappingProxyType(dict(self.info))

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._cv2 is not None:
                if self._previous_threads is not None:
                    self._cv2.setNumThreads(self._previous_threads)
                if self._previous_opencl is not None and hasattr(self._cv2, "ocl"):
                    self._cv2.ocl.setUseOpenCL(self._previous_opencl)
        finally:
            _OPENCV_SCOPE_LOCK.release()


def _frame_seed(random_seed: int, graph_hash: str, source_index: int) -> int:
    payload = f"{int(random_seed)}:{graph_hash}:{source_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _validate_master_frame(
    frame: Any,
    *,
    width: int,
    height: int,
    stage: str,
) -> np.ndarray:
    array = np.asarray(frame)
    expected = (height, width, 3)
    if array.shape != expected:
        raise OfflineMasterError(
            f"{stage} frame has shape {array.shape}; master path requires {expected}"
        )
    if array.dtype != np.uint8:
        raise OfflineMasterError(
            f"{stage} frame has dtype {array.dtype}; master path requires uint8 RGB"
        )
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples:
        return {
            "samples": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    ordered = sorted(float(value) * 1000.0 for value in samples)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "samples": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
    }


class OfflineMasterRenderer:
    """Serial deterministic renderer with an isolated encoder egress branch."""

    def __init__(self, config: OfflineMasterConfig) -> None:
        self.config = config

    def render(
        self,
        source_frames: Iterable[np.ndarray],
        *,
        source_frame_count: int,
        processor: MasterFrameProcessor,
        encoder: MasterFrameSink,
    ) -> OfflineMasterResult:
        with _DeterministicOpenCvScope() as opencv_determinism:
            result = self._render_impl(
                source_frames,
                source_frame_count=source_frame_count,
                processor=processor,
                encoder=encoder,
            )
        telemetry = dict(result.telemetry)
        telemetry["opencv_determinism"] = dict(opencv_determinism)
        return OfflineMasterResult(telemetry=telemetry, encoder=result.encoder)

    def _render_impl(
        self,
        source_frames: Iterable[np.ndarray],
        *,
        source_frame_count: int,
        processor: MasterFrameProcessor,
        encoder: MasterFrameSink,
    ) -> OfflineMasterResult:
        config = self.config
        if int(source_frame_count) <= 0:
            raise OfflineMasterError(
                "offline master encode requires at least one source frame"
            )
        schedule = build_frame_schedule(
            source_frame_count,
            config.source_fps,
            config.output_fps,
            config.fps_semantics,
        )
        grouped_outputs = schedule.output_indices_by_source()
        instrumentation = OfflineInstrumentation(config.node_provider_assignment)
        render_times: list[float] = []
        encode_times: list[float] = []
        processed = 0

        try:
            for source_index, raw_frame in enumerate(source_frames):
                if source_index >= source_frame_count:
                    raise OfflineMasterError(
                        "source yielded more frames than source_frame_count"
                    )
                source = _validate_master_frame(
                    raw_frame,
                    width=config.width,
                    height=config.height,
                    stage="ingress",
                )
                seed = _frame_seed(config.random_seed, config.graph_hash, source_index)
                context = OfflineFrameContext(
                    source_index=source_index,
                    source_time_seconds=Fraction(source_index, 1) / schedule.source_fps,
                    output_indices=grouped_outputs[source_index],
                    frame_seed=seed,
                    graph_hash=config.graph_hash,
                    master_resolution=(config.width, config.height),
                    instrumentation=instrumentation,
                    numpy_rng=np.random.default_rng(seed),
                    python_rng=random.Random(seed),
                )
                started = time.perf_counter()
                with _DeterministicRngScope(seed), use_numpy_rng(context.numpy_rng):
                    processed_frame = processor(source, context)
                master = _validate_master_frame(
                    processed_frame,
                    width=config.width,
                    height=config.height,
                    stage="processor output",
                )
                render_times.append(time.perf_counter() - started)
                processed += 1

                # FPS placement is strictly an egress operation.  Temporal graph
                # state has already advanced once, even when this loop writes the
                # same immutable master result more than once.
                for _output_index in grouped_outputs[source_index]:
                    encode_started = time.perf_counter()
                    encoder.write_frame(master)
                    encode_times.append(time.perf_counter() - encode_started)

            if processed != source_frame_count:
                raise OfflineMasterError(
                    f"source yielded {processed} frames, expected {source_frame_count}"
                )
            encode_result = encoder.close()
        except BaseException:
            encoder.abort()
            raise

        if encode_result.frames_written != schedule.output_frame_count:
            raise OfflineMasterError(
                "encoder frame count does not match deterministic schedule: "
                f"{encode_result.frames_written} != {schedule.output_frame_count}"
            )

        transfers = instrumentation.transfers
        measured_bitrate = encode_result.readback.get("bit_rate")
        encoder_frame_count = max(1, encode_result.frames_written)
        encoder_timing = {
            "frames": encode_result.frames_written,
            "process_elapsed_seconds": encode_result.elapsed_seconds,
            "end_to_end_mean_ms": (
                encode_result.elapsed_seconds * 1000.0 / encoder_frame_count
            ),
            "stdin_write": _timing_summary(encode_times),
        }
        history = {
            str(name): [int(value[0]), int(value[1])]
            for name, value in sorted(config.history_buffer_resolution.items())
        }
        if not history:
            history = {"master_temporal_state": [config.width, config.height]}
        telemetry: dict[str, Any] = {
            "mode": "offline_master",
            "resolution": [config.width, config.height],
            "source_fps": float(schedule.source_fps),
            "output_fps": float(schedule.output_fps),
            "fps_semantics": schedule.semantics.value,
            "frame_count": {
                "source": source_frame_count,
                "master_processed": processed,
                "encoded": schedule.output_frame_count,
            },
            "duration": {
                "source_seconds": float(schedule.source_duration_seconds),
                "output_seconds": float(schedule.output_duration_seconds),
            },
            "node_provider_assignment": instrumentation.provider_assignments,
            "cpu_gpu_transfers": {
                "count": len(transfers),
                "bytes": sum(transfer.byte_count for transfer in transfers),
                "events": [
                    {
                        "direction": transfer.direction,
                        "bytes": transfer.byte_count,
                        "source_index": transfer.source_index,
                    }
                    for transfer in transfers
                ],
            },
            "history_buffer_resolution": history,
            # Offline scheduling never drops a frame due to lateness/backpressure.
            "dropped_frames": 0,
            "duplicated_frames": schedule.duplicated_frames,
            "source_frames_omitted_at_egress": schedule.source_frames_omitted_at_egress,
            "render_time_per_frame": _timing_summary(render_times),
            # End-to-end includes FFmpeg startup, pipe backpressure, flush, and
            # final muxing.  Stdin write latency remains separately visible.
            "encode_time_per_frame": encoder_timing,
            "bitrate": measured_bitrate,
            "graph_hash": config.graph_hash,
            "random_seed": int(config.random_seed),
            "encoder": {
                "frames_written": encode_result.frames_written,
                "bytes_written": encode_result.bytes_written,
                "returncode": encode_result.returncode,
                "readback": dict(encode_result.readback),
            },
        }
        return OfflineMasterResult(telemetry=telemetry, encoder=encode_result)


@dataclass(frozen=True)
class FFmpegMasterConfig:
    output_path: str | os.PathLike[str]
    width: int
    height: int
    fps: int | float | str | Fraction
    codec: MasterCodec = MasterCodec.H264
    crf: int = 16
    preset: str = "slow"
    output_pixel_format: str = "yuv420p"
    ffmpeg_executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"
    overwrite: bool = True
    extra_output_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("encoder dimensions must be positive")
        pixel_format = str(self.output_pixel_format).lower()
        if pixel_format in {"yuv420p", "yuvj420p", "nv12", "nv21"} and (
            int(self.width) % 2 or int(self.height) % 2
        ):
            raise ValueError(
                f"{pixel_format} requires even master dimensions; refusing "
                "implicit encoder scaling/padding"
            )
        _positive_rate(self.fps, "fps")
        MasterCodec(self.codec)
        crf = int(self.crf)
        if not 0 <= crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        if not str(self.output_path):
            raise ValueError("output_path is required")


class FFmpegEncodeError(OfflineMasterError):
    def __init__(self, message: str, result: FFmpegEncodeResult) -> None:
        super().__init__(message)
        self.result = result


def build_ffmpeg_master_command(config: FFmpegMasterConfig) -> tuple[str, ...]:
    rate = _positive_rate(config.fps, "fps")
    rate_text = f"{rate.numerator}/{rate.denominator}"
    codec = MasterCodec(config.codec)
    video_codec = "libx264" if codec is MasterCodec.H264 else "libx265"
    command = [
        resolve_ffmpeg_executable(config.ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if config.overwrite else "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{int(config.width)}x{int(config.height)}",
        "-framerate",
        rate_text,
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        video_codec,
        "-preset",
        str(config.preset),
        "-crf",
        str(int(config.crf)),
        "-pix_fmt",
        str(config.output_pixel_format),
        "-fps_mode",
        "passthrough",
    ]
    if codec is MasterCodec.H265:
        command.extend(("-tag:v", "hvc1"))
    command.extend(str(value) for value in config.extra_output_args)
    command.append(os.fspath(config.output_path))
    return tuple(command)


def probe_encoded_media(
    path: str | os.PathLike[str],
    *,
    ffprobe_executable: str = "ffprobe",
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    command = (
        resolve_ffprobe_executable(ffprobe_executable),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,nb_read_frames,bit_rate,duration:format=duration,bit_rate",
        "-of",
        "json",
        os.fspath(path),
    )
    try:
        completed = run(command, capture_output=True, check=False)
    except OSError as error:
        return {"available": False, "error": str(error), "command": list(command)}
    stdout = (
        completed.stdout.decode("utf-8", "replace")
        if isinstance(completed.stdout, bytes)
        else str(completed.stdout or "")
    )
    stderr = (
        completed.stderr.decode("utf-8", "replace")
        if isinstance(completed.stderr, bytes)
        else str(completed.stderr or "")
    )
    if int(completed.returncode) != 0:
        return {
            "available": False,
            "returncode": int(completed.returncode),
            "error": stderr[-4096:],
            "command": list(command),
        }
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        return {"available": False, "error": f"invalid ffprobe JSON: {error}"}
    stream = (payload.get("streams") or [{}])[0]
    format_data = payload.get("format") or {}
    return {
        "available": True,
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pixel_format": stream.get("pix_fmt"),
        "average_frame_rate": stream.get("avg_frame_rate"),
        "frame_count": stream.get("nb_read_frames") or stream.get("nb_frames"),
        "duration_seconds": stream.get("duration") or format_data.get("duration"),
        "bit_rate": stream.get("bit_rate") or format_data.get("bit_rate"),
    }


class FFmpegMasterEncoder:
    """Raw RGB master-frame sink backed by one direct FFmpeg stdin pipe."""

    _STDERR_LIMIT = 64 * 1024

    def __init__(
        self,
        config: FFmpegMasterConfig,
        *,
        process_factory: Callable[..., Any] = subprocess.Popen,
        probe: Callable[..., Mapping[str, Any]] = probe_encoded_media,
    ) -> None:
        self.config = config
        self.command = build_ffmpeg_master_command(config)
        self._process_factory = process_factory
        self._probe = probe
        self._process: Any | None = None
        self._stderr_chunks: list[bytes] = []
        self._stderr_bytes = 0
        self._stderr_thread: threading.Thread | None = None
        self._frames_written = 0
        self._bytes_written = 0
        self._started_at: float | None = None
        self._result: FFmpegEncodeResult | None = None

    def _drain_stderr(self, stream: BinaryIO) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "replace")
            self._stderr_bytes += len(chunk)
            self._stderr_chunks.append(bytes(chunk))
            joined_size = sum(len(value) for value in self._stderr_chunks)
            while self._stderr_chunks and joined_size > self._STDERR_LIMIT:
                removed = self._stderr_chunks.pop(0)
                joined_size -= len(removed)

    def start(self) -> None:
        if self._process is not None:
            return
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        try:
            process = self._process_factory(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            empty = FFmpegEncodeResult(
                self.command, 0, 0, -1, str(error), 0.0, {"available": False}
            )
            raise FFmpegEncodeError(
                f"could not start FFmpeg: {error}", empty
            ) from error
        if process.stdin is None or process.stderr is None:
            process.terminate()
            raise OfflineMasterError("FFmpeg process did not expose stdin/stderr pipes")
        self._process = process
        self._started_at = started_at
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="megaglitch-ffmpeg-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def write_frame(self, frame: np.ndarray) -> None:
        if self._result is not None:
            raise OfflineMasterError("cannot write after encoder close")
        self.start()
        array = _validate_master_frame(
            frame,
            width=int(self.config.width),
            height=int(self.config.height),
            stage="encoder egress",
        )
        payload = memoryview(array).cast("B")
        assert self._process is not None and self._process.stdin is not None
        try:
            written = self._process.stdin.write(payload)
        except (BrokenPipeError, OSError) as error:
            self.abort()
            raise OfflineMasterError(f"FFmpeg stdin failed: {error}") from error
        expected = payload.nbytes
        if written is not None and int(written) != expected:
            self.abort()
            raise OfflineMasterError(
                f"FFmpeg stdin accepted {written} of {expected} frame bytes"
            )
        self._frames_written += 1
        self._bytes_written += expected

    def _stderr_text(self) -> str:
        return b"".join(self._stderr_chunks).decode("utf-8", "replace")

    def close(self) -> FFmpegEncodeResult:
        if self._result is not None:
            return self._result
        self.start()
        assert self._process is not None
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        returncode = int(self._process.wait())
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=10.0)
        elapsed = time.perf_counter() - (self._started_at or time.perf_counter())
        stderr = self._stderr_text()
        readback: Mapping[str, Any] = {"available": False}
        if returncode == 0:
            output_path = Path(self.config.output_path)
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                result = FFmpegEncodeResult(
                    command=self.command,
                    frames_written=self._frames_written,
                    bytes_written=self._bytes_written,
                    returncode=returncode,
                    stderr=stderr,
                    elapsed_seconds=elapsed,
                    readback={
                        "available": False,
                        "error": "output file is missing or empty",
                    },
                )
                self._result = result
                raise FFmpegEncodeError(
                    "FFmpeg exited successfully but produced no non-empty output file",
                    result,
                )
            readback = self._probe(
                self.config.output_path,
                ffprobe_executable=self.config.ffprobe_executable,
            )
        result = FFmpegEncodeResult(
            command=self.command,
            frames_written=self._frames_written,
            bytes_written=self._bytes_written,
            returncode=returncode,
            stderr=stderr,
            elapsed_seconds=elapsed,
            readback=dict(readback),
        )
        self._result = result
        if returncode != 0:
            raise FFmpegEncodeError(
                f"FFmpeg exited with status {returncode}: {stderr[-4096:]}", result
            )
        if readback.get("available"):
            mismatches: list[str] = []
            frame_count = readback.get("frame_count")
            try:
                if (
                    frame_count not in (None, "", "N/A")
                    and int(frame_count) != self._frames_written
                ):
                    mismatches.append(
                        f"frame count {frame_count} != submitted {self._frames_written}"
                    )
            except (TypeError, ValueError):
                mismatches.append(f"invalid frame count {frame_count!r}")
            for key, expected in (
                ("width", int(self.config.width)),
                ("height", int(self.config.height)),
            ):
                actual = readback.get(key)
                try:
                    if actual not in (None, "", "N/A") and int(actual) != expected:
                        mismatches.append(f"{key} {actual} != submitted {expected}")
                except (TypeError, ValueError):
                    mismatches.append(f"invalid {key} {actual!r}")
            if mismatches:
                raise FFmpegEncodeError(
                    "FFmpeg readback does not match submitted master frames: "
                    + "; ".join(mismatches),
                    result,
                )
        return result

    def abort(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5.0)

    def __enter__(self) -> "FFmpegMasterEncoder":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def adapt_existing_frame_callback(
    callback: Callable[[np.ndarray], np.ndarray],
) -> MasterFrameProcessor:
    """Adapt a current one-argument MegaGlitch frame callback without cloning effects."""

    def process(frame: np.ndarray, _context: OfflineFrameContext) -> np.ndarray:
        return callback(frame)

    return process
