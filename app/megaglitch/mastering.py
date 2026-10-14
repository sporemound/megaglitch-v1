"""
Offline Mastering Pipeline with Direct FFmpeg Streaming & Explicit FPS Semantics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


@dataclass
class FPSPlan:
    """Exact rational representation of source and output FPS and duration schedules."""
    source_fps: Fraction
    output_fps: Fraction
    source_frames: int
    fps_semantics: str  # "true_accelerated" or "duration_preserving"
    output_frames: int = 0
    source_duration: Fraction = field(default_factory=lambda: Fraction(0, 1))
    output_duration: Fraction = field(default_factory=lambda: Fraction(0, 1))
    source_frame_indices: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.source_frames <= 0:
            raise ValueError("source_frames must be positive")
        if self.source_fps <= 0 or self.output_fps <= 0:
            raise ValueError("FPS values must be positive")

        self.source_duration = Fraction(self.source_frames, 1) / self.source_fps

        if self.fps_semantics == "true_accelerated":
            # True acceleration retains each source frame exactly once, placed on output timebase
            self.output_frames = self.source_frames
            self.output_duration = Fraction(self.output_frames, 1) / self.output_fps
            self.source_frame_indices = list(range(self.source_frames))
        elif self.fps_semantics == "duration_preserving":
            # Duration preserving maintains total duration by interpolating/repeating frames
            total_duration_sec = float(self.source_duration)
            self.output_frames = max(1, int(round(total_duration_sec * float(self.output_fps))))
            self.output_duration = Fraction(self.output_frames, 1) / self.output_fps
            indices = []
            for out_idx in range(self.output_frames):
                t_sec = float(out_idx) / float(self.output_fps)
                src_idx = min(self.source_frames - 1, int(round(t_sec * float(self.source_fps))))
                indices.append(src_idx)
            self.source_frame_indices = indices
        else:
            raise ValueError(f"Unknown fps_semantics: {self.fps_semantics}")


def create_fps_plan(
    source_frames: int,
    source_fps: Union[int, float, str, Fraction] = 30,
    output_fps: Union[int, float, str, Fraction] = 55,
    fps_semantics: str = "true_accelerated",
) -> FPSPlan:
    """Create a validated FPSPlan with exact rational arithmetic."""
    s_fps = Fraction(str(source_fps)) if not isinstance(source_fps, Fraction) else source_fps
    o_fps = Fraction(str(output_fps)) if not isinstance(output_fps, Fraction) else output_fps
    return FPSPlan(
        source_fps=s_fps,
        output_fps=o_fps,
        source_frames=int(source_frames),
        fps_semantics=str(fps_semantics),
    )


def find_ffmpeg_binary() -> str:
    """Locate the preferred ffmpeg executable."""
    env_bin = os.environ.get("MEGAGLITCH_FFMPEG_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    workspace_candidates = [
        Path(r"V:\MegaGlitch-Render-Quality-20260827\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"),
        Path(r"V:\ffmpeg\bin\ffmpeg.exe"),
    ]
    for candidate in workspace_candidates:
        if candidate.is_file():
            return str(candidate)

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    return "ffmpeg"


@dataclass
class DirectEncoderConfig:
    width: int = 1920
    height: int = 1080
    output_fps: Union[int, float, str, Fraction] = 55
    codec: str = "libx264"
    crf: int = 16  # Target CRF 14-18
    preset: str = "slow"
    pix_fmt: str = "yuv420p"
    sink: str = "memory_hash"  # "memory_hash", "file", "pipe"
    output_path: Optional[str] = None
    extra_ffmpeg_args: List[str] = field(default_factory=list)


class DirectFFmpegMasterEncoder:
    """
    High-Quality Direct Master Encoder.
    Pipes uncompressed RGB24 master-resolution frames directly to FFmpeg.
    """

    def __init__(self, config: DirectEncoderConfig):
        self.config = config
        self.width = int(config.width)
        self.height = int(config.height)
        self.output_fps = Fraction(str(config.output_fps))
        self.sink = str(config.sink)
        self.output_path = config.output_path

        self.frames_written = 0
        self.bytes_written = 0
        self.stdin_write_times: List[float] = []
        self._hasher = hashlib.sha256() if self.sink == "memory_hash" else None
        self.process: Optional[subprocess.Popen] = None
        self.start_time = 0.0
        self.end_time = 0.0
        self.returncode: Optional[int] = None

        self._init_encoder()

    def _init_encoder(self):
        self.start_time = time.perf_counter()
        if self.sink == "memory_hash" and not self.output_path:
            return

        ffmpeg_bin = find_ffmpeg_binary()
        dest = self.output_path or "output_master.mp4"
        fps_str = f"{self.output_fps.numerator}/{self.output_fps.denominator}"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", str(self.config.codec),
            "-crf", str(int(self.config.crf)),
            "-preset", str(self.config.preset),
            "-pix_fmt", str(self.config.pix_fmt),
        ]
        cmd.extend(self.config.extra_ffmpeg_args)
        cmd.append(str(dest))

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            logging.error(f"Failed to spawn FFmpeg process: {exc}")
            self.process = None

    def write_frame(self, frame: np.ndarray):
        """Write a master RGB24 frame."""
        if frame is None:
            return

        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            raise ValueError(f"Frame resolution {(w, h)} does not match encoder resolution {(self.width, self.height)}")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[..., :3]

        raw_bytes = frame.tobytes()
        t0 = time.perf_counter()

        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.write(raw_bytes)
        elif self.sink == "memory_hash":
            if self._hasher is not None:
                self._hasher.update(raw_bytes)

        t1 = time.perf_counter()
        write_ms = (t1 - t0) * 1000.0
        self.stdin_write_times.append(write_ms)

        self.frames_written += 1
        self.bytes_written += len(raw_bytes)

    def close(self) -> Dict[str, Any]:
        """Finalize the stream and return telemetry metrics."""
        self.end_time = time.perf_counter()
        elapsed_total = max(1e-6, self.end_time - self.start_time)

        output_sha256 = None
        bit_rate = None

        if self.process is not None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                stdout, stderr = self.process.communicate(timeout=60.0)
                self.returncode = self.process.returncode
                if self.returncode != 0:
                    logging.warning(f"FFmpeg returned code {self.returncode}: {stderr.decode('utf-8', errors='ignore')}")
            except Exception as exc:
                logging.error(f"Error during FFmpeg shutdown: {exc}")
                self.returncode = -1

            if self.output_path and os.path.isfile(self.output_path):
                file_size = os.path.getsize(self.output_path)
                with open(self.output_path, "rb") as f:
                    output_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
                duration_sec = float(self.frames_written) / float(self.output_fps)
                if duration_sec > 0:
                    bit_rate = int((file_size * 8) / duration_sec)
        elif self.sink == "memory_hash" and self._hasher is not None:
            output_sha256 = self._hasher.hexdigest()
            self.returncode = 0

        # Compute write statistics
        if self.stdin_write_times:
            mean_ms = float(np.mean(self.stdin_write_times))
            p50_ms = float(np.percentile(self.stdin_write_times, 50))
            p95_ms = float(np.percentile(self.stdin_write_times, 95))
            p99_ms = float(np.percentile(self.stdin_write_times, 99))
            max_ms = float(np.max(self.stdin_write_times))
        else:
            mean_ms = p50_ms = p95_ms = p99_ms = max_ms = 0.0

        return {
            "frames": self.frames_written,
            "bytes_written": self.bytes_written,
            "process_elapsed_seconds": elapsed_total,
            "end_to_end_mean_ms": (elapsed_total / max(1, self.frames_written)) * 1000.0,
            "returncode": self.returncode,
            "output_sha256": output_sha256,
            "bit_rate": bit_rate,
            "stdin_write": {
                "samples": len(self.stdin_write_times),
                "mean_ms": mean_ms,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "max_ms": max_ms,
            }
        }
