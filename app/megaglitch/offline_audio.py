"""Deterministic source-audio analysis for Offline Master modulation."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .external_tools import resolve_ffmpeg_executable


class OfflineAudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfflineAudioFeatures:
    fft_bands: dict[str, float]
    transient: float
    beat_state: bool


@dataclass
class OfflineAudioAnalysis:
    samples: np.ndarray
    sample_rate: int
    source_path: str = ""

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.samples, dtype=np.float32).reshape(-1)
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.all(np.isfinite(values)):
            raise ValueError("audio samples must be finite")
        self.samples = np.clip(values, -1.0, 1.0)
        self.sample_rate = int(self.sample_rate)

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.size) / float(self.sample_rate)

    def _window(self, timestamp_seconds: float, duration_seconds: float) -> np.ndarray:
        start = max(0, int(round(float(timestamp_seconds) * self.sample_rate)))
        size = max(256, int(round(float(duration_seconds) * self.sample_rate)))
        end = min(self.samples.size, start + size)
        window = self.samples[start:end]
        if window.size < size:
            window = np.pad(window, (0, size - window.size))
        return window.astype(np.float32, copy=False)

    def features_at(
        self, timestamp_seconds: float, *, frame_duration_seconds: float
    ) -> OfflineAudioFeatures:
        duration = max(float(frame_duration_seconds), 2048.0 / self.sample_rate)
        window = self._window(timestamp_seconds, duration)
        previous = self._window(max(0.0, timestamp_seconds - duration), duration)
        taper = np.hanning(window.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(window * taper)).astype(np.float32)
        frequencies = np.fft.rfftfreq(window.size, 1.0 / self.sample_rate)
        peak = float(np.max(spectrum)) if spectrum.size else 0.0
        if peak > 1e-12:
            spectrum /= peak
        definitions = (
            ("Sub", 20.0, 60.0),
            ("Bass", 60.0, 250.0),
            ("Low Mid", 250.0, 500.0),
            ("Mid", 500.0, 2000.0),
            ("High Mid", 2000.0, 4000.0),
            ("Presence", 4000.0, 6000.0),
            ("Brilliance", 6000.0, 12000.0),
            ("Air", 12000.0, min(20000.0, self.sample_rate / 2.0)),
        )
        bands = {}
        for name, low, high in definitions:
            selected = spectrum[(frequencies >= low) & (frequencies < high)]
            bands[name] = float(np.clip(np.mean(selected) if selected.size else 0.0, 0.0, 1.0))
        rms = float(np.sqrt(np.mean(window * window)))
        previous_rms = float(np.sqrt(np.mean(previous * previous)))
        transient = float(np.clip((rms - previous_rms) * 8.0, 0.0, 1.0))
        return OfflineAudioFeatures(
            fft_bands=bands,
            transient=transient,
            beat_state=bool(transient >= 0.35),
        )

    def document(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "sample_rate": self.sample_rate,
            "sample_count": int(self.samples.size),
            "duration_seconds": self.duration_seconds,
            "pcm_sha256": hashlib.sha256(self.samples.tobytes()).hexdigest(),
        }


def decode_source_audio(
    source_path: str | os.PathLike[str],
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    sample_rate: int = 48000,
    ffmpeg_executable: str = "ffmpeg",
    run: Callable[..., Any] = subprocess.run,
) -> OfflineAudioAnalysis:
    path = Path(source_path)
    if not path.is_file():
        raise OfflineAudioError(f"source audio path does not exist: {path}")
    command = [
        resolve_ffmpeg_executable(ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(start_seconds)):.9f}",
        "-i",
        os.fspath(path),
    ]
    if duration_seconds is not None:
        command.extend(("-t", f"{max(0.0, float(duration_seconds)):.9f}"))
    command.extend(
        ("-vn", "-ac", "1", "-ar", str(int(sample_rate)), "-f", "f32le", "pipe:1")
    )
    completed = run(tuple(command), capture_output=True, check=False)
    if int(completed.returncode) != 0:
        stderr = (
            completed.stderr.decode("utf-8", "replace")
            if isinstance(completed.stderr, bytes)
            else str(completed.stderr or "")
        )
        raise OfflineAudioError(f"FFmpeg source-audio decode failed: {stderr[-4096:]}")
    payload = completed.stdout or b""
    if len(payload) % np.dtype(np.float32).itemsize:
        raise OfflineAudioError("FFmpeg returned a truncated float32 audio payload")
    samples = np.frombuffer(payload, dtype="<f4").copy()
    return OfflineAudioAnalysis(samples, int(sample_rate), os.fspath(path))
