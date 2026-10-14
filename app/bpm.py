"""Realtime BPM and peak detection helpers for MegaGlitch.

This module provides a lightweight, pure-Python translation of the original
MegaGlitch BPM detector so that beat information can be consumed inside the
node modulation system.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable, Iterable, Optional

import numpy as np


class BPMPeakDetector:
    """Track beat-related information from incoming audio blocks.

    The detector keeps a rolling energy history and flags a beat when the
    instantaneous RMS energy crosses a dynamic threshold. Beat intervals are
    averaged over a short history to obtain a stable BPM estimate. The class is
    thread-safe so it can be updated from the audio capture thread and queried
    from the UI thread.
    """

    def __init__(
        self,
        sample_rate: float = 44100.0,
        history_seconds: float = 6.0,
        beat_threshold: float = 1.3,
        peak_threshold: float = 2.0,
        min_interval: float = 0.2,
        max_interval: float = 2.0,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.beat_threshold = float(beat_threshold)
        self.peak_threshold = float(peak_threshold)
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)

        history_size = max(1, int(self.sample_rate * history_seconds / 1024))
        self._volume_history: deque[float] = deque(maxlen=history_size)
        self._beat_history: deque[float] = deque(maxlen=10)

        # State protected by _lock
        self._lock = threading.Lock()
        self._last_beat_time: float = 0.0
        self._last_peak_time: float = 0.0
        self._beat_interval: float = 60.0 / 120.0  # default 120 BPM
        self._bpm: float = 120.0
        self._last_energy: float = 0.0
        self._last_update_time: float = 0.0
        self._beat_event_id: int = 0
        self._peak_event_id: int = 0

        self._beat_callbacks: list[Callable[[float, float], None]] = []
        self._peak_callbacks: list[Callable[[float, float], None]] = []

    # ------------------------------------------------------------------
    # Public API
    def register_beat_callback(self, callback: Callable[[float, float], None]) -> None:
        with self._lock:
            if callback not in self._beat_callbacks:
                self._beat_callbacks.append(callback)

    def register_peak_callback(self, callback: Callable[[float, float], None]) -> None:
        with self._lock:
            if callback not in self._peak_callbacks:
                self._peak_callbacks.append(callback)

    def clear_callbacks(self) -> None:
        with self._lock:
            self._beat_callbacks.clear()
            self._peak_callbacks.clear()

    def set_bpm(self, bpm: float) -> None:
        bpm = max(1.0, float(bpm))
        with self._lock:
            self._bpm = bpm
            self._beat_interval = 60.0 / bpm
            self._beat_history.clear()
            self._last_beat_time = time.monotonic()

    # ------------------------------------------------------------------
    # Update path
    def update_from_block(
        self,
        block: np.ndarray,
        fft_spectrum: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Feed an audio block into the detector.

        Args:
            block: 1-D numpy array containing mono audio samples.
            fft_spectrum: Optional magnitude spectrum for additional context.
            sample_rate: Override the detector's sample-rate hint.
            timestamp: Explicit timestamp (seconds, monotonic). Defaults to
                ``time.monotonic()`` when omitted.
        """
        if block is None:
            return
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        sr = float(sample_rate) if sample_rate else self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(block))))
        spectral_energy = float(np.mean(np.abs(fft_spectrum))) if fft_spectrum is not None else rms
        energy = max(rms, spectral_energy)
        self._update_energy(energy, timestamp or time.monotonic())

    def update_from_energy(self, energy: float, timestamp: Optional[float] = None) -> None:
        self._update_energy(float(energy), timestamp or time.monotonic())

    # ------------------------------------------------------------------
    # Query helpers
    def get_bpm(self) -> float:
        with self._lock:
            return float(self._bpm)

    def get_normalized_bpm(self, min_bpm: float = 60.0, max_bpm: float = 200.0) -> float:
        bpm = self.get_bpm()
        if max_bpm <= min_bpm:
            return 0.0
        norm = (bpm - min_bpm) / (max_bpm - min_bpm)
        return float(min(1.0, max(0.0, norm)))

    def get_beat_phase(self, timestamp: Optional[float] = None) -> float:
        with self._lock:
            last = self._last_beat_time
            interval = self._beat_interval
        if interval <= 1e-6 or last <= 0.0:
            return 0.0
        now = timestamp or time.monotonic()
        phase = (now - last) / interval
        return float(phase % 1.0)

    def get_pulse_value(self, timestamp: Optional[float] = None) -> float:
        with self._lock:
            last = self._last_beat_time
        if last <= 0.0:
            return 0.0
        phase = self.get_beat_phase(timestamp)
        # Cosine pulse: 1 at beat, falls toward 0 over half the cycle
        return float(max(0.0, math.cos(math.pi * phase)))

    def get_last_energy(self) -> float:
        with self._lock:
            return float(self._last_energy)

    def get_beat_event_id(self) -> int:
        with self._lock:
            return int(self._beat_event_id)

    def get_peak_event_id(self) -> int:
        with self._lock:
            return int(self._peak_event_id)

    # ------------------------------------------------------------------
    # Internal helpers
    def _update_energy(self, energy: float, timestamp: float) -> None:
        beat_callbacks: Iterable[Callable[[float, float], None]] = ()
        peak_callbacks: Iterable[Callable[[float, float], None]] = ()
        with self._lock:
            self._last_update_time = timestamp
            self._last_energy = energy
            self._volume_history.append(energy)
            moving_average = float(np.mean(self._volume_history)) if self._volume_history else energy
            beat_threshold = moving_average * self.beat_threshold
            peak_threshold = moving_average * self.peak_threshold

            is_beat = energy > beat_threshold and (timestamp - self._last_beat_time) >= self.min_interval
            is_peak = energy > peak_threshold and (timestamp - self._last_peak_time) >= 0.05

            if is_beat:
                if self._last_beat_time > 0.0:
                    interval = timestamp - self._last_beat_time
                    if self.min_interval <= interval <= self.max_interval:
                        self._beat_history.append(interval)
                        self._beat_interval = sum(self._beat_history) / len(self._beat_history)
                        self._bpm = 60.0 / max(self._beat_interval, 1e-6)
                self._last_beat_time = timestamp
                self._beat_event_id += 1
                beat_callbacks = tuple(self._beat_callbacks)

            if is_peak:
                self._last_peak_time = timestamp
                self._peak_event_id += 1
                peak_callbacks = tuple(self._peak_callbacks)

        for callback in beat_callbacks:
            try:
                callback(self.get_bpm(), energy)
            except Exception:
                pass
        for callback in peak_callbacks:
            try:
                callback(energy, energy)
            except Exception:
                pass


__all__ = ["BPMPeakDetector"]
