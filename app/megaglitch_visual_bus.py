from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields
from typing import Dict


VISUAL_METRIC_NAMES = (
    "brightness",
    "contrast",
    "motion",
    "entropy",
    "edge_density",
    "red_energy",
    "green_energy",
    "blue_energy",
    "glitch_spike",
    "center_mass_x",
    "center_mass_y",
)


@dataclass(frozen=True)
class VisualMetrics:
    brightness: float = 0.0
    contrast: float = 0.0
    motion: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    red_energy: float = 0.0
    green_energy: float = 0.0
    blue_energy: float = 0.0
    glitch_spike: float = 0.0
    center_mass_x: float = 0.5
    center_mass_y: float = 0.5

    def as_dict(self) -> Dict[str, float]:
        return {field.name: _clamp01(getattr(self, field.name)) for field in fields(self)}


class VisualBus:
    """Thread-safe snapshot store for visual analysis metrics."""

    def __init__(self, smoothing: float = 0.28):
        self._lock = threading.Lock()
        self._metrics = VisualMetrics()
        self._timestamp = 0.0
        self._smoothing = _clamp01(smoothing)

    def update(self, metrics: VisualMetrics | Dict[str, float]) -> None:
        if isinstance(metrics, VisualMetrics):
            incoming = metrics.as_dict()
        else:
            incoming = {name: _clamp01(metrics.get(name, 0.0)) for name in VISUAL_METRIC_NAMES}

        with self._lock:
            current = self._metrics.as_dict()
            alpha = self._smoothing
            smoothed = {}
            for name in VISUAL_METRIC_NAMES:
                if name in ("center_mass_x", "center_mass_y"):
                    default = 0.5
                else:
                    default = 0.0
                prev = current.get(name, default)
                new = incoming.get(name, default)
                smoothed[name] = _clamp01(prev + (new - prev) * alpha)
            self._metrics = VisualMetrics(**smoothed)
            self._timestamp = time.monotonic()

    def get_latest(self) -> Dict[str, float]:
        with self._lock:
            return self._metrics.as_dict()

    def get_snapshot(self) -> tuple[Dict[str, float], float]:
        with self._lock:
            return self._metrics.as_dict(), self._timestamp


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0
