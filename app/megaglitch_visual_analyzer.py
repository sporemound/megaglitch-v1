from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - app already depends on cv2, but keep import-safe.
    cv2 = None

from megaglitch_visual_bus import VisualBus, VisualMetrics


@dataclass
class VisualAnalyzerConfig:
    max_width: int = 160
    max_height: int = 90
    color_order: str = "RGB"


class VisualAnalyzer:
    """Fast normalized frame metrics for RGB or BGR uint8 frames."""

    def __init__(self, config: Optional[VisualAnalyzerConfig] = None):
        self.config = config or VisualAnalyzerConfig()
        self._previous_gray: Optional[np.ndarray] = None
        self._motion_history: list[float] = []

    def analyze(self, frame) -> VisualMetrics:
        if frame is None:
            return VisualMetrics()

        arr = np.asarray(frame)
        if arr.size == 0:
            return VisualMetrics()
        if arr.ndim == 2:
            rgb = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3:
            rgb = arr[:, :, :3]
        else:
            return VisualMetrics()

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = self._downsample(rgb)
        if self.config.color_order.upper() == "BGR":
            rgb = rgb[:, :, ::-1]

        rgb_f = rgb.astype(np.float32) / 255.0
        red = rgb_f[:, :, 0]
        green = rgb_f[:, :, 1]
        blue = rgb_f[:, :, 2]
        gray = (0.2126 * red + 0.7152 * green + 0.0722 * blue).astype(np.float32)

        brightness = float(np.mean(gray))
        contrast = float(np.clip(np.std(gray) * 2.5, 0.0, 1.0))
        entropy = self._entropy(gray)
        edge_density = self._edge_density(gray)
        motion = self._motion(gray)
        glitch_spike = self._glitch_spike(motion, contrast, edge_density)
        center_x, center_y = self._center_mass(gray)

        return VisualMetrics(
            brightness=brightness,
            contrast=contrast,
            motion=motion,
            entropy=entropy,
            edge_density=edge_density,
            red_energy=float(np.mean(red)),
            green_energy=float(np.mean(green)),
            blue_energy=float(np.mean(blue)),
            glitch_spike=glitch_spike,
            center_mass_x=center_x,
            center_mass_y=center_y,
        )

    def _downsample(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        if width <= self.config.max_width and height <= self.config.max_height:
            return rgb
        scale = min(self.config.max_width / max(1, width), self.config.max_height / max(1, height))
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        if cv2 is not None:
            return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
        y_idx = np.linspace(0, height - 1, new_size[1]).astype(np.int32)
        x_idx = np.linspace(0, width - 1, new_size[0]).astype(np.int32)
        return rgb[np.ix_(y_idx, x_idx)]

    def _entropy(self, gray: np.ndarray) -> float:
        hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=False)
        total = float(np.sum(hist))
        if total <= 0:
            return 0.0
        probs = hist.astype(np.float32) / total
        probs = probs[probs > 0]
        return float(np.clip(-np.sum(probs * np.log2(probs)) / 5.0, 0.0, 1.0))

    def _edge_density(self, gray: np.ndarray) -> float:
        if cv2 is not None:
            gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
            edges = cv2.Canny(gray_u8, 48, 128)
            return float(np.mean(edges > 0))
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx * gx + gy * gy)
        return float(np.clip(np.mean(mag > 0.12), 0.0, 1.0))

    def _motion(self, gray: np.ndarray) -> float:
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            self._previous_gray = gray.copy()
            return 0.0
        diff = np.abs(gray - self._previous_gray)
        self._previous_gray = gray.copy()
        return float(np.clip(np.mean(diff) * 3.0, 0.0, 1.0))

    def _glitch_spike(self, motion: float, contrast: float, edge_density: float) -> float:
        composite = float(np.clip(motion * 0.62 + contrast * 0.18 + edge_density * 0.20, 0.0, 1.0))
        self._motion_history.append(composite)
        if len(self._motion_history) > 32:
            self._motion_history.pop(0)
        baseline = float(np.mean(self._motion_history)) if self._motion_history else composite
        spread = float(np.std(self._motion_history)) if len(self._motion_history) > 1 else 0.03
        threshold = baseline + max(0.04, spread * 1.8)
        return float(np.clip((composite - threshold) / max(0.08, 1.0 - threshold), 0.0, 1.0))

    def _center_mass(self, gray: np.ndarray) -> tuple[float, float]:
        mass = gray + 1e-4
        total = float(np.sum(mass))
        if total <= 0:
            return 0.5, 0.5
        height, width = gray.shape[:2]
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
        center_x = float(np.sum(mass * xs[None, :]) / total)
        center_y = float(np.sum(mass * ys[:, None]) / total)
        return float(np.clip(center_x, 0.0, 1.0)), float(np.clip(center_y, 0.0, 1.0))


class VisualAnalyzerWorker:
    """Single-frame worker queue so analysis does not block DearPyGui rendering."""

    def __init__(
        self,
        visual_bus: VisualBus,
        analyzer: Optional[VisualAnalyzer] = None,
        on_metrics: Optional[Callable[[VisualMetrics], None]] = None,
    ):
        self.visual_bus = visual_bus
        self.analyzer = analyzer or VisualAnalyzer()
        self.on_metrics = on_metrics
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="VisualAnalyzerWorker", daemon=True)
        self._thread.start()
        logging.info("Visual analyzer worker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def submit_frame(self, frame) -> None:
        if frame is None:
            return
        try:
            arr = np.asarray(frame)
            if arr.size == 0:
                return
            frame_copy = arr[:, :, :3].copy() if arr.ndim == 3 else arr.copy()
        except Exception:
            return
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame_copy)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                metrics = self.analyzer.analyze(frame)
                self.visual_bus.update(metrics)
                if self.on_metrics:
                    self.on_metrics(metrics)
            except Exception:
                logging.debug("Visual analysis failed", exc_info=True)
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass
