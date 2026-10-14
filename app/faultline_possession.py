from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List

import cv2
import numpy as np


@dataclass
class Territory:
    x: float
    y: float
    vx: float
    vy: float
    style: int
    aggressiveness: float
    decay: float
    temporal_rate: float
    block_size: int
    strength: float
    hue_shift: int
    invert_prob: float
    stutter: float


class FaultlinePossessionEngine:
    """
    Hostile, evolving glitch system with:
      - persistent territorial segmentation
      - drifting seeds / moving region ownership
      - faultline rupture masks
      - memory scars + stale history
      - abrupt takeover / possession events
      - territory-specific glitch behaviors
    """

    STYLES = {
        0: "displace",
        1: "blocks",
        2: "channel_bleed",
        3: "temporal_bite",
        4: "edge_burn",
        5: "scan_rupture",
    }

    def __init__(
        self,
        width: int,
        height: int,
        n_territories: int = 7,
        mutation_rate: float = 0.018,
        history_offset: float = 4.0,
        memory_decay: float = 0.93,
        event_hazard: float = 0.05,
        clock_quantization: float = 0.0,
        sync_desync: float = 0.0,
        phase_reset_chance: float = 0.01,
        local_time_warp: float = 0.0,
        rupture_threshold: float = 0.5,
        hysteresis_speed: float = 0.08,
        modulator_coupling: float = 0.0,
        scar_decay: float = 0.985,
        takeover_rate: float = 0.04,
        rupture_gain: float = 1.4,
        reset_probability: float = 0.0015,
        max_displacement: float = 18.0,
        seed: int | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.n_territories = int(max(3, n_territories))

        self.mutation_rate = float(mutation_rate)
        self.history_offset = float(np.clip(history_offset, 0.0, 64.0))
        self.memory_decay = float(np.clip(memory_decay, 0.0, 0.9999))
        self.event_hazard = float(np.clip(event_hazard, 0.0, 1.0))
        self.clock_quantization = float(np.clip(clock_quantization, 0.0, 1.0))
        self.sync_desync = float(np.clip(sync_desync, -1.0, 1.0))
        self.phase_reset_chance = float(np.clip(phase_reset_chance, 0.0, 1.0))
        self.local_time_warp = float(np.clip(local_time_warp, -3.0, 3.0))
        self.rupture_threshold = float(max(0.01, rupture_threshold))
        self.hysteresis_speed = float(np.clip(hysteresis_speed, 0.0, 1.0))
        self.modulator_coupling = float(np.clip(modulator_coupling, 0.0, 1.0))
        self.scar_decay = float(np.clip(scar_decay, 0.0, 0.99999))
        self.takeover_rate = float(np.clip(takeover_rate, 0.0, 1.0))
        self.rupture_gain = float(max(0.0, rupture_gain))
        self.reset_probability = float(np.clip(reset_probability, 0.0, 1.0))
        self.max_displacement = float(max(1.0, max_displacement))

        self.rng = np.random.default_rng(seed)

        self.frame_index = 0
        self.global_phase = 0.0
        self.climate = 0.5
        self.previous_gray: np.ndarray | None = None
        self.feedback = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.short_history = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.scar_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.rupture_heat = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_output = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.history_buffer: List[np.ndarray] = []
        self.history_capacity = 40

        yy, xx = np.mgrid[0:self.height, 0:self.width]
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)

        self.territories: List[Territory] = [self._random_territory() for _ in range(self.n_territories)]
        self.territory_phase_offsets = self.rng.normal(0.0, 0.35, size=self.n_territories).astype(np.float32)
        self.territory_time_scale = np.ones(self.n_territories, dtype=np.float32)
        self.territory_stress = self.rng.uniform(0.0, 0.15, size=self.n_territories).astype(np.float32)
        self.owner_map = np.zeros((self.height, self.width), dtype=np.int32)
        self.boundary_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.distortion_field_x = np.zeros((self.height, self.width), dtype=np.float32)
        self.distortion_field_y = np.zeros((self.height, self.width), dtype=np.float32)
        self._recompute_maps(force=True)

    def configure(
        self,
        *,
        n_territories: int | None = None,
        mutation_rate: float | None = None,
        history_offset: float | None = None,
        memory_decay: float | None = None,
        event_hazard: float | None = None,
        clock_quantization: float | None = None,
        sync_desync: float | None = None,
        phase_reset_chance: float | None = None,
        local_time_warp: float | None = None,
        rupture_threshold: float | None = None,
        hysteresis_speed: float | None = None,
        modulator_coupling: float | None = None,
        scar_decay: float | None = None,
        takeover_rate: float | None = None,
        rupture_gain: float | None = None,
        reset_probability: float | None = None,
        max_displacement: float | None = None,
    ) -> None:
        if mutation_rate is not None:
            self.mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))
        if history_offset is not None:
            self.history_offset = float(np.clip(history_offset, 0.0, 64.0))
        if memory_decay is not None:
            self.memory_decay = float(np.clip(memory_decay, 0.0, 0.9999))
        if event_hazard is not None:
            self.event_hazard = float(np.clip(event_hazard, 0.0, 1.0))
        if clock_quantization is not None:
            self.clock_quantization = float(np.clip(clock_quantization, 0.0, 1.0))
        if sync_desync is not None:
            self.sync_desync = float(np.clip(sync_desync, -1.0, 1.0))
        if phase_reset_chance is not None:
            self.phase_reset_chance = float(np.clip(phase_reset_chance, 0.0, 1.0))
        if local_time_warp is not None:
            self.local_time_warp = float(np.clip(local_time_warp, -3.0, 3.0))
        if rupture_threshold is not None:
            self.rupture_threshold = float(max(0.01, rupture_threshold))
        if hysteresis_speed is not None:
            self.hysteresis_speed = float(np.clip(hysteresis_speed, 0.0, 1.0))
        if modulator_coupling is not None:
            self.modulator_coupling = float(np.clip(modulator_coupling, 0.0, 1.0))
        if scar_decay is not None:
            self.scar_decay = float(np.clip(scar_decay, 0.0, 0.99999))
        if takeover_rate is not None:
            self.takeover_rate = float(np.clip(takeover_rate, 0.0, 1.0))
        if rupture_gain is not None:
            self.rupture_gain = float(max(0.0, rupture_gain))
        if reset_probability is not None:
            self.reset_probability = float(np.clip(reset_probability, 0.0, 1.0))
        if max_displacement is not None:
            self.max_displacement = float(max(1.0, max_displacement))

        if n_territories is not None:
            target = int(max(3, n_territories))
            if target != self.n_territories:
                self.n_territories = target
                while len(self.territories) < self.n_territories:
                    self.territories.append(self._random_territory())
                if len(self.territories) > self.n_territories:
                    self.territories = self.territories[:self.n_territories]
                self._resize_temporal_state()
                self._recompute_maps(force=True)

    def _resize_temporal_state(self) -> None:
        target = int(self.n_territories)
        current = int(getattr(self.territory_phase_offsets, "shape", [0])[0]) if hasattr(self, "territory_phase_offsets") else 0
        if current == target:
            return

        def _resize(arr, fill_fn):
            if current == 0:
                return fill_fn(target)
            if target < current:
                return arr[:target].copy()
            extra = fill_fn(target - current)
            return np.concatenate([arr, extra], axis=0)

        self.territory_phase_offsets = _resize(
            getattr(self, "territory_phase_offsets", np.empty((0,), dtype=np.float32)),
            lambda count: self.rng.normal(0.0, 0.35, size=count).astype(np.float32),
        )
        self.territory_time_scale = _resize(
            getattr(self, "territory_time_scale", np.empty((0,), dtype=np.float32)),
            lambda count: np.ones(count, dtype=np.float32),
        )
        self.territory_stress = _resize(
            getattr(self, "territory_stress", np.empty((0,), dtype=np.float32)),
            lambda count: self.rng.uniform(0.0, 0.15, size=count).astype(np.float32),
        )

    def _sample_history(self, offset: float) -> np.ndarray:
        if not self.history_buffer:
            return self.feedback.copy()
        max_depth = max(0.0, float(len(self.history_buffer) - 1))
        depth = float(np.clip(offset, 0.0, max_depth))
        low = int(math.floor(depth))
        high = min(len(self.history_buffer) - 1, low + 1)
        frac = depth - low
        newer = self.history_buffer[-1 - low]
        older = self.history_buffer[-1 - high]
        return newer * (1.0 - frac) + older * frac

    def _territory_phase(self, idx: int, t: Territory, local_drive: float = 0.0) -> float:
        coupling = self.modulator_coupling
        warp_signal = (
            0.55 * math.sin(self.global_phase * (0.45 + t.temporal_rate) + idx * 0.73) +
            0.45 * math.cos(self.global_phase * (0.18 + t.temporal_rate * 0.7) - idx * 0.29)
        )
        desired_scale = 1.0 + self.local_time_warp * (warp_signal + local_drive * (0.6 + 0.8 * coupling))
        desired_scale *= 1.0 + coupling * (self.climate - 0.5) * 0.35
        desired_scale = float(np.clip(desired_scale, 0.08, 3.2))

        follow = 0.01 + 0.45 * self.hysteresis_speed
        self.territory_time_scale[idx] += (desired_scale - self.territory_time_scale[idx]) * follow

        if self.sync_desync >= 0.0:
            self.territory_phase_offsets[idx] *= max(0.0, 1.0 - (0.02 + 0.18 * self.sync_desync) * follow)
        else:
            self.territory_phase_offsets[idx] += float(
                self.rng.normal(0.0, (0.01 + abs(self.sync_desync) * 0.08) * (1.0 + t.aggressiveness))
            )

        self.territory_phase_offsets[idx] += self.territory_stress[idx] * (0.002 + 0.006 * coupling)
        local_phase = self.global_phase * float(self.territory_time_scale[idx]) + float(self.territory_phase_offsets[idx])

        if self.clock_quantization > 1e-6:
            step = max(0.015, self.clock_quantization * math.tau)
            local_phase = round(local_phase / step) * step
        return float(local_phase)

    def _random_territory(self) -> Territory:
        return Territory(
            x=float(self.rng.uniform(0, self.width)),
            y=float(self.rng.uniform(0, self.height)),
            vx=float(self.rng.uniform(-1.7, 1.7)),
            vy=float(self.rng.uniform(-1.7, 1.7)),
            style=int(self.rng.integers(0, len(self.STYLES))),
            aggressiveness=float(self.rng.uniform(0.25, 1.0)),
            decay=float(self.rng.uniform(0.82, 0.985)),
            temporal_rate=float(self.rng.uniform(0.25, 2.2)),
            block_size=int(self.rng.integers(6, 32)),
            strength=float(self.rng.uniform(0.2, 1.0)),
            hue_shift=int(self.rng.integers(-32, 33)),
            invert_prob=float(self.rng.uniform(0.0, 0.18)),
            stutter=float(self.rng.uniform(0.0, 0.35)),
        )

    def _mutate_territory(self, t: Territory, hard: bool = False) -> None:
        amp = 1.0 if hard else 0.35
        if self.rng.random() < (0.8 if hard else 0.25):
            t.style = int(self.rng.integers(0, len(self.STYLES)))
        t.aggressiveness = float(np.clip(t.aggressiveness + self.rng.normal(0, 0.18 * amp), 0.05, 1.4))
        t.decay = float(np.clip(t.decay + self.rng.normal(0, 0.04 * amp), 0.65, 0.995))
        t.temporal_rate = float(np.clip(t.temporal_rate + self.rng.normal(0, 0.45 * amp), 0.05, 4.0))
        t.block_size = int(np.clip(round(t.block_size + self.rng.normal(0, 7 * amp)), 4, 72))
        t.strength = float(np.clip(t.strength + self.rng.normal(0, 0.2 * amp), 0.05, 1.6))
        t.hue_shift = int(np.clip(round(t.hue_shift + self.rng.normal(0, 14 * amp)), -90, 90))
        t.invert_prob = float(np.clip(t.invert_prob + self.rng.normal(0, 0.06 * amp), 0.0, 0.55))
        t.stutter = float(np.clip(t.stutter + self.rng.normal(0, 0.08 * amp), 0.0, 0.95))
        t.vx = float(np.clip(t.vx + self.rng.normal(0, 0.35 * amp), -3.5, 3.5))
        t.vy = float(np.clip(t.vy + self.rng.normal(0, 0.35 * amp), -3.5, 3.5))

    def _step_territories(self) -> None:
        self.global_phase += 0.013 + 0.02 * self.climate
        for idx, t in enumerate(self.territories):
            jitter = 0.12 + 0.35 * t.aggressiveness
            t.vx = float(np.clip(t.vx + self.rng.normal(0, jitter * 0.06), -3.5, 3.5))
            t.vy = float(np.clip(t.vy + self.rng.normal(0, jitter * 0.06), -3.5, 3.5))

            local_phase = self._territory_phase(idx, t, local_drive=float(self.territory_stress[idx]))
            drift_gain = 0.25 + self.climate
            t.x += t.vx + math.sin(local_phase * t.temporal_rate + t.y * 0.003) * drift_gain
            t.y += t.vy + math.cos(local_phase * (0.9 + t.temporal_rate) + t.x * 0.0025) * drift_gain

            if t.x < 0:
                t.x = 0
                t.vx *= -0.8
            elif t.x > self.width - 1:
                t.x = self.width - 1
                t.vx *= -0.8

            if t.y < 0:
                t.y = 0
                t.vy *= -0.8
            elif t.y > self.height - 1:
                t.y = self.height - 1
                t.vy *= -0.8

            hazard = self.event_hazard * (0.35 + 0.65 * self.climate) * (1.0 + self.modulator_coupling * self.territory_stress[idx])
            if self.rng.random() < self.phase_reset_chance * (0.25 + self.climate) * (1.0 + hazard):
                self.territory_phase_offsets[idx] = float(self.rng.normal(0.0, math.pi))
                self.territory_time_scale[idx] = float(self.rng.uniform(0.2, 1.9))
                if self.rng.random() < 0.45:
                    self._mutate_territory(t, hard=True)

            if self.rng.random() < self.mutation_rate * (0.35 + t.aggressiveness) * (1.0 + hazard):
                self._mutate_territory(t, hard=False)

    def _recompute_maps(self, force: bool = False) -> None:
        seeds = np.array([(t.x, t.y) for t in self.territories], dtype=np.float32)
        dx = self.xx[None, :, :] - seeds[:, 0][:, None, None]
        dy = self.yy[None, :, :] - seeds[:, 1][:, None, None]
        dist2 = dx * dx + dy * dy

        owner = np.argmin(dist2, axis=0).astype(np.int32)
        smallest = np.take_along_axis(dist2, owner[None, :, :], axis=0)[0]

        part = np.partition(dist2, 1, axis=0)
        second = part[1]

        diff = np.abs(second - smallest)
        boundary = np.exp(-diff / (1200.0 + 1600.0 * self.climate)).astype(np.float32)
        boundary = cv2.GaussianBlur(boundary, (0, 0), 1.6)
        boundary = np.clip(boundary * self.rupture_gain, 0.0, 1.0)

        owner_norm = owner.astype(np.float32) / max(1.0, (self.n_territories - 1))
        fx = cv2.GaussianBlur(owner_norm, (0, 0), 9.0)
        fy = cv2.GaussianBlur(np.flipud(owner_norm), (0, 0), 9.0)
        gradx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
        grady = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)

        self.owner_map = owner
        self.boundary_mask = boundary
        self.distortion_field_x = gradx
        self.distortion_field_y = grady

        if force:
            self.rupture_heat = np.maximum(self.rupture_heat, boundary)

    def _takeover_event(self) -> None:
        predator_idx = int(self.rng.integers(0, self.n_territories))
        prey_idx = int(self.rng.integers(0, self.n_territories - 1))
        if prey_idx >= predator_idx:
            prey_idx += 1

        predator = self.territories[predator_idx]
        prey = self.territories[prey_idx]

        prey.style = predator.style if self.rng.random() < 0.8 else prey.style
        prey.aggressiveness = float(np.clip(0.7 * predator.aggressiveness + 0.3 * prey.aggressiveness + self.rng.normal(0, 0.08), 0.05, 1.5))
        prey.temporal_rate = float(np.clip(0.65 * predator.temporal_rate + 0.35 * prey.temporal_rate + self.rng.normal(0, 0.12), 0.05, 4.0))
        prey.block_size = int(np.clip(round(0.6 * predator.block_size + 0.4 * prey.block_size + self.rng.normal(0, 3)), 4, 72))
        prey.strength = float(np.clip(0.6 * predator.strength + 0.4 * prey.strength + self.rng.normal(0, 0.08), 0.05, 1.6))
        prey.hue_shift = int(np.clip(round(0.5 * predator.hue_shift + 0.5 * prey.hue_shift + self.rng.normal(0, 8)), -90, 90))
        prey.x += self.rng.normal(0, self.width * 0.08)
        prey.y += self.rng.normal(0, self.height * 0.08)
        prey.x = float(np.clip(prey.x, 0, self.width - 1))
        prey.y = float(np.clip(prey.y, 0, self.height - 1))
        self._mutate_territory(prey, hard=True)

        local_burst = np.zeros_like(self.scar_map)
        cv2.circle(
            local_burst,
            (int(prey.x), int(prey.y)),
            int(self.rng.integers(max(12, min(self.width, self.height) // 18), max(24, min(self.width, self.height) // 6))),
            1.0,
            -1,
        )
        local_burst = cv2.GaussianBlur(local_burst, (0, 0), 13.0)
        self.scar_map = np.maximum(self.scar_map, local_burst * (0.45 + 0.55 * predator.aggressiveness))
        self.rupture_heat = np.maximum(self.rupture_heat, local_burst)

    def _catastrophic_reset(self) -> None:
        dead_zone = np.zeros_like(self.scar_map)
        for _ in range(int(self.rng.integers(1, 4))):
            x1 = int(self.rng.integers(0, self.width))
            y1 = int(self.rng.integers(0, self.height))
            x2 = int(np.clip(x1 + self.rng.normal(0, self.width * 0.2), 0, self.width - 1))
            y2 = int(np.clip(y1 + self.rng.normal(0, self.height * 0.2), 0, self.height - 1))
            thickness = int(self.rng.integers(10, max(18, min(self.width, self.height) // 10)))
            cv2.line(dead_zone, (x1, y1), (x2, y2), 1.0, thickness)

        dead_zone = cv2.GaussianBlur(dead_zone, (0, 0), 9.0)
        self.scar_map = np.maximum(self.scar_map, dead_zone)
        self.feedback = self.feedback[:, :, ::-1]
        self.short_history = cv2.GaussianBlur(self.short_history, (0, 0), 7.0)
        if self.history_buffer:
            self.history_buffer = [cv2.GaussianBlur(frame, (0, 0), 5.0)[:, :, ::-1] for frame in self.history_buffer[-self.history_capacity:]]
        for t in self.territories:
            self._mutate_territory(t, hard=True)
        self.territory_phase_offsets = self.rng.normal(0.0, math.pi * 0.5, size=self.n_territories).astype(np.float32)
        self.territory_time_scale = self.rng.uniform(0.3, 1.8, size=self.n_territories).astype(np.float32)
        self.territory_stress *= 0.2
        self.rupture_heat = np.maximum(self.rupture_heat, dead_zone)

    def _motion_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            motion = np.zeros_like(gray, dtype=np.float32)
        else:
            diff = cv2.absdiff(gray, self.previous_gray)
            motion = cv2.GaussianBlur(diff.astype(np.float32) / 255.0, (0, 0), 1.1)
        self.previous_gray = gray
        return motion

    def _edge_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 180)
        return cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 1.0)

    def _style_displace(
        self,
        src: np.ndarray,
        mask: np.ndarray,
        motion: np.ndarray,
        rupture: np.ndarray,
        t: Territory,
        phase: float,
    ) -> np.ndarray:
        amp = self.max_displacement * (0.3 + 0.7 * t.strength)
        dx = (self.distortion_field_x * 45.0 + np.sin(self.yy * 0.022 + phase) * 3.0 + motion * 28.0) * amp / 20.0
        dy = (self.distortion_field_y * 45.0 + np.cos(self.xx * 0.018 - phase * 1.2) * 3.0 + rupture * 18.0) * amp / 20.0
        mapx = np.clip(self.xx + dx * (0.15 + 0.85 * mask), 0, self.width - 1).astype(np.float32)
        mapy = np.clip(self.yy + dy * (0.15 + 0.85 * mask), 0, self.height - 1).astype(np.float32)
        return cv2.remap(src, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _style_blocks(
        self,
        src: np.ndarray,
        motion: np.ndarray,
        rupture: np.ndarray,
        t: Territory,
        phase: float,
    ) -> np.ndarray:
        block = max(4, int(t.block_size))
        small_w = max(1, self.width // block)
        small_h = max(1, self.height // block)
        tmp = cv2.resize(src, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        if t.hue_shift != 0:
            hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = np.mod(hsv[:, :, 0].astype(np.int16) + t.hue_shift, 180).astype(np.uint8)
            tmp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out = cv2.resize(tmp, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        row_count = max(1, self.height // block)
        for r in range(row_count):
            chance = 0.12 + 0.45 * t.aggressiveness + 0.18 * float(np.mean(rupture))
            if self.rng.random() < chance:
                y1 = r * block
                y2 = min(self.height, y1 + block)
                shift = int(self.rng.integers(-block * 2, block * 2 + 1) + math.sin(phase + r * 0.33) * block)
                out[y1:y2] = np.roll(out[y1:y2], shift, axis=1)

        return out

    def _style_channel_bleed(
        self,
        src: np.ndarray,
        motion: np.ndarray,
        rupture: np.ndarray,
        t: Territory,
        phase: float,
    ) -> np.ndarray:
        b, g, r = cv2.split(src)
        s1 = int(round((2 + 10 * t.strength) * math.sin(phase * (0.7 + t.temporal_rate))))
        s2 = int(round((2 + 14 * t.strength) * math.cos(phase * (1.1 + t.temporal_rate))))
        r2 = np.roll(r, s1, axis=1)
        g2 = np.roll(g, -s2, axis=0)
        b2 = np.roll(b, s1 // 2 - s2 // 3, axis=1)
        merged = cv2.merge((b2, g2, r2))

        edge_boost = np.clip(rupture * 1.8 + motion * 0.8, 0.0, 1.0)
        if t.invert_prob > 0 and self.rng.random() < t.invert_prob:
            merged = 255 - merged
        merged = np.clip(merged.astype(np.float32) * (1.0 + edge_boost[..., None] * (0.15 + 0.6 * t.strength)), 0, 255).astype(np.uint8)
        return merged

    def _style_temporal_bite(
        self,
        src: np.ndarray,
        mask: np.ndarray,
        motion: np.ndarray,
        rupture: np.ndarray,
        t: Territory,
        history_frame: np.ndarray,
    ) -> np.ndarray:
        curr = src.astype(np.float32)
        hist = history_frame
        short = self.short_history

        bite = np.clip(mask * (0.25 + 0.75 * t.aggressiveness) + motion * 0.4 + rupture * 0.6, 0.0, 1.0)
        mix_a = 0.15 + 0.55 * t.strength
        mix_b = 0.18 + 0.55 * t.stutter

        out = curr * (1.0 - bite[..., None] * mix_a) + hist * (bite[..., None] * mix_a)
        out = out * (1.0 - rupture[..., None] * mix_b) + short * (rupture[..., None] * mix_b)

        if self.rng.random() < t.stutter * 0.25:
            out = np.roll(out, int(self.rng.integers(-14, 15)), axis=0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _style_edge_burn(
        self,
        src: np.ndarray,
        rupture: np.ndarray,
        t: Territory,
    ) -> np.ndarray:
        edges = cv2.Canny(src, 80, 160).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (0, 0), 1.2)
        burn = np.clip(edges * (0.35 + 1.1 * t.strength) + rupture * 0.8 + self.scar_map * 0.6, 0.0, 1.0)

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + burn * 110.0, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] - burn * (70.0 + 80.0 * t.aggressiveness), 0, 255)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + t.hue_shift * burn, 180)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        out = out.astype(np.float32)
        out[:, :, 0] += burn * 65.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def _style_scan_rupture(
        self,
        src: np.ndarray,
        mask: np.ndarray,
        motion: np.ndarray,
        rupture: np.ndarray,
        t: Territory,
        phase: float,
    ) -> np.ndarray:
        out = src.copy().astype(np.float32)
        row_wave = np.sin((self.yy * (0.06 + t.temporal_rate * 0.01)) + phase) * (2.0 + 12.0 * t.strength)
        row_shift = (row_wave + rupture * 16.0 + motion * 10.0).astype(np.int32)

        for y in range(self.height):
            out[y] = np.roll(out[y], int(row_shift[y, 0]), axis=0)

        gate = ((np.sin(self.yy * 0.22 + phase * 2.0) > 0.72).astype(np.float32) * (0.25 + 0.75 * mask))
        out *= 1.0 - gate[..., None] * 0.42

        if self.rng.random() < 0.15 + 0.5 * t.aggressiveness:
            band_h = int(self.rng.integers(4, max(8, self.height // 8)))
            y = int(self.rng.integers(0, max(1, self.height - band_h)))
            out[y:y + band_h] = np.roll(out[y:y + band_h], int(self.rng.integers(-40, 41)), axis=1)

        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_style(
        self,
        idx: int,
        src: np.ndarray,
        mask: np.ndarray,
        motion: np.ndarray,
        rupture: np.ndarray,
        t: Territory,
    ) -> np.ndarray:
        phase = self._territory_phase(idx, t, local_drive=float(np.mean(mask)))
        history_frame = self._sample_history(self.history_offset + self.territory_stress[idx] * (2.0 + self.modulator_coupling * 6.0))
        if t.style == 0:
            return self._style_displace(src, mask, motion, rupture, t, phase)
        if t.style == 1:
            return self._style_blocks(src, motion, rupture, t, phase)
        if t.style == 2:
            return self._style_channel_bleed(src, motion, rupture, t, phase)
        if t.style == 3:
            return self._style_temporal_bite(src, mask, motion, rupture, t, history_frame)
        if t.style == 4:
            return self._style_edge_burn(src, rupture, t)
        return self._style_scan_rupture(src, mask, motion, rupture, t, phase)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected BGR frame with shape (H, W, 3)")
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError(
                f"Frame size {frame_bgr.shape[1]}x{frame_bgr.shape[0]} does not match engine size "
                f"{self.width}x{self.height}"
            )
        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)

        self.frame_index += 1
        motion = self._motion_map(frame_bgr)
        edges = self._edge_map(frame_bgr)

        motion_energy = float(np.mean(motion))
        self.climate = float(np.clip(
            0.94 * self.climate + 0.06 * (0.35 + motion_energy * 4.0 + 0.25 * (0.5 + 0.5 * math.sin(self.global_phase * 0.37))),
            0.05,
            1.6,
        ))

        self._step_territories()
        self._recompute_maps()

        takeover_hazard = (self.takeover_rate + self.event_hazard * 0.12) * (0.4 + self.climate * (1.0 + 0.35 * self.modulator_coupling))
        if self.rng.random() < takeover_hazard:
            self._takeover_event()

        catastrophic_hazard = self.reset_probability * (0.8 + self.climate * 1.8) * (1.0 + self.event_hazard * (0.8 + 0.5 * self.modulator_coupling))
        if self.rng.random() < catastrophic_hazard:
            self._catastrophic_reset()

        src = frame_bgr.copy()
        composite = src.astype(np.float32)

        self.scar_map *= self.scar_decay
        self.rupture_heat *= 0.965
        rupture = np.clip(self.boundary_mask * (0.65 + self.climate) + self.rupture_heat * 0.8 + edges * 0.35, 0.0, 1.0)
        sampled_history = self._sample_history(self.history_offset + self.modulator_coupling * self.climate * 3.0)

        for idx, t in enumerate(self.territories):
            mask = (self.owner_map == idx).astype(np.float32)
            if not np.any(mask):
                continue

            soft = cv2.GaussianBlur(mask, (0, 0), 4.0 + 8.0 * t.aggressiveness)
            local_mask = np.clip(soft + rupture * 0.22 * t.aggressiveness, 0.0, 1.0)

            local_energy = float(np.mean(local_mask * (motion * 1.4 + rupture * 1.1 + self.scar_map * 0.6)))
            stress_target = (
                local_energy * (0.8 + t.aggressiveness) +
                self.event_hazard * (0.35 + self.climate) +
                self.modulator_coupling * abs(self.local_time_warp) * 0.18
            )
            response = 0.01 + 0.45 * self.hysteresis_speed
            self.territory_stress[idx] += (stress_target - self.territory_stress[idx]) * response
            self.territory_stress[idx] = float(np.clip(self.territory_stress[idx], 0.0, 3.0))

            if self.rng.random() < 0.06 * (0.3 + t.aggressiveness) * (1.0 + self.event_hazard * 0.8):
                burst = np.zeros_like(mask)
                cv2.circle(
                    burst,
                    (int(t.x), int(t.y)),
                    int(self.rng.integers(8, max(12, min(self.width, self.height) // 9))),
                    1.0,
                    -1,
                )
                burst = cv2.GaussianBlur(burst, (0, 0), 11.0)
                self.scar_map = np.maximum(self.scar_map, burst * (0.18 + 0.55 * t.strength))
                local_mask = np.clip(local_mask + burst * 0.8, 0.0, 1.0)

            if self.territory_stress[idx] > self.rupture_threshold:
                burst = np.zeros_like(mask)
                burst_radius = int(self.rng.integers(10, max(16, min(self.width, self.height) // 7)))
                cv2.circle(burst, (int(t.x), int(t.y)), burst_radius, 1.0, -1)
                burst = cv2.GaussianBlur(burst, (0, 0), 13.0)
                self.rupture_heat = np.maximum(self.rupture_heat, burst)
                self.scar_map = np.maximum(self.scar_map, burst * (0.25 + 0.65 * t.strength))
                local_mask = np.clip(local_mask + burst * 0.9, 0.0, 1.0)
                self.territory_stress[idx] *= 0.18
                if self.rng.random() < 0.35 + self.phase_reset_chance * 0.8:
                    self.territory_phase_offsets[idx] = float(self.rng.normal(0.0, math.pi))
                    self.territory_time_scale[idx] = float(self.rng.uniform(0.25, 1.9))
                    self._mutate_territory(t, hard=True)

            styled = self._apply_style(idx, src, local_mask, motion, rupture, t).astype(np.float32)
            alpha = np.clip(local_mask * (0.25 + 0.8 * t.strength), 0.0, 1.0)[..., None]
            composite = composite * (1.0 - alpha) + styled * alpha

        scar_rgb = np.repeat(np.clip(self.scar_map * (0.3 + self.climate), 0.0, 1.0)[..., None], 3, axis=2)
        composite = composite * (1.0 - scar_rgb * 0.18) + sampled_history * (scar_rgb * (0.08 + 0.25 * self.climate))

        tear_strength = np.clip(rupture * (0.4 + self.climate), 0.0, 1.0)
        if float(np.mean(tear_strength)) > 0.06:
            shifted = np.roll(composite, int(3 + 9 * self.climate + self.event_hazard * 11.0), axis=1)
            composite = composite * (1.0 - tear_strength[..., None] * 0.2) + shifted * (tear_strength[..., None] * 0.2)

        out = np.clip(composite, 0, 255).astype(np.uint8)

        out_f = out.astype(np.float32)
        self.short_history = self.short_history * 0.72 + out_f * 0.28
        self.feedback = self.feedback * self.memory_decay + out_f * (1.0 - self.memory_decay)
        self.history_buffer.append(out_f.copy())
        if len(self.history_buffer) > self.history_capacity:
            self.history_buffer = self.history_buffer[-self.history_capacity:]

        self.scar_map = np.clip(self.scar_map + motion * 0.08 + rupture * 0.03, 0.0, 1.0)
        self.last_output = out_f
        return out
