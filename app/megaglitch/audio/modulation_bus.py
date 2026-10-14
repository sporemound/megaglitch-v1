"""Deterministic bidirectional named-value modulation routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading


@dataclass
class ModulationMapping:
    source: str
    target: str
    source_range: tuple[float, float] = (0.0, 1.0)
    target_range: tuple[float, float] = (0.0, 1.0)
    curve: str = "linear"
    smoothing_ms: float = 100.0
    slew_limit: float | None = None
    deadzone: float = 0.0
    enabled: bool = True
    blend_mode: str = "replace"
    steps: int = 2
    current: float | None = None


class ModulationBus:
    def __init__(self):
        self._values = {}
        self._mappings: list[ModulationMapping] = []
        self._manual_overrides = {}
        self._lock = threading.RLock()

    def publish(self, name: str, value: float):
        with self._lock:
            self._values[str(name)] = float(value)

    def get(self, name: str, default=0.0):
        with self._lock:
            return self._values.get(str(name), default)

    def add_mapping(self, mapping=None, **kwargs):
        if isinstance(mapping, dict):
            mapping = ModulationMapping(**mapping)
        elif mapping is None:
            mapping = ModulationMapping(**kwargs)
        with self._lock:
            self._mappings.append(mapping)
        return mapping

    def remove_mapping(self, mapping):
        with self._lock:
            if mapping in self._mappings:
                self._mappings.remove(mapping)

    def set_manual_override(self, target, value):
        with self._lock:
            self._manual_overrides[str(target)] = float(value)

    def clear_manual_override(self, target):
        with self._lock:
            self._manual_overrides.pop(str(target), None)

    @staticmethod
    def _curve(value, curve, steps):
        value = min(1.0, max(0.0, value))
        if curve in {"smoothstep", "ease_in_out"}:
            return value * value * (3.0 - 2.0 * value)
        if curve == "exponential":
            return value * value
        if curve == "logarithmic":
            return math.log1p(9.0 * value) / math.log(10.0)
        if curve == "stepped":
            count = max(2, int(steps))
            return round(value * (count - 1)) / (count - 1)
        return value

    def _map_value(self, mapping, source_value):
        src_lo, src_hi = map(float, mapping.source_range)
        dst_lo, dst_hi = map(float, mapping.target_range)
        denominator = src_hi - src_lo
        normalized = 0.0 if abs(denominator) < 1e-12 else (float(source_value) - src_lo) / denominator
        normalized = min(1.0, max(0.0, normalized))
        if normalized < max(0.0, float(mapping.deadzone)):
            normalized = 0.0
        shaped = self._curve(normalized, mapping.curve, mapping.steps)
        return dst_lo + shaped * (dst_hi - dst_lo)

    @staticmethod
    def _blend(base, value, mode):
        if base is None or mode == "replace":
            return value
        if mode == "add":
            return base + value
        if mode == "multiply":
            return base * value
        if mode == "max":
            return max(base, value)
        if mode == "min":
            return min(base, value)
        return value

    def update(self, dt: float):
        dt = max(0.0, float(dt))
        with self._lock:
            results = {}
            bounds = {}
            for mapping in self._mappings:
                if not mapping.enabled or mapping.source not in self._values:
                    continue
                target_value = self._map_value(mapping, self._values[mapping.source])
                if mapping.current is None or mapping.smoothing_ms <= 0.0:
                    current = target_value
                else:
                    tau = max(1e-6, mapping.smoothing_ms * 0.001)
                    alpha = 1.0 - math.exp(-dt / tau)
                    current = mapping.current + (target_value - mapping.current) * alpha
                if mapping.slew_limit is not None and mapping.current is not None:
                    maximum_delta = max(0.0, float(mapping.slew_limit)) * dt
                    current = min(mapping.current + maximum_delta, max(mapping.current - maximum_delta, current))
                mapping.current = current
                results[mapping.target] = self._blend(results.get(mapping.target), current, mapping.blend_mode)
                lo, hi = sorted(map(float, mapping.target_range))
                bounds[mapping.target] = (
                    min(bounds.get(mapping.target, (lo, hi))[0], lo),
                    max(bounds.get(mapping.target, (lo, hi))[1], hi),
                )

            for target, value in self._manual_overrides.items():
                results[target] = value
            for target, value in list(results.items()):
                if target in bounds and target not in self._manual_overrides:
                    lo, hi = bounds[target]
                    results[target] = min(hi, max(lo, value))
                self._values[target] = results[target]
            return dict(results)

    def inspect(self):
        with self._lock:
            return {
                "values": dict(self._values),
                "manual_overrides": dict(self._manual_overrides),
                "mappings": [asdict(mapping) for mapping in self._mappings],
            }
