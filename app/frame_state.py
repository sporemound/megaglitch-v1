from dataclasses import dataclass, field
from typing import List, Optional, Any
from collections import deque
import numpy as np


@dataclass
class FrameState:
    """
    Holds per-stream temporal state for effects.
    OPTIMIZED: All frames stored as numpy arrays (H,W,3 uint8) to avoid PIL conversions.
    """
    # All stored as numpy arrays (H, W, 3) dtype=uint8 for zero-copy operations
    previous_frame: Optional[np.ndarray] = None      # For P-frame effects
    feedback_image: Optional[np.ndarray] = None      # For feedback blend
    trail_image: Optional[np.ndarray] = None         # For trailing/persistence
    frame_buffer: List[np.ndarray] = field(default_factory=list)  # Temporal effects
    
    # Slit-scan effect state
    slit_scan_buffer: Optional[Any] = None
    slit_scan_dir: Optional[str] = None

    # Stateful generator memory keyed by node id
    generator_states: dict = field(default_factory=dict)
    effect_states: dict = field(default_factory=dict)
    
    # Buffer limit (can be overridden by nodes)
    buffer_limit: int = 10
    
    # Frame counter for consistent temporal effects (increments per frame, not wall-clock)
    frame_count: int = 0
    
    # Preallocated scratch buffers for avoiding allocations
    _scratch_float: Optional[np.ndarray] = None
    _scratch_uint8: Optional[np.ndarray] = None
    _last_shape: Optional[tuple] = None
    temporal_history: Optional[Any] = None
    # Resolution contract for every image-domain temporal buffer owned by this
    # stream.  A change is an explicit state reset, never an implicit resize.
    master_shape: Optional[tuple] = None
    master_generation: int = 0
    resolution_change_count: int = 0

    def reset(self):
        """Reset all state."""
        seen_resources = set()

        def release_resource(value):
            if value is None or id(value) in seen_resources:
                return
            if isinstance(value, dict):
                for nested in value.values():
                    release_resource(nested)
                return
            if isinstance(value, (list, tuple, set, deque)):
                for nested in value:
                    release_resource(nested)
                return
            for method_name in ("destroy", "dispose", "release"):
                method = getattr(value, method_name, None)
                if callable(method):
                    seen_resources.add(id(value))
                    try:
                        method()
                    except Exception:
                        pass
                    return

        for state_map in (self.generator_states, self.effect_states):
            for value in state_map.values():
                release_resource(value)
        self.previous_frame = None
        self.feedback_image = None
        self.trail_image = None
        self.frame_buffer.clear()
        self.slit_scan_buffer = None
        self.slit_scan_dir = None
        self.generator_states.clear()
        self.effect_states.clear()
        self.buffer_limit = 10
        self.frame_count = 0
        self._scratch_float = None
        self._scratch_uint8 = None
        self._last_shape = None
        self.master_shape = None
        self.master_generation = 0
        self.resolution_change_count = 0
        if self.temporal_history is not None:
            try:
                self.temporal_history.shutdown()
            finally:
                self.temporal_history = None

    @staticmethod
    def _normalize_master_shape(shape: tuple) -> tuple:
        normalized = tuple(int(value) for value in shape)
        if len(normalized) != 3 or normalized[0] < 1 or normalized[1] < 1:
            raise ValueError(f"invalid master frame shape: {shape!r}")
        if normalized[2] != 3:
            raise ValueError(f"master frame must have exactly 3 RGB channels: {shape!r}")
        return normalized

    def bind_master_shape(self, shape: tuple, *, allow_resolution_change: bool = False) -> bool:
        """Bind all temporal state to one master resolution.

        Returns ``True`` when an already-bound stream changed resolution.  Such
        a change clears temporal history and effect state so no preview-sized or
        stale-resolution frame can be reconciled by scaling.
        """
        normalized = self._normalize_master_shape(shape)
        changed = self.master_shape is not None and self.master_shape != normalized
        if changed and not allow_resolution_change:
            raise ValueError(
                f"master resolution changed from {self.master_shape} to {normalized}"
            )

        if changed:
            next_generation = int(self.master_generation) + 1
            next_change_count = int(self.resolution_change_count) + 1
            self.reset()
            self.master_generation = next_generation
            self.resolution_change_count = next_change_count

        if self.master_shape is None:
            self.master_shape = normalized

        for name in ("previous_frame", "feedback_image", "trail_image"):
            value = getattr(self, name)
            if value is None:
                setattr(self, name, np.zeros(normalized, dtype=np.uint8))
                continue
            if not isinstance(value, np.ndarray) or value.shape != normalized:
                raise ValueError(
                    f"temporal buffer {name} has shape {getattr(value, 'shape', None)}; "
                    f"expected master shape {normalized}"
                )

        invalid_history = [
            getattr(frame, "shape", None)
            for frame in self.frame_buffer
            if not isinstance(frame, np.ndarray) or frame.shape != normalized
        ]
        if invalid_history:
            raise ValueError(
                f"frame history contains non-master shapes {invalid_history}; "
                f"expected {normalized}"
            )
        return changed

    def assert_master_integrity(self) -> None:
        """Fail fast if any image-domain temporal buffer escaped the master contract."""
        if self.master_shape is None:
            raise ValueError("master resolution has not been bound")
        normalized = self._normalize_master_shape(self.master_shape)
        for name in ("previous_frame", "feedback_image", "trail_image"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != normalized:
                raise ValueError(
                    f"temporal buffer {name} violates master shape {normalized}: "
                    f"{getattr(value, 'shape', None)}"
                )
        for index, value in enumerate(self.frame_buffer):
            if not isinstance(value, np.ndarray) or value.shape != normalized:
                raise ValueError(
                    f"frame_buffer[{index}] violates master shape {normalized}: "
                    f"{getattr(value, 'shape', None)}"
                )
        inventory = self.temporal_buffer_inventory()
        violations = {
            path: record
            for path, record in inventory.items()
            if not record.get("master_resolution", False)
        }
        if violations:
            details = ", ".join(
                f"{path}={record.get('shape')}"
                for path, record in sorted(violations.items())
            )
            raise ValueError(
                f"nested temporal buffers violate master shape {normalized}: {details}"
            )

    def temporal_buffer_inventory(self) -> dict:
        """Measure nested image-history ownership rather than declaring it.

        Full-resolution 2-D masks and 2-channel flow fields are valid master
        state. Reduced artistic control lattices are ignored unless their path
        identifies them as an image/history buffer (for example ``prev_gray``,
        ``dye``, ``feedback``, or ``buffer``).
        """
        if self.master_shape is None:
            return {}
        master = self._normalize_master_shape(self.master_shape)
        spatial = master[:2]
        strong_tokens = (
            "frame",
            "image",
            "output",
            "history",
            "buffer",
            "feedback",
            "trail",
            "previous",
            "prev_gray",
            "flow",
            "mask",
            "dye",
            "accum",
            "held",
            "reference",
            "motion",
            "luma",
        )
        inventory = {}
        seen = set()

        def record_array(path, value, provider="cpu"):
            shape = tuple(int(part) for part in value.shape)
            if len(shape) < 2:
                return
            full_resolution = shape[:2] == spatial or (
                len(shape) >= 4 and shape[-3:-1] == spatial
            )
            looks_temporal = any(token in path.lower() for token in strong_tokens)
            if not full_resolution and not looks_temporal:
                return
            inventory[path] = {
                "shape": shape,
                "resolution": [
                    int(spatial[1]) if full_resolution else int(shape[1]),
                    int(spatial[0]) if full_resolution else int(shape[0]),
                ],
                "dtype": str(getattr(value, "dtype", "unknown")),
                "provider": provider,
                "master_resolution": bool(full_resolution),
            }

        def visit(value, path):
            if value is None or isinstance(value, (str, bytes, int, float, bool)):
                return
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            if isinstance(value, np.ndarray):
                record_array(path, value)
                return
            if isinstance(value, dict):
                for key, nested in value.items():
                    visit(nested, f"{path}.{key}")
                return
            if isinstance(value, (list, tuple, set)):
                for index, nested in enumerate(value):
                    visit(nested, f"{path}[{index}]")
                return
            size_method = getattr(value, "size", None)
            if callable(size_method) and value.__class__.__name__.endswith("GpuMat"):
                try:
                    width, height = size_method()
                    proxy = np.empty((int(height), int(width), 1), dtype=np.uint8)
                    record_array(path, proxy, provider="cuda")
                except Exception:
                    pass
                return
            module = str(getattr(value.__class__, "__module__", ""))
            if module.startswith("megaglitch") or module in {
                "collections",
                "collections.deque",
            }:
                try:
                    for name, nested in vars(value).items():
                        visit(nested, f"{path}.{name}")
                except TypeError:
                    pass

        visit(self.effect_states, "effect_states")
        visit(self.generator_states, "generator_states")
        visit(self.slit_scan_buffer, "slit_scan_buffer")
        if self.temporal_history is not None:
            try:
                stats = self.temporal_history.get_statistics()
                configured = bool(stats.get("configured"))
                width = int(stats.get("width", 0))
                height = int(stats.get("height", 0))
                if configured:
                    inventory["native_temporal_ring"] = {
                        "shape": (height, width, int(stats.get("channels", 0))),
                        "resolution": [width, height],
                        "dtype": "uint8",
                        "provider": str(stats.get("backend", "native")),
                        "master_resolution": (height, width) == spatial,
                    }
            except Exception:
                inventory["native_temporal_ring"] = {
                    "shape": None,
                    "resolution": None,
                    "dtype": "unknown",
                    "provider": "native",
                    "master_resolution": False,
                }
        return inventory

    def get_temporal_history(self, width: int, height: int, capacity: int,
                             channels: int = 3, backend: Optional[str] = None):
        """Create or reconfigure the opt-in native/Python temporal prototype."""
        from megaglitch.native_temporal import TemporalFrameRing

        if self.temporal_history is None:
            self.temporal_history = TemporalFrameRing(backend)
        stats = self.temporal_history.get_statistics()
        expected = (int(width), int(height), int(channels), int(capacity))
        actual = (
            stats.get("width", 0), stats.get("height", 0),
            stats.get("channels", 0), stats.get("capacity", 0),
        )
        if not stats.get("configured") or actual != expected:
            self.temporal_history.configure(width, height, capacity, channels)
        return self.temporal_history
    
    def ensure_scratch_buffers(self, shape: tuple):
        """Preallocate scratch buffers for the given shape to avoid per-frame allocations."""
        if self._last_shape != shape:
            self._scratch_float = np.zeros(shape, dtype=np.float32)
            self._scratch_uint8 = np.zeros(shape, dtype=np.uint8)
            self._last_shape = shape
    
    def get_scratch_float(self, shape: tuple) -> np.ndarray:
        """Get preallocated float32 scratch buffer."""
        self.ensure_scratch_buffers(shape)
        return self._scratch_float
    
    def get_scratch_uint8(self, shape: tuple) -> np.ndarray:
        """Get preallocated uint8 scratch buffer."""
        self.ensure_scratch_buffers(shape)
        return self._scratch_uint8
