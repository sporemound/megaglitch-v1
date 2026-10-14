"""Low-latency, preview-only graph execution.

Preview Engine V1 deliberately owns its temporal state, graph snapshot, adaptive
resolution controller, and newest-frame mailbox.  It never accepts or returns a
master ``FrameState`` and it never writes preview-sized buffers into the
authoritative render path.
"""

from __future__ import annotations

import copy
import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from frame_state import FrameState
from megaglitch.determinism import use_numpy_rng
from node_pipeline_runtime import execute_flat_node_pipeline


PREVIEW_TIERS: tuple[tuple[int, int], ...] = (
    (1280, 720),
    (960, 540),
    (720, 480),
    (640, 360),
)
DEFAULT_PREVIEW_TIER = (960, 540)

# Milestone 1 intentionally covers the current Rust default graph and the saved
# practical P-frame workflow.  Everything else is visible as unsupported rather
# than silently passing through while claiming production coverage.
MILESTONE_1_IMPLEMENTED_NODES = frozenset(
    {
        "P-Frame Glitches",
        "Invert Colors",
        "Color Adjust",
        "Temporal Slicing",
        "Mesh Tear",
        "Channel Flow Offset",
        "Quantization / Solarization",
        "Edge Reconstruction",
        "Structural Restoration",
        "Feedback / Trail",
        "Spectral Shift",
    }
)
MILESTONE_1_APPROXIMATED_NODES = frozenset()


def _percentile(samples: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        return 0.0
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def _json_snapshot(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


@dataclass(frozen=True)
class PreviewGraphSnapshot:
    """Immutable, revisioned graph/control input consumed at frame boundaries."""

    graph_revision: int
    pipeline_json: str
    source_position: float | None = None
    source_timestamp: float | None = None
    automation_json: str = "{}"
    controls_json: str = "{}"

    @classmethod
    def from_pipeline(
        cls,
        pipeline: Sequence[Mapping[str, Any]],
        graph_revision: int,
        *,
        source_position: float | None = None,
        source_timestamp: float | None = None,
        automation: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> "PreviewGraphSnapshot":
        return cls(
            graph_revision=int(graph_revision),
            pipeline_json=_json_snapshot(list(pipeline)),
            source_position=None if source_position is None else float(source_position),
            source_timestamp=None if source_timestamp is None else float(source_timestamp),
            automation_json=_json_snapshot(dict(automation or {})),
            controls_json=_json_snapshot(dict(controls or {})),
        )

    def materialize_pipeline(self) -> list[dict[str, Any]]:
        materialized = json.loads(self.pipeline_json)
        if not isinstance(materialized, list):
            raise ValueError("preview pipeline snapshot must contain a list")
        return materialized

    def materialize_controls(self) -> dict[str, Any]:
        automation = json.loads(self.automation_json)
        controls = json.loads(self.controls_json)
        return {**dict(automation or {}), **dict(controls or {})}


@dataclass(frozen=True)
class PreviewNodeStatus:
    node_id: str
    label: str
    status: str
    provider: str

    def as_dict(self) -> dict[str, str]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "status": self.status,
            "provider": self.provider,
        }


def build_preview_plan(
    pipeline: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[PreviewNodeStatus, ...]]:
    plan = copy.deepcopy(list(pipeline))
    statuses: list[PreviewNodeStatus] = []
    for index, step in enumerate(plan):
        label = str(step.get("effect_type") or step.get("label") or "Unknown")
        node_id = str(step.get("node_id") or f"preview-node-{index}")
        node_type = str(step.get("node_type") or "effect")
        params = dict(step.get("params") or {})
        enabled = _enabled(step.get("enabled", params.get("enabled", True)))

        if node_type == "source" or label in {"Display", "Global Time Macros"}:
            status, provider = "implemented", "preview_core"
        elif not enabled:
            status, provider = "disabled", "disabled"
        elif label in MILESTONE_1_IMPLEMENTED_NODES:
            status, provider = "implemented", "preview_scaled_existing"
        elif label in MILESTONE_1_APPROXIMATED_NODES:
            status, provider = "approximated", "preview_approximation"
        else:
            status, provider = "unsupported", "none"
            # Keep graph stack semantics, but make this node an explicit no-op.
            # Its unsupported status is carried in every result/snapshot.
            params["enabled"] = False
            step["params"] = params

        statuses.append(PreviewNodeStatus(node_id, label, status, provider))
    unsupported_structural = any(
        status.status == "unsupported"
        and str(step.get("node_type") or "effect")
        in {"generator", "mixer", "lua_multi_input", "genesis_priority_composite", "genesis_mask_key"}
        for step, status in zip(plan, statuses)
    )
    if unsupported_structural:
        # Structural nodes are handled before the ordinary enabled check in the
        # legacy flat executor.  A source/display-only fallback is the only safe
        # M1 plan that guarantees an unsupported generator/mixer cannot execute.
        sources = [step for step in plan if step.get("node_type") == "source"]
        for index, (step, status) in enumerate(zip(plan, statuses)):
            retained = (bool(sources) and step is sources[0]) or status.label in {
                "Display", "Global Time Macros"
            }
            if not retained and status.status in {"implemented", "approximated"}:
                # Coverage describes this executable plan, not a node's support
                # in isolation. The bridge/UI already surface unsupported nodes.
                statuses[index] = PreviewNodeStatus(
                    status.node_id, status.label, "unsupported", "source_display_fallback"
                )
        plan = sources[:1] + [{"effect_type": "Display", "params": {}, "node_type": "display"}]
    return plan, tuple(statuses)


def _topology_signature(pipeline: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(step.get("node_id") or ""),
            str(step.get("effect_type") or step.get("label") or ""),
            str(step.get("node_type") or "effect"),
        )
        for step in pipeline
    )


def _history_limit(pipeline: Sequence[Mapping[str, Any]]) -> int:
    limit = 10
    for step in pipeline:
        params = dict(step.get("params") or {})
        for key in ("frame_buffer_size", "buffer_size", "temporal_range", "bleed_distance"):
            try:
                value = int(params.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if key == "bleed_distance" and value > 0:
                value += 1
            limit = max(limit, value)
    return max(1, min(120, limit))


@dataclass
class PreviewTemporalState(FrameState):
    """State owned exclusively by one PreviewExecutor worker."""

    preview_shape: tuple[int, int, int] | None = None
    resolution_generation: int = 0
    temporal_reset_count: int = 0
    random_seed: int = 0
    random_state: np.random.Generator = field(init=False, repr=False)
    python_random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.random_state = np.random.default_rng(self.random_seed)
        self.python_random = random.Random(self.random_seed)

    @property
    def analysis_cache(self) -> dict[str, Any]:
        return self.effect_states

    @property
    def pframe_history(self) -> list[np.ndarray]:
        return self.frame_buffer

    @property
    def temporal_slicing_history(self) -> list[np.ndarray]:
        return self.frame_buffer

    def reset_preview(
        self,
        *,
        shape: tuple[int, int, int] | None = None,
        resolution_change: bool = False,
    ) -> None:
        FrameState.reset(self)
        self.random_state = np.random.default_rng(self.random_seed)
        self.python_random = random.Random(self.random_seed)
        self.preview_shape = shape
        self.temporal_reset_count += 1
        if resolution_change:
            self.resolution_generation += 1


class AdaptivePreviewController:
    """Bounded p95 controller with sustained windows and tier-change cooldown."""

    def __init__(
        self,
        tiers: Sequence[tuple[int, int]] = PREVIEW_TIERS,
        preferred_tier: tuple[int, int] = DEFAULT_PREVIEW_TIER,
        *,
        high_budget_ms: float = 16.67,
        low_budget_ms: float = 11.0,
        cooldown_seconds: float = 2.0,
        min_samples: int = 60,
        window_size: int = 120,
        clock=time.perf_counter,
    ) -> None:
        normalized = tuple((int(width), int(height)) for width, height in tiers)
        if not normalized or any(width <= 0 or height <= 0 for width, height in normalized):
            raise ValueError("preview tiers must contain positive dimensions")
        if preferred_tier not in normalized:
            raise ValueError("preferred preview tier is not in the configured tier list")
        if low_budget_ms >= high_budget_ms:
            raise ValueError("adaptive low budget must be below high budget")
        self.tiers = normalized
        self._index = normalized.index(preferred_tier)
        self.high_budget_ms = float(high_budget_ms)
        self.low_budget_ms = float(low_budget_ms)
        self.cooldown_seconds = float(cooldown_seconds)
        self.min_samples = int(min_samples)
        self._samples: deque[float] = deque(maxlen=max(int(window_size), self.min_samples))
        self._clock = clock
        self._last_change = float("-inf")

    @property
    def current_tier(self) -> tuple[int, int]:
        return self.tiers[self._index]

    @property
    def p95_ms(self) -> float:
        return _percentile(self._samples, 0.95)

    def set_tier(self, tier: tuple[int, int], *, now: float | None = None) -> bool:
        tier = (int(tier[0]), int(tier[1]))
        if tier not in self.tiers:
            raise ValueError(f"unsupported preview tier {tier!r}")
        next_index = self.tiers.index(tier)
        changed = next_index != self._index
        self._index = next_index
        self._samples.clear()
        if changed:
            self._last_change = self._clock() if now is None else float(now)
        return changed

    def observe(self, render_ms: float, *, now: float | None = None) -> tuple[int, int] | None:
        self._samples.append(float(render_ms))
        if len(self._samples) < self.min_samples:
            return None
        now_value = self._clock() if now is None else float(now)
        if now_value - self._last_change < self.cooldown_seconds:
            return None
        p95 = self.p95_ms
        next_index = self._index
        if p95 > self.high_budget_ms and self._index < len(self.tiers) - 1:
            next_index += 1
        elif p95 < self.low_budget_ms and self._index > 0:
            next_index -= 1
        if next_index == self._index:
            return None
        self._index = next_index
        self._last_change = now_value
        self._samples.clear()
        return self.current_tier


@dataclass(frozen=True)
class PreviewRequest:
    sequence: int
    source_frame: np.ndarray = field(compare=False, repr=False)
    source_timestamp: float
    submitted_at: float
    graph_revision: int
    source_position: float | None = None


@dataclass(frozen=True)
class PreviewResult:
    sequence: int
    source_timestamp: float
    submitted_at: float
    rendered_at: float
    graph_revision: int
    source_position: float | None
    frame: np.ndarray = field(compare=False, repr=False)
    render_ms: float
    p95_ms: float
    tier: tuple[int, int]
    resolution_generation: int
    node_statuses: tuple[PreviewNodeStatus, ...]


@dataclass(frozen=True)
class PreviewExecutorTelemetry:
    preview_requests: int
    preview_rendered: int
    preview_superseded: int
    preview_dropped: int
    preview_sequence: int
    preview_render_fps: float
    preview_frame_ms: float
    preview_p95_ms: float
    preview_resolution: tuple[int, int]
    configured_tier: tuple[int, int]
    graph_revision: int
    last_rendered_graph_revision: int
    resolution_generation: int
    node_statuses: tuple[PreviewNodeStatus, ...]
    last_rendered_node_statuses: tuple[PreviewNodeStatus, ...]
    last_error: dict[str, str] | None


def _resize_preview_ingress(frame: np.ndarray, tier: tuple[int, int]) -> np.ndarray:
    source = np.asarray(frame)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError(f"preview source must be HxWx3+, got {source.shape!r}")
    source = source[:, :, :3]
    width, height = tier
    if source.shape[1] <= width and source.shape[0] <= height:
        return np.array(source, dtype=np.uint8, order="C", copy=True)
    try:
        import cv2

        interpolation = cv2.INTER_AREA if source.shape[1] >= width and source.shape[0] >= height else cv2.INTER_LINEAR
        resized = cv2.resize(source, (width, height), interpolation=interpolation)
        return np.ascontiguousarray(resized, dtype=np.uint8)
    except ImportError:
        x_indices = np.linspace(0, source.shape[1] - 1, width).astype(np.intp)
        y_indices = np.linspace(0, source.shape[0] - 1, height).astype(np.intp)
        return np.ascontiguousarray(source[y_indices][:, x_indices], dtype=np.uint8)


class PreviewExecutor:
    """Dedicated preview worker with a capacity-one newest-frame mailbox."""

    def __init__(
        self,
        graph_snapshot: PreviewGraphSnapshot | None = None,
        *,
        preferred_tier: tuple[int, int] = DEFAULT_PREVIEW_TIER,
        adaptive: bool = True,
        autostart: bool = True,
        controller: AdaptivePreviewController | None = None,
    ) -> None:
        self.temporal_state = PreviewTemporalState()
        self.controller = controller or AdaptivePreviewController(preferred_tier=preferred_tier)
        self.adaptive = bool(adaptive)
        self._condition = threading.Condition()
        self._pending: PreviewRequest | None = None
        self._latest_result: PreviewResult | None = None
        self._stop_requested = False
        self._reset_requested = True
        self._thread: threading.Thread | None = None
        self._snapshot = graph_snapshot or PreviewGraphSnapshot.from_pipeline(
            [
                {"effect_type": "Source", "params": {}, "node_type": "source", "source_key": "primary"},
                {"effect_type": "Display", "params": {}, "node_type": "display"},
            ],
            0,
        )
        self._pipeline, self._node_statuses = build_preview_plan(self._snapshot.materialize_pipeline())
        self._topology = _topology_signature(self._snapshot.materialize_pipeline())
        self._preview_requests = 0
        self._preview_rendered = 0
        self._preview_superseded = 0
        self._preview_dropped = 0
        self._preview_sequence = 0
        self._last_render_ms = 0.0
        self._render_timestamps: deque[float] = deque()
        # Protocol revisions are unsigned; zero is the explicit pre-render sentinel.
        self._last_rendered_graph_revision = 0
        self._last_rendered_node_statuses: tuple[PreviewNodeStatus, ...] = ()
        self._last_rendered_resolution = (0, 0)
        self._last_error: dict[str, str] | None = None
        if autostart:
            self.start()

    @property
    def mailbox_capacity(self) -> int:
        return 1

    @property
    def graph_snapshot(self) -> PreviewGraphSnapshot:
        with self._condition:
            return self._snapshot

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested = False
            self._thread = threading.Thread(target=self._worker_loop, name="mg-preview-v1", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))
            if thread.is_alive():
                raise TimeoutError("preview worker did not stop cleanly")
        with self._condition:
            self._thread = None
            self._pending = None
        self.temporal_state.reset_preview()

    def update_graph(self, snapshot: PreviewGraphSnapshot) -> None:
        if not isinstance(snapshot, PreviewGraphSnapshot):
            raise TypeError("preview graph update requires PreviewGraphSnapshot")
        materialized = snapshot.materialize_pipeline()
        pipeline, statuses = build_preview_plan(materialized)
        topology = _topology_signature(materialized)
        with self._condition:
            if snapshot.graph_revision < self._snapshot.graph_revision:
                raise ValueError("preview graph revisions must be monotonic")
            self._snapshot = snapshot
            self._pipeline = pipeline
            self._node_statuses = statuses
            if topology != self._topology:
                self._reset_requested = True
            self._topology = topology
            self._condition.notify_all()

    def submit_latest(
        self,
        source_frame: np.ndarray,
        source_timestamp: float,
        graph_revision: int,
        *,
        sequence: int | None = None,
        source_position: float | None = None,
    ) -> int:
        # The mailbox owns an immutable copy so an authoritative executor cannot
        # mutate the same source allocation while preview ingress is scaling it.
        owned_frame = np.array(source_frame, dtype=np.uint8, order="C", copy=True)
        submitted_at = time.time()
        with self._condition:
            if int(graph_revision) > self._snapshot.graph_revision:
                raise ValueError("source request references a graph revision that is not configured")
            self._preview_requests += 1
            request_sequence = self._preview_requests if sequence is None else int(sequence)
            if self._pending is not None:
                self._preview_superseded += 1
            self._pending = PreviewRequest(
                sequence=request_sequence,
                source_frame=owned_frame,
                source_timestamp=float(source_timestamp),
                submitted_at=submitted_at,
                graph_revision=int(graph_revision),
                source_position=None if source_position is None else float(source_position),
            )
            self._condition.notify_all()
            return request_sequence

    def reset_temporal_state(self) -> None:
        with self._condition:
            self._reset_requested = True
            self._condition.notify_all()

    def set_tier(self, tier: tuple[int, int]) -> None:
        with self._condition:
            if self.controller.set_tier(tier):
                self._reset_requested = True
            self._condition.notify_all()

    def set_adaptive(self, enabled: bool) -> None:
        with self._condition:
            self.adaptive = bool(enabled)

    def _claim_latest(self, *, wait: bool) -> tuple[PreviewRequest, PreviewGraphSnapshot, list[dict[str, Any]], tuple[PreviewNodeStatus, ...], tuple[int, int], bool] | None:
        with self._condition:
            while wait and self._pending is None and not self._stop_requested:
                self._condition.wait(timeout=0.25)
            if self._pending is None:
                return None
            request = self._pending
            self._pending = None
            snapshot = self._snapshot
            pipeline = self._pipeline
            statuses = self._node_statuses
            tier = self.controller.current_tier
            reset_requested = self._reset_requested
            self._reset_requested = False
            return request, snapshot, pipeline, statuses, tier, reset_requested

    def render_latest(self) -> PreviewResult | None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("synchronous render_latest cannot run while preview worker is active")
        claimed = self._claim_latest(wait=False)
        if claimed is None:
            return None
        request, snapshot, pipeline, statuses, tier, reset_requested = claimed
        return self._render_claimed(request, snapshot, pipeline, statuses, tier, reset_requested)

    def _render_claimed(
        self,
        request: PreviewRequest,
        snapshot: PreviewGraphSnapshot,
        pipeline: list[dict[str, Any]],
        statuses: tuple[PreviewNodeStatus, ...],
        tier: tuple[int, int],
        reset_requested: bool,
    ) -> PreviewResult:
        started = time.perf_counter()
        preview_input = _resize_preview_ingress(request.source_frame, tier)
        shape = tuple(int(value) for value in preview_input.shape)
        resolution_change = self.temporal_state.preview_shape not in {None, shape}
        if reset_requested or self.temporal_state.preview_shape != shape:
            self.temporal_state.reset_preview(shape=shape, resolution_change=resolution_change)

        state = self.temporal_state
        extra_params = snapshot.materialize_controls()
        extra_params["_preview_python_rng"] = state.python_random
        extra_params["_preview_numpy_rng"] = state.random_state
        with use_numpy_rng(state.random_state):
            output, previous, feedback, trail = execute_flat_node_pipeline(
                pipeline,
                preview_input,
                extra_params=extra_params,
                previous_frame=state.previous_frame,
                feedback_image=state.feedback_image,
                trail_image=state.trail_image,
                frame_buffer=state.frame_buffer,
                worker_state=state,
                fallback_size=(preview_input.shape[1], preview_input.shape[0]),
                source_copy_policy="borrow_unique",
            )
        state.previous_frame = previous
        state.feedback_image = feedback
        state.trail_image = trail
        state.frame_count += 1
        state.frame_buffer.append(np.ascontiguousarray(output, dtype=np.uint8).copy())
        limit = _history_limit(pipeline)
        if len(state.frame_buffer) > limit:
            del state.frame_buffer[:-limit]
        rendered_at = time.time()
        render_ms = (time.perf_counter() - started) * 1000.0

        with self._condition:
            triggering_p95 = self.controller.p95_ms
            changed_tier = self.controller.observe(render_ms) if self.adaptive else None
            reported_p95 = triggering_p95 if changed_tier is not None else self.controller.p95_ms
            if changed_tier is not None:
                self._reset_requested = True

        result = PreviewResult(
            sequence=request.sequence,
            source_timestamp=request.source_timestamp,
            submitted_at=request.submitted_at,
            rendered_at=rendered_at,
            graph_revision=snapshot.graph_revision,
            source_position=request.source_position,
            frame=np.ascontiguousarray(output, dtype=np.uint8),
            render_ms=render_ms,
            p95_ms=reported_p95,
            tier=tier,
            resolution_generation=state.resolution_generation,
            node_statuses=statuses,
        )
        self._record_result(result)
        return result

    def _record_result(self, result: PreviewResult) -> None:
        now = time.perf_counter()
        with self._condition:
            self._preview_rendered += 1
            self._preview_sequence = result.sequence
            self._last_render_ms = result.render_ms
            self._last_rendered_graph_revision = result.graph_revision
            self._last_rendered_node_statuses = result.node_statuses
            self._last_rendered_resolution = (int(result.frame.shape[1]), int(result.frame.shape[0]))
            self._last_error = None
            self._render_timestamps.append(now)
            cutoff = now - 2.0
            while self._render_timestamps and self._render_timestamps[0] < cutoff:
                self._render_timestamps.popleft()
            if self._latest_result is not None:
                self._preview_dropped += 1
            self._latest_result = result
            self._condition.notify_all()

    def consume_latest(self) -> PreviewResult | None:
        with self._condition:
            result = self._latest_result
            self._latest_result = None
            return result

    def wait_for_result(self, *, after_sequence: int = 0, timeout: float = 5.0) -> PreviewResult:
        deadline = time.perf_counter() + float(timeout)
        with self._condition:
            while True:
                if self._latest_result is not None and self._latest_result.sequence > after_sequence:
                    result = self._latest_result
                    self._latest_result = None
                    return result
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for a new preview result")
                self._condition.wait(timeout=remaining)

    def telemetry(self) -> PreviewExecutorTelemetry:
        now = time.perf_counter()
        with self._condition:
            cutoff = now - 2.0
            while self._render_timestamps and self._render_timestamps[0] < cutoff:
                self._render_timestamps.popleft()
            if len(self._render_timestamps) >= 2:
                elapsed = self._render_timestamps[-1] - self._render_timestamps[0]
                render_fps = (len(self._render_timestamps) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                render_fps = 0.0
            return PreviewExecutorTelemetry(
                preview_requests=self._preview_requests,
                preview_rendered=self._preview_rendered,
                preview_superseded=self._preview_superseded,
                preview_dropped=self._preview_dropped,
                preview_sequence=self._preview_sequence,
                preview_render_fps=render_fps,
                preview_frame_ms=self._last_render_ms,
                preview_p95_ms=self.controller.p95_ms,
                preview_resolution=self._last_rendered_resolution,
                configured_tier=self.controller.current_tier,
                graph_revision=self._snapshot.graph_revision,
                last_rendered_graph_revision=self._last_rendered_graph_revision,
                resolution_generation=self.temporal_state.resolution_generation,
                node_statuses=self._node_statuses,
                last_rendered_node_statuses=self._last_rendered_node_statuses,
                last_error=None if self._last_error is None else dict(self._last_error),
            )

    def _worker_loop(self) -> None:
        while True:
            claimed = self._claim_latest(wait=True)
            if claimed is None:
                with self._condition:
                    if self._stop_requested:
                        return
                continue
            request, snapshot, pipeline, statuses, tier, reset_requested = claimed
            try:
                self._render_claimed(request, snapshot, pipeline, statuses, tier, reset_requested)
            except Exception as error:
                with self._condition:
                    self._preview_dropped += 1
                    self._last_error = {
                        "code": "preview_render_failed",
                        "message": str(error),
                        "exception": type(error).__name__,
                    }
                    self._condition.notify_all()

    def __enter__(self) -> "PreviewExecutor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
