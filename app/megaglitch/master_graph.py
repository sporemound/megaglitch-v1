"""Deterministic dual-deck executor for the master-resolution graph domain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import gpu_ops
from frame_state import FrameState
from node_pipeline_runtime import blend_decks, execute_flat_node_pipeline, record_frame_copy

from .render_contract import FrameRole, MasterResolution, validate_master_frame


def _provider_for_node(node: Mapping[str, Any]) -> str:
    declared = str(node.get("provider") or node.get("backend") or "").strip().lower()
    if declared:
        return declared
    label = str(node.get("effect_type") or "").lower()
    if "cuda" in label or label.startswith("gpu "):
        return "cuda"
    if "lua" in label or "glsl" in label or "shader" in label:
        return "opengl"
    return "cpu"


def describe_provider_assignments(
    pipeline_a: Sequence[Mapping[str, Any]] | None,
    pipeline_b: Sequence[Mapping[str, Any]] | None,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for deck_name, pipeline in (("a", pipeline_a), ("b", pipeline_b)):
        for index, node in enumerate(pipeline or ()):
            if node.get("effect_type") == "Display":
                continue
            node_key = str(
                node.get("node_id")
                or f"deck_{deck_name}:{index}:{node.get('effect_type', 'unknown')}"
            )
            assignments[node_key] = _provider_for_node(node)
    assignments["master_deck_blend"] = "cpu"
    return dict(sorted(assignments.items()))


def _buffer_limit(pipeline: Sequence[Mapping[str, Any]] | None, default: int) -> int:
    limit = max(1, int(default))
    for node in pipeline or ():
        params = node.get("params") or {}
        for name in ("frame_buffer_size", "buffer_size", "temporal_range"):
            if name not in params:
                continue
            try:
                limit = max(limit, int(float(params[name])))
            except (TypeError, ValueError):
                continue
    return limit


def _normalize_state_frame(
    value: Any,
    resolution: MasterResolution,
    name: str,
) -> np.ndarray:
    normalized = validate_master_frame(
        value,
        resolution,
        role=FrameRole.MASTER_PROGRAM,
        name=name,
    )
    if normalized is None:
        return np.zeros(resolution.shape, dtype=np.uint8)
    return normalized


@dataclass(frozen=True)
class MasterGraphResult:
    frame: np.ndarray
    deck_a: Optional[np.ndarray]
    deck_b: Optional[np.ndarray]
    frame_copy_stats: Mapping[str, Mapping[str, int]]
    history_depth_a: int
    history_depth_b: int
    temporal_inventory_a: Mapping[str, Mapping[str, Any]]
    temporal_inventory_b: Mapping[str, Mapping[str, Any]]


class MasterGraphExecutor:
    """Serial, fixed-resolution graph state shared by live-safe offline rendering.

    The executor has no preview, OBS, virtual-camera, queue, or encoder hooks.
    Every persistent image buffer is validated against ``resolution`` after
    every frame.  A mismatch is an error; it is never repaired by resizing.
    """

    def __init__(self, resolution: MasterResolution, *, use_cuda: bool = False) -> None:
        self.resolution = resolution
        self.use_cuda = bool(use_cuda)
        self.state_a = FrameState()
        self.state_b = FrameState()
        self.state_a.bind_master_shape(resolution.shape)
        self.state_b.bind_master_shape(resolution.shape)

    def reset(self) -> None:
        self.state_a.reset()
        self.state_b.reset()
        self.state_a.bind_master_shape(self.resolution.shape)
        self.state_b.bind_master_shape(self.resolution.shape)

    @property
    def history_buffer_resolution(self) -> dict[str, tuple[int, int]]:
        size = (self.resolution.width, self.resolution.height)
        measured = {
            "deck_a_previous_output": size,
            "deck_a_feedback": size,
            "deck_a_held_and_pframe_history": size,
            "deck_b_previous_output": size,
            "deck_b_feedback": size,
            "deck_b_held_and_pframe_history": size,
        }
        for deck_name, state in (("a", self.state_a), ("b", self.state_b)):
            for path, record in state.temporal_buffer_inventory().items():
                resolution = record.get("resolution")
                if resolution and len(resolution) == 2:
                    measured[f"deck_{deck_name}.{path}"] = tuple(
                        int(value) for value in resolution
                    )
        return measured

    def _execute_deck(
        self,
        pipeline: Sequence[Mapping[str, Any]] | None,
        primary: Optional[np.ndarray],
        secondary: Optional[np.ndarray],
        extra_params: Mapping[str, Any] | None,
        state: FrameState,
        copy_stats: dict[str, dict[str, int]],
        deck_name: str,
    ) -> Optional[np.ndarray]:
        if not pipeline:
            return None
        output, previous, feedback, trail = execute_flat_node_pipeline(
            pipeline,
            primary,
            secondary_frame=secondary,
            extra_params=dict(extra_params or {}),
            previous_frame=state.previous_frame,
            feedback_image=state.feedback_image,
            trail_image=state.trail_image,
            frame_buffer=state.frame_buffer,
            worker_state=state,
            fallback_size=(self.resolution.width, self.resolution.height),
            copy_stats=copy_stats,
            source_copy_policy="copy",
        )
        output = _normalize_state_frame(
            output, self.resolution, f"deck {deck_name} master output"
        )
        state.previous_frame = _normalize_state_frame(
            previous, self.resolution, f"deck {deck_name} previous output"
        )
        state.feedback_image = _normalize_state_frame(
            feedback, self.resolution, f"deck {deck_name} feedback"
        )
        state.trail_image = _normalize_state_frame(
            trail, self.resolution, f"deck {deck_name} trail"
        )
        history_frame = output.copy()
        record_frame_copy(copy_stats, "history", history_frame)
        state.frame_buffer.append(history_frame)
        limit = _buffer_limit(pipeline, state.buffer_limit)
        while len(state.frame_buffer) > limit:
            state.frame_buffer.pop(0)
        state.buffer_limit = limit
        state.frame_count += 1
        state.assert_master_integrity()
        return output

    def process(
        self,
        primary: Any,
        secondary: Any = None,
        *,
        pipeline_a: Sequence[Mapping[str, Any]] | None,
        pipeline_b: Sequence[Mapping[str, Any]] | None = None,
        extra_params_a: Mapping[str, Any] | None = None,
        extra_params_b: Mapping[str, Any] | None = None,
        master_mix: float = 0.0,
        master_transition: str = "Normal Mix",
        instrumentation: Any = None,
        source_index: int = 0,
    ) -> MasterGraphResult:
        primary_frame = validate_master_frame(
            primary,
            self.resolution,
            role=FrameRole.MASTER_INGRESS,
            name="offline primary ingress",
        )
        secondary_frame = validate_master_frame(
            secondary,
            self.resolution,
            role=FrameRole.MASTER_INGRESS,
            name="offline secondary ingress",
        )
        copy_stats: dict[str, dict[str, int]] = {}
        gpu_ops.set_use_cuda(self.use_cuda)

        def transfer_observer(direction: str, byte_count: int) -> None:
            if instrumentation is not None:
                instrumentation.record_cpu_gpu_transfer(
                    direction, byte_count, int(source_index)
                )

        with gpu_ops.observe_transfers(transfer_observer):
            frame_a = self._execute_deck(
                pipeline_a,
                primary_frame,
                secondary_frame,
                extra_params_a,
                self.state_a,
                copy_stats,
                "A",
            )
            frame_b = self._execute_deck(
                pipeline_b,
                secondary_frame,
                secondary_frame,
                extra_params_b,
                self.state_b,
                copy_stats,
                "B",
            )
            if frame_a is None and frame_b is None:
                if primary_frame is not None:
                    master = primary_frame.copy()
                elif secondary_frame is not None:
                    master = secondary_frame.copy()
                else:
                    master = np.zeros(self.resolution.shape, dtype=np.uint8)
            else:
                master = blend_decks(
                    frame_a,
                    frame_b,
                    master_mix,
                    master_transition,
                    fallback_size=(self.resolution.width, self.resolution.height),
                )
        master = _normalize_state_frame(master, self.resolution, "master deck blend")
        self.state_a.assert_master_integrity()
        self.state_b.assert_master_integrity()
        return MasterGraphResult(
            frame=master,
            deck_a=frame_a,
            deck_b=frame_b,
            frame_copy_stats=copy_stats,
            history_depth_a=len(self.state_a.frame_buffer),
            history_depth_b=len(self.state_b.frame_buffer),
            temporal_inventory_a=self.state_a.temporal_buffer_inventory(),
            temporal_inventory_b=self.state_b.temporal_buffer_inventory(),
        )
