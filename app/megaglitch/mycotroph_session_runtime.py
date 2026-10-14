"""UI-independent authoritative dual-deck session execution.

This module is the coarse-grained seam between a host and MegaGlitch's
existing Python visual runtime.  It deliberately owns no files, cameras,
windows, or synthetic sources: callers supply the primary and secondary
frames for each render.

Session replacement has two explicit phases.  :meth:`prepare_session`
normalizes and compiles both deck graphs without touching live temporal
state.  :meth:`commit_session` then swaps the fully prepared value in one
critical section.  A rejected candidate therefore leaves the last accepted
session, pipelines, graph generations, and per-deck state intact.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from frame_state import FrameState
from node_pipeline_runtime import blend_decks, execute_flat_node_pipeline

from .session_document import normalize_session_document
from .session_graph_compiler import GraphCompileError, compile_preset_graph


MASTER_TRANSITION_MODES = (
    "Normal Mix",
    "Luma Wipe",
    "Glitch Dissolve",
    "Horizontal Wipe",
    "Vertical Wipe",
    "Radial Wipe",
    "Checker Wipe",
    "Hard Cut",
    "Additive",
    "Multiply",
    "Screen",
    "Difference",
)

_SOURCE_KEYS = frozenset({"primary", "secondary"})


class MycotrophSessionError(ValueError):
    """Stable, machine-readable failure at the session-runtime boundary."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = _json_detach(details)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": _json_detach(self.details),
        }


def _json_detach(value: Any) -> Any:
    """Detach JSON data and reject non-portable values."""

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _sha256_frame(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes(order="C")).hexdigest()


def _source_key(binding: Any) -> str:
    if isinstance(binding, Mapping):
        binding = binding.get("source_key")
    return str(binding or "").strip()


def _canonical_rgb(frame: Any, fallback_size: tuple[int, int]) -> np.ndarray:
    """Use the production executor's established source conversion path."""

    output, _previous, _feedback, _trail = execute_flat_node_pipeline(
        [],
        frame,
        fallback_size=fallback_size,
        source_copy_policy="copy",
    )
    array = np.asarray(output)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"executor returned non-RGB8 frame {array.shape}/{array.dtype}")
    return np.ascontiguousarray(array).copy()


def _frame_description(frame: np.ndarray) -> dict[str, Any]:
    height, width = frame.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "channels": 3,
        "pixel_format": "RGB8",
        "sha256": _sha256_frame(frame),
    }


@dataclass(frozen=True)
class _PreparedDeck:
    canonical: Mapping[str, Any]
    pipeline: tuple[Mapping[str, Any], ...]
    source_namespace: Mapping[str, Any]
    executable_sha256: str


@dataclass(frozen=True)
class PreparedSession:
    """Opaque immutable result of successful normalization and compilation."""

    canonical: Mapping[str, Any]
    decks: Mapping[str, _PreparedDeck]
    master_mix: float
    master_transition: str
    macros: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class SessionRender:
    """One canonical dual-deck render and deterministic, timing-free metadata."""

    deck_a: np.ndarray
    deck_b: np.ndarray
    program: np.ndarray
    telemetry: Mapping[str, Any]

    def __iter__(self):
        yield self.deck_a
        yield self.deck_b
        yield self.program
        yield self.telemetry


@dataclass
class _DeckRuntime:
    prepared: _PreparedDeck
    state: FrameState
    generation: int


class MycotrophSessionRuntime:
    """Own and execute one accepted canonical dual-deck session."""

    def __init__(
        self,
        session: Mapping[str, Any] | None = None,
        *,
        fallback_size: tuple[int, int] = (1280, 720),
    ) -> None:
        try:
            width, height = (int(fallback_size[0]), int(fallback_size[1]))
        except (IndexError, TypeError, ValueError) as error:
            raise MycotrophSessionError(
                "invalid_fallback_size",
                "fallback_size must contain integer width and height",
            ) from error
        if width <= 0 or height <= 0:
            raise MycotrophSessionError(
                "invalid_fallback_size",
                "fallback dimensions must be positive",
                width=width,
                height=height,
            )

        self._fallback_size = (width, height)
        self._lock = threading.RLock()
        self._accepted: PreparedSession | None = None
        self._decks: dict[str, _DeckRuntime] = {}
        self._session_revision = 0
        self._last_sequence: int | None = None
        if session is not None:
            self.replace_session(session)

    @staticmethod
    def prepare_session(document: Mapping[str, Any]) -> PreparedSession:
        """Normalize and compile a complete candidate without live mutation."""

        try:
            canonical = normalize_session_document(document)
        except (TypeError, ValueError) as error:
            raise MycotrophSessionError(
                "invalid_session_document",
                str(error),
                exception=type(error).__name__,
            ) from error

        prepared_decks: dict[str, _PreparedDeck] = {}
        for deck_name in ("a", "b"):
            deck = canonical["decks"][deck_name]
            namespace = deck["source_bindings"]
            for endpoint, binding in namespace.items():
                source_key = _source_key(binding)
                if source_key not in _SOURCE_KEYS:
                    raise MycotrophSessionError(
                        "invalid_source_namespace",
                        f"deck {deck_name} endpoint {endpoint!r} has unsupported source key",
                        deck=deck_name,
                        endpoint=str(endpoint),
                        source_key=source_key,
                        allowed=sorted(_SOURCE_KEYS),
                    )
            try:
                pipeline = compile_preset_graph(deck["graph"], namespace)
            except GraphCompileError as error:
                raise MycotrophSessionError(
                    "deck_graph_invalid",
                    f"deck {deck_name} graph was rejected: {error}",
                    deck=deck_name,
                    graph_code=error.code,
                    graph_details=error.details,
                ) from error

            executable = {
                "graph": deck["graph"],
                "source_bindings": namespace,
            }
            prepared_decks[deck_name] = _PreparedDeck(
                canonical=_freeze_json(deck),
                pipeline=tuple(_freeze_json(step) for step in pipeline),
                source_namespace=_freeze_json(namespace),
                executable_sha256=_sha256_json(executable),
            )

        master = canonical.get("master")
        if not isinstance(master, Mapping):
            raise MycotrophSessionError(
                "invalid_master",
                "session.master must be a JSON object",
            )
        raw_mix = master.get("mix", 0.5)
        if isinstance(raw_mix, bool) or not isinstance(raw_mix, (int, float)):
            raise MycotrophSessionError(
                "invalid_master_mix",
                "session.master.mix must be numeric",
                received_type=type(raw_mix).__name__,
            )
        master_mix = float(raw_mix)
        if not math.isfinite(master_mix):
            raise MycotrophSessionError(
                "invalid_master_mix",
                "session.master.mix must be finite",
            )
        master_mix = min(1.0, max(0.0, master_mix))

        transition = master.get("transition", "Normal Mix")
        if transition not in MASTER_TRANSITION_MODES:
            raise MycotrophSessionError(
                "unsupported_master_transition",
                f"unsupported master transition {transition!r}",
                transition=transition,
                supported=list(MASTER_TRANSITION_MODES),
            )

        session_macros = canonical.get("macros")
        if not isinstance(session_macros, Mapping):
            raise MycotrophSessionError(
                "invalid_session_macros",
                "session.macros must be a JSON object",
            )

        frozen_canonical = _freeze_json(canonical)
        return PreparedSession(
            canonical=frozen_canonical,
            decks=MappingProxyType(prepared_decks),
            master_mix=master_mix,
            master_transition=str(transition),
            macros=_freeze_json(session_macros),
            sha256=_sha256_json(frozen_canonical),
        )

    def commit_session(self, prepared: PreparedSession) -> dict[str, Any]:
        """Atomically accept a successfully prepared session."""

        if not isinstance(prepared, PreparedSession):
            raise MycotrophSessionError(
                "invalid_prepared_session",
                "commit_session requires a PreparedSession returned by prepare_session",
                received_type=type(prepared).__name__,
            )

        retired_states: list[FrameState] = []
        with self._lock:
            candidate_decks: dict[str, _DeckRuntime] = {}
            try:
                prepared_decks = {
                    deck_name: prepared.decks[deck_name] for deck_name in ("a", "b")
                }
            except (KeyError, TypeError) as error:
                raise MycotrophSessionError(
                    "invalid_prepared_session",
                    "prepared session does not contain both compiled decks",
                ) from error

            for deck_name, next_deck in prepared_decks.items():
                if not isinstance(next_deck, _PreparedDeck):
                    raise MycotrophSessionError(
                        "invalid_prepared_session",
                        "prepared session contains an invalid deck payload",
                        deck=deck_name,
                    )
                current = self._decks.get(deck_name)
                if (
                    current is not None
                    and current.prepared.executable_sha256 == next_deck.executable_sha256
                ):
                    candidate_decks[deck_name] = _DeckRuntime(
                        prepared=next_deck,
                        state=current.state,
                        generation=current.generation,
                    )
                else:
                    generation = 1 if current is None else current.generation + 1
                    candidate_decks[deck_name] = _DeckRuntime(
                        prepared=next_deck,
                        state=FrameState(),
                        generation=generation,
                    )
                    if current is not None:
                        retired_states.append(current.state)

            # This is the single commit point.  Every fallible normalization
            # and compilation operation has already completed above.
            self._accepted = prepared
            self._decks = candidate_decks
            self._session_revision += 1
            snapshot = self._snapshot_unlocked()

        # Resource retirement is best-effort and cannot partially roll back an
        # already committed session.  No live deck references these states.
        for state in retired_states:
            try:
                state.reset()
            except Exception:
                pass
        return snapshot

    def replace_session(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare then atomically commit a complete session document."""

        return self.commit_session(self.prepare_session(document))

    @property
    def session_document(self) -> dict[str, Any] | None:
        with self._lock:
            if self._accepted is None:
                return None
            return _thaw_json(self._accepted.canonical)

    @property
    def session_revision(self) -> int:
        with self._lock:
            return self._session_revision

    @property
    def deck_generations(self) -> dict[str, int]:
        with self._lock:
            return {
                deck_name: runtime.generation
                for deck_name, runtime in self._decks.items()
            }

    def _temporal_description(self, state: FrameState) -> dict[str, Any]:
        return {
            "frame_count": int(state.frame_count),
            "history_depth": len(state.frame_buffer),
            "has_previous": state.previous_frame is not None,
            "has_feedback": state.feedback_image is not None,
            "has_trail": state.trail_image is not None,
            "generator_state_count": len(state.generator_states),
            "effect_state_count": len(state.effect_states),
        }

    def _snapshot_unlocked(self) -> dict[str, Any]:
        if self._accepted is None:
            return {
                "backend": "python_flat_dual_deck",
                "session_revision": self._session_revision,
                "accepted": False,
                "decks": {},
                "last_sequence": self._last_sequence,
            }
        return {
            "backend": "python_flat_dual_deck",
            "session_revision": self._session_revision,
            "accepted": True,
            "session_sha256": self._accepted.sha256,
            "master": {
                "mix": self._accepted.master_mix,
                "transition": self._accepted.master_transition,
            },
            "decks": {
                deck_name: {
                    "graph_generation": runtime.generation,
                    "executable_sha256": runtime.prepared.executable_sha256,
                    "temporal": self._temporal_description(runtime.state),
                }
                for deck_name, runtime in self._decks.items()
            },
            "last_sequence": self._last_sequence,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return detached deterministic control and temporal state metadata."""

        with self._lock:
            return _json_detach(self._snapshot_unlocked())

    @staticmethod
    def _buffer_limit(pipeline: tuple[Mapping[str, Any], ...], current: int) -> int:
        limit = int(current)
        for effect in pipeline:
            params = effect.get("params")
            if not isinstance(params, Mapping):
                continue
            for key in ("frame_buffer_size", "buffer_size"):
                if key not in params:
                    continue
                try:
                    limit = max(0, int(params[key]))
                except (TypeError, ValueError):
                    pass
                break
        return limit

    def _render_deck(
        self,
        deck_name: str,
        primary: np.ndarray | None,
        secondary: np.ndarray | None,
        extra_params: Mapping[str, Any],
        fallback_size: tuple[int, int],
    ) -> np.ndarray:
        runtime = self._decks[deck_name]
        pipeline = runtime.prepared.pipeline

        # The compiler's empty graph is the legacy pass-through shape.  Select
        # its base from that deck's explicit SOURCE binding rather than letting
        # the executor implicitly prefer the global primary source.
        if not pipeline:
            base_key = _source_key(runtime.prepared.source_namespace.get("SOURCE"))
            base = secondary if base_key == "secondary" else primary
            call_primary = base
            call_secondary = None
        else:
            call_primary = primary
            call_secondary = secondary

        try:
            output, previous, feedback, trail = execute_flat_node_pipeline(
                pipeline,
                call_primary,
                secondary_frame=call_secondary,
                extra_params=extra_params,
                previous_frame=runtime.state.previous_frame,
                feedback_image=runtime.state.feedback_image,
                trail_image=runtime.state.trail_image,
                frame_buffer=runtime.state.frame_buffer,
                worker_state=runtime.state,
                fallback_size=fallback_size,
                source_copy_policy="copy",
            )
        except Exception as error:
            raise MycotrophSessionError(
                "deck_render_failed",
                f"deck {deck_name} render failed: {error}",
                deck=deck_name,
                exception=type(error).__name__,
            ) from error

        runtime.state.previous_frame = previous
        runtime.state.feedback_image = feedback
        runtime.state.trail_image = trail
        result = np.asarray(output)
        if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
            raise MycotrophSessionError(
                "invalid_deck_output",
                f"deck {deck_name} did not produce canonical RGB8",
                deck=deck_name,
                shape=list(result.shape),
                dtype=str(result.dtype),
            )
        result = np.ascontiguousarray(result).copy()

        if pipeline:
            limit = self._buffer_limit(pipeline, runtime.state.buffer_limit)
            runtime.state.frame_buffer.append(result.copy())
            while len(runtime.state.frame_buffer) > limit:
                runtime.state.frame_buffer.pop(0)
            runtime.state.buffer_limit = limit
        runtime.state.frame_count += 1
        return result

    def render(
        self,
        primary: Any,
        secondary: Any,
        sequence: int,
        time_seconds: float,
        macros: Mapping[str, Any] | None = None,
    ) -> SessionRender:
        """Render both decks and their master program from caller-owned frames."""

        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise MycotrophSessionError(
                "invalid_sequence",
                "sequence must be a non-negative integer",
                received=sequence,
            )
        if isinstance(time_seconds, bool) or not isinstance(time_seconds, (int, float)):
            raise MycotrophSessionError(
                "invalid_time",
                "time_seconds must be numeric",
                received_type=type(time_seconds).__name__,
            )
        time_value = float(time_seconds)
        if not math.isfinite(time_value):
            raise MycotrophSessionError(
                "invalid_time",
                "time_seconds must be finite",
            )
        if macros is not None and not isinstance(macros, Mapping):
            raise MycotrophSessionError(
                "invalid_macros",
                "macros must be a JSON object or null",
                received_type=type(macros).__name__,
            )
        try:
            render_macros = _json_detach(dict(macros or {}))
        except (TypeError, ValueError) as error:
            raise MycotrophSessionError(
                "invalid_macros",
                "macros must contain finite JSON values",
                exception=type(error).__name__,
            ) from error

        with self._lock:
            if self._accepted is None:
                raise MycotrophSessionError(
                    "session_not_ready",
                    "a session must be committed before rendering",
                )

            try:
                primary_rgb = (
                    None if primary is None else _canonical_rgb(primary, self._fallback_size)
                )
                secondary_rgb = (
                    None if secondary is None else _canonical_rgb(secondary, self._fallback_size)
                )
            except Exception as error:
                raise MycotrophSessionError(
                    "invalid_source_frame",
                    str(error),
                    exception=type(error).__name__,
                ) from error

            available = primary_rgb if primary_rgb is not None else secondary_rgb
            fallback_size = (
                (int(available.shape[1]), int(available.shape[0]))
                if available is not None
                else self._fallback_size
            )
            extra_params = _thaw_json(self._accepted.macros)
            extra_params.update(render_macros)
            extra_params["time_accum"] = time_value
            extra_params.setdefault("noise_warp_time", time_value)
            extra_params["frame_sequence"] = int(sequence)

            deck_a = self._render_deck(
                "a", primary_rgb, secondary_rgb, extra_params, fallback_size
            )
            deck_b = self._render_deck(
                "b", primary_rgb, secondary_rgb, extra_params, fallback_size
            )
            try:
                program = blend_decks(
                    deck_a,
                    deck_b,
                    self._accepted.master_mix,
                    self._accepted.master_transition,
                    fallback_size=fallback_size,
                )
            except Exception as error:
                raise MycotrophSessionError(
                    "master_render_failed",
                    f"master transition failed: {error}",
                    transition=self._accepted.master_transition,
                    exception=type(error).__name__,
                ) from error
            program = np.asarray(program)
            if program.dtype != np.uint8 or program.ndim != 3 or program.shape[2] != 3:
                raise MycotrophSessionError(
                    "invalid_program_output",
                    "master transition did not produce canonical RGB8",
                    shape=list(program.shape),
                    dtype=str(program.dtype),
                )
            program = np.ascontiguousarray(program).copy()
            self._last_sequence = int(sequence)

            telemetry = {
                "backend": "python_flat_dual_deck",
                "session_revision": self._session_revision,
                "session_sha256": self._accepted.sha256,
                "sequence": int(sequence),
                "time_seconds": time_value,
                "macros_sha256": _sha256_json(extra_params),
                "master": {
                    "mix": self._accepted.master_mix,
                    "transition": self._accepted.master_transition,
                },
                "decks": {
                    "a": {
                        "graph_generation": self._decks["a"].generation,
                        "executable_sha256": self._decks["a"].prepared.executable_sha256,
                        "frame": _frame_description(deck_a),
                        "temporal": self._temporal_description(self._decks["a"].state),
                    },
                    "b": {
                        "graph_generation": self._decks["b"].generation,
                        "executable_sha256": self._decks["b"].prepared.executable_sha256,
                        "frame": _frame_description(deck_b),
                        "temporal": self._temporal_description(self._decks["b"].state),
                    },
                },
                "program": _frame_description(program),
            }
            return SessionRender(
                deck_a=deck_a,
                deck_b=deck_b,
                program=program,
                telemetry=_json_detach(telemetry),
            )


__all__ = [
    "MASTER_TRANSITION_MODES",
    "MycotrophSessionError",
    "MycotrophSessionRuntime",
    "PreparedSession",
    "SessionRender",
]
