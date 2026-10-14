"""Headless, deterministic automation evaluation for Mycotroph hosts.

The legacy automation classes are coupled to Dear PyGui and live device
providers.  This module is the serialization/runtime boundary for a new host:
it consumes a canonical ``megaglitch.session`` v2 document, compiles its
timeline and modulation configuration without opening a UI or a device, and
evaluates immutable parameter patches from recorded source snapshots.

Evaluation order is deliberate and stable:

1. configured time-domain macro values form the global baseline;
2. timeline tracks replace their target value using ``ParameterTimeline``'s
   linear/ease/step rules;
3. modulation routes are applied in document order using
   ``signal * depth + offset`` and the requested blend/clamp policy.

Targets that cannot be resolved against the session's deck/node/parameter
inventory are preserved in the configuration snapshot and never applied.
Replacing configuration is transactional: a structurally invalid document
returns structured errors and leaves the last accepted revision active.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import threading
from types import MappingProxyType
from typing import Any, TypeVar

from megaglitch.session_document import (
    SESSION_FORMAT,
    SESSION_FORMAT_VERSION,
    normalize_session_document,
)


_T = TypeVar("_T")


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, bool, float, type(None))):
        return value
    raise TypeError(f"unsupported immutable snapshot value: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class FrozenMap(Mapping[str, Any]):
    """A recursively immutable mapping used by public service snapshots."""

    __slots__ = ("__proxy", "__hash")

    def __init__(self, values: Mapping[str, Any] | None = None):
        frozen = {str(key): _freeze(value) for key, value in (values or {}).items()}
        object.__setattr__(self, "_FrozenMap__proxy", MappingProxyType(frozen))
        object.__setattr__(self, "_FrozenMap__hash", None)

    def __getitem__(self, key: str) -> Any:
        return self.__proxy[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__proxy)

    def __len__(self) -> int:
        return len(self.__proxy)

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self.__proxy)!r})"

    def __hash__(self) -> int:
        cached = self.__hash
        if cached is None:
            cached = hash(tuple(sorted((key, _hashable(value)) for key, value in self.items())))
            object.__setattr__(self, "_FrozenMap__hash", cached)
        return cached

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self)


def _hashable(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return hash(value)
    if isinstance(value, tuple):
        return tuple(_hashable(item) for item in value)
    return value


@dataclass(frozen=True, order=True)
class AutomationIssue:
    """Stable machine-readable validation or evaluation diagnostic."""

    code: str
    path: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ConfigUpdate:
    """Result of a transactional configuration replacement."""

    accepted: bool
    revision: int
    errors: tuple[AutomationIssue, ...] = ()
    warnings: tuple[AutomationIssue, ...] = ()

    @property
    def issues(self) -> tuple[AutomationIssue, ...]:
        return self.errors + self.warnings


@dataclass(frozen=True)
class RecordedSourceSnapshot:
    """Recorded automation inputs; every field is optional and device-free.

    MIDI note and controller values use their native 0..127 range.  Routes see
    normalized 0..1 values for the legacy ``MIDI Note`` and ``MIDI CC`` source
    names.  ``visual`` accepts the existing VisualMetrics names as well as
    future numeric metrics.
    """

    audio_bands: FrozenMap = field(default_factory=FrozenMap)
    onset_value: float | None = None
    onset_event_id: int | None = None
    bpm_value: float | None = None
    bpm_normalized: float | None = None
    bpm_phase: float | None = None
    bpm_pulse: float | None = None
    beat_event_id: int | None = None
    midi_note: float | None = None
    midi_velocity: float | None = None
    midi_cc_number: int | None = None
    midi_cc_value: float | None = None
    midi_cc: FrozenMap = field(default_factory=FrozenMap)
    visual: FrozenMap = field(default_factory=FrozenMap)
    extra: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        for name in ("audio_bands", "midi_cc", "visual", "extra"):
            value = getattr(self, name)
            if not isinstance(value, FrozenMap):
                object.__setattr__(self, name, FrozenMap(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RecordedSourceSnapshot":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("recorded source snapshot must be a mapping")

        audio = value.get("audio") if isinstance(value.get("audio"), Mapping) else {}
        bpm = value.get("bpm") if isinstance(value.get("bpm"), Mapping) else {}
        midi = value.get("midi") if isinstance(value.get("midi"), Mapping) else {}
        visual_value = value.get("visual", value.get("visual_metrics", {}))
        if hasattr(visual_value, "as_dict"):
            visual_value = visual_value.as_dict()
        if not isinstance(visual_value, Mapping):
            visual_value = {}

        bands = value.get("audio_bands", audio.get("bands", value.get("fft_bands", {})))
        controllers = value.get("midi_cc", midi.get("controllers", {}))
        if not isinstance(controllers, Mapping):
            controllers = {}
        extra = value.get("extra", value.get("sources", {}))
        if not isinstance(extra, Mapping):
            extra = {}

        latest_cc = midi.get("cc_value", value.get("midi_cc_value"))
        if latest_cc is None and "value" in midi and "cc_number" in midi:
            latest_cc = midi.get("value")
        return cls(
            audio_bands=FrozenMap(bands if isinstance(bands, Mapping) else {}),
            onset_value=_optional_float(value.get("onset_value", audio.get("onset"))),
            onset_event_id=_optional_int(value.get("onset_event_id", audio.get("onset_event_id"))),
            bpm_value=_optional_float(value.get("bpm_value", bpm.get("value"))),
            bpm_normalized=_optional_float(value.get("bpm_normalized", bpm.get("normalized"))),
            bpm_phase=_optional_float(value.get("bpm_phase", bpm.get("phase"))),
            bpm_pulse=_optional_float(value.get("bpm_pulse", bpm.get("pulse"))),
            beat_event_id=_optional_int(value.get("beat_event_id", bpm.get("event_id"))),
            midi_note=_optional_float(value.get("midi_note", midi.get("note"))),
            midi_velocity=_optional_float(value.get("midi_velocity", midi.get("velocity"))),
            midi_cc_number=_optional_int(value.get("midi_cc_number", midi.get("cc_number"))),
            midi_cc_value=_optional_float(latest_cc),
            midi_cc=FrozenMap(controllers),
            visual=FrozenMap(visual_value),
            extra=FrozenMap(extra),
        )


@dataclass(frozen=True)
class AutomationSnapshot:
    """One immutable fixed-time evaluation result."""

    config_revision: int
    sequence: int
    time_seconds: float
    panicked: bool
    decks: FrozenMap
    global_macros: FrozenMap
    unresolved_tracks: tuple[Any, ...]
    unresolved_routes: tuple[Any, ...]
    issues: tuple[AutomationIssue, ...] = ()

    @property
    def deck_a(self) -> FrozenMap:
        value = self.decks.get("a")
        return value if isinstance(value, FrozenMap) else FrozenMap()

    @property
    def deck_b(self) -> FrozenMap:
        value = self.decks.get("b")
        return value if isinstance(value, FrozenMap) else FrozenMap()

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_revision": self.config_revision,
            "sequence": self.sequence,
            "time_seconds": self.time_seconds,
            "panicked": self.panicked,
            "decks": self.decks.to_dict(),
            "global_macros": self.global_macros.to_dict(),
            "unresolved_tracks": _thaw(self.unresolved_tracks),
            "unresolved_routes": _thaw(self.unresolved_routes),
            "issues": [item.as_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class AutomationConfigurationSnapshot:
    revision: int
    session: FrozenMap
    unresolved_tracks: tuple[Any, ...]
    unresolved_routes: tuple[Any, ...]
    warnings: tuple[AutomationIssue, ...]


@dataclass(frozen=True, order=True)
class _Address:
    scope: str
    deck: str = ""
    node_id: str = ""
    parameter: str = ""


@dataclass(frozen=True)
class _ParameterMeta:
    value_type: str
    base: Any
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class _Keyframe:
    time: float
    value: Any
    interpolation: str


@dataclass(frozen=True)
class _Track:
    track_id: str
    address: _Address
    value_type: str
    keyframes: tuple[_Keyframe, ...]


@dataclass(frozen=True)
class _Route:
    route_id: str
    source: str
    addresses: tuple[_Address, ...]
    enabled: bool
    depth: float
    offset: float
    blend_mode: str
    source_range: tuple[float, float]
    target_range: tuple[float, float] | None
    clamp: tuple[float | None, float | None] | None
    curve: str
    steps: int
    deadzone: float
    bipolar: bool
    raw: FrozenMap


@dataclass(frozen=True)
class _CompiledConfig:
    session: FrozenMap
    inventory: FrozenMap
    macro_meta: FrozenMap
    tracks: tuple[_Track, ...]
    routes: tuple[_Route, ...]
    timeline_enabled: bool
    modulation_enabled: bool
    unresolved_tracks: tuple[Any, ...]
    unresolved_routes: tuple[Any, ...]
    warnings: tuple[AutomationIssue, ...]


_TIME_MACROS: dict[str, tuple[float, float, float, float]] = {
    # name: (default, minimum, maximum, legacy modulation span)
    "history_offset": (0.0, 0.0, 32.0, 8.0),
    "memory_decay": (0.0, 0.0, 0.999, 0.08),
    "event_hazard": (0.0, 0.0, 1.0, 0.25),
    "clock_quantization": (0.0, 0.0, 0.25, 0.08),
    "sync_desync": (0.0, -1.0, 1.0, 0.6),
    "phase_reset_chance": (0.0, 0.0, 0.5, 0.12),
    "local_time_warp": (0.0, -2.0, 2.0, 1.0),
    "rupture_threshold": (0.0, 0.0, 2.0, 0.6),
    "hysteresis_speed": (0.0, 0.0, 1.0, 0.25),
    "modulator_coupling": (0.0, 0.0, 1.0, 0.4),
    "time_reversal": (0.0, 0.0, 1.0, 1.0),
}

_CURVES = {"linear", "smoothstep", "ease_in_out", "exponential", "logarithmic", "stepped"}
_BLEND_MODES = {"replace", "add", "offset", "multiply", "max", "min"}


class MycotrophAutomationService:
    """Thread-safe owner of one accepted automation configuration."""

    def __init__(self, session: Mapping[str, Any] | None = None):
        self._lock = threading.RLock()
        self._config = _empty_compiled_config()
        self._revision = 0
        self._sequence = 0
        self._panicked = False
        self._last_rejection: ConfigUpdate | None = None
        self._last_snapshot = _empty_snapshot()
        if session is not None:
            self.replace_config(session)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def panicked(self) -> bool:
        with self._lock:
            return self._panicked

    @property
    def last_rejection(self) -> ConfigUpdate | None:
        with self._lock:
            return self._last_rejection

    def replace_config(self, session: Mapping[str, Any]) -> ConfigUpdate:
        """Compile and atomically replace a canonical session-v2 configuration."""

        compiled, errors = _compile_session(session)
        if errors:
            with self._lock:
                result = ConfigUpdate(False, self._revision, tuple(errors), ())
                self._last_rejection = result
                return result
        assert compiled is not None
        with self._lock:
            self._revision += 1
            self._config = compiled
            self._panicked = False
            self._sequence = 0
            self._last_rejection = None
            self._last_snapshot = self._evaluate_locked(0.0, RecordedSourceSnapshot(), sequence=0)
            return ConfigUpdate(True, self._revision, (), compiled.warnings)

    replace_configuration = replace_config

    def evaluate(
        self,
        time_seconds: float,
        sources: RecordedSourceSnapshot | Mapping[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Evaluate a deterministic patch at an explicit session time."""

        seconds = _finite_float(time_seconds, "time_seconds")
        if seconds < 0.0:
            raise ValueError("time_seconds must be non-negative")
        source_snapshot = (
            sources
            if isinstance(sources, RecordedSourceSnapshot)
            else RecordedSourceSnapshot.from_mapping(sources)
        )
        with self._lock:
            self._sequence += 1
            self._last_snapshot = self._evaluate_locked(seconds, source_snapshot, self._sequence)
            return self._last_snapshot

    def panic(self) -> AutomationSnapshot:
        """Suppress all patches until reset or a configuration replacement."""

        with self._lock:
            self._panicked = True
            self._sequence += 1
            self._last_snapshot = AutomationSnapshot(
                config_revision=self._revision,
                sequence=self._sequence,
                time_seconds=self._last_snapshot.time_seconds,
                panicked=True,
                decks=FrozenMap({"a": {}, "b": {}}),
                global_macros=FrozenMap(),
                unresolved_tracks=self._config.unresolved_tracks,
                unresolved_routes=self._config.unresolved_routes,
                issues=(
                    AutomationIssue(
                        "automation.panicked",
                        "/runtime",
                        "automation output is suppressed until reset",
                        "warning",
                    ),
                ),
            )
            return self._last_snapshot

    def reset(self) -> AutomationSnapshot:
        """Clear panic/runtime state and deterministically re-evaluate time zero."""

        with self._lock:
            self._panicked = False
            self._sequence = 0
            self._last_snapshot = self._evaluate_locked(0.0, RecordedSourceSnapshot(), sequence=0)
            return self._last_snapshot

    def snapshot(self) -> AutomationSnapshot:
        with self._lock:
            return self._last_snapshot

    def configuration_snapshot(self) -> AutomationConfigurationSnapshot:
        with self._lock:
            return AutomationConfigurationSnapshot(
                revision=self._revision,
                session=self._config.session,
                unresolved_tracks=self._config.unresolved_tracks,
                unresolved_routes=self._config.unresolved_routes,
                warnings=self._config.warnings,
            )

    def _evaluate_locked(
        self,
        seconds: float,
        sources: RecordedSourceSnapshot,
        sequence: int,
    ) -> AutomationSnapshot:
        config = self._config
        if self._panicked:
            return AutomationSnapshot(
                self._revision,
                sequence,
                seconds,
                True,
                FrozenMap({"a": {}, "b": {}}),
                FrozenMap(),
                config.unresolved_tracks,
                config.unresolved_routes,
                (),
            )

        values: dict[_Address, Any] = {}
        touched: set[_Address] = set()
        macro_values: dict[str, Any] = {
            name: meta.base for name, meta in config.macro_meta.items()
        }
        for name, value in macro_values.items():
            values[_Address("global_macro", parameter=name)] = value

        if config.timeline_enabled:
            for track in config.tracks:
                value = _evaluate_track(track, seconds)
                if value is None:
                    continue
                meta = _meta_for_address(config, track.address)
                value = _coerce_result(value, meta or _ParameterMeta(track.value_type, value))
                values[track.address] = value
                touched.add(track.address)

        runtime_unresolved: list[Any] = list(config.unresolved_routes)
        runtime_issues: list[AutomationIssue] = []
        if config.modulation_enabled:
            for index, route in enumerate(config.routes):
                if not route.enabled:
                    continue
                source_value = _resolve_source(route.source, sources, seconds, route.route_id)
                if source_value is None:
                    runtime_unresolved.append(route.raw)
                    runtime_issues.append(
                        AutomationIssue(
                            "modulation.source_unavailable",
                            f"/modulation/routes/{index}/source",
                            f"recorded source {route.source!r} is unavailable",
                            "warning",
                        )
                    )
                    continue
                signal = _shape_source(source_value, route)
                for address in route.addresses:
                    meta = _meta_for_address(config, address)
                    base = values.get(address, meta.base if meta is not None else 0.0)
                    result = _blend_value(base, signal, route.blend_mode)
                    clamp = route.clamp
                    if clamp is None and meta is not None:
                        clamp = (meta.minimum, meta.maximum)
                    result = _clamp_optional(result, clamp)
                    values[address] = _coerce_result(result, meta)
                    touched.add(address)

        decks: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
        for address in sorted(touched):
            value = values[address]
            if address.scope == "global_macro":
                macro_values[address.parameter] = value
                continue
            deck = decks.setdefault(address.deck, {})
            deck.setdefault(address.node_id, {})[address.parameter] = value

        # Global macros are a complete baseline, not a sparse patch.  This
        # mirrors the legacy worker, which receives every time-domain macro on
        # each frame and lets node-local parameters override those values.
        for name in tuple(macro_values):
            address = _Address("global_macro", parameter=name)
            if address in values:
                macro_values[name] = values[address]

        return AutomationSnapshot(
            config_revision=self._revision,
            sequence=sequence,
            time_seconds=seconds,
            panicked=False,
            decks=FrozenMap(decks),
            global_macros=FrozenMap(macro_values),
            unresolved_tracks=config.unresolved_tracks,
            unresolved_routes=tuple(_freeze(item) for item in runtime_unresolved),
            issues=tuple(runtime_issues),
        )


def _compile_session(
    session: Mapping[str, Any],
) -> tuple[_CompiledConfig | None, list[AutomationIssue]]:
    errors: list[AutomationIssue] = []
    if not isinstance(session, Mapping):
        return None, [
            AutomationIssue("config.type", "/", "session must be a JSON object")
        ]
    if session.get("format") != SESSION_FORMAT:
        return None, [
            AutomationIssue(
                "config.format",
                "/format",
                f"expected {SESSION_FORMAT!r}",
            )
        ]
    if str(session.get("format_version")) not in {"2", "2.0"}:
        return None, [
            AutomationIssue(
                "config.version",
                "/format_version",
                f"expected canonical format version {SESSION_FORMAT_VERSION}",
            )
        ]
    try:
        normalized = normalize_session_document(session)
    except (TypeError, ValueError, OverflowError) as exc:
        return None, [AutomationIssue("config.invalid_document", "/", str(exc))]

    inventory, ambiguous_nodes, inventory_issues = _build_inventory(normalized.get("decks"))
    warnings = list(inventory_issues)
    macros, macro_meta = _compile_macros(normalized.get("macros"), errors)
    tracks, unresolved_tracks, timeline_enabled = _compile_timeline(
        normalized.get("timeline"),
        inventory,
        ambiguous_nodes,
        macro_meta,
        errors,
        warnings,
    )
    routes, unresolved_routes, modulation_enabled = _compile_modulation(
        normalized.get("modulation"),
        inventory,
        ambiguous_nodes,
        macro_meta,
        errors,
        warnings,
    )
    if errors:
        return None, errors

    # Store configured macro values as the immutable meta baseline.
    macro_meta = {
        name: _ParameterMeta(meta.value_type, macros[name], meta.minimum, meta.maximum)
        for name, meta in macro_meta.items()
    }
    return (
        _CompiledConfig(
            session=FrozenMap(normalized),
            inventory=FrozenMap(inventory),
            macro_meta=FrozenMap(macro_meta),
            tracks=tuple(tracks),
            routes=tuple(routes),
            timeline_enabled=timeline_enabled,
            modulation_enabled=modulation_enabled,
            unresolved_tracks=tuple(_freeze(item) for item in unresolved_tracks),
            unresolved_routes=tuple(_freeze(item) for item in unresolved_routes),
            warnings=tuple(warnings),
        ),
        [],
    )


def _build_inventory(
    decks_value: Any,
) -> tuple[dict[str, Any], set[tuple[str, str]], list[AutomationIssue]]:
    inventory: dict[str, dict[str, dict[str, _ParameterMeta]]] = {}
    ambiguous: set[tuple[str, str]] = set()
    issues: list[AutomationIssue] = []
    if not isinstance(decks_value, Mapping):
        return inventory, ambiguous, issues
    for raw_deck, deck_value in decks_value.items():
        deck = str(raw_deck)
        inventory[deck] = {}
        graph = deck_value.get("graph") if isinstance(deck_value, Mapping) else None
        nodes = graph.get("nodes", []) if isinstance(graph, Mapping) else []
        if not isinstance(nodes, list):
            continue
        counts: dict[str, int] = {}
        for index, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            counts[node_id] = counts.get(node_id, 0) + 1
            if counts[node_id] > 1:
                ambiguous.add((deck, node_id))
                inventory[deck].pop(node_id, None)
                issues.append(
                    AutomationIssue(
                        "inventory.ambiguous_node_id",
                        f"/decks/{_pointer(deck)}/graph/nodes/{index}/id",
                        f"node id {node_id!r} is not unique in deck {deck!r}",
                        "warning",
                    )
                )
                continue
            params = node.get("params")
            if not isinstance(params, Mapping):
                params = {}
            specs = _parameter_specs(node)
            node_params: dict[str, _ParameterMeta] = {}
            for raw_name, base in params.items():
                name = str(raw_name)
                spec = specs.get(name, {})
                value_type = _value_type(spec.get("type"), base)
                minimum = _optional_finite(spec.get("min_value", spec.get("min")))
                maximum = _optional_finite(spec.get("max_value", spec.get("max")))
                node_params[name] = _ParameterMeta(value_type, base, minimum, maximum)
            inventory[deck][node_id] = node_params
    return inventory, ambiguous, issues


def _parameter_specs(node: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = node.get("parameter_specs", node.get("params_def", node.get("param_specs", {})))
    if isinstance(raw, Mapping):
        return {
            str(name): value
            for name, value in raw.items()
            if isinstance(value, Mapping)
        }
    if isinstance(raw, list):
        return {
            str(item["name"]): item
            for item in raw
            if isinstance(item, Mapping) and str(item.get("name", "")).strip()
        }
    return {}


def _compile_macros(
    value: Any,
    errors: list[AutomationIssue],
) -> tuple[dict[str, Any], dict[str, _ParameterMeta]]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        errors.append(AutomationIssue("macros.type", "/macros", "macros must be an object"))
        value = {}
    result: dict[str, Any] = {}
    meta: dict[str, _ParameterMeta] = {}
    for name, (default, minimum, maximum, _span) in _TIME_MACROS.items():
        raw = value.get(name, default)
        try:
            number = _finite_float(raw, name)
        except (TypeError, ValueError) as exc:
            errors.append(AutomationIssue("macros.value", f"/macros/{_pointer(name)}", str(exc)))
            number = default
        number = min(maximum, max(minimum, number))
        result[name] = number
        meta[name] = _ParameterMeta("float", number, minimum, maximum)
    for raw_name, raw in value.items():
        name = str(raw_name)
        if name in result:
            continue
        try:
            number = _finite_float(raw, name)
        except (TypeError, ValueError) as exc:
            errors.append(AutomationIssue("macros.value", f"/macros/{_pointer(name)}", str(exc)))
            continue
        result[name] = number
        meta[name] = _ParameterMeta("float", number)
    return result, meta


def _compile_timeline(
    value: Any,
    inventory: Mapping[str, Any],
    ambiguous_nodes: set[tuple[str, str]],
    macro_meta: Mapping[str, _ParameterMeta],
    errors: list[AutomationIssue],
    warnings: list[AutomationIssue],
) -> tuple[list[_Track], list[Any], bool]:
    if value is None:
        return [], [], False
    if not isinstance(value, Mapping):
        errors.append(AutomationIssue("timeline.type", "/timeline", "timeline must be an object or null"))
        return [], [], False
    enabled = bool(value.get("enabled", False))
    if value.get("timebase", "seconds") != "seconds":
        errors.append(
            AutomationIssue(
                "timeline.timebase",
                "/timeline/timebase",
                "only the deterministic seconds timebase is supported",
            )
        )
    raw_tracks = value.get("tracks", [])
    if not isinstance(raw_tracks, list):
        errors.append(AutomationIssue("timeline.tracks_type", "/timeline/tracks", "tracks must be an array"))
        raw_tracks = []
    unresolved = list(value.get("unresolved_tracks", [])) if isinstance(value.get("unresolved_tracks", []), list) else []
    tracks: list[_Track] = []
    target_ids: set[_Address] = set()
    track_ids: set[str] = set()
    for index, raw in enumerate(raw_tracks):
        path = f"/timeline/tracks/{index}"
        if not isinstance(raw, Mapping):
            errors.append(AutomationIssue("timeline.track_type", path, "track must be an object"))
            continue
        track_id = str(raw.get("id", f"track:{index + 1}")).strip() or f"track:{index + 1}"
        if track_id in track_ids:
            errors.append(AutomationIssue("timeline.duplicate_id", f"{path}/id", f"duplicate track id {track_id!r}"))
            continue
        track_ids.add(track_id)
        address = _timeline_address(raw.get("target"))
        if address is None or not _address_exists(address, inventory, ambiguous_nodes, macro_meta):
            unresolved.append(raw)
            warnings.append(
                AutomationIssue(
                    "timeline.target_unresolved",
                    f"{path}/target",
                    "track target is absent or ambiguous in the accepted inventory",
                    "warning",
                )
            )
            continue
        if address in target_ids:
            errors.append(AutomationIssue("timeline.duplicate_target", f"{path}/target", "multiple tracks target the same parameter"))
            continue
        target_ids.add(address)
        value_type = str(raw.get("value_type", "float"))
        if value_type not in {"float", "int", "checkbox"}:
            errors.append(AutomationIssue("timeline.value_type", f"{path}/value_type", "value_type must be float, int, or checkbox"))
            continue
        raw_keyframes = raw.get("keyframes", [])
        if not isinstance(raw_keyframes, list):
            errors.append(AutomationIssue("timeline.keyframes_type", f"{path}/keyframes", "keyframes must be an array"))
            continue
        keyframes: list[_Keyframe] = []
        seen_times: set[float] = set()
        for key_index, raw_keyframe in enumerate(raw_keyframes):
            key_path = f"{path}/keyframes/{key_index}"
            if not isinstance(raw_keyframe, Mapping):
                errors.append(AutomationIssue("timeline.keyframe_type", key_path, "keyframe must be an object"))
                continue
            try:
                key_time = _finite_float(raw_keyframe.get("time", 0.0), "keyframe time")
                if key_time < 0.0:
                    raise ValueError("keyframe time must be non-negative")
            except (TypeError, ValueError) as exc:
                errors.append(AutomationIssue("timeline.keyframe_time", f"{key_path}/time", str(exc)))
                continue
            if key_time in seen_times:
                errors.append(AutomationIssue("timeline.duplicate_time", f"{key_path}/time", "duplicate keyframe time"))
                continue
            seen_times.add(key_time)
            interpolation = str(raw_keyframe.get("interpolation", raw_keyframe.get("interp", "linear")))
            if interpolation not in {"linear", "ease", "step"}:
                errors.append(AutomationIssue("timeline.interpolation", f"{key_path}/interpolation", "interpolation must be linear, ease, or step"))
                continue
            try:
                key_value = _coerce_keyframe_value(raw_keyframe.get("value"), value_type)
            except (TypeError, ValueError) as exc:
                errors.append(AutomationIssue("timeline.keyframe_value", f"{key_path}/value", str(exc)))
                continue
            keyframes.append(_Keyframe(key_time, key_value, interpolation))
        keyframes.sort(key=lambda item: item.time)
        tracks.append(_Track(track_id, address, value_type, tuple(keyframes)))
    return tracks, unresolved, enabled


def _compile_modulation(
    value: Any,
    inventory: Mapping[str, Any],
    ambiguous_nodes: set[tuple[str, str]],
    macro_meta: Mapping[str, _ParameterMeta],
    errors: list[AutomationIssue],
    warnings: list[AutomationIssue],
) -> tuple[list[_Route], list[Any], bool]:
    if value is None:
        return [], [], False
    if not isinstance(value, Mapping):
        errors.append(AutomationIssue("modulation.type", "/modulation", "modulation must be an object or null"))
        return [], [], False
    enabled = bool(value.get("enabled", False))
    raw_routes = value.get("routes", [])
    if not isinstance(raw_routes, list):
        errors.append(AutomationIssue("modulation.routes_type", "/modulation/routes", "routes must be an array"))
        raw_routes = []
    legacy_unresolved = value.get("unresolved_legacy_connections", [])
    unresolved = list(legacy_unresolved) if isinstance(legacy_unresolved, list) else [legacy_unresolved]
    routes: list[_Route] = []
    route_ids: set[str] = set()
    for index, raw in enumerate(raw_routes):
        path = f"/modulation/routes/{index}"
        if not isinstance(raw, Mapping):
            errors.append(AutomationIssue("modulation.route_type", path, "route must be an object"))
            continue
        route_id = str(raw.get("id", f"route:{index + 1}")).strip() or f"route:{index + 1}"
        if route_id in route_ids:
            errors.append(AutomationIssue("modulation.duplicate_id", f"{path}/id", f"duplicate route id {route_id!r}"))
            continue
        route_ids.add(route_id)
        source = str(raw.get("source", "")).strip()
        if not source:
            errors.append(AutomationIssue("modulation.source", f"{path}/source", "route source must be a non-empty string"))
            continue
        addresses = _route_addresses(raw.get("target"), inventory, ambiguous_nodes, macro_meta)
        if not addresses:
            unresolved.append(raw)
            warnings.append(
                AutomationIssue(
                    "modulation.target_unresolved",
                    f"{path}/target",
                    "route target is absent or ambiguous in the accepted inventory",
                    "warning",
                )
            )
            continue
        try:
            depth = _finite_float(raw.get("depth", 1.0), "route depth")
            offset = _finite_float(raw.get("offset", 0.0), "route offset")
            source_range = _number_pair(raw.get("source_range", (0.0, 1.0)), "source_range")
            target_range = (
                None
                if raw.get("target_range") is None
                else _number_pair(raw.get("target_range"), "target_range")
            )
            clamp = _parse_clamp(raw.get("clamp"))
            deadzone = _finite_float(raw.get("deadzone", 0.0), "deadzone")
            if deadzone < 0.0 or deadzone > 1.0:
                raise ValueError("deadzone must be between 0 and 1")
            steps = int(raw.get("steps", 2))
            if steps < 2:
                raise ValueError("steps must be at least 2")
        except (TypeError, ValueError) as exc:
            errors.append(AutomationIssue("modulation.route_value", path, str(exc)))
            continue
        blend_mode = str(raw.get("blend_mode", raw.get("mode", "add"))).lower()
        if blend_mode not in _BLEND_MODES:
            errors.append(AutomationIssue("modulation.blend_mode", f"{path}/blend_mode", f"unsupported blend mode {blend_mode!r}"))
            continue
        curve = str(raw.get("curve", "linear")).lower()
        if curve not in _CURVES:
            errors.append(AutomationIssue("modulation.curve", f"{path}/curve", f"unsupported curve {curve!r}"))
            continue
        routes.append(
            _Route(
                route_id=route_id,
                source=source,
                addresses=tuple(addresses),
                enabled=bool(raw.get("enabled", True)),
                depth=depth,
                offset=offset,
                blend_mode=blend_mode,
                source_range=source_range,
                target_range=target_range,
                clamp=clamp,
                curve=curve,
                steps=steps,
                deadzone=deadzone,
                bipolar=bool(raw.get("bipolar", False)),
                raw=FrozenMap(raw),
            )
        )
    return routes, unresolved, enabled


def _timeline_address(target: Any) -> _Address | None:
    if not isinstance(target, Mapping):
        return None
    scope = str(target.get("scope", "node_parameter"))
    parameter = str(target.get("parameter", "")).strip()
    if not parameter:
        return None
    if scope == "global_macro":
        return _Address("global_macro", parameter=parameter)
    deck = str(target.get("deck", "")).strip()
    node_id = str(target.get("node_id", "")).strip()
    if not deck or not node_id:
        return None
    return _Address("node_parameter", deck, node_id, parameter)


def _route_addresses(
    target: Any,
    inventory: Mapping[str, Any],
    ambiguous_nodes: set[tuple[str, str]],
    macro_meta: Mapping[str, _ParameterMeta],
) -> list[_Address]:
    if not isinstance(target, Mapping):
        return []
    scope = str(target.get("scope", "node_parameter"))
    parameter = str(target.get("parameter", "")).strip()
    if not parameter:
        return []
    if scope == "global_macro":
        address = _Address("global_macro", parameter=parameter)
        return [address] if _address_exists(address, inventory, ambiguous_nodes, macro_meta) else []
    deck = str(target.get("deck", "")).strip()
    if scope == "deck_parameter_name":
        nodes = inventory.get(deck, {}) if isinstance(inventory.get(deck, {}), Mapping) else {}
        return [
            _Address("node_parameter", deck, node_id, parameter)
            for node_id, params in nodes.items()
            if isinstance(params, Mapping) and parameter in params
        ]
    node_id = str(target.get("node_id", "")).strip()
    address = _Address("node_parameter", deck, node_id, parameter)
    return [address] if _address_exists(address, inventory, ambiguous_nodes, macro_meta) else []


def _address_exists(
    address: _Address,
    inventory: Mapping[str, Any],
    ambiguous_nodes: set[tuple[str, str]],
    macro_meta: Mapping[str, _ParameterMeta],
) -> bool:
    if address.scope == "global_macro":
        return address.parameter in macro_meta
    if (address.deck, address.node_id) in ambiguous_nodes:
        return False
    deck = inventory.get(address.deck)
    if not isinstance(deck, Mapping):
        return False
    node = deck.get(address.node_id)
    return isinstance(node, Mapping) and address.parameter in node


def _meta_for_address(config: _CompiledConfig, address: _Address) -> _ParameterMeta | None:
    if address.scope == "global_macro":
        value = config.macro_meta.get(address.parameter)
        return value if isinstance(value, _ParameterMeta) else None
    deck = config.inventory.get(address.deck)
    node = deck.get(address.node_id) if isinstance(deck, Mapping) else None
    value = node.get(address.parameter) if isinstance(node, Mapping) else None
    return value if isinstance(value, _ParameterMeta) else None


def _evaluate_track(track: _Track, seconds: float) -> Any | None:
    """Match ParameterTimeline.evaluate_track, without importing Dear PyGui."""

    keyframes = track.keyframes
    if not keyframes:
        return None
    if seconds <= keyframes[0].time:
        return keyframes[0].value
    if seconds >= keyframes[-1].time:
        return keyframes[-1].value
    previous = keyframes[0]
    following = keyframes[-1]
    for index in range(1, len(keyframes)):
        if seconds <= keyframes[index].time:
            previous = keyframes[index - 1]
            following = keyframes[index]
            break
    interpolation = following.interpolation or previous.interpolation or "linear"
    if interpolation == "step" or track.value_type == "checkbox":
        return previous.value
    amount = 0.0 if following.time <= previous.time else (
        (seconds - previous.time) / (following.time - previous.time)
    )
    amount = min(1.0, max(0.0, amount))
    if interpolation == "ease":
        amount = amount * amount * (3.0 - 2.0 * amount)
    try:
        value = float(previous.value) * (1.0 - amount) + float(following.value) * amount
    except (TypeError, ValueError):
        return previous.value
    return int(round(value)) if track.value_type == "int" else value


def _resolve_source(
    name: str,
    snapshot: RecordedSourceSnapshot,
    seconds: float,
    route_id: str,
) -> float | None:
    if name in snapshot.extra:
        return _optional_float(snapshot.extra[name])
    lowered = name.strip().lower()
    for key, value in snapshot.extra.items():
        if key.lower() == lowered:
            return _optional_float(value)

    if lowered in {"time", "lfo"}:
        return (math.sin(seconds) + 1.0) * 0.5
    if lowered == "random":
        payload = f"{route_id}\0{seconds:.12f}".encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return integer / float((1 << 64) - 1)

    band_name = None
    if lowered.startswith("audio.band."):
        band_name = name[len("audio.band.") :]
    elif lowered.startswith("audio.bands."):
        band_name = name[len("audio.bands.") :]
    elif lowered.startswith("fft band "):
        band_name = name[len("FFT Band ") :]
        if band_name.isdigit():
            band_name = f"Band {int(band_name):02d}"
    elif lowered in {
        "sub", "bass", "low mid", "mid", "high mid", "presence", "brilliance", "air"
    }:
        band_name = name
    elif lowered == "bass amplitude":
        band_name = "Bass"
    if band_name is not None:
        value = _mapping_get_casefold(snapshot.audio_bands, band_name)
        return _optional_float(value)
    if lowered == "treble amplitude":
        values = [
            _optional_float(_mapping_get_casefold(snapshot.audio_bands, band))
            for band in ("Presence", "Brilliance", "Air")
        ]
        present = [value for value in values if value is not None]
        return max(present) if present else None

    if lowered in {"audio.onset", "onset", "onset energy", "onset trigger"}:
        return snapshot.onset_value
    if lowered in {"audio.onset_event_id", "onset event id"}:
        return None if snapshot.onset_event_id is None else float(snapshot.onset_event_id)
    if lowered in {"bpm", "bpm.value", "bpm value"}:
        return snapshot.bpm_value
    if lowered in {"bpm.normalized", "bpm (normalized)", "normalized bpm"}:
        if snapshot.bpm_normalized is not None:
            return snapshot.bpm_normalized
        if snapshot.bpm_value is None:
            return None
        return min(1.0, max(0.0, (snapshot.bpm_value - 60.0) / 140.0))
    if lowered in {"bpm.phase", "beat phase"}:
        return snapshot.bpm_phase
    if lowered in {"bpm.pulse", "beat pulse", "beat trigger"}:
        return snapshot.bpm_pulse
    if lowered in {"bpm.event_id", "beat event id"}:
        return None if snapshot.beat_event_id is None else float(snapshot.beat_event_id)

    if lowered in {"midi note", "midi.note_velocity", "midi.velocity"}:
        value = snapshot.midi_velocity
        return None if value is None else min(1.0, max(0.0, value / 127.0))
    if lowered == "midi.note":
        value = snapshot.midi_note
        return None if value is None else min(1.0, max(0.0, value / 127.0))
    if lowered in {"midi cc", "midi.cc"}:
        value = snapshot.midi_cc_value
        return None if value is None else min(1.0, max(0.0, value / 127.0))
    if lowered.startswith("midi.cc."):
        key = name.rsplit(".", 1)[-1]
        value = snapshot.midi_cc.get(key)
        if value is None:
            try:
                value = snapshot.midi_cc.get(str(int(key)))
            except ValueError:
                pass
        numeric = _optional_float(value)
        return None if numeric is None else min(1.0, max(0.0, numeric / 127.0))

    visual_name = name[len("visual.") :] if lowered.startswith("visual.") else name
    visual = _mapping_get_casefold(snapshot.visual, visual_name)
    return _optional_float(visual)


def _shape_source(source: float, route: _Route) -> float:
    low, high = route.source_range
    denominator = high - low
    normalized = 0.0 if abs(denominator) < 1e-12 else (source - low) / denominator
    normalized = min(1.0, max(0.0, normalized))
    if normalized < route.deadzone:
        normalized = 0.0
    shaped = _curve(normalized, route.curve, route.steps)
    if route.bipolar:
        shaped = shaped * 2.0 - 1.0
    if route.target_range is not None:
        target_low, target_high = route.target_range
        shaped = target_low + shaped * (target_high - target_low)
    return shaped * route.depth + route.offset


def _curve(value: float, curve: str, steps: int) -> float:
    # These curves intentionally match ModulationBus._curve.
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


def _blend_value(base: Any, signal: float, mode: str) -> float:
    try:
        numeric_base = float(base)
    except (TypeError, ValueError):
        numeric_base = 0.0
    if mode == "replace":
        return signal
    if mode in {"add", "offset"}:
        return numeric_base + signal
    if mode == "multiply":
        return numeric_base * signal
    if mode == "max":
        return max(numeric_base, signal)
    if mode == "min":
        return min(numeric_base, signal)
    return signal


def _coerce_result(value: Any, meta: _ParameterMeta | None) -> Any:
    if meta is None:
        return value
    value = _clamp_optional(value, (meta.minimum, meta.maximum))
    if meta.value_type == "checkbox":
        return bool(value)
    if meta.value_type == "int":
        return int(round(float(value)))
    return float(value)


def _coerce_keyframe_value(value: Any, value_type: str) -> Any:
    if value_type == "checkbox":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    number = _finite_float(value, "keyframe value")
    return int(round(number)) if value_type == "int" else number


def _clamp_optional(
    value: Any,
    bounds: tuple[float | None, float | None] | None,
) -> float:
    number = float(value)
    if bounds is None:
        return number
    low, high = bounds
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def _parse_clamp(value: Any) -> tuple[float | None, float | None] | None:
    if value is None or value is False:
        return None
    if value is True:
        return None
    if isinstance(value, Mapping):
        low = _optional_finite(value.get("min", value.get("minimum")))
        high = _optional_finite(value.get("max", value.get("maximum")))
    else:
        low, high = _number_pair(value, "clamp")
    if low is not None and high is not None and low > high:
        raise ValueError("clamp minimum must not exceed maximum")
    return low, high


def _number_pair(value: Any, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{label} must contain two numbers")
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"{label} must contain exactly two numbers")
    return _finite_float(items[0], label), _finite_float(items[1], label)


def _value_type(explicit: Any, base: Any) -> str:
    name = str(explicit or "").lower()
    if name in {"checkbox", "bool", "boolean"} or isinstance(base, bool):
        return "checkbox"
    if name in {"int", "integer"} or (isinstance(base, int) and not isinstance(base, bool)):
        return "int"
    return "float"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(value, "value")


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _mapping_get_casefold(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    lowered = key.casefold()
    for candidate, value in mapping.items():
        if candidate.casefold() == lowered:
            return value
    return None


def _pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _empty_session() -> dict[str, Any]:
    return {
        "format": SESSION_FORMAT,
        "format_version": SESSION_FORMAT_VERSION,
        "decks": {
            "a": {"graph": {"version": "1.0", "nodes": [], "links": []}},
            "b": {"graph": {"version": "1.0", "nodes": [], "links": []}},
        },
        "macros": {},
        "timeline": {"enabled": False, "timebase": "seconds", "tracks": [], "unresolved_tracks": []},
        "modulation": {"enabled": False, "routes": [], "unresolved_legacy_connections": []},
    }


def _empty_compiled_config() -> _CompiledConfig:
    compiled, errors = _compile_session(_empty_session())
    if errors or compiled is None:  # pragma: no cover - internal invariant
        raise RuntimeError(f"could not construct empty automation configuration: {errors!r}")
    return compiled


def _empty_snapshot() -> AutomationSnapshot:
    return AutomationSnapshot(
        config_revision=0,
        sequence=0,
        time_seconds=0.0,
        panicked=False,
        decks=FrozenMap({"a": {}, "b": {}}),
        global_macros=FrozenMap({name: spec[0] for name, spec in _TIME_MACROS.items()}),
        unresolved_tracks=(),
        unresolved_routes=(),
        issues=(),
    )


AutomationService = MycotrophAutomationService


__all__ = [
    "AutomationConfigurationSnapshot",
    "AutomationIssue",
    "AutomationService",
    "AutomationSnapshot",
    "ConfigUpdate",
    "FrozenMap",
    "MycotrophAutomationService",
    "RecordedSourceSnapshot",
]
