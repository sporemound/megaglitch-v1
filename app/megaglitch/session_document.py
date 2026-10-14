"""Pure, lossless session-document normalization for MegaGlitch.

The live application historically saved two related JSON shapes: a version-1
node-editor graph and a version-2 ``dual_deck_session`` wrapper.  Neither shape
is a complete description of the visual machine.  This module provides the
UI-independent boundary used by new hosts: every supported input is migrated
to ``megaglitch.session`` version 2 without opening media, enumerating devices,
or starting any runtime service.

Migration is deliberately conservative.  Stable node ids and references are
added only when the saved information identifies one destination.  Ambiguous
links, timeline tracks, and modulation connections remain in the document so
that a later UI or migration tool can resolve them with more context.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any


SESSION_FORMAT = "megaglitch.session"
SESSION_FORMAT_VERSION = 2

_DECK_A_BINDINGS = {"SOURCE": "primary", "SOURCE_B": "secondary"}
_DECK_B_BINDINGS = {"SOURCE": "secondary", "SOURCE_B": "secondary"}
_SPECIAL_GRAPH_ENDPOINTS = {"SOURCE", "SOURCE_B", "DISPLAY"}

_LEGACY_GRAPH_FIELDS = {
    "version",
    "nodes",
    "links",
    "source_pos",
    "source_b_pos",
    "display_pos",
}
_LEGACY_APP_FIELDS = {
    "preset_type",
    "decks",
    "master_mix",
    "master_transition",
    "display_macros",
    "media_audio",
    "mod_matrix_state",
    "parameter_timeline_state",
}
_LEGACY_PRESENCE_FIELDS = (
    "master_mix",
    "master_transition",
    "display_macros",
    "media_audio",
    "mod_matrix_state",
    "parameter_timeline_state",
)

_GLOBAL_MACRO_NAMES = {
    "history_offset",
    "memory_decay",
    "event_hazard",
    "clock_quantization",
    "sync_desync",
    "phase_reset_chance",
    "local_time_warp",
    "rupture_threshold",
    "hysteresis_speed",
    "modulator_coupling",
    "time_reversal",
}


def _json_clone(value: Any) -> Any:
    """Return a detached value containing JSON data only.

    Normalization is a serialization boundary, so accepting objects that can
    never be written as JSON would only defer an error to a less useful place.
    The round trip also converts tuples and non-string mapping keys to their
    JSON representation without retaining references to caller-owned values.
    """

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _empty_graph() -> dict[str, Any]:
    return {"version": "1.0", "nodes": [], "links": []}


def _empty_modulation(enabled: bool = False) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "routes": [],
        "unresolved_legacy_connections": [],
    }


def _empty_timeline(enabled: bool = False) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "timebase": "seconds",
        "tracks": [],
        "unresolved_tracks": [],
    }


def _default_bindings(deck: str) -> dict[str, str]:
    return dict(_DECK_B_BINDINGS if str(deck).lower() == "b" else _DECK_A_BINDINGS)


def _default_deck(deck: str) -> dict[str, Any]:
    return {"source_bindings": _default_bindings(deck), "graph": _empty_graph()}


def _canonical_skeleton() -> dict[str, Any]:
    return {
        "format": SESSION_FORMAT,
        "format_version": SESSION_FORMAT_VERSION,
        "metadata": {},
        "decks": {"a": _default_deck("a"), "b": _default_deck("b")},
        "sources": {},
        "source_compositor": {},
        "master": {},
        "macros": {},
        "modulation": _empty_modulation(),
        "timeline": _empty_timeline(),
        "audio": {},
        "transport": {},
        "render": {},
        "outputs": {},
        "ui": {},
        "extensions": {},
    }


def _string_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_graph(raw_graph: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw_graph is None:
        return _empty_graph()
    if not isinstance(raw_graph, Mapping):
        raise TypeError("a deck graph must be a JSON object")

    graph = _json_clone(raw_graph)
    graph.setdefault("version", "1.0")
    graph.setdefault("nodes", [])
    graph.setdefault("links", [])
    if not isinstance(graph["nodes"], list):
        raise TypeError("graph.nodes must be a JSON array")
    if not isinstance(graph["links"], list):
        raise TypeError("graph.links must be a JSON array")

    # Existing ids remain authoritative.  Missing/null/empty ids receive the
    # same per-type ordinal form used by the Python NodeEditor serializer.
    used_ids = {
        node_id
        for node in graph["nodes"]
        if isinstance(node, Mapping)
        if (node_id := _string_id(node.get("id"))) is not None
    }
    type_counts: dict[str, int] = defaultdict(int)
    normalized_nodes: list[Any] = []
    for raw_node in graph["nodes"]:
        if not isinstance(raw_node, Mapping):
            normalized_nodes.append(_json_clone(raw_node))
            continue
        node = _json_clone(raw_node)
        type_name = _string_id(node.get("type")) or "Node"
        type_counts[type_name] += 1
        node_id = _string_id(node.get("id"))
        if node_id is None:
            ordinal = type_counts[type_name]
            candidate = f"{type_name}:{ordinal}"
            while candidate in used_ids:
                ordinal += 1
                candidate = f"{type_name}:{ordinal}"
            type_counts[type_name] = ordinal
            node_id = candidate
            node["id"] = node_id
            used_ids.add(node_id)
        elif node.get("id") != node_id:
            node["id"] = node_id
        normalized_nodes.append(node)
    graph["nodes"] = normalized_nodes

    ids: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in normalized_nodes:
        if not isinstance(node, dict):
            continue
        node_id = _string_id(node.get("id"))
        label = _string_id(node.get("label"))
        if node_id:
            ids[node_id].append(node)
        if label:
            labels[label].append(node)

    normalized_links: list[Any] = []
    for raw_link in graph["links"]:
        if not isinstance(raw_link, Mapping):
            normalized_links.append(_json_clone(raw_link))
            continue
        link = _json_clone(raw_link)
        for role in ("from", "to"):
            id_field = f"{role}_node_id"
            label_field = f"{role}_node"
            # An explicit, non-empty id is never silently retargeted, even if
            # it is now dangling or duplicated.
            if _string_id(link.get(id_field)) is not None:
                continue
            identifier = _string_id(link.get(label_field))
            if identifier is None or identifier in _SPECIAL_GRAPH_ENDPOINTS:
                continue
            candidates = ids.get(identifier, []) or labels.get(identifier, [])
            if len(candidates) != 1:
                continue
            candidate_id = _string_id(candidates[0].get("id"))
            if candidate_id and len(ids.get(candidate_id, [])) == 1:
                link[id_field] = candidate_id
        normalized_links.append(link)
    graph["links"] = normalized_links
    return graph


def _node_inventory(decks: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for raw_deck, raw_deck_state in decks.items():
        deck = str(raw_deck)
        if not isinstance(raw_deck_state, Mapping):
            continue
        graph = raw_deck_state.get("graph")
        if not isinstance(graph, Mapping):
            continue
        nodes = graph.get("nodes", [])
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_id = _string_id(node.get("id"))
            if node_id is None:
                continue
            params = node.get("params")
            inventory.append(
                {
                    "deck": deck,
                    "node_id": node_id,
                    "label": _string_id(node.get("label")),
                    "params": set(str(key) for key in params) if isinstance(params, Mapping) else set(),
                }
            )
    return inventory


def _resolve_track_node(
    track: Mapping[str, Any],
    legacy_key: str,
    inventory: list[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    target = track.get("target")
    target = target if isinstance(target, Mapping) else {}
    deck_hint = _string_id(target.get("deck") or track.get("deck"))
    node_id = _string_id(target.get("node_id") or track.get("node_id"))
    parameter = _string_id(target.get("parameter") or track.get("param"))
    if parameter is None:
        target_id = _string_id(track.get("target_id")) or legacy_key
        if ":" in target_id:
            parameter = _string_id(target_id.rsplit(":", 1)[1])
    if parameter is None:
        return None

    candidates = inventory
    if deck_hint:
        candidates = [node for node in candidates if node["deck"] == deck_hint]

    # A stable id is sufficient only when it occurs once in the relevant
    # scope.  Old Dear PyGui numeric tags generally miss this branch and fall
    # through to label+parameter matching.
    if node_id:
        by_id = [node for node in candidates if node["node_id"] == node_id]
        if len(by_id) == 1:
            return by_id[0], parameter

    node_label = _string_id(track.get("node_label"))
    if node_label:
        by_label_and_param = [
            node
            for node in candidates
            if node["label"] == node_label and parameter in node["params"]
        ]
        if len(by_label_and_param) == 1:
            return by_label_and_param[0], parameter
    return None


def _legacy_keyframes(raw_keyframes: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_keyframes, list):
        return None
    keyframes: list[dict[str, Any]] = []
    for raw_keyframe in raw_keyframes:
        if not isinstance(raw_keyframe, Mapping):
            return None
        interpolation = str(
            raw_keyframe.get("interpolation", raw_keyframe.get("interp", "linear"))
        )
        if interpolation not in {"linear", "ease", "step"}:
            interpolation = "linear"
        keyframes.append(
            {
                "time": _json_clone(raw_keyframe.get("time", 0.0)),
                "value": _json_clone(raw_keyframe.get("value")),
                "interpolation": interpolation,
            }
        )
    return keyframes


def _unresolved_track(legacy_key: str, raw_track: Any) -> dict[str, Any]:
    if isinstance(raw_track, Mapping):
        value = _json_clone(raw_track)
        value.setdefault("legacy_key", legacy_key)
        return value
    return {"legacy_key": legacy_key, "legacy_track": _json_clone(raw_track)}


def _migrate_legacy_timeline(
    raw_timeline: Any,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw_timeline, Mapping):
        result = _empty_timeline()
        result["unresolved_tracks"].append({"legacy_payload": _json_clone(raw_timeline)})
        return result

    result = {
        key: _json_clone(value)
        for key, value in raw_timeline.items()
        if key not in {"enabled", "timebase", "tracks", "unresolved_tracks"}
    }
    result.update(
        {
            "enabled": bool(raw_timeline.get("enabled", True)),
            "timebase": "seconds",
            "tracks": [],
            "unresolved_tracks": [],
        }
    )
    raw_tracks = raw_timeline.get("tracks", {})
    if isinstance(raw_tracks, Mapping):
        track_items = [(str(key), value) for key, value in raw_tracks.items()]
    elif isinstance(raw_tracks, list):
        track_items = [
            (_string_id(value.get("id")) or f"legacy-track:{index + 1}", value)
            if isinstance(value, Mapping)
            else (f"legacy-track:{index + 1}", value)
            for index, value in enumerate(raw_tracks)
        ]
    else:
        result["unresolved_tracks"].append(
            {"legacy_key": "tracks", "legacy_track": _json_clone(raw_tracks)}
        )
        return result

    resolutions: list[tuple[str, Any, tuple[dict[str, Any], str] | None]] = []
    proposed_ids: list[str] = []
    for legacy_key, raw_track in track_items:
        resolution = (
            _resolve_track_node(raw_track, legacy_key, inventory)
            if isinstance(raw_track, Mapping)
            else None
        )
        resolutions.append((legacy_key, raw_track, resolution))
        if resolution is not None:
            node, parameter = resolution
            proposed_ids.append(f"{node['deck']}:{node['node_id']}:{parameter}")
    duplicate_targets = {
        track_id for track_id, count in Counter(proposed_ids).items() if count > 1
    }

    for legacy_key, raw_track, resolution in resolutions:
        if resolution is None or not isinstance(raw_track, Mapping):
            result["unresolved_tracks"].append(_unresolved_track(legacy_key, raw_track))
            continue
        node, parameter = resolution
        track_id = f"{node['deck']}:{node['node_id']}:{parameter}"
        keyframes = _legacy_keyframes(raw_track.get("keyframes", []))
        if track_id in duplicate_targets or keyframes is None:
            result["unresolved_tracks"].append(_unresolved_track(legacy_key, raw_track))
            continue
        value_type = str(raw_track.get("value_type", raw_track.get("type", "float")))
        if value_type not in {"float", "int", "checkbox"}:
            value_type = "float"
        result["tracks"].append(
            {
                "id": track_id,
                "target": {
                    "deck": node["deck"],
                    "node_id": node["node_id"],
                    "parameter": parameter,
                },
                "value_type": value_type,
                "label": _json_clone(raw_track.get("label") or f"{node['label']} / {parameter}"),
                "keyframes": keyframes,
                "legacy": {"track_key": legacy_key, "track": _json_clone(raw_track)},
            }
        )
    return result


def _normalize_canonical_timeline(raw_timeline: Any, inventory: list[dict[str, Any]]) -> Any:
    if raw_timeline is None:
        return None
    if not isinstance(raw_timeline, Mapping):
        raise TypeError("session.timeline must be an object or null")
    if isinstance(raw_timeline.get("tracks"), Mapping):
        return _migrate_legacy_timeline(raw_timeline, inventory)
    timeline = _json_clone(raw_timeline)
    timeline.setdefault("enabled", False)
    timeline.setdefault("timebase", "seconds")
    timeline.setdefault("tracks", [])
    timeline.setdefault("unresolved_tracks", [])
    if not isinstance(timeline["tracks"], list):
        raise TypeError("session.timeline.tracks must be an array")
    if not isinstance(timeline["unresolved_tracks"], list):
        raise TypeError("session.timeline.unresolved_tracks must be an array")
    for index, track in enumerate(timeline["tracks"]):
        if isinstance(track, dict) and not _string_id(track.get("id")):
            target = track.get("target") if isinstance(track.get("target"), Mapping) else {}
            deck = _string_id(target.get("deck")) or "track"
            node_id = _string_id(target.get("node_id")) or str(index + 1)
            parameter = _string_id(target.get("parameter")) or "value"
            track["id"] = f"{deck}:{node_id}:{parameter}"
    return timeline


def _parameter_decks(parameter: str, inventory: list[dict[str, Any]]) -> set[str]:
    return {node["deck"] for node in inventory if parameter in node["params"]}


def _legacy_route(
    connection: str,
    route_index: int,
    inventory: list[dict[str, Any]],
    macro_names: set[str],
) -> dict[str, Any] | None:
    if connection.count("->") != 1:
        return None
    source, parameter = (part.strip() for part in connection.split("->", 1))
    if not source or not parameter:
        return None
    if parameter in macro_names:
        target: dict[str, Any] = {"scope": "global_macro", "parameter": parameter}
    else:
        owners = _parameter_decks(parameter, inventory)
        if len(owners) != 1:
            return None
        target = {
            "scope": "deck_parameter_name",
            "deck": next(iter(owners)),
            "parameter": parameter,
        }
    return {
        "id": f"legacy-route:{route_index}",
        "enabled": True,
        "source": source,
        "target": target,
        "legacy_connection": connection,
    }


def _migrate_legacy_modulation(
    raw_modulation: Any,
    inventory: list[dict[str, Any]],
    macros: Any,
) -> dict[str, Any]:
    if not isinstance(raw_modulation, Mapping):
        result = _empty_modulation()
        result["unresolved_legacy_connections"].append(
            json.dumps(_json_clone(raw_modulation), ensure_ascii=False, sort_keys=True)
        )
        return result
    result = {
        key: _json_clone(value)
        for key, value in raw_modulation.items()
        if key not in {"enabled", "connections", "routes", "unresolved_legacy_connections"}
    }
    result.update(
        {
            "enabled": bool(raw_modulation.get("enabled", True)),
            "routes": [],
            "unresolved_legacy_connections": [],
        }
    )
    macro_names = set(_GLOBAL_MACRO_NAMES)
    if isinstance(macros, Mapping):
        macro_names.update(str(key) for key in macros)
    connections = raw_modulation.get("connections", [])
    if not isinstance(connections, list):
        result["unresolved_legacy_connections"].append(
            json.dumps(_json_clone(connections), ensure_ascii=False, sort_keys=True)
        )
        return result
    for index, raw_connection in enumerate(connections, 1):
        if isinstance(raw_connection, str):
            route = _legacy_route(raw_connection, index, inventory, macro_names)
            if route is not None:
                result["routes"].append(route)
            else:
                result["unresolved_legacy_connections"].append(raw_connection)
            continue
        if isinstance(raw_connection, Mapping):
            connection = _json_clone(raw_connection)
            target = connection.get("target")
            if isinstance(target, Mapping) and target.get("scope") in {
                "global_macro",
                "deck_parameter_name",
                "node_parameter",
            }:
                connection.setdefault("id", f"legacy-route:{index}")
                connection.setdefault("enabled", True)
                result["routes"].append(connection)
                continue
        result["unresolved_legacy_connections"].append(
            json.dumps(_json_clone(raw_connection), ensure_ascii=False, sort_keys=True)
        )
    return result


def _normalize_canonical_modulation(
    raw_modulation: Any,
    inventory: list[dict[str, Any]],
    macros: Any,
) -> Any:
    if raw_modulation is None:
        return None
    if not isinstance(raw_modulation, Mapping):
        raise TypeError("session.modulation must be an object or null")
    if "connections" in raw_modulation and "routes" not in raw_modulation:
        return _migrate_legacy_modulation(raw_modulation, inventory, macros)
    modulation = _json_clone(raw_modulation)
    modulation.setdefault("enabled", False)
    modulation.setdefault("routes", [])
    modulation.setdefault("unresolved_legacy_connections", [])
    if not isinstance(modulation["routes"], list):
        raise TypeError("session.modulation.routes must be an array")
    if not isinstance(modulation["unresolved_legacy_connections"], list):
        raise TypeError("session.modulation.unresolved_legacy_connections must be an array")
    used_ids = {
        route_id
        for route in modulation["routes"]
        if isinstance(route, Mapping)
        if (route_id := _string_id(route.get("id"))) is not None
    }
    for index, route in enumerate(modulation["routes"], 1):
        if not isinstance(route, dict) or _string_id(route.get("id")):
            continue
        candidate = f"route:{index}"
        ordinal = index
        while candidate in used_ids:
            ordinal += 1
            candidate = f"route:{ordinal}"
        route["id"] = candidate
        used_ids.add(candidate)
    return modulation


def _legacy_provenance(
    raw: Mapping[str, Any],
    source_format: str,
    recognized_fields: set[str],
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "source_format": source_format,
        "field_presence": {field: field in raw for field in _LEGACY_PRESENCE_FIELDS},
        "unmapped_root": {
            str(key): _json_clone(value)
            for key, value in raw.items()
            if key not in recognized_fields
        },
    }
    service_payloads = {
        field: _json_clone(raw[field])
        for field in ("mod_matrix_state", "parameter_timeline_state")
        if field in raw
    }
    if service_payloads:
        provenance["service_payloads"] = service_payloads
    return provenance


def _legacy_service(raw: Mapping[str, Any], field: str, missing_default: Any) -> Any:
    if field not in raw:
        return _json_clone(missing_default)
    if raw[field] is None:
        return None
    return _json_clone(raw[field])


def _migrate_legacy_graph(raw: Mapping[str, Any]) -> dict[str, Any]:
    session = _canonical_skeleton()
    graph_payload = {
        key: _json_clone(value) for key, value in raw.items() if key in _LEGACY_GRAPH_FIELDS
    }
    session["decks"]["a"]["graph"] = _normalize_graph(graph_payload)
    inventory = _node_inventory(session["decks"])
    session["macros"] = _legacy_service(raw, "display_macros", {})
    session["audio"] = _legacy_service(raw, "media_audio", {})
    raw_modulation = _legacy_service(raw, "mod_matrix_state", {})
    session["modulation"] = (
        None
        if raw_modulation is None
        else _migrate_legacy_modulation(raw_modulation, inventory, session["macros"])
    )
    raw_timeline = _legacy_service(raw, "parameter_timeline_state", {})
    session["timeline"] = (
        None
        if raw_timeline is None
        else _migrate_legacy_timeline(raw_timeline, inventory)
    )
    session["extensions"] = {
        "legacy": _legacy_provenance(
            raw,
            "graph_v1",
            _LEGACY_GRAPH_FIELDS | _LEGACY_APP_FIELDS,
        )
    }
    return session


def _migrate_legacy_dual_deck(raw: Mapping[str, Any]) -> dict[str, Any]:
    session = _canonical_skeleton()
    raw_decks = raw.get("decks", {})
    if not isinstance(raw_decks, Mapping):
        raise TypeError("dual_deck_session.decks must be a JSON object")
    decks: dict[str, Any] = {}
    for raw_name, raw_graph in raw_decks.items():
        name = str(raw_name)
        decks[name] = {
            "source_bindings": _default_bindings(name),
            "graph": _normalize_graph(raw_graph),
        }
    decks.setdefault("a", _default_deck("a"))
    decks.setdefault("b", _default_deck("b"))
    session["decks"] = decks
    if "master_mix" in raw or "master_transition" in raw:
        session["master"] = {}
        if "master_mix" in raw:
            session["master"]["mix"] = _json_clone(raw["master_mix"])
        if "master_transition" in raw:
            session["master"]["transition"] = _json_clone(raw["master_transition"])
    session["macros"] = _legacy_service(raw, "display_macros", {})
    session["audio"] = _legacy_service(raw, "media_audio", {})
    inventory = _node_inventory(session["decks"])
    raw_modulation = _legacy_service(raw, "mod_matrix_state", {})
    session["modulation"] = (
        None
        if raw_modulation is None
        else _migrate_legacy_modulation(raw_modulation, inventory, session["macros"])
    )
    raw_timeline = _legacy_service(raw, "parameter_timeline_state", {})
    session["timeline"] = (
        None
        if raw_timeline is None
        else _migrate_legacy_timeline(raw_timeline, inventory)
    )
    session["extensions"] = {
        "legacy": _legacy_provenance(
            raw,
            "dual_deck_session_v2",
            {"version"} | _LEGACY_APP_FIELDS,
        )
    }
    return session


def _normalize_canonical_deck(name: str, raw_deck: Any) -> dict[str, Any]:
    if not isinstance(raw_deck, Mapping):
        raise TypeError(f"session.decks.{name} must be a JSON object")
    deck = _json_clone(raw_deck)
    bindings = deck.get("source_bindings")
    if bindings is None:
        bindings = _default_bindings(name)
    elif not isinstance(bindings, Mapping):
        raise TypeError(f"session.decks.{name}.source_bindings must be an object")
    else:
        bindings = _json_clone(bindings)
        for endpoint, source in _default_bindings(name).items():
            bindings.setdefault(endpoint, source)
    deck["source_bindings"] = bindings
    deck["graph"] = _normalize_graph(deck.get("graph"))
    return deck


def _normalize_canonical(raw: Mapping[str, Any]) -> dict[str, Any]:
    version = raw.get("format_version")
    if str(version) not in {"2", "2.0"}:
        raise ValueError(f"unsupported {SESSION_FORMAT} format_version: {version!r}")
    session = _json_clone(raw)
    session["format"] = SESSION_FORMAT
    session["format_version"] = SESSION_FORMAT_VERSION
    defaults = _canonical_skeleton()
    for key, default in defaults.items():
        session.setdefault(key, _json_clone(default))

    if not isinstance(session["decks"], Mapping):
        raise TypeError("session.decks must be a JSON object")
    decks = {
        str(name): _normalize_canonical_deck(str(name), deck)
        for name, deck in session["decks"].items()
    }
    decks.setdefault("a", _default_deck("a"))
    decks.setdefault("b", _default_deck("b"))
    session["decks"] = decks
    inventory = _node_inventory(decks)
    session["modulation"] = _normalize_canonical_modulation(
        session["modulation"], inventory, session["macros"]
    )
    session["timeline"] = _normalize_canonical_timeline(session["timeline"], inventory)
    return session


def normalize_session_document(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a legacy or canonical mapping without activating the machine.

    The caller's mapping is never mutated.  Unsupported roots and versions are
    rejected instead of being guessed, while unknown fields inside supported
    documents are retained.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError("a MegaGlitch session document must be a JSON object")
    raw = _json_clone(mapping)
    if raw.get("format") == SESSION_FORMAT:
        return _normalize_canonical(raw)
    if raw.get("preset_type") == "dual_deck_session":
        if str(raw.get("version")) not in {"2", "2.0"}:
            raise ValueError(
                f"unsupported dual_deck_session version: {raw.get('version')!r}"
            )
        return _migrate_legacy_dual_deck(raw)
    if str(raw.get("version", "1.0")) in {"1", "1.0"} and (
        "nodes" in raw or "links" in raw
    ):
        return _migrate_legacy_graph(raw)
    raise ValueError("unsupported MegaGlitch session document format")


def load_session_document(path: str | PathLike[str]) -> dict[str, Any]:
    """Read and normalize a session JSON file, performing no runtime I/O."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("a MegaGlitch session file must contain a JSON object")
    return normalize_session_document(payload)


def _legacy_extensions(session: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = session.get("extensions")
    if not isinstance(extensions, Mapping):
        return {}
    legacy = extensions.get("legacy")
    return legacy if isinstance(legacy, Mapping) else {}


def _should_export_field(legacy: Mapping[str, Any], field: str) -> bool:
    presence = legacy.get("field_presence")
    if isinstance(presence, Mapping) and field in presence:
        return bool(presence[field])
    return True


def _export_timeline(timeline: Any) -> Any:
    if timeline is None or not isinstance(timeline, Mapping):
        return _json_clone(timeline)
    result = {
        key: _json_clone(value)
        for key, value in timeline.items()
        if key not in {"timebase", "tracks", "unresolved_tracks"}
    }
    result["enabled"] = bool(timeline.get("enabled", False))
    tracks: dict[str, Any] = {}
    for track in timeline.get("tracks", []) if isinstance(timeline.get("tracks"), list) else []:
        if not isinstance(track, Mapping):
            continue
        target = track.get("target") if isinstance(track.get("target"), Mapping) else {}
        node_id = _string_id(target.get("node_id"))
        parameter = _string_id(target.get("parameter"))
        if node_id is None or parameter is None:
            continue
        legacy = track.get("legacy") if isinstance(track.get("legacy"), Mapping) else {}
        raw_track = legacy.get("track") if isinstance(legacy.get("track"), Mapping) else {}
        value = _json_clone(raw_track)
        value["target_id"] = f"{node_id}:{parameter}"
        value["node_id"] = node_id
        value["param"] = parameter
        value["type"] = track.get("value_type", value.get("type", "float"))
        value["label"] = _json_clone(track.get("label", value.get("label", parameter)))
        value["keyframes"] = [
            {
                "time": _json_clone(keyframe.get("time", 0.0)),
                "value": _json_clone(keyframe.get("value")),
                "interp": str(keyframe.get("interpolation", "linear")),
            }
            for keyframe in track.get("keyframes", [])
            if isinstance(keyframe, Mapping)
        ]
        tracks[f"{node_id}:{parameter}"] = value
    unresolved = timeline.get("unresolved_tracks", [])
    if isinstance(unresolved, list):
        for index, item in enumerate(unresolved, 1):
            if not isinstance(item, Mapping):
                tracks[f"unresolved:{index}"] = _json_clone(item)
                continue
            key = _string_id(item.get("legacy_key")) or f"unresolved:{index}"
            if "legacy_track" in item:
                value = _json_clone(item["legacy_track"])
            else:
                value = {k: _json_clone(v) for k, v in item.items() if k != "legacy_key"}
            candidate = key
            suffix = 2
            while candidate in tracks:
                candidate = f"{key}#{suffix}"
                suffix += 1
            tracks[candidate] = value
    result["tracks"] = tracks
    return result


def _route_connection(route: Mapping[str, Any]) -> str | None:
    legacy = route.get("legacy_connection")
    if isinstance(legacy, str):
        return legacy
    source = _string_id(route.get("source"))
    target = route.get("target") if isinstance(route.get("target"), Mapping) else {}
    parameter = _string_id(target.get("parameter"))
    if source and parameter:
        return f"{source} -> {parameter}"
    return None


def _export_modulation(modulation: Any, legacy: Mapping[str, Any]) -> Any:
    if modulation is None or not isinstance(modulation, Mapping):
        return _json_clone(modulation)
    result = {
        key: _json_clone(value)
        for key, value in modulation.items()
        if key not in {"routes", "unresolved_legacy_connections"}
    }
    result["enabled"] = bool(modulation.get("enabled", False))
    current = [
        connection
        for route in modulation.get("routes", [])
        if isinstance(route, Mapping)
        if (connection := _route_connection(route)) is not None
    ]
    unresolved = modulation.get("unresolved_legacy_connections", [])
    if isinstance(unresolved, list):
        current.extend(str(value) for value in unresolved)

    # Retain the original interleaving when the same legacy strings still
    # exist, then append routes added in canonical form.
    payloads = legacy.get("service_payloads")
    raw_modulation = payloads.get("mod_matrix_state") if isinstance(payloads, Mapping) else None
    original = raw_modulation.get("connections") if isinstance(raw_modulation, Mapping) else None
    ordered: list[str] = []
    remaining = list(current)
    if isinstance(original, list):
        for value in original:
            if isinstance(value, str) and value in remaining:
                ordered.append(value)
                remaining.remove(value)
    ordered.extend(remaining)
    result["connections"] = ordered
    return result


def export_legacy_dual_deck(session: Mapping[str, Any]) -> dict[str, Any]:
    """Export a canonical-compatible session in the legacy v2 wrapper shape.

    Canonical-only source/device/output settings have no v2 representation and
    remain in the caller's session.  Graphs, mixer state, macros, modulation,
    timeline, audio payloads, nulls, and legacy unknown root fields are emitted
    without activating or validating any backend service.
    """

    canonical = normalize_session_document(session)
    legacy = _legacy_extensions(canonical)
    unmapped = legacy.get("unmapped_root")
    result = _json_clone(unmapped) if isinstance(unmapped, Mapping) else {}
    result.update(
        {
            "version": "2.0",
            "preset_type": "dual_deck_session",
            "decks": {
                str(name): _json_clone(deck.get("graph", _empty_graph()))
                for name, deck in canonical["decks"].items()
                if isinstance(deck, Mapping)
            },
        }
    )
    master = canonical.get("master")
    master = master if isinstance(master, Mapping) else {}
    if _should_export_field(legacy, "master_mix") and "mix" in master:
        result["master_mix"] = _json_clone(master["mix"])
    if _should_export_field(legacy, "master_transition") and "transition" in master:
        result["master_transition"] = _json_clone(master["transition"])
    if _should_export_field(legacy, "display_macros"):
        result["display_macros"] = _json_clone(canonical.get("macros"))
    if _should_export_field(legacy, "media_audio"):
        result["media_audio"] = _json_clone(canonical.get("audio"))
    if _should_export_field(legacy, "mod_matrix_state"):
        result["mod_matrix_state"] = _export_modulation(canonical.get("modulation"), legacy)
    if _should_export_field(legacy, "parameter_timeline_state"):
        result["parameter_timeline_state"] = _export_timeline(canonical.get("timeline"))
    return result


__all__ = [
    "SESSION_FORMAT",
    "SESSION_FORMAT_VERSION",
    "export_legacy_dual_deck",
    "load_session_document",
    "normalize_session_document",
]
