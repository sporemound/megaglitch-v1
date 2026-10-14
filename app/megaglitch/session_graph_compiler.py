"""Compile saved Dear PyGui node graphs without importing Dear PyGui.

The legacy :class:`node_editor.NodeEditor` builds a recursive graph from the
socket connected to ``DISPLAY.image_in`` and then flattens that graph for
``node_pipeline_runtime.execute_flat_node_pipeline``.  This module preserves
that execution order while making validation explicit.  It intentionally
compiles one version-1 editor graph at a time; callers of a version-2 dual-deck
session select ``decks.a`` or ``decks.b`` and provide that deck's source
namespace.

``source_namespace`` maps saved endpoint names to runtime source keys.  The
usual deck-A mapping is ``{"SOURCE": "primary", "SOURCE_B": "secondary"}``;
deck B can map ``SOURCE`` to ``secondary``.  A value may instead be a mapping
with ``source_key``, optional ``label``, and optional ``outputs`` fields.

Supported visual node semantics are the ones consumed by the flat runtime:

* ordinary GlitchEffectNode/CUDA nodes (``image_in`` -> ``image_out``),
* GeneratorNode subclasses, including selected auxiliary outputs,
* Genesis priority/mask multi-input nodes,
* VideoMixerNode,
* LuaShaderNode and the two shipped multi-input Lua subclasses.

The v1 preset format does not serialize arbitrary runtime class inheritance or
dynamic socket metadata.  Unknown ``*GeneratorNode`` classes therefore use the
legacy generator default (no inputs, ``image_out``), while unknown non-special
node classes use the legacy effect default.  ``FrameDifferenceNode``,
``FeedbackLoopNode``, bare ``MultiInputNode``, and audio-only ``FMSynthNode``
cannot be represented by the flat visual executor and are rejected if wired.
Lua path/metadata parameters are copied verbatim; the sockets that v1 can
represent are validated from the shipped class contracts below.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class GraphCompileError(ValueError):
    """A preset cannot be represented faithfully by the flat frame runtime."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details)


@dataclass(frozen=True)
class _Contract:
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    runtime_node_type: str


@dataclass(frozen=True)
class _Source:
    identifier: str
    source_key: str
    label: str
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class _Node:
    index: int
    stable_id: str
    backend_id: str
    label: str
    params: dict[str, Any]
    enabled: Any
    contract: _Contract


@dataclass(frozen=True)
class _Endpoint:
    kind: str
    identifier: str
    node: _Node | None = None
    source: _Source | None = None


@dataclass(frozen=True)
class _Link:
    index: int
    source: _Endpoint
    target: _Endpoint
    from_attr: str
    to_attr: str


_EFFECT = _Contract("effect", ("image_in",), ("image_out",), "effect")
_GENERATOR = _Contract("generator", (), ("image_out",), "generator")
_MIXER = _Contract(
    "mixer", ("image_a", "image_b"), ("image_out",), "mixer"
)
_GENESIS_PRIORITY = _Contract(
    "multi_input",
    (
        "base_image",
        "overlay_image",
        "priority_mask",
        "luma_mask",
        "corruption_mask",
    ),
    ("image_out",),
    "genesis_priority_composite",
)
_GENESIS_MASK = _Contract(
    "multi_input",
    ("image_in", "mask_in", "luma_mask", "priority_mask", "corruption_mask"),
    ("image_out",),
    "genesis_mask_key",
)
_LUA = _Contract(
    "lua", ("image_in", "feedback_in"), ("image_out",), "lua_multi_input"
)
_ABYSSAL_LUA = _Contract(
    "lua",
    (
        "image_in",
        "bubble_mask",
        "bubble_normal",
        "bubble_depth",
        "bubble_refraction",
        "bubble_highlight",
    ),
    ("image_out",),
    "lua_multi_input",
)
_BUBBLE_GENERATOR = _Contract(
    "generator",
    ("image_in",),
    (
        "image_out",
        "bubble_mask",
        "bubble_normal",
        "bubble_depth",
        "bubble_refraction",
        "bubble_highlight",
        "bubble_debug",
    ),
    "generator",
)
_GENESIS_VDP_GENERATOR = _Contract(
    "generator",
    ("feedback_in", "mask_in"),
    (
        "image_out",
        "final_frame",
        "plane_a",
        "plane_b",
        "sprite_layer",
        "priority_mask",
        "palette_index_buffer",
        "luma_mask",
        "corruption_mask",
        "scanline_mod_buffer",
        "crt_prepass",
    ),
    "generator",
)
_NO_VISUAL_OUTPUT = _Contract("unsupported", (), (), "unsupported")
_FRAME_DIFFERENCE_UNSUPPORTED = _Contract(
    "unsupported", ("frame_a", "frame_b"), ("result",), "unsupported"
)
_FEEDBACK_LOOP_UNSUPPORTED = _Contract(
    "unsupported",
    ("current_frame", "feedback_frame"),
    ("result",),
    "unsupported",
)

_SPECIAL_CONTRACTS = {
    "VideoMixerNode": _MIXER,
    "GenesisPriorityCompositeNode": _GENESIS_PRIORITY,
    "GenesisMaskKeyNode": _GENESIS_MASK,
    "LuaShaderNode": _LUA,
    "KaleidoFeedbackShaderNode": _LUA,
    "AbyssalSignalFeedbackShaderNode": _ABYSSAL_LUA,
    "BubbleSimGeneratorNode": _BUBBLE_GENERATOR,
    "BubbleSimGenerator2Node": _BUBBLE_GENERATOR,
    "GenesisVDPGeneratorNode": _GENESIS_VDP_GENERATOR,
    "FMSynthNode": _NO_VISUAL_OUTPUT,
    "FrameDifferenceNode": _FRAME_DIFFERENCE_UNSUPPORTED,
    "FeedbackLoopNode": _FEEDBACK_LOOP_UNSUPPORTED,
}

_UNSUPPORTED_NODE_TYPES = {
    "MultiInputNode",
}

_RESERVED_IDENTIFIERS = {"DISPLAY"}
_VALID_SOURCE_KEYS = {"primary", "secondary"}


def _fail(code: str, message: str, **details: Any) -> None:
    raise GraphCompileError(code, message, **details)


def _nonempty_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_graph", f"{description} must be a non-empty string")
    return value.strip()


def _string_sequence(value: Any, description: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        _fail("invalid_source_namespace", f"{description} must be a sequence")
    result = tuple(_nonempty_string(item, description) for item in value)
    if not result or len(set(result)) != len(result):
        _fail(
            "invalid_source_namespace",
            f"{description} must contain unique socket names",
        )
    return result


def _normalise_sources(source_namespace: Mapping[str, Any]) -> dict[str, _Source]:
    if not isinstance(source_namespace, Mapping):
        _fail("invalid_source_namespace", "source_namespace must be a mapping")
    sources: dict[str, _Source] = {}
    for raw_identifier, raw_config in source_namespace.items():
        identifier = _nonempty_string(raw_identifier, "source identifier")
        if identifier in _RESERVED_IDENTIFIERS:
            _fail(
                "invalid_source_namespace",
                f"source identifier {identifier!r} is reserved",
            )
        if isinstance(raw_config, str):
            source_key = _nonempty_string(raw_config, f"source key for {identifier}")
            label = "Source B" if identifier == "SOURCE_B" else "Source"
            outputs = ("image_out",)
        elif isinstance(raw_config, Mapping):
            source_key = _nonempty_string(
                raw_config.get("source_key"), f"source key for {identifier}"
            )
            label = _nonempty_string(
                raw_config.get("label")
                or ("Source B" if identifier == "SOURCE_B" else "Source"),
                f"source label for {identifier}",
            )
            outputs = _string_sequence(
                raw_config.get("outputs", ("image_out",)),
                f"outputs for source {identifier}",
            )
        else:
            _fail(
                "invalid_source_namespace",
                f"source {identifier!r} needs a source key or configuration mapping",
            )
        if source_key not in _VALID_SOURCE_KEYS:
            _fail(
                "invalid_source_namespace",
                f"source {identifier!r} uses unsupported runtime key {source_key!r}",
                supported=sorted(_VALID_SOURCE_KEYS),
            )
        sources[identifier] = _Source(identifier, source_key, label, outputs)
    return sources


def _contract_for(backend_id: str, raw_node: Mapping[str, Any]) -> _Contract:
    contract = _SPECIAL_CONTRACTS.get(backend_id)
    if contract is not None:
        return contract
    if backend_id in _UNSUPPORTED_NODE_TYPES:
        return _NO_VISUAL_OUTPUT

    serialized_kind = raw_node.get("node_type")
    if serialized_kind == "mixer":
        return _MIXER
    if serialized_kind == "genesis_priority_composite":
        return _GENESIS_PRIORITY
    if serialized_kind == "genesis_mask_key":
        return _GENESIS_MASK
    if serialized_kind == "lua_multi_input":
        return _LUA
    if serialized_kind == "generator" or backend_id.endswith("GeneratorNode"):
        return _GENERATOR

    params = raw_node.get("params")
    if isinstance(params, Mapping) and {
        "shader_path",
        "lua_path",
    }.issubset(params):
        return _LUA
    return _EFFECT


def _normalise_nodes(raw_nodes: Any) -> list[_Node]:
    if raw_nodes is None:
        raw_nodes = []
    if isinstance(raw_nodes, (str, bytes)) or not isinstance(raw_nodes, Sequence):
        _fail("invalid_graph", "preset nodes must be a list")

    type_counts: dict[str, int] = defaultdict(int)
    nodes: list[_Node] = []
    seen_ids: dict[str, int] = {}
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            _fail("invalid_node", f"node {index} must be an object", node_index=index)
        backend_id = _nonempty_string(raw_node.get("type"), f"node {index} type")
        label = _nonempty_string(raw_node.get("label"), f"node {index} label")
        type_counts[backend_id] += 1
        raw_id = raw_node.get("id")
        stable_id = (
            _nonempty_string(raw_id, f"node {index} id")
            if raw_id is not None
            else f"{backend_id}:{type_counts[backend_id]}"
        )
        if stable_id in _RESERVED_IDENTIFIERS:
            _fail("duplicate_node_id", f"node id {stable_id!r} is reserved")
        if stable_id in seen_ids:
            _fail(
                "duplicate_node_id",
                f"duplicate node id {stable_id!r} at nodes {seen_ids[stable_id]} and {index}",
                node_id=stable_id,
            )
        seen_ids[stable_id] = index

        raw_params = raw_node.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, Mapping):
            _fail("invalid_node", f"node {stable_id!r} params must be an object")
        params = dict(raw_params)

        enabled = params.get("enabled", True)
        if "enabled" in raw_node:
            if not isinstance(raw_node["enabled"], bool):
                _fail("invalid_node", f"node {stable_id!r} enabled must be boolean")
            enabled = raw_node["enabled"]
            params["enabled"] = enabled
        if "active" in raw_node:
            if not isinstance(raw_node["active"], bool):
                _fail("invalid_node", f"node {stable_id!r} active must be boolean")
            enabled = raw_node["active"]
            params["enabled"] = enabled

        nodes.append(
            _Node(
                index=index,
                stable_id=stable_id,
                backend_id=backend_id,
                label=label,
                params=params,
                enabled=enabled,
                contract=_contract_for(backend_id, raw_node),
            )
        )
    return nodes


class _Compiler:
    def __init__(
        self,
        preset: Mapping[str, Any],
        source_namespace: Mapping[str, Any],
    ) -> None:
        self.sources = _normalise_sources(source_namespace)
        self.nodes = _normalise_nodes(preset.get("nodes"))
        source_id_collisions = sorted(
            node.stable_id for node in self.nodes if node.stable_id in self.sources
        )
        if source_id_collisions:
            _fail(
                "duplicate_node_id",
                "node ids collide with source endpoints: "
                + ", ".join(source_id_collisions),
                node_ids=source_id_collisions,
            )
        self.by_id = {node.stable_id: node for node in self.nodes}
        self.by_label: dict[str, list[_Node]] = defaultdict(list)
        for node in self.nodes:
            self.by_label[node.label].append(node)

        self.links = self._normalise_links(preset.get("links"))
        self.incoming: dict[tuple[str, str, str], _Link] = {}
        for link in self.links:
            key = (link.target.kind, link.target.identifier, link.to_attr)
            previous = self.incoming.get(key)
            if previous is not None:
                _fail(
                    "duplicate_input",
                    f"input {link.target.identifier}.{link.to_attr} has more than one link",
                    links=[previous.index, link.index],
                )
            self.incoming[key] = link

        self._validate_cycles()
        self.used_links: set[int] = set()
        self.reached_nodes: set[str] = set()

    def _resolve_endpoint(self, identifier: str, *, role: str, link_index: int) -> _Endpoint:
        if identifier == "DISPLAY":
            return _Endpoint("display", "DISPLAY")
        source = self.sources.get(identifier)
        if source is not None:
            return _Endpoint("source", identifier, source=source)
        node = self.by_id.get(identifier)
        if node is not None:
            return _Endpoint("node", node.stable_id, node=node)
        candidates = self.by_label.get(identifier, [])
        if len(candidates) > 1:
            _fail(
                "ambiguous_node_label",
                f"link {link_index} {role} label {identifier!r} matches multiple nodes; use the stable node id",
                label=identifier,
                node_ids=[candidate.stable_id for candidate in candidates],
            )
        if len(candidates) == 1:
            node = candidates[0]
            return _Endpoint("node", node.stable_id, node=node)
        _fail(
            "missing_endpoint",
            f"link {link_index} {role} endpoint {identifier!r} does not exist",
            link_index=link_index,
            endpoint=identifier,
        )

    @staticmethod
    def _endpoint_identifier(raw_link: Mapping[str, Any], role: str, index: int) -> str:
        id_field = f"{role}_node_id"
        label_field = f"{role}_node"
        value = raw_link.get(id_field) or raw_link.get(label_field)
        return _nonempty_string(value, f"link {index} {role} endpoint")

    @staticmethod
    def _available_sockets(endpoint: _Endpoint, role: str) -> tuple[str, ...]:
        if endpoint.kind == "source":
            return endpoint.source.outputs if role == "from" and endpoint.source else ()
        if endpoint.kind == "display":
            return ("image_in",) if role == "to" else ()
        if endpoint.node is None:
            return ()
        return endpoint.node.contract.outputs if role == "from" else endpoint.node.contract.inputs

    def _normalise_links(self, raw_links: Any) -> list[_Link]:
        if raw_links is None:
            raw_links = []
        if isinstance(raw_links, (str, bytes)) or not isinstance(raw_links, Sequence):
            _fail("invalid_graph", "preset links must be a list")
        links: list[_Link] = []
        for index, raw_link in enumerate(raw_links):
            if not isinstance(raw_link, Mapping):
                _fail("invalid_link", f"link {index} must be an object")
            source_identifier = self._endpoint_identifier(raw_link, "from", index)
            target_identifier = self._endpoint_identifier(raw_link, "to", index)
            source = self._resolve_endpoint(source_identifier, role="from", link_index=index)
            target = self._resolve_endpoint(target_identifier, role="to", link_index=index)
            from_attr = _nonempty_string(raw_link.get("from_attr"), f"link {index} from_attr")
            to_attr = _nonempty_string(raw_link.get("to_attr"), f"link {index} to_attr")

            source_sockets = self._available_sockets(source, "from")
            if from_attr not in source_sockets:
                _fail(
                    "missing_socket",
                    f"link {index} source socket {source.identifier}.{from_attr} does not exist",
                    available=list(source_sockets),
                )
            target_sockets = self._available_sockets(target, "to")
            if to_attr not in target_sockets:
                _fail(
                    "missing_socket",
                    f"link {index} target socket {target.identifier}.{to_attr} does not exist",
                    available=list(target_sockets),
                )
            links.append(_Link(index, source, target, from_attr, to_attr))
        return links

    def _validate_cycles(self) -> None:
        adjacency: dict[str, list[str]] = defaultdict(list)
        for link in self.links:
            if link.source.kind == "node" and link.target.kind == "node":
                adjacency[link.source.identifier].append(link.target.identifier)

        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(node_id: str) -> None:
            marker = state.get(node_id, 0)
            if marker == 2:
                return
            if marker == 1:
                try:
                    start = stack.index(node_id)
                except ValueError:
                    start = 0
                cycle = stack[start:] + [node_id]
                _fail("cycle", f"cycle detected: {' -> '.join(cycle)}", cycle=cycle)
            state[node_id] = 1
            stack.append(node_id)
            for target in adjacency.get(node_id, ()):
                visit(target)
            stack.pop()
            state[node_id] = 2

        for node in self.nodes:
            visit(node.stable_id)

    @staticmethod
    def _with_enabled(step: dict[str, Any], node: _Node) -> dict[str, Any]:
        # The Python executor reads params.enabled.  The coarse native selector
        # also understands a top-level enabled flag, so emit both without
        # coercing legacy parameter values.
        step["enabled"] = node.enabled
        return step

    def _emit_link(self, link: _Link, pipeline: list[dict[str, Any]]) -> None:
        self.used_links.add(link.index)
        if link.source.kind == "source":
            source = link.source.source
            assert source is not None
            pipeline.append(
                {
                    "effect_type": source.label,
                    "params": {},
                    "node_type": "source",
                    "source_key": source.source_key,
                }
            )
            return

        node = link.source.node
        assert node is not None
        self.reached_nodes.add(node.stable_id)
        contract = node.contract
        if contract.kind == "unsupported":
            _fail(
                "unsupported_node",
                f"node {node.stable_id!r} ({node.backend_id}) has no flat visual runtime contract",
                node_id=node.stable_id,
                backend_id=node.backend_id,
            )

        connected_inputs: list[str] = []
        for input_name in contract.inputs:
            input_link = self.incoming.get(("node", node.stable_id, input_name))
            if input_link is not None:
                self._emit_link(input_link, pipeline)
                connected_inputs.append(input_name)

        common = {
            "effect_type": node.label,
            "params": dict(node.params),
            "node_id": node.stable_id,
        }
        if contract.kind == "generator":
            step = {
                **common,
                "is_generator": True,
                "node_type": "generator",
                "selected_output": link.from_attr,
                "input_order": connected_inputs,
            }
        elif contract.kind in {"multi_input", "lua"}:
            step = {
                **common,
                "node_type": contract.runtime_node_type,
                "input_order": connected_inputs,
            }
        elif contract.kind == "mixer":
            step = {**common, "node_type": "mixer"}
        else:
            step = {**common, "node_type": "effect"}
        pipeline.append(self._with_enabled(step, node))

    def compile(self) -> list[dict[str, Any]]:
        if not self.nodes and not self.links:
            return []

        display_link = self.incoming.get(("display", "DISPLAY", "image_in"))
        if display_link is None:
            _fail(
                "missing_display_input",
                "non-empty graph must connect exactly one output to DISPLAY.image_in",
            )

        pipeline: list[dict[str, Any]] = []
        self._emit_link(display_link, pipeline)

        unreachable = sorted(
            node.stable_id for node in self.nodes if node.stable_id not in self.reached_nodes
        )
        if unreachable:
            _fail(
                "unreachable_nodes",
                "preset contains nodes that are not part of the DISPLAY path: "
                + ", ".join(unreachable),
                node_ids=unreachable,
            )
        unused_links = sorted(link.index for link in self.links if link.index not in self.used_links)
        if unused_links:
            _fail(
                "unreachable_links",
                "preset contains links that do not contribute to DISPLAY: "
                + ", ".join(map(str, unused_links)),
                link_indices=unused_links,
            )

        pipeline.append({"effect_type": "Display", "params": {}, "node_type": "display"})
        return pipeline


def compile_preset_graph(
    preset: Mapping[str, Any],
    source_namespace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile one saved v1 editor graph into the production flat pipeline.

    The function does not import, instantiate, or inspect Dear PyGui nodes.  It
    returns fresh dictionaries and never mutates ``preset`` or
    ``source_namespace``.
    """

    if not isinstance(preset, Mapping):
        _fail("invalid_graph", "preset must be an object")
    if str(preset.get("version", "1.0")) != "1.0":
        _fail(
            "unsupported_version",
            "compile_preset_graph accepts one Python node preset version 1.0 graph; "
            "select a deck before compiling a dual-deck session",
            version=preset.get("version"),
        )
    return _Compiler(preset, source_namespace).compile()


__all__ = ["GraphCompileError", "compile_preset_graph"]
