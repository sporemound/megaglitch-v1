from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

import dearpygui.dearpygui as dpg


@dataclass
class ParameterTarget:
    editor_name: str
    node: object
    node_label: str
    node_tag: int
    param_name: str
    control_tag: int
    param_def: Dict[str, Any]

    @property
    def label(self) -> str:
        param_label = self.param_def.get("label") or self.param_def.get("compact_label") or self.param_name
        return f"{self.editor_name} / {self.node_label} / {param_label}"


class HardwareMappingManager:
    """Ableton-style MIDI learn for node parameters.

    Mapping mode flow:
    1. Enable mapping mode.
    2. Left-click a node parameter control.
    3. Move a hardware CC control.
    """

    def __init__(self, node_editors: Iterable[Tuple[str, object]]):
        self.node_editors = list(node_editors)
        self.enabled = False
        self.status_message = "MIDI mapping mode off"
        self._pending_target: Optional[ParameterTarget] = None
        self._target_by_control: Dict[int, ParameterTarget] = {}
        self._handler_registry_by_control: Dict[int, int] = {}
        self._mappings: Dict[str, ParameterTarget] = {}
        self._events: Deque[Tuple[str, int]] = deque()
        self._lock = threading.RLock()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            self.status_message = "MIDI mapping: click a parameter"
        else:
            self._pending_target = None
            self.status_message = f"MIDI mapping off ({len(self._mappings)} mapped)"

    def clear(self) -> None:
        with self._lock:
            self._mappings.clear()
            self._events.clear()
        self._pending_target = None
        self.status_message = "MIDI mappings cleared"

    def scan_controls(self) -> None:
        for editor_name, editor in self.node_editors:
            for node in list(getattr(editor, "nodes", []) or []):
                param_controls = getattr(node, "param_controls", {}) or {}
                if not param_controls:
                    continue
                param_defs = self._param_defs_for_node(node)
                for param_name, control_tag in list(param_controls.items()):
                    if not control_tag or control_tag in self._handler_registry_by_control:
                        continue
                    param_def = dict(param_defs.get(param_name) or {})
                    param_type = str(param_def.get("type") or self._infer_param_type(control_tag))
                    if param_type not in {"float", "int", "checkbox", "combo"}:
                        continue
                    target = ParameterTarget(
                        editor_name=editor_name,
                        node=node,
                        node_label=str(getattr(node, "label", "Node")),
                        node_tag=int(getattr(node, "node_tag", 0) or 0),
                        param_name=str(param_name),
                        control_tag=int(control_tag),
                        param_def=param_def,
                    )
                    self._target_by_control[int(control_tag)] = target
                    self._bind_control(target)

    def handle_midi(self, msg) -> bool:
        key = self._hardware_key(msg)
        if not key:
            return False
        value = int(getattr(msg, "value", 0))
        consumed = False
        with self._lock:
            if self.enabled and self._pending_target is not None:
                self._mappings[key] = self._pending_target
                self.status_message = f"Mapped {key} -> {self._pending_target.label}"
                self._pending_target = None
                consumed = True
            if key in self._mappings:
                self._events.append((key, value))
                consumed = True
        return consumed

    def update(self) -> None:
        self.scan_controls()
        while True:
            with self._lock:
                if not self._events:
                    break
                key, value = self._events.popleft()
                target = self._mappings.get(key)
            if target is not None:
                self._apply_value(target, value)

    def _bind_control(self, target: ParameterTarget) -> None:
        try:
            with dpg.item_handler_registry() as registry:
                dpg.add_item_clicked_handler(
                    button=dpg.mvMouseButton_Left,
                    callback=self._parameter_clicked,
                    user_data=target.control_tag,
                )
            dpg.bind_item_handler_registry(target.control_tag, registry)
            self._handler_registry_by_control[target.control_tag] = registry
        except Exception:
            logging.debug("Could not bind MIDI mapping click handler", exc_info=True)

    def _parameter_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        if not self.enabled:
            return
        target = self._target_by_control.get(int(user_data or 0))
        if not target:
            return
        self._pending_target = target
        self.status_message = f"Move a hardware control for {target.label}"

    def _apply_value(self, target: ParameterTarget, value: int) -> None:
        if not dpg.does_item_exist(target.control_tag):
            return
        mapped_value = self._map_value(target, value)
        node = target.node
        try:
            if hasattr(node, "_set_param_value"):
                node._set_param_value(target.param_name, mapped_value, update_control=True)
            elif hasattr(node, "update_param"):
                node.update_param(target.param_name, mapped_value)
                dpg.set_value(target.control_tag, mapped_value)
            else:
                getattr(node, "param_values", {})[target.param_name] = mapped_value
                dpg.set_value(target.control_tag, mapped_value)
            self.status_message = f"{target.label}: {mapped_value}"
        except Exception:
            logging.debug("Could not apply hardware mapped value", exc_info=True)

    def _map_value(self, target: ParameterTarget, value: int):
        param_type = str(target.param_def.get("type") or self._infer_param_type(target.control_tag))
        norm = max(0.0, min(1.0, float(value) / 127.0))
        if param_type == "checkbox":
            return value >= 64
        if param_type == "combo":
            items = list(target.param_def.get("items") or [])
            if not items:
                try:
                    config = dpg.get_item_configuration(target.control_tag)
                    items = list(config.get("items") or [])
                except Exception:
                    items = []
            if not items:
                return dpg.get_value(target.control_tag)
            index = max(0, min(len(items) - 1, int(round(norm * (len(items) - 1)))))
            return items[index]
        low, high = self._range_for_target(target)
        mapped = low + norm * (high - low)
        if param_type == "int":
            return int(round(mapped))
        return float(mapped)

    def _range_for_target(self, target: ParameterTarget) -> Tuple[float, float]:
        param_def = target.param_def
        low = param_def.get("min_value", param_def.get("min", None))
        high = param_def.get("max_value", param_def.get("max", None))
        if low is None or high is None:
            try:
                config = dpg.get_item_configuration(target.control_tag)
                low = config.get("min_value", 0.0) if low is None else low
                high = config.get("max_value", 1.0) if high is None else high
            except Exception:
                low = 0.0 if low is None else low
                high = 1.0 if high is None else high
        try:
            return float(low), float(high)
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def _hardware_key(msg) -> str:
        msg_type = getattr(msg, "type", "")
        if msg_type == "control_change":
            channel = int(getattr(msg, "channel", 0))
            control = int(getattr(msg, "control", 0))
            return f"ch{channel + 1}:cc{control}"
        if msg_type == "pitchwheel":
            channel = int(getattr(msg, "channel", 0))
            return f"ch{channel + 1}:pitchwheel"
        return ""

    @staticmethod
    def _infer_param_type(control_tag: int) -> str:
        try:
            item_type = str(dpg.get_item_type(control_tag) or "").lower()
        except Exception:
            return "float"
        if "checkbox" in item_type:
            return "checkbox"
        if "combo" in item_type:
            return "combo"
        if "int" in item_type:
            return "int"
        return "float"

    @staticmethod
    def _param_defs_for_node(node) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for attr_name in ("params", "params_def", "base_param_defs", "dynamic_param_defs"):
            params = getattr(node, attr_name, None)
            if isinstance(params, dict):
                params = params.values()
            for param in list(params or []):
                if isinstance(param, dict) and param.get("name"):
                    out[str(param["name"])] = dict(param)
        metadata = getattr(node, "param_metadata", None)
        if isinstance(metadata, dict):
            for name, param in metadata.items():
                if isinstance(param, dict):
                    out[str(name)] = dict(param)
        return out
