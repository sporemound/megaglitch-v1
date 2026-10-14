import logging
from typing import Any, Callable, Dict, List, Optional

import dearpygui.dearpygui as dpg


class ParameterTimeline:
    """Node-specific parameter keyframes evaluated against the render timeline."""

    def __init__(
        self,
        node_editor_provider: Callable[[], Any],
        current_time_provider: Callable[[], float],
    ):
        self.node_editor_provider = node_editor_provider
        self.current_time_provider = current_time_provider
        self.enabled = True
        self.tracks: Dict[str, Dict[str, Any]] = {}
        self.window_tag = "parameter_timeline_window"
        self.target_combo_tag = "parameter_timeline_target"
        self.time_input_tag = "parameter_timeline_time"
        self.value_input_tag = "parameter_timeline_value"
        self_interp_tag = "parameter_timeline_interp"
        self.interp_combo_tag = self_interp_tag
        self.keyframe_list_tag = "parameter_timeline_keyframes"
        self.status_tag = "parameter_timeline_status"
        self._target_cache: Dict[str, Dict[str, Any]] = {}

    def _collect_targets(self) -> List[Dict[str, Any]]:
        editor = self.node_editor_provider() if callable(self.node_editor_provider) else None
        targets: List[Dict[str, Any]] = []
        if editor is None:
            return targets

        for node in getattr(editor, "nodes", []) or []:
            params = getattr(node, "params", None) or getattr(node, "params_def", None)
            values = getattr(node, "param_values", None)
            if not params or not isinstance(values, dict):
                continue
            if isinstance(params, dict):
                params = list(params.values())

            node_id = str(getattr(node, "node_tag", ""))
            if not node_id:
                continue
            node_label = str(getattr(node, "label", node.__class__.__name__))
            node_suffix = node_id[-5:]

            for param in params:
                if not isinstance(param, dict):
                    continue
                name = param.get("name")
                param_type = str(param.get("type") or "")
                if not name or name == "enabled":
                    continue
                if param_type not in {"float", "int", "checkbox"}:
                    continue
                target_id = f"{node_id}:{name}"
                label = f"{node_label} [{node_suffix}] / {param.get('label') or name}"
                targets.append({
                    "id": target_id,
                    "label": label,
                    "node_id": node_id,
                    "node_label": node_label,
                    "param": name,
                    "type": param_type,
                    "min": param.get("min_value", param.get("min", 0.0)),
                    "max": param.get("max_value", param.get("max", 1.0)),
                    "value": values.get(name, param.get("default_value", 0.0)),
                })

        targets.sort(key=lambda item: item["label"].lower())
        self._target_cache = {item["id"]: item for item in targets}
        return targets

    def _coerce_value(self, target: Dict[str, Any], value: Any) -> Any:
        param_type = target.get("type")
        if param_type == "checkbox":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        try:
            mn = float(target.get("min", numeric))
            mx = float(target.get("max", numeric))
            numeric = max(mn, min(mx, numeric))
        except (TypeError, ValueError):
            pass
        if param_type == "int":
            return int(round(numeric))
        return float(numeric)

    def _track_for_target(self, target_id: str) -> Dict[str, Any]:
        target = self._target_cache.get(target_id)
        if target is None:
            targets = self._collect_targets()
            target = next((item for item in targets if item["id"] == target_id), None)
        if target is None:
            raise KeyError(f"Unknown timeline target: {target_id}")

        track = self.tracks.setdefault(target_id, {
            "target_id": target_id,
            "node_id": target["node_id"],
            "node_label": target["node_label"],
            "param": target["param"],
            "type": target["type"],
            "label": target["label"],
            "keyframes": [],
        })
        track.update({
            "node_id": target["node_id"],
            "node_label": target["node_label"],
            "param": target["param"],
            "type": target["type"],
            "label": target["label"],
        })
        return track

    def add_keyframe(self, target_id: str, seconds: float, value: Any, interpolation: str = "linear"):
        track = self._track_for_target(target_id)
        target = self._target_cache[target_id]
        seconds = max(0.0, float(seconds))
        value = self._coerce_value(target, value)
        interpolation = interpolation if interpolation in {"linear", "ease", "step"} else "linear"
        keyframes = track.setdefault("keyframes", [])
        keyframes[:] = [kf for kf in keyframes if abs(float(kf.get("time", 0.0)) - seconds) > 1e-4]
        keyframes.append({"time": seconds, "value": value, "interp": interpolation})
        keyframes.sort(key=lambda kf: float(kf.get("time", 0.0)))

    def remove_keyframe_label(self, label: str):
        if not label:
            return
        for target_id, track in list(self.tracks.items()):
            keyframes = track.get("keyframes", [])
            kept = []
            for kf in keyframes:
                if self._format_keyframe(track, kf) != label:
                    kept.append(kf)
            track["keyframes"] = kept
            if not kept:
                self.tracks.pop(target_id, None)

    def evaluate_track(self, track: Dict[str, Any], seconds: float) -> Optional[Any]:
        keyframes = sorted(track.get("keyframes", []), key=lambda kf: float(kf.get("time", 0.0)))
        if not keyframes:
            return None
        seconds = float(seconds)
        if seconds <= float(keyframes[0].get("time", 0.0)):
            return keyframes[0].get("value")
        if seconds >= float(keyframes[-1].get("time", 0.0)):
            return keyframes[-1].get("value")

        prev = keyframes[0]
        nxt = keyframes[-1]
        for index in range(1, len(keyframes)):
            candidate = keyframes[index]
            if seconds <= float(candidate.get("time", 0.0)):
                prev = keyframes[index - 1]
                nxt = candidate
                break

        interp = str(nxt.get("interp", prev.get("interp", "linear")) or "linear")
        if interp == "step" or track.get("type") == "checkbox":
            return prev.get("value")

        t0 = float(prev.get("time", 0.0))
        t1 = float(nxt.get("time", t0))
        amount = 0.0 if t1 <= t0 else (seconds - t0) / (t1 - t0)
        amount = max(0.0, min(1.0, amount))
        if interp == "ease":
            amount = amount * amount * (3.0 - 2.0 * amount)
        try:
            value = float(prev.get("value", 0.0)) * (1.0 - amount) + float(nxt.get("value", 0.0)) * amount
        except (TypeError, ValueError):
            return prev.get("value")
        if track.get("type") == "int":
            return int(round(value))
        return value

    def apply_to_pipeline(self, pipeline: List[Dict[str, Any]], seconds: float) -> Dict[str, Dict[str, Any]]:
        if not self.enabled or not self.tracks:
            return {}

        changes: Dict[str, Dict[str, Any]] = {}
        for track in self.tracks.values():
            value = self.evaluate_track(track, seconds)
            if value is None:
                continue
            node_id = str(track.get("node_id", ""))
            param = str(track.get("param", ""))
            if not node_id or not param:
                continue
            changes.setdefault(node_id, {})[param] = value

        if not changes:
            return {}

        for effect_config in pipeline or []:
            node_id = str(effect_config.get("node_id", ""))
            node_changes = changes.get(node_id)
            if not node_changes:
                continue
            params = dict(effect_config.get("params") or {})
            for name, value in node_changes.items():
                if name in params:
                    params[name] = value
            effect_config["params"] = params
        return changes

    def sync_controls(self, editors: List[Any], changes: Dict[str, Dict[str, Any]]):
        if not changes:
            return
        for editor in editors:
            if editor is None:
                continue
            for node in getattr(editor, "nodes", []) or []:
                node_id = str(getattr(node, "node_tag", ""))
                node_changes = changes.get(node_id)
                if not node_changes:
                    continue
                for name, value in node_changes.items():
                    control = getattr(node, "param_controls", {}).get(name)
                    if control and dpg.does_item_exist(control):
                        try:
                            dpg.set_value(control, value)
                        except Exception:
                            pass

    def _format_keyframe(self, track: Dict[str, Any], keyframe: Dict[str, Any]) -> str:
        return (
            f"{track.get('label', track.get('param', 'Param'))} | "
            f"{float(keyframe.get('time', 0.0)):.3f}s = {keyframe.get('value')} "
            f"({keyframe.get('interp', 'linear')})"
        )

    def _keyframe_labels(self) -> List[str]:
        labels: List[str] = []
        for track in sorted(self.tracks.values(), key=lambda item: str(item.get("label", "")).lower()):
            for keyframe in sorted(track.get("keyframes", []), key=lambda kf: float(kf.get("time", 0.0))):
                labels.append(self._format_keyframe(track, keyframe))
        return labels

    def _refresh_keyframe_list(self):
        if dpg.does_item_exist(self.keyframe_list_tag):
            dpg.configure_item(self.keyframe_list_tag, items=self._keyframe_labels())

    def _selected_target(self) -> Optional[Dict[str, Any]]:
        if not dpg.does_item_exist(self.target_combo_tag):
            return None
        selected_label = dpg.get_value(self.target_combo_tag)
        for target in self._target_cache.values():
            if target["label"] == selected_label:
                return target
        return None

    def _capture_current_value(self):
        target = self._selected_target()
        if target is None:
            return
        if dpg.does_item_exist(self.value_input_tag):
            try:
                dpg.set_value(self.value_input_tag, target.get("value", 0.0))
            except Exception:
                pass

    def _set_current_time(self):
        if dpg.does_item_exist(self.time_input_tag):
            try:
                dpg.set_value(self.time_input_tag, float(self.current_time_provider()))
            except Exception:
                pass

    def _add_from_ui(self):
        target = self._selected_target()
        if target is None:
            return
        try:
            seconds = float(dpg.get_value(self.time_input_tag))
            value = dpg.get_value(self.value_input_tag)
            interp = str(dpg.get_value(self.interp_combo_tag) or "linear")
            self.add_keyframe(target["id"], seconds, value, interp)
            self._refresh_keyframe_list()
            if dpg.does_item_exist(self.status_tag):
                dpg.set_value(self.status_tag, f"Added {target['label']} at {seconds:.3f}s")
        except Exception as e:
            logging.warning(f"Could not add timeline keyframe: {e}")
            if dpg.does_item_exist(self.status_tag):
                dpg.set_value(self.status_tag, f"Could not add keyframe: {e}")

    def _remove_from_ui(self):
        selected = dpg.get_value(self.keyframe_list_tag) if dpg.does_item_exist(self.keyframe_list_tag) else ""
        self.remove_keyframe_label(selected)
        self._refresh_keyframe_list()

    def _refresh_targets_ui(self):
        targets = self._collect_targets()
        labels = [target["label"] for target in targets]
        if dpg.does_item_exist(self.target_combo_tag):
            dpg.configure_item(self.target_combo_tag, items=labels)
            if labels:
                current = dpg.get_value(self.target_combo_tag)
                if current not in labels:
                    dpg.set_value(self.target_combo_tag, labels[0])
                    self._capture_current_value()
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, f"{len(labels)} numeric/boolean targets")

    def show_window(self):
        targets = self._collect_targets()
        labels = [target["label"] for target in targets]
        current_target = labels[0] if labels else ""
        current_value = targets[0].get("value", 0.0) if targets else 0.0

        if dpg.does_item_exist(self.window_tag):
            dpg.delete_item(self.window_tag)

        with dpg.window(label="Parameter Timeline", width=720, height=460, tag=self.window_tag):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Enable", default_value=self.enabled, callback=lambda s, a: setattr(self, "enabled", bool(a)))
                dpg.add_button(label="Use Current Time", callback=lambda: self._set_current_time())
                dpg.add_button(label="Refresh Targets", callback=lambda: self._refresh_targets_ui())

            dpg.add_combo(
                label="Target",
                items=labels,
                default_value=current_target,
                width=520,
                tag=self.target_combo_tag,
                callback=lambda s, a: self._capture_current_value(),
            )
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="Time", default_value=float(self.current_time_provider()), width=160, tag=self.time_input_tag, format="%.3f")
                dpg.add_input_float(label="Value", default_value=float(current_value or 0.0), width=160, tag=self.value_input_tag, format="%.4f")
                dpg.add_combo(label="Interp", items=["linear", "ease", "step"], default_value="linear", width=130, tag=self.interp_combo_tag)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Capture Value", callback=lambda: self._capture_current_value())
                dpg.add_button(label="Add / Update Key", callback=lambda: self._add_from_ui())
                dpg.add_button(label="Remove Selected", callback=lambda: self._remove_from_ui())

            dpg.add_listbox(items=self._keyframe_labels(), width=-1, num_items=12, tag=self.keyframe_list_tag)
            dpg.add_text(f"{len(labels)} numeric/boolean targets", tag=self.status_tag)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "tracks": self.tracks,
        }

    def from_dict(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        self.enabled = bool(data.get("enabled", True))
        tracks = data.get("tracks", {})
        if isinstance(tracks, dict):
            self._collect_targets()
            remapped: Dict[str, Dict[str, Any]] = {}
            for old_target_id, old_track in tracks.items():
                if not isinstance(old_track, dict):
                    continue
                target = self._target_cache.get(str(old_target_id))
                if target is None:
                    node_label = str(old_track.get("node_label", ""))
                    param = str(old_track.get("param", ""))
                    target = next(
                        (
                            item for item in self._target_cache.values()
                            if item.get("node_label") == node_label and item.get("param") == param
                        ),
                        None,
                    )
                if target is None:
                    continue
                new_track = dict(old_track)
                new_track.update({
                    "target_id": target["id"],
                    "node_id": target["node_id"],
                    "node_label": target["node_label"],
                    "param": target["param"],
                    "type": target["type"],
                    "label": target["label"],
                })
                remapped[target["id"]] = new_track
            self.tracks = remapped
        self._refresh_keyframe_list()
