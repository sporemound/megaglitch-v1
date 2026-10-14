import dearpygui.dearpygui as dpg
from glitch_effects import CorruptionMode
import logging
import json
import os
from collections import OrderedDict
from pathlib import Path

try:
    from lupa import LuaRuntime
except ImportError:
    LuaRuntime = None


# Narrow control width to keep nodes compact on small monitors
CONTROL_WIDTH = 200
_AUDIOVISUAL_MODULATION_BUSES = {}


def register_audiovisual_modulation_bus(node_editor_tag, bus):
    _AUDIOVISUAL_MODULATION_BUSES[node_editor_tag] = bus


def get_audiovisual_modulation_bus(node_editor_tag):
    return _AUDIOVISUAL_MODULATION_BUSES.get(node_editor_tag)

class Node:
    def __init__(self, label, parent_node_editor):
        self.label = label
        self.parent_node_editor = parent_node_editor
        self.inputs = []
        self.outputs = []
        self.node_tag = dpg.generate_uuid()
        self.input_attrs = {}
        self.output_attrs = {}

    def create_node(self, pos):
        with dpg.node(label=self.label, tag=self.node_tag, parent=self.parent_node_editor, pos=pos):
            self.add_attributes()

    def add_attributes(self):
        pass

    def process(self, input_data):
        pass


class MultiInputNode(Node):
    """Small base for utility nodes that consume multiple image-like inputs."""

    def __init__(self, label, parent_node_editor, params=None, input_defs=None, output_defs=None, node_type_key="multi_input"):
        super().__init__(label, parent_node_editor)
        self.params = [
            {"name": "enabled", "type": "checkbox", "label": "Enabled", "default_value": True}
        ] + list(params or [])
        self.param_values = {param["name"]: param["default_value"] for param in self.params}
        self.param_controls = {}
        self.input_defs = list(input_defs or [])
        self.output_defs = list(output_defs or [{"name": "image_out", "label": "Output", "text": "Image Out"}])
        self.node_type_key = str(node_type_key or "multi_input")

    def add_attributes(self):
        for input_def in self.input_defs:
            attr_label = str(input_def.get("label") or input_def.get("attr_label") or "Input")
            socket_text = str(input_def.get("text") or input_def.get("label") or input_def.get("name") or "Input")
            with dpg.node_attribute(label=attr_label, attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr:
                dpg.add_text(socket_text)
                self.input_attrs[str(input_def["name"])] = attr

        with dpg.node_attribute(label="Parameters", attribute_type=dpg.mvNode_Attr_Static):
            for param_info in self.params:
                self._add_param_control(param_info)

        for output_def in self.output_defs:
            attr_label = str(output_def.get("label") or output_def.get("attr_label") or "Output")
            socket_text = str(output_def.get("text") or output_def.get("label") or output_def.get("name") or "Output")
            with dpg.node_attribute(label=attr_label, attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr:
                dpg.add_text(socket_text)
                self.output_attrs[str(output_def["name"])] = attr

    def _add_param_control(self, param_info):
        param_name = param_info["name"]
        param_type = param_info["type"]
        default_value = param_info["default_value"]

        control_tag = None
        if param_type == "float":
            control_tag = dpg.add_slider_float(
                label=param_info["label"],
                min_value=param_info["min_value"],
                max_value=param_info["max_value"],
                default_value=default_value,
                width=CONTROL_WIDTH,
                format="%.3f",
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
            )
        elif param_type == "int":
            control_tag = dpg.add_slider_int(
                label=param_info["label"],
                min_value=param_info["min_value"],
                max_value=param_info["max_value"],
                default_value=default_value,
                width=CONTROL_WIDTH,
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
            )
        elif param_type == "combo":
            control_tag = dpg.add_combo(
                label=param_info["label"],
                items=param_info["items"],
                default_value=default_value,
                width=CONTROL_WIDTH,
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
            )
        elif param_type == "checkbox":
            control_tag = dpg.add_checkbox(
                label=param_info["label"],
                default_value=default_value,
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
            )

        if control_tag:
            self.param_controls[param_name] = control_tag
            tooltip = param_info.get("tooltip")
            if tooltip:
                try:
                    with dpg.tooltip(parent=control_tag):
                        dpg.add_text(str(tooltip), wrap=260)
                except Exception:
                    pass

    def update_param(self, param_name, value):
        self.param_values[param_name] = value

class GlitchEffectNode(Node):
    def __init__(self, label, parent_node_editor, params):
        super().__init__(label, parent_node_editor)
        self.params = [
            {"name": "enabled", "type": "checkbox", "label": "Enabled", "default_value": True},
            {
                "name": "multimodal_amount",
                "type": "float",
                "label": "Multimodal Select",
                "min_value": 0.0,
                "max_value": 1.0,
                "default_value": 0.0,
                "tooltip": (
                    "Spatially gates this node with the upstream master-resolution "
                    "motion/edge/luma/entropy/spectrum/audio analysis mask."
                ),
            },
        ] + list(params)
        self.param_values = {}
        self.param_controls = {}  # Store UI control tags for each parameter

    @staticmethod
    def with_evolution_params(params, prefix, amount_label="Evolve", speed_label="Evolve Speed",
                              amount_default=0.0, speed_default=1.0):
        return params + [
            {"name": f"{prefix}_evolution", "type": "float", "label": amount_label,
             "min_value": 0.0, "max_value": 1.0, "default_value": amount_default},
            {"name": f"{prefix}_evolution_speed", "type": "float", "label": speed_label,
             "min_value": 0.0, "max_value": 5.0, "default_value": speed_default},
        ]

    def add_attributes(self):
        with dpg.node_attribute(label="Input", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Image In")
            self.input_attrs["image_in"] = attr

        with dpg.node_attribute(label="Parameters", attribute_type=dpg.mvNode_Attr_Static):
            for param_info in self.params:
                self._add_param_control(param_info)

        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Image Out")
            self.output_attrs["image_out"] = attr

    def _add_param_control(self, param_info, parent=None):
        param_name = param_info["name"]
        param_type = param_info["type"]
        param_label = param_info.get("compact_label", param_info["label"])
        default_value = param_info["default_value"]
        control_width = int(param_info.get("control_width", CONTROL_WIDTH))

        kwargs = {}
        if parent is not None:
            kwargs["parent"] = parent

        control_tag = None
        if param_type == "float":
            control_tag = dpg.add_slider_float(
                label=param_label,
                min_value=param_info["min_value"],
                max_value=param_info["max_value"],
                default_value=default_value,
                width=control_width,
                format="%.3f",
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )
        elif param_type == "int":
            control_tag = dpg.add_slider_int(
                label=param_label,
                min_value=param_info["min_value"],
                max_value=param_info["max_value"],
                default_value=default_value,
                width=control_width,
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )
        elif param_type == "combo":
            control_tag = dpg.add_combo(
                label=param_label,
                items=param_info["items"],
                default_value=default_value,
                width=control_width,
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )
        elif param_type == "checkbox":
            control_tag = dpg.add_checkbox(
                label=param_label,
                default_value=default_value,
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )
        elif param_type == "text":
            control_tag = dpg.add_input_text(
                label=param_label,
                default_value=str(default_value) if default_value is not None else "",
                width=control_width,
                user_data=param_name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )

        if control_tag:
            self.param_controls[param_name] = control_tag
            tooltip = param_info.get("tooltip")
            if tooltip:
                try:
                    with dpg.tooltip(parent=control_tag):
                        dpg.add_text(str(tooltip), wrap=260)
                except Exception:
                    pass

        self.param_values[param_name] = default_value
        return control_tag

    def update_param(self, param_name, value):
        """Update parameter value and ensure control stays enabled."""
        self.param_values[param_name] = value
        
        # Ensure the control stays enabled/responsive
        if param_name in self.param_controls:
            try:
                control_tag = self.param_controls[param_name]
                if dpg.does_item_exist(control_tag):
                    # Re-enable if it got disabled somehow
                    dpg.configure_item(control_tag, enabled=True)
            except Exception:
                pass

class LuaShaderNode(GlitchEffectNode):
    """Lua + GLSL Shader node with metadata-driven parameters."""

    def __init__(self, parent_node_editor):
        self.lua_input_defs = [
            {"name": "image_in", "attr_label": "Input", "text": "Image In"},
            {"name": "feedback_in", "attr_label": "Feedback Input", "text": "Feedback In"},
        ]
        self.base_param_defs = [
            {"name": "shader_preset", "type": "combo", "label": "Shader Pack", "items": [], "default_value": "effect"},
            {"name": "shader_path", "type": "text", "label": "Frag Shader", "default_value": "lua_glsl_demo/shaders/effect.frag"},
            {"name": "lua_path", "type": "text", "label": "Lua Script", "default_value": "lua_glsl_demo/scripts/effect.lua"},
            {"name": "metadata_path", "type": "text", "label": "Effect Metadata", "default_value": "lua_glsl_demo/scripts/effect.effect"},
            {"name": "use_input", "type": "checkbox", "label": "Use Input", "default_value": True},
            {"name": "freeze_time", "type": "checkbox", "label": "Freeze Time", "default_value": False},
            {"name": "flip_y", "type": "checkbox", "label": "Flip Y (readback)", "default_value": True},
        ]
        super().__init__("Lua Shader", parent_node_editor, self.base_param_defs)
        self.base_param_names = {p["name"] for p in self.base_param_defs}
        self.param_metadata = {}
        self.param_controls = {}
        self.param_values = {}
        self.dynamic_param_defs = []
        self.dynamic_param_names = set()
        self.dynamic_group_tag = None
        self.dynamic_header_tag = None
        self.dynamic_status_tag = None
        self.base_group_tag = None
        self.base_dir = Path(__file__).resolve().parent
        preset_items = self._shader_preset_names()
        self.base_param_defs[0]["items"] = preset_items
        self.base_param_defs[0]["default_value"] = "effect" if "effect" in preset_items else (preset_items[0] if preset_items else "")
        self._lua_runtime = None
        self._lua_runtime_failed = False
        self._suppress_callbacks = False

    def add_attributes(self):
        for input_def in self.lua_input_defs:
            with dpg.node_attribute(
                label=str(input_def.get("attr_label") or "Input"),
                attribute_type=dpg.mvNode_Attr_Input,
                tag=dpg.generate_uuid(),
            ) as attr:
                dpg.add_text(str(input_def.get("text") or input_def["name"]))
                self.input_attrs[str(input_def["name"])] = attr

        with dpg.node_attribute(label="Paths & Options", attribute_type=dpg.mvNode_Attr_Static, tag=dpg.generate_uuid()):
            with dpg.group(horizontal=False) as self.base_group_tag:
                for param_def in self.base_param_defs:
                    coerced = self._coerce_value(param_def, param_def.get("default_value"))
                    self.param_metadata[param_def["name"]] = param_def
                    self.param_values[param_def["name"]] = coerced
                    ctrl = self._create_control(param_def, coerced)
                    self.param_controls[param_def["name"]] = ctrl

        with dpg.node_attribute(label="Effect Parameters", attribute_type=dpg.mvNode_Attr_Static, tag=dpg.generate_uuid()):
            self.dynamic_header_tag = dpg.add_text("Effect Parameters", show=False)
            self.dynamic_status_tag = dpg.add_text("", show=False, wrap=CONTROL_WIDTH)
            self.dynamic_group_tag = dpg.add_group(horizontal=False)

        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Image Out")
            self.output_attrs["image_out"] = attr

        self._reload_effect_params()

    def update_param(self, param_name, value):
        self._set_param_value(param_name, value, update_control=False)

    # ---- metadata + discovery helpers ----
    def _ensure_lua_runtime(self):
        if self._lua_runtime_failed:
            return None
        if self._lua_runtime is None:
            if LuaRuntime is None:
                logging.warning("LuaRuntime (lupa) not available; Lua parameter metadata disabled.")
                self._lua_runtime_failed = True
                return None
            try:
                self._lua_runtime = LuaRuntime(unpack_returned_tuples=True)
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to initialise Lua runtime: %s", exc)
                self._lua_runtime_failed = True
                return None
        return self._lua_runtime

    def _reload_effect_params(self):
        if not self.dynamic_group_tag:
            return

        lua_value = str(self.param_values.get("lua_path") or "").strip()
        meta_value = str(self.param_values.get("metadata_path") or "").strip()

        candidate_paths = []
        if meta_value:
            candidate_paths.append(self._resolve_path(meta_value))
        if lua_value:
            if lua_value.lower().endswith(".effect"):
                candidate_paths.append(self._resolve_path(lua_value))
            else:
                lua_path = self._resolve_path(lua_value)
                if lua_path is not None:
                    candidate_paths.append(lua_path.with_suffix(".effect"))

        metadata_path = None
        for candidate in candidate_paths:
            if candidate is None:
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.exists():
                metadata_path = resolved
                break

        if metadata_path is None:
            suggestion = {}
            shader_value = self.param_values.get("shader_path")
            if shader_value:
                suggestion = self._find_effect_for_shader(shader_value) or {}
            if meta_value:
                name_hint = Path(meta_value).stem
                if name_hint:
                    fallback = self._find_effect_by_name(name_hint) or {}
                    suggestion.setdefault("effect", fallback.get("effect"))
                    suggestion.setdefault("lua", fallback.get("lua"))

            effect_suggestion = suggestion.get("effect")
            lua_suggestion = suggestion.get("lua")
            if effect_suggestion and effect_suggestion != self.param_values.get("metadata_path"):
                self._set_param_value("metadata_path", effect_suggestion, update_control=True)
                return
            if lua_suggestion and lua_suggestion != self.param_values.get("lua_path"):
                self._set_param_value("lua_path", lua_suggestion, update_control=True)
                return

            missing_label = meta_value if meta_value else "(none)"
            self._apply_dynamic_params([], message=f"Effect metadata not found: {missing_label}")
            return

        normalized = self._to_param_path(metadata_path)
        if normalized != meta_value:
            self._set_param_value("metadata_path", normalized, update_control=True)
            return

        if not lua_value or lua_value.lower().endswith(".effect"):
            linked = self._find_effect_by_name(metadata_path.stem)
            lua_candidate = linked.get("lua") if linked else None
            if lua_candidate and lua_candidate != self.param_values.get("lua_path"):
                self._set_param_value("lua_path", lua_candidate, update_control=True)
                return

        param_defs, error = self._load_effect_param_defs(metadata_path)
        if error:
            self._apply_dynamic_params([], message=error)
        else:
            self._apply_dynamic_params(param_defs, message=None)

    def _load_effect_param_defs(self, metadata_path: Path):
        ext = metadata_path.suffix.lower()
        if ext == ".effect":
            return self._load_effect_file(metadata_path)
        param_defs, error = self._load_lua_param_defs(metadata_path)
        if param_defs:
            return param_defs, ""
        if error:
            fallback = metadata_path.with_suffix(".effect")
            if fallback.exists():
                return self._load_effect_file(fallback)
            return [], error
        fallback = metadata_path.with_suffix(".effect")
        if fallback.exists():
            return self._load_effect_file(fallback)
        return [], "Effect script defines no parameters."

    def _load_lua_param_defs(self, lua_path: Path):
        runtime = self._ensure_lua_runtime()
        if runtime is None:
            return [], "Lua runtime unavailable (install lupa)."
        try:
            script = lua_path.read_text(encoding="utf-8")
        except OSError as exc:
            logging.warning("LuaShaderNode: failed to read %s: %s", lua_path, exc)
            return [], f"Unable to read effect: {exc}"
        try:
            module = runtime.execute(script)
        except Exception as exc:  # pragma: no cover
            logging.warning("LuaShaderNode: execution error in %s: %s", lua_path, exc)
            return [], f"Lua error: {exc}"

        params_table = None
        order_table = None
        if hasattr(module, "items"):
            for key, value in module.items():
                key_str = str(key)
                if key_str == "params":
                    params_table = value
                elif key_str == "params_order":
                    order_table = value
        if params_table is None:
            return [], "Effect script does not expose a params table."

        py_params = self._lua_table_to_python(params_table)
        order_list = self._lua_table_to_python(order_table) if order_table is not None else None
        param_defs = self._extract_param_definitions(py_params, order_list)
        return param_defs, ""

    def _load_effect_file(self, effect_path: Path):
        try:
            try:
                raw = effect_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw = effect_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logging.warning("LuaShaderNode: failed to read %s: %s", effect_path, exc)
            return [], f"Unable to read effect metadata: {exc}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logging.warning("LuaShaderNode: invalid metadata in %s: %s", effect_path, exc)
            return [], f"Invalid effect metadata: {exc}"

        params_data = data.get("params")
        if params_data is None:
            return [], "Effect metadata missing params."

        if isinstance(params_data, list):
            params_dict = OrderedDict()
            for entry in params_data:
                if isinstance(entry, dict) and entry.get("name"):
                    params_dict[str(entry["name"])] = dict(entry)
        elif isinstance(params_data, dict):
            params_dict = OrderedDict(
                (str(name), dict(meta))
                for name, meta in params_data.items()
                if isinstance(meta, dict)
            )
        else:
            return [], "Effect metadata has unexpected params structure."

        if not params_dict:
            return [], "Effect metadata defines no parameters."

        order_value = data.get("order") or data.get("params_order")
        order_list = None
        if isinstance(order_value, list):
            order_list = [str(item) for item in order_value]

        return self._extract_param_definitions(params_dict, order_list), ""

    def _lua_table_to_python(self, obj):
        if hasattr(obj, "items"):
            dict_entries = OrderedDict()
            array_entries = []
            for key, value in obj.items():
                py_value = self._lua_table_to_python(value)
                if isinstance(key, (int, float)) and int(key) == key:
                    array_entries.append((int(key), py_value))
                else:
                    dict_entries[str(key)] = py_value
            if dict_entries and not array_entries:
                return dict_entries
            if not dict_entries:
                array_entries.sort(key=lambda pair: pair[0])
                return [val for _, val in array_entries]
            for idx, val in array_entries:
                dict_entries[str(idx)] = val
            return dict_entries
        return obj

    def _extract_param_definitions(self, params, order):
        if not isinstance(params, (dict, OrderedDict)):
            return []
        names = []
        if isinstance(order, (list, tuple)):
            for entry in order:
                if isinstance(entry, str) and entry in params and entry not in names:
                    names.append(entry)
        for name in params.keys():
            if name not in names:
                names.append(name)

        definitions = []
        for name in names:
            meta = params.get(name)
            if not isinstance(meta, dict):
                continue
            param_type = str(meta.get("type", "")).strip().lower()
            items = meta.get("items") or meta.get("options")
            label = meta.get("label") or name
            default = meta.get("value")
            min_value = meta.get("min")
            max_value = meta.get("max")
            step = meta.get("step")

            if not param_type:
                if items:
                    param_type = "combo"
                elif isinstance(default, bool):
                    param_type = "checkbox"
                elif isinstance(default, int):
                    param_type = "int"
                elif isinstance(default, (int, float)):
                    param_type = "float"
                else:
                    param_type = "text"

            if param_type == "combo":
                if isinstance(items, dict):
                    items = list(items.values())
                items = [str(x) for x in (items or [])]
                if default is None and items:
                    default = items[0]
                elif default is not None:
                    default = str(default)
            elif param_type == "checkbox":
                default = bool(default)
            elif param_type == "int":
                default = int(round(float(default or 0)))
                if min_value is not None:
                    min_value = int(round(float(min_value)))
                if max_value is not None:
                    max_value = int(round(float(max_value)))
                if step is not None:
                    step = max(1, int(round(float(step))))
            elif param_type == "float":
                default = float(default if default is not None else 0.0)
                if min_value is not None:
                    min_value = float(min_value)
                if max_value is not None:
                    max_value = float(max_value)
                if step is not None:
                    step = float(step)
            else:
                default = "" if default is None else str(default)

            definitions.append({
                "name": name,
                "type": param_type,
                "label": str(label),
                "default_value": default,
                "min_value": min_value,
                "max_value": max_value,
                "step": step,
                "items": [str(x) for x in items] if param_type == "combo" else None,
            })
        return definitions

    def _apply_dynamic_params(self, param_defs, message):
        dpg.delete_item(self.dynamic_group_tag, children_only=True)
        new_names = {p["name"] for p in param_defs}
        for name in list(self.param_metadata.keys()):
            if name not in self.base_param_names and name not in new_names:
                self.param_metadata.pop(name, None)
        for name in list(self.param_controls.keys()):
            if name not in self.base_param_names and name not in new_names:
                self.param_controls.pop(name, None)
        for name in list(self.param_values.keys()):
            if name not in self.base_param_names and name not in new_names:
                self.param_values.pop(name, None)

        self.dynamic_param_defs = param_defs
        self.dynamic_param_names = new_names

        dpg.configure_item(self.dynamic_header_tag, show=bool(param_defs))

        if not param_defs:
            info = message or "Effect metadata defines no parameters."
            self._show_dynamic_status(info)
            return

        self._show_dynamic_status("")
        dpg.push_container_stack(self.dynamic_group_tag)
        try:
            for param_def in param_defs:
                coerced = self._coerce_value(param_def, self.param_values.get(param_def["name"], param_def.get("default_value")))
                self.param_metadata[param_def["name"]] = param_def
                self.param_values[param_def["name"]] = coerced
                ctrl = self._create_control(param_def, coerced)
                self.param_controls[param_def["name"]] = ctrl
        finally:
            dpg.pop_container_stack()

    def _show_dynamic_status(self, message):
        if not self.dynamic_status_tag:
            return
        message = (message or "").strip()
        if message:
            dpg.set_value(self.dynamic_status_tag, message)
            dpg.configure_item(self.dynamic_status_tag, show=True)
        else:
            dpg.set_value(self.dynamic_status_tag, "")
            dpg.configure_item(self.dynamic_status_tag, show=False)

    def _coerce_value(self, param_def, value):
        param_type = param_def.get("type", "float")
        name = param_def.get("name", "")
        if param_type == "checkbox":
            return bool(value)
        if param_type == "int":
            try:
                result = int(round(float(value if value is not None else param_def.get("default_value") or 0)))
            except (TypeError, ValueError):
                result = int(param_def.get("default_value") or 0)
            min_value = param_def.get("min_value")
            max_value = param_def.get("max_value")
            if min_value is not None:
                result = max(result, int(min_value))
            if max_value is not None:
                result = min(result, int(max_value))
            return result
        if param_type == "float":
            try:
                result = float(value if value is not None else param_def.get("default_value") or 0.0)
            except (TypeError, ValueError):
                result = float(param_def.get("default_value") or 0.0)
            min_value = param_def.get("min_value")
            max_value = param_def.get("max_value")
            if min_value is not None:
                result = max(result, float(min_value))
            if max_value is not None:
                result = min(result, float(max_value))
            return result
        if param_type == "combo":
            items = param_def.get("items") or []
            as_text = str(value if value is not None else param_def.get("default_value") or (items[0] if items else ""))
            if items and as_text not in items:
                as_text = items[0]
            return as_text
        if param_type == "text":
            text = "" if value is None else str(value)
            if name in {"shader_path", "lua_path", "metadata_path"}:
                return self._normalize_path_input(text)
            return text
        return value

    def _create_control(self, param_def, value):
        param_name = param_def["name"]
        param_type = param_def.get("type", "float")
        label = param_def.get("label", param_name)
        common = {"label": label, "user_data": param_name, "callback": self._on_param_changed}
        if param_type == "checkbox":
            return dpg.add_checkbox(default_value=bool(value), **common)
        if param_type == "combo":
            items = param_def.get("items") or []
            return dpg.add_combo(items=items, default_value=str(value), width=CONTROL_WIDTH, **common)
        if param_type == "text":
            return dpg.add_input_text(default_value=str(value or ""), width=CONTROL_WIDTH, on_enter=True, **common)
        if param_type == "int":
            min_value = param_def.get("min_value")
            max_value = param_def.get("max_value")
            if min_value is None:
                min_value = 0
            if max_value is None:
                max_value = max(int(min_value) + 1, 100)
            return dpg.add_slider_int(min_value=int(min_value), max_value=int(max_value), default_value=int(value), width=CONTROL_WIDTH, **common)
        min_value = float(param_def.get("min_value", 0.0))
        max_value = float(param_def.get("max_value", max(float(min_value) + 1.0, 1.0)))
        return dpg.add_slider_float(min_value=min_value, max_value=max_value, default_value=float(value), width=CONTROL_WIDTH, format="%.3f", **common)

    def _on_param_changed(self, sender, app_data, param_name):
        if self._suppress_callbacks:
            return
        meta = self.param_metadata.get(param_name)
        if not meta:
            self.param_values[param_name] = app_data
            return
        value = self._coerce_value(meta, app_data)
        update_control = False
        if meta.get("type") == "text":
            original = "" if app_data is None else str(app_data)
            update_control = original != value
        self._set_param_value(param_name, value, update_control=update_control)

    def _set_param_value(self, param_name, value, update_control=False):
        self.param_values[param_name] = value
        if update_control and param_name in self.param_controls:
            self._suppress_callbacks = True
            try:
                dpg.set_value(self.param_controls[param_name], value)
            finally:
                self._suppress_callbacks = False
        self._after_param_change(param_name, value)

    def _after_param_change(self, param_name, value):
        if param_name == "shader_path":
            suggestion = self._find_effect_for_shader(value)
            effect_path = suggestion.get("effect") if suggestion else None
            lua_path = suggestion.get("lua") if suggestion else None
            if effect_path and effect_path != self.param_values.get("metadata_path"):
                self._set_param_value("metadata_path", effect_path, update_control=True)
                return
            if lua_path and lua_path != self.param_values.get("lua_path"):
                self._set_param_value("lua_path", lua_path, update_control=True)
                return
            self._reload_effect_params()
        elif param_name == "metadata_path":
            self._reload_effect_params()
        elif param_name == "lua_path":
            self._reload_effect_params()
        elif param_name == "shader_preset":
            self._apply_shader_preset(value)

    def _shader_preset_names(self):
        scripts_dir = self.base_dir / "lua_glsl_demo" / "scripts"
        shaders_dir = self.base_dir / "lua_glsl_demo" / "shaders"
        names = []
        try:
            for effect_path in sorted(scripts_dir.glob("*.effect")):
                stem = effect_path.stem
                if (scripts_dir / f"{stem}.lua").exists() and (shaders_dir / f"{stem}.frag").exists():
                    names.append(stem)
        except Exception:
            pass
        return names

    def _apply_shader_preset(self, preset_name):
        stem = str(preset_name or "").strip()
        if not stem:
            return
        linked = self._find_effect_by_name(stem)
        effect_path = linked.get("effect") if linked else None
        lua_path = linked.get("lua") if linked else None
        shader_path = self._to_param_path(self.base_dir / "lua_glsl_demo" / "shaders" / f"{stem}.frag")
        if shader_path:
            self._set_param_value("shader_path", shader_path, update_control=True)
        if lua_path:
            self._set_param_value("lua_path", lua_path, update_control=True)
        if effect_path:
            self._set_param_value("metadata_path", effect_path, update_control=True)

    def _find_effect_by_name(self, stem):
        result = {"effect": None, "lua": None}
        if not stem:
            return result
        stem = str(stem).strip()
        if not stem:
            return result

        preferred_dirs = [
            self.base_dir / "lua_glsl_demo" / "scripts",
            self.base_dir / "lua_glsl_demo" / "shaders",
        ]
        effect_candidates = []
        lua_candidates = []
        for directory in preferred_dirs:
            effect_candidates.append(directory / f"{stem}.effect")
            lua_candidates.append(directory / f"{stem}.lua")

        try:
            for candidate in self.base_dir.rglob(f"{stem}.effect"):
                effect_candidates.append(candidate)
        except Exception:
            pass
        try:
            for candidate in self.base_dir.rglob(f"{stem}.lua"):
                lua_candidates.append(candidate)
        except Exception:
            pass

        result["effect"] = self._first_existing(effect_candidates)
        result["lua"] = self._first_existing(lua_candidates)
        return result

    def _first_existing(self, candidates):
        for candidate in candidates:
            if not candidate:
                continue
            try:
                resolved = Path(candidate).resolve()
            except OSError:
                continue
            if resolved.exists():
                return self._to_param_path(resolved)
        return None

    def _find_effect_for_shader(self, shader_value):
        result = {"effect": None, "lua": None}
        shader_path = self._resolve_path(shader_value)
        if shader_path is None:
            return result

        stem = shader_path.stem
        effect_candidates = [shader_path.with_suffix(".effect")]
        lua_candidates = [shader_path.with_suffix(".lua")]

        parent = shader_path.parent
        if parent:
            effect_candidates.append(parent / f"{stem}.effect")
            lua_candidates.append(parent / f"{stem}.lua")
            try:
                parent_parent = parent.parent
            except AttributeError:
                parent_parent = None
            if parent_parent and parent_parent != parent:
                scripts_dir = parent_parent / "scripts"
                effect_candidates.append(scripts_dir / f"{stem}.effect")
                lua_candidates.append(scripts_dir / f"{stem}.lua")

        default_scripts = self.base_dir / "lua_glsl_demo" / "scripts"
        effect_candidates.append(default_scripts / f"{stem}.effect")
        lua_candidates.append(default_scripts / f"{stem}.lua")

        result["effect"] = self._first_existing(effect_candidates)
        result["lua"] = self._first_existing(lua_candidates)
        return result

    def _normalize_path_input(self, value: str) -> str:
        value = (value or "").strip()
        if not value:
            return ""
        value = value.strip().strip('"').strip("'")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1].strip()
        value = value.strip().strip('"').strip("'")
        return value.replace('\\', '/')



    def _resolve_path(self, value):
        if not value:
            return None
        try:
            candidate = Path(value).expanduser()
        except Exception:
            return None
        if not candidate.is_absolute():
            candidate = (self.base_dir / candidate).resolve()
        return candidate

    def _to_param_path(self, path_obj: Path) -> str:
        try:
            relative = path_obj.relative_to(self.base_dir)
            return str(relative).replace(os.sep, "/")
        except ValueError:
            return str(path_obj)


class AbyssalSignalFeedbackShaderNode(LuaShaderNode):
    """Lua Shader shortcut preloaded with the Abyssal Signal Feedback pack."""

    def __init__(self, parent_node_editor):
        super().__init__(parent_node_editor)
        self.label = "Abyssal Signal Feedback"
        self.lua_input_defs = [
            {"name": "image_in", "attr_label": "Input", "text": "Image In"},
            {"name": "bubble_mask", "attr_label": "Bubble Mask", "text": "Bubble Mask"},
            {"name": "bubble_normal", "attr_label": "Bubble Normal", "text": "Bubble Normal"},
            {"name": "bubble_depth", "attr_label": "Bubble Depth", "text": "Bubble Depth"},
            {"name": "bubble_refraction", "attr_label": "Bubble Refract", "text": "Bubble Refraction"},
            {"name": "bubble_highlight", "attr_label": "Bubble Highlight", "text": "Bubble Highlight"},
        ]
        self.base_param_defs[0]["default_value"] = "abyssal_signal_feedback"
        self.base_param_defs[1]["default_value"] = "lua_glsl_demo/shaders/abyssal_signal_feedback.frag"
        self.base_param_defs[2]["default_value"] = "lua_glsl_demo/scripts/abyssal_signal_feedback.lua"
        self.base_param_defs[3]["default_value"] = "lua_glsl_demo/scripts/abyssal_signal_feedback.effect"


class KaleidoFeedbackShaderNode(LuaShaderNode):
    """Lua Shader shortcut preloaded with the Kaleido Feedback pack."""

    def __init__(self, parent_node_editor):
        super().__init__(parent_node_editor)
        self.label = "Kaleido Feedback"
        self.lua_input_defs = [
            {"name": "image_in", "attr_label": "Input", "text": "Image In"},
            {"name": "feedback_in", "attr_label": "Feedback Input", "text": "Feedback In"},
        ]
        self.base_param_defs[0]["default_value"] = "kaleido_feedback"
        self.base_param_defs[1]["default_value"] = "lua_glsl_demo/shaders/kaleido_feedback.frag"
        self.base_param_defs[2]["default_value"] = "lua_glsl_demo/scripts/kaleido_feedback.lua"
        self.base_param_defs[3]["default_value"] = "lua_glsl_demo/scripts/kaleido_feedback.effect"


class GlitchAmountNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Glitch Amount", parent_node_editor, params=[
            {"name": "glitch_amount_x", "type": "float", "label": "Glitch X", "min_value": 0.0, "max_value": 50.0, "default_value": 0.0},
            {"name": "glitch_amount_y", "type": "float", "label": "Glitch Y", "min_value": 0.0, "max_value": 50.0, "default_value": 0.0},
            {"name": "glitch_freq_x", "type": "float", "label": "Freq X", "min_value": 1.0, "max_value": 100.0, "default_value": 1.0},
            {"name": "glitch_freq_y", "type": "float", "label": "Freq Y", "min_value": 1.0, "max_value": 100.0, "default_value": 1.0},
            {"name": "uv_size", "type": "float", "label": "UV Size", "min_value": 0.1, "max_value": 5.0, "default_value": 0.1},
            {"name": "time_scale", "type": "float", "label": "Time Scale", "min_value": 0.0, "max_value": 5.0, "default_value": 1.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class PixelSortingNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Pixel Sorting", parent_node_editor, params=self.with_evolution_params([
            {"name": "pixel_sorting_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "pixel_sorting_direction", "type": "combo", "label": "Direction", "items": ["Horizontal", "Vertical"], "default_value": "Horizontal"},
            {"name": "sort_chance", "type": "int", "label": "Sort %", "min_value": 0, "max_value": 100, "default_value": 0},
        ], "pixel_sort", amount_label="Band Drift", speed_label="Band Speed", speed_default=1.4))
    def add_attributes(self):
        super().add_attributes()

class DatabendingNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Databending", parent_node_editor, params=[
            {"name": "databend_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "databend_mode", "type": "combo", "label": "Mode", "items": [e.value for e in CorruptionMode], "default_value": CorruptionMode.RANDOM.value},
            {"name": "databend_layers", "type": "int", "label": "Layers", "min_value": 1, "max_value": 10, "default_value": 1},
            {"name": "databend_format", "type": "combo", "label": "Format", "items": ['jpeg', 'png', 'bmp'], "default_value": 'jpeg'},
            {"name": "databend_quality", "type": "int", "label": "JPEG Q", "min_value": 1, "max_value": 100, "default_value": 75},
            {"name": "preserve_keyframes", "type": "checkbox", "label": "Keep KFs", "default_value": True},
            {"name": "frame_skip", "type": "int", "label": "Skip", "min_value": 0, "max_value": 10, "default_value": 0},
            {"name": "start_time", "type": "float", "label": "Start", "min_value": 0.0, "max_value": 60.0, "default_value": 0.0},
            {"name": "duration", "type": "float", "label": "Duration", "min_value": 0.0, "max_value": 60.0, "default_value": 10.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class RollingShutterWobbleNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Rolling Shutter Wobble", parent_node_editor, params=[
            {"name": "wobble_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "wobble_freq", "type": "float", "label": "Freq", "min_value": 0.1, "max_value": 10.0, "default_value": 3.0},
            {"name": "wobble_speed", "type": "float", "label": "Speed", "min_value": 0.0, "max_value": 5.0, "default_value": 0.5},
        ])
    def add_attributes(self):
        super().add_attributes()

class ScanlineShuffleNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Scanline Shuffle", parent_node_editor, params=self.with_evolution_params([
            {"name": "scan_shuffle", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "scan_span", "type": "int", "label": "Max Span", "min_value": 1, "max_value": 128, "default_value": 32},
            {"name": "scan_prob", "type": "float", "label": "Prob", "min_value": 0.0, "max_value": 1.0, "default_value": 0.1},
        ], "scan_shuffle", amount_label="Phase Drift", speed_label="Phase Speed", speed_default=1.3))
    def add_attributes(self):
        super().add_attributes()

class ChromaLeakNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Chroma Leak", parent_node_editor, params=[
            {"name": "chroma_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "chroma_mode", "type": "combo", "label": "Mode", "items": ["420", "422", "none"], "default_value": "420"},
            {"name": "chroma_drift", "type": "int", "label": "Drift px", "min_value": 0, "max_value": 10, "default_value": 2},
        ])
    def add_attributes(self):
        super().add_attributes()

class FFTPhaseScrambleNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("FFT Phase Scramble", parent_node_editor, params=[
            {"name": "fft_phase", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "fft_bands", "type": "int", "label": "Bands", "min_value": 1, "max_value": 10, "default_value": 3},
        ])
    def add_attributes(self):
        super().add_attributes()

class RowTemporalStutterNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Row Temporal Stutter", parent_node_editor, params=[
            {"name": "row_stutter", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "row_stutter_max_delay", "type": "int", "label": "Max Delay", "min_value": 1, "max_value": 10, "default_value": 4},
        ])
    def add_attributes(self):
        super().add_attributes()

class PaletteCrunchNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Palette Crunch", parent_node_editor, params=[
            {"name": "palette_levels", "type": "int", "label": "Levels", "min_value": 2, "max_value": 256, "default_value": 6},
            {"name": "palette_dither", "type": "checkbox", "label": "Dither", "default_value": True},
        ])
    def add_attributes(self):
        super().add_attributes()

class ChannelFlowOffsetNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Channel Flow Offset", parent_node_editor, params=self.with_evolution_params([
            {"name": "channel_flow", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "channel_flow_max", "type": "int", "label": "Max px", "min_value": 0, "max_value": 20, "default_value": 8},
        ], "channel_flow", amount_label="Orbit", speed_label="Orbit Speed", speed_default=1.1))
    def add_attributes(self):
        super().add_attributes()

class MeshTearNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Mesh Tear", parent_node_editor, params=self.with_evolution_params([
            {"name": "mesh_tear", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "mesh_grid", "type": "int", "label": "Grid", "min_value": 2, "max_value": 20, "default_value": 6},
        ], "mesh_tear", amount_label="Morph", speed_label="Morph Speed", speed_default=0.9))
    def add_attributes(self):
        super().add_attributes()

class DataMoshingNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        # NOTE: Non-zero defaults for visible effect out of the box
        super().__init__("Data Moshing", parent_node_editor, params=self.with_evolution_params([
            {"name": "block_size", "type": "int", "label": "Block", "min_value": 1, "max_value": 64, "default_value": 16},
            {"name": "block_displacement", "type": "float", "label": "Displace", "min_value": 0.0, "max_value": 1.0, "default_value": 0.3},
            {"name": "displacement_direction", "type": "combo", "label": "Dir", "items": ["random", "horizontal", "vertical"], "default_value": "random"},
            {"name": "block_duplication", "type": "float", "label": "Duplicate", "min_value": 0.0, "max_value": 1.0, "default_value": 0.2},
            {"name": "duplication_pattern", "type": "combo", "label": "Pattern", "items": ["random", "horizontal", "vertical", "grid"], "default_value": "horizontal"},
            {"name": "block_corruption", "type": "float", "label": "Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.1},
            {"name": "corruption_type", "type": "combo", "label": "Type", "items": ["noise", "invert", "blur", "median"], "default_value": "noise"},
            {"name": "block_sorting", "type": "float", "label": "Sort", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "sort_method", "type": "combo", "label": "Method", "items": ["brightness", "red", "green", "blue"], "default_value": "brightness"},
            {"name": "motion_vectors", "type": "float", "label": "Vectors", "min_value": 0.0, "max_value": 1.0, "default_value": 0.15},
            {"name": "vector_length", "type": "int", "label": "Vec Len", "min_value": 5, "max_value": 100, "default_value": 20},
            {"name": "bitplane_corruption", "type": "float", "label": "Bitplane", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "bitplane", "type": "int", "label": "Plane (0-7)", "min_value": 0, "max_value": 7, "default_value": 1},
            {"name": "dct_glitch_intensity", "type": "float", "label": "DCT Glitch", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "slit_scan_intensity", "type": "float", "label": "Slit Scan", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "slit_scan_direction", "type": "combo", "label": "Slit Dir", "items": ["right_to_left", "left_to_right", "top_to_bottom", "bottom_to_top"], "default_value": "right_to_left"},
            {"name": "slit_scan_size", "type": "int", "label": "Slit Size", "min_value": 1, "max_value": 128, "default_value": 32},
            {"name": "slit_scan_pos", "type": "float", "label": "Slit Pos", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
        ], "data_mosh", amount_label="Mutation", speed_label="Mutation Speed", speed_default=1.0))
    def add_attributes(self):
        super().add_attributes()

class PFrameGlitchesNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("P-Frame Glitches", parent_node_editor, params=[
            {"name": "motion_vector_corruption", "type": "float", "label": "MV Corr", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "motion_vector_direction", "type": "combo", "label": "MV Dir", "items": ['random', 'horizontal', 'vertical', 'diagonal', 'radial', 'spiral', 'wave'], "default_value": 'random'},
            {"name": "prediction_error_amplify", "type": "float", "label": "Pred Err Amp", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "prediction_error_direction", "type": "combo", "label": "Err Dir", "items": ['uniform', 'horizontal', 'vertical', 'diagonal', 'radial'], "default_value": 'uniform'},
            {"name": "temporal_bleeding", "type": "float", "label": "Bleeding", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "temporal_bleeding_direction", "type": "combo", "label": "Bleed Dir", "items": ['random', 'forward', 'backward', 'ping_pong', 'center_out'], "default_value": 'random'},
            {"name": "macroblock_scramble", "type": "float", "label": "MB Scramble", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "scrambling_pattern", "type": "combo", "label": "Scramble", "items": ['random', 'horizontal_shift', 'vertical_shift', 'diagonal', 'spiral', 'checkerboard'], "default_value": 'random'},
            {"name": "residual_injection", "type": "float", "label": "Residual Inj", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "residual_direction", "type": "combo", "label": "Residual Dir", "items": ['uniform', 'horizontal_gradient', 'vertical_gradient', 'radial', 'corner_focus'], "default_value": 'uniform'},
            {"name": "reference_corruption", "type": "float", "label": "Ref Corr", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "reference_corruption_pattern", "type": "combo", "label": "Ref Corr Patt", "items": ['scattered', 'horizontal_bands', 'vertical_bands', 'diagonal_sweep', 'center_corruption'], "default_value": 'scattered'},
            {"name": "macroblock_size", "type": "combo", "label": "MB Size", "items": ["8", "16", "32"], "default_value": "16"},
            {"name": "error_threshold", "type": "int", "label": "Err Thresh", "min_value": 10, "max_value": 100, "default_value": 30},
            {"name": "bleed_distance", "type": "int", "label": "Bleed Dist", "min_value": 1, "max_value": 10, "default_value": 3},
            {"name": "residual_noise_type", "type": "combo", "label": "Residual Noise", "items": ['gaussian', 'block_artifacts', 'salt_pepper'], "default_value": 'gaussian'},
            {"name": "reference_corruption_type", "type": "combo", "label": "Ref Corr Type", "items": ['random_blocks', 'temporal_artifacts'], "default_value": 'random_blocks'},
            {"name": "buffer_size", "type": "int", "label": "P-Frame Buf", "min_value": 1, "max_value": 50, "default_value": 10},
        ])
    def add_attributes(self):
        super().add_attributes()

class RotationKaleidoscopeNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Rotation / Kaleidoscope", parent_node_editor, params=[
            {"name": "rotation_angle", "type": "float", "label": "Rotation Angle", "min_value": 0.0, "max_value": 360.0, "default_value": 0.0},
            {"name": "rotation_speed", "type": "float", "label": "Rotation Speed", "min_value": 0.0, "max_value": 10.0, "default_value": 0.0},
            {"name": "kaleidoscope_segments", "type": "int", "label": "Kaleidoscope Segments", "min_value": 1, "max_value": 12, "default_value": 1},
            {"name": "kaleidoscope_speed", "type": "float", "label": "Kaleidoscope Speed", "min_value": 0.0, "max_value": 100.0, "default_value": 0.2},
        ])
    def add_attributes(self):
        super().add_attributes()

class PixelationChromaticNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Pixelation / Chromatic", parent_node_editor, params=[
            {"name": "pixelation_intensity", "type": "int", "label": "Pixelation Intensity", "min_value": 1, "max_value": 50, "default_value": 1},
            {"name": "chromatic_aberration", "type": "float", "label": "Chromatic Aberration", "min_value": 0.0, "max_value": 10.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class ScanLinesNoiseWaveNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("ScanLines / Noise / Wave", parent_node_editor, params=[
            {"name": "scan_lines_intensity", "type": "float", "label": "Scan Lines Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "noise_intensity", "type": "float", "label": "Noise Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "wave_distortion", "type": "float", "label": "Wave Distortion Amount", "min_value": 0.0, "max_value": 50.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class InvertColorNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Invert Colors", parent_node_editor, params=[
            {"name": "invert_chance", "type": "int", "label": "Invert Chance (%)", "min_value": 0, "max_value": 100, "default_value": 0},
        ])
    def add_attributes(self):
        super().add_attributes()

class SimpleDataMoshingIntensityNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Data Moshing Intensity", parent_node_editor, params=[
            {"name": "data_moshing_intensity", "type": "float", "label": "Data Moshing Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class MultimodalAnalysisNode(GlitchEffectNode):
    """Pass-through analysis node that publishes typed master-resolution masks."""

    def __init__(self, parent_node_editor):
        super().__init__("Multimodal Analysis", parent_node_editor, params=[
            {"name": "entropy_window", "type": "int", "label": "Entropy Window", "min_value": 3, "max_value": 31, "default_value": 9},
            {"name": "flow_scale_pixels", "type": "float", "label": "Flow Scale", "min_value": 1.0, "max_value": 64.0, "default_value": 12.0},
            {"name": "motion_low", "type": "float", "label": "Motion Low", "min_value": 0.0, "max_value": 0.95, "default_value": 0.035},
            {"name": "motion_high", "type": "float", "label": "Motion High", "min_value": 0.01, "max_value": 1.0, "default_value": 0.42},
            {"name": "edge_low", "type": "float", "label": "Edge Low", "min_value": 0.0, "max_value": 0.95, "default_value": 0.025},
            {"name": "edge_high", "type": "float", "label": "Edge High", "min_value": 0.01, "max_value": 1.0, "default_value": 0.32},
            {"name": "scene_change_threshold", "type": "float", "label": "Scene Reset", "min_value": 0.0, "max_value": 1.0, "default_value": 0.32},
            {"name": "modulation_gamma", "type": "float", "label": "Mask Gamma", "min_value": 0.25, "max_value": 3.0, "default_value": 1.12},
        ])

    def add_attributes(self):
        super().add_attributes()


class OpticalFlowRemapNode(GlitchEffectNode):
    """Provider-independent remap using the canonical full-resolution flow."""

    def __init__(self, parent_node_editor):
        super().__init__("Optical Flow Remap", parent_node_editor, params=[
            {"name": "flow_remap_strength", "type": "float", "label": "Strength", "min_value": 0.0, "max_value": 2.0, "default_value": 0.65},
            {"name": "flow_displacement_cap", "type": "float", "label": "Cap px", "min_value": 0.0, "max_value": 128.0, "default_value": 24.0},
            {"name": "flow_direction", "type": "combo", "label": "Direction", "items": ["backward", "forward"], "default_value": "backward"},
            {"name": "flow_border_mode", "type": "combo", "label": "Border", "items": ["reflect", "replicate", "wrap", "constant"], "default_value": "reflect"},
        ])

    def add_attributes(self):
        super().add_attributes()


class ControlledRecursiveFeedbackNode(GlitchEffectNode):
    """Motion-sensitive recursive feedback with hard clean-source bounds."""

    def __init__(self, parent_node_editor):
        super().__init__("Controlled Recursive Feedback", parent_node_editor, params=[
            {"name": "feedback_opacity", "type": "float", "label": "Opacity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.42},
            {"name": "opacity_cap", "type": "float", "label": "Opacity Cap", "min_value": 0.0, "max_value": 1.0, "default_value": 0.68},
            {"name": "feedback_decay", "type": "float", "label": "Decay", "min_value": 0.0, "max_value": 1.0, "default_value": 0.92},
            {"name": "clean_source_floor", "type": "float", "label": "Clean Floor", "min_value": 0.0, "max_value": 1.0, "default_value": 0.32},
            {"name": "motion_suppression", "type": "float", "label": "Motion Protect", "min_value": 0.0, "max_value": 1.0, "default_value": 0.72},
            {"name": "edge_protection", "type": "float", "label": "Edge Protect", "min_value": 0.0, "max_value": 1.0, "default_value": 0.28},
            {"name": "scene_change_threshold", "type": "float", "label": "Scene Reset", "min_value": 0.0, "max_value": 1.0, "default_value": 0.32},
        ])

    def add_attributes(self):
        super().add_attributes()


class FeedbackTrailNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Feedback / Trail", parent_node_editor, params=[
            {"name": "feedback_intensity", "type": "float", "label": "Feedback Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "trailing_intensity", "type": "float", "label": "Trailing Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class FrameDropRGBSplitNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("FrameDrop / RGB Split", parent_node_editor, params=[
            {"name": "frame_drop_chance", "type": "int", "label": "Frame Drop Chance (%)", "min_value": 0, "max_value": 100, "default_value": 0},
            {"name": "rgb_split_distance", "type": "int", "label": "RGB Split Distance", "min_value": 0, "max_value": 50, "default_value": 0},
        ])
    def add_attributes(self):
        super().add_attributes()

class TextOverlayNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Text Overlay", parent_node_editor, params=[
            {"name": "glitch_text", "type": "text", "label": "Glitch Text", "default_value": ""},
            {"name": "glitch_text_intensity", "type": "float", "label": "Text Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class LumaKeyNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Luma Key", parent_node_editor, params=[
            {"name": "key_mode", "type": "combo", "label": "Key Mode", "items": ["luma bright", "luma dark", "green screen"], "default_value": "luma bright"},
            {"name": "key_threshold", "type": "float", "label": "Threshold", "min_value": 0.0, "max_value": 1.0, "default_value": 0.62},
            {"name": "key_softness", "type": "float", "label": "Softness", "min_value": 0.0, "max_value": 1.0, "default_value": 0.12},
            {"name": "key_opacity", "type": "float", "label": "Opacity", "min_value": 0.0, "max_value": 1.0, "default_value": 1.0},
            {"name": "fill_mode", "type": "combo", "label": "Fill", "items": ["black", "white", "color", "invert", "posterize"], "default_value": "color"},
            {"name": "fill_red", "type": "int", "label": "Fill R", "min_value": 0, "max_value": 255, "default_value": 0},
            {"name": "fill_green", "type": "int", "label": "Fill G", "min_value": 0, "max_value": 255, "default_value": 255},
            {"name": "fill_blue", "type": "int", "label": "Fill B", "min_value": 0, "max_value": 255, "default_value": 180},
            {"name": "edge_glow", "type": "float", "label": "Edge Glow", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "invert_matte", "type": "checkbox", "label": "Invert Matte", "default_value": False},
        ])
    def add_attributes(self):
        super().add_attributes()

class LineGlitchNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Line Glitch", parent_node_editor, params=[
            {"name": "line_count", "type": "int", "label": "Line Count", "min_value": 0, "max_value": 100, "default_value": 0},
            {"name": "max_offset", "type": "int", "label": "Max Offset", "min_value": 0, "max_value": 100, "default_value": 0},
            {"name": "glitch_frequency", "type": "int", "label": "Glitch Frequency (%)", "min_value": 0, "max_value": 100, "default_value": 0},
            {"name": "glitch_intensity", "type": "int", "label": "Glitch Intensity", "min_value": 0, "max_value": 100, "default_value": 0},
        ])
    def add_attributes(self):
        super().add_attributes()

class UnfoldEuclidRandomNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Unfold / Euclid / Random", parent_node_editor, params=[
            {"name": "unfolding_speed", "type": "float", "label": "Unfolding Speed", "min_value": 0.0, "max_value": 5.0, "default_value": 0.0},
            {"name": "euclidean_parameter", "type": "float", "label": "Euclidean Parameter", "min_value": 0.0, "max_value": 10.0, "default_value": 0.0},
            {"name": "randomize_intensity", "type": "int", "label": "Randomize Intensity", "min_value": 0, "max_value": 100, "default_value": 0},
        ])
    def add_attributes(self):
        super().add_attributes()

class FrameBufferNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Frame Buffer", parent_node_editor, params=[
            {"name": "frame_buffer_size", "type": "int", "label": "Frame Buffer Size", "min_value": 1, "max_value": 10, "default_value": 1},
        ])
    def add_attributes(self):
        super().add_attributes()

# ---- New Conventional Effect Nodes ----

class BrightnessContrastNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Brightness / Contrast", parent_node_editor, params=[
            {"name": "brightness", "type": "int", "label": "Bright", "min_value": -100, "max_value": 100, "default_value": 0},
            {"name": "contrast", "type": "int", "label": "Contrast", "min_value": -100, "max_value": 100, "default_value": 0},
        ])
    def add_attributes(self):
        super().add_attributes()

class ColorAdjustNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Color Adjust", parent_node_editor, params=[
            {"name": "hue_shift", "type": "int", "label": "Hue", "min_value": -180, "max_value": 180, "default_value": 0},
            {"name": "saturation", "type": "float", "label": "Sat", "min_value": 0.0, "max_value": 3.0, "default_value": 1.0},
            {"name": "gamma", "type": "float", "label": "Gamma", "min_value": 0.1, "max_value": 3.0, "default_value": 1.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class VignetteNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Vignette", parent_node_editor, params=[
            {"name": "vignette_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "vignette_radius", "type": "float", "label": "Radius", "min_value": 0.1, "max_value": 1.5, "default_value": 0.75},
        ])
    def add_attributes(self):
        super().add_attributes()

class BlurSharpenNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Blur / Sharpen", parent_node_editor, params=[
            {"name": "blur_radius", "type": "float", "label": "Blur", "min_value": 0.0, "max_value": 20.0, "default_value": 0.0},
            {"name": "sharpen_amount", "type": "float", "label": "Sharpen", "min_value": 0.0, "max_value": 2.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()


class QuantizationSolarizationNode(GlitchEffectNode):
    """Strict master-resolution palette quantization and solarization."""

    def __init__(self, parent_node_editor):
        super().__init__("Quantization / Solarization", parent_node_editor, params=[
            {"name": "quantization_levels", "type": "int", "label": "Quantization Levels", "min_value": 2, "max_value": 256, "default_value": 192},
            {"name": "solarization_threshold", "type": "int", "label": "Solar Threshold", "min_value": 0, "max_value": 255, "default_value": 208},
            {"name": "solarization_blend", "type": "float", "label": "Solar Blend", "min_value": 0.0, "max_value": 1.0, "default_value": 0.10},
        ])

    def add_attributes(self):
        super().add_attributes()


class EdgeReconstructionNode(GlitchEffectNode):
    """Restrained edge-only luminance reinforcement before final restoration."""

    def __init__(self, parent_node_editor):
        super().__init__("Edge Reconstruction", parent_node_editor, params=[
            {"name": "edge_luma_reinforcement", "type": "float", "label": "Edge Luma", "min_value": 0.0, "max_value": 0.25, "default_value": 0.035},
            {"name": "edge_sensitivity", "type": "float", "label": "Edge Sensitivity", "min_value": 0.001, "max_value": 0.25, "default_value": 0.055},
        ])

    def add_attributes(self):
        super().add_attributes()


class StructuralRestorationNode(GlitchEffectNode):
    """Final master-path restoration; place after all destructive operations."""

    def __init__(self, parent_node_editor):
        super().__init__("Structural Restoration", parent_node_editor, params=[
            {"name": "unsharp_amount", "type": "float", "label": "Unsharp Amount", "min_value": 0.0, "max_value": 1.0, "default_value": 0.20},
            {"name": "unsharp_radius", "type": "float", "label": "Unsharp Radius", "min_value": 0.4, "max_value": 3.0, "default_value": 1.10},
            {"name": "local_contrast", "type": "float", "label": "Local Contrast", "min_value": 0.0, "max_value": 0.75, "default_value": 0.12},
            {"name": "local_contrast_radius", "type": "float", "label": "Contrast Radius", "min_value": 2.0, "max_value": 32.0, "default_value": 10.0},
            {"name": "edge_luma_reinforcement", "type": "float", "label": "Edge Luma", "min_value": 0.0, "max_value": 0.5, "default_value": 0.05},
            {"name": "edge_sensitivity", "type": "float", "label": "Edge Sensitivity", "min_value": 0.005, "max_value": 0.25, "default_value": 0.05},
            {"name": "grain_normalization", "type": "float", "label": "Grain Normalize", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "grain_target", "type": "float", "label": "Grain Target", "min_value": 0.0, "max_value": 0.08, "default_value": 0.012},
            {"name": "final_clamp_low", "type": "int", "label": "Final Clamp Low", "min_value": 0, "max_value": 64, "default_value": 0},
            {"name": "final_clamp_high", "type": "int", "label": "Final Clamp High", "min_value": 191, "max_value": 255, "default_value": 255},
        ])

    def add_attributes(self):
        super().add_attributes()

class BloomNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Bloom", parent_node_editor, params=[
            {"name": "bloom_strength", "type": "float", "label": "Strength", "min_value": 0.0, "max_value": 2.0, "default_value": 0.0},
            {"name": "bloom_threshold", "type": "int", "label": "Thresh", "min_value": 0, "max_value": 255, "default_value": 200},
        ])
    def add_attributes(self):
        super().add_attributes()

class EdgeEmbossNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Edges / Emboss", parent_node_editor, params=[
            {"name": "edge_mix", "type": "float", "label": "Edges", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "emboss_strength", "type": "float", "label": "Emboss", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

# ---- New Obscure Effect Nodes ----

class BarrelDistortionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Barrel/Pincushion", parent_node_editor, params=[
            {"name": "barrel_amount", "type": "float", "label": "Amount", "min_value": -0.5, "max_value": 0.5, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class SwirlNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Swirl/Twirl", parent_node_editor, params=[
            {"name": "swirl_amount", "type": "float", "label": "Amount", "min_value": -6.28, "max_value": 6.28, "default_value": 0.0},
            {"name": "swirl_radius_factor", "type": "float", "label": "Radius", "min_value": 0.1, "max_value": 1.5, "default_value": 0.75},
        ])
    def add_attributes(self):
        super().add_attributes()

class RippleNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Ripple", parent_node_editor, params=[
            {"name": "ripple_amp", "type": "float", "label": "Amp", "min_value": 0.0, "max_value": 20.0, "default_value": 0.0},
            {"name": "ripple_freq", "type": "float", "label": "Freq", "min_value": 0.1, "max_value": 20.0, "default_value": 3.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class HalftoneNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Halftone", parent_node_editor, params=[
            {"name": "halftone_cell", "type": "int", "label": "Cell", "min_value": 2, "max_value": 32, "default_value": 8},
            {"name": "halftone_contrast", "type": "float", "label": "Contrast", "min_value": 0.5, "max_value": 2.0, "default_value": 1.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class VHSJitterNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("VHS Jitter", parent_node_editor, params=[
            {"name": "vhs_jitter", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "vhs_max_shift", "type": "int", "label": "Max Shift", "min_value": 1, "max_value": 32, "default_value": 8},
        ])
    def add_attributes(self):
        super().add_attributes()

# ---- New Effect Nodes (from new_effects.py) ----

class KaleidoscopeNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Kaleidoscope", parent_node_editor, params=[
            {"name": "kaleido_segments", "type": "int", "label": "Segments", "min_value": 2, "max_value": 16, "default_value": 6},
            {"name": "kaleido_angle", "type": "float", "label": "Angle", "min_value": -3.14, "max_value": 3.14, "default_value": 0.0},
            {"name": "kaleido_radius", "type": "float", "label": "Radius", "min_value": 0.2, "max_value": 2.0, "default_value": 1.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class VectorscopeNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Vectorscope XY", parent_node_editor, params=[
            {"name": "vectorscope_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "freq_x", "type": "float", "label": "Frequency X", "min_value": 0.1, "max_value": 10.0, "default_value": 1.0},
            {"name": "freq_y", "type": "float", "label": "Frequency Y", "min_value": 0.1, "max_value": 10.0, "default_value": 2.0},
            {"name": "phase_offset", "type": "float", "label": "Phase", "min_value": 0.0, "max_value": 6.28, "default_value": 0.0},
            {"name": "circular_distortion", "type": "float", "label": "Circular Distort", "min_value": 0.0, "max_value": 1.0, "default_value": 0.3},
        ])
    def add_attributes(self):
        super().add_attributes()

class NoiseWarpNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Noise Warp", parent_node_editor, params=[
            {"name": "noise_warp_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "noise_warp_scale", "type": "float", "label": "Scale", "min_value": 4.0, "max_value": 256.0, "default_value": 64.0},
            {"name": "noise_warp_time", "type": "float", "label": "Time", "min_value": 0.0, "max_value": 999.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class WatercolorPosterizeNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Watercolor Posterize", parent_node_editor, params=[
            {"name": "watercolor_levels", "type": "int", "label": "Levels", "min_value": 2, "max_value": 32, "default_value": 6},
            {"name": "watercolor_sigma", "type": "int", "label": "Sigma", "min_value": 4, "max_value": 64, "default_value": 12},
            {"name": "watercolor_passes", "type": "int", "label": "Passes", "min_value": 1, "max_value": 5, "default_value": 2},
        ])
    def add_attributes(self):
        super().add_attributes()

class CRTEmulationNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("CRT Emulation", parent_node_editor, params=[
            {"name": "crt_scanline", "type": "float", "label": "Scanline", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "crt_mask", "type": "combo", "label": "Mask", "items": ["none", "rgb", "shadow"], "default_value": "rgb"},
            {"name": "crt_curvature", "type": "float", "label": "Curvature", "min_value": -0.5, "max_value": 0.5, "default_value": 0.0},
            {"name": "crt_bloom", "type": "float", "label": "Bloom", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class VHSDamageNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("VHS Damage", parent_node_editor, params=self.with_evolution_params([
            {"name": "vhs_damage", "type": "float", "label": "Damage", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "vhs_band_min", "type": "int", "label": "Band min", "min_value": 2, "max_value": 40, "default_value": 6},
            {"name": "vhs_band_max", "type": "int", "label": "Band max", "min_value": 4, "max_value": 80, "default_value": 24},
            {"name": "vhs_jitter", "type": "float", "label": "Jitter", "min_value": 0.0, "max_value": 1.0, "default_value": 0.2},
            {"name": "vhs_noise", "type": "float", "label": "Noise", "min_value": 0.0, "max_value": 1.0, "default_value": 0.2},
        ], "vhs_tracking", amount_label="Tracking Drift", speed_label="Tracking Speed", speed_default=0.8))
    def add_attributes(self):
        super().add_attributes()

# Additional Glitch Effect Nodes for your system

# ==== COMPRESSION & FORMAT ARTIFACTS ====

class JPEGCompressionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("JPEG Compression", parent_node_editor, params=[
            {"name": "jpeg_quality", "type": "int", "label": "Quality", "min_value": 1, "max_value": 100, "default_value": 75},
            {"name": "jpeg_passes", "type": "int", "label": "Passes", "min_value": 1, "max_value": 10, "default_value": 1},
            {"name": "dct_corruption", "type": "float", "label": "DCT Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "quantization_corruption", "type": "float", "label": "Quant Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class WebPCorruptionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("WebP Corruption", parent_node_editor, params=[
            {"name": "webp_quality", "type": "int", "label": "Quality", "min_value": 0, "max_value": 100, "default_value": 80},
            {"name": "lossless_corruption", "type": "float", "label": "Lossless Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class H264ArtifactsNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("H264 Artifacts", parent_node_editor, params=[
            {"name": "iframe_corruption", "type": "float", "label": "I-Frame Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "pframe_corruption", "type": "float", "label": "P-Frame Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "bframe_corruption", "type": "float", "label": "B-Frame Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "bitrate_starve", "type": "float", "label": "Bitrate Starve", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

# ==== ADVANCED TEMPORAL EFFECTS ====

class FrameInterpolationGlitchNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Frame Interpolation Glitch", parent_node_editor, params=[
            {"name": "interp_corruption", "type": "float", "label": "Interp Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "motion_estimation_error", "type": "float", "label": "Motion Est Err", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "blending_artifacts", "type": "float", "label": "Blend Artifacts", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class TemporalSlicingNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Temporal Slicing", parent_node_editor, params=[
            {"name": "slice_width", "type": "int", "label": "Slice Width", "min_value": 1, "max_value": 64, "default_value": 8},
            {"name": "time_offset_range", "type": "int", "label": "Time Offset", "min_value": 1, "max_value": 30, "default_value": 5},
            {"name": "slice_direction", "type": "combo", "label": "Direction", "items": ["vertical", "horizontal", "diagonal", "radial"], "default_value": "vertical"},
        ])
    def add_attributes(self):
        super().add_attributes()

class TimeDisplacementNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Time Displacement", parent_node_editor, params=[
            {"name": "displacement_map_scale", "type": "float", "label": "Map Scale", "min_value": 0.1, "max_value": 5.0, "default_value": 1.0},
            {"name": "temporal_range", "type": "int", "label": "Temporal Range", "min_value": 1, "max_value": 60, "default_value": 10},
            {"name": "displacement_strength", "type": "float", "label": "Strength", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

# ==== ANALOG/HARDWARE SIMULATION ====

class CRTPhosphorNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("CRT Phosphor Decay", parent_node_editor, params=[
            {"name": "phosphor_decay", "type": "float", "label": "Decay Rate", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "phosphor_color", "type": "combo", "label": "Phosphor", "items": ["p22", "p45", "p53", "amber", "green"], "default_value": "p22"},
            {"name": "burn_in_intensity", "type": "float", "label": "Burn-in", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class AnalogTVNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Analog TV", parent_node_editor, params=[
            {"name": "signal_noise", "type": "float", "label": "Signal Noise", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "vsync_corruption", "type": "float", "label": "V-Sync Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "hsync_corruption", "type": "float", "label": "H-Sync Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "ghosting", "type": "float", "label": "Ghosting", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "rf_interference", "type": "float", "label": "RF Interference", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class TapeDropoutNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Tape Dropout", parent_node_editor, params=[
            {"name": "dropout_frequency", "type": "float", "label": "Frequency", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "dropout_size_min", "type": "int", "label": "Size Min", "min_value": 1, "max_value": 100, "default_value": 5},
            {"name": "dropout_size_max", "type": "int", "label": "Size Max", "min_value": 1, "max_value": 200, "default_value": 50},
            {"name": "tape_head_clog", "type": "float", "label": "Head Clog", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

# ==== DIGITAL ARTIFACT SIMULATION ====

class NetworkPacketLossNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Network Packet Loss", parent_node_editor, params=[
            {"name": "packet_loss_rate", "type": "float", "label": "Loss Rate %", "min_value": 0.0, "max_value": 50.0, "default_value": 0.0},
            {"name": "burst_loss", "type": "checkbox", "label": "Burst Loss", "default_value": False},
            {"name": "frame_reconstruction", "type": "combo", "label": "Reconstruction", "items": ["none", "repeat_last", "interpolate", "noise_fill"], "default_value": "repeat_last"},
        ])
    def add_attributes(self):
        super().add_attributes()

class BufferUnderrunNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Buffer Underrun", parent_node_editor, params=[
            {"name": "underrun_frequency", "type": "float", "label": "Frequency", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "freeze_duration", "type": "int", "label": "Freeze Duration", "min_value": 1, "max_value": 30, "default_value": 3},
            {"name": "skip_frames", "type": "int", "label": "Skip Frames", "min_value": 0, "max_value": 10, "default_value": 2},
        ])
    def add_attributes(self):
        super().add_attributes()

class DigitalCLIPNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Digital Clipping", parent_node_editor, params=[
            {"name": "clip_threshold", "type": "int", "label": "Threshold", "min_value": 0, "max_value": 255, "default_value": 235},
            {"name": "soft_clip", "type": "checkbox", "label": "Soft Clip", "default_value": False},
            {"name": "clip_color", "type": "combo", "label": "Clip Color", "items": ["white", "black", "red", "blue", "random"], "default_value": "white"},
        ])
    def add_attributes(self):
        super().add_attributes()

# ==== MATHEMATICAL/ALGORITHMIC GLITCHES ====

class WaveletCorruptionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Wavelet Corruption", parent_node_editor, params=[
            {"name": "wavelet_type", "type": "combo", "label": "Wavelet", "items": ["db4", "db8", "haar", "bior2.2"], "default_value": "db4"},
            {"name": "corruption_level", "type": "int", "label": "Level", "min_value": 1, "max_value": 6, "default_value": 3},
            {"name": "coefficient_corruption", "type": "float", "label": "Coeff Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class FractalNoiseNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Fractal Noise", parent_node_editor, params=[
            {"name": "fractal_octaves", "type": "int", "label": "Octaves", "min_value": 1, "max_value": 8, "default_value": 4},
            {"name": "fractal_persistence", "type": "float", "label": "Persistence", "min_value": 0.1, "max_value": 1.0, "default_value": 0.5},
            {"name": "fractal_scale", "type": "float", "label": "Scale", "min_value": 1.0, "max_value": 100.0, "default_value": 20.0},
            {"name": "displacement_strength", "type": "float", "label": "Displacement", "min_value": 0.0, "max_value": 50.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class CellularAutomataNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Cellular Automata", parent_node_editor, params=[
            {"name": "ca_rule", "type": "int", "label": "Rule", "min_value": 0, "max_value": 255, "default_value": 30},
            {"name": "ca_generations", "type": "int", "label": "Generations", "min_value": 1, "max_value": 100, "default_value": 10},
            {"name": "ca_animation_speed", "type": "float", "label": "Animation Speed", "min_value": 0.0, "max_value": 10.0, "default_value": 1.0},
            {"name": "ca_time_offset", "type": "float", "label": "Time Offset", "min_value": 0.0, "max_value": 100.0, "default_value": 0.0},
            {"name": "seed_from_image", "type": "checkbox", "label": "Seed from Image", "default_value": True},
            {"name": "blend_mode", "type": "combo", "label": "Blend", "items": ["overlay", "multiply", "screen", "difference"], "default_value": "overlay"},
        ])
    def add_attributes(self):
        super().add_attributes()

class FaultlinePossessionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        params = [
            {"name": "fp_territory_count", "type": "int", "label": "Territories", "min_value": 3, "max_value": 24, "default_value": 8,
             "compact_label": "Territories",
             "tooltip": "Number of territorial seeds competing for regional ownership."},
            {"name": "fp_mutation_rate", "type": "float", "label": "Mutation Rate", "min_value": 0.0, "max_value": 0.2, "default_value": 0.025,
             "compact_label": "Mutation",
             "tooltip": "Base mutation rate for territory behavior and style rewrites."},
            {"name": "fp_scar_decay", "type": "float", "label": "Scar Decay", "min_value": 0.9, "max_value": 0.9999, "default_value": 0.989,
             "compact_label": "Scar Decay",
             "tooltip": "How slowly territorial scars fade after rupture events."},
            {"name": "fp_takeover_rate", "type": "float", "label": "Takeover Rate", "min_value": 0.0, "max_value": 0.2, "default_value": 0.055,
             "compact_label": "Takeover",
             "tooltip": "Base rate for hostile territory possession events."},
            {"name": "fp_rupture_gain", "type": "float", "label": "Rupture Gain", "min_value": 0.0, "max_value": 3.0, "default_value": 1.6,
             "compact_label": "Rupture",
             "tooltip": "Overall gain on boundary rupture masks and fault activity."},
            {"name": "fp_reset_probability", "type": "float", "label": "Catastrophic Reset", "min_value": 0.0, "max_value": 0.02, "default_value": 0.002,
             "compact_label": "Cat Reset",
             "tooltip": "Chance of large-scale dead-zone reset events that scramble the whole field."},
            {"name": "fp_max_displacement", "type": "float", "label": "Max Displace", "min_value": 1.0, "max_value": 48.0, "default_value": 22.0,
             "compact_label": "Displace",
             "tooltip": "Upper bound for territorial displacement amplitude."},

            {"name": "history_offset", "type": "float", "label": "History Offset", "min_value": 0.0, "max_value": 32.0, "default_value": 4.0,
             "compact_label": "Hist Off", "control_width": 110, "mod_mode": "offset", "mod_span": 8.0,
             "tooltip": "Which history buffer tap is sampled for stale frame recall and temporal misremembering."},
            {"name": "memory_decay", "type": "float", "label": "Memory Decay", "min_value": 0.5, "max_value": 0.999, "default_value": 0.93,
             "compact_label": "Mem Decay", "control_width": 110, "mod_mode": "offset", "mod_span": 0.08,
             "tooltip": "Persistence of feedback and memory state across frames."},

            {"name": "event_hazard", "type": "float", "label": "Event Hazard", "min_value": 0.0, "max_value": 1.0, "default_value": 0.05,
             "compact_label": "Hazard", "control_width": 110, "mod_mode": "offset", "mod_span": 0.25,
             "tooltip": "Hazard rate for discontinuous ruptures, seizures, possession bursts, and fault events."},
            {"name": "phase_reset_chance", "type": "float", "label": "Phase Reset", "min_value": 0.0, "max_value": 0.5, "default_value": 0.01,
             "compact_label": "Phase Reset", "control_width": 110, "mod_mode": "offset", "mod_span": 0.12,
             "tooltip": "Chance of abrupt local phase wipes, resets, and regime flips."},
            {"name": "rupture_threshold", "type": "float", "label": "Rupture Threshold", "min_value": 0.0, "max_value": 2.0, "default_value": 0.5,
             "compact_label": "Threshold", "control_width": 110, "mod_mode": "offset", "mod_span": 0.6,
             "tooltip": "Stress threshold required before accumulated pressure releases as a rupture."},

            {"name": "clock_quantization", "type": "float", "label": "Clock Quantize", "min_value": 0.0, "max_value": 0.25, "default_value": 0.0,
             "compact_label": "Clock Q", "control_width": 110, "mod_mode": "offset", "mod_span": 0.08,
             "tooltip": "Temporal step size for gated, quantized local clocks instead of smooth flow."},
            {"name": "sync_desync", "type": "float", "label": "Sync / Desync", "min_value": -1.0, "max_value": 1.0, "default_value": 0.0,
             "compact_label": "Sync", "control_width": 110, "mod_mode": "offset", "mod_span": 0.6,
             "tooltip": "Negative values push territories into drift, positive values pull them back toward the master clock."},
            {"name": "local_time_warp", "type": "float", "label": "Local Time Warp", "min_value": -2.0, "max_value": 2.0, "default_value": 0.0,
             "compact_label": "Time Warp", "control_width": 110, "mod_mode": "offset", "mod_span": 1.0,
             "tooltip": "Regional speed-up, slowdown, lag, and warped local time scaling."},
            {"name": "hysteresis_speed", "type": "float", "label": "Hysteresis", "min_value": 0.0, "max_value": 1.0, "default_value": 0.08,
             "compact_label": "Hysteresis", "control_width": 110, "mod_mode": "offset", "mod_span": 0.25,
             "tooltip": "Attack and release speed for temporal targets, relocking, and stress response."},
            {"name": "time_reversal", "type": "float", "label": "Time Reversal", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0,
             "compact_label": "Reverse", "control_width": 110, "mod_mode": "offset", "mod_span": 1.0,
             "tooltip": "Reverse the flow of local time. 0 = forward, 1 = backwards."},

            {"name": "modulator_coupling", "type": "float", "label": "Mod Coupling", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0,
             "compact_label": "Mod Couple", "control_width": 110, "mod_mode": "offset", "mod_span": 0.4,
             "tooltip": "How strongly modulation processes feed back into other modulation behaviors."},
        ]
        super().__init__("Faultline Possession", parent_node_editor, params=params)

    def add_attributes(self):
        with dpg.node_attribute(label="Input", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Image In")
            self.input_attrs["image_in"] = attr

        with dpg.node_attribute(label="Parameters", attribute_type=dpg.mvNode_Attr_Static):
            categories = [
                ("Core", [
                    "fp_territory_count", "fp_mutation_rate", "fp_max_displacement",
                    "fp_scar_decay", "fp_takeover_rate", "fp_rupture_gain", "fp_reset_probability",
                ]),
                ("Memory", [
                    "history_offset", "memory_decay",
                ]),
                ("Events", [
                    "event_hazard", "phase_reset_chance", "rupture_threshold",
                ]),
                ("Clock", [
                    "clock_quantization", "sync_desync", "local_time_warp", "hysteresis_speed", "time_reversal",
                ]),
                ("Meta Mod", [
                    "modulator_coupling",
                ]),
            ]
            param_map = {param["name"]: param for param in self.params}
            for category_label, names in categories:
                dpg.add_text(category_label, color=(190, 210, 235))
                for name in names:
                    param_info = param_map.get(name)
                    if param_info:
                        self._add_param_control(param_info)
                dpg.add_spacer(height=4)

        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Image Out")
            self.output_attrs["image_out"] = attr

# ==== ADVANCED COLOR/FREQUENCY EFFECTS ====

class SpectralShiftNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Spectral Shift", parent_node_editor, params=[
            {"name": "frequency_shift", "type": "float", "label": "Freq Shift", "min_value": -1.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "phase_randomize", "type": "float", "label": "Phase Random", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "magnitude_corrupt", "type": "float", "label": "Mag Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class ColorSpaceCorruptionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("ColorSpace Corruption", parent_node_editor, params=[
            {"name": "source_space", "type": "combo", "label": "From", "items": ["RGB", "HSV", "LAB", "YUV", "XYZ"], "default_value": "RGB"},
            {"name": "target_space", "type": "combo", "label": "To", "items": ["RGB", "HSV", "LAB", "YUV", "XYZ"], "default_value": "HSV"},
            {"name": "channel_swap", "type": "checkbox", "label": "Swap Channels", "default_value": False},
            {"name": "quantization_levels", "type": "int", "label": "Quantize", "min_value": 2, "max_value": 256, "default_value": 256},
        ])
    def add_attributes(self):
        super().add_attributes()

class LUTCorruptionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("LUT Corruption", parent_node_editor, params=[
            {"name": "lut_size", "type": "combo", "label": "LUT Size", "items": ["16", "32", "64"], "default_value": "32"},
            {"name": "corruption_amount", "type": "float", "label": "Corruption", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "interpolation_error", "type": "float", "label": "Interp Error", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

# ==== GEOMETRY/MESH DISTORTIONS ====

class VertexShaderGlitchNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Vertex Shader Glitch", parent_node_editor, params=[
            {"name": "vertex_displacement", "type": "float", "label": "Displacement", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "tessellation_error", "type": "float", "label": "Tessellation Err", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "triangle_corruption", "type": "float", "label": "Triangle Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

class PerspectiveGlitchNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Perspective Glitch", parent_node_editor, params=[
            {"name": "perspective_error", "type": "float", "label": "Perspective Err", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "depth_corruption", "type": "float", "label": "Depth Corrupt", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "projection_matrix_error", "type": "float", "label": "Proj Matrix Err", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()

# ==== IMPROVEMENTS TO EXISTING EFFECTS ====

class EnhancedDataMoshingNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Enhanced Data Moshing", parent_node_editor, params=[
            # All original parameters from DataMoshingNode
            {"name": "block_size", "type": "int", "label": "Block", "min_value": 0, "max_value": 64, "default_value": 16},
            {"name": "block_displacement", "type": "float", "label": "Displace", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            # New enhanced parameters
            {"name": "motion_blur_simulation", "type": "float", "label": "Motion Blur", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "compression_artifacts", "type": "float", "label": "Compression Art", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "keyframe_preservation", "type": "float", "label": "Keyframe Preserve", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "adaptive_block_size", "type": "checkbox", "label": "Adaptive Blocks", "default_value": False},
            {"name": "temporal_coherence", "type": "float", "label": "Temporal Coherence", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
        ])
    def add_attributes(self):
        super().add_attributes()

class EnhancedPixelSortingNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Enhanced Pixel Sorting", parent_node_editor, params=[
            # Original parameters
            {"name": "pixel_sorting_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "pixel_sorting_direction", "type": "combo", "label": "Direction", "items": ["Horizontal", "Vertical"], "default_value": "Horizontal"},
            # New enhanced parameters
            {"name": "sorting_criteria", "type": "combo", "label": "Sort By", "items": ["brightness", "hue", "saturation", "red", "green", "blue", "random"], "default_value": "brightness"},
            {"name": "threshold_mode", "type": "combo", "label": "Threshold", "items": ["global", "adaptive", "edge_based"], "default_value": "global"},
            {"name": "sort_length_variation", "type": "float", "label": "Length Variation", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "reverse_sort_chance", "type": "int", "label": "Reverse %", "min_value": 0, "max_value": 100, "default_value": 0},
        ])
    def add_attributes(self):
        super().add_attributes()

# ==== UTILITY NODES FOR EFFECT COMBINATIONS ====

class EffectBlenderNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Effect Blender", parent_node_editor, params=[
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference", "exclusion"], "default_value": "normal"},
            {"name": "blend_amount", "type": "float", "label": "Blend Amount", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "mask_source", "type": "combo", "label": "Mask", "items": ["none", "luminance", "edges", "noise"], "default_value": "none"},
        ])
    def add_attributes(self):
        super().add_attributes()

class ConditionalEffectNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Conditional Effect", parent_node_editor, params=[
            {"name": "trigger_condition", "type": "combo", "label": "Trigger", "items": ["always", "frame_number", "luminance", "motion", "audio_reactive"], "default_value": "always"},
            {"name": "threshold_value", "type": "float", "label": "Threshold", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "fade_in_frames", "type": "int", "label": "Fade In", "min_value": 0, "max_value": 60, "default_value": 0},
            {"name": "fade_out_frames", "type": "int", "label": "Fade Out", "min_value": 0, "max_value": 60, "default_value": 0},
        ])
    def add_attributes(self):
        super().add_attributes()


# ==============================================================================
# Quality-First Overhaul Nodes
# ==============================================================================

class MultimodalAnalysisNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Multimodal Analysis", parent_node_editor, params=[
            {"name": "edge_threshold", "type": "float", "label": "Edge Thresh", "min_value": 0.0, "max_value": 1.0, "default_value": 0.15},
            {"name": "audio_boost", "type": "float", "label": "Audio Boost", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "debug_view", "type": "combo", "label": "Debug View", "items": ["passthrough", "optical_flow", "modulation_mask", "motion_mask", "edges", "local_entropy"], "default_value": "passthrough"},
        ])
    def add_attributes(self):
        super().add_attributes()


class OpticalFlowRemapNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Optical Flow Remap", parent_node_editor, params=self.with_evolution_params([
            {"name": "intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 2.0, "default_value": 1.0},
            {"name": "flow_scale", "type": "float", "label": "Flow Scale", "min_value": 0.0, "max_value": 20.0, "default_value": 4.0},
        ], "intensity", amount_label="Displace", speed_label="Speed", speed_default=1.0))
    def add_attributes(self):
        super().add_attributes()


class ControlledRecursiveFeedbackNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Controlled Recursive Feedback", parent_node_editor, params=[
            {"name": "feedback_amount", "type": "float", "label": "Feedback", "min_value": 0.0, "max_value": 0.95, "default_value": 0.65},
            {"name": "decay", "type": "float", "label": "Decay", "min_value": 0.0, "max_value": 1.0, "default_value": 0.92},
            {"name": "clean_mix", "type": "float", "label": "Clean Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.20},
            {"name": "scene_change_threshold", "type": "float", "label": "Scene Cut", "min_value": 0.05, "max_value": 1.0, "default_value": 0.40},
            {"name": "zoom", "type": "float", "label": "Zoom", "min_value": 0.90, "max_value": 1.10, "default_value": 1.01},
            {"name": "rotation", "type": "float", "label": "Rotation", "min_value": -10.0, "max_value": 10.0, "default_value": 0.0},
            {"name": "offset_x", "type": "float", "label": "Offset X", "min_value": -50.0, "max_value": 50.0, "default_value": 0.0},
            {"name": "offset_y", "type": "float", "label": "Offset Y", "min_value": -50.0, "max_value": 50.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()


class QuantizationSolarizationNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Quantization / Solarization", parent_node_editor, params=[
            {"name": "levels", "type": "int", "label": "Levels", "min_value": 2, "max_value": 256, "default_value": 16},
            {"name": "solarize_threshold", "type": "float", "label": "Solarize Thresh", "min_value": 0.0, "max_value": 1.0, "default_value": 0.65},
            {"name": "solarize_mix", "type": "float", "label": "Solarize Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.40},
            {"name": "intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 1.0},
        ])
    def add_attributes(self):
        super().add_attributes()


class EdgeReconstructionNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Edge Reconstruction", parent_node_editor, params=[
            {"name": "strength", "type": "float", "label": "Strength", "min_value": 0.0, "max_value": 2.0, "default_value": 0.75},
        ])
    def add_attributes(self):
        super().add_attributes()


class StructuralRestorationNode(GlitchEffectNode):
    def __init__(self, parent_node_editor):
        super().__init__("Structural Restoration", parent_node_editor, params=[
            {"name": "unsharp_amount", "type": "float", "label": "Unsharp", "min_value": 0.0, "max_value": 2.0, "default_value": 0.70},
            {"name": "unsharp_radius", "type": "float", "label": "Radius", "min_value": 0.5, "max_value": 5.0, "default_value": 1.2},
            {"name": "unsharp_threshold", "type": "float", "label": "Threshold", "min_value": 0.0, "max_value": 20.0, "default_value": 4.0},
            {"name": "local_contrast_clip", "type": "float", "label": "Contrast", "min_value": 0.0, "max_value": 4.0, "default_value": 1.5},
            {"name": "edge_reinforce", "type": "float", "label": "Edge Boost", "min_value": 0.0, "max_value": 1.0, "default_value": 0.35},
            {"name": "grain_amount", "type": "float", "label": "Grain", "min_value": 0.0, "max_value": 0.20, "default_value": 0.02},
        ])
    def add_attributes(self):
        super().add_attributes()


# ==== MULTI-INPUT NODES ====

class FrameDifferenceNode(Node):
    def __init__(self, parent_node_editor):
        super().__init__("Frame Difference", parent_node_editor)

    def add_attributes(self):
        with dpg.node_attribute(label="Input A", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr_a:
            dpg.add_text("Frame A")
            self.input_attrs["frame_a"] = attr_a

        with dpg.node_attribute(label="Input B", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr_b:
            dpg.add_text("Frame B")
            self.input_attrs["frame_b"] = attr_b

        with dpg.node_attribute(label="Parameters", attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_slider_float(label="Difference Threshold", min_value=0.0, max_value=1.0, default_value=0.1, width=CONTROL_WIDTH)
            dpg.add_combo(label="Output Mode", items=["difference", "motion_mask", "blend"], default_value="difference", width=CONTROL_WIDTH)

        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr_out:
            dpg.add_text("Result")
            self.output_attrs["result"] = attr_out


class FeedbackLoopNode(Node):
    def __init__(self, parent_node_editor):
        super().__init__("Feedback Loop", parent_node_editor)

    def add_attributes(self):
        with dpg.node_attribute(label="Input", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr_in:
            dpg.add_text("Current Frame")
            self.input_attrs["current_frame"] = attr_in

        with dpg.node_attribute(label="Feedback Input", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr_fb:
            dpg.add_text("Feedback Frame")
            self.input_attrs["feedback_frame"] = attr_fb

        with dpg.node_attribute(label="Parameters", attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_slider_float(label="Feedback Amount", min_value=0.0, max_value=1.0, default_value=0.5, width=CONTROL_WIDTH)
            dpg.add_slider_float(label="Zoom", min_value=0.9, max_value=1.1, default_value=1.0, width=CONTROL_WIDTH)
            dpg.add_slider_float(label="Rotation", min_value=-5.0, max_value=5.0, default_value=0.0, width=CONTROL_WIDTH)

        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr_out:
            dpg.add_text("Result")
            self.output_attrs["result"] = attr_out


class VideoMixerNode(Node):
    def __init__(self, parent_node_editor):
        super().__init__("Video Mixer", parent_node_editor)
        self.params = [
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "blend_mode", "type": "combo", "label": "Mode",
             "items": ["crossfade", "add", "multiply", "screen", "difference", "lighten", "darken"],
             "default_value": "crossfade"},
        ]
        self.param_values = {param["name"]: param["default_value"] for param in self.params}
        self.param_controls = {}

    def add_attributes(self):
        with dpg.node_attribute(label="Input A", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr_a:
            dpg.add_text("Channel A")
            self.input_attrs["image_a"] = attr_a

        with dpg.node_attribute(label="Input B", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr_b:
            dpg.add_text("Channel B")
            self.input_attrs["image_b"] = attr_b

        with dpg.node_attribute(label="Parameters", attribute_type=dpg.mvNode_Attr_Static):
            for param_info in self.params:
                param_name = param_info["name"]
                param_type = param_info["type"]
                default_value = param_info["default_value"]

                control_tag = None
                if param_type == "float":
                    control_tag = dpg.add_slider_float(
                        label=param_info["label"],
                        min_value=param_info["min_value"],
                        max_value=param_info["max_value"],
                        default_value=default_value,
                        width=CONTROL_WIDTH,
                        user_data=param_name,
                        callback=lambda s, a, u: self.update_param(u, a),
                    )
                elif param_type == "combo":
                    control_tag = dpg.add_combo(
                        label=param_info["label"],
                        items=param_info["items"],
                        default_value=default_value,
                        width=CONTROL_WIDTH,
                        user_data=param_name,
                        callback=lambda s, a, u: self.update_param(u, a),
                    )

                if control_tag:
                    self.param_controls[param_name] = control_tag

        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr_out:
            dpg.add_text("Mixed Out")
            self.output_attrs["image_out"] = attr_out

    def update_param(self, param_name, value):
        self.param_values[param_name] = value


class GenesisPriorityCompositeNode(MultiInputNode):
    def __init__(self, parent_node_editor):
        super().__init__(
            "Genesis Priority Composite",
            parent_node_editor,
            params=[
                {
                    "name": "mode",
                    "type": "combo",
                    "label": "Mode",
                    "items": ["strict_priority", "soft_priority", "priority_plus_luma"],
                    "default_value": "strict_priority",
                },
                {
                    "name": "threshold",
                    "type": "float",
                    "label": "Threshold",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 0.55,
                },
                {
                    "name": "softness",
                    "type": "float",
                    "label": "Softness",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 0.18,
                },
                {
                    "name": "mix",
                    "type": "float",
                    "label": "Mix",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 1.0,
                },
                {
                    "name": "luma_influence",
                    "type": "float",
                    "label": "Luma Bias",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 0.35,
                },
                {
                    "name": "corruption_influence",
                    "type": "float",
                    "label": "Corrupt Bias",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 0.25,
                },
                {
                    "name": "invert_priority",
                    "type": "checkbox",
                    "label": "Invert Priority",
                    "default_value": False,
                },
            ],
            input_defs=[
                {"name": "base_image", "label": "Base", "text": "Base Image"},
                {"name": "overlay_image", "label": "Overlay", "text": "Overlay Image"},
                {"name": "priority_mask", "label": "Priority", "text": "Priority Mask"},
                {"name": "luma_mask", "label": "Luma", "text": "Luma Mask"},
                {"name": "corruption_mask", "label": "Corruption", "text": "Corruption Mask"},
            ],
            output_defs=[{"name": "image_out", "label": "Output", "text": "Composite Out"}],
            node_type_key="genesis_priority_composite",
        )


class GenesisMaskKeyNode(MultiInputNode):
    def __init__(self, parent_node_editor):
        super().__init__(
            "Genesis Mask Key",
            parent_node_editor,
            params=[
                {
                    "name": "mode",
                    "type": "combo",
                    "label": "Mode",
                    "items": ["multiply_mask", "threshold_key", "invert_key", "luma_gate", "corruption_gate"],
                    "default_value": "multiply_mask",
                },
                {
                    "name": "threshold",
                    "type": "float",
                    "label": "Threshold",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 0.5,
                },
                {
                    "name": "softness",
                    "type": "float",
                    "label": "Softness",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 0.12,
                },
                {
                    "name": "strength",
                    "type": "float",
                    "label": "Strength",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "default_value": 1.0,
                },
                {
                    "name": "invert",
                    "type": "checkbox",
                    "label": "Invert",
                    "default_value": False,
                },
            ],
            input_defs=[
                {"name": "image_in", "label": "Image", "text": "Image In"},
                {"name": "mask_in", "label": "Mask", "text": "Mask In"},
                {"name": "luma_mask", "label": "Luma", "text": "Luma Mask"},
                {"name": "priority_mask", "label": "Priority", "text": "Priority Mask"},
                {"name": "corruption_mask", "label": "Corruption", "text": "Corruption Mask"},
            ],
            output_defs=[{"name": "image_out", "label": "Output", "text": "Keyed Out"}],
            node_type_key="genesis_mask_key",
        )


# ===== QUANTUM GESTALT COGNITIVE GLITCH NODES =====

class QuantumGestaltCognitiveNode(GlitchEffectNode):
    """AI-driven cognitive glitch effects based on brainwave states."""
    def __init__(self, parent_node_editor):
        super().__init__("Quantum Gestalt (AI)", parent_node_editor, params=[
            {"name": "qg_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "qg_state", "type": "combo", "label": "State", 
             "items": ["Auto (AI)", "Delta (Relaxed)", "Theta (Meditative)", "Alpha (Calm)", "Beta (Active)", "Gamma (Hyper-focus)"], 
             "default_value": "Auto (AI)"},
            {"name": "qg_progress", "type": "float", "label": "Animation", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "qg_auto_animate", "type": "checkbox", "label": "Auto Animate", "default_value": True},
        ])
    def add_attributes(self):
        super().add_attributes()


class QuantumGestaltDeltaNode(GlitchEffectNode):
    """Delta wave (relaxed) glitch: Subtle, flowing effects."""
    def __init__(self, parent_node_editor):
        super().__init__("QG Delta Wave", parent_node_editor, params=[
            {"name": "qg_delta_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.3},
            {"name": "qg_delta_progress", "type": "float", "label": "Progress", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()


class QuantumGestaltThetaNode(GlitchEffectNode):
    """Theta wave (meditative) glitch: Wave-like, smooth distortions."""
    def __init__(self, parent_node_editor):
        super().__init__("QG Theta Flow", parent_node_editor, params=[
            {"name": "qg_theta_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.4},
            {"name": "qg_theta_progress", "type": "float", "label": "Progress", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()


class QuantumGestaltAlphaNode(GlitchEffectNode):
    """Alpha wave (calm) glitch: Balanced, moderate effects."""
    def __init__(self, parent_node_editor):
        super().__init__("QG Alpha Balance", parent_node_editor, params=[
            {"name": "qg_alpha_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "qg_alpha_progress", "type": "float", "label": "Progress", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()


class QuantumGestaltBetaNode(GlitchEffectNode):
    """Beta wave (active) glitch: Sharp, aggressive effects."""
    def __init__(self, parent_node_editor):
        super().__init__("QG Beta Pulse", parent_node_editor, params=[
            {"name": "qg_beta_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.7},
            {"name": "qg_beta_progress", "type": "float", "label": "Progress", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()


class QuantumGestaltGammaNode(GlitchEffectNode):
    """Gamma wave (hyper-focus) glitch: Extreme, chaotic distortions."""
    def __init__(self, parent_node_editor):
        super().__init__("QG Gamma Chaos", parent_node_editor, params=[
            {"name": "qg_gamma_intensity", "type": "float", "label": "Intensity", "min_value": 0.0, "max_value": 1.0, "default_value": 0.9},
            {"name": "qg_gamma_progress", "type": "float", "label": "Progress", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ])
    def add_attributes(self):
        super().add_attributes()


# ==== PROCEDURAL GENERATOR NODES ====
# These nodes generate content without requiring input

class GeneratorNode(Node):
    """Base class for pure generator nodes (no input, only output)."""
    def __init__(self, label, parent_node_editor):
        super().__init__(label, parent_node_editor)
        self.is_generator = True
        self.params_def = [
            {"name": "enabled", "type": "checkbox", "label": "Enabled", "default_value": True}
        ]
        self.param_values = {}
        self.param_controls = {}
        self.input_defs = []
        self.param_ui_height = 0
        self.param_ui_width = CONTROL_WIDTH + 34
        self.param_control_width = CONTROL_WIDTH
        self.param_label_width = 120
        self.param_group_collapsible = False
        self.param_open_groups = set()
        self.output_defs = [
            {
                "name": "image_out",
                "attr_label": "Output",
                "text": "Generated Output",
                "primary": True,
            }
        ]
    
    def add_attributes(self):
        for input_def in self.input_defs:
            attr_label = str(input_def.get("attr_label") or input_def.get("label") or "Input")
            socket_text = str(input_def.get("text") or input_def.get("label") or input_def.get("name") or "Input")
            with dpg.node_attribute(label=attr_label, attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr:
                dpg.add_text(socket_text)
                self.input_attrs[str(input_def["name"])] = attr

        with dpg.node_attribute(label="Parameters", attribute_type=dpg.mvNode_Attr_Static):
            self.create_param_ui()

        for output_def in self.output_defs:
            attr_label = str(output_def.get("attr_label") or output_def.get("label") or "Output")
            socket_text = str(output_def.get("text") or output_def.get("label") or output_def.get("name") or "Output")
            with dpg.node_attribute(
                label=attr_label,
                attribute_type=dpg.mvNode_Attr_Output,
                tag=dpg.generate_uuid()
            ) as attr:
                dpg.add_text(socket_text)
                self.output_attrs[str(output_def["name"])] = attr
    
    def create_param_ui(self):
        """Create UI controls for parameters."""
        container_parent = None
        effective_param_ui_width = self._get_effective_param_ui_width()
        if self.param_ui_height and int(self.param_ui_height) > 0:
            with dpg.child_window(
                width=int(effective_param_ui_width),
                height=int(self.param_ui_height),
                border=False,
                autosize_x=False,
            ) as container_parent:
                self._build_param_ui_content(container_parent)
        else:
            self._build_param_ui_content(container_parent)

    def _get_effective_param_ui_width(self):
        label_padding = 36
        return max(
            int(self.param_ui_width),
            int(self.param_control_width) + int(self.param_label_width) + label_padding,
        )

    def _build_param_ui_content(self, parent=None):
        current_group = None
        current_parent = parent
        for param in self.params_def:
            group = param.get("group")

            if group and group != current_group:
                if current_group is not None:
                    self._add_group_gap(parent=parent)
                if self.param_group_collapsible:
                    current_parent = dpg.add_collapsing_header(
                        label=str(group),
                        default_open=(not self.param_open_groups or str(group) in self.param_open_groups),
                        parent=parent,
                    )
                else:
                    dpg.add_text(str(group), parent=parent)
                    self._add_group_gap(height=3, parent=parent)
                    current_parent = parent
                current_group = group

            self._create_param_control(param, parent=current_parent)

    def _add_group_gap(self, height=6, parent=None):
        kwargs = {}
        if parent is not None:
            kwargs["parent"] = parent
        dpg.add_spacer(height=int(height), **kwargs)

    def _create_param_control(self, param, parent=None):
        name = param["name"]
        ptype = param["type"]
        label = param["label"]
        default = param["default"]

        kwargs = {}
        if parent is not None:
            kwargs["parent"] = parent

        control_tag = None
        if ptype == "float":
            control_tag = dpg.add_slider_float(
                label=label,
                min_value=param["min"],
                max_value=param["max"],
                default_value=default,
                width=int(self.param_control_width),
                user_data=name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )
        elif ptype == "int":
            control_tag = dpg.add_slider_int(
                label=label,
                min_value=param["min"],
                max_value=param["max"],
                default_value=default,
                width=int(self.param_control_width),
                user_data=name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )
        elif ptype == "combo":
            control_tag = dpg.add_combo(
                label=label,
                items=param["items"],
                default_value=default,
                width=int(self.param_control_width),
                user_data=name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )
        elif ptype == "checkbox":
            control_tag = dpg.add_checkbox(
                label=label,
                default_value=default,
                user_data=name,
                callback=lambda s, a, u: self.update_param(u, a),
                **kwargs,
            )

        if control_tag:
            self.param_controls[name] = control_tag
            tooltip = param.get("tooltip")
            if tooltip:
                try:
                    with dpg.tooltip(parent=control_tag):
                        dpg.add_text(str(tooltip), wrap=260)
                except Exception:
                    pass
    
    def update_param(self, param_name, value):
        """Update parameter value from UI callback."""
        import logging
        self.param_values[param_name] = value
        logging.debug(f"Generator {self.label}: param '{param_name}' updated to {value}")

    def _common_generator_params(self):
        return [
            {"name": "mutation_strength", "type": "float", "label": "Mutation", "min": 0.0, "max": 1.0, "default": 0.0},
            {"name": "mutation_rate", "type": "float", "label": "Mutation Rate", "min": 0.0, "max": 5.0, "default": 1.0},
            {"name": "mutation_warp", "type": "float", "label": "Mutation Warp", "min": 0.0, "max": 2.0, "default": 0.5},
            {"name": "mutation_palette", "type": "float", "label": "Palette Drift", "min": 0.0, "max": 2.0, "default": 0.4},
            {"name": "mutation_fracture", "type": "float", "label": "Fracture", "min": 0.0, "max": 1.0, "default": 0.4},
            {"name": "mutation_echo", "type": "float", "label": "Echo", "min": 0.0, "max": 1.0, "default": 0.35},
        ]

    def _finalize_generator_params(self):
        names = {param["name"] for param in self.params_def}
        for param in self._common_generator_params():
            if param["name"] not in names:
                self.params_def.append(param)
        for param in self.params_def:
            self.param_values[param["name"]] = param["default"]


class PerlinNoiseGeneratorNode(GeneratorNode):
    """Multi-octave Perlin noise generator - pure procedural content."""
    def __init__(self, parent_node_editor):
        super().__init__("Perlin Noise Gen", parent_node_editor)
        self.params_def = [
            {"name": "scale", "type": "float", "label": "Scale", "min": 1.0, "max": 512.0, "default": 80.0},
            {"name": "octaves", "type": "int", "label": "Octaves", "min": 1, "max": 8, "default": 3},
            {"name": "persistence", "type": "float", "label": "Persistence", "min": 0.0, "max": 1.0, "default": 0.6},
            {"name": "lacunarity", "type": "float", "label": "Lacunarity", "min": 1.0, "max": 4.0, "default": 2.0},
            {"name": "time_scale", "type": "float", "label": "Time Scale", "min": 0.0, "max": 5.0, "default": 1.5},
            {"name": "color_variance", "type": "float", "label": "Color Variance", "min": 0.0, "max": 1.0, "default": 0.4},
            {"name": "brightness", "type": "float", "label": "Brightness", "min": 0.0, "max": 2.0, "default": 1.2},
            {"name": "contrast", "type": "float", "label": "Contrast", "min": 0.0, "max": 3.0, "default": 1.3},
            # Blend mode parameters
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
            # Retro mode parameters
            {"name": "retro_mode", "type": "checkbox", "label": "16-bit Mode", "default": False},
            {"name": "bits_per_channel", "type": "int", "label": "Bits/Channel", "min": 1, "max": 8, "default": 4},
            {"name": "dither", "type": "checkbox", "label": "Dither", "default": True},
        ]
        self._finalize_generator_params()


class GradientGeneratorNode(GeneratorNode):
    """Linear/radial/angular gradient generator."""
    def __init__(self, parent_node_editor):
        super().__init__("Gradient Gen", parent_node_editor)
        self.params_def = [
            {"name": "mode", "type": "combo", "label": "Mode", "items": ["linear", "radial", "angle"], "default": "linear"},
            {"name": "angle", "type": "float", "label": "Angle", "min": 0.0, "max": 360.0, "default": 0.0},
            {"name": "color1_r", "type": "int", "label": "Color 1 R", "min": 0, "max": 255, "default": 0},
            {"name": "color1_g", "type": "int", "label": "Color 1 G", "min": 0, "max": 255, "default": 0},
            {"name": "color1_b", "type": "int", "label": "Color 1 B", "min": 0, "max": 255, "default": 0},
            {"name": "color2_r", "type": "int", "label": "Color 2 R", "min": 0, "max": 255, "default": 255},
            {"name": "color2_g", "type": "int", "label": "Color 2 G", "min": 0, "max": 255, "default": 255},
            {"name": "color2_b", "type": "int", "label": "Color 2 B", "min": 0, "max": 255, "default": 255},
            # Blend mode parameters
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
            # Retro mode parameters
            {"name": "retro_mode", "type": "checkbox", "label": "16-bit Mode", "default": False},
            {"name": "bits_per_channel", "type": "int", "label": "Bits/Channel", "min": 1, "max": 8, "default": 4},
            {"name": "dither", "type": "checkbox", "label": "Dither", "default": True},
        ]
        self._finalize_generator_params()


class PlasmaGeneratorNode(GeneratorNode):
    """Colorful animated plasma generator - psychedelic visuals."""
    def __init__(self, parent_node_editor):
        super().__init__("Plasma Gen", parent_node_editor)
        self.params_def = [
            {"name": "frequency", "type": "float", "label": "Frequency", "min": 0.1, "max": 10.0, "default": 2.0},
            {"name": "speed", "type": "float", "label": "Speed", "min": 0.0, "max": 5.0, "default": 1.0},
            {"name": "hue_shift", "type": "float", "label": "Hue Shift", "min": 0.0, "max": 360.0, "default": 0.0},
            {"name": "saturation", "type": "float", "label": "Saturation", "min": 0.0, "max": 1.0, "default": 0.9},
            {"name": "complexity", "type": "int", "label": "Complexity", "min": 1, "max": 5, "default": 3},
            # Blend mode parameters
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
            # Retro mode parameters
            {"name": "retro_mode", "type": "checkbox", "label": "16-bit Mode", "default": False},
            {"name": "bits_per_channel", "type": "int", "label": "Bits/Channel", "min": 1, "max": 8, "default": 4},
            {"name": "dither", "type": "checkbox", "label": "Dither", "default": True},
        ]
        self._finalize_generator_params()


class VoronoiGeneratorNode(GeneratorNode):
    """Organic cellular/Voronoi pattern generator."""
    def __init__(self, parent_node_editor):
        super().__init__("Voronoi Gen", parent_node_editor)
        self.params_def = [
            {"name": "num_cells", "type": "int", "label": "Num Cells", "min": 5, "max": 200, "default": 30},
            {"name": "speed", "type": "float", "label": "Speed", "min": 0.0, "max": 3.0, "default": 0.5},
            {"name": "color_mode", "type": "combo", "label": "Color Mode", "items": ["rainbow", "distance", "cell_id"], "default": "rainbow"},
            {"name": "brightness", "type": "float", "label": "Brightness", "min": 0.5, "max": 2.0, "default": 1.2},
            # Blend mode parameters
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
            # Retro mode parameters
            {"name": "retro_mode", "type": "checkbox", "label": "16-bit Mode", "default": False},
            {"name": "bits_per_channel", "type": "int", "label": "Bits/Channel", "min": 1, "max": 8, "default": 4},
            {"name": "dither", "type": "checkbox", "label": "Dither", "default": True},
        ]
        self._finalize_generator_params()
        self.cell_positions = None


class FractalGeneratorNode(GeneratorNode):
    """Mandelbrot fractal generator with infinite zoom capability."""
    def __init__(self, parent_node_editor):
        super().__init__("Fractal Gen", parent_node_editor)
        self.params_def = [
            {"name": "zoom", "type": "float", "label": "Zoom", "min": 0.1, "max": 100.0, "default": 1.0},
            {"name": "center_x", "type": "float", "label": "Center X", "min": -2.0, "max": 2.0, "default": -0.5},
            {"name": "center_y", "type": "float", "label": "Center Y", "min": -2.0, "max": 2.0, "default": 0.0},
            {"name": "max_iter", "type": "int", "label": "Max Iterations", "min": 10, "max": 100, "default": 30},
            {"name": "color_speed", "type": "float", "label": "Color Speed", "min": 0.0, "max": 5.0, "default": 1.0},
            {"name": "continuous_zoom", "type": "checkbox", "label": "Infinite Zoom", "default": False},
            {"name": "zoom_speed", "type": "float", "label": "Zoom Speed", "min": 0.1, "max": 5.0, "default": 1.0},
            # Blend mode parameters
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
            # Retro mode parameters
            {"name": "retro_mode", "type": "checkbox", "label": "16-bit Mode", "default": False},
            {"name": "bits_per_channel", "type": "int", "label": "Bits/Channel", "min": 1, "max": 8, "default": 4},
            {"name": "dither", "type": "checkbox", "label": "Dither", "default": True},
        ]
        self._finalize_generator_params()


class CheckerboardGeneratorNode(GeneratorNode):
    """Animated geometric checkerboard/grid patterns."""
    def __init__(self, parent_node_editor):
        super().__init__("Checkerboard Gen", parent_node_editor)
        self.params_def = [
            {"name": "grid_size", "type": "int", "label": "Grid Size", "min": 2, "max": 50, "default": 8},
            {"name": "rotation", "type": "float", "label": "Rotation", "min": 0.0, "max": 360.0, "default": 0.0},
            {"name": "rotation_speed", "type": "float", "label": "Rotation Speed", "min": 0.0, "max": 5.0, "default": 0.0},
            {"name": "color1_r", "type": "int", "label": "Color 1 R", "min": 0, "max": 255, "default": 255},
            {"name": "color1_g", "type": "int", "label": "Color 1 G", "min": 0, "max": 255, "default": 0},
            {"name": "color1_b", "type": "int", "label": "Color 1 B", "min": 0, "max": 255, "default": 255},
            {"name": "color2_r", "type": "int", "label": "Color 2 R", "min": 0, "max": 255, "default": 0},
            {"name": "color2_g", "type": "int", "label": "Color 2 G", "min": 0, "max": 255, "default": 255},
            {"name": "color2_b", "type": "int", "label": "Color 2 B", "min": 0, "max": 255, "default": 255},
            # Blend mode parameters
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
            # Retro mode parameters
            {"name": "retro_mode", "type": "checkbox", "label": "16-bit Mode", "default": False},
            {"name": "bits_per_channel", "type": "int", "label": "Bits/Channel", "min": 1, "max": 8, "default": 4},
            {"name": "dither", "type": "checkbox", "label": "Dither", "default": True},
        ]
        self._finalize_generator_params()


class ScanFeedbackGeneratorNode(GeneratorNode):
    """Broken sync / scan feedback source with retrace bursts and unstable channel drift."""
    def __init__(self, parent_node_editor):
        super().__init__("Scan Feedback Gen", parent_node_editor)
        self.params_def = [
            {"name": "stripe_density", "type": "float", "label": "Stripe Density", "min": 8.0, "max": 160.0, "default": 72.0},
            {"name": "drift_speed", "type": "float", "label": "Drift Speed", "min": 0.0, "max": 5.0, "default": 1.2},
            {"name": "tear_amount", "type": "float", "label": "Tear Amount", "min": 0.0, "max": 1.0, "default": 0.35},
            {"name": "feedback", "type": "float", "label": "Feedback", "min": 0.0, "max": 1.0, "default": 0.45},
            {"name": "noise_amount", "type": "float", "label": "Noise", "min": 0.0, "max": 0.6, "default": 0.12},
            {"name": "channel_split", "type": "float", "label": "Channel Split", "min": 0.0, "max": 1.0, "default": 0.3},
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
        ]
        self._finalize_generator_params()


class XorLatticeGeneratorNode(GeneratorNode):
    """Bitwise interference lattice with hard digital contouring and coordinate warp."""
    def __init__(self, parent_node_editor):
        super().__init__("XOR Lattice Gen", parent_node_editor)
        self.params_def = [
            {"name": "lattice_scale", "type": "float", "label": "Lattice Scale", "min": 4.0, "max": 48.0, "default": 18.0},
            {"name": "warp", "type": "float", "label": "Warp", "min": 0.0, "max": 1.5, "default": 0.55},
            {"name": "speed", "type": "float", "label": "Speed", "min": 0.0, "max": 5.0, "default": 1.0},
            {"name": "palette_shift", "type": "float", "label": "Palette Shift", "min": 0.0, "max": 2.0, "default": 0.0},
            {"name": "contrast", "type": "float", "label": "Contrast", "min": 0.5, "max": 3.0, "default": 1.3},
            {"name": "bit_depth", "type": "int", "label": "Bit Depth", "min": 1, "max": 8, "default": 8},
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
        ]
        self._finalize_generator_params()


class MacroblockEchoGeneratorNode(GeneratorNode):
    """Predictive-compression inspired source with drifting macroblocks and echo ghosts."""
    def __init__(self, parent_node_editor):
        super().__init__("Macroblock Echo Gen", parent_node_editor)
        self.params_def = [
            {"name": "block_size", "type": "int", "label": "Block Size", "min": 4, "max": 48, "default": 18},
            {"name": "drift", "type": "float", "label": "Drift", "min": 0.0, "max": 4.0, "default": 1.0},
            {"name": "echo_strength", "type": "float", "label": "Echo Strength", "min": 0.0, "max": 1.0, "default": 0.5},
            {"name": "corruption", "type": "float", "label": "Corruption", "min": 0.0, "max": 1.0, "default": 0.35},
            {"name": "palette_levels", "type": "int", "label": "Palette Levels", "min": 2, "max": 16, "default": 6},
            {"name": "blend_mode", "type": "combo", "label": "Blend Mode", "items": ["replace", "add", "multiply", "screen", "overlay", "difference", "lighten", "darken", "color_dodge", "color_burn"], "default": "replace"},
            {"name": "blend_opacity", "type": "float", "label": "Blend Opacity", "min": 0.0, "max": 1.0, "default": 1.0},
        ]
        self._finalize_generator_params()


class BubbleSimGeneratorNode(GeneratorNode):
    """Bubble simulation source preview with parameters matching the OpenGL bubble texture generator."""
    def __init__(self, parent_node_editor):
        super().__init__("Bubble Sim Gen", parent_node_editor)
        self.param_ui_height = 420
        self.param_ui_width = 430
        self.param_control_width = 230
        self.param_label_width = 170
        self.param_group_collapsible = True
        self.param_open_groups = {"Simulation", "Render"}
        self.input_defs = [
            {
                "name": "image_in",
                "attr_label": "Input",
                "text": "Background / Pass Through",
            }
        ]
        self.output_defs = [
            {"name": "image_out", "attr_label": "Output", "text": "Bubble Composite", "primary": True},
            {"name": "bubble_mask", "attr_label": "Mask", "text": "Bubble Mask"},
            {"name": "bubble_normal", "attr_label": "Normal", "text": "Bubble Normal"},
            {"name": "bubble_depth", "attr_label": "Depth", "text": "Bubble Depth"},
            {"name": "bubble_refraction", "attr_label": "Refraction", "text": "Bubble Refraction"},
            {"name": "bubble_highlight", "attr_label": "Highlight", "text": "Bubble Highlight"},
            {"name": "bubble_debug", "attr_label": "Debug", "text": "Bubble Debug"},
        ]
        self.params_def = [
            {"name": "seed", "type": "int", "label": "Seed", "min": 0, "max": 999999, "default": 1977, "group": "Simulation"},
            {"name": "particle_texture_size", "type": "int", "label": "Particle Tex Size", "min": 16, "max": 512, "default": 128, "group": "Simulation"},
            {"name": "density", "type": "float", "label": "Density", "min": 0.0, "max": 1.0, "default": 0.55, "group": "Simulation"},
            {"name": "base_radius", "type": "float", "label": "Base Radius", "min": 0.001, "max": 0.08, "default": 0.018, "group": "Simulation"},
            {"name": "radius_variation", "type": "float", "label": "Radius Variation", "min": 0.0, "max": 1.0, "default": 0.65, "group": "Simulation"},
            {"name": "buoyancy", "type": "float", "label": "Buoyancy", "min": 0.0, "max": 2.0, "default": 0.16, "group": "Simulation"},
            {"name": "drag", "type": "float", "label": "Drag", "min": 0.0, "max": 1.0, "default": 0.92, "group": "Simulation"},
            {"name": "turbulence", "type": "float", "label": "Turbulence", "min": 0.0, "max": 2.0, "default": 0.35, "group": "Simulation"},
            {"name": "flow_speed", "type": "float", "label": "Flow Speed", "min": 0.0, "max": 4.0, "default": 0.25, "group": "Simulation"},
            {"name": "depth_spread", "type": "float", "label": "Depth Spread", "min": 0.0, "max": 1.0, "default": 0.8, "group": "Simulation"},
            {"name": "depth_motion", "type": "float", "label": "Depth Motion", "min": 0.0, "max": 2.0, "default": 0.7, "group": "Simulation"},
            {"name": "orbit_strength", "type": "float", "label": "Orbit Strength", "min": 0.0, "max": 2.0, "default": 0.8, "group": "Simulation"},
            {"name": "perspective", "type": "float", "label": "Perspective", "min": 0.0, "max": 2.0, "default": 1.0, "group": "Simulation"},
            {"name": "show_energy", "type": "float", "label": "Show Energy", "min": 0.0, "max": 1.0, "default": 0.0, "group": "Simulation"},
            {"name": "show_chaos", "type": "float", "label": "Show Chaos", "min": 0.0, "max": 1.0, "default": 0.0, "group": "Simulation"},
            {"name": "refraction_strength", "type": "float", "label": "Refraction", "min": 0.0, "max": 0.15, "default": 0.025, "group": "Render"},
            {"name": "highlight_strength", "type": "float", "label": "Highlight", "min": 0.0, "max": 2.0, "default": 0.45, "group": "Render"},
            {"name": "sparkle", "type": "float", "label": "Sparkle", "min": 0.0, "max": 1.0, "default": 0.25, "group": "Render"},
            {"name": "debug_mode", "type": "combo", "label": "Output View", "items": ["composite", "mask", "normal", "depth", "refraction", "highlight"], "default": "composite", "group": "Render"},
        ]
        self._finalize_generator_params()


class BubbleSimGenerator2Node(GeneratorNode):
    """Persistent multipass refractive bubble generator with CPU fallback."""

    def __init__(self, parent_node_editor):
        super().__init__("Bubble Sim Gen (2)", parent_node_editor)
        self.param_ui_height = 440
        self.param_ui_width = 430
        self.param_control_width = 230
        self.param_label_width = 170
        self.param_group_collapsible = True
        self.param_open_groups = {"Simulation", "Render"}
        self.input_defs = [
            {"name": "image_in", "attr_label": "Input", "text": "Background / Pass Through"}
        ]
        self.output_defs = [
            {"name": "image_out", "attr_label": "Output", "text": "Bubble Beauty", "primary": True},
            {"name": "bubble_mask", "attr_label": "Mask", "text": "Bubble Mask"},
            {"name": "bubble_normal", "attr_label": "Normal", "text": "Bubble Normal"},
            {"name": "bubble_depth", "attr_label": "Depth", "text": "Bubble Depth / Age / Radius"},
            {"name": "bubble_refraction", "attr_label": "Refraction", "text": "Bubble Refraction / Rim"},
            {"name": "bubble_highlight", "attr_label": "Highlight", "text": "Bubble Highlight"},
            {"name": "bubble_debug", "attr_label": "Debug", "text": "Runtime / Debug View"},
        ]
        self.params_def = [
            {"name": "enabled", "type": "checkbox", "label": "Enabled", "default": True, "group": "Simulation"},
            {"name": "seed", "type": "int", "label": "Seed", "min": 0, "max": 999999, "default": 1977, "group": "Simulation"},
            {"name": "particle_texture_size", "type": "int", "label": "Particle Tex Size", "min": 16, "max": 512, "default": 64, "group": "Simulation"},
            {"name": "render_scale", "type": "float", "label": "Render Scale", "min": 0.25, "max": 1.0, "default": 0.5, "group": "Simulation"},
            {"name": "density", "type": "float", "label": "Density", "min": 0.0, "max": 1.0, "default": 0.12, "group": "Simulation"},
            {"name": "base_radius", "type": "float", "label": "Base Radius", "min": 0.001, "max": 0.08, "default": 0.021, "group": "Simulation"},
            {"name": "radius_variation", "type": "float", "label": "Radius Variation", "min": 0.0, "max": 1.0, "default": 0.8, "group": "Simulation"},
            {"name": "buoyancy", "type": "float", "label": "Buoyancy", "min": 0.0, "max": 2.0, "default": 0.02, "group": "Simulation"},
            {"name": "drag", "type": "float", "label": "Drag", "min": 0.0, "max": 1.0, "default": 0.94, "group": "Simulation"},
            {"name": "turbulence", "type": "float", "label": "Turbulence", "min": 0.0, "max": 2.0, "default": 0.30, "group": "Simulation"},
            {"name": "flow_speed", "type": "float", "label": "Flow Speed", "min": 0.0, "max": 4.0, "default": 0.22, "group": "Simulation"},
            {"name": "motion_speed", "type": "float", "label": "Motion Speed", "min": 0.0, "max": 4.0, "default": 1.0, "group": "Simulation"},
            {"name": "motion_style", "type": "combo", "label": "Motion Style", "items": ["Fluid Drift", "Rising", "Vortex", "Burst"], "default": "Vortex", "group": "Simulation"},
            {"name": "depth_spread", "type": "float", "label": "Depth Spread", "min": 0.0, "max": 1.0, "default": 0.95, "group": "Simulation"},
            {"name": "depth_motion", "type": "float", "label": "Depth Motion", "min": 0.0, "max": 2.0, "default": 0.35, "group": "Simulation"},
            {"name": "orbit_strength", "type": "float", "label": "Orbit Strength", "min": 0.0, "max": 2.0, "default": 0.70, "group": "Simulation"},
            {"name": "perspective", "type": "float", "label": "Perspective", "min": 0.0, "max": 2.0, "default": 1.2, "group": "Simulation"},
            {"name": "show_energy", "type": "float", "label": "Show Energy", "min": 0.0, "max": 1.0, "default": 0.0, "group": "Simulation"},
            {"name": "show_chaos", "type": "float", "label": "Show Chaos", "min": 0.0, "max": 1.0, "default": 0.0, "group": "Simulation"},
            {"name": "refraction_strength", "type": "float", "label": "Refraction", "min": 0.0, "max": 0.15, "default": 0.040, "group": "Render"},
            {"name": "highlight_strength", "type": "float", "label": "Highlight", "min": 0.0, "max": 2.0, "default": 0.34, "group": "Render"},
            {"name": "sparkle", "type": "float", "label": "Sparkle", "min": 0.0, "max": 1.0, "default": 0.04, "group": "Render"},
            {"name": "debug_mode", "type": "combo", "label": "Output View", "items": ["composite", "mask", "normal", "depth", "refraction", "highlight"], "default": "composite", "group": "Render"},
        ]
        self._finalize_generator_params()


class FMSynthNode(GeneratorNode):
    """Real-time four-operator FM audio generator and modulation source."""

    def __init__(self, parent_node_editor):
        super().__init__("FM Synth", parent_node_editor)
        from megaglitch.nodes.fm_synth_node import FMSynthNode as FMSynthBackend

        self.is_audio_node = True
        self.output_defs = []
        self.param_ui_height = 460
        self.param_ui_width = 430
        self.param_control_width = 230
        self.param_label_width = 170
        self.param_group_collapsible = True
        self.param_open_groups = {"Voice", "Envelope"}
        self.params_def = [
            {"name": "enabled", "type": "checkbox", "label": "Audio Enabled", "default": True, "group": "Voice"},
            {"name": "master_gain", "type": "float", "label": "Master Gain", "min": 0.0, "max": 1.0, "default": 0.35, "group": "Voice"},
            {"name": "carrier_freq", "type": "float", "label": "Carrier Hz", "min": 20.0, "max": 4000.0, "default": 110.0, "group": "Voice"},
            {"name": "modulation_index", "type": "float", "label": "Mod Index", "min": 0.0, "max": 24.0, "default": 1.5, "group": "Voice"},
            {"name": "feedback", "type": "float", "label": "Feedback", "min": 0.0, "max": 0.95, "default": 0.05, "group": "Voice"},
            {"name": "detune", "type": "float", "label": "Detune Cents", "min": -50.0, "max": 50.0, "default": 0.0, "group": "Voice"},
            {"name": "voice_mode", "type": "combo", "label": "Voice Mode", "items": ["tx802", "legacy"], "default": "tx802", "group": "TX802"},
            {"name": "algorithm", "type": "int", "label": "Algorithm", "min": 1, "max": 32, "default": 1, "group": "TX802"},
            {"name": "polyphony", "type": "int", "label": "Polyphony", "min": 1, "max": 16, "default": 16, "group": "TX802"},
            {"name": "operator_ratio_1", "type": "float", "label": "Op 1 Ratio", "min": 0.125, "max": 16.0, "default": 1.0, "group": "Operators"},
            {"name": "operator_ratio_2", "type": "float", "label": "Op 2 Ratio", "min": 0.125, "max": 16.0, "default": 2.0, "group": "Operators"},
            {"name": "operator_ratio_3", "type": "float", "label": "Op 3 Ratio", "min": 0.125, "max": 16.0, "default": 3.0, "group": "Operators"},
            {"name": "operator_ratio_4", "type": "float", "label": "Op 4 Ratio", "min": 0.125, "max": 16.0, "default": 5.0, "group": "Operators"},
            {"name": "operator_ratio_5", "type": "float", "label": "Op 5 Ratio", "min": 0.125, "max": 32.0, "default": 0.5, "group": "Operators"},
            {"name": "operator_ratio_6", "type": "float", "label": "Op 6 Ratio", "min": 0.125, "max": 32.0, "default": 7.0, "group": "Operators"},
            {"name": "operator_detune_1", "type": "float", "label": "Op 1 Detune", "min": -100.0, "max": 100.0, "default": 0.0, "group": "Operators"},
            {"name": "operator_detune_2", "type": "float", "label": "Op 2 Detune", "min": -100.0, "max": 100.0, "default": 0.0, "group": "Operators"},
            {"name": "operator_detune_3", "type": "float", "label": "Op 3 Detune", "min": -100.0, "max": 100.0, "default": 0.0, "group": "Operators"},
            {"name": "operator_detune_4", "type": "float", "label": "Op 4 Detune", "min": -100.0, "max": 100.0, "default": 0.0, "group": "Operators"},
            {"name": "operator_detune_5", "type": "float", "label": "Op 5 Detune", "min": -100.0, "max": 100.0, "default": 0.0, "group": "Operators"},
            {"name": "operator_detune_6", "type": "float", "label": "Op 6 Detune", "min": -100.0, "max": 100.0, "default": 0.0, "group": "Operators"},
            {"name": "operator_mix_1", "type": "float", "label": "Op 1 Mix", "min": 0.0, "max": 1.0, "default": 1.0, "group": "Operator Mix"},
            {"name": "operator_mix_2", "type": "float", "label": "Op 2 Mix", "min": 0.0, "max": 1.0, "default": 0.65, "group": "Operator Mix"},
            {"name": "operator_mix_3", "type": "float", "label": "Op 3 Mix", "min": 0.0, "max": 1.0, "default": 0.45, "group": "Operator Mix"},
            {"name": "operator_mix_4", "type": "float", "label": "Op 4 Mix", "min": 0.0, "max": 1.0, "default": 0.3, "group": "Operator Mix"},
            {"name": "operator_level_5", "type": "float", "label": "Op 5 Level", "min": 0.0, "max": 1.0, "default": 0.22, "group": "Operator Mix"},
            {"name": "operator_level_6", "type": "float", "label": "Op 6 Level", "min": 0.0, "max": 1.0, "default": 0.18, "group": "Operator Mix"},
            {"name": "operator_pan_1", "type": "float", "label": "Op 1 Pan", "min": -1.0, "max": 1.0, "default": -0.2, "group": "Operator Mix"},
            {"name": "operator_pan_2", "type": "float", "label": "Op 2 Pan", "min": -1.0, "max": 1.0, "default": 0.2, "group": "Operator Mix"},
            {"name": "operator_pan_3", "type": "float", "label": "Op 3 Pan", "min": -1.0, "max": 1.0, "default": -0.55, "group": "Operator Mix"},
            {"name": "operator_pan_4", "type": "float", "label": "Op 4 Pan", "min": -1.0, "max": 1.0, "default": 0.55, "group": "Operator Mix"},
            {"name": "operator_pan_5", "type": "float", "label": "Op 5 Pan", "min": -1.0, "max": 1.0, "default": -0.35, "group": "Operator Mix"},
            {"name": "operator_pan_6", "type": "float", "label": "Op 6 Pan", "min": -1.0, "max": 1.0, "default": 0.35, "group": "Operator Mix"},
            {"name": "operator_fixed_5", "type": "checkbox", "label": "Op 5 Fixed", "default": False, "group": "TX802"},
            {"name": "operator_fixed_freq_5", "type": "float", "label": "Op 5 Fixed Hz", "min": 0.1, "max": 12000.0, "default": 440.0, "group": "TX802"},
            {"name": "operator_fixed_6", "type": "checkbox", "label": "Op 6 Fixed", "default": False, "group": "TX802"},
            {"name": "operator_fixed_freq_6", "type": "float", "label": "Op 6 Fixed Hz", "min": 0.1, "max": 12000.0, "default": 440.0, "group": "TX802"},
            {"name": "attack", "type": "float", "label": "Attack", "min": 0.001, "max": 10.0, "default": 0.02, "group": "Envelope"},
            {"name": "decay", "type": "float", "label": "Decay", "min": 0.001, "max": 20.0, "default": 0.4, "group": "Envelope"},
            {"name": "sustain", "type": "float", "label": "Sustain", "min": 0.0, "max": 1.0, "default": 0.6, "group": "Envelope"},
            {"name": "release", "type": "float", "label": "Release", "min": 0.001, "max": 30.0, "default": 1.2, "group": "Envelope"},
            {"name": "lfo_rate", "type": "float", "label": "LFO Rate", "min": 0.0, "max": 30.0, "default": 0.1, "group": "Motion"},
            {"name": "lfo_depth", "type": "float", "label": "LFO Depth", "min": 0.0, "max": 1.0, "default": 0.1, "group": "Motion"},
            {"name": "lfo_sync", "type": "checkbox", "label": "Tempo Sync", "default": False, "group": "Motion"},
            {"name": "lfo_division", "type": "float", "label": "Beat Division", "min": 0.125, "max": 8.0, "default": 1.0, "group": "Motion"},
            {"name": "tempo_bpm", "type": "float", "label": "Tempo BPM", "min": 20.0, "max": 300.0, "default": 120.0, "group": "Motion"},
            {"name": "velocity_sensitivity", "type": "float", "label": "Velocity", "min": 0.0, "max": 1.0, "default": 0.8, "group": "Performance"},
            {"name": "scale_root", "type": "int", "label": "Scale Root", "min": 0, "max": 127, "default": 36, "group": "Pitch"},
            {"name": "scale_mode", "type": "combo", "label": "Scale", "items": ["chromatic", "major", "minor", "minor_pentatonic", "whole_tone", "diminished"], "default": "minor_pentatonic", "group": "Pitch"},
            {"name": "quantize_pitch", "type": "checkbox", "label": "Quantize Pitch", "default": True, "group": "Pitch"},
            {"name": "show_energy", "type": "float", "label": "Show Energy", "min": 0.0, "max": 1.0, "default": 0.0, "group": "Show"},
            {"name": "show_chaos", "type": "float", "label": "Show Chaos", "min": 0.0, "max": 1.0, "default": 0.0, "group": "Show"},
            {"name": "trigger", "type": "checkbox", "label": "Trigger", "default": False, "group": "Performance"},
        ]
        self.param_values = {param["name"]: param["default"] for param in self.params_def}
        self.audio_node = FMSynthBackend(
            modulation_bus=get_audiovisual_modulation_bus(parent_node_editor)
        )
        for name, value in self.param_values.items():
            if name != "trigger":
                self.audio_node.set_param(name, value)

    def create_node(self, pos):
        super().create_node(pos)
        self.audio_node.start()

    def update_param(self, param_name, value):
        super().update_param(param_name, value)
        try:
            self.audio_node.set_param(param_name, value)
            if param_name == "enabled":
                if bool(value):
                    self.audio_node.start()
                else:
                    self.audio_node.panic()
                    self.audio_node.stop()
        except Exception:
            logging.exception("Could not update FM synth parameter '%s'", param_name)

    def update_audio(self, time_value, dt, slot_time=None):
        return self.audio_node.update(time_value, dt, time_value if slot_time is None else slot_time)

    def get_analysis_outputs(self):
        return self.audio_node.get_outputs()

    def destroy(self):
        self.audio_node.destroy()


class ColonyReefGeneratorNode(GeneratorNode):
    """Stateful multi-colony ecosystem generator with persistent territorial memory."""
    def __init__(self, parent_node_editor):
        super().__init__("Colony Reef Gen", parent_node_editor)
        self.params_def = [
            {"name": "mutation_rate", "type": "float", "label": "Mutation Rate", "min": 0.0, "max": 3.0, "default": 0.28},
            {"name": "memory_decay", "type": "float", "label": "Memory Decay", "min": 0.0, "max": 1.0, "default": 0.14},
            {"name": "colony_count", "type": "int", "label": "Colony Count", "min": 3, "max": 24, "default": 9},
            {"name": "phase_instability", "type": "float", "label": "Phase Instability", "min": 0.0, "max": 1.0, "default": 0.42},
            {"name": "region_spread_speed", "type": "float", "label": "Region Spread", "min": 0.1, "max": 3.0, "default": 1.0},
            {"name": "catastrophic_reset_probability", "type": "float", "label": "Reset Prob", "min": 0.0, "max": 0.25, "default": 0.01},
            {"name": "nutrient_flux", "type": "float", "label": "Nutrient Flux", "min": 0.0, "max": 2.0, "default": 1.0},
            {"name": "predator_pressure", "type": "float", "label": "Predator Pressure", "min": 0.0, "max": 2.0, "default": 0.7},
            {"name": "luminous_residue", "type": "float", "label": "Luminous Residue", "min": 0.0, "max": 2.0, "default": 0.9},
        ]
        for param in self.params_def:
            self.param_values[param["name"]] = param["default"]


class GenesisVDPGeneratorNode(GeneratorNode):
    """Mini VDP-style procedural renderer with tilemaps, sprites, raster logic, and corruption."""
    def __init__(self, parent_node_editor):
        super().__init__("Genesis VDP Gen", parent_node_editor)
        from genesis_vdp_renderer import get_genesis_vdp_io_defs, get_genesis_vdp_param_defs

        self.param_ui_height = 420
        self.param_ui_width = 430
        self.param_control_width = 230
        self.param_label_width = 170
        self.param_group_collapsible = True
        self.param_open_groups = {"Scene", "Palette", "Pattern"}
        self.params_def = get_genesis_vdp_param_defs()
        self.input_defs, self.output_defs = get_genesis_vdp_io_defs()
        for param in self.params_def:
            self.param_values[param["name"]] = param["default"]
