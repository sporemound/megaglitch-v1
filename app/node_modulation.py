import dearpygui.dearpygui as dpg
import math
import time
import random
from typing import Any, Callable, Dict, List, Tuple


class NodeModulationMatrix:
    """
    Lightweight modulation matrix for the node-based app.

    Optional providers can be supplied for external signals:
        'fft_data'      -> Callable returning np.ndarray or None
        'midi_note'     -> Callable returning float in 0..127
        'midi_cc'       -> Callable returning float in 0..127
        'bpm_phase'     -> Callable returning float 0..1
        'bpm_pulse'     -> Callable returning float 0..1
        'bpm_normalized'-> Callable returning float 0..1
        'bpm_value'     -> Callable returning raw BPM value
        'fft_bands'     -> Callable returning {band_name: normalized value}
        'onset_value'   -> Callable returning float 0..1
        'onset_event_id'-> Callable returning monotonically increasing onset id
        'beat_event_id' -> Callable returning monotonically increasing beat id
    """

    def __init__(self, node_editor, providers: Dict[str, Callable[[], Any]] | None = None,
                 extra_targets_provider: Callable[[], List[Dict[str, Any]]] | None = None):
        self.node_editor = node_editor
        self.enabled = True
        self.connections: List[str] = []  # e.g., ["Time -> glitch_amount_x"]
        self.param_ranges: Dict[str, Tuple[float, float]] = {}
        self.param_labels: Dict[str, str] = {}
        self.param_types: Dict[str, str] = {}
        self.param_items: Dict[str, List[Any]] = {}
        self.param_mod_modes: Dict[str, str] = {}
        self.param_mod_spans: Dict[str, float] = {}
        self._discrete_event_state: Dict[str, Tuple[int, int]] = {}
        self.window_tag = "modulation_matrix_window"
        self.conn_list_tag = "mod_connections_list"
        self.source_checks: Dict[str, int] = {}
        self.target_checks: Dict[str, int] = {}
        self.providers = providers or {}
        self.extra_targets_provider = extra_targets_provider
        self.targets_child_tag = None
        self.presets = {
            "None": [],
            "Wavy Glitch": ["Time -> glitch_amount_x", "Time -> glitch_amount_y"],
            "Random Pixelation": ["Random -> pixelation_intensity"],
            "Pulsing Aberration": ["Time -> chromatic_aberration"],
            "Wave Distortion": ["Time -> wave_distortion"],
            "Data Moshing": ["Random -> data_moshing_intensity"],
            "Beat Pulses": ["Beat Pulse -> glitch_amount_x", "Beat Pulse -> glitch_amount_y"],
            "Bass Drive": ["Bass -> glitch_amount_x", "Low Mid -> pixel_sorting_intensity"],
            "Transient Switches": ["Onset Trigger -> palette_mode", "Beat Trigger -> blend_mode"],
        }

    def _collect_targets_and_ranges(self):
        targets = set()
        ranges: Dict[str, Tuple[float, float]] = {}
        labels: Dict[str, str] = {}
        types: Dict[str, str] = {}
        items: Dict[str, List[Any]] = {}
        mod_modes: Dict[str, str] = {}
        mod_spans: Dict[str, float] = {}
        
        try:
            for node in self.node_editor.nodes:
                try:
                    # Lua shader uniforms are metadata-driven and are not part
                    # of the inherited base ``params`` list.
                    params = []
                    regular_params = getattr(node, "params", None)
                    generator_params = getattr(node, "params_def", None)
                    dynamic_params = getattr(node, "dynamic_param_defs", None)
                    if regular_params:
                        params.extend(
                            regular_params.values() if isinstance(regular_params, dict) else regular_params
                        )
                    elif generator_params:
                        params.extend(
                            generator_params.values() if isinstance(generator_params, dict) else generator_params
                        )
                    if dynamic_params:
                        dynamic_values = (
                            dynamic_params.values() if isinstance(dynamic_params, dict) else dynamic_params
                        )
                        known_names = {
                            str(param.get("name"))
                            for param in params
                            if isinstance(param, dict) and param.get("name")
                        }
                        params.extend(
                            param for param in dynamic_values
                            if isinstance(param, dict) and str(param.get("name")) not in known_names
                        )
                    
                    if not params:
                        continue
                    
                    for p in params:
                        if not isinstance(p, dict):
                            continue
                        
                        name = p.get("name")
                        if not name:
                            continue
                        
                        targets.add(name)
                        labels[name] = str(p.get("label") or name)
                        types[name] = str(p.get("type") or "float")
                        if isinstance(p.get("items"), (list, tuple)):
                            items[name] = list(p.get("items") or [])
                        mod_modes[name] = str(p.get("mod_mode") or "replace")
                        
                        # Handle both "min_value"/"max_value" (effect nodes) and "min"/"max" (generator nodes)
                        mn = p.get("min_value")
                        if mn is None:
                            mn = p.get("min", 0.0)
                        mx = p.get("max_value")
                        if mx is None:
                            mx = p.get("max", 1.0)
                        
                        # Ensure min/max are valid numbers
                        try:
                            mn = float(mn)
                            mx = float(mx)
                        except (TypeError, ValueError):
                            mn, mx = 0.0, 1.0
                        
                        ranges[name] = (mn, mx)
                        try:
                            mod_span = p.get("mod_span", (mx - mn) * 0.5)
                            mod_spans[name] = float(mod_span)
                        except (TypeError, ValueError):
                            mod_spans[name] = float(max(0.0, (mx - mn) * 0.5))
                except Exception as e:
                    # Skip problematic nodes but continue processing others
                    import logging
                    logging.debug(f"Skipping node during target collection: {e}")
                    continue
            extra_targets = []
            if callable(self.extra_targets_provider):
                try:
                    extra_targets = list(self.extra_targets_provider() or [])
                except Exception as e:
                    import logging
                    logging.debug(f"Skipping external modulation targets: {e}")
                    extra_targets = []

            for p in extra_targets:
                if not isinstance(p, dict):
                    continue
                name = p.get("name")
                if not name:
                    continue
                targets.add(name)
                labels[name] = str(p.get("label") or name)
                types[name] = str(p.get("type") or "float")
                if isinstance(p.get("items"), (list, tuple)):
                    items[name] = list(p.get("items") or [])
                mod_modes[name] = str(p.get("mod_mode") or "replace")
                try:
                    mn = float(p.get("min_value", p.get("min", 0.0)))
                    mx = float(p.get("max_value", p.get("max", 1.0)))
                except (TypeError, ValueError):
                    mn, mx = 0.0, 1.0
                ranges[name] = (mn, mx)
                try:
                    mod_span = p.get("mod_span", (mx - mn) * 0.5)
                    mod_spans[name] = float(mod_span)
                except (TypeError, ValueError):
                    mod_spans[name] = float(max(0.0, (mx - mn) * 0.5))
        except Exception as e:
            import logging
            logging.error(f"Error collecting modulation targets: {e}")
        
        self.param_ranges = ranges
        self.param_labels = labels
        self.param_types = types
        self.param_items = items
        self.param_mod_modes = mod_modes
        self.param_mod_spans = mod_spans
        return sorted(targets, key=lambda name: labels.get(name, name).lower())

    def _reset_ui_state(self):
        self.source_checks.clear()
        self.target_checks.clear()
        self.targets_child_tag = None

    def _sync_checks_from_connections(self):
        for widget in self.source_checks.values():
            if dpg.does_item_exist(widget):
                dpg.set_value(widget, False)
        for widget in self.target_checks.values():
            if dpg.does_item_exist(widget):
                dpg.set_value(widget, False)

        for connection in self.connections:
            try:
                src, tgt = [x.strip() for x in connection.split("->")]
            except Exception:
                continue
            source_widget = self.source_checks.get(src)
            target_widget = self.target_checks.get(tgt)
            if source_widget and dpg.does_item_exist(source_widget):
                dpg.set_value(source_widget, True)
            if target_widget and dpg.does_item_exist(target_widget):
                dpg.set_value(target_widget, True)

    def show_window(self):
        sources = [
            "Time",
            "Random",
            "Beat Pulse",
            "Beat Trigger",
            "Beat Phase",
            "BPM (Normalized)",
            "Onset Energy",
            "Onset Trigger",
            "Bass Amplitude",
            "Sub",
            "Low Mid",
            "Mid",
            "High Mid",
            "Presence",
            "Brilliance",
            "Air",
            "Treble Amplitude",
            "FFT Band 01",
            "FFT Band 02",
            "FFT Band 03",
            "FFT Band 04",
            "FFT Band 05",
            "FFT Band 06",
            "FFT Band 07",
            "FFT Band 08",
            "FFT Band 09",
            "FFT Band 10",
            "FFT Band 11",
            "FFT Band 12",
            "FFT Band 13",
            "FFT Band 14",
            "FFT Band 15",
            "FFT Band 16",
            "MIDI Note",
            "MIDI CC",
        ]
        
        try:
            targets = self._collect_targets_and_ranges()
        except Exception as e:
            import logging
            logging.error(f"Failed to collect modulation targets: {e}")
            targets = []

        # Rebuild on every open so the window can recover from hidden/deleted state
        # and pick up newly available node parameters.
        if dpg.does_item_exist(self.window_tag):
            dpg.delete_item(self.window_tag)
        self._reset_ui_state()
        
        if not targets:
            import logging
            logging.warning("No modulation targets found - add some effect nodes first")
            # Show empty window with helpful message
            with dpg.window(label="Modulation Matrix", width=500, height=200, tag=self.window_tag):
                dpg.add_text("No parameters available for modulation.", color=(255, 200, 100))
                dpg.add_text("Add some effect nodes to the pipeline first!")
                dpg.add_spacer(height=10)
                dpg.add_button(label="Close", callback=lambda: dpg.delete_item(self.window_tag), width=-1)
            return
        
        targets = targets  # Now guaranteed to have targets
        self.targets_child_tag = dpg.generate_uuid()

        with dpg.window(label="Modulation Matrix", width=640, height=460, tag=self.window_tag):
            import analog_clock
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Enable", default_value=self.enabled, callback=lambda s, a, u: setattr(self, 'enabled', bool(a)))
                
                # Add clock to the top right of the modulation matrix
                # Use a dummy spacer to push the clock to the right
                dpg.add_spacer(width=460)
                analog_clock.create_clock(parent=self.window_tag, radius=25, pos=[580, 30])

            def _apply_preset(sender, app_data):
                name = app_data
                self.connections = list(self.presets.get(name, []))
                if dpg.does_item_exist(self.conn_list_tag):
                    dpg.configure_item(self.conn_list_tag, items=self.connections)
                self._sync_checks_from_connections()

            dpg.add_combo(label="Preset", items=list(self.presets.keys()), default_value="None", callback=_apply_preset)

            with dpg.group(horizontal=True):
                with dpg.child_window(width=190, height=300, border=True):
                    dpg.add_text("Sources")
                    for s in sources:
                        self.source_checks[s] = dpg.add_checkbox(label=s, default_value=False)
                with dpg.child_window(width=260, height=300, border=True, tag=self.targets_child_tag):
                    dpg.add_text("Targets")
                    for t in targets:
                        self.target_checks[t] = dpg.add_checkbox(label=self.param_labels.get(t, t), default_value=False)
                with dpg.child_window(width=150, height=300, border=True):
                    dpg.add_text("Connections")
                    dpg.add_listbox(items=self.connections, tag=self.conn_list_tag, num_items=12)

            def _add_connections():
                sel_sources = [k for k, v in self.source_checks.items() if dpg.get_value(v)]
                sel_targets = [k for k, v in self.target_checks.items() if dpg.get_value(v)]
                for s in sel_sources:
                    for t in sel_targets:
                        conn = f"{s} -> {t}"
                        if conn not in self.connections:
                            self.connections.append(conn)
                dpg.configure_item(self.conn_list_tag, items=self.connections)

            def _remove_selected():
                try:
                    sel = dpg.get_value(self.conn_list_tag)
                except Exception:
                    sel = None
                if sel and sel in self.connections:
                    self.connections.remove(sel)
                    dpg.configure_item(self.conn_list_tag, items=self.connections)

            def _refresh_targets():
                try:
                    targets = self._collect_targets_and_ranges()
                    if dpg.does_item_exist(self.targets_child_tag):
                        dpg.delete_item(self.targets_child_tag, children_only=True)
                        dpg.add_text("Targets", parent=self.targets_child_tag)
                        self.target_checks.clear()
                        for t in targets:
                            self.target_checks[t] = dpg.add_checkbox(label=self.param_labels.get(t, t), default_value=False, parent=self.targets_child_tag)
                        self._sync_checks_from_connections()
                except Exception:
                    pass

            with dpg.group(horizontal=True):
                dpg.add_button(label="Add", callback=lambda: _add_connections())
                dpg.add_button(label="Remove", callback=lambda: _remove_selected())
                dpg.add_button(label="Refresh Targets", callback=lambda: _refresh_targets())

        if dpg.does_item_exist(self.conn_list_tag):
            dpg.configure_item(self.conn_list_tag, items=self.connections)
        self._sync_checks_from_connections()

    def set_providers(self, providers: Dict[str, Callable[[], Any]] | None):
        self.providers = providers or {}

    def apply(self, params: Dict[str, float], explicit_time: float = None, time_scale: float = 1.0) -> Dict[str, float]:
        """
        Apply modulation to parameters.
        
        Args:
            params: Parameter dictionary to modulate
            explicit_time: Optional explicit time value (for batch rendering)
                          If None, uses time.time() (for real-time rendering)
            time_scale: Time multiplier for modulation speed (default 1.0)
        """
        if not self.enabled or not self.connections:
            return params
        out = dict(params)
        now = explicit_time if explicit_time is not None else time.time()
        # Apply time scale for modulation speed control
        if not hasattr(self, '_time_base'):
            self._time_base = now
        scaled_time = self._time_base + (now - self._time_base) * time_scale

        def _call_provider(key: str, default: float = 0.0):
            provider = self.providers.get(key)
            if not callable(provider):
                return default
            try:
                return provider()  # may return arrays or floats
            except Exception:
                return default

        def source_norm(src: str) -> float:
            norm = 0.0
            def band_value(name: str) -> float:
                bands = _call_provider('fft_bands', {})
                try:
                    if isinstance(bands, dict):
                        return float(bands.get(name, 0.0))
                except Exception:
                    pass
                return 0.0

            if src == "Time":
                norm = (math.sin(scaled_time) + 1.0) * 0.5
            elif src == "Random":
                norm = random.random()
            elif src == "Beat Pulse":
                norm = float(_call_provider('bpm_pulse', 0.0))
            elif src == "Beat Trigger":
                norm = float(_call_provider('bpm_pulse', 0.0))
            elif src == "Beat Phase":
                norm = float(_call_provider('bpm_phase', 0.0))
            elif src == "BPM (Normalized)":
                norm = float(_call_provider('bpm_normalized', 0.0))
            elif src == "Onset Energy":
                norm = float(_call_provider('onset_value', 0.0))
            elif src == "Onset Trigger":
                norm = float(_call_provider('onset_value', 0.0))
            elif src in {"Sub", "Bass", "Low Mid", "Mid", "High Mid", "Presence", "Brilliance", "Air"}:
                norm = band_value(src)
            elif src.startswith("FFT Band "):
                norm = band_value(src.replace("FFT ", ""))
            elif src == "Bass Amplitude":
                norm = band_value("Bass")
            elif src == "Treble Amplitude":
                norm = max(band_value("Presence"), band_value("Brilliance"), band_value("Air"))
            elif src == "MIDI Note":
                try:
                    vel = float(_call_provider('midi_note', 0.0))
                    norm = min(1.0, max(0.0, vel / 127.0))
                except Exception:
                    norm = 0.0
            elif src == "MIDI CC":
                try:
                    cc = float(_call_provider('midi_cc', 0.0))
                    norm = min(1.0, max(0.0, cc / 127.0))
                except Exception:
                    norm = 0.0
            else:
                norm = 0.0
            return float(min(1.0, max(0.0, norm)))

        def mod_value(src: str, pmin: float, pmax: float) -> float:
            norm = source_norm(src)
            return float(pmin) + float(norm) * (float(pmax) - float(pmin))

        def event_id_for_source(src: str) -> int:
            if src == "Onset Trigger":
                return int(_call_provider('onset_event_id', 0))
            if src == "Beat Trigger":
                return int(_call_provider('beat_event_id', 0))
            return 0

        def discrete_value(src: str, tgt: str, current_value: Any) -> Any:
            choices = self.param_items.get(tgt) or []
            if not choices:
                return current_value
            event_id = event_id_for_source(src)
            conn_key = f"{src}->{tgt}"
            if event_id > 0:
                last_event, last_index = self._discrete_event_state.get(conn_key, (0, -1))
                if event_id != last_event:
                    next_index = (last_index + 1) % len(choices)
                    self._discrete_event_state[conn_key] = (event_id, next_index)
                    return choices[next_index]
                if 0 <= last_index < len(choices):
                    return choices[last_index]
                return current_value
            index = int(round(source_norm(src) * (len(choices) - 1)))
            index = max(0, min(len(choices) - 1, index))
            return choices[index]

        for c in list(self.connections):
            try:
                src, tgt = [x.strip() for x in c.split("->")]
            except Exception:
                continue
            if tgt not in out:
                continue
            pmin, pmax = self.param_ranges.get(tgt, (0.0, 1.0))
            if self.param_types.get(tgt) == "combo":
                out[tgt] = discrete_value(src, tgt, out.get(tgt))
                continue
            mod_mode = self.param_mod_modes.get(tgt, "replace")
            if mod_mode == "offset":
                norm = source_norm(src)
                span = float(self.param_mod_spans.get(tgt, max(0.0, (pmax - pmin) * 0.5)))
                try:
                    base_value = float(out[tgt])
                except Exception:
                    base_value = float(pmin)
                modulated = base_value + ((norm * 2.0) - 1.0) * span
                modulated = min(float(pmax), max(float(pmin), modulated))
                if self.param_types.get(tgt) == "int":
                    out[tgt] = int(round(modulated))
                else:
                    out[tgt] = modulated
            else:
                out[tgt] = mod_value(src, pmin, pmax)
        return out

    def refresh_targets_ui(self):
        try:
            targets = self._collect_targets_and_ranges()
            if dpg.does_item_exist(self.targets_child_tag):
                dpg.delete_item(self.targets_child_tag, children_only=True)
                dpg.add_text("Targets", parent=self.targets_child_tag)
                self.target_checks.clear()
                for t in targets:
                    self.target_checks[t] = dpg.add_checkbox(label=self.param_labels.get(t, t), default_value=False, parent=self.targets_child_tag)
        except Exception:
            pass
