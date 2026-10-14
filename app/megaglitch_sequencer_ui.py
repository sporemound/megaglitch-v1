from __future__ import annotations

import logging
from typing import Dict, List, Optional

import dearpygui.dearpygui as dpg

from lane_presets import LANE_PRESETS
from megaglitch_live_sequencer import LANE_MODES, MAX_CC_SLOTS, SCALES, STEP_DIVISIONS, LiveSequencer
from megaglitch_midi_router import MidiDeviceManager
from megaglitch_visual_bus import VISUAL_METRIC_NAMES, VisualBus


VISIBLE_METERS = (
    "brightness",
    "motion",
    "entropy",
    "edge_density",
    "glitch_spike",
    "center_mass_x",
    "center_mass_y",
)

MORPH_SLOT_COUNT = 16
MORPH_LANE_ITEMS = ["All Lanes"] + [f"Lane {index + 1}" for index in range(8)]
MORPH_AXIS_ITEMS = ["X", "Y"]
MORPH_TARGET_PARAMS = {
    "root_note": ("Root Note", 0.0, 127.0, True),
    "min_note": ("Min Note", 0.0, 127.0, True),
    "max_note": ("Max Note", 0.0, 127.0, True),
    "probability": ("Probability", 0.0, 1.0, False),
    "gate_length": ("Gate", 0.02, 1.0, False),
    "velocity_min": ("Velocity Min", 1.0, 127.0, True),
    "velocity_max": ("Velocity Max", 1.0, 127.0, True),
    "visual_amount": ("Visual Amount", 0.0, 1.0, False),
    "mutation_amount": ("Mutation", 0.0, 1.0, False),
}
MORPH_TARGET_ITEMS = [item[0] for item in MORPH_TARGET_PARAMS.values()]
MORPH_LABEL_TO_PARAM = {label: key for key, (label, _, _, _) in MORPH_TARGET_PARAMS.items()}


class LiveSequencerUI:
    def __init__(self, sequencer: LiveSequencer, visual_bus: VisualBus, midi_manager: MidiDeviceManager, hardware_controls: Optional[object] = None):
        self.sequencer = sequencer
        self.visual_bus = visual_bus
        self.midi_manager = midi_manager
        self.hardware_controls = hardware_controls
        self.metric_meter_tags: Dict[str, str] = {}
        self.metric_value_tags: Dict[str, str] = {}
        self.status_tag = "live_seq_status"
        self.hardware_status_tag = "live_seq_hardware_status"
        self.port_status_tag = "live_seq_port_status"
        self.enabled_tag = "live_seq_enabled_checkbox"
        self.lane_tags: List[Dict[str, str]] = []
        self._last_port_names: List[str] = []
        self.morph_x = 0.5
        self.morph_y = 0.5
        self.morph_pad_size = 160
        self.morph_pad_tag = None
        self.morph_knob_tag = None
        self.morph_x_slider_tag = None
        self.morph_y_slider_tag = None
        self.morph_value_tag = "live_seq_morph_xy_value"
        self.morph_slots: List[Dict[str, object]] = [self._default_morph_slot() for _ in range(MORPH_SLOT_COUNT)]
        self.morph_slot_tags: List[Dict[str, str]] = []

    def build(self, parent) -> None:
        with dpg.child_window(parent=parent, border=False, autosize_x=True, autosize_y=True):
            dpg.add_text("Global Clock")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Sequencer Enabled",
                    default_value=False,
                    tag=self.enabled_tag,
                    callback=lambda s, a: self.sequencer.update_global("enabled", a),
                )
                dpg.add_button(label="Play", callback=self._play_pressed, width=70)
                dpg.add_button(label="Stop", callback=lambda s=None, a=None, u=None: self.sequencer.pause(), width=70)
                dpg.add_button(label="Reset", callback=lambda s=None, a=None, u=None: self.sequencer.reset(), width=70)
                dpg.add_button(label="MIDI Panic", callback=lambda s=None, a=None, u=None: self.sequencer.panic_all(), width=110)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="BPM",
                    min_value=20.0,
                    max_value=300.0,
                    default_value=120.0,
                    width=220,
                    callback=lambda s, a: self.sequencer.update_global("bpm", a),
                )
                dpg.add_slider_float(
                    label="Swing",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=0.0,
                    width=180,
                    callback=lambda s, a: self.sequencer.update_global("swing", a),
                )
                dpg.add_slider_float(
                    label="Humanize ms",
                    min_value=0.0,
                    max_value=80.0,
                    default_value=0.0,
                    width=180,
                    callback=lambda s, a: self.sequencer.update_global("humanize_ms", a),
                )
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="Global Density",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=1.0,
                    width=220,
                    callback=lambda s, a: self.sequencer.update_global("global_density", a),
                )
                dpg.add_slider_float(
                    label="Global Mutation",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=0.0,
                    width=220,
                    callback=lambda s, a: self.sequencer.update_global("global_mutation", a),
                )
            dpg.add_text("Stopped", tag=self.status_tag, color=(170, 210, 255))
            dpg.add_text("", tag=self.hardware_status_tag, color=(160, 220, 170))
            dpg.add_separator()

            self._build_xy_morph_controls()
            dpg.add_separator()

            dpg.add_text("Visual Mod Sources")
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True):
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                for metric in VISIBLE_METERS:
                    with dpg.table_row():
                        dpg.add_text(metric)
                        self.metric_meter_tags[metric] = dpg.add_progress_bar(default_value=0.0, width=-1)
                        self.metric_value_tags[metric] = dpg.add_text("0.00")
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_text("Device Lanes")
                dpg.add_button(label="Refresh MIDI Ports", callback=self.refresh_ports)
                dpg.add_button(label="Test First Output", callback=self._test_first_output, width=130)
                dpg.add_button(label="Broadcast MIDI Test", callback=self._broadcast_midi_test, width=150)
                dpg.add_text("", tag=self.port_status_tag, color=(210, 190, 120))

            with dpg.tab_bar():
                for lane_index in range(8):
                    with dpg.tab(label=f"Lane {lane_index + 1}"):
                        self._build_lane(lane_index)
            self.refresh_ports()

    def _build_lane(self, lane_index: int) -> None:
        tags: Dict[str, str] = {}
        self.lane_tags.append(tags)
        lane = self.sequencer.get_lane_config(lane_index)

        with dpg.group(horizontal=True):
            tags["enabled"] = dpg.add_checkbox(
                label="Enabled",
                default_value=lane.enabled,
                callback=self._lane_value_changed,
                user_data=(lane_index, "enabled"),
            )
            dpg.add_button(label="Panic Lane", callback=self._panic_lane, user_data=lane_index)
            tags["preset"] = dpg.add_combo(
                label="Preset",
                items=list(LANE_PRESETS.keys()),
                default_value="Default Mono",
                width=180,
                callback=self._preset_changed,
                user_data=lane_index,
            )
        with dpg.group(horizontal=True):
            tags["output_port"] = dpg.add_combo(
                label="MIDI Output",
                items=["No MIDI outputs"],
                default_value="",
                width=360,
                callback=self._output_port_changed,
                user_data=lane_index,
            )
            dpg.add_button(label="Refresh", callback=self.refresh_ports)
            dpg.add_button(label="Test Note", callback=self._test_lane_note, user_data=lane_index, width=90)
            tags["midi_channel"] = dpg.add_slider_int(
                label="MIDI Channel",
                min_value=1,
                max_value=16,
                default_value=lane.midi_channel,
                width=180,
                callback=self._lane_value_changed,
                user_data=(lane_index, "midi_channel"),
            )

        with dpg.group(horizontal=True):
            tags["mode"] = dpg.add_combo(
                label="Mode",
                items=list(LANE_MODES),
                default_value=lane.mode,
                width=130,
                callback=self._lane_value_changed,
                user_data=(lane_index, "mode"),
            )
            tags["scale"] = dpg.add_combo(
                label="Scale",
                items=list(SCALES.keys()),
                default_value=lane.scale,
                width=150,
                callback=self._lane_value_changed,
                user_data=(lane_index, "scale"),
            )
            tags["step_division"] = dpg.add_combo(
                label="Step",
                items=list(STEP_DIVISIONS.keys()),
                default_value=lane.step_division,
                width=110,
                callback=self._lane_value_changed,
                user_data=(lane_index, "step_division"),
            )

        with dpg.group(horizontal=True):
            self._slider(tags, lane_index, "root_note", "Root Note", 0, 127, lane.root_note, int_slider=True)
            self._slider(tags, lane_index, "min_note", "Min Note", 0, 127, lane.min_note, int_slider=True)
            self._slider(tags, lane_index, "max_note", "Max Note", 0, 127, lane.max_note, int_slider=True)
        with dpg.group(horizontal=True):
            self._slider(tags, lane_index, "probability", "Probability", 0.0, 1.0, lane.probability)
            self._slider(tags, lane_index, "gate_length", "Gate", 0.02, 1.0, lane.gate_length)
            self._slider(tags, lane_index, "visual_amount", "Visual Amount", 0.0, 1.0, lane.visual_amount)
            self._slider(tags, lane_index, "mutation_amount", "Mutation", 0.0, 1.0, lane.mutation_amount)
        with dpg.group(horizontal=True):
            self._slider(tags, lane_index, "velocity_min", "Velocity Min", 1, 127, lane.velocity_min, int_slider=True)
            self._slider(tags, lane_index, "velocity_max", "Velocity Max", 1, 127, lane.velocity_max, int_slider=True)

        with dpg.group(horizontal=True):
            for field_name, label in (
                ("pitch_source", "Pitch Source"),
                ("velocity_source", "Velocity Source"),
                ("probability_source", "Probability Source"),
            ):
                tags[field_name] = dpg.add_combo(
                    label=label,
                    items=list(VISUAL_METRIC_NAMES),
                    default_value=getattr(lane, field_name),
                    width=180,
                    callback=self._lane_value_changed,
                    user_data=(lane_index, field_name),
                )

        dpg.add_separator()
        dpg.add_text("CC Mapping Slots")
        with dpg.table(header_row=True, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True, borders_outerV=True, borders_innerH=True):
            dpg.add_table_column(label="Slot")
            dpg.add_table_column(label="Enable")
            dpg.add_table_column(label="CC")
            dpg.add_table_column(label="Source")
            dpg.add_table_column(label="Send")
            for slot_index in range(MAX_CC_SLOTS):
                slot = lane.cc_slots[slot_index]
                with dpg.table_row():
                    dpg.add_text(str(slot_index + 1))
                    tags[f"cc_slot_{slot_index}_enabled"] = dpg.add_checkbox(
                        label="",
                        default_value=slot.enabled,
                        callback=self._cc_slot_changed,
                        user_data=(lane_index, slot_index, "enabled"),
                    )
                    tags[f"cc_slot_{slot_index}_number"] = dpg.add_slider_int(
                        label="",
                        min_value=0,
                        max_value=127,
                        default_value=slot.number,
                        width=120,
                        callback=self._cc_slot_changed,
                        user_data=(lane_index, slot_index, "number"),
                    )
                    tags[f"cc_slot_{slot_index}_source"] = dpg.add_combo(
                        label="",
                        items=list(VISUAL_METRIC_NAMES),
                        default_value=slot.source,
                        width=-1,
                        callback=self._cc_slot_changed,
                        user_data=(lane_index, slot_index, "source"),
                    )
                    dpg.add_button(
                        label="Send",
                        width=70,
                        callback=self._send_cc_slot,
                        user_data=(lane_index, slot_index),
                    )

    def _default_morph_slot(self) -> Dict[str, object]:
        label, min_value, max_value, _ = MORPH_TARGET_PARAMS["root_note"]
        return {
            "enabled": False,
            "lane": "All Lanes",
            "parameter": label,
            "axis": "X",
            "min": min_value,
            "max": max_value,
        }

    def _build_xy_morph_controls(self) -> None:
        dpg.add_text("XY Note Morph")
        with dpg.group(horizontal=True):
            with dpg.group():
                self.morph_pad_tag = dpg.generate_uuid()
                with dpg.drawlist(width=self.morph_pad_size, height=self.morph_pad_size, tag=self.morph_pad_tag):
                    dpg.draw_rectangle((0, 0), (self.morph_pad_size - 1, self.morph_pad_size - 1), color=(120, 150, 180, 255), thickness=1)
                    dpg.draw_line((self.morph_pad_size / 2, 0), (self.morph_pad_size / 2, self.morph_pad_size), color=(80, 95, 110, 255), thickness=1)
                    dpg.draw_line((0, self.morph_pad_size / 2), (self.morph_pad_size, self.morph_pad_size / 2), color=(80, 95, 110, 255), thickness=1)
                    self.morph_knob_tag = dpg.draw_circle((self.morph_pad_size / 2, self.morph_pad_size / 2), 7, color=(240, 240, 255, 255), fill=(80, 180, 255, 255))
                dpg.add_text("X 0.50  Y 0.50", tag=self.morph_value_tag, color=(170, 210, 255))
            with dpg.group():
                self.morph_x_slider_tag = dpg.add_slider_float(
                    label="X",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=self.morph_x,
                    width=260,
                    callback=self._morph_axis_changed,
                    user_data="X",
                )
                self.morph_y_slider_tag = dpg.add_slider_float(
                    label="Y",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=self.morph_y,
                    width=260,
                    callback=self._morph_axis_changed,
                    user_data="Y",
                )
                dpg.add_button(label="Center", callback=self._center_morph_xy, width=90)

        with dpg.table(header_row=True, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True, borders_outerV=True, borders_innerH=True):
            dpg.add_table_column(label="Slot")
            dpg.add_table_column(label="Enable")
            dpg.add_table_column(label="Lane")
            dpg.add_table_column(label="Parameter")
            dpg.add_table_column(label="Axis")
            dpg.add_table_column(label="Min")
            dpg.add_table_column(label="Max")
            for slot_index in range(MORPH_SLOT_COUNT):
                slot = self.morph_slots[slot_index]
                tags: Dict[str, str] = {}
                self.morph_slot_tags.append(tags)
                with dpg.table_row():
                    dpg.add_text(str(slot_index + 1))
                    tags["enabled"] = dpg.add_checkbox(
                        label="",
                        default_value=bool(slot["enabled"]),
                        callback=self._morph_slot_changed,
                        user_data=(slot_index, "enabled"),
                    )
                    tags["lane"] = dpg.add_combo(
                        label="",
                        items=MORPH_LANE_ITEMS,
                        default_value=str(slot["lane"]),
                        width=120,
                        callback=self._morph_slot_changed,
                        user_data=(slot_index, "lane"),
                    )
                    tags["parameter"] = dpg.add_combo(
                        label="",
                        items=MORPH_TARGET_ITEMS,
                        default_value=str(slot["parameter"]),
                        width=160,
                        callback=self._morph_slot_changed,
                        user_data=(slot_index, "parameter"),
                    )
                    tags["axis"] = dpg.add_combo(
                        label="",
                        items=MORPH_AXIS_ITEMS,
                        default_value=str(slot["axis"]),
                        width=70,
                        callback=self._morph_slot_changed,
                        user_data=(slot_index, "axis"),
                    )
                    tags["min"] = dpg.add_input_float(
                        label="",
                        default_value=float(slot["min"]),
                        width=90,
                        format="%.3f",
                        callback=self._morph_slot_changed,
                        user_data=(slot_index, "min"),
                    )
                    tags["max"] = dpg.add_input_float(
                        label="",
                        default_value=float(slot["max"]),
                        width=90,
                        format="%.3f",
                        callback=self._morph_slot_changed,
                        user_data=(slot_index, "max"),
                    )

    def _morph_axis_changed(self, sender=None, app_data=None, user_data=None) -> None:
        axis = str(user_data or "")
        if axis == "X":
            self.morph_x = max(0.0, min(1.0, float(app_data)))
        elif axis == "Y":
            self.morph_y = max(0.0, min(1.0, float(app_data)))
        self._update_morph_pad()
        self._apply_morph_slots()

    def _center_morph_xy(self, sender=None, app_data=None, user_data=None) -> None:
        self.morph_x = 0.5
        self.morph_y = 0.5
        if self.morph_x_slider_tag and dpg.does_item_exist(self.morph_x_slider_tag):
            dpg.set_value(self.morph_x_slider_tag, self.morph_x)
        if self.morph_y_slider_tag and dpg.does_item_exist(self.morph_y_slider_tag):
            dpg.set_value(self.morph_y_slider_tag, self.morph_y)
        self._update_morph_pad()
        self._apply_morph_slots()

    def _update_morph_pad(self) -> None:
        if self.morph_knob_tag and dpg.does_item_exist(self.morph_knob_tag):
            x = self.morph_x * self.morph_pad_size
            y = (1.0 - self.morph_y) * self.morph_pad_size
            dpg.configure_item(self.morph_knob_tag, center=(x, y))
        if dpg.does_item_exist(self.morph_value_tag):
            dpg.set_value(self.morph_value_tag, f"X {self.morph_x:.2f}  Y {self.morph_y:.2f}")

    def _morph_slot_changed(self, sender=None, app_data=None, user_data=None) -> None:
        if not user_data:
            return
        slot_index, field_name = user_data
        slot_index = int(slot_index)
        slot = self.morph_slots[slot_index]
        slot[str(field_name)] = app_data
        if field_name == "parameter":
            param_key = MORPH_LABEL_TO_PARAM.get(str(app_data), "root_note")
            _, min_value, max_value, _ = MORPH_TARGET_PARAMS[param_key]
            slot["min"] = min_value
            slot["max"] = max_value
            tags = self.morph_slot_tags[slot_index] if 0 <= slot_index < len(self.morph_slot_tags) else {}
            if dpg.does_item_exist(tags.get("min", "")):
                dpg.set_value(tags["min"], min_value)
            if dpg.does_item_exist(tags.get("max", "")):
                dpg.set_value(tags["max"], max_value)
        self._apply_morph_slots()

    def _apply_morph_slots(self) -> None:
        for slot in self.morph_slots:
            if not bool(slot.get("enabled", False)):
                continue
            param_key = MORPH_LABEL_TO_PARAM.get(str(slot.get("parameter")), "root_note")
            _, default_min, default_max, is_int = MORPH_TARGET_PARAMS[param_key]
            axis_value = self.morph_y if str(slot.get("axis")) == "Y" else self.morph_x
            try:
                min_value = float(slot.get("min", default_min))
                max_value = float(slot.get("max", default_max))
            except Exception:
                min_value = default_min
                max_value = default_max
            value = min_value + axis_value * (max_value - min_value)
            if is_int:
                value = int(round(value))
            targets = self._morph_target_lanes(str(slot.get("lane", "All Lanes")))
            for lane_index in targets:
                self.sequencer.update_lane(lane_index, param_key, value)
                self._sync_lane_control_value(lane_index, param_key, value)

    def _morph_target_lanes(self, lane_label: str) -> List[int]:
        if lane_label == "All Lanes":
            return list(range(8))
        if lane_label.startswith("Lane "):
            try:
                index = int(lane_label.split(" ", 1)[1]) - 1
            except Exception:
                return []
            return [index] if 0 <= index < 8 else []
        return []

    def _sync_lane_control_value(self, lane_index: int, field_name: str, value) -> None:
        if not 0 <= lane_index < len(self.lane_tags):
            return
        tag = self.lane_tags[lane_index].get(field_name)
        if tag and dpg.does_item_exist(tag):
            try:
                dpg.set_value(tag, value)
            except Exception:
                pass

    def _slider(self, tags, lane_index, field_name, label, min_value, max_value, default_value, int_slider=False):
        add_fn = dpg.add_slider_int if int_slider else dpg.add_slider_float
        tags[field_name] = add_fn(
            label=label,
            min_value=min_value,
            max_value=max_value,
            default_value=default_value,
            width=180,
            callback=self._lane_value_changed,
            user_data=(lane_index, field_name),
        )

    def _lane_value_changed(self, sender=None, app_data=None, user_data=None) -> None:
        if not user_data:
            return
        lane_index, field_name = user_data
        lane_index = int(lane_index)
        logging.info("Live sequencer lane %d %s changed to %r", lane_index + 1, field_name, app_data)
        self.sequencer.update_lane(lane_index, str(field_name), app_data)

    def _cc_slot_changed(self, sender=None, app_data=None, user_data=None) -> None:
        if not user_data:
            return
        lane_index, slot_index, field_name = user_data
        lane_index = int(lane_index)
        slot_index = int(slot_index)
        logging.info(
            "Live sequencer lane %d CC slot %d %s changed to %r",
            lane_index + 1,
            slot_index + 1,
            field_name,
            app_data,
        )
        self.sequencer.update_lane_cc_slot(lane_index, slot_index, str(field_name), app_data)

    def _send_cc_slot(self, sender=None, app_data=None, user_data=None) -> None:
        if not user_data:
            return
        lane_index, slot_index = user_data
        lane_index = int(lane_index)
        slot_index = int(slot_index)
        combo = self.lane_tags[lane_index].get("output_port") if 0 <= lane_index < len(self.lane_tags) else None
        if combo and dpg.does_item_exist(combo):
            selected_port = str(dpg.get_value(combo) or "")
            if selected_port and selected_port != "No MIDI outputs":
                self.sequencer.update_lane(lane_index, "output_port", selected_port)
        self.sequencer.send_cc_slot(lane_index, slot_index, value=127)
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, self.sequencer.status_message)

    def _preset_changed(self, sender=None, app_data=None, user_data=None) -> None:
        if user_data is None:
            return
        self._apply_preset(int(user_data), str(app_data or ""))

    def _panic_lane(self, sender=None, app_data=None, user_data=None) -> None:
        if user_data is None:
            return
        self.sequencer.panic_lane(int(user_data))

    def _apply_preset(self, lane_index: int, preset_name: str) -> None:
        preset = LANE_PRESETS.get(preset_name)
        if not preset:
            return
        self.sequencer.apply_lane_preset(lane_index, preset)
        self.sync_lane_controls(lane_index)

    def refresh_ports(self, sender=None, app_data=None, user_data=None) -> None:
        ports = self.midi_manager.refresh_ports()
        self._last_port_names = list(ports)
        items = ports if ports else ["No MIDI outputs"]
        if dpg.does_item_exist(self.port_status_tag):
            dpg.set_value(self.port_status_tag, f"{len(ports)} MIDI output(s)" if ports else "No MIDI outputs available")
        for lane_index, tags in enumerate(self.lane_tags):
            combo = tags.get("output_port")
            if combo and dpg.does_item_exist(combo):
                dpg.configure_item(combo, items=items)
                current = self.sequencer.get_lane_config(lane_index).output_port
                if current and current in ports:
                    dpg.set_value(combo, current)
                elif not ports:
                    dpg.set_value(combo, "")
        logging.info("Live sequencer MIDI ports refreshed: %d outputs: %s", len(ports), ports)

    def _test_refreshed_output(self, ports: List[str]) -> None:
        loop_ports = [port for port in ports if "loopmidi" in port.lower()]
        port_name = loop_ports[0] if loop_ports else (ports[0] if ports else "")
        logging.info("Live sequencer refresh-triggered output test: %r", port_name)
        self.sequencer.test_output_port(port_name)
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, self.sequencer.status_message)

    def _output_port_changed(self, sender=None, app_data=None, user_data=None) -> None:
        lane_index = int(user_data if user_data is not None else 0)
        selected_port = str(app_data or "")
        logging.info("Live sequencer lane %d output combo changed to %r", lane_index + 1, selected_port)
        self.sequencer.update_lane(lane_index, "output_port", selected_port)
        if selected_port and selected_port != "No MIDI outputs":
            self.sequencer.test_output_port(selected_port, channel=self.sequencer.get_lane_config(lane_index).midi_channel)
            if dpg.does_item_exist(self.status_tag):
                dpg.set_value(self.status_tag, self.sequencer.status_message)

    def _test_first_output(self, sender=None, app_data=None, user_data=None) -> None:
        ports = self.midi_manager.get_output_names() or self.midi_manager.refresh_ports()
        port_name = ports[0] if ports else ""
        logging.info("Live sequencer first-output test pressed: %r", port_name)
        self.sequencer.test_output_port(port_name)
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, self.sequencer.status_message)

    def _broadcast_midi_test(self, sender=None, app_data=None, user_data=None) -> None:
        ports = self.midi_manager.get_output_names() or self.midi_manager.refresh_ports()
        loop_ports = [port for port in ports if "loopmidi" in port.lower()]
        targets = loop_ports or ports
        logging.info("Live sequencer broadcast MIDI test pressed: %s", targets)
        sent_count = 0
        for port_name in targets:
            if self.sequencer.test_output_port(port_name, channel=1, note=60, velocity=120, duration=1.0):
                sent_count += 1
        self.sequencer.status_message = f"Broadcast test sent to {sent_count}/{len(targets)} MIDI output(s)"
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, self.sequencer.status_message)

    def _test_lane_note(self, sender=None, app_data=None, user_data=None) -> None:
        lane_index = int(user_data if user_data is not None else 0)
        logging.info("Live sequencer lane test button pressed: lane=%d", lane_index + 1)
        if 0 <= lane_index < len(self.lane_tags):
            combo = self.lane_tags[lane_index].get("output_port")
            if combo and dpg.does_item_exist(combo):
                selected_port = str(dpg.get_value(combo) or "")
                if selected_port and selected_port != "No MIDI outputs":
                    self.sequencer.update_lane(lane_index, "output_port", selected_port)
        self.sequencer.test_lane_note(lane_index)
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, self.sequencer.status_message)

    def _play_pressed(self, sender=None, app_data=None, user_data=None) -> None:
        self.sequencer.play()
        if dpg.does_item_exist(self.enabled_tag):
            dpg.set_value(self.enabled_tag, True)

    def sync_lane_controls(self, lane_index: int) -> None:
        if not 0 <= lane_index < len(self.lane_tags):
            return
        lane = self.sequencer.get_lane_config(lane_index)
        tags = self.lane_tags[lane_index]
        for field_name, tag in tags.items():
            if field_name == "preset":
                continue
            if field_name.startswith("cc_slot_"):
                parts = field_name.split("_")
                if len(parts) == 4:
                    slot_index = int(parts[2])
                    slot_field = parts[3]
                    if 0 <= slot_index < len(lane.cc_slots) and dpg.does_item_exist(tag):
                        try:
                            dpg.set_value(tag, getattr(lane.cc_slots[slot_index], slot_field))
                        except Exception:
                            pass
                continue
            if dpg.does_item_exist(tag) and hasattr(lane, field_name):
                try:
                    dpg.set_value(tag, getattr(lane, field_name))
                except Exception:
                    pass

    def update(self) -> None:
        metrics = self.visual_bus.get_latest()
        for name, tag in self.metric_meter_tags.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, float(metrics.get(name, 0.0)))
        for name, tag in self.metric_value_tags.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, f"{float(metrics.get(name, 0.0)):.2f}")
        if dpg.does_item_exist(self.status_tag):
            snapshot = self.sequencer.get_config_snapshot()
            state = self.sequencer.status_message
            if not snapshot.enabled:
                state = "Disabled"
            elif not snapshot.playing:
                state = "Stopped"
            dpg.set_value(self.status_tag, state)
        if self.hardware_controls and dpg.does_item_exist(self.hardware_status_tag):
            dpg.set_value(self.hardware_status_tag, getattr(self.hardware_controls, "status_message", ""))


def build_live_sequencer_ui(parent, sequencer: LiveSequencer, visual_bus: VisualBus, midi_manager: MidiDeviceManager, hardware_controls: Optional[object] = None) -> LiveSequencerUI:
    ui = LiveSequencerUI(sequencer, visual_bus, midi_manager, hardware_controls=hardware_controls)
    ui.build(parent)
    return ui
