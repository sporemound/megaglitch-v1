
import dearpygui.dearpygui as dpg
import numpy as np
import effect_nodes
import logging
from pathlib import Path
import json

from megaglitch.audio.modulation_bus import ModulationBus

class SourceNode(effect_nodes.Node):
    def __init__(self, parent_node_editor):
        super().__init__("Source", parent_node_editor)

    def add_attributes(self):
        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Image Out")
            self.output_attrs["image_out"] = attr

class SecondarySourceNode(effect_nodes.Node):
    def __init__(self, parent_node_editor):
        super().__init__("Source B", parent_node_editor)

    def add_attributes(self):
        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Channel B Out")
            self.output_attrs["image_out"] = attr

class DisplayNode(effect_nodes.Node):
    def __init__(self, parent_node_editor):
        super().__init__("Display", parent_node_editor)
        self.texture_tag = dpg.generate_uuid()
        self.display_width = 800
        self.display_height = 600
        self.display_image_item = None

    def add_attributes(self):
        with dpg.node_attribute(label="Input", attribute_type=dpg.mvNode_Attr_Input, tag=dpg.generate_uuid()) as attr:
            dpg.add_text("Image In")
            self.input_attrs["image_in"] = attr

class NodeEditor:
    _zoom_fonts = {}

    def __init__(self, primary_source_key="primary"):
        self.nodes = []
        self.links = []
        self.primary_source_key = (
            "secondary" if str(primary_source_key).lower() == "secondary" else "primary"
        )
        self.node_editor_tag = dpg.generate_uuid()
        self.modulation_bus = ModulationBus()
        effect_nodes.register_audiovisual_modulation_bus(self.node_editor_tag, self.modulation_bus)
        self._load_default_audiovisual_mappings()
        self._right_pan_handler_tag = dpg.generate_uuid()
        self._right_pan_release_handler_tag = dpg.generate_uuid()
        self._wheel_zoom_handler_tag = dpg.generate_uuid()
        self._right_pan_handler_registry = None
        self._right_pan_last_mouse_pos = None
        self.zoom_level = 1.0
        self.min_zoom_level = 0.5
        self.max_zoom_level = 2.0
        self._zoom_item_dimensions = {}
        self._prepare_zoom_fonts()
        self.reset_queue = None  # Will be set by main app
        self.verbose_pipeline_logging = False
        self._pipeline_dirty = True
        self._cached_pipeline = []
        self._ensure_presets_directory()

    def mark_pipeline_dirty(self):
        self._pipeline_dirty = True

    def _copy_pipeline_with_current_params(self, pipeline):
        node_by_id = {getattr(node, "node_tag", None): node for node in self.nodes}
        copied = []
        for step in pipeline:
            step_copy = dict(step)
            node_id = step_copy.get("node_id")
            node = node_by_id.get(node_id)
            if node is not None and hasattr(node, "param_values"):
                step_copy["params"] = dict(getattr(node, "param_values", {}) or {})
            else:
                step_copy["params"] = dict(step_copy.get("params", {}) or {})
            copied.append(step_copy)
        return copied
    
    def _ensure_presets_directory(self):
        """Ensure the node_presets directory exists."""
        import os
        if not os.path.exists("node_presets"):
            os.makedirs("node_presets")

    def _load_default_audiovisual_mappings(self):
        path = Path(__file__).parent / "shows" / "one_hour_audio_visual_mappings.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for mapping in payload.get("mappings", []):
                self.modulation_bus.add_mapping(mapping)
        except Exception:
            logging.exception("Could not load audiovisual modulation mappings")

    def create_node_editor(self, parent_window):
        with dpg.node_editor(
            callback=self.link_callback, 
            delink_callback=self.delink_callback, 
            tag=self.node_editor_tag, 
            parent=parent_window,
            minimap=True,  # Enable minimap for navigation
            minimap_location=dpg.mvNodeMiniMap_Location_BottomRight  # Position minimap
        ):
            source_node = SourceNode(self.node_editor_tag)
            source_node.create_node(pos=[50, 50])
            self.nodes.append(source_node)

            display_node = DisplayNode(self.node_editor_tag)
            display_node.create_node(pos=[600, 50])
            self.nodes.append(display_node)

        self._install_right_click_pan_handlers()

    def _install_right_click_pan_handlers(self):
        if self._right_pan_handler_registry is not None:
            return

        with dpg.handler_registry() as registry:
            self._right_pan_handler_registry = registry
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Right,
                threshold=0.0,
                callback=self._right_click_pan_callback,
                tag=self._right_pan_handler_tag,
            )
            dpg.add_mouse_release_handler(
                button=dpg.mvMouseButton_Right,
                callback=self._right_click_pan_release_callback,
                tag=self._right_pan_release_handler_tag,
            )
            if hasattr(dpg, "add_mouse_wheel_handler"):
                dpg.add_mouse_wheel_handler(
                    callback=self._mouse_wheel_zoom_callback,
                    tag=self._wheel_zoom_handler_tag,
                )
            else:
                logging.warning("DearPyGui mouse wheel handler unavailable; node editor wheel zoom disabled.")

    def _right_click_pan_release_callback(self, sender=None, app_data=None, user_data=None):
        self._right_pan_last_mouse_pos = None

    def _mouse_wheel_zoom_callback(self, sender=None, app_data=None, user_data=None):
        if not self._is_editor_navigation_hovered():
            return

        try:
            wheel_delta = float(app_data)
        except (TypeError, ValueError):
            wheel_delta = 0.0
        if abs(wheel_delta) < 0.01:
            return

        step = 1.10 if wheel_delta > 0.0 else 1.0 / 1.10
        old_zoom = float(self.zoom_level)
        new_zoom = float(np.clip(old_zoom * step, self.min_zoom_level, self.max_zoom_level))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        factor = new_zoom / max(old_zoom, 1e-6)
        pivot = self._zoom_pivot()
        self._zoom_all_nodes(factor, pivot)
        self.zoom_level = new_zoom

    def _right_click_pan_callback(self, sender=None, app_data=None, user_data=None):
        if not self._is_editor_navigation_hovered():
            self._right_pan_last_mouse_pos = None
            return

        try:
            mouse_pos = dpg.get_mouse_pos(local=False)
        except Exception:
            mouse_pos = dpg.get_mouse_pos()

        if not mouse_pos or len(mouse_pos) < 2:
            self._right_pan_last_mouse_pos = None
            return

        current = (float(mouse_pos[0]), float(mouse_pos[1]))
        if self._right_pan_last_mouse_pos is None:
            self._right_pan_last_mouse_pos = current
            return

        dx = current[0] - self._right_pan_last_mouse_pos[0]
        dy = current[1] - self._right_pan_last_mouse_pos[1]
        self._right_pan_last_mouse_pos = current

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return

        self._pan_all_nodes(dx, dy)

    def _is_editor_navigation_hovered(self):
        try:
            if dpg.is_item_hovered(self.node_editor_tag):
                return True
        except Exception:
            pass

        # Some DearPyGui versions only report the child node as hovered, not
        # the node editor itself. Treat hovering a node as canvas navigation too.
        for node in self.nodes:
            try:
                if dpg.is_item_hovered(node.node_tag):
                    return True
            except Exception:
                pass
        return False

    def _zoom_pivot(self):
        # Node positions are expressed in node-editor canvas coordinates.
        # Mouse and item-rectangle positions are viewport coordinates, so using
        # either as a pivot causes the whole graph to drift diagonally.
        centers = []
        for node in self.nodes:
            try:
                pos = dpg.get_item_pos(node.node_tag)
                if pos is None or len(pos) < 2:
                    continue
                size = dpg.get_item_rect_size(node.node_tag)
                width = float(size[0]) if size and len(size) >= 2 else 0.0
                height = float(size[1]) if size and len(size) >= 2 else 0.0
                centers.append((float(pos[0]) + width * 0.5, float(pos[1]) + height * 0.5))
            except Exception:
                continue

        if not centers:
            return (0.0, 0.0)

        return (
            sum(center[0] for center in centers) / len(centers),
            sum(center[1] for center in centers) / len(centers),
        )

    def _zoom_all_nodes(self, factor, pivot):
        px, py = float(pivot[0]), float(pivot[1])
        target_zoom = float(self.zoom_level * factor)
        for node in self.nodes:
            try:
                pos = dpg.get_item_pos(node.node_tag)
                if pos is None or len(pos) < 2:
                    continue
                size = dpg.get_item_rect_size(node.node_tag)
                width = float(size[0]) if size and len(size) >= 2 else 0.0
                height = float(size[1]) if size and len(size) >= 2 else 0.0
                center_x = float(pos[0]) + width * 0.5
                center_y = float(pos[1]) + height * 0.5
                scaled_center_x = px + (center_x - px) * factor
                scaled_center_y = py + (center_y - py) * factor
                expected_width = width * factor
                expected_height = height * factor
                x = scaled_center_x - expected_width * 0.5
                y = scaled_center_y - expected_height * 0.5
                dpg.set_item_pos(node.node_tag, [x, y])
                self._scale_node_contents(node.node_tag, target_zoom)
            except Exception as e:
                logging.debug(f"Mouse-wheel zoom skipped node {getattr(node, 'label', '?')}: {e}")

    @classmethod
    def _prepare_zoom_fonts(cls):
        if cls._zoom_fonts:
            return

        font_path = Path(__file__).parent / "fonts" / "LazenbyCompSmooth.ttf"
        if not font_path.exists():
            return

        try:
            with dpg.font_registry():
                for font_size in range(8, 33, 2):
                    cls._zoom_fonts[font_size] = dpg.add_font(str(font_path), font_size)
        except Exception as e:
            cls._zoom_fonts.clear()
            logging.debug(f"Could not prepare node zoom fonts: {e}")

    @classmethod
    def _get_zoom_font(cls, zoom_level):
        if not cls._zoom_fonts:
            return None

        requested_size = float(np.clip(16.0 * zoom_level, 8, 32))
        font_size = min(cls._zoom_fonts, key=lambda size: abs(size - requested_size))
        return cls._zoom_fonts[font_size]

    @staticmethod
    def _walk_item_tree(root):
        pending = [root]
        seen = set()
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            yield item
            for slot in range(4):
                try:
                    pending.extend(dpg.get_item_children(item, slot=slot) or [])
                except Exception:
                    continue

    def _scale_node_contents(self, node_tag, zoom_level):
        font = self._get_zoom_font(zoom_level)
        for item in self._walk_item_tree(node_tag):
            try:
                if font is not None:
                    dpg.bind_item_font(item, font)
            except Exception:
                pass

            try:
                config = dpg.get_item_configuration(item) or {}
            except Exception:
                continue

            baseline = self._zoom_item_dimensions.get(item)
            if baseline is None:
                current_width = float(config.get("width", 0) or 0)
                current_height = float(config.get("height", 0) or 0)
                baseline = {
                    "width": current_width / max(self.zoom_level, 1e-6),
                    "height": current_height / max(self.zoom_level, 1e-6),
                }
                self._zoom_item_dimensions[item] = baseline

            updates = {}
            if baseline["width"] > 0:
                updates["width"] = max(1, int(round(baseline["width"] * zoom_level)))
            if baseline["height"] > 0:
                updates["height"] = max(1, int(round(baseline["height"] * zoom_level)))
            if updates:
                try:
                    dpg.configure_item(item, **updates)
                except Exception:
                    pass

    def _pan_all_nodes(self, dx, dy):
        for node in self.nodes:
            try:
                pos = dpg.get_item_pos(node.node_tag)
                if pos is None or len(pos) < 2:
                    continue
                dpg.set_item_pos(node.node_tag, [float(pos[0]) + dx, float(pos[1]) + dy])
            except Exception as e:
                logging.debug(f"Right-click pan skipped node {getattr(node, 'label', '?')}: {e}")

    def link_callback(self, sender, app_data):
        # app_data: (attr_1, attr_2)
        new_out, new_in = app_data[0], app_data[1]
        import logging
        logging.info(f"LINK CREATED: output_attr={new_out} -> input_attr={new_in}")
        
        # Find which nodes these belong to
        for node in self.nodes:
            if new_out in node.output_attrs.values():
                logging.info(f"  Source: {node.label}")
            if new_in in node.input_attrs.values():
                logging.info(f"  Target: {node.label}")
        
        # Ensure only one link into an input attribute: remove existing link targeting new_in
        to_remove = []
        for link_tag in list(self.links):
            cfg = dpg.get_item_configuration(link_tag)
            if cfg.get("attr_2") == new_in:
                to_remove.append(link_tag)
        for link_tag in to_remove:
            try:
                dpg.delete_item(link_tag)
            except Exception:
                pass
            if link_tag in self.links:
                self.links.remove(link_tag)

        link = dpg.add_node_link(new_out, new_in, parent=sender)
        self.links.append(link)
        self.mark_pipeline_dirty()
        logging.info(f"  Link tag={link}, Total links: {len(self.links)}")
        
        # Reset worker state when pipeline changes
        if self.reset_queue is not None:
            try:
                self.reset_queue.put("RESET")
                logging.debug("Worker state reset triggered by link")
            except Exception as e:
                logging.error(f"Failed to reset worker: {e}")

    def delink_callback(self, sender, app_data):
        dpg.delete_item(app_data)
        self.links.remove(app_data)
        self.mark_pipeline_dirty()
        
        # Reset worker state when pipeline changes
        if self.reset_queue is not None:
            try:
                self.reset_queue.put("RESET")
                import logging
                logging.debug("Worker state reset triggered by delink")
            except Exception as e:
                import logging
                logging.error(f"Failed to reset worker: {e}")

    def clear_all_links(self):
        for link_tag in list(self.links):
            try:
                dpg.delete_item(link_tag)
            except Exception:
                pass
        self.links.clear()
        self.mark_pipeline_dirty()

    def delete_selected_links(self):
        try:
            selected = dpg.get_selected_links(self.node_editor_tag)
        except Exception:
            selected = []
        for link_tag in selected:
            try:
                dpg.delete_item(link_tag)
            except Exception:
                pass
            if link_tag in self.links:
                self.links.remove(link_tag)
                self.mark_pipeline_dirty()
    
    def delete_selected_node(self):
        """Delete the currently selected node(s) from the node editor."""
        try:
            selected_nodes = dpg.get_selected_nodes(self.node_editor_tag)
        except Exception:
            selected_nodes = []
        
        if not selected_nodes:
            logging.info("No nodes selected for deletion")
            return
        
        for node_tag in selected_nodes:
            # Find the node object
            node_to_delete = None
            for node in self.nodes:
                if node.node_tag == node_tag:
                    node_to_delete = node
                    break
            
            if node_to_delete:
                # Don't allow deleting Source or Display nodes
                if isinstance(node_to_delete, SourceNode):
                    logging.warning("Cannot delete Source node")
                    continue
                if isinstance(node_to_delete, DisplayNode):
                    logging.warning("Cannot delete Display node")
                    continue
                
                # Delete all links connected to this node
                links_to_remove = []
                for link_tag in self.links:
                    try:
                        link_data = dpg.get_item_configuration(link_tag)
                        attr_1 = link_data.get("attr_1")
                        attr_2 = link_data.get("attr_2")
                        
                        # Check if this link is connected to the node being deleted
                        if (attr_1 in node_to_delete.input_attrs.values() or 
                            attr_1 in node_to_delete.output_attrs.values() or
                            attr_2 in node_to_delete.input_attrs.values() or 
                            attr_2 in node_to_delete.output_attrs.values()):
                            links_to_remove.append(link_tag)
                    except Exception:
                        pass
                
                # Remove the links
                for link_tag in links_to_remove:
                    try:
                        dpg.delete_item(link_tag)
                    except Exception:
                        pass
                    if link_tag in self.links:
                        self.links.remove(link_tag)
                
                # Delete the node from DearPyGui
                try:
                    dpg.delete_item(node_tag)
                except Exception:
                    pass

                try:
                    destroy = getattr(node_to_delete, "destroy", None)
                    if callable(destroy):
                        destroy()
                except Exception:
                    logging.exception("Failed to destroy node: %s", getattr(node_to_delete, "label", "?"))
                
                # Remove from our node list
                self.nodes.remove(node_to_delete)
                self.mark_pipeline_dirty()
                logging.info(f"Deleted node: {node_to_delete.label}")
                
                # Trigger state reset
                if self.reset_queue:
                    self.reset_queue.put("RESET")

    def copy_selected_node(self):
        """Duplicate selected node(s), preserving parameter values but not links."""
        try:
            selected_nodes = dpg.get_selected_nodes(self.node_editor_tag)
        except Exception:
            selected_nodes = []

        if not selected_nodes:
            logging.info("No nodes selected for copy")
            return []

        copied_nodes = []
        for node_tag in selected_nodes:
            node_to_copy = None
            for node in self.nodes:
                if node.node_tag == node_tag:
                    node_to_copy = node
                    break

            if node_to_copy is None:
                continue

            if isinstance(node_to_copy, (SourceNode, SecondarySourceNode, DisplayNode)):
                logging.warning(f"Cannot copy node: {node_to_copy.label}")
                continue

            try:
                new_node = type(node_to_copy)(self.node_editor_tag)
            except Exception as e:
                logging.error(f"Failed to copy node {getattr(node_to_copy, 'label', '?')}: {e}")
                continue

            try:
                pos = dpg.get_item_pos(node_to_copy.node_tag)
            except Exception:
                pos = [200, 150]
            if not pos or len(pos) < 2:
                pos = [200, 150]

            new_node.create_node(pos=[float(pos[0]) + 40.0, float(pos[1]) + 40.0])

            source_params = dict(getattr(node_to_copy, "param_values", {}) or {})
            self._restore_node_params(new_node, source_params)

            self.nodes.append(new_node)
            copied_nodes.append(new_node)
            logging.info(f"Copied node: {node_to_copy.label}")

        if copied_nodes:
            self.mark_pipeline_dirty()
            if self.reset_queue:
                try:
                    self.reset_queue.put("RESET")
                except Exception as e:
                    logging.error(f"Failed to reset worker after copy: {e}")

        return copied_nodes

    def _restore_node_params(self, node, params):
        """Best-effort parameter restore for regular, generator, and Lua shader nodes."""
        if not hasattr(node, "param_values"):
            return

        # Apply path/preset fields first because LuaShaderNode rebuilds dynamic controls
        # when these change.
        priority = ["shader_preset", "shader_path", "lua_path", "metadata_path"]
        ordered_names = priority + [name for name in params.keys() if name not in priority]

        for param_name in ordered_names:
            if param_name not in params:
                continue
            value = params[param_name]
            try:
                if hasattr(node, "_set_param_value"):
                    node._set_param_value(param_name, value, update_control=True)
                else:
                    node.param_values[param_name] = value
                    control_tag = getattr(node, "param_controls", {}).get(param_name)
                    if control_tag:
                        try:
                            dpg.set_value(control_tag, value)
                        except Exception:
                            pass
            except Exception as e:
                logging.debug(f"Could not restore copied param {param_name}: {e}")

    def add_node(self, node_type, params=None):
        if node_type == "SecondarySourceNode":
            return self.add_secondary_source_node()

        # Try to import from effect_nodes first, then effect_nodes_cuda
        try:
            node = getattr(effect_nodes, node_type)(self.node_editor_tag)
        except AttributeError:
            try:
                import effect_nodes_cuda
                node = getattr(effect_nodes_cuda, node_type)(self.node_editor_tag)
            except (ImportError, AttributeError) as e:
                import logging
                logging.error(f"Failed to create node {node_type}: {e}")
                return None
        
        node.create_node(pos=[200, 150])
        self.nodes.append(node)
        self.mark_pipeline_dirty()
        return node

    def add_secondary_source_node(self):
        existing = self.get_secondary_source_node()
        if existing:
            logging.info("Source B already exists")
            return existing
        node = SecondarySourceNode(self.node_editor_tag)
        node.create_node(pos=[50, 220])
        self.nodes.append(node)
        self.mark_pipeline_dirty()
        return node

    def _build_attr_maps(self):
        attr_to_node_map = {}
        incoming_links = {}

        for node in self.nodes:
            for attr_tag in node.input_attrs.values():
                attr_to_node_map[attr_tag] = node
            for attr_tag in node.output_attrs.values():
                attr_to_node_map[attr_tag] = node

        for link_tag in self.links:
            try:
                cfg = dpg.get_item_configuration(link_tag)
                out_attr = cfg.get("attr_1")
                in_attr = cfg.get("attr_2")
                if out_attr is not None and in_attr is not None:
                    incoming_links[in_attr] = out_attr
            except Exception:
                pass

        return attr_to_node_map, incoming_links

    def _compile_node_from_input(self, input_attr, attr_to_node_map, incoming_links, visiting=None):
        if input_attr is None:
            return None
        if visiting is None:
            visiting = set()

        output_attr = incoming_links.get(input_attr)
        if output_attr is None:
            return None

        parent_node = attr_to_node_map.get(output_attr)
        if parent_node is None:
            return None

        selected_output = None
        for output_name, attr_tag in parent_node.output_attrs.items():
            if attr_tag == output_attr:
                selected_output = output_name
                break

        node_key = getattr(parent_node, "node_tag", None)
        if node_key in visiting:
            logging.warning("Cycle detected while compiling processing graph")
            return None

        next_visiting = set(visiting)
        if node_key is not None:
            next_visiting.add(node_key)

        if isinstance(parent_node, SourceNode):
            return {
                "node_type": "source",
                "source_key": self.primary_source_key,
                "label": parent_node.label,
            }

        if isinstance(parent_node, SecondarySourceNode):
            return {"node_type": "source", "source_key": "secondary", "label": parent_node.label}

        if isinstance(parent_node, effect_nodes.GeneratorNode):
            generator_inputs = {}
            input_order = []
            for input_name, attr_tag in parent_node.input_attrs.items():
                input_order.append(input_name)
                generator_inputs[input_name] = self._compile_node_from_input(attr_tag, attr_to_node_map, incoming_links, next_visiting)
            return {
                "node_type": "generator",
                "effect_type": parent_node.label,
                "params": dict(parent_node.param_values),
                "is_generator": True,
                "node_id": getattr(parent_node, "node_tag", None),
                "selected_output": selected_output or "image_out",
                "inputs": generator_inputs,
                "input_order": input_order,
            }

        if isinstance(parent_node, effect_nodes.MultiInputNode):
            node_inputs = {}
            input_order = []
            for input_name, attr_tag in parent_node.input_attrs.items():
                input_order.append(input_name)
                node_inputs[input_name] = self._compile_node_from_input(attr_tag, attr_to_node_map, incoming_links, next_visiting)
            return {
                "node_type": getattr(parent_node, "node_type_key", "multi_input"),
                "effect_type": parent_node.label,
                "params": dict(parent_node.param_values),
                "node_id": getattr(parent_node, "node_tag", None),
                "inputs": node_inputs,
                "input_order": input_order,
            }

        if isinstance(parent_node, effect_nodes.VideoMixerNode):
            return {
                "node_type": "mixer",
                "effect_type": parent_node.label,
                "params": dict(parent_node.param_values),
                "input_a": self._compile_node_from_input(parent_node.input_attrs.get("image_a"), attr_to_node_map, incoming_links, next_visiting),
                "input_b": self._compile_node_from_input(parent_node.input_attrs.get("image_b"), attr_to_node_map, incoming_links, next_visiting),
            }

        if isinstance(parent_node, effect_nodes.LuaShaderNode):
            node_inputs = {}
            input_order = []
            for input_name, attr_tag in parent_node.input_attrs.items():
                input_order.append(input_name)
                node_inputs[input_name] = self._compile_node_from_input(
                    attr_tag, attr_to_node_map, incoming_links, next_visiting
                )
            return {
                "node_type": "lua_multi_input",
                "effect_type": parent_node.label,
                "params": dict(parent_node.param_values),
                "node_id": getattr(parent_node, "node_tag", None),
                "inputs": node_inputs,
                "input_order": input_order,
            }

        if isinstance(parent_node, effect_nodes.GlitchEffectNode):
            input_attr_tag = parent_node.input_attrs.get("image_in")
            if input_attr_tag is None:
                input_attr_tag = next(iter(parent_node.input_attrs.values()), None)
            return {
                "node_type": "effect",
                "effect_type": parent_node.label,
                "params": dict(parent_node.param_values),
                "node_id": getattr(parent_node, "node_tag", None),
                "input": self._compile_node_from_input(
                    input_attr_tag,
                    attr_to_node_map,
                    incoming_links,
                    next_visiting,
                ),
            }

        logging.warning(f"Unsupported node type in processing graph: {type(parent_node)}")
        return None

    def get_processing_graph(self):
        display_node = self.get_display_node()
        if not display_node:
            return None

        display_input = next(iter(display_node.input_attrs.values()), None)
        if display_input is None:
            return None

        attr_to_node_map, incoming_links = self._build_attr_maps()
        graph = self._compile_node_from_input(display_input, attr_to_node_map, incoming_links)
        if self.verbose_pipeline_logging:
            logging.debug(f"Processing graph: {graph}")
        return graph

    def _flatten_processing_graph(self, graph, pipeline):
        if not graph:
            return

        node_type = graph.get("node_type")
        if node_type == "source":
            pipeline.append({
                "effect_type": graph.get("label", "Source"),
                "params": {},
                "node_type": node_type,
                "source_key": graph.get("source_key", "primary"),
            })
            return

        if node_type == "generator":
            connected_input_order = []
            generator_inputs = dict(graph.get("inputs", {}) or {})
            for input_name in graph.get("input_order", []) or []:
                input_graph = generator_inputs.get(input_name)
                if input_graph:
                    self._flatten_processing_graph(input_graph, pipeline)
                    connected_input_order.append(input_name)
            pipeline.append({
                "effect_type": graph.get("effect_type", "Generator"),
                "params": dict(graph.get("params", {}) or {}),
                "is_generator": True,
                "node_type": node_type,
                "node_id": graph.get("node_id"),
                "selected_output": graph.get("selected_output", "image_out"),
                "input_order": connected_input_order,
            })
            return

        if node_type in {"genesis_priority_composite", "genesis_mask_key", "lua_multi_input"}:
            connected_input_order = []
            node_inputs = dict(graph.get("inputs", {}) or {})
            for input_name in graph.get("input_order", []) or []:
                input_graph = node_inputs.get(input_name)
                if input_graph:
                    self._flatten_processing_graph(input_graph, pipeline)
                    connected_input_order.append(input_name)
            pipeline.append({
                "effect_type": graph.get("effect_type", "Multi Input"),
                "params": dict(graph.get("params", {}) or {}),
                "node_type": node_type,
                "node_id": graph.get("node_id"),
                "input_order": connected_input_order,
            })
            return

        if node_type == "effect":
            self._flatten_processing_graph(graph.get("input"), pipeline)
            pipeline.append({
                "effect_type": graph.get("effect_type", "Unknown"),
                "params": dict(graph.get("params", {}) or {}),
                "node_type": node_type,
                "node_id": graph.get("node_id"),
            })
            return

        if node_type == "mixer":
            self._flatten_processing_graph(graph.get("input_a"), pipeline)
            self._flatten_processing_graph(graph.get("input_b"), pipeline)
            pipeline.append({
                "effect_type": graph.get("effect_type", "Video Mixer"),
                "params": dict(graph.get("params", {}) or {}),
                "node_type": node_type,
            })

    def get_processing_pipeline(self):
        if not self._pipeline_dirty:
            return self._copy_pipeline_with_current_params(self._cached_pipeline)

        pipeline = []

        graph = self.get_processing_graph()
        self._flatten_processing_graph(graph, pipeline)
        if pipeline:
            pipeline.append({"effect_type": "Display", "params": {}, "node_type": "display"})

        if self.verbose_pipeline_logging:
            logging.debug(f"Pipeline: {pipeline}")
        self._cached_pipeline = self._copy_pipeline_with_current_params(pipeline)
        self._pipeline_dirty = False
        return self._copy_pipeline_with_current_params(self._cached_pipeline)

    def get_source_node(self):
        for node in self.nodes:
            if isinstance(node, SourceNode):
                return node
        return None

    def get_secondary_source_node(self):
        for node in self.nodes:
            if isinstance(node, SecondarySourceNode):
                return node
        return None

    def get_display_node(self):
        for node in self.nodes:
            if isinstance(node, DisplayNode):
                return node
        return None

    def get_node_params(self):
        params = {}
        for node in self.nodes:
            if isinstance(node, (effect_nodes.GlitchEffectNode, effect_nodes.MultiInputNode, effect_nodes.GeneratorNode)):
                params[node.node_tag] = node.param_values
        return params

    def update_audio_nodes(self, time_value, dt, slot_time=None):
        node_aliases = {}
        for node in tuple(self.nodes):
            alias = self._modulation_node_alias(node)
            node_aliases[alias] = node
            for name, value in dict(getattr(node, "param_values", {}) or {}).items():
                if isinstance(value, (int, float, bool)):
                    self.modulation_bus.publish(f"{alias}.{name}", float(value))
            get_outputs = getattr(node, "get_analysis_outputs", None)
            if callable(get_outputs):
                for name, value in get_outputs().items():
                    self.modulation_bus.publish(f"{alias}.{name}", float(value))

        routed = self.modulation_bus.update(dt)
        for target, value in routed.items():
            node_name, separator, param_name = str(target).partition(".")
            if not separator:
                continue
            node = node_aliases.get(node_name)
            if node is None or param_name not in getattr(node, "param_values", {}):
                continue
            try:
                node.update_param(param_name, value)
                control = getattr(node, "param_controls", {}).get(param_name)
                if control and dpg.does_item_exist(control):
                    dpg.set_value(control, value)
            except Exception:
                logging.exception("Could not route modulation target: %s", target)

        outputs = {}
        for node in tuple(self.nodes):
            update_audio = getattr(node, "update_audio", None)
            if not callable(update_audio):
                continue
            try:
                outputs[node.node_tag] = update_audio(time_value, dt, slot_time)
            except Exception:
                logging.exception("Audio node update failed: %s", getattr(node, "label", "?"))
        return outputs

    @staticmethod
    def _modulation_node_alias(node):
        if isinstance(node, effect_nodes.FMSynthNode):
            return "fm_synth_node"
        if isinstance(node, effect_nodes.BubbleSimGeneratorNode):
            return "bubble_sim_generator_node"
        if isinstance(node, effect_nodes.BubbleSimGenerator2Node):
            return "bubble_sim_generator2_node"
        if isinstance(node, effect_nodes.AbyssalSignalFeedbackShaderNode):
            return "abyssal_signal_feedback"
        label = str(getattr(node, "label", node.__class__.__name__)).strip().lower()
        return "_".join(part for part in label.replace("-", " ").split() if part)

    def destroy_audio_nodes(self):
        for node in tuple(self.nodes):
            if not getattr(node, "is_audio_node", False):
                continue
            try:
                node.destroy()
            except Exception:
                logging.exception("Audio node shutdown failed: %s", getattr(node, "label", "?"))

    def get_node_preset_data(self):
        """Return a serializable snapshot of this editor without writing a file."""
        preset_data = {
            "nodes": [],
            "links": [],
            "version": "1.0",
        }
        preset_ids = {}
        type_counts = {}

        for node in self.nodes:
            if isinstance(node, (SourceNode, SecondarySourceNode, DisplayNode)):
                pos = dpg.get_item_pos(node.node_tag)
                if isinstance(node, SourceNode):
                    preset_data["source_pos"] = pos
                elif isinstance(node, SecondarySourceNode):
                    preset_data["source_b_pos"] = pos
                elif isinstance(node, DisplayNode):
                    preset_data["display_pos"] = pos
                continue

            if isinstance(node, effect_nodes.Node):
                node_type = node.__class__.__name__
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
                preset_id = f"{node_type}:{type_counts[node_type]}"
                preset_ids[node.node_tag] = preset_id
                preset_data["nodes"].append({
                    "id": preset_id,
                    "type": node_type,
                    "label": node.label,
                    "position": dpg.get_item_pos(node.node_tag),
                    "params": dict(getattr(node, "param_values", {}) or {}),
                })

        for link_tag in self.links:
            try:
                cfg = dpg.get_item_configuration(link_tag)
                attr_1 = cfg.get("attr_1")
                attr_2 = cfg.get("attr_2")
                from_node_tag = None
                to_node_tag = None
                from_attr_name = None
                to_attr_name = None

                for node in self.nodes:
                    for attr_name, attr_tag in node.output_attrs.items():
                        if attr_tag == attr_1:
                            from_node_tag = node.node_tag
                            from_attr_name = attr_name
                            break
                    for attr_name, attr_tag in node.input_attrs.items():
                        if attr_tag == attr_2:
                            to_node_tag = node.node_tag
                            to_attr_name = attr_name
                            break

                if from_node_tag and to_node_tag:
                    preset_data["links"].append({
                        "from_node": self._get_node_identifier(from_node_tag),
                        "to_node": self._get_node_identifier(to_node_tag),
                        "from_node_id": preset_ids.get(from_node_tag),
                        "to_node_id": preset_ids.get(to_node_tag),
                        "from_attr": from_attr_name,
                        "to_attr": to_attr_name,
                    })
            except Exception as e:
                logging.warning("Could not serialize link: %s", e)

        return preset_data
    
    def save_node_preset(self, preset_name):
        """Save a legacy single-editor preset.

        The application-level save command writes dual-deck session presets.
        This method remains for compatibility with direct callers, but must not
        replace a newer session file with one deck.
        """
        import json
        import logging
        
        if not preset_name:
            logging.warning("Preset name is required")
            return False
        
        preset_data = {
            'nodes': [],
            'links': [],
            'version': '1.0'
        }
        preset_ids = {}
        type_counts = {}
        
        # Save all nodes (except Source and Display which are always present)
        for node in self.nodes:
            if isinstance(node, (SourceNode, SecondarySourceNode, DisplayNode)):
                # Save position of source and display nodes
                pos = dpg.get_item_pos(node.node_tag)
                if isinstance(node, SourceNode):
                    preset_data['source_pos'] = pos
                elif isinstance(node, SecondarySourceNode):
                    preset_data['source_b_pos'] = pos
                elif isinstance(node, DisplayNode):
                    preset_data['display_pos'] = pos
                continue
            
            if isinstance(node, effect_nodes.Node):
                node_type = node.__class__.__name__
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
                preset_id = f"{node_type}:{type_counts[node_type]}"
                preset_ids[node.node_tag] = preset_id
                node_data = {
                    'id': preset_id,
                    'type': node_type,
                    'label': node.label,
                    'position': dpg.get_item_pos(node.node_tag),
                    'params': dict(getattr(node, 'param_values', {}) or {})
                }
                preset_data['nodes'].append(node_data)
        
        # Save all links
        for link_tag in self.links:
            try:
                cfg = dpg.get_item_configuration(link_tag)
                # Get the node tags from the attribute tags
                attr_1 = cfg.get('attr_1')
                attr_2 = cfg.get('attr_2')
                
                # Find which nodes these attributes belong to
                from_node_tag = None
                to_node_tag = None
                from_attr_name = None
                to_attr_name = None
                
                for node in self.nodes:
                    for attr_name, attr_tag in node.output_attrs.items():
                        if attr_tag == attr_1:
                            from_node_tag = node.node_tag
                            from_attr_name = attr_name
                            break
                    for attr_name, attr_tag in node.input_attrs.items():
                        if attr_tag == attr_2:
                            to_node_tag = node.node_tag
                            to_attr_name = attr_name
                            break
                
                if from_node_tag and to_node_tag:
                    link_data = {
                        'from_node': self._get_node_identifier(from_node_tag),
                        'to_node': self._get_node_identifier(to_node_tag),
                        'from_node_id': preset_ids.get(from_node_tag),
                        'to_node_id': preset_ids.get(to_node_tag),
                        'from_attr': from_attr_name,
                        'to_attr': to_attr_name
                    }
                    preset_data['links'].append(link_data)
            except Exception as e:
                logging.warning(f"Could not save link: {e}")
        
        filepath = f"node_presets/{preset_name}.json"
        try:
            if Path(filepath).exists():
                try:
                    with open(filepath, "r", encoding="utf-8") as existing_file:
                        existing = json.load(existing_file)
                    if isinstance(existing.get("decks"), dict):
                        logging.error(
                            "Refusing to overwrite dual-deck session preset '%s' with one editor",
                            preset_name,
                        )
                        return False
                except Exception:
                    pass
            with open(filepath, 'w', encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2)
            logging.info(f"Node preset saved: {preset_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to save preset: {e}")
            return False
    
    def load_node_preset(self, preset_name, preset_data=None):
        """Load a node configuration from a preset file."""
        import json
        import logging
        
        if preset_data is None:
            filepath = f"node_presets/{preset_name}.json"
            try:
                with open(filepath, 'r') as f:
                    preset_data = json.load(f)
            except FileNotFoundError:
                logging.error(f"Preset not found: {preset_name}")
                return False
            except Exception as e:
                logging.error(f"Failed to load preset: {e}")
                return False
        if not isinstance(preset_data, dict):
            logging.error("Invalid preset data for %s", preset_name)
            return False
        
        # Clear existing nodes (except Source and Display)
        nodes_to_remove = [n for n in self.nodes if not isinstance(n, (SourceNode, DisplayNode))]
        for node in nodes_to_remove:
            try:
                destroy = getattr(node, "destroy", None)
                if callable(destroy):
                    destroy()
                dpg.delete_item(node.node_tag)
                self.nodes.remove(node)
            except Exception:
                pass
        
        # Clear all links
        self.clear_all_links()
        
        # Restore Source and Display positions if saved
        if 'source_pos' in preset_data:
            source = self.get_source_node()
            if source:
                dpg.set_item_pos(source.node_tag, preset_data['source_pos'])
        if 'source_b_pos' in preset_data:
            source_b = self.get_secondary_source_node()
            if not source_b:
                self.add_secondary_source_node()
                source_b = self.get_secondary_source_node()
            if source_b:
                dpg.set_item_pos(source_b.node_tag, preset_data['source_b_pos'])
        if 'display_pos' in preset_data:
            display = self.get_display_node()
            if display:
                dpg.set_item_pos(display.node_tag, preset_data['display_pos'])
        
        # Create nodes
        node_map = {}  # Map explicit preset ids and unique legacy labels to new node objects
        legacy_label_map = {}
        lua_preset_map = {}
        for node_data in preset_data.get('nodes', []):
            try:
                node_class_name = node_data['type']
                node_class = getattr(effect_nodes, node_class_name, None)
                if node_class is None:
                    try:
                        import effect_nodes_cuda
                        node_class = getattr(effect_nodes_cuda, node_class_name, None)
                    except Exception:
                        node_class = None
                
                if node_class:
                    new_node = node_class(self.node_editor_tag)
                    pos = node_data.get('position', [100, 100])
                    new_node.create_node(pos=pos)
                    
                    # Lua shader paths must be applied first so metadata-driven
                    # controls exist before their saved values are restored.
                    self._restore_node_params(new_node, node_data.get('params', {}))
                    
                    self.nodes.append(new_node)
                    self.mark_pipeline_dirty()
                    preset_node_id = node_data.get("id")
                    if preset_node_id:
                        node_map[str(preset_node_id)] = new_node
                    label = str(node_data.get('label') or "")
                    if label:
                        legacy_label_map.setdefault(label, []).append(new_node)
                        node_map.setdefault(label, new_node)
                    params = dict(node_data.get("params", {}) or {})
                    shader_preset = str(params.get("shader_preset") or "").strip()
                    if shader_preset:
                        lua_preset_map.setdefault(shader_preset, []).append(new_node)
                    shader_path = str(params.get("shader_path") or "").replace("\\", "/").strip()
                    if shader_path:
                        lua_preset_map.setdefault(Path(shader_path).stem, []).append(new_node)
                else:
                    logging.error(f"Unknown node type in preset '{preset_name}': {node_class_name}")
            except Exception as e:
                logging.error(f"Failed to create node {node_data.get('type')}: {e}")
        
        # Recreate links
        for link_data in preset_data.get('links', []):
            try:
                from_identifier = link_data['from_node']
                to_identifier = link_data['to_node']
                from_attr = link_data['from_attr']
                to_attr = link_data['to_attr']
                
                from_node = self._find_node_by_identifier(
                    link_data.get("from_node_id") or from_identifier,
                    node_map,
                    fallback_attr=from_attr,
                    label_map=legacy_label_map,
                    lua_preset_map=lua_preset_map,
                    link_role="from",
                    peer_identifier=to_identifier,
                    peer_attr=to_attr,
                )
                to_node = self._find_node_by_identifier(
                    link_data.get("to_node_id") or to_identifier,
                    node_map,
                    fallback_attr=to_attr,
                    label_map=legacy_label_map,
                    lua_preset_map=lua_preset_map,
                    link_role="to",
                    peer_identifier=from_identifier,
                    peer_attr=from_attr,
                )
                
                if from_node and to_node:
                    from_attr_tag = from_node.output_attrs.get(from_attr)
                    to_attr_tag = to_node.input_attrs.get(to_attr)
                    
                    if from_attr_tag and to_attr_tag:
                        link = dpg.add_node_link(from_attr_tag, to_attr_tag, parent=self.node_editor_tag)
                        self.links.append(link)
                        self.mark_pipeline_dirty()
            except Exception as e:
                logging.error(f"Failed to create link: {e}")
        
        logging.info(f"Node preset loaded: {preset_name}")
        return True
    
    def get_available_presets(self):
        """Get valid preset names, excluding truncated or malformed files."""
        import json
        import os

        self._ensure_presets_directory()
        try:
            presets = []
            for filename in os.listdir("node_presets"):
                if not filename.endswith(".json"):
                    continue
                filepath = Path("node_presets") / filename
                try:
                    if filepath.stat().st_size <= 0:
                        logging.warning("Skipping empty preset: %s", filepath)
                        continue
                    with filepath.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    if not isinstance(data, dict):
                        logging.warning("Skipping invalid preset root: %s", filepath)
                        continue
                    if not isinstance(data.get("decks"), dict) and not isinstance(data.get("nodes"), list):
                        logging.warning("Skipping unsupported preset: %s", filepath)
                        continue
                    presets.append(filename[:-5])
                except Exception as exc:
                    logging.warning("Skipping unreadable preset %s: %s", filepath, exc)
            return sorted(presets, key=str.lower)
        except Exception:
            return []
    
    def _get_node_identifier(self, node_tag):
        """Get a unique identifier for a node (label for effect nodes, type for Source/Display)."""
        for node in self.nodes:
            if node.node_tag == node_tag:
                if isinstance(node, SourceNode):
                    return 'SOURCE'
                elif isinstance(node, SecondarySourceNode):
                    return 'SOURCE_B'
                elif isinstance(node, DisplayNode):
                    return 'DISPLAY'
                else:
                    return node.label
        return None
    
    def _find_node_by_identifier(
        self,
        identifier,
        node_map,
        fallback_attr=None,
        label_map=None,
        lua_preset_map=None,
        link_role=None,
        peer_identifier=None,
        peer_attr=None,
    ):
        """Find a node by its identifier."""
        if identifier == 'SOURCE':
            return self.get_source_node()
        elif identifier == 'SOURCE_B':
            return self.get_secondary_source_node()
        elif identifier == 'DISPLAY':
            return self.get_display_node()
        else:
            key = str(identifier or "")
            if key in node_map:
                return node_map.get(key)

            candidates = list((label_map or {}).get(key, []))
            if len(candidates) == 1:
                return candidates[0]

            if candidates and key == "Lua Shader":
                attr = str(fallback_attr or "")
                peer = str(peer_identifier or "")
                peer_attr = str(peer_attr or "")
                if link_role == "from" and peer == "DISPLAY":
                    preferred = (lua_preset_map or {}).get("kaleido_feedback") or []
                    if preferred:
                        return preferred[0]
                if link_role == "from" and peer_attr == "feedback_in":
                    preferred = (lua_preset_map or {}).get("glitch_feedback") or []
                    if preferred:
                        return preferred[0]
                if link_role == "to" and attr == "feedback_in":
                    preferred = (lua_preset_map or {}).get("kaleido_feedback") or []
                    if preferred:
                        return preferred[0]
                if link_role == "to" and attr == "image_in":
                    preferred = (lua_preset_map or {}).get("kaleido_feedback") or []
                    if preferred:
                        return preferred[0]
                if link_role == "from" and attr == "image_out":
                    preferred = (lua_preset_map or {}).get("glitch_feedback") or []
                    if preferred:
                        return preferred[0]
                return candidates[0]

            return None
