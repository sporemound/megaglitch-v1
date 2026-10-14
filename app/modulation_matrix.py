import dearpygui.dearpygui as dpg
import numpy as np
import math
import time

class ModulationMatrix:
    def __init__(self, slider_tags, fm_params, mod_connections_ref, fft_data_ref, latest_midi_note_velocity_ref, latest_midi_cc_value_ref, gif_audio_params_ref, latest_optflow_ref, latest_gan_ref, latest_spectral_ref):
        self.slider_tags = slider_tags
        self.fm_params = fm_params
        self.mod_connections_ref = mod_connections_ref
        self.fft_data_ref = fft_data_ref
        self.latest_midi_note_velocity_ref = latest_midi_note_velocity_ref
        self.latest_midi_cc_value_ref = latest_midi_cc_value_ref
        self.gif_audio_params_ref = gif_audio_params_ref
        self.latest_optflow_ref = latest_optflow_ref
        self.latest_gan_ref = latest_gan_ref
        self.latest_spectral_ref = latest_spectral_ref

        self.mod_presets = {
            "None": [],
            "Wavy Glitch": ["Time -> glitch_amount_x", "Time -> glitch_amount_y"],
            "Random Pixelation": ["Random -> pixelation_intensity"],
            "Pulsing Aberration": ["Time -> chromatic_aberration"],
            "Wave Distortion": ["Time -> wave_distortion"],
            "Data Moshing": ["Random -> data_moshing_intensity"],
            "Bass Pulse": ["Bass Amplitude -> invert_chance"],
            "Treble Flicker": ["Treble Amplitude -> rotation_angle"],
            "MIDI Note": ["MIDI Note -> glitch_intensity"],
            "MIDI CC": ["MIDI CC -> pixelation_intensity"]
        }

    def apply_mod_preset(self, preset_name, mod_sources_tags, mod_targets_tags, mod_connection_list_tag):
        self.mod_connections_ref[:] = self.mod_presets.get(preset_name, []).copy()
        dpg.configure_item(mod_connection_list_tag, items=self.mod_connections_ref)
        for s in mod_sources_tags:
            dpg.set_value(mod_sources_tags[s], False)
        for t in mod_targets_tags:
            dpg.set_value(mod_targets_tags[t], False)
        for c in self.mod_connections_ref:
            src, tgt = c.split(" -> ")
            if src in mod_sources_tags:
                dpg.set_value(mod_sources_tags[src], True)
            if tgt in mod_targets_tags:
                dpg.set_value(mod_targets_tags[tgt], True)

    def add_mod_connection(self, sel_sources_tags, sel_targets_tags, mod_connection_list_tag):
        sel_sources = [s for s, t in sel_sources_tags.items() if dpg.get_value(t)]
        sel_targets = [t for t, tt in sel_targets_tags.items() if dpg.get_value(tt)]
        for src in sel_sources:
            for tgt in sel_targets:
                c = f"{src} -> {tgt}"
                if c not in self.mod_connections_ref:
                    self.mod_connections_ref.append(c)
        dpg.configure_item(mod_connection_list_tag, items=self.mod_connections_ref)

    def remove_mod_connection(self, sel_connection, mod_connection_list_tag):
        if sel_connection:
            if sel_connection in self.mod_connections_ref:
                self.mod_connections_ref.remove(sel_connection)
            dpg.configure_item(mod_connection_list_tag, items=self.mod_connections_ref)

    def apply_modulations(self):
        """
        Return (final_fm_params, final_glitch_params, final_gif_audio_params)
        """
        new_fm = self.fm_params.copy()
        new_glitch = {}
        # Correctly gather all current slider/control values
        for tag, dpg_id in self.slider_tags.items():
            new_glitch[tag] = dpg.get_value(dpg_id)
        
        # Handle non-slider tags separately
        new_glitch['pixel_sorting_direction'] = dpg.get_value('pixel_sorting_direction')
        new_glitch['glitch_text'] = dpg.get_value('glitch_text')

        # Type conversion for specific parameters
        if 'macroblock_size' in new_glitch:
            new_glitch['macroblock_size'] = int(new_glitch['macroblock_size'])

        new_gif_audio_params = self.gif_audio_params_ref.copy()
        current_time = time.time()

        for c in self.mod_connections_ref:
            src, tgt = c.split(" -> ")
            base_val = 0.0
            param_min = 0.0
            param_max = 1.0

            if tgt in self.slider_tags:
                conf = dpg.get_item_configuration(self.slider_tags[tgt])
                param_min = conf.get('min_value', 0)
                param_max = conf.get('max_value', 1)
                base_val = dpg.get_value(self.slider_tags[tgt])
            elif tgt in new_fm:
                base_val = new_fm[tgt]
                if tgt == "carrier_freq": param_min, param_max = 20, 20000
                elif tgt == "modulator_freq": param_min, param_max = 0.1, 2000
                elif tgt == "modulation_index": param_min, param_max = 0, 50
                elif tgt == "lfo_freq": param_min, param_max = 0, 20
                elif tgt == "lfo_depth": param_min, param_max = 0, 100
                elif tgt in ["attack", "decay", "release"]: param_min, param_max = 0, 10
                else: param_min, param_max = 0, 1 # sustain, amplitude
            elif tgt in new_gif_audio_params:
                base_val = new_gif_audio_params[tgt]
                if tgt in ["modulation_index_start", "modulation_index_end"]: param_min, param_max = 0.0, 100.0
                elif tgt in ["motion_mod_index_scale", "color_filter_scale"]: param_min, param_max = 0.0, 10.0
                else: continue

            mod_val = 0
            if src == "Time": mod_val = param_min + ((math.sin(current_time) + 1.0) / 2.0) * (param_max - param_min)
            elif src == "Random": mod_val = param_min + np.random.rand() * (param_max - param_min)
            elif src == "Bass Amplitude":
                bass = np.mean(self.fft_data_ref[0:50]) / (self.fft_data_ref.max() or 1)
                mod_val = param_min + bass * (param_max - param_min)
            elif src == "Treble Amplitude":
                treb = np.mean(self.fft_data_ref[200:300]) / (self.fft_data_ref.max() or 1)
                mod_val = param_min + treb * (param_max - param_min)
            elif src == "MIDI Note": mod_val = param_min + (self.latest_midi_note_velocity_ref / 127.0) * (param_max - param_min)
            elif src == "MIDI CC": mod_val = param_min + (self.latest_midi_cc_value_ref / 127.0) * (param_max - param_min)
            elif src == "Optical Flow": mod_val = param_min + self.latest_optflow_ref * (param_max - param_min)
            elif src == "GAN": mod_val = param_min + self.latest_gan_ref * (param_max - param_min)
            elif src == "Spectral": mod_val = param_min + self.latest_spectral_ref * (param_max - param_min)
            else: continue

            if tgt in self.slider_tags:
                if tgt not in ["pixel_sorting_direction", "glitch_text", "displacement_direction", "duplication_pattern", "corruption_type", "sort_method"]:
                    new_glitch[tgt] = mod_val
            elif tgt in new_fm:
                new_fm[tgt] = mod_val
            elif tgt in new_gif_audio_params:
                new_gif_audio_params[tgt] = mod_val

        return new_fm, new_glitch, new_gif_audio_params
