import os
import json
import dearpygui.dearpygui as dpg

class PresetManager:
    SOLFEGGIO_PRESETS = {
        '174 Hz - Foundation': {
            'carrier_freq': 174.0, 'modulator_freq': 87.0, 'modulation_index': 10.0,
            'carrier_waveform': 'Sine', 'modulator_waveform': 'Sine',
            'modulation_index_start': 1.0, 'modulation_index_end': 1.0, 'mod_index_curve': 'linear',
            'lfo_freq': 0.1, 'lfo_depth': 2.0, 'attack': 0.5, 'decay': 0.5, 'sustain': 0.8, 'release': 0.5,
            'filter_cutoff': 5000.0, 'reverb_decay': 2.0, 'reverb_mix': 0.5, 'delay_time': 0.5, 'delay_decay': 0.5,
            'motion_mod_index_scale': 0.5, 'color_filter_scale': 0.5, 'impulse_response_path': ''
        },
        '285 Hz - Quantum Cognition': {
            'carrier_freq': 285.0, 'modulator_freq': 71.25, 'modulation_index': 15.0,
            'carrier_waveform': 'Triangle', 'modulator_waveform': 'Triangle',
            'modulation_index_start': 2.0, 'modulation_index_end': 1.0, 'mod_index_curve': 'exponential',
            'lfo_freq': 0.2, 'lfo_depth': 3.0, 'attack': 0.3, 'decay': 0.7, 'sustain': 0.6, 'release': 0.3,
            'filter_cutoff': 6000.0, 'reverb_decay': 3.0, 'reverb_mix': 0.6, 'delay_time': 0.4, 'delay_decay': 0.6,
            'motion_mod_index_scale': 0.6, 'color_filter_scale': 0.6, 'impulse_response_path': ''
        },
        '396 Hz - Liberating Guilt and Fear': {
            'carrier_freq': 396.0, 'modulator_freq': 99.0, 'modulation_index': 20.0,
            'carrier_waveform': 'Sawtooth', 'modulator_waveform': 'Sawtooth',
            'modulation_index_start': 1.0, 'modulation_index_end': 5.0, 'mod_index_curve': 'linear',
            'lfo_freq': 0.5, 'lfo_depth': 5.0, 'attack': 0.2, 'decay': 0.5, 'sustain': 0.7, 'release': 0.2,
            'filter_cutoff': 7000.0, 'reverb_decay': 2.5, 'reverb_mix': 0.7, 'delay_time': 0.3, 'delay_decay': 0.7,
            'motion_mod_index_scale': 0.7, 'color_filter_scale': 0.7, 'impulse_response_path': ''
        },
        '417 Hz - Undoing Situations': {
            'carrier_freq': 417.0, 'modulator_freq': 104.25, 'modulation_index': 25.0,
            'carrier_waveform': 'Square', 'modulator_waveform': 'Square',
            'modulation_index_start': 5.0, 'modulation_index_end': 1.0, 'mod_index_curve': 'exponential',
            'lfo_freq': 0.7, 'lfo_depth': 7.0, 'attack': 0.1, 'decay': 0.4, 'sustain': 0.6, 'release': 0.1,
            'filter_cutoff': 8000.0, 'reverb_decay': 3.0, 'reverb_mix': 0.8, 'delay_time': 0.2, 'delay_decay': 0.8,
            'motion_mod_index_scale': 0.8, 'color_filter_scale': 0.8, 'impulse_response_path': ''
        },
        '528 Hz - Transformation and Miracles': {
            'carrier_freq': 528.0, 'modulator_freq': 132.0, 'modulation_index': 30.0,
            'carrier_waveform': 'Sine', 'modulator_waveform': 'Sine',
            'modulation_index_start': 1.0, 'modulation_index_end': 10.0, 'mod_index_curve': 'linear',
            'lfo_freq': 1.0, 'lfo_depth': 10.0, 'attack': 0.05, 'decay': 0.3, 'sustain': 0.9, 'release': 0.05,
            'filter_cutoff': 9000.0, 'reverb_decay': 4.0, 'reverb_mix': 0.9, 'delay_time': 0.1, 'delay_decay': 0.9,
            'motion_mod_index_scale': 0.9, 'color_filter_scale': 0.9, 'impulse_response_path': ''
        },
    }

    def __init__(self, slider_tags, fm_params, effect_params, gif_audio_params, mod_connections_ref, preset_list_tag):
        self.slider_tags = slider_tags
        self.fm_params = fm_params
        self.effect_params = effect_params
        self.gif_audio_params = gif_audio_params # New parameter for GIF audio specific params
        self.mod_connections_ref = mod_connections_ref
        self.preset_list_tag = preset_list_tag
        self._ensure_presets_directory()

    def _ensure_presets_directory(self):
        if not os.path.exists("presets"):
            os.makedirs("presets")

    def get_saved_presets(self):
        self._ensure_presets_directory()
        file_presets = [f.replace(".json", "") for f in os.listdir("presets") if f.endswith(".json")]
        return list(self.SOLFEGGIO_PRESETS.keys()) + file_presets

    def save_preset(self, preset_name):
        if not preset_name:
            print("Enter preset name.")
            return
        preset_data = {}
        for k, tag in self.slider_tags.items():
            if k not in ["pixel_sorting_direction", "glitch_text"]:
                preset_data[k] = dpg.get_value(tag)
        preset_data['pixel_sorting_direction'] = dpg.get_value('pixel_sorting_direction')
        preset_data['glitch_text'] = dpg.get_value('glitch_text')
        preset_data['mod_connections'] = list(self.mod_connections_ref)
        preset_data['fm_params'] = dict(self.fm_params)
        preset_data['effect_params'] = dict(self.effect_params)
        preset_data['gif_audio_params'] = dict(self.gif_audio_params) # Save new GIF audio params

        with open(f"presets/{preset_name}.json", "w") as f:
            json.dump(preset_data, f, indent=4)
        dpg.configure_item(self.preset_list_tag, items=self.get_saved_presets())
        print(f"Preset '{preset_name}' saved.")

    def load_preset(self, sel_preset_name, mod_connection_list_tag):
        if not sel_preset_name:
            print("Select a preset to load.")
            return

        data = None
        if sel_preset_name in self.SOLFEGGIO_PRESETS:
            data = self.SOLFEGGIO_PRESETS[sel_preset_name]
        else:
            try:
                with open(f"presets/{sel_preset_name}.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"Preset file '{sel_preset_name}.json' not found.")
                return

        if data:
            for k, v in data.items():
                if k in self.slider_tags:
                    dpg.set_value(self.slider_tags[k], v)
                elif k == "pixel_sorting_direction":
                    dpg.set_value('pixel_sorting_direction', v)
                elif k == "glitch_text":
                    dpg.set_value('glitch_text', v)
                elif k == "mod_connections":
                    self.mod_connections_ref[:] = v # Update the referenced list in place
                    dpg.configure_item(mod_connection_list_tag, items=[c for c in v])
                elif k == "fm_params":
                    for fm_key, fm_val in v.items():
                        self.fm_params[fm_key] = fm_val
                elif k == "effect_params":
                    for ef_key, ef_val in v.items():
                        self.effect_params[ef_key] = ef_val
                elif k == "gif_audio_params": # Load new GIF audio params
                    for ga_key, ga_val in v.items():
                        self.gif_audio_params[ga_key] = ga_val
            print(f"Preset '{sel_preset_name}' loaded.")

    def delete_preset(self, sel_preset_name):
        if not sel_preset_name:
            print("Select preset to delete.")
            return
        if sel_preset_name in self.SOLFEGGIO_PRESETS:
            print(f"Cannot delete built-in Solfeggio preset: '{sel_preset_name}'.")
            return
        try:
            os.remove(f"presets/{sel_preset_name}.json")
            dpg.configure_item(self.preset_list_tag, items=self.get_saved_presets())
            print(f"Preset '{sel_preset_name}' deleted.")
        except FileNotFoundError:
            print(f"Preset file '{sel_preset_name}.json' not found.")