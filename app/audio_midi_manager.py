import sounddevice as sd
import mido
import threading
import queue
import numpy as np
import time
import logging
import inspect


class AudioMidiManager:
    def __init__(self, sample_rate, buffer_size, fm_params, fm_state, effect_params, fft_size=1024, audio_callbacks=None, midi_callbacks=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.fm_params = fm_params
        self.fm_state = fm_state
        self.effect_params = effect_params
        self.audio_callbacks = list(audio_callbacks) if audio_callbacks else []
        self.midi_callbacks = list(midi_callbacks) if midi_callbacks else []

        self.input_stream = None
        self.output_stream = None
        self.audio_stream = None  # Legacy attribute for stop_audio compatibility
        self.audio_input_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()
        self.stop_input_thread = threading.Event()
        self.stop_output_thread = threading.Event()
        self.stop_midi_thread = threading.Event()
        self.input_thread = None
        self.output_thread = None
        self.midi_thread = None
        self.fft_data = np.zeros(fft_size // 2 + 1)
        self.fft_freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
        self.fft_band_names = [
            "Sub", "Bass", "Low Mid", "Mid", "High Mid", "Presence", "Brilliance", "Air",
            "Band 01", "Band 02", "Band 03", "Band 04", "Band 05", "Band 06", "Band 07", "Band 08",
            "Band 09", "Band 10", "Band 11", "Band 12", "Band 13", "Band 14", "Band 15", "Band 16",
        ]
        self.fft_bands = {name: 0.0 for name in self.fft_band_names}
        self.onset_value = 0.0
        self.onset_event_id = 0
        self._last_spectrum = np.zeros(fft_size // 2 + 1, dtype=np.float32)
        self._flux_history = []
        self._last_onset_time = 0.0
        self.selected_input_device_index = None
        self.selected_input_is_loopback = False
        self.selected_input_channels = 1
        self.selected_output_device_index = None
        self.last_input_error = ""

        self.fft_size = fft_size
        self.audio_buffer = np.zeros(self.fft_size * 4, dtype=np.float32)
        self.audio_pos = 0
        self.spectrogram_history = 200
        self.spectrogram_data = np.zeros((self.fft_size // 2 + 1, self.spectrogram_history), dtype=np.float32)

        self.midi_in = None
        self.selected_midi_input_name = ""
        self.latest_midi_note_velocity = 0
        self.latest_midi_note = 0
        self.latest_midi_cc_value = 0
        self.latest_midi_cc_number = 0

        # Lock used for fft_data and MIDI values accessed from other threads
        self.data_lock = threading.Lock()

    def add_audio_callback(self, callback):
        if callable(callback) and callback not in self.audio_callbacks:
            self.audio_callbacks.append(callback)

    def add_midi_callback(self, callback):
        if callable(callback) and callback not in self.midi_callbacks:
            self.midi_callbacks.append(callback)

    def start_input_stream(self):
        try:
            self.stop_input_thread.clear()
            stream_kwargs = {
                "device": self.selected_input_device_index,
                "channels": self.selected_input_channels,
                "samplerate": self.sample_rate,
                "callback": self.input_audio_callback,
            }
            if self.selected_input_is_loopback:
                if not self._supports_wasapi_loopback():
                    raise RuntimeError(
                        "This sounddevice build does not support WASAPI output loopback. "
                        "Select an input/capture endpoint such as Stereo Mix, VoiceMeeter Output, "
                        "CABLE Output, or your interface loopback input."
                    )
                stream_kwargs["extra_settings"] = sd.WasapiSettings(loopback=True)
            self.input_stream = sd.InputStream(
                **stream_kwargs,
            )
            self.input_stream.start()
            self.input_thread = threading.Thread(target=self.process_input_audio_data, daemon=True)
            self.input_thread.start()
            self.last_input_error = ""
            logging.info("Audio input stream started.")
            return True
        except Exception as e:
            self.last_input_error = str(e)
            logging.warning("Error initializing audio input: %s", e)
            self.input_stream = None
            return False

    @staticmethod
    def _supports_wasapi_loopback():
        try:
            return "loopback" in inspect.signature(sd.WasapiSettings).parameters
        except Exception:
            return False

    def start_output_stream(self):
        try:
            self.stop_output_thread.clear()
            self.output_stream = sd.OutputStream(
                device=self.selected_output_device_index,
                channels=2,
                samplerate=self.sample_rate,
                callback=self.output_audio_callback,
            )
            self.output_stream.start()
            self.output_thread = threading.Thread(target=self.process_output_audio_data, daemon=True)
            self.output_thread.start()
            logging.info("Audio output stream started.")
        except Exception as e:
            logging.warning("Error initializing audio output: %s", e)
            self.output_stream = None

    def stop_input_stream(self):
        if self.input_stream:
            self.stop_input_thread.set()
            if self.input_thread and self.input_thread.is_alive():
                self.input_thread.join()
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            logging.info("Audio input stream stopped.")

    def stop_output_stream(self):
        if self.output_stream:
            self.stop_output_thread.set()
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join()
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            logging.info("Audio output stream stopped.")

    def input_audio_callback(self, indata, frames, time_info, status):
        self.audio_input_queue.put(indata.copy())

    def output_audio_callback(self, outdata, frames, time_info, status):
        try:
            block = self.audio_output_queue.get(timeout=0.1)
            if block.shape[0] == frames:
                outdata[:] = block
            else:
                if block.shape[0] > frames:
                    outdata[:] = block[:frames]
                else:
                    outdata[:block.shape[0]] = block
                    outdata[block.shape[0]:] = 0.0
            self.audio_output_queue.task_done()
        except queue.Empty:
            outdata.fill(0.0)

    def process_input_audio_data(self):
        while not self.stop_input_thread.is_set():
            if not self.audio_input_queue.empty():
                block = self.audio_input_queue.get()
                mono = block[:, 0].astype(np.float32, copy=False)
                fft_res = self._compute_fft(mono)
                bands, onset_value, onset_event = self._analyze_fft_frame(fft_res)
                with self.data_lock:
                    self.fft_data = fft_res
                    self.fft_bands = bands
                    self.onset_value = onset_value
                    if onset_event:
                        self.onset_event_id += 1
                callbacks_snapshot = tuple(self.audio_callbacks)
                timestamp = time.monotonic()
                for cb in callbacks_snapshot:
                    try:
                        cb(mono.copy(), fft_res.copy(), self.sample_rate, timestamp)
                    except Exception:
                        logging.debug("Audio analysis callback failed", exc_info=True)
                self.audio_input_queue.task_done()
            else:
                time.sleep(0.01)

    def _compute_fft(self, mono):
        if mono.size == 0:
            return np.zeros(self.fft_size // 2 + 1, dtype=np.float32)
        windowed = np.zeros(self.fft_size, dtype=np.float32)
        count = min(mono.size, self.fft_size)
        windowed[:count] = mono[:count]
        window = np.hanning(self.fft_size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(windowed * window, n=self.fft_size)).astype(np.float32)
        return spectrum

    def _analyze_fft_frame(self, spectrum):
        max_mag = float(np.max(spectrum)) if spectrum.size else 0.0
        denom = max(max_mag, 1e-6)

        named_ranges = [
            ("Sub", 20.0, 60.0),
            ("Bass", 60.0, 250.0),
            ("Low Mid", 250.0, 500.0),
            ("Mid", 500.0, 2000.0),
            ("High Mid", 2000.0, 4000.0),
            ("Presence", 4000.0, 6000.0),
            ("Brilliance", 6000.0, 12000.0),
            ("Air", 12000.0, min(20000.0, self.sample_rate * 0.5)),
        ]
        bands = {}
        for name, low, high in named_ranges:
            bands[name] = self._band_energy(spectrum, low, high, denom)

        nyquist = max(1000.0, self.sample_rate * 0.5)
        edges = np.geomspace(40.0, nyquist, 17)
        for index in range(16):
            bands[f"Band {index + 1:02d}"] = self._band_energy(spectrum, float(edges[index]), float(edges[index + 1]), denom)

        positive_flux = np.maximum(spectrum - self._last_spectrum, 0.0)
        flux = float(np.mean(positive_flux)) / denom if positive_flux.size else 0.0
        self._last_spectrum = spectrum.copy()
        self._flux_history.append(flux)
        if len(self._flux_history) > 64:
            self._flux_history.pop(0)
        avg_flux = float(np.mean(self._flux_history)) if self._flux_history else 0.0
        std_flux = float(np.std(self._flux_history)) if len(self._flux_history) > 1 else 0.0
        threshold = avg_flux + max(0.03, std_flux * 2.5)
        now = time.monotonic()
        onset_event = flux > threshold and (now - self._last_onset_time) >= 0.08
        if onset_event:
            self._last_onset_time = now
        onset_value = min(1.0, max(0.0, flux / max(threshold, 1e-6)))
        return bands, onset_value, onset_event

    def _band_energy(self, spectrum, low_hz, high_hz, denom):
        mask = (self.fft_freqs >= low_hz) & (self.fft_freqs < high_hz)
        if not np.any(mask):
            return 0.0
        value = float(np.mean(spectrum[mask])) / denom
        return min(1.0, max(0.0, value))

    def process_output_audio_data(self):
        while not self.stop_output_thread.is_set():
            time.sleep(0.1)

    def start_midi_thread(self):
        if self.midi_thread and self.midi_thread.is_alive():
            return
        self.stop_midi_thread.clear()
        self.midi_thread = threading.Thread(target=self.process_midi_data, daemon=True)
        self.midi_thread.start()

    def process_midi_data(self):
        try:
            kwargs = {}
            if self.selected_midi_input_name:
                kwargs["name"] = self.selected_midi_input_name
            self.midi_in = mido.open_input(**kwargs)
            for msg in self.midi_in:
                if self.stop_midi_thread.is_set():
                    break
                with self.data_lock:
                    if msg.type == 'note_on':
                        self.latest_midi_note = msg.note
                        self.latest_midi_note_velocity = msg.velocity
                    elif msg.type == 'control_change':
                        self.latest_midi_cc_number = msg.control
                        self.latest_midi_cc_value = msg.value
                callbacks_snapshot = tuple(self.midi_callbacks)
                for cb in callbacks_snapshot:
                    try:
                        cb(msg)
                    except Exception:
                        logging.debug("MIDI callback failed", exc_info=True)
        except AttributeError as exc:
            logging.warning("MIDI backend unavailable; disabling MIDI input (%s)", exc)
        except Exception as exc:
            logging.debug("MIDI thread error", exc_info=True)
        finally:
            if self.midi_in:
                try:
                    self.midi_in.close()
                except Exception:
                    pass
                self.midi_in = None

    def get_midi_input_names(self):
        try:
            return list(mido.get_input_names())
        except Exception as exc:
            logging.warning("Could not list MIDI inputs: %s", exc)
            return []

    def select_midi_input(self, name):
        self.selected_midi_input_name = str(name or "")
        logging.info("Selected MIDI input: %s", self.selected_midi_input_name or "System Default")
        was_running = self.midi_thread and self.midi_thread.is_alive()
        if was_running:
            self.stop_midi_thread.set()
            if self.midi_in:
                try:
                    self.midi_in.close()
                except Exception:
                    pass
            self.midi_thread.join(timeout=0.5)
            self.midi_thread = None
            self.stop_midi_thread.clear()
            self.start_midi_thread()

    def stop_audio(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
            logging.info("Audio stopped.")

    def select_input_device(self, index):
        self.selected_input_is_loopback = False
        self.selected_input_channels = 1
        self.selected_input_device_index = index
        logging.info("Selected audio input device %s", index)
        return self.restart_input_stream()

    def select_input_source(self, source):
        source = source or {}
        self.selected_input_device_index = source.get("index")
        self.selected_input_is_loopback = bool(source.get("loopback", False))
        try:
            self.selected_input_channels = max(1, int(source.get("channels", 1) or 1))
        except Exception:
            self.selected_input_channels = 1
        logging.info(
            "Selected audio source index=%s loopback=%s channels=%s",
            self.selected_input_device_index,
            self.selected_input_is_loopback,
            self.selected_input_channels,
        )
        return self.restart_input_stream()

    def select_output_device(self, index):
        self.selected_output_device_index = index
        logging.info("Selected audio output device %s", index)
        self.restart_output_stream()

    def restart_input_stream(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.stop_input_thread.set()
            if self.input_thread and self.input_thread.is_alive():
                self.input_thread.join()
            self.stop_input_thread.clear()
        return self.start_input_stream()

    def restart_output_stream(self):
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.stop_output_thread.set()
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join()
            self.stop_output_thread.clear()
        self.start_output_stream()

    def get_input_devices(self):
        devices = sd.query_devices()
        input_devs = [dev for dev in devices if dev.get('max_input_channels', 0) > 0]
        return [{'index': d['index'], 'name': d['name']} for d in input_devs]

    def get_input_sources(self):
        devices = sd.query_devices()
        try:
            hostapis = sd.query_hostapis()
        except Exception:
            hostapis = []
        sources = []
        for dev in devices:
            try:
                index = dev.get("index")
                name = str(dev.get("name") or f"Device {index}")
                hostapi_index = int(dev.get("hostapi", -1))
                hostapi_name = ""
                if 0 <= hostapi_index < len(hostapis):
                    hostapi_name = str(hostapis[hostapi_index].get("name") or "")
                max_input = int(dev.get("max_input_channels", 0) or 0)
                max_output = int(dev.get("max_output_channels", 0) or 0)
                if max_input > 0:
                    sources.append({
                        "index": index,
                        "name": name,
                        "label": f"Input: {index}: {name}",
                        "loopback": False,
                        "channels": 1,
                    })
                if (
                    max_output > 0
                    and "WASAPI" in hostapi_name.upper()
                    and hasattr(sd, "WasapiSettings")
                    and self._supports_wasapi_loopback()
                ):
                    sources.append({
                        "index": index,
                        "name": name,
                        "label": f"Loopback: {index}: {name}",
                        "loopback": True,
                        "channels": min(2, max_output),
                    })
            except Exception:
                continue
        return sources

    def get_output_devices(self):
        devices = sd.query_devices()
        output_devs = [dev for dev in devices if dev.get('max_output_channels', 0) > 0]
        return [{'index': d['index'], 'name': d['name']} for d in output_devs]

    def shutdown(self):
        self.stop_input_thread.set()
        self.stop_output_thread.set()
        self.stop_midi_thread.set()

        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join()
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join()
        if self.midi_thread and self.midi_thread.is_alive():
            self.midi_thread.join(timeout=0.5)

        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        if self.midi_in:
            try:
                self.midi_in.close()
            except Exception:
                pass
            self.midi_in = None
