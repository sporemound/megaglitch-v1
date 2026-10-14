"""Isolated sounddevice backend adapter."""

from __future__ import annotations

import logging
import threading
import time


class AudioDevice:
    def __init__(
        self,
        render_callback,
        sample_rate=48000,
        block_size=256,
        channels=2,
        device=None,
        latency=None,
    ):
        self.render_callback = render_callback
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.channels = int(channels)
        self.device = device
        self.latency = latency
        self.stream = None
        self.available = False
        self.last_error = None
        self._lock = threading.Lock()
        self._last_status_log_time = 0.0
        self._suppressed_status_count = 0
        try:
            import sounddevice as sounddevice
            self._sd = sounddevice
            self.available = True
        except Exception as exc:
            self._sd = None
            self.last_error = str(exc)

    @property
    def running(self):
        return self.stream is not None and bool(getattr(self.stream, "active", False))

    def _callback(self, outdata, frames, time_info, status):
        if status:
            now = time.monotonic()
            if now - self._last_status_log_time >= 1.0:
                suppressed = self._suppressed_status_count
                self._suppressed_status_count = 0
                self._last_status_log_time = now
                if suppressed:
                    logging.debug(
                        "FM audio callback status: %s (%s similar callback statuses suppressed)",
                        status,
                        suppressed,
                    )
                else:
                    logging.debug("FM audio callback status: %s", status)
            else:
                self._suppressed_status_count += 1
        try:
            self.render_callback(frames, out=outdata)
        except Exception:
            logging.exception("FM audio callback failed")
            outdata.fill(0.0)

    def start(self):
        if not self.available:
            logging.warning("sounddevice unavailable; FM synth remains in offline mode: %s", self.last_error)
            return False
        with self._lock:
            if self.running:
                return True
            try:
                self.stream = self._sd.OutputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=self.channels,
                    dtype="float32",
                    callback=self._callback,
                    device=self.device,
                    latency=self.latency,
                )
                self.stream.start()
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self.stream = None
                logging.warning("Could not start FM audio output: %s", exc)
                return False

    def stop(self):
        with self._lock:
            stream, self.stream = self.stream, None
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()

    def destroy(self):
        self.stop()
