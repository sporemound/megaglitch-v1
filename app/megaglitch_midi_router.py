from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

try:
    import mido
except Exception:  # pragma: no cover
    mido = None


class MidiDeviceManager:
    """Lists and reuses MIDI output ports for the live sequencer."""

    def __init__(self):
        self._lock = threading.RLock()
        self._ports: Dict[str, object] = {}
        self._port_names: List[str] = []
        self.refresh_ports()

    def refresh_ports(self) -> List[str]:
        with self._lock:
            if mido is None:
                self._port_names = []
                return []
            try:
                self._port_names = list(mido.get_output_names())
            except Exception as exc:
                logging.warning("Could not list MIDI outputs: %s", exc)
                self._port_names = []
            return list(self._port_names)

    def get_output_names(self) -> List[str]:
        with self._lock:
            return list(self._port_names)

    def get_port(self, name: str):
        if not name:
            return None
        with self._lock:
            port = self._ports.get(name)
            if port is not None:
                return port
            if mido is None:
                logging.warning("MIDI output unavailable: mido is not installed")
                return None
            try:
                port = mido.open_output(name)
                self._ports[name] = port
                logging.info("Opened MIDI output: %s", name)
                return port
            except Exception as exc:
                logging.warning("Could not open MIDI output '%s': %s", name, exc)
                return None

    def send(self, port_name: str, message) -> bool:
        port = self.get_port(port_name)
        if port is None:
            return False
        try:
            port.send(message)
            return True
        except Exception as exc:
            logging.warning("MIDI send failed on '%s': %s", port_name, exc)
            self.close_port(port_name)
            return False

    def close_port(self, name: str) -> None:
        with self._lock:
            port = self._ports.pop(name, None)
        if port is not None:
            try:
                port.close()
                logging.info("Closed MIDI output: %s", name)
            except Exception:
                logging.debug("MIDI output close failed", exc_info=True)

    def close_all(self) -> None:
        with self._lock:
            names = list(self._ports.keys())
        for name in names:
            self.close_port(name)


def make_message(message_type: str, **kwargs):
    if mido is None:
        return None
    try:
        return mido.Message(message_type, **kwargs)
    except Exception:
        logging.debug("Could not create MIDI message", exc_info=True)
        return None
