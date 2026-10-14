"""Measure and validate the Rust UI protocol-v2 Python sidecar boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rust_ui_bridge import (  # noqa: E402
    PROTOCOL_VERSION,
    compile_preset_graph,
    deterministic_frame,
    downscale_preview_rgb,
    process_pipeline,
)


def invert_preset() -> dict[str, Any]:
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "InvertColorNode:1",
                "type": "InvertColorNode",
                "label": "Invert Colors",
                "position": [200.0, 100.0],
                "params": {"enabled": True, "invert_chance": 100},
            }
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Invert Colors",
                "to_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Invert Colors",
                "to_node": "DISPLAY",
                "from_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no samples")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _protocol_integer(value: Any, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RuntimeError(f"protocol field {name!r} must be an integer >= {minimum}")
    return value


def _protocol_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"protocol field {name!r} must be a non-empty string")
    return value


class JsonLineReader:
    """Drain child stdout without letting a blocking readline defeat deadlines."""

    def __init__(self, stream: TextIO) -> None:
        self._items: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.event_count = 0
        self._thread = threading.Thread(
            target=self._drain,
            args=(stream,),
            name="megaglitch-benchmark-jsonl",
            daemon=True,
        )
        self._thread.start()

    def _drain(self, stream: TextIO) -> None:
        try:
            for raw_line in stream:
                line = raw_line.rstrip("\r\n")
                if not line:
                    self._items.put(("error", RuntimeError("bridge emitted a blank stdout line")))
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._items.put(
                        (
                            "error",
                            RuntimeError(f"bridge emitted non-JSON stdout: {line[:240]!r}"),
                        )
                    )
                    continue
                if not isinstance(event, dict):
                    self._items.put(
                        (
                            "error",
                            RuntimeError("bridge emitted JSON that is not a protocol object"),
                        )
                    )
                    continue
                self._items.put(("event", event))
        except BaseException as error:  # pragma: no cover - OS pipe failure
            self._items.put(("error", error))
        finally:
            self._items.put(("eof", None))

    def read(self, deadline: float, context: str) -> dict[str, Any]:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            raise TimeoutError(f"timed out waiting for {context}")
        try:
            kind, value = self._items.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError(f"timed out waiting for {context}") from error
        if kind == "error":
            raise value
        if kind == "eof":
            raise RuntimeError(f"bridge stdout closed while waiting for {context}")
        if value.get("version") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"bridge event used protocol version {value.get('version')!r}; "
                f"expected {PROTOCOL_VERSION}"
            )
        self.event_count += 1
        return value

    def expect_eof(self, deadline: float) -> list[dict[str, Any]]:
        trailing: list[dict[str, Any]] = []
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                raise TimeoutError("timed out waiting for bridge stdout to close")
            try:
                kind, value = self._items.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("timed out waiting for bridge stdout to close") from error
            if kind == "error":
                raise value
            if kind == "eof":
                self._thread.join(timeout=max(0.0, deadline - time.perf_counter()))
                return trailing
            if value.get("version") != PROTOCOL_VERSION:
                raise RuntimeError(
                    f"trailing bridge event used protocol version {value.get('version')!r}"
                )
            self.event_count += 1
            trailing.append(value)


class TextDrain:
    """Continuously consume child stderr so diagnostics cannot fill its pipe."""

    def __init__(self, stream: TextIO) -> None:
        self.text = ""
        self._stream = stream
        self._thread = threading.Thread(
            target=self._drain,
            name="megaglitch-benchmark-stderr",
            daemon=True,
        )
        self._thread.start()

    def _drain(self) -> None:
        self.text = self._stream.read()

    def finish(self, timeout: float) -> str:
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("bridge stderr did not close")
        return self.text


class MappedPreview:
    """Validated read-only consumer for the bridge's fixed-slot RGB8 surface."""

    def __init__(self, descriptor: dict[str, Any]) -> None:
        if descriptor.get("type") != "preview_surface":
            raise RuntimeError(f"expected preview_surface, received {descriptor!r}")
        if descriptor.get("pixel_format") != "RGB8":
            raise RuntimeError("preview surface is not RGB8")
        self.path = Path(_protocol_string(descriptor.get("path"), "path"))
        if not self.path.is_absolute() or not self.path.is_file():
            raise RuntimeError("preview surface path is not an existing absolute file")
        self.generation = _protocol_integer(descriptor.get("generation"), "generation")
        self.slots = _protocol_integer(descriptor.get("slots"), "slots", 1)
        self.slot_bytes = _protocol_integer(descriptor.get("slot_bytes"), "slot_bytes", 1)
        self.max_width = _protocol_integer(descriptor.get("max_width"), "max_width", 1)
        self.max_height = _protocol_integer(descriptor.get("max_height"), "max_height", 1)
        self._file = self.path.open("rb")
        self.surface_bytes = self.path.stat().st_size
        required_bytes = self.slots * self.slot_bytes
        if self.surface_bytes < required_bytes:
            self._file.close()
            raise RuntimeError(
                f"preview surface has {self.surface_bytes} bytes; expected at least {required_bytes}"
            )
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._closed = False

    def read_rgb(self, event: dict[str, Any]) -> bytes:
        if self._closed:
            raise RuntimeError("preview mapping is closed")
        if event.get("type") != "preview_frame":
            raise RuntimeError("expected a preview_frame event")
        generation = _protocol_integer(event.get("generation"), "generation")
        if generation != self.generation:
            raise RuntimeError(
                f"preview generation changed from {self.generation} to {generation}"
            )
        slot = _protocol_integer(event.get("slot"), "slot")
        if slot >= self.slots:
            raise RuntimeError(f"preview slot {slot} is outside {self.slots} slots")
        width = _protocol_integer(event.get("width"), "width", 1)
        height = _protocol_integer(event.get("height"), "height", 1)
        stride = _protocol_integer(event.get("stride"), "stride", 1)
        nbytes = _protocol_integer(event.get("nbytes"), "nbytes", 1)
        if width > self.max_width or height > self.max_height:
            raise RuntimeError("preview frame exceeds the advertised surface dimensions")
        row_bytes = width * 3
        if stride < row_bytes:
            raise RuntimeError(f"preview stride {stride} is smaller than RGB row {row_bytes}")
        required_frame_bytes = (height - 1) * stride + row_bytes
        if nbytes < required_frame_bytes or nbytes > self.slot_bytes:
            raise RuntimeError("preview frame byte range is inconsistent with its dimensions")
        offset = slot * self.slot_bytes
        if offset + nbytes > len(self._mapping):
            raise RuntimeError("preview frame exceeds the mapped surface")
        if stride == row_bytes:
            return bytes(self._mapping[offset : offset + height * row_bytes])
        return b"".join(
            bytes(self._mapping[offset + row * stride : offset + row * stride + row_bytes])
            for row in range(height)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mapping.close()
        self._file.close()

    def __enter__(self) -> "MappedPreview":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def write_command(process: subprocess.Popen[str], command: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("bridge stdin is unavailable")
    process.stdin.write(json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n")
    process.stdin.flush()


def _validate_ready(ready: dict[str, Any]) -> dict[str, Any]:
    if ready.get("type") != "ready":
        raise RuntimeError(f"expected ready event, received {ready!r}")
    capabilities = ready.get("capabilities")
    if not isinstance(capabilities, dict):
        raise RuntimeError("ready event omitted capabilities")
    transactional = capabilities.get("transactional_commands")
    fire_and_forget = capabilities.get("fire_and_forget_commands")
    preview = capabilities.get("preview")
    required_transactions = {"replace_graph", "set_transport", "shutdown"}
    if not isinstance(transactional, list) or not required_transactions.issubset(transactional):
        raise RuntimeError(
            "bridge did not negotiate replace_graph, set_transport, and shutdown"
        )
    if not isinstance(fire_and_forget, list) or "release_preview_slot" not in fire_and_forget:
        raise RuntimeError("bridge did not negotiate preview-slot release")
    if not isinstance(preview, dict):
        raise RuntimeError("bridge did not negotiate preview capabilities")
    if preview.get("transport") != "mmap_fixed_slots" or preview.get("pixel_format") != "RGB8":
        raise RuntimeError("bridge did not negotiate the RGB8 mmap preview contract")
    source_kinds = capabilities.get("source_kinds")
    if not isinstance(source_kinds, list) or "test_pattern" not in source_kinds:
        raise RuntimeError("bridge did not advertise its deterministic test-pattern source")
    if ready.get("source_ready") is not True or ready.get("source_kind") != "test_pattern":
        raise RuntimeError("bridge did not start with a usable deterministic test pattern")
    return {
        "backend": _protocol_string(ready.get("backend"), "backend"),
        "build": _protocol_string(ready.get("build"), "build"),
        "transactional_commands": sorted(str(value) for value in transactional),
        "fire_and_forget_commands": sorted(str(value) for value in fire_and_forget),
        "source_kinds": sorted(str(value) for value in source_kinds),
        "preview": {
            "transport": preview["transport"],
            "pixel_format": preview["pixel_format"],
            "slots": _protocol_integer(preview.get("slots"), "preview.slots", 1),
            "slot_bytes": _protocol_integer(
                preview.get("slot_bytes"), "preview.slot_bytes", 1
            ),
            "max_width": _protocol_integer(preview.get("max_width"), "preview.max_width", 1),
            "max_height": _protocol_integer(
                preview.get("max_height"), "preview.max_height", 1
            ),
        },
    }


def _validate_ack(
    event: dict[str, Any],
    *,
    command_id: int,
    command_type: str,
    previous_revision: int,
) -> tuple[int, int, int]:
    if event.get("type") != "ack" or event.get("id") != command_id:
        raise RuntimeError(f"expected acknowledgement {command_id}, received {event!r}")
    if event.get("command_type") != command_type or event.get("accepted") is not True:
        raise RuntimeError(f"bridge rejected {command_type}: {event!r}")
    if event.get("error") is not None:
        raise RuntimeError(f"accepted {command_type} acknowledgement carried an error")
    revision = _protocol_integer(event.get("state_revision"), "state_revision")
    if revision != previous_revision + 1:
        raise RuntimeError(
            f"{command_type} advanced revision from {previous_revision} to {revision}"
        )
    generation = _protocol_integer(event.get("graph_generation"), "graph_generation")
    effective_sequence = _protocol_integer(
        event.get("effective_frame_sequence"), "effective_frame_sequence"
    )
    return revision, generation, effective_sequence


def _wait_for_ack(
    reader: JsonLineReader,
    process: subprocess.Popen[str],
    preview: MappedPreview | None,
    *,
    command_id: int,
    deadline: float,
    release_counter: list[int],
) -> dict[str, Any]:
    while True:
        event = reader.read(deadline, f"acknowledgement {command_id}")
        event_type = event.get("type")
        if event_type == "runtime_error":
            raise RuntimeError(f"bridge reported a runtime error: {event!r}")
        if event_type == "preview_frame":
            if preview is None:
                raise RuntimeError("bridge published a preview after transport quiesced")
            sequence = _protocol_integer(event.get("sequence"), "sequence", 1)
            preview.read_rgb(event)
            write_command(
                process,
                {
                    "version": PROTOCOL_VERSION,
                    "type": "release_preview_slot",
                    "slot": event["slot"],
                    "generation": event["generation"],
                    "sequence": sequence,
                },
            )
            release_counter[0] += 1
            continue
        if event_type == "ack":
            if event.get("id") != command_id:
                raise RuntimeError(f"received acknowledgement for unexpected command: {event!r}")
            return event


def _portable_python_metadata() -> dict[str, str]:
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "environment": "active dependency-capable interpreter",
    }


def run_benchmark(
    *,
    width: int,
    height: int,
    warmup: int,
    iterations: int,
    startup_timeout: float = 15.0,
    shutdown_timeout: float = 5.0,
) -> dict[str, Any]:
    if min(width, height, iterations) <= 0 or warmup < 0:
        raise ValueError("dimensions and iterations must be positive; warmup cannot be negative")

    preset = invert_preset()
    pipeline = compile_preset_graph(preset)
    direct_ms: list[float] = []
    for sequence in range(1, warmup + iterations + 1):
        started = time.perf_counter()
        frame = deterministic_frame(sequence, width, height)
        output, _ = process_pipeline(pipeline, frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        expected = 255 - frame
        np.testing.assert_array_equal(output, expected)
        if sequence > warmup:
            direct_ms.append(elapsed_ms)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(ROOT / "tools" / "rust_ui_bridge.py"),
            "--width",
            str(width),
            "--height",
            str(height),
            "--hash-mode",
            "sha256",
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("bridge pipes were not created")
    reader = JsonLineReader(process.stdout)
    stderr_drain = TextDrain(process.stderr)
    surface: MappedPreview | None = None
    surface_path: Path | None = None
    parity_failures: list[dict[str, Any]] = []
    preview_read_ms: list[float] = []
    sidecar_ms: list[float] = []
    release_counter = [0]
    last_snapshot: dict[str, Any] | None = None
    startup_started = time.perf_counter()
    trailing_events: list[dict[str, Any]] = []
    try:
        startup_deadline = startup_started + startup_timeout
        ready = reader.read(startup_deadline, "protocol-v2 readiness")
        negotiated = _validate_ready(ready)
        ready_ms = (time.perf_counter() - startup_started) * 1000.0
        state_revision = _protocol_integer(ready.get("state_revision"), "state_revision")
        initial_state_revision = state_revision
        initial_generation = _protocol_integer(
            ready.get("graph_generation"), "graph_generation"
        )

        surface_event = reader.read(startup_deadline, "preview surface")
        surface = MappedPreview(surface_event)
        surface_path = surface.path
        surface_ms = (time.perf_counter() - startup_started) * 1000.0
        preview_contract = negotiated["preview"]
        for name in ("slots", "slot_bytes", "max_width", "max_height"):
            if getattr(surface, name) != preview_contract[name]:
                raise RuntimeError(f"preview descriptor disagrees with ready capability {name}")
        if width > surface.max_width or height > surface.max_height:
            raise RuntimeError(
                f"benchmark dimensions {width}x{height} exceed preview capability "
                f"{surface.max_width}x{surface.max_height}"
            )

        graph_command_id = 1
        graph_started = time.perf_counter()
        write_command(
            process,
            {
                "version": PROTOCOL_VERSION,
                "id": graph_command_id,
                "expected_state_revision": state_revision,
                "type": "replace_graph",
                "graph": preset,
            },
        )
        graph_ack = _wait_for_ack(
            reader,
            process,
            surface,
            command_id=graph_command_id,
            deadline=startup_deadline,
            release_counter=release_counter,
        )
        graph_ack_ms = (time.perf_counter() - graph_started) * 1000.0
        state_revision, graph_generation, effective_sequence = _validate_ack(
            graph_ack,
            command_id=graph_command_id,
            command_type="replace_graph",
            previous_revision=state_revision,
        )
        if graph_generation != initial_generation + 1:
            raise RuntimeError("replace_graph did not advance the graph generation")

        preview_by_sequence: dict[int, dict[str, Any]] = {}
        seen_sequences: set[int] = set()
        required_frames = warmup + iterations
        frame_deadline = time.perf_counter() + max(15.0, required_frames / 10.0)
        while len(seen_sequences) < required_frames:
            event = reader.read(frame_deadline, "Invert preview and snapshot samples")
            event_type = event.get("type")
            if event_type == "runtime_error":
                raise RuntimeError(f"bridge reported a runtime error: {event!r}")
            if event_type == "preview_frame":
                sequence = _protocol_integer(event.get("sequence"), "sequence", 1)
                read_started = time.perf_counter()
                preview_rgb = surface.read_rgb(event)
                preview_read_ms.append((time.perf_counter() - read_started) * 1000.0)
                expected_output = 255 - deterministic_frame(sequence, width, height)
                expected_preview = downscale_preview_rgb(expected_output)
                expected_preview_hash = hashlib.sha256(expected_preview.tobytes()).hexdigest()
                actual_preview_hash = hashlib.sha256(preview_rgb).hexdigest()
                preview_by_sequence[sequence] = {
                    "expected": expected_preview_hash,
                    "actual": actual_preview_hash,
                    "matches": actual_preview_hash == expected_preview_hash,
                }
                write_command(
                    process,
                    {
                        "version": PROTOCOL_VERSION,
                        "type": "release_preview_slot",
                        "slot": event["slot"],
                        "generation": event["generation"],
                        "sequence": sequence,
                    },
                )
                release_counter[0] += 1
                continue
            if event_type != "snapshot":
                if event_type == "ack":
                    raise RuntimeError(f"received an unexpected acknowledgement: {event!r}")
                continue
            last_snapshot = event
            if event.get("graph_generation") != graph_generation:
                continue
            sequence = _protocol_integer(event.get("graph_sequence"), "graph_sequence", 1)
            if sequence < effective_sequence or sequence in seen_sequences:
                continue
            preview_sample = preview_by_sequence.pop(sequence, None)
            if preview_sample is None:
                raise RuntimeError(f"snapshot {sequence} had no corresponding mmap preview frame")
            expected_output = 255 - deterministic_frame(sequence, width, height)
            expected_output_hash = hashlib.sha256(expected_output.tobytes()).hexdigest()
            snapshot_hash = _protocol_string(event.get("output_sha256"), "output_sha256")
            if snapshot_hash != expected_output_hash:
                parity_failures.append(
                    {
                        "sequence": sequence,
                        "layer": "snapshot",
                        "expected": expected_output_hash,
                        "actual": snapshot_hash,
                    }
                )
            if not preview_sample["matches"]:
                parity_failures.append(
                    {
                        "sequence": sequence,
                        "layer": "mmap_preview",
                        "expected": preview_sample["expected"],
                        "actual": preview_sample["actual"],
                    }
                )
            seen_sequences.add(sequence)
            if len(seen_sequences) > warmup:
                sidecar_ms.append(float(event["frame_ms"]))

        stop_command_id = 2
        stop_started = time.perf_counter()
        write_command(
            process,
            {
                "version": PROTOCOL_VERSION,
                "id": stop_command_id,
                "expected_state_revision": state_revision,
                "type": "set_transport",
                "running": False,
            },
        )
        stop_ack = _wait_for_ack(
            reader,
            process,
            surface,
            command_id=stop_command_id,
            deadline=time.perf_counter() + shutdown_timeout,
            release_counter=release_counter,
        )
        stop_ack_ms = (time.perf_counter() - stop_started) * 1000.0
        state_revision, stopped_generation, _ = _validate_ack(
            stop_ack,
            command_id=stop_command_id,
            command_type="set_transport",
            previous_revision=state_revision,
        )
        if stopped_generation != graph_generation:
            raise RuntimeError("set_transport unexpectedly changed the graph generation")

        # Windows cannot unlink the producer-owned file while this consumer map
        # remains live. Quiesce first, then unmap before revisioned shutdown.
        surface.close()
        surface = None

        shutdown_command_id = 3
        shutdown_started = time.perf_counter()
        write_command(
            process,
            {
                "version": PROTOCOL_VERSION,
                "id": shutdown_command_id,
                "expected_state_revision": state_revision,
                "type": "shutdown",
            },
        )
        shutdown_ack = _wait_for_ack(
            reader,
            process,
            None,
            command_id=shutdown_command_id,
            deadline=time.perf_counter() + shutdown_timeout,
            release_counter=release_counter,
        )
        shutdown_ack_ms = (time.perf_counter() - shutdown_started) * 1000.0
        state_revision, shutdown_generation, _ = _validate_ack(
            shutdown_ack,
            command_id=shutdown_command_id,
            command_type="shutdown",
            previous_revision=state_revision,
        )
        if shutdown_generation != graph_generation:
            raise RuntimeError("shutdown unexpectedly changed the graph generation")
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=shutdown_timeout)
        trailing_events = reader.expect_eof(time.perf_counter() + shutdown_timeout)
        stderr_text = stderr_drain.finish(shutdown_timeout)
        if process.returncode != 0:
            raise RuntimeError(f"bridge exited with code {process.returncode}")
        if stderr_text:
            raise RuntimeError(f"bridge wrote to stderr: {stderr_text[:1000]}")
        if surface_path is None:
            raise RuntimeError("bridge did not advertise a preview surface")
        cleanup_deadline = time.perf_counter() + shutdown_timeout
        while surface_path.exists() and time.perf_counter() < cleanup_deadline:
            time.sleep(0.01)
        surface_removed = not surface_path.exists()
        if not surface_removed:
            raise RuntimeError("bridge exited without removing its preview surface")

        direct_mean = statistics.fmean(direct_ms)
        sidecar_mean = statistics.fmean(sidecar_ms)
        performance_limit_ms = max(direct_mean * 2.0, direct_mean + 1.0)
        final_processing_drops = int((last_snapshot or {}).get("processing_drops", 0))
        final_preview_drops = int((last_snapshot or {}).get("preview_drops", 0))
        gates = {
            "protocol_v2": PROTOCOL_VERSION == 2,
            "capabilities": True,
            "snapshot_output_parity": not any(
                failure["layer"] == "snapshot" for failure in parity_failures
            ),
            "mmap_preview_parity": not any(
                failure["layer"] == "mmap_preview" for failure in parity_failures
            ),
            "performance": sidecar_mean <= performance_limit_ms,
            "slot_release": release_counter[0] >= required_frames,
            "no_processing_drops": final_processing_drops == 0,
            "no_preview_drops": final_preview_drops == 0,
            "revisioned_shutdown": state_revision == initial_state_revision + 3,
            "orderly_shutdown": process.returncode == 0,
            "stderr_clean": stderr_text == "",
            "json_only_stdout": True,
            "no_trailing_events": not trailing_events,
            "surface_removed": surface_removed,
        }
        return {
            "schema_version": 2,
            "benchmark": "rust-ui-python-sidecar",
            "protocol_version": PROTOCOL_VERSION,
            "python": _portable_python_metadata(),
            "configuration": {
                "effect": "Invert Colors",
                "width": width,
                "height": height,
                "warmup": warmup,
                "iterations": iterations,
                "source": "deterministic_test_pattern",
                "preview_transport": "mmap_fixed_slots",
                "pixel_format": "RGB8",
            },
            "negotiated": negotiated,
            "direct": {
                "scope": "test-pattern generation plus execute_flat_node_pipeline",
                "mean_ms": direct_mean,
                "p99_ms": percentile(direct_ms, 0.99),
                "fps_from_mean": 1000.0 / direct_mean,
                "reported_samples": len(direct_ms),
            },
            "sidecar": {
                "scope": "bridge-reported source read plus graph processing",
                "mean_ms": sidecar_mean,
                "p99_ms": percentile(sidecar_ms, 0.99),
                "fps_from_mean": 1000.0 / sidecar_mean,
                "replace_graph_ack_ms": graph_ack_ms,
                "performance_limit_ms": performance_limit_ms,
                "reported_samples": len(sidecar_ms),
                "processing_drops": final_processing_drops,
                "preview_drops": final_preview_drops,
            },
            "preview": {
                "mean_mmap_read_ms": statistics.fmean(preview_read_ms),
                "p99_mmap_read_ms": percentile(preview_read_ms, 0.99),
                "mapped_samples": len(preview_read_ms),
                "slot_releases": release_counter[0],
                "surface_bytes": negotiated["preview"]["slots"]
                * negotiated["preview"]["slot_bytes"],
            },
            "lifecycle": {
                "ready_ms": ready_ms,
                "preview_surface_ms": surface_ms,
                "transport_stop_ack_ms": stop_ack_ms,
                "shutdown_ack_ms": shutdown_ack_ms,
                "shutdown_revision": state_revision,
                "exit_code": process.returncode,
                "stderr_bytes": len(stderr_text.encode("utf-8")),
                "stdout_events": reader.event_count,
                "trailing_events": len(trailing_events),
                "surface_removed": surface_removed,
            },
            "gates": gates,
            "parity_failures": parity_failures,
        }
    finally:
        if surface is not None:
            surface.close()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=shutdown_timeout)
        try:
            stderr_drain.finish(1.0)
        except TimeoutError:
            pass
        if surface_path is not None and surface_path.exists() and process.poll() is not None:
            # Failed runs must not leak a temporary surface. Passing cleanup is
            # measured before this best-effort fallback removal.
            try:
                surface_path.unlink()
            except OSError:
                pass


def write_results(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rust-ui-bridge.json"
    csv_path = output_dir / "rust-ui-bridge.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "mean_ms", "p99_ms", "fps_from_mean", "samples"])
        writer.writerow(
            [
                "direct",
                result["direct"]["mean_ms"],
                result["direct"]["p99_ms"],
                result["direct"]["fps_from_mean"],
                result["direct"]["reported_samples"],
            ]
        )
        writer.writerow(
            [
                "sidecar",
                result["sidecar"]["mean_ms"],
                result["sidecar"]["p99_ms"],
                result["sidecar"]["fps_from_mean"],
                result["sidecar"]["reported_samples"],
            ]
        )
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--shutdown-timeout", type=float, default=5.0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "tools" / "benchmark" / "results"
    )
    args = parser.parse_args()
    if min(args.width, args.height, args.iterations) <= 0 or args.warmup < 0:
        parser.error("dimensions and iterations must be positive; warmup cannot be negative")
    if min(args.startup_timeout, args.shutdown_timeout) <= 0.0:
        parser.error("timeouts must be positive")

    result = run_benchmark(
        width=args.width,
        height=args.height,
        warmup=args.warmup,
        iterations=args.iterations,
        startup_timeout=args.startup_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    json_path, csv_path = write_results(result, args.output_dir)
    print(json.dumps(result, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
