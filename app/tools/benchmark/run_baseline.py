"""Reproducible MegaGlitch frame-path baseline.

Bootstrap OpenCV before importing any project module that imports cv2. Results are
written as JSON and flattened CSV. Unsupported capabilities are reported rather
than silently replaced with a different implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESOLUTIONS = ((1280, 720), (1920, 1080), (1200, 800))
HISTORY_LENGTHS = (1, 8, 32, 64, 120)
REPRESENTATIVE_PRESET = ROOT / "node_presets" / "6-20-26.json"
DEFAULT_OPENCV_BIN = r"C:\src\opencv-cuda\build_py312\bin\Release"
DEFAULT_OPENCV_PYTHON = r"C:\src\opencv-cuda\build_py312\lib\python3\Release"
DEFAULT_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"


def _echo_worker(in_queue, out_queue):
    while True:
        item = in_queue.get()
        if item is None:
            return
        out_queue.put(item)


def _load_worker_cv2():
    """Load the custom cv2 binding in a fresh spawned process."""
    opencv_bin = os.environ.get("MEGAGLITCH_OPENCV_BIN", DEFAULT_OPENCV_BIN)
    opencv_python = os.environ.get("MEGAGLITCH_OPENCV_PYTHON", DEFAULT_OPENCV_PYTHON)
    cuda_bin = os.environ.get("MEGAGLITCH_CUDA_BIN", DEFAULT_CUDA_BIN)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(opencv_bin)
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)
    if opencv_python not in sys.path:
        sys.path.insert(0, opencv_python)
    import cv2

    return cv2


def _cuda_probe_worker(out_queue):
    try:
        cv2 = _load_worker_cv2()
        count = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
        out_queue.put(
            {
                "opencv_path": str(getattr(cv2, "__file__", "")),
                "opencv_version": str(getattr(cv2, "__version__", "unknown")),
                "cuda_device_count": count,
                "cuda_available": count > 0,
                "cuda_events_available": bool(
                    hasattr(cv2, "cuda_Event") and hasattr(cv2.cuda, "Event_elapsedTime")
                ),
                "error": None,
            }
        )
    except Exception as exc:
        out_queue.put({"cuda_available": False, "cuda_device_count": 0, "error": repr(exc)})


def _gpu_micro_worker(out_queue, resolutions, warmup: int, iterations: int):
    try:
        cv2 = _load_worker_cv2()
        import numpy as np

        if int(cv2.cuda.getCudaEnabledDeviceCount()) <= 0:
            out_queue.put({"error": "No CUDA device in spawned worker", "cases": []})
            return
        cv2.cuda.setDevice(0)
        stream = cv2.cuda_Stream()

        def elapsed_event(operation):
            start = cv2.cuda_Event()
            end = cv2.cuda_Event()
            host_started = time.perf_counter()
            start.record(stream)
            result = operation()
            end.record(stream)
            end.waitForCompletion()
            host_ms = (time.perf_counter() - host_started) * 1000.0
            event_ms = float(cv2.cuda.Event_elapsedTime(start, end))
            return result, host_ms, event_ms

        cases = []
        for width, height in resolutions:
            frame = _base_frame(np, width, height)
            gpu_input = cv2.cuda_GpuMat()
            gpu_input.upload(frame, stream)
            stream.waitForCompletion()
            gaussian = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.0
            )
            for _ in range(warmup):
                warmed = gaussian.apply(gpu_input, stream=stream)
                _ = warmed.download(stream)
                stream.waitForCompletion()

            upload_events = []
            upload_hosts = []
            for _ in range(iterations):
                target = cv2.cuda_GpuMat()
                _, host_ms, event_ms = elapsed_event(lambda target=target: target.upload(frame, stream))
                upload_hosts.append(host_ms)
                upload_events.append(event_ms)

            kernel_events = []
            kernel_hosts = []
            last_output = None
            for _ in range(iterations):
                last_output, host_ms, event_ms = elapsed_event(
                    lambda: gaussian.apply(gpu_input, stream=stream)
                )
                kernel_hosts.append(host_ms)
                kernel_events.append(event_ms)

            download_events = []
            download_hosts = []
            for _ in range(iterations):
                _, host_ms, event_ms = elapsed_event(lambda: last_output.download(stream))
                download_hosts.append(host_ms)
                download_events.append(event_ms)

            hybrid_total = []
            hybrid_event = []
            for _ in range(iterations):
                def hybrid():
                    uploaded = cv2.cuda_GpuMat()
                    uploaded.upload(frame, stream)
                    filtered = gaussian.apply(uploaded, stream=stream)
                    return filtered.download(stream)

                _, host_ms, event_ms = elapsed_event(hybrid)
                hybrid_total.append(host_ms)
                hybrid_event.append(event_ms)

            resident_total = []
            resident_event = []
            for _ in range(iterations):
                _, host_ms, event_ms = elapsed_event(
                    lambda: gaussian.apply(gpu_input, stream=stream)
                )
                resident_total.append(host_ms)
                resident_event.append(event_ms)

            common = {
                "status": "measured",
                "category": "cuda",
                "width": width,
                "height": height,
                "format": "rgb8",
                "frames": iterations,
                "warmup_frames": warmup,
                "frame_bytes": int(frame.nbytes),
                "dropped_frames": 0,
            }
            cases.extend(
                [
                    {
                        **common,
                        "name": "cuda_upload",
                        "mean_frame_ms": statistics.fmean(upload_hosts),
                        "cuda_event_mean_ms": statistics.fmean(upload_events),
                        "upload_count": iterations,
                        "upload_bytes": int(frame.nbytes) * iterations,
                    },
                    {
                        **common,
                        "name": "cuda_gaussian_kernel_resident",
                        "mean_frame_ms": statistics.fmean(kernel_hosts),
                        "cuda_event_mean_ms": statistics.fmean(kernel_events),
                    },
                    {
                        **common,
                        "name": "cuda_download",
                        "mean_frame_ms": statistics.fmean(download_hosts),
                        "cuda_event_mean_ms": statistics.fmean(download_events),
                        "download_count": iterations,
                        "download_bytes": int(frame.nbytes) * iterations,
                    },
                    {
                        **common,
                        "name": "cuda_hybrid_upload_kernel_download",
                        **_summarize_times(hybrid_total),
                        "cuda_event_mean_ms": statistics.fmean(hybrid_event),
                        "upload_count": iterations,
                        "upload_bytes": int(frame.nbytes) * iterations,
                        "download_count": iterations,
                        "download_bytes": int(frame.nbytes) * iterations,
                    },
                    {
                        **common,
                        "name": "cuda_resident_kernel_no_download",
                        **_summarize_times(resident_total),
                        "cuda_event_mean_ms": statistics.fmean(resident_event),
                        "upload_count": 0,
                        "upload_bytes": 0,
                        "download_count": 0,
                        "download_bytes": 0,
                    },
                ]
            )
        out_queue.put({"error": None, "cases": cases})
    except Exception as exc:
        out_queue.put({"error": repr(exc), "cases": []})


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_times(times_ms: list[float]) -> dict[str, float]:
    total_ms = sum(times_ms)
    mean_ms = statistics.fmean(times_ms) if times_ms else 0.0
    p99_ms = _percentile(times_ms, 0.99)
    return {
        "frames": len(times_ms),
        "elapsed_ms": total_ms,
        "mean_frame_ms": mean_ms,
        "median_frame_ms": statistics.median(times_ms) if times_ms else 0.0,
        "p95_frame_ms": _percentile(times_ms, 0.95),
        "p99_frame_ms": p99_ms,
        "max_frame_ms": max(times_ms) if times_ms else 0.0,
        "average_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "one_percent_low_fps": 1000.0 / p99_ms if p99_ms > 0 else 0.0,
    }


def _process_memory() -> dict[str, int | None]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {
            "rss_bytes": int(info.rss),
            "peak_working_set_bytes": int(getattr(info, "peak_wset", 0)) or None,
            "system_available_bytes": int(psutil.virtual_memory().available),
        }
    except Exception:
        return {"rss_bytes": None, "peak_working_set_bytes": None, "system_available_bytes": None}


def _nvidia_snapshot() -> dict[str, Any] | None:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        line = completed.stdout.strip().splitlines()[0]
        name, utilization, used_mib, total_mib = [part.strip() for part in line.split(",", 3)]
        return {
            "name": name,
            "utilization_percent": float(utilization),
            "memory_used_mib": float(used_mib),
            "memory_total_mib": float(total_mib),
        }
    except Exception:
        return None


def _bootstrap_runtime() -> tuple[Any, dict[str, Any]]:
    bootstrap_error = None
    try:
        from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults

        bootstrap_for_dev_defaults()
    except Exception as exc:
        bootstrap_error = str(exc)

    import cv2

    cuda_devices = 0
    cuda_error = None
    try:
        cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    except Exception as exc:
        cuda_error = str(exc)

    capabilities = {
        "bootstrap_error": bootstrap_error,
        "opencv_path": str(getattr(cv2, "__file__", "")),
        "opencv_version": str(getattr(cv2, "__version__", "unknown")),
        "cuda_namespace": bool(hasattr(cv2, "cuda")),
        "cuda_device_count": cuda_devices,
        "cuda_error": cuda_error,
        "cuda_available": cuda_devices > 0,
        "parent_cuda_available": cuda_devices > 0,
        "worker_cuda_available": False,
        "supported_frame_formats": ["rgb8"],
        "unsupported_frame_formats": ["rgba16f", "rgba32f"],
        "gpu_history_available": False,
        "virtual_camera_benchmark_available": False,
        "preview_measurement_scope": "CPU preparation only; Dear PyGui upload/render excluded",
    }
    return cv2, capabilities


def _probe_cuda_worker() -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_cuda_probe_worker, args=(queue,))
    process.start()
    try:
        result = queue.get(timeout=30)
    finally:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
    return result


def _run_gpu_microbenchmarks(resolutions, warmup: int, iterations: int) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_gpu_micro_worker,
        args=(queue, tuple(resolutions), warmup, iterations),
    )
    process.start()
    try:
        result = queue.get(timeout=120)
    finally:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
    return result


def _base_frame(np, width: int, height: int):
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    return frame


def _minimal_pipeline() -> list[dict[str, Any]]:
    return [
        {"effect_type": "Source", "params": {}, "node_type": "source", "source_key": "primary"},
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]


def _noop_pipeline(node_count: int) -> list[dict[str, Any]]:
    pipeline = _minimal_pipeline()[:-1]
    for index in range(node_count):
        pipeline.append(
            {
                "effect_type": "Pixelation / Chromatic",
                "params": {"enabled": False},
                "node_type": "effect",
                "node_id": f"noop:{index}",
            }
        )
    pipeline.append(_minimal_pipeline()[-1])
    return pipeline


def _history_pipeline(history_length: int) -> list[dict[str, Any]]:
    return [
        {"effect_type": "Source", "params": {}, "node_type": "source", "source_key": "primary"},
        {
            "effect_type": "Row Temporal Stutter",
            "params": {
                "enabled": True,
                "row_stutter": 0.75,
                # Match the product node's normal delay range while varying the
                # retained deck-wide history independently.
                # The current handler treats zero as the fallback value four
                # (`value or 4`), so use one for a one-frame history.
                "row_stutter_max_delay": max(1, min(history_length - 1, 4)),
            },
            "node_type": "effect",
            "node_id": "benchmark:row_stutter",
        },
        {"effect_type": "Display", "params": {}, "node_type": "display"},
    ]


def _representative_pipeline() -> tuple[list[dict[str, Any]], str]:
    data = json.loads(REPRESENTATIVE_PRESET.read_text(encoding="utf-8"))
    deck = data["decks"]["a"]
    pipeline = _minimal_pipeline()[:-1]
    for index, node in enumerate(deck.get("nodes", [])):
        pipeline.append(
            {
                "effect_type": node.get("label", node.get("type", "Unknown")),
                "params": dict(node.get("params", {})),
                "node_type": "effect",
                "node_id": node.get("id", f"preset:{index}"),
            }
        )
    pipeline.append(_minimal_pipeline()[-1])
    return pipeline, str(REPRESENTATIVE_PRESET.relative_to(ROOT))


class TransferCounter:
    def __init__(self, gpu_ops):
        self.gpu_ops = gpu_ops
        self.original_upload = gpu_ops.upload
        self.original_download = gpu_ops.download
        self.reset()

    def reset(self):
        self.upload_count = 0
        self.upload_bytes = 0
        self.download_count = 0
        self.download_bytes = 0
        self.upload_ms = 0.0
        self.download_ms = 0.0

    def __enter__(self):
        def upload(value):
            started = time.perf_counter()
            result = self.original_upload(value)
            self.upload_ms += (time.perf_counter() - started) * 1000.0
            self.upload_count += 1
            self.upload_bytes += int(getattr(value, "nbytes", 0))
            return result

        def download(value):
            started = time.perf_counter()
            result = self.original_download(value)
            self.download_ms += (time.perf_counter() - started) * 1000.0
            self.download_count += 1
            self.download_bytes += int(getattr(result, "nbytes", 0))
            return result

        self.gpu_ops.upload = upload
        self.gpu_ops.download = download
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.gpu_ops.upload = self.original_upload
        self.gpu_ops.download = self.original_download

    def result(self):
        return {
            "upload_count": self.upload_count,
            "upload_bytes": self.upload_bytes,
            "upload_ms": self.upload_ms,
            "download_count": self.download_count,
            "download_bytes": self.download_bytes,
            "download_ms": self.download_ms,
        }


def _run_pipeline_case(np, npr, frame_state_cls, gpu_ops, *, name: str, pipeline, width: int,
                       height: int, history_length: int, warmup: int, iterations: int,
                       preset: str | None = None) -> dict[str, Any]:
    frame = _base_frame(np, width, height)
    state = frame_state_cls()
    state.previous_frame = np.zeros_like(frame)
    state.feedback_image = np.zeros_like(frame)
    state.trail_image = np.zeros_like(frame)
    state.frame_buffer = [frame.copy() for _ in range(history_length)]
    state.buffer_limit = history_length
    node_times: dict[str, list[float]] = {}
    original_apply = npr.apply_node_effect

    def timed_apply(label, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_apply(label, *args, **kwargs)
        finally:
            node_times.setdefault(str(label), []).append((time.perf_counter() - started) * 1000.0)

    def process_one(measure_history: bool):
        started = time.perf_counter()
        output, state.previous_frame, state.feedback_image, state.trail_image = npr.execute_flat_node_pipeline(
            pipeline,
            frame,
            extra_params={"time_accum": state.frame_count / 60.0, "frame_num": state.frame_count},
            previous_frame=state.previous_frame,
            feedback_image=state.feedback_image,
            trail_image=state.trail_image,
            frame_buffer=state.frame_buffer,
            worker_state=state,
            fallback_size=(width, height),
        )
        pipeline_ms = (time.perf_counter() - started) * 1000.0
        history_started = time.perf_counter()
        state.frame_buffer.append(output.copy())
        while len(state.frame_buffer) > history_length:
            state.frame_buffer.pop(0)
        history_ms = (time.perf_counter() - history_started) * 1000.0
        state.frame_count += 1
        return output, pipeline_ms, history_ms if measure_history else 0.0

    npr.apply_node_effect = timed_apply
    try:
        for _ in range(warmup):
            process_one(False)
        node_times.clear()
        transfer_counter = TransferCounter(gpu_ops)
        memory_before = _process_memory()
        wall_times = []
        pipeline_times = []
        history_times = []
        cpu_started = time.process_time()
        with transfer_counter:
            for _ in range(iterations):
                wall_started = time.perf_counter()
                output, pipeline_ms, history_ms = process_one(True)
                wall_times.append((time.perf_counter() - wall_started) * 1000.0)
                pipeline_times.append(pipeline_ms)
                history_times.append(history_ms)
        cpu_ms = (time.process_time() - cpu_started) * 1000.0
        timed_node_means = {key: statistics.fmean(values) for key, values in node_times.items() if values}

        # Allocation tracing is intentionally a separate one-frame sample so
        # tracemalloc overhead does not contaminate the frame-time baseline.
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        output, _, _ = process_one(False)
        snapshot_after = tracemalloc.take_snapshot()
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        allocations = snapshot_after.compare_to(snapshot_before, "lineno")
        allocation_count = sum(max(0, item.count_diff) for item in allocations)
        allocation_bytes = sum(max(0, item.size_diff) for item in allocations)
        memory_after = _process_memory()
    finally:
        npr.apply_node_effect = original_apply

    lookup_repeats = max(100, min(2000, 200000 // max(1, history_length)))
    lookup_started = time.perf_counter()
    for index in range(lookup_repeats):
        _ = state.frame_buffer[-1 - (index % len(state.frame_buffer))]
    direct_lookup_us = (time.perf_counter() - lookup_started) * 1_000_000.0 / lookup_repeats

    from effects_registry import _sample_temporal_history

    sample_started = time.perf_counter()
    for index in range(min(100, lookup_repeats)):
        _sample_temporal_history(state.frame_buffer, state.previous_frame, float(index % history_length), (height, width))
    sampled_lookup_ms = (time.perf_counter() - sample_started) * 1000.0 / min(100, lookup_repeats)

    result = {
        "name": name,
        "status": "measured",
        "category": "pipeline",
        "width": width,
        "height": height,
        "format": "rgb8",
        "history_backend": "python_list",
        "history_length": history_length,
        "warmup_frames": warmup,
        **_summarize_times(wall_times),
        "pipeline_mean_ms": statistics.fmean(pipeline_times),
        "history_write_mean_ms": statistics.fmean(history_times),
        "direct_history_lookup_us": direct_lookup_us,
        "sampled_history_lookup_mean_ms": sampled_lookup_ms,
        "cpu_time_ms": cpu_ms,
        "cpu_time_per_frame_ms": cpu_ms / iterations,
        "cpu_to_wall_ratio": cpu_ms / max(sum(wall_times), 1e-9),
        "python_allocation_count_delta": allocation_count,
        "python_allocation_bytes_delta": allocation_bytes,
        "allocation_sample_frames": 1,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "rss_before_bytes": memory_before["rss_bytes"],
        "rss_after_bytes": memory_after["rss_bytes"],
        "rss_delta_bytes": (
            memory_after["rss_bytes"] - memory_before["rss_bytes"]
            if memory_before["rss_bytes"] is not None and memory_after["rss_bytes"] is not None
            else None
        ),
        "peak_working_set_bytes": memory_after["peak_working_set_bytes"],
        "output_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(output))).hexdigest(),
        "output_bytes": int(output.nbytes),
        "estimated_history_write_bytes_per_frame": int(output.nbytes),
        "dropped_frames": 0,
        "preset": preset,
        "node_mean_ms": timed_node_means,
        **transfer_counter.result(),
    }
    return result


def _run_preallocated_ring_case(np, width: int, height: int, capacity: int, warmup: int,
                                iterations: int) -> dict[str, Any]:
    if capacity > 90:
        return {
            "name": "preallocated_cpu_ring",
            "status": "unsupported",
            "reason": "Existing CUDA Temporal Slit-Scan clamps history capacity to 90",
            "category": "history",
            "width": width,
            "height": height,
            "format": "rgb8",
            "history_backend": "preallocated_cpu_ring",
            "history_length": capacity,
        }
    frame = _base_frame(np, width, height)
    memory_before = _process_memory()
    ring = np.empty((capacity, height, width, 3), dtype=np.uint8)
    for index in range(capacity):
        ring[index] = frame
    write_index = 0
    for _ in range(warmup):
        ring[write_index] = frame
        write_index = (write_index + 1) % capacity
    times = []
    lookup_times = []
    cpu_started = time.process_time()
    for iteration in range(iterations):
        started = time.perf_counter()
        ring[write_index] = frame
        write_index = (write_index + 1) % capacity
        lookup_started = time.perf_counter()
        delayed = ring[(write_index + capacity - 1 - (iteration % capacity)) % capacity]
        lookup_times.append((time.perf_counter() - lookup_started) * 1_000_000.0)
        _ = delayed[0, 0, 0]
        times.append((time.perf_counter() - started) * 1000.0)
    cpu_ms = (time.process_time() - cpu_started) * 1000.0
    memory_after = _process_memory()
    return {
        "name": "preallocated_cpu_ring",
        "status": "measured",
        "category": "history",
        "width": width,
        "height": height,
        "format": "rgb8",
        "history_backend": "preallocated_cpu_ring",
        "history_length": capacity,
        "warmup_frames": warmup,
        **_summarize_times(times),
        "history_write_mean_ms": statistics.fmean(times),
        "direct_history_lookup_us": statistics.fmean(lookup_times),
        "cpu_time_ms": cpu_ms,
        "cpu_time_per_frame_ms": cpu_ms / iterations,
        "ring_bytes": int(ring.nbytes),
        "rss_before_bytes": memory_before["rss_bytes"],
        "rss_after_bytes": memory_after["rss_bytes"],
        "rss_delta_bytes": (
            memory_after["rss_bytes"] - memory_before["rss_bytes"]
            if memory_before["rss_bytes"] is not None and memory_after["rss_bytes"] is not None
            else None
        ),
        "dropped_frames": 0,
    }


def _run_ipc_case(np, width: int, height: int, payload_frames: int, warmup: int,
                  iterations: int) -> dict[str, Any]:
    frame = _base_frame(np, width, height)
    payload = {"frame": frame, "pipeline": _minimal_pipeline()}
    if payload_frames >= 3:
        payload["frame_a"] = frame.copy()
        payload["frame_b"] = frame.copy()
    pickle_bytes = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    context = mp.get_context("spawn")
    in_queue = context.Queue(maxsize=2)
    out_queue = context.Queue(maxsize=2)
    process = context.Process(target=_echo_worker, args=(in_queue, out_queue))
    process.start()
    try:
        for _ in range(warmup):
            in_queue.put(payload)
            out_queue.get(timeout=30)
        times = []
        cpu_started = time.process_time()
        for _ in range(iterations):
            started = time.perf_counter()
            in_queue.put(payload)
            out_queue.get(timeout=30)
            times.append((time.perf_counter() - started) * 1000.0)
        cpu_ms = (time.process_time() - cpu_started) * 1000.0
    finally:
        in_queue.put(None)
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        in_queue.close()
        out_queue.close()
    return {
        "name": f"ipc_roundtrip_{payload_frames}_frame_payload",
        "status": "measured",
        "category": "ipc",
        "width": width,
        "height": height,
        "format": "rgb8",
        "payload_frames": payload_frames,
        "serialized_payload_bytes": pickle_bytes,
        "warmup_frames": warmup,
        **_summarize_times(times),
        "cpu_time_ms": cpu_ms,
        "cpu_time_per_frame_ms": cpu_ms / iterations,
        "dropped_frames": 0,
    }


def _run_worker_roundtrip_case(np, pipeline, width: int, height: int, warmup: int,
                               iterations: int, name: str, use_cuda: bool = False) -> dict[str, Any]:
    """Measure the existing mp.Queue -> real worker -> mp.Queue path."""
    from megaglitchfm_app_new import glitch_process_worker

    frame = _base_frame(np, width, height)
    context = mp.get_context("spawn")
    in_queue = context.Queue(maxsize=2)
    out_queue = context.Queue(maxsize=2)
    stop_event = context.Event()
    process = context.Process(
        target=glitch_process_worker,
        args=(in_queue, out_queue, stop_event, None),
    )
    process.start()

    def payload(frame_number: int):
        return {
            "image": frame,
            "secondary_image": None,
            "pipeline": pipeline,
            "pipeline_b": [],
            "master_mix": 0.0,
            "master_transition": "Normal Mix",
            "extra_params": {
                "time_accum": frame_number / 60.0,
                "frame_num": frame_number,
            },
            "use_cuda": bool(use_cuda),
            "fallback_width": width,
            "fallback_height": height,
            "timestamp": time.perf_counter(),
        }

    times = []
    output = None
    try:
        for index in range(warmup):
            in_queue.put(payload(index), timeout=30)
            output = out_queue.get(timeout=60)
        cpu_started = time.process_time()
        for index in range(iterations):
            item = payload(warmup + index)
            started = time.perf_counter()
            in_queue.put(item, timeout=30)
            output = out_queue.get(timeout=60)
            times.append((time.perf_counter() - started) * 1000.0)
        parent_cpu_ms = (time.process_time() - cpu_started) * 1000.0
    finally:
        stop_event.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        in_queue.close()
        out_queue.close()

    output_frame = output.get("frame") if isinstance(output, dict) else output
    return {
        "name": name,
        "status": "measured",
        "category": "end_to_end",
        "width": width,
        "height": height,
        "format": "rgb8",
        "cuda_requested": bool(use_cuda),
        "scope": "Existing input queue, worker graph/history/deck blend, and output queue; capture and preview excluded",
        "warmup_frames": warmup,
        **_summarize_times(times),
        "parent_cpu_time_ms": parent_cpu_ms,
        "parent_cpu_time_per_frame_ms": parent_cpu_ms / iterations,
        "output_sha256": (
            hashlib.sha256(memoryview(np.ascontiguousarray(output_frame))).hexdigest()
            if output_frame is not None else None
        ),
        "dropped_frames": 0,
    }


def _run_preview_case(np, cv2, width: int, height: int, monitors: int, warmup: int,
                      iterations: int) -> dict[str, Any]:
    frame = _base_frame(np, width, height)
    display_width, display_height = 640, 360
    canvases = [
        (
            np.empty((display_height, display_width, 4), dtype=np.uint8),
            np.empty((display_height, display_width, 4), dtype=np.float32),
        )
        for _ in range(monitors)
    ]

    def prepare(canvas_u8, canvas_f32):
        resized = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        canvas_u8.fill(0)
        canvas_u8[..., :3] = resized
        canvas_u8[..., 3] = 255
        np.multiply(canvas_u8, 1.0 / 255.0, out=canvas_f32, casting="unsafe")
        return canvas_f32.ravel()

    for _ in range(warmup):
        for canvases_for_monitor in canvases:
            prepare(*canvases_for_monitor)
    times = []
    cpu_started = time.process_time()
    for _ in range(iterations):
        started = time.perf_counter()
        for canvases_for_monitor in canvases:
            prepare(*canvases_for_monitor)
        times.append((time.perf_counter() - started) * 1000.0)
    cpu_ms = (time.process_time() - cpu_started) * 1000.0
    return {
        "name": f"preview_cpu_prepare_{monitors}_monitors",
        "status": "measured",
        "category": "preview",
        "width": width,
        "height": height,
        "format": "rgb8",
        "preview_enabled": monitors > 0,
        "preview_monitors": monitors,
        "preview_display_width": display_width,
        "preview_display_height": display_height,
        "scope": "CPU resize/canvas/float conversion only; dpg.set_value and GPU presentation excluded",
        "warmup_frames": warmup,
        **_summarize_times(times),
        "cpu_time_ms": cpu_ms,
        "cpu_time_per_frame_ms": cpu_ms / iterations,
        "dropped_frames": 0,
    }


def _unsupported_cases(capabilities) -> list[dict[str, Any]]:
    cases = [
        {
            "name": "format_rgba16f",
            "status": "unsupported",
            "reason": "Current flat-pipeline contract normalizes output to RGB uint8",
            "category": "format",
            "format": "rgba16f",
        },
        {
            "name": "format_rgba32f",
            "status": "unsupported",
            "reason": "Current flat-pipeline contract normalizes output to RGB uint8",
            "category": "format",
            "format": "rgba32f",
        },
        {
            "name": "virtual_camera_submission",
            "status": "unsupported",
            "reason": "Requires a configured external virtual-camera backend and live application",
            "category": "virtual_camera",
        },
        {
            "name": "gpu_history_matrix",
            "status": "unsupported",
            "reason": "No general device-resident history backend exists in the current pipeline",
            "category": "history",
            "history_backend": "gpu",
        },
    ]
    if not capabilities["worker_cuda_available"]:
        cases.extend(
            [
                {
                    "name": "cuda_pipeline",
                    "status": "unsupported",
                    "reason": "OpenCV reports zero CUDA devices in the benchmark interpreter",
                    "category": "cuda",
                },
                {
                    "name": "gpu_download_enabled_vs_disabled",
                    "status": "unsupported",
                    "reason": "CUDA is unavailable; no valid upload/download comparison can be measured",
                    "category": "cuda",
                },
                {
                    "name": "cuda_event_timing",
                    "status": "unsupported",
                    "reason": "CUDA is unavailable; CPU high-resolution timing is used for supported cases",
                    "category": "cuda",
                },
            ]
        )
    return cases


def _case_iterations(args, category: str) -> tuple[int, int]:
    if args.quick:
        return (1, 3)
    if category == "representative":
        return (2, max(3, args.iterations // 2))
    if category in {"ipc", "preview"}:
        return (3, max(10, args.iterations))
    return (args.warmup, args.iterations)


def run(args) -> dict[str, Any]:
    cv2, capabilities = _bootstrap_runtime()
    worker_cuda = _probe_cuda_worker()
    capabilities["worker_opencv_path"] = worker_cuda.get("opencv_path")
    capabilities["worker_opencv_version"] = worker_cuda.get("opencv_version")
    capabilities["worker_cuda_device_count"] = int(worker_cuda.get("cuda_device_count", 0) or 0)
    capabilities["worker_cuda_available"] = bool(worker_cuda.get("cuda_available", False))
    capabilities["worker_cuda_events_available"] = bool(worker_cuda.get("cuda_events_available", False))
    capabilities["worker_cuda_error"] = worker_cuda.get("error")
    # `cuda_available` means a valid benchmark path exists. The parent/worker
    # split remains explicit in the adjacent fields.
    capabilities["cuda_available"] = capabilities["worker_cuda_available"]
    import numpy as np
    import gpu_ops
    import node_pipeline_runtime as npr
    from frame_state import FrameState

    gpu_ops.set_use_cuda(capabilities["parent_cuda_available"])
    machine = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "nvidia_before": _nvidia_snapshot(),
        "memory_before": _process_memory(),
    }
    cases: list[dict[str, Any]] = []
    selected_resolutions = (RESOLUTIONS[-1],) if args.quick else RESOLUTIONS
    selected_histories = (1, 8) if args.quick else HISTORY_LENGTHS

    if capabilities["worker_cuda_available"]:
        gpu_warmup = 2 if args.quick else 5
        gpu_iterations = 5 if args.quick else 20
        gpu_result = _run_gpu_microbenchmarks(selected_resolutions, gpu_warmup, gpu_iterations)
        capabilities["gpu_microbenchmark_error"] = gpu_result.get("error")
        cases.extend(gpu_result.get("cases", []))

    warmup, iterations = _case_iterations(args, "pipeline")
    for width, height in selected_resolutions:
        cases.append(
            _run_pipeline_case(
                np, npr, FrameState, gpu_ops,
                name="minimal_graph",
                pipeline=_minimal_pipeline(),
                width=width,
                height=height,
                history_length=1,
                warmup=warmup,
                iterations=iterations,
            )
        )

    rep_pipeline, preset_path = _representative_pipeline()

    # Exercise the actual multiprocessing worker boundary at the application's
    # current primary resolution. Keep this bounded because worker startup and
    # the saved production effect are intentionally expensive.
    e2e_width, e2e_height = RESOLUTIONS[-1]
    e2e_warmup = 1
    e2e_iterations = 2 if args.quick else 10
    cases.append(
        _run_worker_roundtrip_case(
            np, _minimal_pipeline(), e2e_width, e2e_height,
            e2e_warmup, e2e_iterations, "worker_roundtrip_minimal_graph",
        )
    )
    if capabilities["worker_cuda_available"]:
        cuda_pipeline = [
            {"effect_type": "Source", "params": {}, "node_type": "source", "source_key": "primary"},
            {
                "effect_type": "CUDA Edge Slicer",
                "params": {"enabled": True, "edge_thresh": 0.2, "shift_x": 24.0, "shift_y": 0.0, "pre_blur": 3},
                "node_type": "effect",
                "node_id": "benchmark:cuda_edge",
            },
            {
                "effect_type": "CUDA Channel Drift",
                "params": {
                    "enabled": True,
                    "r_shift_x": -8.0,
                    "g_shift_x": 0.0,
                    "b_shift_x": 8.0,
                    "drift_speed": 1.0,
                    "rot_var": 0.5,
                    "scale_var": 0.02,
                },
                "node_type": "effect",
                "node_id": "benchmark:cuda_drift",
            },
            {"effect_type": "Display", "params": {}, "node_type": "display"},
        ]
        cases.append(
            _run_worker_roundtrip_case(
                np, cuda_pipeline, e2e_width, e2e_height,
                e2e_warmup, e2e_iterations, "worker_roundtrip_cuda_two_node_graph",
                use_cuda=True,
            )
        )
    cases.append(
        _run_worker_roundtrip_case(
            np, rep_pipeline, e2e_width, e2e_height,
            e2e_warmup, e2e_iterations, "worker_roundtrip_representative_preset",
        )
    )

    rep_warmup, rep_iterations = _case_iterations(args, "representative")
    for width, height in selected_resolutions:
        cases.append(
            _run_pipeline_case(
                np, npr, FrameState, gpu_ops,
                name="representative_saved_preset",
                pipeline=rep_pipeline,
                width=width,
                height=height,
                history_length=1,
                warmup=rep_warmup,
                iterations=rep_iterations,
                preset=preset_path,
            )
        )

    for width, height in selected_resolutions:
        for history_length in selected_histories:
            cases.append(
                _run_pipeline_case(
                    np, npr, FrameState, gpu_ops,
                    name="python_history_row_stutter",
                    pipeline=_history_pipeline(history_length),
                    width=width,
                    height=height,
                    history_length=history_length,
                    warmup=warmup,
                    iterations=iterations,
                )
            )
            cases.append(
                _run_preallocated_ring_case(
                    np, width, height, history_length, warmup=warmup, iterations=iterations
                )
            )

    dispatch_counts = (1, 8) if args.quick else (1, 8, 32)
    # Use a small frame and many samples so image copies do not drown out the
    # Python call/loop cost being isolated.
    dispatch_width, dispatch_height = (64, 64)
    dispatch_warmup = 10 if args.quick else 20
    dispatch_iterations = 50 if args.quick else 500
    for node_count in dispatch_counts:
        cases.append(
            _run_pipeline_case(
                np, npr, FrameState, gpu_ops,
                name=f"python_dispatch_{node_count}_disabled_nodes",
                pipeline=_noop_pipeline(node_count),
                width=dispatch_width,
                height=dispatch_height,
                history_length=1,
                warmup=dispatch_warmup,
                iterations=dispatch_iterations,
            )
        )

    ipc_warmup, ipc_iterations = _case_iterations(args, "ipc")
    for payload_frames in (1, 3):
        cases.append(
            _run_ipc_case(np, RESOLUTIONS[-1][0], RESOLUTIONS[-1][1], payload_frames,
                          ipc_warmup, ipc_iterations)
        )

    preview_warmup, preview_iterations = _case_iterations(args, "preview")
    cases.append(
        {
            "name": "preview_disabled",
            "status": "measured",
            "category": "preview",
            "width": RESOLUTIONS[-1][0],
            "height": RESOLUTIONS[-1][1],
            "format": "rgb8",
            "preview_enabled": False,
            **_summarize_times([0.0] * preview_iterations),
            "scope": "No preview work",
            "dropped_frames": 0,
        }
    )
    for monitors in (1, 2):
        cases.append(
            _run_preview_case(np, cv2, RESOLUTIONS[-1][0], RESOLUTIONS[-1][1], monitors,
                              preview_warmup, preview_iterations)
        )

    cases.extend(_unsupported_cases(capabilities))
    machine["nvidia_after"] = _nvidia_snapshot()
    machine["memory_after"] = _process_memory()
    return {
        "schema_version": 1,
        "benchmark": "megaglitch-framebuffer-baseline",
        "mode": "quick" if args.quick else "baseline",
        "configuration": {
            "resolutions": [list(item) for item in selected_resolutions],
            "history_lengths": list(selected_histories),
            "warmup_frames": warmup,
            "measurement_frames": iterations,
            "representative_preset": preset_path,
            "preview_scope": capabilities["preview_measurement_scope"],
        },
        "capabilities": capabilities,
        "machine": machine,
        "cases": cases,
    }


def _write_csv(path: Path, cases: list[dict[str, Any]]):
    flattened = []
    keys = set()
    for case in cases:
        row = {}
        for key, value in case.items():
            row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        flattened.append(row)
        keys.update(row)
    preferred = [
        "name", "status", "reason", "category", "width", "height", "format",
        "history_backend", "history_length", "frames", "mean_frame_ms", "p99_frame_ms",
        "average_fps", "one_percent_low_fps",
    ]
    fieldnames = preferred + sorted(keys.difference(preferred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tools" / "benchmark" / "results")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke matrix")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    json_path = args.output_dir / f"{args.label}.json"
    csv_path = args.output_dir / f"{args.label}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, result["cases"])
    measured = sum(case.get("status") == "measured" for case in result["cases"])
    unsupported = sum(case.get("status") == "unsupported" for case in result["cases"])
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Cases: {measured} measured, {unsupported} unsupported")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
