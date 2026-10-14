"""Safe worker selector for the explicitly enabled Rust runtime vertical slice."""

from __future__ import annotations

import logging
import queue
import time

import numpy as np

from megaglitch.native_runtime import (
    NativeRuntimeUnavailable,
    RuntimeConfig,
    create_runtime,
    has_active_temporal_macros,
    native_graph_for_pipeline,
    native_mode_for_pipeline,
    native_runtime_available,
)


class _PrefixedQueue:
    """Present one consumed item before delegating to the multiprocessing queue."""

    def __init__(self, first, source):
        self._first = first
        self._source = source

    def get(self, *args, **kwargs):
        if self._first is not None:
            item, self._first = self._first, None
            return item
        return self._source.get(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._source, name)


def _active_nodes(pipeline):
    return [
        node for node in (pipeline or [])
        if node.get("enabled", True)
        and node.get("node_type") not in {"source", "display"}
        and node.get("effect_type") != "Display"
        and str((node.get("params") or {}).get("enabled", True)).lower() != "false"
    ]


def _native_item_mode(item):
    """Backward-compatible classifier; the worker executes `_native_item_graph`."""
    graph = _native_item_graph(item)
    if graph is None:
        return None
    return native_mode_for_pipeline(item.get("pipeline"))


def _native_item_graph(item):
    if not isinstance(item, dict) or item.get("genesis_debug_request"):
        return None
    if _active_nodes(item.get("pipeline_b")) or item.get("secondary_image") is not None:
        return None
    graph = native_graph_for_pipeline(item.get("pipeline"))
    if graph is None:
        return None
    if len(graph.operations) > 2 and has_active_temporal_macros(item.get("extra_params")):
        return None
    return graph


def _latest_put(target, value):
    try:
        target.put_nowait(value)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        try:
            target.put_nowait(value)
        except queue.Full:
            pass


def native_runtime_process_worker(raw_queue, glitched_queue, stop_event, node_params_getter,
                                  python_worker, lifecycle_queue=None):
    """Use Rust for supported frames; permanently fall back on the first unsupported graph."""
    from megaglitch.worker_lifecycle import report_worker_status
    report_worker_status(lifecycle_queue, "warming_up", backend="native_rust_selector")
    if not native_runtime_available():
        raise NativeRuntimeUnavailable(
            "native Rust runtime was explicitly selected but its DLL could not be loaded"
        )
    del node_params_getter
    runtime = None
    runtime_key = None
    runtime_graph = None
    report_worker_status(
        lifecycle_queue, "ready", backend="native_rust_selector",
        frame_transport="legacy",
    )
    try:
        while not stop_event.is_set():
            try:
                item = raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item == "RESET":
                if runtime is not None:
                    runtime.clear()
                _latest_put(glitched_queue, "RESET")
                continue

            graph = _native_item_graph(item)
            if graph is None:
                logging.warning(
                    "Native runtime graph unsupported; switching this worker to the Python fallback"
                )
                if runtime is not None:
                    runtime.close()
                    runtime = None
                return python_worker(
                    _PrefixedQueue(item, raw_queue), glitched_queue, stop_event, None
                )

            frame = np.asarray(item.get("image"))
            if frame.dtype != np.uint8 or frame.ndim != 3:
                logging.warning("Native runtime input contract unsupported; switching to Python fallback")
                if runtime is not None:
                    runtime.close()
                return python_worker(
                    _PrefixedQueue(item, raw_queue), glitched_queue, stop_event, None
                )
            frame = np.ascontiguousarray(frame)
            height, width, channels = frame.shape
            active = _active_nodes(item.get("pipeline"))
            history_capacity = 8
            for node in active:
                params = node.get("params", {}) or {}
                history_capacity = max(
                    history_capacity,
                    int(params.get("history_capacity", 8) or 8),
                    int(params.get("delay", 0) or 0) + 1,
                )
            history_capacity = max(1, history_capacity)
            key = (width, height, channels, history_capacity)
            if runtime is None or runtime_key != key:
                if runtime is not None:
                    runtime.close()
                runtime = create_runtime(
                    RuntimeConfig(
                        width, height, channels, history_capacity,
                        queue_capacity=2, graph=graph,
                    ),
                    backend="native",
                )
                runtime.start()
                runtime_key = key
                runtime_graph = graph
            elif graph != runtime_graph:
                runtime.update_graph(graph)
                runtime_graph = graph
            if not runtime.submit_frame(frame):
                continue
            output, _sequence = runtime.wait_for_frame(timeout=1.0)
            result = {
                "frame": output,
                "frame_a": output,
                "frame_b": output,
                "timestamp": item.get("timestamp", time.perf_counter()),
                "runtime_backend": "native_rust",
                "runtime_stats": runtime.get_statistics(),
            }
            _latest_put(glitched_queue, result)
    except Exception:
        logging.exception("Native runtime worker failed")
        raise
    finally:
        if runtime is not None:
            runtime.close()
