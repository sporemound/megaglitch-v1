"""Deterministic lifecycle for MegaGlitch's spawned processing worker."""

from __future__ import annotations

from enum import Enum
import logging
import os
import time
import traceback
from typing import Any


class ProcessingState(str, Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    WARMING_UP = "warming_up"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class WorkerStartupError(RuntimeError):
    pass


def report_worker_status(status_queue, event: str, **details) -> None:
    """Best-effort structured status reporting from a child process."""
    if status_queue is None:
        return
    message = {
        "event": str(event),
        "pid": os.getpid(),
        "time": time.time(),
        **details,
    }
    try:
        # Lifecycle traffic is deliberately tiny (warming, ready, failed,
        # stopped).  A SimpleQueue writes directly to its pipe and avoids the
        # bounded Queue capacity semaphore and feeder-thread teardown race
        # observed in detached frozen Windows children.
        status_queue.put(message)
    except (BrokenPipeError, EOFError, OSError, ValueError):
        # The parent may already be exiting.  Worker status must never mask the
        # target exception or prevent the child from completing shutdown.
        pass


def _managed_worker_entry(target, args, status_queue) -> None:
    try:
        target(*args, lifecycle_queue=status_queue)
    except BaseException as exc:
        report_worker_status(
            status_queue,
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        report_worker_status(status_queue, "stopped")


class WorkerProcessLifecycle:
    """Own worker readiness, failure reporting, and bounded shutdown."""

    def __init__(self, mp_context, stop_event, *, startup_timeout: float = 20.0,
                 graceful_timeout: float = 5.0, terminate_timeout: float = 2.0):
        self.context = mp_context
        self.stop_event = stop_event
        self.startup_timeout = max(0.05, float(startup_timeout))
        self.graceful_timeout = max(0.05, float(graceful_timeout))
        self.terminate_timeout = max(0.05, float(terminate_timeout))
        self.status_queue = mp_context.SimpleQueue()
        self.process = None
        self.state = ProcessingState.UNINITIALIZED
        self.error: dict[str, Any] | None = None
        self.forced_termination = False
        self.transitions: list[dict[str, Any]] = []
        self.status_channel_error: dict[str, Any] | None = None
        self._closed = False
        self._transition(ProcessingState.UNINITIALIZED)

    def _transition(self, state: ProcessingState, **details) -> None:
        self.state = state
        self.transitions.append({"state": state.value, "time": time.time(), **details})
        if state == ProcessingState.FAILED:
            logging.error("Processing lifecycle: %s (%s)", state.value, details.get("error", ""))
        else:
            logging.info("Processing lifecycle: %s", state.value)

    def _consume(self, message) -> None:
        if not isinstance(message, dict):
            return
        event = message.get("event")
        if event == "warming_up":
            self._transition(ProcessingState.WARMING_UP, **message)
        elif event == "ready":
            self._transition(ProcessingState.READY, **message)
        elif event == "failed":
            self.error = dict(message)
            self._transition(ProcessingState.FAILED, **message)
        elif event == "stopped" and self.state not in {
            ProcessingState.FAILED, ProcessingState.STOPPED,
        }:
            if self.state == ProcessingState.RUNNING and not self.stop_event.is_set():
                self.error = {
                    "event": "failed",
                    "error_type": "WorkerExited",
                    "error": "worker returned while processing was expected to remain active",
                }
                self._transition(ProcessingState.FAILED, **self.error)
            else:
                self._transition(ProcessingState.STOPPED, **message)

    def start(self, target, args=()):
        if self.state not in {ProcessingState.UNINITIALIZED, ProcessingState.STOPPED}:
            raise RuntimeError(f"cannot start worker from {self.state.value}")
        if self.stop_event.is_set():
            raise RuntimeError("worker stop event is already set")
        self.error = None
        self.forced_termination = False
        self._transition(ProcessingState.INITIALIZING)
        self.process = self.context.Process(
            target=_managed_worker_entry,
            args=(target, tuple(args), self.status_queue),
        )
        self.process.start()
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            self.poll()
            if self.state == ProcessingState.READY:
                self._transition(ProcessingState.RUNNING)
                return self.process
            if self.state == ProcessingState.FAILED:
                self._terminate_after_failure()
                raise WorkerStartupError(self.error_message())
            if self.process is not None and not self.process.is_alive():
                # Queue feeder threads can flush a structured child failure a
                # moment after Windows reports the process exit. Give that
                # diagnostic a bounded chance to arrive before replacing it
                # with a generic WorkerExited error.
                diagnostic_deadline = time.monotonic() + 0.25
                while time.monotonic() < diagnostic_deadline:
                    self.poll()
                    if self.state == ProcessingState.FAILED:
                        break
                    time.sleep(0.01)
                if self.state != ProcessingState.FAILED:
                    self.error = {
                        "event": "failed",
                        "error_type": "WorkerExited",
                        "error": f"worker exited during startup with code {self.process.exitcode}",
                    }
                    self._transition(ProcessingState.FAILED, **self.error)
                raise WorkerStartupError(self.error_message())
            time.sleep(0.01)
        self.error = {
            "event": "failed",
            "error_type": "WorkerStartupTimeout",
            "error": f"worker did not become ready within {self.startup_timeout:.2f}s",
        }
        self._transition(ProcessingState.FAILED, **self.error)
        self._terminate_after_failure()
        raise WorkerStartupError(self.error_message())

    def poll(self) -> ProcessingState:
        while True:
            try:
                if self.status_queue.empty():
                    break
                self._consume(self.status_queue.get())
            except (EOFError, OSError, ValueError) as exc:
                details = {
                    "event": "failed",
                    "error_type": "WorkerStatusChannelError",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                self.status_channel_error = details
                if self.state in {ProcessingState.STOPPING, ProcessingState.STOPPED}:
                    logging.warning(
                        "Processing lifecycle status channel failed during shutdown: %s",
                        exc,
                    )
                else:
                    self.error = details
                    self._transition(ProcessingState.FAILED, **details)
                break
        if (
            self.process is not None
            and self.state == ProcessingState.RUNNING
            and not self.process.is_alive()
        ):
            self.error = {
                "event": "failed",
                "error_type": "WorkerExited",
                "error": f"worker exited while running with code {self.process.exitcode}",
            }
            self._transition(ProcessingState.FAILED, **self.error)
        return self.state

    def error_message(self) -> str:
        if not self.error:
            return "processing worker failed"
        return f"{self.error.get('error_type', 'WorkerFailure')}: {self.error.get('error', '')}"

    def _terminate_after_failure(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.forced_termination = True
        if self.process is not None:
            self.process.join(timeout=self.terminate_timeout)

    def stop(self) -> bool:
        """Stop idempotently; return True when forced termination was required."""
        if self.state == ProcessingState.STOPPED:
            self.close_status_queue()
            return self.forced_termination
        self._transition(ProcessingState.STOPPING)
        self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=self.graceful_timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=self.terminate_timeout)
                self.forced_termination = True
        self.poll()
        if self.state != ProcessingState.STOPPED:
            self._transition(
                ProcessingState.STOPPED,
                forced_termination=self.forced_termination,
                exitcode=None if self.process is None else self.process.exitcode,
            )
        self.close_status_queue()
        return self.forced_termination

    def close_status_queue(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.status_queue.close()
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "pid": None if self.process is None else self.process.pid,
            "exitcode": None if self.process is None else self.process.exitcode,
            "forced_termination": self.forced_termination,
            "error": self.error,
            "status_channel_error": self.status_channel_error,
            "transitions": list(self.transitions),
        }
