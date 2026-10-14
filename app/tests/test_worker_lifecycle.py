import multiprocessing as mp
import time
import unittest

from megaglitch.worker_lifecycle import (
    ProcessingState,
    WorkerProcessLifecycle,
    WorkerStartupError,
    report_worker_status,
)


def _ready_worker(stop_event, lifecycle_queue=None):
    report_worker_status(lifecycle_queue, "warming_up", phase="synthetic")
    time.sleep(0.02)
    report_worker_status(lifecycle_queue, "ready", backend="synthetic")
    while not stop_event.is_set():
        time.sleep(0.005)


def _startup_failure_worker(stop_event, lifecycle_queue=None):
    del stop_event, lifecycle_queue
    raise RuntimeError("startup sentinel")


def _stalled_worker(stop_event, lifecycle_queue=None):
    del lifecycle_queue
    while not stop_event.is_set():
        time.sleep(0.01)


def _runtime_failure_worker(stop_event, lifecycle_queue=None):
    del stop_event
    report_worker_status(lifecycle_queue, "warming_up")
    report_worker_status(lifecycle_queue, "ready")
    time.sleep(0.05)
    raise RuntimeError("runtime sentinel")


class _BrokenStatusQueue:
    """Model an unavailable lifecycle status pipe."""

    def __init__(self):
        self.closed = False

    def empty(self):
        raise ValueError("semaphore or lock released too many times")

    def get(self):
        raise AssertionError("get must not run after empty fails")

    def close(self):
        self.closed = True


class WorkerLifecycleTests(unittest.TestCase):
    def context(self):
        return mp.get_context("spawn")

    def test_ready_worker_has_ordered_states_and_graceful_idempotent_stop(self):
        context = self.context()
        stop_event = context.Event()
        lifecycle = WorkerProcessLifecycle(
            context, stop_event, startup_timeout=2.0,
            graceful_timeout=1.0, terminate_timeout=0.5,
        )
        process = lifecycle.start(_ready_worker, (stop_event,))
        self.assertTrue(process.is_alive())
        self.assertEqual(lifecycle.state, ProcessingState.RUNNING)
        self.assertFalse(lifecycle.stop())
        self.assertFalse(lifecycle.stop())
        self.assertFalse(process.is_alive())
        states = [item["state"] for item in lifecycle.snapshot()["transitions"]]
        for required in (
            "initializing", "warming_up", "ready", "running", "stopping", "stopped",
        ):
            self.assertIn(required, states)

    def test_startup_exception_is_structured_and_process_is_cleaned(self):
        context = self.context()
        stop_event = context.Event()
        lifecycle = WorkerProcessLifecycle(
            context, stop_event, startup_timeout=2.0, terminate_timeout=0.5,
        )
        with self.assertRaisesRegex(WorkerStartupError, "startup sentinel"):
            lifecycle.start(_startup_failure_worker, (stop_event,))
        self.assertEqual(lifecycle.state, ProcessingState.FAILED)
        self.assertEqual(lifecycle.error["error_type"], "RuntimeError")
        self.assertFalse(lifecycle.process.is_alive())
        lifecycle.close_status_queue()

    def test_startup_timeout_forces_bounded_cleanup(self):
        context = self.context()
        stop_event = context.Event()
        lifecycle = WorkerProcessLifecycle(
            context, stop_event, startup_timeout=0.1, terminate_timeout=0.5,
        )
        with self.assertRaisesRegex(WorkerStartupError, "did not become ready"):
            lifecycle.start(_stalled_worker, (stop_event,))
        self.assertEqual(lifecycle.state, ProcessingState.FAILED)
        self.assertTrue(lifecycle.forced_termination)
        self.assertFalse(lifecycle.process.is_alive())
        lifecycle.close_status_queue()

    def test_failure_after_readiness_is_reported_by_poll(self):
        context = self.context()
        stop_event = context.Event()
        lifecycle = WorkerProcessLifecycle(
            context, stop_event, startup_timeout=2.0, terminate_timeout=0.5,
        )
        lifecycle.start(_runtime_failure_worker, (stop_event,))
        deadline = time.monotonic() + 3.0
        while lifecycle.poll() != ProcessingState.FAILED and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(lifecycle.state, ProcessingState.FAILED)
        self.assertIn("runtime sentinel", lifecycle.error_message())
        lifecycle.stop()

    def test_three_repeated_start_stop_cycles(self):
        for _ in range(3):
            context = self.context()
            stop_event = context.Event()
            lifecycle = WorkerProcessLifecycle(
                context, stop_event, startup_timeout=2.0,
                graceful_timeout=1.0, terminate_timeout=0.5,
            )
            lifecycle.start(_ready_worker, (stop_event,))
            self.assertFalse(lifecycle.stop())
            self.assertEqual(lifecycle.snapshot()["state"], "stopped")

    def test_stop_tolerates_windows_status_queue_semaphore_teardown_race(self):
        context = self.context()
        stop_event = context.Event()
        lifecycle = WorkerProcessLifecycle(context, stop_event)
        original_queue = lifecycle.status_queue
        original_queue.close()
        broken_queue = _BrokenStatusQueue()
        lifecycle.status_queue = broken_queue

        lifecycle._transition(ProcessingState.RUNNING)
        with self.assertLogs(level="WARNING") as captured:
            self.assertFalse(lifecycle.stop())

        self.assertEqual(lifecycle.state, ProcessingState.STOPPED)
        self.assertTrue(stop_event.is_set())
        self.assertTrue(broken_queue.closed)
        self.assertIn("released too many times", "\n".join(captured.output))
        self.assertEqual(
            lifecycle.snapshot()["status_channel_error"]["error_type"],
            "WorkerStatusChannelError",
        )
        self.assertFalse(lifecycle.stop())

    def test_active_status_channel_failure_is_structured_and_terminal(self):
        context = self.context()
        stop_event = context.Event()
        lifecycle = WorkerProcessLifecycle(context, stop_event)
        original_queue = lifecycle.status_queue
        original_queue.close()
        broken_queue = _BrokenStatusQueue()
        lifecycle.status_queue = broken_queue

        lifecycle._transition(ProcessingState.INITIALIZING)
        self.assertEqual(lifecycle.poll(), ProcessingState.FAILED)
        self.assertEqual(lifecycle.error["error_type"], "WorkerStatusChannelError")
        self.assertIn("released too many times", lifecycle.error_message())
        lifecycle.stop()
        self.assertTrue(broken_queue.closed)

    def test_lifecycle_status_transport_has_no_capacity_semaphore(self):
        context = self.context()
        lifecycle = WorkerProcessLifecycle(context, context.Event())
        try:
            self.assertFalse(hasattr(lifecycle.status_queue, "_sem"))
            self.assertTrue(hasattr(lifecycle.status_queue, "put"))
            self.assertFalse(hasattr(lifecycle.status_queue, "put_nowait"))
        finally:
            lifecycle.close_status_queue()

    def test_real_shared_worker_reports_ready_and_stops_before_owner_unlink(self):
        from megaglitch.shared_frame_transport import SharedFrameTransportOwner
        from megaglitchfm_app_new import glitch_process_worker

        context = self.context()
        stop_event = context.Event()
        raw_queue = context.Queue(maxsize=1)
        output_queue = context.Queue(maxsize=1)
        owner = SharedFrameTransportOwner(
            context, 16, 12, input_slots=2, output_slots=1
        )
        lifecycle = WorkerProcessLifecycle(
            context, stop_event, startup_timeout=30.0,
            graceful_timeout=5.0, terminate_timeout=1.0,
        )
        try:
            lifecycle.start(
                glitch_process_worker,
                (raw_queue, output_queue, stop_event, None, owner.spec()),
            )
            ready = [
                item for item in lifecycle.snapshot()["transitions"]
                if item["state"] == "ready"
            ]
            self.assertEqual(ready[-1]["frame_transport"], "shared_v1")
            raw_queue.put("RESET", timeout=5.0)
            self.assertEqual(output_queue.get(timeout=10.0), "RESET")
            self.assertFalse(lifecycle.stop())
            final_snapshot = lifecycle.snapshot()
            self.assertEqual(final_snapshot["state"], "stopped")
            self.assertEqual(final_snapshot["exitcode"], 0)
            self.assertFalse(final_snapshot["forced_termination"])
            self.assertIsNone(final_snapshot["error"])
            self.assertIsNone(final_snapshot["status_channel_error"])
            self.assertEqual(owner.get_statistics()["input_slots_in_use"], 0)
            self.assertEqual(owner.get_statistics()["output_slots_in_use"], 0)
        finally:
            if lifecycle.state != ProcessingState.STOPPED:
                lifecycle.stop()
            raw_queue.close()
            output_queue.close()
            owner.close()

    def test_real_worker_rejects_broken_required_shared_attachment(self):
        from megaglitch.shared_frame_transport import SharedFrameTransportOwner
        from megaglitchfm_app_new import glitch_process_worker

        context = self.context()
        stop_event = context.Event()
        raw_queue = context.Queue(maxsize=1)
        output_queue = context.Queue(maxsize=1)
        owner = SharedFrameTransportOwner(
            context, 16, 12, input_slots=1, output_slots=1
        )
        broken_spec = owner.spec()
        owner.close()
        lifecycle = WorkerProcessLifecycle(
            context, stop_event, startup_timeout=30.0, terminate_timeout=1.0,
        )
        try:
            with self.assertRaisesRegex(WorkerStartupError, "required shared frame transport"):
                lifecycle.start(
                    glitch_process_worker,
                    (raw_queue, output_queue, stop_event, None, broken_spec),
                )
            self.assertEqual(lifecycle.state, ProcessingState.FAILED)
            self.assertFalse(lifecycle.process.is_alive())
        finally:
            lifecycle.close_status_queue()
            raw_queue.close()
            output_queue.close()


if __name__ == "__main__":
    unittest.main()
