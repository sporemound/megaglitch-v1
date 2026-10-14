import time
import queue
import threading
import unittest

import numpy as np

from megaglitch.native_runtime import (
    EngineState,
    GRAPH_IR_VERSION,
    GraphOpcode,
    GraphOperation,
    GraphSnapshot,
    NativeRuntimeFailure,
    ParameterSnapshot,
    RuntimeConfig,
    UnsupportedNativeGraph,
    create_runtime,
    native_graph_for_pipeline,
    native_mode_for_pipeline,
    native_runtime_available,
)
from megaglitch.native_runtime_worker import _native_item_mode, native_runtime_process_worker


BACKENDS = ["python"] + (["native"] if native_runtime_available() else [])


class RuntimeContractTests(unittest.TestCase):
    def engine(self, backend, mode="passthrough", queue_capacity=2, history_capacity=4):
        engine = create_runtime(
            RuntimeConfig(4, 3, 3, history_capacity, queue_capacity, mode), backend
        )
        self.addCleanup(engine.close)
        return engine

    def test_initialization_passthrough_statistics_and_shutdown(self):
        frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = self.engine(backend)
                self.assertEqual(engine.state, EngineState.READY)
                engine.start()
                self.assertEqual(engine.state, EngineState.RUNNING)
                self.assertTrue(engine.submit_frame(frame))
                output, sequence = engine.wait_for_frame()
                np.testing.assert_array_equal(output, frame)
                self.assertEqual(sequence, 1)
                stats = engine.get_statistics()
                self.assertEqual(stats["submitted"], 1)
                self.assertEqual(stats["processed"], 1)
                self.assertEqual(stats["delivered"], 1)
                engine.shutdown()
                self.assertEqual(engine.state, EngineState.STOPPED)

    def test_parameter_snapshot_is_applied_to_next_frame_and_freeze_holds_history(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = self.engine(backend, mode="temporal_mix")
                engine.start()
                first = np.full((3, 4, 3), 10, np.uint8)
                engine.submit_frame(first)
                engine.wait_for_frame()
                engine.update_parameters(ParameterSnapshot(delay=0, mix=1.0, frozen=True))
                engine.submit_frame(np.full_like(first, 20))
                frozen, _ = engine.wait_for_frame()
                np.testing.assert_array_equal(frozen, first)
                engine.update_parameters(ParameterSnapshot(delay=0, mix=0.0, frozen=False))
                engine.submit_frame(np.full_like(first, 30))
                current, _ = engine.wait_for_frame()
                self.assertTrue(np.all(current == 30))
                self.assertGreaterEqual(engine.get_statistics()["commands_applied"], 2)

    def test_existing_invert_node_is_pixel_exact_in_runtime(self):
        frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = self.engine(backend, mode="invert")
                engine.start()
                self.assertTrue(engine.submit_frame(frame))
                output, sequence = engine.wait_for_frame()
                np.testing.assert_array_equal(output, 255 - frame)
                self.assertEqual(sequence, 1)

    def test_graph_ir_compiles_multi_node_pipeline_and_matches_backends(self):
        pipeline = [
            {"effect_type": "Source", "node_type": "source", "source_key": "primary"},
            {"effect_type": "Brightness / Contrast", "params": {
                "brightness": 17, "contrast": -25,
            }},
            {"effect_type": "Invert Colors", "params": {"invert_chance": 100}},
            {"effect_type": "Display", "node_type": "display"},
        ]
        graph = native_graph_for_pipeline(pipeline)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.version, GRAPH_IR_VERSION)
        self.assertEqual(
            [operation.opcode for operation in graph.operations],
            [
                GraphOpcode.INPUT,
                GraphOpcode.BRIGHTNESS_CONTRAST,
                GraphOpcode.INVERT,
                GraphOpcode.OUTPUT,
            ],
        )
        frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        expected = 255 - np.clip(frame.astype(np.float32) * 0.75 + 17, 0, 255).astype(np.uint8)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = create_runtime(RuntimeConfig(4, 3, graph=graph), backend)
                self.addCleanup(engine.close)
                engine.start()
                engine.submit_frame(frame)
                output, _ = engine.wait_for_frame()
                np.testing.assert_array_equal(output, expected)

    def test_immutable_graph_snapshot_replaces_at_next_frame(self):
        passthrough = GraphSnapshot((
            GraphOperation(GraphOpcode.INPUT),
            GraphOperation(GraphOpcode.OUTPUT),
        ))
        invert = GraphSnapshot((
            GraphOperation(GraphOpcode.INPUT),
            GraphOperation(GraphOpcode.INVERT),
            GraphOperation(GraphOpcode.OUTPUT),
        ))
        frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = create_runtime(RuntimeConfig(4, 3, graph=passthrough), backend)
                self.addCleanup(engine.close)
                engine.start()
                engine.submit_frame(frame)
                first, _ = engine.wait_for_frame()
                np.testing.assert_array_equal(first, frame)
                engine.update_graph(invert)
                engine.submit_frame(frame)
                second, _ = engine.wait_for_frame()
                np.testing.assert_array_equal(second, 255 - frame)
                self.assertGreaterEqual(engine.get_statistics()["commands_applied"], 1)

    def test_graph_temporal_mix_and_blend_use_bounded_history(self):
        for opcode in (GraphOpcode.TEMPORAL_MIX, GraphOpcode.BLEND):
            graph = GraphSnapshot((
                GraphOperation(GraphOpcode.INPUT),
                GraphOperation(opcode, arg0=1.0, arg2=1),
                GraphOperation(GraphOpcode.OUTPUT),
            ))
            for backend in BACKENDS:
                with self.subTest(opcode=opcode.name, backend=backend):
                    engine = create_runtime(
                        RuntimeConfig(4, 3, history_capacity=4, graph=graph), backend
                    )
                    self.addCleanup(engine.close)
                    engine.start()
                    first = np.full((3, 4, 3), 10, np.uint8)
                    engine.submit_frame(first)
                    engine.wait_for_frame()
                    engine.submit_frame(np.full_like(first, 20))
                    delayed, _ = engine.wait_for_frame()
                    np.testing.assert_array_equal(delayed, first)

    def test_invalid_graph_ir_is_rejected(self):
        invalid_graphs = (
            GraphSnapshot((
                GraphOperation(GraphOpcode.INPUT), GraphOperation(GraphOpcode.OUTPUT),
            ), version=GRAPH_IR_VERSION + 1),
            GraphSnapshot((GraphOperation(GraphOpcode.INVERT), GraphOperation(GraphOpcode.OUTPUT))),
            GraphSnapshot((
                GraphOperation(GraphOpcode.INPUT),
                GraphOperation(GraphOpcode.TEMPORAL_MIX, arg0=0.5, arg2=99),
                GraphOperation(GraphOpcode.OUTPUT),
            )),
        )
        for graph in invalid_graphs:
            for backend in BACKENDS:
                with self.subTest(graph=graph, backend=backend):
                    with self.assertRaises(NativeRuntimeFailure):
                        create_runtime(RuntimeConfig(4, 3, history_capacity=4, graph=graph), backend)

    def test_bounded_queue_rejects_newest_and_recovers_after_poll(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = self.engine(backend, queue_capacity=1)
                engine.start()
                frame = np.zeros((3, 4, 3), np.uint8)
                self.assertTrue(engine.submit_frame(frame))
                deadline = time.perf_counter() + 1.0
                while engine.get_statistics()["processed"] < 1 and time.perf_counter() < deadline:
                    time.sleep(0.001)
                self.assertFalse(engine.submit_frame(frame))
                engine.wait_for_frame()
                self.assertTrue(engine.submit_frame(frame))
                engine.wait_for_frame()
                self.assertGreaterEqual(engine.get_statistics()["input_dropped"], 1)

    def test_rapid_parameter_changes_clear_and_shutdown_during_work(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = self.engine(backend, mode="temporal_mix", queue_capacity=4)
                engine.start()
                for index in range(10):
                    engine.update_parameters(ParameterSnapshot(index % 4, index / 10.0, False))
                engine.clear()
                for value in range(4):
                    engine.submit_frame(np.full((3, 4, 3), value, np.uint8))
                engine.shutdown()
                self.assertIn(engine.state, {EngineState.STOPPED, EngineState.FAILED})

    def test_invalid_state_input_and_graph_report_structured_failures(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                engine = self.engine(backend)
                with self.assertRaises((NativeRuntimeFailure, RuntimeError)):
                    engine.submit_frame(np.zeros((3, 4, 3), np.uint8))
                engine.start()
                with self.assertRaises(ValueError):
                    engine.submit_frame(np.zeros((2, 2, 3), np.uint8))
                with self.assertRaises((NativeRuntimeFailure, ValueError)):
                    engine.update_parameters(ParameterSnapshot(delay=4, mix=0.5))
                with self.assertRaises((NativeRuntimeFailure, RuntimeError)):
                    engine.start()
        with self.assertRaises(UnsupportedNativeGraph):
            create_runtime(RuntimeConfig(4, 3, mode="unknown"), "native")

    def test_repeated_start_stop_and_resolution_reconstruction(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                for width, height in ((4, 3), (8, 6), (4, 3)):
                    engine = create_runtime(RuntimeConfig(width, height), backend)
                    engine.start()
                    frame = np.zeros((height, width, 3), np.uint8)
                    engine.submit_frame(frame)
                    engine.wait_for_frame()
                    engine.close()
                    self.assertEqual(engine.state, EngineState.UNINITIALIZED if backend == "native" else EngineState.STOPPED)

    def test_graph_selection_preserves_python_fallback(self):
        self.assertEqual(native_mode_for_pipeline([]), "passthrough")
        self.assertEqual(
            native_mode_for_pipeline([{"effect_type": "Display", "enabled": True}]),
            "passthrough",
        )
        self.assertEqual(
            native_mode_for_pipeline([{"effect_type": "Native Temporal Mix", "enabled": True}]),
            "temporal_mix",
        )
        self.assertEqual(
            native_mode_for_pipeline([
                {"effect_type": "Invert Colors", "params": {"invert_chance": 100}}
            ]),
            "invert",
        )
        self.assertEqual(
            native_mode_for_pipeline([
                {"effect_type": "Source", "node_type": "source", "source_key": "primary"},
                {"effect_type": "Invert Colors", "params": {"invert_chance": 100}},
                {"effect_type": "Display", "node_type": "display"},
            ]),
            "invert",
        )
        self.assertEqual(
            native_mode_for_pipeline([
                {"effect_type": "Invert Colors", "params": {"invert_chance": 0}}
            ]),
            "passthrough",
        )
        self.assertEqual(
            native_mode_for_pipeline([
                {"effect_type": "Invert Colors", "params": {
                    "invert_chance": 100, "enabled": False,
                }}
            ]),
            "passthrough",
        )
        self.assertIsNone(native_mode_for_pipeline([
            {"effect_type": "Source", "node_type": "source", "source_key": "secondary"},
            {"effect_type": "Invert Colors", "params": {"invert_chance": 100}},
            {"effect_type": "Display", "node_type": "display"},
        ]))
        self.assertIsNone(native_mode_for_pipeline([{"effect_type": "P-Frame Glitches"}]))
        self.assertIsNone(native_mode_for_pipeline([
            {"effect_type": "Invert Colors", "params": {
                "invert_chance": 100, "event_hazard": 0.5,
            }}
        ]))
        self.assertIsNone(_native_item_mode({
            "image": np.zeros((3, 4, 3), np.uint8),
            "pipeline": [{"effect_type": "Invert Colors", "params": {"invert_chance": 100}}],
            "extra_params": {"local_time_warp": 0.25},
        }))
        self.assertIsNone(_native_item_mode({
            "image": np.zeros((3, 4, 3), np.uint8),
            "pipeline": [{"effect_type": "P-Frame Glitches"}],
        }))

    @unittest.skipUnless(native_runtime_available(), "native runtime DLL is not built")
    def test_native_worker_integration_and_unsupported_graph_fallback(self):
        raw, output = queue.Queue(), queue.Queue()
        stop = threading.Event()
        frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        raw.put({"image": frame, "pipeline": [], "pipeline_b": [], "timestamp": 1.25})
        thread = threading.Thread(
            target=native_runtime_process_worker,
            args=(raw, output, stop, None, lambda *args: self.fail("unexpected fallback")),
        )
        thread.start()
        result = output.get(timeout=2.0)
        stop.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        np.testing.assert_array_equal(result["frame"], frame)
        self.assertEqual(result["runtime_backend"], "native_rust")

        raw, output = queue.Queue(), queue.Queue()
        stop = threading.Event()
        raw.put({
            "image": frame,
            "pipeline": [
                {"effect_type": "Invert Colors", "params": {"invert_chance": 100}}
            ],
            "pipeline_b": [],
            "timestamp": 1.5,
        })
        thread = threading.Thread(
            target=native_runtime_process_worker,
            args=(raw, output, stop, None, lambda *args: self.fail("unexpected fallback")),
        )
        thread.start()
        result = output.get(timeout=2.0)
        stop.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        np.testing.assert_array_equal(result["frame"], 255 - frame)
        self.assertEqual(result["runtime_backend"], "native_rust")

        raw, output = queue.Queue(), queue.Queue()
        stop = threading.Event()
        fallback_called = threading.Event()

        def fallback(source, target, stop_event, getter):
            del target, stop_event, getter
            recovered = source.get(timeout=0.1)
            self.assertEqual(recovered["pipeline"][0]["effect_type"], "P-Frame Glitches")
            fallback_called.set()

        raw.put({"image": frame, "pipeline": [{"effect_type": "P-Frame Glitches"}]})
        native_runtime_process_worker(raw, output, stop, None, fallback)
        self.assertTrue(fallback_called.is_set())

    @unittest.skipUnless(native_runtime_available(), "native runtime DLL is not built")
    def test_native_worker_applies_immutable_parameter_snapshot_to_next_frame(self):
        raw, output = queue.Queue(), queue.Queue()
        stop = threading.Event()
        frame = np.full((3, 4, 3), 20, np.uint8)

        def item(brightness):
            return {
                "image": frame,
                "pipeline": [
                    {"effect_type": "Source", "node_type": "source", "source_key": "primary"},
                    {"effect_type": "Brightness / Contrast", "params": {
                        "brightness": brightness, "contrast": 0,
                    }},
                    {"effect_type": "Display", "node_type": "display"},
                ],
                "pipeline_b": [],
            }

        thread = threading.Thread(
            target=native_runtime_process_worker,
            args=(raw, output, stop, None, lambda *args: self.fail("unexpected fallback")),
        )
        thread.start()
        raw.put(item(10))
        first = output.get(timeout=2.0)
        raw.put(item(30))
        second = output.get(timeout=2.0)
        stop.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(np.all(first["frame"] == 30))
        self.assertTrue(np.all(second["frame"] == 50))


if __name__ == "__main__":
    unittest.main()
