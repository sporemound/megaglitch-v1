import unittest

import numpy as np

from node_pipeline_runtime import execute_flat_node_pipeline, record_frame_copy


class FrameCopyTelemetryTests(unittest.TestCase):
    def test_source_node_copy_is_counted_without_changing_output(self):
        frame = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        stats = {}
        pipeline = [
            {"effect_type": "Image Source", "node_type": "source", "source_key": "primary"},
            {"effect_type": "Display", "params": {}},
        ]
        output, *_ = execute_flat_node_pipeline(
            pipeline, frame, fallback_size=(6, 4), copy_stats=stats
        )
        np.testing.assert_array_equal(output, frame)
        self.assertFalse(np.shares_memory(output, frame))
        self.assertEqual(stats["initial_node"], {"count": 1, "bytes": frame.nbytes})

    def test_history_copy_accounting_accumulates(self):
        frame = np.zeros((4, 6, 3), np.uint8)
        stats = {}
        record_frame_copy(stats, "history", frame)
        record_frame_copy(stats, "history", frame)
        self.assertEqual(stats["history"], {"count": 2, "bytes": frame.nbytes * 2})

    def test_unique_source_can_be_borrowed_without_output_drift(self):
        frame = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        pipeline = [
            {"effect_type": "Source", "node_type": "source", "source_key": "primary"},
            {"effect_type": "Display", "node_type": "display", "params": {}},
        ]
        copied_stats = {}
        borrowed_stats = {}
        copied, *_ = execute_flat_node_pipeline(
            pipeline, frame, copy_stats=copied_stats, source_copy_policy="copy"
        )
        borrowed, *_ = execute_flat_node_pipeline(
            pipeline, frame, copy_stats=borrowed_stats, source_copy_policy="borrow_unique"
        )
        np.testing.assert_array_equal(borrowed, copied)
        self.assertEqual(copied_stats["initial_node"]["count"], 1)
        self.assertNotIn("initial_node", borrowed_stats)
        self.assertTrue(np.shares_memory(borrowed, frame))

    def test_repeated_source_is_copied_under_borrow_policy(self):
        frame = np.zeros((4, 6, 3), np.uint8)
        stats = {}
        pipeline = [
            {"effect_type": "Source A", "node_type": "source", "source_key": "primary"},
            {"effect_type": "Source B", "node_type": "source", "source_key": "primary"},
            {"effect_type": "Video Mixer", "node_type": "mixer", "params": {}},
            {"effect_type": "Display", "node_type": "display", "params": {}},
        ]
        execute_flat_node_pipeline(
            pipeline, frame, copy_stats=stats, source_copy_policy="borrow_unique"
        )
        self.assertEqual(stats["initial_node"], {"count": 2, "bytes": frame.nbytes * 2})

    def test_unknown_source_copy_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported source copy policy"):
            execute_flat_node_pipeline([], np.zeros((2, 2, 3), np.uint8), source_copy_policy="unsafe")


if __name__ == "__main__":
    unittest.main()
