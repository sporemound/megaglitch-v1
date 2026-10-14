import unittest
import random

import numpy as np

from frame_state import FrameState
from megaglitch.native_temporal import (
    TemporalFrameRing,
    amplify_prediction_errors_uniform,
    native_available,
    process_pframe_spiral_uniform,
)


BACKENDS = ["python"] + (["native"] if native_available() else [])


class TemporalRingContractTests(unittest.TestCase):
    def make_ring(self, backend, capacity=3):
        ring = TemporalFrameRing(backend)
        ring.initialize(4, 3, capacity)
        self.addCleanup(ring.shutdown)
        return ring

    def test_ring_delay_and_wrap_are_deterministic(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                ring = self.make_ring(backend)
                for value in range(1, 5):
                    ring.write_frame(np.full((3, 4, 3), value, np.uint8))
                self.assertTrue(np.all(ring.read_frame(0) == 4))
                self.assertTrue(np.all(ring.read_frame(1) == 3))
                self.assertTrue(np.all(ring.read_frame(2) == 2))

    def test_native_and_python_mix_outputs_match(self):
        if not native_available():
            self.skipTest("native extension is not built")
        current = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        history = np.flip(current, axis=1).copy()
        outputs = []
        for backend in ("python", "native"):
            ring = self.make_ring(backend)
            ring.write_frame(history)
            output = np.empty_like(current)
            returned = ring.process_temporal_mix(current, 0, 0.35, output)
            self.assertIs(returned, output)
            outputs.append(output.copy())
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_freeze_clear_resize_and_invalid_values(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                ring = self.make_ring(backend)
                first = np.full((3, 4, 3), 10, np.uint8)
                second = np.full((3, 4, 3), 20, np.uint8)
                ring.write_frame(first)
                ring.freeze(True)
                self.assertFalse(ring.write_frame(second))
                np.testing.assert_array_equal(ring.read_frame(0), first)
                ring.freeze(False)
                ring.clear()
                with self.assertRaises(ValueError):
                    ring.read_frame(0)
                with self.assertRaises(ValueError):
                    ring.set_delay(-1)
                with self.assertRaises(ValueError):
                    ring.set_mix(1.1)
                with self.assertRaises(ValueError):
                    ring.configure(0, 2, 4)
                ring.configure(2, 2, 4)
                ring.write_frame(np.zeros((2, 2, 3), np.uint8))
                self.assertEqual(ring.get_statistics()["allocated_bytes"], 48)

    def test_repeated_initialize_shutdown_and_exception_cleanup(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                ring = TemporalFrameRing(backend)
                for _ in range(5):
                    ring.configure(2, 2, 2)
                    ring.write_frame(np.zeros((2, 2, 3), np.uint8))
                    ring.shutdown()
                    self.assertFalse(ring.get_statistics()["configured"])
                with self.assertRaisesRegex(RuntimeError, "sentinel"):
                    with TemporalFrameRing(backend) as scoped:
                        scoped.configure(2, 2, 2)
                        raise RuntimeError("sentinel")
                self.assertFalse(scoped.get_statistics()["configured"])

    @unittest.skipUnless(native_available(), "native extension is not built")
    def test_impossible_native_allocation_fails_before_allocating(self):
        ring = TemporalFrameRing("native")
        with self.assertRaises(OverflowError):
            ring.configure(2_000_000_000, 2_000_000_000, 2_000_000_000, 2_000_000_000)
        self.assertFalse(ring.get_statistics()["configured"])

    def test_frame_state_uses_opt_in_boundary_and_releases_it(self):
        state = FrameState()
        ring = state.get_temporal_history(4, 3, 5, backend="python")
        self.assertEqual(ring.get_statistics()["capacity"], 5)
        self.assertIs(ring, state.get_temporal_history(4, 3, 5, backend="python"))
        state.reset()
        self.assertIsNone(state.temporal_history)

    @unittest.skipUnless(native_available(), "native extension is not built")
    def test_uniform_prediction_kernel_exactly_matches_existing_numpy_semantics(self):
        rng = np.random.default_rng(1984)
        for intensity in (0.01, 0.451, 0.978, 1.0):
            for threshold in (0, 1, 30, 255):
                current = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)
                reference = rng.integers(0, 256, current.shape, dtype=np.uint8)
                residual = current.astype(np.float32) - reference.astype(np.float32)
                error_mask = np.abs(residual) > threshold
                amplification = 1 + intensity * 3 * np.ones(current.shape[:2])[..., None]
                expected = np.clip(
                    reference.astype(np.float32)
                    + np.where(error_mask, residual * amplification, residual),
                    0,
                    255,
                ).astype(np.uint8)
                actual = amplify_prediction_errors_uniform(
                    current, reference, intensity, threshold
                )
                np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(native_available(), "native extension is not built")
    def test_uniform_prediction_kernel_supports_preallocated_output(self):
        current = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        reference = np.flip(current, axis=1).copy()
        output = np.empty_like(current)
        returned = amplify_prediction_errors_uniform(
            current, reference, 0.5, 3, output
        )
        self.assertIs(returned, output)
        with self.assertRaises(ValueError):
            amplify_prediction_errors_uniform(current, reference[:, :-1], 0.5, 3)
        with self.assertRaises(ValueError):
            amplify_prediction_errors_uniform(current[:, ::-1], reference, 0.5, 3)

    @unittest.skipUnless(native_available(), "native extension is not built")
    def test_fused_pframe_segment_matches_seeded_python_stages_exactly(self):
        import glitch_effects as ge

        rng = np.random.default_rng(713)
        current = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        reference = rng.integers(0, 256, current.shape, dtype=np.uint8)
        random.seed(1984)
        expected = ge.corrupt_motion_vectors(
            current, reference, 0.451, block_size=16, direction="spiral"
        )
        expected = amplify_prediction_errors_uniform(expected, reference, 0.978, 30)

        block_count = ((current.shape[1] - 1) // 16) * ((current.shape[0] - 1) // 16)
        random.seed(1984)
        decisions = np.fromiter(
            (random.random() for _ in range(block_count)),
            dtype=np.float64,
            count=block_count,
        )
        output = current.copy()
        returned = process_pframe_spiral_uniform(
            output, reference, 0.451, 16, decisions, 0.978, 30, output
        )
        self.assertIs(returned, output)
        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    unittest.main()
