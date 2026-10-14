import hashlib
import random
import unittest

import numpy as np

import glitch_effects as ge
from megaglitch.native_temporal import native_available
from node_pipeline_runtime import blend_decks


class DeterministicVisualRegressionTests(unittest.TestCase):
    @unittest.skipUnless(native_available(), "native extension is not built")
    def test_seeded_fused_pframe_golden_hash(self):
        rng = np.random.default_rng(20260714)
        current = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        reference = rng.integers(0, 256, current.shape, dtype=np.uint8)
        random.seed(1984)
        output = current.copy()
        returned = ge.process_pframe_spiral_uniform(
            output, reference, 0.451, 16, 0.978, 30, output=output
        )
        self.assertIs(returned, output)
        self.assertEqual(
            hashlib.sha256(output.tobytes()).hexdigest(),
            "8d6b58d2cbab628b80b8cb164cb8b00a83676dbcce873cb861f8a0c126c23124",
        )

    def test_deterministic_crossfader_golden_hashes(self):
        deck_a = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3).astype(np.uint8)
        deck_b = np.flip(deck_a, axis=1).copy()
        expected = {
            "Normal Mix": "9596c896c069d4afcf159c140cd486dcdac83147187d9cc305ff16d3d5de26a5",
            "Luma Wipe": "0dae010bbc705405aa394d1f22fa34aa348d1231db809dd00637e8bec9437b77",
            "Horizontal Wipe": "d4676e3c598f882d51032c5991a534aae77cc5fe273c2422cb4922cb81c0763b",
            "Difference": "a952637edd5541d672ba32f6e1435df6fce395ec34a0843af239826a17eeb3d6",
        }
        for mode, digest in expected.items():
            with self.subTest(mode=mode):
                output = blend_decks(deck_a, deck_b, 0.37, mode)
                self.assertEqual(hashlib.sha256(output.tobytes()).hexdigest(), digest)
