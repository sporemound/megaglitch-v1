import multiprocessing as mp
import queue
import unittest

import numpy as np

from megaglitch.shared_frame_transport import (
    SharedFrameTransportEndpoint,
    SharedFrameTransportOwner,
    is_shared_frame,
)


class SharedFrameTransportTests(unittest.TestCase):
    def owner(self, **kwargs):
        owner = SharedFrameTransportOwner(
            mp.get_context("spawn"), 8, 6, input_slots=kwargs.get("input_slots", 2),
            output_slots=kwargs.get("output_slots", 2),
        )
        self.addCleanup(owner.close)
        return owner

    def test_input_slot_roundtrip_and_reuse(self):
        owner = self.owner(input_slots=1)
        endpoint = SharedFrameTransportEndpoint(owner.spec())
        self.addCleanup(endpoint.close)
        frame = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        descriptor = owner.encode_frame(frame)
        self.assertTrue(is_shared_frame(descriptor, "input"))
        view, slot = endpoint.resolve_input(descriptor)
        np.testing.assert_array_equal(view, frame)
        view[0, 0] = 255
        self.assertTrue(np.all(owner._slot_array(owner.input_shm, slot, frame.shape)[0, 0] == 255))
        self.assertIs(owner.encode_frame(frame), False)
        endpoint.release_input_slot(slot)
        self.assertTrue(is_shared_frame(owner.encode_frame(frame), "input"))

    def test_payload_releases_all_acquired_slots(self):
        owner = self.owner(input_slots=2)
        frame = np.zeros((4, 6, 3), np.uint8)
        payload = owner.encode_payload({"image": frame, "secondary_image": frame + 1})
        self.assertTrue(is_shared_frame(payload["image"], "input"))
        self.assertTrue(is_shared_frame(payload["secondary_image"], "input"))
        self.assertEqual(payload["frame_copy_stats"]["shared_input"], {
            "count": 2, "bytes": frame.nbytes * 2,
        })
        owner.release_input_payload(payload)
        self.assertIsNot(owner.encode_frame(frame), False)
        owner.release_input_descriptor(owner.encode_frame(frame))

    def test_secondary_pool_saturation_rolls_back_primary_slot(self):
        owner = self.owner(input_slots=1)
        frame = np.zeros((4, 6, 3), np.uint8)
        self.assertIsNone(owner.encode_payload({
            "image": frame,
            "secondary_image": frame,
        }))
        self.assertEqual(owner.get_statistics()["input_slots_in_use"], 0)

    def test_output_is_canonical_copy_and_stale_slot_is_reclaimed(self):
        owner = self.owner(output_slots=1)
        endpoint = SharedFrameTransportEndpoint(owner.spec())
        self.addCleanup(endpoint.close)
        target = queue.Queue(maxsize=1)
        first = np.full((4, 6, 3), 10, np.uint8)
        second = np.full((4, 6, 3), 20, np.uint8)
        self.assertTrue(endpoint.publish_result(target, first, {"timestamp": 1.0}))
        # Publishing latest reclaims the queued stale slot instead of growing.
        self.assertTrue(endpoint.publish_result(target, second, {"timestamp": 2.0}))
        decoded = owner.decode_result(target.get_nowait())
        np.testing.assert_array_equal(decoded["frame"], second)
        self.assertIsNone(decoded["frame_a"])
        self.assertIsNone(decoded["frame_b"])
        self.assertEqual(decoded["frame_transport"], "shared_v1")
        self.assertEqual(decoded["frame_copy_stats"]["shared_output"], {
            "count": 1, "bytes": second.nbytes,
        })
        self.assertEqual(decoded["frame_copy_stats"]["ui_consumer"], {
            "count": 1, "bytes": second.nbytes,
        })
        stats = owner.get_statistics()
        self.assertEqual(stats["ui_consumer_copy_count"], 1)
        self.assertEqual(stats["ui_consumer_copy_bytes"], second.nbytes)

    def test_input_copy_statistics_report_full_frame_bytes(self):
        owner = self.owner()
        frame = np.zeros((4, 6, 3), np.uint8)
        descriptor = owner.encode_frame(frame)
        self.assertTrue(is_shared_frame(descriptor, "input"))
        stats = owner.get_statistics()
        self.assertEqual(stats["input_copy_count"], 1)
        self.assertEqual(stats["input_copy_bytes"], frame.nbytes)
        owner.release_input_descriptor(descriptor)

    def test_oversize_input_and_output_use_safe_legacy_contract(self):
        owner = self.owner()
        endpoint = SharedFrameTransportEndpoint(owner.spec())
        self.addCleanup(endpoint.close)
        oversize = np.zeros((7, 9, 3), np.uint8)
        self.assertIsNone(owner.encode_frame(oversize))
        target = queue.Queue(maxsize=1)
        self.assertTrue(endpoint.publish_result(target, oversize, {"timestamp": 3.0}))
        result = target.get_nowait()
        self.assertEqual(result["frame_transport"], "legacy_oversize")
        self.assertEqual(result["frame"].shape, oversize.shape)
        self.assertNotIn("frame_a", result)
        self.assertNotIn("frame_b", result)

    def test_close_is_idempotent(self):
        owner = self.owner()
        endpoint = SharedFrameTransportEndpoint(owner.spec())
        endpoint.close()
        endpoint.close()
        owner.close()
        owner.close()

    def test_spawned_worker_canonical_output_owns_crossfader_endpoints_and_soaks_cleanly(self):
        from megaglitchfm_app_new import glitch_process_worker

        context = mp.get_context("spawn")
        owner = SharedFrameTransportOwner(
            context, 32, 24, input_slots=4, output_slots=2
        )
        raw_queue = context.Queue(maxsize=2)
        output_queue = context.Queue(maxsize=2)
        stop_event = context.Event()
        process = context.Process(
            target=glitch_process_worker,
            args=(raw_queue, output_queue, stop_event, None, owner.spec()),
        )
        process.start()
        try:
            import psutil
            worker_metrics = psutil.Process(process.pid)
        except Exception:
            worker_metrics = None
        deck_a = np.full((24, 32, 3), 17, np.uint8)
        deck_b = np.full((24, 32, 3), 231, np.uint8)
        display = [{"effect_type": "Display", "params": {}}]
        inverted = [
            {"effect_type": "Invert Colors", "params": {"invert_chance": 1}},
            {"effect_type": "Display", "params": {}},
        ]
        try:
            preserved_first = None
            steady_rss = None
            steady_handles = None
            for index in range(60):
                mix = 0.0 if index % 2 == 0 else 1.0
                active_pipeline = inverted if index % 10 >= 5 else display
                payload = owner.encode_payload({
                    "image": deck_a,
                    "secondary_image": deck_b,
                    "pipeline": active_pipeline,
                    "pipeline_b": active_pipeline,
                    "master_mix": mix,
                    "master_transition": "Normal Mix",
                    "extra_params": {"frame_num": index, "time_accum": index / 60.0},
                    "use_cuda": False,
                    "fallback_width": 32,
                    "fallback_height": 24,
                    "timestamp": 0.0,
                }, block=True, timeout=10.0)
                self.assertIsNotNone(payload)
                raw_queue.put(payload, timeout=10.0)
                decoded = owner.decode_result(output_queue.get(timeout=30.0))
                expected = deck_a if mix == 0.0 else deck_b
                if active_pipeline is inverted:
                    expected = 255 - expected
                np.testing.assert_array_equal(decoded["frame"], expected)
                self.assertIsNone(decoded["frame_a"])
                self.assertIsNone(decoded["frame_b"])
                if preserved_first is None:
                    preserved_first = decoded["frame"]
                if index == 9 and worker_metrics is not None:
                    steady_rss = worker_metrics.memory_info().rss
                    steady_handles = worker_metrics.num_handles()
                if index == 29:
                    raw_queue.put("RESET", timeout=10.0)
                    self.assertEqual(output_queue.get(timeout=30.0), "RESET")
            # Decoded recording/preview/virtual-camera consumers own a stable copy.
            np.testing.assert_array_equal(preserved_first, deck_a)
            stats = owner.get_statistics()
            self.assertEqual(stats["input_slots_in_use"], 0)
            self.assertEqual(stats["output_slots_in_use"], 0)
            if worker_metrics is not None and steady_rss is not None:
                self.assertLess(worker_metrics.memory_info().rss - steady_rss, 64 * 1024 * 1024)
                self.assertLessEqual(worker_metrics.num_handles() - steady_handles, 8)
        finally:
            stop_event.set()
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            raw_queue.close()
            output_queue.close()
            owner.close()

    def test_repeated_worker_start_stop_releases_all_shared_ownership(self):
        from megaglitchfm_app_new import glitch_process_worker

        context = mp.get_context("spawn")
        frame = np.full((12, 16, 3), 91, np.uint8)
        for _ in range(3):
            owner = SharedFrameTransportOwner(
                context, 16, 12, input_slots=2, output_slots=1
            )
            raw_queue = context.Queue(maxsize=1)
            output_queue = context.Queue(maxsize=1)
            stop_event = context.Event()
            process = context.Process(
                target=glitch_process_worker,
                args=(raw_queue, output_queue, stop_event, None, owner.spec()),
            )
            process.start()
            try:
                payload = owner.encode_payload({
                    "image": frame,
                    "secondary_image": None,
                    "pipeline": [],
                    "pipeline_b": [],
                    "master_mix": 0.0,
                    "extra_params": {},
                    "use_cuda": False,
                    "fallback_width": 16,
                    "fallback_height": 12,
                    "timestamp": 0.0,
                }, block=True, timeout=10.0)
                raw_queue.put(payload, timeout=10.0)
                decoded = owner.decode_result(output_queue.get(timeout=30.0))
                np.testing.assert_array_equal(decoded["frame"], frame)
            finally:
                stop_event.set()
                process.join(timeout=10.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
                self.assertFalse(process.is_alive())
                self.assertEqual(owner.get_statistics()["input_slots_in_use"], 0)
                self.assertEqual(owner.get_statistics()["output_slots_in_use"], 0)
                raw_queue.close()
                output_queue.close()
                owner.close()


if __name__ == "__main__":
    unittest.main()
