import time
import numpy as np
import pytest

from megaglitch.preview_v3_ring import (
    PreviewV3Producer,
    PreviewV3Consumer,
    FMT_RGB8_PACKED,
)


def test_v3_ring_creation_and_descriptor():
    generation = 123456789
    producer = PreviewV3Producer(generation=generation, max_width=1280, max_height=720, slot_count=3)
    desc = producer.descriptor()
    assert desc["version"] == 3
    assert desc["type"] == "preview_v3_ring"
    assert desc["generation"] == generation
    assert desc["slots"] == 3
    assert desc["pixel_format"] == "RGB8_PACKED"
    assert desc["max_width"] == 1280
    assert desc["max_height"] == 720
    producer.close()


def test_v3_ring_lossless_pixel_transport_and_monotonic_timestamp():
    generation = 99887766
    producer = PreviewV3Producer(generation=generation, max_width=640, max_height=360, slot_count=3)
    consumer = PreviewV3Consumer(tag_name=producer.tag_name, total_bytes=producer.total_bytes, generation=generation)

    test_frame = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)
    t_before = time.monotonic_ns()
    producer.write_frame(test_frame, sequence=42)
    t_after = time.monotonic_ns()

    acquired = consumer.acquire_latest_frame()
    assert acquired is not None
    assert acquired["sequence"] == 42
    assert t_before <= acquired["timestamp_ns"] <= t_after
    assert acquired["width"] == 640
    assert acquired["height"] == 360
    assert acquired["pixel_format"] == FMT_RGB8_PACKED
    assert acquired["nbytes"] == 640 * 360 * 3

    # Exact pixel payload equality
    payload_arr = np.frombuffer(acquired["payload"], dtype=np.uint8).reshape((360, 640, 3))
    np.testing.assert_array_equal(payload_arr, test_frame)

    # Subsequent immediate acquire returns None (already consumed)
    assert consumer.acquire_latest_frame() is None

    producer.close()
    consumer.close()


def test_v3_ring_lossy_overwrites_skip_to_newest_frame_without_stalling():
    generation = 44556677
    producer = PreviewV3Producer(generation=generation, max_width=640, max_height=360, slot_count=3)
    consumer = PreviewV3Consumer(tag_name=producer.tag_name, total_bytes=producer.total_bytes, generation=generation)

    # Producer writes 10 frames in rapid succession (consumer did not read intermediate frames)
    last_frame = None
    for seq in range(1, 11):
        frame = np.full((360, 640, 3), seq, dtype=np.uint8)
        producer.write_frame(frame, sequence=seq)
        last_frame = frame

    # Consumer acquires directly the latest frame (sequence 10)
    acquired = consumer.acquire_latest_frame()
    assert acquired is not None
    assert acquired["sequence"] == 10
    payload_arr = np.frombuffer(acquired["payload"], dtype=np.uint8).reshape((360, 640, 3))
    np.testing.assert_array_equal(payload_arr, last_frame)

    producer.close()
    consumer.close()


def test_v3_ring_stale_generation_consumer_is_rejected():
    gen1 = 11111111
    gen2 = 22222222
    producer = PreviewV3Producer(generation=gen2, max_width=640, max_height=360, slot_count=3)
    stale_consumer = PreviewV3Consumer(tag_name=producer.tag_name, total_bytes=producer.total_bytes, generation=gen1)

    frame = np.ones((360, 640, 3), dtype=np.uint8)
    producer.write_frame(frame, sequence=1)

    # Consumer with mismatched generation fails seqlock check
    assert stale_consumer.acquire_latest_frame() is None

    producer.close()
    stale_consumer.close()
