import hashlib
import mmap
from concurrent.futures import Future

import numpy as np
import pytest

from tools.rust_ui_bridge import (
    BridgeEngine,
    BridgeProtocolError,
    PREVIEW_SLOT_BYTES,
    PreviewSurface,
    compile_preset_graph,
    create_source,
    deterministic_frame,
    downscale_preview_rgb,
    process_pipeline,
    _process_authoritative_frame,
)
from tools.benchmark.run_rust_ui_bridge import run_benchmark


def test_downscale_preview_rgb_returns_contiguous_720p_rgb8_from_1080p():
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    source[..., 0] = 17
    source[..., 1] = 83
    source[..., 2] = 241

    preview = downscale_preview_rgb(source)

    assert preview is not None
    assert preview.shape == (720, 1280, 3)
    assert preview.dtype == np.uint8
    assert preview.flags.c_contiguous
    np.testing.assert_array_equal(preview[0, 0], source[0, 0])


def invert_preset():
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "InvertColorNode:1",
                "type": "InvertColorNode",
                "label": "Invert Colors",
                "position": [200, 100],
                "params": {"enabled": True, "invert_chance": 100},
            }
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Invert Colors",
                "to_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Invert Colors",
                "to_node": "DISPLAY",
                "from_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def feedback_trail_preset():
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "FeedbackTrailNode:1",
                "type": "FeedbackTrailNode",
                "label": "Feedback / Trail",
                "params": {
                    "enabled": True,
                    "feedback_intensity": 0.25,
                    "trailing_intensity": 0.5,
                },
            }
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Feedback / Trail",
                "to_node_id": "FeedbackTrailNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Feedback / Trail",
                "from_node_id": "FeedbackTrailNode:1",
                "to_node": "DISPLAY",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def branch_mixer_preset():
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "InvertColorNode:1",
                "type": "InvertColorNode",
                "label": "Invert Colors",
                "params": {"enabled": True, "invert_chance": 100},
            },
            {
                "id": "VideoMixerNode:1",
                "type": "VideoMixerNode",
                "label": "Video Mixer",
                "params": {"enabled": True, "mix": 0.5},
            },
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Invert Colors",
                "to_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Invert Colors",
                "from_node_id": "InvertColorNode:1",
                "to_node": "Video Mixer",
                "to_node_id": "VideoMixerNode:1",
                "from_attr": "image_out",
                "to_attr": "image_a",
            },
            {
                "from_node": "SOURCE",
                "to_node": "Video Mixer",
                "to_node_id": "VideoMixerNode:1",
                "from_attr": "image_out",
                "to_attr": "image_b",
            },
            {
                "from_node": "Video Mixer",
                "from_node_id": "VideoMixerNode:1",
                "to_node": "DISPLAY",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def secondary_mixer_preset():
    preset = branch_mixer_preset()
    preset["nodes"] = [preset["nodes"][1]]
    preset["links"] = [
        {
            "from_node": "SOURCE",
            "to_node": "Video Mixer",
            "to_node_id": "VideoMixerNode:1",
            "from_attr": "image_out",
            "to_attr": "image_a",
        },
        {
            "from_node": "SOURCE_B",
            "to_node": "Video Mixer",
            "to_node_id": "VideoMixerNode:1",
            "from_attr": "image_out",
            "to_attr": "image_b",
        },
        preset["links"][-1],
    ]
    return preset


def test_bridge_compiles_and_executes_existing_invert_effect_exactly():
    pipeline = compile_preset_graph(invert_preset())
    frame = deterministic_frame(7, 64, 36)
    output, _state = process_pipeline(pipeline, frame)
    np.testing.assert_array_equal(output, 255 - frame)
    assert hashlib.sha256(output.tobytes()).hexdigest()


def test_snapshot_labels_draining_authoritative_output_with_its_graph_generation():
    with PreviewSurface(generation=17) as surface:
        engine = BridgeEngine(64, 36, surface=surface, hash_mode="sha256")
        try:
            old_frame = deterministic_frame(14, 64, 36)
            old_result = _process_authoritative_frame(
                engine.pipeline,
                old_frame,
                engine.frame_state,
                14,
                engine._authoritative_epoch,
                engine.graph_generation,
            )
            old_future = Future()
            old_future.set_result(old_result)
            engine._authoritative_future = old_future
            engine._poll_authoritative()

            ack, shutdown = engine.handle_command(
                {
                    "version": 2,
                    "id": "graph-invert",
                    "expected_state_revision": 0,
                    "type": "replace_graph",
                    "graph": invert_preset(),
                }
            )
            assert not shutdown and ack["accepted"] is True

            pending = engine.snapshot()
            assert pending["graph_generation"] == 1
            assert pending["authoritative_graph_generation"] == 0
            assert pending["graph_sequence"] == 14
            assert pending["output_sha256"] == hashlib.sha256(old_frame).hexdigest()

            new_frame = deterministic_frame(15, 64, 36)
            new_result = _process_authoritative_frame(
                engine.pipeline,
                new_frame,
                engine.frame_state,
                15,
                engine._authoritative_epoch,
                engine.graph_generation,
            )
            new_future = Future()
            new_future.set_result(new_result)
            engine._authoritative_future = new_future
            engine._poll_authoritative()

            current = engine.snapshot()
            assert current["graph_generation"] == 1
            assert current["authoritative_graph_generation"] == 1
            assert current["graph_sequence"] == 15
            assert current["output_sha256"] == hashlib.sha256(255 - new_frame).hexdigest()
        finally:
            engine.close()


def test_feedback_trail_sequence_matches_fixed_canonical_hashes():
    expected_hashes = (
        "2979d22411daf9aa1e0c10199609858325047ce652766262e27ddacba27bb270",
        "284275d176f8e29579e761085592419afd66b9ec45b5c5d0159c321910ce0c43",
        "db3104f30d7d9dc70cf9da75fc2cc387aec148eb7293a36ba4996eaff6fcf1c9",
        "e33b153e4cc95b52ee782522fb4aa5cc4da39e21858c128fd7c09b91dfce9e9d",
        "8f7e69bc7fe7607e9532e59e13d640b7140d58a2f632579f7491ddad6bc5d645",
        "8de032a145b97949b28c5c38a8669c8eee6d30708c8eb785e5a39864dd1f7b33",
    )
    pipeline = compile_preset_graph(feedback_trail_preset())
    state = None
    actual_hashes = []
    for sequence in range(1, len(expected_hashes) + 1):
        output, state = process_pipeline(
            pipeline,
            deterministic_frame(sequence, 32, 18),
            state,
        )
        actual_hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
    assert tuple(actual_hashes) == expected_hashes


def test_bridge_compiles_and_executes_branching_mixer_graph():
    pipeline = compile_preset_graph(branch_mixer_preset())
    assert [step["node_type"] for step in pipeline] == [
        "source",
        "effect",
        "source",
        "mixer",
        "display",
    ]

    frame = deterministic_frame(11, 64, 36)
    output, _state = process_pipeline(pipeline, frame)
    np.testing.assert_array_equal(output, np.full_like(frame, 127))


def test_bridge_reports_effects_omitted_by_preview_fallback_and_clears_on_replacement():
    with PreviewSurface(generation=42) as surface:
        engine = BridgeEngine(64, 36, surface=surface)
        try:
            for revision, preset in enumerate((branch_mixer_preset(), invert_preset())):
                ack, shutdown = engine.handle_command(
                    {
                        "version": 2,
                        "id": f"coverage-{revision}",
                        "expected_state_revision": revision,
                        "type": "replace_graph",
                        "graph": preset,
                    }
                )
                assert ack["accepted"] and not shutdown
                snapshot = engine.snapshot()
                statuses = {node["node_id"]: node for node in snapshot["preview_node_statuses"]}
                invert = statuses["InvertColorNode:1"]
                if revision == 0:
                    assert invert["status"] == "unsupported"
                    assert invert["provider"] == "source_display_fallback"
                    assert invert in snapshot["preview_unsupported_nodes"]
                    assert statuses["VideoMixerNode:1"]["status"] == "unsupported"
                else:
                    assert invert["status"] == "implemented"
                    assert snapshot["preview_unsupported_nodes"] == []
        finally:
            engine.close()


def test_bridge_preserves_generator_selected_outputs_and_multi_input_order():
    preset = {
        "version": "1.0",
        "nodes": [
            {
                "id": "BubbleSimGenerator2Node:1",
                "type": "BubbleSimGenerator2Node",
                "label": "Bubble Sim Gen (2)",
                "params": {},
            },
            {
                "id": "GenesisMaskKeyNode:1",
                "type": "GenesisMaskKeyNode",
                "label": "Genesis Mask Key",
                "params": {},
            },
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Bubble Sim Gen (2)",
                "to_node_id": "BubbleSimGenerator2Node:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Bubble Sim Gen (2)",
                "from_node_id": "BubbleSimGenerator2Node:1",
                "to_node": "Genesis Mask Key",
                "to_node_id": "GenesisMaskKeyNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Bubble Sim Gen (2)",
                "from_node_id": "BubbleSimGenerator2Node:1",
                "to_node": "Genesis Mask Key",
                "to_node_id": "GenesisMaskKeyNode:1",
                "from_attr": "bubble_mask",
                "to_attr": "mask_in",
            },
            {
                "from_node": "Genesis Mask Key",
                "from_node_id": "GenesisMaskKeyNode:1",
                "to_node": "DISPLAY",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }

    pipeline = compile_preset_graph(preset)
    generators = [step for step in pipeline if step.get("is_generator")]
    assert [step["selected_output"] for step in generators] == ["image_out", "bubble_mask"]
    assert all(step["input_order"] == ["image_in"] for step in generators)
    assert pipeline[-2]["node_type"] == "genesis_mask_key"
    assert pipeline[-2]["input_order"] == ["image_in", "mask_in"]


def test_bridge_rejects_disconnected_nodes_instead_of_silently_ignoring_them():
    preset = invert_preset()
    preset["nodes"].append(
        {
            "id": "ColorAdjustNode:1",
            "type": "ColorAdjustNode",
            "label": "Color Adjust",
            "position": [400, 100],
            "params": {"enabled": True},
        }
    )
    with pytest.raises(BridgeProtocolError) as caught:
        compile_preset_graph(preset)
    assert caught.value.code == "unreachable_nodes"
    assert caught.value.details["node_ids"] == ["ColorAdjustNode:1"]


def test_transactional_graph_rejection_preserves_last_accepted_pipeline():
    with PreviewSurface(generation=41) as surface:
        engine = BridgeEngine(64, 36, surface=surface)
        try:
            accepted, shutdown = engine.handle_command(
                {
                    "version": 2,
                    "id": "graph-good",
                    "expected_state_revision": 0,
                    "type": "replace_graph",
                    "graph": invert_preset(),
                }
            )
            assert not shutdown
            assert accepted["accepted"] is True
            assert accepted["command_type"] == "replace_graph"
            assert accepted["state_revision"] == 1
            assert accepted["graph_generation"] == 1
            last_accepted_pipeline = engine.pipeline

            rejected_graph = secondary_mixer_preset()
            rejected, shutdown = engine.handle_command(
                {
                    "version": 2,
                    "id": "graph-bad",
                    "expected_state_revision": 1,
                    "type": "replace_graph",
                    "graph": rejected_graph,
                }
            )
            assert not shutdown
            assert rejected["accepted"] is False
            assert rejected["state_revision"] == 1
            assert rejected["graph_generation"] == 1
            assert rejected["error"]["code"] == "secondary_source_unavailable"
            assert rejected["error"]["details"]["source_key"] == "secondary"
            assert engine.pipeline is last_accepted_pipeline

            disconnected = invert_preset()
            disconnected["nodes"].append(
                {
                    "id": "ColorAdjustNode:1",
                    "type": "ColorAdjustNode",
                    "label": "Color Adjust",
                    "params": {"enabled": True},
                }
            )
            invalid, shutdown = engine.handle_command(
                {
                    "version": 2,
                    "id": "graph-disconnected",
                    "expected_state_revision": 1,
                    "type": "replace_graph",
                    "graph": disconnected,
                }
            )
            assert not shutdown
            assert invalid["accepted"] is False
            assert invalid["error"]["code"] == "unreachable_nodes"
            assert invalid["error"]["details"]["node_ids"] == ["ColorAdjustNode:1"]
            assert engine.pipeline is last_accepted_pipeline

            stale, _shutdown = engine.handle_command(
                {
                    "version": 2,
                    "id": "stale-transport",
                    "expected_state_revision": 0,
                    "type": "set_transport",
                    "running": False,
                }
            )
            assert stale["accepted"] is False
            assert stale["error"]["code"] == "state_revision_conflict"
            assert engine.transport_running is True
        finally:
            engine.close()


def test_preview_surface_is_headerless_rgb8_and_requires_matching_release():
    rgb = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    with PreviewSurface(generation=77) as surface:
        event = surface.write_frame(rgb, sequence=9, timestamp=12.5)
        assert event is not None
        assert event["slot"] == 0
        assert event["generation"] == 77
        assert event["stride"] == 6
        assert event["nbytes"] == rgb.nbytes
        assert PREVIEW_SLOT_BYTES >= event["nbytes"]
        expected = rgb.tobytes()
        assert surface.read_slot_bytes(0, len(expected)) == expected
        with surface.path.open("rb") as preview_file:
            mapped = mmap.mmap(preview_file.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                assert mapped[: len(expected)] == expected
            finally:
                mapped.close()
        assert surface.release(0, 77, sequence=8) is False
        assert surface.release(0, 76, sequence=9) is False
        assert surface.release(0, 77, sequence=9) is True


def test_preview_surface_drops_when_all_three_slots_are_leased():
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    with PreviewSurface(generation=5) as surface:
        assert [surface.write_frame(rgb, sequence=i, timestamp=0.0)["slot"] for i in range(3)] == [
            0,
            1,
            2,
        ]
        assert surface.write_frame(rgb, sequence=4, timestamp=0.0) is None


def test_image_source_loads_lazily_and_publishes_rgb_preview(tmp_path):
    cv2 = pytest.importorskip("cv2")
    image_path = tmp_path / "bridge-source.png"
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    assert cv2.imwrite(str(image_path), bgr)

    source = create_source("image", str(image_path))
    try:
        np.testing.assert_array_equal(source.read(1, 2, 1), rgb)
    finally:
        source.close()

    with PreviewSurface(generation=88) as surface:
        engine = BridgeEngine(2, 1, surface=surface)
        try:
            ack, shutdown = engine.handle_command(
                {
                    "version": 2,
                    "id": "source-image",
                    "expected_state_revision": 0,
                    "type": "set_source",
                    "kind": "image",
                    "path": str(image_path),
                }
            )
            assert not shutdown
            assert ack["accepted"] is True
            assert engine.source_kind == "image"
            assert engine.source_label == str(image_path.resolve())
            events = engine.render_frame()
            preview = next(event for event in events if event["type"] == "preview_frame")
            assert surface.read_slot_bytes(preview["slot"], preview["nbytes"]) == rgb.tobytes()
            snapshot = engine.snapshot()
            assert snapshot["source_kind"] == "image"
            assert snapshot["source_label"] == str(image_path.resolve())
            assert snapshot["source_ready"] is True
            assert snapshot["state_revision"] == 1
        finally:
            engine.close()


def test_missing_image_rejection_is_structured_and_does_not_replace_source(tmp_path):
    with PreviewSurface(generation=99) as surface:
        engine = BridgeEngine(2, 1, surface=surface)
        try:
            ack, shutdown = engine.handle_command(
                {
                    "version": 2,
                    "id": "missing-image",
                    "expected_state_revision": 0,
                    "type": "set_source",
                    "kind": "image",
                    "path": str(tmp_path / "not-there.png"),
                }
            )
            assert not shutdown
            assert ack["accepted"] is False
            assert ack["error"]["code"] == "source_not_found"
            assert ack["error"]["recoverable"] is True
            assert ack["error"]["path"] == str((tmp_path / "not-there.png").resolve())
            assert ack["state_revision"] == 0
            assert engine.source_kind == "test_pattern"
            assert engine.source_ready is True
        finally:
            engine.close()


def test_video_source_reads_and_seeks_real_opencv_container(tmp_path):
    cv2 = pytest.importorskip("cv2")
    width, height, fps = 32, 24, 10.0
    source_frames = []
    for index in range(10):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 20 + index * 15
        frame[:, :, 1] = 40 + index * 10
        frame[:, :, 2] = 210 - index * 12
        source_frames.append(frame)

    video_path = None
    attempts = []
    for suffix, codec in ((".avi", "MJPG"), (".avi", "XVID"), (".mp4", "mp4v")):
        candidate = tmp_path / f"bridge-video-{codec}{suffix}"
        writer = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            attempts.append(f"{codec}: writer unavailable")
            continue
        for frame in source_frames:
            writer.write(frame)
        writer.release()
        probe = cv2.VideoCapture(str(candidate))
        opened = probe.isOpened()
        decoded, probe_frame = probe.read() if opened else (False, None)
        probe.release()
        if opened and decoded and probe_frame is not None and probe_frame.size:
            video_path = candidate
            break
        attempts.append(f"{codec}: written container could not be decoded")
    if video_path is None:
        pytest.skip("no temporary OpenCV video codec initialized: " + "; ".join(attempts))

    source = create_source("video", str(video_path))
    try:
        first = source.read(1, width, height)
        second = source.read(2, width, height)
        assert first.shape == (height, width, 3)
        assert np.mean(np.abs(second.astype(np.int16) - first.astype(np.int16))) > 5.0

        source.seek(0.0)
        rewound = source.read(3, width, height)
        assert np.mean(np.abs(rewound.astype(np.int16) - first.astype(np.int16))) < 8.0

        source.seek(0.5)
        sought = source.read(4, width, height)
        nearby_targets = [
            np.ascontiguousarray(source_frames[index][:, :, ::-1])
            for index in (4, 5, 6)
        ]
        sought_error = min(
            float(np.mean(np.abs(sought.astype(np.int16) - target.astype(np.int16))))
            for target in nearby_targets
        )
        first_error = min(
            float(np.mean(np.abs(first.astype(np.int16) - target.astype(np.int16))))
            for target in nearby_targets
        )
        assert sought_error < 20.0
        assert sought_error < first_error
    finally:
        source.close()


def test_protocol_v2_benchmark_maps_pixels_releases_slots_and_cleans_surface():
    result = run_benchmark(
        width=48,
        height=27,
        warmup=1,
        iterations=2,
        startup_timeout=10.0,
        shutdown_timeout=5.0,
    )
    assert result["protocol_version"] == 2
    assert result["preview"]["mapped_samples"] >= 3
    assert result["preview"]["slot_releases"] >= 3
    assert result["lifecycle"]["shutdown_revision"] == 3
    assert result["parity_failures"] == []
    assert all(result["gates"].values())
