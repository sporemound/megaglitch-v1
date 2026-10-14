from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from megaglitch.mycotroph_sources import (
    AnimatedGifSource,
    ImageSource,
    SourceError,
    SourceService,
    SourceSlot,
    TestPatternSource as PatternSource,
    VideoSource,
    create_source,
    deterministic_test_pattern,
    probe_optional_adapters,
)


def _assert_rgb8(frame: np.ndarray, shape: tuple[int, int, int] | None = None) -> None:
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.flags.c_contiguous
    assert frame.flags.owndata
    if shape is not None:
        assert frame.shape == shape


def _write_rgb_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", size, color).save(path)


def _write_gif(path: Path) -> list[np.ndarray]:
    Image = pytest.importorskip("PIL.Image")
    colors = [(240, 10, 20), (20, 230, 30), (30, 40, 220)]
    frames = [Image.new("RGB", (4, 3), color) for color in colors]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[50, 150, 300],
        loop=0,
        disposal=2,
        optimize=False,
    )
    return [np.full((3, 4, 3), color, dtype=np.uint8) for color in colors]


def _write_video(tmp_path: Path) -> tuple[Path, list[np.ndarray], float] | None:
    cv2 = pytest.importorskip("cv2")
    width, height, fps = 24, 16, 10.0
    bgr_frames: list[np.ndarray] = []
    for index in range(6):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 15 + index * 20
        frame[:, :, 1] = 35 + index * 17
        frame[:, :, 2] = 220 - index * 24
        bgr_frames.append(frame)
    attempts: list[str] = []
    for suffix, codec in ((".avi", "MJPG"), (".avi", "XVID"), (".mp4", "mp4v")):
        path = tmp_path / f"source-{codec}{suffix}"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            attempts.append(f"{codec} writer unavailable")
            continue
        for frame in bgr_frames:
            writer.write(frame)
        writer.release()
        capture = cv2.VideoCapture(str(path))
        opened = capture.isOpened()
        decoded, probe = capture.read() if opened else (False, None)
        capture.release()
        if opened and decoded and probe is not None and probe.size:
            rgb_frames = [np.ascontiguousarray(frame[:, :, ::-1]) for frame in bgr_frames]
            return path, rgb_frames, fps
        attempts.append(f"{codec} output was not decodable")
    pytest.skip("no temporary video codec is available: " + "; ".join(attempts))


def _mean_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(np.abs(actual.astype(np.int16) - expected.astype(np.int16))))


def test_module_import_is_ui_independent_and_keeps_optional_decoders_lazy():
    root = Path(__file__).resolve().parents[1]
    script = (
        "import json,sys; "
        "import megaglitch.mycotroph_sources; "
        "print(json.dumps({'cv2': 'cv2' in sys.modules, "
        "'pillow': 'PIL.Image' in sys.modules, "
        "'dpg': any(k == 'dearpygui' or k.startswith('dearpygui.') for k in sys.modules)}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = json.loads(completed.stdout)
    assert imported == {"cv2": False, "pillow": False, "dpg": False}


def test_test_pattern_is_bridge_compatible_rgb8_and_resizes_truthfully():
    source = PatternSource(width=5, height=3)
    first = source.read(sequence=7)
    _assert_rgb8(first, (3, 5, 3))
    np.testing.assert_array_equal(first, deterministic_test_pattern(7, 5, 3))

    resized = source.read(sequence=8, width=2, height=4)
    _assert_rgb8(resized, (4, 2, 3))
    snapshot = source.snapshot()
    assert snapshot["frame"]["width"] == 2
    assert snapshot["frame"]["height"] == 4
    assert snapshot["frame"]["ownership"] == "caller_owned_copy"
    assert snapshot["transport"]["supported"] is False

    source.close()
    source.close()
    with pytest.raises(SourceError, match="closed") as caught:
        source.read()
    assert caught.value.code == "source_closed"


def test_image_alpha_policy_is_explicit_and_pixels_are_composited(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "alpha.png"
    rgba = np.array(
        [[[100, 50, 0, 128], [200, 100, 50, 255], [1, 2, 3, 0]]],
        dtype=np.uint8,
    )
    Image.fromarray(rgba, mode="RGBA").save(path)
    source = ImageSource(path, alpha_background=(10, 20, 30))
    try:
        frame = source.read()
        _assert_rgb8(frame, (1, 3, 3))
        expected = np.array([[[55, 35, 15], [200, 100, 50], [10, 20, 30]]], dtype=np.uint8)
        np.testing.assert_array_equal(frame, expected)

        # Reads are caller-owned; mutating one must not alter the source.
        frame[:] = 0
        np.testing.assert_array_equal(source.read(), expected)

        snapshot = source.snapshot()
        assert snapshot["metadata"]["alpha_present"] is True
        assert snapshot["metadata"]["alpha_handling"] == {
            "input": "straight RGBA",
            "output": "RGB8",
            "policy": "composite_over_solid_background",
            "rounding": "nearest_integer",
            "default_background_rgb": [0, 0, 0],
            "background_rgb": [10, 20, 30],
        }
    finally:
        source.close()
    assert source.snapshot()["ready"] is False


def test_gif_uses_frame_durations_seek_transport_and_loop(tmp_path):
    path = tmp_path / "animated.gif"
    expected = _write_gif(path)
    source = AnimatedGifSource(path, loop=True)
    try:
        assert source.frame_durations == pytest.approx((0.05, 0.15, 0.3), abs=0.001)
        np.testing.assert_array_equal(source.read(), expected[0])

        source.advance(0.06)
        np.testing.assert_array_equal(source.read(), expected[1])
        snapshot = source.snapshot()
        assert snapshot["transport"]["frame_index"] == 1
        assert snapshot["transport"]["position_seconds"] == pytest.approx(0.06)
        assert snapshot["transport"]["frame_durations_seconds"] == pytest.approx(
            [0.05, 0.15, 0.3], abs=0.001
        )

        source.seek(0.21)
        np.testing.assert_array_equal(source.read(), expected[2])
        source.pause()
        source.advance(2.0)
        assert source.snapshot()["transport"]["position_seconds"] == pytest.approx(0.21)

        source.play()
        source.seek(0.49)
        source.advance(0.03)
        looped = source.snapshot()
        assert looped["transport"]["frame_index"] == 0
        assert looped["transport"]["position_seconds"] == pytest.approx(0.02, abs=0.002)
        assert looped["transport"]["loops_completed"] == 1

        source.stop()
        stopped = source.snapshot()["transport"]
        assert stopped["state"] == "stopped"
        assert stopped["position_seconds"] == 0.0
        np.testing.assert_array_equal(source.read(), expected[0])

        source.play()
        source.set_loop(False)
        source.seek(0.49)
        source.advance(0.03)
        ended = source.snapshot()["transport"]
        assert ended["state"] == "ended"
        assert ended["position_seconds"] == pytest.approx(0.5, abs=0.002)
        np.testing.assert_array_equal(source.read(), expected[-1])
    finally:
        source.close()
    source.close()
    assert source.snapshot()["state"] == "closed"


def test_video_reports_real_transport_seek_loop_and_cleanup(tmp_path):
    created = _write_video(tmp_path)
    assert created is not None
    path, expected, source_fps = created
    source = VideoSource(path, loop=True)
    try:
        initial = source.snapshot()
        assert initial["transport"]["fps"] == pytest.approx(source_fps, rel=0.1)
        assert initial["transport"]["duration_seconds"] == pytest.approx(0.6, rel=0.15)
        assert initial["transport"]["frame_count"] == 6
        assert initial["metadata"]["backend_open"] is True
        assert initial["frame"]["width"] == 24
        assert initial["frame"]["height"] == 16

        first = source.read()
        second = source.read()
        _assert_rgb8(first, (16, 24, 3))
        assert _mean_error(first, expected[0]) < 12.0
        assert _mean_error(second, expected[1]) < 12.0

        source.pause()
        paused = source.read()
        np.testing.assert_array_equal(paused, source.read())
        assert source.snapshot()["transport"]["state"] == "paused"

        source.seek(0.3)
        sought = source.read()
        assert _mean_error(sought, expected[3]) < 16.0
        assert source.snapshot()["transport"]["position_seconds"] == pytest.approx(0.3, abs=0.11)

        source.stop()
        assert source.snapshot()["transport"]["state"] == "stopped"
        assert source.snapshot()["transport"]["position_seconds"] == pytest.approx(0.0)
        assert _mean_error(source.read(), expected[0]) < 12.0

        source.play()
        source.seek(0.5)
        assert _mean_error(source.read(), expected[5]) < 16.0
        assert _mean_error(source.read(), expected[0]) < 16.0
        assert source.snapshot()["transport"]["loops_completed"] >= 1

        source.set_loop(False)
        source.seek(0.5)
        source.read()
        final = source.read()
        assert _mean_error(final, expected[5]) < 16.0
        assert source.snapshot()["transport"]["state"] == "ended"
    finally:
        source.close()
    source.close()
    closed = source.snapshot()
    assert closed["ready"] is False
    assert closed["metadata"]["backend_open"] is False


def test_slot_replacement_tracks_resolution_and_failed_connect_is_atomic(tmp_path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    _write_rgb_image(first_path, (2, 1), (10, 20, 30))
    _write_rgb_image(second_path, (5, 4), (40, 50, 60))

    slot = SourceSlot("primary")
    first_snapshot = slot.connect("image", first_path)
    assert first_snapshot["generation"] == 1
    _assert_rgb8(slot.read(), (1, 2, 3))

    second_snapshot = slot.connect("image", second_path)
    assert second_snapshot["generation"] == 2
    _assert_rgb8(slot.read(), (4, 5, 3))
    assert slot.snapshot()["source"]["frame"]["width"] == 5

    with pytest.raises(SourceError) as caught:
        slot.connect("image", tmp_path / "missing.png")
    assert caught.value.code == "source_not_found"
    after_failure = slot.snapshot()
    assert after_failure["generation"] == 2
    assert after_failure["connected"] is True
    assert after_failure["last_error"]["code"] == "source_not_found"
    np.testing.assert_array_equal(slot.read()[0, 0], [40, 50, 60])

    disconnected = slot.disconnect()
    assert disconnected["generation"] == 3
    assert disconnected["connected"] is False
    with pytest.raises(SourceError) as disconnected_error:
        slot.read()
    assert disconnected_error.value.code == "source_disconnected"

    slot.close()
    slot.close()
    with pytest.raises(SourceError) as closed_error:
        slot.connect("test_pattern")
    assert closed_error.value.code == "slot_closed"


def test_service_slots_capabilities_and_deterministic_close():
    service = SourceService(("primary", "secondary"))
    service.connect("primary", "test_pattern", width=3, height=2)
    frame = service.read("primary", sequence=4)
    _assert_rgb8(frame, (2, 3, 3))
    snapshot = service.snapshot()
    assert snapshot["slots"]["primary"]["connected"] is True
    assert snapshot["slots"]["secondary"]["connected"] is False
    assert snapshot["optional_adapters"]["webcam"]["available"] is False
    assert snapshot["optional_adapters"]["window_capture"]["available"] is False

    service.close()
    service.close()
    closed = service.snapshot()
    assert closed["closed"] is True
    assert all(slot["closed"] for slot in closed["slots"].values())
    with pytest.raises(SourceError) as error:
        service.read("primary")
    assert error.value.code == "service_closed"


def test_optional_capture_probes_are_explicit_and_never_fake_frames():
    probes = probe_optional_adapters()
    assert set(probes) == {"webcam", "window_capture"}
    for kind, probe in probes.items():
        assert probe == {
            "version": 1,
            "kind": kind,
            "available": False,
            "reason_code": "adapter_not_registered",
            "reason": f"no {kind} adapter is registered in this runtime",
            "enumerated": False,
        }
        with pytest.raises(SourceError) as caught:
            create_source(kind)
        assert caught.value.code == "source_adapter_unavailable"
        assert caught.value.details["probe"]["available"] is False


@pytest.mark.parametrize(
    ("operation", "expected_code"),
    [
        (lambda: create_source("not-a-source"), "unsupported_source_kind"),
        (lambda: create_source("image", ""), "source_path_required"),
        (lambda: PatternSource(width=0), "invalid_dimensions"),
    ],
)
def test_errors_have_stable_structured_envelopes(operation, expected_code):
    with pytest.raises(SourceError) as caught:
        operation()
    error = caught.value.to_dict()
    assert error["code"] == expected_code
    assert isinstance(error["message"], str) and error["message"]
    assert error["recoverable"] is True


def test_invalid_transport_values_are_rejected_without_state_corruption(tmp_path):
    gif_path = tmp_path / "transport.gif"
    _write_gif(gif_path)
    source = AnimatedGifSource(gif_path)
    try:
        before = source.snapshot()["transport"]
        with pytest.raises(SourceError) as seek_error:
            source.seek(float("nan"))
        assert seek_error.value.code == "invalid_seek"
        with pytest.raises(SourceError) as elapsed_error:
            source.advance(-0.1)
        assert elapsed_error.value.code == "invalid_elapsed"
        with pytest.raises(SourceError) as loop_error:
            source.set_loop(1)  # type: ignore[arg-type]
        assert loop_error.value.code == "invalid_loop"
        after = source.snapshot()["transport"]
        assert after["position_seconds"] == before["position_seconds"]
        assert after["state"] == before["state"]
    finally:
        source.close()
