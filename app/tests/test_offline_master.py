from __future__ import annotations

import io
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

from megaglitch.offline_master import (
    FFmpegEncodeError,
    FFmpegEncodeResult,
    FFmpegMasterConfig,
    FFmpegMasterEncoder,
    FpsSemantics,
    MasterCodec,
    OfflineMasterConfig,
    OfflineMasterError,
    OfflineMasterRenderer,
    build_ffmpeg_master_command,
    build_frame_schedule,
    stable_graph_hash,
)


class _MemorySink:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.aborted = False

    def write_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> FFmpegEncodeResult:
        byte_count = sum(frame.nbytes for frame in self.frames)
        return FFmpegEncodeResult(
            command=("memory",),
            frames_written=len(self.frames),
            bytes_written=byte_count,
            returncode=0,
            stderr="",
            elapsed_seconds=0.0,
            readback={"available": True, "bit_rate": "12000000"},
        )

    def abort(self) -> None:
        self.aborted = True


def _config(
    *,
    semantics: FpsSemantics = FpsSemantics.TRUE_ACCELERATED,
    source_fps: int = 30,
    output_fps: int = 55,
    seed: int = 1234,
) -> OfflineMasterConfig:
    graph_hash = stable_graph_hash({"nodes": ["existing-megaglitch-callback"]})
    return OfflineMasterConfig(
        width=3,
        height=2,
        source_fps=source_fps,
        output_fps=output_fps,
        fps_semantics=semantics,
        graph_hash=graph_hash,
        random_seed=seed,
        history_buffer_resolution={
            "previous_output": (3, 2),
            "feedback": (3, 2),
            "p_frame_history": (3, 2),
        },
        node_provider_assignment={"feedback": "cuda", "restore": "cuda"},
    )


def test_true_acceleration_keeps_each_source_frame_once_and_shortens_duration():
    schedule = build_frame_schedule(30, 30, 55, FpsSemantics.TRUE_ACCELERATED)

    assert [item.source_index for item in schedule.items] == list(range(30))
    assert schedule.output_frame_count == 30
    assert float(schedule.source_duration_seconds) == 1.0
    assert float(schedule.output_duration_seconds) == pytest.approx(30 / 55)
    assert schedule.duplicated_frames == 0
    assert schedule.source_frames_omitted_at_egress == 0


def test_duration_preserving_conversion_duplicates_only_at_encoder_egress():
    schedule = build_frame_schedule(3, 30, 60, FpsSemantics.DURATION_PRESERVING)

    assert [item.source_index for item in schedule.items] == [0, 0, 1, 1, 2, 2]
    assert schedule.output_frame_count == 6
    assert float(schedule.source_duration_seconds) == pytest.approx(0.1)
    assert float(schedule.output_duration_seconds) == pytest.approx(0.1)
    assert schedule.duplicated_frames == 3


def test_lower_duration_preserving_rate_still_processes_every_master_frame():
    frames = [np.full((2, 3, 3), index, np.uint8) for index in range(4)]
    calls: list[int] = []
    sink = _MemorySink()

    def processor(frame, context):
        calls.append(context.source_index)
        return frame

    result = OfflineMasterRenderer(
        _config(
            semantics=FpsSemantics.DURATION_PRESERVING,
            source_fps=30,
            output_fps=15,
        )
    ).render(frames, source_frame_count=4, processor=processor, encoder=sink)

    assert calls == [0, 1, 2, 3]
    assert [int(frame[0, 0, 0]) for frame in sink.frames] == [0, 2]
    assert result.telemetry["dropped_frames"] == 0
    assert result.telemetry["source_frames_omitted_at_egress"] == 2
    assert result.telemetry["frame_count"] == {
        "source": 4,
        "master_processed": 4,
        "encoded": 2,
    }


def test_renderer_is_reproducible_and_restores_process_rng_state():
    frames = [np.full((2, 3, 3), 20 + index, np.uint8) for index in range(3)]

    def run_once(seed: int):
        sink = _MemorySink()

        def existing_callback(frame, context):
            context.instrumentation.record_cpu_gpu_transfer(
                "cpu_to_gpu", frame.nbytes, context.source_index
            )
            noise = np.random.randint(0, 16, frame.shape, dtype=np.uint8)
            modern_noise = context.numpy_rng.integers(0, 4, frame.shape, dtype=np.uint8)
            offset = random.randrange(8) + context.python_rng.randrange(4)
            return np.add(frame, noise + modern_noise + offset, dtype=np.uint8)

        before_python = random.getstate()
        before_numpy = np.random.get_state()
        result = OfflineMasterRenderer(_config(seed=seed)).render(
            frames,
            source_frame_count=3,
            processor=existing_callback,
            encoder=sink,
        )
        assert random.getstate() == before_python
        after_numpy = np.random.get_state()
        assert before_numpy[0] == after_numpy[0]
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        assert before_numpy[2:] == after_numpy[2:]
        return sink.frames, result

    left_frames, left = run_once(917)
    right_frames, right = run_once(917)
    changed_frames, _changed = run_once(918)

    for a, b in zip(left_frames, right_frames, strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(left_frames, changed_frames, strict=True)
    )
    telemetry = left.telemetry
    assert telemetry["mode"] == "offline_master"
    assert telemetry["resolution"] == [3, 2]
    assert telemetry["source_fps"] == 30.0
    assert telemetry["output_fps"] == 55.0
    assert telemetry["fps_semantics"] == "true_accelerated"
    assert telemetry["cpu_gpu_transfers"]["count"] == 3
    assert telemetry["cpu_gpu_transfers"]["bytes"] == frames[0].nbytes * 3
    assert telemetry["history_buffer_resolution"]["feedback"] == [3, 2]
    assert telemetry["node_provider_assignment"]["feedback"] == "cuda"
    assert telemetry["graph_hash"] == _config().graph_hash
    assert telemetry["random_seed"] == 917
    assert telemetry["bitrate"] == "12000000"
    assert telemetry["render_time_per_frame"]["samples"] == 3
    assert telemetry["encode_time_per_frame"]["frames"] == 3
    assert telemetry["encode_time_per_frame"]["stdin_write"]["samples"] == 3


def test_renderer_pins_and_restores_opencv_process_scheduling():
    import cv2

    before_threads = int(cv2.getNumThreads())
    before_opencl = bool(cv2.ocl.useOpenCL())
    observed: list[tuple[int, bool]] = []
    sink = _MemorySink()

    def processor(frame, _context):
        observed.append((int(cv2.getNumThreads()), bool(cv2.ocl.useOpenCL())))
        return frame

    result = OfflineMasterRenderer(_config()).render(
        [np.zeros((2, 3, 3), dtype=np.uint8)],
        source_frame_count=1,
        processor=processor,
        encoder=sink,
    )

    assert observed == [(1, False)]
    assert int(cv2.getNumThreads()) == before_threads
    assert bool(cv2.ocl.useOpenCL()) == before_opencl
    assert result.telemetry["opencv_determinism"]["available"] is True
    assert result.telemetry["opencv_determinism"]["num_threads"] == 1
    assert result.telemetry["opencv_determinism"]["opencl"] is False


def test_renderer_rejects_preview_sized_ingress_and_history_buffers():
    config = _config()
    with pytest.raises(ValueError, match="expected master resolution"):
        OfflineMasterConfig(
            width=3,
            height=2,
            source_fps=30,
            output_fps=30,
            fps_semantics=FpsSemantics.TRUE_ACCELERATED,
            graph_hash=config.graph_hash,
            history_buffer_resolution={"feedback": (2, 1)},
        )

    with pytest.raises(OfflineMasterError, match="master path requires"):
        OfflineMasterRenderer(config).render(
            [np.zeros((1, 2, 3), np.uint8)],
            source_frame_count=1,
            processor=lambda frame, context: frame,
            encoder=_MemorySink(),
        )

    with pytest.raises(OfflineMasterError, match="at least one source frame"):
        OfflineMasterRenderer(config).render(
            [],
            source_frame_count=0,
            processor=lambda frame, context: frame,
            encoder=_MemorySink(),
        )


def test_ffmpeg_commands_use_raw_master_frames_and_high_quality_defaults(tmp_path):
    h264 = build_ffmpeg_master_command(
        FFmpegMasterConfig(
            output_path=tmp_path / "master.mp4",
            width=1920,
            height=1080,
            fps="30000/1001",
        )
    )
    assert h264[0] == "ffmpeg"
    assert h264[h264.index("-f") + 1] == "rawvideo"
    assert h264[h264.index("-pixel_format") + 1] == "rgb24"
    assert h264[h264.index("-video_size") + 1] == "1920x1080"
    assert h264[h264.index("-framerate") + 1] == "30000/1001"
    assert h264[h264.index("-i") + 1] == "pipe:0"
    assert h264[h264.index("-c:v") + 1] == "libx264"
    assert h264[h264.index("-crf") + 1] == "16"
    assert "scale" not in " ".join(h264)

    h265 = build_ffmpeg_master_command(
        FFmpegMasterConfig(
            output_path=tmp_path / "master-hevc.mp4",
            width=3840,
            height=2160,
            fps=55,
            codec=MasterCodec.H265,
            crf=14,
            extra_output_args=("-movflags", "+faststart"),
        )
    )
    assert h265[h265.index("-c:v") + 1] == "libx265"
    assert h265[h265.index("-crf") + 1] == "14"
    assert h265[h265.index("-tag:v") + 1] == "hvc1"
    assert h265[-3:-1] == ("-movflags", "+faststart")


class _CaptureStdin:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.closed = False

    def write(self, value) -> int:
        data = bytes(value)
        self.payload.extend(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, command, *, returncode=0, stderr=b"") -> None:
        self.command = tuple(command)
        self.stdin = _CaptureStdin()
        self.stderr = io.BytesIO(stderr)
        self._returncode = returncode
        self.terminated = False

    def wait(self, timeout=None):
        if self._returncode == 0:
            Path(self.command[-1]).write_bytes(b"encoded")
        return self._returncode

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def test_direct_encoder_streams_exact_rgb_bytes_and_reports_readback(tmp_path):
    holder = {}

    def factory(command, **kwargs):
        holder["process"] = _FakeProcess(command, stderr=b"diagnostic\n")
        return holder["process"]

    def probe(path, **kwargs):
        assert Path(path).read_bytes() == b"encoded"
        return {
            "available": True,
            "codec": "h264",
            "frame_count": "2",
            "bit_rate": "9000000",
        }

    config = FFmpegMasterConfig(
        output_path=tmp_path / "output.mp4",
        width=3,
        height=2,
        fps=55,
        output_pixel_format="yuv444p",
    )
    encoder = FFmpegMasterEncoder(config, process_factory=factory, probe=probe)
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    encoder.write_frame(frame)
    encoder.write_frame(255 - frame)
    result = encoder.close()

    assert (
        bytes(holder["process"].stdin.payload)
        == frame.tobytes() + (255 - frame).tobytes()
    )
    assert result.frames_written == 2
    assert result.bytes_written == frame.nbytes * 2
    assert result.returncode == 0
    assert result.stderr == "diagnostic\n"
    assert result.readback["frame_count"] == "2"


def test_direct_encoder_surfaces_ffmpeg_stderr_and_status(tmp_path):
    def factory(command, **kwargs):
        return _FakeProcess(
            command, returncode=7, stderr=b"codec initialization failed"
        )

    encoder = FFmpegMasterEncoder(
        FFmpegMasterConfig(
            output_path=tmp_path / "failed.mp4",
            width=3,
            height=2,
            fps=30,
            output_pixel_format="yuv444p",
        ),
        process_factory=factory,
    )
    encoder.write_frame(np.zeros((2, 3, 3), np.uint8))
    with pytest.raises(FFmpegEncodeError) as caught:
        encoder.close()

    assert caught.value.result.returncode == 7
    assert caught.value.result.frames_written == 1
    assert "codec initialization failed" in caught.value.result.stderr


def test_direct_encoder_rejects_mismatched_ffprobe_frame_count(tmp_path):
    def factory(command, **kwargs):
        return _FakeProcess(command)

    encoder = FFmpegMasterEncoder(
        FFmpegMasterConfig(
            output_path=tmp_path / "mismatch.mp4",
            width=3,
            height=2,
            fps=30,
            output_pixel_format="yuv444p",
        ),
        process_factory=factory,
        probe=lambda path, **kwargs: {
            "available": True,
            "frame_count": "9",
            "width": 3,
            "height": 2,
        },
    )
    encoder.write_frame(np.zeros((2, 3, 3), np.uint8))

    with pytest.raises(FFmpegEncodeError, match="readback does not match") as caught:
        encoder.close()

    assert caught.value.result.frames_written == 1
    assert caught.value.result.readback["frame_count"] == "9"


def test_ffmpeg_config_rejects_odd_yuv420_master_without_hidden_scaling(
    tmp_path,
):
    with pytest.raises(ValueError, match="requires even master dimensions"):
        FFmpegMasterConfig(
            output_path=tmp_path / "odd.mp4",
            width=17,
            height=16,
            fps=30,
        )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg/FFprobe are not installed",
)
def test_direct_encoder_real_ffmpeg_smoke(tmp_path):
    output = tmp_path / "real-smoke.mp4"
    encoder = FFmpegMasterEncoder(
        FFmpegMasterConfig(output_path=output, width=4, height=4, fps=55)
    )
    for index in range(3):
        encoder.write_frame(np.full((4, 4, 3), 50 + index * 40, np.uint8))
    result = encoder.close()

    assert output.stat().st_size > 0
    assert result.frames_written == 3
    assert result.readback["available"] is True
    assert result.readback["codec"] == "h264"
    assert int(result.readback["frame_count"]) == 3
    assert int(result.readback["width"]) == 4
    assert int(result.readback["height"]) == 4
