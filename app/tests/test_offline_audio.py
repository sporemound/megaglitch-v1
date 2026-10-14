from __future__ import annotations

import subprocess

import numpy as np
import pytest

from megaglitch.offline_audio import (
    OfflineAudioAnalysis,
    OfflineAudioError,
    decode_source_audio,
)


def test_source_audio_features_are_deterministic_and_band_selective() -> None:
    rate = 48000
    time = np.arange(rate, dtype=np.float32) / rate
    samples = (0.7 * np.sin(2.0 * np.pi * 110.0 * time)).astype(np.float32)
    analysis = OfflineAudioAnalysis(samples, rate, "tone.wav")
    first = analysis.features_at(0.25, frame_duration_seconds=1 / 30)
    second = analysis.features_at(0.25, frame_duration_seconds=1 / 30)
    assert first == second
    assert first.fft_bands["Bass"] > first.fft_bands["Presence"]
    assert analysis.document()["pcm_sha256"]


def test_decode_source_audio_builds_direct_pcm_command(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    samples = np.array([0.0, 0.25, -0.5], dtype="<f4")

    def run(command, **kwargs):
        assert "-vn" in command
        assert command[-2:] == ("f32le", "pipe:1")
        return subprocess.CompletedProcess(command, 0, samples.tobytes(), b"")

    analysis = decode_source_audio(source, run=run)
    assert np.allclose(analysis.samples, samples)


def test_decode_source_audio_surfaces_ffmpeg_error(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 4, b"", b"no audio stream")

    with pytest.raises(OfflineAudioError, match="no audio stream"):
        decode_source_audio(source, run=run)
