import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from megaglitch.codec_domain_post import (
    CodecDomainPostConfig,
    build_codec_domain_commands,
)
from megaglitch.external_tools import (
    resolve_ffmpeg_executable,
    resolve_ffprobe_executable,
    resolve_packaged_media_tool,
)
from megaglitch.offline_audio import decode_source_audio
from megaglitch.offline_master import (
    FFmpegMasterConfig,
    build_ffmpeg_master_command,
    probe_encoded_media,
)


def _make_bundle_tools(root: Path) -> tuple[Path, Path]:
    directory = root / "bin" / "ffmpeg"
    directory.mkdir(parents=True)
    ffmpeg = directory / "ffmpeg.exe"
    ffprobe = directory / "ffprobe.exe"
    ffmpeg.write_bytes(b"test ffmpeg")
    ffprobe.write_bytes(b"test ffprobe")
    return ffmpeg.resolve(), ffprobe.resolve()


def _freeze(monkeypatch, root: Path) -> None:
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", os.fspath(root), raising=False)


def test_source_sentinels_and_explicit_overrides_are_preserved(tmp_path):
    assert resolve_ffmpeg_executable("ffmpeg") == "ffmpeg"
    assert resolve_ffprobe_executable("ffprobe.exe") == "ffprobe.exe"
    override = tmp_path / "custom" / "ffmpeg.exe"
    assert resolve_ffmpeg_executable(override, bundle_root=tmp_path) == os.fspath(
        override
    )


def test_bundle_sentinels_resolve_to_absolute_verified_tools(tmp_path):
    ffmpeg, ffprobe = _make_bundle_tools(tmp_path)
    assert resolve_ffmpeg_executable("ffmpeg", bundle_root=tmp_path) == os.fspath(
        ffmpeg
    )
    assert resolve_ffprobe_executable(
        "ffprobe.exe", bundle_root=tmp_path
    ) == os.fspath(ffprobe)


def test_missing_bundle_tool_fails_instead_of_borrowing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing ffmpeg"):
        resolve_ffmpeg_executable("ffmpeg", bundle_root=tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_packaged_media_tool("thing", "not-a-tool", bundle_root=tmp_path)


def test_offline_master_and_probe_use_frozen_absolute_tools(tmp_path, monkeypatch):
    ffmpeg, ffprobe = _make_bundle_tools(tmp_path)
    _freeze(monkeypatch, tmp_path)
    command = build_ffmpeg_master_command(
        FFmpegMasterConfig(
            output_path=tmp_path / "master.mp4", width=4, height=4, fps=55
        )
    )
    assert command[0] == os.fspath(ffmpeg)

    def fake_probe(command, **_kwargs):
        assert command[0] == os.fspath(ffprobe)
        payload = {"streams": [], "format": {}}
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        )

    result = probe_encoded_media(tmp_path / "master.mp4", run=fake_probe)
    assert result["available"] is True


def test_audio_and_codec_network_use_frozen_absolute_ffmpeg(tmp_path, monkeypatch):
    ffmpeg, _ffprobe = _make_bundle_tools(tmp_path)
    _freeze(monkeypatch, tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    def fake_decode(command, **_kwargs):
        assert command[0] == os.fspath(ffmpeg)
        return SimpleNamespace(
            returncode=0,
            stdout=np.zeros(8, dtype="<f4").tobytes(),
            stderr=b"",
        )

    analysis = decode_source_audio(source, run=fake_decode, sample_rate=8000)
    assert analysis.samples.size == 8

    commands = build_codec_domain_commands(
        CodecDomainPostConfig(
            deterministic_input=source,
            final_output=tmp_path / "final.mp4",
            work_directory=tmp_path / "codec-work",
            corruption_command=(
                "corruptor.exe",
                "{prepared}",
                "{corrupted}",
                "{seed}",
            ),
        )
    )
    assert commands.preparation[0] == os.fspath(ffmpeg)
    assert commands.tolerant_decode[0] == os.fspath(ffmpeg)
    assert commands.final_encode[0] == os.fspath(ffmpeg)
