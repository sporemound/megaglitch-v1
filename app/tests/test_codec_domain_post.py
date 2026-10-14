from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from megaglitch.codec_domain_post import (
    CodecDomainPostConfig,
    CodecDomainStageError,
    GlitchDomain,
    build_codec_domain_commands,
    parse_corruption_command_template,
    run_codec_domain_post_network,
)
from megaglitch.offline_master import MasterCodec


def _config(tmp_path: Path, **overrides) -> CodecDomainPostConfig:
    values = {
        "deterministic_input": tmp_path / "deterministic-master.mp4",
        "final_output": tmp_path / "final" / "codec-master.mp4",
        "work_directory": tmp_path / "codec-work",
        "corruption_command": (
            "pframe-mosh",
            "--input",
            "{prepared}",
            "--output",
            "{corrupted}",
            "--seed",
            "{seed}",
        ),
        "corruption_seed": 77,
    }
    values.update(overrides)
    return CodecDomainPostConfig(**values)


def test_codec_domain_plan_keeps_external_corruption_between_codec_boundaries(tmp_path):
    commands = build_codec_domain_commands(_config(tmp_path))

    assert commands.preparation[commands.preparation.index("-c:v") + 1] == "mpeg4"
    assert commands.preparation[commands.preparation.index("-g") + 1] == "120"
    assert commands.corruption[0] == "pframe-mosh"
    assert str(commands.prepared_path) in commands.corruption
    assert str(commands.corrupted_path) in commands.corruption
    assert commands.corruption[commands.corruption.index("--seed") + 1] == "77"
    assert (
        commands.tolerant_decode[commands.tolerant_decode.index("-err_detect") + 1]
        == "ignore_err"
    )
    assert (
        commands.tolerant_decode[commands.tolerant_decode.index("-c:v") + 1] == "ffv1"
    )
    assert commands.final_encode[commands.final_encode.index("-c:v") + 1] == "libx264"
    assert commands.final_encode[commands.final_encode.index("-crf") + 1] == "16"
    assert not any(
        "filter" in token for command in commands.ordered for token in command[1]
    )


def test_codec_domain_plan_requires_a_real_external_transform(tmp_path):
    with pytest.raises(ValueError, match="explicitly map"):
        _config(tmp_path, corruption_command=("ffmpeg", "-i", "{prepared}"))


def test_gui_command_template_preserves_quoted_windows_paths_and_placeholders():
    parsed = parse_corruption_command_template(
        '"C:\\Program Files\\PFrame Tool\\mosh.exe" --input "{prepared}" '
        '--output "{corrupted}" --seed {seed}',
        windows=True,
    )
    assert parsed == (
        "C:\\Program Files\\PFrame Tool\\mosh.exe",
        "--input",
        "{prepared}",
        "--output",
        "{corrupted}",
        "--seed",
        "{seed}",
    )


def test_codec_domain_runner_is_ordered_and_reports_domain_separation(tmp_path):
    calls = []
    config = _config(tmp_path)
    expected_outputs = (
        config.work_directory / "megaglitch-long-gop-prepared.avi",
        config.work_directory / "megaglitch-codec-corrupted.avi",
        config.work_directory / "megaglitch-error-tolerant-recovery.mkv",
        config.final_output,
    )

    def runner(command, **kwargs):
        calls.append(tuple(command))
        output = Path(expected_outputs[len(calls) - 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"stage output")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    result = run_codec_domain_post_network(config, run=runner)

    assert len(calls) == 4
    assert [stage.stage for stage in result.stages] == [
        "mpeg4_long_gop_preparation",
        "external_codec_corruption",
        "error_tolerant_decode",
        "final_master_encode",
    ]
    assert result.domains == (
        GlitchDomain.IMAGE.value,
        GlitchDomain.TEMPORAL_NETWORK.value,
        GlitchDomain.CODEC.value,
    )


def test_codec_domain_runner_stops_at_first_failed_stage(tmp_path):
    calls = []
    config = _config(tmp_path)

    def runner(command, **kwargs):
        calls.append(tuple(command))
        status = 9 if len(calls) == 2 else 0
        if len(calls) == 1:
            prepared = config.work_directory / "megaglitch-long-gop-prepared.avi"
            prepared.write_bytes(b"prepared")
        return subprocess.CompletedProcess(command, status, b"", b"bad packet map")

    with pytest.raises(CodecDomainStageError) as caught:
        run_codec_domain_post_network(config, run=runner)

    assert len(calls) == 2
    assert caught.value.result.stage == "external_codec_corruption"
    assert caught.value.result.returncode == 9
    assert "bad packet map" in caught.value.result.stderr


def test_codec_domain_runner_rejects_success_without_stage_output(tmp_path):
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, b"", b"")

    with pytest.raises(CodecDomainStageError) as caught:
        run_codec_domain_post_network(_config(tmp_path), run=runner)

    assert caught.value.result.stage == "mpeg4_long_gop_preparation"
    assert caught.value.result.returncode == -2
    assert "no non-empty output" in caught.value.result.stderr


def test_h265_final_codec_uses_quality_override_without_changing_corruption(tmp_path):
    commands = build_codec_domain_commands(
        _config(tmp_path, final_codec=MasterCodec.H265, final_crf=18)
    )

    assert commands.final_encode[commands.final_encode.index("-c:v") + 1] == "libx265"
    assert commands.final_encode[commands.final_encode.index("-crf") + 1] == "18"
    assert commands.final_encode[commands.final_encode.index("-tag:v") + 1] == "hvc1"
    assert commands.corruption[0] == "pframe-mosh"
