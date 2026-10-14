"""Optional traditional codec-domain corruption after deterministic rendering.

This is intentionally separate from MegaGlitch's image-domain and temporal
network.  FFmpeg prepares and recovers codecs, but the corruption command is a
required external CLI supplied by the caller.  No generic FFmpeg filter is
presented as a replacement for packet/P-frame/datamosh logic.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .offline_master import MasterCodec
from .external_tools import resolve_ffmpeg_executable


class GlitchDomain(str, Enum):
    IMAGE = "image_domain_glitch"
    TEMPORAL_NETWORK = "temporal_network_glitch"
    CODEC = "codec_domain_glitch"


def parse_corruption_command_template(
    command_text: str, *, windows: bool | None = None
) -> tuple[str, ...]:
    """Parse the GUI command field into subprocess-safe arguments.

    Windows paths retain backslashes and quoted spaces. Placeholders remain
    literal tokens until :func:`build_codec_domain_commands` expands them.
    """
    text = str(command_text or "").strip()
    if not text:
        raise ValueError("codec corruption command cannot be empty")
    use_windows_rules = os.name == "nt" if windows is None else bool(windows)
    if not use_windows_rules:
        values = shlex.split(text, posix=True)
    else:
        lexer = shlex.shlex(text, posix=False)
        lexer.whitespace_split = True
        lexer.commenters = ""
        values = list(lexer)
        values = [
            value[1:-1]
            if len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
            else value
            for value in values
        ]
    if not values:
        raise ValueError("codec corruption command cannot be empty")
    return tuple(values)


@dataclass(frozen=True)
class CodecDomainPostConfig:
    deterministic_input: str | os.PathLike[str]
    final_output: str | os.PathLike[str]
    work_directory: str | os.PathLike[str]
    corruption_command: tuple[str, ...]
    ffmpeg_executable: str = "ffmpeg"
    long_gop_size: int = 120
    preparation_quality: int = 2
    corruption_seed: int = 0
    final_codec: MasterCodec = MasterCodec.H264
    final_crf: int = 16
    final_preset: str = "slow"

    def __post_init__(self) -> None:
        if int(self.long_gop_size) < 2:
            raise ValueError("long_gop_size must be at least 2")
        if not 1 <= int(self.preparation_quality) <= 31:
            raise ValueError("preparation_quality must be between 1 and 31")
        if not 0 <= int(self.final_crf) <= 51:
            raise ValueError("final_crf must be between 0 and 51")
        MasterCodec(self.final_codec)
        command_text = "\0".join(str(value) for value in self.corruption_command)
        if "{prepared}" not in command_text or "{corrupted}" not in command_text:
            raise ValueError(
                "corruption_command must explicitly map {prepared} to {corrupted}"
            )


@dataclass(frozen=True)
class CodecDomainCommands:
    preparation: tuple[str, ...]
    corruption: tuple[str, ...]
    tolerant_decode: tuple[str, ...]
    final_encode: tuple[str, ...]
    prepared_path: Path
    corrupted_path: Path
    recovered_path: Path

    @property
    def ordered(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (
            ("mpeg4_long_gop_preparation", self.preparation),
            ("external_codec_corruption", self.corruption),
            ("error_tolerant_decode", self.tolerant_decode),
            ("final_master_encode", self.final_encode),
        )


def build_codec_domain_commands(config: CodecDomainPostConfig) -> CodecDomainCommands:
    work = Path(config.work_directory)
    prepared = work / "megaglitch-long-gop-prepared.avi"
    corrupted = work / "megaglitch-codec-corrupted.avi"
    recovered = work / "megaglitch-error-tolerant-recovery.mkv"
    source = os.fspath(config.deterministic_input)
    target = os.fspath(config.final_output)
    ffmpeg = resolve_ffmpeg_executable(config.ffmpeg_executable)

    preparation = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        str(int(config.preparation_quality)),
        "-g",
        str(int(config.long_gop_size)),
        "-bf",
        "2",
        "-pix_fmt",
        "yuv420p",
        os.fspath(prepared),
    )
    replacements = {
        "prepared": os.fspath(prepared),
        "corrupted": os.fspath(corrupted),
        "seed": str(int(config.corruption_seed)),
    }
    corruption = tuple(
        str(value)
        .replace("{prepared}", replacements["prepared"])
        .replace("{corrupted}", replacements["corrupted"])
        .replace("{seed}", replacements["seed"])
        for value in config.corruption_command
    )
    tolerant_decode = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-err_detect",
        "ignore_err",
        "-fflags",
        "+discardcorrupt+genpts",
        "-i",
        os.fspath(corrupted),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        os.fspath(recovered),
    )
    final_video_codec = (
        "libx264" if MasterCodec(config.final_codec) is MasterCodec.H264 else "libx265"
    )
    final_encode_parts = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        os.fspath(recovered),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        final_video_codec,
        "-preset",
        str(config.final_preset),
        "-crf",
        str(int(config.final_crf)),
        "-pix_fmt",
        "yuv420p",
    ]
    if MasterCodec(config.final_codec) is MasterCodec.H265:
        final_encode_parts.extend(("-tag:v", "hvc1"))
    final_encode_parts.append(target)

    return CodecDomainCommands(
        preparation=preparation,
        corruption=corruption,
        tolerant_decode=tolerant_decode,
        final_encode=tuple(final_encode_parts),
        prepared_path=prepared,
        corrupted_path=corrupted,
        recovered_path=recovered,
    )


@dataclass(frozen=True)
class CodecStageResult:
    stage: str
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    stderr: str


@dataclass(frozen=True)
class CodecDomainPostResult:
    domains: tuple[str, ...]
    stages: tuple[CodecStageResult, ...]
    final_output: str


class CodecDomainStageError(RuntimeError):
    def __init__(self, result: CodecStageResult) -> None:
        super().__init__(
            f"codec-domain stage {result.stage} failed with status "
            f"{result.returncode}: {result.stderr[-4096:]}"
        )
        self.result = result


def run_codec_domain_post_network(
    config: CodecDomainPostConfig,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> CodecDomainPostResult:
    """Run the explicitly requested post-network stages, stopping on first error."""

    Path(config.work_directory).mkdir(parents=True, exist_ok=True)
    Path(config.final_output).parent.mkdir(parents=True, exist_ok=True)
    commands = build_codec_domain_commands(config)
    results: list[CodecStageResult] = []
    expected_outputs = (
        commands.prepared_path,
        commands.corrupted_path,
        commands.recovered_path,
        Path(config.final_output),
    )
    for (stage, command), expected_output in zip(
        commands.ordered, expected_outputs, strict=True
    ):
        started = time.perf_counter()
        try:
            completed = run(command, capture_output=True, check=False)
        except OSError as error:
            result = CodecStageResult(
                stage=stage,
                command=command,
                returncode=-1,
                elapsed_seconds=time.perf_counter() - started,
                stderr=str(error),
            )
            raise CodecDomainStageError(result) from error
        stderr = (
            completed.stderr.decode("utf-8", "replace")
            if isinstance(completed.stderr, bytes)
            else str(completed.stderr or "")
        )
        result = CodecStageResult(
            stage=stage,
            command=command,
            returncode=int(completed.returncode),
            elapsed_seconds=time.perf_counter() - started,
            stderr=stderr,
        )
        results.append(result)
        if result.returncode != 0:
            raise CodecDomainStageError(result)
        if not expected_output.is_file() or expected_output.stat().st_size <= 0:
            missing = CodecStageResult(
                stage=stage,
                command=command,
                returncode=-2,
                elapsed_seconds=result.elapsed_seconds,
                stderr=f"stage produced no non-empty output: {expected_output}",
            )
            results[-1] = missing
            raise CodecDomainStageError(missing)

    return CodecDomainPostResult(
        domains=(
            GlitchDomain.IMAGE.value,
            GlitchDomain.TEMPORAL_NETWORK.value,
            GlitchDomain.CODEC.value,
        ),
        stages=tuple(results),
        final_output=os.fspath(config.final_output),
    )
