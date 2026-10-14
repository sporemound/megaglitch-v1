"""Resolve packaged media command-line tools without weakening GUI overrides.

The source application deliberately keeps ``ffmpeg`` and ``ffprobe`` as
user-editable values.  In a PyInstaller onedir build those sentinel names map
to the audited executables stored under ``_internal/bin/ffmpeg``.  Explicit
absolute or relative paths supplied by the user are returned unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


_BUNDLED_DIRECTORY = Path("bin") / "ffmpeg"
_SENTINELS = {
    "ffmpeg": {"ffmpeg", "ffmpeg.exe"},
    "ffprobe": {"ffprobe", "ffprobe.exe"},
}


def frozen_bundle_root() -> Path | None:
    """Return PyInstaller's immutable bundle root, if this is a frozen app."""

    if not bool(getattr(sys, "frozen", False)):
        return None
    root = getattr(sys, "_MEIPASS", None)
    if not root:
        return None
    return Path(root)


def resolve_packaged_media_tool(
    configured: str | os.PathLike[str],
    tool: str,
    *,
    bundle_root: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve a default FFmpeg-family name to its packaged absolute path.

    ``bundle_root`` is an explicit test/release-verifier seam.  Normal source
    runs leave sentinel names alone so PATH behavior and existing tests remain
    compatible.  A user-entered path is never silently replaced.
    """

    tool_name = str(tool).strip().lower()
    if tool_name not in _SENTINELS:
        raise ValueError(f"unsupported packaged media tool: {tool!r}")

    requested = os.fspath(configured).strip()
    if not requested:
        raise ValueError(f"{tool_name} executable cannot be empty")
    if requested.lower() not in _SENTINELS[tool_name]:
        return requested

    root = Path(bundle_root) if bundle_root is not None else frozen_bundle_root()
    if root is None:
        return requested
    executable = (root / _BUNDLED_DIRECTORY / f"{tool_name}.exe").resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"frozen MegaGlitch bundle is missing {tool_name}: {executable}"
        )
    return os.fspath(executable)


def resolve_ffmpeg_executable(
    configured: str | os.PathLike[str] = "ffmpeg",
    *,
    bundle_root: str | os.PathLike[str] | None = None,
) -> str:
    return resolve_packaged_media_tool(
        configured, "ffmpeg", bundle_root=bundle_root
    )


def resolve_ffprobe_executable(
    configured: str | os.PathLike[str] = "ffprobe",
    *,
    bundle_root: str | os.PathLike[str] | None = None,
) -> str:
    return resolve_packaged_media_tool(
        configured, "ffprobe", bundle_root=bundle_root
    )
