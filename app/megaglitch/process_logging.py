"""Process-isolated logging paths for source and frozen MegaGlitch hosts."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping


class SafeRotatingFileHandler(RotatingFileHandler):
    """Keep logging alive when Windows cannot rotate an open process log."""

    def doRollover(self) -> None:  # noqa: N802 - logging API name
        try:
            super().doRollover()
        except (OSError, PermissionError):
            # Logging must never stop rendering or worker shutdown.
            return


def process_log_candidates(
    *,
    frozen: bool | None = None,
    environ: Mapping[str, str] | None = None,
    process_id: int | None = None,
    temp_directory: str | os.PathLike[str] | None = None,
    bundle_directory: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Return ordered log destinations for this process.

    Source launches preserve the established repository-relative
    ``glitch_app.log``. Frozen launches never use the current working directory:
    each process receives a PID-qualified log under LocalAppData, followed by a
    temp-directory fallback. This also prevents a UI host and its spawned frame
    worker from sharing a Windows file handle.
    """

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if not is_frozen:
        return (Path("glitch_app.log"),)

    environment = os.environ if environ is None else environ
    pid = os.getpid() if process_id is None else int(process_id)
    filename = f"glitch_app-{pid}.log"
    temp_root = Path(
        tempfile.gettempdir() if temp_directory is None else temp_directory
    )
    bundle_root = Path(
        Path(sys.executable).resolve().parent
        if bundle_directory is None
        else bundle_directory
    )
    bundle_identity = os.path.normcase(os.path.abspath(bundle_root))

    roots: list[Path] = []
    local_app_data = environment.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        roots.append(Path(local_app_data))
    roots.append(temp_root)

    candidates: list[Path] = []
    identities: set[str] = set()
    for root in roots:
        candidate = Path(
            os.path.abspath(root / "MegaGlitchNode" / "logs" / filename)
        )
        identity = os.path.normcase(os.path.abspath(candidate))
        try:
            common = os.path.commonpath((bundle_identity, identity))
        except ValueError:
            common = ""
        if os.path.normcase(common) == bundle_identity:
            # Reject an invalid/hostile environment that points a writable log
            # root into the frozen onedir. The next candidate may still work;
            # otherwise create_process_log_handler degrades to NullHandler.
            continue
        if identity not in identities:
            identities.add(identity)
            candidates.append(candidate)
    return tuple(candidates)


def create_process_log_handler(
    *,
    max_bytes: int,
    backup_count: int,
    candidates: Iterable[Path] | None = None,
) -> logging.Handler:
    """Create a rotating handler, degrading safely if no path is writable."""

    destinations = (
        process_log_candidates() if candidates is None else tuple(candidates)
    )
    for destination in destinations:
        path = Path(destination)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return SafeRotatingFileHandler(
                path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                errors="backslashreplace",
            )
        except (OSError, PermissionError):
            continue

    return logging.NullHandler()
