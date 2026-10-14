from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from megaglitch.process_logging import (
    create_process_log_handler,
    process_log_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_MODULE = REPO_ROOT / "megaglitchfm_app_new.py"


def test_source_mode_preserves_repository_relative_log_name() -> None:
    assert process_log_candidates(frozen=False) == (Path("glitch_app.log"),)


def test_application_uses_process_isolated_handler() -> None:
    source = APP_MODULE.read_text(encoding="utf-8")
    assert "from megaglitch.process_logging import create_process_log_handler" in source
    assert "_handler = create_process_log_handler(" in source
    assert "_handler = SafeRotatingFileHandler(" not in source


def test_frozen_logs_are_outside_cwd_and_unique_per_process(tmp_path: Path) -> None:
    local_app_data = tmp_path / "local"
    bundle = tmp_path / "bundle"
    first = process_log_candidates(
        frozen=True,
        environ={"LOCALAPPDATA": str(local_app_data)},
        process_id=101,
        temp_directory=tmp_path / "temp",
        bundle_directory=bundle,
    )
    second = process_log_candidates(
        frozen=True,
        environ={"LOCALAPPDATA": str(local_app_data)},
        process_id=202,
        temp_directory=tmp_path / "temp",
        bundle_directory=bundle,
    )

    assert first[0] == (
        local_app_data / "MegaGlitchNode" / "logs" / "glitch_app-101.log"
    )
    assert second[0].name == "glitch_app-202.log"
    assert first[0] != second[0]
    assert all(path.is_absolute() for path in first)
    assert all(bundle not in path.parents for path in first)


def test_frozen_log_uses_temp_when_local_app_data_is_unavailable(
    tmp_path: Path,
) -> None:
    candidates = process_log_candidates(
        frozen=True,
        environ={},
        process_id=303,
        temp_directory=tmp_path,
        bundle_directory=tmp_path / "bundle",
    )

    assert candidates == (
        tmp_path / "MegaGlitchNode" / "logs" / "glitch_app-303.log",
    )


def test_frozen_log_rejects_candidates_inside_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    candidates = process_log_candidates(
        frozen=True,
        environ={"LOCALAPPDATA": str(bundle)},
        process_id=304,
        temp_directory=bundle,
        bundle_directory=bundle,
    )

    assert candidates == ()


def test_handler_falls_back_after_unwritable_candidate(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    fallback = tmp_path / "fallback" / "glitch_app-404.log"

    handler = create_process_log_handler(
        max_bytes=1024,
        backup_count=1,
        candidates=(blocked_parent / "glitch_app-404.log", fallback),
    )
    try:
        assert isinstance(handler, RotatingFileHandler)
        assert Path(handler.baseFilename) == fallback
    finally:
        handler.close()


def test_handler_degrades_to_null_handler_when_all_candidates_fail(
    tmp_path: Path,
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    handler = create_process_log_handler(
        max_bytes=1024,
        backup_count=1,
        candidates=(blocked_parent / "glitch_app-505.log",),
    )

    assert isinstance(handler, logging.NullHandler)
