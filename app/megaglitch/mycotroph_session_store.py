"""Transactional, UI-independent persistence for canonical Mycotroph sessions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .session_document import normalize_session_document


class SessionStoreError(RuntimeError):
    """A stable persistence error suitable for the Engine V2 protocol."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.path = path
        self.recoverable = bool(recoverable)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "recoverable": self.recoverable,
        }
        if self.path:
            result["path"] = self.path
        return result


@dataclass(frozen=True)
class SessionStoreSnapshot:
    generation: int
    dirty: bool
    path: str | None
    sha256: str


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _normalize(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return normalize_session_document(document)
    except (TypeError, ValueError, OverflowError) as error:
        raise SessionStoreError("invalid_session", str(error)) from error


class SessionStore:
    """Own one canonical document with prepare-before-commit file operations."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self._lock = threading.RLock()
        self._document = _normalize(document)
        self._generation = 0
        self._dirty = False
        self._path: Path | None = None

    def document(self) -> dict[str, Any]:
        with self._lock:
            return _clone_json(self._document)

    def snapshot(self) -> SessionStoreSnapshot:
        with self._lock:
            payload = _canonical_bytes(self._document)
            return SessionStoreSnapshot(
                generation=self._generation,
                dirty=self._dirty,
                path=str(self._path) if self._path is not None else None,
                sha256=hashlib.sha256(payload).hexdigest(),
            )

    def replace(self, document: Mapping[str, Any]) -> SessionStoreSnapshot:
        candidate = _normalize(document)
        with self._lock:
            self._document = candidate
            self._generation += 1
            self._dirty = True
            return self.snapshot()

    def load(self, path_value: str | os.PathLike[str]) -> SessionStoreSnapshot:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise SessionStoreError(
                "session_not_found",
                f"session file does not exist: {path}",
                path=str(path),
            )
        try:
            with path.open("r", encoding="utf-8-sig") as source:
                payload = json.load(source)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SessionStoreError(
                "session_read_failed",
                f"could not read session {path}: {error}",
                path=str(path),
            ) from error
        if not isinstance(payload, Mapping):
            raise SessionStoreError(
                "invalid_session",
                "session file must contain one JSON object",
                path=str(path),
            )
        try:
            candidate = _normalize(payload)
        except SessionStoreError as error:
            raise SessionStoreError(
                error.code,
                str(error),
                path=str(path),
                recoverable=error.recoverable,
            ) from error
        with self._lock:
            self._document = candidate
            self._generation += 1
            self._dirty = False
            self._path = path
            return self.snapshot()

    def save(self, path_value: str | os.PathLike[str] | None = None) -> SessionStoreSnapshot:
        with self._lock:
            path = Path(path_value).expanduser().resolve() if path_value is not None else self._path
            if path is None:
                raise SessionStoreError(
                    "session_path_required",
                    "save requires a destination path until the session has been saved once",
                )
            if path.exists() and not path.is_file():
                raise SessionStoreError(
                    "session_path_invalid",
                    f"session destination is not a file: {path}",
                    path=str(path),
                )
            parent = path.parent
            if not parent.is_dir():
                raise SessionStoreError(
                    "session_directory_not_found",
                    f"session destination directory does not exist: {parent}",
                    path=str(path),
                )
            document = _clone_json(self._document)
            payload = _canonical_bytes(document)

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
            with path.open("rb") as saved:
                readback = saved.read()
            if readback != payload:
                raise SessionStoreError(
                    "session_readback_mismatch",
                    f"saved session did not match the committed bytes: {path}",
                    path=str(path),
                    recoverable=False,
                )
        except SessionStoreError:
            raise
        except OSError as error:
            raise SessionStoreError(
                "session_write_failed",
                f"could not save session {path}: {error}",
                path=str(path),
            ) from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

        with self._lock:
            # A concurrent replace during I/O remains dirty; only the exact
            # document copied above may be marked as persisted.
            if self._document == document:
                self._dirty = False
                self._path = path
            return self.snapshot()


__all__ = ["SessionStore", "SessionStoreError", "SessionStoreSnapshot"]
