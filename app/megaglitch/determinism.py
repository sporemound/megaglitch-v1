"""Render-scoped deterministic entropy helpers.

Live mode keeps entropy-backed generators. Offline Master installs one seeded
generator per source frame so legacy effects that previously constructed a new
unseeded ``default_rng`` participate in the reproducible graph sequence.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import numpy as np


_active_numpy_rng: ContextVar[np.random.Generator | None] = ContextVar(
    "megaglitch_active_numpy_rng", default=None
)


@contextmanager
def use_numpy_rng(rng: np.random.Generator) -> Iterator[np.random.Generator]:
    token = _active_numpy_rng.set(rng)
    try:
        yield rng
    finally:
        _active_numpy_rng.reset(token)


def current_numpy_rng() -> np.random.Generator:
    rng = _active_numpy_rng.get()
    return rng if rng is not None else np.random.default_rng()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode(
        "utf-8", errors="surrogatepass"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
