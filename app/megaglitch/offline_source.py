"""Deterministic source enumeration and timestamp sampling for Offline Master."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Iterable, Sequence


class OfflineSourceKind(str, Enum):
    VIDEO = "video"
    GIF = "gif"
    STILL = "still"
    GENERATOR = "generator"


class OfflineSourceRange(str, Enum):
    ENTIRE_SOURCE = "entire_source"
    FROM_PLAYHEAD = "from_playhead"


def _positive_fraction(value: int | float | str | Fraction, name: str) -> Fraction:
    rate = value if isinstance(value, Fraction) else Fraction(str(value))
    if rate <= 0:
        raise ValueError(f"{name} must be positive")
    return rate


@dataclass(frozen=True)
class OfflineSourcePlan:
    kind: OfflineSourceKind
    source_rate: Fraction
    source_indices: tuple[int, ...]
    timestamps_seconds: tuple[Fraction, ...]
    duration_seconds: Fraction
    start_index: int = 0

    def __post_init__(self) -> None:
        if not self.source_indices:
            raise ValueError("offline source plan requires at least one frame")
        if len(self.source_indices) != len(self.timestamps_seconds):
            raise ValueError("source index/timestamp lengths differ")
        if self.duration_seconds <= 0:
            raise ValueError("offline source duration must be positive")
        if any(b < a for a, b in zip(self.timestamps_seconds, self.timestamps_seconds[1:])):
            raise ValueError("offline source timestamps must be monotonic")

    @property
    def frame_count(self) -> int:
        return len(self.source_indices)


def plan_video_source(
    *,
    frame_count: int,
    fps: int | float | str | Fraction,
    playhead_index: int = 0,
    source_range: OfflineSourceRange | str = OfflineSourceRange.ENTIRE_SOURCE,
) -> OfflineSourcePlan:
    """Enumerate each selected video frame exactly once, with no EOF looping."""
    count = int(frame_count)
    if count <= 0:
        raise ValueError("video frame_count must be positive")
    rate = _positive_fraction(fps, "video fps")
    mode = OfflineSourceRange(source_range)
    start = 0 if mode is OfflineSourceRange.ENTIRE_SOURCE else max(0, int(playhead_index))
    start = min(start, count - 1)
    indices = tuple(range(start, count))
    timestamps = tuple(Fraction(index - start, 1) / rate for index in indices)
    return OfflineSourcePlan(
        kind=OfflineSourceKind.VIDEO,
        source_rate=rate,
        source_indices=indices,
        timestamps_seconds=timestamps,
        duration_seconds=Fraction(len(indices), 1) / rate,
        start_index=start,
    )


def plan_gif_source(
    durations_ms: Sequence[int | float],
    *,
    playhead_index: int = 0,
    source_range: OfflineSourceRange | str = OfflineSourceRange.ENTIRE_SOURCE,
) -> OfflineSourcePlan:
    """Enumerate selected GIF frames once while retaining variable timestamps."""
    durations = tuple(Fraction(str(value)) / 1000 for value in durations_ms)
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("GIF frame durations must all be positive")
    mode = OfflineSourceRange(source_range)
    start = 0 if mode is OfflineSourceRange.ENTIRE_SOURCE else max(0, int(playhead_index))
    start = min(start, len(durations) - 1)
    selected = durations[start:]
    timestamps = []
    elapsed = Fraction(0, 1)
    for duration in selected:
        timestamps.append(elapsed)
        elapsed += duration
    # A CFR encoder timebase can preserve the exact aggregate GIF duration by
    # using frame_count / total_duration as the declared source rate. Per-frame
    # control evaluation still uses the original variable timestamps above.
    rate = Fraction(len(selected), 1) / elapsed
    return OfflineSourcePlan(
        kind=OfflineSourceKind.GIF,
        source_rate=rate,
        source_indices=tuple(range(start, len(durations))),
        timestamps_seconds=tuple(timestamps),
        duration_seconds=elapsed,
        start_index=start,
    )


def plan_timed_source(
    *,
    duration_seconds: int | float | str | Fraction,
    fps: int | float | str | Fraction,
    kind: OfflineSourceKind | str,
) -> OfflineSourcePlan:
    """Plan a still/generator timeline; duration is explicit by design."""
    duration = _positive_fraction(duration_seconds, "duration_seconds")
    rate = _positive_fraction(fps, "source fps")
    source_kind = OfflineSourceKind(kind)
    if source_kind not in {OfflineSourceKind.STILL, OfflineSourceKind.GENERATOR}:
        raise ValueError("timed source kind must be still or generator")
    count = max(1, int(duration * rate + Fraction(1, 2)))
    indices = tuple(range(count))
    timestamps = tuple(Fraction(index, 1) / rate for index in indices)
    return OfflineSourcePlan(
        kind=source_kind,
        source_rate=rate,
        source_indices=indices,
        timestamps_seconds=timestamps,
        duration_seconds=Fraction(count, 1) / rate,
    )


def sample_cfr_index_at_time(
    timestamp_seconds: int | float | str | Fraction,
    *,
    fps: int | float | str | Fraction,
    frame_count: int,
) -> int:
    """Sample a secondary CFR source by timestamp, clamped without looping."""
    count = int(frame_count)
    if count <= 0:
        raise ValueError("frame_count must be positive")
    timestamp = max(Fraction(0, 1), Fraction(str(timestamp_seconds)))
    rate = _positive_fraction(fps, "fps")
    return min(count - 1, int(timestamp * rate))


def sample_gif_index_at_time(
    timestamp_seconds: int | float | str | Fraction,
    durations_ms: Iterable[int | float],
) -> int:
    """Sample a secondary variable-duration GIF by timestamp, clamped at EOF."""
    durations = tuple(Fraction(str(value)) / 1000 for value in durations_ms)
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("GIF frame durations must all be positive")
    boundaries = []
    elapsed = Fraction(0, 1)
    for duration in durations:
        elapsed += duration
        boundaries.append(elapsed)
    timestamp = max(Fraction(0, 1), Fraction(str(timestamp_seconds)))
    return min(len(durations) - 1, bisect_right(boundaries, timestamp))
