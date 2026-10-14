from fractions import Fraction

import pytest

from megaglitch.offline_source import (
    OfflineSourceKind,
    OfflineSourceRange,
    plan_gif_source,
    plan_timed_source,
    plan_video_source,
    sample_cfr_index_at_time,
    sample_gif_index_at_time,
)


def test_video_plan_enumerates_every_source_frame_once() -> None:
    plan = plan_video_source(frame_count=90, fps=30)
    assert plan.source_indices == tuple(range(90))
    assert len(set(plan.source_indices)) == 90
    assert plan.duration_seconds == 3


def test_video_plan_from_playhead_stops_at_eof_without_looping() -> None:
    plan = plan_video_source(
        frame_count=10,
        fps=25,
        playhead_index=7,
        source_range=OfflineSourceRange.FROM_PLAYHEAD,
    )
    assert plan.source_indices == (7, 8, 9)
    assert plan.timestamps_seconds == (0, Fraction(1, 25), Fraction(2, 25))


def test_variable_duration_gif_retains_each_frame_and_total_duration() -> None:
    plan = plan_gif_source([100, 250, 50])
    assert plan.source_indices == (0, 1, 2)
    assert plan.timestamps_seconds == (0, Fraction(1, 10), Fraction(7, 20))
    assert plan.duration_seconds == Fraction(2, 5)
    assert plan.source_rate == Fraction(15, 2)


def test_timed_source_is_explicitly_duration_driven() -> None:
    plan = plan_timed_source(duration_seconds=2, fps=24, kind=OfflineSourceKind.STILL)
    assert plan.frame_count == 48
    assert plan.source_indices[0] == 0
    assert plan.source_indices[-1] == 47


def test_secondary_timestamp_sampling_clamps_without_looping() -> None:
    assert sample_cfr_index_at_time(0.9, fps=10, frame_count=100) == 9
    assert sample_cfr_index_at_time(99, fps=10, frame_count=100) == 99
    assert sample_gif_index_at_time(0.0, [100, 200, 300]) == 0
    assert sample_gif_index_at_time(0.1, [100, 200, 300]) == 1
    assert sample_gif_index_at_time(99, [100, 200, 300]) == 2


@pytest.mark.parametrize("count", [0, -1])
def test_video_plan_rejects_unknown_or_empty_source(count: int) -> None:
    with pytest.raises(ValueError):
        plan_video_source(frame_count=count, fps=30)
