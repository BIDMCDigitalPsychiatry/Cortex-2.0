import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.screen_bouts import (
    build_screen_bouts,
    compute_daily_screen_features,
    local_clock_intervals,
    screen_supported_duration_ms,
)
from cortex2_engine.time import local_day_bounds_ms

SECOND = 1000
MINUTE = 60_000
HOUR = 3_600_000
DAY = 24 * HOUR


def events(*pairs):
    return pd.DataFrame(
        pairs,
        columns=["timestamp_ms", "event"],
    )


def test_on_off_pairs_make_bouts_and_unlock_does_not_restart():

    screen = events(
        (1 * HOUR, "screen_on"),
        (1 * HOUR + 2 * SECOND, "unlock"),
        (1 * HOUR + 10 * MINUTE, "lock"),
        (1 * HOUR + 10 * MINUTE + 500, "screen_off"),
        (5 * HOUR, "unlock"),
        (5 * HOUR + 20 * SECOND, "screen_off"),
    )

    bouts = build_screen_bouts(screen, window_start_ms=0, window_end_ms=DAY)

    assert bouts["bout_start_ms"].tolist() == [1 * HOUR, 5 * HOUR]
    assert bouts["duration_ms"].tolist() == [10 * MINUTE, 20 * SECOND]
    assert bouts["bout_status"].tolist() == ["observed", "observed"]
    assert bouts["censor_reason"].isna().all()


def test_unknown_events_are_ignored():

    screen = events(
        (HOUR, "screen_on"),
        (HOUR + MINUTE, "unknown"),
        (HOUR + 2 * MINUTE, "screen_off"),
    )

    bouts = build_screen_bouts(screen, window_start_ms=0, window_end_ms=DAY)

    assert bouts["duration_ms"].tolist() == [2 * MINUTE]


def test_long_bout_is_kept_and_censored_not_dropped():

    screen = events(
        (HOUR, "screen_on"),
        (HOUR + 6 * HOUR, "screen_off"),
    )

    bouts = build_screen_bouts(screen, window_start_ms=0, window_end_ms=DAY)

    assert len(bouts) == 1
    assert bouts.loc[0, "bout_status"] == "censored"
    assert bouts.loc[0, "censor_reason"] == "exceeds_max_bout"
    assert bouts.loc[0, "duration_ms"] == 6 * HOUR
    assert bouts.loc[0, "counted_duration_ms"] == 4 * HOUR


def test_open_bout_at_window_end_is_censored():

    screen = events((DAY - HOUR, "unlock"))

    bouts = build_screen_bouts(screen, window_start_ms=0, window_end_ms=DAY)

    assert bouts.loc[0, "bout_end_ms"] == DAY
    assert bouts.loc[0, "censor_reason"] == "right_window"


def test_state_carried_from_before_window_is_left_censored():

    screen = events(
        (-10 * MINUTE, "screen_on"),
        (5 * MINUTE, "screen_off"),
    )

    bouts = build_screen_bouts(screen, window_start_ms=0, window_end_ms=DAY)

    assert bouts.loc[0, "bout_start_ms"] == 0
    assert bouts.loc[0, "duration_ms"] == 5 * MINUTE
    assert bouts.loc[0, "censor_reason"] == "left_window"


def test_leading_off_without_history_implies_on_at_start():

    bouts = build_screen_bouts(
        events((3 * MINUTE, "lock")),
        window_start_ms=0,
        window_end_ms=DAY,
    )

    assert bouts.loc[0, "duration_ms"] == 3 * MINUTE
    assert bouts.loc[0, "bout_status"] == "censored"


def test_debounce_merges_flicker():

    screen = events(
        (HOUR, "screen_on"),
        (HOUR + MINUTE, "screen_off"),
        (HOUR + MINUTE + 300, "screen_on"),
        (HOUR + 2 * MINUTE, "screen_off"),
    )

    bouts = build_screen_bouts(screen, window_start_ms=0, window_end_ms=DAY)

    assert len(bouts) == 1
    assert bouts.loc[0, "duration_ms"] == 2 * MINUTE

    no_debounce = build_screen_bouts(
        screen,
        window_start_ms=0,
        window_end_ms=DAY,
        debounce_ms=0,
    )

    assert len(no_debounce) == 2


def test_empty_and_invalid_inputs():

    assert build_screen_bouts(
        events(),
        window_start_ms=0,
        window_end_ms=DAY,
    ).empty

    with pytest.raises(ValueError):
        build_screen_bouts(events(), window_start_ms=1, window_end_ms=1)

    with pytest.raises(ValueError):
        build_screen_bouts(
            events(),
            window_start_ms=0,
            window_end_ms=1,
            max_bout_ms=0,
        )

    with pytest.raises(ValueError):
        build_screen_bouts(
            pd.DataFrame({"timestamp_ms": [1]}),
            window_start_ms=0,
            window_end_ms=DAY,
        )


def test_daily_features():

    screen = events(
        (1 * HOUR, "screen_on"),
        (1 * HOUR + 30 * MINUTE, "screen_off"),
        (2 * HOUR, "screen_on"),
        (2 * HOUR + 10 * SECOND, "screen_off"),
        (3 * HOUR, "screen_on"),
        (3 * HOUR + 29 * MINUTE + 50 * SECOND, "screen_off"),
    )

    bouts = build_screen_bouts(screen, window_start_ms=0, window_end_ms=DAY)

    result = compute_daily_screen_features(
        bouts,
        supported_duration_ms=20 * HOUR,
        window_start_ms=0,
        window_end_ms=DAY,
    )

    assert result["screen_on_ms"] == pytest.approx(HOUR)
    assert result["screen_on_ms_status"] == "observed"
    assert result["n_bouts"] == 3
    assert result["median_bout_ms"] == pytest.approx(29 * MINUTE + 50 * SECOND)
    assert result["micro_bout_rate_per_hour"] == pytest.approx(1 / 20)
    assert result["screen_fragmentation_index"] == pytest.approx(3.0)

    # 01:00-03:30 UTC falls inside 22:00-06:00 UTC.
    assert result["nocturnal_screen_on_ms"] == pytest.approx(HOUR)
    assert result["nocturnal_n_bouts"] == 3
    assert result["n_censored_bouts"] == 0


def test_no_bouts_on_supported_day_is_observed_zero():

    result = compute_daily_screen_features(
        build_screen_bouts(events(), window_start_ms=0, window_end_ms=DAY),
        supported_duration_ms=DAY,
        window_start_ms=0,
        window_end_ms=DAY,
    )

    assert result["screen_on_ms"] == 0.0
    assert result["screen_on_ms_status"] == "observed_zero"
    assert result["n_bouts_status"] == "observed_zero"
    assert np.isnan(result["median_bout_ms"])
    assert result["median_bout_ms_status"] == "not_applicable"
    assert result["screen_fragmentation_index_status"] == "not_applicable"


def test_unsupported_day_is_never_zero():

    result = compute_daily_screen_features(
        build_screen_bouts(events(), window_start_ms=0, window_end_ms=DAY),
        supported_duration_ms=2 * HOUR,
        window_start_ms=0,
        window_end_ms=DAY,
    )

    assert np.isnan(result["screen_on_ms"])
    assert result["screen_on_ms_status"] == "insufficient_observability"


def test_censored_bout_marks_total_censored():

    bouts = build_screen_bouts(
        events((DAY - HOUR, "screen_on")),
        window_start_ms=0,
        window_end_ms=DAY,
    )

    result = compute_daily_screen_features(
        bouts,
        supported_duration_ms=DAY,
        window_start_ms=0,
        window_end_ms=DAY,
    )

    assert result["screen_on_ms"] == pytest.approx(HOUR)
    assert result["screen_on_ms_status"] == "censored"
    assert result["n_censored_bouts"] == 1


def test_nocturnal_without_window_is_not_applicable():

    result = compute_daily_screen_features(
        build_screen_bouts(
            events((HOUR, "screen_on"), (2 * HOUR, "screen_off")),
            window_start_ms=0,
            window_end_ms=DAY,
        ),
        supported_duration_ms=DAY,
    )

    assert result["nocturnal_screen_on_ms_status"] == "not_applicable"


def test_daily_features_validation():

    with pytest.raises(ValueError):
        compute_daily_screen_features(pd.DataFrame(), supported_duration_ms=1)

    bouts = build_screen_bouts(events(), window_start_ms=0, window_end_ms=DAY)

    with pytest.raises(ValueError):
        compute_daily_screen_features(bouts, supported_duration_ms=-1)


def test_supported_duration():

    screen = events(
        (1 * HOUR, "unknown"),
        (2 * HOUR, "screen_on"),
        (20 * HOUR, "screen_off"),
    )

    # Gaps: 1 h, 1 h, 18 h (too long), 4 h.
    assert screen_supported_duration_ms(
        screen,
        window_start_ms=0,
        window_end_ms=DAY,
    ) == pytest.approx(6 * HOUR)

    assert screen_supported_duration_ms(
        events(),
        window_start_ms=0,
        window_end_ms=DAY,
    ) == 0.0


def test_nocturnal_intervals_across_dst():

    tz = "America/New_York"
    start, end = local_day_bounds_ms("2026-03-08", tz)

    pieces = local_clock_intervals(
        start,
        end,
        tz=tz,
        start_local="22:00",
        end_local="06:00",
    )

    # 00:00-06:00 local on a spring-forward day is 5 elapsed hours; 22-24 is 2.
    assert [b - a for a, b in pieces] == [5 * HOUR, 2 * HOUR]
