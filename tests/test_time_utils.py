import datetime as dt

import numpy as np
import pytest

from cortex2_engine.config import StudyConfig
from cortex2_engine.time import (
    DAY_MS,
    HOUR_MS,
    circular_mean_hours,
    circular_sd_hours,
    iter_local_days,
    local_dates,
    local_day_bounds_ms,
    parse_clock,
    to_local_clock_hours,
)

NY = "America/New_York"

# 2026 US DST: spring forward Sunday 2026-03-08, fall back Sunday 2026-11-01.
SPRING = dt.date(2026, 3, 8)
FALL = dt.date(2026, 11, 1)


def test_utc_day_is_24_hours():

    start, end = local_day_bounds_ms("2026-03-08", "UTC")

    assert end - start == DAY_MS
    assert start == 1_772_928_000_000


def test_spring_forward_day_is_23_hours():

    start, end = local_day_bounds_ms(SPRING, NY)

    assert end - start == 23 * HOUR_MS
    # Local midnight EST is 05:00 UTC.
    assert start == 1_772_928_000_000 + 5 * HOUR_MS


def test_fall_back_day_is_25_hours():

    start, end = local_day_bounds_ms(FALL, NY)

    assert end - start == 25 * HOUR_MS


def test_day_start_offset():

    start, end = local_day_bounds_ms(
        "2026-06-01",
        NY,
        day_start="04:00",
    )

    midnight, _ = local_day_bounds_ms("2026-06-01", NY)

    assert start - midnight == 4 * HOUR_MS
    assert end - start == DAY_MS


def test_iter_local_days_clips_and_flags_partial_days():

    first_start, _ = local_day_bounds_ms(dt.date(2026, 3, 7), NY)

    window_start = first_start + 12 * HOUR_MS
    _, window_end = local_day_bounds_ms(dt.date(2026, 3, 9), NY)
    window_end -= 6 * HOUR_MS

    days = list(iter_local_days(window_start, window_end, NY))

    assert [d.date for d in days] == [
        dt.date(2026, 3, 7),
        dt.date(2026, 3, 8),
        dt.date(2026, 3, 9),
    ]

    assert [d.clipped for d in days] == [True, False, True]
    assert days[0].duration_ms == 12 * HOUR_MS
    assert days[1].duration_ms == 23 * HOUR_MS
    assert days[1].day_duration_ms == 23 * HOUR_MS
    assert days[2].duration_ms == 18 * HOUR_MS

    # Days tile the window exactly.
    assert sum(d.duration_ms for d in days) == window_end - window_start


def test_iter_local_days_rejects_empty_window():

    with pytest.raises(ValueError):
        list(iter_local_days(10, 10, NY))


def test_local_dates_respect_day_start():

    start, _ = local_day_bounds_ms("2026-06-02", NY)

    stamps = [start - 1, start, start + 3 * HOUR_MS]

    assert list(local_dates(stamps, NY)) == [
        dt.date(2026, 6, 1),
        dt.date(2026, 6, 2),
        dt.date(2026, 6, 2),
    ]

    # With a 04:00 day start, 03:00 local belongs to the previous day.
    assert list(local_dates(stamps, NY, day_start="04:00")) == [
        dt.date(2026, 6, 1),
        dt.date(2026, 6, 1),
        dt.date(2026, 6, 1),
    ]


def test_local_clock_hours_across_dst():

    start, _ = local_day_bounds_ms(SPRING, NY)

    hours = to_local_clock_hours(
        [start + 1 * HOUR_MS, start + 3 * HOUR_MS],
        NY,
    )

    # 01:00 EST, then two elapsed hours later the wall clock reads 04:00 EDT.
    assert hours.tolist() == pytest.approx([1.0, 4.0])


def test_local_clock_hours_missing_is_nan():

    assert np.isnan(to_local_clock_hours([None], NY)[0])


def test_circular_sd_wraps_midnight():

    assert circular_sd_hours(np.array([23.5, 0.5])) == pytest.approx(
        0.5,
        abs=0.01,
    )

    assert circular_mean_hours(np.array([23.5, 0.5])) == pytest.approx(
        0.0,
        abs=1e-9,
    ) or circular_mean_hours(np.array([23.5, 0.5])) == pytest.approx(24.0)


def test_circular_sd_edge_cases():

    assert circular_sd_hours(np.array([7.0, 7.0, 7.0])) == 0.0
    assert np.isnan(circular_sd_hours(np.array([])))
    assert circular_sd_hours(np.array([0.0, 12.0])) == float("inf")
    assert np.isnan(circular_mean_hours(np.array([0.0, 12.0])))
    assert np.isnan(circular_mean_hours(np.array([np.nan])))


def test_circular_sd_small_spread_matches_linear_sd():

    hours = np.array([7.0, 7.5, 8.0, 8.5, 9.0])

    assert circular_sd_hours(hours) == pytest.approx(
        np.std(hours),
        rel=0.02,
    )


def test_parse_clock():

    assert parse_clock("06:30") == dt.time(6, 30)
    assert parse_clock(dt.time(1, 2)) == dt.time(1, 2)

    with pytest.raises(ValueError):
        parse_clock("25:99")


def _config(**overrides):

    raw = {
        "name": "s",
        "output_dir": "out",
        "windows": {"path": "w.csv"},
    }
    raw.update(overrides)

    return StudyConfig.model_validate(raw)


def test_study_config_day_start_and_timezone():

    config = _config(timezone=NY, day_start="04:00")

    assert config.day_start == "04:00"
    assert _config().day_start == "00:00"

    with pytest.raises(ValueError):
        _config(timezone="Mars/Olympus")

    with pytest.raises(ValueError):
        _config(day_start="noon")
