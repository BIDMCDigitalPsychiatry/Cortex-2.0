import datetime as dt

import numpy as np
import pandas as pd
import pytest
from gps_synthetic import HOUR_MS, local_ms, offset, track

from cortex2_engine.features.anchors import ANCHOR_COLUMNS, build_daily_anchors
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.social_rhythm import (
    circular_difference_hours,
    social_rhythm_regularity,
)

TZ = "America/New_York"


def _home():
    lat, lon = offset(0.0, 0.0)
    return {"home_status": "observed", "home_latitude": lat, "home_longitude": lon, "home_radius_m": 50.0}


def test_build_daily_anchors_from_all_sources():
    date = "2026-05-04"
    screen = pd.DataFrame({
        "timestamp_ms": [local_ms(date, 1.0), local_ms(date, 7.0), local_ms(date, 7.25)],
        "event": ["unlock", "screen_on", "unlock"],
    })
    start, leave = local_ms(date, 6.0), local_ms(date, 8.5)
    gps = track(start, local_ms(date, 10.0), lambda t: (0.0, 0.0) if t < leave else (3000.0, 0.0))
    edges = build_gps_intervals(gps, window_start_ms=start, window_end_ms=local_ms(date, 10.0), max_gap_ms=600_000)
    tel = pd.DataFrame({
        "timestamp_ms": [local_ms(date, 3.0), local_ms(date, 9.5), local_ms(date, 22.0)],
        "channel": ["call", "sms", "call"],
    })
    sleep = {"sleep_offset_ms": local_ms(date, 6.5), "sleep_offset_local_h_status": "observed"}

    a = build_daily_anchors(date=date, tz=TZ, sleep=sleep, screen_events=screen,
                            gps_edges=edges, home=_home(), telephony=tel)
    got = dict(zip(a["anchor"], a["clock_hours"], strict=True))

    assert list(a.columns) == ANCHOR_COLUMNS
    assert got["sleep_offset"] == pytest.approx(6.5)
    assert got["first_unlock"] == pytest.approx(7.25)  # 01:00 belongs to the night before
    assert got["first_home_departure"] == pytest.approx(8.5 - 1 / 60)
    assert got["first_communication"] == pytest.approx(9.5)  # 03:00 is before 04:00
    assert got["last_communication"] == pytest.approx(22.0)


def test_anchors_fall_back_to_screen_on_and_skip_unobserved():
    date = dt.date(2026, 5, 4)
    screen = pd.DataFrame({"timestamp_ms": [local_ms("2026-05-04", 8.0)], "event": ["screen_on"]})
    a = build_daily_anchors(
        date=date, tz=TZ, screen_events=screen,
        sleep={"sleep_offset_ms": np.nan},
        home={"home_status": "insufficient_observability"},
        gps_edges=pd.DataFrame({"x": [1]}),
    )
    assert a["anchor"].tolist() == ["first_unlock"]

    empty = build_daily_anchors(date=date, tz=TZ)
    assert empty.empty and list(empty.columns) == ANCHOR_COLUMNS


def _anchors(times_by_anchor, start="2026-05-01"):
    rows = []
    d0 = dt.date.fromisoformat(start)
    for name, hours in times_by_anchor.items():
        for i, h in enumerate(hours):
            if h is not None:
                rows.append({"date_local": (d0 + dt.timedelta(days=i)).isoformat(), "anchor": name, "clock_hours": h})
    return pd.DataFrame(rows)


def test_circular_difference_wraps_midnight():
    assert circular_difference_hours(np.array([0.5]), 23.5)[0] == pytest.approx(1.0)
    assert circular_difference_hours(np.array([23.5]), 0.5)[0] == pytest.approx(-1.0)


def test_regular_week_scores_one_even_across_midnight():
    a = _anchors({
        "sleep_offset": [7.0, 7.2, 6.9, 7.1, 7.0, 6.8, 7.3],
        "last_communication": [23.8, 0.2, 23.9, 0.1, 0.0, 23.7, 0.3],
    })
    out = social_rhythm_regularity(a)
    assert out["social_rhythm_regularity"] == 1.0
    assert out["social_rhythm_regularity_status"] == "observed"
    assert out["last_communication_circular_sd_h"] < 0.5
    assert out["social_rhythm_n_anchors"] == 2


def test_irregular_and_gated_weeks():
    a = _anchors({
        "sleep_offset": [5.0, 9.0, 6.0, 11.0, 7.0, 10.0, 8.0],
        "first_unlock": [7.0, 7.1, 7.0, None, None, None, 7.2],
    })
    out = social_rhythm_regularity(a)
    assert out["sleep_offset_hit_fraction"] < 0.5
    assert out["first_unlock_hit_fraction"] == 1.0
    assert 0 < out["social_rhythm_regularity"] < 1

    sparse = social_rhythm_regularity(a, min_days_per_anchor=5)
    assert sparse["social_rhythm_regularity_status"] == "insufficient_observability"
    assert np.isnan(sparse["first_unlock_hit_fraction"])

    # Times evenly spread round the clock have no habitual time.
    spread = _anchors({"sleep_offset": [0, 6, 12, 18], "first_unlock": [7, 7, 7, 7]})
    assert social_rhythm_regularity(spread, min_anchors=1)["social_rhythm_n_anchors"] == 1

    outside = social_rhythm_regularity(a, end_date="2026-01-01")
    assert outside["social_rhythm_regularity_status"] == "no_raw_data"

    none = social_rhythm_regularity(pd.DataFrame(columns=["date_local", "anchor", "clock_hours"]))
    assert none["social_rhythm_mean_circular_sd_h_status"] == "no_raw_data"


def test_window_and_validation():
    a = _anchors({"sleep_offset": [7.0] * 14, "first_unlock": [8.0] * 14})
    out = social_rhythm_regularity(a, end_date=dt.date(2026, 5, 14), window_days=7)
    assert out["sleep_offset_n_days"] == 7

    with pytest.raises(ValueError):
        social_rhythm_regularity(a, window_days=0)
    with pytest.raises(ValueError):
        social_rhythm_regularity(a, regular_threshold_min=0)
    with pytest.raises(ValueError, match="Missing required anchor columns"):
        social_rhythm_regularity(pd.DataFrame({"anchor": []}))
    assert HOUR_MS
