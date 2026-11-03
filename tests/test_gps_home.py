import numpy as np
import pandas as pd
import pytest
from gps_synthetic import HOUR_MS, assignment, local_ms, offset, track

from cortex2_engine.features.gps_home import (
    compute_daily_home_time,
    infer_home_location,
)
from cortex2_engine.features.gps_intervals import build_gps_intervals

TZ = "America/New_York"


def _places():
    rows = []
    for pid, east in ((1, 0.0), (2, 5000.0), (3, 9000.0)):
        lat, lon = offset(east, 0.0)
        rows.append({
            "place_candidate_id": pid,
            "representative_latitude": lat,
            "representative_longitude": lon,
            "seed_support_radius_m": 30.0 if pid == 1 else 80.0,
        })
    return pd.DataFrame(rows)


def _nights(n_home, n_other, *, night_hours=6.0, start="2026-03-01"):
    rows = []
    day0 = pd.Timestamp(start)
    for i in range(n_home + n_other):
        date = (day0 + pd.Timedelta(days=i)).date().isoformat()
        place = 1 if i < n_home else 2
        rows.append(assignment(local_ms(date, 0), local_ms(date, night_hours), place))
        rows.append(assignment(local_ms(date, 9), local_ms(date, 17), 3))
    return pd.DataFrame(rows)


def test_home_is_the_place_that_wins_most_nights_across_dst():
    # Spans the 2026-03-08 spring-forward night (a 23 h day).
    home = infer_home_location(_nights(8, 2), _places(), tz=TZ)

    assert home["home_status"] == "observed"
    assert home["home_place_candidate_id"] == 1
    assert home["home_n_valid_nights"] == 10
    assert home["home_confidence"] == pytest.approx(0.8)
    assert home["home_radius_m"] == 50.0  # floor beats the 30 m support radius
    assert home["reference_start_date"] == "2026-03-01"
    assert home["reference_end_date"] == "2026-03-14"


def test_daytime_place_never_becomes_home():
    # Place 3 holds 8 h a day, more than any night place, but it is daytime.
    home = infer_home_location(_nights(6, 0), _places(), tz=TZ)
    assert home["home_place_candidate_id"] == 1


def test_short_nights_are_not_valid():
    home = infer_home_location(_nights(8, 0, night_hours=1.5), _places(), tz=TZ)
    assert home["home_status"] == "insufficient_observability"
    assert home["home_reason"] == "no_valid_nights"
    assert np.isnan(home["home_latitude"])


def test_too_few_nights_and_inconsistent_nights():
    few = infer_home_location(_nights(3, 0), _places(), tz=TZ)
    assert few["home_reason"] == "too_few_valid_nights"
    assert few["home_place_candidate_id"] == 1

    mixed = infer_home_location(_nights(4, 4), _places(), tz=TZ, min_confidence=0.6)
    assert mixed["home_reason"] == "inconsistent_night_place"
    assert mixed["home_status"] == "insufficient_observability"


def test_reference_period_limits_nights():
    home = infer_home_location(_nights(20, 0), _places(), tz=TZ, reference_days=7)
    assert home["home_n_valid_nights"] == 7


def test_home_inputs_validated():
    no_rows = pd.DataFrame(columns=["episode_start_ms", "episode_end_ms", "place_candidate_id"])
    empty = infer_home_location(no_rows, _places(), tz=TZ)
    assert empty["home_status"] == "no_raw_data"

    with pytest.raises(ValueError, match="Missing required assignments"):
        infer_home_location(pd.DataFrame({"x": [1]}), _places(), tz=TZ)
    with pytest.raises(ValueError):
        infer_home_location(_nights(5, 0), _places(), tz=TZ, min_confidence=2)
    with pytest.raises(ValueError):
        infer_home_location(_nights(5, 0), _places(), tz=TZ, min_nights=0)
    with pytest.raises(ValueError, match="missing from places"):
        infer_home_location(_nights(5, 0), _places().iloc[1:], tz=TZ)


def _day_edges(home_hours, away_hours):
    start = local_ms("2026-03-20", 0)
    switch = start + int(home_hours * HOUR_MS)
    end = switch + int(away_hours * HOUR_MS)
    gps = track(start, end, lambda t: (0.0, 0.0) if t < switch else (5000.0, 0.0))
    return build_gps_intervals(gps, window_start_ms=start, window_end_ms=end, max_gap_ms=600_000)


def test_daily_home_time_counts_only_supported_time_near_home():
    home = infer_home_location(_nights(8, 0), _places(), tz=TZ)
    out = compute_daily_home_time(_day_edges(8, 8), home)

    # The departure edge counts half.
    assert out["home_time_ms"] == pytest.approx(8 * HOUR_MS - 30_000, rel=1e-6)
    assert out["home_time_fraction"] == pytest.approx(0.5, abs=1e-3)
    assert out["home_time_fraction_status"] == "observed"


def test_daily_home_time_zero_and_gates():
    home = infer_home_location(_nights(8, 0), _places(), tz=TZ)

    away = compute_daily_home_time(_day_edges(0, 8), home)
    assert away["home_time_ms"] == 0.0
    assert away["home_time_ms_status"] == "observed_zero"

    short = compute_daily_home_time(_day_edges(2, 1), home)
    assert np.isnan(short["home_time_ms"])
    assert short["home_time_ms_status"] == "insufficient_observability"

    unknown = compute_daily_home_time(
        _day_edges(8, 8), infer_home_location(_nights(2, 0), _places(), tz=TZ)
    )
    assert np.isnan(unknown["home_time_fraction"])
    assert unknown["home_time_fraction_status"] == "insufficient_observability"

    with pytest.raises(ValueError, match="Missing required GPS edge columns"):
        compute_daily_home_time(pd.DataFrame({"supported": []}), home)
