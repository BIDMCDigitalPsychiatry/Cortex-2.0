import datetime as dt

import numpy as np
import pandas as pd
import pytest
from gps_synthetic import HOUR_MS, assignment, local_ms, track

from cortex2_engine.features.gps_circadian_movement import (
    compute_circadian_movement,
    weighted_sinusoid_r2,
)
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_routine import (
    hourly_place_sequence,
    location_novelty_fraction,
    routine_index,
)

TZ = "America/New_York"


def _day(date, blocks):
    """``blocks`` = [(start_hour, end_hour, place), ...] in wall-clock hours."""
    return [assignment(local_ms(date, a, TZ), local_ms(date, b, TZ), p) for a, b, p in blocks]


ROUTINE = [(0, 9, 1), (9, 17, 2), (17, 24, 1)]


def _history(end_date, n_days, blocks=ROUTINE):
    end = dt.date.fromisoformat(end_date)
    rows = []
    for k in range(n_days, 0, -1):
        rows += _day((end - dt.timedelta(days=k)).isoformat(), blocks)
    return rows


def test_identical_day_has_routine_one_and_no_novelty():
    date = "2026-10-20"
    a = pd.DataFrame(_history(date, 28) + _day(date, ROUTINE))

    r = routine_index(a, date=date, tz=TZ)
    assert r["routine_index"] == pytest.approx(1.0)
    assert r["routine_index_status"] == "observed"
    assert r["routine_n_days"] == 7

    n = location_novelty_fraction(a, date=date, tz=TZ)
    assert n["location_novelty_fraction"] == 0.0
    assert n["location_novelty_fraction_status"] == "observed_zero"
    assert n["novelty_history_days"] == 28


def test_new_place_raises_novelty_and_lowers_routine():
    date = "2026-10-20"
    today = [(0, 9, 1), (9, 15, 5), (15, 17, 2), (17, 24, 1)]
    a = pd.DataFrame(_history(date, 28) + _day(date, today))

    n = location_novelty_fraction(a, date=date, tz=TZ)
    assert n["location_novelty_fraction"] == pytest.approx(6 / 24)

    r = routine_index(a, date=date, tz=TZ)
    assert r["routine_index"] == pytest.approx(18 / 24)


def test_routine_on_a_25_hour_dst_day():
    date = "2026-11-01"  # fall back: 01:00-02:00 occurs twice
    a = pd.DataFrame(_history(date, 7) + _day(date, ROUTINE))

    seq = hourly_place_sequence(a, date=date, tz=TZ)
    assert sorted(seq) == list(range(24))
    assert routine_index(a, date=date, tz=TZ)["routine_index"] == pytest.approx(1.0)


def test_routine_and_novelty_gates():
    date = "2026-10-20"
    sparse_history = pd.DataFrame(_history(date, 2) + _day(date, ROUTINE))
    r = routine_index(sparse_history, date=date, tz=TZ)
    assert r["routine_index_status"] == "insufficient_observability"
    assert r["routine_n_days"] == 2

    n = location_novelty_fraction(sparse_history, date=date, tz=TZ)
    assert n["location_novelty_fraction_status"] == "insufficient_observability"
    assert np.isnan(n["location_novelty_fraction"])

    # A day with only 4 h of assignments is too thin for novelty.
    thin = pd.DataFrame(_history(date, 28) + _day(date, [(0, 4, 1)]))
    assert location_novelty_fraction(thin, date=date, tz=TZ)[
        "location_novelty_fraction_status"
    ] == "insufficient_observability"

    # Totally different, fully observed day: routine is an observed zero.
    other = pd.DataFrame(_history(date, 7) + _day(date, [(0, 24, 9)]))
    zero = routine_index(other, date=date, tz=TZ)
    assert zero["routine_index"] == 0.0
    assert zero["routine_index_status"] == "observed_zero"

    empty = pd.DataFrame(columns=["episode_start_ms", "episode_end_ms", "place_candidate_id"])
    assert routine_index(empty, date=date, tz=TZ)["routine_index_status"] == "no_raw_data"

    with pytest.raises(ValueError):
        routine_index(other, date=date, tz=TZ, min_days=0)
    with pytest.raises(ValueError):
        location_novelty_fraction(other, date=date, tz=TZ, history_days=0)
    with pytest.raises(ValueError, match="Missing required assignment"):
        routine_index(pd.DataFrame({"x": [1]}), date=date, tz=TZ)


def test_weighted_sinusoid_r2():
    t = np.arange(0, 3 * 24 * HOUR_MS, HOUR_MS, dtype=float)
    y = np.sin(2 * np.pi * t / (24 * HOUR_MS))
    w = np.ones_like(t)
    assert weighted_sinusoid_r2(t, y, w, period_ms=24 * HOUR_MS) == pytest.approx(1.0)
    assert weighted_sinusoid_r2(t, np.zeros_like(t), w, period_ms=24 * HOUR_MS) == 0.0


def _edges(positions, days=4.0, start=0):
    end = start + int(days * 24 * HOUR_MS)
    gps = track(start, end, positions, step_ms=5 * 60_000)
    return build_gps_intervals(gps, window_start_ms=start, window_end_ms=end, max_gap_ms=900_000)


def test_regular_commuter_has_higher_circadian_movement_than_random():
    def commuter(t):
        hour = (t / HOUR_MS) % 24
        return (3000.0, 0.0) if 9 <= hour < 17 else (0.0, 0.0)

    rng = np.random.default_rng(2)
    noise = rng.uniform(0, 3000, (5000, 2))

    regular = compute_circadian_movement(_edges(commuter))
    random = compute_circadian_movement(
        _edges(lambda t: tuple(noise[int(t // (5 * 60_000)) % 5000]))
    )

    assert regular["circadian_movement_status"] == "observed"
    assert regular["circadian_energy_east"] > 0.5
    assert regular["circadian_energy_north"] == 0.0
    assert regular["circadian_movement"] > random["circadian_movement"] + 2


def test_circadian_movement_gates():
    still = compute_circadian_movement(_edges(lambda t: (0.0, 0.0)))
    assert still["circadian_movement_status"] == "not_applicable"
    assert np.isnan(still["circadian_movement"])

    # 16 h of data: enough supported time but less than one cycle.
    short = compute_circadian_movement(_edges(lambda t: (t / 1000.0, 0.0), days=16 / 24))
    assert short["circadian_movement_status"] == "insufficient_observability"

    thin = compute_circadian_movement(_edges(lambda t: (0.0, 0.0), days=0.25))
    assert thin["circadian_movement_status"] == "insufficient_observability"

    none = compute_circadian_movement(_edges(lambda t: (0.0, 0.0)).iloc[0:0])
    assert none["circadian_movement_status"] == "no_raw_data"

    with pytest.raises(ValueError):
        compute_circadian_movement(_edges(lambda t: (0.0, 0.0)), period_band_h=(25, 23))
