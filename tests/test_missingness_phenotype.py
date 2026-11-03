import numpy as np
import pandas as pd
import pytest

from cortex2_engine.missingness.phenotype import (
    PHENOTYPES,
    classify_day,
    daily_missingness_phenotype,
)

H = 3_600_000


def _hours(gps, acc, *, battery=None, unavailable=None, episode="E1", start_ms=0):
    n = len(gps)
    df = pd.DataFrame({
        "episode_id": episode,
        "participant_id": "P1",
        "hour_index": np.arange(n),
        "start_ms": start_ms + np.arange(n) * H,
        "gps_observed": np.asarray(gps, bool),
        "acc_observed": np.asarray(acc, bool),
        "screen_event_observed": True,
        "gps_dq": 1.0,
        "acc_dq": 1.0,
        "gps_source_unavailable": False if unavailable is None else unavailable,
        "acc_source_unavailable": False,
        "gps_partial_source": False,
        "acc_partial_source": False,
        "gps_boundary_unknown": False,
        "acc_boundary_unknown": False,
    })
    if battery is not None:
        df["battery_last"] = battery
    return df


def _one(gps, acc, **kw):
    out = daily_missingness_phenotype(_hours(gps, acc, **kw), tz="UTC")
    assert len(out) == 1
    return out.iloc[0]


ON = [True] * 24


def test_complete_and_no_signal():
    assert _one(ON, ON)["missingness_phenotype"] == "complete"
    assert _one(ON, ON)["completeness_weight"] == 1.0

    nothing = _one([False] * 24, [False] * 24)
    assert nothing["missingness_phenotype"] == "no_signal"
    assert nothing["completeness_weight"] == 0.0


def test_gps_specific_dropout():
    gps = [True] * 12 + [False] * 12
    row = _one(gps, ON)
    assert row["missingness_phenotype"] == "gps_specific_dropout"
    assert row["n_gps_specific_onsets"] == 1
    assert row["completeness_weight"] == pytest.approx(np.sqrt(0.25))


def test_shared_interruption_versus_battery_depletion():
    gps = [True] * 14 + [False] * 10
    battery_high = [0.8] * 24
    battery_low = [0.8] * 13 + [0.1] + [np.nan] * 10

    interrupted = _one(gps, gps, battery=battery_high)
    assert interrupted["missingness_phenotype"] == "shared_interruption"
    assert interrupted["n_shared_onsets"] == 1

    depleted = _one(gps, gps, battery=battery_low)
    assert depleted["missingness_phenotype"] == "battery_depletion"
    assert depleted["n_low_battery_shared_onsets"] == 1

    # Without battery data a shared gap is never labelled battery depletion.
    assert _one(gps, gps)["missingness_phenotype"] == "shared_interruption"


def test_mixed_and_eligibility():
    gps = [True] * 8 + [False] * 4 + [True] * 12
    acc = [True] * 16 + [False] * 4 + [True] * 4
    assert _one(gps, acc)["missingness_phenotype"] == "mixed"

    unavailable = [True] * 16 + [False] * 8
    short = _one(ON, ON, unavailable=unavailable)
    assert short["missingness_phenotype_status"] == "insufficient_observability"
    assert np.isnan(short["missingness_phenotype"])

    absent = _one(ON, ON, unavailable=[True] * 24)
    assert absent["missingness_phenotype_status"] == "source_absent"
    assert np.isnan(absent["completeness_weight"])


def test_days_split_on_local_dates_with_dst():
    # 2026-03-08 in New York is 23 h long: 47 hours from local midnight on 03-07
    # cover exactly two local days.
    start = int(pd.Timestamp("2026-03-07", tz="America/New_York").timestamp() * 1000)
    df = _hours([True] * 47, [True] * 47, start_ms=start)
    out = daily_missingness_phenotype(df, tz="America/New_York")
    assert out["date_local"].tolist() == ["2026-03-07", "2026-03-08"]
    assert out["n_hours"].tolist() == [24, 23]


def test_classify_day_rules_and_validation():
    assert set(PHENOTYPES) == {
        "no_signal", "complete", "battery_depletion",
        "shared_interruption", "gps_specific_dropout", "mixed",
    }
    base = {"n_eligible_hours": 24, "gps_specific_gap_hours": 0, "shared_gap_hours": 0,
                "acc_only_gap_hours": 10, "n_shared_onsets": 0, "n_low_battery_shared_onsets": 0}
    assert classify_day(**base) == "mixed"  # accelerometer-only gaps
    with pytest.raises(ValueError):
        classify_day(**(base | {"n_eligible_hours": 0}))
    with pytest.raises(ValueError):
        daily_missingness_phenotype(_hours(ON, ON), tz="UTC", dominance_fraction=0)
    with pytest.raises(ValueError, match="start_ms"):
        daily_missingness_phenotype(_hours(ON, ON).drop(columns="start_ms"), tz="UTC")
