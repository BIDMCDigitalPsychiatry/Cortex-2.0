import numpy as np
import pytest
from acc_synthetic import HOUR, day_with_still_block, samples

from cortex2_engine.features.acc_daily import (
    compute_daily_activity_features,
    prepare_minute_epochs,
)
from cortex2_engine.features.acc_preprocess import STANDARD_GRAVITY_M_S2
from cortex2_engine.time import local_day_bounds_ms

DAY = 24 * HOUR


def test_daily_features_exclude_nonwear():

    acc = day_with_still_block(0, DAY, 0, 6 * HOUR)

    result = compute_daily_activity_features(
        acc,
        window_start_ms=0,
        window_end_ms=DAY,
    )

    assert result["acc_nonwear_ms"] == pytest.approx(6 * HOUR)
    assert result["acc_valid_minutes"] == 18 * 60
    assert result["mean_enmo_mg"] > 30
    assert result["mean_enmo_mg_status"] == "observed"
    assert result["acc_unit"] == "g"
    assert result["acc_observed_fraction"] == pytest.approx(1.0)
    assert result["algorithm"] == "cortex2.acc_daily"


def test_unit_confusion_g_vs_m_s2_gives_same_features():

    acc_g = samples(0, 12 * HOUR, moving=True, seed=4)
    acc_ms2 = acc_g.assign(
        x=acc_g["x"] * STANDARD_GRAVITY_M_S2,
        y=acc_g["y"] * STANDARD_GRAVITY_M_S2,
        z=acc_g["z"] * STANDARD_GRAVITY_M_S2,
    )

    kwargs = {"window_start_ms": 0, "window_end_ms": 12 * HOUR}

    a = compute_daily_activity_features(acc_g, **kwargs)
    b = compute_daily_activity_features(acc_ms2, **kwargs)

    assert b["acc_unit"] == "m_s2"
    assert b["mean_enmo_mg"] == pytest.approx(a["mean_enmo_mg"])
    assert b["active_minutes"] == a["active_minutes"]

    # Declaring the wrong unit is what the detection guards against: m/s^2
    # read as g inflates ENMO about tenfold.
    wrong = compute_daily_activity_features(acc_ms2, unit="g", **kwargs)
    assert wrong["mean_enmo_mg"] > 5 * a["mean_enmo_mg"]


def test_unknown_unit_is_extraction_failed():

    acc = samples(0, 12 * HOUR, moving=False).assign(z=0.0)

    result = compute_daily_activity_features(
        acc,
        window_start_ms=0,
        window_end_ms=12 * HOUR,
    )

    assert result["mean_enmo_mg_status"] == "extraction_failed"
    assert np.isnan(result["mean_enmo_mg"])
    assert result["acc_error_type"] == "accelerometer_unit_unknown"


def test_no_samples_is_no_raw_data():

    result = compute_daily_activity_features(
        samples(0, 0, moving=True),
        window_start_ms=0,
        window_end_ms=DAY,
    )

    assert result["mean_enmo_mg_status"] == "no_raw_data"
    assert result["astp_per_min_status"] == "no_raw_data"
    assert result["acc_n_samples"] == 0


def test_dst_spring_forward_day_uses_23_hour_denominator():

    tz = "America/New_York"
    start, end = local_day_bounds_ms("2026-03-08", tz)

    acc = samples(start, end, moving=True, seed=9)

    result = compute_daily_activity_features(
        acc,
        window_start_ms=start,
        window_end_ms=end,
    )

    assert result["day_duration_ms"] == 23 * HOUR
    assert result["acc_observed_fraction"] == pytest.approx(1.0)
    assert result["acc_valid_minutes"] == 23 * 60


def test_prepare_minute_epochs_shapes():

    prepared = prepare_minute_epochs(
        samples(0, 2 * HOUR, moving=True),
        window_start_ms=0,
        window_end_ms=3 * HOUR,
    )

    assert len(prepared.minutes) == 180
    assert prepared.minutes["valid"].sum() == 120
    assert prepared.n_samples == 7200
