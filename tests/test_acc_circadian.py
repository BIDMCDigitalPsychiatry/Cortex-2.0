import math

import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.acc_circadian import (
    compute_circadian_features,
    hourly_activity_matrix,
    interdaily_stability,
    intradaily_variability,
    relative_amplitude,
)
from cortex2_engine.time import local_day_bounds_ms

HOURS = np.arange(24)


def sinusoid(days=7, amplitude=40.0, mean=50.0):
    profile = mean + amplitude * np.sin(2 * np.pi * HOURS / 24)
    return np.tile(profile, (days, 1)), np.ones((days, 24), dtype=bool)


def brute_force_iv(matrix):
    x = matrix.ravel()
    n = x.size
    return n * np.sum(np.diff(x) ** 2) / ((n - 1) * np.sum((x - x.mean()) ** 2))


def test_identical_sinusoidal_days_have_is_one():

    values, mask = sinusoid()

    result = interdaily_stability(values, mask)

    assert result["interdaily_stability"] == pytest.approx(1.0)
    assert result["interdaily_stability_status"] == "observed"


def test_sinusoid_iv_matches_formula_and_analytic_value():

    values, mask = sinusoid()

    result = intradaily_variability(values, mask)

    assert result["intradaily_variability"] == pytest.approx(
        brute_force_iv(values)
    )

    # Smooth 24 h sinusoid: IV -> 4 sin^2(pi/24).
    assert result["intradaily_variability"] == pytest.approx(
        4 * math.sin(math.pi / 24) ** 2,
        rel=0.02,
    )

    assert result["intradaily_variability_n_pairs"] == values.size - 1


def test_white_noise_has_iv_near_two_and_low_is():

    rng = np.random.default_rng(0)
    values = rng.normal(50, 10, (60, 24))
    mask = np.ones_like(values, dtype=bool)

    iv = intradaily_variability(values, mask)["intradaily_variability"]
    is_ = interdaily_stability(values, mask)["interdaily_stability"]

    assert iv == pytest.approx(2.0, abs=0.15)
    assert is_ < 0.1


def test_day_to_day_offsets_lower_is():

    values, mask = sinusoid()
    shifted = values + np.arange(7)[:, None] * 15.0

    assert interdaily_stability(shifted, mask)["interdaily_stability"] < 0.9


def test_relative_amplitude_m10_l5():

    profile = np.full(24, 50.0)
    profile[8:18] = 100.0
    profile[0:5] = 10.0

    values = np.tile(profile, (7, 1))
    mask = np.ones_like(values, dtype=bool)

    result = relative_amplitude(values, mask)

    assert result["m10_mg"] == pytest.approx(100.0)
    assert result["l5_mg"] == pytest.approx(10.0)
    assert result["relative_amplitude"] == pytest.approx(90.0 / 110.0)
    assert result["m10_onset_local_h"] == 8.0
    assert result["l5_onset_local_h"] == 0.0
    assert result["relative_amplitude_status"] == "observed"


def test_l5_window_wraps_midnight():

    profile = np.full(24, 50.0)
    profile[[22, 23, 0, 1, 2]] = 5.0

    values = np.tile(profile, (6, 1))
    result = relative_amplitude(values, np.ones_like(values, dtype=bool))

    assert result["l5_onset_local_h"] == 22.0
    assert result["l5_mg"] == pytest.approx(5.0)


def test_gating_min_days_and_hours():

    values, mask = sinusoid(days=4)

    result = compute_circadian_features(values, mask)

    assert np.isnan(result["interdaily_stability"])
    assert result["interdaily_stability_status"] == "insufficient_observability"
    assert result["relative_amplitude_status"] == "insufficient_observability"

    values, mask = sinusoid(days=7)
    mask[:, :7] = False

    result = compute_circadian_features(values, mask)

    assert result["n_days_valid"] == 0
    assert result["intradaily_variability_status"] == "insufficient_observability"


def test_excluded_fraction_cap():

    values, mask = sinusoid(days=7)
    mask[:, :6] = False

    assert interdaily_stability(values, mask)[
        "interdaily_stability_status"
    ] == "insufficient_observability"

    assert interdaily_stability(values, mask, max_excluded_fraction=0.3)[
        "interdaily_stability_status"
    ] == "observed"


def test_excluded_day_breaks_iv_chain():

    values, mask = sinusoid(days=8)
    mask[3, :] = False

    result = intradaily_variability(values, mask)

    # 7 valid days in two runs (3 + 4): 3*24-1 + 4*24-1 pairs.
    assert result["intradaily_variability_n_pairs"] == 72 - 1 + 96 - 1


def test_constant_signal_is_not_applicable():

    values = np.full((7, 24), 20.0)
    mask = np.ones_like(values, dtype=bool)

    result = compute_circadian_features(values, mask)

    assert result["interdaily_stability_status"] == "not_applicable"
    assert result["intradaily_variability_status"] == "not_applicable"
    assert result["relative_amplitude"] == pytest.approx(0.0)


def test_empty_and_shape_errors():

    result = interdaily_stability(np.zeros((0, 24)), np.zeros((0, 24), bool))
    assert result["interdaily_stability_status"] == "no_raw_data"

    with pytest.raises(ValueError):
        interdaily_stability(np.zeros((3, 23)), np.ones((3, 23), bool))

    with pytest.raises(ValueError):
        interdaily_stability(np.zeros((3, 24)), np.ones((2, 24), bool))


def _minutes(start, end, value=10.0):
    t = np.arange(start, end, 60_000, dtype=np.int64)
    return pd.DataFrame({
        "epoch_start_ms": t,
        "enmo_mg": np.full(t.size, value),
        "valid": np.ones(t.size, dtype=bool),
    })


def test_hourly_matrix_handles_dst_and_missing_days():

    tz = "America/New_York"
    s1, e1 = local_day_bounds_ms("2026-03-07", tz)
    s2, e2 = local_day_bounds_ms("2026-03-08", tz)  # 23 h
    s4, e4 = local_day_bounds_ms("2026-03-10", tz)

    minutes = pd.concat(
        [_minutes(s1, e1), _minutes(s2, e2), _minutes(s4, e4)],
        ignore_index=True,
    )

    dates, values, observed = hourly_activity_matrix(minutes, tz=tz)

    assert [d.isoformat() for d in dates] == [
        "2026-03-07",
        "2026-03-08",
        "2026-03-09",
        "2026-03-10",
    ]

    assert observed[0].all()
    assert observed[1].sum() == 23
    assert not observed[1, 2]
    assert not observed[2].any()
    assert values[0, 5] == pytest.approx(10.0)


def test_hourly_matrix_requires_half_hour_coverage():

    tz = "UTC"
    minutes = _minutes(0, 20 * 60_000)

    _, _, observed = hourly_activity_matrix(minutes, tz=tz)

    assert not observed[0, 0]

    with pytest.raises(ValueError):
        hourly_activity_matrix(minutes.drop(columns="valid"), tz=tz)

    dates, values, _ = hourly_activity_matrix(minutes.iloc[0:0], tz=tz)
    assert dates == [] and values.shape == (0, 24)
