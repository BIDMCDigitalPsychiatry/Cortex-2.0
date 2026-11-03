import numpy as np
import pandas as pd
import pytest
from acc_synthetic import HOUR, MINUTE, day_with_still_block, samples

from cortex2_engine.features.acc_nonwear import detect_nonwear
from cortex2_engine.features.acc_preprocess import resample_to_epochs

DAY = 24 * HOUR


def _epochs(acc, start=0, end=DAY):
    return resample_to_epochs(acc, window_start_ms=start, window_end_ms=end)


def test_six_hours_of_injected_nonwear_is_recovered():

    acc = day_with_still_block(0, DAY, 2 * HOUR, 8 * HOUR)

    result = detect_nonwear(_epochs(acc))

    assert result.summary["acc_nonwear_ms"] == pytest.approx(6 * HOUR)
    assert result.summary["acc_wear_ms"] == pytest.approx(18 * HOUR)
    assert result.summary["acc_nonwear_fraction"] == pytest.approx(0.25)
    assert result.summary["acc_nonwear_fraction_status"] == "observed"

    assert result.intervals[["start_ms", "end_ms"]].values.tolist() == [
        [2 * HOUR, 8 * HOUR],
    ]


def test_still_period_shorter_than_window_is_not_nonwear():

    acc = day_with_still_block(0, 6 * HOUR, 2 * HOUR, 2 * HOUR + 45 * MINUTE)

    result = detect_nonwear(_epochs(acc, end=6 * HOUR))

    assert result.summary["acc_nonwear_ms"] == 0.0
    assert result.summary["acc_nonwear_fraction_status"] == "observed_zero"
    assert result.intervals.empty


def test_unobserved_time_is_never_nonwear():

    # Still phone for 1 h, then 5 h with no samples at all.
    acc = samples(0, HOUR, moving=False)

    result = detect_nonwear(_epochs(acc, end=6 * HOUR))

    assert result.summary["acc_observed_ms"] == pytest.approx(HOUR)
    assert result.summary["acc_nonwear_ms"] == pytest.approx(HOUR)
    assert result.nonwear.sum() == 3600


def test_sparse_window_is_not_judged():

    # 10 minutes of still samples inside an otherwise empty hour.
    acc = samples(0, 10 * MINUTE, moving=False)

    result = detect_nonwear(_epochs(acc, end=2 * HOUR))

    assert result.summary["acc_nonwear_ms"] == 0.0
    assert result.summary["acc_evaluable_ms"] == 0.0
    assert np.isnan(result.summary["acc_nonwear_fraction"])
    assert (
        result.summary["acc_nonwear_fraction_status"]
        == "insufficient_observability"
    )


def test_one_moving_axis_still_counts_as_nonwear_with_two_still_axes():

    acc = samples(0, 2 * HOUR, moving=False)
    rng = np.random.default_rng(3)
    acc["x"] = rng.normal(0.0, 0.2, len(acc))

    assert detect_nonwear(_epochs(acc, end=2 * HOUR)).summary[
        "acc_nonwear_ms"
    ] == pytest.approx(2 * HOUR)

    assert detect_nonwear(_epochs(acc, end=2 * HOUR), min_axes=3).summary[
        "acc_nonwear_ms"
    ] == 0.0


def test_charging_annotation_from_battery():

    acc = day_with_still_block(0, 12 * HOUR, 2 * HOUR, 6 * HOUR)

    battery = pd.DataFrame({
        "timestamp_ms": [2 * HOUR + MINUTE, 5 * HOUR],
        "battery_level_fraction": [0.30, 0.90],
    })

    result = detect_nonwear(_epochs(acc, end=12 * HOUR), battery=battery)

    assert result.intervals["charging"].tolist() == [True]
    assert result.summary["acc_nonwear_charging_ms"] == pytest.approx(4 * HOUR)


def test_empty_and_invalid_inputs():

    empty = detect_nonwear(_epochs(samples(0, 0, moving=False), end=HOUR))

    assert empty.summary["acc_nonwear_fraction_status"] == "no_raw_data"

    epochs = _epochs(samples(0, HOUR, moving=False), end=HOUR)

    with pytest.raises(ValueError):
        detect_nonwear(epochs, step_ms=7 * MINUTE)

    with pytest.raises(ValueError):
        detect_nonwear(epochs, min_axes=4)

    with pytest.raises(ValueError):
        detect_nonwear(epochs, sd_threshold_mg=0)

    with pytest.raises(ValueError):
        detect_nonwear(epochs.drop(columns="sum_x_g"))

    # Battery without its level column is rejected once there is an interval.
    with pytest.raises(ValueError):
        detect_nonwear(
            _epochs(samples(0, 2 * HOUR, moving=False), end=2 * HOUR),
            battery=pd.DataFrame({"timestamp_ms": [0]}),
        )

    assert detect_nonwear(epochs.iloc[0:0]).intervals.empty
