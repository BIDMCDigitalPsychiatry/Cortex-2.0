import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.acc_fragmentation import (
    activity_fragmentation,
    gini_coefficient,
)


def minutes(states, valid=None):
    states = np.asarray(states, dtype=bool)
    return pd.DataFrame({
        "enmo_mg": np.where(states, 80.0, 5.0),
        "valid": np.ones(states.size, dtype=bool) if valid is None else valid,
    })


def test_transition_probabilities():

    # 600 sedentary + active bouts of 10, 20, 30 separated by sedentary.
    pattern = [0] * 300 + [1] * 10 + [0] * 100 + [1] * 20 + [0] * 100 + [1] * 30
    pattern += [0] * 100

    result = activity_fragmentation(minutes(pattern), min_valid_minutes=600)

    assert result["active_minutes"] == 60
    assert result["n_active_bouts"] == 3
    assert result["astp_per_min"] == pytest.approx(3 / 60)
    assert result["mean_active_bout_min"] == pytest.approx(20.0)
    assert result["satp_per_min"] == pytest.approx(4 / 600)
    assert result["active_bout_gini"] == pytest.approx(gini_coefficient([10, 20, 30]))
    assert result["astp_per_min_status"] == "observed"


def test_invalid_minutes_break_bouts_and_are_not_sedentary():

    pattern = [1] * 20 + [0] * 700
    valid = np.ones(len(pattern), dtype=bool)
    valid[10] = False

    result = activity_fragmentation(minutes(pattern, valid))

    assert result["n_active_bouts"] == 2
    assert result["active_minutes"] == 19
    assert result["sedentary_minutes"] == 700
    assert result["n_censored_bouts"] == 3


def test_no_activity_is_observed_zero_with_undefined_bout_metrics():

    result = activity_fragmentation(minutes([0] * 700))

    assert result["active_minutes"] == 0.0
    assert result["active_minutes_status"] == "observed_zero"
    assert np.isnan(result["astp_per_min"])
    assert result["astp_per_min_status"] == "not_applicable"
    assert result["mean_active_bout_min_status"] == "not_applicable"
    assert result["active_bout_gini_status"] == "not_applicable"


def test_below_minimum_wear_is_insufficient():

    result = activity_fragmentation(minutes([1] * 100 + [0] * 100))

    assert np.isnan(result["active_minutes"])
    assert result["active_minutes_status"] == "insufficient_observability"


def test_empty_input_is_no_raw_data():

    result = activity_fragmentation(minutes([]))

    assert result["astp_per_min_status"] == "no_raw_data"


def test_gini():

    assert gini_coefficient([5, 5, 5]) == pytest.approx(0.0)
    assert np.isnan(gini_coefficient([5]))
    assert gini_coefficient([0, 0, 10]) == pytest.approx(2 / 3)


def test_invalid_arguments():

    with pytest.raises(ValueError):
        activity_fragmentation(pd.DataFrame({"enmo_mg": [1.0]}))

    with pytest.raises(ValueError):
        activity_fragmentation(minutes([0]), active_threshold_mg=-1)
