"""Home/away entropy identity and missingness semantics."""
import numpy as np
import pandas as pd

from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
    compute_home_away_location_entropy,
)


def a(values):
    return pd.DataFrame({
        "place_candidate_id": list(values),
        "duration_ms": list(values.values()),
    })


def test_identity_and_effective_places():
    x = a({1: 6, 2: 2, 3: 2})
    result = compute_home_away_location_entropy(
        x, home_place_candidate_id=1, home_status="observed_or_reconstructed"
    )
    total = compute_daily_location_features(x)
    assert np.isclose(result["away_location_entropy_nats"], np.log(2))
    assert np.isclose(result["away_effective_number_places"], 2)
    assert np.isclose(result["home_stationary_fraction"], 0.6)
    assert np.isclose(result["entropy_decomposition_residual"], 0, atol=1e-12)
    assert np.isclose(
        total["location_entropy_nats"],
        result["home_away_binary_entropy_nats"]
        + (1-result["home_stationary_fraction"])*result["away_location_entropy_nats"],
    )


def test_missing_home_does_not_claim_zero_away_places():
    result = compute_home_away_location_entropy(a({1: 1, 2: 2}), home_place_candidate_id=None)
    assert result["away_location_entropy_status"] == "unknown_home"
    assert np.isnan(result["away_location_entropy_nats"])
    assert np.isnan(result["away_place_count"])


def test_only_home_away_undefined():
    result = compute_home_away_location_entropy(
        a({1: 5}), home_place_candidate_id=1, home_status="observed_or_reconstructed"
    )
    assert result["away_location_entropy_status"] == "no_observed_away_time"
    assert np.isnan(result["away_location_entropy_nats"])
    assert result["away_place_count"] == 0


def test_absent_home_during_day_gives_zero_home_fraction():
    result = compute_home_away_location_entropy(
        a({2: 2, 3: 2}), home_place_candidate_id=1, home_status="observed_or_reconstructed"
    )
    assert result["home_stationary_fraction"] == 0
    assert np.isclose(result["away_location_entropy_nats"], np.log(2))
    assert np.isclose(result["entropy_decomposition_residual"], 0, atol=1e-12)


def test_weekly_is_pooled_occupancy_not_mean_daily_entropy():
    d1 = compute_daily_location_features(a({1: 6, 2: 4}))
    d2 = compute_daily_location_features(a({1: 6, 3: 4}))
    weekly = compute_daily_location_features(a({1: 12, 2: 4, 3: 4}))
    assert weekly["location_entropy_nats"] > (
        d1["location_entropy_nats"] + d2["location_entropy_nats"]
    ) / 2
