
import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)
from cortex2_engine.features.gps_spatial_semantics import (
    SPATIAL_STATUS_ACCURACY_OVERLAP,
    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
)
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)


def test_empty_supported_evidence():
    edges = pd.DataFrame(
        {
            "supported": [False],
            "duration_ms": [60_000],
            "start_latitude": [42.0],
            "start_longitude": [-71.0],
            "end_latitude": [42.0],
            "end_longitude": [-71.0],
            "distance_m": [0.0],
            "spatial_status": ["temporal_gap"],
        }
    )

    edges = add_gps_trajectory_plausibility(
        edges
    )

    result = compute_daily_mobility_features(
        edges
    )

    # No supported time: distance is unknown, not a behavioural zero.
    assert np.isnan(
        result[
            "supported_displacement_distance_m"
        ]
    )

    assert (
        result[
            "supported_displacement_distance_m_status"
        ]
        == "insufficient_observability"
    )

    assert np.isnan(
        result[
            "implausible_transition_fraction"
        ]
    )

    assert (
        result[
            "displacement_edge_count"
        ]
        == 0
    )

    assert np.isnan(
        result[
            "supported_radius_of_gyration_m"
        ]
    )


def test_distance_uses_only_displacement_evidence():
    edges = pd.DataFrame(
        {
            "supported":
                [True, True, True],

            "duration_ms":
                [60_000, 60_000, 60_000],

            "start_latitude":
                [42.0, 42.0, 42.001],

            "start_longitude":
                [-71.0, -71.0, -71.001],

            "end_latitude":
                [42.0, 42.001, 42.002],

            "end_longitude":
                [-71.0, -71.001, -71.002],

            "distance_m":
                [20.0, 150.0, 200.0],

            "spatial_status":
                [
                    SPATIAL_STATUS_ACCURACY_OVERLAP,
                    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
                    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
                ],
        }
    )

    edges = add_gps_trajectory_plausibility(
        edges
    )

    result = compute_daily_mobility_features(
        edges
    )

    assert (
        result[
            "supported_displacement_distance_m"
        ]
        == pytest.approx(350.0)
    )

    assert (
        result[
            "displacement_edge_count"
        ]
        == 2
    )


def test_radius_zero_when_location_does_not_change():
    edges = pd.DataFrame(
        {
            "supported": [True, True],
            "duration_ms": [60_000, 120_000],
            "start_latitude": [42.0, 42.0],
            "start_longitude": [-71.0, -71.0],
            "end_latitude": [42.0, 42.0],
            "end_longitude": [-71.0, -71.0],
            "distance_m": [0.0, 0.0],
            "spatial_status": [
                SPATIAL_STATUS_ACCURACY_OVERLAP,
                SPATIAL_STATUS_ACCURACY_OVERLAP,
            ],
        }
    )

    edges = add_gps_trajectory_plausibility(
        edges
    )

    result = compute_daily_mobility_features(
        edges
    )

    assert (
        result[
            "supported_radius_of_gyration_m"
        ]
        == pytest.approx(0.0)
    )


def test_radius_positive_when_locations_are_spread():
    edges = pd.DataFrame(
        {
            "supported": [True, True],
            "duration_ms": [60_000, 60_000],
            "start_latitude": [42.0, 42.01],
            "start_longitude": [-71.0, -71.01],
            "end_latitude": [42.0, 42.01],
            "end_longitude": [-71.0, -71.01],
            "distance_m": [0.0, 0.0],
            "spatial_status": [
                SPATIAL_STATUS_ACCURACY_OVERLAP,
                SPATIAL_STATUS_ACCURACY_OVERLAP,
            ],
        }
    )

    edges = add_gps_trajectory_plausibility(
        edges
    )

    result = compute_daily_mobility_features(
        edges
    )

    assert (
        result[
            "supported_radius_of_gyration_m"
        ]
        > 0
    )


def test_missing_columns_raise():
    with pytest.raises(
        ValueError,
        match="Missing required GPS columns",
    ):
        compute_daily_mobility_features(
            pd.DataFrame(
                {
                    "supported": [True],
                }
            )
        )


def _stationary_edges(duration_ms):
    edges = pd.DataFrame(
        {
            "supported": [True],
            "duration_ms": [duration_ms],
            "start_latitude": [42.0],
            "start_longitude": [-71.0],
            "end_latitude": [42.0],
            "end_longitude": [-71.0],
            "distance_m": [0.0],
            "spatial_status": [SPATIAL_STATUS_ACCURACY_OVERLAP],
        }
    )

    return add_gps_trajectory_plausibility(edges)


def test_observed_stationary_day_is_observed_zero():

    result = compute_daily_mobility_features(
        _stationary_edges(3_600_000)
    )

    assert result["supported_displacement_distance_m"] == 0.0
    assert (
        result["supported_displacement_distance_m_status"]
        == "observed_zero"
    )
    assert result["supported_radius_of_gyration_m_status"] == "observed_zero"

    # No displacement candidates: undefined, not zero.
    assert np.isnan(result["implausible_transition_fraction"])
    assert result["implausible_transition_fraction_status"] == "not_applicable"


def test_minimum_supported_duration_gates_values():

    result = compute_daily_mobility_features(
        _stationary_edges(3_600_000),
        minimum_supported_duration_ms=2 * 3_600_000,
    )

    assert np.isnan(result["supported_displacement_distance_m"])
    assert (
        result["supported_displacement_distance_m_status"]
        == "insufficient_observability"
    )
    assert result["spatially_supported_duration_ms"] == 3_600_000


def test_negative_minimum_raises():

    with pytest.raises(ValueError):
        compute_daily_mobility_features(
            _stationary_edges(60_000),
            minimum_supported_duration_ms=-1,
        )
