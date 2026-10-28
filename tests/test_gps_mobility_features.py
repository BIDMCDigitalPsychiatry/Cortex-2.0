import math

import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)

from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)

from cortex2_engine.features.gps_spatial_semantics import (
    SPATIAL_STATUS_ACCURACY_OVERLAP,
    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
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

    assert (
        result[
            "supported_displacement_distance_m"
        ]
        == 0.0
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
