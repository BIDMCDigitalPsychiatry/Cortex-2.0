import pandas as pd
import pytest

from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
    TRAJECTORY_STATUS_NOT_DISPLACEMENT,
    TRAJECTORY_STATUS_PLAUSIBLE_DISPLACEMENT,
    TRAJECTORY_STATUS_IMPLAUSIBLE_SPEED,
)

from cortex2_engine.features.gps_spatial_semantics import (
    SPATIAL_STATUS_ACCURACY_OVERLAP,
    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
)


def test_plausible_displacement():
    x = pd.DataFrame(
        {
            "supported": [True],
            "duration_ms": [10_000],
            "distance_m": [1000.0],
            "spatial_status": [
                SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
            ],
        }
    )

    result = add_gps_trajectory_plausibility(x)

    assert (
        result.loc[
            0,
            "derived_speed_kmh",
        ]
        == pytest.approx(360.0)
    )

    assert (
        result.loc[
            0,
            "trajectory_status",
        ]
        == TRAJECTORY_STATUS_PLAUSIBLE_DISPLACEMENT
    )


def test_implausible_near_instantaneous_jump():
    x = pd.DataFrame(
        {
            "supported": [True],
            "duration_ms": [1],
            "distance_m": [1_500_000.0],
            "spatial_status": [
                SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
            ],
        }
    )

    result = add_gps_trajectory_plausibility(x)

    assert (
        result.loc[
            0,
            "trajectory_status",
        ]
        == TRAJECTORY_STATUS_IMPLAUSIBLE_SPEED
    )


def test_accuracy_overlap_is_not_movement():
    x = pd.DataFrame(
        {
            "supported": [True],
            "duration_ms": [1000],
            "distance_m": [100.0],
            "spatial_status": [
                SPATIAL_STATUS_ACCURACY_OVERLAP
            ],
        }
    )

    result = add_gps_trajectory_plausibility(x)

    assert (
        result.loc[
            0,
            "trajectory_status",
        ]
        == TRAJECTORY_STATUS_NOT_DISPLACEMENT
    )


def test_custom_threshold():
    x = pd.DataFrame(
        {
            "supported": [True],
            "duration_ms": [10_000],
            "distance_m": [1000.0],
            "spatial_status": [
                SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
            ],
        }
    )

    result = add_gps_trajectory_plausibility(
        x,
        max_plausible_speed_kmh=300.0,
    )

    assert (
        result.loc[
            0,
            "trajectory_status",
        ]
        == TRAJECTORY_STATUS_IMPLAUSIBLE_SPEED
    )
