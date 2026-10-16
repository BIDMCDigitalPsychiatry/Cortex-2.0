import numpy as np
import pandas as pd
import pytest

from raphe.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
    SPATIAL_STATUS_TEMPORAL_GAP,
    SPATIAL_STATUS_ACCURACY_UNAVAILABLE,
    SPATIAL_STATUS_ACCURACY_OVERLAP,
    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
)


def _edge(
    *,
    supported=True,
    distance_km=0.010,
    start_accuracy=10.0,
    end_accuracy=10.0,
):
    return pd.DataFrame({
        "supported": [
            supported
        ],
        "distance_km": [
            distance_km
        ],
        "start_accuracy": [
            start_accuracy
        ],
        "end_accuracy": [
            end_accuracy
        ],
    })


def test_accuracy_overlap():
    edges = _edge(
        distance_km=0.010,
        start_accuracy=10,
        end_accuracy=10,
    )

    result = add_gps_spatial_semantics(
        edges
    )

    row = result.iloc[0]

    assert row["distance_m"] == pytest.approx(
        10.0
    )

    assert row["combined_accuracy_m"] == pytest.approx(
        20.0
    )

    assert row["displacement_accuracy_ratio"] == pytest.approx(
        0.5
    )

    assert row["displacement_excess_m"] == pytest.approx(
        0.0
    )

    assert bool(
        row["accuracy_regions_overlap"]
    )

    assert (
        row["spatial_status"]
        == SPATIAL_STATUS_ACCURACY_OVERLAP
    )


def test_displacement_exceeds_accuracy():
    edges = _edge(
        distance_km=0.050,
        start_accuracy=10,
        end_accuracy=10,
    )

    result = add_gps_spatial_semantics(
        edges
    )

    row = result.iloc[0]

    assert row["distance_m"] == pytest.approx(
        50.0
    )

    assert row["combined_accuracy_m"] == pytest.approx(
        20.0
    )

    assert row["displacement_accuracy_ratio"] == pytest.approx(
        2.5
    )

    assert row["displacement_excess_m"] == pytest.approx(
        30.0
    )

    assert not bool(
        row["accuracy_regions_overlap"]
    )

    assert (
        row["spatial_status"]
        ==
        SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
    )


def test_exact_boundary_counts_as_overlap():
    edges = _edge(
        distance_km=0.020,
        start_accuracy=10,
        end_accuracy=10,
    )

    result = add_gps_spatial_semantics(
        edges
    )

    row = result.iloc[0]

    assert row["displacement_accuracy_ratio"] == pytest.approx(
        1.0
    )

    assert bool(
        row["accuracy_regions_overlap"]
    )

    assert (
        row["spatial_status"]
        == SPATIAL_STATUS_ACCURACY_OVERLAP
    )


def test_temporal_gap_takes_priority():
    edges = _edge(
        supported=False,
        distance_km=1.0,
        start_accuracy=5,
        end_accuracy=5,
    )

    result = add_gps_spatial_semantics(
        edges
    )

    row = result.iloc[0]

    assert (
        row["spatial_status"]
        == SPATIAL_STATUS_TEMPORAL_GAP
    )


def test_missing_accuracy_is_unavailable():
    edges = _edge(
        start_accuracy=np.nan,
        end_accuracy=10,
    )

    result = add_gps_spatial_semantics(
        edges
    )

    row = result.iloc[0]

    assert pd.isna(
        row["combined_accuracy_m"]
    )

    assert pd.isna(
        row["displacement_accuracy_ratio"]
    )

    assert pd.isna(
        row["accuracy_regions_overlap"]
    )

    assert (
        row["spatial_status"]
        == SPATIAL_STATUS_ACCURACY_UNAVAILABLE
    )


def test_nonpositive_accuracy_is_unavailable():
    edges = _edge(
        start_accuracy=0,
        end_accuracy=10,
    )

    result = add_gps_spatial_semantics(
        edges
    )

    assert (
        result.iloc[0][
            "spatial_status"
        ]
        == SPATIAL_STATUS_ACCURACY_UNAVAILABLE
    )


def test_multiple_edge_statuses():
    edges = pd.DataFrame({
        "supported": [
            True,
            True,
            False,
        ],
        "distance_km": [
            0.005,
            0.100,
            1.000,
        ],
        "start_accuracy": [
            10,
            10,
            10,
        ],
        "end_accuracy": [
            10,
            10,
            10,
        ],
    })

    result = add_gps_spatial_semantics(
        edges
    )

    assert result[
        "spatial_status"
    ].tolist() == [
        SPATIAL_STATUS_ACCURACY_OVERLAP,
        SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
        SPATIAL_STATUS_TEMPORAL_GAP,
    ]


def test_missing_required_column_raises():
    edges = pd.DataFrame({
        "supported": [True],
        "distance_km": [0.0],
    })

    with pytest.raises(ValueError):
        add_gps_spatial_semantics(
            edges
        )
