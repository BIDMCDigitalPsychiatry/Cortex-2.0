import pandas as pd
import pytest

from raphe.features.gps_intervals import (
    build_gps_intervals,
    haversine_km,
)


MINUTE = 60_000


def test_haversine_zero_distance():
    assert (
        haversine_km(
            42.0,
            -71.0,
            42.0,
            -71.0,
        )
        == pytest.approx(
            0.0
        )
    )


def test_haversine_known_distance():
    # Roughly one degree latitude.
    distance = haversine_km(
        0.0,
        0.0,
        1.0,
        0.0,
    )

    assert distance == pytest.approx(
        111.2,
        rel=0.01,
    )


def test_regular_points_create_supported_edges():
    gps = pd.DataFrame({
        "timestamp": [
            0,
            MINUTE,
            2 * MINUTE,
        ],
        "latitude": [
            42.0,
            42.0,
            42.0,
        ],
        "longitude": [
            -71.0,
            -71.0,
            -71.0,
        ],
    })

    result = build_gps_intervals(
        gps,
        window_start_ms=0,
        window_end_ms=3 * MINUTE,
        max_gap_ms=MINUTE,
    )

    assert len(result) == 2
    assert result["supported"].all()
    assert (
        result["duration_ms"]
        == MINUTE
    ).all()


def test_long_gap_is_preserved_but_marked_unsupported():
    gps = pd.DataFrame({
        "timestamp": [
            0,
            MINUTE,
            10 * MINUTE,
        ],
        "latitude": [
            42.0,
            42.0,
            42.1,
        ],
        "longitude": [
            -71.0,
            -71.0,
            -71.1,
        ],
    })

    result = build_gps_intervals(
        gps,
        window_start_ms=0,
        window_end_ms=11 * MINUTE,
        max_gap_ms=2 * MINUTE,
    )

    assert len(result) == 2

    assert bool(
        result.iloc[0]["supported"]
    )

    assert not bool(
        result.iloc[1]["supported"]
    )

    assert (
        result.iloc[1]["gap_ms"]
        == 9 * MINUTE
    )


def test_duplicate_timestamp_does_not_create_zero_time_edge():
    gps = pd.DataFrame({
        "timestamp": [
            0,
            0,
            MINUTE,
        ],
        "latitude": [
            42.0,
            43.0,
            42.0,
        ],
        "longitude": [
            -71.0,
            -72.0,
            -71.0,
        ],
    })

    result = build_gps_intervals(
        gps,
        window_start_ms=0,
        window_end_ms=2 * MINUTE,
        max_gap_ms=MINUTE,
    )

    assert len(result) == 1

    assert (
        result.iloc[0]["duration_ms"]
        == MINUTE
    )


def test_end_boundary_is_excluded():
    gps = pd.DataFrame({
        "timestamp": [
            0,
            MINUTE,
            2 * MINUTE,
        ],
        "latitude": [
            42.0,
            42.0,
            42.0,
        ],
        "longitude": [
            -71.0,
            -71.0,
            -71.0,
        ],
    })

    result = build_gps_intervals(
        gps,
        window_start_ms=0,
        window_end_ms=2 * MINUTE,
        max_gap_ms=MINUTE,
    )

    assert len(result) == 1

    assert (
        result.iloc[0]["end_ms"]
        == MINUTE
    )


def test_speed_units_are_kmh():
    gps = pd.DataFrame({
        "timestamp": [
            0,
            3_600_000,
        ],
        "latitude": [
            0.0,
            1.0,
        ],
        "longitude": [
            0.0,
            0.0,
        ],
    })

    result = build_gps_intervals(
        gps,
        window_start_ms=0,
        window_end_ms=3_600_001,
        max_gap_ms=3_600_000,
    )

    assert result.iloc[0]["distance_km"] == pytest.approx(
        111.2,
        rel=0.01,
    )

    assert result.iloc[0]["speed_kmh"] == pytest.approx(
        111.2,
        rel=0.01,
    )


def test_empty_input_returns_empty_schema():
    gps = pd.DataFrame(
        columns=[
            "timestamp",
            "latitude",
            "longitude",
        ]
    )

    result = build_gps_intervals(
        gps,
        window_start_ms=0,
        window_end_ms=MINUTE,
        max_gap_ms=MINUTE,
    )

    assert result.empty

    assert {
        "supported",
        "distance_km",
        "speed_kmh",
    }.issubset(
        result.columns
    )


def test_missing_required_column_raises():
    gps = pd.DataFrame({
        "timestamp": [0],
        "latitude": [42.0],
    })

    with pytest.raises(ValueError):
        build_gps_intervals(
            gps,
            window_start_ms=0,
            window_end_ms=MINUTE,
            max_gap_ms=MINUTE,
        )
