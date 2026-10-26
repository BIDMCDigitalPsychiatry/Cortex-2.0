import numpy as np
import pandas as pd

from cortex2_engine.quality.daily import (
    daily_stream_quality,
)


DAY = 86_400_000


def test_available_empty_gps_is_zero_quality():
    df = pd.DataFrame({
        "timestamp_ms": [],
    })

    r = daily_stream_quality(
        df,
        stream="gps",
        start_ms=0,
        end_ms=DAY,
        source_end_ms=2 * DAY,
    )

    assert r["source_unavailable"] is False
    assert r["partial_source"] is False
    assert r["raw_data_quality"] == 0.0
    assert r["data_quality"] == 0.0


def test_outside_source_boundary_is_na():
    df = pd.DataFrame({
        "timestamp_ms": [],
    })

    r = daily_stream_quality(
        df,
        stream="gps",
        start_ms=DAY,
        end_ms=2 * DAY,
        source_end_ms=DAY,
    )

    assert r["source_status"] == (
        "outside_declared_end"
    )
    assert r["source_unavailable"] is True
    assert r["raw_data_quality"] == 0.0
    assert np.isnan(
        r["data_quality"]
    )


def test_partial_source_day_is_na():
    df = pd.DataFrame({
        "timestamp_ms": [
            1_000,
        ],
    })

    r = daily_stream_quality(
        df,
        stream="gps",
        start_ms=0,
        end_ms=DAY,
        source_end_ms=DAY // 2,
    )

    assert r["source_unavailable"] is False
    assert r["partial_source"] is True
    assert (
        r["source_available_fraction"]
        == 0.5
    )
    assert np.isnan(
        r["data_quality"]
    )


def test_unknown_boundary_preserves_dq():
    df = pd.DataFrame({
        "timestamp_ms": [
            1_000,
        ],
    })

    r = daily_stream_quality(
        df,
        stream="gps",
        start_ms=0,
        end_ms=DAY,
    )

    assert r["boundary_unknown"] is True
    assert not np.isnan(
        r["data_quality"]
    )


def test_screen_does_not_invent_scalar_dq():
    df = pd.DataFrame({
        "timestamp_ms": [
            1_000,
        ],
    })

    r = daily_stream_quality(
        df,
        stream="screen",
        start_ms=0,
        end_ms=DAY,
        source_end_ms=2 * DAY,
    )

    assert "data_quality" not in r
    assert r["observed"] == 1
    assert r["raw_observed"] == 1
