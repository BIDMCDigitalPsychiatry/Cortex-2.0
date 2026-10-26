import numpy as np
import pandas as pd

from cortex2_engine.quality.observability import (
    temporal_quality_metrics,
)

from cortex2_engine.quality.streams import (
    gps_quality,
    screen_quality,
)


DAY_MS = 86_400_000


def test_half_day_coverage():

    # GPS: one observation in each of first 72
    # ten-minute bins = exactly 50% coverage.
    timestamps = np.arange(
        72
    ) * 600_000

    m = temporal_quality_metrics(
        timestamps,
        start_ms=0,
        end_ms=DAY_MS,
        bin_seconds=600,
    )

    assert m["n_bins"] == 144
    assert m["n_occupied_bins"] == 72
    assert np.isclose(
        m["data_quality"],
        0.5,
    )

    assert (
        m["longest_missing_bins"]
        == 72
    )

    assert (
        m["longest_missing_seconds"]
        == 43200
    )


def test_same_quality_different_missingness():

    # Pattern A: first half observed,
    # second half missing.
    a = np.arange(
        72
    ) * 600_000

    # Pattern B: alternating occupied bins.
    b = np.arange(
        0,
        144,
        2,
    ) * 600_000

    ma = temporal_quality_metrics(
        a,
        start_ms=0,
        end_ms=DAY_MS,
        bin_seconds=600,
    )

    mb = temporal_quality_metrics(
        b,
        start_ms=0,
        end_ms=DAY_MS,
        bin_seconds=600,
    )

    assert np.isclose(
        ma["data_quality"],
        mb["data_quality"],
    )

    assert (
        ma["longest_missing_bins"]
        >
        mb["longest_missing_bins"]
    )


def test_gps_accuracy():

    df = pd.DataFrame({
        "timestamp_ms": [
            1000,
            601000,
            1201000,
        ],
        "accuracy_m": [
            5.0,
            10.0,
            20.0,
        ],
    })

    m = gps_quality(
        df,
        start_ms=0,
        end_ms=DAY_MS,
    )

    assert m["n_observations"] == 3
    assert m["accuracy_m_median"] == 10.0


def test_screen_no_data():

    df = pd.DataFrame({
        "timestamp_ms": [],
    })

    m = screen_quality(
        df,
        start_ms=0,
        end_ms=DAY_MS,
    )

    assert m["observed"] == 0
    assert m["n_events"] == 0
