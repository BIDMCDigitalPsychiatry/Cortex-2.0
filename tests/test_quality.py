import numpy as np

from cortex2_engine.quality.binning import temporal_bin_coverage


def test_empty_window_returns_zero():
    result = temporal_bin_coverage(
        timestamps_ms=np.array([], dtype=np.int64),
        start_ms=0,
        end_ms=86_400_000,
        bin_seconds=600,
    )

    assert result == 0.0


def test_single_gps_bin():
    result = temporal_bin_coverage(
        timestamps_ms=np.array([1000, 2000, 3000]),
        start_ms=0,
        end_ms=86_400_000,
        bin_seconds=600,
    )

    assert np.isclose(
        result,
        1 / 144,
    )


def test_duplicate_points_do_not_inflate_quality():
    result = temporal_bin_coverage(
        timestamps_ms=np.array(
            [1000] * 1000
        ),
        start_ms=0,
        end_ms=86_400_000,
        bin_seconds=600,
    )

    assert np.isclose(
        result,
        1 / 144,
    )
