import numpy as np
import pandas as pd
import pytest

from raphe.features.gps_degradation import (
    apply_missing_bin_mask_to_points,
    make_missing_bin_mask,
)


MINUTE = 60_000


def test_full_coverage_removes_nothing():

    mask, info = make_missing_bin_mask(
        window_start_ms=0,
        window_end_ms=60 * MINUTE,
        target_coverage=1.0,
        pattern="distributed",
        bin_size_ms=5 * MINUTE,
        seed=1,
    )

    assert not mask.any()

    assert info[
        "planned_coverage"
    ] == pytest.approx(
        1.0
    )


def test_same_coverage_same_number_missing():

    distributed, _ = make_missing_bin_mask(
        window_start_ms=0,
        window_end_ms=60 * MINUTE,
        target_coverage=0.5,
        pattern="distributed",
        bin_size_ms=5 * MINUTE,
        seed=1,
    )

    contiguous, _ = make_missing_bin_mask(
        window_start_ms=0,
        window_end_ms=60 * MINUTE,
        target_coverage=0.5,
        pattern="contiguous",
        bin_size_ms=5 * MINUTE,
        seed=1,
    )

    assert distributed.sum() == 6
    assert contiguous.sum() == 6


def test_contiguous_mask_is_single_block():

    mask, _ = make_missing_bin_mask(
        window_start_ms=0,
        window_end_ms=60 * MINUTE,
        target_coverage=0.5,
        pattern="contiguous",
        bin_size_ms=5 * MINUTE,
        seed=5,
    )

    positions = np.flatnonzero(
        mask
    )

    assert np.all(
        np.diff(
            positions
        )
        == 1
    )


def test_point_inside_missing_bin_removed():

    points = pd.DataFrame({
        "timestamp": [
            1 * MINUTE,
            6 * MINUTE,
            11 * MINUTE,
        ],
        "latitude": [
            0.0,
            0.0,
            0.0,
        ],
        "longitude": [
            0.0,
            0.0,
            0.0,
        ],
    })

    mask = np.array([
        False,
        True,
        False,
    ])

    result = apply_missing_bin_mask_to_points(
        points,
        window_start_ms=0,
        window_end_ms=15 * MINUTE,
        bin_size_ms=5 * MINUTE,
        missing_mask=mask,
    )

    assert result[
        "timestamp"
    ].tolist() == [
        1 * MINUTE,
        11 * MINUTE,
    ]


def test_duplicate_points_in_missing_bin_all_removed():

    points = pd.DataFrame({
        "timestamp": [
            6 * MINUTE,
            6 * MINUTE,
        ]
    })

    mask = np.array([
        False,
        True,
    ])

    result = apply_missing_bin_mask_to_points(
        points,
        window_start_ms=0,
        window_end_ms=10 * MINUTE,
        bin_size_ms=5 * MINUTE,
        missing_mask=mask,
    )

    assert result.empty


def test_distributed_mask_is_reproducible():

    first, _ = make_missing_bin_mask(
        window_start_ms=0,
        window_end_ms=60 * MINUTE,
        target_coverage=0.75,
        pattern="distributed",
        bin_size_ms=5 * MINUTE,
        seed=123,
    )

    second, _ = make_missing_bin_mask(
        window_start_ms=0,
        window_end_ms=60 * MINUTE,
        target_coverage=0.75,
        pattern="distributed",
        bin_size_ms=5 * MINUTE,
        seed=123,
    )

    assert np.array_equal(
        first,
        second,
    )


def test_invalid_pattern_raises():

    with pytest.raises(
        ValueError
    ):

        make_missing_bin_mask(
            window_start_ms=0,
            window_end_ms=60 * MINUTE,
            target_coverage=0.5,
            pattern="bad",
            bin_size_ms=5 * MINUTE,
            seed=1,
        )
