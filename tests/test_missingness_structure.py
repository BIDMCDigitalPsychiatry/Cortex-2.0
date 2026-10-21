import numpy as np
import pytest

from raphe.features.missingness_structure import (
    compute_missingness_structure,
)


MINUTE = 60_000


def test_no_missingness():

    x = compute_missingness_structure(
        np.array(
            [
                False,
                False,
                False,
                False,
            ]
        ),
        bin_size_ms=5 * MINUTE,
    )

    assert x["coverage"] == 1.0
    assert x["n_missing_bins"] == 0
    assert x["missing_run_count"] == 0
    assert x["longest_missing_bins"] == 0
    assert x["missing_fragmentation"] == 0.0


def test_all_missing_is_one_run():

    x = compute_missingness_structure(
        np.array(
            [
                True,
                True,
                True,
                True,
            ]
        ),
        bin_size_ms=5 * MINUTE,
    )

    assert x["coverage"] == 0.0
    assert x["missing_run_count"] == 1
    assert x["longest_missing_bins"] == 4
    assert x["longest_missing_ms"] == 20 * MINUTE
    assert (
        x[
            "longest_missing_fraction_of_missing"
        ]
        == 1.0
    )
    assert (
        x[
            "missing_run_concentration"
        ]
        == 1.0
    )


def test_distributed_single_bin_runs():

    x = compute_missingness_structure(
        np.array(
            [
                True,
                False,
                True,
                False,
                True,
                False,
            ]
        ),
        bin_size_ms=5 * MINUTE,
    )

    assert x["n_missing_bins"] == 3
    assert x["missing_run_count"] == 3
    assert x["longest_missing_bins"] == 1
    assert x["mean_missing_run_bins"] == 1.0
    assert x["missing_fragmentation"] == 1.0

    assert x[
        "missing_run_concentration"
    ] == pytest.approx(
        1 / 3
    )


def test_same_missing_fraction_different_structure():

    contiguous = compute_missingness_structure(
        np.array(
            [
                True,
                True,
                True,
                False,
                False,
                False,
            ]
        ),
        bin_size_ms=5 * MINUTE,
    )

    distributed = compute_missingness_structure(
        np.array(
            [
                True,
                False,
                True,
                False,
                True,
                False,
            ]
        ),
        bin_size_ms=5 * MINUTE,
    )

    assert (
        contiguous[
            "missing_fraction"
        ]
        == distributed[
            "missing_fraction"
        ]
    )

    assert (
        contiguous[
            "longest_missing_bins"
        ]
        > distributed[
            "longest_missing_bins"
        ]
    )

    assert (
        contiguous[
            "missing_run_concentration"
        ]
        > distributed[
            "missing_run_concentration"
        ]
    )

    assert (
        contiguous[
            "missing_fragmentation"
        ]
        < distributed[
            "missing_fragmentation"
        ]
    )


def test_two_runs():

    x = compute_missingness_structure(
        np.array(
            [
                False,
                True,
                True,
                False,
                True,
                False,
            ]
        ),
        bin_size_ms=5 * MINUTE,
    )

    assert x["missing_run_count"] == 2
    assert x["longest_missing_bins"] == 2
    assert x["mean_missing_run_bins"] == pytest.approx(
        1.5
    )
    assert x["median_missing_run_bins"] == pytest.approx(
        1.5
    )


def test_invalid_empty_mask():

    with pytest.raises(
        ValueError
    ):
        compute_missingness_structure(
            np.array(
                [],
                dtype=bool,
            ),
            bin_size_ms=5 * MINUTE,
        )


def test_invalid_bin_size():

    with pytest.raises(
        ValueError
    ):
        compute_missingness_structure(
            np.array(
                [
                    True,
                    False,
                ]
            ),
            bin_size_ms=0,
        )
