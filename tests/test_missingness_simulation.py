import numpy as np
import pytest

from raphe.features.missingness_simulation import (
    make_balanced_k_run_missing_bin_mask,
)

from raphe.features.missingness_structure import (
    compute_missingness_structure,
)


MINUTE = 60_000
DAY = 24 * 60 * MINUTE


@pytest.mark.parametrize(
    "n_runs",
    [
        1,
        2,
        4,
        8,
        16,
    ],
)
def test_exact_number_of_runs(
    n_runs,
):

    mask, info = (
        make_balanced_k_run_missing_bin_mask(
            window_start_ms=0,
            window_end_ms=DAY,
            target_coverage=0.50,
            n_runs=n_runs,
            bin_size_ms=5 * MINUTE,
            seed=123,
        )
    )


    structure = (
        compute_missingness_structure(
            mask,
            bin_size_ms=5 * MINUTE,
        )
    )


    assert len(mask) == 288

    assert (
        structure[
            "missing_run_count"
        ]
        == n_runs
    )

    assert (
        info[
            "requested_n_runs"
        ]
        == n_runs
    )

    assert (
        structure[
            "n_missing_bins"
        ]
        == 144
    )


def test_deterministic():

    kwargs = dict(
        window_start_ms=0,
        window_end_ms=DAY,
        target_coverage=0.75,
        n_runs=4,
        bin_size_ms=5 * MINUTE,
        seed=9876,
    )


    a, _ = (
        make_balanced_k_run_missing_bin_mask(
            **kwargs
        )
    )

    b, _ = (
        make_balanced_k_run_missing_bin_mask(
            **kwargs
        )
    )


    assert np.array_equal(
        a,
        b,
    )


def test_balanced_run_lengths():

    mask, _ = (
        make_balanced_k_run_missing_bin_mask(
            window_start_ms=0,
            window_end_ms=DAY,
            target_coverage=0.25,
            n_runs=8,
            bin_size_ms=5 * MINUTE,
            seed=1,
        )
    )


    padded = np.concatenate(
        [
            [False],
            mask,
            [False],
        ]
    ).astype(
        np.int8
    )


    transitions = np.diff(
        padded
    )


    starts = np.flatnonzero(
        transitions == 1
    )

    ends = np.flatnonzero(
        transitions == -1
    )


    lengths = (
        ends
        - starts
    )


    assert len(
        lengths
    ) == 8

    assert (
        lengths.max()
        - lengths.min()
        <= 1
    )


def test_coverage_matches_rounding_policy():

    mask, info = (
        make_balanced_k_run_missing_bin_mask(
            window_start_ms=0,
            window_end_ms=DAY,
            target_coverage=0.90,
            n_runs=4,
            bin_size_ms=5 * MINUTE,
            seed=10,
        )
    )


    # Same rounding policy as the frozen validation:
    # round(0.10 * 288) = 29 missing bins.
    assert mask.sum() == 29

    assert info[
        "planned_coverage"
    ] == pytest.approx(
        259 / 288
    )


def test_invalid_too_many_runs():

    with pytest.raises(
        ValueError
    ):

        make_balanced_k_run_missing_bin_mask(
            window_start_ms=0,
            window_end_ms=DAY,
            target_coverage=0.90,
            n_runs=100,
            bin_size_ms=5 * MINUTE,
            seed=1,
        )
