"""Controlled temporal-missingness simulations for Cortex 2.0.

This module creates missingness geometries over equal-width temporal bins.

It is intentionally separate from the original frozen GPS degradation
implementation so that the held-out validation experiment remains unchanged.

The balanced k-run design fixes:
- total missing time
- exact number of missing runs

while randomizing their temporal placement.

Missing-run lengths are as balanced as integer binning permits.
"""

from __future__ import annotations

import math

import numpy as np


def make_balanced_k_run_missing_bin_mask(
    *,
    window_start_ms: int,
    window_end_ms: int,
    target_coverage: float,
    n_runs: int,
    bin_size_ms: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Create a deterministic mask containing exactly ``n_runs`` missing runs."""

    window_start_ms = int(
        window_start_ms
    )

    window_end_ms = int(
        window_end_ms
    )

    bin_size_ms = int(
        bin_size_ms
    )

    n_runs = int(
        n_runs
    )


    if window_end_ms <= window_start_ms:
        raise ValueError(
            "window_end_ms must be greater than window_start_ms"
        )


    if (
        not math.isfinite(
            target_coverage
        )
        or target_coverage <= 0
        or target_coverage >= 1
    ):
        raise ValueError(
            "target_coverage must be in (0, 1)"
        )


    if bin_size_ms <= 0:
        raise ValueError(
            "bin_size_ms must be > 0"
        )


    if n_runs <= 0:
        raise ValueError(
            "n_runs must be > 0"
        )


    duration_ms = (
        window_end_ms
        - window_start_ms
    )


    n_bins = int(
        math.ceil(
            duration_ms
            / bin_size_ms
        )
    )


    n_missing = int(
        round(
            (1.0 - target_coverage)
            * n_bins
        )
    )


    n_missing = min(
        max(
            n_missing,
            0,
        ),
        n_bins,
    )


    n_observed = (
        n_bins
        - n_missing
    )


    if n_missing == 0:
        raise ValueError(
            "Requested coverage produces zero missing bins."
        )


    # To create k distinct missing runs:
    # - every run needs >=1 missing bin
    # - adjacent missing runs need >=1 observed bin between them
    max_runs = min(
        n_missing,
        n_observed + 1,
    )


    if n_runs > max_runs:
        raise ValueError(
            f"Cannot create {n_runs} distinct missing runs "
            f"with {n_missing} missing and "
            f"{n_observed} observed bins."
        )


    rng = np.random.default_rng(
        int(
            seed
        )
    )


    # --------------------------------------------------------
    # Missing-run lengths
    #
    # Keep them as balanced as possible. If there is a remainder,
    # randomly choose which runs receive one extra missing bin.
    # --------------------------------------------------------

    base_missing = (
        n_missing
        // n_runs
    )

    missing_remainder = (
        n_missing
        % n_runs
    )


    run_lengths = np.full(
        n_runs,
        base_missing,
        dtype=np.int64,
    )


    if missing_remainder:

        extra_runs = rng.choice(
            n_runs,
            size=missing_remainder,
            replace=False,
        )

        run_lengths[
            extra_runs
        ] += 1


    # --------------------------------------------------------
    # Observed gaps
    #
    # There are k+1 gaps:
    # before first run,
    # k-1 internal gaps,
    # after last run.
    #
    # Internal gaps receive at least one observed bin so that
    # the missing blocks remain distinct.
    # --------------------------------------------------------

    observed_gaps = np.zeros(
        n_runs + 1,
        dtype=np.int64,
    )


    if n_runs > 1:

        observed_gaps[
            1:n_runs
        ] = 1


    remaining_observed = (
        n_observed
        - (
            n_runs
            - 1
        )
    )


    if remaining_observed < 0:
        raise RuntimeError(
            "Internal error: negative remaining observed bins."
        )


    if remaining_observed:

        probabilities = np.full(
            n_runs + 1,
            1.0
            / (
                n_runs
                + 1
            ),
        )


        observed_gaps += (
            rng.multinomial(
                remaining_observed,
                probabilities,
            )
        )


    # --------------------------------------------------------
    # Assemble mask
    # --------------------------------------------------------

    mask = np.zeros(
        n_bins,
        dtype=bool,
    )


    cursor = 0


    for run_index in range(
        n_runs
    ):

        cursor += int(
            observed_gaps[
                run_index
            ]
        )


        run_length = int(
            run_lengths[
                run_index
            ]
        )


        mask[
            cursor:
            cursor + run_length
        ] = True


        cursor += run_length


    cursor += int(
        observed_gaps[
            -1
        ]
    )


    if cursor != n_bins:
        raise RuntimeError(
            f"Mask assembly ended at {cursor}, "
            f"expected {n_bins}."
        )


    return (
        mask,
        {
            "n_bins":
                n_bins,

            "n_missing_bins":
                int(
                    mask.sum()
                ),

            "n_observed_bins":
                int(
                    (
                        ~mask
                    ).sum()
                ),

            "planned_coverage":
                float(
                    1.0
                    - mask.mean()
                ),

            "requested_n_runs":
                n_runs,

            "run_lengths_bins":
                run_lengths.tolist(),

            "observed_gap_lengths_bins":
                observed_gaps.tolist(),

            "bin_size_ms":
                bin_size_ms,
        },
    )
