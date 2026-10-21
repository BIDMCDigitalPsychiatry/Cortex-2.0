"""Temporal missingness-structure descriptors for RAPHE.

These descriptors separate *how much* time is missing from *how that
missingness is arranged in time*.

The input is a Boolean sequence over equal-width temporal bins:

    False = observable / retained
    True  = missing

The bin width is an explicit representation parameter. These descriptors
do not themselves define whether a feature is reliable.
"""

from __future__ import annotations

import numpy as np


def compute_missingness_structure(
    missing_mask,
    *,
    bin_size_ms: int,
) -> dict:
    """Summarize the temporal structure of a missing-bin mask."""

    mask = np.asarray(
        missing_mask,
        dtype=bool,
    )

    if mask.ndim != 1:
        raise ValueError(
            "missing_mask must be one-dimensional"
        )

    if len(mask) == 0:
        raise ValueError(
            "missing_mask must not be empty"
        )

    if int(bin_size_ms) <= 0:
        raise ValueError(
            "bin_size_ms must be > 0"
        )


    n_bins = int(
        len(mask)
    )

    n_missing_bins = int(
        mask.sum()
    )

    n_observed_bins = int(
        n_bins
        - n_missing_bins
    )


    coverage = (
        n_observed_bins
        / n_bins
    )

    missing_fraction = (
        n_missing_bins
        / n_bins
    )


    # --------------------------------------------------------
    # Missing runs
    # --------------------------------------------------------

    padded = np.concatenate(
        [
            np.array(
                [False],
                dtype=bool,
            ),
            mask,
            np.array(
                [False],
                dtype=bool,
            ),
        ]
    )

    transitions = np.diff(
        padded.astype(
            np.int8
        )
    )

    run_starts = np.flatnonzero(
        transitions == 1
    )

    run_ends = np.flatnonzero(
        transitions == -1
    )

    run_lengths_bins = (
        run_ends
        - run_starts
    ).astype(
        np.int64
    )


    missing_run_count = int(
        len(
            run_lengths_bins
        )
    )


    if missing_run_count:

        longest_missing_bins = int(
            run_lengths_bins.max()
        )

        mean_missing_run_bins = float(
            run_lengths_bins.mean()
        )

        median_missing_run_bins = float(
            np.median(
                run_lengths_bins
            )
        )

        q90_missing_run_bins = float(
            np.quantile(
                run_lengths_bins,
                0.90,
            )
        )


        longest_missing_fraction_of_missing = (
            longest_missing_bins
            / n_missing_bins
            if n_missing_bins
            else 0.0
        )


        # High when missingness is concentrated in a few large runs;
        # low when distributed across many smaller runs.
        run_proportions = (
            run_lengths_bins
            / n_missing_bins
        )

        missing_run_concentration = float(
            np.sum(
                np.square(
                    run_proportions
                )
            )
        )


        # High when many separate runs exist relative to the total
        # amount of missing time.
        missing_fragmentation = float(
            missing_run_count
            / n_missing_bins
        )

    else:

        longest_missing_bins = 0
        mean_missing_run_bins = 0.0
        median_missing_run_bins = 0.0
        q90_missing_run_bins = 0.0
        longest_missing_fraction_of_missing = 0.0
        missing_run_concentration = 0.0
        missing_fragmentation = 0.0


    return {
        "n_bins":
            n_bins,

        "n_observed_bins":
            n_observed_bins,

        "n_missing_bins":
            n_missing_bins,

        "coverage":
            float(
                coverage
            ),

        "missing_fraction":
            float(
                missing_fraction
            ),

        "missing_run_count":
            missing_run_count,

        "longest_missing_bins":
            longest_missing_bins,

        "longest_missing_ms":
            int(
                longest_missing_bins
                * int(
                    bin_size_ms
                )
            ),

        "mean_missing_run_bins":
            mean_missing_run_bins,

        "mean_missing_run_ms":
            float(
                mean_missing_run_bins
                * int(
                    bin_size_ms
                )
            ),

        "median_missing_run_bins":
            median_missing_run_bins,

        "median_missing_run_ms":
            float(
                median_missing_run_bins
                * int(
                    bin_size_ms
                )
            ),

        "q90_missing_run_bins":
            q90_missing_run_bins,

        "q90_missing_run_ms":
            float(
                q90_missing_run_bins
                * int(
                    bin_size_ms
                )
            ),

        "longest_missing_fraction_of_missing":
            float(
                longest_missing_fraction_of_missing
            ),

        "missing_run_concentration":
            float(
                missing_run_concentration
            ),

        "missing_fragmentation":
            float(
                missing_fragmentation
            ),
    }
