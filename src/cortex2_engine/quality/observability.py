import numpy as np
import pandas as pd


def occupied_bins(
    timestamps_ms,
    *,
    start_ms: int,
    end_ms: int,
    bin_seconds: int,
):
    """
    Return a Boolean array indicating which temporal bins
    contain >=1 observation.
    """

    bin_ms = int(bin_seconds * 1000)

    n_bins = int(
        np.ceil(
            (end_ms - start_ms)
            / bin_ms
        )
    )

    if n_bins <= 0:
        raise ValueError(
            "end_ms must be greater than start_ms"
        )

    occupied = np.zeros(
        n_bins,
        dtype=bool,
    )

    ts = pd.to_numeric(
        pd.Series(timestamps_ms),
        errors="coerce",
    ).dropna()

    if len(ts) == 0:
        return occupied

    ts = ts.to_numpy(
        dtype=np.int64
    )

    ts = ts[
        (ts >= start_ms)
        &
        (ts < end_ms)
    ]

    if len(ts) == 0:
        return occupied

    idx = (
        (ts - start_ms)
        // bin_ms
    ).astype(np.int64)

    idx = idx[
        (idx >= 0)
        &
        (idx < n_bins)
    ]

    occupied[
        np.unique(idx)
    ] = True

    return occupied


def longest_false_run(mask):
    """
    Longest consecutive run of False values.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    best = 0
    current = 0

    for value in mask:

        if value:
            current = 0

        else:
            current += 1
            best = max(
                best,
                current,
            )

    return int(best)


def count_false_runs(mask):
    """
    Number of separate missing runs.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if len(mask) == 0:
        return 0

    missing = ~mask

    starts = (
        missing
        &
        np.concatenate(
            [
                [True],
                ~missing[:-1],
            ]
        )
    )

    return int(
        starts.sum()
    )


def temporal_quality_metrics(
    timestamps_ms,
    *,
    start_ms: int,
    end_ms: int,
    bin_seconds: int,
):
    """
    Compute standardized Cortex 2.0 temporal observability metrics.
    """

    occupied = occupied_bins(
        timestamps_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        bin_seconds=bin_seconds,
    )

    n_bins = len(
        occupied
    )

    n_occupied = int(
        occupied.sum()
    )

    quality = (
        n_occupied / n_bins
        if n_bins
        else np.nan
    )

    longest_missing_bins = (
        longest_false_run(
            occupied
        )
    )

    return {
        "data_quality":
            quality,

        "n_bins":
            n_bins,

        "n_occupied_bins":
            n_occupied,

        "n_missing_bins":
            int(
                n_bins - n_occupied
            ),

        "missing_run_count":
            count_false_runs(
                occupied
            ),

        "longest_missing_bins":
            longest_missing_bins,

        "longest_missing_seconds":
            (
                longest_missing_bins
                * bin_seconds
            ),
    }
