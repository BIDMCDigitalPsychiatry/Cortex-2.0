import numpy as np


DAY_MS = 86_400_000


def temporal_bin_coverage(
    timestamps_ms,
    start_ms,
    end_ms,
    bin_seconds,
):
    """
    Fraction of fixed-width temporal bins containing >=1 observation.

    Parameters
    ----------
    timestamps_ms
        Sensor timestamps in Unix milliseconds.
    start_ms
        Inclusive window start in Unix milliseconds.
    end_ms
        Exclusive window end in Unix milliseconds.
    bin_seconds
        Width of temporal bins in seconds.

    Returns
    -------
    float
        Proportion of temporal bins containing at least one observation.
    """

    bin_ms = int(bin_seconds * 1000)

    if end_ms <= start_ms:
        raise ValueError(
            "end_ms must be greater than start_ms"
        )

    n_bins = int(
        np.ceil(
            (end_ms - start_ms) / bin_ms
        )
    )

    timestamps = np.asarray(
        timestamps_ms,
        dtype=np.int64,
    )

    timestamps = timestamps[
        (timestamps >= start_ms)
        & (timestamps < end_ms)
    ]

    if timestamps.size == 0:
        return 0.0

    indices = (
        (timestamps - start_ms)
        // bin_ms
    ).astype(np.int64)

    occupied = np.unique(indices).size

    return occupied / n_bins
