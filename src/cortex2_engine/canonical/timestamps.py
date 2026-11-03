"""Resolution-independent datetime <-> epoch-millisecond conversion.

pandas 2 introduced non-nanosecond datetime resolutions and pandas 3 made
microsecond resolution the default for many constructors. Code that converts
datetimes with ``series.astype("int64") // 1_000_000`` silently assumes
nanoseconds and is wrong by a factor of 1000 under any other resolution.

This module is the single place where that conversion happens.
"""

from __future__ import annotations

import pandas as pd

_EPOCH_UTC = pd.Timestamp(0, tz="UTC")
_ONE_MS = pd.Timedelta(milliseconds=1)


def datetime_to_epoch_ms(
    values: pd.Series,
) -> pd.Series:
    """Convert timezone-aware or naive-UTC datetimes to epoch milliseconds.

    Parameters
    ----------
    values
        Datetime-like series. Naive values are interpreted as UTC.

    Returns
    -------
    pandas.Series
        Nullable ``Int64`` epoch milliseconds (floor), with ``<NA>`` where
        the input is ``NaT``.

    Notes
    -----
    The conversion uses timedelta arithmetic, so it is correct for any
    datetime resolution (s, ms, us, ns).
    """

    dt = pd.to_datetime(
        values,
        utc=True,
        errors="coerce",
    )

    delta = dt - _EPOCH_UTC

    return (
        delta
        // _ONE_MS
    ).astype("Int64")
