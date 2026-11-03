"""Shared local-time utilities for Cortex 2.0 features.

Every daily feature is defined on a *local* day: the half-open interval
``[day_start, next day_start)`` in the study timezone. A local day is therefore
not always 24 h long. Across a daylight-saving transition it is 23 h or 25 h,
and features must use the true duration of the day as their denominator
(rule 1: observed time, never a fixed 24 h).

All conversions go through :mod:`zoneinfo`; no fixed UTC offsets are used.

This module does not decide which day a *behaviour* belongs to (for example a
night of sleep is attributed to the date on which it ends). That is the job of
each feature module.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from collections.abc import Iterator
from typing import NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cortex2_engine.canonical.timestamps import datetime_to_epoch_ms

__all__ = [
    "DAY_MS",
    "HOUR_MS",
    "LocalDay",
    "circular_mean_hours",
    "circular_sd_hours",
    "coerce_timestamp_ms",
    "datetime_to_epoch_ms",
    "iter_local_days",
    "local_dates",
    "local_day_bounds_ms",
    "parse_clock",
    "to_local_clock_hours",
]

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
HOURS_PER_DAY = 24.0

# Mean resultant lengths below this are treated as zero (no mean direction):
# opposite clock times cancel only up to floating-point error.
_ZERO_RESULTANT = 1e-12


class LocalDay(NamedTuple):
    """One local day, optionally clipped to an enclosing window.

    Attributes
    ----------
    date
        Local calendar date whose ``day_start`` opens the interval.
    start_ms, end_ms
        Half-open epoch-ms bounds after clipping.
    day_start_ms, day_end_ms
        Unclipped bounds of the full local day.
    clipped
        True when the window truncates the day (the day is partial).
    """

    date: dt.date
    start_ms: int
    end_ms: int
    day_start_ms: int
    day_end_ms: int
    clipped: bool

    @property
    def duration_ms(self) -> int:
        """Duration of the (possibly clipped) interval."""
        return self.end_ms - self.start_ms

    @property
    def day_duration_ms(self) -> int:
        """Duration of the full local day (23, 24 or 25 h)."""
        return self.day_end_ms - self.day_start_ms


def parse_clock(
    value: str | dt.time,
) -> dt.time:
    """Parse an ``"HH:MM"`` (or ``"HH:MM:SS"``) local clock time.

    Parameters
    ----------
    value
        Clock text or an existing :class:`datetime.time`.

    Returns
    -------
    datetime.time
        Naive clock time.
    """

    if isinstance(value, dt.time):
        return value

    try:
        return dt.time.fromisoformat(
            str(value).strip()
        )

    except ValueError as exc:
        raise ValueError(
            f"Invalid clock time {value!r}; expected HH:MM"
        ) from exc


def _zone(
    tz: str | ZoneInfo,
) -> ZoneInfo:

    if isinstance(tz, ZoneInfo):
        return tz

    return ZoneInfo(str(tz))


def _local_instant_ms(
    date: dt.date,
    clock: dt.time,
    zone: ZoneInfo,
) -> int:
    """Epoch ms of a local wall-clock instant.

    Nonexistent wall times (inside a spring-forward gap) resolve with
    ``fold=0``, i.e. using the offset in force before the transition.
    """

    local = dt.datetime.combine(
        date,
        clock,
        tzinfo=zone,
    )

    return round(
        local.timestamp() * 1000
    )


def local_day_bounds_ms(
    date: dt.date | str,
    tz: str | ZoneInfo,
    day_start: str | dt.time = "00:00",
) -> tuple[int, int]:
    """Return the half-open epoch-ms bounds of one local day.

    Parameters
    ----------
    date
        Local calendar date (``datetime.date`` or ISO ``YYYY-MM-DD``).
    tz
        IANA timezone name, e.g. ``StudyConfig.timezone``.
    day_start
        Local clock time at which a day begins (default midnight).

    Returns
    -------
    tuple of int
        ``(start_ms, end_ms)``. ``end_ms - start_ms`` is 23 h, 24 h or 25 h
        depending on daylight-saving transitions.
    """

    if isinstance(date, str):
        date = dt.date.fromisoformat(date)

    zone = _zone(tz)
    clock = parse_clock(day_start)

    start_ms = _local_instant_ms(
        date,
        clock,
        zone,
    )

    end_ms = _local_instant_ms(
        date + dt.timedelta(days=1),
        clock,
        zone,
    )

    return start_ms, end_ms


def local_dates(
    timestamp_ms: np.ndarray | pd.Series | list[int],
    tz: str | ZoneInfo,
    day_start: str | dt.time = "00:00",
) -> np.ndarray:
    """Assign epoch-ms instants to local days.

    Parameters
    ----------
    timestamp_ms
        Epoch milliseconds.
    tz
        IANA timezone name.
    day_start
        Local clock time at which a day begins.

    Returns
    -------
    numpy.ndarray
        Object array of :class:`datetime.date` (``None`` for missing input).
    """

    clock = parse_clock(day_start)

    offset = pd.Timedelta(
        hours=clock.hour,
        minutes=clock.minute,
        seconds=clock.second,
    )

    values = pd.to_numeric(
        pd.Series(np.asarray(timestamp_ms)),
        errors="coerce",
    )

    local = (
        pd.to_datetime(
            values,
            unit="ms",
            utc=True,
        )
        .dt.tz_convert(str(_zone(tz)))
        .dt.tz_localize(None)
        - offset
    )

    return np.array(
        [
            None if pd.isna(v) else v.date()
            for v in local
        ],
        dtype=object,
    )


def iter_local_days(
    start_ms: int,
    end_ms: int,
    tz: str | ZoneInfo,
    *,
    day_start: str | dt.time = "00:00",
) -> Iterator[LocalDay]:
    """Iterate over the local days overlapping ``[start_ms, end_ms)``.

    Parameters
    ----------
    start_ms, end_ms
        Half-open epoch-ms window, typically an episode.
    tz
        IANA timezone name.
    day_start
        Local clock time at which a day begins.

    Yields
    ------
    LocalDay
        One record per overlapping day, clipped to the window. ``clipped`` is
        True for a partial first or last day.
    """

    start_ms = int(start_ms)
    end_ms = int(end_ms)

    if end_ms <= start_ms:
        raise ValueError(
            "end_ms must be greater than start_ms"
        )

    zone = _zone(tz)

    date = local_dates(
        [start_ms],
        zone,
        day_start,
    )[0]

    while True:

        day_start_ms, day_end_ms = local_day_bounds_ms(
            date,
            zone,
            day_start,
        )

        if day_start_ms >= end_ms:
            return

        lo = max(day_start_ms, start_ms)
        hi = min(day_end_ms, end_ms)

        if hi > lo:
            yield LocalDay(
                date=date,
                start_ms=lo,
                end_ms=hi,
                day_start_ms=day_start_ms,
                day_end_ms=day_end_ms,
                clipped=bool(
                    lo != day_start_ms
                    or hi != day_end_ms
                ),
            )

        date = date + dt.timedelta(days=1)


def to_local_clock_hours(
    timestamp_ms: np.ndarray | pd.Series | list[int],
    tz: str | ZoneInfo,
) -> np.ndarray:
    """Convert epoch-ms instants to local clock time in decimal hours.

    Parameters
    ----------
    timestamp_ms
        Epoch milliseconds.
    tz
        IANA timezone name.

    Returns
    -------
    numpy.ndarray
        Float hours in ``[0, 24)`` since local midnight (NaN for missing
        input). On a DST day the clock reading is what a wall clock shows, so
        the mapping from elapsed time to clock hours is not linear that day.
    """

    values = pd.to_numeric(
        pd.Series(np.asarray(timestamp_ms)),
        errors="coerce",
    )

    local = pd.to_datetime(
        values,
        unit="ms",
        utc=True,
    ).dt.tz_convert(str(_zone(tz)))

    hours = (
        local.dt.hour
        + local.dt.minute / 60.0
        + local.dt.second / 3600.0
        + local.dt.microsecond / 3.6e9
    )

    return hours.to_numpy(
        dtype=float,
        na_value=np.nan,
    )


def _circular_resultant(
    hours: np.ndarray,
) -> tuple[float, float] | None:
    """Mean resultant angle (radians) and length of clock hours."""

    h = np.asarray(hours, dtype=float)
    h = h[np.isfinite(h)]

    if h.size == 0:
        return None

    theta = 2.0 * math.pi * h / HOURS_PER_DAY

    c = float(np.mean(np.cos(theta)))
    s = float(np.mean(np.sin(theta)))

    return math.atan2(s, c), math.hypot(c, s)


def circular_mean_hours(
    hours: np.ndarray,
) -> float:
    """Circular mean of clock times on the 24 h circle.

    Parameters
    ----------
    hours
        Clock hours in ``[0, 24)``; non-finite values are ignored.

    Returns
    -------
    float
        Mean clock hour in ``[0, 24)``; NaN for empty input or when the
        resultant length is numerically zero (no defined mean direction).
    """

    result = _circular_resultant(hours)

    if result is None:
        return float("nan")

    angle, length = result

    if length < _ZERO_RESULTANT:
        return float("nan")

    return (angle * HOURS_PER_DAY / (2.0 * math.pi)) % HOURS_PER_DAY


def circular_sd_hours(
    hours: np.ndarray,
) -> float:
    """Circular standard deviation of clock times, in hours.

    Parameters
    ----------
    hours
        Clock hours in ``[0, 24)``; non-finite values are ignored.

    Returns
    -------
    float
        ``sqrt(-2 ln R) * 24 / (2 pi)``, where ``R`` is the mean resultant
        length of the clock angles. 0 for identical times, NaN for empty
        input, ``inf`` when ``R`` is zero (uniformly spread times).

    Notes
    -----
    Times either side of midnight are close (23:30 and 00:30 have a circular SD
    of about 0.5 h, not 16 h), which is why a linear SD must not be used for
    clock-time regularity.
    """

    result = _circular_resultant(hours)

    if result is None:
        return float("nan")

    _, length = result

    length = min(length, 1.0)

    if length < _ZERO_RESULTANT:
        return float("inf")

    if length >= 1.0 - 1e-15:
        return 0.0

    return (
        math.sqrt(-2.0 * math.log(length))
        * HOURS_PER_DAY
        / (2.0 * math.pi)
    )


def coerce_timestamp_ms(
    df: pd.DataFrame,
    *,
    caller: str,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a canonical ``timestamp_ms`` column.

    Parameters
    ----------
    df
        Input frame carrying ``timestamp_ms`` (canonical) or the legacy
        ``timestamp`` column (epoch ms).
    caller
        Function name used in the warning and error text.

    Returns
    -------
    pandas.DataFrame
        Copy with ``timestamp_ms``. ``timestamp_ms`` wins when both exist.

    Warns
    -----
    DeprecationWarning
        When only the legacy ``timestamp`` column is present.
    """

    if "timestamp_ms" in df.columns:
        return df.copy()

    if "timestamp" in df.columns:
        warnings.warn(
            f"{caller}: the 'timestamp' column is deprecated; pass "
            "canonical 'timestamp_ms' (epoch milliseconds) instead.",
            DeprecationWarning,
            stacklevel=3,
        )

        out = df.copy()
        out["timestamp_ms"] = out["timestamp"]

        return out

    raise ValueError(
        f"{caller}: missing required column timestamp_ms "
        "(or legacy timestamp)"
    )
