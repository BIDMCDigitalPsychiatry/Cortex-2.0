"""Non-parametric circadian rest-activity metrics (Van Someren et al. 1999).

Construct
---------
Regularity and amplitude of the 24 h rest-activity pattern over several days,
computed from an hourly activity matrix (days x 24 local clock hours) with an
explicit observed-hour mask:

- **Interdaily stability (IS)**: how closely each day follows the average
  24 h profile (0 = no day-to-day coupling, 1 = identical days).
- **Intradaily variability (IV)**: hour-to-hour fragmentation (about 0 for a
  smooth sinusoid, about 2 for white noise).
- **Relative amplitude (RA)**: ``(M10 - L5) / (M10 + L5)`` from the most
  active 10 h and least active 5 h of the average profile.

Exact formulas, with x_i the observed hourly values in time order, n their
number, x_bar their mean, and x_bar_h the mean of clock hour h over valid
days (p = 24)::

    IS = n * sum_h (x_bar_h - x_bar)^2 / (p * sum_i (x_i - x_bar)^2)

``p`` is the number of clock hours observed on at least one valid day (24
with complete data), and the sum runs over those hours.
    IV = n * sum_pairs (x_i - x_{i-1})^2 / (n_pairs * sum_i (x_i - x_bar)^2)

``n_pairs`` counts only pairs of *consecutive clock hours that are both
observed* (including 23:00 -> 00:00 across consecutive valid days). With no
missing hours, ``n_pairs = n - 1`` and the formula is Van Someren's.

What this is NOT
----------------
- Not a daily feature: IS/IV/RA need several days (``min_days``).
- Not sleep: L5 is the least *active* 5 h, which may include phone-on-desk
  time. Non-wear is excluded upstream when the matrix is built.
- Not cosinor amplitude or acrophase (parametric; not implemented).

DST
---
Columns are local *clock* hours. On a 25 h day the repeated hour is averaged;
on a 23 h day the skipped hour is unobserved.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.time import (
    local_dates,
    local_day_bounds_ms,
    to_local_clock_hours,
)

HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
MINUTE_MS = 60_000
M10_HOURS = 10
L5_HOURS = 5


def hourly_activity_matrix(
    minute_epochs: pd.DataFrame,
    *,
    tz: str,
    day_start: str = "00:00",
    min_hour_observed_fraction: float = 0.5,
) -> tuple[list[dt.date], np.ndarray, np.ndarray]:
    """Build a days x 24 matrix of mean ENMO per local clock hour.

    Parameters
    ----------
    minute_epochs
        Output of ``acc_preprocess.aggregate_epochs`` (``epoch_start_ms``,
        ``enmo_mg``, ``valid``), with non-wear already excluded.
    tz
        IANA timezone.
    day_start
        Local day boundary used to assign rows to days.
    min_hour_observed_fraction
        Minimum share of the hour's valid minutes for the cell to count as
        observed (default half the hour).

    Returns
    -------
    tuple
        ``(dates, values_mg, observed)``: a contiguous range of local dates
        (days without data are all-unobserved rows), a float matrix
        of hourly mean ENMO in mg (NaN where unobserved), and its boolean
        observed mask.
    """

    required = {"epoch_start_ms", "enmo_mg", "valid"}
    missing = required - set(minute_epochs.columns)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    if minute_epochs.empty:
        return [], np.zeros((0, HOURS_PER_DAY)), np.zeros(
            (0, HOURS_PER_DAY),
            dtype=bool,
        )

    t = minute_epochs["epoch_start_ms"].to_numpy(dtype=np.int64)

    frame = pd.DataFrame({
        "date": local_dates(t, tz, day_start),
        "hour": np.floor(to_local_clock_hours(t, tz)).astype(int),
        "enmo_mg": minute_epochs["enmo_mg"].to_numpy(dtype=float),
        "valid": minute_epochs["valid"].to_numpy(dtype=bool),
    })

    n_valid = frame.groupby(["date", "hour"], sort=True)["valid"].sum()

    means = (
        frame[frame["valid"]]
        .groupby(["date", "hour"], sort=True)["enmo_mg"]
        .mean()
    )

    # Contiguous calendar range: a day with no rows is an unobserved row,
    # not a missing row, so IV never chains non-adjacent days.
    present = sorted(frame["date"].unique())
    n_days = (present[-1] - present[0]).days + 1
    dates = [present[0] + dt.timedelta(days=i) for i in range(n_days)]
    index = {d: i for i, d in enumerate(dates)}

    values = np.full((len(dates), HOURS_PER_DAY), np.nan)
    observed = np.zeros((len(dates), HOURS_PER_DAY), dtype=bool)

    # Minutes each clock hour really contains: 60, 120 for the repeated hour
    # of a fall-back day, 0 for the skipped hour of a spring-forward day.
    nominal = {}

    for date in dates:
        start_ms, end_ms = local_day_bounds_ms(date, tz, day_start)
        grid = np.arange(start_ms, end_ms, MINUTE_MS, dtype=np.int64)
        hours, counts = np.unique(
            np.floor(to_local_clock_hours(grid, tz)).astype(int),
            return_counts=True,
        )
        nominal.update({
            (date, int(h)): int(c) for h, c in zip(hours, counts)
        })

    for (date, hour), n in n_valid.items():
        share = n / nominal.get((date, hour), MINUTES_PER_HOUR)

        if share >= min_hour_observed_fraction and (date, hour) in means:
            values[index[date], hour] = means[(date, hour)]
            observed[index[date], hour] = True

    return dates, values, observed


def _valid_days(
    observed: np.ndarray,
    *,
    min_observed_hours_per_day: int,
) -> np.ndarray:

    return observed.sum(axis=1) >= min_observed_hours_per_day


def _gate(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    min_days: int,
    min_observed_hours_per_day: int,
    max_excluded_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FeatureStatus, dict]:
    """Select valid days and decide whether the metrics may be reported.

    Returns the valid-day submatrix and mask, the full-length mask with
    invalid days blanked (rows are consecutive calendar days), the status and
    context.
    """

    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool) & np.isfinite(values)

    if values.ndim != 2 or values.shape[1] != HOURS_PER_DAY:
        raise ValueError("matrix must have shape (days, 24)")

    if values.shape != observed.shape:
        raise ValueError("mask must have the same shape as the matrix")

    keep = _valid_days(
        observed,
        min_observed_hours_per_day=min_observed_hours_per_day,
    )

    v = values[keep]
    m = observed[keep]
    full_mask = observed & keep[:, None]

    n_cells = m.size
    excluded_fraction = 1.0 - m.sum() / n_cells if n_cells else np.nan

    context = {
        "n_days_total": int(values.shape[0]),
        "n_days_valid": int(keep.sum()),
        "excluded_hour_fraction": float(excluded_fraction),
    }

    if values.shape[0] == 0 or not observed.any():
        status = FeatureStatus.NO_RAW_DATA

    elif (
        keep.sum() < min_days
        or excluded_fraction > max_excluded_fraction
    ):
        status = FeatureStatus.INSUFFICIENT_OBSERVABILITY

    else:
        status = FeatureStatus.OBSERVED

    return v, m, full_mask, status, context


def _hour_means(
    values: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Mean of each clock hour over the days it was observed (NaN if never)."""

    counts = observed.sum(axis=0)
    sums = np.where(observed, values, 0.0).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _series(
    values: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Observed hourly values in time order (missing as NaN)."""

    return np.where(observed, values, np.nan).ravel()


def interdaily_stability(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    min_days: int = 5,
    min_observed_hours_per_day: int = 18,
    max_excluded_fraction: float = 0.2,
) -> dict:
    """Interdaily stability (IS) over valid days.

    Parameters
    ----------
    values
        Days x 24 hourly activity matrix.
    observed
        Matching boolean mask.
    min_days
        Minimum number of valid days.
    min_observed_hours_per_day
        A day is valid when at least this many hours are observed.
    max_excluded_fraction
        Maximum share of hourly cells, within valid days, that may be
        unobserved.

    Returns
    -------
    dict
        ``interdaily_stability`` (dimensionless, [0, 1]), its ``_status``,
        and the day/coverage context.
    """

    v, m, _, status, context = _gate(
        values,
        observed,
        min_days=min_days,
        min_observed_hours_per_day=min_observed_hours_per_day,
        max_excluded_fraction=max_excluded_fraction,
    )

    value = np.nan

    if status is FeatureStatus.OBSERVED:
        x = v[m]
        grand = x.mean()
        total = float(((x - grand) ** 2).sum())

        hour_means = _hour_means(v, m)
        hour_means = hour_means[np.isfinite(hour_means)]

        if total == 0.0 or hour_means.size == 0:
            status = FeatureStatus.NOT_APPLICABLE

        else:
            value = float(
                x.size * ((hour_means - grand) ** 2).sum()
                / (hour_means.size * total)
            )

    return {
        "interdaily_stability": value,
        "interdaily_stability_status": status.value,
        **context,
    }


def intradaily_variability(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    min_days: int = 5,
    min_observed_hours_per_day: int = 18,
    max_excluded_fraction: float = 0.2,
) -> dict:
    """Intradaily variability (IV) over valid days.

    Rows must be consecutive calendar days. Hours are chained across midnight
    only between adjacent valid days; an excluded day breaks the series.

    Parameters
    ----------
    values, observed, min_days, min_observed_hours_per_day, max_excluded_fraction
        As for :func:`interdaily_stability`.

    Returns
    -------
    dict
        ``intradaily_variability`` (dimensionless, >= 0), its ``_status``, the
        number of consecutive observed pairs, and day/coverage context.
    """

    _, _, full_mask, status, context = _gate(
        values,
        observed,
        min_days=min_days,
        min_observed_hours_per_day=min_observed_hours_per_day,
        max_excluded_fraction=max_excluded_fraction,
    )

    value = np.nan
    n_pairs = 0

    if status is FeatureStatus.OBSERVED:
        series = _series(np.asarray(values, dtype=float), full_mask)
        x = series[np.isfinite(series)]
        grand = x.mean()
        total = float(((x - grand) ** 2).sum())

        diffs = np.diff(series)
        diffs = diffs[np.isfinite(diffs)]
        n_pairs = int(diffs.size)

        if total == 0.0 or n_pairs == 0:
            status = FeatureStatus.NOT_APPLICABLE

        else:
            value = float(
                x.size * (diffs ** 2).sum() / (n_pairs * total)
            )

    return {
        "intradaily_variability": value,
        "intradaily_variability_status": status.value,
        "intradaily_variability_n_pairs": n_pairs,
        **context,
    }


def _circular_window_means(
    profile: np.ndarray,
    width: int,
) -> np.ndarray:

    padded = np.concatenate([profile, profile[: width - 1]])
    kernel = np.ones(width) / width

    return np.convolve(padded, kernel, mode="valid")[:HOURS_PER_DAY]


def relative_amplitude(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    min_days: int = 5,
    min_observed_hours_per_day: int = 18,
    max_excluded_fraction: float = 0.2,
) -> dict:
    """Relative amplitude with M10 and L5 from the average 24 h profile.

    Parameters
    ----------
    values, observed, min_days, min_observed_hours_per_day, max_excluded_fraction
        As for :func:`interdaily_stability`.

    Returns
    -------
    dict
        ``relative_amplitude`` (dimensionless, [0, 1]), ``m10_mg``, ``l5_mg``,
        ``m10_onset_local_h`` and ``l5_onset_local_h`` (clock hour at which
        the window starts; windows wrap around midnight), each with a status.

    Notes
    -----
    Ties are broken by the earliest onset hour.
    """

    v, m, _, status, context = _gate(
        values,
        observed,
        min_days=min_days,
        min_observed_hours_per_day=min_observed_hours_per_day,
        max_excluded_fraction=max_excluded_fraction,
    )

    out = {
        "relative_amplitude": np.nan,
        "m10_mg": np.nan,
        "l5_mg": np.nan,
        "m10_onset_local_h": np.nan,
        "l5_onset_local_h": np.nan,
    }

    if status is FeatureStatus.OBSERVED:

        profile = _hour_means(v, m)

        if not np.isfinite(profile).all():
            status = FeatureStatus.INSUFFICIENT_OBSERVABILITY

        else:
            m10 = _circular_window_means(profile, M10_HOURS)
            l5 = _circular_window_means(profile, L5_HOURS)

            m10_onset = int(np.argmax(m10))
            l5_onset = int(np.argmin(l5))

            out["m10_mg"] = float(m10[m10_onset])
            out["l5_mg"] = float(l5[l5_onset])
            out["m10_onset_local_h"] = float(m10_onset)
            out["l5_onset_local_h"] = float(l5_onset)

            denominator = out["m10_mg"] + out["l5_mg"]

            if denominator > 0:
                out["relative_amplitude"] = (
                    (out["m10_mg"] - out["l5_mg"]) / denominator
                )

    ra_status = status

    if status is FeatureStatus.OBSERVED and np.isnan(out["relative_amplitude"]):
        ra_status = FeatureStatus.NOT_APPLICABLE

    for key in list(out):
        out[f"{key}_status"] = (
            ra_status if key == "relative_amplitude" else status
        ).value

    out.update(context)

    return out


def compute_circadian_features(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    min_days: int = 5,
    min_observed_hours_per_day: int = 18,
    max_excluded_fraction: float = 0.2,
) -> dict:
    """IS, IV and RA (with M10/L5) for one multi-day window.

    Parameters
    ----------
    values, observed, min_days, min_observed_hours_per_day, max_excluded_fraction
        As for :func:`interdaily_stability`.

    Returns
    -------
    dict
        Union of the three metric dicts.
    """

    kwargs = {
        "min_days": min_days,
        "min_observed_hours_per_day": min_observed_hours_per_day,
        "max_excluded_fraction": max_excluded_fraction,
    }

    out = {}
    out.update(interdaily_stability(values, observed, **kwargs))
    out.update(intradaily_variability(values, observed, **kwargs))
    out.update(relative_amplitude(values, observed, **kwargs))

    return out
