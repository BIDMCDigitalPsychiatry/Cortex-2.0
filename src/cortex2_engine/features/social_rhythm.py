"""Social rhythm regularity from daily anchors.

Construct
---------
How consistently the participant's daily anchors (wake, first unlock, first
departure, first and last communication; see ``anchors``) fall at the same
clock time across a week. This is a passive analogue of the Social Rhythm
Metric (Monk et al. 1990, 1991).

Estimator
---------
For each anchor with at least ``min_days_per_anchor`` occurrences in the
``window_days`` local days ending on ``end_date``:

- the *habitual time* is the circular mean of its clock times;
- a *hit* is a day whose time lies within ``regular_threshold_min``
  (± 45 min, as in the SRM) of the habitual time, on the 24 h circle;
- ``<anchor>_hit_fraction`` = hits / occurrences;
- ``<anchor>_circular_sd_h`` = ``circular_sd_hours`` of its times.

``social_rhythm_regularity`` is the mean hit fraction over eligible anchors,
in [0, 1]. ``social_rhythm_mean_circular_sd_h`` is the mean circular SD.
Both need at least ``min_anchors`` eligible anchors.

Circular statistics avoid the midnight wrap: 23:30 and 00:30 are one hour
apart, not 23.

What this is NOT
----------------
Not the self-report SRM, and not sleep regularity alone. Anchors that did
not occur are missing, not "irregular".
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cortex2_engine.features.anchors import ANCHOR_NAMES
from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.time import circular_mean_hours, circular_sd_hours


def circular_difference_hours(a: np.ndarray, b: float) -> np.ndarray:
    """Signed smallest difference ``a - b`` on the 24 h circle, in (-12, 12]."""

    d = (np.asarray(a, dtype=float) - b) % 24.0
    return np.where(d > 12.0, d - 24.0, d)


def social_rhythm_regularity(
    anchors_long: pd.DataFrame,
    *,
    end_date: dt.date | str | None = None,
    window_days: int = 7,
    min_days_per_anchor: int = 4,
    regular_threshold_min: float = 45.0,
    min_anchors: int = 2,
) -> dict:
    """Social rhythm regularity over the window ending on ``end_date``.

    Parameters
    ----------
    anchors_long
        Concatenated ``build_daily_anchors`` output (``date_local``,
        ``anchor``, ``clock_hours``).
    end_date
        Last local date of the window (inclusive). Defaults to the latest
        date in ``anchors_long``.
    window_days
        Window length (7).
    min_days_per_anchor
        Occurrences an anchor needs to be eligible (4).
    regular_threshold_min
        Half-width of a hit, in minutes (45).
    min_anchors
        Eligible anchors needed for the summary values (2).

    Returns
    -------
    dict
        ``social_rhythm_regularity`` and ``social_rhythm_mean_circular_sd_h``
        with ``_status``; ``social_rhythm_n_anchors``; and per anchor
        ``<anchor>_hit_fraction``, ``<anchor>_circular_sd_h`` and
        ``<anchor>_n_days``.
    """

    if window_days < 1 or min_days_per_anchor < 1 or min_anchors < 1:
        raise ValueError("window_days, min_days_per_anchor, min_anchors must be >= 1")

    if regular_threshold_min <= 0:
        raise ValueError("regular_threshold_min must be > 0")

    missing = {"date_local", "anchor", "clock_hours"} - set(anchors_long.columns)

    if missing:
        raise ValueError(
            "Missing required anchor columns: " + ", ".join(sorted(missing))
        )

    out: dict = {
        "social_rhythm_regularity": np.nan,
        "social_rhythm_mean_circular_sd_h": np.nan,
        "social_rhythm_n_anchors": 0,
    }

    dates = pd.to_datetime(anchors_long["date_local"], errors="coerce").dt.date

    if end_date is None:
        if dates.dropna().empty:
            for key in ("social_rhythm_regularity", "social_rhythm_mean_circular_sd_h"):
                out[f"{key}_status"] = FeatureStatus.NO_RAW_DATA.value
            return out
        end_date = max(dates.dropna())

    end = end_date if isinstance(end_date, dt.date) else dt.date.fromisoformat(end_date)
    first = end - dt.timedelta(days=window_days - 1)

    x = anchors_long[((dates >= first) & (dates <= end)).fillna(False).to_numpy()]

    hit_fractions = []
    sds = []

    for name in ANCHOR_NAMES:
        hours = pd.to_numeric(
            x.loc[x["anchor"] == name, "clock_hours"], errors="coerce"
        ).dropna().to_numpy()

        out[f"{name}_n_days"] = int(hours.size)
        out[f"{name}_hit_fraction"] = np.nan
        out[f"{name}_circular_sd_h"] = np.nan

        if hours.size < min_days_per_anchor:
            continue

        centre = circular_mean_hours(hours)
        sd = circular_sd_hours(hours)

        if not np.isfinite(centre) or not np.isfinite(sd):
            # Times spread evenly round the clock: no habitual time exists.
            continue

        hits = np.abs(circular_difference_hours(hours, centre)) <= regular_threshold_min / 60.0
        out[f"{name}_hit_fraction"] = float(hits.mean())
        out[f"{name}_circular_sd_h"] = float(sd)

        hit_fractions.append(float(hits.mean()))
        sds.append(float(sd))

    out["social_rhythm_n_anchors"] = len(hit_fractions)

    if x.empty:
        status = FeatureStatus.NO_RAW_DATA
    elif len(hit_fractions) < min_anchors:
        status = FeatureStatus.INSUFFICIENT_OBSERVABILITY
    else:
        status = None

    if status is not None:
        for key in ("social_rhythm_regularity", "social_rhythm_mean_circular_sd_h"):
            out[f"{key}_status"] = status.value
        return out

    for key, value in (
        ("social_rhythm_regularity", float(np.mean(hit_fractions))),
        ("social_rhythm_mean_circular_sd_h", float(np.mean(sds))),
    ):
        out[key] = value
        out[f"{key}_status"] = (
            FeatureStatus.OBSERVED_ZERO if value == 0 else FeatureStatus.OBSERVED
        ).value

    return out
