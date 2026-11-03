"""Within-session reaction-time variability for smartphone cognitive tasks.

Construct
---------
Intra-individual variability of response times within one task session,
which is a marker of attentional lapses that mean RT misses (MacDonald et al.
2006). Computed per session and then summarised by local week.

Estimators (per session)
------------------------
Trials are kept when ``rt_min_ms <= response_time_ms <= rt_max_multiple_of_median
× median``. Anticipations and lapses are trimmed, and the upper cut adapts to
the person and task. A session needs at least ``min_trials`` kept trials.

- ``rt_cov`` — coefficient of variation, SD / mean (sample SD, ddof = 1).
- ``rt_tau_ms`` — the exponential component of an ex-Gaussian fitted by the
  method of moments (Olivier & Norberg 2010): with sample skewness
  ``g = m3 / s³``, ``tau = s × (g / 2)^(1/3)``, and ``g`` capped at 2 (the
  ex-Gaussian maximum, where tau = s). With ``g <= 0`` the ex-Gaussian does
  not fit, so tau is ``not_applicable`` and is not set to 0.
- ``rt_slope_ms_per_trial`` — OLS slope of RT on kept-trial order (positive
  values mean slowing, for example fatigue).

Weekly summary
--------------
:func:`weekly_cognitive_summary` gives, per task and local week (Monday
start), the median of each session metric over sessions with an
``observed`` value, plus the number of sessions.

What this is NOT
----------------
Not accuracy and not a normed cognitive score. Values are not comparable
across tasks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.time import local_dates

SESSION_KEYS = ["rt_cov", "rt_tau_ms", "rt_slope_ms_per_trial"]


def ex_gaussian_tau_mom(rt_ms: np.ndarray) -> float:
    """Method-of-moments ex-Gaussian tau (NaN when skewness <= 0)."""

    x = np.asarray(rt_ms, dtype=float)

    if x.size < 3:
        return np.nan

    s = float(np.std(x, ddof=1))

    if s <= 0:
        return np.nan

    m3 = float(np.mean((x - x.mean()) ** 3))
    g = m3 / s**3

    if g <= 0:
        return np.nan

    return float(s * (min(g, 2.0) / 2.0) ** (1.0 / 3.0))


def session_rt_variability(
    trials: pd.DataFrame,
    *,
    min_trials: int = 8,
    rt_min_ms: float = 150.0,
    rt_max_multiple_of_median: float = 3.0,
) -> dict:
    """Reaction-time variability for the trials of one session.

    Parameters
    ----------
    trials
        Canonical cognitive rows of one session (``timestamp_ms``,
        ``response_time_ms``).
    min_trials
        Kept trials needed (8).
    rt_min_ms
        Anticipation cut-off (150 ms).
    rt_max_multiple_of_median
        Upper cut-off as a multiple of the session median (3).

    Returns
    -------
    dict
        ``SESSION_KEYS`` with ``_status``, plus ``n_trials``,
        ``n_trials_kept`` and ``rt_mean_ms``.
    """

    if min_trials < 3:
        raise ValueError("min_trials must be >= 3")

    x = trials.sort_values("timestamp_ms", kind="mergesort")
    rt = pd.to_numeric(x["response_time_ms"], errors="coerce").to_numpy(float)
    rt = rt[np.isfinite(rt)]

    out: dict = {"n_trials": int(rt.size)}

    kept = rt[rt >= rt_min_ms]

    if kept.size:
        kept = kept[kept <= rt_max_multiple_of_median * np.median(kept)]

    out["n_trials_kept"] = int(kept.size)
    out["rt_mean_ms"] = float(kept.mean()) if kept.size else np.nan

    if rt.size == 0:
        status = FeatureStatus.NO_RAW_DATA
    elif kept.size < min_trials:
        status = FeatureStatus.INSUFFICIENT_OBSERVABILITY
    else:
        status = None

    if status is not None:
        for key in SESSION_KEYS:
            out[key] = np.nan
            out[f"{key}_status"] = status.value
        return out

    values = {
        "rt_cov": float(np.std(kept, ddof=1) / kept.mean()),
        "rt_tau_ms": ex_gaussian_tau_mom(kept),
        "rt_slope_ms_per_trial": float(np.polyfit(np.arange(kept.size), kept, 1)[0]),
    }

    for key, value in values.items():
        if not np.isfinite(value):
            key_status = FeatureStatus.NOT_APPLICABLE
            value = np.nan
        elif value == 0:
            key_status = FeatureStatus.OBSERVED_ZERO
        else:
            key_status = FeatureStatus.OBSERVED

        out[key] = value
        out[f"{key}_status"] = key_status.value

    return out


def compute_session_variability(
    cognitive: pd.DataFrame,
    *,
    tz: str,
    **kwargs,
) -> pd.DataFrame:
    """Per-session variability for every task session.

    Parameters
    ----------
    cognitive
        Canonical cognitive rows (``task``, ``session_id``, ``timestamp_ms``,
        ``response_time_ms``).
    tz
        IANA timezone for ``date_local``.
    **kwargs
        Passed to :func:`session_rt_variability`.

    Returns
    -------
    pandas.DataFrame
        One row per ``(task, session_id)`` with ``session_start_ms``,
        ``date_local`` and the session metrics.
    """

    missing = {"task", "session_id", "timestamp_ms", "response_time_ms"} - set(
        cognitive.columns
    )

    if missing:
        raise ValueError(
            "Missing required cognitive columns: " + ", ".join(sorted(missing))
        )

    rows = []

    for (task, session), trials in cognitive.groupby(["task", "session_id"], sort=True):
        start = int(pd.to_numeric(trials["timestamp_ms"]).min())
        row = {"task": task, "session_id": session, "session_start_ms": start}
        row.update(session_rt_variability(trials, **kwargs))
        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame(columns=["task", "session_id", "session_start_ms", "date_local"])

    out.insert(
        3,
        "date_local",
        [d.isoformat() for d in local_dates(out["session_start_ms"].to_numpy(), tz)],
    )

    return out


def weekly_cognitive_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    """Median session metrics per task and local week (Monday start).

    Parameters
    ----------
    sessions
        :func:`compute_session_variability` output.

    Returns
    -------
    pandas.DataFrame
        ``task``, ``week_start``, ``n_sessions`` and, per metric,
        ``<metric>_weekly_median``, ``<metric>_n_sessions`` and
        ``<metric>_weekly_median_status`` (``insufficient_observability``
        when no session in the week has an observed value).
    """

    if sessions.empty:
        return pd.DataFrame(columns=["task", "week_start", "n_sessions"])

    s = sessions.copy()
    dates = pd.to_datetime(s["date_local"])
    s["week_start"] = (dates - pd.to_timedelta(dates.dt.day_of_week, unit="D")).dt.date.astype(str)

    rows = []
    ok = {FeatureStatus.OBSERVED.value, FeatureStatus.OBSERVED_ZERO.value}

    for (task, week), group in s.groupby(["task", "week_start"], sort=True):
        row = {"task": task, "week_start": week, "n_sessions": len(group)}

        for key in SESSION_KEYS:
            valid = group.loc[group[f"{key}_status"].isin(ok), key]
            row[f"{key}_n_sessions"] = len(valid)

            if len(valid):
                value = float(valid.median())
                row[f"{key}_weekly_median"] = value
                row[f"{key}_weekly_median_status"] = (
                    FeatureStatus.OBSERVED_ZERO if value == 0 else FeatureStatus.OBSERVED
                ).value
            else:
                row[f"{key}_weekly_median"] = np.nan
                row[f"{key}_weekly_median_status"] = (
                    FeatureStatus.INSUFFICIENT_OBSERVABILITY.value
                )

        rows.append(row)

    return pd.DataFrame(rows)
