"""Location novelty and routine index from candidate-place assignments.

Construct
---------
- ``location_novelty_fraction``: the share of a day's place-assigned time
  spent at places with no assigned time in the preceding ``history_days``
  local days. 0 means everywhere visited was familiar.
- ``routine_index``: how closely the day's hour-by-hour place sequence
  repeats the previous ``window_days`` days. For each earlier day, similarity
  is the share of *shared observed hours* (hours where both days have a
  place) in which the place is the same. The index is the mean similarity
  over comparison days with at least ``min_shared_hours`` shared hours, and
  needs at least ``min_days`` such days. It is in [0, 1].

Both use place-assignment time, which is observed time (rule 1). Unobserved
hours are excluded from the comparison, never treated as "away".

Hours are local clock hours (``zoneinfo``). On a 25 h DST day the repeated
hour is merged into one clock-hour label, and on a 23 h day the skipped hour
is absent, so days stay comparable hour by hour.

Inputs
------
``build_place_candidates`` assignments computed over the whole
participant-episode, so ``place_candidate_id`` is stable across days.

What this is NOT
----------------
Not entropy of places visited, and not a measure of how far the
participant travelled.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus, status_from_counts
from cortex2_engine.time import HOUR_MS, local_day_bounds_ms, to_local_clock_hours

ROUTINE_KEYS = ["location_novelty_fraction", "routine_index"]


def _clean_assignments(assignments: pd.DataFrame) -> pd.DataFrame:

    required = {"episode_start_ms", "episode_end_ms", "place_candidate_id"}
    missing = required - set(assignments.columns)

    if missing:
        raise ValueError(
            "Missing required assignment columns: "
            + ", ".join(sorted(missing))
        )

    a = assignments[sorted(required)].apply(pd.to_numeric, errors="coerce")
    a = a.dropna()

    return a[a["episode_end_ms"] > a["episode_start_ms"]]


def _time_by_place(a: pd.DataFrame, start_ms: int, end_ms: int) -> pd.Series:
    """Assigned milliseconds per place inside ``[start_ms, end_ms)``."""

    overlap = np.clip(
        np.minimum(a["episode_end_ms"].to_numpy(float), end_ms)
        - np.maximum(a["episode_start_ms"].to_numpy(float), start_ms),
        0,
        None,
    )

    by_place = (
        pd.Series(overlap)
        .groupby(a["place_candidate_id"].to_numpy(dtype=int))
        .sum()
    )

    return by_place[by_place > 0]


def _as_date(date: dt.date | str) -> dt.date:

    return date if isinstance(date, dt.date) else dt.date.fromisoformat(date)


def location_novelty_fraction(
    assignments: pd.DataFrame,
    *,
    date: dt.date | str,
    tz: str,
    history_days: int = 28,
    min_history_days: int = 7,
    min_supported_ms: float = 21_600_000,
    min_day_observed_ms: float = 3_600_000,
) -> dict:
    """Share of a day's observed place time spent at unfamiliar places.

    Parameters
    ----------
    assignments
        Episode-wide place assignments.
    date
        Target local date.
    tz
        IANA timezone.
    history_days
        Look-back window in local days (28).
    min_history_days
        History days with at least ``min_day_observed_ms`` of assigned time
        needed before novelty is meaningful (7).
    min_supported_ms
        Minimum assigned time on the target day (6 h).
    min_day_observed_ms
        Assigned time for a history day to count (1 h).

    Returns
    -------
    dict
        ``location_novelty_fraction`` with ``_status``,
        ``novelty_history_days`` and ``novelty_supported_ms``.
    """

    if history_days < 1 or min_history_days < 1:
        raise ValueError("history_days and min_history_days must be >= 1")

    a = _clean_assignments(assignments)
    day = _as_date(date)
    start, end = local_day_bounds_ms(day, tz)

    today = _time_by_place(a, start, end)
    supported = float(today.sum())

    history_places: set[int] = set()
    n_history_days = 0

    for offset in range(1, history_days + 1):
        h_start, h_end = local_day_bounds_ms(day - dt.timedelta(days=offset), tz)
        by_place = _time_by_place(a, h_start, h_end)

        if by_place.sum() >= min_day_observed_ms:
            n_history_days += 1

        history_places.update(int(p) for p in by_place.index)

    value = (
        float(today[~today.index.isin(history_places)].sum() / supported)
        if supported > 0
        else np.nan
    )

    status = status_from_counts(
        supported,
        min_supported_ms,
        value,
        n_records=len(a) if len(a) else 0,
    )

    if (
        status in (FeatureStatus.OBSERVED, FeatureStatus.OBSERVED_ZERO)
        and n_history_days < min_history_days
    ):
        status = FeatureStatus.INSUFFICIENT_OBSERVABILITY

    return {
        "location_novelty_fraction": (
            value
            if status in (FeatureStatus.OBSERVED, FeatureStatus.OBSERVED_ZERO)
            else np.nan
        ),
        "location_novelty_fraction_status": status.value,
        "novelty_history_days": n_history_days,
        "novelty_supported_ms": supported,
    }


def hourly_place_sequence(
    assignments: pd.DataFrame,
    *,
    date: dt.date | str,
    tz: str,
    min_hour_observed_ms: float = 1_800_000,
) -> dict[int, int]:
    """Dominant place for each local clock hour of one day.

    Parameters
    ----------
    assignments
        Episode-wide place assignments.
    date
        Local date.
    tz
        IANA timezone.
    min_hour_observed_ms
        Assigned time an hour needs to receive a place (30 min).

    Returns
    -------
    dict
        ``{clock_hour: place_candidate_id}`` for observed hours only. Ties go
        to the lowest place id.
    """

    a = _clean_assignments(assignments)
    start, end = local_day_bounds_ms(_as_date(date), tz)

    hour_starts = np.arange(start, end, HOUR_MS, dtype=np.int64)
    labels = np.floor(to_local_clock_hours(hour_starts, tz)).astype(int)

    per_label: dict[int, pd.Series] = {}

    for hour_start, label in zip(hour_starts, labels, strict=True):
        by_place = _time_by_place(a, int(hour_start), int(min(hour_start + HOUR_MS, end)))

        if label in per_label:
            by_place = per_label[label].add(by_place, fill_value=0)

        per_label[label] = by_place

    sequence = {}

    for label, by_place in per_label.items():
        if by_place.sum() >= min_hour_observed_ms:
            sequence[int(label)] = int(by_place.sort_index().idxmax())

    return sequence


def routine_index(
    assignments: pd.DataFrame,
    *,
    date: dt.date | str,
    tz: str,
    window_days: int = 7,
    min_shared_hours: int = 12,
    min_days: int = 3,
    min_hour_observed_ms: float = 1_800_000,
) -> dict:
    """Mean hour-by-hour place agreement with the previous days.

    Parameters
    ----------
    assignments
        Episode-wide place assignments.
    date
        Target local date.
    tz
        IANA timezone.
    window_days
        Number of preceding days compared (7).
    min_shared_hours
        Shared observed hours for a comparison day to count (12).
    min_days
        Valid comparison days required (3).
    min_hour_observed_ms
        Assigned time for an hour to receive a place (30 min).

    Returns
    -------
    dict
        ``routine_index`` with ``_status``, ``routine_n_days`` and
        ``routine_mean_shared_hours``.
    """

    if window_days < 1 or min_days < 1 or min_shared_hours < 1:
        raise ValueError("window_days, min_days, min_shared_hours must be >= 1")

    day = _as_date(date)

    a = _clean_assignments(assignments)

    out = {
        "routine_index": np.nan,
        "routine_index_status": FeatureStatus.NO_RAW_DATA.value,
        "routine_n_days": 0,
        "routine_mean_shared_hours": np.nan,
    }

    if a.empty:
        return out

    target = hourly_place_sequence(
        a, date=day, tz=tz, min_hour_observed_ms=min_hour_observed_ms
    )

    similarities = []
    shared_counts = []

    for offset in range(1, window_days + 1):
        other = hourly_place_sequence(
            a,
            date=day - dt.timedelta(days=offset),
            tz=tz,
            min_hour_observed_ms=min_hour_observed_ms,
        )

        shared = sorted(set(target) & set(other))

        if len(shared) < min_shared_hours:
            continue

        same = sum(target[h] == other[h] for h in shared)
        similarities.append(same / len(shared))
        shared_counts.append(len(shared))

    out["routine_n_days"] = len(similarities)

    if len(similarities) < min_days:
        out["routine_index_status"] = (
            FeatureStatus.INSUFFICIENT_OBSERVABILITY.value
        )
        return out

    value = float(np.mean(similarities))

    out["routine_index"] = value
    out["routine_index_status"] = (
        FeatureStatus.OBSERVED_ZERO if value == 0.0 else FeatureStatus.OBSERVED
    ).value
    out["routine_mean_shared_hours"] = float(np.mean(shared_counts))

    return out
