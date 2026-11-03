"""Participant-level home inference and daily home time.

Construct
---------
``home`` is a participant-episode-level latent location, the place where the
participant repeatedly spends the night, as the GPS design document requires
(section 4). It is *not* the dominant place of a day, which is what legacy
Cortex's ``rank == 0`` significant location measures.

Inference (:func:`infer_home_location`)
---------------------------------------
1. Reference period: the first ``reference_days`` local days that contain
   assignments.
2. Each local night (``night_start_local`` to ``night_end_local``) is
   *valid* when the candidate-place assignments cover at least
   ``min_night_observed_ms`` of it (time-weighted, never point counts).
3. A valid night's place is the place with the most observed night time.
4. Home is the place that wins the most valid nights;
   ``home_confidence = nights won / valid nights``.
5. ``home_status`` is ``observed`` when there are at least ``min_nights``
   valid nights and ``home_confidence >= min_confidence``; otherwise
   ``insufficient_observability`` with ``home_reason`` saying why.

Daily home time (:func:`compute_daily_home_time`)
-------------------------------------------------
Supported GPS edge time within ``home_radius_m`` of home, where
``home_radius_m = max(r_min_m, place support radius)``. An edge with both
endpoints inside counts fully; with exactly one endpoint inside it counts
half (a departure or arrival).

What this is NOT
----------------
Not a residential address and not "time not travelling". Unobserved time is
never assumed to be home (rule 1).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_geometry import haversine_m, supported_intervals
from cortex2_engine.features.screen_bouts import local_clock_intervals
from cortex2_engine.features.status import (
    FeatureStatus,
    gated_value,
    status_from_counts,
)
from cortex2_engine.quality.completeness import window_weight
from cortex2_engine.time import local_dates, local_day_bounds_ms


def infer_home_location(
    assignments: pd.DataFrame,
    places: pd.DataFrame,
    *,
    tz: str,
    reference_days: int = 14,
    min_nights: int = 5,
    night_start_local: str = "00:00",
    night_end_local: str = "06:00",
    min_night_observed_ms: float = 7_200_000,
    min_confidence: float = 0.5,
    r_min_m: float = 50.0,
) -> dict:
    """Infer one participant-episode home location from night occupancy.

    Parameters
    ----------
    assignments
        ``build_place_candidates`` assignments for the whole episode
        (``episode_start_ms``, ``episode_end_ms``, ``place_candidate_id``).
        Place ids must be shared across days, so cluster the episode
        together.
    places
        ``build_place_candidates`` places (``place_candidate_id``,
        ``representative_latitude``, ``representative_longitude``, optional
        ``seed_support_radius_m``).
    tz
        IANA timezone.
    reference_days
        Number of local days, from the first day with data, used as the
        reference period.
    min_nights
        Minimum valid nights.
    night_start_local, night_end_local
        Local night interval (00:00–06:00).
    min_night_observed_ms
        Assigned time a night needs to be valid (2 h).
    min_confidence
        Minimum share of valid nights won by the home place.
    r_min_m
        Floor for ``home_radius_m``.

    Returns
    -------
    dict
        ``home_place_candidate_id``, ``home_latitude``, ``home_longitude``,
        ``home_radius_m``, ``home_confidence``, ``home_n_valid_nights``,
        ``home_n_nights_at_home``, ``home_status``, ``home_reason``,
        ``reference_start_date`` and ``reference_end_date``.
    """

    if reference_days < 1 or min_nights < 1:
        raise ValueError("reference_days and min_nights must be >= 1")

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    for name, frame, columns in (
        (
            "assignments",
            assignments,
            {"episode_start_ms", "episode_end_ms", "place_candidate_id"},
        ),
        (
            "places",
            places,
            {
                "place_candidate_id",
                "representative_latitude",
                "representative_longitude",
            },
        ),
    ):
        missing = columns - set(frame.columns)

        if missing:
            raise ValueError(
                f"Missing required {name} columns: "
                + ", ".join(sorted(missing))
            )

    out = {
        "home_place_candidate_id": np.nan,
        "home_latitude": np.nan,
        "home_longitude": np.nan,
        "home_radius_m": np.nan,
        "home_confidence": np.nan,
        "home_n_valid_nights": 0,
        "home_n_nights_at_home": 0,
        "home_status": FeatureStatus.NO_RAW_DATA.value,
        "home_reason": "no_assignments",
        "reference_start_date": None,
        "reference_end_date": None,
    }

    a = assignments[
        ["episode_start_ms", "episode_end_ms", "place_candidate_id"]
    ].apply(pd.to_numeric, errors="coerce").dropna()

    a = a[a["episode_end_ms"] > a["episode_start_ms"]]

    if a.empty:
        return out

    first_date = local_dates([int(a["episode_start_ms"].min())], tz)[0]
    last_date = first_date + dt.timedelta(days=reference_days - 1)

    out["reference_start_date"] = first_date.isoformat()
    out["reference_end_date"] = last_date.isoformat()

    ref_start, _ = local_day_bounds_ms(first_date, tz)
    _, ref_end = local_day_bounds_ms(last_date, tz)

    nights = local_clock_intervals(
        ref_start,
        ref_end,
        tz=tz,
        start_local=night_start_local,
        end_local=night_end_local,
    )

    starts = a["episode_start_ms"].to_numpy(dtype=float)
    ends = a["episode_end_ms"].to_numpy(dtype=float)
    place_ids = a["place_candidate_id"].to_numpy(dtype=int)

    winners = []

    for night_start, night_end in nights:
        overlap = np.clip(
            np.minimum(ends, night_end) - np.maximum(starts, night_start),
            0,
            None,
        )

        if overlap.sum() < min_night_observed_ms:
            continue

        by_place = pd.Series(overlap).groupby(place_ids).sum()
        by_place = by_place[by_place > 0]

        # Most night time; ties go to the lowest place id (deterministic).
        winners.append(int(by_place.sort_index().idxmax()))

    n_valid = len(winners)
    out["home_n_valid_nights"] = n_valid

    if n_valid == 0:
        out["home_status"] = FeatureStatus.INSUFFICIENT_OBSERVABILITY.value
        out["home_reason"] = "no_valid_nights"
        return out

    tally = pd.Series(winners).value_counts().sort_index()
    home_id = int(tally.idxmax())
    n_home = int(tally.max())
    confidence = n_home / n_valid

    out["home_n_nights_at_home"] = n_home
    out["home_confidence"] = confidence

    place = places[
        pd.to_numeric(places["place_candidate_id"], errors="coerce") == home_id
    ]

    if place.empty:
        raise ValueError(
            f"place_candidate_id {home_id} missing from places"
        )

    place = place.iloc[0]
    support = float(place.get("seed_support_radius_m", np.nan))

    out["home_place_candidate_id"] = home_id
    out["home_latitude"] = float(place["representative_latitude"])
    out["home_longitude"] = float(place["representative_longitude"])
    out["home_radius_m"] = (
        max(r_min_m, support) if np.isfinite(support) else float(r_min_m)
    )

    if n_valid < min_nights:
        out["home_status"] = FeatureStatus.INSUFFICIENT_OBSERVABILITY.value
        out["home_reason"] = "too_few_valid_nights"

    elif confidence < min_confidence:
        out["home_status"] = FeatureStatus.INSUFFICIENT_OBSERVABILITY.value
        out["home_reason"] = "inconsistent_night_place"

    else:
        out["home_status"] = FeatureStatus.OBSERVED.value
        out["home_reason"] = None

    return out


def compute_daily_home_time(
    edges: pd.DataFrame,
    home: dict,
    *,
    r_min_m: float = 50.0,
    min_supported_ms: float = 21_600_000,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
) -> dict:
    """Observed time at home for one participant-day.

    Parameters
    ----------
    edges
        ``build_gps_intervals`` edges for the day (``supported``,
        ``duration_ms``, start/end coordinates).
    home
        Output of :func:`infer_home_location`.
    r_min_m
        Minimum home radius (used when the home record has none).
    min_supported_ms
        Minimum supported GPS time for the day (6 h).
    window_start_ms, window_end_ms
        Optional day bounds for ``completeness_weight`` (NaN without them).

    Returns
    -------
    dict
        ``home_time_ms`` and ``home_time_fraction`` (of supported time), each
        with ``_status``, plus ``supported_duration_ms``. When home is not
        established, both are NaN with the home status.
    """

    required = {
        "supported",
        "duration_ms",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
    }

    missing = required - set(edges.columns)

    if missing:
        raise ValueError(
            "Missing required GPS edge columns: " + ", ".join(sorted(missing))
        )

    x = edges[edges["supported"].fillna(False).astype(bool)].copy()

    for column in required - {"supported"}:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    x = x[(x["duration_ms"] > 0)].dropna(subset=list(required - {"supported"}))

    supported_ms = float(x["duration_ms"].sum())
    weight = window_weight(
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        intervals=supported_intervals(edges) if window_start_ms is not None else None,
    )

    home_status = FeatureStatus(home.get("home_status", "no_raw_data"))

    if home_status is not FeatureStatus.OBSERVED:
        status = (
            FeatureStatus.INSUFFICIENT_OBSERVABILITY
            if home_status is FeatureStatus.NO_RAW_DATA
            else home_status
        )

        return {
            "home_time_ms": np.nan,
            "home_time_ms_status": status.value,
            "home_time_fraction": np.nan,
            "home_time_fraction_status": status.value,
            "supported_duration_ms": supported_ms,
            "completeness_weight": weight,
        }

    radius = float(home.get("home_radius_m") or r_min_m)
    radius = max(radius, r_min_m)

    inside_start = haversine_m(
        home["home_latitude"],
        home["home_longitude"],
        x["start_latitude"].to_numpy(),
        x["start_longitude"].to_numpy(),
    ) <= radius

    inside_end = haversine_m(
        home["home_latitude"],
        home["home_longitude"],
        x["end_latitude"].to_numpy(),
        x["end_longitude"].to_numpy(),
    ) <= radius

    weight = (inside_start.astype(float) + inside_end.astype(float)) / 2.0
    home_ms = float((weight * x["duration_ms"].to_numpy()).sum())

    fraction = home_ms / supported_ms if supported_ms > 0 else np.nan

    out = {}

    for key, value in (
        ("home_time_ms", home_ms),
        ("home_time_fraction", fraction),
    ):
        status = status_from_counts(
            supported_ms,
            min_supported_ms,
            value,
            n_records=len(edges),
        )

        out[key] = gated_value(value, status)
        out[f"{key}_status"] = status.value

    out["supported_duration_ms"] = supported_ms
    out["completeness_weight"] = weight

    return out
