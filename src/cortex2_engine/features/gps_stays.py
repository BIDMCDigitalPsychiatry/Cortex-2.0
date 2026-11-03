"""Stays: places where the participant dwelt, with a sampling-aware threshold.

Construct
---------
A *stay* is a run of consecutive anchor-compatible location episodes
(``gps_anchor_episodes``) whose representative points remain within
``r_stay_m`` of the stay's duration-weighted centre, and whose **observed**
duration reaches the dwell threshold::

    dwell_threshold_ms = max(t_floor_ms, dwell_multiplier * sampling_interval_ms)

The dynamic term stops sparse sampling from manufacturing stays. When fixes
arrive every 5 min, two fixes 10 min apart are weak evidence of dwelling, and
the threshold rises to 15 min (3 intervals). With dense sampling the 10 min
floor applies.

Time between merged episodes (an unobserved gap) is part of the stay's *span*
but not of its *observed duration*; rule 1 keeps observed time as the unit of
evidence.

What this is NOT
----------------
Not a significant place (``gps_place_candidates`` clusters across days) and
not home (``gps_home``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_geometry import haversine_m

STAY_COLUMNS = [
    "stay_id",
    "start_ms",
    "end_ms",
    "span_ms",
    "observed_duration_ms",
    "latitude",
    "longitude",
    "n_episodes",
    "max_radius_m",
    "dwell_threshold_ms",
]


def dwell_threshold_ms(
    *,
    t_floor_ms: float,
    dwell_multiplier: float,
    sampling_interval_ms: float,
) -> float:
    """Dynamic dwell threshold ``max(t_floor_ms, multiplier * interval)``."""

    return float(max(t_floor_ms, dwell_multiplier * sampling_interval_ms))


def build_stays(
    anchor_episodes: pd.DataFrame,
    *,
    sampling_interval_ms: float,
    r_stay_m: float = 50.0,
    t_floor_ms: float = 600_000,
    dwell_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Merge consecutive nearby location episodes into stays.

    Parameters
    ----------
    anchor_episodes
        ``build_anchor_compatible_location_episodes`` output
        (``episode_start_ms``, ``episode_end_ms``, ``duration_ms``,
        ``representative_latitude``, ``representative_longitude``).
    sampling_interval_ms
        Typical GPS sampling interval for the window (for example the
        ``median_sampling_interval_ms`` from ``gps_observability``).
    r_stay_m
        Maximum distance of a member episode from the running stay centre.
    t_floor_ms
        Minimum dwell time regardless of sampling (10 min).
    dwell_multiplier
        Sampling intervals a stay must span (3).

    Returns
    -------
    pandas.DataFrame
        One row per stay (``STAY_COLUMNS``), in time order.
    """

    if not np.isfinite(sampling_interval_ms) or sampling_interval_ms <= 0:
        raise ValueError("sampling_interval_ms must be finite and > 0")

    if r_stay_m <= 0 or t_floor_ms < 0 or dwell_multiplier < 0:
        raise ValueError(
            "r_stay_m must be > 0; t_floor_ms and dwell_multiplier >= 0"
        )

    required = {
        "episode_start_ms",
        "episode_end_ms",
        "duration_ms",
        "representative_latitude",
        "representative_longitude",
    }

    missing = required - set(anchor_episodes.columns)

    if missing:
        raise ValueError(
            "Missing required episode columns: " + ", ".join(sorted(missing))
        )

    threshold = dwell_threshold_ms(
        t_floor_ms=t_floor_ms,
        dwell_multiplier=dwell_multiplier,
        sampling_interval_ms=sampling_interval_ms,
    )

    x = anchor_episodes[list(required)].apply(pd.to_numeric, errors="coerce")
    x = x.dropna()
    x = x[x["duration_ms"] > 0].sort_values("episode_start_ms", kind="mergesort")

    groups: list[dict] = []
    current: dict | None = None

    for row in x.itertuples(index=False):

        lat = float(row.representative_latitude)
        lon = float(row.representative_longitude)
        weight = float(row.duration_ms)

        if current is not None:
            distance = float(
                haversine_m(current["lat"], current["lon"], lat, lon)
            )

            if distance <= r_stay_m:
                total = current["weight"] + weight
                current["lat"] = (current["lat"] * current["weight"] + lat * weight) / total
                current["lon"] = (current["lon"] * current["weight"] + lon * weight) / total
                current["weight"] = total
                current["end"] = max(current["end"], float(row.episode_end_ms))
                current["points"].append((lat, lon))
                continue

            groups.append(current)

        current = {
            "start": float(row.episode_start_ms),
            "end": float(row.episode_end_ms),
            "lat": lat,
            "lon": lon,
            "weight": weight,
            "points": [(lat, lon)],
        }

    if current is not None:
        groups.append(current)

    rows = []

    for group in groups:

        if group["weight"] < threshold:
            continue

        points = np.asarray(group["points"])

        radius = float(
            np.max(
                haversine_m(group["lat"], group["lon"], points[:, 0], points[:, 1])
            )
        )

        rows.append({
            "start_ms": int(group["start"]),
            "end_ms": int(group["end"]),
            "span_ms": group["end"] - group["start"],
            "observed_duration_ms": group["weight"],
            "latitude": group["lat"],
            "longitude": group["lon"],
            "n_episodes": len(group["points"]),
            "max_radius_m": radius,
            "dwell_threshold_ms": threshold,
        })

    out = pd.DataFrame(rows, columns=STAY_COLUMNS[1:])
    out.insert(0, "stay_id", np.arange(len(out), dtype=int))

    return out[STAY_COLUMNS]
