"""RAPHE-native GPS interval construction.

Consecutive GPS observations are converted into explicit temporal edges.

An edge is considered temporally supported only when the time difference
between the two observations is <= max_gap_ms.

The module does not infer behavior. It only describes what portion of the
trajectory is sufficiently observed to support downstream feature estimation.

No Cortex functions are used.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance between two WGS84 coordinates."""

    values = [
        lat1,
        lon1,
        lat2,
        lon2,
    ]

    if any(
        not math.isfinite(float(x))
        for x in values
    ):
        return float("nan")

    lat1_rad = math.radians(
        float(lat1)
    )

    lat2_rad = math.radians(
        float(lat2)
    )

    dlat = math.radians(
        float(lat2)
        - float(lat1)
    )

    dlon = math.radians(
        float(lon2)
        - float(lon1)
    )

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    c = (
        2
        * math.atan2(
            math.sqrt(a),
            math.sqrt(
                1 - a
            ),
        )
    )

    return (
        EARTH_RADIUS_KM
        * c
    )


def build_gps_intervals(
    gps: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    max_gap_ms: int,
) -> pd.DataFrame:
    """Convert irregular GPS observations to consecutive temporal edges.

    Parameters
    ----------
    gps
        DataFrame containing at least timestamp, latitude, longitude.
    window_start_ms
        Inclusive beginning of analysis window.
    window_end_ms
        Exclusive end of analysis window.
    max_gap_ms
        Maximum temporal separation for an edge to be considered supported.

    Returns
    -------
    pandas.DataFrame
        One row per pair of consecutive unique-timestamp observations.

        Important fields include:

        - start_ms
        - end_ms
        - duration_ms
        - gap_ms
        - supported
        - distance_km
        - speed_kmh
        - start/end coordinates
        - start/end positional accuracy when available
    """

    required = {
        "timestamp",
        "latitude",
        "longitude",
    }

    missing = (
        required
        - set(
            gps.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing required GPS columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    window_start_ms = int(
        window_start_ms
    )

    window_end_ms = int(
        window_end_ms
    )

    max_gap_ms = int(
        max_gap_ms
    )

    if (
        window_end_ms
        <= window_start_ms
    ):
        raise ValueError(
            "window_end_ms must be greater than window_start_ms"
        )

    if max_gap_ms <= 0:
        raise ValueError(
            "max_gap_ms must be greater than zero"
        )

    columns = [
        "start_ms",
        "end_ms",
        "duration_ms",
        "gap_ms",
        "supported",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "distance_km",
        "speed_kmh",
        "start_accuracy",
        "end_accuracy",
    ]

    if gps.empty:
        return pd.DataFrame(
            columns=columns
        )

    x = gps.copy()

    for col in [
        "timestamp",
        "latitude",
        "longitude",
    ]:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    if "accuracy" in x.columns:
        x["accuracy"] = pd.to_numeric(
            x["accuracy"],
            errors="coerce",
        )

    else:
        x["accuracy"] = np.nan

    x = (
        x.dropna(
            subset=[
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .copy()
    )

    x = x[
        (
            x["timestamp"]
            >= window_start_ms
        )
        & (
            x["timestamp"]
            < window_end_ms
        )
    ].copy()

    if x.empty:
        return pd.DataFrame(
            columns=columns
        )

    x["timestamp"] = (
        x["timestamp"]
        .astype("int64")
    )

    # Duplicate timestamps cannot contribute additional temporal support.
    # Keep one deterministic observation per timestamp.
    x = (
        x.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "timestamp",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    if len(x) < 2:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    for left, right in zip(
        x.iloc[:-1].itertuples(
            index=False
        ),
        x.iloc[1:].itertuples(
            index=False
        ),
    ):

        start_ms = int(
            left.timestamp
        )

        end_ms = int(
            right.timestamp
        )

        gap_ms = (
            end_ms
            - start_ms
        )

        supported = (
            gap_ms
            <= max_gap_ms
        )

        distance_km = haversine_km(
            left.latitude,
            left.longitude,
            right.latitude,
            right.longitude,
        )

        duration_hours = (
            gap_ms
            / 3_600_000
        )

        if (
            duration_hours > 0
            and math.isfinite(
                distance_km
            )
        ):
            speed_kmh = (
                distance_km
                / duration_hours
            )

        else:
            speed_kmh = np.nan

        rows.append({
            "start_ms":
                start_ms,

            "end_ms":
                end_ms,

            "duration_ms":
                gap_ms,

            "gap_ms":
                gap_ms,

            "supported":
                bool(
                    supported
                ),

            "start_latitude":
                float(
                    left.latitude
                ),

            "start_longitude":
                float(
                    left.longitude
                ),

            "end_latitude":
                float(
                    right.latitude
                ),

            "end_longitude":
                float(
                    right.longitude
                ),

            "distance_km":
                distance_km,

            "speed_kmh":
                speed_kmh,

            "start_accuracy":
                (
                    float(
                        left.accuracy
                    )
                    if pd.notna(
                        left.accuracy
                    )
                    else np.nan
                ),

            "end_accuracy":
                (
                    float(
                        right.accuracy
                    )
                    if pd.notna(
                        right.accuracy
                    )
                    else np.nan
                ),
        })

    return pd.DataFrame(
        rows,
        columns=columns,
    )
