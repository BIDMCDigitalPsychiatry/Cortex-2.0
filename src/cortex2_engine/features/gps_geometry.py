"""Vectorised geodesy helpers for the Phase 5 GPS modules.

Two operations, both with explicit argument order (latitude first) and units
in the names, because legacy Cortex swapped latitude/longitude in its
Haversine call and took ``cos`` of a latitude in degrees:

- :func:`haversine_m` — great-circle distance in metres between arrays of
  points (WGS84 mean radius, as ``gps_intervals.EARTH_RADIUS_KM``).
- :func:`project_local_m` — local equirectangular projection to metres around
  a reference latitude/longitude, using ``cos(radians(lat0))``. Accurate to
  well under 1 % over the tens of kilometres of a daily activity space.

What this is NOT: a general map projection. Do not use ``project_local_m``
over hundreds of kilometres or near the poles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_intervals import EARTH_RADIUS_KM

EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0


def haversine_m(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in metres, element-wise (broadcasting).

    Parameters
    ----------
    lat1_deg, lon1_deg, lat2_deg, lon2_deg
        Coordinates in decimal degrees.

    Returns
    -------
    numpy.ndarray
        Distances in metres (NaN where any input is NaN).
    """

    lat1 = np.radians(np.asarray(lat1_deg, dtype=float))
    lat2 = np.radians(np.asarray(lat2_deg, dtype=float))
    dlat = lat2 - lat1
    dlon = np.radians(
        np.asarray(lon2_deg, dtype=float) - np.asarray(lon1_deg, dtype=float)
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * EARTH_RADIUS_M * np.arctan2(
        np.sqrt(a),
        np.sqrt(np.clip(1.0 - a, 0.0, None)),
    )


def project_local_m(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project to local east/north metres around ``(lat0_deg, lon0_deg)``.

    Parameters
    ----------
    lat_deg, lon_deg
        Coordinates in decimal degrees.
    lat0_deg, lon0_deg
        Projection origin in decimal degrees.

    Returns
    -------
    tuple of numpy.ndarray
        ``(east_m, north_m)``.
    """

    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))

    dlon = (lon - lon0 + np.pi) % (2.0 * np.pi) - np.pi

    east = EARTH_RADIUS_M * dlon * np.cos(lat0)
    north = EARTH_RADIUS_M * (lat - lat0)

    return east, north


def edge_midpoints(
    edges,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint latitude/longitude, mid-time and duration of supported edges.

    Parameters
    ----------
    edges
        ``build_gps_intervals`` output (optionally with spatial semantics).

    Returns
    -------
    tuple of numpy.ndarray
        ``(lat_deg, lon_deg, mid_time_ms, duration_ms)`` for supported edges
        with finite coordinates and positive duration.
    """

    required = {
        "supported",
        "start_ms",
        "end_ms",
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

    x = edges.copy()

    for column in required - {"supported"}:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    keep = (
        x["supported"].fillna(False).astype(bool)
        & (x["duration_ms"] > 0)
        & np.isfinite(
            x[
                [
                    "start_latitude",
                    "start_longitude",
                    "end_latitude",
                    "end_longitude",
                ]
            ]
        ).all(axis=1)
    )

    x = x[keep]

    return (
        ((x["start_latitude"] + x["end_latitude"]) / 2.0).to_numpy(),
        ((x["start_longitude"] + x["end_longitude"]) / 2.0).to_numpy(),
        ((x["start_ms"] + x["end_ms"]) / 2.0).to_numpy(),
        x["duration_ms"].to_numpy(dtype=float),
    )


def supported_intervals(edges) -> list[tuple[float, float]]:
    """``(start_ms, end_ms)`` of supported edges with positive duration."""

    if edges is None or len(edges) == 0 or "supported" not in edges.columns:
        return []

    x = edges[edges["supported"].fillna(False).astype(bool)]
    start = pd.to_numeric(x["start_ms"], errors="coerce").to_numpy(float)
    end = pd.to_numeric(x["end_ms"], errors="coerce").to_numpy(float)
    keep = np.isfinite(start) & np.isfinite(end) & (end > start)

    return list(zip(start[keep], end[keep], strict=True))
