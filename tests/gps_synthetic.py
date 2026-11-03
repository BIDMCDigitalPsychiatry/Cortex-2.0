"""Synthetic GPS helpers for the Phase 5 tests."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_geometry import EARTH_RADIUS_M

BASE_LAT = 42.3370
BASE_LON = -71.1060
MIN_MS = 60_000
HOUR_MS = 3_600_000


def offset(east_m: float, north_m: float, *, lat0: float = BASE_LAT, lon0: float = BASE_LON):
    """Latitude/longitude ``east_m`` / ``north_m`` metres from the origin."""

    lat = lat0 + np.degrees(north_m / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(east_m / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lat, lon


def local_ms(date: str, hour: float, tz: str = "America/New_York") -> int:
    """Epoch ms of a local wall-clock time."""

    d = dt.date.fromisoformat(date)
    base = dt.datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(tz))
    return int((base + dt.timedelta(hours=hour)).timestamp() * 1000)


def track(start_ms: int, end_ms: int, positions, *, step_ms: int = MIN_MS, accuracy_m: float = 10.0):
    """GPS rows every ``step_ms``; ``positions(t_ms) -> (east_m, north_m)``."""

    t = np.arange(start_ms, end_ms, step_ms, dtype=np.int64)
    east, north = np.vectorize(positions)(t)
    lat, lon = offset(np.asarray(east, float), np.asarray(north, float))
    return pd.DataFrame({
        "timestamp_ms": t,
        "latitude": lat,
        "longitude": lon,
        "accuracy_m": accuracy_m,
    })


def assignment(start_ms: int, end_ms: int, place: int) -> dict:
    """One place-assignment row."""

    return {
        "episode_start_ms": start_ms,
        "episode_end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "place_candidate_id": place,
    }
