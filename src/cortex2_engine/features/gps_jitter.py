"""Pre-interval GPS point filtering: invalid fixes, poor accuracy, spikes.

Construct
---------
Removes GPS fixes that cannot be positions of the participant, *before*
``build_gps_intervals`` turns points into edges. Every removal is counted by
reason, so the filter is auditable and degradation experiments can report how
much evidence it removed.

Reasons
-------
``missing_coordinates``
    Latitude or longitude absent, or timestamp absent.
``invalid_coordinates``
    Outside [-90, 90] / [-180, 180], or exactly (0, 0) ("null island", a
    common placeholder for a failed fix).
``accuracy_exceeds_max``
    Reported horizontal accuracy worse than ``accuracy_max_m``. Fixes with no
    accuracy are kept, because they cannot be judged, and are counted in
    ``n_accuracy_missing``.
``speed_spike``
    A single fix B between neighbours A and C such that the implied speed
    A->B and B->C both exceed ``max_speed_kmh``: the classic out-and-back
    jump. A sustained fast trajectory (A->B fast, B->C plausible) is kept;
    ``gps_trajectory_plausibility`` judges it later at the edge level.

What this is NOT
----------------
Not smoothing (no position is moved) and not stay detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_geometry import haversine_m
from cortex2_engine.time import coerce_timestamp_ms

DROP_REASONS = (
    "missing_coordinates",
    "invalid_coordinates",
    "accuracy_exceeds_max",
    "speed_spike",
)

MS_PER_HOUR = 3_600_000.0
M_PER_KM = 1000.0


def _speed_kmh(
    lat_a: np.ndarray,
    lon_a: np.ndarray,
    t_a: np.ndarray,
    lat_b: np.ndarray,
    lon_b: np.ndarray,
    t_b: np.ndarray,
) -> np.ndarray:

    distance_km = haversine_m(lat_a, lon_a, lat_b, lon_b) / M_PER_KM
    hours = (t_b - t_a) / MS_PER_HOUR

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hours > 0, distance_km / hours, np.inf)


def filter_gps_jitter(
    gps: pd.DataFrame,
    *,
    accuracy_max_m: float = 100.0,
    max_speed_kmh: float = 250.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop invalid, inaccurate and spike GPS fixes, counting each reason.

    Parameters
    ----------
    gps
        Canonical GPS rows (``timestamp_ms``, ``latitude``, ``longitude``,
        optional ``accuracy_m``; legacy ``timestamp`` / ``accuracy``
        accepted).
    accuracy_max_m
        Largest acceptable reported accuracy radius in metres.
    max_speed_kmh
        Speed above which a single out-and-back fix is a spike (250 km/h
        rejects jumps while keeping highway and rail travel).

    Returns
    -------
    tuple
        ``(filtered, n_dropped_by_reason)``. ``filtered`` keeps the input
        columns and order of the retained rows. The dict has one count per
        ``DROP_REASONS`` entry plus ``n_input``, ``n_kept`` and
        ``n_accuracy_missing``. Reasons are assigned in the order listed, so
        each dropped row is counted once.
    """

    if accuracy_max_m <= 0 or max_speed_kmh <= 0:
        raise ValueError("accuracy_max_m and max_speed_kmh must be > 0")

    x = coerce_timestamp_ms(gps, caller="filter_gps_jitter")

    missing = {"latitude", "longitude"} - set(x.columns)

    if missing:
        raise ValueError(
            "Missing required GPS columns: " + ", ".join(sorted(missing))
        )

    counts = {reason: 0 for reason in DROP_REASONS}
    counts["n_input"] = len(x)
    counts["n_accuracy_missing"] = 0

    if x.empty:
        counts["n_kept"] = 0
        return gps.iloc[0:0].copy(), counts

    t = pd.to_numeric(x["timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(x["latitude"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(x["longitude"], errors="coerce").to_numpy(dtype=float)

    accuracy_column = (
        "accuracy_m"
        if "accuracy_m" in x.columns
        else ("accuracy" if "accuracy" in x.columns else None)
    )

    accuracy = (
        pd.to_numeric(x[accuracy_column], errors="coerce").to_numpy(dtype=float)
        if accuracy_column
        else np.full(len(x), np.nan)
    )

    keep = np.ones(len(x), dtype=bool)

    missing_mask = ~(np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon))
    counts["missing_coordinates"] = int(missing_mask.sum())
    keep &= ~missing_mask

    invalid = keep & (
        (np.abs(lat) > 90)
        | (np.abs(lon) > 180)
        | ((lat == 0) & (lon == 0))
    )
    counts["invalid_coordinates"] = int(invalid.sum())
    keep &= ~invalid

    counts["n_accuracy_missing"] = int((keep & ~np.isfinite(accuracy)).sum())

    inaccurate = keep & np.isfinite(accuracy) & (accuracy > accuracy_max_m)
    counts["accuracy_exceeds_max"] = int(inaccurate.sum())
    keep &= ~inaccurate

    # Spike pass over the surviving fixes in time order.
    index = np.flatnonzero(keep)
    index = index[np.argsort(t[index], kind="mergesort")]

    if index.size >= 3:
        a, b, c = index[:-2], index[1:-1], index[2:]

        into = _speed_kmh(lat[a], lon[a], t[a], lat[b], lon[b], t[b])
        out_of = _speed_kmh(lat[b], lon[b], t[b], lat[c], lon[c], t[c])
        across = _speed_kmh(lat[a], lon[a], t[a], lat[c], lon[c], t[c])

        spike = (into > max_speed_kmh) & (out_of > max_speed_kmh) & (
            across <= max_speed_kmh
        )

        counts["speed_spike"] = int(spike.sum())
        keep[b[spike]] = False

    counts["n_kept"] = int(keep.sum())

    return gps.iloc[np.flatnonzero(keep)].copy(), counts
