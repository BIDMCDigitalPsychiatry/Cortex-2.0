"""Circadian movement: 24 h periodicity of location (Saeb et al. 2015).

Construct
---------
How strongly the participant's position follows a 24-hour cycle over a
multi-day window. Regular daily routines (home at night, elsewhere by day)
give high values.

Estimator
---------
Saeb et al. (2015) apply least-squares spectral analysis to latitude and
longitude separately, take the spectral energy in the 23.5–24.5 h band and
report ``log(E_lat + E_lon)``. Here:

1. Points are the midpoints of supported GPS edges, weighted by edge
   duration (observed time, rule 1). They are projected to local east/north
   metres (:func:`gps_geometry.project_local_m`). Each axis is a linear
   rescaling of latitude or longitude, and the normalised power below is
   scale-invariant, so this matches the lat/lon formulation while keeping
   the fit numerically well conditioned.
2. For each period ``P`` on an evenly spaced grid of ``n_periods`` values in
   ``period_band_h``, a weighted least-squares fit of
   ``c0 + c1 cos(2 pi t / P) + c2 sin(2 pi t / P)`` gives the share of
   weighted variance explained, ``R²(P)``. This is the normalised
   Lomb–Scargle power with a floating mean.
3. ``circadian_energy_<axis>`` is the mean ``R²`` over the band, in [0, 1].
4. ``circadian_movement = log(E_east + E_north)``.

An axis with no variance (no movement along it) has energy 0. When both axes
are still the feature is ``not_applicable``: a 24 h rhythm is undefined
without movement. This is not zero and not missing.

Minimum observability: ``min_supported_ms`` of supported time (12 h) *and* a
time span of at least ``min_span_ms`` (24 h). A 24 h sinusoid cannot be
identified from less than one cycle.

What this is NOT
----------------
Not the accelerometer IS/IV (``acc_circadian``) and not a sleep measure.
Values from windows of different lengths are not directly comparable, so
keep the window fixed within a study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_geometry import edge_midpoints, project_local_m
from cortex2_engine.features.status import (
    FeatureStatus,
    gated_value,
    status_from_counts,
)

HOUR_MS = 3_600_000


def weighted_sinusoid_r2(
    t_ms: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    period_ms: float,
) -> float:
    """Weighted share of variance explained by one sinusoid plus mean.

    Parameters
    ----------
    t_ms
        Sample times in milliseconds.
    y
        Values.
    weights
        Non-negative weights.
    period_ms
        Sinusoid period.

    Returns
    -------
    float
        ``R²`` in [0, 1]; 0 when ``y`` has no weighted variance.
    """

    t = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    mean = np.average(y, weights=w)
    total = float(np.sum(w * (y - mean) ** 2))

    if total <= 0:
        return 0.0

    phase = 2.0 * np.pi * (t - t.min()) / period_ms
    design = np.column_stack([np.ones_like(t), np.cos(phase), np.sin(phase)])

    root_w = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)

    residual = float(np.sum(w * (y - design @ coef) ** 2))

    return float(np.clip(1.0 - residual / total, 0.0, 1.0))


def compute_circadian_movement(
    edges: pd.DataFrame,
    *,
    period_band_h: tuple[float, float] = (23.5, 24.5),
    n_periods: int = 11,
    min_supported_ms: float = 43_200_000,
    min_span_ms: float = 86_400_000,
) -> dict:
    """Saeb et al. (2015) circadian movement over a multi-day window.

    Parameters
    ----------
    edges
        ``build_gps_intervals`` edges for the whole window (several days).
    period_band_h
        Period band in hours (23.5–24.5).
    n_periods
        Periods evaluated across the band.
    min_supported_ms
        Minimum supported GPS time (12 h).
    min_span_ms
        Minimum span between first and last supported midpoint (24 h).

    Returns
    -------
    dict
        ``circadian_movement`` with ``_status``, plus
        ``circadian_energy_east``, ``circadian_energy_north``,
        ``circadian_span_ms`` and ``supported_duration_ms``.
    """

    low, high = period_band_h

    if not 0 < low <= high or n_periods < 1:
        raise ValueError("period_band_h must be 0 < low <= high; n_periods >= 1")

    lat, lon, mid_ms, duration = edge_midpoints(edges)
    supported_ms = float(duration.sum())
    span_ms = float(mid_ms.max() - mid_ms.min()) if mid_ms.size else 0.0

    out = {
        "circadian_movement": np.nan,
        "circadian_energy_east": np.nan,
        "circadian_energy_north": np.nan,
        "circadian_span_ms": span_ms,
        "supported_duration_ms": supported_ms,
    }

    status = status_from_counts(
        supported_ms,
        min_supported_ms,
        1.0,
        n_records=len(edges),
    )

    if status is FeatureStatus.OBSERVED and span_ms < min_span_ms:
        status = FeatureStatus.INSUFFICIENT_OBSERVABILITY

    if status is not FeatureStatus.OBSERVED:
        out["circadian_movement_status"] = status.value
        return out

    lat0 = float(np.average(lat, weights=duration))
    lon0 = float(np.average(lon, weights=duration))
    east, north = project_local_m(lat, lon, lat0_deg=lat0, lon0_deg=lon0)

    periods_ms = np.linspace(low, high, n_periods) * HOUR_MS

    energies = {}

    for axis, values in (("east", east), ("north", north)):
        energies[axis] = float(
            np.mean([
                weighted_sinusoid_r2(mid_ms, values, duration, period_ms=p)
                for p in periods_ms
            ])
        )

    out["circadian_energy_east"] = energies["east"]
    out["circadian_energy_north"] = energies["north"]

    total = energies["east"] + energies["north"]
    value = float(np.log(total)) if total > 0 else np.nan

    status = status_from_counts(supported_ms, min_supported_ms, value)

    out["circadian_movement"] = gated_value(value, status)
    out["circadian_movement_status"] = status.value

    return out
