"""Daily activity space: convex hull and standard deviational ellipse.

Construct
---------
The geographic extent of the places a participant occupied during one local
day, estimated from supported GPS edges only.

Estimators
----------
Points are the midpoints of supported ``build_gps_intervals`` edges, weighted
by edge duration (observed time, rule 1). They are projected to local metres
with :func:`gps_geometry.project_local_m` around their duration-weighted
centre, using ``cos(radians(lat0))``. Legacy Cortex took the cosine of a
latitude in degrees.

- :func:`convex_hull_area_km2` — Andrew's monotone-chain hull and the
  shoelace area. Unweighted by construction.
- :func:`standard_deviational_ellipse` — the duration-weighted 2x2 covariance
  of the projected points, eigendecomposed in closed form. Semi-axes are
  ``sqrt(eigenvalue)`` (one standard deviation),
  ``sde_area_km2 = pi * a * b``, ``anisotropy = 1 - b / a`` (0 = circle) and
  ``major_axis_bearing_deg`` is the compass bearing of the major axis in
  [0, 180).

Zero handling (rule 2)
----------------------
An area below the squared median GPS accuracy radius is indistinguishable
from staying still, so it is emitted as ``0.0`` with status
``observed_zero``. An area is never zero just because data are missing. With
fewer than 3 distinct points the hull is degenerate; its area is
``observed_zero`` when supported time is sufficient.

What this is NOT
----------------
Not radius of gyration (``gps_mobility_features``) and not a multi-day home
range. The ellipse is a descriptive spread measure, not a confidence region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_geometry import (
    edge_midpoints,
    project_local_m,
    supported_intervals,
)
from cortex2_engine.features.status import (
    FeatureStatus,
    gated_value,
    status_from_counts,
)
from cortex2_engine.quality.completeness import window_weight

M2_PER_KM2 = 1_000_000.0

ACTIVITY_SPACE_KEYS = [
    "convex_hull_area_km2",
    "sde_area_km2",
    "anisotropy",
    "major_axis_bearing_deg",
]


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:

    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points_xy: np.ndarray) -> np.ndarray:
    """Convex hull vertices (counter-clockwise) by Andrew's monotone chain.

    Parameters
    ----------
    points_xy
        ``(n, 2)`` array of planar coordinates.

    Returns
    -------
    numpy.ndarray
        Hull vertices, without repeating the first vertex. Fewer than three
        rows when the points are degenerate (all identical or collinear).
    """

    pts = np.unique(np.asarray(points_xy, dtype=float), axis=0)

    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower: list[np.ndarray] = []

    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []

    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1])


def polygon_area(vertices_xy: np.ndarray) -> float:
    """Shoelace area of a simple polygon (0 when fewer than 3 vertices)."""

    v = np.asarray(vertices_xy, dtype=float)

    if len(v) < 3:
        return 0.0

    x, y = v[:, 0], v[:, 1]

    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def convex_hull_area_km2(points_xy_m: np.ndarray) -> float:
    """Convex hull area in km² of projected points given in metres."""

    return polygon_area(convex_hull(points_xy_m)) / M2_PER_KM2


def standard_deviational_ellipse(
    points_xy_m: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Weighted standard deviational ellipse of projected points.

    Parameters
    ----------
    points_xy_m
        ``(n, 2)`` east/north metres.
    weights
        Non-negative weights (observed duration). Uniform when ``None``.

    Returns
    -------
    dict
        ``sde_area_km2``, ``sde_major_m``, ``sde_minor_m``, ``anisotropy``
        and ``major_axis_bearing_deg``. NaN when the weights sum to zero or
        there are no points. ``anisotropy`` and the bearing are NaN for a
        circular (or point) ellipse, where the major axis is undefined.
    """

    pts = np.asarray(points_xy_m, dtype=float)
    nan = {
        "sde_area_km2": np.nan,
        "sde_major_m": np.nan,
        "sde_minor_m": np.nan,
        "anisotropy": np.nan,
        "major_axis_bearing_deg": np.nan,
    }

    if pts.size == 0:
        return nan

    w = np.ones(len(pts)) if weights is None else np.asarray(weights, float)

    if np.any(w < 0) or w.sum() <= 0:
        return nan

    w = w / w.sum()
    centre = w @ pts
    d = pts - centre

    sxx = float(w @ (d[:, 0] ** 2))
    syy = float(w @ (d[:, 1] ** 2))
    sxy = float(w @ (d[:, 0] * d[:, 1]))

    # Closed-form eigenvalues of [[sxx, sxy], [sxy, syy]].
    half_trace = (sxx + syy) / 2.0
    root = float(np.hypot((sxx - syy) / 2.0, sxy))
    lam_major = max(half_trace + root, 0.0)
    lam_minor = max(half_trace - root, 0.0)

    a = float(np.sqrt(lam_major))
    b = float(np.sqrt(lam_minor))

    out = {
        "sde_area_km2": np.pi * a * b / M2_PER_KM2,
        "sde_major_m": a,
        "sde_minor_m": b,
        "anisotropy": np.nan,
        "major_axis_bearing_deg": np.nan,
    }

    # Relative tolerance: a circle has root == 0 up to rounding.
    if a > 0 and root > 1e-12 * max(half_trace, 1e-300):
        out["anisotropy"] = 1.0 - b / a

        # Angle of the major eigenvector from east, counter-clockwise.
        theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)

        # Compass bearing: clockwise from north, axis so modulo 180.
        out["major_axis_bearing_deg"] = float(
            (90.0 - np.degrees(theta)) % 180.0
        )

    return out


def compute_daily_activity_space(
    edges: pd.DataFrame,
    *,
    min_supported_ms: float = 21_600_000,
    median_accuracy_m: float | None = None,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
) -> dict:
    """Activity-space features for one participant-day.

    Parameters
    ----------
    edges
        ``build_gps_intervals`` edges for the day.
    min_supported_ms
        Minimum supported GPS time (6 h).
    median_accuracy_m
        Median reported accuracy radius. When ``None`` it is the median of
        the edges' ``start_accuracy``/``end_accuracy``; if those are absent
        the ``observed_zero`` floor is not applied.
    window_start_ms, window_end_ms
        Optional day bounds for ``completeness_weight`` (NaN without them).

    Returns
    -------
    dict
        ``ACTIVITY_SPACE_KEYS`` values with ``_status``, plus
        ``sde_major_m``, ``sde_minor_m``, ``activity_space_n_points``,
        ``supported_duration_ms`` and ``zero_area_floor_km2``.
    """

    lat, lon, _, duration = edge_midpoints(edges)
    supported_ms = float(duration.sum())
    n_points = int(lat.size)

    if median_accuracy_m is None:
        accuracy = pd.concat(
            [
                pd.to_numeric(edges.get(column), errors="coerce")
                for column in ("start_accuracy", "end_accuracy")
                if column in edges.columns
            ]
            or [pd.Series(dtype=float)]
        ).dropna()

        median_accuracy_m = float(accuracy.median()) if len(accuracy) else None

    floor_km2 = (
        float(median_accuracy_m) ** 2 / M2_PER_KM2
        if median_accuracy_m is not None and np.isfinite(median_accuracy_m)
        else 0.0
    )

    hull = np.nan
    sde = standard_deviational_ellipse(np.empty((0, 2)))

    if n_points:
        lat0 = float(np.average(lat, weights=duration))
        lon0 = float(np.average(lon, weights=duration))
        east, north = project_local_m(lat, lon, lat0_deg=lat0, lon0_deg=lon0)
        xy = np.column_stack([east, north])

        hull = convex_hull_area_km2(xy)
        sde = standard_deviational_ellipse(xy, duration)

    out: dict = {}

    for key, value, is_area in (
        ("convex_hull_area_km2", hull, True),
        ("sde_area_km2", sde["sde_area_km2"], True),
        ("anisotropy", sde["anisotropy"], False),
        ("major_axis_bearing_deg", sde["major_axis_bearing_deg"], False),
    ):
        if is_area and np.isfinite(value) and value <= floor_km2:
            value = 0.0

        status = status_from_counts(
            supported_ms,
            min_supported_ms,
            value,
            n_records=len(edges),
        )

        # A NaN shape with enough support (circle, single point) comes back
        # as not_applicable: undefined, not missing.
        out[key] = gated_value(value, status)
        out[f"{key}_status"] = status.value

    # Orientation of a sub-noise ellipse is noise: suppress it.
    if out["sde_area_km2_status"] == FeatureStatus.OBSERVED_ZERO.value:
        for key in ("anisotropy", "major_axis_bearing_deg"):
            out[key] = np.nan
            out[f"{key}_status"] = FeatureStatus.NOT_APPLICABLE.value

    out["sde_major_m"] = sde["sde_major_m"]
    out["sde_minor_m"] = sde["sde_minor_m"]
    out["activity_space_n_points"] = n_points
    out["supported_duration_ms"] = supported_ms
    out["zero_area_floor_km2"] = floor_km2
    out["completeness_weight"] = window_weight(
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        intervals=supported_intervals(edges) if window_start_ms is not None else None,
    )

    return out
