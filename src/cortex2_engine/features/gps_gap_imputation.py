"""Participant-specific GPS gap imputation for Cortex-2.0.

Imputed timelines must be processed through the same QC, stays,
trips, home-time and distance functions.

This is a candidate implementation, not yet a validated Forest replica.
"""
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

EARTH_RADIUS_M = 6371008.8


@dataclass(frozen=True)
class ImputationConfig:
    max_supported_gap_ms: int = 60_000
    m: int = 20
    seed: int = 20261117
    min_donor_fixes: int = 3
    min_template_span_ms: int = 60_000
    maximum_bridge_speed_m_s: float = 45.0
    maximum_time_stretch: float = 4.0
    endpoint_distance_tolerance_m: float = 1000.0
    stay_compatibility_multiplier: float = 1.0

    def __post_init__(self):
        if self.max_supported_gap_ms <= 0 or self.m < 2:
            raise ValueError("Invalid gap threshold or imputation count")
        if self.maximum_bridge_speed_m_s <= 0:
            raise ValueError("Invalid maximum speed")
        if self.maximum_time_stretch < 1:
            raise ValueError("Invalid time-stretch setting")


def _require_fixes(df):
    required = {"timestamp", "latitude", "longitude", "accuracy"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    out = df[list(required)].copy()
    if "imputed" in df.columns:
        out["imputed"] = df["imputed"].fillna(False).astype(bool)

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.loc[
        out.latitude.between(-90, 90)
        & out.longitude.between(-180, 180)
        & (out.accuracy > 0)
    ]
    return (
        out.sort_values("timestamp", kind="mergesort")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def _haversine(lat1, lon1, lat2, lon2):
    a1, a2 = np.deg2rad([lat1, lat2])
    dl = np.deg2rad(lon2 - lon1)
    a = (
        np.sin((a2 - a1) / 2) ** 2
        + np.cos(a1) * np.cos(a2) * np.sin(dl / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find_long_gaps(fixes, max_gap_ms):
    f = _require_fixes(fixes)
    rows = []

    for i in range(len(f) - 1):
        left, right = f.iloc[i], f.iloc[i + 1]
        gap = int(right.timestamp - left.timestamp)

        if gap > max_gap_ms:
            rows.append({
                "left_index": i,
                "right_index": i + 1,
                "start_ms": int(left.timestamp),
                "end_ms": int(right.timestamp),
                "gap_ms": gap,
                "endpoint_distance_m": _haversine(
                    left.latitude, left.longitude,
                    right.latitude, right.longitude,
                ),
            })

    return pd.DataFrame(rows, columns=[
        "left_index", "right_index", "start_ms",
        "end_ms", "gap_ms", "endpoint_distance_m"
    ])


def observed_trip_templates(
    fixes, observed_trips,
    start_col="trip_start_ms",
    end_col="trip_end_ms",
    min_fixes=3,
):
    """Build donor templates from this participant's observed trips."""
    f = _require_fixes(fixes)
    if "imputed" in f.columns:
        f = f.loc[~f["imputed"]].copy()

    if not {start_col, end_col}.issubset(observed_trips.columns):
        raise ValueError("Observed trips lack start/end timestamps")

    donors = []
    for trip in observed_trips.itertuples(index=False):
        start = getattr(trip, start_col)
        end = getattr(trip, end_col)

        subset = f.loc[f.timestamp.between(start, end)].copy()
        if len(subset) >= min_fixes:
            donors.append(subset.reset_index(drop=True))

    return donors


def _bridge(left, right, donor, cfg):
    """Transform a participant's observed trip to bridge a GPS gap."""
    dt = float(right.timestamp - left.timestamp)
    if dt <= 0:
        return None

    endpoint_m = _haversine(
        left.latitude, left.longitude,
        right.latitude, right.longitude,
    )

    if endpoint_m / (dt / 1000) > cfg.maximum_bridge_speed_m_s:
        return None

    d = _require_fixes(donor)
    if len(d) < cfg.min_donor_fixes:
        return None

    donor_dt = float(d.timestamp.iloc[-1] - d.timestamp.iloc[0])
    if donor_dt < cfg.min_template_span_ms:
        return None

    stretch = dt / donor_dt
    if not (
        1 / cfg.maximum_time_stretch
        <= stretch <= cfg.maximum_time_stretch
    ):
        return None

    tau = (
        (d.timestamp.to_numpy() - d.timestamp.iloc[0]) / donor_dt
    )

    lat0 = float(left.latitude)
    lon0 = float(left.longitude)
    coslat = np.cos(np.deg2rad(lat0))

    if abs(coslat) < 1e-8:
        return None

    x = (
        EARTH_RADIUS_M * np.deg2rad(d.longitude.to_numpy() - lon0)
        * coslat
    )
    y = EARTH_RADIUS_M * np.deg2rad(d.latitude.to_numpy() - lat0)

    x -= x[0]
    y -= y[0]

    xe = (
        EARTH_RADIUS_M * np.deg2rad(right.longitude - lon0)
        * coslat
    )
    ye = EARTH_RADIUS_M * np.deg2rad(right.latitude - lat0)

    x = x + tau * (xe - x[-1])
    y = y + tau * (ye - y[-1])

    lat = lat0 + np.rad2deg(y / EARTH_RADIUS_M)
    lon = lon0 + np.rad2deg(x / (EARTH_RADIUS_M * coslat))

    # CORTEX2_BRIDGE_RESAMPLING_V1
    # Ensure the completed gap has no interval longer than G.
    # These are synthetic positions, not additional observations.
    n_intervals = int(np.ceil(dt / cfg.max_supported_gap_ms))

    if n_intervals < 2:
        return None

    # Protect against excessive synthetic trajectory generation.
    if n_intervals > 10000:
        return None

    grid = np.linspace(0.0, 1.0, n_intervals + 1)

    # Interpolate in local metric coordinates, avoiding
    # discontinuities from direct longitude interpolation.
    x_grid = np.interp(grid, tau, x)
    y_grid = np.interp(grid, tau, y)

    time_grid = np.rint(
        left.timestamp + grid * dt
    ).astype(np.int64)

    mid = (
        (time_grid > left.timestamp)
        & (time_grid < right.timestamp)
    )

    if not mid.any():
        return None

    lat_grid = lat0 + np.rad2deg(
        y_grid / EARTH_RADIUS_M
    )
    lon_grid = lon0 + np.rad2deg(
        x_grid / (EARTH_RADIUS_M * coslat)
    )

    # Donor accuracy is retained only as a computational proxy.
    # It must not be interpreted as measured accuracy of
    # the synthetic locations.
    accuracy_grid = np.interp(
        grid,
        tau,
        d.accuracy.to_numpy(dtype=float),
    )

    return pd.DataFrame({
        "timestamp": time_grid[mid],
        "latitude": lat_grid[mid],
        "longitude": lon_grid[mid],
        "accuracy": accuracy_grid[mid],
        "imputed": True,
    })


def impute_timelines(
    fixes,
    donors,
    config=None,
    stay_donors=None,
):
    """Generate M timelines and an auditable gap-disposition table."""
    cfg = config or ImputationConfig()
    f = _require_fixes(fixes)
    gaps = find_long_gaps(f, cfg.max_supported_gap_ms)

    rng = np.random.default_rng(cfg.seed)
    stay_donors = stay_donors or []
    timelines = []
    diagnostics = []

    for imp in range(cfg.m):
        pieces = [f.assign(imputed=False)]

        for gap in gaps.itertuples(index=False):
            left = f.iloc[gap.left_index]
            right = f.iloc[gap.right_index]

            # Distinguish stationary (accuracy-compatible) gaps from trips.
            # If stationary, use only the participant's observed stay donors.
            stay_possible = gap.endpoint_distance_m <= (
                (float(left.accuracy) + float(right.accuracy))
                * cfg.stay_compatibility_multiplier
            )
            donor_set = stay_donors if stay_possible else donors
            candidates = []
            for j, donor in enumerate(donor_set):
                d = _require_fixes(donor)
                if len(d) < cfg.min_donor_fixes:
                    continue

                duration = d.timestamp.iloc[-1] - d.timestamp.iloc[0]
                if duration <= 0:
                    continue

                ratio = gap.gap_ms / duration
                if not (
                    1 / cfg.maximum_time_stretch
                    <= ratio <= cfg.maximum_time_stretch
                ):
                    continue

                distance = _haversine(
                    d.latitude.iloc[0], d.longitude.iloc[0],
                    d.latitude.iloc[-1], d.longitude.iloc[-1],
                )

                if abs(distance - gap.endpoint_distance_m) <= (
                    cfg.endpoint_distance_tolerance_m
                ):
                    candidates.append(j)

            piece = None
            chosen = None

            if candidates:
                chosen = int(rng.choice(candidates))
                piece = _bridge(left, right, donor_set[chosen], cfg)

            if piece is not None:
                pieces.append(piece)

            diagnostics.append({
                "imputation": imp,
                "start_ms": gap.start_ms,
                "end_ms": gap.end_ms,
                "gap_ms": gap.gap_ms,
                "imputed": piece is not None,
                "donor_index": chosen if piece is not None else pd.NA,
                "gap_type": "stay" if stay_possible else "trip",
            })

        completed = (
            pd.concat(pieces, ignore_index=True)
            .sort_values("timestamp", kind="mergesort")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )
        timelines.append(completed)

    return timelines, pd.DataFrame(diagnostics)


def rerun_shared_features(
    timelines: list[pd.DataFrame],
    compute_features: Callable[[pd.DataFrame], Mapping[str, float]],
):
    """Recompute all mobility outputs from each completed timeline."""
    rows = []
    for i, timeline in enumerate(timelines):
        features = dict(compute_features(timeline))
        rows.append({"imputation": i, **features})
    return pd.DataFrame(rows)


def pool_multiple_imputations(estimates, within_variances=None):
    """Pool estimates; distinguish Rubin intervals from imputation spread."""
    features = [c for c in estimates if c != "imputation"]
    rows = []

    for feature in features:
        values = pd.to_numeric(
            estimates[feature], errors="coerce"
        ).dropna().to_numpy(dtype=float)

        if len(values) < 2:
            rows.append({
                "feature": feature,
                "estimate": np.nan,
                "lower95": np.nan,
                "upper95": np.nan,
                "interval_type": "insufficient",
            })
            continue

        mean = float(values.mean())
        between = float(values.var(ddof=1))
        m = len(values)

        row = {
            "feature": feature,
            "estimate": mean,
            "between_var": between,
            "m": m,
        }

        if within_variances is None:
            row.update({
                "lower95": float(np.quantile(values, 0.025)),
                "upper95": float(np.quantile(values, 0.975)),
                "interval_type": "imputation_spread_not_rubin",
            })
        else:
            if feature not in within_variances:
                raise ValueError(f"Missing variance for {feature}")

            mask = estimates[feature].notna()
            u = pd.to_numeric(
                within_variances.loc[mask, feature],
                errors="coerce",
            ).to_numpy(dtype=float)

            if len(u) != m or not np.isfinite(u).all() or (u < 0).any():
                raise ValueError(f"Invalid within variance for {feature}")

            ubar = float(u.mean())
            total = ubar + (1 + 1 / m) * between

            if between == 0:
                df = np.inf
            else:
                df = (m - 1) * (
                    1 + ubar / ((1 + 1 / m) * between)
                ) ** 2

            critical = (
                float(student_t.ppf(0.975, df))
                if np.isfinite(df)
                else 1.95996398454
            )
            half = critical * np.sqrt(total)

            row.update({
                "within_var": ubar,
                "total_var": total,
                "lower95": mean - half,
                "upper95": mean + half,
                "interval_type": "rubin",
            })

        rows.append(row)

    return pd.DataFrame(rows)


def observed_stay_templates(fixes, observed_visits, *,
                            start_col="visit_start_ms", end_col="visit_end_ms",
                            min_fixes=3):
    """Create within-participant stationary donor tracks from observed visits.

    Callers must pass the visit boundaries produced by their existing stay
    segmentation. A different visit schema can be named explicitly.
    """
    return observed_trip_templates(
        fixes, observed_visits, start_col=start_col,
        end_col=end_col, min_fixes=min_fixes,
    )
