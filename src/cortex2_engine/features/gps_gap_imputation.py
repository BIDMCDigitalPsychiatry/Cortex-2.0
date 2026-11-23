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
        if (len(subset) >= min_fixes
                and ("imputed" not in subset or not subset["imputed"].any())
                and np.diff(subset.timestamp.to_numpy()).max(initial=0) <= 60_000):
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

    # Rotate and uniformly scale the donor's displacement to the target
    # endpoint. This preserves the donor's relative turning geometry instead
    # of imposing a large artificial linear correction on an opposite trip.
    donor_endpoint = np.array([x[-1], y[-1]], dtype=float)
    target_endpoint = np.array([xe, ye], dtype=float)
    donor_norm = float(np.linalg.norm(donor_endpoint))
    target_norm = float(np.linalg.norm(target_endpoint))
    # Near-zero endpoint displacement does NOT imply a stationary path:
    # a round trip can start and finish at the same place. Preserve the
    # donor's loop geometry, adding only a small linear endpoint adjustment.
    close_threshold = max(1.0, float(left.accuracy + right.accuracy))
    if target_norm <= close_threshold:
        if donor_norm <= close_threshold:
            x = x + tau * (xe - x[-1])
            y = y + tau * (ye - y[-1])
        else:
            # A one-way donor can't be stretched to zero displacement
            # without destroying its geometry; reject instead.
            return None
    elif donor_norm <= 1e-6:
        return None
    else:
        angle = np.arctan2(target_endpoint[1], target_endpoint[0]) - np.arctan2(donor_endpoint[1], donor_endpoint[0])
        scale = target_norm / donor_norm
        if not np.isfinite(scale) or scale > 4.0:
            return None
        cc, ss = np.cos(angle), np.sin(angle)
        xx = scale * (cc * x - ss * y)
        yy = scale * (ss * x + cc * y)
        x, y = xx, yy

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



# CORTEX2_GEOGRAPHIC_ROUTE_V1
def _matching_route_sections(left, right, trip_donors, cfg):
    """Find observed route subsections near both gap endpoints."""
    matches = []

    for donor_index, donor in enumerate(trip_donors):
        d = _require_fixes(donor)
        if len(d) < cfg.min_donor_fixes:
            continue
        if "imputed" in d and d.imputed.any():
            continue

        times = d.timestamp.to_numpy(dtype=np.int64)
        if np.diff(times).max(initial=0) > cfg.max_supported_gap_ms:
            continue

        lat = d.latitude.to_numpy(dtype=float)
        lon = d.longitude.to_numpy(dtype=float)

        def distances(anchor):
            a = np.deg2rad(float(anchor.latitude))
            b = np.deg2rad(lat)
            dl = np.deg2rad(lon - float(anchor.longitude))
            h = (np.sin((b-a)/2)**2
                 + np.cos(a)*np.cos(b)*np.sin(dl/2)**2)
            return 2*EARTH_RADIUS_M*np.arcsin(
                np.sqrt(np.clip(h, 0, 1))
            )

        ds = distances(left)
        de = distances(right)

        k = min(8, len(d))
        starts = np.argpartition(ds, k-1)[:k]
        ends = np.argpartition(de, k-1)[:k]

        for i in starts:
            for j in ends:
                if abs(int(i)-int(j))+1 < cfg.min_donor_fixes:
                    continue

                a, b = sorted((int(i), int(j)))
                if times[b]-times[a] < cfg.min_template_span_ms:
                    continue

                score = float(ds[i]+de[j])

                # Compare endpoint mismatch with gap displacement,
                # without imposing a donor-duration limit.
                displacement = float(_haversine(
                    left.latitude, left.longitude,
                    right.latitude, right.longitude
                ))
                if score >= displacement:
                    continue

                route = d.iloc[a:b+1].copy().reset_index(drop=True)
                if i > j:
                    route = route.iloc[::-1].reset_index(drop=True)

                matches.append((score, donor_index, route))

    matches.sort(key=lambda x: x[0])
    return matches



def _bridge_matched_route(left, right, route, cfg):
    """Adapt geographically matched donor geometry to a missing GPS interval.

    Preserve observed route vertices, including bends, while allocating
    timestamps according to cumulative distance rather than donor duration.
    """
    d = route.copy().reset_index(drop=True)

    if len(d) < cfg.min_donor_fixes:
        return None

    dt = int(right.timestamp - left.timestamp)
    if dt <= cfg.max_supported_gap_ms:
        return None

    lat0 = float(left.latitude)
    lon0 = float(left.longitude)
    coslat = np.cos(np.deg2rad(lat0))

    if abs(coslat) < 1e-8:
        return None

    # Local metric coordinates, relative to the observed gap start.
    x = (
        EARTH_RADIUS_M
        * np.deg2rad(d.longitude.to_numpy(dtype=float) - lon0)
        * coslat
    )
    y = (
        EARTH_RADIUS_M
        * np.deg2rad(d.latitude.to_numpy(dtype=float) - lat0)
    )

    x -= x[0]
    y -= y[0]

    # Distance-based progress along the original donor geometry.
    step = np.hypot(np.diff(x), np.diff(y))
    cumulative = np.r_[0.0, np.cumsum(step)]
    total_distance = float(cumulative[-1])

    if not np.isfinite(total_distance) or total_distance <= 0:
        return None

    progress = cumulative / total_distance

    # Remove exact repeated positions, not meaningful route bends.
    keep = np.r_[True, np.diff(progress) > 1e-12]
    progress = progress[keep]
    x = x[keep]
    y = y[keep]
    accuracy = d.accuracy.to_numpy(dtype=float)[keep]

    if len(progress) < 2:
        return None

    # Observed target endpoint in the same coordinate system.
    xe = (
        EARTH_RADIUS_M
        * np.deg2rad(float(right.longitude) - lon0)
        * coslat
    )
    ye = (
        EARTH_RADIUS_M
        * np.deg2rad(float(right.latitude) - lat0)
    )

    # Small, gradual translation to align observed donor endpoints.
    # No whole-route rotation or scaling.
    x = x + progress * (xe - x[-1])
    y = y + progress * (ye - y[-1])


    # CORTEX2_COMPACT_GEOGRAPHIC_ROUTE_V1
    # Simplify by the additional path length lost when removing
    # each vertex, rather than by spatial distance alone.
    # This prioritizes preservation of bends and traveled distance.
    import heapq

    n_original = len(progress)
    target_vertices = max(3, int(dt // 1000) + 1)

    if n_original > target_vertices:
        prev = np.arange(n_original, dtype=int) - 1
        nxt = np.arange(n_original, dtype=int) + 1
        nxt[-1] = -1
        alive = np.ones(n_original, dtype=bool)
        heap = []

        def removal_loss(i):
            a, b = int(prev[i]), int(nxt[i])
            if a < 0 or b < 0:
                return float("inf")
            before = (
                np.hypot(x[i]-x[a], y[i]-y[a])
                + np.hypot(x[b]-x[i], y[b]-y[i])
            )
            after = np.hypot(x[b]-x[a], y[b]-y[a])
            return max(0.0, float(before-after))

        for i in range(1, n_original-1):
            heapq.heappush(
                heap, (removal_loss(i), i, int(prev[i]), int(nxt[i]))
            )

        remaining = n_original

        while remaining > target_vertices and heap:
            loss, i, a, b = heapq.heappop(heap)

            if not alive[i] or prev[i] != a or nxt[i] != b:
                continue

            alive[i] = False
            remaining -= 1
            nxt[a] = b
            prev[b] = a

            for j in (a, b):
                if 0 < j < n_original-1 and alive[j]:
                    heapq.heappush(
                        heap,
                        (removal_loss(j), j, int(prev[j]), int(nxt[j]))
                    )

        progress = progress[alive]
        x = x[alive]
        y = y[alive]
        accuracy = accuracy[alive]

    # Add support points only when needed to avoid long temporal
    # gaps. These are interpolated from the retained geometry.
    regular_intervals = max(2, int(np.ceil(dt / 30000)))
    support_progress = np.linspace(0, 1, regular_intervals+1)
    sampling_progress = np.unique(
        np.r_[progress, support_progress, 0.0, 1.0]
    )

    timestamps = np.rint(
        int(left.timestamp) + dt*sampling_progress
    ).astype(np.int64)

    xx = np.interp(sampling_progress, progress, x)
    yy = np.interp(sampling_progress, progress, y)
    aa = np.interp(sampling_progress, progress, accuracy)

    valid = (
        (timestamps > int(left.timestamp))
        & (timestamps < int(right.timestamp))
    )

    out = pd.DataFrame({
        "timestamp": timestamps[valid],
        "latitude": lat0 + np.rad2deg(
            yy[valid] / EARTH_RADIUS_M
        ),
        "longitude": lon0 + np.rad2deg(
            xx[valid] / (EARTH_RADIUS_M*coslat)
        ),
        "accuracy": aa[valid],
        "imputed": True,
    })

    out = (
        out.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return out if not out.empty else None



def impute_timelines(
    fixes,
    donors,
    config=None,
    stay_donors=None,
    window_start_ms=None,
    window_end_ms=None,
):
    """Generate M timelines and an auditable gap-disposition table."""
    cfg = config or ImputationConfig()
    f = _require_fixes(fixes)
    if "imputed" in f.columns and f.imputed.any():
        raise ValueError("Observed source fixes must not contain imputed points")
    gaps = find_long_gaps(f, cfg.max_supported_gap_ms)
    if (window_start_ms is None) != (window_end_ms is None):
        raise ValueError("Both window boundaries must be supplied together")
    boundary_gaps = []
    if window_start_ms is not None:
        if window_end_ms <= window_start_ms:
            raise ValueError("Invalid window boundaries")
        if f.empty:
            boundary_gaps = [(int(window_start_ms), int(window_end_ms))]
        else:
            if int(f.timestamp.iloc[0]) > int(window_start_ms):
                boundary_gaps.append((int(window_start_ms), int(f.timestamp.iloc[0])))
            if int(f.timestamp.iloc[-1]) < int(window_end_ms):
                boundary_gaps.append((int(f.timestamp.iloc[-1]), int(window_end_ms)))

    rng = np.random.default_rng(cfg.seed)
    stay_donors = stay_donors or []
    timelines = []
    diagnostics = []

    for imp in range(cfg.m):
        pieces = [f.assign(imputed=False)]
        for lo, hi in boundary_gaps:
            # No bracketing observation: cannot condition a donor bridge.
            # Report the entire unobserved boundary span explicitly.
            diagnostics.append({
                "imputation": imp, "start_ms": lo, "end_ms": hi,
                "gap_ms": hi - lo, "imputed": False,
                "donor_index": pd.NA, "gap_type": "unbounded",
                "n_candidate_donors": 0,
            })

        for gap in gaps.itertuples(index=False):
            left = f.iloc[gap.left_index]
            right = f.iloc[gap.right_index]

            # Close endpoints can hide a round-trip excursion. Never force
            # all such gaps to be stationary: eligible trip AND stay templates
            # compete, with donor identity sampled uniformly (provisional).
            stay_possible = gap.endpoint_distance_m <= (
                (float(left.accuracy) + float(right.accuracy))
                * cfg.stay_compatibility_multiplier
            )
            candidate_pairs = []
            for kind, donor_set in (
                ("trip", donors),
                ("stay", stay_donors if stay_possible else []),
            ):
                for j, donor in enumerate(donor_set):
                    d = _require_fixes(donor)
                    if len(d) < cfg.min_donor_fixes:
                        continue
                    if ("imputed" in d and d["imputed"].any()) or (
                        np.diff(d.timestamp.to_numpy()).max(initial=0)
                        > cfg.max_supported_gap_ms
                    ):
                        continue
                    duration = d.timestamp.iloc[-1] - d.timestamp.iloc[0]
                    if duration <= 0:
                        continue
                    ratio = gap.gap_ms / duration
                    if not (1 / cfg.maximum_time_stretch <= ratio
                            <= cfg.maximum_time_stretch):
                        continue
                    distance = _haversine(
                        d.latitude.iloc[0], d.longitude.iloc[0],
                        d.latitude.iloc[-1], d.longitude.iloc[-1])
                    if abs(distance - gap.endpoint_distance_m) <= (
                        cfg.endpoint_distance_tolerance_m
                    ):
                        candidate_pairs.append((kind, j, donor))

            piece = None
            chosen = None
            kind = "unresolved"
            geographic_error_m = np.nan

            route_matches = _matching_route_sections(
                left, right, donors, cfg
            )

            for score, donor_index, route in route_matches:
                proposed = _bridge_matched_route(
                    left, right, route, cfg
                )
                if proposed is not None:
                    piece = proposed
                    chosen = donor_index
                    kind = "geographic_trip"
                    geographic_error_m = score
                    break

            if piece is None and candidate_pairs:
                for candidate_idx in rng.permutation(len(candidate_pairs)):
                    candidate_kind, candidate_index, donor = candidate_pairs[int(candidate_idx)]
                    candidate_piece = _bridge(left, right, donor, cfg)
                    if candidate_piece is not None:
                        kind = candidate_kind
                        chosen = candidate_index
                        piece = candidate_piece
                        break
            if piece is not None:
                pieces.append(piece)

            diagnostics.append({
                "imputation": imp,
                "start_ms": gap.start_ms,
                "end_ms": gap.end_ms,
                "gap_ms": gap.gap_ms,
                "imputed": piece is not None,
                "donor_index": chosen if piece is not None else pd.NA,
                "gap_type": kind,
                "n_candidate_donors": len(candidate_pairs),
                "geographic_endpoint_error_m": geographic_error_m,
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
