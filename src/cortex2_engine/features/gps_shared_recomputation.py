"""Cortex-2.0 shared recomputation on observed and imputed GPS timelines.

The same location episode, place, visit, trip and smoothing functions are
used on each timeline. The observed-only output remains separate from the
completed-timeline output. Quantiles are NOT calibrated confidence intervals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_trajectory_plausibility import add_gps_trajectory_plausibility
from cortex2_engine.features.gps_anchor_episodes import build_anchor_compatible_location_episodes
from cortex2_engine.features.gps_place_candidates import build_place_candidates
from cortex2_engine.features.gps_trip_features import (
    build_place_visits, build_observed_trips, add_smoothed_trip_path_distances,
)
from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig, impute_timelines, find_long_gaps,
)
from cortex2_engine.features.gps_home_shared import HomeConfig, calculate_home_time


def annotate_interval_provenance(edges, fixes):
    result = edges.copy()
    if result.empty:
        result["involves_imputation"] = pd.Series(dtype=bool)
        return result
    if "imputed" not in fixes.columns:
        result["involves_imputation"] = False
        return result
    x = fixes[["timestamp", "imputed"]].copy()
    x["timestamp"] = pd.to_numeric(x.timestamp, errors="coerce")
    lookup = x.dropna(subset=["timestamp"]).groupby("timestamp").imputed.any()
    start = result.start_ms.map(lookup)
    end = result.end_ms.map(lookup)
    # Unmatched endpoints are not asserted to be observed.
    if start.isna().any() or end.isna().any():
        raise ValueError("Interval endpoints could not be matched to GPS provenance")
    result["involves_imputation"] = start.astype(bool) | end.astype(bool)
    return result


def annotate_trip_imputation_provenance(trips, edges):
    out = trips.copy()
    if out.empty:
        out["trip_provenance"] = pd.Series(dtype="object")
        return out
    if not {"trip_start_ms", "trip_end_ms"}.issubset(out.columns):
        raise ValueError("Trips are missing time boundaries")
    labels = []
    for trip in out.itertuples(index=False):
        e = edges.loc[(edges.start_ms < trip.trip_end_ms) & (edges.end_ms > trip.trip_start_ms)]
        if e.empty or not e.supported.fillna(False).all():
            labels.append("unknown")
        elif e.involves_imputation.any():
            labels.append("imputed")
        else:
            labels.append("observed")
    out["trip_provenance"] = labels
    return out


def recompute_timeline(fixes, *, start_ms, end_ms, max_gap_ms=60_000,
                       compatibility_multiplier=1.0,
                       min_endpoint_dwell_ms=600_000,
                       acceleration_noise_density_m_s1p5=1.0,
                       home_config=None):
    """Recompute mobility in one window using the same Cortex place substrate.

    For home inference, fixes must span enough reference nights; otherwise
    home_time_ms remains missing. Never infer home from a single day's mode.
    """
    edges = build_gps_intervals(
        fixes, window_start_ms=int(start_ms), window_end_ms=int(end_ms),
        max_gap_ms=int(max_gap_ms),
    )
    edges = annotate_interval_provenance(edges, fixes)
    results = {
        "trip_count": np.nan, "path_length_m": np.nan,
        "displacement_m": np.nan, "raw_path_length_m": np.nan,
        "home_time_ms": np.nan, "home_status": "insufficient_observability",
        "supported_duration_ms": 0.0, "observed_supported_duration_ms": 0.0,
        "imputed_supported_duration_ms": 0.0,
    }
    if edges.empty:
        return results
    good = edges.supported.fillna(False).astype(bool)
    results["observed_supported_duration_ms"] = float(
        edges.loc[good & ~edges.involves_imputation, "duration_ms"].sum())
    results["imputed_supported_duration_ms"] = float(
        edges.loc[good & edges.involves_imputation, "duration_ms"].sum())
    results["supported_duration_ms"] = (
        results["observed_supported_duration_ms"] +
        results["imputed_supported_duration_ms"])
    if not good.any():
        return results
    spatial_all = add_gps_spatial_semantics(edges.copy())
    trajectory = add_gps_trajectory_plausibility(
        spatial_all, max_plausible_speed_kmh=1000.0)
    # Location microsegments use only temporally supported edges.
    episodes = build_anchor_compatible_location_episodes(
        spatial_all.loc[good].copy())
    if episodes.empty:
        return results
    assignments, places = build_place_candidates(
        episodes, compatibility_multiplier=compatibility_multiplier)
    if assignments.empty or places.empty:
        return results
    visits = build_place_visits(assignments, min_endpoint_dwell_ms=min_endpoint_dwell_ms)
    trips = build_observed_trips(trajectory, visits, places)
    trips = annotate_trip_imputation_provenance(trips, edges)
    results["trip_count"] = int(len(trips))
    results["trip_count_imputed"] = int((trips.trip_provenance == "imputed").sum())
    if len(trips):
        trips = add_smoothed_trip_path_distances(
            trajectory, trips,
            acceleration_noise_density_m_s1p5=acceleration_noise_density_m_s1p5)
        for out, col in (
            ("path_length_m", "path_distance_smooth_obs_m"),
            ("displacement_m", "endpoint_displacement_obs_m"),
            ("raw_path_length_m", "path_distance_raw_obs_m"),
        ):
            if col not in trips.columns:
                raise ValueError(f"Trip estimator missing column {col}")
            results[out] = float(trips[col].sum(min_count=1))
    else:
        # No trips can be a valid zero; avoid zero on unsupported windows.
        results.update(path_length_m=0.0, displacement_m=0.0, raw_path_length_m=0.0)
    home = calculate_home_time(
        assignments, supported_duration_ms=results["supported_duration_ms"],
        start_ms=start_ms, end_ms=end_ms, config=home_config or HomeConfig())
    results.update(home)
    return results


def audit_completed_timeline(timeline, max_gap_ms=60_000):
    gaps = find_long_gaps(timeline, max_gap_ms)
    return {
        "temporally_complete": bool(len(timeline) >= 2 and gaps.empty),
        "n_remaining_gaps": len(gaps),
        "max_remaining_gap_ms": float(gaps.gap_ms.max()) if len(gaps) else 0.0,
    }


def pool_completed_mobility_features(estimates, completion):
    """Summarize the M complete timelines. No confidence-interval claim."""
    names = ("trip_count", "path_length_m", "displacement_m", "raw_path_length_m", "home_time_ms")
    if estimates.empty or completion.empty or len(estimates) != len(completion):
        raise ValueError("Empty or misaligned imputation results")
    if not estimates.imputation.equals(completion.imputation):
        raise ValueError("Misaligned imputation IDs")
    complete = bool(completion.temporally_complete.fillna(False).all())
    rows = []
    for name in names:
        values = pd.to_numeric(estimates[name], errors="coerce").to_numpy(float)
        valid = complete and np.isfinite(values).all()
        rows.append({
            "feature": name,
            "estimate": float(values.mean()) if valid else np.nan,
            "lower95": float(np.quantile(values, 0.025)) if valid else np.nan,
            "upper95": float(np.quantile(values, 0.975)) if valid else np.nan,
            "interval_type": ("descriptive_imputation_range_not_CI" if valid
                              else "incomplete_or_missing"),
            "n_imputations": int(np.isfinite(values).sum()),
        })
    return pd.DataFrame(rows)


def recompute_multiple_timelines(fixes, donors, *, start_ms, end_ms,
                                 stay_donors=None,
                                 imputation_config=None, **kwargs):
    cfg = imputation_config or ImputationConfig()
    observed = fixes.loc[~fixes.imputed.fillna(False).astype(bool)].copy() if "imputed" in fixes else fixes.copy()
    observed_features = recompute_timeline(
        observed, start_ms=start_ms, end_ms=end_ms, max_gap_ms=cfg.max_supported_gap_ms, **kwargs)
    timelines, diagnostics = impute_timelines(observed, donors, config=cfg, stay_donors=stay_donors)
    records, checks = [], []
    for i, timeline in enumerate(timelines):
        check = audit_completed_timeline(timeline, cfg.max_supported_gap_ms)
        checks.append({"imputation": i, **check})
        features = recompute_timeline(
            timeline, start_ms=start_ms, end_ms=end_ms,
            max_gap_ms=cfg.max_supported_gap_ms, **kwargs)
        records.append({"imputation": i, **features})
    estimates = pd.DataFrame(records)
    completion = pd.DataFrame(checks)
    pooled = pool_completed_mobility_features(estimates, completion)
    unresolved = int((~diagnostics.imputed.astype(bool)).sum()) if len(diagnostics) else 0
    complete = bool(completion.temporally_complete.all()) and unresolved == 0
    status = "observed_only" if diagnostics.empty else (
        "candidate_imputation_complete" if complete else "incomplete_imputation")
    if not complete:
        pooled.loc[:, ["estimate", "lower95", "upper95"]] = np.nan
        pooled["interval_type"] = "incomplete_imputation"
    return {
        "observed_features": observed_features,
        "completed_timeline_estimates": estimates,
        "estimates": estimates,
        "pooled": pooled,
        "gap_diagnostics": diagnostics,
        "completion_diagnostics": completion,
        "imputation_status": status,
        "n_gaps": len(diagnostics.drop_duplicates(["start_ms", "end_ms"])) if len(diagnostics) else 0,
        "n_unresolved_imputation_attempts": unresolved,
    }
