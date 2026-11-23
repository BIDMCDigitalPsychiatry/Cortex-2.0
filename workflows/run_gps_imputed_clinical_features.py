"""Review-gated Cortex-2.0 window-aligned observed/imputed GPS features.

Default mode does NOT read data. --execute explicitly enables participant access.
Observed and imputed use identical QC and one 7-day place model per timeline.
Boundary gaps are unresolved, not falsely imputed without anchors. Per-day
path allocation is not claimed to be exact. Home outputs are provisional and
never overwrite validated home-time results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig, observed_trip_templates,
)
from cortex2_engine.features.gps_pipeline_qc import GPSQCConfig, clean_gps_fixes
from cortex2_engine.features.gps_shared_recomputation import recompute_multiple_timelines
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_anchor_episodes import build_anchor_compatible_location_episodes
from cortex2_engine.features.gps_place_candidates import build_place_candidates
from cortex2_engine.features.gps_trajectory_plausibility import add_gps_trajectory_plausibility
from cortex2_engine.features.gps_trip_features import build_place_visits, build_observed_trips

ROOT = Path('/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical')
RAW = ROOT / 'gps_clinical_weekly_raw_points.parquet'
REQUESTED = ROOT / 'gps_clinical_weekly_requested_days.parquet'
OUT = ROOT / 'imputed_mobility_candidate_review'
PREVIOUS_AMO327 = ROOT / 'home_time/trip_distance_validation/AMO327_daily_trip_distance_comparison.csv'
COLUMNS = ['timestamp', 'latitude', 'longitude', 'accuracy']
PLACE_MULTIPLIER = 1.0  # validated home-time and AMO327 trip parameter
MAX_GAP_MS = 60_000
MIN_DWELL_MS = 600_000


def build_participant_donors(cleaned_window_fixes, *, max_gap_ms=MAX_GAP_MS):
    """Donors are observed, continuous segments from the SAME window."""
    fixes = cleaned_window_fixes
    if len(fixes) < 2:
        return [], [], {'n_observed_trips': 0, 'n_observed_stays': 0}
    edges = build_gps_intervals(
        fixes, window_start_ms=int(fixes.timestamp.min()),
        window_end_ms=int(fixes.timestamp.max()) + 1,
        max_gap_ms=max_gap_ms)
    if edges.empty:
        return [], [], {'n_observed_trips': 0, 'n_observed_stays': 0}
    spatial = add_gps_spatial_semantics(edges.copy())
    supported = spatial.loc[spatial.supported.fillna(False)].copy()
    if supported.empty:
        return [], [], {'n_observed_trips': 0, 'n_observed_stays': 0}
    episodes = build_anchor_compatible_location_episodes(supported)
    if episodes.empty:
        return [], [], {'n_observed_trips': 0, 'n_observed_stays': 0}
    assignments, places = build_place_candidates(
        episodes, compatibility_multiplier=PLACE_MULTIPLIER)
    if assignments.empty or places.empty:
        return [], [], {'n_observed_trips': 0, 'n_observed_stays': 0}
    visits = build_place_visits(assignments, min_endpoint_dwell_ms=MIN_DWELL_MS)
    trajectory = add_gps_trajectory_plausibility(
        spatial, max_plausible_speed_kmh=1000.0)
    trips = build_observed_trips(trajectory, visits, places)
    if trips.empty:
        trips = pd.DataFrame(columns=['trip_start_ms', 'trip_end_ms'])
    trip_donors = observed_trip_templates(fixes, trips, min_fixes=3)

    # CORTEX2_CONTINUOUS_ROUTE_DONORS_V1
    # Keep detected trip donors, but also make continuously
    # observed GPS runs available for geographic subsection
    # matching. Split only at unsupported GPS gaps.
    observed = (
        fixes.sort_values("timestamp")
        .reset_index(drop=True)
    )

    if "imputed" in observed.columns:
        observed = observed.loc[
            ~observed.imputed.fillna(False)
        ].reset_index(drop=True)

    if len(observed) >= 2:
        t = observed.timestamp.to_numpy(dtype=np.int64)

        # Split continuous donor trajectories only at actual
        # unsupported GPS gaps, not at UTC midnight.
        boundaries = np.diff(t) > max_gap_ms

        cuts = np.r_[
            0,
            np.flatnonzero(boundaries) + 1,
            len(observed)
        ]

        for a, b in zip(cuts[:-1], cuts[1:]):
            segment = (
                observed.iloc[a:b]
                .copy()
                .reset_index(drop=True)
            )

            if len(segment) < 3:
                continue

            if (
                segment.timestamp.iloc[-1]
                - segment.timestamp.iloc[0]
                < 60000
            ):
                continue

            trip_donors.append(segment)

    stays = []
    for episode in episodes.itertuples(index=False):
        lo, hi = int(episode.episode_start_ms), int(episode.episode_end_ms)
        if hi - lo < MIN_DWELL_MS:
            continue
        donor = fixes.loc[fixes.timestamp.between(lo, hi)].copy()
        if len(donor) >= 3 and np.diff(donor.timestamp.to_numpy()).max(initial=0) <= max_gap_ms:
            stays.append(donor)
    return trip_donors, stays, {
        'n_observed_trips': len(trips),
        'n_observed_trip_donors': len(trip_donors),
        'n_observed_stays': len(stays),
    }


def day_coverage(fixes, days, *, max_gap_ms=MAX_GAP_MS):
    """Day-boundary missingness is accounted explicitly, never silently zero."""
    records = []
    for day in days.drop_duplicates('date').itertuples(index=False):
        lo, hi = int(day.start_ms), int(day.end_ms)
        # Window padding: neighboring fixes are accessible for the window
        # reconstruction; day coverage is clipped to day boundaries.
        local = fixes.loc[(fixes.timestamp >= lo) & (fixes.timestamp < hi)]
        if len(local) < 2:
            observed_ms = 0.0
        else:
            diffs = np.diff(local.timestamp.to_numpy(dtype=np.int64))
            observed_ms = float(diffs[diffs <= max_gap_ms].sum())
        records.append({
            'date': str(day.date),
            'observed_supported_ms': observed_ms,
            'observed_fraction_of_day': observed_ms / (hi - lo),
            'unobserved_ms': (hi - lo) - observed_ms,
            'n_clean_fixes': len(local),
            'day_complete': bool(observed_ms >= (hi - lo) - max_gap_ms),
        })
    return pd.DataFrame(records)


def run_participant(participant, *, m, output_dir):
    import pyarrow.parquet as pq
    requested = pq.read_table(
        REQUESTED, filters=[('participant_id', '=', participant)]).to_pandas()
    raw = pq.read_table(
        RAW, filters=[('participant_id', '=', participant)]).to_pandas()
    if requested.empty or raw.empty:
        raise RuntimeError('Missing requested days or GPS observations')
    if not set(COLUMNS).issubset(raw):
        raise ValueError('Raw GPS cache missing required columns')
    for table in (requested, raw):
        table['date'] = table.date.astype(str)
    cfg = ImputationConfig(m=m, stay_compatibility_multiplier=1.0)
    qc = GPSQCConfig(max_gap_ms=MAX_GAP_MS)
    summary, pooled_rows, diagnostics, daily_coverage_rows = [], [], [], []
    # Previous comparison is read-only; it is never regenerated or overwritten.
    previous = pd.read_csv(PREVIOUS_AMO327) if participant == 'AMO327' and PREVIOUS_AMO327.exists() else None
    if previous is not None and not {'date', 'old_supported_displacement_m'}.issubset(previous.columns):
        raise ValueError('Previous AMO327 baseline schema changed')
    if previous is not None:
        previous['date'] = pd.to_datetime(previous.date).dt.strftime('%Y-%m-%d')
    for window_id, days in requested.groupby('clinical_window_id', sort=False):
        days = days.drop_duplicates('date').sort_values('start_ms')
        start_ms, end_ms = int(days.start_ms.min()), int(days.end_ms.max())
        # Select by timestamps, not preassigned date labels; this preserves
        # neighboring-day fix context where the requested cache has it.
        # Full 7-day window supplies the place model, not one day at a time.
        inside = raw.loc[(raw.timestamp >= start_ms) & (raw.timestamp < end_ms), COLUMNS]
        # Do not treat outside-window fixes as observations inside the
        # assessed window. Missing time at either boundary is explicitly
        # reported as unresolved (not zero movement).
        cleaned = clean_gps_fixes(inside, qc)
        donor_fixes = cleaned.loc[(cleaned.timestamp >= start_ms) & (cleaned.timestamp < end_ms)]
        key = {'participant_id': participant, 'clinical_window_id': window_id}
        if len(cleaned) < 2:
            summary.append({**key, 'imputation_status': 'insufficient_observations'})
            continue
        trip_donors, stay_donors, donor_stats = build_participant_donors(donor_fixes)
        result = recompute_multiple_timelines(
            cleaned, trip_donors, stay_donors=stay_donors,
            start_ms=start_ms, end_ms=end_ms,
            imputation_config=cfg, qc_config=qc,
            compatibility_multiplier=PLACE_MULTIPLIER,
            acceleration_noise_density_m_s1p5=1.0)
        observed = result['observed_features']
        baseline = np.nan
        if previous is not None:
            window_dates = set(days.date.astype(str))
            baseline_rows = previous.loc[previous.date.isin(window_dates)]
            if not baseline_rows.empty:
                baseline = float(pd.to_numeric(baseline_rows.old_supported_displacement_m, errors='coerce').sum(min_count=1))
        summary.append({
            **key, **donor_stats,
            'window_start_ms': start_ms, 'window_end_ms': end_ms,
            'imputation_status': result['imputation_status'],
            'n_gaps': result['n_gaps'],
            'n_unresolved_imputation_attempts': result['n_unresolved_imputation_attempts'],
            'unresolved_gap_fraction': float(result['pooled']['unresolved_gap_duration_fraction'].iloc[0]),
            'previous_consecutive_fix_supported_displacement_m': baseline,
            'observed_trip_count': observed.get('trip_count'),
            'observed_path_length_m': observed.get('path_length_m'),
            'observed_displacement_m': observed.get('displacement_m'),
            'observed_endpoint_displacement_m': observed.get('endpoint_displacement_m'),
            'observed_supported_duration_ms': observed.get('observed_supported_duration_ms'),
            'mean_imputed_supported_duration_ms': result['estimates']['imputed_supported_duration_ms'].mean(),
            'scope': 'full_clinical_window_with_explicit_unresolved_boundaries',
        })
        daily_cov = day_coverage(cleaned, days)
        for rec in daily_cov.to_dict('records'):
            daily_coverage_rows.append({**key, **rec})
        for rec in result['pooled'].to_dict('records'):
            pooled_rows.append({**key, **rec})
        if not result['gap_diagnostics'].empty:
            diagnostics.extend(
                {**key, **rec}
                for rec in result['gap_diagnostics'].to_dict('records'))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output_dir / f'{participant}_window_status.csv', index=False)
    pd.DataFrame(pooled_rows).to_csv(output_dir / f'{participant}_window_pooled_descriptive.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / f'{participant}_gap_diagnostics.csv', index=False)
    pd.DataFrame(daily_coverage_rows).to_csv(output_dir / f'{participant}_daily_coverage.csv', index=False)
    print('Saved window-aligned candidate imputation outputs to', output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant')
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if not args.execute:
        print('REVIEW ONLY: no participant data accessed.')
        print('Window-based, place multiplier=1.0; unresolved boundary time remains missing.')
        return
    if not args.participant:
        parser.error('--participant required with --execute')
    if args.m < 2:
        parser.error('--m must be >= 2')
    run_participant(args.participant, m=args.m, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
