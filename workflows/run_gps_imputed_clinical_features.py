"""Opt-in GPS imputation workflow. Does not replace validated home-time outputs.

Default invocation performs static configuration checks only. --execute is
required to read participant data. Outputs are placed in a new directory.
All imputation intervals are DESCRIPTIVE, not calibrated confidence intervals.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig, observed_trip_templates,
)
from cortex2_engine.features.gps_shared_recomputation import recompute_multiple_timelines
from cortex2_engine.features.gps_pipeline_qc import GPSQCConfig, clean_gps_fixes
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
COLUMNS = ['timestamp', 'latitude', 'longitude', 'accuracy']


def build_participant_donors(window_fixes, *, compatibility_multiplier=2.0,
                             min_dwell_ms=600_000, max_gap_ms=60_000):
    """Build trip and stay donors using this window's Cortex place substrate.

    Donors come ONLY from the target participant and analysis window. No
    pooling across participants, and no imputed fixes become observed donors.
    """
    fixes = clean_gps_fixes(window_fixes[COLUMNS].copy(),
                            GPSQCConfig(max_gap_ms=max_gap_ms, max_accuracy_m=None))
    if len(fixes) < 2:
        return [], [], {'n_observed_trips': 0, 'n_observed_stays': 0}
    edges = build_gps_intervals(
        fixes, window_start_ms=int(fixes.timestamp.min()),
        window_end_ms=int(fixes.timestamp.max()) + 1, max_gap_ms=max_gap_ms)
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
        episodes, compatibility_multiplier=compatibility_multiplier)
    if assignments.empty or places.empty:
        return [], [], {'n_observed_trips': 0, 'n_observed_stays': 0}
    visits = build_place_visits(assignments, min_endpoint_dwell_ms=min_dwell_ms)
    trajectory = add_gps_trajectory_plausibility(
        spatial, max_plausible_speed_kmh=1000.0)
    trips = build_observed_trips(trajectory, visits, places)
    if trips.empty:
        trips = pd.DataFrame(columns=['trip_start_ms', 'trip_end_ms'])
    trip_donors = observed_trip_templates(fixes, trips, min_fixes=3)
    stay_donors = []
    for episode in episodes.itertuples(index=False):
        lo, hi = int(episode.episode_start_ms), int(episode.episode_end_ms)
        if hi - lo < min_dwell_ms:
            continue
        donor = fixes.loc[fixes.timestamp.between(lo, hi)].copy()
        if len(donor) >= 3:
            stay_donors.append(donor)
    return trip_donors, stay_donors, {
        'n_observed_trips': int(len(trips)),
        'n_observed_stays': int(len(stay_donors)),
    }


def run_participant(participant, *, m, output_dir):
    import pyarrow.parquet as pq
    requested = pq.read_table(
        REQUESTED, filters=[('participant_id', '=', participant)]).to_pandas()
    if requested.empty:
        raise RuntimeError(f'No requested days for participant {participant}')
    raw = pq.read_table(RAW, filters=[('participant_id', '=', participant)]).to_pandas()
    if raw.empty:
        raise RuntimeError(f'No GPS observations for participant {participant}')
    required = set(COLUMNS) | {'date'}
    if not required.issubset(raw):
        raise ValueError(f'Missing GPS fields: {sorted(required - set(raw))}')
    raw['date'] = raw.date.astype(str)
    requested['date'] = requested.date.astype(str)
    cfg = ImputationConfig(m=m, stay_compatibility_multiplier=2.0)
    rows, pooled_rows, diagnostic_rows = [], [], []
    for window_id, days in requested.groupby('clinical_window_id', sort=False):
        dates = set(days.date)
        window_raw = raw.loc[raw.date.isin(dates)].copy()
        trip_donors, stay_donors, donor_summary = build_participant_donors(window_raw)
        for day in days.drop_duplicates('date').itertuples(index=False):
            fixes = window_raw.loc[window_raw.date.eq(str(day.date)), COLUMNS].copy()
            key = {'participant_id': participant, 'clinical_window_id': window_id,
                   'date': str(day.date)}
            if len(fixes) < 2:
                rows.append({**key, 'imputation_status': 'insufficient_gps',
                             **donor_summary})
                continue
            result = recompute_multiple_timelines(
                fixes, trip_donors, stay_donors=stay_donors,
                start_ms=int(day.start_ms), end_ms=int(day.end_ms),
                imputation_config=cfg, compatibility_multiplier=2.0,
                acceleration_noise_density_m_s1p5=1.0,
            )
            observed = result['observed_features']
            rows.append({**key, **donor_summary,
                         'imputation_status': result['imputation_status'],
                         'n_gaps': result['n_gaps'],
                         'observed_trip_count': observed.get('trip_count'),
                         'observed_path_length_m': observed.get('path_length_m'),
                         'observed_displacement_m': observed.get('displacement_m'),
                         'observed_supported_duration_ms': observed.get('observed_supported_duration_ms'),
                         'imputed_supported_duration_ms': observed.get('imputed_supported_duration_ms')})
            for record in result['pooled'].to_dict('records'):
                pooled_rows.append({**key, **record})
            if not result['gap_diagnostics'].empty:
                diagnostic_rows.extend({**key, **record} for record in
                                       result['gap_diagnostics'].to_dict('records'))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f'{participant}_daily_status.csv', index=False)
    pd.DataFrame(pooled_rows).to_csv(output_dir / f'{participant}_pooled_descriptive.csv', index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / f'{participant}_gap_diagnostics.csv', index=False)
    print(f'Saved candidate imputation review outputs to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', help='Specific participant, required for execution')
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--execute', action='store_true', help='Actually process participant data')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if not args.execute:
        print('REVIEW MODE: no participant data read or written.')
        print('Explicit --execute and --participant required after code review.')
        return
    if not args.participant:
        parser.error('--participant is required with --execute')
    if args.m < 2:
        parser.error('--m must be at least 2')
    run_participant(args.participant, m=args.m, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
