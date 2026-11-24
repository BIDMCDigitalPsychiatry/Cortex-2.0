"""Build one retrospective place/home reference from BOTH Cortex raw GPS caches.

Usage: PYTHONPATH=src python workflows/build_gps_shared_place_reference.py --participant AMO327
Existing references are never overwritten; remove a stale participant directory
explicitly only after inspecting why its coverage/source scope changed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from cortex2_engine.features.gps_shared_place_reference import save_reference, REFERENCE_ROOT
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_anchor_episodes import build_anchor_compatible_location_episodes
from cortex2_engine.features.gps_place_candidates import build_place_candidates
from cortex2_engine.features.gps_home_shared import calculate_home_time
from cortex2_engine.features.gps_pipeline_qc import clean_gps_fixes, GPSQCConfig

DC = Path('/shared/Digital Clinic/cortex2_outputs/digital_clinic')
SOURCES = (
    DC / 'feature_reliability/gps_clinical/gps_clinical_weekly_raw_points.parquet',
    DC / 'clinical/daily_ema/daily_ema_gps_raw_points.parquet',
)
REQUESTED = DC / 'feature_reliability/gps_clinical/gps_clinical_weekly_requested_days.parquet'
EMA = DC / 'clinical/daily_ema/daily_ema_gps_participants.csv'
MAX_GAP_MS = 60_000


def requested_dates(pid):
    dates = set()
    if REQUESTED.exists():
        q = pd.read_parquet(REQUESTED, columns=['participant_id', 'date'])
        dates.update(q.loc[q.participant_id.astype(str).eq(pid), 'date'].astype(str))
    if EMA.exists():
        e = pd.read_csv(EMA, usecols=lambda c: c in ('participant_id', 'cortex_participant_id', 'date'))
        identifier = 'participant_id' if 'participant_id' in e else 'cortex_participant_id'
        dates.update(pd.to_datetime(e.loc[e[identifier].astype(str).eq(pid), 'date'], errors='coerce').dt.strftime('%Y-%m-%d').dropna())
    return dates


def build(pid, compatibility_multiplier):
    dates = requested_dates(pid)
    frames = []
    sources_used = []
    for path in SOURCES:
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format='parquet')
        if 'participant_id' not in dataset.schema.names:
            raise ValueError(f'No participant_id in {path}')
        table = dataset.to_table(filter=ds.field('participant_id') == pid)
        if table.num_rows:
            frame = table.to_pandas()
            needed = {'timestamp', 'latitude', 'longitude', 'accuracy', 'date'}
            if not needed.issubset(frame):
                raise ValueError(f'Missing columns in {path}: {needed - set(frame)}')
            frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            frames.append(frame)
            sources_used.append(path.name)
            dates.update(frame['date'].dropna().astype(str))
    if not frames:
        raise RuntimeError(f'No GPS observations for {pid} in either raw cache')
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=['date', 'timestamp', 'latitude', 'longitude'])
    raw = raw.drop_duplicates(['timestamp', 'latitude', 'longitude', 'accuracy'])
    raw = raw.sort_values(['date', 'timestamp'])
    sections = []
    support_ms = 0.0
    for date, day in raw.groupby('date', sort=True):
        dt = pd.Timestamp(date, tz='UTC')
        start_ms = int(dt.timestamp() * 1000)
        end_ms = start_ms + 86_400_000
        fixes = clean_gps_fixes(day[['timestamp', 'latitude', 'longitude', 'accuracy']].copy(),
                                GPSQCConfig(max_gap_ms=MAX_GAP_MS))
        edges = build_gps_intervals(fixes, window_start_ms=start_ms,
                                    window_end_ms=end_ms, max_gap_ms=MAX_GAP_MS)
        if edges.empty:
            continue
        supported = edges.loc[edges.supported.fillna(False)].copy()
        support_ms += float(supported.duration_ms.sum())
        if supported.empty:
            continue
        segments = build_anchor_compatible_location_episodes(add_gps_spatial_semantics(supported))
        if not segments.empty:
            segments['date'] = date
            sections.append(segments)
    if not sections:
        raise RuntimeError(f'No supported location episodes for {pid}')
    segments = pd.concat(sections, ignore_index=True)
    assignments, places = build_place_candidates(
        segments, compatibility_multiplier=compatibility_multiplier)
    if 'date' not in assignments:
        if len(assignments) != len(segments):
            raise ValueError('Clustering lost episode-to-date alignment')
        assignments['date'] = segments['date'].to_numpy()
    home = calculate_home_time(
        assignments,
        supported_duration_ms=support_ms,
        start_ms=int(min(pd.Timestamp(d, tz='UTC').timestamp() for d in dates) * 1000),
        end_ms=int((max(pd.Timestamp(d, tz='UTC') for d in dates).timestamp() * 1000) + 86_400_000),
    )
    path = save_reference(pid, assignments, places, home, dates,
                          compatibility_multiplier=compatibility_multiplier,
                          source_names=sources_used)
    print(f'{pid}: {len(raw):,} unique fixes, {len(places)} places, home={home["home_status"]}')
    print('Reference saved:', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', required=True, help='Participant ID; build one at a time')
    parser.add_argument('--compatibility-multiplier', type=float, default=1.0)
    args = parser.parse_args()
    build(args.participant, args.compatibility_multiplier)
