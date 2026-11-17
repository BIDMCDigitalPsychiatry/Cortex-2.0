"""Allocate smoothed path length over dates; keep D at the trip level.

When per-edge smoothed path segments are unavailable, we allocate a trip's
smoothed total proportional to temporally supported raw edge lengths by day.
This is a flagged proxy, not exact per-day smoothed path length. If edge data
is absent, use a time-proportional proxy. Trip displacement is assigned to the
departure date ONLY for bookkeeping, not as a daily movement measure.
"""
from datetime import timedelta
import numpy as np
import pandas as pd

PATH_FIELDS = {
    'trip_path_length_obs_m': 'path_distance_smooth_obs_m',
    'trip_path_raw_obs_m': 'path_distance_raw_obs_m',
}
DISPLACEMENT_FIELDS = {
    'trip_displacement_obs_m': 'place_displacement_obs_m',
    'trip_endpoint_displacement_obs_m': 'endpoint_displacement_obs_m',
}


def allocate_trips_to_days(trips, *, edges=None, timezone='America/New_York'):
    columns = ['date', 'trip_count_obs', *PATH_FIELDS, *DISPLACEMENT_FIELDS,
               'trip_smoothing_coverage', 'trip_allocation_proxy_method',
               'cross_midnight_trip']
    if trips.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for trip in trips.itertuples(index=False):
        start, end = int(trip.trip_start_ms), int(trip.trip_end_ms)
        if end <= start:
            raise ValueError('Invalid trip duration')
        departure = pd.Timestamp(start, unit='ms', tz='UTC').tz_convert(timezone)
        finish = pd.Timestamp(end, unit='ms', tz='UTC').tz_convert(timezone)
        pieces = []
        current = departure
        while current < finish:
            midnight = pd.Timestamp(current.date() + timedelta(days=1)).tz_localize(timezone)
            part_end = min(midnight, finish)
            lo, hi = int(current.timestamp() * 1000), int(part_end.timestamp() * 1000)
            pieces.append((str(current.date()), lo, hi))
            current = part_end
        cross = len(pieces) > 1
        weights = np.array([hi - lo for _, lo, hi in pieces], dtype=float)
        method = 'time_proportional_proxy'
        if edges is not None and {'start_ms', 'end_ms', 'distance_m', 'supported'}.issubset(edges.columns):
            e = edges.loc[(edges.end_ms > start) & (edges.start_ms < end)
                          & edges.supported.fillna(False)].copy()
            lengths = []
            for _, lo, hi in pieces:
                overlap = np.minimum(e.end_ms.to_numpy(float), hi) - np.maximum(e.start_ms.to_numpy(float), lo)
                duration = e.end_ms.to_numpy(float) - e.start_ms.to_numpy(float)
                good = (overlap > 0) & (duration > 0)
                lengths.append(float(np.sum(np.maximum(overlap[good], 0) /
                                                    duration[good] * e.distance_m.to_numpy(float)[good])))
            candidate = np.asarray(lengths)
            if np.isfinite(candidate).all() and candidate.sum() > 0:
                weights, method = candidate, 'raw_edge_length_proportional_proxy'
        weights /= weights.sum()
        for idx, ((date, lo, hi), weight) in enumerate(zip(pieces, weights)):
            rec = {
                'date': date,
                'trip_count_obs': int(idx == 0),
                'trip_smoothing_coverage': getattr(trip, 'smoothing_coverage_fraction', np.nan),
                'trip_allocation_proxy_method': method if cross else 'whole_trip',
                'cross_midnight_trip': cross,
            }
            for name, source in PATH_FIELDS.items():
                value = getattr(trip, source, np.nan)
                rec[name] = float(value) * weight if pd.notna(value) else np.nan
            for name, source in DISPLACEMENT_FIELDS.items():
                value = getattr(trip, source, np.nan)
                rec[name] = float(value) if idx == 0 and pd.notna(value) else (0.0 if idx else np.nan)
            rows.append(rec)
    pieces = pd.DataFrame(rows)
    aggregations = {'trip_count_obs': 'sum',
                    **{name: lambda x: x.sum(min_count=1) for name in [*PATH_FIELDS, *DISPLACEMENT_FIELDS]},
                    'trip_smoothing_coverage': 'mean',
                    'cross_midnight_trip': 'max',
                    'trip_allocation_proxy_method': lambda x: 'raw_edge_length_proportional_proxy' if 'raw_edge_length_proportional_proxy' in set(x) else ('time_proportional_proxy' if 'time_proportional_proxy' in set(x) else 'whole_trip')}
    return pieces.groupby('date', as_index=False).agg(aggregations)
