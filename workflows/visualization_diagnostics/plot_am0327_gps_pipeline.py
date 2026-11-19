#!/usr/bin/env python3
"""Plot the actual Cortex-2.0 GPS processing stages for one AMO327 week.

Reads existing raw cache, saved AMO327 CSV outputs, and Cortex processing
functions. Does not rerun imputation, edit the source, or write patient tables.
Figure panels A-D are calculated from the real raw GPS cache; E-F read saved
outputs. No invented trajectories or simulated locations are drawn.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from cortex2_engine.features.gps_pipeline_qc import GPSQCConfig, clean_gps_fixes
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_trajectory_plausibility import add_gps_trajectory_plausibility
from cortex2_engine.features.gps_anchor_episodes import build_anchor_compatible_location_episodes
from cortex2_engine.features.gps_place_candidates import build_place_candidates
from cortex2_engine.features.gps_trip_features import build_place_visits, build_observed_trips, add_smoothed_trip_path_distances

RAW = Path('/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/gps_clinical_weekly_raw_points.parquet')
SAVED = Path('outputs/am0327_final_gps_comparison')
BLUE = '#17639A'
TEAL = '#139B91'
ORANGE = '#E69F32'
RED = '#C84D4D'
SLATE = '#6E7F8F'
PALE = '#E9F1F5'
DARK = '#15354C'


def local_xy(lon, lat, lon0, lat0):
    """Equirectangular local XY in km. Only used for plotting."""
    return (np.asarray(lon, dtype=float) - lon0) * (111.195 * np.cos(np.deg2rad(lat0))), (np.asarray(lat, dtype=float) - lat0) * 111.195


def fig_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 10.0, 'axes.titlesize': 12.0,
        'axes.titleweight': 'bold', 'axes.labelsize': 9.3,
        'axes.edgecolor': '#C8D5DE', 'xtick.color': '#496277',
        'ytick.color': '#496277', 'text.color': DARK,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'svg.fonttype': 'none',
    })


def map_axes(ax, width, center_label=None):
    ax.set_xlim(-width, width)
    ax.set_ylim(-width * .76, width * .76)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('East–west distance (km)')
    ax.set_ylabel('North–south distance (km)')
    ax.grid(True, color='#E9EEF2', linewidth=.7, zorder=0)
    if center_label:
        ax.text(.02, .98, center_label, transform=ax.transAxes,
                va='top', ha='left', fontsize=8.3, color=SLATE,
                bbox=dict(facecolor='white', alpha=.92, edgecolor='none', pad=3))


def draw_edges(ax, edges, lon0, lat0, width, *, color=BLUE, alpha=.75, linewidth=.72, max_lines=15000):
    if edges.empty:
        return 0
    x1, y1 = local_xy(edges.start_longitude, edges.start_latitude, lon0, lat0)
    x2, y2 = local_xy(edges.end_longitude, edges.end_latitude, lon0, lat0)
    good = np.isfinite(x1 + y1 + x2 + y2)
    good &= (np.abs(x1) <= width) & (np.abs(x2) <= width)
    good &= (np.abs(y1) <= width * .76) & (np.abs(y2) <= width * .76)
    indices = np.flatnonzero(good)
    if len(indices) > max_lines:
        indices = indices[np.linspace(0, len(indices)-1, max_lines, dtype=int)]
    if len(indices):
        segments = np.stack([
            np.stack([x1[indices], y1[indices]], axis=1),
            np.stack([x2[indices], y2[indices]], axis=1)
        ], axis=1)
        ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth, alpha=alpha, zorder=3))
    return len(indices)


def title(ax, tag, heading, sub=None):
    ax.set_title(f'{tag}  {heading}', loc='left', pad=12, fontsize=11.4, color=DARK, fontweight='bold')
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=SLATE,
                va='bottom', ha='left', fontsize=8.1)


def overlap_by_day(gap_frame, days, imputed):
    out = np.zeros(len(days), dtype=float)
    for _, r in gap_frame.loc[gap_frame.imputed.eq(imputed)].iterrows():
        starts = np.maximum(days.start_ms.to_numpy(dtype=np.int64), int(r.start_ms))
        ends = np.minimum(days.end_ms.to_numpy(dtype=np.int64), int(r.end_ms))
        out += np.maximum(0, ends-starts)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--participant', default='AMO327')
    p.add_argument('--window', default='AMO327__20231212')
    p.add_argument('--saved', type=Path, default=SAVED)
    p.add_argument('--out', type=Path, default=Path.home() / 'cortex2_visualizations' / 'outputs')
    args = p.parse_args()
    participant, window = args.participant, args.window
    if not window.startswith(participant + '__'):
        raise SystemExit('Window ID must belong to selected participant.')
    import pyarrow.parquet as pq

    status = pd.read_csv(args.saved / f'{participant}_window_status.csv')
    coverage = pd.read_csv(args.saved / f'{participant}_daily_coverage.csv')
    gaps = pd.read_csv(args.saved / f'{participant}_gap_diagnostics.csv')
    pooled = pd.read_csv(args.saved / f'{participant}_window_pooled_descriptive.csv')
    row = status.loc[status.clinical_window_id.eq(window)]
    if len(row) != 1 or not np.isfinite(row.iloc[0].window_start_ms):
        raise SystemExit(f'No valid window in saved status: {window}')
    s = row.iloc[0]
    start_ms, end_ms = int(s.window_start_ms), int(s.window_end_ms)
    days = coverage.loc[coverage.clinical_window_id.eq(window)].copy()
    if days.empty:
        raise SystemExit('Missing saved daily coverage for window')
    days['date'] = pd.to_datetime(days.date)
    days = days.sort_values('date').reset_index(drop=True)
    # Reconstruct day boundaries from actual window start and day ordering.
    days['start_ms'] = start_ms + np.arange(len(days), dtype=np.int64) * 86_400_000
    days['end_ms'] = np.minimum(days.start_ms + 86_400_000, end_ms)
    if int(days.end_ms.iloc[-1]) != end_ms:
        raise RuntimeError('Saved daily coverage is not a consecutive complete window; check requested days.')

    raw = pq.read_table(RAW, filters=[('participant_id', '=', participant)]).to_pandas()
    raw = raw.loc[(raw.timestamp >= start_ms) & (raw.timestamp < end_ms),
                  ['timestamp', 'latitude', 'longitude', 'accuracy']].copy()
    if len(raw) < 2:
        raise SystemExit('Too few raw GPS observations')
    qc = GPSQCConfig(max_gap_ms=60_000)
    clean = clean_gps_fixes(raw, qc)
    edges = build_gps_intervals(clean, window_start_ms=start_ms,
                                window_end_ms=end_ms, max_gap_ms=60_000)
    spatial = add_gps_spatial_semantics(edges.copy())
    trajectory = add_gps_trajectory_plausibility(spatial, max_plausible_speed_kmh=qc.max_speed_kmh)
    good = trajectory.supported.fillna(False).astype(bool) & ~trajectory.trajectory_status.eq('implausible_speed')
    episodes = build_anchor_compatible_location_episodes(spatial.loc[good].copy())
    assignments, places = build_place_candidates(episodes, compatibility_multiplier=1.0)
    visits = build_place_visits(assignments, min_endpoint_dwell_ms=600_000)
    trips = build_observed_trips(trajectory, visits, places)
    # Preserve exactly the same observed segmentation as recompute_timeline.
    assert int(len(trips)) == int(s.observed_trip_count), (
        f'Trip counts do not match saved pipeline: {len(trips)} vs {s.observed_trip_count}. '
        'Stop rather than presenting a non-equivalent figure.'
    )

    lon0, lat0 = float(clean.longitude.median()), float(clean.latitude.median())
    cx, cy = local_xy(clean.longitude, clean.latitude, lon0, lat0)
    finite = np.isfinite(cx + cy)
    # Robust map window: unclipped data remain in all calculations; only geographic view zooms.
    width = max(1.5, float(np.quantile(np.maximum(np.abs(cx[finite]), np.abs(cy[finite]) / .76), .96)) * 1.12)
    width = min(width, 100.0)  # limit zoom extent to preserve nearby point detail
    visible_n = int((finite & (np.abs(cx) <= width) & (np.abs(cy) <= .76*width)).sum())

    gg = gaps.loc[gaps.clinical_window_id.eq(window) & gaps.imputation.eq(0)].copy()
    filled_ms = overlap_by_day(gg, days, True)
    unresolved_ms = overlap_by_day(gg, days, False)
    durations = (days.end_ms - days.start_ms).to_numpy(dtype=float)
    observed = np.clip(days.observed_supported_ms.to_numpy(float) / durations, 0, 1)
    filled = np.clip(filled_ms / durations, 0, 1)
    unresolved = np.clip(unresolved_ms / durations, 0, 1)
    # Do not silently hide discrepancies among independently recorded support measures.
    discrepancy = np.abs(observed + filled + unresolved - 1)
    if (discrepancy > .025).any():
        print('WARNING: Daily stacked time fractions are not exhaustive on some dates;',
              'showing remaining unclassified time separately.')
    remainder = np.maximum(0, 1-observed-filled-unresolved)

    def pooled_feature(feature):
        x = pooled.loc[pooled.clinical_window_id.eq(window) & pooled.feature.eq(feature)]
        return float(x.estimate.iloc[0]) if len(x) else np.nan

    # Compute real trip-level path length with exactly the parameter used for AMO327.
    trips = add_smoothed_trip_path_distances(
        trajectory, trips, acceleration_noise_density_m_s1p5=1.0
    ) if len(trips) else trips

    # Select an actual well-observed trip for explaining L versus D. This is
    # illustrative selection for the slide, NOT a modification of the analysis.
    candidates = []
    for tr in trips.itertuples(index=False):
        start, end = int(tr.trip_start_ms), int(tr.trip_end_ms)
        tr_edges = trajectory.loc[
            good & (trajectory.start_ms >= start) & (trajectory.end_ms <= end)
        ].sort_values('start_ms')
        if len(tr_edges) < 4 or end <= start:
            continue
        support = float((tr_edges.end_ms - tr_edges.start_ms).clip(lower=0).sum()) / (end-start)
        L = float(getattr(tr, 'path_distance_smooth_obs_m', np.nan))
        D = float(getattr(tr, 'endpoint_displacement_obs_m', np.nan))
        if not (np.isfinite(L) and np.isfinite(D) and
                300 <= D <= L * 1.03 and 500 <= L <= 25000 and support >= .90):
            continue
        # Prefer visually interpretable moderately curved routes rather than
        # automatically selecting the longest trip.
        score = (min(L / max(D, 1), 2.5) - 1) * 5 + support
        candidates.append((score, tr, tr_edges, support, L, D))
    chosen = max(candidates, key=lambda x: x[0]) if candidates else None

    # Scientific illustration using the existing, validated AMO327 processing.
    # Panel A: actual candidate-place episode support. Panel B: an observed trip
    # between its assigned origin/destination. Panel C: a REAL reconstructed gap
    # shown in time (the pipeline does not export imputed path coordinates).
    # Panel D: observed trip geometry and calculated Kalman/RTS scalar L.
    # Prefer a supported trip between TWO distinct detected candidate places.
    place_ids = set(places.place_candidate_id.to_numpy())
    eligible = [v for v in candidates
        if getattr(v[1], 'origin_place_candidate_id', None) in place_ids
        and getattr(v[1], 'destination_place_candidate_id', None) in place_ids
        and getattr(v[1], 'origin_place_candidate_id') != getattr(v[1], 'destination_place_candidate_id')]
    if not eligible:
        raise RuntimeError('No well-supported trip between distinct candidate places; refusing to fabricate a route.')
    pair_candidates = [
        v for v in eligible
        if {
            int(v[1].origin_place_candidate_id),
            int(v[1].destination_place_candidate_id)
        } == {0, 3}
    ]

    filled_trip_gaps = gg.loc[gg.gap_type.eq('trip')]
    complete = []
    for v in pair_candidates:
        tr = v[1]
        t0, t1 = int(tr.trip_start_ms), int(tr.trip_end_ms)
        overlaps = filled_trip_gaps.loc[
            (filled_trip_gaps.start_ms < t1) &
            (filled_trip_gaps.end_ms > t0)
        ]
        if overlaps.empty:
            complete.append(v)

    if not complete:
        raise RuntimeError(
            "No eligible, fully observed trip for place pair (0, 3)."
        )

    _, example_trip, example_edges, support, example_L, example_D = max(
        complete, key=lambda v: v[0]
    )

    tr_id_start = getattr(example_trip, 'origin_place_candidate_id', None)
    tr_id_end = getattr(example_trip, 'destination_place_candidate_id', None)
    if tr_id_start is None or tr_id_end is None:
        raise RuntimeError('Trip lacks place IDs; cannot claim a place-to-place route.')
    tr_id_start, tr_id_end = int(tr_id_start), int(tr_id_end)
    place_index = places.set_index('place_candidate_id', drop=False)
    if tr_id_start not in place_index.index or tr_id_end not in place_index.index:
        raise RuntimeError('Trip origin/destination place IDs missing from place table.')
    if tr_id_start == tr_id_end:
        raise RuntimeError('Selected trip has same origin and destination place; choose another.')

    # Actual trip origin and destination candidate places determine the focus.
    pa = place_index.loc[tr_id_start]
    pb = place_index.loc[tr_id_end]
    midlon = (float(pa.representative_longitude) + float(pb.representative_longitude)) / 2
    midlat = (float(pa.representative_latitude) + float(pb.representative_latitude)) / 2
    trip_lon = np.r_[example_edges.start_longitude.to_numpy(float),
                     example_edges.end_longitude.to_numpy(float)]
    trip_lat = np.r_[example_edges.start_latitude.to_numpy(float),
                     example_edges.end_latitude.to_numpy(float)]
    trip_x, trip_y = local_xy(trip_lon, trip_lat, midlon, midlat)
    centers_lon = [float(pa.representative_longitude), float(pb.representative_longitude)]
    centers_lat = [float(pa.representative_latitude), float(pb.representative_latitude)]
    center_x, center_y = local_xy(centers_lon, centers_lat, midlon, midlat)
    extent = np.r_[np.abs(trip_x), np.abs(trip_y)/.76, np.abs(center_x), np.abs(center_y)/.76]
    focus_width = max(.38, 1.25 * float(np.nanmax(extent)))
    focus_width = min(focus_width, 12.0)

    # Place evidence comes from the episode-to-place assignments actually used
    # in this analysis: durations, not arbitrary point density or a fixed radius.
    needed = {'place_candidate_id', 'episode_start_ms', 'episode_end_ms'}
    if not needed.issubset(assignments.columns):
        raise RuntimeError('Cannot plot real place evidence: assignment schema changed.')
    ass = assignments.copy()
    ass['episode_duration_ms'] = (ass.episode_end_ms-ass.episode_start_ms).clip(lower=0)
    durations_by_place = ass.groupby('place_candidate_id').episode_duration_ms.sum()
    # Route-linked A-C v2: two places actually joined by the selected trip.
    ids = [0, 3, 7]
    if not set(ids).issubset(place_index.index):
        raise RuntimeError("Required detected places are missing.")

    palette = [ORANGE, '#8E76B4', TEAL]
    labels = ['Place 1', 'Place 3', 'Place 2']
    place_colors = dict(zip(ids, palette))
    place_labels = dict(zip(ids, labels))
    rsort = raw.sort_values('timestamp').reset_index(drop=True)
    times = rsort.timestamp.to_numpy(dtype=np.int64)
    lon_array, lat_array = rsort.longitude.to_numpy(float), rsort.latitude.to_numpy(float)
    place_evidence = []
    for pid in ids:
        ep = ass.loc[ass.place_candidate_id.eq(pid)]
        samples = []
        for t0, t1 in ep[['episode_start_ms','episode_end_ms']].itertuples(index=False, name=None):
            if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
                continue
            l = np.searchsorted(times, int(t0), 'left')
            r = np.searchsorted(times, int(t1), 'right')
            if r > l:
                samples.extend(np.linspace(l, r-1, min(150, r-l), dtype=int))
        sample = np.unique(samples)
        x, y = local_xy(lon_array[sample], lat_array[sample], midlon, midlat)
        center = place_index.loc[pid]
        xx, yy = local_xy([center.representative_longitude],
                          [center.representative_latitude], midlon, midlat)
        place_evidence.append(dict(id=pid, x=float(xx[0]), y=float(yy[0]),
            fixes_x=x, fixes_y=y, episodes=len(ep),
            hours=float(ep.episode_duration_ms.sum())/3_600_000,
            label=place_labels[pid], color=place_colors[pid]))

    def draw_place_evidence(ax, detailed=False):
        for rec in place_evidence:
            if len(rec['fixes_x']):
                ax.scatter(rec['fixes_x'], rec['fixes_y'],
                           color=rec['color'], s=9 if detailed else 4,
                           alpha=.38 if detailed else .18,
                           rasterized=True, edgecolors='none', zorder=4)
            ax.scatter([rec['x']], [rec['y']], s=115, color=rec['color'],
                       edgecolors='white', linewidth=1.3, zorder=8)
            ax.annotate(rec['label'], (rec['x'], rec['y']), xytext=(8,10),
                        textcoords='offset points', fontsize=10.3, color=rec['color'],
                        fontweight='bold', zorder=9,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.4))

    fig_style()
    plt.rcParams.update({'font.size': 11.5, 'axes.titlesize': 14, 'axes.labelsize': 11})
    fig, axs = plt.subplots(2, 2, figsize=(16.8, 9.7))
    fig.subplots_adjust(left=.075, right=.965, top=.88, bottom=.18,
                        wspace=.22, hspace=.32)
    a, b, c, d = axs.flat
    fig.suptitle('How GPS becomes a mobility measure', fontsize=24,
                 fontweight='bold', color=DARK, y=.97)
    fig.text(.5, .927, f'{participant}  |  Dec 06–12, 2023  |  real GPS observations',
             ha='center', fontsize=12.5, color=SLATE)

    # A. Real observed routes between the three detected places.
    # Place 1 = ID 0, Place 3 = ID 3, Place 2 = ID 7.
    if set(ids) != {0, 3, 7}:
        raise RuntimeError("Unexpected place IDs; no relabeling allowed.")

    def observed_pair_route(first, second):
        options = []
        for tr in trips.itertuples(index=False):
            pair = {
                int(tr.origin_place_candidate_id),
                int(tr.destination_place_candidate_id)
            }
            if pair != {first, second}:
                continue

            t0, t1 = int(tr.trip_start_ms), int(tr.trip_end_ms)
            edges = trajectory.loc[
                good &
                (trajectory.start_ms >= t0) &
                (trajectory.end_ms <= t1)
            ].sort_values('start_ms')

            if len(edges) < 4 or t1 <= t0:
                continue

            coverage = float(
                (edges.end_ms - edges.start_ms).clip(lower=0).sum()
            ) / (t1 - t0)

            if coverage >= .90:
                options.append((coverage, len(edges), edges))

        if not options:
            raise RuntimeError(
                f"No sufficiently observed trip for places {first}, {second}"
            )

        return max(options, key=lambda v: (v[0], v[1]))[2]

    # Use the SAME real trip as B/D for Place 1–Place 3.
    route_13 = example_edges

    # An independently observed real trip for Place 1–Place 2.
    route_12 = observed_pair_route(0, 7)

    # Calculate the map extent using actual locations and both paths.
    all_lon = np.r_[
        route_13.start_longitude.to_numpy(float),
        route_13.end_longitude.to_numpy(float),
        route_12.start_longitude.to_numpy(float),
        route_12.end_longitude.to_numpy(float),
        [place_index.loc[k].representative_longitude for k in ids]
    ]
    all_lat = np.r_[
        route_13.start_latitude.to_numpy(float),
        route_13.end_latitude.to_numpy(float),
        route_12.start_latitude.to_numpy(float),
        route_12.end_latitude.to_numpy(float),
        [place_index.loc[k].representative_latitude for k in ids]
    ]
    mx, my = local_xy(all_lon, all_lat, midlon, midlat)
    a_focus = max(
        focus_width,
        1.12 * float(np.nanmax(
            np.r_[np.abs(mx), np.abs(my) / .76]
        ))
    )

    title(
        a, 'A', 'Three detected places',
        'Observed GPS routes connecting Place 1 with Places 2 and 3'
    )
    map_axes(a, a_focus)

    rx, ry = local_xy(
        rsort.longitude, rsort.latitude, midlon, midlat
    )
    visible = np.flatnonzero(
        np.isfinite(rx + ry) &
        (np.abs(rx) <= a_focus) &
        (np.abs(ry) <= a_focus * .76)
    )
    step = max(1, len(visible) // 11000)

    a.scatter(
        rx[visible[::step]], ry[visible[::step]],
        s=1.4, color=SLATE, alpha=.06,
        edgecolors='none', rasterized=True, zorder=1
    )

    # Plot real GPS observations as points, not dotted line segments.
    # No interpolated coordinates or straight-line connectors.
    # Panel A: all observed GPS segment endpoints, without thinning.
    # Both routes have identical dot size and opacity.
    # No lines, interpolation, or invented coordinates.
    # Uniformly spaced display dots along actual GPS-supported edges.
    # These are rendering samples, not additional observed GPS fixes.
    # Each edge is handled independently; unsupported intervals are not bridged.
    def draw_gps_points(edges, color):
        x0, y0 = local_xy(
            edges.start_longitude.to_numpy(float),
            edges.start_latitude.to_numpy(float),
            midlon, midlat
        )
        x1, y1 = local_xy(
            edges.end_longitude.to_numpy(float),
            edges.end_latitude.to_numpy(float),
            midlon, midlat
        )

        dots_x, dots_y = [], []
        spacing_km = .008  # One display dot every 8 meters.

        for xa, ya, xb, yb in zip(x0, y0, x1, y1):
            if not np.isfinite([xa, ya, xb, yb]).all():
                continue

            length = float(np.hypot(xb - xa, yb - ya))

            if length == 0:
                dots_x.append(xa)
                dots_y.append(ya)
                continue

            n = max(1, int(np.ceil(length / spacing_km)))
            t = np.linspace(0, 1, n + 1)

            dots_x.extend(xa + t * (xb - xa))
            dots_y.extend(ya + t * (yb - ya))

        a.scatter(
            dots_x, dots_y,
            s=2.2,
            c=color,
            alpha=.95,
            marker='o',
            edgecolors='none',
            linewidths=0,
            rasterized=True,
            zorder=6
        )

    # Both paths have identical dot size, density, and opacity.
    draw_gps_points(route_13, BLUE)
    draw_gps_points(route_12, '#159A87')

    # Preserve the three real detected place centers and episode fixes.
    for rec in place_evidence:
        xx = np.asarray(rec['fixes_x'])
        yy = np.asarray(rec['fixes_y'])
        nearby = np.hypot(
            xx - rec['x'], yy - rec['y']
        ) <= .12

        a.scatter(
            xx[nearby], yy[nearby],
            s=9, c=rec['color'], alpha=.46,
            edgecolors='none', rasterized=True, zorder=5
        )
        a.scatter(
            [rec['x']], [rec['y']],
            s=135, c=rec['color'],
            edgecolors='white', linewidth=1.4, zorder=8
        )
        a.annotate(
            rec['label'], (rec['x'], rec['y']),
            xytext=(8, 10), textcoords='offset points',
            fontsize=11, fontweight='bold',
            color=rec['color'], zorder=9,
            bbox=dict(
                facecolor='white', edgecolor='none',
                alpha=.93, pad=1.5
            )
        )

    # B. Exactly the real observed trip linked to the corresponding place IDs.
    title(b,'B','Movement between detected places',
          f'Observed trip: {place_labels[tr_id_start]} to {place_labels[tr_id_end]}')
    map_axes(b,focus_width)
    draw_edges(b,trajectory.loc[good],midlon,midlat,focus_width,
               color='#CAD8DF',alpha=.30,linewidth=.62)
    draw_edges(b,example_edges,midlon,midlat,focus_width,
               color=BLUE,alpha=.96,linewidth=2.4)
    for rec in place_evidence[:2]:
        b.scatter([rec['x']],[rec['y']],s=155,c=rec['color'],
                  edgecolors='white',linewidth=1.4,zorder=8)
        b.annotate(rec['label'],(rec['x'],rec['y']),xytext=(8,9),
                   textcoords='offset points',fontsize=11,color=rec['color'],
                   fontweight='bold',zorder=9,
                   bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=1.4))
    # Repeat count only for the exactly matched origin/destination pair.
    has_pair={'origin_place_candidate_id','destination_place_candidate_id'}.issubset(trips.columns)
    n_pair=int(((trips.origin_place_candidate_id.eq(tr_id_start) &
                 trips.destination_place_candidate_id.eq(tr_id_end)) |
                (trips.origin_place_candidate_id.eq(tr_id_end) &
                 trips.destination_place_candidate_id.eq(tr_id_start))).sum()) if has_pair else 0
    b.text(.02,.03, f'{n_pair} detected trips between these places in this week',
           transform=b.transAxes,fontsize=10.1,color=DARK,
           bbox=dict(facecolor='white',edgecolor='#DCE7ED',alpha=.96,pad=3))

    # C. Actual imputed gap on the same place pair (0, 3).
    title(c, 'C', 'Imputation on this same place-to-place route',
          'A different trip between these places with a verified GPS gap')
    map_axes(c, focus_width)

    for rec in place_evidence[:2]:
        c.scatter(
            [rec['x']], [rec['y']], s=130,
            c=rec['color'], edgecolors='white',
            linewidth=1.4, zorder=8
        )
        c.annotate(
            rec['label'], (rec['x'], rec['y']),
            xytext=(8, 10), textcoords='offset points',
            fontsize=10.5, fontweight='bold',
            color=rec['color'],
            bbox=dict(
                facecolor='white', alpha=.94,
                edgecolor='none', pad=1.5
            )
        )

    pair_trips=trips.loc[
        (trips.origin_place_candidate_id.eq(tr_id_start)&
         trips.destination_place_candidate_id.eq(tr_id_end))|
        (trips.origin_place_candidate_id.eq(tr_id_end)&
         trips.destination_place_candidate_id.eq(tr_id_start))].copy()
    filled_g=gg.loc[gg.imputed.eq(True)&gg.gap_type.eq('trip')]
    matching=[]
    for tr in pair_trips.itertuples(index=False):
        gs=filled_g.loc[(filled_g.start_ms>=int(tr.trip_start_ms))&
                        (filled_g.end_ms<=int(tr.trip_end_ms))]
        for gr in gs.itertuples(index=False):
            matching.append((float(gr.gap_ms),tr,gr))
    if matching:
        _, tr, gr=max(matching,key=lambda t:t[0])
        t0,t1=int(gr.start_ms),int(gr.end_ms)
        date=pd.to_datetime(t0,unit='ms',utc=True)
        obs_edges=trajectory.loc[
            good & (trajectory.start_ms>=int(tr.trip_start_ms)) &
            (trajectory.end_ms<=int(tr.trip_end_ms)) &
            ~((trajectory.start_ms<t1)&(trajectory.end_ms>t0))]
        draw_edges(c,obs_edges,midlon,midlat,focus_width,
                   color=BLUE,alpha=.96,linewidth=2.4)
        sorted_clean=clean.sort_values('timestamp')
        before=sorted_clean.loc[sorted_clean.timestamp<=t0].tail(1)
        after=sorted_clean.loc[sorted_clean.timestamp>=t1].head(1)
        if len(before) and len(after):
            bx,by=local_xy([float(before.longitude.iloc[0]),float(after.longitude.iloc[0])],
                           [float(before.latitude.iloc[0]),float(after.latitude.iloc[0])],
                           midlon,midlat)
            c.plot(bx,by,ls=':',color=ORANGE,lw=2.5,zorder=5)
            c.scatter(bx,by,s=60,color=ORANGE,edgecolors='white',zorder=7)
        donors=int(gr.n_candidate_donors) if pd.notna(gr.n_candidate_donors) else 0
        fig.text(
            .075, .078,
            f'{date:%b %d, %Y %H:%M} UTC | '
            f'{gr.gap_ms/60000:.1f} min GPS gap\n'
            f'Imputed (candidate donors: {donors}); '
            'dotted line connects observed gap-boundary fixes only',
            ha='left', va='top',
            fontsize=9.5, color=DARK
        )
    else:
        raise RuntimeError(
            "No verified imputed trip gap on pair (0, 3)."
        )

    # D. Same actual well-supported trip, with path length and straight-line D.
    t0=pd.to_datetime(int(example_trip.trip_start_ms),unit='ms',utc=True)
    t1=pd.to_datetime(int(example_trip.trip_end_ms),unit='ms',utc=True)
    title(d,'D','Two ways to measure a trip',f'Real observed trip: {t0:%b %d, %H:%M}–{t1:%H:%M} UTC')
    d.set_xlabel('East–west distance relative to trip (km)')
    d.set_ylabel('North–south distance relative to trip (km)')
    d.grid(True,color='#E9EEF2',linewidth=.75)
    start_lon=float(example_edges.start_longitude.iloc[0])
    start_lat=float(example_edges.start_latitude.iloc[0])
    x1,y1=local_xy(example_edges.start_longitude,example_edges.start_latitude,start_lon,start_lat)
    x2,y2=local_xy(example_edges.end_longitude,example_edges.end_latitude,start_lon,start_lat)
    seg=np.stack([np.stack([x1,y1],axis=1),np.stack([x2,y2],axis=1)],axis=1)
    d.add_collection(LineCollection(seg,colors=BLUE,linewidths=2.7,alpha=.96,zorder=4))
    d.plot([x1[0],x2[-1]],[y1[0],y2[-1]],'--',color=ORANGE,lw=3,zorder=5)
    allx=np.r_[x1,x2]; ally=np.r_[y1,y2]
    pad=max(.08,.12*max(np.ptp(allx),np.ptp(ally)))
    d.set_xlim(allx.min()-pad,allx.max()+pad)
    d.set_ylim(ally.min()-pad,ally.max()+pad)
    d.set_aspect('equal',adjustable='datalim')
    for xx, yy, pid in [
        (float(x1[0]), float(y1[0]), tr_id_start),
        (float(x2[-1]), float(y2[-1]), tr_id_end)
    ]:
        d.annotate(
            place_labels[pid], (xx, yy),
            xytext=(0, 12), textcoords='offset points',
            ha='center', va='bottom',
            fontsize=10.5, fontweight='bold',
            color=place_colors[pid], zorder=10,
            bbox=dict(
                facecolor='white', alpha=.94,
                edgecolor='none', pad=1.5
            )
        )

    d.text(.025,.96,f'L = {example_L/1000:.2f} km  ·  smoothed supported path',
           transform=d.transAxes,fontsize=11.1,fontweight='bold',color=BLUE,
           va='top',bbox=dict(facecolor='white',alpha=.94,edgecolor='none',pad=3))
    d.text(.025,.885,f'D = {example_D/1000:.2f} km  ·  endpoint separation',
           transform=d.transAxes,fontsize=11.1,fontweight='bold',color='#B67316',
           va='top',bbox=dict(facecolor='white',alpha=.94,edgecolor='none',pad=3))
    d.legend(handles=[Line2D([],[],color=BLUE,lw=2.8,label='Actual GPS-supported route'),
                      Line2D([],[],color=ORANGE,lw=2.8,ls='--',label='Straight-line separation')],
             loc='lower right',fontsize=9.7,framealpha=.96)
    print(f'Illustrative trip: {t0.isoformat()} to {t1.isoformat()}'
          f' | L={example_L:.1f}m D_endpoint={example_D:.1f}m support={support:.1%}')

    # ------------------------------------------------------------
    # SLIDE 2. Missingness and weekly outputs
    # ------------------------------------------------------------
    fig2, (e, f) = plt.subplots(1, 2, figsize=(15.4, 8.0),
                               gridspec_kw={'width_ratios': [1.2, 1]})
    fig2.subplots_adjust(left=.085, right=.96, top=.825, bottom=.145, wspace=.25)
    fig2.suptitle('What happens when GPS is missing?', fontsize=23,
                  fontweight='bold', color=DARK, y=.955)
    fig2.text(.5, .902, f'{participant}  |  Dec 06–12, 2023  |  observed and imputed are reported separately',
              ha='center', fontsize=12, color=SLATE)

    title(e, 'E', 'Where GPS is observed or missing',
          'Proportion of each day; based on actual saved gap diagnostics')
    yy = np.arange(len(days))
    e.barh(yy, observed*100, height=.59, color=BLUE, label='Observed GPS')
    e.barh(yy, filled*100, left=observed*100, height=.59,
           color=TEAL, label='Imputed gaps')
    e.barh(yy, unresolved*100, left=(observed+filled)*100, height=.59,
           color=ORANGE, label='Unresolved GPS time')
    if np.any(remainder > .001):
        e.barh(yy, remainder*100, left=(observed+filled+unresolved)*100,
               height=.59, color='#D8E0E6', label='Other unclassified time')
    e.set_yticks(yy, [x.strftime('%b %d') for x in days.date])
    e.invert_yaxis()
    e.set_xlim(0, 100)
    e.set_xlabel('Share of the day (%)')
    e.grid(axis='x', color='#E9EEF2', linewidth=.85)
    e.set_axisbelow(True)
    e.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2,
             fontsize=10, frameon=False)
    e.spines[['top', 'right']].set_visible(False)

    title(f, 'F', 'Weekly mobility: observed vs. imputed',
          'Actual saved AMO327 outputs; incomplete window')
    f.spines[:].set_visible(False)
    f.set_xticks([]); f.set_yticks([])
    f.set_xlim(0, 1); f.set_ylim(0, 1)
    imp_l = pooled_feature('path_length_m') / 1000
    observed_l = float(s.observed_path_length_m) / 1000
    maximum = max(observed_l, imp_l, 1) * 1.15
    f.text(.025, .86, 'Path length (km)', fontsize=14, fontweight='bold', color=DARK)
    for y, label, value, color in [(.68, 'Observed GPS-supported L', observed_l, BLUE),
                                    (.45, 'With reconstructed gaps (exploratory)', imp_l, TEAL)]:
        f.text(.025, y+.085, label, fontsize=11.5, color=DARK)
        f.add_patch(plt.Rectangle((.025, y-.008), .68*value/maximum, .084,
                                  color=color, transform=f.transAxes, linewidth=0))
        f.text(.98, y+.015, f'{value:.2f}', ha='right', va='center',
               fontsize=16, fontweight='bold', color=color)
    observed_share = float(s.observed_supported_duration_ms)/(end_ms-start_ms)
    imputed_share = float(filled_ms.sum())/(end_ms-start_ms)
    unresolved_share = float(unresolved_ms.sum())/(end_ms-start_ms)
    f.text(.04, .33,
           f'{int(s.observed_trip_count)} observed trips\n'
           f'{observed_share:.1%} observed GPS time  |  {imputed_share:.1%} reconstructed\n'
           f'{unresolved_share:.1%} still unresolved',
           va='top', fontsize=12.1, linespacing=1.7,
           bbox=dict(boxstyle='round,pad=.65', facecolor=PALE,
                     edgecolor='none'))
    f.text(.035, .035, 'Not a complete estimate of travel; no verified ground truth for gaps.',
           fontsize=10, color=SLATE)
    fig2.text(.5, .039, 'Imputed time is the proportion of the window represented by gap reconstructions; '
              'it does not imply that the reconstructed movement is correct.',
              fontsize=9.5, ha='center', color=SLATE)

    args.out.mkdir(parents=True, exist_ok=True)
    for figure, suffix in [(fig, '01_observed_gps_to_trips'),
                           (fig2, '02_missingness_and_weekly_mobility')]:
        stem = f'{participant}_{window.split("__")[-1]}_{suffix}'
        for ext in ('png', 'pdf', 'svg'):
            dest = args.out / f'{stem}.{ext}'
            figure.savefig(dest, dpi=220, bbox_inches='tight', pad_inches=.18)
            print('Saved:', dest)
        plt.close(figure)
    print('Verified observed trip count:', len(trips))
    print('Raw fixes:', len(raw), '| QC fixes:', len(clean),
          '| places:', len(places), '| visits:', len(visits))


if __name__ == '__main__':
    main()
