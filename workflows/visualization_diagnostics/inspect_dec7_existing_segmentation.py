"""Read-only audit of existing Cortex trip and visit boundaries."""

from pathlib import Path
import runpy
import pandas as pd

from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes
)
from cortex2_engine.features.gps_place_candidates import build_place_candidates
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility
)
from cortex2_engine.features.gps_trip_features import (
    build_place_visits, build_observed_trips
)

# Reuse the existing, verified Dec 7 reverse-direction match.
ctx = runpy.run_path(
    "workflows/visualization_diagnostics/check_dec7_reverse_route.py"
)

clean = ctx["clean"]
section = ctx["reverse_section"]

lo = int(section.timestamp.min())
hi = int(section.timestamp.max())

# Same full clinical-window place model as the existing workflow.
edges = build_gps_intervals(
    clean,
    window_start_ms=int(clean.timestamp.min()),
    window_end_ms=int(clean.timestamp.max()) + 1,
    max_gap_ms=60000,
)

spatial = add_gps_spatial_semantics(edges.copy())
supported = spatial.loc[
    spatial.supported.fillna(False)
].copy()

episodes = build_anchor_compatible_location_episodes(supported)
assignments, places = build_place_candidates(
    episodes, compatibility_multiplier=1.0
)
visits = build_place_visits(
    assignments, min_endpoint_dwell_ms=600000
)

trajectory = add_gps_trajectory_plausibility(
    spatial, max_plausible_speed_kmh=1000.0
)
trips = build_observed_trips(trajectory, visits, places)


def utc(ms):
    return str(pd.to_datetime(int(ms), unit="ms", utc=True))


def show_overlap(name, frame, possible_pairs):
    print(f"\n{name}")

    if frame.empty:
        print("No records.")
        return

    pair = next(
        (
            (a, b) for a, b in possible_pairs
            if a in frame.columns and b in frame.columns
        ),
        None
    )

    if pair is None:
        print("Time column names:", list(frame.columns))
        print("Cannot establish overlap without known time columns.")
        return

    a, b = pair

    selected = frame.loc[
        (pd.to_numeric(frame[a], errors="coerce") <= hi)
        & (pd.to_numeric(frame[b], errors="coerce") >= lo)
    ].copy()

    print("Overlapping records:", len(selected))

    if selected.empty:
        return

    selected["start_utc"] = pd.to_datetime(
        selected[a], unit="ms", utc=True
    )
    selected["end_utc"] = pd.to_datetime(
        selected[b], unit="ms", utc=True
    )
    selected["duration_min"] = (
        pd.to_numeric(selected[b]) - pd.to_numeric(selected[a])
    ) / 60000

    useful = [
        "start_utc", "end_utc", "duration_min",
        "place_candidate_id", "place_id",
        "start_place_candidate_id", "end_place_candidate_id",
        "start_place_id", "end_place_id",
    ]
    cols = [c for c in useful if c in selected.columns]

    print(selected[cols].to_string(index=False, max_rows=25))


print("\nDEC 7 — EXISTING SEGMENTATION")
print("=" * 55)
print("Matched section:", utc(lo), "to", utc(hi))
print("Duration (minutes):", round((hi - lo) / 60000, 2))

show_overlap(
    "LOCATION EPISODES",
    episodes,
    [
        ("episode_start_ms", "episode_end_ms"),
        ("start_ms", "end_ms"),
    ],
)

show_overlap(
    "DETECTED PLACE VISITS",
    visits,
    [
        ("visit_start_ms", "visit_end_ms"),
        ("start_ms", "end_ms"),
        ("arrival_ms", "departure_ms"),
    ],
)

show_overlap(
    "OBSERVED TRIPS",
    trips,
    [
        ("trip_start_ms", "trip_end_ms"),
        ("start_ms", "end_ms"),
    ],
)
