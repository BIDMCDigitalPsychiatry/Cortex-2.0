#!/usr/bin/env python3

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch


# ============================================================
# PATHS
# ============================================================

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))

from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)
from cortex2_engine.features.gps_place_candidates import build_place_candidates


PARTICIPANT = "RSH634"

DISPLAY_START = pd.Timestamp("2025-04-23")
DISPLAY_END = pd.Timestamp("2025-05-07")

WORKFLOW = REPO / "workflows" / "build_weekly_home_time.py"

OUTPUT_ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)

RAW_GPS = (
    OUTPUT_ROOT
    / "feature_reliability"
    / "gps_clinical"
    / "gps_clinical_weekly_raw_points.parquet"
)

OUTDIR = REPO / "home_derivation_RSH634"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ THE EXACT CONSTANTS FROM THE CURRENT WORKFLOW
# ============================================================

def read_literal_constants(path: Path):
    source = path.read_text()
    tree = ast.parse(source)

    wanted = {
        "MAX_GAP_MS",
        "COMPATIBILITY_MULTIPLIER",
        "LOCAL_TIMEZONE",
        "NIGHT_START_HOUR",
        "NIGHT_END_HOUR",
    }

    values = {}

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            value_node = node.value
        else:
            target = node.target
            value_node = node.value

        if not isinstance(target, ast.Name):
            continue

        if target.id not in wanted:
            continue

        try:
            values[target.id] = ast.literal_eval(value_node)
        except Exception:
            pass

    missing = wanted - set(values)

    if missing:
        raise RuntimeError(
            "Could not recover these constants directly from "
            f"{path}: {sorted(missing)}\n"
            "I am stopping instead of inventing their values."
        )

    return values


C = read_literal_constants(WORKFLOW)

MAX_GAP_MS = int(C["MAX_GAP_MS"])
COMPATIBILITY_MULTIPLIER = float(C["COMPATIBILITY_MULTIPLIER"])
LOCAL_TIMEZONE = str(C["LOCAL_TIMEZONE"])
NIGHT_START_HOUR = int(C["NIGHT_START_HOUR"])
NIGHT_END_HOUR = int(C["NIGHT_END_HOUR"])

print("=" * 80)
print("CURRENT CORTEX-2.0 HOME-TIME PARAMETERS")
print("=" * 80)
print("MAX_GAP_MS:", MAX_GAP_MS)
print("COMPATIBILITY_MULTIPLIER:", COMPATIBILITY_MULTIPLIER)
print("LOCAL_TIMEZONE:", LOCAL_TIMEZONE)
print(
    "NIGHT WINDOW:",
    f"{NIGHT_START_HOUR:02d}:00–{NIGHT_END_HOUR:02d}:00",
)


# ============================================================
# LOAD REAL PARTICIPANT GPS
# ============================================================

print("\nLoading raw GPS for", PARTICIPANT)

try:
    raw = pd.read_parquet(
        RAW_GPS,
        filters=[
            ("participant_id", "==", PARTICIPANT),
        ],
    )
except Exception:
    raw = pd.read_parquet(RAW_GPS)
    raw = raw[
        raw["participant_id"].astype(str).eq(PARTICIPANT)
    ].copy()

if raw.empty:
    raise RuntimeError(
        f"No raw GPS found for {PARTICIPANT}"
    )

for c in ["timestamp", "latitude", "longitude", "accuracy"]:
    raw[c] = pd.to_numeric(
        raw[c],
        errors="coerce",
    )

raw = raw.dropna(
    subset=["timestamp", "latitude", "longitude"]
).copy()

raw["timestamp"] = raw["timestamp"].astype("int64")

if "date" not in raw.columns:
    raw["date"] = (
        pd.to_datetime(
            raw["timestamp"],
            unit="ms",
            utc=True,
        )
        .dt.date
        .astype(str)
    )

print("Raw rows:", f"{len(raw):,}")
print(
    "Participant dates:",
    raw["date"].min(),
    "to",
    raw["date"].max(),
)


# ============================================================
# REPRODUCE THE EXACT HOME-TIME WORKFLOW
# ============================================================

all_microsegments = []
day_supported_duration = {}
daily_supported_intervals = {}


for date, day_raw in raw.groupby(
    "date",
    sort=True,
):

    day_raw = (
        day_raw
        .sort_values("timestamp")
        .copy()
    )

    if (
        "day_start_ms" in day_raw.columns
        and "day_end_ms" in day_raw.columns
    ):
        start_ms = int(
            day_raw["day_start_ms"].iloc[0]
        )
        end_ms = int(
            day_raw["day_end_ms"].iloc[0]
        )

    else:
        date_utc = pd.Timestamp(
            date,
            tz="UTC",
        )

        start_ms = int(
            date_utc.timestamp()
            * 1000
        )

        end_ms = int(
            (
                date_utc
                + pd.Timedelta(days=1)
            ).timestamp()
            * 1000
        )

    intervals = build_gps_intervals(
        day_raw[
            [
                "timestamp",
                "latitude",
                "longitude",
                "accuracy",
            ]
        ],
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        max_gap_ms=MAX_GAP_MS,
    )

    if intervals.empty:
        day_supported_duration[date] = 0.0
        daily_supported_intervals[date] = intervals
        continue

    supported = intervals[
        intervals["supported"]
        .fillna(False)
        .astype(bool)
    ].copy()

    daily_supported_intervals[date] = supported.copy()

    supported_duration_ms = float(
        pd.to_numeric(
            supported["duration_ms"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    day_supported_duration[date] = (
        supported_duration_ms
    )

    if supported.empty:
        continue

    spatial = add_gps_spatial_semantics(
        supported
    )

    microsegments = (
        build_anchor_compatible_location_episodes(
            spatial
        )
    )

    if microsegments.empty:
        continue

    microsegments = microsegments.copy()

    microsegments["participant_id"] = PARTICIPANT
    microsegments["date"] = date

    midpoint_ms = (
        (
            pd.to_numeric(
                microsegments["episode_start_ms"],
                errors="coerce",
            )
            +
            pd.to_numeric(
                microsegments["episode_end_ms"],
                errors="coerce",
            )
        )
        / 2.0
    )

    local_time = (
        pd.to_datetime(
            midpoint_ms,
            unit="ms",
            utc=True,
        )
        .dt.tz_convert(
            LOCAL_TIMEZONE
        )
    )

    microsegments["local_hour"] = (
        local_time.dt.hour
    )

    microsegments["is_nighttime"] = (
        (
            microsegments["local_hour"]
            >= NIGHT_START_HOUR
        )
        &
        (
            microsegments["local_hour"]
            < NIGHT_END_HOUR
        )
    )

    all_microsegments.append(
        microsegments
    )


if not all_microsegments:
    raise RuntimeError(
        "No location microsegments were created."
    )


participant_segments = pd.concat(
    all_microsegments,
    ignore_index=True,
)


assignments, places = build_place_candidates(
    participant_segments,
    compatibility_multiplier=(
        COMPATIBILITY_MULTIPLIER
    ),
)


if assignments.empty:
    raise RuntimeError(
        "No persistent place assignments."
    )


# Preserve metadata exactly as the workflow does.
for col in [
    "date",
    "local_hour",
    "is_nighttime",
]:
    if col not in assignments.columns:
        assignments[col] = (
            participant_segments[col]
            .to_numpy()
        )


# ============================================================
# HOME = MOST NIGHTTIME-SUPPORTED PLACE
# ============================================================

nighttime = assignments[
    assignments["is_nighttime"]
    .fillna(False)
    .astype(bool)
].copy()


if nighttime.empty:
    raise RuntimeError(
        "No nighttime location evidence."
    )


night_by_place = (
    nighttime.groupby(
        "place_candidate_id",
        as_index=False,
    )["duration_ms"]
    .sum()
    .sort_values(
        "duration_ms",
        ascending=False,
    )
    .reset_index(drop=True)
)


home_place_id = int(
    night_by_place.iloc[0][
        "place_candidate_id"
    ]
)


total_night_duration = float(
    night_by_place["duration_ms"].sum()
)

home_night_duration = float(
    night_by_place.iloc[0]["duration_ms"]
)

home_night_fraction = (
    home_night_duration
    / total_night_duration
)


print("\n" + "=" * 80)
print("HOME IDENTIFICATION")
print("=" * 80)
print("Persistent places:", len(places))
print("Home place ID:", home_place_id)
print(
    "Observed nighttime duration:",
    round(
        total_night_duration
        / 3_600_000,
        2,
    ),
    "hours",
)
print(
    "Nighttime duration at home:",
    round(
        home_night_duration
        / 3_600_000,
        2,
    ),
    "hours",
)
print(
    "Nighttime fraction at home:",
    round(home_night_fraction, 3),
)


# ============================================================
# SAVE REAL INTERMEDIATE DATA
# ============================================================

assignments.to_csv(
    OUTDIR / "RSH634_place_assignments.csv",
    index=False,
)

places.to_csv(
    OUTDIR / "RSH634_candidate_places.csv",
    index=False,
)

night_by_place.to_csv(
    OUTDIR / "RSH634_nighttime_occupancy_by_place.csv",
    index=False,
)


# ============================================================
# DAILY HOME TIME
# ============================================================

daily_rows = []

for date in sorted(
    day_supported_duration
):

    supported_duration_ms = float(
        day_supported_duration[date]
    )

    day_assignments = assignments[
        assignments["date"]
        .astype(str)
        .eq(str(date))
    ].copy()

    home_duration_ms = float(
        pd.to_numeric(
            day_assignments.loc[
                day_assignments[
                    "place_candidate_id"
                ].eq(home_place_id),
                "duration_ms",
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    home_fraction = (
        home_duration_ms
        / supported_duration_ms
        if supported_duration_ms > 0
        else np.nan
    )

    daily_rows.append({
        "date": str(date),
        "supported_duration_ms":
            supported_duration_ms,
        "home_duration_ms":
            home_duration_ms,
        "time_at_home_fraction":
            home_fraction,
    })


daily = pd.DataFrame(daily_rows)

daily.to_csv(
    OUTDIR / "RSH634_daily_home_time_rebuilt.csv",
    index=False,
)


# ============================================================
# DISPLAY WINDOW
# ============================================================

display_assignments = assignments[
    pd.to_datetime(
        assignments["date"]
    ).between(
        DISPLAY_START,
        DISPLAY_END
        - pd.Timedelta(days=1),
    )
].copy()


display_raw = raw[
    pd.to_datetime(
        raw["date"]
    ).between(
        DISPLAY_START,
        DISPLAY_END
        - pd.Timedelta(days=1),
    )
].copy()


display_daily = daily[
    pd.to_datetime(
        daily["date"]
    ).between(
        DISPLAY_START,
        DISPLAY_END
        - pd.Timedelta(days=1),
    )
].copy()


# ============================================================
# CHOOSE A REPRESENTATIVE REAL DAY
#
# Choose a well-observed day whose home fraction is closest
# to the participant's 14-day home fraction.
# ============================================================

display_daily["supported_h"] = (
    display_daily["supported_duration_ms"]
    / 3_600_000
)

display_daily["home_h"] = (
    display_daily["home_duration_ms"]
    / 3_600_000
)

window_supported = (
    display_daily["supported_duration_ms"]
    .sum()
)

window_home = (
    display_daily["home_duration_ms"]
    .sum()
)

window_home_fraction = (
    window_home / window_supported
)


eligible_days = display_daily[
    display_daily["supported_h"] >= 20
].copy()

if eligible_days.empty:
    eligible_days = display_daily.copy()


eligible_days["distance_from_window"] = (
    eligible_days[
        "time_at_home_fraction"
    ]
    .sub(window_home_fraction)
    .abs()
)


representative_day_row = (
    eligible_days
    .sort_values(
        [
            "distance_from_window",
            "supported_h",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .iloc[0]
)


REPRESENTATIVE_DAY = str(
    representative_day_row["date"]
)


print("\nRepresentative display day:")
print(REPRESENTATIVE_DAY)
print(
    "Supported:",
    round(
        representative_day_row[
            "supported_h"
        ],
        2,
    ),
    "h",
)
print(
    "Home:",
    round(
        representative_day_row[
            "home_h"
        ],
        2,
    ),
    "h",
)
print(
    "Home fraction:",
    round(
        representative_day_row[
            "time_at_home_fraction"
        ],
        3,
    ),
)


# ============================================================
# GEOGRAPHIC HELPERS
# ============================================================

home_row = places[
    places["place_candidate_id"]
    .eq(home_place_id)
].iloc[0]


home_lat = float(
    home_row["representative_latitude"]
)

home_lon = float(
    home_row["representative_longitude"]
)


EARTH_RADIUS_M = 6_371_000.0


def project_m(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lat0 = math.radians(home_lat)

    x = (
        EARTH_RADIUS_M
        * np.deg2rad(
            lon - home_lon
        )
        * math.cos(lat0)
    )

    y = (
        EARTH_RADIUS_M
        * np.deg2rad(
            lat - home_lat
        )
    )

    return x, y


def convex_hull(points):
    """
    Monotonic-chain convex hull.
    Used only to visualize the real observed footprint of
    assigned microsegments. The hull is NOT a Cortex threshold.
    """

    pts = sorted(
        set(
            (
                float(x),
                float(y),
            )
            for x, y in points
            if np.isfinite(x)
            and np.isfinite(y)
        )
    )

    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (
            (a[0] - o[0])
            * (b[1] - o[1])
            -
            (a[1] - o[1])
            * (b[0] - o[0])
        )

    lower = []

    for p in pts:
        while (
            len(lower) >= 2
            and cross(
                lower[-2],
                lower[-1],
                p,
            ) <= 0
        ):
            lower.pop()

        lower.append(p)

    upper = []

    for p in reversed(pts):
        while (
            len(upper) >= 2
            and cross(
                upper[-2],
                upper[-1],
                p,
            ) <= 0
        ):
            upper.pop()

        upper.append(p)

    return (
        lower[:-1]
        + upper[:-1]
    )


# ============================================================
# SELECT TOP PLACES FOR DISPLAY
# ============================================================

display_place_time = (
    display_assignments
    .groupby(
        "place_candidate_id"
    )["duration_ms"]
    .sum()
    .sort_values(
        ascending=False
    )
)


top_place_ids = list(
    display_place_time
    .head(6)
    .index
)


if home_place_id not in top_place_ids:
    top_place_ids = (
        [home_place_id]
        + top_place_ids[:5]
    )


# ============================================================
# CREATE FIGURE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
})


fig = plt.figure(
    figsize=(16, 9)
)

gs = fig.add_gridspec(
    2,
    2,
    left=0.055,
    right=0.97,
    top=0.86,
    bottom=0.08,
    wspace=0.20,
    hspace=0.32,
)


fig.suptitle(
    "How Cortex-2.0 derives time at home from raw GPS",
    x=0.055,
    ha="left",
    fontsize=24,
    fontweight="bold",
)


fig.text(
    0.055,
    0.90,
    (
        "Real Digital Clinic participant RSH634 | "
        "candidate places derived from supported, "
        "accuracy-aware GPS episodes"
    ),
    fontsize=11.5,
)


# ============================================================
# PANEL 1 — RAW GPS → CANDIDATE PLACES
# ============================================================

ax1 = fig.add_subplot(gs[0, 0])

raw_x, raw_y = project_m(
    display_raw["latitude"],
    display_raw["longitude"],
)


# Show every real raw point, rasterized for efficient rendering.
ax1.scatter(
    raw_x,
    raw_y,
    s=1,
    alpha=0.08,
    color="#9ca3af",
    rasterized=True,
    label="Raw GPS observations",
)


palette = [
    "#2563eb",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#dc2626",
    "#65a30d",
]


place_color = {}

for idx, place_id in enumerate(
    top_place_ids
):

    color = palette[
        idx % len(palette)
    ]

    if place_id == home_place_id:
        color = "#15803d"

    place_color[place_id] = color

    group = display_assignments[
        display_assignments[
            "place_candidate_id"
        ].eq(place_id)
    ]

    gx, gy = project_m(
        group[
            "representative_latitude"
        ],
        group[
            "representative_longitude"
        ],
    )

    ax1.scatter(
        gx,
        gy,
        s=18,
        alpha=0.55,
        color=color,
        rasterized=True,
    )

    hull = convex_hull(
        zip(gx, gy)
    )

    if len(hull) >= 3:

        hx = [
            p[0]
            for p in hull
        ] + [hull[0][0]]

        hy = [
            p[1]
            for p in hull
        ] + [hull[0][1]]

        ax1.plot(
            hx,
            hy,
            linewidth=2.0,
            color=color,
        )

        ax1.fill(
            hx,
            hy,
            color=color,
            alpha=0.07,
        )


# Actual Cortex seed/support circle for home place.
if (
    "seed_latitude" in home_row.index
    and
    "seed_longitude" in home_row.index
    and
    "seed_support_radius_m" in home_row.index
):

    seed_x, seed_y = project_m(
        [float(home_row["seed_latitude"])],
        [float(home_row["seed_longitude"])],
    )

    support_radius = float(
        home_row["seed_support_radius_m"]
    )

    ax1.add_patch(
        Circle(
            (
                seed_x[0],
                seed_y[0],
            ),
            support_radius,
            fill=False,
            linestyle="--",
            linewidth=2,
            edgecolor="#15803d",
        )
    )


ax1.scatter(
    [0],
    [0],
    s=130,
    marker="*",
    color="#15803d",
    edgecolor="black",
    linewidth=0.5,
    zorder=20,
)


ax1.set_title(
    "1. Raw GPS → candidate places",
    loc="left",
    fontsize=16,
    fontweight="bold",
)

ax1.set_xlabel(
    "Meters east/west of home"
)

ax1.set_ylabel(
    "Meters north/south of home"
)

ax1.set_aspect(
    "equal",
    adjustable="datalim",
)


ax1.text(
    0.02,
    0.02,
    (
        "Outlined footprints = observed locations of assigned "
        "microsegments\n"
        "Dashed circle = actual Cortex support region for home seed"
    ),
    transform=ax1.transAxes,
    fontsize=9,
    va="bottom",
)


# ============================================================
# PANEL 2 — NIGHTTIME OCCUPANCY → HOME
# ============================================================

ax2 = fig.add_subplot(gs[0, 1])


night_display = display_assignments[
    display_assignments[
        "is_nighttime"
    ]
    .fillna(False)
    .astype(bool)
].copy()


# Draw place footprints lightly first.
for place_id in top_place_ids:

    group = display_assignments[
        display_assignments[
            "place_candidate_id"
        ].eq(place_id)
    ]

    gx, gy = project_m(
        group[
            "representative_latitude"
        ],
        group[
            "representative_longitude"
        ],
    )

    hull = convex_hull(
        zip(gx, gy)
    )

    if len(hull) >= 3:

        hx = [
            p[0]
            for p in hull
        ] + [hull[0][0]]

        hy = [
            p[1]
            for p in hull
        ] + [hull[0][1]]

        ax2.fill(
            hx,
            hy,
            color=place_color[place_id],
            alpha=0.05,
        )

        ax2.plot(
            hx,
            hy,
            color=place_color[place_id],
            alpha=0.5,
            linewidth=1.4,
        )


for place_id in top_place_ids:

    ng = night_display[
        night_display[
            "place_candidate_id"
        ].eq(place_id)
    ]

    if ng.empty:
        continue

    nx, ny = project_m(
        ng[
            "representative_latitude"
        ],
        ng[
            "representative_longitude"
        ],
    )

    ax2.scatter(
        nx,
        ny,
        s=25,
        color=place_color[place_id],
        alpha=0.80,
        rasterized=True,
    )


ax2.scatter(
    [0],
    [0],
    s=150,
    marker="*",
    color="#15803d",
    edgecolor="black",
    linewidth=0.5,
    zorder=20,
)


ax2.set_title(
    "2. Nighttime occupancy identifies home",
    loc="left",
    fontsize=16,
    fontweight="bold",
)

ax2.set_xlabel(
    "Meters east/west of home"
)

ax2.set_ylabel(
    "Meters north/south of home"
)

ax2.set_aspect(
    "equal",
    adjustable="datalim",
)


ax2.text(
    0.02,
    0.97,
    (
        r"$Home = \arg\max_p\;"
        r"T_{night}(p)$"
        "\n"
        f"Night = "
        f"{NIGHT_START_HOUR:02d}:00–"
        f"{NIGHT_END_HOUR:02d}:00 local time"
        "\n"
        f"Home captured "
        f"{home_night_fraction * 100:.1f}% "
        "of observed nighttime place occupancy"
    ),
    transform=ax2.transAxes,
    va="top",
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        alpha=0.88,
        edgecolor="#d1d5db",
    ),
)


# ============================================================
# PANEL 3 — REAL DAY TIMELINE
# ============================================================

ax3 = fig.add_subplot(gs[1, 0])


day_date = str(
    pd.Timestamp(
        REPRESENTATIVE_DAY
    ).date()
)


supported_day = (
    daily_supported_intervals.get(
        day_date,
        pd.DataFrame(),
    )
)


# Sometimes dictionary keys may be date-like but not exact strings.
if supported_day.empty:
    for key, val in (
        daily_supported_intervals.items()
    ):
        if str(key) == day_date:
            supported_day = val
            break


day_assign = assignments[
    assignments["date"]
    .astype(str)
    .eq(day_date)
].copy()


date_utc = pd.Timestamp(
    day_date,
    tz="UTC",
)

day_start_ms = int(
    date_utc.timestamp()
    * 1000
)


def hour_from_ms(ms):
    return (
        (
            float(ms)
            - day_start_ms
        )
        / 3_600_000
    )


# Entire day starts as unobserved.
ax3.axvspan(
    0,
    24,
    color="#e5e7eb",
    alpha=1.0,
)


# Overlay all supported GPS intervals.
for row in supported_day.itertuples():

    x0 = hour_from_ms(
        row.start_ms
    )

    x1 = hour_from_ms(
        row.end_ms
    )

    ax3.axvspan(
        x0,
        x1,
        ymin=0.20,
        ymax=0.80,
        color="#60a5fa",
        alpha=0.95,
        linewidth=0,
    )


# Overlay supported episodes belonging to home.
home_day = day_assign[
    day_assign[
        "place_candidate_id"
    ].eq(home_place_id)
]


for row in home_day.itertuples():

    x0 = hour_from_ms(
        row.episode_start_ms
    )

    x1 = hour_from_ms(
        row.episode_end_ms
    )

    ax3.axvspan(
        x0,
        x1,
        ymin=0.20,
        ymax=0.80,
        color="#16a34a",
        alpha=1.0,
        linewidth=0,
    )


ax3.set_xlim(
    0,
    24,
)

ax3.set_ylim(
    0,
    1,
)

ax3.set_yticks([])

ax3.set_xticks(
    [0, 6, 12, 18, 24]
)

ax3.set_xticklabels(
    [
        "12 AM",
        "6 AM",
        "12 PM",
        "6 PM",
        "12 AM",
    ]
)


ax3.set_title(
    (
        "3. Only supported time is classified"
        f" — {day_date}"
    ),
    loc="left",
    fontsize=16,
    fontweight="bold",
)


legend_handles = [
    Patch(
        facecolor="#16a34a",
        label="At home",
    ),
    Patch(
        facecolor="#60a5fa",
        label="Observed, not home",
    ),
    Patch(
        facecolor="#e5e7eb",
        label="Unobserved / unknown",
    ),
]


ax3.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=9,
)


rep_supported_h = float(
    representative_day_row[
        "supported_h"
    ]
)

rep_home_h = float(
    representative_day_row[
        "home_h"
    ]
)

rep_unknown_h = (
    24.0
    - rep_supported_h
)


ax3.text(
    0.02,
    0.04,
    (
        f"Home = {rep_home_h:.1f} h   |   "
        f"Observed = {rep_supported_h:.1f} h   |   "
        f"Unknown = {rep_unknown_h:.1f} h"
    ),
    transform=ax3.transAxes,
    fontsize=10,
)


# ============================================================
# PANEL 4 — SIMPLE MATH + REAL VALUES
# ============================================================

ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")


ax4.set_title(
    "4. Sum the supported intervals assigned to home",
    loc="left",
    fontsize=16,
    fontweight="bold",
)


ax4.text(
    0.03,
    0.77,
    (
        r"$T_{home} = "
        r"\sum_{i:\;place_i=home} "
        r"\Delta t_i$"
    ),
    fontsize=23,
)


ax4.text(
    0.03,
    0.58,
    (
        "Representative day\n"
        f"{day_date}\n\n"
        f"Home time = {rep_home_h:.1f} h\n"
        f"Supported GPS time = {rep_supported_h:.1f} h\n"
        f"Time at home = "
        f"{100 * rep_home_h / rep_supported_h:.1f}% "
        "of supported time"
    ),
    fontsize=13,
    linespacing=1.5,
)


window_home_h = (
    window_home
    / 3_600_000
)

window_supported_h = (
    window_supported
    / 3_600_000
)


ax4.text(
    0.57,
    0.58,
    (
        "14-day example window\n"
        "Apr 23–May 6\n\n"
        f"Home time = {window_home_h:.1f} h\n"
        f"Supported GPS time = "
        f"{window_supported_h:.1f} h\n"
        f"Time at home = "
        f"{100 * window_home_fraction:.1f}%"
    ),
    fontsize=13,
    linespacing=1.5,
)


ax4.text(
    0.03,
    0.08,
    (
        "Missing GPS is never converted into assumed home time. "
        "It remains unobserved."
    ),
    fontsize=12,
    fontweight="bold",
)


# ============================================================
# SAVE
# ============================================================

png = OUTDIR / "RSH634_home_derivation.png"
pdf = OUTDIR / "RSH634_home_derivation.pdf"

fig.savefig(
    png,
    dpi=240,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

for p in [
    OUTDIR / "RSH634_place_assignments.csv",
    OUTDIR / "RSH634_candidate_places.csv",
    OUTDIR / "RSH634_nighttime_occupancy_by_place.csv",
    OUTDIR / "RSH634_daily_home_time_rebuilt.csv",
    png,
    pdf,
]:
    print(p)

