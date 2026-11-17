#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch


# ============================================================
# INPUTS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/Cortex-2.0/home_derivation_RSH634"
)

ASSIGNMENTS = ROOT / "RSH634_place_assignments.csv"
PLACES = ROOT / "RSH634_candidate_places.csv"
NIGHT = ROOT / "RSH634_nighttime_occupancy_by_place.csv"
DAILY = ROOT / "RSH634_daily_home_time_rebuilt.csv"

OUT_PNG = ROOT / "RSH634_home_visual_clean.png"
OUT_PDF = ROOT / "RSH634_home_visual_clean.pdf"

DISPLAY_START = pd.Timestamp("2025-04-23")
DISPLAY_END = pd.Timestamp("2025-05-07")


# ============================================================
# LOAD REAL DERIVED DATA
# ============================================================

assign = pd.read_csv(ASSIGNMENTS)
places = pd.read_csv(PLACES)
night = pd.read_csv(NIGHT)
daily = pd.read_csv(DAILY)

assign["date"] = pd.to_datetime(assign["date"])
daily["date"] = pd.to_datetime(daily["date"])

home_place = int(
    night.sort_values(
        "duration_ms",
        ascending=False,
    ).iloc[0]["place_candidate_id"]
)

display = assign[
    assign["date"].between(
        DISPLAY_START,
        DISPLAY_END - pd.Timedelta(days=1),
    )
].copy()


# ============================================================
# SELECT THREE MOST IMPORTANT PLACES
# ============================================================

place_duration = (
    display.groupby("place_candidate_id")["duration_ms"]
    .sum()
    .sort_values(ascending=False)
)

top_places = list(
    place_duration.head(3).index.astype(int)
)

if home_place not in top_places:
    top_places[-1] = home_place

top_places = list(dict.fromkeys(top_places))


# ============================================================
# HOME COORDINATE
# ============================================================

home_row = places[
    places["place_candidate_id"].eq(home_place)
].iloc[0]

home_lat = float(home_row["representative_latitude"])
home_lon = float(home_row["representative_longitude"])

EARTH_R = 6_371_000.0


def xy(lat, lon, lat0=home_lat, lon0=home_lon):

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    x = (
        EARTH_R
        * np.deg2rad(lon - lon0)
        * math.cos(math.radians(lat0))
    )

    y = (
        EARTH_R
        * np.deg2rad(lat - lat0)
    )

    return x, y


# ============================================================
# SIMPLE CONVEX HULL — VISUAL ONLY
# ============================================================

def hull(points):

    pts = sorted(
        set(
            (float(x), float(y))
            for x, y in points
            if np.isfinite(x) and np.isfinite(y)
        )
    )

    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            -
            (a[1] - o[1]) * (b[0] - o[0])
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

    return lower[:-1] + upper[:-1]


# ============================================================
# LOCAL PLACE-CENTERED COORDINATES
#
# Instead of showing the participant's entire travel trajectory,
# display each real candidate place separately around its own
# center. This makes the spatial clustering visually understandable.
# ============================================================

def local_place_xy(group):

    lat0 = float(
        np.average(
            group["representative_latitude"],
            weights=group["duration_ms"],
        )
    )

    lon0 = float(
        np.average(
            group["representative_longitude"],
            weights=group["duration_ms"],
        )
    )

    lat = group["representative_latitude"].to_numpy(float)
    lon = group["representative_longitude"].to_numpy(float)

    x = (
        EARTH_R
        * np.deg2rad(lon - lon0)
        * math.cos(math.radians(lat0))
    )

    y = (
        EARTH_R
        * np.deg2rad(lat - lat0)
    )

    return x, y


# ============================================================
# CHOOSE REAL REPRESENTATIVE DAY
# ============================================================

d = daily[
    daily["date"].between(
        DISPLAY_START,
        DISPLAY_END - pd.Timedelta(days=1),
    )
].copy()

d["supported_h"] = (
    d["supported_duration_ms"] / 3_600_000
)

d["home_h"] = (
    d["home_duration_ms"] / 3_600_000
)

window_fraction = (
    d["home_duration_ms"].sum()
    / d["supported_duration_ms"].sum()
)

eligible = d[
    d["supported_h"] >= 20
].copy()

eligible["difference"] = (
    eligible["time_at_home_fraction"]
    - window_fraction
).abs()

rep_day = (
    eligible.sort_values(
        ["difference", "supported_h"],
        ascending=[True, False],
    )
    .iloc[0]
)

day = rep_day["date"]

day_assign = assign[
    assign["date"].eq(day)
].copy()


# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
})


fig = plt.figure(
    figsize=(16, 5.5)
)

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1.15, 1.15, 1.35],
    left=0.045,
    right=0.975,
    top=0.86,
    bottom=0.14,
    wspace=0.28,
)


# ============================================================
# PANEL 1 — THREE REAL CANDIDATE PLACES
# ============================================================

ax1 = fig.add_subplot(gs[0, 0])

colors = [
    "#2563eb",
    "#d97706",
    "#16a34a",
]

for idx, place_id in enumerate(top_places):

    g = display[
        display["place_candidate_id"].eq(place_id)
    ].copy()

    if g.empty:
        continue

    gx, gy = local_place_xy(g)

    # vertically separate the actual places for visual comparison
    offset_y = (
        (len(top_places) - 1) / 2 - idx
    ) * 500

    gy_display = gy + offset_y

    color = (
        "#16a34a"
        if place_id == home_place
        else colors[idx]
    )

    ax1.scatter(
        gx,
        gy_display,
        s=9,
        alpha=0.40,
        color=color,
        rasterized=True,
    )

    hh = hull(
        zip(
            gx,
            gy_display,
        )
    )

    if len(hh) >= 3:

        hx = [p[0] for p in hh] + [hh[0][0]]
        hy = [p[1] for p in hh] + [hh[0][1]]

        ax1.fill(
            hx,
            hy,
            color=color,
            alpha=0.10,
        )

        ax1.plot(
            hx,
            hy,
            color=color,
            linewidth=2,
        )

    ax1.text(
        0,
        offset_y,
        (
            "HOME"
            if place_id == home_place
            else f"Place {place_id}"
        ),
        fontsize=12,
        fontweight=(
            "bold"
            if place_id == home_place
            else "normal"
        ),
        ha="center",
        va="center",
    )


ax1.set_title(
    "Candidate places",
    fontsize=17,
    fontweight="bold",
)

ax1.set_xticks([])
ax1.set_yticks([])

for spine in ax1.spines.values():
    spine.set_visible(False)


# ============================================================
# PANEL 2 — NIGHTTIME OCCUPANCY
# ============================================================

ax2 = fig.add_subplot(gs[0, 1])


night_plot = night[
    night["place_candidate_id"]
    .isin(top_places)
].copy()

night_plot["hours"] = (
    night_plot["duration_ms"]
    / 3_600_000
)

night_plot = night_plot.sort_values(
    "hours",
    ascending=True,
)


labels = []

bar_colors = []

for pid in night_plot["place_candidate_id"]:

    pid = int(pid)

    if pid == home_place:

        labels.append("HOME")
        bar_colors.append("#16a34a")

    else:

        labels.append(
            f"Place {pid}"
        )

        bar_colors.append(
            "#94a3b8"
        )


y = np.arange(
    len(night_plot)
)


ax2.barh(
    y,
    night_plot["hours"],
    color=bar_colors,
    height=0.55,
)


ax2.set_yticks(y)
ax2.set_yticklabels(labels)

ax2.set_xlabel(
    "Observed nighttime hours"
)

ax2.set_title(
    "12 AM–6 AM",
    fontsize=17,
    fontweight="bold",
)


for i, value in enumerate(
    night_plot["hours"]
):

    ax2.text(
        value,
        i,
        f"  {value:.1f} h",
        va="center",
        fontsize=11,
    )


ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)


# ============================================================
# PANEL 3 — REAL DAY HOME / AWAY / UNKNOWN
# ============================================================

ax3 = fig.add_subplot(gs[0, 2])


day_start = pd.Timestamp(
    day.date(),
    tz="UTC",
)

day_start_ms = int(
    day_start.timestamp() * 1000
)


# background = unknown
ax3.axvspan(
    0,
    24,
    ymin=0.34,
    ymax=0.66,
    color="#e5e7eb",
)


# classify candidate-place episodes
for row in day_assign.itertuples():

    start_h = (
        float(row.episode_start_ms)
        - day_start_ms
    ) / 3_600_000

    end_h = (
        float(row.episode_end_ms)
        - day_start_ms
    ) / 3_600_000

    is_home = (
        int(row.place_candidate_id)
        == home_place
    )

    color = (
        "#16a34a"
        if is_home
        else "#60a5fa"
    )

    ax3.axvspan(
        start_h,
        end_h,
        ymin=0.34,
        ymax=0.66,
        color=color,
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

ax3.set_yticks([])


ax3.set_title(
    day.strftime("%b %d"),
    fontsize=17,
    fontweight="bold",
)


ax3.legend(
    handles=[
        Patch(
            facecolor="#16a34a",
            label="Home",
        ),
        Patch(
            facecolor="#60a5fa",
            label="Away",
        ),
        Patch(
            facecolor="#e5e7eb",
            label="Unknown",
        ),
    ],
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.87),
)


ax3.text(
    0.5,
    0.18,
    (
        f"{rep_day['home_h']:.1f} h at home"
    ),
    transform=ax3.transAxes,
    ha="center",
    fontsize=18,
    fontweight="bold",
)


for spine in [
    "top",
    "right",
    "left",
]:
    ax3.spines[spine].set_visible(False)


# ============================================================
# ARROWS BETWEEN PANELS
# ============================================================

fig.text(
    0.355,
    0.50,
    "→",
    fontsize=30,
    ha="center",
    va="center",
)

fig.text(
    0.665,
    0.50,
    "→",
    fontsize=30,
    ha="center",
    va="center",
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=250,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


print("Saved:")
print(OUT_PNG)
print(OUT_PDF)

print()
print("Home place:", home_place)
print(
    "Representative day:",
    day.strftime("%Y-%m-%d"),
)
print(
    "Home hours:",
    round(
        float(rep_day["home_h"]),
        2,
    ),
)
print(
    "Supported hours:",
    round(
        float(rep_day["supported_h"]),
        2,
    ),
)
