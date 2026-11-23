from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import runpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig,
    _matching_route_sections,
    _bridge_matched_route,
    _haversine,
)

# ------------------------------------------------------------
# 1. VERIFIED REAL GPS DATA
# ------------------------------------------------------------

with redirect_stdout(StringIO()):
    ctx = runpy.run_path(
        "workflows/visualization_diagnostics/check_dec7_reverse_route.py"
    )

clean = ctx["clean"].copy()
left, right = ctx["left"], ctx["right"]

clean["timestamp"] = pd.to_numeric(
    clean["timestamp"], errors="coerce"
)

clean = (
    clean.dropna(
        subset=["timestamp", "latitude", "longitude"]
    )
    .sort_values("timestamp")
    .reset_index(drop=True)
)

cfg = ImputationConfig()

t0 = int(left.timestamp)
t1 = int(right.timestamp)

DEC8 = pd.Timestamp("2023-12-08", tz="UTC")
DEC12 = pd.Timestamp("2023-12-12", tz="UTC")

BLUE = "#287CB2"
GREEN = "#16876B"
GRAY = "#B8C8D2"
ORANGE = "#E47730"

OUT = Path(
    "outputs/gps_mobility_figures/"
    "AMO327_real_data_three_panel.png"
)


def get_day(day):
    start = day.value // 1_000_000
    end = (day + pd.Timedelta(days=1)).value // 1_000_000

    return clean.loc[
        (clean.timestamp >= start)
        & (clean.timestamp < end)
    ].copy().reset_index(drop=True)


dec8 = get_day(DEC8)
dec12 = get_day(DEC12)

assert not dec8.empty, "No Dec 8 observations"
assert not dec12.empty, "No Dec 12 observations"

# ------------------------------------------------------------
# 2. ORIGINAL DEC 12 INCOMPLETE TRIP
#
# Extract the actual continuous observed segments adjoining
# the verified gap endpoints.
#
# Do not plot the entire day: other Dec 12 trips can overlap
# the missing geographic corridor and make the gap look filled.
# ------------------------------------------------------------

def adjacent_segment(day, boundary_ms, side):
    day = day.sort_values("timestamp").reset_index(drop=True)
    t = day.timestamp.to_numpy(dtype=np.int64)

    # Find the actual observation nearest the known boundary.
    i = int(np.argmin(np.abs(t - boundary_ms)))

    if abs(int(t[i]) - boundary_ms) > 1000:
        raise RuntimeError(
            f"Could not find observed {side} gap boundary"
        )

    if side == "before":
        start = i

        # Follow only the supported original trajectory
        # backwards from the gap-start observation.
        while (
            start > 0
            and t[start] - t[start - 1] <= 60000
        ):
            start -= 1

        return day.iloc[start:i + 1].copy()

    end = i

    # Follow only the supported original trajectory
    # forwards from the gap-end observation.
    while (
        end + 1 < len(day)
        and t[end + 1] - t[end] <= 60000
    ):
        end += 1

    return day.iloc[i:end + 1].copy()


before = adjacent_segment(dec12, t0, "before")
after = adjacent_segment(dec12, t1, "after")

if len(before) < 2 or len(after) < 2:
    raise RuntimeError(
        "Insufficient observed GPS around the Dec 12 gap"
    )


# Restrict the plotted observations to the trip surrounding
# the known Dec 12 gap, excluding earlier movement through
# the same geographic corridor.
#
# These are visualization windows, not donor eligibility rules.
before = before.loc[
    (before.timestamp >= t0 - 10 * 60_000)
    & (before.timestamp <= t0)
].copy()

after = after.loc[
    (after.timestamp >= t1)
    & (after.timestamp <= t1 + 20 * 60_000)
].copy()

if len(before) < 2 or len(after) < 2:
    raise RuntimeError("Insufficient observed GPS around the gap")

# The original incomplete target trip consists ONLY of these
# two recorded portions. No trajectory is drawn between them.
target = pd.concat(
    [before, after], ignore_index=True
)

# ------------------------------------------------------------
# 3. REAL DEC 8 DONOR SUBSECTION
#
# Search genuine Dec 8 GPS observations.
# Do not manufacture a donor trajectory.
# ------------------------------------------------------------

timestamps = dec8.timestamp.to_numpy(dtype=np.int64)

cuts = np.r_[
    0,
    np.flatnonzero(
        np.diff(timestamps) > cfg.max_supported_gap_ms
    ) + 1,
    len(dec8)
]

donor_segments = []

for start, end in zip(cuts[:-1], cuts[1:]):
    segment = (
        dec8.iloc[start:end]
        .copy()
        .reset_index(drop=True)
    )

    if len(segment) < cfg.min_donor_fixes:
        continue

    span = (
        int(segment.timestamp.iloc[-1])
        - int(segment.timestamp.iloc[0])
    )

    if span >= cfg.min_template_span_ms:
        donor_segments.append(segment)

matches = _matching_route_sections(
    left, right, donor_segments, cfg
)

if not matches:
    raise RuntimeError(
        "No Dec 8 geographic donor match. "
        "No reconstructed trajectory has been fabricated."
    )

score, donor_index, donor_route = matches[0]

# Complete continuously observed source movement containing
# the highlighted matching subsection.
full_donor = donor_segments[donor_index]

# ------------------------------------------------------------
# 4. ACTUAL COMPACT IMPUTATION FUNCTION
# ------------------------------------------------------------

filled = _bridge_matched_route(
    left, right, donor_route, cfg
)

if filled is None or filled.empty:
    raise RuntimeError("Compact reconstruction failed")

reconstructed = pd.concat(
    [
        pd.DataFrame([left]),
        filled,
        pd.DataFrame([right]),
    ],
    ignore_index=True
).sort_values("timestamp")

# ------------------------------------------------------------
# 5. GEOGRAPHIC COORDINATES
# ------------------------------------------------------------

R = 6371000.0
lat0 = float(left.latitude)
lon0 = float(left.longitude)
coslat = np.cos(np.deg2rad(lat0))


def xy(frame):
    latitude = frame.latitude.to_numpy(dtype=float)
    longitude = frame.longitude.to_numpy(dtype=float)

    x = (
        R * np.deg2rad(longitude - lon0) * coslat / 1000
    )
    y = (
        R * np.deg2rad(latitude - lat0) / 1000
    )

    return x, y


def plot_line(ax, frame, color, width=2.5, alpha=1):
    if len(frame) < 2:
        return

    x, y = xy(frame)

    ax.plot(
        x, y,
        color=color,
        lw=width,
        alpha=alpha,
        solid_capstyle="round",
        zorder=3
    )


def limits(*frames):
    xx = []
    yy = []

    for frame in frames:
        x, y = xy(frame)
        xx.extend(x)
        yy.extend(y)

    xx = np.asarray(xx)
    yy = np.asarray(yy)

    pad = max(
        .12,
        .07 * max(np.ptp(xx), np.ptp(yy))
    )

    return (
        (xx.min() - pad, xx.max() + pad),
        (yy.min() - pad, yy.max() + pad)
    )


gap_endpoints = pd.DataFrame([left, right])
gx, gy = xy(gap_endpoints)

# ------------------------------------------------------------
# 6. THE THREE PANELS
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(17.7, 6.0)
)

a, b, c = axes

# PANEL A
# Original Dec 12 observations only. Leave the gap empty.
plot_line(a, before, BLUE, width=2.6)
plot_line(a, after, BLUE, width=2.6)

a.scatter(
    gx, gy, s=55, color=ORANGE,
    edgecolor="white", linewidth=.9, zorder=6
)

a.set_title(
    "A  Dec 12 — Original incomplete trip",
    loc="left", fontsize=12, weight="bold"
)

# PANEL B
# Actual observed Dec 8 source movement in gray.
# The matching donor subsection is highlighted in green.
plot_line(b, full_donor, GRAY, width=2.0)
plot_line(b, donor_route, GREEN, width=3.2)

bx, by = xy(donor_route)

b.scatter(
    [bx[0], bx[-1]],
    [by[0], by[-1]],
    color=GREEN,
    s=45,
    edgecolor="white",
    linewidth=.8,
    zorder=6
)

b.set_title(
    "B  Dec 8 — Observed donor route",
    loc="left",
    fontsize=12,
    weight="bold"
)

# PANEL C
# Same original observed pieces as Panel A.
# ONLY here is the missing interval reconstructed.
plot_line(c, before, BLUE, width=2.6)
plot_line(c, after, BLUE, width=2.6)
plot_line(c, reconstructed, GREEN, width=3.3)

c.scatter(
    gx, gy,
    color=ORANGE,
    s=55,
    edgecolor="white",
    linewidth=.9,
    zorder=6
)

c.set_title(
    "C  Dec 12 — Donor-based reconstruction",
    loc="left",
    fontsize=12,
    weight="bold"
)

# A and C: identical scale, based on the ORIGINAL trip.
target_xlim, target_ylim = limits(
    before, after, reconstructed
)

for ax in (a, c):
    ax.set_xlim(*target_xlim)
    ax.set_ylim(*target_ylim)

# B: full geographic extent of the observed donor movement.
donor_xlim, donor_ylim = limits(full_donor)

b.set_xlim(*donor_xlim)
b.set_ylim(*donor_ylim)

for ax in axes:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East–west distance (km)")
    ax.set_ylabel("North–south distance (km)")
    ax.grid(color="#E7EDF0", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

fig.suptitle(
    "GPS gap reconstruction from observed movement",
    fontsize=18,
    weight="bold",
    y=.975
)

fig.legend(
    handles=[
        Line2D([0], [0], color=BLUE, lw=2.5,
               label="Observed Dec 12 trip"),
        Line2D([0], [0], color=GRAY, lw=2.5,
               label="Observed Dec 8 movement"),
        Line2D([0], [0], color=GREEN, lw=3.2,
               label="Donor / reconstructed section"),
        Line2D([0], [0], marker="o",
               linestyle="None", markersize=7,
               markerfacecolor=ORANGE,
               markeredgecolor="white",
               label="Gap endpoints"),
    ],
    loc="upper center",
    bbox_to_anchor=(.5, .87),
    ncol=2,
    frameon=False,
    fontsize=9.5
)

fig.subplots_adjust(
    left=.055,
    right=.985,
    top=.76,
    bottom=.12,
    wspace=.23
)

OUT.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(OUT, dpi=220, facecolor="white")
fig.savefig(
    OUT.with_suffix(".pdf"),
    facecolor="white"
)

plt.close(fig)

def distance(frame):
    coords = frame[
        ["latitude", "longitude"]
    ].to_numpy(dtype=float)

    return float(sum(
        _haversine(*p, *q)
        for p, q in zip(coords[:-1], coords[1:])
    ))

print("\nREAL GPS THREE-PANEL FIGURE")
print("=" * 42)
print("Dec 12 before-gap fixes:", len(before))
print("Dec 12 after-gap fixes:", len(after))
print("Gap duration (min):", round((t1-t0)/60000, 2))
print("Dec 8 selected donor fixes:", len(donor_route))
print("Endpoint mismatch (m):", round(score, 1))
print("Compact reconstructed fixes:", len(filled))
print("Reconstructed distance (m):", round(distance(reconstructed), 1))
print("Saved:", OUT)
