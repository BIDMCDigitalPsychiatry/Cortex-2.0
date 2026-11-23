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

# ============================================================
# 1. REAL GPS OBSERVATIONS + CURRENT AUTOMATIC DONOR SELECTION
# ============================================================

with redirect_stdout(StringIO()):
    ctx = runpy.run_path(
        "workflows/visualization_diagnostics/check_dec7_reverse_route.py"
    )
    wf = runpy.run_path(
        "workflows/run_gps_imputed_clinical_features.py"
    )

clean = ctx["clean"].copy()
left, right = ctx["left"], ctx["right"]

clean["timestamp"] = pd.to_numeric(
    clean["timestamp"], errors="coerce"
)
clean = (
    clean.dropna(subset=[
        "timestamp", "latitude", "longitude"
    ])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

cfg = ImputationConfig()

donors, _, _ = wf["build_participant_donors"](clean)

matches = _matching_route_sections(
    left, right, donors, cfg
)
assert matches, "No geographic donors found"

endpoint_error, donor_index, donor_route = matches[0]
donor_source = donors[donor_index].copy()

filled = _bridge_matched_route(
    left, right, donor_route, cfg
)
assert filled is not None and not filled.empty

reconstructed = pd.concat(
    [pd.DataFrame([left]), filled, pd.DataFrame([right])],
    ignore_index=True
).sort_values("timestamp")


# CORTEX2_ALTERNATIVE_DONORS_PANEL_B_V2
alternative_donors = []
seen_sources = {donor_index}

for error, index, candidate_route in matches:
    if index in seen_sources:
        continue

    seen_sources.add(index)
    alternative_donors.append(
        (index, error, candidate_route)
    )

# ============================================================
# 2. ORIGINAL DECEMBER 12 INCOMPLETE TRIP
#    Preserve the previously approved Panel A time windows.
# ============================================================

DEC12 = pd.Timestamp("2023-12-12", tz="UTC")
start_ms = int(DEC12.value // 1_000_000)
end_ms = int(
    (DEC12 + pd.Timedelta(days=1)).value // 1_000_000
)

dec12 = clean.loc[
    (clean.timestamp >= start_ms)
    & (clean.timestamp < end_ms)
].sort_values("timestamp").reset_index(drop=True)

t0 = int(left.timestamp)
t1 = int(right.timestamp)

assert not dec12.empty

# Extract the observed sequences adjacent to the real gap.
t = dec12.timestamp.to_numpy(dtype=np.int64)

i0 = int(np.argmin(np.abs(t - t0)))
i1 = int(np.argmin(np.abs(t - t1)))

assert abs(int(t[i0]) - t0) <= 1000
assert abs(int(t[i1]) - t1) <= 1000
assert i0 < i1

# Supported sequence before the gap.
a0 = i0
while a0 > 0 and t[a0] - t[a0 - 1] <= 60000:
    a0 -= 1

# Supported sequence after the gap.
a1 = i1
while a1 + 1 < len(t) and t[a1 + 1] - t[a1] <= 60000:
    a1 += 1

before = dec12.iloc[a0:i0 + 1].copy()
after = dec12.iloc[i1:a1 + 1].copy()

# Same plotting windows that produced the approved Panel A.
before = before.loc[
    before.timestamp >= t0 - 10 * 60000
].copy()

after = after.loc[
    after.timestamp <= t1 + 20 * 60000
].copy()

assert len(before) >= 2 and len(after) >= 2, (
    "Insufficient original observations around gap"
)

# ============================================================
# 3. OBSERVED DONOR CONTEXT
#    Show the entire matched passage and nearby real movement.
#    The selected donor subsection remains unmodified.
# ============================================================

donor_source = donor_source.sort_values(
    "timestamp"
).reset_index(drop=True)

route_t = donor_route.timestamp.to_numpy(dtype=np.int64)
route_start = int(route_t.min())
route_end = int(route_t.max())

context_before_ms = 10 * 60000
context_after_ms = 10 * 60000

donor_context = donor_source.loc[
    (donor_source.timestamp >= route_start - context_before_ms)
    & (donor_source.timestamp <= route_end + context_after_ms)
].copy()

assert not donor_context.empty

donor_start_label = pd.to_datetime(
    route_start, unit="ms", utc=True
).strftime("%b %d")

donor_end_label = pd.to_datetime(
    route_end, unit="ms", utc=True
).strftime("%b %d")

donor_label = (
    donor_start_label if donor_start_label == donor_end_label
    else f"{donor_start_label}–{donor_end_label}"
)

# ============================================================
# 4. GEOGRAPHIC COORDINATES
# ============================================================

R = 6371000.0
lat0 = float(left.latitude)
lon0 = float(left.longitude)
coslat = np.cos(np.deg2rad(lat0))

BLUE = "#287CB2"
GREEN = "#14876C"
GRAY = "#B7C6CF"
ORANGE = "#DF7437"


def xy(frame):
    lat = frame.latitude.to_numpy(dtype=float)
    lon = frame.longitude.to_numpy(dtype=float)

    x = R * np.deg2rad(lon - lon0) * coslat / 1000
    y = R * np.deg2rad(lat - lat0) / 1000
    return x, y


def plot_observed(ax, frame, color, lw=2.4, alpha=1):
    """Never connect observed GPS across unsupported intervals."""
    frame = frame.sort_values("timestamp")
    if len(frame) < 2:
        return

    tt = frame.timestamp.to_numpy(dtype=np.int64)
    x, y = xy(frame)

    cuts = np.r_[
        0,
        np.flatnonzero(np.diff(tt) > 60000) + 1,
        len(frame)
    ]

    for i, j in zip(cuts[:-1], cuts[1:]):
        if j - i >= 2:
            ax.plot(
                x[i:j], y[i:j],
                color=color, lw=lw, alpha=alpha,
                solid_capstyle="round",
                zorder=3
            )


def plot_route(ax, frame, color=GREEN, lw=3.3):
    x, y = xy(frame)
    ax.plot(
        x, y, color=color, lw=lw,
        solid_capstyle="round", zorder=5
    )


def extent(*frames):
    xx, yy = [], []

    for frame in frames:
        x, y = xy(frame)
        xx.extend(x)
        yy.extend(y)

    xx = np.asarray(xx)
    yy = np.asarray(yy)

    margin = max(
        0.12,
        0.07 * max(np.ptp(xx), np.ptp(yy))
    )

    return (
        (float(xx.min() - margin), float(xx.max() + margin)),
        (float(yy.min() - margin), float(yy.max() + margin))
    )


gap_ends = pd.DataFrame([left, right])
gx, gy = xy(gap_ends)

# ============================================================
# 5. EXACTLY THREE PANELS
# ============================================================

fig, (a, b, c) = plt.subplots(
    1, 3, figsize=(18, 6.2)
)

# PANEL A: ORIGINAL INCOMPLETE TRIP — NO IMPUTATION
plot_observed(a, before, BLUE, lw=2.6)
plot_observed(a, after, BLUE, lw=2.6)

a.scatter(
    gx, gy, s=58, color=ORANGE,
    edgecolors="white", linewidth=1, zorder=7
)

a.set_title(
    "A  Dec 12 | Original incomplete trip",
    loc="left", fontsize=12, weight="bold"
)

# PANEL B: SELECTED DONOR AND OTHER GEOGRAPHIC CANDIDATES
plot_observed(b, donor_context, GRAY, lw=1.5, alpha=0.45)

# Plot all distinct alternative donor sources in different colors.
# Each line represents an actual observed geographic subsection.
cmap = plt.get_cmap("tab20")

for i, (index, error, candidate_route) in enumerate(alternative_donors):
    x_alt, y_alt = xy(candidate_route)

    b.plot(
        x_alt,
        y_alt,
        color=cmap(i % 20),
        linewidth=1.1,
        alpha=0.58,
        zorder=2,
    )

# Selected donor stays visually dominant.
plot_route(b, donor_route, GREEN, lw=1.3)

bx, by = xy(donor_route)
b.scatter(
    [bx[0], bx[-1]],
    [by[0], by[-1]],
    s=48, color=GREEN,
    edgecolors="white", linewidth=1, zorder=7
)

b.set_title(
    "B  Candidate observed donor routes",
    loc="left", fontsize=12, weight="bold"
)

# PANEL C: SAME ORIGINAL TRIP + COMPACT RECONSTRUCTION
plot_observed(c, before, BLUE, lw=2.6)
plot_observed(c, after, BLUE, lw=2.6)

plot_route(c, reconstructed, GREEN, lw=3.5)

c.scatter(
    gx, gy, s=58, color=ORANGE,
    edgecolors="white", linewidth=1, zorder=7
)

c.set_title(
    "C  Dec 12 | Reconstructed missing section",
    loc="left", fontsize=12, weight="bold"
)

# Panels A/C have IDENTICAL geographic framing.
target_xlim, target_ylim = extent(
    before, after, reconstructed
)

# Panel B displays the full selected observed passage
# with nearby recorded context, not the entire 27-hour source.
donor_xlim, donor_ylim = extent(
    donor_context,
    donor_route,
    *[candidate for _, _, candidate in alternative_donors]
)

for ax in (a, c):
    ax.set_xlim(*target_xlim)
    ax.set_ylim(*target_ylim)

b.set_xlim(*donor_xlim)
b.set_ylim(*donor_ylim)

for ax in (a, b, c):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East–west distance (km)")
    ax.set_ylabel("North–south distance (km)")
    ax.grid(color="#E7EDF0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

fig.suptitle(
    "GPS gap reconstruction using observed movement",
    fontsize=17, weight="bold", y=.975
)

fig.legend(
    handles=[
        Line2D([0], [0], color=BLUE, lw=2.6,
               label="Observed Dec 12 trip"),
        Line2D([0], [0], color=GRAY, lw=2,
               label="Other observed donor movement"),
        Line2D([0], [0], color="#967CB3", lw=1.8,
               label="Other matched donor routes"),
        Line2D([0], [0], color=GREEN, lw=3.4,
               label="Selected donor / reconstructed route"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=ORANGE,
               markeredgecolor="white", markersize=8,
               label="Actual gap endpoints")
    ],
    loc="upper center",
    bbox_to_anchor=(.5, .88),
    ncol=2, frameon=False, fontsize=9.5
)

fig.subplots_adjust(
    left=.055, right=.985,
    top=.77, bottom=.13, wspace=.25
)

# ============================================================
# 6. SAVE + CONCISE DIAGNOSTICS
# ============================================================

out = Path(
    "outputs/gps_mobility_figures/"
    "AMO327_real_data_three_panel.png"
)
out.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(out, dpi=220, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
plt.close(fig)


def length(frame):
    coords = frame[
        ["latitude", "longitude"]
    ].to_numpy(dtype=float)
    return float(sum(
        _haversine(*p, *q)
        for p, q in zip(coords[:-1], coords[1:])
    ))


print("\nAMO327 THREE-PANEL FIGURE")
print("=" * 46)
print("Original before-gap fixes:", len(before))
print("Original after-gap fixes:", len(after))
print("Gap duration:", round((t1 - t0) / 60000, 2), "min")
print("Automatic donor index:", donor_index)
print("Donor source:", donor_label)
print("Donor subsection fixes:", len(donor_route))
print("Endpoint mismatch:", round(endpoint_error, 1), "m")
print("Observed donor route length:", round(length(donor_route), 1), "m")
print("Compact imputed fixes:", len(filled))
print("Reconstructed length:", round(length(reconstructed), 1), "m")
print("Alternative donor sources displayed:")
for index, error, candidate in alternative_donors:
    print(
        f"  Donor {index}: "
        f"endpoint mismatch {error:.1f} m, "
        f"{len(candidate)} observed fixes"
    )
print("Saved:", out)
