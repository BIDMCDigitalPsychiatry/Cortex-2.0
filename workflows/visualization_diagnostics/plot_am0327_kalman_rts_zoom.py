from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from cortex2_engine.features.gps_pipeline_qc import GPSQCConfig, clean_gps_fixes
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_trajectory_plausibility import add_gps_trajectory_plausibility
from cortex2_engine.features.gps_trip_features import _kalman_rts_xy

START = pd.Timestamp("2023-12-09T22:20:00Z").value // 10**6
END = pd.Timestamp("2023-12-09T22:40:00Z").value // 10**6
BLUE, PURPLE, RED = "#267BB5", "#8143B5", "#D95749"

path = (
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/gps_clinical_weekly_raw_points.parquet"
)

raw = pq.read_table(
    path, filters=[("participant_id", "=", "AMO327")]
).to_pandas()

raw = raw.loc[
    raw.timestamp.between(START, END),
    ["timestamp", "latitude", "longitude", "accuracy"]
].sort_values("timestamp").copy()

qc = GPSQCConfig(max_gap_ms=60000)
clean = clean_gps_fixes(raw, qc)
edges = build_gps_intervals(
    clean, window_start_ms=int(START),
    window_end_ms=int(END), max_gap_ms=60000
)
spatial = add_gps_spatial_semantics(edges.copy())
trajectory = add_gps_trajectory_plausibility(
    spatial, max_plausible_speed_kmh=qc.max_speed_kmh
)

good = (
    trajectory.supported.fillna(False).astype(bool)
    & ~trajectory.trajectory_status.eq("implausible_speed")
)
supported = trajectory.loc[good].sort_values("start_ms")

if supported.empty:
    raise RuntimeError("No supported GPS intervals.")

# Do not join across missing/unsupported intervals.
blocks = []
current = []
for e in supported.itertuples(index=False):
    if current and int(e.start_ms) != int(current[-1].end_ms):
        blocks.append(current)
        current = []
    current.append(e)
if current:
    blocks.append(current)

lookup = clean.drop_duplicates("timestamp").set_index("timestamp")["accuracy"]
R = 6371000.0
lat0 = float(clean.latitude.iloc[0])
lon0 = float(clean.longitude.iloc[0])

def xy(lat, lon):
    x = R * np.cos(np.radians(lat0)) * np.radians(np.asarray(lon) - lon0)
    y = R * np.radians(np.asarray(lat) - lat0)
    return x, y

tracks = []

for block in blocks:
    if len(block) < 10:
        continue

    t = np.array(
        [int(e.start_ms) for e in block] + [int(block[-1].end_ms)],
        dtype=np.int64
    )
    lat = np.array(
        [float(e.start_latitude) for e in block] +
        [float(block[-1].end_latitude)]
    )
    lon = np.array(
        [float(e.start_longitude) for e in block] +
        [float(block[-1].end_longitude)]
    )
    acc = lookup.reindex(t).to_numpy(float)

    if (
        np.any(np.diff(t) <= 0)
        or not np.isfinite(acc).all()
        or not (acc > 0).all()
        or not np.isfinite(lat).all()
        or not np.isfinite(lon).all()
    ):
        continue

    x, y = xy(lat, lon)
    sx, sy = _kalman_rts_xy(
        t, x, y, acc,
        acceleration_noise_density_m_s1p5=1.0
    )

    if not np.isfinite(sx).all() or not np.isfinite(sy).all():
        continue

    correction = np.hypot(x - sx, y - sy)

    tracks.append(dict(
        t=t, x=x, y=y, sx=sx, sy=sy,
        correction=correction
    ))

if not tracks:
    raise RuntimeError("No valid supported block with reported GPS accuracy.")

# Choose the real subsection with the largest observed smoothing adjustment.
target_ms = int(
    pd.Timestamp("2023-12-09T22:29:31.481Z").value // 10**6
)

matches = []
for candidate in tracks:
    indices = np.flatnonzero(
        np.abs(candidate["t"].astype(np.int64) - target_ms) <= 1500
    )
    for idx in indices:
        if idx + 60 <= len(candidate["t"]):
            matches.append((candidate, int(idx)))

if not matches:
    raise RuntimeError("No matching continuous GPS section found.")

# Choose the observation closest to the exact target timestamp.
matches.sort(
    key=lambda item: abs(int(item[0]["t"][item[1]]) - target_ms)
)
track, lo = matches[0]

print(
    "Matched GPS timestamp:",
    pd.to_datetime(int(track["t"][lo]), unit="ms", utc=True)
)
hi = lo + 60
sl = slice(lo, hi)

dt = np.diff(track["t"][sl]) / 1000
if np.any(dt <= 0) or np.max(dt) > 30:
    raise RuntimeError("Selected section has an unexpected GPS gap.")

peak = lo + int(np.argmax(track["correction"][sl]))

raw_distance = float(np.hypot(
    np.diff(track["x"][sl]),
    np.diff(track["y"][sl])
).sum())

smooth_distance = float(np.hypot(
    np.diff(track["sx"][sl]),
    np.diff(track["sy"][sl])
).sum())

print(f"Selected section: {pd.to_datetime(track['t'][lo], unit='ms', utc=True)}")
print(f"Raw observed path: {raw_distance:.1f} m")
print(f"Kalman/RTS path: {smooth_distance:.1f} m")
print(f"GPS observations: {hi - lo}")


# Visualize only the continuous supported GPS block containing
# the verified 60-observation example. No disconnected trip blocks.
fig, (a, b) = plt.subplots(
    1, 2, figsize=(15.5, 7.3),
    gridspec_kw={"width_ratios": [1, 1.12]}
)
fig.subplots_adjust(
    left=.075, right=.965, top=.735, bottom=.16, wspace=.28
)

# PANEL A: full continuous block, real observed coordinates.
a.plot(
    track["x"]/1000, track["y"]/1000,
    color=BLUE, lw=1.4, ls="--", alpha=.70, zorder=2
)
a.scatter(
    track["x"]/1000, track["y"]/1000,
    s=5, facecolors="none", edgecolors=RED,
    linewidths=.5, alpha=.60, rasterized=True, zorder=3
)
a.plot(
    track["sx"]/1000, track["sy"]/1000,
    color=PURPLE, lw=2.1, alpha=.96, zorder=4
)

# Highlight exactly the selected 60 observations.
selected_x = np.r_[
    track["x"][sl], track["sx"][sl]
] / 1000
selected_y = np.r_[
    track["y"][sl], track["sy"][sl]
] / 1000

xmin, xmax = float(selected_x.min()), float(selected_x.max())
ymin, ymax = float(selected_y.min()), float(selected_y.max())

pad = max(.012, .15 * max(xmax-xmin, ymax-ymin))
xmin -= pad
xmax += pad
ymin -= pad
ymax += pad

a.add_patch(Rectangle(
    (xmin, ymin), xmax-xmin, ymax-ymin,
    fill=False, edgecolor="#BD782D",
    linewidth=2, linestyle="--", zorder=8
))
a.annotate(
    "Magnified in B", (xmin, ymax),
    xytext=(3, 7), textcoords="offset points",
    fontsize=10, fontweight="bold", color="#A66C25"
)

a.set_title(
    "A  Continuous GPS trajectory",
    loc="left", fontsize=14, fontweight="bold", pad=13
)
a.set_xlabel("East–west distance (km)")
a.set_ylabel("North–south distance (km)")
a.set_aspect("equal", adjustable="datalim")

# PANEL B: same 60 observations, rotated for readability.
# This rigid rotation preserves every geographic distance.
xx = track["x"][sl].astype(float)
yy = track["y"][sl].astype(float)
sx = track["sx"][sl].astype(float)
sy = track["sy"][sl].astype(float)

# Orient the horizontal axis along the overall observed travel direction.
dx = float(sx[-1] - sx[0])
dy = float(sy[-1] - sy[0])
norm = float(np.hypot(dx, dy))

if norm < 1e-6:
    raise RuntimeError("Selected section has no usable travel direction.")

ux, uy = dx/norm, dy/norm
cx, cy = float(sx[0]), float(sy[0])

def orient(x, y):
    px, py = x-cx, y-cy
    along = px*ux + py*uy
    across = -px*uy + py*ux
    return along, across

raw_along, raw_across = orient(xx, yy)
smooth_along, smooth_across = orient(sx, sy)

b.plot(
    raw_along, raw_across,
    ls="--", lw=1.5, color=BLUE, alpha=.78,
    zorder=2, label="Raw GPS path"
)
b.scatter(
    raw_along, raw_across,
    s=23, facecolors="white", edgecolors=RED,
    linewidths=1.0, alpha=.85, zorder=4,
    label="Observed GPS locations"
)
b.plot(
    smooth_along, smooth_across,
    lw=3, color=PURPLE, alpha=.98, zorder=3,
    label="Kalman/RTS-smoothed path"
)

b.set_title(
    "B  Effect of Kalman/RTS smoothing",
    loc="left", fontsize=14, fontweight="bold", pad=34
)

# This line is outside the data area, beneath the panel heading.
b.text(
    0, 1.035,
    f"Raw path: {raw_distance:.1f} m   |   "
    f"Kalman/RTS: {smooth_distance:.1f} m",
    transform=b.transAxes,
    ha="left", va="bottom",
    fontsize=10.6, color="#354F63",
    clip_on=False
)

b.set_xlabel("Distance along trip direction (m)")
b.set_ylabel("Distance across trip direction (m)")
b.set_aspect("equal", adjustable="datalim")

# Same physical scale horizontally and vertically.
for ax in (a, b):
    ax.grid(color="#E7EDF1", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)

# Titles and legend have separate vertical space.
fig.suptitle(
    "How Kalman/RTS smooths real GPS observations",
    fontsize=20, fontweight="bold", y=.976
)
fig.text(
    .5, .920,
    "AMO327 | Dec 9, 2023 | "
    "60 consecutive observations in one supported GPS block",
    ha="center", fontsize=11.3, color="#64798A"
)

from matplotlib.lines import Line2D

fig.legend(
    handles=[
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor="white", markeredgecolor=RED,
               markersize=6, label="Observed GPS locations"),
        Line2D([], [], color=BLUE, linestyle="--",
               lw=1.8, label="Raw GPS path"),
        Line2D([], [], color=PURPLE,
               lw=3, label="Kalman/RTS-smoothed path")
    ],
    loc="upper center",
    bbox_to_anchor=(.5, .872),
    ncol=3, frameon=False, fontsize=10.6
)

fig.text(
    .5, .06,
    "Both paths use the same real GPS observations. "
    "Distances are estimates, not independent ground truth.",
    ha="center", fontsize=10, color="#64798A"
)

difference_m = float(track["correction"][peak])

out = Path.home() / "cortex2_visualizations/outputs"
out.mkdir(parents=True, exist_ok=True)

for ext in ("png", "pdf"):
    dest = out / f"AMO327_Kalman_RTS_zoom.{ext}"
    fig.savefig(dest, dpi=240, bbox_inches="tight")
    print("Saved:", dest)

print(f"Maximum observed-to-smoothed position difference: {difference_m:.2f} m")
print(f"Supported blocks plotted: {len(tracks)}")
plt.close(fig)
