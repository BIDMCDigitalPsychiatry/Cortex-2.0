from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_anchor_episodes import build_anchor_compatible_location_episodes
from cortex2_engine.features.gps_place_candidates import build_place_candidates

# ============================================================
# CONFIG
# ============================================================

PARTICIPANT_ID = "AMO327"
LOCAL_TIMEZONE = "America/New_York"
MAX_GAP_MS = 60_000
COMPATIBILITY_MULTIPLIER = 1.0
REFERENCE_DAYS = 14

RAW_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/gps_clinical_weekly_raw_points.parquet"
)
ANCHOR_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/home_time/participant_home_anchors.csv"
)
OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/home_time/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / f"home_geography_assignment_based_{PARTICIPANT_ID}.png"
OUT_PDF = OUT_DIR / f"home_geography_assignment_based_{PARTICIPANT_ID}.pdf"

# ============================================================
# HELPERS
# ============================================================

def to_timestamp_ms(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        # Decide whether already milliseconds or seconds
        vals = pd.to_numeric(series, errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            raise ValueError("Timestamp column has no finite values.")
        # crude heuristic: unix milliseconds are usually >1e11
        if np.nanmedian(finite) > 1e11:
            return vals.astype("Int64")
        else:
            return (vals * 1000.0).round().astype("Int64")
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return (dt.astype("int64") // 1_000_000).astype("Int64")

def ms_to_local_datetime(ms):
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert(LOCAL_TIMEZONE)

def latlon_to_xy_m(lat, lon, lat0, lon0):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(lat0)
    lon0 = float(lon0)

    r = 6371000.0
    x = np.radians(lon - lon0) * r * np.cos(np.radians((lat + lat0) / 2.0))
    y = np.radians(lat - lat0) * r
    return x, y

def add_xy(df, lat0, lon0):
    x, y = latlon_to_xy_m(df["latitude"], df["longitude"], lat0, lon0)
    out = df.copy()
    out["x_m"] = x
    out["y_m"] = y
    out["x_km"] = x / 1000.0
    out["y_km"] = y / 1000.0
    return out

# ============================================================
# LOAD
# ============================================================

raw = pd.read_parquet(RAW_PATH)
anchors = pd.read_csv(ANCHOR_PATH)

raw["participant_id"] = raw["participant_id"].astype(str)
anchors["participant_id"] = anchors["participant_id"].astype(str)

raw = raw.loc[raw["participant_id"] == PARTICIPANT_ID].copy()
if raw.empty:
    raise ValueError(f"No raw GPS rows found for participant {PARTICIPANT_ID}")

anchor_row = anchors.loc[anchors["participant_id"] == PARTICIPANT_ID].copy()
if anchor_row.empty:
    raise ValueError(f"No anchor row found for participant {PARTICIPANT_ID}")
anchor_row = anchor_row.iloc[0]

home_place_id = int(anchor_row["home_place_candidate_id"])
home_confidence = float(anchor_row["home_confidence"])
n_valid_nights = int(anchor_row["home_n_valid_nights"])
n_nights_at_home = int(anchor_row["home_n_nights_at_home"])
home_lat = float(anchor_row["representative_latitude"])
home_lon = float(anchor_row["representative_longitude"])

raw["timestamp_ms"] = to_timestamp_ms(raw["timestamp"])
raw = raw.loc[raw["timestamp_ms"].notna()].copy()
raw["timestamp_ms"] = raw["timestamp_ms"].astype(np.int64)
raw["timestamp_local"] = raw["timestamp_ms"].apply(ms_to_local_datetime)
raw["local_date"] = raw["timestamp_local"].dt.date

# first 14 local days for plotting
first_local_date = raw["local_date"].min()
ref_start_local = pd.Timestamp(f"{first_local_date} 00:00:00").tz_localize(LOCAL_TIMEZONE)
ref_end_local = ref_start_local + pd.Timedelta(days=REFERENCE_DAYS)

raw_ref = raw.loc[
    (raw["timestamp_local"] >= ref_start_local) &
    (raw["timestamp_local"] < ref_end_local)
].copy()

if raw_ref.empty:
    raise ValueError("No raw GPS rows found inside the first 14 local days.")

# ============================================================
# BUILD CANDIDATE-PLACE ASSIGNMENTS ON FULL AVAILABLE DATA
# ============================================================

full_start_ms = int(raw["timestamp_ms"].min())
full_end_ms = int(raw["timestamp_ms"].max()) + 1

work_raw = (
    raw[["timestamp_ms", "latitude", "longitude", "accuracy"]]
    .rename(columns={"timestamp_ms": "timestamp"})
    .sort_values("timestamp")
    .reset_index(drop=True)
)

intervals = build_gps_intervals(
    work_raw,
    window_start_ms=full_start_ms,
    window_end_ms=full_end_ms,
    max_gap_ms=MAX_GAP_MS,
)

if intervals.empty:
    raise ValueError("No GPS intervals were built.")

supported = intervals.loc[
    intervals["supported"].fillna(False).astype(bool)
].copy()

if supported.empty:
    raise ValueError("No supported intervals were found.")

supported = add_gps_spatial_semantics(supported)
segments = build_anchor_compatible_location_episodes(supported)

if segments.empty:
    raise ValueError("No location microsegments were built.")

assignments, places = build_place_candidates(
    segments,
    compatibility_multiplier=COMPATIBILITY_MULTIPLIER,
)

if assignments.empty:
    raise ValueError("No place assignments were built.")

required_cols = ["episode_start_ms", "episode_end_ms", "place_candidate_id"]
for c in required_cols:
    if c not in assignments.columns:
        raise ValueError(f"Assignments are missing required column: {c}")

assignments = assignments.copy()
assignments["episode_start_ms"] = pd.to_numeric(assignments["episode_start_ms"], errors="coerce")
assignments["episode_end_ms"] = pd.to_numeric(assignments["episode_end_ms"], errors="coerce")
assignments["place_candidate_id"] = pd.to_numeric(assignments["place_candidate_id"], errors="coerce")
assignments = assignments.dropna(subset=["episode_start_ms", "episode_end_ms", "place_candidate_id"]).copy()
assignments["place_candidate_id"] = assignments["place_candidate_id"].astype(int)

# ============================================================
# MAP RAW POINTS TO ASSIGNMENT STATUS
# ============================================================

raw_ref["status"] = "unsupported_or_unassigned"

# Use assignment windows to label points
for row in assignments.itertuples(index=False):
    mask = (
        (raw_ref["timestamp_ms"] >= int(row.episode_start_ms)) &
        (raw_ref["timestamp_ms"] <= int(row.episode_end_ms))
    )
    if not mask.any():
        continue
    if int(row.place_candidate_id) == home_place_id:
        raw_ref.loc[mask, "status"] = "home_assigned"
    else:
        # do not overwrite home
        raw_ref.loc[
            mask & raw_ref["status"].ne("home_assigned"),
            "status"
        ] = "other_place_assigned"

raw_ref = add_xy(raw_ref, home_lat, home_lon)

# ============================================================
# MAIN PANEL GEOMETRY
# ============================================================

# Zoom box based on actual home-assigned points
home_pts = raw_ref.loc[raw_ref["status"] == "home_assigned"].copy()
if home_pts.empty:
    raise ValueError("No raw points were assigned to the inferred home place in the 14-day reference period.")

qx = np.nanpercentile(np.abs(home_pts["x_m"]), 95)
qy = np.nanpercentile(np.abs(home_pts["y_m"]), 95)
zoom_half_width_m = max(200.0, float(max(qx, qy) + 40.0))

x0_m = -zoom_half_width_m
x1_m = zoom_half_width_m
y0_m = -zoom_half_width_m
y1_m = zoom_half_width_m

# ============================================================
# PLOT
# ============================================================

plt.close("all")
fig = plt.figure(figsize=(14, 7), dpi=160)
fig.patch.set_facecolor("white")

# ---- overall title
fig.suptitle(
    f"Participant {PARTICIPANT_ID}: geographic context of inferred home",
    fontsize=18,
    fontweight="bold",
    y=0.96,
)
fig.text(
    0.5, 0.915,
    f"Home confidence = {home_confidence:.2f}   |   Valid nights = {n_valid_nights}   |   Nights won by home = {n_nights_at_home}",
    ha="center",
    va="center",
    fontsize=12,
)

# ---- left main axis
ax_main = fig.add_axes([0.08, 0.18, 0.48, 0.50])
ax_main.set_title(
    "GPS observations during the 14-day home-reference period",
    fontsize=16,
    fontweight="bold",
    pad=8,
)

# Plot order: gray first, then blue, then orange
for status, color, alpha, size, label in [
    ("unsupported_or_unassigned", "#bdbdbd", 0.35, 5, "Unsupported / unassigned"),
    ("other_place_assigned", "#2c7fb8", 0.55, 6, "Assigned to other candidate place"),
    ("home_assigned", "#ff8c00", 0.70, 7, "Assigned to inferred home place"),
]:
    d = raw_ref.loc[raw_ref["status"] == status]
    if len(d):
        ax_main.scatter(
            d["x_km"], d["y_km"],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            label=label,
            rasterized=True,
        )

# star at inferred home
ax_main.scatter(
    [0], [0],
    marker="*",
    s=160,
    c="#2ca25f",
    edgecolors="none",
    zorder=5,
    label="Inferred home center",
)

ax_main.set_xlabel("East–west displacement from home (km)", fontsize=13)
ax_main.set_ylabel("North–south displacement from home (km)", fontsize=13)
ax_main.grid(True, alpha=0.20)

# ---- inset axis
ax_in = fig.add_axes([0.63, 0.48, 0.28, 0.34])
ax_in.set_title("Zoom around inferred home", fontsize=15, fontweight="bold", pad=6)

for status, color, alpha, size in [
    ("unsupported_or_unassigned", "#bdbdbd", 0.35, 6),
    ("other_place_assigned", "#2c7fb8", 0.60, 7),
    ("home_assigned", "#ff8c00", 0.75, 8),
]:
    d = raw_ref.loc[raw_ref["status"] == status]
    if len(d):
        ax_in.scatter(
            d["x_m"], d["y_m"],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )

ax_in.scatter(
    [0], [0],
    marker="*",
    s=180,
    c="#2ca25f",
    edgecolors="none",
    zorder=5,
)

ax_in.set_xlim(x0_m, x1_m)
ax_in.set_ylim(y0_m, y1_m)
ax_in.set_xlabel("Meters east / west", fontsize=12)
ax_in.set_ylabel("Meters north / south", fontsize=12)
ax_in.grid(True, alpha=0.20)

# ---- rectangle on main corresponding to inset
rect = Rectangle(
    (x0_m / 1000.0, y0_m / 1000.0),
    (x1_m - x0_m) / 1000.0,
    (y1_m - y0_m) / 1000.0,
    fill=False,
    edgecolor="black",
    linewidth=1.2,
)
ax_main.add_patch(rect)

# ---- connectors
fig.add_artist(ConnectionPatch(
    xyA=(x1_m / 1000.0, y1_m / 1000.0), coordsA=ax_main.transData,
    xyB=(0, 1), coordsB=ax_in.transAxes,
    color="black", linewidth=1.0, alpha=0.7
))
fig.add_artist(ConnectionPatch(
    xyA=(x1_m / 1000.0, y0_m / 1000.0), coordsA=ax_main.transData,
    xyB=(0, 0), coordsB=ax_in.transAxes,
    color="black", linewidth=1.0, alpha=0.7
))

# ---- legend
legend_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff8c00", markersize=7, label="Assigned to inferred home place"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#2c7fb8", markersize=7, label="Assigned to other candidate place"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#bdbdbd", markersize=7, label="Unsupported / unassigned"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="#2ca25f", markersize=11, label="Inferred home center"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.77, 0.43),
    frameon=False,
    fontsize=11,
    ncol=1,
    title="GPS point classification",
    title_fontsize=11,
)

# ---- caption
caption = (
    "Points are colored by the actual Cortex-2.0 candidate-place assignment used in home inference. "
    f"Candidate places are built from the participant’s available GPS data; the figure displays only the first {REFERENCE_DAYS} local days. "
    "Home is the candidate place that wins the most valid 12 AM–6 AM local nights."
)
fig.text(0.08, 0.08, caption, ha="left", va="center", fontsize=11)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(OUT_PNG)
print(OUT_PDF)

print()
print("Counts in displayed 14-day period:")
print(raw_ref["status"].value_counts(dropna=False).to_string())
