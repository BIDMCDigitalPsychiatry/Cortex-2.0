#!/usr/bin/env python3

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch


# ============================================================
# SETTINGS
# ============================================================

PARTICIPANT_ID = "AMO327"
LOCAL_TIMEZONE = "America/New_York"

HOME_ANCHOR_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "participant_home_anchors.csv"
)

RAW_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_clinical_weekly_raw_points.parquet"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PNG = (
    OUT_DIR /
    f"home_geography_inset_{PARTICIPANT_ID}.png"
)

OUT_PDF = (
    OUT_DIR /
    f"home_geography_inset_{PARTICIPANT_ID}.pdf"
)


# Visual zoom only — NOT home inference thresholds
INSET_HALF_WIDTH_M = 250.0
HOME_HIGHLIGHT_RADIUS_M = 120.0


# ============================================================
# HELPERS
# ============================================================

def pick_col(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find any of these columns: {candidates}"
    )


def to_utc_datetime(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().mean() > 0.9:

        maximum = numeric.dropna().abs().max()

        unit = (
            "ms"
            if maximum > 1e11
            else "s"
        )

        return pd.to_datetime(
            numeric,
            unit=unit,
            utc=True,
            errors="coerce",
        )

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    )


def latlon_to_xy_m(
    lat,
    lon,
    lat0,
    lon0,
):

    R = 6371000.0

    lat = np.asarray(
        lat,
        dtype=float,
    )

    lon = np.asarray(
        lon,
        dtype=float,
    )

    lat0_rad = math.radians(
        lat0
    )

    x = (
        np.radians(
            lon - lon0
        )
        * math.cos(lat0_rad)
        * R
    )

    y = (
        np.radians(
            lat - lat0
        )
        * R
    )

    return x, y


# ============================================================
# LOAD HOME ANCHOR
# ============================================================

anchors = pd.read_csv(
    HOME_ANCHOR_PATH
)

anchors["participant_id"] = (
    anchors["participant_id"]
    .astype(str)
)

anchor = anchors.loc[
    anchors["participant_id"]
    == PARTICIPANT_ID
].copy()

if anchor.empty:
    raise ValueError(
        f"No home anchor found for {PARTICIPANT_ID}"
    )


home_lat = float(
    anchor.iloc[0][
        "representative_latitude"
    ]
)

home_lon = float(
    anchor.iloc[0][
        "representative_longitude"
    ]
)


home_conf = (
    anchor.iloc[0]["home_confidence"]
    if "home_confidence" in anchor.columns
    else np.nan
)

valid_nights = (
    anchor.iloc[0]["home_n_valid_nights"]
    if "home_n_valid_nights" in anchor.columns
    else np.nan
)

home_nights = (
    anchor.iloc[0]["home_n_nights_at_home"]
    if "home_n_nights_at_home" in anchor.columns
    else np.nan
)


print(
    f"[INFO] Home center: "
    f"{home_lat:.6f}, {home_lon:.6f}"
)


# ============================================================
# LOAD REAL RAW GPS
# ============================================================

if not RAW_PATH.exists():
    raise FileNotFoundError(
        RAW_PATH
    )


print(
    f"[INFO] Using raw GPS file: {RAW_PATH}"
)


# Filter during parquet read if pyarrow supports it.
try:

    raw = pd.read_parquet(
        RAW_PATH,
        filters=[
            (
                "participant_id",
                "==",
                PARTICIPANT_ID,
            )
        ],
    )

except Exception:

    raw = pd.read_parquet(
        RAW_PATH
    )

    raw = raw.loc[
        raw["participant_id"]
        .astype(str)
        == PARTICIPANT_ID
    ].copy()


if raw.empty:
    raise ValueError(
        f"No raw GPS rows found for "
        f"{PARTICIPANT_ID}"
    )


print(
    f"[INFO] Raw GPS rows for "
    f"{PARTICIPANT_ID}: "
    f"{len(raw):,}"
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

lat_col = pick_col(
    raw,
    [
        "latitude",
        "lat",
        "Latitude",
    ],
)

lon_col = pick_col(
    raw,
    [
        "longitude",
        "lon",
        "Longitude",
    ],
)

time_col = pick_col(
    raw,
    [
        "timestamp",
        "timestamp_ms",
        "timestamp_utc_ms",
        "time",
        "datetime",
        "UTC_timestamp",
    ],
)


raw["timestamp_utc"] = (
    to_utc_datetime(
        raw[time_col]
    )
)


raw = raw.dropna(
    subset=[
        "timestamp_utc",
        lat_col,
        lon_col,
    ]
).copy()


# ============================================================
# FIRST 14 LOCAL DAYS
# ============================================================

raw["timestamp_local"] = (
    raw["timestamp_utc"]
    .dt.tz_convert(
        LOCAL_TIMEZONE
    )
)

raw["local_date"] = (
    raw["timestamp_local"]
    .dt.date
)


first_date = min(
    raw["local_date"]
)

reference_dates = {
    first_date
    + pd.Timedelta(days=i)
    for i in range(14)
}


sub = raw.loc[
    raw["local_date"].isin(
        reference_dates
    )
].copy()


if sub.empty:
    raise ValueError(
        "No GPS data in first 14 local days."
    )


print(
    "[INFO] Reference period:",
    min(sub["local_date"]),
    "to",
    max(sub["local_date"]),
)

print(
    "[INFO] GPS points in reference period:",
    f"{len(sub):,}",
)


# ============================================================
# CONVERT TO METERS RELATIVE TO HOME
# ============================================================

x_m, y_m = latlon_to_xy_m(
    sub[lat_col].to_numpy(),
    sub[lon_col].to_numpy(),
    home_lat,
    home_lon,
)

sub["x_m"] = x_m
sub["y_m"] = y_m

sub["dist_home_m"] = np.sqrt(
    sub["x_m"] ** 2
    + sub["y_m"] ** 2
)


# This highlighting is ONLY for the visual inset.
sub["near_home_visual"] = (
    sub["dist_home_m"]
    <= HOME_HIGHLIGHT_RADIUS_M
)


# ============================================================
# ROBUST OVERVIEW LIMITS
# ============================================================

xmin, xmax = np.nanpercentile(
    sub["x_m"],
    [0.5, 99.5],
)

ymin, ymax = np.nanpercentile(
    sub["y_m"],
    [0.5, 99.5],
)


xspan = xmax - xmin
yspan = ymax - ymin

xpad = max(
    100,
    xspan * 0.08,
)

ypad = max(
    100,
    yspan * 0.08,
)

xmin -= xpad
xmax += xpad
ymin -= ypad
ymax += ypad


# ============================================================
# PLOT
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
})


fig = plt.figure(
    figsize=(10.5, 6.2)
)


# Main map
ax = fig.add_axes(
    [
        0.08,
        0.15,
        0.53,
        0.70,
    ]
)


ax.scatter(
    sub["x_m"] / 1000,
    sub["y_m"] / 1000,
    s=3,
    alpha=0.35,
    linewidths=0,
)


# Inferred home center
ax.scatter(
    0,
    0,
    s=55,
    marker="*",
    zorder=10,
    label="Inferred home",
)


ax.set_xlim(
    xmin / 1000,
    xmax / 1000,
)

ax.set_ylim(
    ymin / 1000,
    ymax / 1000,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.set_xlabel(
    "East–west displacement from home (km)"
)

ax.set_ylabel(
    "North–south displacement from home (km)"
)

ax.set_title(
    "GPS observations during the 14-day home-reference period",
    fontweight="bold",
)

ax.grid(
    alpha=0.15
)

ax.legend(
    frameon=False,
    loc="upper right",
)


# ============================================================
# ZOOM BOX
# ============================================================

half = INSET_HALF_WIDTH_M

rect = Rectangle(
    (
        -half / 1000,
        -half / 1000,
    ),
    2 * half / 1000,
    2 * half / 1000,
    fill=False,
    linewidth=1.3,
)

ax.add_patch(
    rect
)


# ============================================================
# INSET
# ============================================================

ax2 = fig.add_axes(
    [
        0.68,
        0.48,
        0.27,
        0.35,
    ]
)


zoom = sub.loc[
    (sub["x_m"].abs() <= half)
    &
    (sub["y_m"].abs() <= half)
].copy()


away_zoom = zoom.loc[
    ~zoom["near_home_visual"]
]

home_zoom = zoom.loc[
    zoom["near_home_visual"]
]


ax2.scatter(
    away_zoom["x_m"],
    away_zoom["y_m"],
    s=5,
    alpha=0.35,
    linewidths=0,
    label="Other GPS",
)


ax2.scatter(
    home_zoom["x_m"],
    home_zoom["y_m"],
    s=7,
    alpha=0.65,
    linewidths=0,
    label="Near inferred home",
)


ax2.scatter(
    0,
    0,
    marker="*",
    s=90,
    zorder=10,
)


ax2.set_xlim(
    -half,
    half,
)

ax2.set_ylim(
    -half,
    half,
)

ax2.set_aspect(
    "equal",
    adjustable="box",
)

ax2.set_title(
    "Zoom around inferred home",
    fontsize=11,
    fontweight="bold",
)

ax2.set_xlabel(
    "Meters east / west",
    fontsize=9,
)

ax2.set_ylabel(
    "Meters north / south",
    fontsize=9,
)

ax2.tick_params(
    labelsize=8
)

ax2.grid(
    alpha=0.15
)


# Connecting lines
fig.add_artist(
    ConnectionPatch(
        xyA=(
            half / 1000,
            half / 1000,
        ),
        coordsA=ax.transData,
        xyB=(0, 1),
        coordsB=ax2.transAxes,
        linewidth=0.8,
        alpha=0.5,
    )
)

fig.add_artist(
    ConnectionPatch(
        xyA=(
            half / 1000,
            -half / 1000,
        ),
        coordsA=ax.transData,
        xyB=(0, 0),
        coordsB=ax2.transAxes,
        linewidth=0.8,
        alpha=0.5,
    )
)


# ============================================================
# ANNOTATION
# ============================================================

fig.suptitle(
    f"Participant {PARTICIPANT_ID}: geographic context of inferred home",
    fontsize=15,
    fontweight="bold",
    y=0.96,
)


info = (
    f"Home confidence = {home_conf:.2f}"
    if pd.notna(home_conf)
    else ""
)

if pd.notna(valid_nights):
    info += (
        f"   |   Valid nights = "
        f"{int(valid_nights)}"
    )

if pd.notna(home_nights):
    info += (
        f"   |   Nights won by home = "
        f"{int(home_nights)}"
    )


fig.text(
    0.08,
    0.90,
    info,
    fontsize=10,
)


fig.text(
    0.08,
    0.045,
    (
        "The star marks the home location inferred by the Cortex-2.0 "
        "night-by-night candidate-place procedure. "
        "The inset radius is for visualization only and is not used "
        "to define home."
    ),
    fontsize=9,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
