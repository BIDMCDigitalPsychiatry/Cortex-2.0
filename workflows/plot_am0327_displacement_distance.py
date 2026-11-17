from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

PARTICIPANT = "AMO327"
TZ = "America/New_York"
REFERENCE_DAYS = 14

RAW_GPS = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_clinical_weekly_raw_points.parquet"
)

HOME_ANCHORS = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "participant_home_anchors.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_OUT = (
    OUT_DIR
    / "AMO327_displacement_distance_real_data.png"
)

PDF_OUT = (
    OUT_DIR
    / "AMO327_displacement_distance_real_data.pdf"
)

# Same support rule used in the validated Cortex-2.0 GPS pipeline.
MAX_GAP_MS = 60_000

# Trajectory plausibility ceiling used in Cortex-2.0.
MAX_SPEED_KMH = 1000.0

# ============================================================
# HELPERS
# ============================================================

def first_present(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    r = 6371000.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * r
        * np.arcsin(
            np.sqrt(a)
        )
    )


def latlon_to_local_m(
    lat,
    lon,
    lat0,
    lon0,
):
    r = 6371000.0

    x = (
        np.radians(
            lon - lon0
        )
        * r
        * np.cos(
            np.radians(lat0)
        )
    )

    y = (
        np.radians(
            lat - lat0
        )
        * r
    )

    return x, y


# ============================================================
# LOAD RAW GPS
# ============================================================

df = pd.read_parquet(
    RAW_GPS
)

print("=" * 100)
print("AMO327 CORTEX-2.0 DISPLACEMENT DISTANCE")
print("=" * 100)

print("\nRaw columns:")
print(list(df.columns))

pid_col = first_present(
    df.columns,
    [
        "participant_id",
        "clinic_id",
    ],
)

time_col = first_present(
    df.columns,
    [
        "timestamp",
        "timestamp_ms",
        "time",
        "datetime",
    ],
)

lat_col = first_present(
    df.columns,
    [
        "latitude",
        "lat",
    ],
)

lon_col = first_present(
    df.columns,
    [
        "longitude",
        "lon",
        "lng",
    ],
)

acc_col = first_present(
    df.columns,
    [
        "accuracy",
        "accuracy_m",
        "horizontal_accuracy",
    ],
)

if pid_col is None:
    raise RuntimeError(
        "Could not identify participant column."
    )

if time_col is None:
    raise RuntimeError(
        "Could not identify timestamp column."
    )

if lat_col is None or lon_col is None:
    raise RuntimeError(
        "Could not identify latitude/longitude columns."
    )

if acc_col is None:
    raise RuntimeError(
        "Could not identify GPS accuracy column."
    )

df = df[
    df[
        pid_col
    ].astype(str).eq(
        PARTICIPANT
    )
].copy()

print(
    "\nParticipant raw GPS rows:",
    len(df),
)

if df.empty:
    raise RuntimeError(
        f"No GPS rows found for {PARTICIPANT}."
    )

# ============================================================
# TIMESTAMP
# ============================================================

t = pd.to_numeric(
    df[
        time_col
    ],
    errors="coerce",
)

if (
    t.notna().mean() > 0.95
    and t.median() > 1e11
):
    df["datetime_utc"] = pd.to_datetime(
        t,
        unit="ms",
        utc=True,
        errors="coerce",
    )

else:
    df["datetime_utc"] = pd.to_datetime(
        df[
            time_col
        ],
        utc=True,
        errors="coerce",
    )

df["datetime_local"] = (
    df[
        "datetime_utc"
    ]
    .dt.tz_convert(
        TZ
    )
)

df[
    lat_col
] = pd.to_numeric(
    df[
        lat_col
    ],
    errors="coerce",
)

df[
    lon_col
] = pd.to_numeric(
    df[
        lon_col
    ],
    errors="coerce",
)

df[
    acc_col
] = pd.to_numeric(
    df[
        acc_col
    ],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "datetime_local",
        lat_col,
        lon_col,
        acc_col,
    ]
).copy()

df = df.sort_values(
    "datetime_local"
).reset_index(
    drop=True
)

# ============================================================
# SAME 14-DAY REFERENCE PERIOD AS HOME FIGURE
# ============================================================

first_day = (
    df[
        "datetime_local"
    ]
    .dt.floor(
        "D"
    )
    .min()
)

window_end = (
    first_day
    +
    pd.Timedelta(
        days=REFERENCE_DAYS
    )
)

ref = df[
    (
        df[
            "datetime_local"
        ]
        >= first_day
    )
    &
    (
        df[
            "datetime_local"
        ]
        < window_end
    )
].copy()

print()
print("Reference start:", first_day)
print("Reference end:", window_end)
print("Reference GPS rows:", len(ref))

# ============================================================
# BUILD CONSECUTIVE GPS EDGES
# ============================================================

ref["next_time"] = (
    ref[
        "datetime_local"
    ]
    .shift(-1)
)

ref["next_lat"] = (
    ref[
        lat_col
    ]
    .shift(-1)
)

ref["next_lon"] = (
    ref[
        lon_col
    ]
    .shift(-1)
)

ref["next_accuracy"] = (
    ref[
        acc_col
    ]
    .shift(-1)
)

ref["gap_ms"] = (
    (
        ref[
            "next_time"
        ]
        -
        ref[
            "datetime_local"
        ]
    )
    .dt.total_seconds()
    * 1000.0
)

ref["distance_m"] = haversine_m(
    ref[
        lat_col
    ],
    ref[
        lon_col
    ],
    ref[
        "next_lat"
    ],
    ref[
        "next_lon"
    ],
)

# Cortex-2.0 spatial support:
# observed displacement must exceed the summed
# endpoint accuracy radii.
ref["accuracy_threshold_m"] = (
    ref[
        acc_col
    ]
    +
    ref[
        "next_accuracy"
    ]
)

ref["temporally_supported"] = (
    ref[
        "gap_ms"
    ].gt(0)
    &
    ref[
        "gap_ms"
    ].le(
        MAX_GAP_MS
    )
)

ref["displacement_exceeds_accuracy"] = (
    ref[
        "distance_m"
    ]
    >
    ref[
        "accuracy_threshold_m"
    ]
)

ref["speed_kmh"] = (
    ref[
        "distance_m"
    ]
    /
    (
        ref[
            "gap_ms"
        ]
        / 1000.0
    )
    * 3.6
)

ref["plausible_speed"] = (
    ref[
        "speed_kmh"
    ]
    <= MAX_SPEED_KMH
)

ref["plausible_displacement"] = (
    ref[
        "temporally_supported"
    ]
    &
    ref[
        "displacement_exceeds_accuracy"
    ]
    &
    ref[
        "plausible_speed"
    ]
)

# ============================================================
# DAILY DISPLACEMENT
# ============================================================

ref["date"] = (
    ref[
        "datetime_local"
    ]
    .dt.date
)

daily = (
    ref[
        ref[
            "plausible_displacement"
        ]
    ]
    .groupby(
        "date",
        as_index=False,
    )
    .agg(
        displacement_distance_m=(
            "distance_m",
            "sum",
        ),
        displacement_edges=(
            "distance_m",
            "size",
        ),
    )
)

daily[
    "displacement_distance_km"
] = (
    daily[
        "displacement_distance_m"
    ]
    / 1000.0
)

total_distance_km = (
    daily[
        "displacement_distance_km"
    ]
    .sum()
)

print()
print("=" * 100)
print("CORTEX-2.0 DISPLACEMENT SUMMARY")
print("=" * 100)

print(
    "Plausible supported displacement edges:",
    int(
        ref[
            "plausible_displacement"
        ].sum()
    ),
)

print(
    "Total displacement distance across 14 days:",
    f"{total_distance_km:.2f} km",
)

print()
print(daily.to_string(index=False))

# ============================================================
# HOME ANCHOR FOR SAME VISUAL REFERENCE
# ============================================================

anchors = pd.read_csv(
    HOME_ANCHORS,
    low_memory=False,
)

anchor = anchors[
    anchors[
        "participant_id"
    ].astype(str).eq(
        PARTICIPANT
    )
].copy()

home_lat = None
home_lon = None

for lat_name in [
    "home_latitude",
    "representative_latitude",
    "home_center_latitude",
]:

    if lat_name in anchor.columns:
        home_lat = pd.to_numeric(
            anchor.iloc[0][lat_name],
            errors="coerce",
        )
        break

for lon_name in [
    "home_longitude",
    "representative_longitude",
    "home_center_longitude",
]:

    if lon_name in anchor.columns:
        home_lon = pd.to_numeric(
            anchor.iloc[0][lon_name],
            errors="coerce",
        )
        break

# Fall back to first GPS point only if anchor coordinates
# are not stored in the anchor table.
if (
    home_lat is None
    or home_lon is None
    or not np.isfinite(home_lat)
    or not np.isfinite(home_lon)
):
    home_lat = float(
        ref[
            lat_col
        ].iloc[0]
    )

    home_lon = float(
        ref[
            lon_col
        ].iloc[0]
    )

# ============================================================
# LOCAL-METER COORDINATES
# ============================================================

x, y = latlon_to_local_m(
    ref[
        lat_col
    ].to_numpy(),
    ref[
        lon_col
    ].to_numpy(),
    home_lat,
    home_lon,
)

ref["x_m"] = x
ref["y_m"] = y

next_x, next_y = latlon_to_local_m(
    ref[
        "next_lat"
    ].to_numpy(),
    ref[
        "next_lon"
    ].to_numpy(),
    home_lat,
    home_lon,
)

ref["next_x_m"] = next_x
ref["next_y_m"] = next_y

# ============================================================
# FIGURE
# ============================================================

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 11

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        15,
        6,
    ),
)

# ------------------------------------------------------------
# PANEL A: REAL GPS + COUNTED DISPLACEMENT EDGES
# ------------------------------------------------------------

ax = axes[0]

ax.scatter(
    ref[
        "x_m"
    ] / 1000.0,
    ref[
        "y_m"
    ] / 1000.0,
    s=5,
    alpha=0.15,
    label="Observed GPS",
)

valid = ref[
    ref[
        "plausible_displacement"
    ]
].copy()

for row in valid.itertuples():

    ax.plot(
        [
            row.x_m / 1000.0,
            row.next_x_m / 1000.0,
        ],
        [
            row.y_m / 1000.0,
            row.next_y_m / 1000.0,
        ],
        linewidth=0.8,
        alpha=0.55,
    )

ax.scatter(
    0,
    0,
    marker="*",
    s=160,
    zorder=10,
    label="Inferred home",
)

ax.set_title(
    "A. GPS movement contributing to displacement distance"
)

ax.set_xlabel(
    "East–west displacement from home (km)"
)

ax.set_ylabel(
    "North–south displacement from home (km)"
)

ax.legend(
    frameon=False,
    loc="best",
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

# ------------------------------------------------------------
# PANEL B: DAILY DISPLACEMENT DISTANCE
# ------------------------------------------------------------

ax = axes[1]

daily_plot = daily.copy()

daily_plot["date"] = pd.to_datetime(
    daily_plot[
        "date"
    ]
)

ax.plot(
    daily_plot[
        "date"
    ],
    daily_plot[
        "displacement_distance_km"
    ],
    marker="o",
    linewidth=1.8,
)

ax.set_title(
    "B. Daily Cortex-2.0 displacement distance"
)

ax.set_xlabel(
    "Day"
)

ax.set_ylabel(
    "Supported displacement distance (km/day)"
)

ax.tick_params(
    axis="x",
    rotation=45,
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

ax.text(
    0.03,
    0.97,
    (
        f"14-day total = {total_distance_km:.1f} km\n"
        f"Counted edges = "
        f"{int(ref['plausible_displacement'].sum()):,}"
    ),
    transform=ax.transAxes,
    va="top",
)

fig.suptitle(
    "Participant AMO327: Cortex-2.0 displacement distance\n"
    "Same 14-day reference period used for home inference",
    fontsize=16,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.93,
    ]
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("=" * 100)
print("SAVED")
print("=" * 100)
print(PNG_OUT)
print(PDF_OUT)
