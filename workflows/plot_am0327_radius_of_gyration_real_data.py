from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pyarrow.parquet as pq

# ============================================================
# CONFIG
# ============================================================

PARTICIPANT = "AMO327"
TZ = "America/New_York"
REFERENCE_DAYS = 14
MAX_GAP_MS = 60_000
MAX_SPEED_KMH = 1000.0

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

RAW_GPS = ROOT / "gps_clinical_weekly_raw_points.parquet"

DAILY_FEATURES = (
    ROOT
    / "gps_clinical_weekly_daily_features.parquet"
)

HOME_ANCHORS = (
    ROOT
    / "home_time"
    / "participant_home_anchors.csv"
)

OUT_DIR = (
    ROOT
    / "home_time"
    / "figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_OUT = (
    OUT_DIR
    / "AMO327_radius_of_gyration_real_data.png"
)

PDF_OUT = (
    OUT_DIR
    / "AMO327_radius_of_gyration_real_data.pdf"
)

CSV_OUT = (
    OUT_DIR
    / "AMO327_radius_of_gyration_daily_summary.csv"
)

# ============================================================
# EXACT CORTEX FUNCTIONS
# ============================================================

from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)


def locate_function(name):

    for py in Path("src").rglob("*.py"):

        text = py.read_text(
            errors="ignore"
        )

        if f"def {name}" in text:

            rel = (
                py.relative_to("src")
                .with_suffix("")
            )

            module_name = ".".join(
                rel.parts
            )

            module = importlib.import_module(
                module_name
            )

            return getattr(
                module,
                name,
            )

    raise RuntimeError(
        f"Could not locate {name}"
    )


build_gps_intervals = locate_function(
    "build_gps_intervals"
)

add_gps_spatial_semantics = locate_function(
    "add_gps_spatial_semantics"
)

add_gps_trajectory_plausibility = locate_function(
    "add_gps_trajectory_plausibility"
)


# ============================================================
# HELPERS
# ============================================================

def first_present(columns, candidates):

    for c in candidates:

        if c in columns:
            return c

    return None


def latlon_to_local_km(
    lat,
    lon,
    lat0,
    lon0,
):

    r = 6371000.0

    lat = np.asarray(
        lat,
        dtype=float,
    )

    lon = np.asarray(
        lon,
        dtype=float,
    )

    x = (
        np.radians(
            lon - lon0
        )
        * r
        * np.cos(
            np.radians(lat0)
        )
        / 1000.0
    )

    y = (
        np.radians(
            lat - lat0
        )
        * r
        / 1000.0
    )

    return x, y


# ============================================================
# LOAD AMO327 RAW GPS ONLY
# ============================================================

schema_cols = (
    pq.ParquetFile(
        RAW_GPS
    )
    .schema
    .names
)

pid_col = first_present(
    schema_cols,
    [
        "participant_id",
        "clinic_id",
    ],
)

if pid_col is None:
    raise RuntimeError(
        "Could not identify participant column."
    )

df = pd.read_parquet(
    RAW_GPS,
    filters=[
        (
            pid_col,
            "==",
            PARTICIPANT,
        )
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

if any(
    c is None
    for c in [
        time_col,
        lat_col,
        lon_col,
        acc_col,
    ]
):

    raise RuntimeError(
        "Missing GPS columns."
    )

for c in [
    lat_col,
    lon_col,
    acc_col,
]:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )


# ============================================================
# TIME
# ============================================================

numeric_time = pd.to_numeric(
    df[time_col],
    errors="coerce",
)

if (
    numeric_time.notna().mean() > 0.95
    and numeric_time.median() > 1e11
):

    df["timestamp_ms"] = (
        numeric_time
    )

    df["datetime_utc"] = (
        pd.to_datetime(
            numeric_time,
            unit="ms",
            utc=True,
            errors="coerce",
        )
    )

else:

    df["datetime_utc"] = (
        pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce",
        )
    )

    df["timestamp_ms"] = (
        df["datetime_utc"]
        .astype("int64")
        // 1_000_000
    )

df["datetime_local"] = (
    df["datetime_utc"]
    .dt.tz_convert(TZ)
)

df = (
    df
    .dropna(
        subset=[
            "timestamp_ms",
            "datetime_local",
            lat_col,
            lon_col,
            acc_col,
        ]
    )
    .sort_values(
        "timestamp_ms"
    )
    .reset_index(
        drop=True
    )
)

reference_start = (
    df["datetime_local"]
    .dt.floor("D")
    .min()
)

reference_end = (
    reference_start
    + pd.Timedelta(
        days=REFERENCE_DAYS
    )
)

ref = df[
    (
        df["datetime_local"]
        >= reference_start
    )
    &
    (
        df["datetime_local"]
        < reference_end
    )
].copy()


# ============================================================
# HOME CENTER
#
# Only used as a privacy-preserving coordinate origin.
# Home is NOT part of the ROG calculation.
# ============================================================

anchors = pd.read_csv(
    HOME_ANCHORS,
    low_memory=False,
)

anchor = anchors[
    anchors[
        "participant_id"
    ]
    .astype(str)
    .eq(PARTICIPANT)
].copy()

home_lat_col = first_present(
    anchor.columns,
    [
        "home_latitude",
        "representative_latitude",
        "home_center_latitude",
    ],
)

home_lon_col = first_present(
    anchor.columns,
    [
        "home_longitude",
        "representative_longitude",
        "home_center_longitude",
    ],
)

home_lat = float(
    anchor.iloc[0][
        home_lat_col
    ]
)

home_lon = float(
    anchor.iloc[0][
        home_lon_col
    ]
)


# ============================================================
# PRODUCTION DAILY CORTEX VALUES
# ============================================================

prod = pd.read_parquet(
    DAILY_FEATURES,
    filters=[
        (
            "participant_id",
            "==",
            PARTICIPANT,
        )
    ],
)

prod["date"] = (
    pd.to_datetime(
        prod["date"]
    )
    .dt.date
)

reference_dates = [
    d.date()
    for d in pd.date_range(
        reference_start,
        periods=REFERENCE_DAYS,
        freq="D",
    )
]

prod = prod[
    prod["date"].isin(
        reference_dates
    )
].copy()

# Same participant-date can occur in multiple clinical windows.
prod = (
    prod
    .sort_values(
        [
            "date",
            "clinical_window_id",
        ]
    )
    .drop_duplicates(
        subset=[
            "participant_id",
            "date",
        ],
        keep="first",
    )
)

daily = pd.DataFrame(
    {
        "date":
            reference_dates
    }
)

daily = daily.merge(
    prod[
        [
            "date",
            "has_raw_gps",
            "n_raw_rows",
            "n_supported_edges",
            "supported_fraction_of_day",
            "supported_radius_of_gyration_m",
            "feature_day_usable",
        ]
    ],
    on="date",
    how="left",
)

daily[
    "radius_of_gyration_km"
] = (
    pd.to_numeric(
        daily[
            "supported_radius_of_gyration_m"
        ],
        errors="coerce",
    )
    / 1000.0
)

daily[
    "gps_coverage_fraction"
] = pd.to_numeric(
    daily[
        "supported_fraction_of_day"
    ],
    errors="coerce",
)

daily[
    "feature_day_usable"
] = (
    daily[
        "feature_day_usable"
    ]
    .fillna(False)
    .astype(bool)
)

daily.to_csv(
    CSV_OUT,
    index=False,
)


# ============================================================
# REPRESENTATIVE DAY
#
# Choose valid day nearest median ROG.
# ============================================================

valid = daily[
    daily[
        "radius_of_gyration_km"
    ].notna()
].copy()

if (
    "feature_day_usable"
    in valid.columns
    and valid[
        "feature_day_usable"
    ].any()
):

    example_pool = valid[
        valid[
            "feature_day_usable"
        ]
    ].copy()

else:

    example_pool = valid.copy()

median_rog = (
    example_pool[
        "radius_of_gyration_km"
    ]
    .median()
)

example_idx = (
    (
        example_pool[
            "radius_of_gyration_km"
        ]
        - median_rog
    )
    .abs()
    .idxmin()
)

example_row = daily.loc[
    example_idx
]

example_date = (
    example_row[
        "date"
    ]
)

example_rog_km = float(
    example_row[
        "radius_of_gyration_km"
    ]
)


# ============================================================
# BUILD EXACT CORTEX TRAJECTORY FOR EXAMPLE DAY
# ============================================================

day = pd.Timestamp(
    example_date,
    tz=TZ,
)

next_day = (
    day
    + pd.Timedelta(
        days=1
    )
)

day_raw = ref[
    (
        ref[
            "datetime_local"
        ]
        >= day
    )
    &
    (
        ref[
            "datetime_local"
        ]
        < next_day
    )
].copy()

start_ms = int(
    day.tz_convert(
        "UTC"
    ).timestamp()
    * 1000
)

end_ms = int(
    next_day.tz_convert(
        "UTC"
    ).timestamp()
    * 1000
)

gps_input = pd.DataFrame(
    {
        "timestamp":
            day_raw[
                "timestamp_ms"
            ].to_numpy(),

        "latitude":
            day_raw[
                lat_col
            ].to_numpy(),

        "longitude":
            day_raw[
                lon_col
            ].to_numpy(),

        "accuracy":
            day_raw[
                acc_col
            ].to_numpy(),
    }
)

intervals = build_gps_intervals(
    gps_input,
    window_start_ms=start_ms,
    window_end_ms=end_ms,
    max_gap_ms=MAX_GAP_MS,
)

supported = (
    intervals[
        intervals[
            "supported"
        ]
    ]
    .copy()
)

spatial = (
    add_gps_spatial_semantics(
        supported
    )
)

trajectory = (
    add_gps_trajectory_plausibility(
        spatial,
        max_plausible_speed_kmh=
            MAX_SPEED_KMH,
    )
)

# Verify our production feature result.
check = (
    compute_daily_mobility_features(
        trajectory
    )
)

check_rog_km = (
    check[
        "supported_radius_of_gyration_m"
    ]
    / 1000.0
)

print(
    "Production daily ROG:",
    example_rog_km,
)

print(
    "Recomputed ROG:",
    check_rog_km,
)


# ============================================================
# ROG MIDPOINT GEOMETRY
#
# Cortex uses spatially supported edge midpoints.
# Implausible-speed transitions do not contribute.
# ============================================================

required = [
    "duration_ms",
    "start_latitude",
    "start_longitude",
    "end_latitude",
    "end_longitude",
]

for c in required:

    trajectory[c] = (
        pd.to_numeric(
            trajectory[c],
            errors="coerce",
        )
    )

valid_spatial = (
    trajectory[
        required
    ]
    .notna()
    .all(
        axis=1
    )
    &
    trajectory[
        "duration_ms"
    ].gt(0)
    &
    ~trajectory[
        "trajectory_status"
    ]
    .astype(str)
    .eq(
        "implausible_speed"
    )
)

z = trajectory.loc[
    valid_spatial,
    required,
].copy()

z[
    "mid_latitude"
] = (
    z[
        "start_latitude"
    ]
    +
    z[
        "end_latitude"
    ]
) / 2.0

z[
    "mid_longitude"
] = (
    z[
        "start_longitude"
    ]
    +
    z[
        "end_longitude"
    ]
) / 2.0

weights = (
    z[
        "duration_ms"
    ]
    .to_numpy(
        dtype=float
    )
)

center_lat = float(
    np.average(
        z[
            "mid_latitude"
        ],
        weights=weights,
    )
)

center_lon = float(
    np.average(
        z[
            "mid_longitude"
        ],
        weights=weights,
    )
)

mid_x, mid_y = (
    latlon_to_local_km(
        z[
            "mid_latitude"
        ],
        z[
            "mid_longitude"
        ],
        home_lat,
        home_lon,
    )
)

center_x, center_y = (
    latlon_to_local_km(
        [center_lat],
        [center_lon],
        home_lat,
        home_lon,
    )
)

center_x = float(
    center_x[0]
)

center_y = float(
    center_y[0]
)

z[
    "x_km"
] = mid_x

z[
    "y_km"
] = mid_y

z[
    "duration_min"
] = (
    z[
        "duration_ms"
    ]
    / 60000.0
)


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
        15.5,
        6.2,
    ),
)


# ============================================================
# PANEL A
# ============================================================

ax = axes[0]

# Duration-weighted spatial density.
z["duration_share_pct"] = (
    z["duration_ms"]
    / z["duration_ms"].sum()
    * 100.0
)

hb = ax.hexbin(
    z["x_km"],
    z["y_km"],
    C=z["duration_share_pct"],
    reduce_C_function=np.sum,
    gridsize=34,
    mincnt=1,
    linewidths=0,
)

cb = fig.colorbar(
    hb,
    ax=ax,
    fraction=0.045,
    pad=0.02,
)

cb.set_label(
    "% of supported observed time"
)

# Weighted activity center.
ax.scatter(
    center_x,
    center_y,
    marker="X",
    s=145,
    color="black",
    zorder=10,
)

# Subtle ROG reference circle.
ax.text(
    0.03,
    0.97,
    (
        f"ROG = {example_rog_km:.2f} km\n"
        "Duration-weighted RMS distance\n"
        "from activity center"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="none",
        alpha=0.88,
    ),
    zorder=20,
)

ax.annotate(
    "Duration-weighted\nactivity center",
    xy=(
        center_x,
        center_y,
    ),
    xytext=(
        -78,
        -42,
    ),
    textcoords="offset points",
    ha="right",
    va="top",
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
        "shrinkA": 2,
        "shrinkB": 4,
    },
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="none",
        alpha=0.88,
    ),
    zorder=20,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.autoscale()

ax.set_xlabel(
    "East–west position relative to home (km)"
)

ax.set_ylabel(
    "North–south position relative to home (km)"
)

ax.set_title(
    (
        "A. Where supported time was spent\n"
        f"{pd.Timestamp(example_date).strftime('%b %d')}"
    )
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# ============================================================
# PANEL B
# ============================================================

ax = axes[1]

x = np.arange(
    len(daily)
)

values = (
    daily[
        "radius_of_gyration_km"
    ]
    .to_numpy(
        dtype=float
    )
)

example_pos = int(
    daily.index[
        daily[
            "date"
        ].eq(
            example_date
        )
    ][0]
)

# Gray bars; highlight only example day.
for i, value in enumerate(
    values
):

    if not np.isfinite(
        value
    ):
        continue

    if i == example_pos:

        ax.bar(
            i,
            value,
            width=0.72,
            color="0.25",
            zorder=3,
        )

    else:

        ax.bar(
            i,
            value,
            width=0.72,
            color="0.72",
            zorder=2,
        )

# Make very small values readable numerically.
for i, value in enumerate(
    values
):

    if (
        np.isfinite(value)
        and value > 0
        and value < 0.5
    ):

        ax.text(
            i,
            value + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

# Example-day value.
ax.text(
    example_pos,
    values[
        example_pos
    ]
    + 0.35,
    f"{values[example_pos]:.2f}",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
)

labels = [
    pd.Timestamp(d)
    .strftime(
        "%b %d"
    )
    for d in daily[
        "date"
    ]
]

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right",
)

ax.set_ylabel(
    "Radius of gyration (km)"
)

ax.set_title(
    "B. Daily Cortex-2.0 radius of gyration"
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# ============================================================
# COVERAGE STRIP
#
# Separate inset below Panel B so coverage never overlaps
# the x-axis labels.
# ============================================================

pos = ax.get_position()

coverage_ax = fig.add_axes(
    [
        pos.x0,
        pos.y0 - 0.075,
        pos.width,
        0.028,
    ]
)

coverage_ax.set_xlim(
    -0.5,
    len(daily) - 0.5,
)

coverage_ax.set_ylim(
    0,
    1,
)

for i, row in daily.iterrows():

    cov = row[
        "gps_coverage_fraction"
    ]

    if not np.isfinite(cov):
        continue

    shade = (
        0.92
        - 0.72
        * float(cov)
    )

    shade = min(
        max(
            shade,
            0.18,
        ),
        0.92,
    )

    coverage_ax.add_patch(
        Rectangle(
            (
                i - 0.36,
                0.15,
            ),
            0.72,
            0.70,
            facecolor=str(shade),
            edgecolor="none",
        )
    )

coverage_ax.set_xticks([])
coverage_ax.set_yticks([])

for spine in coverage_ax.spines.values():
    spine.set_visible(False)

coverage_ax.text(
    -0.02,
    0.5,
    "GPS coverage",
    transform=coverage_ax.transAxes,
    ha="right",
    va="center",
    fontsize=8,
)

coverage_ax.text(
    0.0,
    -0.55,
    "lower coverage",
    transform=coverage_ax.transAxes,
    ha="left",
    va="top",
    fontsize=7.5,
)

coverage_ax.text(
    1.0,
    -0.55,
    "higher coverage",
    transform=coverage_ax.transAxes,
    ha="right",
    va="top",
    fontsize=7.5,
)

# ============================================================
# PANEL B SUMMARY
# ============================================================

median_all = (
    daily[
        "radius_of_gyration_km"
    ]
    .median()
)

n_usable = int(
    daily[
        "feature_day_usable"
    ]
    .sum()
)

ax.text(
    0.98,
    0.95,
    (
        "14-day window\n"
        f"{daily['radius_of_gyration_km'].notna().sum()} days with computed ROG\n"
        f"Median daily ROG = {median_all:.2f} km"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    (
        "AMO327: How Cortex-2.0 calculates radius of gyration\n"
        "Duration-weighted spatial spread of supported GPS "
        "edge midpoints around the activity center"
    ),
    fontsize=16,
)

fig.subplots_adjust(
    top=0.78,
    bottom=0.27,
    wspace=0.28,
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# CONSOLE
# ============================================================

print()
print("=" * 100)
print("AMO327 RADIUS OF GYRATION FIGURE")
print("=" * 100)

print(
    "\nReference window:",
    reference_start,
    "to",
    reference_end,
)

print(
    "\nRepresentative day:",
    example_date,
)

print(
    "Production ROG:",
    f"{example_rog_km:.3f} km",
)

print(
    "Recomputed ROG:",
    f"{check_rog_km:.3f} km",
)

print(
    "Production-QC usable days:",
    n_usable,
    "/",
    REFERENCE_DAYS,
)

print(
    "\nDaily summary:"
)

print(
    daily[
        [
            "date",
            "radius_of_gyration_km",
            "gps_coverage_fraction",
            "feature_day_usable",
        ]
    ]
    .to_string(
        index=False
    )
)

print()
print("Saved:")
print(PNG_OUT)
print(PDF_OUT)
print(CSV_OUT)
