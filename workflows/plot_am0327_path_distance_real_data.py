from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

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

DAILY_FEATURES = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_clinical_weekly_daily_features.parquet"
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUT_DIR / "AMO327_path_distance_real_data.png"
PDF_OUT = OUT_DIR / "AMO327_path_distance_real_data.pdf"
CSV_OUT = OUT_DIR / "AMO327_path_distance_daily_summary.csv"

MAX_GAP_MS = 60_000
MAX_SPEED_KMH = 1000.0

# ============================================================
# HELPERS
# ============================================================

def first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * r * np.arcsin(np.sqrt(a))


def latlon_to_local_km(lat, lon, lat0, lon0):
    r = 6371000.0

    x_m = (
        np.radians(lon - lon0)
        * r
        * np.cos(np.radians(lat0))
    )

    y_m = (
        np.radians(lat - lat0)
        * r
    )

    return x_m / 1000.0, y_m / 1000.0


# ============================================================
# LOAD GPS
# ============================================================

# Read schema first, then use Parquet predicate pushdown so we do
# not load the full ~400M-row GPS dataset for one participant.
import pyarrow.parquet as pq

schema_cols = pq.ParquetFile(RAW_GPS).schema.names

pid_col = first_present(
    schema_cols,
    ["participant_id", "clinic_id"],
)

if pid_col is None:
    raise RuntimeError(
        "Could not identify participant column."
    )

df = pd.read_parquet(
    RAW_GPS,
    filters=[
        (pid_col, "==", PARTICIPANT)
    ],
)


time_col = first_present(
    df.columns,
    ["timestamp", "timestamp_ms", "time", "datetime"],
)

lat_col = first_present(
    df.columns,
    ["latitude", "lat"],
)

lon_col = first_present(
    df.columns,
    ["longitude", "lon", "lng"],
)

acc_col = first_present(
    df.columns,
    [
        "accuracy",
        "accuracy_m",
        "horizontal_accuracy",
    ],
)

required = {
    "participant": pid_col,
    "timestamp": time_col,
    "latitude": lat_col,
    "longitude": lon_col,
    "accuracy": acc_col,
}

missing = [
    k
    for k, v in required.items()
    if v is None
]

if missing:
    raise RuntimeError(
        f"Could not identify columns: {missing}"
    )

df = df[
    df[pid_col].astype(str).eq(PARTICIPANT)
].copy()

if df.empty:
    raise RuntimeError(
        f"No GPS rows found for {PARTICIPANT}"
    )

# ============================================================
# PARSE TIMESTAMP
# ============================================================

numeric_time = pd.to_numeric(
    df[time_col],
    errors="coerce",
)

if (
    numeric_time.notna().mean() > 0.95
    and numeric_time.median() > 1e11
):
    df["datetime_utc"] = pd.to_datetime(
        numeric_time,
        unit="ms",
        utc=True,
        errors="coerce",
    )
else:
    df["datetime_utc"] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce",
    )

df["datetime_local"] = (
    df["datetime_utc"]
    .dt.tz_convert(TZ)
)

for c in [lat_col, lon_col, acc_col]:
    df[c] = pd.to_numeric(
        df[c],
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
).reset_index(drop=True)

# ============================================================
# SAME 14-DAY HOME-REFERENCE WINDOW
#
# Home workflow used the first 14 local calendar days
# to infer the home location.
# ============================================================

reference_start = (
    df["datetime_local"]
    .dt.floor("D")
    .min()
)

reference_end = (
    reference_start
    + pd.Timedelta(days=REFERENCE_DAYS)
)

ref = df[
    (df["datetime_local"] >= reference_start)
    & (df["datetime_local"] < reference_end)
].copy()

if ref.empty:
    raise RuntimeError(
        "No GPS points in reference period."
    )

ref["date"] = (
    ref["datetime_local"]
    .dt.floor("D")
)

# ============================================================
# HOME CENTER
# ============================================================

anchors = pd.read_csv(
    HOME_ANCHORS,
    low_memory=False,
)

anchor = anchors[
    anchors["participant_id"]
    .astype(str)
    .eq(PARTICIPANT)
].copy()

if anchor.empty:
    raise RuntimeError(
        f"No home anchor for {PARTICIPANT}"
    )

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

if (
    home_lat_col is None
    or home_lon_col is None
):
    raise RuntimeError(
        "Could not find home-center coordinates "
        "in participant_home_anchors.csv"
    )

home_lat = float(
    pd.to_numeric(
        anchor.iloc[0][home_lat_col],
        errors="raise",
    )
)

home_lon = float(
    pd.to_numeric(
        anchor.iloc[0][home_lon_col],
        errors="raise",
    )
)

# ============================================================
# BUILD EXACT PRODUCTION CORTEX-2.0 TRAJECTORY EDGES
#
# raw GPS
# -> temporal intervals
# -> supported intervals
# -> spatial semantics
# -> trajectory plausibility
# ============================================================

def locate_function(function_name):

    for py in Path("src").rglob("*.py"):

        txt = py.read_text(
            errors="ignore"
        )

        if f"def {function_name}" in txt:

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

            return (
                getattr(
                    module,
                    function_name,
                ),
                module_name,
            )

    raise RuntimeError(
        f"Could not locate {function_name}"
    )


build_gps_intervals, intervals_module = (
    locate_function(
        "build_gps_intervals"
    )
)

add_gps_spatial_semantics, spatial_module = (
    locate_function(
        "add_gps_spatial_semantics"
    )
)

add_gps_trajectory_plausibility, trajectory_module = (
    locate_function(
        "add_gps_trajectory_plausibility"
    )
)

print(
    "Production interval builder:",
    intervals_module,
)

print(
    "Production spatial semantics:",
    spatial_module,
)

print(
    "Production trajectory plausibility:",
    trajectory_module,
)

# Cortex interval builder expects epoch milliseconds.
ref["timestamp_ms"] = (
    ref["datetime_local"]
    .dt.tz_convert("UTC")
    .astype("int64")
    // 1_000_000
)

trajectory_frames = []

for day in pd.date_range(
    start=reference_start,
    periods=REFERENCE_DAYS,
    freq="D",
    tz=TZ,
):

    next_day = (
        day
        + pd.Timedelta(days=1)
    )

    day_raw = ref[
        (
            ref["datetime_local"] >= day
        )
        &
        (
            ref["datetime_local"] < next_day
        )
    ].copy()

    if day_raw.empty:
        continue

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

    if supported.empty:
        continue

    spatial = (
        add_gps_spatial_semantics(
            supported
        )
    )

    trajectory = (
        add_gps_trajectory_plausibility(
            spatial,
            max_plausible_speed_kmh=MAX_SPEED_KMH,
        )
    )

    trajectory["date"] = day

    trajectory_frames.append(
        trajectory
    )

if not trajectory_frames:
    raise RuntimeError(
        "No production trajectory edges were created."
    )

traj = pd.concat(
    trajectory_frames,
    ignore_index=True,
)

# Production displacement counts only spatially meaningful,
# plausible trajectory edges.
if "trajectory_status" not in traj.columns:
    raise RuntimeError(
        "Production trajectory table lacks trajectory_status."
    )

# Inspect actual statuses so figure labels come from production data.
print(
    "\nProduction trajectory statuses:"
)

print(
    traj[
        "trajectory_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

# Mark edges that contribute to displacement.
#
# compute_daily_mobility_features requires:
# supported = True
# valid spatial endpoints
# trajectory plausibility
# distance beyond endpoint accuracy uncertainty.
#
# add_gps_spatial_semantics already supplies distance_m
# and the spatial classification used by production.
# Exact Cortex-2.0 displacement definition:
# only plausible_displacement edges contribute to distance.
traj["counted"] = (
    traj["trajectory_status"]
    .astype(str)
    .eq("plausible_displacement")
)

# ============================================================
# LOCAL HOME-CENTERED COORDINATES
# ============================================================

x, y = latlon_to_local_km(
    traj[
        "start_latitude"
    ].to_numpy(),
    traj[
        "start_longitude"
    ].to_numpy(),
    home_lat,
    home_lon,
)

traj["x_km"] = x
traj["y_km"] = y

next_x, next_y = latlon_to_local_km(
    traj[
        "end_latitude"
    ].to_numpy(),
    traj[
        "end_longitude"
    ].to_numpy(),
    home_lat,
    home_lon,
)

traj["next_x_km"] = next_x
traj["next_y_km"] = next_y

# Raw GPS points are still shown faintly in Panel A as geographic context.
raw_x, raw_y = latlon_to_local_km(
    ref[lat_col].to_numpy(),
    ref[lon_col].to_numpy(),
    home_lat,
    home_lon,
)

ref["x_km"] = raw_x
ref["y_km"] = raw_y

# ============================================================
# DAILY PRODUCTION CORTEX-2.0 FEATURES
#
# Use the same daily mobility values and supported-coverage
# definition as the clinical feature pipeline.
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

prod["date"] = pd.to_datetime(
    prod["date"]
).dt.date

reference_dates = [
    d.date()
    for d in pd.date_range(
        start=reference_start,
        periods=REFERENCE_DAYS,
        freq="D",
    )
]

prod = prod[
    prod["date"].isin(
        reference_dates
    )
].copy()

# A calendar day can appear in more than one clinical window.
# Daily mobility features are participant-day quantities, so
# retain one copy of each participant-date.
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
        "date": reference_dates
    }
)

daily = daily.merge(
    prod[
        [
            "date",
            "has_raw_gps",
            "n_raw_rows",
            "n_supported_edges",
            "supported_duration_ms",
            "supported_fraction_of_day",
            "supported_displacement_distance_m",
            "feature_day_usable",
        ]
    ],
    on="date",
    how="left",
)

daily["path_distance_km"] = (
    pd.to_numeric(
        daily[
            "supported_displacement_distance_m"
        ],
        errors="coerce",
    )
    / 1000.0
)

# IMPORTANT:
# Coverage now means Cortex-supported fraction of the day,
# not the earlier custom fraction of 10-minute bins.
daily["gps_coverage_fraction"] = (
    pd.to_numeric(
        daily[
            "supported_fraction_of_day"
        ],
        errors="coerce",
    )
)

daily["gps_points"] = (
    pd.to_numeric(
        daily["n_raw_rows"],
        errors="coerce",
    )
    .fillna(0)
)

daily["has_gps"] = (
    daily[
        "has_raw_gps"
    ]
    .fillna(False)
    .astype(bool)
)

daily.to_csv(
    CSV_OUT,
    index=False,
)

n_days_with_gps = int(
    daily[
        "has_gps"
    ].sum()
)

total_km = float(
    daily[
        "path_distance_km"
    ]
    .sum(
        skipna=True
    )
)

# ============================================================
# DAY INDEX / COLOR
# ============================================================

traj["day_index"] = (
    (
        pd.to_datetime(
            traj["date"]
        )
        - reference_start
    )
    .dt.days
)

# Raw GPS points are still displayed in Panel A for geographic
# context, so give them the same 0-13 day index used for colors.
ref["day_index"] = (
    (
        ref["date"]
        - reference_start
    )
    .dt.days
)

# Use matplotlib's default categorical palette.
cycle = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]

day_colors = {
    i: cycle[i % len(cycle)]
    for i in range(REFERENCE_DAYS)
}

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6.2),
)

# ------------------------------------------------------------
# PANEL A
# ------------------------------------------------------------

ax = axes[0]

# Very light raw points colored by day
for day_idx in range(REFERENCE_DAYS):
    g = ref[
        ref["day_index"].eq(day_idx)
    ]

    if g.empty:
        continue

    ax.scatter(
        g["x_km"],
        g["y_km"],
        s=7,
        alpha=0.25,
        color=day_colors[day_idx],
        zorder=1,
    )

# Excluded edges in grey
excluded = traj[
    ~traj["counted"]
].copy()

excluded_segments = np.stack(
    [
        excluded[["x_km", "y_km"]].to_numpy(),
        excluded[["next_x_km", "next_y_km"]].to_numpy(),
    ],
    axis=1,
)

if len(excluded_segments):
    excluded_lc = LineCollection(
        excluded_segments,
        linewidths=0.7,
        alpha=0.20,
        colors="0.55",
        zorder=2,
    )
    ax.add_collection(excluded_lc)

# Counted edges colored by day
counted = traj[
    traj["counted"]
].copy()

counted_segments = np.stack(
    [
        counted[["x_km", "y_km"]].to_numpy(),
        counted[["next_x_km", "next_y_km"]].to_numpy(),
    ],
    axis=1,
)

counted_colors = [
    day_colors[int(i)]
    for i in counted["day_index"].to_numpy()
]

if len(counted_segments):
    counted_lc = LineCollection(
        counted_segments,
        linewidths=1.15,
        alpha=0.75,
        colors=counted_colors,
        zorder=3,
    )
    ax.add_collection(counted_lc)

ax.autoscale()

ax.scatter(
    0,
    0,
    marker="*",
    s=180,
    color="black",
    zorder=10,
)

ax.set_title(
    "A. GPS edges contributing to path distance"
)

ax.set_xlabel(
    "East–west position relative to home (km)"
)

ax.set_ylabel(
    "North–south position relative to home (km)"
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_elements = [
    Line2D(
        [0],
        [0],
        color="0.25",
        lw=1.5,
        label="Counted edges (colored by day)",
    ),
    Line2D(
        [0],
        [0],
        color="0.65",
        lw=1.0,
        alpha=0.5,
        label="Excluded edges",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="black",
        linestyle="None",
        markersize=11,
        label="Inferred home",
    ),
]

ax.legend(
    handles=legend_elements,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.0, -0.12),
    ncol=3,
)

# ------------------------------------------------------------
# PANEL B
# ------------------------------------------------------------

ax = axes[1]

xpos = np.arange(
    REFERENCE_DAYS
)

bar_colors = [
    day_colors[i]
    for i in range(
        REFERENCE_DAYS
    )
]

# Opacity reflects GPS coverage.
# Give data-containing days a visible minimum opacity.
bar_alpha = (
    0.25
    + 0.75
    * daily[
        "gps_coverage_fraction"
    ].clip(
        0,
        1,
    )
)

for i, row in daily.iterrows():

    if row["has_gps"]:

        ax.bar(
            i,
            row["path_distance_km"],
            width=0.72,
            color=bar_colors[i],
            alpha=float(
                bar_alpha.iloc[i]
            ),
        )

    else:

        # Explicit no-GPS marker.
        ax.scatter(
            i,
            0,
            marker="x",
            s=55,
            color="black",
            zorder=5,
        )

        ax.text(
            i,
            0,
            "no GPS",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_title(
    "B. Daily path distance and GPS coverage"
)

ax.set_ylabel(
    "Supported path distance (km/day)"
)

date_labels = [
    d.strftime("%b %d")
    for d in daily["date"]
]

ax.set_xticks(
    xpos
)

ax.set_xticklabels(
    date_labels,
    rotation=45,
    ha="right",
)

ax.spines["top"].set_visible(False)

# GPS coverage on secondary y-axis
ax2 = ax.twinx()

coverage_valid = daily[
    daily["has_gps"]
]

ax2.scatter(
    coverage_valid.index,
    coverage_valid[
        "gps_coverage_fraction"
    ] * 100.0,
    marker="o",
    facecolors="none",
    edgecolors="black",
    s=42,
    linewidth=1.1,
    zorder=6,
)

ax2.set_ylabel(
    "Supported GPS coverage (%)"
)

coverage_legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        linestyle="None",
        markersize=7,
        label="GPS coverage",
    )
]

ax2.legend(
    handles=coverage_legend,
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.0, 0.78),
)

ax2.set_ylim(
    0,
    105,
)

ax2.spines["top"].set_visible(False)

# Annotation moved away from peak
ax.text(
    0.97,
    0.97,
    (
        f"14-day window\n"
        f"{n_days_with_gps} days with GPS\n"
        f"Total path distance = {total_km:.1f} km"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
)

# ------------------------------------------------------------
# MAIN TITLE
# ------------------------------------------------------------

fig.suptitle(
    (
        "AMO327: How Cortex-2.0 calculates daily path distance\n"
        "Sum of spatially and temporally supported distances "
        "between consecutive GPS observations"
    ),
    fontsize=16,
)

# ------------------------------------------------------------
# DAY COLOR KEY
# ------------------------------------------------------------

from matplotlib.patches import Patch

day_handles = []

for i, d in enumerate(daily["date"]):

    label = pd.Timestamp(d).strftime("%b %d")

    if daily.loc[i, "has_gps"]:

        day_handles.append(
            Patch(
                facecolor=day_colors[i],
                edgecolor="none",
                label=label,
            )
        )

    else:

        day_handles.append(
            Patch(
                facecolor="0.85",
                edgecolor="0.5",
                hatch="//",
                label=label,
            )
        )

fig.legend(
    handles=day_handles,
    title="Day in 14-day window",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=7,
    frameon=False,
    fontsize=8,
    title_fontsize=9,
)

fig.tight_layout(
    rect=[
        0,
        0.10,
        1,
        0.91,
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

# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("=" * 100)
print("AMO327 PATH DISTANCE FIGURE")
print("=" * 100)

print(
    "\nReference window:",
    reference_start,
    "to",
    reference_end,
)

print(
    "Days with GPS:",
    n_days_with_gps,
    "/",
    REFERENCE_DAYS,
)

print(
    "Total supported path distance:",
    f"{total_km:.2f} km",
)

print("\nEdge status:")
print(
    traj["trajectory_status"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nDaily summary:")
print(
    daily[
        [
            "date",
            "path_distance_km",
            "gps_coverage_fraction",
            "gps_points",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved:")
print(PNG_OUT)
print(PDF_OUT)
print(CSV_OUT)
