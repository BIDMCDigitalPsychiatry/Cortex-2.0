from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

PID = "AMO327"

TZ = "America/New_York"

WINDOWS = [
    3,
    5,
    7,
    10,
    14,
    21,
    30,
]

MAX_GAP_MS = 60_000

# Exact current home-time workflow value
COMPATIBILITY_MULTIPLIER = 1.0

# Existing production home rules
MIN_HOME_CONFIDENCE = 0.50

NIGHT_HOUR_THRESHOLDS = [
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
]

VALID_NIGHT_THRESHOLDS = [
    2,
    3,
    4,
    5,
    7,
]


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

RAW_PATH = (
    ROOT
    / "gps_clinical_weekly_raw_points.parquet"
)

OUT_DIR = (
    ROOT
    / "home_time"
    / "home_reference_stability"
    / PID
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_OUT = (
    OUT_DIR
    / "AMO327_home_convergence.csv"
)

PNG_OUT = (
    OUT_DIR
    / "AMO327_home_convergence.png"
)

PDF_OUT = (
    OUT_DIR
    / "AMO327_home_convergence.pdf"
)


# ============================================================
# FIND CORTEX FUNCTIONS
# ============================================================

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
        f"Could not locate Cortex function: {name}"
    )


build_gps_intervals = locate_function(
    "build_gps_intervals"
)

add_gps_spatial_semantics = locate_function(
    "add_gps_spatial_semantics"
)

build_anchor_compatible_location_episodes = (
    locate_function(
        "build_anchor_compatible_location_episodes"
    )
)

build_place_candidates = locate_function(
    "build_place_candidates"
)


# ============================================================
# HELPERS
# ============================================================

def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):

    R = 6371000.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dp = np.radians(
        lat2 - lat1
    )

    dl = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dp / 2.0) ** 2
        +
        np.cos(p1)
        * np.cos(p2)
        * np.sin(dl / 2.0) ** 2
    )

    return (
        2.0
        * R
        * np.arcsin(
            np.sqrt(a)
        )
    )


def infer_home(
    assignments,
    places,
    reference_dates,
    min_valid_nights,
    min_valid_night_hours,
):

    """
    Apply the production-style nighttime home voting rules
    to an already constructed place model.
    """

    valid_nights = []

    for local_date in reference_dates:

        night_start = pd.Timestamp(
            local_date,
            tz=TZ,
        )

        night_end = (
            night_start
            + pd.Timedelta(
                hours=6
            )
        )

        night_start_ms = int(
            night_start
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )

        night_end_ms = int(
            night_end
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )

        # Episodes that overlap this night's 00:00–06:00 interval.
        x = assignments[
            (
                assignments[
                    "episode_end_ms"
                ]
                > night_start_ms
            )
            &
            (
                assignments[
                    "episode_start_ms"
                ]
                < night_end_ms
            )
        ].copy()

        if x.empty:
            continue

        overlap_start = np.maximum(
            x[
                "episode_start_ms"
            ].to_numpy(
                dtype=float
            ),
            night_start_ms,
        )

        overlap_end = np.minimum(
            x[
                "episode_end_ms"
            ].to_numpy(
                dtype=float
            ),
            night_end_ms,
        )

        x["night_overlap_ms"] = np.maximum(
            0.0,
            overlap_end
            - overlap_start,
        )

        x = x[
            x[
                "night_overlap_ms"
            ]
            > 0
        ].copy()

        if x.empty:
            continue

        observed_night_ms = float(
            x[
                "night_overlap_ms"
            ].sum()
        )

        min_valid_night_ms = (
            min_valid_night_hours
            * 60
            * 60
            * 1000
        )

        if (
            observed_night_ms
            < min_valid_night_ms
        ):
            continue

        by_place = (
            x.groupby(
                "place_candidate_id",
                as_index=False,
            )[
                "night_overlap_ms"
            ]
            .sum()
        )

        by_place = (
            by_place
            .sort_values(
                [
                    "night_overlap_ms",
                    "place_candidate_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        winner = by_place.iloc[0]

        valid_nights.append(
            {
                "date":
                    str(local_date),

                "winner":
                    int(
                        winner[
                            "place_candidate_id"
                        ]
                    ),

                "observed_night_ms":
                    observed_night_ms,

                "winner_duration_ms":
                    float(
                        winner[
                            "night_overlap_ms"
                        ]
                    ),
            }
        )

    n_valid = len(
        valid_nights
    )

    if n_valid < min_valid_nights:

        return {
            "home_status":
                "not_inferred",

            "home_reason":
                "too_few_valid_nights",

            "n_valid_nights":
                n_valid,

            "n_nights_won":
                np.nan,

            "home_confidence":
                np.nan,

            "home_latitude":
                np.nan,

            "home_longitude":
                np.nan,

            "home_place_candidate_id":
                np.nan,
        }

    winners = pd.DataFrame(
        valid_nights
    )

    vote_counts = (
        winners.groupby(
            "winner"
        )
        .size()
        .reset_index(
            name="n_nights_won"
        )
        .sort_values(
            [
                "n_nights_won",
                "winner",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    home_id = int(
        vote_counts.iloc[0][
            "winner"
        ]
    )

    n_won = int(
        vote_counts.iloc[0][
            "n_nights_won"
        ]
    )

    confidence = (
        n_won
        / n_valid
    )

    if (
        confidence
        < MIN_HOME_CONFIDENCE
    ):

        return {
            "home_status":
                "not_inferred",

            "home_reason":
                "low_home_confidence",

            "n_valid_nights":
                n_valid,

            "n_nights_won":
                n_won,

            "home_confidence":
                confidence,

            "home_latitude":
                np.nan,

            "home_longitude":
                np.nan,

            "home_place_candidate_id":
                home_id,
        }

    place_row = places[
        places[
            "place_candidate_id"
        ].eq(
            home_id
        )
    ]

    if place_row.empty:

        raise RuntimeError(
            f"Home place {home_id} missing from places table."
        )

    place_row = place_row.iloc[0]

    lat_col = (
        "representative_latitude"
        if "representative_latitude"
        in place_row.index
        else "anchor_latitude"
    )

    lon_col = (
        "representative_longitude"
        if "representative_longitude"
        in place_row.index
        else "anchor_longitude"
    )

    return {
        "home_status":
            "observed",

        "home_reason":
            "accepted",

        "n_valid_nights":
            n_valid,

        "n_nights_won":
            n_won,

        "home_confidence":
            confidence,

        "home_latitude":
            float(
                place_row[
                    lat_col
                ]
            ),

        "home_longitude":
            float(
                place_row[
                    lon_col
                ]
            ),

        "home_place_candidate_id":
            home_id,
    }


# ============================================================
# LOAD AMO327 ONCE
# ============================================================

print("=" * 90)
print("AMO327 FAST HOME-CONVERGENCE ANALYSIS")
print("=" * 90)

table = pq.read_table(
    RAW_PATH,
    filters=[
        (
            "participant_id",
            "=",
            PID,
        )
    ],
)

raw = table.to_pandas()

if raw.empty:
    raise RuntimeError(
        f"No raw GPS found for {PID}"
    )

raw["date"] = (
    raw[
        "date"
    ]
    .astype(str)
)

raw["timestamp"] = pd.to_numeric(
    raw["timestamp"],
    errors="coerce",
)

raw["latitude"] = pd.to_numeric(
    raw["latitude"],
    errors="coerce",
)

raw["longitude"] = pd.to_numeric(
    raw["longitude"],
    errors="coerce",
)

raw["accuracy"] = pd.to_numeric(
    raw["accuracy"],
    errors="coerce",
)

raw = raw.dropna(
    subset=[
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
    ]
)

all_dates = sorted(
    pd.to_datetime(
        raw[
            "date"
        ].unique()
    )
)

first_date = min(
    all_dates
).normalize()

last_date = max(
    all_dates
).normalize()

full_span_days = int(
    (
        last_date
        - first_date
    ).days
    + 1
)

print(
    "First GPS date:",
    first_date.date(),
)

print(
    "Last GPS date:",
    last_date.date(),
)

print(
    "Full calendar span:",
    full_span_days,
    "days",
)

print(
    "Raw GPS rows:",
    len(raw),
)


# ============================================================
# BUILD DAILY MICROSEGMENTS ONCE
# ============================================================

print()
print(
    "Building daily Cortex GPS microsegments once..."
)

day_microsegments = {}

calendar_dates = pd.date_range(
    first_date,
    last_date,
    freq="D",
)

for i, local_date in enumerate(
    calendar_dates,
    start=1,
):

    date_string = (
        local_date
        .strftime(
            "%Y-%m-%d"
        )
    )

    day_raw = raw[
        raw[
            "date"
        ].eq(
            date_string
        )
    ].copy()

    if day_raw.empty:

        day_microsegments[
            date_string
        ] = pd.DataFrame()

        continue

    local_start = pd.Timestamp(
        date_string,
        tz=TZ,
    )

    local_end = (
        local_start
        + pd.Timedelta(
            days=1
        )
    )

    start_ms = int(
        local_start
        .tz_convert("UTC")
        .timestamp()
        * 1000
    )

    end_ms = int(
        local_end
        .tz_convert("UTC")
        .timestamp()
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

    supported = intervals[
        intervals[
            "supported"
        ]
    ].copy()

    if supported.empty:

        day_microsegments[
            date_string
        ] = pd.DataFrame()

        continue

    spatial = (
        add_gps_spatial_semantics(
            supported
        )
    )

    segments = (
        build_anchor_compatible_location_episodes(
            spatial
        )
    )

    if not segments.empty:

        segments = (
            segments.copy()
        )

        segments[
            "date"
        ] = date_string

    day_microsegments[
        date_string
    ] = segments

    print(
        f"\r  {i}/{len(calendar_dates)} calendar days",
        end="",
        flush=True,
    )

print()
print(
    "Daily microsegments complete."
)


# ============================================================
# RUN ONE CALIBRATION WINDOW
# ============================================================

def run_window(
    n_days,
    label,
    min_valid_nights,
    min_valid_night_hours,
):

    reference_dates = (
        pd.date_range(
            first_date,
            periods=n_days,
            freq="D",
        )
    )

    date_strings = [
        x.strftime(
            "%Y-%m-%d"
        )
        for x in reference_dates
    ]

    frames = []

    for date_string in date_strings:

        seg = day_microsegments.get(
            date_string,
            pd.DataFrame(),
        )

        if (
            seg is not None
            and not seg.empty
        ):
            frames.append(
                seg
            )

    if not frames:

        return {
            "calibration_days":
                n_days,

            "calibration_label":
                label,

            "home_status":
                "not_inferred",

            "home_reason":
                "no_location_segments",

            "n_valid_nights":
                0,

            "n_nights_won":
                np.nan,

            "home_confidence":
                np.nan,

            "home_latitude":
                np.nan,

            "home_longitude":
                np.nan,

            "home_place_candidate_id":
                np.nan,

            "n_place_candidates":
                0,
        }

    segments = pd.concat(
        frames,
        ignore_index=True,
    )

    assignments, places = (
        build_place_candidates(
            segments,
            compatibility_multiplier=(
                COMPATIBILITY_MULTIPLIER
            ),
        )
    )

    result = infer_home(
        assignments,
        places,
        reference_dates,
        min_valid_nights=min_valid_nights,
        min_valid_night_hours=min_valid_night_hours,
    )

    result[
        "calibration_days"
    ] = n_days

    result[
        "calibration_label"
    ] = label

    result[
        "n_place_candidates"
    ] = len(
        places
    )

    return result


# ============================================================
# REFERENCE HOME
#
# Use the previously validated current production-rule reference:
# full 48-day study, >=2 h/night, >=5 valid nights.
# This is a provisional reference for sensitivity analysis,
# not external ground truth.
# ============================================================

print()
print("Building provisional 48-day reference home...")

reference_result = run_window(
    full_span_days,
    "Full study",
    min_valid_nights=5,
    min_valid_night_hours=2.0,
)

if reference_result["home_status"] != "observed":
    raise RuntimeError(
        "Could not infer provisional full-study reference home."
    )

ref_lat = reference_result["home_latitude"]
ref_lon = reference_result["home_longitude"]


# ============================================================
# THRESHOLD SENSITIVITY GRID
# ============================================================

CALIBRATION_WINDOWS = [
    7,
    10,
    14,
    21,
    30,
    full_span_days,
]

rows = []

for n_days in CALIBRATION_WINDOWS:

    for night_hours in NIGHT_HOUR_THRESHOLDS:

        for min_nights in VALID_NIGHT_THRESHOLDS:

            print(
                f"\rDays={n_days:2d} | "
                f"night >= {night_hours:3.1f} h | "
                f"valid nights >= {min_nights}",
                end="",
                flush=True,
            )

            result = run_window(
                n_days,
                f"{n_days} days",
                min_valid_nights=min_nights,
                min_valid_night_hours=night_hours,
            )

            if result["home_status"] == "observed":

                distance_m = haversine_m(
                    result["home_latitude"],
                    result["home_longitude"],
                    ref_lat,
                    ref_lon,
                )

            else:
                distance_m = np.nan

            rows.append(
                {
                    "calibration_days":
                        n_days,

                    "min_night_observation_hours":
                        night_hours,

                    "min_valid_nights":
                        min_nights,

                    "home_status":
                        result["home_status"],

                    "home_reason":
                        result["home_reason"],

                    "observed_valid_nights":
                        result["n_valid_nights"],

                    "n_nights_won":
                        result["n_nights_won"],

                    "home_confidence":
                        result["home_confidence"],

                    "distance_to_reference_home_m":
                        distance_m,

                    "home_latitude":
                        result["home_latitude"],

                    "home_longitude":
                        result["home_longitude"],

                    "n_place_candidates":
                        result["n_place_candidates"],
                }
            )

print()

results = pd.DataFrame(rows)

CSV_OUT = (
    OUT_DIR
    / "AMO327_home_threshold_sensitivity.csv"
)

results.to_csv(
    CSV_OUT,
    index=False,
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 110)
print("AMO327 HOME-THRESHOLD SENSITIVITY")
print("=" * 110)

summary = (
    results
    .groupby(
        [
            "min_night_observation_hours",
            "min_valid_nights",
        ],
        as_index=False,
    )
    .agg(
        n_windows_tested=(
            "calibration_days",
            "count",
        ),
        n_home_inferred=(
            "home_status",
            lambda x: int(
                (x == "observed").sum()
            ),
        ),
        median_distance_m=(
            "distance_to_reference_home_m",
            "median",
        ),
        max_distance_m=(
            "distance_to_reference_home_m",
            "max",
        ),
    )
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# FIGURE 1 — DISTANCE BY CALIBRATION WINDOW
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10.5

fig, axes = plt.subplots(
    1,
    len(NIGHT_HOUR_THRESHOLDS),
    figsize=(18, 4.3),
    sharey=True,
)

for ax, night_hours in zip(
    axes,
    NIGHT_HOUR_THRESHOLDS,
):

    x = results[
        results[
            "min_night_observation_hours"
        ].eq(night_hours)
    ]

    for min_nights in VALID_NIGHT_THRESHOLDS:

        z = x[
            x[
                "min_valid_nights"
            ].eq(min_nights)
        ]

        ax.plot(
            z["calibration_days"],
            z["distance_to_reference_home_m"],
            marker="o",
            label=f">={min_nights} nights",
        )

    ax.set_title(
        f">= {night_hours:g} h/night"
    )

    ax.set_xlabel(
        "Calibration days"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(
    "Distance to 48-day reference home (m)"
)

axes[-1].legend(
    frameon=False,
    fontsize=8,
)

fig.suptitle(
    "AMO327: sensitivity of home inference to nighttime evidence thresholds",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.92]
)

FIG1 = (
    OUT_DIR
    / "AMO327_home_threshold_distance.png"
)

fig.savefig(
    FIG1,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# FIGURE 2 — WHETHER HOME CAN BE INFERRED AT EACH SETTING
# ============================================================

heat = results[
    results[
        "calibration_days"
    ].eq(14)
].copy()

heat["inferred"] = (
    heat["home_status"]
    .eq("observed")
    .astype(int)
)

matrix = heat.pivot(
    index="min_night_observation_hours",
    columns="min_valid_nights",
    values="inferred",
)

fig, ax = plt.subplots(
    figsize=(6.5, 4.5)
)

im = ax.imshow(
    matrix.to_numpy(),
    aspect="auto",
    vmin=0,
    vmax=1,
    cmap="Greys",
)

ax.set_xticks(
    np.arange(
        len(matrix.columns)
    )
)

ax.set_xticklabels(
    matrix.columns
)

ax.set_yticks(
    np.arange(
        len(matrix.index)
    )
)

ax.set_yticklabels(
    matrix.index
)

ax.set_xlabel(
    "Minimum valid nights"
)

ax.set_ylabel(
    "Minimum observed nighttime GPS (hours)"
)

ax.set_title(
    "AMO327: successful home inference at 14 days"
)

for i in range(
    matrix.shape[0]
):
    for j in range(
        matrix.shape[1]
    ):

        ax.text(
            j,
            i,
            (
                "Yes"
                if matrix.iloc[i, j] == 1
                else "No"
            ),
            ha="center",
            va="center",
        )

FIG2 = (
    OUT_DIR
    / "AMO327_home_threshold_14d_success.png"
)

fig.tight_layout()

fig.savefig(
    FIG2,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


print()
print("Reference home:")
print(
    "  48 days, >=2 h/night, >=5 valid nights"
)

print()
print("Saved:")
print(CSV_OUT)
print(FIG1)
print(FIG2)
