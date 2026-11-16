from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

PID = "CJO550"

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
MIN_VALID_NIGHTS = 5
MIN_HOME_CONFIDENCE = 0.50

# A valid night requires >=2 h observed GPS during 00:00–06:00
MIN_VALID_NIGHT_MS = (
    2
    * 60
    * 60
    * 1000
)


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
    / "CJO550_home_convergence.csv"
)

PNG_OUT = (
    OUT_DIR
    / "CJO550_home_convergence.png"
)

PDF_OUT = (
    OUT_DIR
    / "CJO550_home_convergence.pdf"
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

        if (
            observed_night_ms
            < MIN_VALID_NIGHT_MS
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

    if n_valid < MIN_VALID_NIGHTS:

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
# LOAD CJO550 ONCE
# ============================================================

print("=" * 90)
print("CJO550 FAST HOME-CONVERGENCE ANALYSIS")
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
# SHORT WINDOWS
# ============================================================

results = []

for n_days in WINDOWS:

    print()
    print(
        f"Inferring home from first {n_days} days..."
    )

    result = run_window(
        n_days,
        f"{n_days} days",
    )

    results.append(
        result
    )

    print(
        "  status:",
        result[
            "home_status"
        ],
        "| valid nights:",
        result[
            "n_valid_nights"
        ],
        "| confidence:",
        result[
            "home_confidence"
        ],
    )


# ============================================================
# FULL STUDY REFERENCE
# ============================================================

print()
print(
    f"Inferring full-study reference home ({full_span_days} days)..."
)

full_result = run_window(
    full_span_days,
    "Full study",
)

results.append(
    full_result
)

if (
    full_result[
        "home_status"
    ]
    != "observed"
):

    raise RuntimeError(
        "Could not infer full-study reference home."
    )


# ============================================================
# DISTANCE TO FULL-STUDY HOME
# ============================================================

reference_lat = (
    full_result[
        "home_latitude"
    ]
)

reference_lon = (
    full_result[
        "home_longitude"
    ]
)

for result in results:

    if (
        result[
            "home_status"
        ]
        != "observed"
    ):

        result[
            "distance_to_full_home_m"
        ] = np.nan

        continue

    result[
        "distance_to_full_home_m"
    ] = haversine_m(
        result[
            "home_latitude"
        ],
        result[
            "home_longitude"
        ],
        reference_lat,
        reference_lon,
    )


summary = pd.DataFrame(
    results
)

summary = summary[
    [
        "calibration_days",
        "calibration_label",
        "home_status",
        "home_reason",
        "n_valid_nights",
        "n_nights_won",
        "home_confidence",
        "distance_to_full_home_m",
        "n_place_candidates",
        "home_place_candidate_id",
        "home_latitude",
        "home_longitude",
    ]
]

summary.to_csv(
    CSV_OUT,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 100)
print("CJO550 HOME-INFERENCE CONVERGENCE")
print("=" * 100)

print(
    summary[
        [
            "calibration_label",
            "home_status",
            "n_valid_nights",
            "n_nights_won",
            "home_confidence",
            "distance_to_full_home_m",
            "n_place_candidates",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# FIGURE
# ============================================================

plot_df = summary[
    summary[
        "calibration_label"
    ]
    != "Full study"
].copy()

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 10.5

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        14,
        4.3,
    ),
)


# ------------------------------------------------------------
# A — Distance to full-study reference
# ------------------------------------------------------------

ax = axes[0]

valid = plot_df[
    plot_df[
        "distance_to_full_home_m"
    ].notna()
]

ax.plot(
    valid[
        "calibration_days"
    ],
    valid[
        "distance_to_full_home_m"
    ],
    marker="o",
)

ax.set_xlabel(
    "Calibration period (days)"
)

ax.set_ylabel(
    "Distance to full-study home (m)"
)

ax.set_title(
    "A. Home-location convergence"
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# ------------------------------------------------------------
# B — Home confidence
# ------------------------------------------------------------

ax = axes[1]

confidence = plot_df[
    plot_df[
        "home_confidence"
    ].notna()
]

ax.plot(
    confidence[
        "calibration_days"
    ],
    confidence[
        "home_confidence"
    ],
    marker="o",
)

ax.axhline(
    MIN_HOME_CONFIDENCE,
    linestyle="--",
    linewidth=1,
    color="0.5",
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xlabel(
    "Calibration period (days)"
)

ax.set_ylabel(
    "Home confidence"
)

ax.set_title(
    "B. Night-vote confidence"
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# ------------------------------------------------------------
# C — Evidence accumulation
# ------------------------------------------------------------

ax = axes[2]

ax.plot(
    plot_df[
        "calibration_days"
    ],
    plot_df[
        "n_valid_nights"
    ],
    marker="o",
    label="Valid nights",
)

ax.plot(
    plot_df[
        "calibration_days"
    ],
    plot_df[
        "n_nights_won"
    ],
    marker="o",
    label="Nights won by home",
)

ax.axhline(
    MIN_VALID_NIGHTS,
    linestyle="--",
    linewidth=1,
    color="0.5",
)

ax.set_xlabel(
    "Calibration period (days)"
)

ax.set_ylabel(
    "Number of nights"
)

ax.set_title(
    "C. Accumulated nighttime evidence"
)

ax.legend(
    frameon=False,
    fontsize=8.5,
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


fig.suptitle(
    (
        "CJO550: minimum observation period for stable home inference\n"
        f"Full-study reference = {full_span_days} calendar days"
    ),
    fontsize=14,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.90,
    ]
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

print()
print("Saved:")
print(CSV_OUT)
print(PNG_OUT)
print(PDF_OUT)
