"""Build daily Cortex 2.0 GPS features aligned to daily EMA.

One row = one participant × EMA day.

Preserves all EMA days, including days without raw GPS.

GPS semantics match the validated weekly clinical pipeline:
- UTC day windows
- max supported temporal gap = 60 seconds
- accuracy-aware spatial semantics
- trajectory plausibility <= 1000 km/h
- fixed-anchor location episodes
- place compatibility multiplier = 2.0
- time-weighted location features
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cortex2_engine.features.gps_observability import (
    compute_gps_temporal_observability,
)
from cortex2_engine.features.gps_intervals import (
    build_gps_intervals,
)
from cortex2_engine.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)
from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)
from cortex2_engine.features.gps_place_candidates import (
    build_place_candidates,
)
from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
)


# ============================================================
# CONFIG
# ============================================================

DC = Path("/shared/Digital Clinic")

ROOT = (
    DC
    / "cortex2_outputs/digital_clinic/"
      "clinical/daily_ema"
)

EMA_PATH = (
    ROOT
    / "daily_ema_gps_participants.csv"
)

RAW_PATH = (
    ROOT
    / "daily_ema_gps_raw_points.parquet"
)

RAW_COVERAGE_PATH = (
    ROOT
    / "daily_ema_gps_raw_coverage.csv"
)

OUT_PATH = (
    ROOT
    / "daily_ema_gps_features.csv"
)

PARQUET_OUT = (
    ROOT
    / "daily_ema_gps_features.parquet"
)

MAX_GAP_MS = 60_000

COMPATIBILITY_MULTIPLIER = 2.0

MAX_PLAUSIBLE_SPEED_KMH = 1000.0


# ============================================================
# EMA TABLE
# ============================================================

ema = pd.read_csv(
    EMA_PATH,
    low_memory=False,
)

ema["date"] = (
    pd.to_datetime(
        ema["date"],
        errors="coerce",
    )
    .dt.strftime("%Y-%m-%d")
)

ema = (
    ema.rename(
        columns={
            "cortex_participant_id":
                "participant_id"
        }
    )
)

keep_ema_cols = [
    "participant_id",
    "date",
    "anxiety",
    "mood",
    "daily_function",
]

missing = [
    c for c in keep_ema_cols
    if c not in ema.columns
]

if missing:
    raise RuntimeError(
        f"EMA table missing columns: {missing}"
    )

ema = (
    ema[
        keep_ema_cols
    ]
    .drop_duplicates(
        subset=[
            "participant_id",
            "date",
        ]
    )
    .sort_values(
        [
            "participant_id",
            "date",
        ]
    )
    .reset_index(
        drop=True
    )
)


coverage = pd.read_csv(
    RAW_COVERAGE_PATH,
    low_memory=False,
)

coverage["date"] = (
    pd.to_datetime(
        coverage["date"],
        errors="coerce",
    )
    .dt.strftime("%Y-%m-%d")
)


# ============================================================
# RAW PARQUET
# ============================================================

pf = pq.ParquetFile(
    RAW_PATH
)

print("=" * 100)
print("CORTEX 2.0 DAILY EMA GPS FEATURES")
print("=" * 100)

print(
    "EMA participant-days:",
    len(ema),
)

print(
    "Participants:",
    ema[
        "participant_id"
    ].nunique(),
)

print(
    "Raw GPS parquet:",
    RAW_PATH,
)

print(
    "Raw parquet rows:",
    pf.metadata.num_rows,
)


# ============================================================
# LOAD RAW GPS ONCE
# ============================================================

print()
print("Loading cached GPS columns...")

raw = pd.read_parquet(
    RAW_PATH,
    columns=[
        "participant_id",
        "date",
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
        "day_start_ms",
        "day_end_ms",
    ],
)

raw["participant_id"] = (
    raw[
        "participant_id"
    ].astype(str)
)

raw["date"] = (
    raw[
        "date"
    ].astype(str)
)

print(
    "Loaded raw rows:",
    f"{len(raw):,}",
)


# ============================================================
# DAILY FEATURE EXTRACTION
# ============================================================

feature_rows = []

n_days = len(ema)


for i, ema_row in ema.iterrows():

    participant_id = str(
        ema_row[
            "participant_id"
        ]
    )

    date = str(
        ema_row[
            "date"
        ]
    )

    if (
        i == 0
        or (i + 1) % 25 == 0
        or i + 1 == n_days
    ):

        print(
            f"{i + 1:4d}/{n_days} | "
            f"{participant_id} | {date}",
            flush=True,
        )


    day_raw = raw[
        (
            raw[
                "participant_id"
            ].eq(
                participant_id
            )
        )
        & (
            raw[
                "date"
            ].eq(
                date
            )
        )
    ].copy()


    row = {
        "participant_id":
            participant_id,

        "date":
            date,

        "anxiety":
            ema_row[
                "anxiety"
            ],

        "mood":
            ema_row[
                "mood"
            ],

        "daily_function":
            ema_row[
                "daily_function"
            ],

        "n_raw_gps_points":
            len(day_raw),

        "has_raw_gps":
            bool(
                len(day_raw)
            ),
    }


    # ========================================================
    # NO RAW GPS
    # ========================================================

    if day_raw.empty:

        row.update(
            {
                "gps_feature_status":
                    "no_raw_gps",
            }
        )

        feature_rows.append(
            row
        )

        continue


    start_ms = int(
        day_raw[
            "day_start_ms"
        ].iloc[0]
    )

    end_ms = int(
        day_raw[
            "day_end_ms"
        ].iloc[0]
    )


    # ========================================================
    # TEMPORAL OBSERVABILITY
    # ========================================================

    observability = (
        compute_gps_temporal_observability(
            day_raw[
                "timestamp"
            ].tolist(),
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            max_gap_ms=MAX_GAP_MS,
        )
    )

    # Prefix observability fields so names remain explicit.
    for key, value in (
        observability.items()
    ):

        row[
            f"gps_observability_{key}"
        ] = value


    # ========================================================
    # INTERVALS
    # ========================================================

    try:

        intervals = (
            build_gps_intervals(
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
        )

    except Exception as exc:

        row[
            "gps_feature_status"
        ] = (
            "interval_error"
        )

        row[
            "gps_feature_error"
        ] = repr(exc)

        feature_rows.append(
            row
        )

        continue


    row[
        "n_gps_intervals"
    ] = len(intervals)


    if intervals.empty:

        row[
            "gps_feature_status"
        ] = (
            "no_intervals"
        )

        feature_rows.append(
            row
        )

        continue


    supported = intervals[
        intervals[
            "supported"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()


    row[
        "n_supported_intervals"
    ] = len(
        supported
    )

    row[
        "supported_duration_ms"
    ] = float(
        pd.to_numeric(
            supported[
                "duration_ms"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    row[
        "supported_fraction_of_day"
    ] = (
        row[
            "supported_duration_ms"
        ]
        / float(
            end_ms
            - start_ms
        )
    )


    if supported.empty:

        row[
            "gps_feature_status"
        ] = (
            "no_supported_intervals"
        )

        feature_rows.append(
            row
        )

        continue


    # ========================================================
    # SPATIAL SEMANTICS
    # ========================================================

    spatial = (
        add_gps_spatial_semantics(
            supported
        )
    )


    if (
        "spatial_status"
        in spatial.columns
    ):

        spatial_counts = (
            spatial[
                "spatial_status"
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        for status, count in (
            spatial_counts.items()
        ):

            row[
                f"n_spatial_{status}"
            ] = int(
                count
            )


    # ========================================================
    # TRAJECTORY PLAUSIBILITY + MOBILITY
    # ========================================================

    trajectory = (
        add_gps_trajectory_plausibility(
            spatial,
            max_plausible_speed_kmh=(
                MAX_PLAUSIBLE_SPEED_KMH
            ),
        )
    )


    mobility = (
        compute_daily_mobility_features(
            trajectory
        )
    )

    row.update(
        mobility
    )


    # ========================================================
    # LOCATION MICROSEGMENTS
    # ========================================================

    microsegments = (
        build_anchor_compatible_location_episodes(
            spatial
        )
    )

    row[
        "n_anchor_location_episodes"
    ] = len(
        microsegments
    )


    # ========================================================
    # PLACE CANDIDATES
    # ========================================================

    if not microsegments.empty:

        (
            assignments,
            places,
        ) = build_place_candidates(
            microsegments,
            compatibility_multiplier=(
                COMPATIBILITY_MULTIPLIER
            ),
        )

    else:

        assignments = (
            pd.DataFrame(
                columns=[
                    "place_candidate_id",
                    "duration_ms",
                ]
            )
        )

        places = pd.DataFrame()


    row[
        "n_place_candidates_model"
    ] = len(
        places
    )


    # ========================================================
    # DAILY LOCATION FEATURES
    # ========================================================

    location = (
        compute_daily_location_features(
            assignments,
            supported_duration_ms=(
                row[
                    "supported_duration_ms"
                ]
            ),
        )
    )

    row.update(
        location
    )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    row[
        "gps_feature_status"
    ] = "ok"

    feature_rows.append(
        row
    )


# ============================================================
# BUILD TABLE
# ============================================================

features = pd.DataFrame(
    feature_rows
)

features = (
    features.sort_values(
        [
            "participant_id",
            "date",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE
# ============================================================

features.to_csv(
    OUT_PATH,
    index=False,
)

features.to_parquet(
    PARQUET_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 100)
print("DAILY CLINICAL FEATURE TABLE")
print("=" * 100)

print(
    "Rows:",
    len(features),
)

print(
    "Participants:",
    features[
        "participant_id"
    ].nunique(),
)

print()
print(
    "GPS feature status:"
)

print(
    features[
        "gps_feature_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print()
print("=" * 100)
print("SUPPORTED GPS COVERAGE")
print("=" * 100)

if (
    "supported_fraction_of_day"
    in features.columns
):

    print(
        features[
            "supported_fraction_of_day"
        ]
        .describe(
            percentiles=[
                .10,
                .25,
                .50,
                .75,
                .90,
            ]
        )
        .to_string()
    )


print()
print("=" * 100)
print("CORE FEATURE DESCRIPTIVES")
print("=" * 100)

core = [
    "supported_displacement_distance_m",
    "location_entropy_nats",
    "effective_number_places",
    "dominant_place_fraction",
    "n_observed_places",
    "implausible_transition_fraction",
]

available = [
    c for c in core
    if c in features.columns
]

print(
    features[
        available
    ]
    .describe()
    .T
    .to_string()
)


print()
print("=" * 100)
print("USABLE DAYS BY PARTICIPANT")
print("=" * 100)

usable = (
    features.assign(
        usable_gps=(
            features[
                "gps_feature_status"
            ].eq(
                "ok"
            )
        )
    )
    .groupby(
        "participant_id"
    )
    .agg(
        ema_days=(
            "date",
            "size",
        ),
        gps_feature_days=(
            "usable_gps",
            "sum",
        ),
    )
    .reset_index()
)

usable[
    "gps_feature_fraction"
] = (
    usable[
        "gps_feature_days"
    ]
    / usable[
        "ema_days"
    ]
)

print(
    usable.sort_values(
        "gps_feature_days",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("THRESHOLDS")
print("=" * 100)

for threshold in [
    5,
    10,
    15,
    20,
    25,
    30,
]:

    n = int(
        (
            usable[
                "gps_feature_days"
            ]
            >= threshold
        ).sum()
    )

    print(
        f">={threshold:2d} usable GPS+EMA days: "
        f"{n} participants"
    )


print()
print("Saved:")
print(" ", OUT_PATH)
print(" ", PARQUET_OUT)


del raw
gc.collect()
