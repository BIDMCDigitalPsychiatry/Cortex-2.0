"""Derive participant-specific home location and weekly time-at-home.

Home is operationally defined as the persistent Cortex place candidate
with the greatest observed nighttime occupancy (00:00–06:00 local time)
across that participant's available clinical GPS days.

Output:
- participant home anchors
- daily home-time summaries
- weekly clinical-window home-time summaries
- weekly clinical feature table augmented with time at home
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_intervals import (
    build_gps_intervals,
)
from cortex2_engine.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)
from cortex2_engine.features.gps_place_candidates import (
    build_place_candidates,
)


# ============================================================
# CONFIG
# ============================================================

DC = Path("/shared/Digital Clinic")

ROOT = (
    DC
    / "cortex2_outputs/digital_clinic/"
      "feature_reliability/gps_clinical"
)

RAW_PATH = (
    ROOT
    / "gps_clinical_weekly_raw_points.parquet"
)

REQUESTED_DAYS_PATH = (
    ROOT
    / "gps_clinical_weekly_requested_days.parquet"
)

WEEKLY_FEATURE_PATH = (
    ROOT
    / "gps_clinical_weekly_features.parquet"
)

OUT_DIR = (
    ROOT
    / "home_time"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HOME_ANCHOR_OUT = (
    OUT_DIR
    / "participant_home_anchors.csv"
)

DAILY_OUT = (
    OUT_DIR
    / "daily_home_time.csv"
)

WEEKLY_OUT = (
    OUT_DIR
    / "weekly_home_time.csv"
)

ENHANCED_WEEKLY_OUT = (
    ROOT
    / "gps_clinical_weekly_features_with_home.parquet"
)


MAX_GAP_MS = 60_000

COMPATIBILITY_MULTIPLIER = 2.0

LOCAL_TIMEZONE = "America/New_York"

NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6


# ============================================================
# LOAD
# ============================================================

raw = pd.read_parquet(
    RAW_PATH
)

requested = pd.read_parquet(
    REQUESTED_DAYS_PATH
)

weekly_features = pd.read_parquet(
    WEEKLY_FEATURE_PATH
)


raw["participant_id"] = (
    raw[
        "participant_id"
    ].astype(str)
)

raw["date"] = (
    pd.to_datetime(
        raw["date"]
    )
    .dt.strftime("%Y-%m-%d")
)

requested["participant_id"] = (
    requested[
        "participant_id"
    ].astype(str)
)

requested["date"] = (
    pd.to_datetime(
        requested["date"]
    )
    .dt.strftime("%Y-%m-%d")
)


print("=" * 100)
print("CORTEX 2.0 WEEKLY TIME AT HOME")
print("=" * 100)

print(
    "Raw GPS rows:",
    f"{len(raw):,}",
)

print(
    "Participants:",
    raw[
        "participant_id"
    ].nunique(),
)

print(
    "Requested participant-days:",
    requested[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "Timezone:",
    LOCAL_TIMEZONE,
)

print(
    "Nighttime reference:",
    f"{NIGHT_START_HOUR:02d}:00–"
    f"{NIGHT_END_HOUR:02d}:00",
)


# ============================================================
# PARTICIPANT PROCESSING
# ============================================================

home_anchor_rows = []
daily_rows = []


for participant_id, participant_raw in raw.groupby(
    "participant_id",
    sort=True,
):

    print()
    print("-" * 90)

    print(
        participant_id,
        f"raw_rows={len(participant_raw):,}",
    )

    participant_raw = (
        participant_raw
        .sort_values(
            "timestamp"
        )
        .copy()
    )


    all_microsegments = []

    day_supported_duration = {}


    # ========================================================
    # BUILD DAILY LOCATION MICROSEGMENTS
    # ========================================================

    for date, day_raw in participant_raw.groupby(
        "date",
        sort=True,
    ):

        day_raw = (
            day_raw
            .sort_values(
                "timestamp"
            )
            .copy()
        )


        if (
            "day_start_ms"
            in day_raw.columns
        ):

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

        else:

            date_utc = pd.Timestamp(
                date,
                tz="UTC",
            )

            start_ms = int(
                date_utc.timestamp()
                * 1000
            )

            end_ms = int(
                (
                    date_utc
                    + pd.Timedelta(
                        days=1
                    )
                ).timestamp()
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


        if intervals.empty:

            day_supported_duration[
                date
            ] = 0.0

            continue


        supported = intervals[
            intervals[
                "supported"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()


        supported_duration_ms = float(
            pd.to_numeric(
                supported[
                    "duration_ms"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )


        day_supported_duration[
            date
        ] = supported_duration_ms


        if supported.empty:
            continue


        spatial = (
            add_gps_spatial_semantics(
                supported
            )
        )


        microsegments = (
            build_anchor_compatible_location_episodes(
                spatial
            )
        )


        if microsegments.empty:
            continue


        microsegments = (
            microsegments.copy()
        )

        microsegments[
            "participant_id"
        ] = participant_id

        microsegments[
            "date"
        ] = date


        midpoint_ms = (
            (
                pd.to_numeric(
                    microsegments[
                        "episode_start_ms"
                    ],
                    errors="coerce",
                )
                + pd.to_numeric(
                    microsegments[
                        "episode_end_ms"
                    ],
                    errors="coerce",
                )
            )
            / 2.0
        )


        local_time = (
            pd.to_datetime(
                midpoint_ms,
                unit="ms",
                utc=True,
            )
            .dt.tz_convert(
                LOCAL_TIMEZONE
            )
        )


        microsegments[
            "local_hour"
        ] = (
            local_time.dt.hour
        )


        microsegments[
            "is_nighttime"
        ] = (
            (
                microsegments[
                    "local_hour"
                ]
                >= NIGHT_START_HOUR
            )
            & (
                microsegments[
                    "local_hour"
                ]
                < NIGHT_END_HOUR
            )
        )


        all_microsegments.append(
            microsegments
        )


    # ========================================================
    # NO LOCATION EVIDENCE
    # ========================================================

    if not all_microsegments:

        print(
            "No location microsegments."
        )

        continue


    participant_segments = (
        pd.concat(
            all_microsegments,
            ignore_index=True,
        )
    )


    # ========================================================
    # CLUSTER PERSISTENT PLACES ACROSS PARTICIPANT PERIOD
    # ========================================================

    assignments, places = (
        build_place_candidates(
            participant_segments,
            compatibility_multiplier=(
                COMPATIBILITY_MULTIPLIER
            ),
        )
    )


    if assignments.empty:

        print(
            "No persistent place assignments."
        )

        continue


    # Ensure metadata remain available even if the clustering
    # implementation does not propagate every extra column.
    for col in [
        "date",
        "local_hour",
        "is_nighttime",
    ]:

        if (
            col
            not in assignments.columns
        ):

            assignments[
                col
            ] = participant_segments[
                col
            ].to_numpy()


    # ========================================================
    # HOME = PLACE WITH MOST NIGHTTIME OBSERVED DURATION
    # ========================================================

    nighttime = assignments[
        assignments[
            "is_nighttime"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()


    if nighttime.empty:

        print(
            "No nighttime location evidence."
        )

        continue


    night_by_place = (
        nighttime.groupby(
            "place_candidate_id",
            as_index=False,
        )[
            "duration_ms"
        ]
        .sum()
        .sort_values(
            "duration_ms",
            ascending=False,
        )
    )


    home_place_id = (
        night_by_place.iloc[0][
            "place_candidate_id"
        ]
    )


    total_night_duration = float(
        night_by_place[
            "duration_ms"
        ].sum()
    )

    home_night_duration = float(
        night_by_place.iloc[0][
            "duration_ms"
        ]
    )

    home_night_fraction = (
        home_night_duration
        / total_night_duration
        if total_night_duration > 0
        else np.nan
    )


    home_place_row = places[
        places[
            "place_candidate_id"
        ].eq(
            home_place_id
        )
    ]


    home_anchor = {
        "participant_id":
            participant_id,

        "home_place_candidate_id":
            home_place_id,

        "nighttime_observed_duration_ms":
            total_night_duration,

        "home_nighttime_duration_ms":
            home_night_duration,

        "home_nighttime_fraction":
            home_night_fraction,

        "n_persistent_places":
            len(places),
    }


    if len(home_place_row):

        for candidate_col in [
            "representative_latitude",
            "representative_longitude",
            "anchor_latitude",
            "anchor_longitude",
        ]:

            if (
                candidate_col
                in home_place_row.columns
            ):

                home_anchor[
                    candidate_col
                ] = (
                    home_place_row[
                        candidate_col
                    ].iloc[0]
                )


    home_anchor_rows.append(
        home_anchor
    )


    # ========================================================
    # DAILY TIME AT HOME
    # ========================================================

    for date in sorted(
        day_supported_duration
    ):

        supported_duration_ms = float(
            day_supported_duration[
                date
            ]
        )


        day_assignments = assignments[
            assignments[
                "date"
            ].astype(str)
            .eq(
                str(date)
            )
        ].copy()


        home_duration_ms = float(
            pd.to_numeric(
                day_assignments.loc[
                    day_assignments[
                        "place_candidate_id"
                    ].eq(
                        home_place_id
                    ),
                    "duration_ms",
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )


        if (
            supported_duration_ms
            > 0
        ):

            home_fraction = (
                home_duration_ms
                / supported_duration_ms
            )

        else:

            home_fraction = np.nan


        daily_rows.append(
            {
                "participant_id":
                    participant_id,

                "date":
                    date,

                "home_place_candidate_id":
                    home_place_id,

                "supported_duration_ms":
                    supported_duration_ms,

                "home_duration_ms":
                    home_duration_ms,

                "time_at_home_fraction":
                    home_fraction,

                "time_at_home_pct":
                    (
                        100.0
                        * home_fraction
                        if np.isfinite(
                            home_fraction
                        )
                        else np.nan
                    ),
            }
        )


    print(
        "persistent_places=",
        len(places),
        "| home_night_fraction=",
        f"{home_night_fraction:.3f}",
    )


# ============================================================
# SAVE HOME + DAILY
# ============================================================

home_anchors = pd.DataFrame(
    home_anchor_rows
)

daily_home = pd.DataFrame(
    daily_rows
)


home_anchors.to_csv(
    HOME_ANCHOR_OUT,
    index=False,
)

daily_home.to_csv(
    DAILY_OUT,
    index=False,
)


# ============================================================
# WEEKLY AGGREGATION
# ============================================================

weekly_days = requested.merge(
    daily_home[
        [
            "participant_id",
            "date",
            "supported_duration_ms",
            "home_duration_ms",
            "time_at_home_fraction",
        ]
    ],
    on=[
        "participant_id",
        "date",
    ],
    how="left",
)


weekly_home = (
    weekly_days.groupby(
        [
            "clinical_window_id",
            "participant_id",
        ],
        as_index=False,
    )
    .agg(
        home_duration_ms=(
            "home_duration_ms",
            "sum",
        ),

        supported_duration_ms=(
            "supported_duration_ms",
            "sum",
        ),

        mean_daily_home_fraction=(
            "time_at_home_fraction",
            "mean",
        ),

        n_home_feature_days=(
            "time_at_home_fraction",
            "count",
        ),
    )
)


weekly_home[
    "time_at_home_fraction"
] = np.where(
    weekly_home[
        "supported_duration_ms"
    ] > 0,
    (
        weekly_home[
            "home_duration_ms"
        ]
        / weekly_home[
            "supported_duration_ms"
        ]
    ),
    np.nan,
)


weekly_home[
    "time_at_home_pct"
] = (
    100.0
    * weekly_home[
        "time_at_home_fraction"
    ]
)


weekly_home.to_csv(
    WEEKLY_OUT,
    index=False,
)


# ============================================================
# MERGE INTO WEEKLY CLINICAL TABLE
# ============================================================

enhanced = weekly_features.merge(
    weekly_home[
        [
            "clinical_window_id",
            "participant_id",
            "time_at_home_fraction",
            "time_at_home_pct",
            "mean_daily_home_fraction",
            "n_home_feature_days",
        ]
    ],
    on=[
        "clinical_window_id",
        "participant_id",
    ],
    how="left",
)


enhanced.to_parquet(
    ENHANCED_WEEKLY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 100)
print("HOME ANCHORS")
print("=" * 100)

print(
    "Participants with home anchor:",
    home_anchors[
        "participant_id"
    ].nunique(),
)

print()

print(
    home_anchors[
        "home_nighttime_fraction"
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
print("DAILY TIME AT HOME")
print("=" * 100)

print(
    "Participant-days:",
    len(daily_home),
)

print(
    daily_home[
        "time_at_home_pct"
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
print("WEEKLY TIME AT HOME")
print("=" * 100)

print(
    "Clinical windows:",
    len(weekly_home),
)

print(
    "Windows with home time:",
    weekly_home[
        "time_at_home_pct"
    ].notna().sum(),
)

print()

print(
    weekly_home[
        "time_at_home_pct"
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
print("Saved:")
print(" ", HOME_ANCHOR_OUT)
print(" ", DAILY_OUT)
print(" ", WEEKLY_OUT)
print(" ", ENHANCED_WEEKLY_OUT)
