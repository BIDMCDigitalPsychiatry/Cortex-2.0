from pathlib import Path
import os

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)
from cortex2_engine.features.gps_place_candidates import build_place_candidates
from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)
from cortex2_engine.features.gps_trip_features import (
    build_place_visits,
    build_observed_trips,
    add_smoothed_trip_path_distances,
)


# ============================================================
# CONFIG
# ============================================================

PARTICIPANT_ID = "AMO327"

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

OUT_DIR = (
    ROOT
    / "home_time"
    / "trip_distance_validation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_GAP_MS = 60_000
COMPATIBILITY_MULTIPLIER = 1.0
MIN_ENDPOINT_DWELL_MS = 10 * 60 * 1000


# ============================================================
# LOAD PARTICIPANT
# ============================================================

raw = pd.read_parquet(
    RAW_PATH,
)

raw["participant_id"] = (
    raw["participant_id"]
    .astype(str)
)

raw["date"] = (
    pd.to_datetime(
        raw["date"]
    )
    .dt.strftime("%Y-%m-%d")
)

p = (
    raw.loc[
        raw["participant_id"]
        == PARTICIPANT_ID
    ]
    .copy()
)

if p.empty:
    raise RuntimeError(
        f"No GPS data found for {PARTICIPANT_ID}"
    )

print("=" * 100)
print("CORTEX 2.0 TRIP-DISTANCE VALIDATION")
print("=" * 100)
print("Participant:", PARTICIPANT_ID)
print("Raw rows:", f"{len(p):,}")
print(
    "Dates:",
    p["date"].min(),
    "to",
    p["date"].max(),
)
print(
    "Unique dates:",
    p["date"].nunique(),
)


# ============================================================
# BUILD DAILY EDGES + MICROSEGMENTS
# ============================================================

all_edges = []
all_microsegments = []
daily_old_feature_rows = []

day_groups = list(
    p.groupby(
        "date",
        sort=True,
    )
)

smoke_days = int(
    os.environ.get(
        "CORTEX2_TRIP_SMOKE_DAYS",
        "0",
    )
)

if smoke_days > 0:
    day_groups = day_groups[:smoke_days]

    print(
        f"SMOKE MODE: processing "
        f"{len(day_groups)} day(s)"
    )

for date, day in tqdm(
    day_groups,
    total=len(day_groups),
    desc=f"{PARTICIPANT_ID}: processing GPS days",
    unit="day",
):
    day = (
        day.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(day) < 2:
        continue

    day_start_values = (
        pd.to_numeric(
            day["day_start_ms"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    day_end_values = (
        pd.to_numeric(
            day["day_end_ms"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if (
        len(day_start_values) == 0
        or len(day_end_values) == 0
    ):
        tqdm.write(
            f"Skipping {date}: missing day boundaries"
        )
        continue

    if (
        len(day_start_values) > 1
        or len(day_end_values) > 1
    ):
        tqdm.write(
            f"WARNING {date}: multiple day boundaries found "
            f"(start={len(day_start_values)}, "
            f"end={len(day_end_values)})"
        )

    window_start_ms = int(
        day_start_values[0]
    )

    window_end_ms = int(
        day_end_values[0]
    )

    intervals = build_gps_intervals(
        day,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        max_gap_ms=MAX_GAP_MS,
    )

    if intervals.empty:
        continue

    intervals = (
        add_gps_spatial_semantics(
            intervals
        )
    )

    intervals = (
        add_gps_trajectory_plausibility(
            intervals
        )
    )

    intervals = intervals.copy()
    intervals["participant_id"] = (
        PARTICIPANT_ID
    )
    intervals["date"] = date

    all_edges.append(
        intervals
    )

    old = compute_daily_mobility_features(
        intervals
    )

    daily_old_feature_rows.append(
        {
            "participant_id":
                PARTICIPANT_ID,
            "date":
                date,
            "old_supported_displacement_m":
                old[
                    "supported_displacement_distance_m"
                ],
            "old_displacement_edge_count":
                old[
                    "displacement_edge_count"
                ],
            "old_implausible_transition_count":
                old[
                    "implausible_transition_count"
                ],
        }
    )

    supported = (
        intervals.loc[
            intervals["supported"]
            .fillna(False)
            .astype(bool)
        ]
        .copy()
    )

    if supported.empty:
        continue

    microsegments = (
        build_anchor_compatible_location_episodes(
            supported
        )
    )

    if microsegments.empty:
        continue

    microsegments = (
        microsegments.copy()
    )

    microsegments[
        "participant_id"
    ] = PARTICIPANT_ID

    microsegments["date"] = date

    all_microsegments.append(
        microsegments
    )


if not all_edges:
    raise RuntimeError(
        "No GPS edges were created."
    )

if not all_microsegments:
    raise RuntimeError(
        "No anchor-compatible microsegments were created."
    )


edges = pd.concat(
    all_edges,
    ignore_index=True,
)

microsegments = pd.concat(
    all_microsegments,
    ignore_index=True,
)


# ============================================================
# BUILD PERSISTENT PLACES
# SAME LOGIC AS HOME-TIME BUILDER
# ============================================================

assignments, places = (
    build_place_candidates(
        microsegments,
        compatibility_multiplier=(
            COMPATIBILITY_MULTIPLIER
        ),
    )
)

if assignments.empty:
    raise RuntimeError(
        "No place assignments were created."
    )

if places.empty:
    raise RuntimeError(
        "No place candidates were created."
    )


# ============================================================
# BUILD VISITS
# ============================================================

visits = build_place_visits(
    assignments,
    min_endpoint_dwell_ms=(
        MIN_ENDPOINT_DWELL_MS
    ),
)

print()
print("=" * 100)
print("PLACE / VISIT SUMMARY")
print("=" * 100)

print(
    "Microsegments:",
    len(microsegments),
)

print(
    "Persistent places:",
    len(places),
)

print(
    "Observed visits:",
    len(visits),
)

print(
    "Qualifying endpoint visits:",
    int(
        visits[
            "qualifying_endpoint"
        ].sum()
    ),
)


# ============================================================
# BUILD TRIPS
# ============================================================

trips = build_observed_trips(
    edges,
    visits,
    places,
)

print()
print("=" * 100)
print("TRIP SUMMARY")
print("=" * 100)

print(
    "Trips:",
    len(trips),
)

if not trips.empty:
    print(
        "Between-place trips:",
        int(
            (
                trips["trip_type"]
                == "between_places"
            ).sum()
        ),
    )

    print(
        "Same-place excursions:",
        int(
            (
                trips["trip_type"]
                == "same_place_excursion"
            ).sum()
        ),
    )

    print(
        "Complete observed paths:",
        int(
            trips[
                "path_complete"
            ].sum()
        ),
    )

    print(
        "Median support fraction:",
        round(
            float(
                trips[
                    "supported_fraction"
                ].median()
            ),
            3,
        ),
    )


# ============================================================
# KALMAN / RTS SMOOTHING SENSITIVITY
# ============================================================

SMOOTHING_ACCELERATION_SD_GRID = [
    0.5,
    1.0,
    2.0,
]

smoothed_versions = {}

if not trips.empty:

    print()
    print("=" * 100)
    print("KALMAN / RTS SMOOTHING SENSITIVITY")
    print("=" * 100)

    sensitivity_rows = []

    for acceleration_sd in (
        SMOOTHING_ACCELERATION_SD_GRID
    ):
        print(
            f"Smoothing trips with "
            f"acceleration SD = "
            f"{acceleration_sd:.1f} m/s^2 ..."
        )

        smoothed = (
            add_smoothed_trip_path_distances(
                edges,
                trips,
                acceleration_sd_m_s2=(
                    acceleration_sd
                ),
            )
        )

        smoothed_versions[
            acceleration_sd
        ] = smoothed

        valid = smoothed.dropna(
            subset=[
                "path_distance_raw_obs_m",
                "path_distance_smooth_obs_m",
            ]
        )

        if valid.empty:
            sensitivity_rows.append(
                {
                    "acceleration_sd_m_s2":
                        acceleration_sd,
                    "n_trips_smoothed":
                        0,
                    "median_smoothing_coverage":
                        np.nan,
                    "raw_total_km":
                        np.nan,
                    "smooth_total_km":
                        np.nan,
                    "median_smooth_raw_ratio":
                        np.nan,
                    "median_trip_change_km":
                        np.nan,
                }
            )

            continue

        ratio = (
            valid[
                "path_distance_smooth_obs_m"
            ]
            /
            valid[
                "path_distance_raw_obs_m"
            ].replace(
                0,
                np.nan,
            )
        )

        sensitivity_rows.append(
            {
                "acceleration_sd_m_s2":
                    acceleration_sd,
                "n_trips_smoothed":
                    len(valid),
                "median_smoothing_coverage":
                    float(
                        valid[
                            "smoothing_coverage_fraction"
                        ].median()
                    ),
                "raw_total_km":
                    float(
                        valid[
                            "path_distance_raw_obs_m"
                        ].sum()
                        / 1000.0
                    ),
                "smooth_total_km":
                    float(
                        valid[
                            "path_distance_smooth_obs_m"
                        ].sum()
                        / 1000.0
                    ),
                "median_smooth_raw_ratio":
                    float(
                        ratio.median()
                    ),
                "median_trip_change_km":
                    float(
                        (
                            valid[
                                "path_distance_smooth_obs_m"
                            ]
                            - valid[
                                "path_distance_raw_obs_m"
                            ]
                        ).median()
                        / 1000.0
                    ),
            }
        )

    smoothing_sensitivity = (
        pd.DataFrame(
            sensitivity_rows
        )
    )

    print()
    print(
        smoothing_sensitivity
        .to_string(
            index=False
        )
    )

    smoothing_sensitivity.to_csv(
        OUT_DIR
        / (
            f"{PARTICIPANT_ID}_"
            "smoothing_sensitivity.csv"
        ),
        index=False,
    )


    # --------------------------------------------------------
    # Detailed trip-level comparison for each sensitivity value
    # --------------------------------------------------------

    comparison_rows = []

    for acceleration_sd, smoothed in (
        smoothed_versions.items()
    ):
        z = smoothed[
            [
                "trip_id",
                "trip_type",
                "path_distance_raw_obs_m",
                "path_distance_smooth_obs_m",
                "endpoint_displacement_obs_m",
                "smoothing_coverage_fraction",
                "n_supported_blocks",
                "n_smoothed_blocks",
                "n_unsmoothed_blocks",
            ]
        ].copy()

        z[
            "acceleration_sd_m_s2"
        ] = acceleration_sd

        z[
            "raw_km"
        ] = (
            z[
                "path_distance_raw_obs_m"
            ]
            / 1000.0
        )

        z[
            "smooth_km"
        ] = (
            z[
                "path_distance_smooth_obs_m"
            ]
            / 1000.0
        )

        z[
            "endpoint_km"
        ] = (
            z[
                "endpoint_displacement_obs_m"
            ]
            / 1000.0
        )

        z[
            "smooth_over_raw"
        ] = (
            z[
                "path_distance_smooth_obs_m"
            ]
            /
            z[
                "path_distance_raw_obs_m"
            ].replace(
                0,
                np.nan,
            )
        )

        comparison_rows.append(
            z
        )

    smoothing_trip_comparison = pd.concat(
        comparison_rows,
        ignore_index=True,
    )

    smoothing_trip_comparison.to_csv(
        OUT_DIR
        / (
            f"{PARTICIPANT_ID}_"
            "smoothing_trip_comparison.csv"
        ),
        index=False,
    )


    # --------------------------------------------------------
    # Print strongest smoothing effects
    # using middle sensitivity value only.
    # This is diagnostic, not yet a declared primary parameter.
    # --------------------------------------------------------

    middle = (
        smoothed_versions[
            1.0
        ].copy()
    )

    middle[
        "raw_km"
    ] = (
        middle[
            "path_distance_raw_obs_m"
        ]
        / 1000.0
    )

    middle[
        "smooth_km"
    ] = (
        middle[
            "path_distance_smooth_obs_m"
        ]
        / 1000.0
    )

    middle[
        "endpoint_km"
    ] = (
        middle[
            "endpoint_displacement_obs_m"
        ]
        / 1000.0
    )

    middle[
        "smooth_over_raw"
    ] = (
        middle[
            "path_distance_smooth_obs_m"
        ]
        /
        middle[
            "path_distance_raw_obs_m"
        ].replace(
            0,
            np.nan,
        )
    )

    middle[
        "distance_reduction_km"
    ] = (
        (
            middle[
                "path_distance_raw_obs_m"
            ]
            - middle[
                "path_distance_smooth_obs_m"
            ]
        )
        / 1000.0
    )

    print()
    print(
        "LARGEST RAW → SMOOTH DISTANCE REDUCTIONS "
        "(acceleration SD = 1.0 m/s^2)"
    )
    print("-" * 100)

    print(
        middle.sort_values(
            "distance_reduction_km",
            ascending=False,
        )
        [
            [
                "trip_id",
                "trip_type",
                "raw_km",
                "smooth_km",
                "endpoint_km",
                "smooth_over_raw",
                "smoothing_coverage_fraction",
                "n_supported_blocks",
                "n_unsmoothed_blocks",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

# ============================================================
# ADD LOCAL DATES FOR DAILY AGGREGATION
# ============================================================

if not trips.empty:
    trip_start_dt = pd.to_datetime(
        trips["trip_start_ms"],
        unit="ms",
        utc=True,
    )

    trip_end_dt = pd.to_datetime(
        trips["trip_end_ms"],
        unit="ms",
        utc=True,
    )

    # Same timezone currently used by home-time workflow.
    trip_start_local = (
        trip_start_dt
        .dt.tz_convert(
            "America/New_York"
        )
    )

    trip_end_local = (
        trip_end_dt
        .dt.tz_convert(
            "America/New_York"
        )
    )

    trips["departure_date"] = (
        trip_start_local
        .dt.strftime("%Y-%m-%d")
    )

    trips["arrival_date"] = (
        trip_end_local
        .dt.strftime("%Y-%m-%d")
    )

    # Retain legacy date field for diagnostic displays.
    trips["date"] = (
        trips["departure_date"]
    )

    trips["cross_date"] = (
        trips["departure_date"]
        != trips["arrival_date"]
    )

    departure_day = pd.to_datetime(
        trips["departure_date"]
    )

    arrival_day = pd.to_datetime(
        trips["arrival_date"]
    )

    trips["trip_span_days"] = (
        (
            arrival_day
            - departure_day
        ).dt.days
        + 1
    )



# ============================================================
# SMOOTHING VALIDITY AUDIT
# ============================================================

if (
    not trips.empty
    and smoothed_versions
):

    print()
    print("=" * 100)
    print("SMOOTHING VALIDITY AUDIT")
    print("=" * 100)

    validity_rows = []

    for acceleration_sd, smoothed in (
        smoothed_versions.items()
    ):

        audit = (
            smoothed.merge(
                trips[
                    [
                        "trip_id",
                        "cross_date",
                    ]
                ],
                on="trip_id",
                how="left",
                validate="one_to_one",
            )
        )

        clean = audit.loc[
            (
                audit["trip_type"]
                == "between_places"
            )
            &
            (
                audit["path_complete"]
                .fillna(False)
                .astype(bool)
            )
            &
            (
                ~audit["cross_date"]
                .fillna(True)
                .astype(bool)
            )
            &
            (
                audit[
                    "smoothing_coverage_fraction"
                ]
                >= 0.999999
            )
            &
            audit[
                "path_distance_smooth_obs_m"
            ].notna()
            &
            audit[
                "endpoint_displacement_obs_m"
            ].notna()
        ].copy()

        if clean.empty:
            validity_rows.append(
                {
                    "acceleration_sd_m_s2":
                        acceleration_sd,
                    "n_clean_trips":
                        0,
                    "raw_total_km":
                        np.nan,
                    "smooth_total_km":
                        np.nan,
                    "endpoint_total_km":
                        np.nan,
                    "median_smooth_raw_ratio":
                        np.nan,
                    "median_smooth_endpoint_ratio":
                        np.nan,
                    "n_endpoint_gt_smooth":
                        0,
                    "max_endpoint_minus_smooth_m":
                        np.nan,
                }
            )
            continue

        clean[
            "smooth_raw_ratio"
        ] = (
            clean[
                "path_distance_smooth_obs_m"
            ]
            /
            clean[
                "path_distance_raw_obs_m"
            ].replace(
                0,
                np.nan,
            )
        )

        clean[
            "smooth_endpoint_ratio"
        ] = (
            clean[
                "path_distance_smooth_obs_m"
            ]
            /
            clean[
                "endpoint_displacement_obs_m"
            ].replace(
                0,
                np.nan,
            )
        )

        clean[
            "endpoint_minus_smooth_m"
        ] = (
            clean[
                "endpoint_displacement_obs_m"
            ]
            -
            clean[
                "path_distance_smooth_obs_m"
            ]
        )

        violations = clean.loc[
            clean[
                "endpoint_minus_smooth_m"
            ]
            > 0
        ]

        validity_rows.append(
            {
                "acceleration_sd_m_s2":
                    acceleration_sd,
                "n_clean_trips":
                    len(clean),
                "raw_total_km":
                    float(
                        clean[
                            "path_distance_raw_obs_m"
                        ].sum()
                        / 1000.0
                    ),
                "smooth_total_km":
                    float(
                        clean[
                            "path_distance_smooth_obs_m"
                        ].sum()
                        / 1000.0
                    ),
                "endpoint_total_km":
                    float(
                        clean[
                            "endpoint_displacement_obs_m"
                        ].sum()
                        / 1000.0
                    ),
                "median_smooth_raw_ratio":
                    float(
                        clean[
                            "smooth_raw_ratio"
                        ].median()
                    ),
                "median_smooth_endpoint_ratio":
                    float(
                        clean[
                            "smooth_endpoint_ratio"
                        ].median()
                    ),
                "n_endpoint_gt_smooth":
                    len(violations),
                "max_endpoint_minus_smooth_m":
                    (
                        float(
                            violations[
                                "endpoint_minus_smooth_m"
                            ].max()
                        )
                        if not violations.empty
                        else 0.0
                    ),
            }
        )

    validity = pd.DataFrame(
        validity_rows
    )

    print()
    print(
        "COMPLETE SAME-DAY BETWEEN-PLACE TRIPS"
    )
    print("-" * 100)

    print(
        validity.to_string(
            index=False
        )
    )

    validity.to_csv(
        OUT_DIR
        / (
            f"{PARTICIPANT_ID}_"
            "smoothing_validity_clean_trips.csv"
        ),
        index=False,
    )


    # --------------------------------------------------------
    # Parameter sensitivity at the individual-trip level
    # --------------------------------------------------------

    per_parameter = []

    for acceleration_sd, smoothed in (
        smoothed_versions.items()
    ):
        z = smoothed[
            [
                "trip_id",
                "path_distance_smooth_obs_m",
            ]
        ].copy()

        z[
            "acceleration_sd_m_s2"
        ] = acceleration_sd

        per_parameter.append(
            z
        )

    parameter_long = pd.concat(
        per_parameter,
        ignore_index=True,
    )

    parameter_wide = (
        parameter_long.pivot(
            index="trip_id",
            columns="acceleration_sd_m_s2",
            values="path_distance_smooth_obs_m",
        )
        .reset_index()
    )

    distance_columns = [
        column
        for column in parameter_wide.columns
        if column != "trip_id"
    ]

    parameter_wide[
        "smooth_min_m"
    ] = (
        parameter_wide[
            distance_columns
        ].min(
            axis=1
        )
    )

    parameter_wide[
        "smooth_max_m"
    ] = (
        parameter_wide[
            distance_columns
        ].max(
            axis=1
        )
    )

    parameter_wide[
        "smooth_range_m"
    ] = (
        parameter_wide[
            "smooth_max_m"
        ]
        -
        parameter_wide[
            "smooth_min_m"
        ]
    )

    parameter_wide[
        "smooth_midpoint_m"
    ] = (
        (
            parameter_wide[
                "smooth_max_m"
            ]
            +
            parameter_wide[
                "smooth_min_m"
            ]
        )
        / 2.0
    )

    parameter_wide[
        "relative_parameter_range"
    ] = (
        parameter_wide[
            "smooth_range_m"
        ]
        /
        parameter_wide[
            "smooth_midpoint_m"
        ].replace(
            0,
            np.nan,
        )
    )

    parameter_wide = (
        parameter_wide.merge(
            trips[
                [
                    "trip_id",
                    "trip_type",
                    "path_complete",
                    "cross_date",
                    "supported_fraction",
                    "path_distance_raw_obs_m",
                    "endpoint_displacement_obs_m",
                ]
            ],
            on="trip_id",
            how="left",
            validate="one_to_one",
        )
    )

    parameter_wide.to_csv(
        OUT_DIR
        / (
            f"{PARTICIPANT_ID}_"
            "smoothing_parameter_trip_stability.csv"
        ),
        index=False,
    )

    print()
    print(
        "TRIPS MOST SENSITIVE TO PROCESS NOISE"
    )
    print("-" * 100)

    sensitive = (
        parameter_wide
        .dropna(
            subset=[
                "relative_parameter_range"
            ]
        )
        .sort_values(
            "relative_parameter_range",
            ascending=False,
        )
        .head(15)
        .copy()
    )

    sensitive[
        "raw_km"
    ] = (
        sensitive[
            "path_distance_raw_obs_m"
        ]
        / 1000.0
    )

    sensitive[
        "endpoint_km"
    ] = (
        sensitive[
            "endpoint_displacement_obs_m"
        ]
        / 1000.0
    )

    sensitive[
        "smooth_range_km"
    ] = (
        sensitive[
            "smooth_range_m"
        ]
        / 1000.0
    )

    print(
        sensitive[
            [
                "trip_id",
                "trip_type",
                "path_complete",
                "cross_date",
                "supported_fraction",
                "raw_km",
                "endpoint_km",
                "smooth_range_km",
                "relative_parameter_range",
            ]
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Why did some trips not receive a smoothed estimate?
    # --------------------------------------------------------

    reference = (
        smoothed_versions[
            1.0
        ].merge(
            trips[
                [
                    "trip_id",
                    "cross_date",
                ]
            ],
            on="trip_id",
            how="left",
            validate="one_to_one",
        )
    )

    unsmoothed = (
        reference.loc[
            reference[
                "path_distance_smooth_obs_m"
            ].isna()
        ]
        .copy()
    )

    print()
    print(
        "TRIPS WITHOUT A SMOOTHED ESTIMATE "
        "(acceleration SD = 1.0 m/s^2)"
    )
    print("-" * 100)

    if unsmoothed.empty:
        print("None.")

    else:
        print(
            unsmoothed[
                [
                    "trip_id",
                    "trip_type",
                    "path_complete",
                    "cross_date",
                    "supported_fraction",
                    "path_distance_raw_obs_m",
                    "endpoint_displacement_obs_m",
                    "n_supported_blocks",
                    "n_smoothed_blocks",
                    "n_unsmoothed_blocks",
                    "n_smoothed_fixes",
                ]
            ]
            .to_string(
                index=False
            )
        )

# ============================================================
# SUSPICIOUS-TRIP DIAGNOSTICS
# ============================================================

if not trips.empty:
    diagnostic_rows = []

    for trip in trips.itertuples(index=False):
        te = edges.loc[
            (edges["start_ms"] >= trip.trip_start_ms)
            & (edges["end_ms"] <= trip.trip_end_ms)
        ].copy()

        if te.empty:
            diagnostic_rows.append(
                {
                    "trip_id": trip.trip_id,
                    "date": trip.date,
                    "trip_type": trip.trip_type,
                    "trip_duration_min":
                        trip.trip_elapsed_duration_ms / 60000.0,
                    "L_raw_km":
                        trip.path_distance_raw_obs_m / 1000.0,
                    "D_km":
                        trip.place_displacement_obs_m / 1000.0,
                    "supported_fraction":
                        trip.supported_fraction,
                    "n_edges": 0,
                    "n_supported_edges_diag": 0,
                    "max_speed_kmh": np.nan,
                    "median_speed_kmh": np.nan,
                    "p95_speed_kmh": np.nan,
                    "max_edge_distance_m": np.nan,
                    "n_accuracy_overlap_edges": 0,
                    "n_displacement_evidence_edges": 0,
                    "n_implausible_speed_edges_diag": 0,
                }
            )
            continue

        supported_mask = (
            te["supported"]
            .fillna(False)
            .astype(bool)
        )

        ts = te.loc[supported_mask].copy()

        if "speed_kmh" in ts.columns:
            speeds = pd.to_numeric(
                ts["speed_kmh"],
                errors="coerce",
            )
        else:
            speeds = pd.Series(
                np.nan,
                index=ts.index,
            )

        if "distance_m" in ts.columns:
            distances = pd.to_numeric(
                ts["distance_m"],
                errors="coerce",
            )
        elif "distance_km" in ts.columns:
            distances = (
                pd.to_numeric(
                    ts["distance_km"],
                    errors="coerce",
                )
                * 1000.0
            )
        else:
            distances = pd.Series(
                np.nan,
                index=ts.index,
            )

        if "spatial_status" in ts.columns:
            spatial = ts["spatial_status"].astype(str)

            n_overlap = int(
                spatial.eq(
                    "accuracy_overlap"
                ).sum()
            )

            n_displacement = int(
                (
                    ~spatial.eq(
                        "accuracy_overlap"
                    )
                ).sum()
            )
        else:
            n_overlap = 0
            n_displacement = 0

        if "trajectory_status" in ts.columns:
            trajectory = (
                ts["trajectory_status"]
                .astype(str)
            )

            n_implausible = int(
                trajectory.str.contains(
                    "implausible",
                    case=False,
                    na=False,
                ).sum()
            )
        else:
            n_implausible = 0

        finite_speeds = speeds[
            np.isfinite(speeds)
        ]

        finite_distances = distances[
            np.isfinite(distances)
        ]

        diagnostic_rows.append(
            {
                "trip_id": int(
                    trip.trip_id
                ),
                "date": trip.date,
                "trip_type":
                    trip.trip_type,
                "trip_duration_min":
                    (
                        trip.trip_elapsed_duration_ms
                        / 60000.0
                    ),
                "L_raw_km":
                    (
                        trip.path_distance_raw_obs_m
                        / 1000.0
                    ),
                "D_km":
                    (
                        trip.place_displacement_obs_m
                        / 1000.0
                    ),
                "supported_fraction":
                    trip.supported_fraction,
                "n_edges":
                    len(te),
                "n_supported_edges_diag":
                    len(ts),
                "max_speed_kmh":
                    (
                        float(
                            finite_speeds.max()
                        )
                        if len(
                            finite_speeds
                        )
                        else np.nan
                    ),
                "median_speed_kmh":
                    (
                        float(
                            finite_speeds.median()
                        )
                        if len(
                            finite_speeds
                        )
                        else np.nan
                    ),
                "p95_speed_kmh":
                    (
                        float(
                            finite_speeds.quantile(
                                0.95
                            )
                        )
                        if len(
                            finite_speeds
                        )
                        else np.nan
                    ),
                "max_edge_distance_m":
                    (
                        float(
                            finite_distances.max()
                        )
                        if len(
                            finite_distances
                        )
                        else np.nan
                    ),
                "n_accuracy_overlap_edges":
                    n_overlap,
                "n_displacement_evidence_edges":
                    n_displacement,
                "n_implausible_speed_edges_diag":
                    n_implausible,
            }
        )

    trip_diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    trip_diagnostics.to_csv(
        OUT_DIR
        / f"{PARTICIPANT_ID}_trip_diagnostics.csv",
        index=False,
    )

    suspicious = (
        trip_diagnostics.loc[
            (
                trip_diagnostics[
                    "trip_duration_min"
                ]
                < 5
            )
            |
            (
                trip_diagnostics[
                    "L_raw_km"
                ]
                > 2
                * trip_diagnostics[
                    "D_km"
                ].clip(lower=0.1)
            )
        ]
        .sort_values(
            [
                "trip_duration_min",
                "L_raw_km",
            ]
        )
    )

    print()
    print("=" * 100)
    print("SUSPICIOUS TRIP DIAGNOSTICS")
    print("=" * 100)

    if suspicious.empty:
        print(
            "No trips met diagnostic criteria."
        )
    else:
        cols = [
            "trip_id",
            "date",
            "trip_type",
            "trip_duration_min",
            "L_raw_km",
            "D_km",
            "supported_fraction",
            "n_supported_edges_diag",
            "max_speed_kmh",
            "median_speed_kmh",
            "p95_speed_kmh",
            "max_edge_distance_m",
            "n_accuracy_overlap_edges",
            "n_displacement_evidence_edges",
            "n_implausible_speed_edges_diag",
        ]

        print(
            suspicious[
                cols
            ].to_string(
                index=False
            )
        )



# ============================================================
# ENDPOINT / PLACE-DISPLACEMENT AUDIT
# ============================================================

if not trips.empty:

    place_index = (
        places
        .set_index(
            "place_candidate_id"
        )
    )

    endpoint_rows = []

    for trip in trips.itertuples(
        index=False
    ):
        origin_id = int(
            trip.origin_place_candidate_id
        )

        destination_id = int(
            trip.destination_place_candidate_id
        )

        origin_place = (
            place_index.loc[origin_id]
            if origin_id in place_index.index
            else None
        )

        destination_place = (
            place_index.loc[destination_id]
            if destination_id in place_index.index
            else None
        )

        L = float(
            trip.path_distance_raw_obs_m
        )

        D = float(
            trip.endpoint_displacement_obs_m
        )

        row = {
            "trip_id":
                int(trip.trip_id),
            "date":
                trip.date,
            "trip_type":
                trip.trip_type,
            "trip_duration_min":
                (
                    trip.trip_elapsed_duration_ms
                    / 60000.0
                ),
            "supported_fraction":
                trip.supported_fraction,
            "path_complete":
                trip.path_complete,
            "L_raw_km":
                L / 1000.0,
            "D_place_km":
                D / 1000.0,
            "D_minus_L_km":
                (D - L) / 1000.0,
            "D_over_L":
                (
                    D / L
                    if L > 0
                    else np.nan
                ),
            "origin_place_id":
                origin_id,
            "destination_place_id":
                destination_id,
        }

        for prefix, place in [
            (
                "origin",
                origin_place,
            ),
            (
                "destination",
                destination_place,
            ),
        ]:
            if place is None:
                row[
                    f"{prefix}_rep_lat"
                ] = np.nan

                row[
                    f"{prefix}_rep_lon"
                ] = np.nan

                row[
                    f"{prefix}_seed_lat"
                ] = np.nan

                row[
                    f"{prefix}_seed_lon"
                ] = np.nan

                row[
                    f"{prefix}_seed_support_radius_m"
                ] = np.nan

                row[
                    f"{prefix}_max_seed_distance_m"
                ] = np.nan

                row[
                    f"{prefix}_max_seed_normalized_distance"
                ] = np.nan

                row[
                    f"{prefix}_n_microsegments"
                ] = np.nan

                row[
                    f"{prefix}_total_duration_h"
                ] = np.nan

                continue

            row[
                f"{prefix}_rep_lat"
            ] = float(
                place[
                    "representative_latitude"
                ]
            )

            row[
                f"{prefix}_rep_lon"
            ] = float(
                place[
                    "representative_longitude"
                ]
            )

            row[
                f"{prefix}_seed_lat"
            ] = float(
                place[
                    "seed_latitude"
                ]
            )

            row[
                f"{prefix}_seed_lon"
            ] = float(
                place[
                    "seed_longitude"
                ]
            )

            row[
                f"{prefix}_seed_support_radius_m"
            ] = float(
                place[
                    "seed_support_radius_m"
                ]
            )

            row[
                f"{prefix}_max_seed_distance_m"
            ] = float(
                place[
                    "max_seed_distance_m"
                ]
            )

            row[
                f"{prefix}_max_seed_normalized_distance"
            ] = float(
                place[
                    "max_seed_normalized_distance"
                ]
            )

            row[
                f"{prefix}_n_microsegments"
            ] = int(
                place[
                    "n_microsegments"
                ]
            )

            row[
                f"{prefix}_total_duration_h"
            ] = (
                float(
                    place[
                        "total_duration_ms"
                    ]
                )
                / 3_600_000.0
            )

        endpoint_rows.append(
            row
        )

    endpoint_audit = pd.DataFrame(
        endpoint_rows
    )

    endpoint_audit.to_csv(
        OUT_DIR
        / (
            f"{PARTICIPANT_ID}_"
            "endpoint_displacement_audit.csv"
        ),
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "ENDPOINT DISPLACEMENT AUDIT"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Complete trips where the geodesic distance between
    # Cortex place representatives exceeds observed path.
    #
    # For a fully observed trajectory, this should not normally
    # happen if the endpoint coordinates accurately represent
    # the actual origin/destination visits.
    # --------------------------------------------------------

    complete_violation = (
        endpoint_audit.loc[
            endpoint_audit[
                "path_complete"
            ].astype(bool)
            &
            (
                endpoint_audit[
                    "D_place_km"
                ]
                >
                endpoint_audit[
                    "L_raw_km"
                ]
            )
        ]
        .sort_values(
            "D_minus_L_km",
            ascending=False,
        )
    )

    print()
    print(
        "COMPLETE TRIPS WITH "
        "D_PLACE > L_RAW"
    )
    print("-" * 100)

    if complete_violation.empty:
        print("None.")
    else:
        print(
            complete_violation[
                [
                    "trip_id",
                    "date",
                    "trip_type",
                    "trip_duration_min",
                    "L_raw_km",
                    "D_place_km",
                    "D_minus_L_km",
                    "origin_place_id",
                    "destination_place_id",
                    "origin_seed_support_radius_m",
                    "destination_seed_support_radius_m",
                    "origin_max_seed_distance_m",
                    "destination_max_seed_distance_m",
                    "origin_n_microsegments",
                    "destination_n_microsegments",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Largest endpoint displacements regardless of completeness.
    # --------------------------------------------------------

    largest = (
        endpoint_audit
        .sort_values(
            "D_place_km",
            ascending=False,
        )
        .head(15)
    )

    print()
    print(
        "15 LARGEST PLACE-TO-PLACE "
        "DISPLACEMENTS"
    )
    print("-" * 100)

    print(
        largest[
            [
                "trip_id",
                "date",
                "trip_type",
                "trip_duration_min",
                "supported_fraction",
                "path_complete",
                "L_raw_km",
                "D_place_km",
                "D_over_L",
                "origin_place_id",
                "destination_place_id",
                "origin_rep_lat",
                "origin_rep_lon",
                "destination_rep_lat",
                "destination_rep_lon",
                "origin_seed_support_radius_m",
                "destination_seed_support_radius_m",
                "origin_max_seed_distance_m",
                "destination_max_seed_distance_m",
            ]
        ].to_string(
            index=False
        )
    )

# ============================================================
# DAILY NEW FEATURES
#
# IMPORTANT:
#   Path distance is allocated by the actual date of each
#   observed edge.
#
#   Endpoint displacement is aggregated only for trips whose
#   departure and arrival occur on the same local calendar day.
#
#   Cross-date trips remain in the trip-level table but their
#   full endpoint displacement is not assigned to one day.
# ============================================================

daily_path_rows = []

if not trips.empty:

    retained_intervals = []

    for trip in trips.itertuples(
        index=False
    ):
        te = edges.loc[
            (
                edges["start_ms"]
                >= trip.trip_start_ms
            )
            &
            (
                edges["end_ms"]
                <= trip.trip_end_ms
            )
        ].copy()

        if te.empty:
            continue

        te = te.loc[
            te["supported"]
            .fillna(False)
            .astype(bool)
        ].copy()

        if te.empty:
            continue

        if "distance_m" in te.columns:
            te["_distance_m"] = (
                pd.to_numeric(
                    te["distance_m"],
                    errors="coerce",
                )
            )

        elif "distance_km" in te.columns:
            te["_distance_m"] = (
                pd.to_numeric(
                    te["distance_km"],
                    errors="coerce",
                )
                * 1000.0
            )

        else:
            raise RuntimeError(
                "Edges require distance_m "
                "or distance_km."
            )

        duration_ms = pd.to_numeric(
            te["duration_ms"],
            errors="coerce",
        )

        if "speed_kmh" in te.columns:
            speed_kmh = pd.to_numeric(
                te["speed_kmh"],
                errors="coerce",
            )

        else:
            speed_kmh = (
                (
                    te["_distance_m"]
                    / 1000.0
                )
                /
                (
                    duration_ms
                    / 3_600_000.0
                )
            )

        valid_path = (
            te["_distance_m"].notna()
            &
            np.isfinite(
                te["_distance_m"]
            )
            &
            (
                te["_distance_m"]
                >= 0
            )
            &
            duration_ms.notna()
            &
            np.isfinite(
                duration_ms
            )
            &
            (
                duration_ms
                > 0
            )
            &
            (
                speed_kmh.isna()
                |
                (
                    speed_kmh
                    <= 1000.0
                )
            )
        )

        te = te.loc[
            valid_path
        ].copy()

        if te.empty:
            continue

        # Edges already carry their source local calendar date
        # from the daily Cortex processing loop.
        if "date" not in te.columns:
            raise RuntimeError(
                "Expected edge-level date column "
                "for daily path allocation."
            )

        te["_trip_id"] = (
            trip.trip_id
        )

        retained_intervals.append(
            te[
                [
                    "date",
                    "_trip_id",
                    "_distance_m",
                ]
            ]
        )

    if retained_intervals:
        path_edges = pd.concat(
            retained_intervals,
            ignore_index=True,
        )

        daily_path = (
            path_edges.groupby(
                "date",
                as_index=False,
            )
            .agg(
                new_trip_path_raw_obs_m=(
                    "_distance_m",
                    "sum",
                ),
            )
        )

    else:
        daily_path = pd.DataFrame(
            columns=[
                "date",
                "new_trip_path_raw_obs_m",
            ]
        )


    # --------------------------------------------------------
    # Same-calendar-day endpoint displacement only
    # --------------------------------------------------------

    same_day_trips = (
        trips.loc[
            ~trips[
                "cross_date"
            ].fillna(True)
        ]
        .copy()
    )

    if same_day_trips.empty:
        daily_endpoint = pd.DataFrame(
            columns=[
                "date",
                "new_endpoint_displacement_obs_m",
                "new_n_same_day_trips",
                "new_mean_supported_fraction",
            ]
        )

    else:
        daily_endpoint = (
            same_day_trips.groupby(
                "departure_date",
                as_index=False,
            )
            .agg(
                new_endpoint_displacement_obs_m=(
                    "endpoint_displacement_obs_m",
                    "sum",
                ),
                new_n_same_day_trips=(
                    "trip_id",
                    "count",
                ),
                new_mean_supported_fraction=(
                    "supported_fraction",
                    "mean",
                ),
            )
            .rename(
                columns={
                    "departure_date":
                        "date",
                }
            )
        )


    # --------------------------------------------------------
    # Departure counts, including cross-date trips
    # --------------------------------------------------------

    daily_departures = (
        trips.groupby(
            "departure_date",
            as_index=False,
        )
        .agg(
            new_n_trip_departures=(
                "trip_id",
                "count",
            ),
            new_n_cross_date_departures=(
                "cross_date",
                "sum",
            ),
        )
        .rename(
            columns={
                "departure_date":
                    "date",
            }
        )
    )


    daily_new = (
        daily_path
        .merge(
            daily_endpoint,
            on="date",
            how="outer",
        )
        .merge(
            daily_departures,
            on="date",
            how="outer",
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

else:
    daily_new = pd.DataFrame(
        columns=[
            "date",
            "new_trip_path_raw_obs_m",
            "new_endpoint_displacement_obs_m",
            "new_n_same_day_trips",
            "new_mean_supported_fraction",
            "new_n_trip_departures",
            "new_n_cross_date_departures",
        ]
    )


# Keep old output column name temporarily so downstream
# comparison/printing code still runs.
daily_new[
    "new_place_displacement_obs_m"
] = (
    daily_new[
        "new_endpoint_displacement_obs_m"
    ]
)

daily_new[
    "new_n_trips"
] = (
    daily_new[
        "new_n_same_day_trips"
    ]
)


# ============================================================
# CROSS-DATE TRIP SUMMARY
# ============================================================

print()
print("=" * 100)
print("CROSS-DATE TRIP SUMMARY")
print("=" * 100)

if trips.empty:
    print("No trips.")

else:
    cross = (
        trips.loc[
            trips["cross_date"]
        ]
        .copy()
    )

    print(
        "Cross-date trips:",
        len(cross),
    )

    if cross.empty:
        print("None.")

    else:
        print(
            cross[
                [
                    "trip_id",
                    "trip_type",
                    "departure_date",
                    "arrival_date",
                    "trip_span_days",
                    "trip_elapsed_duration_ms",
                    "supported_fraction",
                    "path_distance_raw_obs_m",
                    "endpoint_displacement_obs_m",
                ]
            ]
            .assign(
                trip_duration_h=lambda x:
                    (
                        x[
                            "trip_elapsed_duration_ms"
                        ]
                        / 3_600_000.0
                    ),
                L_raw_km=lambda x:
                    (
                        x[
                            "path_distance_raw_obs_m"
                        ]
                        / 1000.0
                    ),
                D_endpoint_km=lambda x:
                    (
                        x[
                            "endpoint_displacement_obs_m"
                        ]
                        / 1000.0
                    ),
            )
            [
                [
                    "trip_id",
                    "trip_type",
                    "departure_date",
                    "arrival_date",
                    "trip_span_days",
                    "trip_duration_h",
                    "supported_fraction",
                    "L_raw_km",
                    "D_endpoint_km",
                ]
            ]
            .to_string(
                index=False
            )
        )


daily_old = pd.DataFrame(
    daily_old_feature_rows
)


comparison = (
    daily_old.merge(
        daily_new,
        on="date",
        how="outer",
    )
    .sort_values(
        "date"
    )
    .reset_index(drop=True)
)


comparison[
    "new_minus_old_m"
] = (
    comparison[
        "new_trip_path_raw_obs_m"
    ]
    - comparison[
        "old_supported_displacement_m"
    ]
)


comparison[
    "new_over_old_ratio"
] = np.nan

ratio_mask = (
    pd.to_numeric(
        comparison[
            "old_supported_displacement_m"
        ],
        errors="coerce",
    )
    > 0
)

comparison.loc[
    ratio_mask,
    "new_over_old_ratio",
] = (
    pd.to_numeric(
        comparison.loc[
            ratio_mask,
            "new_trip_path_raw_obs_m",
        ],
        errors="coerce",
    )
    / pd.to_numeric(
        comparison.loc[
            ratio_mask,
            "old_supported_displacement_m",
        ],
        errors="coerce",
    )
)


# ============================================================
# SAVE
# ============================================================

assignments.to_parquet(
    OUT_DIR
    / f"{PARTICIPANT_ID}_place_assignments.parquet",
    index=False,
)

places.to_csv(
    OUT_DIR
    / f"{PARTICIPANT_ID}_places.csv",
    index=False,
)

visits.to_csv(
    OUT_DIR
    / f"{PARTICIPANT_ID}_visits.csv",
    index=False,
)

trips.to_csv(
    OUT_DIR
    / f"{PARTICIPANT_ID}_trips.csv",
    index=False,
)

comparison.to_csv(
    OUT_DIR
    / f"{PARTICIPANT_ID}_daily_trip_distance_comparison.csv",
    index=False,
)


# ============================================================
# PRINT MOST INFORMATIVE TRIPS
# ============================================================

print()
print("=" * 100)
print("FIRST 20 TRIPS")
print("=" * 100)

if trips.empty:
    print("No trips detected.")

else:
    display_cols = [
        "date",
        "trip_id",
        "origin_place_candidate_id",
        "destination_place_candidate_id",
        "trip_type",
        "trip_elapsed_duration_ms",
        "supported_fraction",
        "path_distance_raw_obs_m",
        "endpoint_displacement_obs_m",
        "place_displacement_obs_m",
        "path_complete",
    ]

    z = (
        trips[
            display_cols
        ]
        .head(20)
        .copy()
    )

    z[
        "trip_duration_min"
    ] = (
        z[
            "trip_elapsed_duration_ms"
        ]
        / 60_000.0
    )

    z[
        "L_raw_km"
    ] = (
        z[
            "path_distance_raw_obs_m"
        ]
        / 1000.0
    )

    z[
        "D_km"
    ] = (
        z[
            "endpoint_displacement_obs_m"
        ]
        / 1000.0
    )

    print(
        z[
            [
                "date",
                "trip_id",
                "origin_place_candidate_id",
                "destination_place_candidate_id",
                "trip_type",
                "trip_duration_min",
                "supported_fraction",
                "L_raw_km",
                "D_km",
                "path_complete",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# DAILY COMPARISON
# ============================================================

print()
print("=" * 100)
print(
    "DAILY NEW TRIP PATH VS EXISTING "
    "SUPPORTED DISPLACEMENT"
)
print("=" * 100)

print(
    comparison[
        [
            "date",
            "old_supported_displacement_m",
            "new_trip_path_raw_obs_m",
            "new_place_displacement_obs_m",
            "new_n_trips",
            "new_mean_supported_fraction",
            "new_minus_old_m",
            "new_over_old_ratio",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("OVERALL COMPARISON")
print("=" * 100)

valid = comparison.dropna(
    subset=[
        "old_supported_displacement_m",
        "new_trip_path_raw_obs_m",
    ]
)

if valid.empty:
    print(
        "No overlapping daily values available."
    )

else:
    print(
        "Days compared:",
        len(valid),
    )

    print(
        "Old distance total (km):",
        round(
            valid[
                "old_supported_displacement_m"
            ].sum()
            / 1000.0,
            2,
        ),
    )

    print(
        "New raw observed trip path total (km):",
        round(
            valid[
                "new_trip_path_raw_obs_m"
            ].sum()
            / 1000.0,
            2,
        ),
    )

    print(
        "Median new / old ratio:",
        round(
            float(
                valid[
                    "new_over_old_ratio"
                ].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .median()
            ),
            3,
        ),
    )

    print(
        "Median daily difference (km):",
        round(
            float(
                valid[
                    "new_minus_old_m"
                ].median()
            )
            / 1000.0,
            3,
        ),
    )


print()
print("Saved outputs to:")
print(OUT_DIR)
