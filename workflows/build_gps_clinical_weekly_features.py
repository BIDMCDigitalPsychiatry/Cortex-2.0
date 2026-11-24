"""Build Cortex 2.0 GPS features for 7-day clinical windows.

For each weekly clinical assessment:

    assessment_date - 6 days ... assessment_date

The pipeline is unchanged from the validated GPS reliability work:

raw GPS
→ temporal intervals
→ supported intervals
→ spatial semantics
→ anchor-compatible microsegments
→ place candidates within that 7-day window
→ daily time-weighted location features
→ weekly summary

Important:
- MAX_GAP_MS = 60 seconds
- compatibility_multiplier = 1.0
- place candidates are built independently inside each clinical window
  to avoid using future GPS information.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
    compute_home_away_location_entropy,
)
from cortex2_engine.features.gps_home_shared import calculate_home_time, HomeConfig
from cortex2_engine.features.gps_shared_place_reference import load_reference

from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)

from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)



from cortex2_engine.features.gps_trip_daily_allocation import allocate_trips_to_days
from cortex2_engine.features.gps_pipeline_qc import GPSQCConfig, clean_gps_fixes

from cortex2_engine.features.gps_trip_features import (
    build_place_visits,
    build_observed_trips,
    add_smoothed_trip_path_distances,
)

# CORTEX2_OBSERVED_TRIPS_V1
MIN_ENDPOINT_DWELL_MS = 600_000
TRIP_ACCELERATION_NOISE = 1.0

MAX_GAP_MS = 60_000
COMPATIBILITY_MULTIPLIER = 1.0


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

REQUESTED_PATH = (
    ROOT
    / "gps_clinical_weekly_requested_days.parquet"
)

ASSESSMENT_PATH = (
    ROOT
    / "gps_clinical_weekly_assessments.csv"
)

DAILY_OUT = (
    ROOT
    / "gps_clinical_weekly_daily_features_with_trips.parquet"
)

WEEKLY_OUT = (
    ROOT
    / "gps_clinical_weekly_features_with_trips.parquet"
)

SUMMARY_OUT = (
    ROOT
    / "gps_clinical_weekly_feature_summary_with_trips.csv"
)


# ============================================================
# LOAD MANIFESTS
# ============================================================

requested = pd.read_parquet(
    REQUESTED_PATH
)

assessments = pd.read_csv(
    ASSESSMENT_PATH
)


# CORTEX2_SINGLE_PARTICIPANT_ENTROPY_TEST
import os

test_participant = os.environ.get("CORTEX2_PARTICIPANT_ID")

if test_participant:
    requested = requested.loc[
        requested["participant_id"].astype(str).eq(test_participant)
    ].copy()

    if requested.empty:
        raise ValueError(
            f"No clinical windows for {test_participant}"
        )

    # Never overwrite the primary cohort outputs during testing.
    DAILY_OUT = DAILY_OUT.with_name(
        f"{DAILY_OUT.stem}_{test_participant}{DAILY_OUT.suffix}"
    )
    WEEKLY_OUT = WEEKLY_OUT.with_name(
        f"{WEEKLY_OUT.stem}_{test_participant}{WEEKLY_OUT.suffix}"
    )
    SUMMARY_OUT = SUMMARY_OUT.with_name(
        f"{SUMMARY_OUT.stem}_{test_participant}{SUMMARY_OUT.suffix}"
    )

requested["date"] = (
    pd.to_datetime(
        requested["date"]
    )
    .dt.strftime("%Y-%m-%d")
)

requested["assessment_date"] = (
    pd.to_datetime(
        requested["assessment_date"]
    )
)

assessments["assessment_date"] = (
    pd.to_datetime(
        assessments["assessment_date"]
    )
)


print("=" * 80)
print("Cortex 2.0 GPS CLINICAL WEEKLY FEATURE BUILD")
print("=" * 80)

print(
    "Clinical windows:",
    requested[
        "clinical_window_id"
    ].nunique(),
)

print(
    "Participants:",
    requested[
        "participant_id"
    ].nunique(),
)

print(
    "Requested window-days:",
    len(requested),
)

print(
    "Unique participant-days:",
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
    "Frozen max gap:",
    MAX_GAP_MS,
)

print(
    "Frozen compatibility multiplier:",
    COMPATIBILITY_MULTIPLIER,
)


# ============================================================
# PROCESS PARTICIPANT BY PARTICIPANT
# ============================================================

daily_feature_rows = []
shared_weekly_rows = []


for participant_id in sorted(
    requested[
        "participant_id"
    ].unique()
):

    print()
    print("-" * 80)
    print(participant_id)

    participant_requests = (
        requested[
            requested[
                "participant_id"
            ].eq(participant_id)
        ]
        .copy()
    )

    uid_values = (
        participant_requests[
            "mindlamp_uid"
        ]
        .astype(str)
        .unique()
    )

    if len(uid_values) != 1:
        raise RuntimeError(
            f"{participant_id}: expected one MindLAMP UID, "
            f"found {uid_values}."
        )

    uid = str(
        uid_values[0]
    )

    # Read only this participant from the large raw cache.
    table = pq.read_table(
        RAW_PATH,
        filters=[
            (
                "participant_id",
                "=",
                participant_id,
            )
        ],
    )

    participant_raw = (
        table.to_pandas()
    )

    if len(participant_raw):
        participant_raw["date"] = (
            participant_raw[
                "date"
            ].astype(str)
        )

    unique_days = (
        participant_requests[
            [
                "participant_id",
                "mindlamp_uid",
                "date",
                "start_ms",
                "end_ms",
            ]
        ]
        .drop_duplicates()
        .sort_values("date")
    )

    # --------------------------------------------------------
    # Compute temporal/spatial microsegments ONCE per day.
    # These can then be reused in overlapping clinical windows.
    # --------------------------------------------------------

    day_edges = {}
    day_microsegments = {}
    day_metadata = {}

    for _, day in unique_days.iterrows():

        date = str(
            day["date"]
        )

        start_ms = int(
            day["start_ms"]
        )

        end_ms = int(
            day["end_ms"]
        )

        if len(participant_raw):

            day_raw = participant_raw[
                participant_raw[
                    "date"
                ].eq(date)
            ].copy()

        else:

            day_raw = pd.DataFrame(
                columns=[
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            )

        n_raw_rows = len(
            day_raw
        )

        if n_raw_rows == 0:

            day_metadata[date] = {
                "n_raw_rows": 0,
                "n_unique_timestamps": 0,
                "n_intervals": 0,
                "n_supported_edges": 0,
                "supported_duration_ms": 0.0,
                "supported_fraction_of_day": 0.0,
                "has_raw_gps": False,

                "supported_displacement_distance_m":
                    np.nan,

                "supported_radius_of_gyration_m":
                    np.nan,

                "displacement_edge_count":
                    0,

                "implausible_transition_count":
                    0,

                "implausible_transition_fraction":
                    0.0,

                "spatially_supported_duration_ms":
                    0.0,
            }

            day_microsegments[
                date
            ] = pd.DataFrame()

            continue

        # Shared accuracy/deduplication QC before interval construction.
        day_raw = clean_gps_fixes(day_raw, GPSQCConfig(
            max_gap_ms=MAX_GAP_MS, max_accuracy_m=None))

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

        day_edges[date] = intervals.copy()
        supported = (
            intervals[
                intervals[
                    "supported"
                ]
            ]
            .copy()
        )

        supported_duration_ms = (
            float(supported["duration_ms"].sum())
            if "duration_ms" in supported.columns
            else 0.0
        )

        window_duration_ms = float(
            end_ms
            - start_ms
        )

        supported_fraction = (
            supported_duration_ms
            / window_duration_ms
            if window_duration_ms > 0
            else np.nan
        )

        day_metadata[
            date
        ] = {
            "n_raw_rows":
                n_raw_rows,

            "n_unique_timestamps":
                day_raw[
                    "timestamp"
                ].nunique(),

            "n_intervals":
                len(intervals),

            "n_supported_edges":
                len(supported),

            "supported_duration_ms":
                supported_duration_ms,

            "supported_fraction_of_day":
                supported_fraction,

            "has_raw_gps":
                True,
        }

        if supported.empty:

            day_microsegments[
                date
            ] = pd.DataFrame()

            continue

        spatial = (
            add_gps_spatial_semantics(
                supported
            )
        )

        trajectory = (
            add_gps_trajectory_plausibility(
                spatial,
                max_plausible_speed_kmh=1000.0,
            )
        )

        mobility = (
            compute_daily_mobility_features(
                trajectory
            )
        )

        day_metadata[
            date
        ].update(
            mobility
        )

        microsegments = (
            build_anchor_compatible_location_episodes(
                spatial
            )
        )

        if not microsegments.empty:

            microsegments = (
                microsegments.copy()
            )

            microsegments[
                "participant_id"
            ] = participant_id

            microsegments[
                "mindlamp_uid"
            ] = uid

            microsegments[
                "date"
            ] = date

        day_microsegments[
            date
        ] = microsegments


    # CORTEX2_CANONICAL_SHARED_REFERENCE_V1
    # Shared entropy uses the same persisted participant places and home
    # reference as the daily EMA workflow (retrospective, not prospective).
    shared_assignments, shared_places, shared_home, shared_reference_meta = (
        load_reference(
            participant_id,
            required_dates=unique_days["date"].astype(str).tolist(),
        )
    )

    # --------------------------------------------------------
    # Each clinical window gets its OWN place candidate model.
    # --------------------------------------------------------

    participant_windows = (
        participant_requests[
            [
                "clinical_window_id",
                "assessment_date",
                "cohort",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "assessment_date"
        )
    )

    for _, window in (
        participant_windows.iterrows()
    ):

        window_id = (
            window[
                "clinical_window_id"
            ]
        )

        window_days = (
            participant_requests[
                participant_requests[
                    "clinical_window_id"
                ].eq(window_id)
            ]
            .sort_values(
                "window_day"
            )
        )

        segment_frames = []

        for date in (
            window_days[
                "date"
            ].astype(str)
        ):

            seg = (
                day_microsegments[
                    date
                ]
            )

            if (
                seg is not None
                and not seg.empty
            ):

                segment_frames.append(
                    seg
                )

        if segment_frames:

            window_segments = pd.concat(
                segment_frames,
                ignore_index=True,
            )

            assignments, places = (
                build_place_candidates(
                    window_segments,
                    compatibility_multiplier=(
                        COMPATIBILITY_MULTIPLIER
                    ),
                )
            )

        else:

            assignments = pd.DataFrame(
                columns=[
                    "date",
                    "place_candidate_id",
                    "duration_ms",
                ]
            )

            places = pd.DataFrame()


        # Pooled weekly entropy uses persistent place occupancy,
        # not the mean of daily entropy values.
        shared_window = shared_assignments.loc[
            shared_assignments["date"].astype(str).isin(
                window_days["date"].astype(str)
            )
        ]
        shared_support_ms = sum(
            float(day_metadata[str(d)]["supported_duration_ms"])
            for d in window_days["date"]
        )
        pooled_total = compute_daily_location_features(
            shared_window,
            supported_duration_ms=shared_support_ms,
        )
        pooled_away = compute_home_away_location_entropy(
            shared_window,
            home_place_candidate_id=shared_home["home_place_candidate_id"],
            home_status=shared_home["home_status"],
        )
        # CORTEX2_ENTROPY_COVERAGE_V1
        eligible_dates = []
        for d in window_days["date"].astype(str).unique():
            support = float(
                day_metadata[d]["supported_duration_ms"]
            )
            day_places = shared_window.loc[
                shared_window["date"].astype(str).eq(d)
            ]
            day_total = compute_daily_location_features(
                day_places,
                supported_duration_ms=support,
            )
            if (
                np.isfinite(support)
                and support >= HomeConfig().min_daily_supported_ms
                and day_total["assigned_place_time_ms"] > 0
            ):
                eligible_dates.append(d)

        eligible_window = shared_window.loc[
            shared_window["date"].astype(str).isin(
                eligible_dates
            )
        ]

        eligible_support_ms = sum(
            float(day_metadata[d]["supported_duration_ms"])
            for d in eligible_dates
        )

        eligible_total = compute_daily_location_features(
            eligible_window,
            supported_duration_ms=eligible_support_ms,
        )
        eligible_away = compute_home_away_location_entropy(
            eligible_window,
            home_place_candidate_id=shared_home[
                "home_place_candidate_id"
            ],
            home_status=shared_home["home_status"],
        )

        shared_weekly_rows.append({
            "clinical_window_id": window_id,
            "pooled_location_entropy_nats":
                pooled_total["location_entropy_nats"],
            "pooled_away_location_entropy_nats":
                pooled_away["away_location_entropy_nats"],
            "pooled_home_stationary_fraction":
                pooled_away["home_stationary_fraction"],
            "pooled_away_place_count":
                pooled_away["away_place_count"],
            "pooled_away_effective_number_places":
                pooled_away["away_effective_number_places"],
            "pooled_entropy_decomposition_residual":
                pooled_away["entropy_decomposition_residual"],
            "pooled_away_location_entropy_status":
                pooled_away["away_location_entropy_status"],
            "pooled_place_time_fraction":
                pooled_total["place_time_fraction_of_supported"],
            "shared_home_status": shared_home["home_status"],
            "shared_home_place_candidate_id": shared_home["home_place_candidate_id"],
            "shared_home_confidence": shared_home.get("home_confidence", np.nan),
            "shared_place_reference_scope": "retrospective_union_of_configured_raw_sources",
            "pooled_eligible_days": len(eligible_dates),
            "pooled_eligible_supported_duration_ms":
                eligible_support_ms,
            "pooled_eligible_location_entropy_nats":
                eligible_total["location_entropy_nats"],
            "pooled_eligible_place_count":
                eligible_total["n_observed_places"],
            "pooled_eligible_effective_number_places":
                eligible_total["effective_number_places"],
            "pooled_eligible_away_place_count":
                eligible_away["away_place_count"],
            "pooled_eligible_away_effective_number_places":
                eligible_away["away_effective_number_places"],
            "pooled_eligible_entropy_decomposition_residual":
                eligible_away["entropy_decomposition_residual"],
            "pooled_eligible_away_location_entropy_nats":
                eligible_away["away_location_entropy_nats"],
            "pooled_eligible_home_stationary_fraction":
                eligible_away["home_stationary_fraction"],
            "pooled_eligible_away_entropy_status":
                eligible_away["away_location_entropy_status"],
            "pooled_eligible_place_time_fraction":
                eligible_total[
                    "place_time_fraction_of_supported"
                ],
        })

        trip_daily = {}

        edge_frames = [
            day_edges[d]
            for d in window_days["date"].astype(str)
            if d in day_edges and not day_edges[d].empty
        ]

        if edge_frames and not assignments.empty and not places.empty:
            window_edges = pd.concat(
                edge_frames, ignore_index=True
            )

            window_edges = add_gps_spatial_semantics(
                window_edges
            )

            window_edges = add_gps_trajectory_plausibility(
                window_edges,
                max_plausible_speed_kmh=1000.0,
            )

            visits = build_place_visits(
                assignments,
                min_endpoint_dwell_ms=MIN_ENDPOINT_DWELL_MS,
            )

            trips = build_observed_trips(
                window_edges,
                visits,
                places,
            )

            if not trips.empty:
                trips = add_smoothed_trip_path_distances(
                    window_edges,
                    trips,
                    acceleration_noise_density_m_s1p5=(
                        TRIP_ACCELERATION_NOISE
                    ),
                )

                # Allocate cross-midnight trips rather than dropping them.
                # Daily length is a time-proportional proxy; weekly totals
                # preserve the whole-trip estimates.
                allocated = allocate_trips_to_days(trips, edges=window_edges)
                for item in allocated.to_dict("records"):
                    trip_daily[item.pop("date")] = item

        n_window_places = (
            len(places)
            if places is not None
            else 0
        )

        for _, day in (
            window_days.iterrows()
        ):

            date = str(
                day["date"]
            )

            meta = (
                day_metadata[
                    date
                ]
            )

            if assignments.empty:

                day_assignments = (
                    pd.DataFrame(
                        columns=[
                            "place_candidate_id",
                            "duration_ms",
                        ]
                    )
                )

            else:

                day_assignments = (
                    assignments[
                        assignments[
                            "date"
                        ].astype(str)
                        .eq(date)
                    ]
                )

            result = (
                compute_daily_location_features(
                    day_assignments,
                    supported_duration_ms=(
                        meta[
                            "supported_duration_ms"
                        ]
                    ),
                )
            )

            # Daily shared-place entropy, using the same
            # participant-level home and place identifiers.
            shared_day = shared_window.loc[
                shared_window["date"].astype(str).eq(date)
            ]
            shared_total = compute_daily_location_features(
                shared_day,
                supported_duration_ms=meta["supported_duration_ms"],
            )
            shared_away = compute_home_away_location_entropy(
                shared_day,
                home_place_candidate_id=shared_home["home_place_candidate_id"],
                home_status=shared_home["home_status"],
            )
            result["shared_location_entropy_nats"] = (
                shared_total["location_entropy_nats"]
            )
            result.update(shared_away)
            result["shared_place_time_fraction"] = (
                shared_total["place_time_fraction_of_supported"]
            )
            result["shared_home_status"] = shared_home["home_status"]
            result["shared_home_place_candidate_id"] = shared_home["home_place_candidate_id"]
            result["shared_home_confidence"] = shared_home.get("home_confidence", np.nan)
            result["shared_place_reference_scope"] = "retrospective_union_of_configured_raw_sources"

            # CORTEX2_WEEKLY_DAILY_ENTROPY_ELIGIBILITY_V1
            day_support = float(meta["supported_duration_ms"])
            eligible = (
                np.isfinite(day_support)
                and day_support >= HomeConfig().min_daily_supported_ms
                and shared_total["assigned_place_time_ms"] > 0
            )
            result["shared_entropy_day_eligible"] = bool(eligible)
            result["shared_entropy_coverage_status"] = (
                "eligible" if eligible
                else "insufficient_supported_time"
                if not np.isfinite(day_support)
                or day_support < HomeConfig().min_daily_supported_ms
                else "no_assigned_place_time"
            )
            result["shared_eligible_location_entropy_nats"] = (
                shared_total["location_entropy_nats"]
                if eligible else np.nan
            )
            result["shared_eligible_away_entropy_nats"] = (
                shared_away["away_location_entropy_nats"]
                if eligible else np.nan
            )


            result.update(
                {
                    "clinical_window_id":
                        window_id,

                    "participant_id":
                        participant_id,

                    "mindlamp_uid":
                        uid,

                    "cohort":
                        window[
                            "cohort"
                        ],

                    "assessment_date":
                        window[
                            "assessment_date"
                        ],

                    "date":
                        date,

                    "window_day":
                        int(
                            day[
                                "window_day"
                            ]
                        ),

                    "compatibility_multiplier":
                        COMPATIBILITY_MULTIPLIER,

                    "n_window_place_candidates":
                        n_window_places,

                    "has_raw_gps":
                        meta[
                            "has_raw_gps"
                        ],

                    "supported_fraction_of_day":
                        meta[
                            "supported_fraction_of_day"
                        ],

                    "supported_duration_ms":
                        meta[
                            "supported_duration_ms"
                        ],

                    "n_raw_rows":
                        meta[
                            "n_raw_rows"
                        ],

                    "n_unique_timestamps":
                        meta[
                            "n_unique_timestamps"
                        ],

                    "n_intervals":
                        meta[
                            "n_intervals"
                        ],

                    "n_supported_edges":
                        meta[
                            "n_supported_edges"
                        ],

                    "supported_displacement_distance_m":
                        meta.get(
                            "supported_displacement_distance_m",
                            np.nan,
                        ),

                    "supported_radius_of_gyration_m":
                        meta.get(
                            "supported_radius_of_gyration_m",
                            np.nan,
                        ),

                    "displacement_edge_count":
                        meta.get(
                            "displacement_edge_count",
                            np.nan,
                        ),

                    "implausible_transition_count":
                        meta.get(
                            "implausible_transition_count",
                            0,
                        ),

                    "implausible_transition_fraction":
                        meta.get(
                            "implausible_transition_fraction",
                            np.nan,
                        ),

                    "spatially_supported_duration_ms":
                        meta.get(
                            "spatially_supported_duration_ms",
                            np.nan,
                        ),
                }
            )


            trip_values = trip_daily.get(date)

            if trip_values is not None:
                result.update(trip_values)
            else:
                has_support = (
                    meta["supported_duration_ms"] > 0
                )

                result.update({
                    "trip_count_obs": (
                        0 if has_support and not assignments.empty and not places.empty else np.nan
                    ),
                    "trip_path_length_obs_m": (
                        0.0 if has_support and not assignments.empty and not places.empty else np.nan
                    ),
                    "trip_path_raw_obs_m": (
                        0.0 if has_support and not assignments.empty and not places.empty else np.nan
                    ),
                    "trip_displacement_obs_m": (
                        0.0 if has_support and not assignments.empty and not places.empty else np.nan
                    ),
                    "trip_endpoint_displacement_obs_m": (
                        0.0 if has_support and not assignments.empty and not places.empty else np.nan
                    ),
                    "trip_smoothing_coverage": np.nan,
                    "trip_allocation_proxy_method": "no_trips",
                    "cross_midnight_trip": False,
                })

            daily_feature_rows.append(
                result
            )

    print(
        "  unique days:",
        len(unique_days),
        "| clinical windows:",
        len(participant_windows),
    )


# ============================================================
# DAILY OUTPUT
# ============================================================

daily = pd.DataFrame(
    daily_feature_rows
)

if len(daily) != len(requested):

    raise RuntimeError(
        f"Expected {len(requested)} window-day rows, "
        f"found {len(daily)}."
    )

if daily[
    [
        "clinical_window_id",
        "date",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate clinical-window/day rows found."
    )


# ============================================================
# WEEKLY AGGREGATION
# ============================================================

# A feature day is usable only when the derived feature exists.
daily[
    "feature_day_usable"
] = (
    daily[
        [
            "location_entropy_nats",
            "effective_number_places",
            "dominant_place_fraction",
        ]
    ]
    .notna()
    .all(axis=1)
)


weekly = (
    daily.groupby(
        [
            "clinical_window_id",
            "participant_id",
            "mindlamp_uid",
            "cohort",
            "assessment_date",
        ],
        as_index=False,
    )
    .agg(
        requested_days=(
            "date",
            "size",
        ),

        days_with_raw=(
            "has_raw_gps",
            "sum",
        ),

        usable_feature_days=(
            "feature_day_usable",
            "sum",
        ),

        mean_supported_fraction=(
            "supported_fraction_of_day",
            "mean",
        ),

        median_supported_fraction=(
            "supported_fraction_of_day",
            "median",
        ),

        location_entropy_mean=(
            "location_entropy_nats",
            "mean",
        ),

        location_entropy_median=(
            "location_entropy_nats",
            "median",
        ),

        effective_places_mean=(
            "effective_number_places",
            "mean",
        ),

        effective_places_median=(
            "effective_number_places",
            "median",
        ),

        dominant_place_fraction_mean=(
            "dominant_place_fraction",
            "mean",
        ),

        dominant_place_fraction_median=(
            "dominant_place_fraction",
            "median",
        ),

        observed_places_mean=(
            "n_observed_places",
            "mean",
        ),

        observed_places_median=(
            "n_observed_places",
            "median",
        ),

        displacement_distance_mean_m=(
            "supported_displacement_distance_m",
            "mean",
        ),

        displacement_distance_median_m=(
            "supported_displacement_distance_m",
            "median",
        ),

        radius_of_gyration_mean_m=(
            "supported_radius_of_gyration_m",
            "mean",
        ),

        radius_of_gyration_median_m=(
            "supported_radius_of_gyration_m",
            "median",
        ),
    )
)



weekly = weekly.merge(
    pd.DataFrame(shared_weekly_rows),
    on="clinical_window_id",
    how="left",
    validate="one_to_one",
)

trip_weekly = (
    daily.groupby("clinical_window_id", as_index=False)
    .agg(
        trip_count_obs_weekly=(
            "trip_count_obs",
            lambda x: x.sum(min_count=1),
        ),
        trip_path_length_obs_weekly_m=(
            "trip_path_length_obs_m",
            lambda x: x.sum(min_count=1),
        ),
        trip_path_raw_obs_weekly_m=(
            "trip_path_raw_obs_m",
            lambda x: x.sum(min_count=1),
        ),
        trip_displacement_obs_weekly_m=(
            "trip_displacement_obs_m",
            lambda x: x.sum(min_count=1),
        ),
        trip_days_with_estimates=(
            "trip_path_length_obs_m",
            "count",
        ),
        trip_smoothing_coverage_mean=(
            "trip_smoothing_coverage",
            "mean",
        ),
    )
)

weekly = weekly.merge(
    trip_weekly,
    on="clinical_window_id",
    how="left",
    validate="one_to_one",
)

weekly = weekly.merge(
    assessments[
        [
            "clinical_window_id",
            "depression_score",
            "anxiety_score",
        ]
    ],
    on="clinical_window_id",
    how="left",
)


# ============================================================
# SAVE
# ============================================================

daily.to_parquet(
    DAILY_OUT,
    index=False,
)

weekly.to_parquet(
    WEEKLY_OUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

coverage_summary = []

for minimum_days in [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]:

    x = weekly[
        weekly[
            "usable_feature_days"
        ]
        >= minimum_days
    ]

    coverage_summary.append(
        {
            "minimum_usable_days":
                minimum_days,

            "clinical_windows":
                len(x),

            "participants":
                x[
                    "participant_id"
                ].nunique(),

            "depression_windows":
                x[
                    "depression_score"
                ].notna().sum(),

            "anxiety_windows":
                x[
                    "anxiety_score"
                ].notna().sum(),
        }
    )

summary = pd.DataFrame(
    coverage_summary
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


print()
print("=" * 80)
print("WEEKLY GPS FEATURE SUMMARY")
print("=" * 80)

print(
    "Daily window rows:",
    len(daily),
)

print(
    "Clinical windows:",
    len(weekly),
)

print(
    "Participants:",
    weekly[
        "participant_id"
    ].nunique(),
)

print(
    "\nUsable feature days per 7-day window:"
)

print(
    weekly[
        "usable_feature_days"
    ]
    .describe()
    .to_string()
)

print(
    "\nWindow availability by minimum usable days:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved daily features:",
    DAILY_OUT,
)

print(
    "Saved weekly features:",
    WEEKLY_OUT,
)

print(
    "Saved coverage summary:",
    SUMMARY_OUT,
)
