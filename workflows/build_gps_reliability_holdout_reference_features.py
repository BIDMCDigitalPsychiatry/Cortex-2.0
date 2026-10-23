"""Build frozen full-data GPS location features for held-out validation.

Validation design is already locked:
- 16 held-out participants
- 7 reference days per participant
- 112 participant-days
- RAPHE 60-second continuity policy
- compatibility_multiplier = 2.0

No parameter tuning is performed here.

Pipeline
--------
raw GPS
→ temporal intervals
→ supported intervals
→ spatial semantics
→ fixed-anchor microsegments
→ participant-level place candidates
→ daily time-weighted location features
"""

from pathlib import Path

import numpy as np
import pandas as pd

from raphe.features.gps_intervals import (
    build_gps_intervals,
)

from raphe.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)

from raphe.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)

from raphe.features.gps_place_candidates import (
    build_place_candidates,
)

from raphe.features.gps_location_features import (
    compute_daily_location_features,
)


MAX_GAP_MS = 60_000
COMPATIBILITY_MULTIPLIER = 2.0


ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

RAW_PATH = (
    ROOT
    / "gps_reliability_holdout_raw_points.parquet"
)

FEATURE_OUT = (
    ROOT
    / "gps_reliability_holdout_reference_location_features.parquet"
)

ASSIGNMENT_OUT = (
    ROOT
    / "gps_reliability_holdout_reference_place_assignments.parquet"
)

PLACE_OUT = (
    ROOT
    / "gps_reliability_holdout_reference_place_candidates.parquet"
)

SUMMARY_OUT = (
    ROOT
    / "gps_reliability_holdout_reference_feature_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

raw = pd.read_parquet(
    RAW_PATH
)

raw["date"] = (
    raw["date"]
    .astype(str)
)


print("=" * 80)
print("RAPHE INDEPENDENT RELIABILITY HOLDOUT GPS REFERENCE FEATURE BUILD")
print("=" * 80)

print(
    "Raw rows:",
    f"{len(raw):,}",
)

print(
    "Participants:",
    raw[
        "participant_id"
    ].nunique(),
)

print(
    "Participant-days:",
    raw[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "Frozen compatibility multiplier:",
    COMPATIBILITY_MULTIPLIER,
)


# ============================================================
# RAW GPS → MICROSEGMENTS
# ============================================================

participant_microsegments = {}
day_metadata = {}


for (
    participant_id,
    uid,
), participant_raw in raw.groupby(
    [
        "participant_id",
        "mindlamp_uid",
    ],
    sort=True,
):

    print()
    print("-" * 80)
    print(
        participant_id,
        uid,
    )


    microsegment_frames = []


    for date, day_raw in participant_raw.groupby(
        "date",
        sort=True,
    ):

        start_values = (
            day_raw[
                "day_start_ms"
            ]
            .dropna()
            .unique()
        )

        end_values = (
            day_raw[
                "day_end_ms"
            ]
            .dropna()
            .unique()
        )

        reference_observability_values = (
            day_raw[
                "reference_observability"
            ]
            .dropna()
            .unique()
        )

        validation_day_values = (
            day_raw[
                "holdout_day_number"
            ]
            .dropna()
            .unique()
        )

        quartile_values = (
            day_raw[
                "sampling_density_quartile"
            ]
            .dropna()
            .unique()
        )


        if len(start_values) != 1:
            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected exactly one day_start_ms."
            )

        if len(end_values) != 1:
            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected exactly one day_end_ms."
            )

        if len(reference_observability_values) != 1:
            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected exactly one reference observability."
            )


        start_ms = int(
            start_values[0]
        )

        end_ms = int(
            end_values[0]
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


        supported = (
            intervals[
                intervals[
                    "supported"
                ]
            ]
            .copy()
        )


        supported_duration_ms = float(
            supported[
                "duration_ms"
            ].sum()
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


        reference_observability = float(
            reference_observability_values[0]
        )


        day_metadata[
            (
                participant_id,
                str(date),
            )
        ] = {
            "mindlamp_uid":
                uid,

            "sampling_density_quartile":
                str(
                    quartile_values[0]
                ),

            "holdout_day_number":
                int(
                    validation_day_values[0]
                ),

            "start_ms":
                start_ms,

            "end_ms":
                end_ms,

            "reference_observability":
                reference_observability,

            "supported_fraction_of_day":
                supported_fraction,

            "observability_difference":
                (
                    supported_fraction
                    - reference_observability
                ),

            "n_raw_rows":
                len(
                    day_raw
                ),

            "n_unique_timestamps":
                day_raw[
                    "timestamp"
                ].nunique(),

            "n_intervals":
                len(
                    intervals
                ),

            "n_supported_edges":
                len(
                    supported
                ),

            "supported_duration_ms":
                supported_duration_ms,
        }


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


        microsegments[
            "participant_id"
        ] = participant_id

        microsegments[
            "mindlamp_uid"
        ] = uid

        microsegments[
            "date"
        ] = str(
            date
        )


        microsegment_frames.append(
            microsegments
        )


    if microsegment_frames:

        participant_segments = pd.concat(
            microsegment_frames,
            ignore_index=True,
        )

    else:

        participant_segments = pd.DataFrame()


    participant_microsegments[
        participant_id
    ] = participant_segments


    participant_meta = [
        value
        for (
            pid,
            _
        ), value
        in day_metadata.items()
        if pid == participant_id
    ]


    median_supported = float(
        np.median(
            [
                x[
                    "supported_fraction_of_day"
                ]
                for x in participant_meta
            ]
        )
    )


    print(
        "  days:",
        len(
            participant_meta
        ),
    )

    print(
        "  microsegments:",
        f"{len(participant_segments):,}",
    )

    print(
        "  median supported fraction:",
        f"{median_supported:.4f}",
    )


# ============================================================
# MICROSEGMENTS → FROZEN PARTICIPANT-LEVEL PLACES
# ============================================================

feature_rows = []
assignment_frames = []
place_frames = []


for participant_id, microsegments in (
    participant_microsegments.items()
):

    dates = sorted(
        raw.loc[
            raw[
                "participant_id"
            ]
            == participant_id,
            "date",
        ]
        .unique()
        .tolist()
    )


    if microsegments.empty:

        assignments = pd.DataFrame(
            columns=[
                "date",
                "place_candidate_id",
                "duration_ms",
            ]
        )

        places = pd.DataFrame()

    else:

        assignments, places = (
            build_place_candidates(
                microsegments,
                compatibility_multiplier=(
                    COMPATIBILITY_MULTIPLIER
                ),
            )
        )


    if not assignments.empty:

        assignments = (
            assignments
            .copy()
        )

        assignments[
            "participant_id"
        ] = participant_id

        assignments[
            "compatibility_multiplier"
        ] = COMPATIBILITY_MULTIPLIER

        assignment_frames.append(
            assignments
        )


    if not places.empty:

        places = places.copy()

        places[
            "participant_id"
        ] = participant_id

        places[
            "compatibility_multiplier"
        ] = COMPATIBILITY_MULTIPLIER

        place_frames.append(
            places
        )


    for date in dates:

        meta = day_metadata[
            (
                participant_id,
                str(date),
            )
        ]


        if assignments.empty:

            day_assignments = pd.DataFrame(
                columns=[
                    "place_candidate_id",
                    "duration_ms",
                ]
            )

        else:

            day_assignments = assignments[
                assignments[
                    "date"
                ].astype(str)
                == str(date)
            ]


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


        result.update({
            "participant_id":
                participant_id,

            "mindlamp_uid":
                meta[
                    "mindlamp_uid"
                ],

            "date":
                str(
                    date
                ),

            "holdout_day_number":
                meta[
                    "holdout_day_number"
                ],

            "sampling_density_quartile":
                meta[
                    "sampling_density_quartile"
                ],

            "compatibility_multiplier":
                COMPATIBILITY_MULTIPLIER,

            "reference_observability":
                meta[
                    "reference_observability"
                ],

            "supported_fraction_of_day":
                meta[
                    "supported_fraction_of_day"
                ],

            "observability_difference":
                meta[
                    "observability_difference"
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
        })


        feature_rows.append(
            result
        )


features = pd.DataFrame(
    feature_rows
)


# ============================================================
# SAVE
# ============================================================

features.to_parquet(
    FEATURE_OUT,
    index=False,
)


if assignment_frames:

    all_assignments = pd.concat(
        assignment_frames,
        ignore_index=True,
    )

else:

    all_assignments = pd.DataFrame()


all_assignments.to_parquet(
    ASSIGNMENT_OUT,
    index=False,
)


if place_frames:

    all_places = pd.concat(
        place_frames,
        ignore_index=True,
    )

else:

    all_places = pd.DataFrame()


all_places.to_parquet(
    PLACE_OUT,
    index=False,
)


# ============================================================
# VALIDATION
# ============================================================

if len(features) != 112:

    raise RuntimeError(
        f"Expected 112 feature rows, "
        f"found {len(features)}."
    )


if (
    features[
        "participant_id"
    ].nunique()
    != 16
):

    raise RuntimeError(
        "Expected 16 participants."
    )


if (
    features[
        [
            "participant_id",
            "date",
        ]
    ]
    .duplicated()
    .any()
):

    raise RuntimeError(
        "Duplicate participant-day feature rows found."
    )


max_observability_difference = float(
    features[
        "observability_difference"
    ]
    .abs()
    .max()
)


# ============================================================
# SUMMARY BY SAMPLING STRATUM
# ============================================================

summary = (
    features
    .groupby(
        "sampling_density_quartile",
        as_index=False,
    )
    .agg(
        participants=(
            "participant_id",
            "nunique",
        ),

        days=(
            "date",
            "size",
        ),

        median_supported_fraction=(
            "supported_fraction_of_day",
            "median",
        ),

        median_entropy=(
            "location_entropy_nats",
            "median",
        ),

        median_effective_places=(
            "effective_number_places",
            "median",
        ),

        median_dominant_fraction=(
            "dominant_place_fraction",
            "median",
        ),

        median_observed_places=(
            "n_observed_places",
            "median",
        ),

        median_place_time_fraction=(
            "place_time_fraction_of_supported",
            "median",
        ),
    )
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("REFERENCE BUILD CHECK")
print("=" * 80)

print(
    "Feature rows:",
    len(
        features
    ),
)

print(
    "Participants:",
    features[
        "participant_id"
    ].nunique(),
)

print(
    "Participant-days:",
    features[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "Maximum |interval observability - "
    "locked observability|:",
    f"{max_observability_difference:.12f}",
)

print(
    "Minimum supported fraction:",
    f"{features['supported_fraction_of_day'].min():.4f}",
)

print(
    "Minimum place-time fraction of supported:",
    f"{features['place_time_fraction_of_supported'].min():.4f}",
)


print()
print("=" * 80)
print("REFERENCE FEATURE SUMMARY BY DENSITY QUARTILE")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(FEATURE_OUT)
print(ASSIGNMENT_OUT)
print(PLACE_OUT)
print(SUMMARY_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
