"""Validate complete Cortex 2.0 GPS reconstruction from cached raw observations.

Purpose
-------
Rebuild the complete Cortex 2.0 location pipeline from the cached raw GPS points:

raw GPS
→ temporal intervals
→ supported edges
→ spatial semantics
→ fixed-anchor microsegments
→ participant-level candidate places
→ daily time-weighted location features

The rebuilt features are compared with the previously generated reference
features. This must be checked before controlled missingness is introduced.
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

from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
)


DAY_MS = 86_400_000
MAX_GAP_MS = 60_000

MULTIPLIERS = [
    0.75,
    1.00,
    1.50,
    2.00,
]


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

RAW_PATH = (
    ROOT
    / "gps_reference_raw_points.parquet"
)

REFERENCE_PATH = (
    ROOT
    / "gps_reference_location_features_daily.parquet"
)

REBUILT_OUT = (
    ROOT
    / "gps_reference_location_features_rebuilt_from_raw.parquet"
)

COMPARISON_OUT = (
    ROOT
    / "gps_reference_rebuild_comparison.csv"
)


# ============================================================
# LOAD
# ============================================================

raw = pd.read_parquet(
    RAW_PATH
)

reference = pd.read_parquet(
    REFERENCE_PATH
)


raw["date"] = (
    raw["date"]
    .astype(str)
)

reference["date"] = (
    reference["date"]
    .astype(str)
)


print("=" * 80)
print("Cortex 2.0 COMPLETE-DATA REBUILD VALIDATION")
print("=" * 80)

print(
    "Raw rows:",
    f"{len(raw):,}",
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
    "Reference feature rows:",
    len(reference),
)


# ============================================================
# REBUILD MICROSEGMENTS FROM RAW GPS
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

        day_start_values = (
            day_raw[
                "day_start_ms"
            ]
            .dropna()
            .unique()
        )


        if len(
            day_start_values
        ) != 1:

            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected exactly one day_start_ms"
            )


        day_start_ms = int(
            day_start_values[
                0
            ]
        )

        day_end_ms = (
            day_start_ms
            + DAY_MS
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
            window_start_ms=(
                day_start_ms
            ),
            window_end_ms=(
                day_end_ms
            ),
            max_gap_ms=(
                MAX_GAP_MS
            ),
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


        day_metadata[
            (
                participant_id,
                str(date),
            )
        ] = {
            "mindlamp_uid":
                uid,

            "supported_duration_ms":
                supported_duration_ms,

            "n_raw_rows":
                len(
                    day_raw
                ),

            "n_intervals":
                len(
                    intervals
                ),

            "n_supported_edges":
                len(
                    supported
                ),
        }


        if supported.empty:
            continue


        supported = (
            add_gps_spatial_semantics(
                supported
            )
        )


        microsegments = (
            build_anchor_compatible_location_episodes(
                supported
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

        all_microsegments = pd.concat(
            microsegment_frames,
            ignore_index=True,
        )

    else:

        all_microsegments = pd.DataFrame()


    participant_microsegments[
        participant_id
    ] = all_microsegments


    print(
        "  days:",
        participant_raw[
            "date"
        ].nunique(),
    )

    print(
        "  microsegments:",
        f"{len(all_microsegments):,}",
    )


# ============================================================
# REBUILD PLACES + DAILY FEATURES
# ============================================================

feature_rows = []


for participant_id, microsegments in (
    participant_microsegments.items()
):

    participant_days = sorted(
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


    for multiplier in MULTIPLIERS:

        if microsegments.empty:

            assignments = pd.DataFrame(
                columns=[
                    "date",
                    "place_candidate_id",
                    "duration_ms",
                ]
            )

        else:

            assignments, _ = (
                build_place_candidates(
                    microsegments,
                    compatibility_multiplier=(
                        multiplier
                    ),
                )
            )


        for date in participant_days:

            metadata = day_metadata[
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
                        metadata[
                            "supported_duration_ms"
                        ]
                    ),
                )
            )


            result.update({
                "compatibility_multiplier":
                    float(
                        multiplier
                    ),

                "participant_id":
                    participant_id,

                "mindlamp_uid":
                    metadata[
                        "mindlamp_uid"
                    ],

                "date":
                    str(
                        date
                    ),

                "n_raw_rows":
                    metadata[
                        "n_raw_rows"
                    ],

                "n_intervals":
                    metadata[
                        "n_intervals"
                    ],

                "n_supported_edges":
                    metadata[
                        "n_supported_edges"
                    ],
            })


            feature_rows.append(
                result
            )


rebuilt = pd.DataFrame(
    feature_rows
)


rebuilt.to_parquet(
    REBUILT_OUT,
    index=False,
)


# ============================================================
# COMPARE TO EXISTING REFERENCE
# ============================================================

metrics = [
    "location_entropy_nats",
    "effective_number_places",
    "dominant_place_time_ms",
    "dominant_place_fraction",
    "n_observed_places",
    "assigned_place_time_ms",
    "place_time_fraction_of_supported",
]


reference_keep = reference[
    [
        "compatibility_multiplier",
        "participant_id",
        "date",
    ]
    + metrics
].copy()


comparison = rebuilt.merge(
    reference_keep,
    on=[
        "compatibility_multiplier",
        "participant_id",
        "date",
    ],
    how="outer",
    suffixes=(
        "_rebuilt",
        "_reference",
    ),
    indicator=True,
    validate="one_to_one",
)


comparison_rows = []


for metric in metrics:

    rebuilt_column = (
        metric
        + "_rebuilt"
    )

    reference_column = (
        metric
        + "_reference"
    )


    a = pd.to_numeric(
        comparison[
            rebuilt_column
        ],
        errors="coerce",
    )

    b = pd.to_numeric(
        comparison[
            reference_column
        ],
        errors="coerce",
    )


    difference = (
        a
        - b
    ).abs()


    finite_difference = difference[
        difference.notna()
        & np.isfinite(
            difference
        )
    ]


    comparison_rows.append({
        "metric":
            metric,

        "rows":
            len(
                comparison
            ),

        "matching_nonmissing":
            int(
                (
                    a.notna()
                    & b.notna()
                )
                .sum()
            ),

        "max_abs_difference":
            (
                float(
                    finite_difference.max()
                )
                if len(
                    finite_difference
                )
                else np.nan
            ),

        "median_abs_difference":
            (
                float(
                    finite_difference.median()
                )
                if len(
                    finite_difference
                )
                else np.nan
            ),
    })


comparison_summary = pd.DataFrame(
    comparison_rows
)


comparison_summary.to_csv(
    COMPARISON_OUT,
    index=False,
)


print()
print("=" * 80)
print("JOIN CHECK")
print("=" * 80)

print(
    comparison[
        "_merge"
    ]
    .value_counts()
    .to_string()
)


print()
print("=" * 80)
print("FEATURE REBUILD DIFFERENCES")
print("=" * 80)

print(
    comparison_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.12f}",
    )
)


print()
print("=" * 80)
print("REBUILT DATASET")
print("=" * 80)

print(
    "Rows:",
    len(
        rebuilt
    ),
)

print(
    "Participants:",
    rebuilt[
        "participant_id"
    ].nunique(),
)

print(
    "Days:",
    rebuilt[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    REBUILT_OUT
)

print(
    COMPARISON_OUT
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
