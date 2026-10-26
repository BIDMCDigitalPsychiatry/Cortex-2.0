"""Audit Cortex 2.0 time-weighted GPS location features.

Computes daily location entropy and dominant-place occupancy across the
candidate-place sensitivity models.

No Cortex feature functions are used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
)


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

ASSIGNMENT_PATH = (
    ROOT
    / "gps_reference_place_candidate_assignments.parquet"
)

DAILY_OBSERVABILITY_PATH = (
    ROOT
    / "gps_reference_location_episode_daily.csv"
)

OUT_PATH = (
    ROOT
    / "gps_reference_location_features_daily.parquet"
)

SUMMARY_OUT = (
    ROOT
    / "gps_reference_location_feature_summary.csv"
)

SENSITIVITY_OUT = (
    ROOT
    / "gps_reference_location_feature_sensitivity_vs_1.csv"
)


assignments = pd.read_parquet(
    ASSIGNMENT_PATH
)

daily_obs = pd.read_csv(
    DAILY_OBSERVABILITY_PATH
)


assignments["date"] = (
    assignments["date"]
    .astype(str)
)

daily_obs["date"] = (
    daily_obs["date"]
    .astype(str)
)


print("=" * 80)
print("Cortex 2.0 TIME-WEIGHTED LOCATION FEATURE AUDIT")
print("=" * 80)

print(
    "Assignment rows:",
    f"{len(assignments):,}",
)

print(
    "Participants:",
    assignments[
        "participant_id"
    ].nunique(),
)

print(
    "Multipliers:",
    sorted(
        assignments[
            "compatibility_multiplier"
        ]
        .unique()
        .tolist()
    ),
)


# ============================================================
# DAILY FEATURES
# ============================================================

rows = []


for (
    multiplier,
    participant_id,
    uid,
    date,
), x in assignments.groupby(
    [
        "compatibility_multiplier",
        "participant_id",
        "mindlamp_uid",
        "date",
    ],
    sort=True,
):

    obs = daily_obs[
        (
            daily_obs[
                "participant_id"
            ]
            == participant_id
        )
        & (
            daily_obs[
                "date"
            ]
            == date
        )
    ]


    supported_duration_ms = (
        float(
            obs[
                "supported_duration_ms"
            ]
            .iloc[0]
        )
        if len(obs)
        else None
    )


    result = (
        compute_daily_location_features(
            x,
            supported_duration_ms=(
                supported_duration_ms
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
            uid,

        "date":
            date,
    })


    rows.append(
        result
    )


daily = pd.DataFrame(
    rows
)


daily[
    "assigned_place_time_hours"
] = (
    daily[
        "assigned_place_time_ms"
    ]
    / 3_600_000
)

daily[
    "dominant_place_time_hours"
] = (
    daily[
        "dominant_place_time_ms"
    ]
    / 3_600_000
)


daily.to_parquet(
    OUT_PATH,
    index=False,
)


# ============================================================
# PARTICIPANT × MULTIPLIER SUMMARY
# ============================================================

summary_rows = []


for (
    multiplier,
    participant_id,
), x in daily.groupby(
    [
        "compatibility_multiplier",
        "participant_id",
    ],
    sort=True,
):

    summary_rows.append({
        "compatibility_multiplier":
            multiplier,

        "participant_id":
            participant_id,

        "days":
            len(x),

        "median_entropy_nats":
            x[
                "location_entropy_nats"
            ].median(),

        "median_effective_places":
            x[
                "effective_number_places"
            ].median(),

        "median_dominant_fraction":
            x[
                "dominant_place_fraction"
            ].median(),

        "median_dominant_hours":
            x[
                "dominant_place_time_hours"
            ].median(),

        "median_observed_places":
            x[
                "n_observed_places"
            ].median(),

        "p90_observed_places":
            x[
                "n_observed_places"
            ].quantile(
                0.90
            ),

        "median_assigned_hours":
            x[
                "assigned_place_time_hours"
            ].median(),

        "median_place_time_fraction_supported":
            x[
                "place_time_fraction_of_supported"
            ].median(),
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# SENSITIVITY RELATIVE TO MULTIPLIER = 1.0
# ============================================================

reference = (
    daily[
        daily[
            "compatibility_multiplier"
        ]
        == 1.0
    ][
        [
            "participant_id",
            "date",
            "location_entropy_nats",
            "dominant_place_fraction",
            "effective_number_places",
        ]
    ]
    .rename(
        columns={
            "location_entropy_nats":
                "entropy_reference",

            "dominant_place_fraction":
                "dominant_reference",

            "effective_number_places":
                "effective_places_reference",
        }
    )
)


comparison = daily.merge(
    reference,
    on=[
        "participant_id",
        "date",
    ],
    how="left",
)


comparison[
    "abs_entropy_difference"
] = (
    comparison[
        "location_entropy_nats"
    ]
    - comparison[
        "entropy_reference"
    ]
).abs()


comparison[
    "abs_dominant_fraction_difference"
] = (
    comparison[
        "dominant_place_fraction"
    ]
    - comparison[
        "dominant_reference"
    ]
).abs()


comparison[
    "abs_effective_places_difference"
] = (
    comparison[
        "effective_number_places"
    ]
    - comparison[
        "effective_places_reference"
    ]
).abs()


sensitivity_rows = []


for (
    multiplier,
    participant_id,
), x in comparison.groupby(
    [
        "compatibility_multiplier",
        "participant_id",
    ],
    sort=True,
):

    sensitivity_rows.append({
        "compatibility_multiplier":
            multiplier,

        "participant_id":
            participant_id,

        "days":
            len(x),

        "median_abs_entropy_diff_vs_1":
            x[
                "abs_entropy_difference"
            ].median(),

        "p90_abs_entropy_diff_vs_1":
            x[
                "abs_entropy_difference"
            ].quantile(
                0.90
            ),

        "median_abs_dominant_diff_vs_1":
            x[
                "abs_dominant_fraction_difference"
            ].median(),

        "p90_abs_dominant_diff_vs_1":
            x[
                "abs_dominant_fraction_difference"
            ].quantile(
                0.90
            ),

        "median_abs_effective_places_diff_vs_1":
            x[
                "abs_effective_places_difference"
            ].median(),
    })


sensitivity = pd.DataFrame(
    sensitivity_rows
)

sensitivity.to_csv(
    SENSITIVITY_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("DAILY FEATURE SUMMARY")
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
print("FEATURE SENSITIVITY RELATIVE TO MULTIPLIER 1.0")
print("=" * 80)

print(
    sensitivity.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(OUT_PATH)
print(SUMMARY_OUT)
print(SENSITIVITY_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
