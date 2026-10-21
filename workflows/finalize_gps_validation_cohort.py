"""Lock the held-out RAPHE GPS validation cohort.

Rules were fixed before examining held-out feature reliability:

- development participants remain excluded
- reference-day RAPHE 60-second observability >= 0.99
- >=7 eligible reference days required per participant
- 4 participants per sampling-density quartile
- participant choice uses the original deterministic selection order
- 7 temporally spread eligible reference days per selected participant
- GPS location algorithm is frozen at compatibility_multiplier = 2.0

No GPS feature values are used for cohort selection.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

PARTICIPANT_CANDIDATES = (
    ROOT
    / "gps_validation_candidate_participants.csv"
)

OBSERVABILITY_PATH = (
    ROOT
    / "gps_validation_candidate_observability_60s.parquet"
)

PARTICIPANT_OUT = (
    ROOT
    / "gps_validation_locked_participants.csv"
)

DAY_OUT = (
    ROOT
    / "gps_validation_locked_days.parquet"
)

DESIGN_OUT = (
    ROOT
    / "gps_validation_design.json"
)


REFERENCE_OBSERVABILITY = 0.99
DAYS_PER_PARTICIPANT = 7
PARTICIPANTS_PER_QUARTILE = 4

EXPECTED_QUARTILES = [
    "Q1_low",
    "Q2",
    "Q3",
    "Q4_high",
]

DEV_PARTICIPANTS = {
    "CHO370",
    "KRE551",
    "RBE684",
}


participants = pd.read_csv(
    PARTICIPANT_CANDIDATES
)

daily = pd.read_parquet(
    OBSERVABILITY_PATH
)


participants[
    "sampling_density_quartile"
] = participants[
    "sampling_density_quartile"
].astype(str)

daily[
    "sampling_density_quartile"
] = daily[
    "sampling_density_quartile"
].astype(str)

daily[
    "date"
] = daily[
    "date"
].astype(str)


# ============================================================
# REFERENCE-DAY ELIGIBILITY
# ============================================================

eligible_days = daily[
    daily[
        "observability"
    ] >= REFERENCE_OBSERVABILITY
].copy()


eligible_counts = (
    eligible_days
    .groupby(
        "participant_id"
    )
    .size()
    .rename(
        "eligible_reference_days"
    )
    .reset_index()
)


participant_pool = participants.merge(
    eligible_counts,
    on="participant_id",
    how="left",
)

participant_pool[
    "eligible_reference_days"
] = (
    participant_pool[
        "eligible_reference_days"
    ]
    .fillna(0)
    .astype(int)
)


participant_pool = participant_pool[
    participant_pool[
        "eligible_reference_days"
    ] >= DAYS_PER_PARTICIPANT
].copy()


# ============================================================
# LOCK 4 PARTICIPANTS PER QUARTILE
# ============================================================

selected_frames = []


for quartile in EXPECTED_QUARTILES:

    x = participant_pool[
        participant_pool[
            "sampling_density_quartile"
        ] == quartile
    ].copy()


    if len(x) < PARTICIPANTS_PER_QUARTILE:

        raise RuntimeError(
            f"{quartile}: need "
            f"{PARTICIPANTS_PER_QUARTILE} participants, "
            f"but only {len(x)} are eligible."
        )


    # Preserve the deterministic random order used during
    # candidate selection. Do not rank by feature performance
    # or exact observability magnitude.
    x = (
        x
        .sort_values(
            [
                "selection_order",
                "participant_id",
            ]
        )
        .head(
            PARTICIPANTS_PER_QUARTILE
        )
        .copy()
    )


    selected_frames.append(
        x
    )


selected_participants = pd.concat(
    selected_frames,
    ignore_index=True,
)


# ============================================================
# LOCK 7 TEMPORALLY SPREAD DAYS/PARTICIPANT
# ============================================================

selected_day_frames = []


for _, participant in selected_participants.iterrows():

    participant_id = participant[
        "participant_id"
    ]


    x = eligible_days[
        eligible_days[
            "participant_id"
        ] == participant_id
    ].copy()


    x = (
        x
        .sort_values(
            [
                "start_ms",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    if len(x) < DAYS_PER_PARTICIPANT:

        raise RuntimeError(
            f"{participant_id}: only {len(x)} "
            "eligible reference days."
        )


    indices = np.rint(
        np.linspace(
            0,
            len(x) - 1,
            DAYS_PER_PARTICIPANT,
        )
    ).astype(int)


    if len(
        np.unique(
            indices
        )
    ) != DAYS_PER_PARTICIPANT:

        raise RuntimeError(
            f"{participant_id}: failed to select "
            "7 unique temporally spread days."
        )


    chosen = x.iloc[
        indices
    ].copy()


    chosen[
        "validation_day_number"
    ] = np.arange(
        1,
        DAYS_PER_PARTICIPANT + 1,
    )


    selected_day_frames.append(
        chosen
    )


selected_days = pd.concat(
    selected_day_frames,
    ignore_index=True,
)


# ============================================================
# ASSERTIONS
# ============================================================

assert (
    selected_participants[
        "participant_id"
    ].nunique()
    == 16
)

assert len(
    selected_days
) == 112

assert (
    selected_days[
        "participant_id"
    ].nunique()
    == 16
)

assert (
    selected_days[
        "observability"
    ].min()
    >= REFERENCE_OBSERVABILITY
)

assert not (
    set(
        selected_participants[
            "participant_id"
        ]
    )
    & DEV_PARTICIPANTS
)


quartile_counts = (
    selected_participants[
        "sampling_density_quartile"
    ]
    .value_counts()
)


for quartile in EXPECTED_QUARTILES:

    assert (
        quartile_counts[
            quartile
        ]
        == PARTICIPANTS_PER_QUARTILE
    )


day_counts = (
    selected_days
    .groupby(
        "participant_id"
    )
    .size()
)

assert (
    day_counts
    == DAYS_PER_PARTICIPANT
).all()


# ============================================================
# SAVE
# ============================================================

selected_participants.to_csv(
    PARTICIPANT_OUT,
    index=False,
)

selected_days.to_parquet(
    DAY_OUT,
    index=False,
)


design = {
    "design_name":
        "RAPHE GPS location reliability held-out validation v1",

    "development_participants": [
        "CHO370",
        "KRE551",
        "RBE684",
    ],

    "validation_reference_observability_min":
        REFERENCE_OBSERVABILITY,

    "continuity_max_gap_ms":
        60000,

    "validation_participants":
        16,

    "sampling_density_quartiles":
        4,

    "participants_per_quartile":
        4,

    "reference_days_per_participant":
        7,

    "total_reference_days":
        112,

    "participant_selection":
        (
            "original deterministic candidate selection order "
            "among participants with >=7 eligible reference days"
        ),

    "reference_day_selection":
        (
            "7 temporally spread days among days with "
            "RAPHE 60-second observability >=0.99"
        ),

    "frozen_location_algorithm": {
        "compatibility_multiplier":
            2.0,

        "selection_basis":
            (
                "chosen from development benchmark before "
                "held-out feature reliability evaluation"
            ),
    },

    "degradation_experiment": {
        "target_coverages": [
            0.90,
            0.75,
            0.50,
            0.25,
        ],

        "missingness_patterns": [
            "distributed",
            "contiguous",
        ],

        "replicates":
            5,

        "mask_bin_size_ms":
            300000,
    },
}


with open(
    DESIGN_OUT,
    "w",
) as f:

    json.dump(
        design,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("RAPHE GPS HELD-OUT VALIDATION COHORT LOCKED")
print("=" * 80)

print(
    "Participants:",
    selected_participants[
        "participant_id"
    ].nunique(),
)

print(
    "Reference days:",
    len(
        selected_days
    ),
)

print(
    "Minimum observability:",
    f"{selected_days['observability'].min():.4f}",
)

print(
    "Median observability:",
    f"{selected_days['observability'].median():.4f}",
)


print()
print("=" * 80)
print("LOCKED PARTICIPANTS")
print("=" * 80)

print(
    selected_participants[
        [
            "sampling_density_quartile",
            "participant_id",
            "median_n_observations",
            "eligible_reference_days",
        ]
    ]
    .sort_values(
        [
            "sampling_density_quartile",
            "median_n_observations",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.1f}",
    )
)


print()
print("=" * 80)
print("REFERENCE DAYS BY QUARTILE")
print("=" * 80)

print(
    selected_days
    .groupby(
        "sampling_density_quartile"
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
        min_observability=(
            "observability",
            "min",
        ),
        median_observability=(
            "observability",
            "median",
        ),
        median_unique_timestamps=(
            "unique_timestamps",
            "median",
        ),
    )
    .reset_index()
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(PARTICIPANT_OUT)
print(DAY_OUT)
print(DESIGN_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
