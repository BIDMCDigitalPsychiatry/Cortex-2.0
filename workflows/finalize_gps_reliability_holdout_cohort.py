"""Lock the independent Cortex 2.0 GPS reliability-validation cohort.

Selection was fixed before examining feature degradation outcomes.

Rules
-----
- sampling-density quartiles retain the original validation definition
- exclude all algorithm-development participants
- exclude all 16 reliability-calibration participants
- reference-day 60-second observability >= 0.99
- >=7 eligible reference days required per participant
- select 4 participants per original density quartile
- preserve deterministic holdout candidate selection order
- select 7 temporally spread eligible days per participant
- no GPS feature values are used for cohort selection

Final cohort:
    16 participants
    112 reference days
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

PARTICIPANT_CANDIDATES = (
    ROOT
    / "gps_reliability_holdout_candidate_participants.csv"
)

OBSERVABILITY_PATH = (
    ROOT
    / "gps_reliability_holdout_candidate_observability_60s.parquet"
)

CALIBRATION_PARTICIPANTS_PATH = (
    ROOT
    / "gps_validation_locked_participants.csv"
)

PARTICIPANT_OUT = (
    ROOT
    / "gps_reliability_holdout_locked_participants.csv"
)

DAY_OUT = (
    ROOT
    / "gps_reliability_holdout_locked_days.parquet"
)

DESIGN_OUT = (
    ROOT
    / "gps_reliability_holdout_design.json"
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


# ============================================================
# LOAD
# ============================================================

participants = pd.read_csv(
    PARTICIPANT_CANDIDATES
)

daily = pd.read_parquet(
    OBSERVABILITY_PATH
)

calibration = pd.read_csv(
    CALIBRATION_PARTICIPANTS_PATH
)


CALIBRATION_PARTICIPANTS = set(
    calibration[
        "participant_id"
    ]
    .astype(str)
)


participants[
    "participant_id"
] = participants[
    "participant_id"
].astype(str)

daily[
    "participant_id"
] = daily[
    "participant_id"
].astype(str)

daily[
    "date"
] = daily[
    "date"
].astype(str)


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


# ============================================================
# LEAKAGE ASSERTIONS
# ============================================================

candidate_ids = set(
    participants[
        "participant_id"
    ]
)


if candidate_ids & DEV_PARTICIPANTS:
    raise RuntimeError(
        "Development participant leaked into holdout candidates."
    )


if candidate_ids & CALIBRATION_PARTICIPANTS:
    raise RuntimeError(
        "Calibration participant leaked into holdout candidates."
    )


# ============================================================
# REFERENCE-DAY ELIGIBILITY
# ============================================================

eligible_days = daily[
    pd.to_numeric(
        daily[
            "observability"
        ],
        errors="coerce",
    )
    >= REFERENCE_OBSERVABILITY
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
    validate="one_to_one",
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
    ]
    >= DAYS_PER_PARTICIPANT
].copy()


# ============================================================
# LOCK 4 PARTICIPANTS PER ORIGINAL QUARTILE
# ============================================================

selected_frames = []


for quartile in EXPECTED_QUARTILES:

    x = participant_pool[
        participant_pool[
            "sampling_density_quartile"
        ]
        == quartile
    ].copy()


    if len(x) < PARTICIPANTS_PER_QUARTILE:
        raise RuntimeError(
            f"{quartile}: need "
            f"{PARTICIPANTS_PER_QUARTILE} participants, "
            f"but only {len(x)} are eligible."
        )


    # Preserve deterministic candidate selection order.
    # Do NOT rank by exact observability or feature behavior.
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
# LOCK 7 TEMPORALLY SPREAD ELIGIBLE DAYS/PARTICIPANT
# ============================================================

selected_day_frames = []


for _, participant in selected_participants.iterrows():

    participant_id = participant[
        "participant_id"
    ]


    x = eligible_days[
        eligible_days[
            "participant_id"
        ]
        == participant_id
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
        "holdout_day_number"
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

assert len(
    selected_participants
) == 16

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


selected_ids = set(
    selected_participants[
        "participant_id"
    ]
)


assert not (
    selected_ids
    & DEV_PARTICIPANTS
)

assert not (
    selected_ids
    & CALIBRATION_PARTICIPANTS
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
    "cohort_role":
        "independent_holdout_reliability_validation",

    "selection_uses_feature_outcomes":
        False,

    "quartile_definition":
        "original validation sampling-density quartiles",

    "reference_observability_threshold":
        REFERENCE_OBSERVABILITY,

    "days_per_participant":
        DAYS_PER_PARTICIPANT,

    "participants_per_quartile":
        PARTICIPANTS_PER_QUARTILE,

    "selected_participants":
        16,

    "selected_reference_days":
        112,

    "excluded_algorithm_development_participants":
        sorted(
            DEV_PARTICIPANTS
        ),

    "excluded_reliability_calibration_participants":
        sorted(
            CALIBRATION_PARTICIPANTS
        ),

    "frozen_gps_pipeline": {
        "maximum_supported_gap_ms":
            60000,

        "degradation_bin_size_ms":
            300000,

        "place_compatibility_multiplier":
            2.0,

        "location_features": [
            "entropy",
            "dominant_fraction",
            "effective_places",
        ],
    },
}


DESIGN_OUT.write_text(
    json.dumps(
        design,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("Cortex 2.0 GPS INDEPENDENT RELIABILITY HOLDOUT — LOCKED")
print("=" * 80)

print(
    "Eligible candidate participants:",
    len(
        participant_pool
    ),
)

print(
    "Locked participants:",
    len(
        selected_participants
    ),
)

print(
    "Locked reference days:",
    len(
        selected_days
    ),
)

print(
    "Minimum locked-day observability:",
    f"{selected_days['observability'].min():.4f}",
)

print(
    "Development overlap:",
    len(
        selected_ids
        & DEV_PARTICIPANTS
    ),
)

print(
    "Calibration overlap:",
    len(
        selected_ids
        & CALIBRATION_PARTICIPANTS
    ),
)


print()
print("Locked cohort:")

print(
    selected_participants[
        [
            "sampling_density_quartile",
            "participant_id",
            "median_n_observations",
            "eligible_reference_days",
            "selection_order",
        ]
    ]
    .sort_values(
        [
            "sampling_density_quartile",
            "selection_order",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("Quartile counts:")

print(
    quartile_counts
    .reindex(
        EXPECTED_QUARTILES
    )
    .to_string()
)


print()
print("Locked-day observability by quartile:")

print(
    selected_days
    .groupby(
        "sampling_density_quartile"
    )[
        "observability"
    ]
    .agg(
        [
            "min",
            "median",
            "max",
        ]
    )
    .to_string(
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("Outputs:")
print(PARTICIPANT_OUT)
print(DAY_OUT)
print(DESIGN_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
