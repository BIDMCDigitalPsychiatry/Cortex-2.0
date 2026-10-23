"""Select a fresh participant holdout for RAPHE GPS reliability validation.

Important design rule
---------------------
Sampling-density quartiles are reproduced using the ORIGINAL eligibility
population (excluding only the three original algorithm-development
participants). The 16 calibration participants are excluded only AFTER
quartile assignment.

This prevents density-stratum definitions from shifting between the
calibration and independent-holdout stages.

Selection uses no GPS feature values or reliability outcomes.

Design
------
- original coarse prefilter: >=14 days with GPS data_quality == 1
- original sampling-density quartile definition
- exclude:
    * 3 algorithm-development participants
    * 16 reliability-calibration participants
- deterministically choose 16 candidates per quartile
- choose 14 temporally spread coarse-perfect candidate days/person
- final eligibility will later require 60-second observability >=0.99
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


QUALITY_PATH = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "quality/gps_daily.parquet"
)

ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

CALIBRATION_PARTICIPANTS_PATH = (
    ROOT
    / "gps_validation_locked_participants.csv"
)

PARTICIPANT_OUT = (
    ROOT
    / "gps_reliability_holdout_candidate_participants.csv"
)

DAY_OUT = (
    ROOT
    / "gps_reliability_holdout_candidate_days.parquet"
)


DEV_PARTICIPANTS = {
    "CHO370",
    "KRE551",
    "RBE684",
}

MIN_PERFECT_DAYS = 14

PARTICIPANTS_PER_QUARTILE = 16

CANDIDATE_DAYS_PER_PARTICIPANT = 14

SELECTION_NAMESPACE = (
    "raphe_gps_reliability_holdout_v1"
)

EXPECTED_QUARTILES = [
    "Q1_low",
    "Q2",
    "Q3",
    "Q4_high",
]


def stable_order_value(
    participant_id,
):
    text = (
        f"{SELECTION_NAMESPACE}|"
        f"{participant_id}"
    )

    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    QUALITY_PATH
)

df = df[
    df["stream"].astype(str)
    == "gps"
].copy()

df["date"] = (
    df["date"]
    .astype(str)
)


calibration = pd.read_csv(
    CALIBRATION_PARTICIPANTS_PATH
)

CALIBRATION_PARTICIPANTS = set(
    calibration[
        "participant_id"
    ]
    .astype(str)
    .tolist()
)


if len(
    CALIBRATION_PARTICIPANTS
) != 16:
    raise RuntimeError(
        f"Expected 16 calibration participants, "
        f"found {len(CALIBRATION_PARTICIPANTS)}."
    )


if (
    CALIBRATION_PARTICIPANTS
    & DEV_PARTICIPANTS
):
    raise RuntimeError(
        "Calibration and development participant sets overlap."
    )


# ============================================================
# REPRODUCE ORIGINAL ELIGIBILITY POPULATION
#
# IMPORTANT:
# Calibration participants are intentionally NOT removed yet.
# Quartiles must be assigned from the same population used for
# the original held-out validation.
# ============================================================

perfect_original_population = df[
    (
        ~df[
            "participant_id"
        ].isin(
            DEV_PARTICIPANTS
        )
    )
    & (
        df[
            "data_quality"
        ]
        == 1.0
    )
].copy()


participant_stats = (
    perfect_original_population
    .groupby(
        "participant_id",
        as_index=False,
    )
    .agg(
        perfect_days=(
            "date",
            "size",
        ),

        episodes=(
            "episode_id",
            "nunique",
        ),

        median_n_observations=(
            "n_observations",
            "median",
        ),
    )
)


eligible_original_population = (
    participant_stats[
        participant_stats[
            "perfect_days"
        ]
        >= MIN_PERFECT_DAYS
    ]
    .copy()
)


# ============================================================
# REPRODUCE ORIGINAL QUARTILE ASSIGNMENT
# ============================================================

eligible_original_population[
    "density_rank"
] = (
    eligible_original_population[
        "median_n_observations"
    ]
    .rank(
        method="first"
    )
)


eligible_original_population[
    "sampling_density_quartile"
] = pd.qcut(
    eligible_original_population[
        "density_rank"
    ],
    q=4,
    labels=EXPECTED_QUARTILES,
)


original_quartile_counts = (
    eligible_original_population[
        "sampling_density_quartile"
    ]
    .value_counts(
        sort=False
    )
)


# ============================================================
# NOW EXCLUDE ALL PREVIOUSLY USED PARTICIPANTS
# ============================================================

excluded_participants = (
    DEV_PARTICIPANTS
    | CALIBRATION_PARTICIPANTS
)


holdout_pool = (
    eligible_original_population[
        ~eligible_original_population[
            "participant_id"
        ].isin(
            excluded_participants
        )
    ]
    .copy()
)


if (
    set(
        holdout_pool[
            "participant_id"
        ]
    )
    & excluded_participants
):
    raise RuntimeError(
        "Previously used participant leaked into holdout pool."
    )


# New deterministic order for the independent holdout.
holdout_pool[
    "selection_order"
] = (
    holdout_pool[
        "participant_id"
    ]
    .map(
        stable_order_value
    )
)


# ============================================================
# SELECT BROAD CANDIDATE POOL
# ============================================================

selected_frames = []


for quartile in EXPECTED_QUARTILES:

    x = holdout_pool[
        holdout_pool[
            "sampling_density_quartile"
        ].astype(str)
        == quartile
    ].copy()


    if len(x) < PARTICIPANTS_PER_QUARTILE:
        raise RuntimeError(
            f"{quartile}: need at least "
            f"{PARTICIPANTS_PER_QUARTILE} remaining participants, "
            f"found {len(x)}."
        )


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


selected = pd.concat(
    selected_frames,
    ignore_index=True,
)


expected_selected = (
    PARTICIPANTS_PER_QUARTILE
    * 4
)


if len(
    selected
) != expected_selected:
    raise RuntimeError(
        f"Expected {expected_selected} selected candidates, "
        f"found {len(selected)}."
    )


if (
    selected[
        "participant_id"
    ].nunique()
    != expected_selected
):
    raise RuntimeError(
        "Candidate participant IDs are not unique."
    )


selected.to_csv(
    PARTICIPANT_OUT,
    index=False,
)


# ============================================================
# SELECT 14 TEMPORALLY SPREAD COARSE-PERFECT DAYS
# ============================================================

day_frames = []


for _, participant in selected.iterrows():

    participant_id = str(
        participant[
            "participant_id"
        ]
    )


    # Use the same original coarse-perfect table.
    days = (
        perfect_original_population[
            perfect_original_population[
                "participant_id"
            ]
            == participant_id
        ]
        .sort_values(
            [
                "start_ms",
                "episode_id",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    n = len(
        days
    )


    if n < CANDIDATE_DAYS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{participant_id}: only {n} candidate days."
        )


    indices = np.rint(
        np.linspace(
            0,
            n - 1,
            CANDIDATE_DAYS_PER_PARTICIPANT,
        )
    ).astype(int)


    if len(
        np.unique(
            indices
        )
    ) != CANDIDATE_DAYS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{participant_id}: temporal selection "
            "did not produce 14 unique days."
        )


    chosen = days.iloc[
        indices
    ].copy()


    chosen[
        "sampling_density_quartile"
    ] = str(
        participant[
            "sampling_density_quartile"
        ]
    )


    chosen[
        "participant_median_n_observations"
    ] = participant[
        "median_n_observations"
    ]


    chosen[
        "participant_perfect_days"
    ] = participant[
        "perfect_days"
    ]


    day_frames.append(
        chosen
    )


candidate_days = pd.concat(
    day_frames,
    ignore_index=True,
)


expected_days = (
    expected_selected
    * CANDIDATE_DAYS_PER_PARTICIPANT
)


if len(
    candidate_days
) != expected_days:
    raise RuntimeError(
        f"Expected {expected_days} candidate days, "
        f"found {len(candidate_days)}."
    )


candidate_days.to_parquet(
    DAY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("RAPHE GPS INDEPENDENT RELIABILITY-HOLDOUT CANDIDATES")
print("=" * 80)

print(
    "Original eligible population:",
    len(
        eligible_original_population
    ),
)

print(
    "Calibration participants excluded:",
    len(
        CALIBRATION_PARTICIPANTS
    ),
)

print(
    "Remaining eligible holdout pool:",
    len(
        holdout_pool
    ),
)

print()
print("Original quartile counts:")
print(
    original_quartile_counts.to_string()
)

print()
print("Remaining pool by original quartile:")
print(
    holdout_pool[
        "sampling_density_quartile"
    ]
    .value_counts(
        sort=False
    )
    .to_string()
)

print()
print("Selected candidates by original quartile:")
print(
    selected[
        "sampling_density_quartile"
    ]
    .value_counts(
        sort=False
    )
    .to_string()
)

print()
print(
    "Candidate participants:",
    len(
        selected
    ),
)

print(
    "Candidate days:",
    len(
        candidate_days
    ),
)

print(
    "Overlap with development participants:",
    len(
        set(
            selected[
                "participant_id"
            ]
        )
        & DEV_PARTICIPANTS
    ),
)

print(
    "Overlap with calibration participants:",
    len(
        set(
            selected[
                "participant_id"
            ]
        )
        & CALIBRATION_PARTICIPANTS
    ),
)

print()
print("Density range by quartile:")

density_summary = (
    selected
    .groupby(
        "sampling_density_quartile",
        observed=True,
    )[
        "median_n_observations"
    ]
    .agg(
        [
            "min",
            "median",
            "max",
        ]
    )
)

print(
    density_summary.to_string(
        float_format=lambda x:
            f"{x:.1f}",
    )
)

print()
print("Outputs:")
print(PARTICIPANT_OUT)
print(DAY_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
