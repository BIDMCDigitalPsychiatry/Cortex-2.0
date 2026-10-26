"""Select participant-held-out candidates for Cortex 2.0 GPS reliability validation.

Selection is performed BEFORE looking at validation feature reliability.

Design
------
1. Exclude the three development participants.
2. Require >=14 days with existing coarse daily GPS DQ == 1.
3. Stratify eligible participants into quartiles of median GPS observation
   count on those days.
4. Deterministically select 8 participants per quartile.
5. Select 14 temporally spread coarse-perfect candidate days per participant.

The existing 10-minute-bin DQ is used ONLY as an efficient prefilter.
Final reference-day eligibility will use Cortex 2.0's 60-second temporal
observability calculation.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


QUALITY_PATH = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "quality/gps_daily.parquet"
)

OUT_ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

PARTICIPANT_OUT = (
    OUT_ROOT
    / "gps_validation_candidate_participants.csv"
)

DAY_OUT = (
    OUT_ROOT
    / "gps_validation_candidate_days.parquet"
)


DEV_PARTICIPANTS = {
    "CHO370",
    "KRE551",
    "RBE684",
}

MIN_PERFECT_DAYS = 14
PARTICIPANTS_PER_STRATUM = 8
CANDIDATE_DAYS_PER_PARTICIPANT = 14

SELECTION_NAMESPACE = (
    "cortex2_gps_validation_v1"
)


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


df = pd.read_parquet(
    QUALITY_PATH
)

df = df[
    df["stream"].astype(str) == "gps"
].copy()

df["date"] = (
    df["date"]
    .astype(str)
)


# ============================================================
# COARSE-PERFECT HELD-OUT DAYS
# ============================================================

perfect = df[
    (~df["participant_id"].isin(
        DEV_PARTICIPANTS
    ))
    & (
        df["data_quality"] == 1.0
    )
].copy()


participant_stats = (
    perfect
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


eligible = participant_stats[
    participant_stats[
        "perfect_days"
    ] >= MIN_PERFECT_DAYS
].copy()


# ============================================================
# SAMPLING-DENSITY STRATA
# ============================================================

# rank(method="first") guarantees unique values for qcut.
eligible[
    "density_rank"
] = eligible[
    "median_n_observations"
].rank(
    method="first"
)


eligible[
    "sampling_density_quartile"
] = pd.qcut(
    eligible[
        "density_rank"
    ],
    q=4,
    labels=[
        "Q1_low",
        "Q2",
        "Q3",
        "Q4_high",
    ],
)


eligible[
    "selection_order"
] = eligible[
    "participant_id"
].map(
    stable_order_value
)


selected_frames = []

for quartile, x in eligible.groupby(
    "sampling_density_quartile",
    observed=True,
    sort=True,
):

    x = (
        x
        .sort_values(
            [
                "selection_order",
                "participant_id",
            ]
        )
        .head(
            PARTICIPANTS_PER_STRATUM
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


if len(selected) != 32:
    raise RuntimeError(
        f"Expected 32 selected participants, "
        f"found {len(selected)}."
    )


selected.to_csv(
    PARTICIPANT_OUT,
    index=False,
)


# ============================================================
# TEMPORALLY SPREAD CANDIDATE DAYS
# ============================================================

day_frames = []


for _, participant in selected.iterrows():

    participant_id = participant[
        "participant_id"
    ]

    days = (
        perfect[
            perfect[
                "participant_id"
            ] == participant_id
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


    n = len(days)

    if n < CANDIDATE_DAYS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{participant_id}: only {n} candidate days."
        )


    indices = np.linspace(
        0,
        n - 1,
        CANDIDATE_DAYS_PER_PARTICIPANT,
    )

    indices = np.rint(
        indices
    ).astype(int)

    indices = np.unique(
        indices
    )


    if len(indices) != CANDIDATE_DAYS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{participant_id}: temporal selection "
            "did not produce 14 unique days."
        )


    chosen = days.iloc[
        indices
    ].copy()


    chosen[
        "sampling_density_quartile"
    ] = participant[
        "sampling_density_quartile"
    ]

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


candidate_days.to_parquet(
    DAY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("Cortex 2.0 GPS VALIDATION CANDIDATE SELECTION")
print("=" * 80)

print(
    "Eligible participants with >=14 coarse-perfect days:",
    len(eligible),
)

print(
    "Selected participants:",
    len(selected),
)

print(
    "Selected candidate days:",
    len(candidate_days),
)


print()
print("=" * 80)
print("SELECTED PARTICIPANTS BY SAMPLING-DENSITY QUARTILE")
print("=" * 80)

summary = (
    selected
    .groupby(
        "sampling_density_quartile",
        observed=True,
    )
    .agg(
        participants=(
            "participant_id",
            "size",
        ),
        min_median_observations=(
            "median_n_observations",
            "min",
        ),
        median_median_observations=(
            "median_n_observations",
            "median",
        ),
        max_median_observations=(
            "median_n_observations",
            "max",
        ),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.1f}",
    )
)


print()
print("=" * 80)
print("SELECTED PARTICIPANTS")
print("=" * 80)

print(
    selected[
        [
            "sampling_density_quartile",
            "participant_id",
            "perfect_days",
            "median_n_observations",
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
print("OUTPUTS")
print("=" * 80)

print(PARTICIPANT_OUT)
print(DAY_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
