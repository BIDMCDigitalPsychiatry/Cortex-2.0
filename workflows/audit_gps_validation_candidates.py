"""Audit candidate GPS days for held-out RAPHE reliability validation.

This uses the existing 10-minute-bin daily DQ table only as a candidate
prefilter. It does NOT treat that DQ measure as the final 60-second RAPHE
temporal observability measure.
"""

from pathlib import Path

import pandas as pd


PATH = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "quality/gps_daily.parquet"
)

DEV_PARTICIPANTS = {
    "CHO370",
    "KRE551",
    "RBE684",
}

THRESHOLDS = [
    1.00,
    0.99,
    0.95,
    0.90,
    0.75,
]


df = pd.read_parquet(PATH)

# GPS table should already be GPS-only, but make the assumption explicit.
df = df[
    df["stream"].astype(str) == "gps"
].copy()

heldout = df[
    ~df["participant_id"].isin(
        DEV_PARTICIPANTS
    )
].copy()


print("=" * 80)
print("RAPHE GPS HELD-OUT VALIDATION CANDIDATE AUDIT")
print("=" * 80)

print(
    "All GPS episode-days:",
    f"{len(df):,}",
)

print(
    "All participants:",
    df["participant_id"].nunique(),
)

print(
    "Held-out episode-days:",
    f"{len(heldout):,}",
)

print(
    "Held-out participants:",
    heldout["participant_id"].nunique(),
)


print()
print("=" * 80)
print("COARSE DAILY DQ THRESHOLDS")
print("=" * 80)

rows = []

for threshold in THRESHOLDS:

    x = heldout[
        heldout["data_quality"]
        >= threshold
    ]

    rows.append({
        "threshold":
            threshold,

        "episode_days":
            len(x),

        "participants":
            x["participant_id"].nunique(),

        "episodes":
            x["episode_id"].nunique(),

        "median_n_observations":
            x["n_observations"].median(),

        "median_longest_missing_seconds":
            x["longest_missing_seconds"].median(),
    })


summary = pd.DataFrame(rows)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("DATA_QUALITY = 1.0 BY PARTICIPANT")
print("=" * 80)

perfect = heldout[
    heldout["data_quality"] == 1.0
].copy()

by_participant = (
    perfect
    .groupby(
        "participant_id",
        as_index=False,
    )
    .agg(
        days=(
            "date",
            "size",
        ),
        episodes=(
            "episode_id",
            "nunique",
        ),
        median_observations=(
            "n_observations",
            "median",
        ),
        min_observations=(
            "n_observations",
            "min",
        ),
        max_observations=(
            "n_observations",
            "max",
        ),
    )
    .sort_values(
        [
            "days",
            "participant_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


print(
    by_participant.head(
        30
    ).to_string(
        index=False,
    )
)


print()
print("=" * 80)
print("PERFECT-DQ PARTICIPANT DEPTH")
print("=" * 80)

for minimum_days in [
    1,
    3,
    5,
    7,
    10,
    14,
]:

    n = int(
        (
            by_participant["days"]
            >= minimum_days
        ).sum()
    )

    print(
        f">= {minimum_days:2d} days: "
        f"{n} participants"
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
