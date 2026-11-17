from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT = Path(
    "outputs/trip_distance_validation/geolife/"
    "full_cohort/validation_cohort.csv"
)

OUT_DIR = Path(
    "outputs/trip_distance_validation/geolife/"
    "validation_sample"
)

PRIMARY_MODES = [
    "walk",
    "bike",
    "car",
    "taxi",
    "bus",
    "subway",
    "train",
]

TARGET_BLOCKS_PER_MODE = 50

# Prevent any one participant from dominating a mode.
MAX_BLOCKS_PER_USER_PER_MODE = 5

RANDOM_SEED = 20261116


def sample_mode(
    mode_df: pd.DataFrame,
    target_n: int,
) -> pd.DataFrame:

    capped_parts = []

    for user_id, user_df in mode_df.groupby(
        "user_id",
        sort=True,
    ):
        n = min(
            len(user_df),
            MAX_BLOCKS_PER_USER_PER_MODE,
        )

        sampled = user_df.sample(
            n=n,
            random_state=(
                RANDOM_SEED
                + int(user_id)
            ),
        )

        capped_parts.append(
            sampled
        )

    capped = pd.concat(
        capped_parts,
        ignore_index=True,
    )

    if len(capped) <= target_n:
        return capped

    return capped.sample(
        n=target_n,
        random_state=RANDOM_SEED,
    )


def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort = pd.read_csv(
        INPUT,
        dtype={
            "user_id": str,
        },
    )

    cohort = cohort.loc[
        cohort[
            "transportation_mode"
        ].isin(
            PRIMARY_MODES
        )
    ].copy()

    selected_parts = []

    print("=" * 100)
    print("GEOLIFE STRATIFIED VALIDATION SAMPLE")
    print("=" * 100)

    for mode in PRIMARY_MODES:

        mode_df = cohort.loc[
            cohort[
                "transportation_mode"
            ] == mode
        ].copy()

        sampled = sample_mode(
            mode_df,
            TARGET_BLOCKS_PER_MODE,
        )

        selected_parts.append(
            sampled
        )

        print(
            f"{mode:<10} "
            f"available={len(mode_df):>4} | "
            f"selected={len(sampled):>3} | "
            f"users={sampled['user_id'].nunique():>2}"
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected = selected.sort_values(
        [
            "transportation_mode",
            "user_id",
            "segment_id",
        ]
    ).reset_index(
        drop=True
    )

    selected.to_csv(
        OUT_DIR
        / "validation_sample.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL SAMPLE")
    print("=" * 100)

    print(
        f"Total blocks: "
        f"{len(selected)}"
    )

    print(
        f"Unique users: "
        f"{selected['user_id'].nunique()}"
    )

    print()
    print("BY MODE")
    print("-" * 100)

    summary = (
        selected.groupby(
            "transportation_mode"
        )
        .agg(
            n_blocks=(
                "segment_id",
                "count",
            ),
            n_users=(
                "user_id",
                "nunique",
            ),
            median_duration_min=(
                "duration_min",
                "median",
            ),
            median_points=(
                "n_points",
                "median",
            ),
        )
    )

    print(
        summary.to_string()
    )

    print()
    print("MAXIMUM USER CONTRIBUTION WITHIN EACH MODE")
    print("-" * 100)

    contribution = (
        selected.groupby(
            [
                "transportation_mode",
                "user_id",
            ]
        )
        .size()
        .groupby(
            level=0
        )
        .max()
    )

    print(
        contribution.to_string()
    )

    print()
    print(
        "Saved validation sample to:"
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()
