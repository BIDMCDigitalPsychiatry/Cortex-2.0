"""Sensitivity analysis for uncertainty-scaled GPS place candidates.

Uses fixed-anchor microsegments from high-observability reference days.

The compatibility multiplier is deliberately varied rather than chosen
a priori. Final parameterization should later be evaluated by downstream
feature degradation reliability.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_place_candidates import (
    build_place_candidates,
)


INPUT_PATH = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability/"
    "gps_reference_anchor_compatible_episodes.parquet"
)

OUT_DIR = INPUT_PATH.parent

SUMMARY_OUT = (
    OUT_DIR
    / "gps_reference_place_candidate_sensitivity.csv"
)

ASSIGNMENT_OUT = (
    OUT_DIR
    / "gps_reference_place_candidate_assignments.parquet"
)

PLACE_OUT = (
    OUT_DIR
    / "gps_reference_place_candidates.parquet"
)


MULTIPLIERS = [
    0.75,
    1.00,
    1.50,
    2.00,
]


episodes = pd.read_parquet(
    INPUT_PATH
)


print("=" * 80)
print("Cortex 2.0 PLACE-CANDIDATE SENSITIVITY")
print("=" * 80)

print(
    "Microsegments:",
    f"{len(episodes):,}",
)

print(
    "Participants:",
    episodes[
        "participant_id"
    ].nunique(),
)


summary_rows = []
assignment_frames = []
place_frames = []


for multiplier in MULTIPLIERS:

    print()
    print(
        f"Multiplier = {multiplier:.2f}"
    )


    for participant_id, x in episodes.groupby(
        "participant_id",
        sort=True,
    ):

        assignments, places = (
            build_place_candidates(
                x,
                compatibility_multiplier=multiplier,
            )
        )


        assignments[
            "participant_id"
        ] = participant_id

        assignments[
            "compatibility_multiplier"
        ] = multiplier


        places[
            "participant_id"
        ] = participant_id

        places[
            "compatibility_multiplier"
        ] = multiplier


        total_time = (
            places[
                "total_duration_ms"
            ].sum()
        )


        ranked = (
            places.sort_values(
                "total_duration_ms",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        fractions = (
            ranked[
                "total_duration_ms"
            ]
            / total_time
        )


        entropy = -float(
            np.sum(
                fractions
                * np.log(
                    fractions
                )
            )
        )


        effective_places = float(
            np.exp(
                entropy
            )
        )


        summary_rows.append({
            "compatibility_multiplier":
                multiplier,

            "participant_id":
                participant_id,

            "microsegments":
                len(
                    assignments
                ),

            "candidate_places":
                len(
                    places
                ),

            "median_microsegments_per_place":
                places[
                    "n_microsegments"
                ].median(),

            "singleton_place_fraction":
                (
                    places[
                        "n_microsegments"
                    ]
                    == 1
                ).mean(),

            "top1_time_fraction":
                fractions.iloc[
                    :1
                ].sum(),

            "top3_time_fraction":
                fractions.iloc[
                    :3
                ].sum(),

            "top5_time_fraction":
                fractions.iloc[
                    :5
                ].sum(),

            "effective_number_places":
                effective_places,

            "median_place_max_seed_distance_m":
                places[
                    "max_seed_distance_m"
                ].median(),

            "p90_place_max_seed_distance_m":
                places[
                    "max_seed_distance_m"
                ].quantile(
                    0.90
                ),

            "p99_place_max_seed_distance_m":
                places[
                    "max_seed_distance_m"
                ].quantile(
                    0.99
                ),
        })


        assignment_frames.append(
            assignments
        )

        place_frames.append(
            places
        )


summary = pd.DataFrame(
    summary_rows
)

assignments = pd.concat(
    assignment_frames,
    ignore_index=True,
)

places = pd.concat(
    place_frames,
    ignore_index=True,
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)

assignments.to_parquet(
    ASSIGNMENT_OUT,
    index=False,
)

places.to_parquet(
    PLACE_OUT,
    index=False,
)


print()
print("=" * 80)
print("PLACE-CANDIDATE SENSITIVITY")
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

print(SUMMARY_OUT)
print(ASSIGNMENT_OUT)
print(PLACE_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
