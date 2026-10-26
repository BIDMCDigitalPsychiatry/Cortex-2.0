from pathlib import Path

import pandas as pd


PATH = Path(
    "../cortex2_outputs/digital_clinic/quality/"
    "hourly_missingness/gap_runs.parquet"
)

runs = pd.read_parquet(PATH)


# ------------------------------------------------------------
# GPS-specific recurrence
# ------------------------------------------------------------

gps = runs[
    runs["gap_pattern"].eq(
        "gps_specific_gap"
    )
].copy()


participant = (
    gps.groupby(
        [
            "participant_id",
            "episode_id",
        ],
        as_index=False,
    )
    .agg(
        gps_gap_runs=(
            "run_id",
            "size",
        ),

        gps_gap_hours=(
            "duration_hours",
            "sum",
        ),

        median_run_hours=(
            "duration_hours",
            "median",
        ),

        max_run_hours=(
            "duration_hours",
            "max",
        ),

        mean_supporting_acc_dq=(
            "mean_supporting_dq",
            "mean",
        ),

        median_supporting_acc_dq=(
            "mean_supporting_dq",
            "median",
        ),

        runs_with_screen_activity=(
            "screen_event_hours",
            lambda s: int(
                s.gt(0).sum()
            ),
        ),

        total_screen_event_hours=(
            "screen_event_hours",
            "sum",
        ),
    )
)


participant[
    "screen_supported_run_fraction"
] = (
    participant[
        "runs_with_screen_activity"
    ]
    /
    participant[
        "gps_gap_runs"
    ]
)


print("=" * 70)
print("GPS-SPECIFIC GAP RECURRENCE")
print("=" * 70)

print(
    "Participants/episodes with >=1 GPS-specific run:",
    len(participant),
)

print("\nRuns per participant:")
print(
    participant[
        "gps_gap_runs"
    ]
    .describe(
        percentiles=[
            .25,
            .50,
            .75,
            .90,
            .95,
        ]
    )
    .to_string()
)


for threshold in [
    1,
    2,
    3,
    5,
    10,
]:
    print(
        f"Participants with >= {threshold} runs:",
        int(
            participant[
                "gps_gap_runs"
            ]
            .ge(threshold)
            .sum()
        ),
    )


print("\nTotal GPS-specific hours per participant:")

print(
    participant[
        "gps_gap_hours"
    ]
    .describe(
        percentiles=[
            .25,
            .50,
            .75,
            .90,
            .95,
        ]
    )
    .to_string()
)


print(
    "\nParticipants with largest total "
    "GPS-specific missingness:"
)

print(
    participant.sort_values(
        [
            "gps_gap_hours",
            "gps_gap_runs",
        ],
        ascending=False,
    )
    .head(25)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ------------------------------------------------------------
# Run duration distribution
# ------------------------------------------------------------

print(
    "\n" + "=" * 70
)
print(
    "GPS-SPECIFIC RUN DURATION"
)
print(
    "=" * 70
)


bins = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

for threshold in bins:
    print(
        f"runs >= {threshold:3d} h:",
        int(
            gps[
                "duration_hours"
            ]
            .ge(threshold)
            .sum()
        ),
    )


# ------------------------------------------------------------
# Screen support and ACC support by duration
# ------------------------------------------------------------

gps["duration_band"] = pd.cut(
    gps["duration_hours"],
    bins=[
        0,
        1,
        3,
        6,
        12,
        24,
        48,
        168,
        float("inf"),
    ],
    labels=[
        "1h",
        "2-3h",
        "4-6h",
        "7-12h",
        "13-24h",
        "25-48h",
        "49-168h",
        ">168h",
    ],
)


duration_summary = (
    gps.groupby(
        "duration_band",
        observed=True,
    )
    .agg(
        runs=(
            "run_id",
            "size",
        ),

        participants=(
            "participant_id",
            "nunique",
        ),

        total_hours=(
            "duration_hours",
            "sum",
        ),

        median_supporting_acc_dq=(
            "mean_supporting_dq",
            "median",
        ),

        median_min_acc_dq=(
            "min_supporting_dq",
            "median",
        ),

        runs_with_screen_activity=(
            "screen_event_hours",
            lambda s: int(
                s.gt(0).sum()
            ),
        ),

        median_screen_event_fraction=(
            "screen_event_fraction",
            "median",
        ),
    )
)


duration_summary[
    "fraction_runs_with_screen_activity"
] = (
    duration_summary[
        "runs_with_screen_activity"
    ]
    /
    duration_summary[
        "runs"
    ]
)


print(
    "\nGPS gap evidence by duration:"
)

print(
    duration_summary.to_string(
        float_format=lambda x: f"{x:.3f}",
    )
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT = (
    PATH.parent
    / "gps_specific_participant_summary.csv"
)

participant.to_csv(
    OUTPUT,
    index=False,
)

print("\nSaved:")
print(OUTPUT)
