from pathlib import Path

import pandas as pd

from cortex2_engine.canonical.episodes import (
    build_episodes_from_windows,
)
from cortex2_engine.canonical.validation import (
    validate_canonical,
)


WINDOWS = Path(
    "../cortex_longitudinal/data/"
    "study_windows.csv"
)

OUTPUT = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/episodes.parquet"
)


windows = pd.read_csv(WINDOWS)

episodes = build_episodes_from_windows(
    windows,
    study_id="digital_clinic",
    participant_id_column="participant_id",
    source_participant_id_column="mindlamp_uid",
    start_column="clinical_start",
    end_column="clinical_end",
    inclusive_end_date=True,
    source_platform="mindlamp",
    window_name="clinical",
)


# --------------------------------------------------
# Validate canonical schema
# --------------------------------------------------

report = validate_canonical(
    episodes,
    "episodes",
)

if not report.valid:
    raise RuntimeError(
        f"Episode validation failed: {report.errors}"
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

episodes.to_parquet(
    OUTPUT,
    index=False,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n=== DIGITAL CLINIC EPISODES ===")

print(
    "Episodes:",
    f"{len(episodes):,}",
)

print(
    "Unique participant IDs:",
    f"{episodes['participant_id'].nunique():,}",
)

print(
    "Unique source IDs:",
    f"{episodes['source_participant_id'].nunique():,}",
)

print(
    "Unique episode IDs:",
    f"{episodes['episode_id'].nunique():,}",
)


# --------------------------------------------------
# Source IDs linked to >1 participant
# --------------------------------------------------

source_summary = (
    episodes.groupby(
        "source_participant_id"
    )
    .agg(
        n_episodes=(
            "episode_id",
            "size",
        ),
        n_participants=(
            "participant_id",
            "nunique",
        ),
    )
    .reset_index()
)

reused = source_summary[
    source_summary[
        "n_participants"
    ] > 1
]

print(
    "\nSource IDs linked to "
    ">1 participant:"
)

print(len(reused))


if len(reused):

    detailed = (
        episodes[
            episodes[
                "source_participant_id"
            ].isin(
                reused[
                    "source_participant_id"
                ]
            )
        ]
        .sort_values(
            [
                "source_participant_id",
                "start_ms",
            ]
        )
    )

    print(
        "\n=== REUSED SOURCE IDS ==="
    )

    print(
        detailed[
            [
                "source_participant_id",
                "participant_id",
                "episode_id",
                "start_utc",
                "end_utc",
            ]
        ].to_string(
            index=False
        )
    )


# --------------------------------------------------
# Episode duration
# --------------------------------------------------

duration_days = (
    (
        episodes["end_ms"]
        - episodes["start_ms"]
    )
    / 86_400_000
)

print(
    "\nEpisode duration days:"
)

print(
    duration_days.describe()
)


print(
    "\nSaved:"
)

print(OUTPUT)
