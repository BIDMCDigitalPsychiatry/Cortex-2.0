from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.mindlamp import (
    MindLAMPAdapter,
)
from cortex2_engine.canonical.reader import (
    EpisodeReader,
)


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

RISK = (
    ROOT
    / "quality/hourly_missingness"
    / "gps_onset_contextual_risk_set.parquet"
)

HOUR_MS = 3_600_000


x = pd.read_parquet(RISK)

bad = x[
    x[
        "ctx_prev_device_reported_fraction"
    ].gt(1.01)
].copy()

bad = bad.sort_values(
    "ctx_prev_device_reported_fraction",
    ascending=False,
)


print("=" * 72)
print("SUSPICIOUS PREVIOUS-HOUR DEVICE COVERAGE")
print("=" * 72)

print(
    bad[
        [
            "participant_id",
            "episode_id",
            "hour_index",
            "start_ms",
            "onset_type",
            "ctx_prev_device_reported_fraction",
            "ctx_prev_device_usage_record_count",
            "ctx_prev_unlock_count",
            "ctx_prev_screen_wake_count",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


episodes = pd.read_parquet(
    ROOT
    / "inventory/episodes.parquet"
)


participant_map = episodes[
    [
        "participant_id",
        "source_participant_id",
    ]
].drop_duplicates()

map_path = (
    ROOT
    / "inventory"
    / "cortex2_participant_map.csv"
)

participant_map.to_csv(
    map_path,
    index=False,
)


adapter = MindLAMPAdapter(
    study_id="digital_clinic",
    passive_root=Path(
        "../passive"
    ),
    participant_map=map_path,
    source_id_column=(
        "source_participant_id"
    ),
    participant_id_column=(
        "participant_id"
    ),
    source_manifest_dir=(
        ROOT
        / "inventory/source_files"
    ),
)

reader = EpisodeReader(
    adapter,
    episodes,
)


print(
    "\n"
    + "=" * 72
)

print(
    "RAW RECORD AUDIT FOR EXTREME HOURS"
)

print(
    "=" * 72
)


# Inspect the six most extreme episode/hour examples.
examples = (
    bad.head(20)
    .drop_duplicates(
        "episode_id"
    )
    .head(6)
)


for row in examples.itertuples(
    index=False
):

    prev_start = (
        int(row.start_ms)
        - HOUR_MS
    )

    prev_end = int(
        row.start_ms
    )

    frames = []

    for chunk in reader.iter_stream(
        "device_usage",
        episode_id=str(
            row.episode_id
        ),
        chunksize=250_000,
    ):

        z = chunk[
            chunk[
                "timestamp_ms"
            ].between(
                prev_start,
                prev_end,
                inclusive="left",
            )
        ].copy()

        if len(z):
            frames.append(z)

    print(
        "\n"
        + "-" * 72
    )

    print(
        row.participant_id,
        row.episode_id,
    )

    print(
        "Outcome:",
        row.onset_type,
    )

    print(
        "Reported fraction:",
        row.ctx_prev_device_reported_fraction,
    )

    if not frames:
        print(
            "No underlying records found."
        )
        continue

    z = pd.concat(
        frames,
        ignore_index=True,
    )

    counts = (
        z.groupby(
            "timestamp_ms"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        "Records:",
        len(z),
    )

    print(
        "Unique timestamps:",
        z[
            "timestamp_ms"
        ].nunique(),
    )

    print(
        "Duplicated rows by timestamp:",
        int(
            z[
                "timestamp_ms"
            ].duplicated(
                keep=False
            ).sum()
        ),
    )

    print(
        "Maximum records at one timestamp:",
        int(
            counts.max()
        ),
    )

    duration = pd.to_numeric(
        z[
            "interval_duration_ms"
        ],
        errors="coerce",
    )

    print(
        "Summed reported duration / hour:",
        duration.sum()
        / HOUR_MS,
    )

    print(
        "\nMost repeated timestamps:"
    )

    print(
        counts.head(10).to_string()
    )

    print(
        "\nSource files:",
        z[
            "source_file"
        ]
        .value_counts()
        .head(15)
        .to_string()
    )

    core = [
        "timestamp_ms",
        "interval_duration_ms",
        "total_unlock_duration_ms",
        "total_unlocks",
        "total_screen_wakes",
    ]

    print(
        "\nFirst records:"
    )

    print(
        z[
            core
        ]
        .sort_values(
            "timestamp_ms"
        )
        .head(20)
        .to_string(
            index=False
        )
    )
