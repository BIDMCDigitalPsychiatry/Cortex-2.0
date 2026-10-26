from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.mindlamp import (
    MindLAMPAdapter,
)
from cortex2_engine.canonical.reader import (
    EpisodeReader,
)
from cortex2_engine.missingness.hourly_context import (
    episode_hourly_context,
)


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

QUALITY = ROOT / "quality"

OUTPUT_DIR = (
    QUALITY
    / "hourly_context"
    / "episodes"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


episodes = pd.read_parquet(
    ROOT / "inventory/episodes.parquet"
)

context = pd.read_parquet(
    QUALITY
    / "episode_stream_context.parquet"
)


# GPS + ACC episodes only.
core = (
    context[
        context["stream"].isin(
            [
                "gps",
                "accelerometer",
            ]
        )
    ]
    .pivot(
        index="episode_id",
        columns="stream",
        values="has_stream_evidence",
    )
    .fillna(False)
)

eligible_ids = (
    core[
        core["gps"]
        &
        core["accelerometer"]
    ]
    .index
    .astype(str)
)

selected = episodes[
    episodes[
        "episode_id"
    ]
    .astype(str)
    .isin(eligible_ids)
].copy()


# Stream-evidence lookup.
evidence = (
    context.pivot(
        index="episode_id",
        columns="stream",
        values="has_stream_evidence",
    )
    .fillna(False)
)


participant_map = episodes[
    [
        "participant_id",
        "source_participant_id",
    ]
].drop_duplicates()

participant_map_path = (
    ROOT
    / "inventory"
    / "cortex2_participant_map.csv"
)

participant_map.to_csv(
    participant_map_path,
    index=False,
)


adapter = MindLAMPAdapter(
    study_id="digital_clinic",
    passive_root=Path(
        "../passive"
    ),
    participant_map=participant_map_path,
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


summary = []


for i, ep in enumerate(
    selected.itertuples(
        index=False
    ),
    start=1,
):

    episode_id = str(
        ep.episode_id
    )

    output = (
        OUTPUT_DIR
        / f"{episode_id}.parquet"
    )

    print(
        f"[{i}/{len(selected)}] "
        f"{ep.participant_id}",
        flush=True,
    )

    if output.exists():

        x = pd.read_parquet(
            output
        )

    else:

        e = (
            evidence.loc[
                episode_id
            ]
            if episode_id
            in evidence.index
            else pd.Series(
                dtype=bool
            )
        )

        stream_evidence = {
            stream: bool(
                e.get(
                    stream,
                    False,
                )
            )
            for stream in [
                "screen",
                "device_usage",
                "nearby_device",
            ]
        }

        x = episode_hourly_context(
            reader,
            episode_id=episode_id,
            stream_evidence=(
                stream_evidence
            ),
        )

        x.to_parquet(
            output,
            index=False,
        )

    summary.append({
        "participant_id":
            str(ep.participant_id),

        "episode_id":
            episode_id,

        "hours":
            len(x),

        "screen_episode_evidence":
            bool(
                x[
                    "screen_episode_evidence"
                ].iloc[0]
            ),

        "device_usage_episode_evidence":
            bool(
                x[
                    "device_usage_episode_evidence"
                ].iloc[0]
            ),

        "nearby_device_episode_evidence":
            bool(
                x[
                    "nearby_device_episode_evidence"
                ].iloc[0]
            ),

        "hours_with_battery":
            int(
                x[
                    "battery_observation_count"
                ]
                .gt(0)
                .sum()
            ),

        "hours_with_device_usage":
            int(
                x[
                    "device_usage_record_count"
                ]
                .gt(0)
                .sum()
            ),

        "hours_with_nearby_observation":
            int(
                x[
                    "nearby_record_count"
                ]
                .gt(0)
                .sum()
            ),
    })


summary = pd.DataFrame(
    summary
)


summary_path = (
    QUALITY
    / "hourly_context"
    / "episode_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print(
    "\n" + "=" * 70
)

print(
    "HOURLY CONTEXT SUMMARY"
)

print(
    "=" * 70
)

print(
    "Episodes:",
    len(summary),
)

print(
    "Screen evidence:",
    int(
        summary[
            "screen_episode_evidence"
        ].sum()
    ),
)

print(
    "Device usage evidence:",
    int(
        summary[
            "device_usage_episode_evidence"
        ].sum()
    ),
)

print(
    "Nearby-device evidence:",
    int(
        summary[
            "nearby_device_episode_evidence"
        ].sum()
    ),
)

print(
    "\nTotal context-positive hours:"
)

print(
    "Battery:",
    int(
        summary[
            "hours_with_battery"
        ].sum()
    ),
)

print(
    "Device usage:",
    int(
        summary[
            "hours_with_device_usage"
        ].sum()
    ),
)

print(
    "Nearby device:",
    int(
        summary[
            "hours_with_nearby_observation"
        ].sum()
    ),
)


print(
    "\nPer-episode positive-hour fractions:"
)

for col in [
    "hours_with_battery",
    "hours_with_device_usage",
    "hours_with_nearby_observation",
]:

    fraction = (
        summary[col]
        /
        summary["hours"]
    )

    print(
        "\n",
        col,
        sep="",
    )

    print(
        fraction.describe().to_string()
    )


print(
    "\nSaved:"
)

print(summary_path)
