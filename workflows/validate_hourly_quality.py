from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.mindlamp import (
    MindLAMPAdapter,
)
from cortex2_engine.canonical.reader import (
    EpisodeReader,
)
from cortex2_engine.quality.hourly import (
    episode_hourly_quality,
)


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

episodes = pd.read_parquet(
    ROOT / "inventory/episodes.parquet"
)

source_intervals = pd.read_parquet(
    ROOT / "inventory/source_intervals.parquet"
)

target = episodes[
    episodes["participant_id"]
    .astype(str)
    .eq("CHO370")
]

if len(target) != 1:
    raise RuntimeError(
        f"Expected one CHO370 episode, found {len(target)}"
    )

episode_id = str(
    target.iloc[0][
        "episode_id"
    ]
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
    / "validation_participant_map.csv"
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
    source_id_column="source_participant_id",
    participant_id_column="participant_id",
    source_manifest_dir=(
        ROOT
        / "inventory/source_files"
    ),
)

reader = EpisodeReader(
    adapter,
    episodes,
)


frames = []

for stream in [
    "gps",
    "accelerometer",
    "screen",
]:

    print(
        f"\nBuilding {stream}..."
    )

    x = episode_hourly_quality(
        reader,
        source_intervals,
        episode_id=episode_id,
        stream=stream,
    )

    frames.append(x)

    print(
        "rows:",
        len(x)
    )

    print(
        "observed hours:",
        int(
            x["hour_observed"].sum()
        )
    )

    print(
        "source unavailable:",
        int(
            x["source_unavailable"].sum()
        )
    )

    if stream != "screen":

        print(
            "zero-DQ hours:",
            int(
                x["data_quality"]
                .eq(0)
                .sum()
            )
        )

        print(
            "mean DQ:",
            x[
                "data_quality"
            ].mean()
        )


out = pd.concat(
    frames,
    ignore_index=True,
    sort=False,
)


OUTPUT = (
    ROOT
    / "quality"
    / "validation"
    / "CHO370_hourly_quality.parquet"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_parquet(
    OUTPUT,
    index=False,
)


print(
    "\nTotal rows:",
    f"{len(out):,}"
)

print(
    "\nSaved:"
)

print(OUTPUT)
