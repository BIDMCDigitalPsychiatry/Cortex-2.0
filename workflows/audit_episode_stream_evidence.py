from pathlib import Path

import pandas as pd

from cortex2_engine.io.source_index import SourceFileIndex


EPISODES_PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/episodes.parquet"
)

MANIFEST_DIR = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/source_files"
)

OUTPUT_PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/episode_stream_evidence.csv"
)

STREAMS = [
    "gps",
    "accelerometer",
    "screen",
    "device_usage",
    "nearby_device",
]


episodes = pd.read_parquet(
    EPISODES_PATH
)

index = SourceFileIndex(
    MANIFEST_DIR
)

rows = []


for ep in episodes.itertuples(
    index=False
):

    for stream in STREAMS:

        files = index.candidate_records(
            stream,
            source_participant_id=(
                ep.source_participant_id
            ),
            start_ms=int(ep.start_ms),
            end_ms=int(ep.end_ms),
            verify_files=False,
        )

        has_stream_evidence = (
            len(files) > 0
        )

        rows.append({
            "participant_id":
                str(ep.participant_id),

            "episode_id":
                str(ep.episode_id),

            "source_participant_id":
                str(
                    ep.source_participant_id
                ),

            "stream":
                stream,

            "episode_start_ms":
                int(ep.start_ms),

            "episode_end_ms":
                int(ep.end_ms),

            "overlapping_files":
                int(len(files)),

            "overlapping_bytes":
                int(
                    files[
                        "size_bytes"
                    ].sum()
                )
                if len(files)
                else 0,

            "has_stream_evidence":
                bool(
                    has_stream_evidence
                ),

            "observed_min_ms":
                (
                    int(
                        files[
                            "min_timestamp_ms"
                        ].min()
                    )
                    if len(files)
                    else None
                ),

            "observed_max_ms":
                (
                    int(
                        files[
                            "max_timestamp_ms"
                        ].max()
                    )
                    if len(files)
                    else None
                ),
        })


out = pd.DataFrame(
    rows
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUTPUT_PATH,
    index=False,
)


print(
    "Rows:",
    len(out)
)

print(
    "Episodes:",
    out["episode_id"].nunique()
)

print("\nStream evidence:")

summary = (
    out.groupby("stream")
    .agg(
        episodes=(
            "episode_id",
            "size",
        ),
        episodes_with_evidence=(
            "has_stream_evidence",
            "sum",
        ),
        total_overlapping_files=(
            "overlapping_files",
            "sum",
        ),
    )
)

summary[
    "episodes_without_evidence"
] = (
    summary["episodes"]
    - summary[
        "episodes_with_evidence"
    ]
)

summary[
    "fraction_with_evidence"
] = (
    summary[
        "episodes_with_evidence"
    ]
    / summary["episodes"]
)

print(
    summary.to_string()
)



wide = (
    out.pivot(
        index="episode_id",
        columns="stream",
        values="has_stream_evidence",
    )
    .fillna(False)
    .astype(bool)
)


CORE_STREAMS = [
    "gps",
    "accelerometer",
    "screen",
]

CONTEXT_STREAMS = [
    "device_usage",
    "nearby_device",
]

ALL_STREAMS = (
    CORE_STREAMS
    + CONTEXT_STREAMS
)


for stream in ALL_STREAMS:
    if stream not in wide.columns:
        wide[stream] = False


wide["all_core_three"] = (
    wide["gps"]
    & wide["accelerometer"]
    & wide["screen"]
)

wide["gps_acc"] = (
    wide["gps"]
    & wide["accelerometer"]
)

wide["all_five"] = (
    wide[ALL_STREAMS]
    .all(axis=1)
)

wide["none_any"] = ~(
    wide[ALL_STREAMS]
    .any(axis=1)
)

wide[
    "gps_acc_plus_device_usage"
] = (
    wide["gps"]
    & wide["accelerometer"]
    & wide["device_usage"]
)

wide[
    "gps_acc_plus_nearby_device"
] = (
    wide["gps"]
    & wide["accelerometer"]
    & wide["nearby_device"]
)

wide[
    "gps_acc_plus_both_context"
] = (
    wide["gps"]
    & wide["accelerometer"]
    & wide["device_usage"]
    & wide["nearby_device"]
)


print(
    "\nEpisode overlap:"
)

print(
    "GPS + ACC:",
    int(
        wide["gps_acc"].sum()
    ),
)

print(
    "Core three "
    "(GPS + ACC + screen):",
    int(
        wide[
            "all_core_three"
        ].sum()
    ),
)

print(
    "GPS + ACC + device_usage:",
    int(
        wide[
            "gps_acc_plus_device_usage"
        ].sum()
    ),
)

print(
    "GPS + ACC + nearby_device:",
    int(
        wide[
            "gps_acc_plus_nearby_device"
        ].sum()
    ),
)

print(
    "GPS + ACC + both contextual streams:",
    int(
        wide[
            "gps_acc_plus_both_context"
        ].sum()
    ),
)

print(
    "All five streams:",
    int(
        wide["all_five"].sum()
    ),
)

print(
    "No evidence for any stream:",
    int(
        wide["none_any"].sum()
    ),
)


print(
    "\nFive-stream combination counts:"
)

print(
    wide[
        ALL_STREAMS
    ]
    .value_counts()
    .to_string()
)


print(
    "\nContext availability among GPS+ACC episodes:"
)

gps_acc = wide[
    wide["gps_acc"]
].copy()

print(
    "GPS+ACC episodes:",
    len(gps_acc),
)

print(
    "with device_usage:",
    int(
        gps_acc[
            "device_usage"
        ].sum()
    ),
)

print(
    "with nearby_device:",
    int(
        gps_acc[
            "nearby_device"
        ].sum()
    ),
)

print(
    "with both:",
    int(
        (
            gps_acc[
                "device_usage"
            ]
            &
            gps_acc[
                "nearby_device"
            ]
        ).sum()
    ),
)


print("\nSaved:")
print(OUTPUT_PATH)
