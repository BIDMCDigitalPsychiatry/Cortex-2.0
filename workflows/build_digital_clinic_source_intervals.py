import json
from pathlib import Path

import pandas as pd


EPISODES_PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/episodes.parquet"
)

METADATA_PATH = Path(
    "../pull_metadata_digital_clinic.json"
)

OUTPUT_PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/source_intervals.parquet"
)

STUDY_ID = "digital_clinic"
SOURCE_PLATFORM = "mindlamp"


STREAM_MAP = {
    "gps": "gps",
    "acc": "accelerometer",
    "screen": "screen",
}


def utc_from_ms(value):
    if value is None:
        return pd.NaT

    return pd.to_datetime(
        int(value),
        unit="ms",
        utc=True,
    )


episodes = pd.read_parquet(
    EPISODES_PATH
)

metadata = json.loads(
    METADATA_PATH.read_text()
)

rows = []


for ep in episodes.itertuples(
    index=False
):
    source_id = str(
        ep.source_participant_id
    )

    item = metadata.get(
        source_id
    )

    if item is None:
        continue

    # Critical for reused source IDs:
    # only attach metadata to the episode whose participant
    # identity agrees with the pull metadata.
    metadata_pid = item.get(
        "participant_id"
    )

    if (
        metadata_pid is not None
        and str(metadata_pid)
        != str(ep.participant_id)
    ):
        continue

    # --------------------------------------------------
    # Participant-level declared export boundary
    # --------------------------------------------------

    participant_end = item.get(
        "last_end_time"
    )

    if participant_end is not None:
        participant_end = int(
            participant_end
        )

        rows.append({
            "study_id":
                STUDY_ID,

            "participant_id":
                str(ep.participant_id),

            "episode_id":
                str(ep.episode_id),

            "source_participant_id":
                source_id,

            "source_platform":
                SOURCE_PLATFORM,

            "stream":
                None,

            "source_start_ms":
                None,

            "source_start_utc":
                pd.NaT,

            "source_end_ms":
                participant_end,

            "source_end_utc":
                utc_from_ms(
                    participant_end
                ),

            "start_inclusive":
                None,

            "end_inclusive":
                False,

            "boundary_source":
                "pull_metadata_participant_last_end_time",

            "provenance_file":
                str(
                    METADATA_PATH.resolve()
                ),
        })

    # --------------------------------------------------
    # Genuine sensor-specific declared boundaries
    # --------------------------------------------------

    sensors = item.get(
        "sensors",
        {}
    )

    for source_name, cortex2_stream in (
        STREAM_MAP.items()
    ):
        sensor = sensors.get(
            source_name
        )

        if not isinstance(
            sensor,
            dict,
        ):
            continue

        sensor_end = sensor.get(
            "last_end_time"
        )

        if sensor_end is None:
            continue

        sensor_end = int(
            sensor_end
        )

        rows.append({
            "study_id":
                STUDY_ID,

            "participant_id":
                str(ep.participant_id),

            "episode_id":
                str(ep.episode_id),

            "source_participant_id":
                source_id,

            "source_platform":
                SOURCE_PLATFORM,

            "stream":
                cortex2_stream,

            "source_start_ms":
                None,

            "source_start_utc":
                pd.NaT,

            "source_end_ms":
                sensor_end,

            "source_end_utc":
                utc_from_ms(
                    sensor_end
                ),

            "start_inclusive":
                None,

            "end_inclusive":
                False,

            "boundary_source":
                (
                    "pull_metadata_sensor_last_end_time"
                ),

            "provenance_file":
                str(
                    METADATA_PATH.resolve()
                ),
        })


out = pd.DataFrame(
    rows
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_parquet(
    OUTPUT_PATH,
    index=False,
)


print(
    "Rows:",
    len(out)
)

print(
    "Episodes with any declared boundary:",
    out["episode_id"].nunique()
)

print(
    "\nBoundary sources:"
)

print(
    out["boundary_source"]
    .value_counts()
)

print(
    "\nStream-specific rows:"
)

print(
    out["stream"]
    .fillna("<participant-level>")
    .value_counts()
)

print(
    "\nReused source ID:"
)

print(
    out[
        out[
            "source_participant_id"
        ].eq(
            "U7886257943"
        )
    ][
        [
            "participant_id",
            "episode_id",
            "stream",
            "source_end_utc",
            "boundary_source",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nCHO370:"
)

print(
    out[
        out[
            "participant_id"
        ].eq(
            "CHO370"
        )
    ][
        [
            "participant_id",
            "stream",
            "source_end_utc",
            "boundary_source",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)
