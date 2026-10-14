from pathlib import Path

import pandas as pd


ROOT = Path("/shared/Digital Clinic")

RISK = Path(
    "../raphe_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)


SENSORKIT_STREAMS = {
    "device_usage":
        ROOT / "passive/device_usage",

    "phone_usage":
        ROOT / "passive/phone_usage",

    "messages_usage":
        ROOT / "passive/messages_usage",

    "visits":
        ROOT / "passive/visits",
}


def extract_uids(
    directory,
    prefix,
):

    uids = set()

    if not directory.exists():
        return uids

    for path in directory.glob(
        f"{prefix}_U*.csv"
    ):

        for part in (
            path.stem.split("_")
        ):

            if (
                part.startswith("U")
                and part[1:].isdigit()
            ):

                uids.add(part)
                break

    return uids


# ============================================================
# SENSOR KIT UID SETS
# ============================================================

stream_uids = {}

for stream, directory in (
    SENSORKIT_STREAMS.items()
):

    stream_uids[stream] = (
        extract_uids(
            directory,
            stream,
        )
    )


ios_union = set().union(
    *stream_uids.values()
)


# ============================================================
# RISK SET
# ============================================================

risk = pd.read_parquet(
    RISK,
    columns=[
        "episode_id",
        "participant_id",
        "source_participant_id",
    ],
).drop_duplicates(
    "episode_id"
)


risk[
    "sensorkit_ios_evidence"
] = (
    risk[
        "source_participant_id"
    ]
    .astype(str)
    .isin(
        ios_union
    )
)


for stream, uids in (
    stream_uids.items()
):

    risk[
        f"has_{stream}"
    ] = (
        risk[
            "source_participant_id"
        ]
        .astype(str)
        .isin(uids)
    )


# ============================================================
# REPORT
# ============================================================

print("=" * 76)
print("IOS SENSORKIT COVERAGE")
print("=" * 76)

for stream, uids in (
    stream_uids.items()
):

    print(
        f"{stream:18s}",
        f"{len(uids):4d}",
        "source UIDs overall",
    )


print()
print(
    "SensorKit union overall:",
    len(ios_union),
)


print()
print("=" * 76)
print("WITHIN GPS ONSET RISK SET")
print("=" * 76)

print(
    "Episodes:",
    len(risk),
)

n_ios = int(
    risk[
        "sensorkit_ios_evidence"
    ].sum()
)

n_unknown = (
    len(risk)
    - n_ios
)

print(
    "Definite iOS "
    "(>=1 SensorKit stream):",
    n_ios,
)

print(
    "OS unresolved:",
    n_unknown,
)


print()
print(
    "Per-stream episode coverage:"
)

for stream in (
    SENSORKIT_STREAMS
):

    n = int(
        risk[
            f"has_{stream}"
        ].sum()
    )

    print(
        f"{stream:18s}",
        n,
    )


print()
print("=" * 76)
print("SENSORKIT COMBINATIONS IN RISK SET")
print("=" * 76)

cols = [
    f"has_{s}"
    for s in SENSORKIT_STREAMS
]

combo = (
    risk[
        cols
    ]
    .value_counts()
    .reset_index(
        name="episodes"
    )
)

print(
    combo.to_string(
        index=False
    )
)


OUT = Path(
    "../raphe_outputs/digital_clinic/"
    "inventory/"
    "risk_episode_ios_sensorkit_evidence.csv"
)

risk.to_csv(
    OUT,
    index=False,
)

print()
print("Saved:")
print(OUT)
