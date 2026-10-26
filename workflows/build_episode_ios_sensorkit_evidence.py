from pathlib import Path
import re

import pandas as pd


ROOT = Path("/shared/Digital Clinic")

RISK_PATH = (
    ROOT
    / "cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

EPISODES_PATH = (
    ROOT
    / "cortex2_outputs/digital_clinic/"
    "inventory/episodes.parquet"
)

OUT_PATH = (
    ROOT
    / "cortex2_outputs/digital_clinic/"
    "inventory/"
    "risk_episode_ios_sensorkit_evidence_timeaware.csv"
)

STREAMS = {
    "device_usage":
        ROOT / "passive/device_usage",

    "phone_usage":
        ROOT / "passive/phone_usage",

    "messages_usage":
        ROOT / "passive/messages_usage",

    "visits":
        ROOT / "passive/visits",
}


# ============================================================
# LOAD RISK EPISODES
# ============================================================

risk = pd.read_parquet(
    RISK_PATH
)

risk_episode = (
    risk[
        [
            "episode_id",
            "participant_id",
            "source_participant_id",
        ]
    ]
    .drop_duplicates()
    .copy()
)


# ============================================================
# LOAD EPISODE WINDOWS
# ============================================================

episodes = pd.read_parquet(
    EPISODES_PATH
)

print("=" * 80)
print("EPISODE TABLE COLUMNS")
print("=" * 80)
print(list(episodes.columns))


# Try common canonical names automatically.
start_candidates = [
    "start_ms",
    "episode_start",
    "episode_start_utc",
    "start_time",
    "start_timestamp",
    "window_start",
    "start",
]

end_candidates = [
    "end_ms",
    "episode_end",
    "episode_end_utc",
    "end_time",
    "end_timestamp",
    "window_end",
    "end",
]


def find_column(
    df,
    candidates,
):

    for col in candidates:

        if col in df.columns:
            return col

    return None


start_col = find_column(
    episodes,
    start_candidates,
)

end_col = find_column(
    episodes,
    end_candidates,
)


if (
    start_col is None
    or end_col is None
):

    raise RuntimeError(
        "Could not automatically identify "
        "episode start/end columns. "
        "Send me the printed column list."
    )


print()
print(
    "Using episode start:",
    start_col,
)

print(
    "Using episode end:",
    end_col,
)


episode_windows = episodes[
    [
        "episode_id",
        start_col,
        end_col,
    ]
].copy()


# ============================================================
# NORMALIZE EPISODE TIMES TO EPOCH MILLISECONDS
# ============================================================

def to_epoch_ms(
    series,
):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    # If mostly numeric, determine units.
    if numeric.notna().mean() > 0.8:

        med = numeric.dropna().abs().median()

        if med > 1e14:
            # nanoseconds
            return numeric / 1e6

        if med > 1e11:
            # milliseconds
            return numeric

        if med > 1e9:
            # seconds
            return numeric * 1000

    dt = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return (
        dt.astype("int64")
        / 1e6
    ).where(
        dt.notna()
    )


episode_windows[
    "episode_start_ms"
] = to_epoch_ms(
    episode_windows[start_col]
)

episode_windows[
    "episode_end_ms"
] = to_epoch_ms(
    episode_windows[end_col]
)


risk_episode = risk_episode.merge(
    episode_windows[
        [
            "episode_id",
            "episode_start_ms",
            "episode_end_ms",
        ]
    ],
    on="episode_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# INDEX SENSOR KIT FILES BY UID
# ============================================================

uid_pattern = re.compile(
    r"(U\d+)"
)

stream_files = {}

for stream, directory in STREAMS.items():

    mapping = {}

    for path in directory.glob(
        f"{stream}_U*.csv"
    ):

        match = uid_pattern.search(
            path.name
        )

        if not match:
            continue

        uid = match.group(1)

        mapping.setdefault(
            uid,
            []
        ).append(
            path
        )

    stream_files[
        stream
    ] = mapping


# ============================================================
# CHECK SENSOR KIT EVENTS WITHIN EACH EPISODE
# ============================================================

rows = []


for i, ep in risk_episode.iterrows():

    uid = str(
        ep["source_participant_id"]
    )

    start_ms = ep[
        "episode_start_ms"
    ]

    end_ms = ep[
        "episode_end_ms"
    ]

    result = {
        "episode_id":
            ep["episode_id"],

        "participant_id":
            ep["participant_id"],

        "source_participant_id":
            uid,

        "episode_start_ms":
            start_ms,

        "episode_end_ms":
            end_ms,
    }


    any_ios = False


    for stream in STREAMS:

        count = 0

        files = (
            stream_files[
                stream
            ]
            .get(
                uid,
                []
            )
        )


        for path in files:

            try:

                df = pd.read_csv(
                    path,
                    usecols=[
                        "timestamp"
                    ],
                )

            except Exception:
                continue


            ts = pd.to_numeric(
                df["timestamp"],
                errors="coerce",
            )


            count += int(
                (
                    (ts >= start_ms)
                    & (ts <= end_ms)
                ).sum()
            )


        result[
            f"{stream}_events_in_episode"
        ] = count

        result[
            f"has_{stream}_in_episode"
        ] = (
            count > 0
        )

        if count > 0:
            any_ios = True


    result[
        "sensorkit_ios_evidence_in_episode"
    ] = any_ios

    rows.append(
        result
    )


out = pd.DataFrame(
    rows
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("TIME-AWARE IOS SENSOR KIT EVIDENCE")
print("=" * 80)

print(
    "Risk-set episodes:",
    len(out),
)

print(
    "Confirmed iOS within episode:",
    int(
        out[
            "sensorkit_ios_evidence_in_episode"
        ].sum()
    ),
)

print(
    "OS unresolved:",
    int(
        (
            ~out[
                "sensorkit_ios_evidence_in_episode"
            ]
        ).sum()
    ),
)


print()
print("Per-stream episode evidence:")

for stream in STREAMS:

    print(
        f"{stream:18s}",
        int(
            out[
                f"has_{stream}_in_episode"
            ].sum()
        ),
    )


print()
print(
    "Confirmed iOS by episode start year:"
)

tmp = out.copy()

tmp[
    "episode_start"
] = pd.to_datetime(
    tmp[
        "episode_start_ms"
    ],
    unit="ms",
    utc=True,
    errors="coerce",
)

tmp[
    "year"
] = (
    tmp[
        "episode_start"
    ]
    .dt.year
)

summary = (
    tmp.groupby(
        "year",
        dropna=False,
    )
    .agg(
        episodes=(
            "episode_id",
            "size",
        ),

        confirmed_iOS=(
            "sensorkit_ios_evidence_in_episode",
            "sum",
        ),
    )
    .reset_index()
)

summary[
    "unresolved"
] = (
    summary["episodes"]
    - summary["confirmed_iOS"]
)

summary[
    "ios_fraction"
] = (
    summary["confirmed_iOS"]
    / summary["episodes"]
)

print(
    summary.to_string(
        index=False
    )
)


OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT_PATH,
    index=False,
)


print()
print("Saved:")
print(OUT_PATH)
