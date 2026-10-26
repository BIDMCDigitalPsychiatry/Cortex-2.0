from pathlib import Path

import pandas as pd


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

episodes = pd.read_parquet(
    ROOT / "inventory/episodes.parquet"
)

evidence = pd.read_csv(
    ROOT / "quality/episode_stream_evidence.csv"
)

gps = pd.read_parquet(
    ROOT / "quality/gps_daily.parquet"
)

acc = pd.read_parquet(
    ROOT / "quality/accelerometer_daily.parquet"
)

screen = pd.read_parquet(
    ROOT / "quality/screen_daily.parquet"
)


# ------------------------------------------------------------
# Episode-level evidence pattern
# ------------------------------------------------------------

wide = (
    evidence.pivot(
        index="episode_id",
        columns="stream",
        values="has_stream_evidence",
    )
    .fillna(False)
    .astype(bool)
    .reset_index()
)

wide = wide.rename(
    columns={
        "accelerometer": "acc",
    }
)


def evidence_pattern(row):
    g = bool(row["gps"])
    a = bool(row["acc"])
    s = bool(row["screen"])

    if g and a and s:
        return "all_three"

    if not g and not a and not s:
        return "none"

    if (not g) and a and s:
        return "acc+screen"

    if (not g) and a and (not s):
        return "acc_only"

    names = []

    if g:
        names.append("gps")

    if a:
        names.append("acc")

    if s:
        names.append("screen")

    return "+".join(names)


wide["evidence_pattern"] = (
    wide.apply(
        evidence_pattern,
        axis=1,
    )
)


episode_meta = episodes[
    [
        "episode_id",
        "participant_id",
        "start_ms",
        "end_ms",
    ]
].merge(
    wide,
    on="episode_id",
    how="left",
    validate="one_to_one",
)


episode_meta["start_date"] = (
    pd.to_datetime(
        episode_meta["start_ms"],
        unit="ms",
        utc=True,
    )
)

episode_meta["start_year"] = (
    episode_meta[
        "start_date"
    ].dt.year
)


print(
    "=" * 70
)

print(
    "EPISODE STREAM EVIDENCE BY START YEAR"
)

print(
    "=" * 70
)

print(
    pd.crosstab(
        episode_meta["start_year"],
        episode_meta[
            "evidence_pattern"
        ],
        margins=True,
    ).to_string()
)


print(
    "\nEarliest episode start by pattern:"
)

print(
    episode_meta.groupby(
        "evidence_pattern"
    )["start_date"]
    .min()
    .to_string()
)


print(
    "\nLatest episode start by pattern:"
)

print(
    episode_meta.groupby(
        "evidence_pattern"
    )["start_date"]
    .max()
    .to_string()
)


# ------------------------------------------------------------
# Daily quality conditional on stream evidence
# ------------------------------------------------------------

def audit_dq(
    df,
    stream,
):
    ev = evidence[
        evidence["stream"].eq(
            stream
        )
    ][
        [
            "episode_id",
            "has_stream_evidence",
        ]
    ]

    z = df.merge(
        ev,
        on="episode_id",
        how="left",
        validate="many_to_one",
    )

    z[
        "has_stream_evidence"
    ] = z[
        "has_stream_evidence"
    ].fillna(False).astype(bool)

    z["dq_zero"] = (
        z["data_quality"]
        .eq(0)
    )

    z["has_observations"] = (
        z["n_observations"]
        .gt(0)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        stream.upper()
    )

    print(
        "=" * 70
    )

    summary = (
        z.groupby(
            [
                "has_stream_evidence",
                "source_status",
            ],
            dropna=False,
        )
        .agg(
            days=(
                "study_day",
                "size",
            ),
            dq_zero=(
                "dq_zero",
                "sum",
            ),
            days_with_observations=(
                "has_observations",
                "sum",
            ),
            mean_dq=(
                "data_quality",
                "mean",
            ),
        )
    )

    print(
        summary.to_string()
    )

    known_zero = z[
        z["source_status"]
        .eq(
            "within_declared_boundary"
        )
        &
        z["data_quality"]
        .eq(0)
    ]

    print(
        "\nZero-DQ days inside declared source boundary:"
    )

    print(
        known_zero[
            "has_stream_evidence"
        ]
        .value_counts(
            dropna=False
        )
        .rename(
            index={
                True:
                    "episode has stream evidence",
                False:
                    "episode has NO stream evidence",
            }
        )
        .to_string()
    )


audit_dq(
    gps,
    "gps",
)

audit_dq(
    acc,
    "accelerometer",
)


# ------------------------------------------------------------
# Screen
# ------------------------------------------------------------

screen_ev = evidence[
    evidence["stream"].eq(
        "screen"
    )
][
    [
        "episode_id",
        "has_stream_evidence",
    ]
]

s = screen.merge(
    screen_ev,
    on="episode_id",
    how="left",
    validate="many_to_one",
)

s[
    "has_stream_evidence"
] = s[
    "has_stream_evidence"
].fillna(False).astype(bool)

s["has_events"] = (
    s["n_events"].gt(0)
)

print(
    "\n" + "=" * 70
)

print("SCREEN")

print("=" * 70)

print(
    s.groupby(
        [
            "has_stream_evidence",
            "source_status",
        ]
    )
    .agg(
        days=(
            "study_day",
            "size",
        ),
        days_with_events=(
            "has_events",
            "sum",
        ),
        total_events=(
            "n_events",
            "sum",
        ),
    )
    .to_string()
)
