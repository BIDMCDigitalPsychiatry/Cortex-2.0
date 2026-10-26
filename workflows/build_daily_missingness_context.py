from pathlib import Path

import pandas as pd

from cortex2_engine.missingness.context import (
    classify_daily_gap_context,
)


INPUT = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/daily_reliability_context.parquet"
)

OUTPUT = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/daily_missingness_context.parquet"
)

KEY = [
    "participant_id",
    "episode_id",
    "source_participant_id",
    "date",
    "study_day",
]

STREAMS = [
    "gps",
    "accelerometer",
    "screen",
]


df = pd.read_parquet(INPUT)

if df.duplicated(
    KEY + ["stream"]
).any():
    raise RuntimeError(
        "Duplicate stream/day rows found."
    )


# ------------------------------------------------------------
# Build positive cross-stream observation indicators
# ------------------------------------------------------------

observed = (
    df.pivot(
        index=KEY,
        columns="stream",
        values="daily_observed",
    )
    .reset_index()
)

for stream in STREAMS:
    if stream not in observed.columns:
        observed[stream] = False

    observed[stream] = (
        observed[stream]
        .fillna(False)
        .astype(bool)
    )


obs_lookup = observed.set_index(
    KEY
)


rows = []

for row in df.itertuples(
    index=False
):

    key = tuple(
        getattr(row, c)
        for c in KEY
    )

    day = obs_lookup.loc[key]

    other_streams = [
        s
        for s in STREAMS
        if s != row.stream
    ]

    observed_others = [
        s
        for s in other_streams
        if bool(day[s])
    ]

    any_other_observed = (
        len(observed_others) > 0
    )

    context = classify_daily_gap_context(
        observability_state=(
            row.observability_state
        ),
        any_other_stream_observed=(
            any_other_observed
        ),
    )

    record = row._asdict()

    record[
        "any_other_stream_observed"
    ] = any_other_observed

    record[
        "n_other_streams_observed"
    ] = len(observed_others)

    record[
        "other_streams_observed"
    ] = ",".join(
        observed_others
    )

    record[
        "daily_gap_context"
    ] = context.value

    rows.append(record)


out = pd.DataFrame(rows)


# ------------------------------------------------------------
# Temporal context
# ------------------------------------------------------------

dates = pd.to_datetime(
    out["date"],
    utc=True,
)

out["day_of_week"] = (
    dates.dt.day_name()
)

out["weekday_number"] = (
    dates.dt.dayofweek
)

out["is_weekend"] = (
    out["weekday_number"] >= 5
)


# Only genuine within-episode stream gaps are candidates for
# missingness-mechanism analysis.
out[
    "missingness_analysis_eligible"
] = out[
    "observability_state"
].eq(
    "daily_gap_with_episode_stream_evidence"
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
    "Rows:",
    f"{len(out):,}"
)

print(
    "Episodes:",
    out["episode_id"].nunique()
)


print(
    "\nDaily gap context by stream:"
)

print(
    pd.crosstab(
        out["stream"],
        out["daily_gap_context"],
        margins=True,
    ).to_string()
)


eligible = out[
    out[
        "missingness_analysis_eligible"
    ]
].copy()


print(
    "\nEligible daily gaps:"
)

print(
    eligible[
        "stream"
    ].value_counts().to_string()
)


print(
    "\nEligible gaps with positive "
    "other-stream activity:"
)

print(
    pd.crosstab(
        eligible["stream"],
        eligible[
            "any_other_stream_observed"
        ],
        margins=True,
    ).to_string()
)


print(
    "\nWeekend vs weekday gaps:"
)

print(
    pd.crosstab(
        eligible["stream"],
        eligible["is_weekend"],
        margins=True,
    ).to_string()
)


print(
    "\nDay-of-week gap counts:"
)

print(
    pd.crosstab(
        eligible["day_of_week"],
        eligible["stream"],
    ).reindex(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
    ).fillna(0).astype(int).to_string()
)


print(
    "\nSaved:"
)

print(OUTPUT)
