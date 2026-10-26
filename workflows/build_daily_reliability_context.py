from pathlib import Path

import pandas as pd

from cortex2_engine.quality.reliability_context import (
    classify_daily_observability,
)


ROOT = Path(
    "../cortex2_outputs/digital_clinic/quality"
)

CONTEXT = ROOT / "episode_stream_context.parquet"

STREAM_FILES = {
    "gps": ROOT / "gps_daily.parquet",
    "accelerometer":
        ROOT / "accelerometer_daily.parquet",
    "screen":
        ROOT / "screen_daily.parquet",
}

OUTPUT = (
    ROOT
    / "daily_reliability_context.parquet"
)


context = pd.read_parquet(
    CONTEXT
)

frames = []


for stream, path in STREAM_FILES.items():

    daily = pd.read_parquet(
        path
    )

    ctx = context[
        context["stream"].eq(
            stream
        )
    ][
        [
            "episode_id",
            "has_stream_evidence",
            "overlapping_files",
            "stream_evidence_context",
        ]
    ]

    daily = daily.merge(
        ctx,
        on="episode_id",
        how="left",
        validate="many_to_one",
    )

    if daily[
        "has_stream_evidence"
    ].isna().any():
        raise RuntimeError(
            f"Missing stream context for {stream}"
        )

    if stream == "screen":
        daily["daily_observed"] = (
            daily["n_events"]
            .gt(0)
        )

    else:
        daily["daily_observed"] = (
            daily["n_observations"]
            .gt(0)
        )

    daily[
        "observability_state"
    ] = [
        classify_daily_observability(
            source_unavailable=bool(su),
            partial_source=bool(ps),
            has_stream_evidence=bool(se),
            daily_observed=bool(obs),
        ).value
        for su, ps, se, obs in zip(
            daily["source_unavailable"],
            daily["partial_source"],
            daily["has_stream_evidence"],
            daily["daily_observed"],
        )
    ]

    frames.append(
        daily
    )


out = pd.concat(
    frames,
    ignore_index=True,
    sort=False,
)

out = out.sort_values(
    [
        "participant_id",
        "episode_id",
        "study_day",
        "stream",
    ]
).reset_index(
    drop=True
)


if len(out) != 3 * 20586:
    raise RuntimeError(
        f"Unexpected daily row count: {len(out)}"
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
    f"{len(out):,}",
)

print(
    "Episodes:",
    out["episode_id"].nunique()
)

print(
    "\nObservability state by stream:"
)

print(
    pd.crosstab(
        out["stream"],
        out["observability_state"],
        margins=True,
    ).to_string()
)


print(
    "\nBoundary scope × observability state:"
)

print(
    pd.crosstab(
        [
            out["stream"],
            out["boundary_scope"]
            .fillna("unknown"),
        ],
        out["observability_state"],
    ).to_string()
)


print(
    "\nZero DQ interpretation:"
)

scalar = out[
    out["stream"].isin(
        [
            "gps",
            "accelerometer",
        ]
    )
].copy()

zeros = scalar[
    scalar["data_quality"].eq(0)
]

print(
    pd.crosstab(
        zeros["stream"],
        zeros["observability_state"],
        margins=True,
    ).to_string()
)


print("\nSaved:")
print(OUTPUT)
