from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.missingness.runs import (
    assign_contiguous_runs,
    EXCLUDED_PATTERNS,
)


ROOT = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness"
)

EPISODE_DIR = (
    ROOT / "episodes"
)

OUTPUT = (
    ROOT / "gap_runs.parquet"
)

SUMMARY_OUTPUT = (
    ROOT / "gap_run_summary.csv"
)


files = sorted(
    EPISODE_DIR.glob("*.parquet")
)

print(
    "Episode checkpoint files:",
    len(files),
)


frames = []

for path in files:

    x = pd.read_parquet(path)

    x = assign_contiguous_runs(
        x
    )

    frames.append(x)


hourly = pd.concat(
    frames,
    ignore_index=True,
    sort=False,
)


hourly = hourly[
    ~hourly["gap_pattern"]
    .isin(EXCLUDED_PATTERNS)
].copy()


rows = []


for (
    episode_id,
    run_id,
), g in hourly.groupby(
    [
        "episode_id",
        "run_id",
    ],
    sort=False,
):

    g = g.sort_values(
        "hour_index"
    )

    pattern = str(
        g["gap_pattern"].iloc[0]
    )

    if g[
        "gap_pattern"
    ].nunique() != 1:
        raise RuntimeError(
            "Mixed gap pattern inside run."
        )


    if pattern == "gps_specific_gap":
        target_stream = "gps"
        supporting_stream = "accelerometer"
        supporting_dq = g["acc_dq"]

    elif pattern == "acc_specific_gap":
        target_stream = "accelerometer"
        supporting_stream = "gps"
        supporting_dq = g["gps_dq"]

    else:
        target_stream = "gps+accelerometer"
        supporting_stream = None
        supporting_dq = pd.Series(
            dtype=float
        )


    screen_hours = int(
        g[
            "screen_event_observed"
        ].sum()
    )


    rows.append({
        "participant_id":
            str(
                g["participant_id"]
                .iloc[0]
            ),

        "episode_id":
            str(episode_id),

        "source_participant_id":
            str(
                g[
                    "source_participant_id"
                ].iloc[0]
            ),

        "run_id":
            int(run_id),

        "gap_pattern":
            pattern,

        "target_stream":
            target_stream,

        "supporting_stream":
            supporting_stream,

        "start_hour_index":
            int(
                g["hour_index"].min()
            ),

        "end_hour_index":
            int(
                g["hour_index"].max()
            ),

        "start_utc":
            g[
                "hour_start_utc"
            ].iloc[0],

        "end_utc":
            g[
                "hour_end_utc"
            ].iloc[-1],

        "duration_hours":
            int(len(g)),

        "screen_event_hours":
            screen_hours,

        "screen_event_fraction":
            (
                screen_hours / len(g)
            ),

        "mean_supporting_dq":
            (
                float(
                    supporting_dq.mean()
                )
                if len(
                    supporting_dq
                )
                else np.nan
            ),

        "min_supporting_dq":
            (
                float(
                    supporting_dq.min()
                )
                if len(
                    supporting_dq
                )
                else np.nan
            ),

        "median_supporting_dq":
            (
                float(
                    supporting_dq.median()
                )
                if len(
                    supporting_dq
                )
                else np.nan
            ),

        "max_supporting_dq":
            (
                float(
                    supporting_dq.max()
                )
                if len(
                    supporting_dq
                )
                else np.nan
            ),

        "any_boundary_unknown":
            bool(
                g[
                    "any_boundary_unknown"
                ].any()
            ),
    })


runs = pd.DataFrame(
    rows
)


runs.to_parquet(
    OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary_rows = []

for pattern, g in runs.groupby(
    "gap_pattern"
):

    summary_rows.append({
        "gap_pattern":
            pattern,

        "runs":
            len(g),

        "participants":
            g[
                "participant_id"
            ].nunique(),

        "episodes":
            g[
                "episode_id"
            ].nunique(),

        "total_gap_hours":
            int(
                g[
                    "duration_hours"
                ].sum()
            ),

        "median_run_hours":
            g[
                "duration_hours"
            ].median(),

        "p90_run_hours":
            g[
                "duration_hours"
            ].quantile(0.90),

        "p95_run_hours":
            g[
                "duration_hours"
            ].quantile(0.95),

        "max_run_hours":
            g[
                "duration_hours"
            ].max(),

        "runs_with_screen_events":
            int(
                g[
                    "screen_event_hours"
                ].gt(0).sum()
            ),

        "median_screen_event_fraction":
            g[
                "screen_event_fraction"
            ].median(),

        "median_supporting_dq":
            g[
                "mean_supporting_dq"
            ].median(),
    })


summary = pd.DataFrame(
    summary_rows
).sort_values(
    "gap_pattern"
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print(
    "\n" + "=" * 70
)

print(
    "GAP RUN SUMMARY"
)

print(
    "=" * 70
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nLongest GPS-specific runs:"
)

print(
    runs[
        runs["gap_pattern"]
        .eq("gps_specific_gap")
    ]
    .sort_values(
        "duration_hours",
        ascending=False,
    )
    [
        [
            "participant_id",
            "episode_id",
            "start_utc",
            "end_utc",
            "duration_hours",
            "mean_supporting_dq",
            "min_supporting_dq",
            "screen_event_fraction",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(OUTPUT)
print(SUMMARY_OUTPUT)
