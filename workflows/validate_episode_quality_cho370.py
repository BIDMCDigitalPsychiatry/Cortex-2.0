from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter
from cortex2_engine.canonical.reader import EpisodeReader
from cortex2_engine.quality.episode import episode_daily_quality


PARTICIPANT_ID = "CHO370"

EPISODES = Path(
    "../cortex2_outputs/digital_clinic/inventory/episodes.parquet"
)

SOURCE_INTERVALS = Path(
    "../cortex2_outputs/digital_clinic/inventory/source_intervals.parquet"
)

SOURCE_MANIFESTS = Path(
    "../cortex2_outputs/digital_clinic/inventory/source_files"
)

PASSIVE_ROOT = Path("../passive")

BENCHMARK = Path(
    "../cortex_longitudinal/data/"
    "benchmark_5_full/"
    "benchmark_5_daily_master.csv"
)


episodes = pd.read_parquet(EPISODES)

source_intervals = pd.read_parquet(
    SOURCE_INTERVALS
)

ep = episodes[
    episodes["participant_id"].eq(
        PARTICIPANT_ID
    )
].iloc[0]

adapter = MindLAMPAdapter(
    study_id="digital_clinic",
    passive_root=PASSIVE_ROOT,
    source_manifest_dir=SOURCE_MANIFESTS,
)

reader = EpisodeReader(
    adapter,
    episodes,
)

benchmark = pd.read_csv(
    BENCHMARK
)

benchmark = benchmark[
    benchmark["participant_id"].eq(
        PARTICIPANT_ID
    )
].copy()

benchmark["date"] = (
    pd.to_datetime(
        benchmark["date"]
    )
    .dt.date
)

print("Participant:", PARTICIPANT_ID)
print("Episode:", ep["episode_id"])
print(
    "Window:",
    ep["start_utc"],
    "->",
    ep["end_utc"],
)


for stream, cortex_column in [
    ("gps", "gps_data_quality"),
    (
        "accelerometer",
        "acc_data_quality",
    ),
]:

    print("\n" + "=" * 70)
    print(stream.upper())
    print("=" * 70)

    cortex2 = episode_daily_quality(
        reader,
        source_intervals,
        episode_id=ep["episode_id"],
        stream=stream,
        chunksize=250_000,
    )

    merged = cortex2.merge(
        benchmark[
            [
                "date",
                "study_day",
                cortex_column,
            ]
        ],
        on=[
            "date",
            "study_day",
        ],
        how="left",
        validate="one_to_one",
    )

    merged["cortex_dq"] = (
        pd.to_numeric(
            merged[cortex_column],
            errors="coerce",
        )
    )

    merged["abs_diff"] = np.abs(
        merged["data_quality"]
        - merged["cortex_dq"]
    )

    comparable = (
        merged["data_quality"].notna()
        &
        merged["cortex_dq"].notna()
    )

    matching = (
        comparable
        &
        (
            merged["abs_diff"]
            <= 1e-12
        )
    )

    print(
        "Days:",
        len(merged),
    )

    print(
        "Comparable:",
        int(comparable.sum()),
    )

    print(
        "Matching:",
        int(matching.sum()),
        "/",
        int(comparable.sum()),
    )

    print(
        "Source unavailable:",
        int(
            merged[
                "source_unavailable"
            ].sum()
        ),
    )

    if comparable.any():
        print(
            "Max abs diff:",
            merged.loc[
                comparable,
                "abs_diff",
            ].max(),
        )

    print("\nFinal 3 days:")

    print(
        merged[
            [
                "date",
                "study_day",
                "source_status",
                "raw_data_quality",
                "data_quality",
                "cortex_dq",
            ]
        ]
        .tail(3)
        .to_string(
            index=False
        )
    )

    assert (
        int(comparable.sum())
        == len(merged) - 1
    )

    assert (
        int(matching.sum())
        == int(comparable.sum())
    )

    assert (
        merged.iloc[-1][
            "source_status"
        ]
        == "outside_declared_end"
    )

    assert np.isnan(
        merged.iloc[-1][
            "data_quality"
        ]
    )


print(
    "\nPASS — reusable episode_daily_quality() "
    "reproduces CHO370 Cortex DQ on every "
    "source-comparable day."
)
