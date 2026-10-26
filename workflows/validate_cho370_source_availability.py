import pandas as pd

from cortex2_engine.quality.source_availability import (
    classify_source_interval,
    resolve_source_boundary,
)


EPISODES = (
    "../cortex2_outputs/digital_clinic/"
    "inventory/episodes.parquet"
)

SOURCE_INTERVALS = (
    "../cortex2_outputs/digital_clinic/"
    "inventory/source_intervals.parquet"
)


episodes = pd.read_parquet(EPISODES)
source_intervals = pd.read_parquet(SOURCE_INTERVALS)

ep = episodes[
    episodes["participant_id"].eq("CHO370")
].iloc[0]


print("Participant:", ep["participant_id"])
print("Episode:", ep["episode_id"])
print(
    "Window:",
    ep["start_utc"],
    "->",
    ep["end_utc"],
)


for stream in [
    "gps",
    "accelerometer",
    "screen",
]:

    boundary = resolve_source_boundary(
        source_intervals,
        episode_id=ep["episode_id"],
        stream=stream,
    )

    print("\n" + "=" * 60)
    print(stream.upper())
    print("=" * 60)

    print(
        "Boundary scope:",
        boundary["boundary_scope"],
    )

    print(
        "Boundary source:",
        boundary["boundary_source"],
    )

    if boundary["source_end_ms"] is not None:
        print(
            "Declared end:",
            pd.to_datetime(
                boundary["source_end_ms"],
                unit="ms",
                utc=True,
            ),
        )

    rows = []

    day_start = ep["start_utc"]

    while day_start < ep["end_utc"]:

        day_end = day_start + pd.Timedelta(
            days=1
        )

        start_ms = int(
            day_start.timestamp() * 1000
        )

        end_ms = int(
            day_end.timestamp() * 1000
        )

        status = classify_source_interval(
            requested_start_ms=start_ms,
            requested_end_ms=end_ms,
            source_start_ms=boundary[
                "source_start_ms"
            ],
            source_end_ms=boundary[
                "source_end_ms"
            ],
        )

        rows.append({
            "date": day_start.date(),
            "status": status.value,
        })

        day_start = day_end

    daily = pd.DataFrame(rows)

    print("\nCounts:")
    print(
        daily["status"]
        .value_counts()
        .to_string()
    )

    print("\nLast 3 days:")
    print(
        daily.tail(3)
        .to_string(index=False)
    )
