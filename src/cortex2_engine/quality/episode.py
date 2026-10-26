import numpy as np
import pandas as pd

from cortex2_engine.quality.observability import (
    count_false_runs,
    longest_false_run,
)
from cortex2_engine.quality.source_availability import (
    SourceBoundaryStatus,
    classify_source_interval,
    resolve_source_boundary,
    source_interval_overlap,
)


DAY_MS = 86_400_000

STREAM_BIN_SECONDS = {
    "gps": 600,
    "accelerometer": 1,
}

OUTSIDE_SOURCE_STATUSES = {
    SourceBoundaryStatus.OUTSIDE_DECLARED_END,
    SourceBoundaryStatus.OUTSIDE_DECLARED_START,
    SourceBoundaryStatus.NO_INTERVAL_OVERLAP,
}


def _occupied_metrics(
    occupied,
    *,
    bin_seconds,
):
    occupied = np.asarray(
        occupied,
        dtype=bool,
    )

    n_bins = len(occupied)
    n_occupied = int(
        occupied.sum()
    )

    longest_missing = (
        longest_false_run(
            occupied
        )
    )

    return {
        "raw_data_quality": (
            n_occupied / n_bins
            if n_bins
            else np.nan
        ),
        "n_bins": n_bins,
        "n_occupied_bins": n_occupied,
        "n_missing_bins": int(
            n_bins - n_occupied
        ),
        "missing_run_count":
            count_false_runs(
                occupied
            ),
        "longest_missing_bins":
            longest_missing,
        "longest_missing_seconds":
            int(
                longest_missing
                * bin_seconds
            ),
    }


def episode_daily_quality(
    reader,
    source_intervals,
    *,
    episode_id,
    stream,
    chunksize=1_000_000,
):
    """
    Compute reliability-aware daily quality for one episode/stream
    while reading the canonical source only once.

    Study days are consecutive 24-hour intervals beginning at the
    episode start. Digital Clinic episodes are UTC-midnight aligned.
    """

    if stream not in {
        "gps",
        "accelerometer",
        "screen",
    }:
        raise ValueError(
            f"Unsupported stream: {stream}"
        )

    ep = reader.episode(
        episode_id
    )

    episode_start = int(
        ep["start_ms"]
    )

    episode_end = int(
        ep["end_ms"]
    )

    if episode_end <= episode_start:
        raise ValueError(
            "Episode end must be greater than start."
        )

    day_starts = np.arange(
        episode_start,
        episode_end,
        DAY_MS,
        dtype=np.int64,
    )

    day_ends = np.minimum(
        day_starts + DAY_MS,
        episode_end,
    )

    n_days = len(
        day_starts
    )

    n_observations = np.zeros(
        n_days,
        dtype=np.int64,
    )

    first_observed = [
        None
        for _ in range(n_days)
    ]

    last_observed = [
        None
        for _ in range(n_days)
    ]

    if stream in STREAM_BIN_SECONDS:
        bin_seconds = (
            STREAM_BIN_SECONDS[
                stream
            ]
        )

        bin_ms = int(
            bin_seconds * 1000
        )

        occupied = [
            np.zeros(
                int(
                    np.ceil(
                        (
                            int(day_ends[i])
                            - int(day_starts[i])
                        )
                        / bin_ms
                    )
                ),
                dtype=bool,
            )
            for i in range(n_days)
        ]

    else:
        bin_seconds = None
        bin_ms = None
        occupied = None

    for chunk in reader.iter_stream(
        stream,
        episode_id=episode_id,
        chunksize=chunksize,
    ):

        ts = pd.to_numeric(
            chunk["timestamp_ms"],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=np.int64
        )

        ts = ts[
            (ts >= episode_start)
            &
            (ts < episode_end)
        ]

        if len(ts) == 0:
            continue

        day_index = (
            (ts - episode_start)
            // DAY_MS
        ).astype(
            np.int64
        )

        valid = (
            (day_index >= 0)
            &
            (day_index < n_days)
        )

        ts = ts[valid]
        day_index = day_index[
            valid
        ]

        if len(ts) == 0:
            continue

        n_observations += np.bincount(
            day_index,
            minlength=n_days,
        )

        unique_days = np.unique(
            day_index
        )

        for d in unique_days:
            values = ts[
                day_index == d
            ]

            lo = int(
                values.min()
            )

            hi = int(
                values.max()
            )

            if (
                first_observed[d]
                is None
                or lo
                < first_observed[d]
            ):
                first_observed[d] = lo

            if (
                last_observed[d]
                is None
                or hi
                > last_observed[d]
            ):
                last_observed[d] = hi

        if occupied is not None:
            offsets = (
                ts
                - day_starts[
                    day_index
                ]
            )

            bin_index = (
                offsets
                // bin_ms
            ).astype(
                np.int64
            )

            for d in unique_days:
                bins = bin_index[
                    day_index == d
                ]

                bins = bins[
                    (bins >= 0)
                    &
                    (
                        bins
                        < len(
                            occupied[d]
                        )
                    )
                ]

                occupied[d][
                    np.unique(bins)
                ] = True

    boundary = resolve_source_boundary(
        source_intervals,
        episode_id=str(
            ep["episode_id"]
        ),
        stream=stream,
    )

    records = []

    for i in range(n_days):

        start_ms = int(
            day_starts[i]
        )

        end_ms = int(
            day_ends[i]
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

        overlap = source_interval_overlap(
            requested_start_ms=start_ms,
            requested_end_ms=end_ms,
            source_start_ms=boundary[
                "source_start_ms"
            ],
            source_end_ms=boundary[
                "source_end_ms"
            ],
        )

        source_unavailable = (
            status
            in OUTSIDE_SOURCE_STATUSES
        )

        source_fraction = float(
            overlap[
                "source_available_fraction"
            ]
        )

        partial_source = (
            not source_unavailable
            and source_fraction < 1.0
        )

        boundary_unknown = (
            status
            == SourceBoundaryStatus.BOUNDARY_UNKNOWN
        )

        record = {
            "study_id":
                ep.get(
                    "study_id",
                    None,
                ),

            "participant_id":
                str(
                    ep["participant_id"]
                ),

            "episode_id":
                str(
                    ep["episode_id"]
                ),

            "source_participant_id":
                str(
                    ep[
                        "source_participant_id"
                    ]
                ),

            "stream":
                stream,

            "date":
                pd.to_datetime(
                    start_ms,
                    unit="ms",
                    utc=True,
                ).date(),

            "study_day":
                i + 1,

            "start_ms":
                start_ms,

            "end_ms":
                end_ms,

            "source_status":
                status.value,

            "source_available_fraction":
                source_fraction,

            "source_unavailable":
                bool(
                    source_unavailable
                ),

            "partial_source":
                bool(
                    partial_source
                ),

            "boundary_unknown":
                bool(
                    boundary_unknown
                ),

            "boundary_scope":
                boundary[
                    "boundary_scope"
                ],

            "boundary_source":
                boundary[
                    "boundary_source"
                ],

            "n_observations":
                int(
                    n_observations[i]
                ),

            "first_observed_ms":
                (
                    first_observed[i]
                    if first_observed[i]
                    is not None
                    else np.nan
                ),

            "last_observed_ms":
                (
                    last_observed[i]
                    if last_observed[i]
                    is not None
                    else np.nan
                ),
        }

        if occupied is not None:

            metrics = _occupied_metrics(
                occupied[i],
                bin_seconds=(
                    bin_seconds
                ),
            )

            record.update(
                metrics
            )

            if (
                source_unavailable
                or partial_source
            ):
                record[
                    "data_quality"
                ] = np.nan

            else:
                record[
                    "data_quality"
                ] = record[
                    "raw_data_quality"
                ]

        else:
            raw_observed = int(
                n_observations[i] > 0
            )

            record[
                "raw_observed"
            ] = raw_observed

            record["n_events"] = int(
                n_observations[i]
            )

            record["observed"] = (
                np.nan
                if (
                    source_unavailable
                    or partial_source
                )
                else raw_observed
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )
