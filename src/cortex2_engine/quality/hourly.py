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


HOUR_MS = 3_600_000

STREAM_BIN_SECONDS = {
    "gps": 600,
    "accelerometer": 1,
}

OUTSIDE_SOURCE_STATUSES = {
    SourceBoundaryStatus.OUTSIDE_DECLARED_END,
    SourceBoundaryStatus.OUTSIDE_DECLARED_START,
    SourceBoundaryStatus.NO_INTERVAL_OVERLAP,
}


def episode_hourly_quality(
    reader,
    source_intervals,
    *,
    episode_id,
    stream,
    chunksize=1_000_000,
):
    """
    Hourly source-aware observability for one episode/stream.

    Hours are consecutive intervals relative to episode_start.
    Timestamps remain UTC; no behavioral local-time interpretation
    is made here.
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

    hour_starts = np.arange(
        episode_start,
        episode_end,
        HOUR_MS,
        dtype=np.int64,
    )

    hour_ends = np.minimum(
        hour_starts + HOUR_MS,
        episode_end,
    )

    n_hours = len(
        hour_starts
    )

    n_observations = np.zeros(
        n_hours,
        dtype=np.int64,
    )

    first_observed = [
        None
        for _ in range(n_hours)
    ]

    last_observed = [
        None
        for _ in range(n_hours)
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
                            int(hour_ends[i])
                            - int(hour_starts[i])
                        )
                        / bin_ms
                    )
                ),
                dtype=bool,
            )
            for i in range(n_hours)
        ]

    else:
        bin_seconds = None
        bin_ms = None
        occupied = None

    # ---------------------------------------------------------
    # One streaming read of this episode/stream
    # ---------------------------------------------------------

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

        hour_index = (
            (ts - episode_start)
            // HOUR_MS
        ).astype(
            np.int64
        )

        valid = (
            (hour_index >= 0)
            &
            (hour_index < n_hours)
        )

        ts = ts[valid]
        hour_index = hour_index[
            valid
        ]

        if len(ts) == 0:
            continue

        n_observations += np.bincount(
            hour_index,
            minlength=n_hours,
        )

        for h in np.unique(
            hour_index
        ):

            values = ts[
                hour_index == h
            ]

            lo = int(
                values.min()
            )

            hi = int(
                values.max()
            )

            if (
                first_observed[h] is None
                or lo < first_observed[h]
            ):
                first_observed[h] = lo

            if (
                last_observed[h] is None
                or hi > last_observed[h]
            ):
                last_observed[h] = hi

        if occupied is not None:

            offsets = (
                ts
                - hour_starts[
                    hour_index
                ]
            )

            bin_index = (
                offsets
                // bin_ms
            ).astype(
                np.int64
            )

            for h in np.unique(
                hour_index
            ):

                bins = bin_index[
                    hour_index == h
                ]

                bins = bins[
                    (bins >= 0)
                    &
                    (
                        bins
                        < len(
                            occupied[h]
                        )
                    )
                ]

                occupied[h][
                    np.unique(bins)
                ] = True

    # ---------------------------------------------------------
    # Source provenance
    # ---------------------------------------------------------

    boundary = resolve_source_boundary(
        source_intervals,
        episode_id=str(
            ep["episode_id"]
        ),
        stream=stream,
    )

    records = []

    for i in range(n_hours):

        start_ms = int(
            hour_starts[i]
        )

        end_ms = int(
            hour_ends[i]
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

        source_fraction = overlap[
            "source_available_fraction"
        ]

        source_unavailable = (
            status
            in OUTSIDE_SOURCE_STATUSES
        )

        partial_source = (
            not source_unavailable
            and pd.notna(
                source_fraction
            )
            and float(
                source_fraction
            ) > 0
            and float(
                source_fraction
            ) < 1
        )

        hour_observed = bool(
            n_observations[i] > 0
        )

        record = {
            "study_id":
                str(ep["study_id"])
                if "study_id" in ep.index
                else None,

            "participant_id":
                str(ep["participant_id"]),

            "episode_id":
                str(ep["episode_id"]),

            "source_participant_id":
                str(
                    ep[
                        "source_participant_id"
                    ]
                ),

            "stream":
                stream,

            "hour_index":
                i,

            "start_ms":
                start_ms,

            "end_ms":
                end_ms,

            "hour_start_utc":
                pd.to_datetime(
                    start_ms,
                    unit="ms",
                    utc=True,
                ),

            "hour_end_utc":
                pd.to_datetime(
                    end_ms,
                    unit="ms",
                    utc=True,
                ),

            "source_status":
                status.value,

            "source_available_fraction":
                source_fraction,

            "source_unavailable":
                source_unavailable,

            "partial_source":
                partial_source,

            "boundary_unknown":
                not bool(
                    boundary[
                        "boundary_known"
                    ]
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

            "hour_observed":
                hour_observed,

            "first_observed_ms":
                first_observed[i],

            "last_observed_ms":
                last_observed[i],
        }

        if occupied is not None:

            n_bins = len(
                occupied[i]
            )

            n_occupied = int(
                occupied[i].sum()
            )

            longest_missing = int(
                longest_false_run(
                    occupied[i]
                )
            )

            raw_dq = (
                n_occupied / n_bins
                if n_bins
                else np.nan
            )

            final_dq = (
                np.nan
                if (
                    source_unavailable
                    or partial_source
                )
                else raw_dq
            )

            record.update({
                "raw_data_quality":
                    raw_dq,

                "data_quality":
                    final_dq,

                "n_bins":
                    n_bins,

                "n_occupied_bins":
                    n_occupied,

                "n_missing_bins":
                    int(
                        n_bins
                        - n_occupied
                    ),

                "missing_run_count":
                    int(
                        count_false_runs(
                            occupied[i]
                        )
                    ),

                "longest_missing_bins":
                    longest_missing,

                "longest_missing_seconds":
                    int(
                        longest_missing
                        * bin_seconds
                    ),
            })

        else:

            record.update({
                "raw_observed":
                    hour_observed,

                "n_events":
                    int(
                        n_observations[i]
                    ),
            })

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )
