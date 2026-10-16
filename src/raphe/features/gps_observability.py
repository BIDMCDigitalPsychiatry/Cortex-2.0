"""Temporal observability primitives for irregular GPS data.

RAPHE does not equate presence of GPS records with continuous observation.

An interval between two consecutive GPS observations is considered temporally
observable only when the separation between those observations is less than or
equal to an explicitly supplied continuity tolerance.

Important
---------
`max_gap_ms` is NOT a universal data-quality threshold. It describes how far
apart two observations may be before RAPHE refuses to treat the intervening
time as observed. It should ultimately come from source/sensor semantics or
sensitivity analysis.

No observation time is extrapolated:

- before the first observation,
- after the final observation,
- across gaps larger than `max_gap_ms`.

Windows are [window_start_ms, window_end_ms), i.e. end-exclusive.
"""

from __future__ import annotations

from statistics import median
import math
from typing import Iterable, Any


def _coerce_timestamp(value: Any) -> int | None:
    """Convert a timestamp-like value to integer milliseconds."""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return int(number)


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge touching/overlapping intervals."""

    if not intervals:
        return []

    intervals = sorted(intervals)

    merged: list[list[int]] = [
        [
            intervals[0][0],
            intervals[0][1],
        ]
    ]

    for start, end in intervals[1:]:
        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(
                previous[1],
                end,
            )
        else:
            merged.append(
                [
                    start,
                    end,
                ]
            )

    return [
        (
            start,
            end,
        )
        for start, end in merged
    ]


def _longest_unobserved_gap(
    *,
    window_start_ms: int,
    window_end_ms: int,
    observed_segments: list[tuple[int, int]],
) -> int:
    """Return longest interval not supported by temporal observation."""

    if not observed_segments:
        return (
            window_end_ms
            - window_start_ms
        )

    cursor = window_start_ms
    longest = 0

    for start, end in observed_segments:
        if start > cursor:
            longest = max(
                longest,
                start - cursor,
            )

        cursor = max(
            cursor,
            end,
        )

    if cursor < window_end_ms:
        longest = max(
            longest,
            window_end_ms - cursor,
        )

    return longest


def compute_gps_temporal_observability(
    timestamps_ms: Iterable[Any],
    *,
    window_start_ms: int,
    window_end_ms: int,
    max_gap_ms: int,
) -> dict[str, Any]:
    """Summarize temporal observability within one analysis window.

    Parameters
    ----------
    timestamps_ms
        GPS observation timestamps in milliseconds. Ordering is irrelevant.
        Invalid timestamps are ignored but counted.
    window_start_ms
        Inclusive beginning of analysis window.
    window_end_ms
        Exclusive end of analysis window.
    max_gap_ms
        Maximum separation between consecutive observations for the
        intervening time to be considered observable.

    Returns
    -------
    dict
        Temporal observability and provenance statistics.
    """

    window_start_ms = int(
        window_start_ms
    )
    window_end_ms = int(
        window_end_ms
    )
    max_gap_ms = int(
        max_gap_ms
    )

    if window_end_ms <= window_start_ms:
        raise ValueError(
            "window_end_ms must be greater than window_start_ms"
        )

    if max_gap_ms <= 0:
        raise ValueError(
            "max_gap_ms must be greater than zero"
        )

    raw_values = list(
        timestamps_ms
    )

    valid_values: list[int] = []
    n_invalid = 0
    n_outside_window = 0

    for value in raw_values:
        timestamp = _coerce_timestamp(
            value
        )

        if timestamp is None:
            n_invalid += 1
            continue

        if not (
            window_start_ms
            <= timestamp
            < window_end_ms
        ):
            n_outside_window += 1
            continue

        valid_values.append(
            timestamp
        )

    n_valid_before_dedup = len(
        valid_values
    )

    timestamps = sorted(
        set(
            valid_values
        )
    )

    n_duplicates = (
        n_valid_before_dedup
        - len(
            timestamps
        )
    )

    window_duration_ms = (
        window_end_ms
        - window_start_ms
    )

    if not timestamps:
        return {
            "status":
                "no_raw_data",
            "window_start_ms":
                window_start_ms,
            "window_end_ms":
                window_end_ms,
            "window_duration_ms":
                window_duration_ms,
            "max_gap_ms":
                max_gap_ms,
            "n_input_timestamps":
                len(raw_values),
            "n_invalid_timestamps":
                n_invalid,
            "n_outside_window":
                n_outside_window,
            "n_duplicate_timestamps":
                n_duplicates,
            "n_observations":
                0,
            "n_supported_edges":
                0,
            "n_observable_segments":
                0,
            "observed_duration_ms":
                0,
            "unobserved_duration_ms":
                window_duration_ms,
            "observability":
                0.0,
            "longest_unobserved_gap_ms":
                window_duration_ms,
            "leading_gap_ms":
                window_duration_ms,
            "trailing_gap_ms":
                window_duration_ms,
            "first_observation_ms":
                None,
            "last_observation_ms":
                None,
            "observation_span_ms":
                0,
            "median_sampling_interval_ms":
                None,
        }

    first_observation = (
        timestamps[0]
    )

    last_observation = (
        timestamps[-1]
    )

    leading_gap_ms = (
        first_observation
        - window_start_ms
    )

    trailing_gap_ms = (
        window_end_ms
        - last_observation
    )

    if len(timestamps) >= 2:
        deltas = [
            right - left
            for left, right in zip(
                timestamps[:-1],
                timestamps[1:],
            )
        ]

        median_sampling_interval_ms = median(
            deltas
        )

    else:
        deltas = []

        median_sampling_interval_ms = None

    supported_edges: list[
        tuple[int, int]
    ] = []

    for left, right in zip(
        timestamps[:-1],
        timestamps[1:],
    ):
        gap = (
            right
            - left
        )

        if gap <= max_gap_ms:
            supported_edges.append(
                (
                    left,
                    right,
                )
            )

    observed_segments = _merge_intervals(
        supported_edges
    )

    observed_duration_ms = sum(
        end - start
        for start, end in observed_segments
    )

    unobserved_duration_ms = (
        window_duration_ms
        - observed_duration_ms
    )

    observability = (
        observed_duration_ms
        / window_duration_ms
    )

    longest_unobserved_gap_ms = (
        _longest_unobserved_gap(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            observed_segments=observed_segments,
        )
    )

    if not observed_segments:
        status = (
            "insufficient_observability"
        )
    else:
        status = (
            "observed_value"
        )

    return {
        "status":
            status,
        "window_start_ms":
            window_start_ms,
        "window_end_ms":
            window_end_ms,
        "window_duration_ms":
            window_duration_ms,
        "max_gap_ms":
            max_gap_ms,
        "n_input_timestamps":
            len(raw_values),
        "n_invalid_timestamps":
            n_invalid,
        "n_outside_window":
            n_outside_window,
        "n_duplicate_timestamps":
            n_duplicates,
        "n_observations":
            len(timestamps),
        "n_supported_edges":
            len(supported_edges),
        "n_observable_segments":
            len(observed_segments),
        "observed_duration_ms":
            observed_duration_ms,
        "unobserved_duration_ms":
            unobserved_duration_ms,
        "observability":
            observability,
        "longest_unobserved_gap_ms":
            longest_unobserved_gap_ms,
        "leading_gap_ms":
            leading_gap_ms,
        "trailing_gap_ms":
            trailing_gap_ms,
        "first_observation_ms":
            first_observation,
        "last_observation_ms":
            last_observation,
        "observation_span_ms":
            last_observation
            - first_observation,
        "median_sampling_interval_ms":
            median_sampling_interval_ms,
    }
