from enum import Enum

import pandas as pd


def _optional_int(value):
    """Convert a nullable scalar to int without inventing values."""
    if pd.isna(value):
        return None

    return int(value)


class SourceBoundaryStatus(str, Enum):
    """
    Relationship between a requested analysis interval and
    known source/export boundaries.
    """

    WITHIN_DECLARED_BOUNDARY = "within_declared_boundary"

    OUTSIDE_DECLARED_END = "outside_declared_end"

    OUTSIDE_DECLARED_START = "outside_declared_start"

    NO_INTERVAL_OVERLAP = "no_interval_overlap"

    BOUNDARY_UNKNOWN = "boundary_unknown"


def classify_source_interval(
    *,
    requested_start_ms: int,
    requested_end_ms: int,
    source_start_ms: int | None = None,
    source_end_ms: int | None = None,
):
    """
    Classify requested interval against known source boundaries.

    Intervals use half-open semantics:
        [start, end)

    Unknown boundaries are never inferred from missing observations.
    """

    if requested_end_ms <= requested_start_ms:
        raise ValueError(
            "requested_end_ms must be greater than "
            "requested_start_ms"
        )

    if (
        source_start_ms is not None
        and source_end_ms is not None
        and source_end_ms <= source_start_ms
    ):
        raise ValueError(
            "source_end_ms must be greater than source_start_ms"
        )

    if source_end_ms is not None:
        if requested_start_ms >= source_end_ms:
            return SourceBoundaryStatus.OUTSIDE_DECLARED_END

    if source_start_ms is not None:
        if requested_end_ms <= source_start_ms:
            return SourceBoundaryStatus.OUTSIDE_DECLARED_START

    if (
        source_start_ms is not None
        and source_end_ms is not None
    ):
        overlap = (
            requested_start_ms < source_end_ms
            and requested_end_ms > source_start_ms
        )

        if not overlap:
            return SourceBoundaryStatus.NO_INTERVAL_OVERLAP

        return SourceBoundaryStatus.WITHIN_DECLARED_BOUNDARY

    if source_end_ms is not None:
        # We know the requested interval is not beyond the
        # declared upper boundary, but the lower boundary is
        # unknown.
        return SourceBoundaryStatus.WITHIN_DECLARED_BOUNDARY

    if source_start_ms is not None:
        return SourceBoundaryStatus.WITHIN_DECLARED_BOUNDARY

    return SourceBoundaryStatus.BOUNDARY_UNKNOWN


def resolve_source_boundary(
    source_intervals,
    *,
    episode_id: str,
    stream: str,
):
    """
    Resolve the best available declared source boundary for an
    episode/stream.

    Priority:
        1. stream-specific boundary
        2. participant-level boundary
        3. no known boundary

    Returns a dictionary rather than inferring availability from
    observed sensor samples.
    """

    ep = source_intervals[
        source_intervals["episode_id"]
        .astype(str)
        .eq(str(episode_id))
    ]

    if ep.empty:
        return {
            "boundary_known": False,
            "boundary_scope": None,
            "boundary_source": None,
            "source_start_ms": None,
            "source_end_ms": None,
        }

    specific = ep[
        ep["stream"]
        .fillna("")
        .astype(str)
        .eq(str(stream))
    ]

    if not specific.empty:
        if len(specific) > 1:
            raise ValueError(
                f"Multiple stream-specific source boundaries "
                f"for episode={episode_id}, stream={stream}"
            )

        row = specific.iloc[0]

        return {
            "boundary_known": True,
            "boundary_scope": "stream",
            "boundary_source": row["boundary_source"],
            "source_start_ms": _optional_int(
                row["source_start_ms"]
            ),
            "source_end_ms": _optional_int(
                row["source_end_ms"]
            ),
        }

    participant_level = ep[
        ep["stream"].isna()
    ]

    if not participant_level.empty:
        if len(participant_level) > 1:
            raise ValueError(
                f"Multiple participant-level source boundaries "
                f"for episode={episode_id}"
            )

        row = participant_level.iloc[0]

        return {
            "boundary_known": True,
            "boundary_scope": "participant",
            "boundary_source": row["boundary_source"],
            "source_start_ms": _optional_int(
                row["source_start_ms"]
            ),
            "source_end_ms": _optional_int(
                row["source_end_ms"]
            ),
        }

    return {
        "boundary_known": False,
        "boundary_scope": None,
        "boundary_source": None,
        "source_start_ms": None,
        "source_end_ms": None,
    }


def source_interval_overlap(
    *,
    requested_start_ms: int,
    requested_end_ms: int,
    source_start_ms: int | None = None,
    source_end_ms: int | None = None,
):
    """
    Return the portion of [requested_start_ms, requested_end_ms)
    that is compatible with known source boundaries.

    Unknown boundaries remain unbounded rather than being inferred
    from observed data.
    """

    if requested_end_ms <= requested_start_ms:
        raise ValueError(
            "requested_end_ms must be greater than requested_start_ms"
        )

    overlap_start = requested_start_ms
    overlap_end = requested_end_ms

    if source_start_ms is not None:
        overlap_start = max(
            overlap_start,
            int(source_start_ms),
        )

    if source_end_ms is not None:
        overlap_end = min(
            overlap_end,
            int(source_end_ms),
        )

    overlap_ms = max(
        0,
        overlap_end - overlap_start,
    )

    requested_ms = (
        requested_end_ms
        - requested_start_ms
    )

    return {
        "overlap_start_ms": (
            overlap_start
            if overlap_ms > 0
            else None
        ),
        "overlap_end_ms": (
            overlap_end
            if overlap_ms > 0
            else None
        ),
        "overlap_ms": overlap_ms,
        "requested_ms": requested_ms,
        "source_available_fraction": (
            overlap_ms / requested_ms
        ),
    }
