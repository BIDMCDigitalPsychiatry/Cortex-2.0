import numpy as np

from cortex2_engine.quality.source_availability import (
    SourceBoundaryStatus,
    classify_source_interval,
    source_interval_overlap,
)
from cortex2_engine.quality.streams import (
    accelerometer_quality,
    gps_quality,
    screen_quality,
)


OUTSIDE_SOURCE_STATUSES = {
    SourceBoundaryStatus.OUTSIDE_DECLARED_END,
    SourceBoundaryStatus.OUTSIDE_DECLARED_START,
    SourceBoundaryStatus.NO_INTERVAL_OVERLAP,
}


def daily_stream_quality(
    df,
    *,
    stream: str,
    start_ms: int,
    end_ms: int,
    source_start_ms: int | None = None,
    source_end_ms: int | None = None,
    boundary_scope: str | None = None,
    boundary_source: str | None = None,
):
    """
    Compute one reliability-aware daily quality record.

    Source availability is evaluated separately from sensor
    observability.

    Semantics
    ---------
    outside declared source interval:
        final data_quality = NA

    partially covered requested interval:
        final data_quality = NA

    fully covered interval with no observations:
        data_quality = 0 for streams with scalar DQ

    unknown source boundary:
        compute DQ from available observations, but explicitly
        preserve boundary_unknown=True.

    Low-level sensor metrics are calculated over the requested
    interval and retained for audit.
    """

    if end_ms <= start_ms:
        raise ValueError(
            "end_ms must be greater than start_ms"
        )

    status = classify_source_interval(
        requested_start_ms=start_ms,
        requested_end_ms=end_ms,
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
    )

    overlap = source_interval_overlap(
        requested_start_ms=start_ms,
        requested_end_ms=end_ms,
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
    )

    source_unavailable = (
        status in OUTSIDE_SOURCE_STATUSES
    )

    source_fraction = float(
        overlap["source_available_fraction"]
    )

    partial_source = (
        not source_unavailable
        and source_fraction < 1.0
    )

    boundary_unknown = (
        status
        == SourceBoundaryStatus.BOUNDARY_UNKNOWN
    )

    if stream == "gps":
        metrics = gps_quality(
            df,
            start_ms=start_ms,
            end_ms=end_ms,
            bin_seconds=600,
        )

    elif stream == "accelerometer":
        metrics = accelerometer_quality(
            df,
            start_ms=start_ms,
            end_ms=end_ms,
            bin_seconds=1,
        )

    elif stream == "screen":
        metrics = screen_quality(
            df,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    else:
        raise ValueError(
            f"Unsupported stream: {stream}"
        )

    out = {
        "stream": stream,
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "source_status": status.value,
        "source_available_fraction": source_fraction,
        "source_unavailable": bool(
            source_unavailable
        ),
        "partial_source": bool(
            partial_source
        ),
        "boundary_unknown": bool(
            boundary_unknown
        ),
        "boundary_scope": boundary_scope,
        "boundary_source": boundary_source,
    }

    out.update(metrics)

    # GPS and accelerometer currently have scalar DQ.
    if "data_quality" in metrics:
        out["raw_data_quality"] = float(
            metrics["data_quality"]
        )

        if (
            source_unavailable
            or partial_source
        ):
            out["data_quality"] = np.nan

    # Screen intentionally does not fabricate a scalar DQ.
    if stream == "screen":
        out["raw_observed"] = int(
            metrics["observed"]
        )

        if (
            source_unavailable
            or partial_source
        ):
            out["observed"] = np.nan

    return out
