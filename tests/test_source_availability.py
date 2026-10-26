from cortex2_engine.quality.source_availability import (
    SourceBoundaryStatus,
    classify_source_interval,
)


DAY = 86_400_000


def test_day_after_export_end():

    status = classify_source_interval(
        requested_start_ms=DAY,
        requested_end_ms=2 * DAY,
        source_end_ms=DAY,
    )

    assert status == (
        SourceBoundaryStatus.OUTSIDE_DECLARED_END
    )


def test_day_before_export_end():

    status = classify_source_interval(
        requested_start_ms=0,
        requested_end_ms=DAY,
        source_end_ms=2 * DAY,
    )

    assert status == (
        SourceBoundaryStatus.WITHIN_DECLARED_BOUNDARY
    )


def test_interval_entirely_before_source():

    status = classify_source_interval(
        requested_start_ms=0,
        requested_end_ms=DAY,
        source_start_ms=2 * DAY,
        source_end_ms=4 * DAY,
    )

    assert status == (
        SourceBoundaryStatus.OUTSIDE_DECLARED_START
    )


def test_unknown_boundaries():

    status = classify_source_interval(
        requested_start_ms=0,
        requested_end_ms=DAY,
    )

    assert status == (
        SourceBoundaryStatus.BOUNDARY_UNKNOWN
    )
