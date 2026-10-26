from cortex2_engine.quality.reliability_context import (
    DailyObservabilityState,
    classify_daily_observability,
)


def test_source_unavailable_has_priority():
    x = classify_daily_observability(
        source_unavailable=True,
        partial_source=False,
        has_stream_evidence=True,
        daily_observed=False,
    )

    assert x == (
        DailyObservabilityState
        .SOURCE_UNAVAILABLE
    )


def test_partial_source():
    x = classify_daily_observability(
        source_unavailable=False,
        partial_source=True,
        has_stream_evidence=True,
        daily_observed=True,
    )

    assert x == (
        DailyObservabilityState
        .PARTIAL_SOURCE
    )


def test_observed():
    x = classify_daily_observability(
        source_unavailable=False,
        partial_source=False,
        has_stream_evidence=True,
        daily_observed=True,
    )

    assert x == (
        DailyObservabilityState
        .OBSERVED
    )


def test_daily_gap_when_stream_exists_elsewhere():
    x = classify_daily_observability(
        source_unavailable=False,
        partial_source=False,
        has_stream_evidence=True,
        daily_observed=False,
    )

    assert x == (
        DailyObservabilityState
        .DAILY_GAP_WITH_EPISODE_STREAM_EVIDENCE
    )


def test_no_episode_stream_evidence():
    x = classify_daily_observability(
        source_unavailable=False,
        partial_source=False,
        has_stream_evidence=False,
        daily_observed=False,
    )

    assert x == (
        DailyObservabilityState
        .NO_EPISODE_STREAM_EVIDENCE
    )
