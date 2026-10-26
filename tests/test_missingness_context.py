from cortex2_engine.missingness.context import (
    DailyGapContext,
    classify_daily_gap_context,
)


def test_observed():
    x = classify_daily_gap_context(
        observability_state="observed",
        any_other_stream_observed=True,
    )

    assert x == DailyGapContext.OBSERVED


def test_source_unavailable():
    x = classify_daily_gap_context(
        observability_state="source_unavailable",
        any_other_stream_observed=False,
    )

    assert x == (
        DailyGapContext.SOURCE_UNAVAILABLE
    )


def test_no_episode_stream_evidence():
    x = classify_daily_gap_context(
        observability_state=(
            "no_episode_stream_evidence"
        ),
        any_other_stream_observed=True,
    )

    assert x == (
        DailyGapContext
        .NO_EPISODE_STREAM_EVIDENCE
    )


def test_gap_with_other_stream_activity():
    x = classify_daily_gap_context(
        observability_state=(
            "daily_gap_with_episode_stream_evidence"
        ),
        any_other_stream_observed=True,
    )

    assert x == (
        DailyGapContext
        .GAP_WITH_OTHER_STREAM_ACTIVITY
    )


def test_gap_without_positive_other_stream_evidence():
    x = classify_daily_gap_context(
        observability_state=(
            "daily_gap_with_episode_stream_evidence"
        ),
        any_other_stream_observed=False,
    )

    assert x == (
        DailyGapContext
        .GAP_WITHOUT_OTHER_STREAM_ACTIVITY_EVIDENCE
    )
