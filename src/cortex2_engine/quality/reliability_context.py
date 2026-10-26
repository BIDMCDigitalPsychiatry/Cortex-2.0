from enum import Enum


class DailyObservabilityState(str, Enum):
    SOURCE_UNAVAILABLE = "source_unavailable"
    PARTIAL_SOURCE = "partial_source"
    OBSERVED = "observed"

    DAILY_GAP_WITH_EPISODE_STREAM_EVIDENCE = (
        "daily_gap_with_episode_stream_evidence"
    )

    NO_EPISODE_STREAM_EVIDENCE = (
        "no_episode_stream_evidence"
    )


def classify_daily_observability(
    *,
    source_unavailable: bool,
    partial_source: bool,
    has_stream_evidence: bool,
    daily_observed: bool,
):
    """
    Classify daily sensing evidence without inferring why data
    are absent.

    This intentionally avoids labels such as 'dropout',
    'not instrumented', or 'device failure'.
    """

    if source_unavailable:
        return (
            DailyObservabilityState
            .SOURCE_UNAVAILABLE
        )

    if partial_source:
        return (
            DailyObservabilityState
            .PARTIAL_SOURCE
        )

    if daily_observed:
        return (
            DailyObservabilityState
            .OBSERVED
        )

    if has_stream_evidence:
        return (
            DailyObservabilityState
            .DAILY_GAP_WITH_EPISODE_STREAM_EVIDENCE
        )

    return (
        DailyObservabilityState
        .NO_EPISODE_STREAM_EVIDENCE
    )
