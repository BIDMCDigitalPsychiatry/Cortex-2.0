from enum import Enum


class DailyGapContext(str, Enum):
    OBSERVED = "observed"

    SOURCE_UNAVAILABLE = (
        "source_unavailable"
    )

    NO_EPISODE_STREAM_EVIDENCE = (
        "no_episode_stream_evidence"
    )

    GAP_WITH_OTHER_STREAM_ACTIVITY = (
        "gap_with_other_stream_activity"
    )

    GAP_WITHOUT_OTHER_STREAM_ACTIVITY_EVIDENCE = (
        "gap_without_other_stream_activity_evidence"
    )


def classify_daily_gap_context(
    *,
    observability_state: str,
    any_other_stream_observed: bool,
):
    """
    Characterize a target stream's daily missingness using
    positive cross-stream evidence.

    This intentionally does not infer MCAR, MAR, MNAR,
    device failure, intentional disabling, or non-wear.
    """

    if observability_state == "observed":
        return DailyGapContext.OBSERVED

    if observability_state == "source_unavailable":
        return DailyGapContext.SOURCE_UNAVAILABLE

    if observability_state == "no_episode_stream_evidence":
        return (
            DailyGapContext
            .NO_EPISODE_STREAM_EVIDENCE
        )

    if (
        observability_state
        == "daily_gap_with_episode_stream_evidence"
    ):
        if any_other_stream_observed:
            return (
                DailyGapContext
                .GAP_WITH_OTHER_STREAM_ACTIVITY
            )

        return (
            DailyGapContext
            .GAP_WITHOUT_OTHER_STREAM_ACTIVITY_EVIDENCE
        )

    if observability_state == "partial_source":
        return DailyGapContext.SOURCE_UNAVAILABLE

    raise ValueError(
        "Unsupported observability_state: "
        f"{observability_state}"
    )
