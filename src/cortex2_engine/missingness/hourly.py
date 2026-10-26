from enum import Enum


class HourlyGapPattern(str, Enum):
    BOTH_OBSERVED = "both_observed"

    GPS_SPECIFIC_GAP = (
        "gps_specific_gap"
    )

    ACC_SPECIFIC_GAP = (
        "acc_specific_gap"
    )

    SHARED_GAP_WITH_SCREEN_ACTIVITY = (
        "shared_gap_with_screen_activity"
    )

    SHARED_GAP_WITHOUT_SCREEN_ACTIVITY_EVIDENCE = (
        "shared_gap_without_screen_activity_evidence"
    )

    SOURCE_UNAVAILABLE_OR_PARTIAL = (
        "source_unavailable_or_partial"
    )


def classify_hourly_gap_pattern(
    *,
    gps_observed: bool,
    acc_observed: bool,
    screen_event_observed: bool,
    gps_source_unavailable: bool,
    acc_source_unavailable: bool,
    gps_partial_source: bool = False,
    acc_partial_source: bool = False,
):
    """
    Describe hourly GPS/ACC missingness using positive
    cross-stream evidence.

    No causal mechanism such as MCAR, MAR, MNAR,
    device failure, non-wear, or intentional disabling
    is inferred here.
    """

    if (
        gps_source_unavailable
        or acc_source_unavailable
        or gps_partial_source
        or acc_partial_source
    ):
        return (
            HourlyGapPattern
            .SOURCE_UNAVAILABLE_OR_PARTIAL
        )

    if gps_observed and acc_observed:
        return (
            HourlyGapPattern.BOTH_OBSERVED
        )

    if (
        not gps_observed
        and acc_observed
    ):
        return (
            HourlyGapPattern
            .GPS_SPECIFIC_GAP
        )

    if (
        gps_observed
        and not acc_observed
    ):
        return (
            HourlyGapPattern
            .ACC_SPECIFIC_GAP
        )

    if screen_event_observed:
        return (
            HourlyGapPattern
            .SHARED_GAP_WITH_SCREEN_ACTIVITY
        )

    return (
        HourlyGapPattern
        .SHARED_GAP_WITHOUT_SCREEN_ACTIVITY_EVIDENCE
    )
