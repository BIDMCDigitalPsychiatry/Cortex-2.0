from cortex2_engine.missingness.hourly import (
    HourlyGapPattern,
    classify_hourly_gap_pattern,
)


def test_both_observed():
    x = classify_hourly_gap_pattern(
        gps_observed=True,
        acc_observed=True,
        screen_event_observed=False,
        gps_source_unavailable=False,
        acc_source_unavailable=False,
    )

    assert x == (
        HourlyGapPattern.BOTH_OBSERVED
    )


def test_gps_specific_gap():
    x = classify_hourly_gap_pattern(
        gps_observed=False,
        acc_observed=True,
        screen_event_observed=True,
        gps_source_unavailable=False,
        acc_source_unavailable=False,
    )

    assert x == (
        HourlyGapPattern.GPS_SPECIFIC_GAP
    )


def test_acc_specific_gap():
    x = classify_hourly_gap_pattern(
        gps_observed=True,
        acc_observed=False,
        screen_event_observed=False,
        gps_source_unavailable=False,
        acc_source_unavailable=False,
    )

    assert x == (
        HourlyGapPattern.ACC_SPECIFIC_GAP
    )


def test_shared_gap_with_screen_activity():
    x = classify_hourly_gap_pattern(
        gps_observed=False,
        acc_observed=False,
        screen_event_observed=True,
        gps_source_unavailable=False,
        acc_source_unavailable=False,
    )

    assert x == (
        HourlyGapPattern
        .SHARED_GAP_WITH_SCREEN_ACTIVITY
    )


def test_shared_gap_without_screen_evidence():
    x = classify_hourly_gap_pattern(
        gps_observed=False,
        acc_observed=False,
        screen_event_observed=False,
        gps_source_unavailable=False,
        acc_source_unavailable=False,
    )

    assert x == (
        HourlyGapPattern
        .SHARED_GAP_WITHOUT_SCREEN_ACTIVITY_EVIDENCE
    )


def test_source_unavailable_has_priority():
    x = classify_hourly_gap_pattern(
        gps_observed=False,
        acc_observed=True,
        screen_event_observed=True,
        gps_source_unavailable=True,
        acc_source_unavailable=False,
    )

    assert x == (
        HourlyGapPattern
        .SOURCE_UNAVAILABLE_OR_PARTIAL
    )
