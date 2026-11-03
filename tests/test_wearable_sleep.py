import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.wearable_sleep import compute_wearable_sleep

HOUR = 3_600_000


def segments(*rows):
    return pd.DataFrame(
        rows,
        columns=["timestamp_ms", "representation", "duration_ms"],
    )


def test_sleep_and_bed_time_attributed_by_end():

    sleep = segments(
        (0, "in_bed", 8 * HOUR),
        (HOUR, "in_sleep", 3 * HOUR),
        (4 * HOUR, "in_awake", HOUR // 2),
        (4 * HOUR + HOUR // 2, "in_sleep", 3 * HOUR),
        (2 * HOUR, "in_sleep", HOUR),  # overlaps the first sleep segment
        (30 * HOUR, "in_sleep", HOUR),  # ends after the night
    )

    result = compute_wearable_sleep(sleep, night_start_ms=0, night_end_ms=12 * HOUR)

    assert result["healthkit_sleep_duration_ms"] == pytest.approx(6 * HOUR)
    assert result["healthkit_in_bed_ms"] == pytest.approx(8 * HOUR)
    assert result["healthkit_awake_ms"] == pytest.approx(HOUR / 2)
    assert result["healthkit_n_segments"] == 5
    assert result["healthkit_sleep_duration_ms_status"] == "observed"


def test_unreported_category_is_not_applicable():

    result = compute_wearable_sleep(
        segments((0, "in_sleep", HOUR)),
        night_start_ms=0,
        night_end_ms=12 * HOUR,
    )

    assert np.isnan(result["healthkit_in_bed_ms"])
    assert result["healthkit_in_bed_ms_status"] == "not_applicable"


def test_no_segments_and_source_absent():

    empty = compute_wearable_sleep(
        segments(),
        night_start_ms=0,
        night_end_ms=HOUR,
    )

    assert empty["healthkit_sleep_duration_ms_status"] == "no_raw_data"

    absent = compute_wearable_sleep(
        segments(),
        night_start_ms=0,
        night_end_ms=HOUR,
        source_available=False,
    )

    assert absent["healthkit_sleep_duration_ms_status"] == "source_absent"


def test_validation():

    with pytest.raises(ValueError):
        compute_wearable_sleep(segments(), night_start_ms=1, night_end_ms=1)

    with pytest.raises(ValueError):
        compute_wearable_sleep(
            pd.DataFrame({"timestamp_ms": [0]}),
            night_start_ms=0,
            night_end_ms=1,
        )
