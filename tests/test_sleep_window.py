import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.screen_bouts import build_screen_bouts
from cortex2_engine.features.sleep_window import (
    detect_sleep_window,
    night_window_ms,
)

MINUTE = 60_000
HOUR = 3_600_000
TZ = "UTC"
NIGHT = "2026-06-02"  # night of June 1 -> June 2


def at(hour_of_june_1):
    start, _ = night_window_ms(NIGHT, TZ)  # June 1 18:00 UTC
    return start + int((hour_of_june_1 - 18) * HOUR)


def acc_minutes(enmo_by_minute=None, start_h=18, end_h=36):
    t = np.arange(at(start_h), at(end_h), MINUTE, dtype=np.int64)
    enmo = np.full(t.size, 2.0)

    if enmo_by_minute is not None:
        for (a, b), value in enmo_by_minute.items():
            enmo[(t >= at(a)) & (t < at(b))] = value

    return pd.DataFrame({
        "epoch_start_ms": t,
        "epoch_end_ms": t + MINUTE,
        "observed": np.ones(t.size, dtype=bool),
        "enmo_mg": enmo,
    })


def screen(*on_off_hours):
    rows = []

    for on, off in on_off_hours:
        rows += [(at(on), "screen_on"), (at(off), "screen_off")]

    s, e = night_window_ms(NIGHT, TZ)

    return build_screen_bouts(
        pd.DataFrame(rows, columns=["timestamp_ms", "event"]),
        window_start_ms=s,
        window_end_ms=e,
    )


def test_longest_dark_run_is_the_night():

    bouts = screen((18, 23), (31, 33))  # use until 23:00, again 07:00-09:00

    result = detect_sleep_window(bouts, acc_minutes(), night_date=NIGHT, tz=TZ)

    assert result["sleep_status"] == "observed"
    assert result["sleep_onset_local_h"] == pytest.approx(23.0)
    assert result["sleep_offset_local_h"] == pytest.approx(7.0)
    assert result["sleep_duration_ms"] == 8 * HOUR
    assert result["sleep_midpoint_local_h"] == pytest.approx(3.0)
    assert result["n_awakenings"] == 0
    assert result["night_coverage_fraction"] == pytest.approx(1.0)


def test_brief_checks_are_awakenings_not_the_end():

    bouts = screen((18, 23), (26, 26 + 5 / 60), (31, 33))

    result = detect_sleep_window(bouts, acc_minutes(), night_date=NIGHT, tz=TZ)

    assert result["n_awakenings"] == 1
    assert result["sleep_onset_local_h"] == pytest.approx(23.0)
    assert result["sleep_offset_local_h"] == pytest.approx(7.0)


def test_long_interruption_splits_and_longest_part_wins():

    bouts = screen((18, 23), (25, 26), (31, 33))  # 1 h of use at 01:00

    result = detect_sleep_window(bouts, acc_minutes(), night_date=NIGHT, tz=TZ)

    assert result["sleep_onset_local_h"] == pytest.approx(2.0)
    assert result["sleep_offset_local_h"] == pytest.approx(7.0)


def test_accelerometer_movement_trims_edges():

    bouts = screen((18, 23), (31, 33))
    acc = acc_minutes({(23, 24): 80.0, (30.5, 31): 80.0})

    result = detect_sleep_window(bouts, acc, night_date=NIGHT, tz=TZ)

    assert result["sleep_onset_local_h"] == pytest.approx(0.0)
    assert result["sleep_offset_local_h"] == pytest.approx(6.5)


def test_low_coverage_is_insufficient_not_sleep():

    bouts = screen((18, 23))
    acc = acc_minutes(start_h=18, end_h=20)

    result = detect_sleep_window(bouts, acc, night_date=NIGHT, tz=TZ)

    assert result["sleep_status"] == "insufficient_observability"
    assert np.isnan(result["sleep_duration_ms"])


def test_no_data_is_no_raw_data():

    result = detect_sleep_window(
        screen(),
        acc_minutes().iloc[0:0],
        night_date=NIGHT,
        tz=TZ,
    )

    assert result["sleep_status"] == "no_raw_data"


def test_dark_run_touching_window_edge_is_censored():

    bouts = screen((18, 30))  # screen on until 06:00, dark until noon

    result = detect_sleep_window(bouts, acc_minutes(), night_date=NIGHT, tz=TZ)

    assert result["sleep_status"] == "censored"
    assert result["sleep_offset_local_h"] == pytest.approx(12.0)


def test_screen_on_all_night_is_observed_zero():

    bouts = screen((18, 36))

    result = detect_sleep_window(bouts, acc_minutes(), night_date=NIGHT, tz=TZ)

    assert result["sleep_duration_ms"] == 0.0
    assert result["sleep_duration_ms_status"] == "observed_zero"
    assert result["sleep_onset_local_h_status"] == "not_applicable"


def test_device_usage_marks_light():

    bouts = screen((18, 23), (31, 33))
    usage = pd.DataFrame({
        "timestamp_ms": [at(25)],
        "interval_duration_ms": [HOUR],
        "total_unlock_duration_ms": [10 * MINUTE],
    })

    result = detect_sleep_window(
        bouts,
        acc_minutes(),
        night_date=NIGHT,
        tz=TZ,
        device_usage=usage,
    )

    assert result["sleep_onset_local_h"] == pytest.approx(2.0)


def test_screen_events_add_coverage():

    bouts = screen((18, 23), (31, 33))
    acc = acc_minutes(start_h=18, end_h=19)
    events = pd.DataFrame({
        "timestamp_ms": np.arange(at(18), at(36), 5 * MINUTE),
    })

    result = detect_sleep_window(
        bouts,
        acc,
        night_date=NIGHT,
        tz=TZ,
        screen_events=events,
    )

    assert result["night_coverage_fraction"] == pytest.approx(1.0)
    assert result["sleep_status"] == "observed"


def test_night_window_dst_and_validation():

    start, end = night_window_ms("2026-03-08", "America/New_York")

    assert end - start == 17 * HOUR  # 18:00 -> 12:00 minus the lost hour

    with pytest.raises(ValueError):
        detect_sleep_window(screen(), acc_minutes(), night_date=NIGHT, tz=TZ, grid_ms=0)

    with pytest.raises(ValueError):
        detect_sleep_window(
            screen(),
            acc_minutes().drop(columns="enmo_mg"),
            night_date=NIGHT,
            tz=TZ,
        )

    with pytest.raises(ValueError):
        detect_sleep_window(
            screen(),
            acc_minutes(),
            night_date=NIGHT,
            tz=TZ,
            min_coverage_fraction=1.5,
        )
