import numpy as np
import pandas as pd
import pytest
from gps_synthetic import HOUR_MS, local_ms

from cortex2_engine.features.communication import (
    COMMUNICATION_KEYS,
    compute_communication_features,
    compute_daily_communication,
    compute_weekly_communication,
    shannon_entropy_nats,
)

TZ = "America/New_York"


def _rows(date, events):
    """events: (hour, channel, direction, duration_s, contact)."""
    return pd.DataFrame([
        {
            "timestamp_ms": local_ms(date, h, TZ),
            "channel": ch,
            "direction": d,
            "duration_ms": None if s is None else s * 1000.0,
            "contact_hash": c,
        }
        for h, ch, d, s, c in events
    ])


DAY = [
    (9, "call", "outgoing", 120, "a"),
    (10, "call", "incoming", 60, "a"),
    (11, "call", "missed", 0, "b"),
    (12, "sms", "outgoing", None, "c"),
    (13, "sms", "incoming", None, "c"),
    (14, "sms", "outgoing", None, "c"),
    (15, "sms", "unknown", None, "d"),
    (16, "sms", "incoming", None, None),
]


def test_daily_counts_contacts_and_ratios():
    out = compute_daily_communication(_rows("2026-04-01", DAY), date="2026-04-01", tz=TZ)

    assert out["call_count"] == 3
    assert out["call_count_missed"] == 1
    assert out["call_duration_ms"] == 180_000
    assert out["call_duration_ms_outgoing"] == 120_000
    assert out["sms_count"] == 4
    assert out["sms_count_incoming"] == 2
    assert out["telephony_n_unknown"] == 1
    assert out["n_unique_contacts"] == 3
    # a: in+out, b: missed only, c: in+out
    assert out["communication_reciprocity"] == pytest.approx(2 / 3)
    assert out["communication_entropy_nats"] == pytest.approx(
        shannon_entropy_nats(np.array([2, 1, 3]))
    )
    # outgoing 3, incoming 3 (missed and unknown excluded)
    assert out["outgoing_fraction"] == pytest.approx(0.5)
    assert out["sms_count_outgoing_status"] == "observed"
    assert all(f"{k}_status" in out for k in COMMUNICATION_KEYS)


def test_zero_counts_need_evidence_that_the_logger_ran():
    empty = _rows("2026-04-01", DAY).iloc[0:0]

    unknown = compute_daily_communication(empty, date="2026-04-01", tz=TZ)
    assert unknown["call_count_status"] == "no_raw_data"
    assert np.isnan(unknown["call_count"])

    quiet = compute_daily_communication(
        empty, date="2026-04-01", tz=TZ,
        stream_active_in_episode=True, source_observed_ms=16 * HOUR_MS,
    )
    assert quiet["call_count"] == 0.0
    assert quiet["call_count_status"] == "observed_zero"
    assert quiet["outgoing_fraction_status"] == "not_applicable"

    thin = compute_daily_communication(
        empty, date="2026-04-01", tz=TZ,
        stream_active_in_episode=True, source_observed_ms=2 * HOUR_MS,
    )
    assert thin["call_count_status"] == "no_raw_data"

    only_unknown = compute_daily_communication(
        _rows("2026-04-01", [(9, "sms", "unknown", None, "x")]), date="2026-04-01", tz=TZ
    )
    assert only_unknown["sms_count_status"] == "insufficient_observability"


def test_ios_is_source_absent():
    out = compute_daily_communication(
        _rows("2026-04-01", DAY), date="2026-04-01", tz=TZ, device_os="iOS"
    )
    assert out["call_count_status"] == "source_absent"
    assert np.isnan(out["outgoing_fraction"])


def test_single_contact_entropy_is_observed_zero_and_no_hash_is_not_applicable():
    one = compute_daily_communication(
        _rows("2026-04-01", [(9, "call", "outgoing", 10, "a"), (10, "call", "outgoing", 10, "a")]),
        date="2026-04-01", tz=TZ,
    )
    assert one["communication_entropy_nats"] == 0.0
    assert one["communication_entropy_nats_status"] == "observed_zero"
    assert one["communication_reciprocity_status"] == "observed_zero"
    assert one["outgoing_fraction"] == 1.0

    rows = _rows("2026-04-01", [(9, "sms", "incoming", None, None)]).drop(
        columns=["contact_hash", "duration_ms"]
    )
    anon = compute_daily_communication(rows, date="2026-04-01", tz=TZ)
    assert anon["n_unique_contacts_status"] == "observed_zero"
    assert anon["communication_reciprocity_status"] == "not_applicable"
    assert anon["outgoing_fraction"] == 0.0


def test_weekly_window_and_dst_day():
    frames = [_rows(f"2026-03-{d:02d}", DAY[:2]) for d in range(2, 10)]
    tel = pd.concat(frames, ignore_index=True)

    week = compute_weekly_communication(tel, date="2026-03-09", tz=TZ)
    assert week["call_count_7d"] == 14  # 03-03 .. 03-09
    assert week["call_count_7d_status"] == "observed"

    # The 23 h spring-forward day still holds its two calls.
    day = compute_daily_communication(tel, date="2026-03-08", tz=TZ)
    assert day["call_count"] == 2

    with pytest.raises(ValueError):
        compute_weekly_communication(tel, date="2026-03-09", tz=TZ, window_days=0)


def test_input_validation():
    with pytest.raises(ValueError, match="Missing required telephony columns"):
        compute_communication_features(pd.DataFrame({"timestamp_ms": []}), window_start_ms=0, window_end_ms=1)
    with pytest.raises(ValueError):
        compute_communication_features(_rows("2026-04-01", DAY), window_start_ms=5, window_end_ms=5)
    assert np.isnan(shannon_entropy_nats(np.array([])))
