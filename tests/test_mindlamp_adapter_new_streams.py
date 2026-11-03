import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter
from cortex2_engine.canonical.validation import validate_canonical

FIXTURES = Path(__file__).parent / "fixtures" / "mindlamp"

SECOND = 1000


@pytest.fixture
def adapter(tmp_path):

    passive = tmp_path / "passive"
    shutil.copytree(FIXTURES, passive)

    return MindLAMPAdapter(
        study_id="test_study",
        passive_root=passive,
    )


def collect(adapter, stream, **kwargs):
    return pd.concat(
        list(
            adapter.iter_canonical(
                stream,
                chunksize=100,
                **kwargs,
            )
        ),
        ignore_index=True,
    )


def test_new_streams_are_discovered(adapter):

    assert {
        "telephony",
        "survey",
        "cognitive",
        "steps",
        "sleep",
    } <= set(adapter.available_streams())


def test_telephony_channel_direction_and_duration(adapter):

    out = collect(adapter, "telephony")

    assert out["channel"].tolist() == [
        "call",
        "call",
        "call",
        "sms",
        "sms",
        "call",
    ]

    assert out["direction"].tolist() == [
        "incoming",
        "outgoing",
        "missed",
        "incoming",
        "outgoing",
        "unknown",
    ]

    # mindLAMP call_duration is seconds; canonical is milliseconds.
    assert out["duration_ms"].iloc[0] == pytest.approx(60 * SECOND)
    assert out["duration_ms"].iloc[1] == pytest.approx(120 * SECOND)
    assert pd.isna(out["duration_ms"].iloc[3])


def test_telephony_contacts_are_hashed_not_propagated(adapter):

    out = collect(adapter, "telephony")

    hashes = out["contact_hash"]

    # Formatting differences and channel do not split one counterpart.
    assert hashes.iloc[0] == hashes.iloc[1] == hashes.iloc[3]
    assert hashes.iloc[0] != hashes.iloc[2]
    assert not pd.isna(hashes.iloc[4])
    assert pd.isna(hashes.iloc[5])

    as_text = out.astype(str).to_string()
    assert "555" not in as_text
    assert "opaque-trace" not in as_text
    assert hashes.dropna().str.fullmatch(r"[0-9a-f]{32}").all()


def test_telephony_hash_is_study_scoped(tmp_path):

    passive = tmp_path / "passive"
    shutil.copytree(FIXTURES, passive)

    a = MindLAMPAdapter(study_id="A", passive_root=passive)
    b = MindLAMPAdapter(study_id="B", passive_root=passive)

    assert (
        collect(a, "telephony")["contact_hash"].iloc[0]
        != collect(b, "telephony")["contact_hash"].iloc[0]
    )


def test_nearby_device_hash_is_unchanged_by_refactor(adapter):

    out = collect(adapter, "nearby_device")

    expected = hashlib.blake2b(
        b"test_study|mindlamp|wifi|aa:bb:cc:dd:ee:ff",
        digest_size=16,
    ).hexdigest()

    assert out["nearby_identifier_hash"].iloc[0] == expected
    assert pd.isna(out["nearby_identifier_hash"].iloc[1])


def test_survey_items(adapter):

    out = collect(adapter, "survey")

    assert out["instrument"].unique().tolist() == ["PHQ-9"]
    assert out["value_numeric"].iloc[0] == 2
    assert pd.isna(out["value_numeric"].iloc[2])
    assert out["value_text"].iloc[2] == "felt ok"
    assert out["response_latency_ms"].iloc[1] == 2100
    assert out["activity_id"].iloc[0] == "act_phq9"


def test_cognitive_trials(adapter):

    out = collect(adapter, "cognitive")

    assert out["task"].tolist() == [
        "jewels_a",
        "jewels_a",
        "jewels_a",
        "trails_b",
    ]

    assert out["session_id"].nunique() == 2
    assert out["correct"].tolist() == [True, True, False, True]
    assert out["response_time_ms"].iloc[0] == 812


def test_steps(adapter):

    out = collect(adapter, "steps")

    assert out["step_count"].tolist() == [120.0, 0.0, 87.0]
    assert out["interval_duration_ms"].iloc[0] == 60_000


def test_sleep_representation_mapping(adapter):

    out = collect(adapter, "sleep")

    assert out["representation"].tolist() == [
        "in_bed",
        "in_sleep",
        "in_awake",
        "in_sleep",
    ]

    assert out["representation_original"].iloc[2] == "2"
    assert out["duration_ms"].iloc[0] == 8 * 3_600_000


def test_time_window_is_half_open(adapter):

    out = collect(
        adapter,
        "telephony",
        start_ms=1_772_885_400_000,
        end_ms=1_772_886_600_000,
    )

    assert out["timestamp_ms"].tolist() == [
        1_772_885_400_000,
        1_772_886_000_000,
    ]


def test_missing_required_source_column_raises(tmp_path):

    passive = tmp_path / "passive"
    (passive / "survey").mkdir(parents=True)

    pd.DataFrame({
        "timestamp": [1_772_884_800_000],
        "question": ["q1"],
        "value": [1],
    }).to_csv(
        passive / "survey" / "survey_U9_all.csv",
        index=False,
    )

    adapter = MindLAMPAdapter(
        study_id="s",
        passive_root=passive,
    )

    with pytest.raises(ValueError, match="instrument"):
        collect(adapter, "survey")


def test_unsupported_stream_raises(adapter):

    with pytest.raises(ValueError):
        list(adapter.iter_canonical("battery"))


def test_allowed_values_are_validated():

    df = pd.DataFrame({
        "study_id": ["s"],
        "timestamp_ms": [1_772_884_800_000],
        "timestamp_utc": pd.to_datetime(
            [1_772_884_800_000],
            unit="ms",
            utc=True,
        ),
        "channel": ["fax"],
        "direction": ["incoming"],
        "source_platform": ["mindlamp"],
        "source_stream": ["telephony"],
        "source_participant_id": ["U1"],
    })

    report = validate_canonical(df, "telephony")

    assert not report.valid
    assert "channel" in report.errors[0]
