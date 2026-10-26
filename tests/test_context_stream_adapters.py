import pandas as pd

from cortex2_engine.adapters.mindlamp import (
    MindLAMPAdapter,
)


def make_adapter(tmp_path):

    passive = tmp_path / "passive"
    passive.mkdir()

    return (
        passive,
        MindLAMPAdapter(
            study_id="test_study",
            passive_root=passive,
        ),
    )


def collect(adapter, stream):
    return pd.concat(
        list(
            adapter.iter_canonical(
                stream,
                chunksize=100,
            )
        ),
        ignore_index=True,
    )


def test_screen_preserves_battery_level(
    tmp_path,
):
    passive, adapter = make_adapter(
        tmp_path
    )

    d = passive / "screen"
    d.mkdir()

    pd.DataFrame({
        "timestamp": [
            1_700_000_000_000,
        ],
        "value": [0],
        "battery_level": [0.45],
        "representation": [
            "screen_on",
        ],
    }).to_csv(
        d / "screen_U1_all.csv",
        index=False,
    )

    out = collect(
        adapter,
        "screen",
    )

    assert (
        out[
            "battery_level_fraction"
        ].iloc[0]
        == 0.45
    )


def test_device_usage_mapping(
    tmp_path,
):
    passive, adapter = make_adapter(
        tmp_path
    )

    d = passive / "device_usage"
    d.mkdir()

    pd.DataFrame({
        "timestamp": [
            1_700_000_000_000,
        ],
        "duration": [900_000],
        "totalUnlockDuration": [
            627_000,
        ],
        "totalUnlocks": [4],
        "totalScreenWakes": [5],
        "applicationUsageByCategory": [
            "{'example': []}"
        ],
        "notificationUsageByCategory": [
            "{}"
        ],
        "webUsageByCategory": [
            "{}"
        ],
    }).to_csv(
        d
        / "device_usage_U1_1.csv",
        index=False,
    )

    out = collect(
        adapter,
        "device_usage",
    )

    row = out.iloc[0]

    assert (
        row[
            "interval_duration_ms"
        ]
        == 900_000
    )

    assert (
        row[
            "total_unlock_duration_ms"
        ]
        == 627_000
    )

    assert (
        row["total_unlocks"]
        == 4
    )

    assert (
        row["total_screen_wakes"]
        == 5
    )


def test_nearby_device_is_pseudonymized(
    tmp_path,
):
    passive, adapter = make_adapter(
        tmp_path
    )

    d = passive / "nearby_device"
    d.mkdir()

    raw_address = (
        "b8:f8:53:80:8:64"
    )

    pd.DataFrame({
        "timestamp": [
            1_700_000_000_000,
            1_700_000_300_000,
        ],
        "strength": [
            0,
            0,
        ],
        "address": [
            raw_address,
            raw_address,
        ],
        "type": [
            "wifi",
            "wifi",
        ],
        "name": [
            "ExampleWifi",
            "ExampleWifi",
        ],
    }).to_csv(
        d
        / "nearby_device_U1_1.csv",
        index=False,
    )

    out = collect(
        adapter,
        "nearby_device",
    )

    assert (
        out[
            "nearby_identifier_hash"
        ].nunique()
        == 1
    )

    assert (
        out[
            "signal_strength_raw"
        ].tolist()
        == [0, 0]
    )

    assert (
        out[
            "device_type"
        ].tolist()
        == ["wifi", "wifi"]
    )

    assert (
        "address"
        not in out.columns
    )

    assert (
        "name"
        not in out.columns
    )

    assert (
        raw_address
        not in out.to_string()
    )
