from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.mindlamp_manifest import (
    file_manifest_record,
    list_mindlamp_files,
)


def test_device_usage_manifest(
    tmp_path,
):
    root = tmp_path / "passive"
    d = root / "device_usage"
    d.mkdir(parents=True)

    path = (
        d
        / "device_usage_U123_1.csv"
    )

    pd.DataFrame({
        "timestamp": [
            1000,
            2000,
            3000,
        ],
        "duration": [
            900000,
            900000,
            900000,
        ],
    }).to_csv(
        path,
        index=False,
    )

    files = list_mindlamp_files(
        root,
        "device_usage",
    )

    assert files == [
        path.resolve()
    ]

    record = file_manifest_record(
        path,
        stream="device_usage",
    )

    assert (
        record[
            "source_participant_id"
        ]
        == "U123"
    )

    assert (
        record[
            "min_timestamp_ms"
        ]
        == 1000
    )

    assert (
        record[
            "max_timestamp_ms"
        ]
        == 3000
    )


def test_nearby_device_manifest(
    tmp_path,
):
    root = tmp_path / "passive"
    d = root / "nearby_device"
    d.mkdir(parents=True)

    path = (
        d
        / "nearby_device_U999_7.csv"
    )

    pd.DataFrame({
        "timestamp": [
            5000,
            6000,
        ],
        "strength": [
            -80,
            -70,
        ],
    }).to_csv(
        path,
        index=False,
    )

    record = file_manifest_record(
        path,
        stream="nearby_device",
    )

    assert (
        record[
            "source_participant_id"
        ]
        == "U999"
    )

    assert (
        record[
            "n_rows"
        ]
        == 2
    )

    assert (
        record[
            "scan_status"
        ]
        == "ok"
    )
