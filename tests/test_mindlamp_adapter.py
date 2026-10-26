from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.mindlamp import (
    MindLAMPAdapter,
)


def test_mindlamp_gps_adapter(tmp_path):

    gps_dir = (
        tmp_path
        / "gps"
    )

    gps_dir.mkdir()

    raw = pd.DataFrame({
        "timestamp": [
            1700000000000,
            1700000001000,
        ],
        "longitude": [
            -71.1,
            -71.2,
        ],
        "altitude": [
            10.0,
            11.0,
        ],
        "latitude": [
            42.3,
            42.4,
        ],
        "accuracy": [
            5.0,
            6.0,
        ],
    })

    raw.to_csv(
        gps_dir
        / "gps_U1234567890_1.csv"
    )

    mapping = pd.DataFrame({
        "mindlamp_uid": [
            "U1234567890"
        ],
        "participant_id": [
            "P001"
        ],
    })

    map_path = (
        tmp_path
        / "map.csv"
    )

    mapping.to_csv(
        map_path,
        index=False,
    )

    adapter = MindLAMPAdapter(
        study_id="test",
        passive_root=tmp_path,
        participant_map=map_path,
    )

    chunks = list(
        adapter.iter_canonical(
            "gps",
            source_participant_id=(
                "U1234567890"
            ),
        )
    )

    assert len(chunks) == 1

    df = chunks[0]

    assert (
        df["participant_id"].iloc[0]
        == "P001"
    )

    assert (
        df["source_participant_id"].iloc[0]
        == "U1234567890"
    )

    assert (
        df["latitude"].iloc[0]
        == 42.3
    )

    assert (
        df["longitude"].iloc[0]
        == -71.1
    )


def test_mindlamp_screen_event_mapping(tmp_path):

    screen_dir = (
        tmp_path
        / "screen"
    )
    screen_dir.mkdir()

    raw = pd.DataFrame({
        "timestamp": [
            1700000000000,
            1700000001000,
            1700000002000,
            1700000003000,
        ],
        "value": [
            0, 1, 2, 3
        ],
        "representation": [
            "screen_on",
            "screen_off",
            "locked",
            "unlocked",
        ],
    })

    raw.to_csv(
        screen_dir
        / "screen_U1234567890_all.csv"
    )

    adapter = MindLAMPAdapter(
        study_id="test",
        passive_root=tmp_path,
    )

    df = next(
        adapter.iter_canonical(
            "screen",
            source_participant_id="U1234567890",
        )
    )

    assert df["event"].tolist() == [
        "screen_on",
        "screen_off",
        "lock",
        "unlock",
    ]

    assert df[
        "representation_original"
    ].tolist() == [
        "screen_on",
        "screen_off",
        "locked",
        "unlocked",
    ]


def test_ambiguous_source_id_is_not_silently_mapped(
    tmp_path,
):

    gps_dir = tmp_path / "gps"
    gps_dir.mkdir()

    pd.DataFrame({
        "timestamp": [
            1700000000000,
        ],
        "latitude": [
            42.3,
        ],
        "longitude": [
            -71.1,
        ],
        "altitude": [
            10.0,
        ],
        "accuracy": [
            5.0,
        ],
    }).to_csv(
        gps_dir
        / "gps_U123_1.csv"
    )

    mapping = pd.DataFrame({
        "mindlamp_uid": [
            "U123",
            "U123",
        ],
        "participant_id": [
            "P001",
            "P002",
        ],
    })

    map_path = (
        tmp_path
        / "map.csv"
    )

    mapping.to_csv(
        map_path,
        index=False,
    )

    adapter = MindLAMPAdapter(
        study_id="test",
        passive_root=tmp_path,
        participant_map=map_path,
    )

    df = next(
        adapter.iter_canonical(
            "gps",
            source_participant_id="U123",
        )
    )

    assert (
        df[
            "source_participant_id"
        ].iloc[0]
        == "U123"
    )

    assert df[
        "participant_id"
    ].isna().all()

    assert (
        "U123"
        in adapter.ambiguous_source_ids
    )
