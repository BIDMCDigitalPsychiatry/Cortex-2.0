import pandas as pd

from cortex2_engine.adapters.mindlamp_manifest import (
    file_manifest_record,
)


def test_mindlamp_file_manifest(tmp_path):

    gps = tmp_path / "gps"
    gps.mkdir()

    path = (
        gps
        / "gps_U123_1.csv"
    )

    pd.DataFrame({
        "timestamp": [
            1700000003000,
            1700000001000,
            1700000002000,
        ],
        "latitude": [
            42.1,
            42.2,
            42.3,
        ],
    }).to_csv(
        path,
        index=False,
    )

    r = file_manifest_record(
        path,
        stream="gps",
        chunksize=2,
    )

    assert (
        r["source_participant_id"]
        == "U123"
    )

    assert r["n_rows"] == 3

    assert (
        r["n_valid_timestamps"]
        == 3
    )

    assert (
        r["min_timestamp_ms"]
        == 1700000001000
    )

    assert (
        r["max_timestamp_ms"]
        == 1700000003000
    )

    assert (
        r["scan_status"]
        == "ok"
    )
