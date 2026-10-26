import pandas as pd

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter
from cortex2_engine.canonical.storage import (
    MaterializedCanonicalStore,
    VirtualCanonicalStore,
)


def make_gps_source(tmp_path):

    gps = tmp_path / "gps"
    gps.mkdir()

    pd.DataFrame({
        "timestamp": [
            1700000000000,
            1700000001000,
        ],
        "latitude": [
            42.3,
            42.4,
        ],
        "longitude": [
            -71.1,
            -71.2,
        ],
        "altitude": [
            10.0,
            11.0,
        ],
        "accuracy": [
            5.0,
            6.0,
        ],
    }).to_csv(
        gps / "gps_U123_1.csv"
    )


def test_virtual_store(tmp_path):

    make_gps_source(
        tmp_path
    )

    adapter = MindLAMPAdapter(
        study_id="test",
        passive_root=tmp_path,
    )

    store = VirtualCanonicalStore(
        adapter
    )

    chunk = next(
        store.iter_stream(
            "gps",
            source_participant_id="U123",
        )
    )

    assert len(chunk) == 2
    assert (
        chunk["participant_id"].iloc[0]
        == "U123"
    )


def test_materialized_store(tmp_path):

    source = (
        tmp_path
        / "source"
    )

    source.mkdir()

    make_gps_source(
        source
    )

    adapter = MindLAMPAdapter(
        study_id="test",
        passive_root=source,
    )

    output = (
        tmp_path
        / "canonical"
    )

    store = MaterializedCanonicalStore(
        output
    )

    manifest = store.materialize(
        adapter,
        streams=["gps"],
        chunksize=100,
        checksum=True,
    )

    assert len(manifest) == 1
    assert manifest["rows"].sum() == 2

    chunks = list(
        store.iter_stream(
            "gps",
            participant_id="U123",
        )
    )

    assert len(chunks) == 1
    assert len(chunks[0]) == 2

    assert (
        chunks[0]["latitude"].tolist()
        == [42.3, 42.4]
    )
