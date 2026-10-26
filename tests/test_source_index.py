from pathlib import Path

import pandas as pd
import pytest

from cortex2_engine.io.source_index import (
    SourceFileIndex,
)


def write_manifest(
    root,
    rows,
):
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_parquet(
        root / "gps.parquet",
        index=False,
    )


def test_source_index_selects_overlap(
    tmp_path,
):

    files = []

    for i in range(3):
        path = (
            tmp_path
            / f"file{i}.csv"
        )

        path.write_text("x\n")

        stat = path.stat()

        files.append({
            "stream":
                "gps",

            "source_participant_id":
                "U1",

            "source_file":
                str(path),

            "size_bytes":
                stat.st_size,

            "modified_ns":
                stat.st_mtime_ns,

            "min_timestamp_ms":
                i * 100,

            "max_timestamp_ms":
                i * 100 + 99,

            "scan_status":
                "ok",
        })

    manifest = (
        tmp_path
        / "manifest"
    )

    write_manifest(
        manifest,
        files,
    )

    index = SourceFileIndex(
        manifest
    )

    selected = index.candidate_files(
        "gps",
        source_participant_id="U1",
        start_ms=100,
        end_ms=200,
    )

    assert selected == [
        Path(
            files[1][
                "source_file"
            ]
        )
    ]


def test_source_index_detects_stale_file(
    tmp_path,
):

    path = tmp_path / "gps.csv"
    path.write_text("original\n")

    stat = path.stat()

    manifest = (
        tmp_path
        / "manifest"
    )

    write_manifest(
        manifest,
        [{
            "stream":
                "gps",

            "source_participant_id":
                "U1",

            "source_file":
                str(path),

            "size_bytes":
                stat.st_size,

            "modified_ns":
                stat.st_mtime_ns,

            "min_timestamp_ms":
                0,

            "max_timestamp_ms":
                100,

            "scan_status":
                "ok",
        }],
    )

    # Alter source after manifest creation.
    path.write_text(
        "source changed\n"
    )

    index = SourceFileIndex(
        manifest
    )

    with pytest.raises(
        RuntimeError,
        match="changed after manifest",
    ):
        index.candidate_files(
            "gps",
            source_participant_id="U1",
        )
