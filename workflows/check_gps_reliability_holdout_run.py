"""Minimal integrity checker for one holdout degradation parquet."""

import os
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


if len(sys.argv) != 2:
    os._exit(2)


path = Path(
    sys.argv[1]
)


try:

    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        os._exit(1)


    parquet_file = pq.ParquetFile(
        path
    )


    if (
        parquet_file.metadata.num_rows
        != 112
    ):
        os._exit(1)


    table = parquet_file.read(
        columns=[
            "participant_id",
        ],
        use_threads=False,
    )


    participant_count = len(
        pc.unique(
            table[
                "participant_id"
            ]
        )
    )


    valid = (
        participant_count == 16
    )


except Exception:

    valid = False


# Avoid Arrow destructor/shutdown issues observed on the cluster.
os._exit(
    0 if valid else 1
)
