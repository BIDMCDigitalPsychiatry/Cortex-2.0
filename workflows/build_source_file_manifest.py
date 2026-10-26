import argparse
from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.mindlamp_manifest import (
    file_manifest_record,
    list_mindlamp_files,
)


PASSIVE_ROOT = Path("../passive")

OUTPUT_ROOT = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/source_files"
)


def load_existing(path):

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stream",
        required=True,
        choices=[
            "gps",
            "accelerometer",
            "screen",
            "device_usage",
            "nearby_device",
        ],
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        OUTPUT_ROOT
        / f"{args.stream}.parquet"
    )

    existing = load_existing(
        output
    )

    # Records are reusable only if the file has not changed.
    reusable = {}

    if len(existing):

        for row in existing.itertuples(
            index=False
        ):

            reusable[
                str(row.source_file)
            ] = {
                "size_bytes":
                    int(row.size_bytes),

                "modified_ns":
                    int(row.modified_ns),

                "record":
                    row._asdict(),
            }

    files = list_mindlamp_files(
        PASSIVE_ROOT,
        args.stream,
    )

    print(
        f"\nStream: {args.stream}"
    )

    print(
        "Files:",
        f"{len(files):,}",
    )

    records = []

    reused_count = 0
    scanned_count = 0

    for i, path in enumerate(
        files,
        1,
    ):

        path = path.resolve()
        stat = path.stat()

        previous = reusable.get(
            str(path)
        )

        if (
            previous is not None
            and previous["size_bytes"]
            == int(stat.st_size)
            and previous["modified_ns"]
            == int(stat.st_mtime_ns)
        ):

            records.append(
                previous["record"]
            )

            reused_count += 1

        else:

            record = file_manifest_record(
                path,
                stream=args.stream,
                chunksize=args.chunksize,
            )

            records.append(
                record
            )

            scanned_count += 1

        if (
            i % args.checkpoint_every == 0
            or i == len(files)
        ):

            df = pd.DataFrame(
                records
            )

            df.to_parquet(
                output,
                index=False,
            )

            print(
                f"{i:,}/{len(files):,} "
                f"| scanned={scanned_count:,} "
                f"| reused={reused_count:,}",
                flush=True,
            )

    manifest = pd.DataFrame(
        records
    )

    print("\n=== SUMMARY ===")

    print(
        "Rows:",
        f"{len(manifest):,}",
    )

    print(
        "Participants:",
        f"{manifest['source_participant_id'].nunique():,}",
    )

    print(
        "Errors:",
        int(
            (
                manifest["scan_status"]
                != "ok"
            ).sum()
        ),
    )

    valid = manifest[
        manifest["scan_status"].eq(
            "ok"
        )
        &
        manifest[
            "min_timestamp_ms"
        ].notna()
    ]

    if len(valid):

        start = pd.to_datetime(
            valid[
                "min_timestamp_ms"
            ].min(),
            unit="ms",
            utc=True,
        )

        end = pd.to_datetime(
            valid[
                "max_timestamp_ms"
            ].max(),
            unit="ms",
            utc=True,
        )

        print(
            "Earliest timestamp:",
            start,
        )

        print(
            "Latest timestamp:",
            end,
        )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()
