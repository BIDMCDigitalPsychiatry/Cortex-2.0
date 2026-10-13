from pathlib import Path

import pandas as pd


ROOTS = [
    Path("../raphe_outputs/digital_clinic"),
    Path("cortex_longitudinal/data"),
]


KEYWORDS = [
    "os",
    "platform",
    "operating",
    "android",
    "ios",
    "iphone",
    "device",
    "phone",
    "model",
]


def candidate_column(col):
    s = str(col).lower()

    return any(
        k in s
        for k in KEYWORDS
    )


def inspect_file(path):

    try:

        if path.suffix == ".parquet":

            # Read only schema / no large data load.
            import pyarrow.parquet as pq

            schema = (
                pq.ParquetFile(path)
                .schema_arrow
            )

            columns = schema.names

            candidates = [
                c for c in columns
                if candidate_column(c)
            ]

            if not candidates:
                return

            df = pd.read_parquet(
                path,
                columns=candidates,
            )

        elif path.suffix == ".csv":

            preview = pd.read_csv(
                path,
                nrows=5,
            )

            candidates = [
                c for c in preview.columns
                if candidate_column(c)
            ]

            if not candidates:
                return

            # Only load candidate columns.
            df = pd.read_csv(
                path,
                usecols=candidates,
            )

        else:
            return

    except Exception as e:

        print(
            "\nREAD ERROR:",
            path,
        )

        print(
            repr(e)
        )

        return


    print()
    print("=" * 80)
    print(path)
    print("=" * 80)

    for col in candidates:

        print()
        print(
            "COLUMN:",
            col,
        )

        x = (
            df[col]
            .dropna()
        )

        print(
            "Nonmissing:",
            f"{len(x):,}",
        )

        if len(x) == 0:
            continue

        vc = (
            x.astype(str)
            .value_counts()
            .head(30)
        )

        print(
            vc.to_string()
        )


files = []

for root in ROOTS:

    if not root.exists():
        continue

    for path in root.rglob("*"):

        if path.suffix.lower() not in {
            ".csv",
            ".parquet",
        }:
            continue

        # Avoid scanning large derived hourly / feature outputs.
        p = str(path).lower()

        if any(
            token in p
            for token in [
                "hourly_context/episodes",
                "hourly_missingness",
                "source_files/",
                "benchmark_5",
            ]
        ):
            continue

        files.append(path)


print(
    "Candidate metadata files:",
    len(files),
)


for i, path in enumerate(
    sorted(files),
    start=1,
):

    inspect_file(path)


print()
print("=" * 80)
print("DONE")
print("=" * 80)
