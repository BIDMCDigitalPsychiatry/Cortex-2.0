from pathlib import Path

import pandas as pd


ROOT = Path("../passive/device_usage")

FILES = sorted(
    ROOT.glob("*.csv")
)

USECOLS = [
    "timestamp",
    "duration",
    "totalUnlockDuration",
    "totalUnlocks",
    "totalScreenWakes",
]


total_rows = 0
unique_timestamp_rows = 0
duplicated_rows = 0

timestamps_with_duplicates = 0
timestamps_with_conflicts = 0

files_with_duplicates = 0
files_with_conflicts = 0

max_multiplicity = 1

examples = []


for i, path in enumerate(
    FILES,
    start=1,
):

    try:
        x = pd.read_csv(
            path,
            usecols=lambda c: c in USECOLS,
        )
    except Exception as e:
        print(
            "READ ERROR:",
            path.name,
            repr(e),
        )
        continue

    if "timestamp" not in x:
        continue

    total_rows += len(x)

    counts = (
        x.groupby(
            "timestamp",
            dropna=False,
        )
        .size()
    )

    unique_timestamp_rows += len(counts)

    dup_ts = counts[
        counts > 1
    ]

    if len(dup_ts) == 0:
        continue

    files_with_duplicates += 1

    timestamps_with_duplicates += len(
        dup_ts
    )

    duplicated_rows += int(
        (dup_ts - 1).sum()
    )

    max_multiplicity = max(
        max_multiplicity,
        int(
            dup_ts.max()
        ),
    )

    file_conflict = False

    z = x[
        x["timestamp"].isin(
            dup_ts.index
        )
    ]

    value_cols = [
        c for c in [
            "duration",
            "totalUnlockDuration",
            "totalUnlocks",
            "totalScreenWakes",
        ]
        if c in z.columns
    ]

    for timestamp, g in z.groupby(
        "timestamp",
        dropna=False,
    ):

        # Number of distinct measurement combinations
        # for the same timestamp.
        distinct = (
            g[
                value_cols
            ]
            .drop_duplicates()
        )

        if len(distinct) > 1:

            timestamps_with_conflicts += 1
            file_conflict = True

            if len(examples) < 20:

                examples.append({
                    "file":
                        path.name,

                    "timestamp":
                        timestamp,

                    "rows":
                        len(g),

                    "distinct_measurements":
                        len(distinct),

                    "values":
                        distinct.to_dict(
                            "records"
                        ),
                })

    if file_conflict:
        files_with_conflicts += 1

    if (
        i % 1000 == 0
        or i == len(FILES)
    ):

        print(
            f"{i:,}/{len(FILES):,}",
            flush=True,
        )


print()
print("=" * 72)
print("DEVICE-USAGE DUPLICATION AUDIT")
print("=" * 72)

print(
    "Files:",
    f"{len(FILES):,}",
)

print(
    "Total rows:",
    f"{total_rows:,}",
)

print(
    "Unique timestamp rows:",
    f"{unique_timestamp_rows:,}",
)

print(
    "Extra duplicated rows:",
    f"{duplicated_rows:,}",
)

print(
    "Files with duplicate timestamps:",
    f"{files_with_duplicates:,}",
)

print(
    "Duplicate timestamps:",
    f"{timestamps_with_duplicates:,}",
)

print(
    "Maximum timestamp multiplicity:",
    f"{max_multiplicity:,}",
)

print()
print(
    "Files with conflicting duplicate timestamps:",
    f"{files_with_conflicts:,}",
)

print(
    "Timestamps with conflicting measurements:",
    f"{timestamps_with_conflicts:,}",
)


print()
print("=" * 72)
print("CONFLICT EXAMPLES")
print("=" * 72)

if not examples:
    print(
        "No conflicting duplicates found."
    )

else:
    for e in examples:
        print()
        print(e)
