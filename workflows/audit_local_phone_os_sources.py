from pathlib import Path

import pandas as pd


ROOT = Path("/shared/Digital Clinic")


# ============================================================
# 1. INSPECT SMALL PARTICIPANT / STUDY FILES
# ============================================================

FILES = [
    ROOT / "completed_participants.csv",
    ROOT / "study_window_diagnostic.csv",
    ROOT / "cortex_validation_5_participants.csv",
]


print("=" * 80)
print("PARTICIPANT / STUDY FILE SCHEMAS")
print("=" * 80)

for path in FILES:

    if not path.exists():
        continue

    try:
        df = pd.read_csv(
            path,
            nrows=5,
        )
    except Exception as e:
        print(
            "\nERROR:",
            path,
            repr(e),
        )
        continue

    print()
    print("-" * 80)
    print(path)
    print("Columns:")
    print(
        list(df.columns)
    )

    print()
    print("First rows:")
    print(
        df.head(3).to_string(
            index=False
        )
    )


# ============================================================
# 2. INSPECT ALIGNED DATA SCHEMAS ONLY
# ============================================================

print()
print("=" * 80)
print("ALIGNED DATA SCHEMAS")
print("=" * 80)

for dirname in [
    ROOT / "aligned_df",
    ROOT / "aligned_df_v1",
]:

    if not dirname.exists():
        continue

    candidates = sorted(
        list(
            dirname.glob("*.csv")
        )
        + list(
            dirname.glob("*.parquet")
        )
    )

    print()
    print(dirname)
    print(
        "Files:",
        len(candidates),
    )

    for path in candidates[:5]:

        try:

            if path.suffix == ".csv":
                df = pd.read_csv(
                    path,
                    nrows=3,
                )

            else:
                df = pd.read_parquet(
                    path
                ).head(3)

        except Exception:
            continue

        print()
        print(
            "FILE:",
            path.name,
        )

        print(
            "Columns:",
            list(df.columns),
        )


# ============================================================
# 3. DEVICE_USAGE VS PHONE_USAGE STREAM OVERLAP
# ============================================================

print()
print("=" * 80)
print("DEVICE_USAGE VS PHONE_USAGE PARTICIPANT OVERLAP")
print("=" * 80)


def get_uids(
    directory,
    prefix,
):

    uids = set()

    for path in directory.glob(
        f"{prefix}_U*.csv"
    ):

        name = path.stem

        parts = name.split("_")

        for part in parts:

            if (
                part.startswith("U")
                and part[1:].isdigit()
            ):
                uids.add(
                    part
                )
                break

    return uids


device_dir = (
    ROOT
    / "passive/device_usage"
)

phone_dir = (
    ROOT
    / "passive/phone_usage"
)


device_uids = get_uids(
    device_dir,
    "device_usage",
)

phone_uids = get_uids(
    phone_dir,
    "phone_usage",
)


print(
    "device_usage participants:",
    len(device_uids),
)

print(
    "phone_usage participants:",
    len(phone_uids),
)

print(
    "both:",
    len(
        device_uids
        & phone_uids
    ),
)

print(
    "device_usage only:",
    len(
        device_uids
        - phone_uids
    ),
)

print(
    "phone_usage only:",
    len(
        phone_uids
        - device_uids
    ),
)


print()
print(
    "Example overlap UIDs:",
    sorted(
        device_uids
        & phone_uids
    )[:20],
)


# ============================================================
# 4. SHOW ONE RAW HEADER FROM EACH STREAM
# ============================================================

print()
print("=" * 80)
print("STREAM SCHEMAS")
print("=" * 80)

for directory, prefix in [
    (
        device_dir,
        "device_usage",
    ),
    (
        phone_dir,
        "phone_usage",
    ),
]:

    files = sorted(
        directory.glob(
            f"{prefix}_U*.csv"
        )
    )

    if not files:
        continue

    path = files[0]

    try:
        df = pd.read_csv(
            path,
            nrows=3,
        )
    except Exception:
        continue

    print()
    print(
        prefix
    )

    print(
        "Example file:",
        path.name,
    )

    print(
        "Columns:",
        list(df.columns),
    )

    print(
        df.head(2).to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
