from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex_longitudinal/data/"
    "benchmark_5_full"
)

FEATURE_ROOT = ROOT / "features"
DAILY_ROOT = ROOT / "participant_daily"


print("=" * 80)
print("CORTEX FEATURE INVENTORY")
print("=" * 80)


# ============================================================
# PARTICIPANT FEATURE DIRECTORIES
# ============================================================

participant_dirs = sorted(
    [
        p
        for p in FEATURE_ROOT.iterdir()
        if p.is_dir()
    ]
)

print(
    "Participant feature directories:",
    len(participant_dirs),
)


# ============================================================
# DAILY FEATURE TYPES
# ============================================================

feature_counts = Counter()
participant_features = defaultdict(list)
sample_files = {}


for participant_dir in participant_dirs:

    daily_parts = (
        participant_dir
        / "_daily_parts"
    )

    if not daily_parts.exists():
        continue

    for feature_dir in sorted(
        daily_parts.iterdir()
    ):

        if not feature_dir.is_dir():
            continue

        files = sorted(
            feature_dir.glob("*.csv")
        )

        if not files:
            continue

        feature = feature_dir.name

        feature_counts[
            feature
        ] += len(files)

        participant_features[
            participant_dir.name
        ].append(
            feature
        )

        sample_files.setdefault(
            feature,
            files[0],
        )


print()
print("=" * 80)
print("DAILY FEATURE TYPES")
print("=" * 80)

for feature, n in sorted(
    feature_counts.items()
):

    n_participants = sum(
        feature
        in features
        for features
        in participant_features.values()
    )

    print(
        f"{feature:30s} "
        f"participants={n_participants:2d} "
        f"daily_files={n:4d}"
    )


# ============================================================
# PARTICIPANT COVERAGE
# ============================================================

print()
print("=" * 80)
print("FEATURES BY PARTICIPANT")
print("=" * 80)

for participant, features in sorted(
    participant_features.items()
):

    print()
    print(participant)

    print(
        ", ".join(
            sorted(features)
        )
    )


# ============================================================
# SAMPLE SCHEMAS
# ============================================================

print()
print("=" * 80)
print("SAMPLE DAILY FEATURE SCHEMAS")
print("=" * 80)

for feature, path in sorted(
    sample_files.items()
):

    try:

        df = pd.read_csv(
            path,
            nrows=3,
        )

    except Exception as e:

        print()
        print(
            feature,
            "ERROR:",
            repr(e),
        )

        continue

    print()
    print("-" * 80)

    print(
        "Feature:",
        feature
    )

    print(
        "File:",
        path
    )

    print(
        "Columns:",
        list(df.columns)
    )

    print(
        df.head(2).to_string(
            index=False
        )
    )


# ============================================================
# FINAL PARTICIPANT-DAILY TABLES
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT_DAILY TABLES")
print("=" * 80)

daily_files = sorted(
    DAILY_ROOT.glob("*.csv")
)

print(
    "Files:",
    len(daily_files),
)

for path in daily_files:

    try:

        df = pd.read_csv(
            path,
            nrows=3,
        )

    except Exception as e:

        print(
            path.name,
            "ERROR:",
            repr(e),
        )

        continue

    print()
    print(
        path.name
    )

    print(
        "Columns:",
        list(df.columns)
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
