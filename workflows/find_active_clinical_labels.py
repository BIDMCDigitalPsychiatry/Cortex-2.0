from pathlib import Path
import pandas as pd

ROOTS = [
    Path("/shared/Digital Clinic"),
    Path(
        "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
    ),
]

NAME_KEYS = [
    "ema",
    "survey",
    "questionnaire",
    "assessment",
    "clinical",
    "phq",
    "gad",
    "mood",
]

COLUMN_KEYS = [
    "participant",
    "subject",
    "id",
    "date",
    "timestamp",
    "mood",
    "anxiety",
    "function",
    "functioning",
    "sleep",
    "energy",
    "fatigue",
    "anhed",
    "stress",
    "social",
    "lonely",
    "phq",
    "gad",
    "score",
    "response",
]

seen = set()
found = []

for root in ROOTS:
    if not root.exists():
        continue

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        if p.suffix.lower() not in {
            ".csv",
            ".parquet",
        }:
            continue

        sp = str(p)

        # Avoid huge raw passive directories
        if any(
            x in sp.lower()
            for x in [
                "/acc/",
                "/gps/",
                "/device_usage/",
                "/ambient_light/",
                "/nearby_device/",
                "/messages_usage/",
            ]
        ):
            continue

        key = str(p.resolve())

        if key in seen:
            continue

        seen.add(key)

        # Prioritize likely active-data filenames
        filename_match = any(
            k in p.name.lower()
            for k in NAME_KEYS
        )

        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(
                    p,
                    nrows=3,
                    low_memory=False,
                )
            else:
                df = pd.read_parquet(p).head(3)

        except Exception:
            continue

        useful = [
            c for c in df.columns
            if any(
                k in str(c).lower()
                for k in COLUMN_KEYS
            )
        ]

        clinical = [
            c for c in useful
            if any(
                k in str(c).lower()
                for k in [
                    "mood",
                    "anxiety",
                    "function",
                    "sleep",
                    "energy",
                    "fatigue",
                    "anhed",
                    "stress",
                    "lonely",
                    "phq",
                    "gad",
                    "score",
                    "response",
                ]
            )
        ]

        if filename_match or clinical:
            found.append(
                (
                    p,
                    useful,
                    clinical,
                )
            )


# Rank files with actual clinical-looking columns first
found.sort(
    key=lambda z: (
        -len(z[2]),
        -len(z[1]),
        str(z[0]),
    )
)

print()
print("=" * 110)
print("CANDIDATE ACTIVE / CLINICAL DATA FILES")
print("=" * 110)

for p, useful, clinical in found[:80]:

    print()
    print("FILE:", p)

    if clinical:
        print("CLINICAL COLUMNS:")
        for c in clinical:
            print("   ", c)

    print("ID / DATE / OTHER USEFUL COLUMNS:")
    for c in useful:
        if c not in clinical:
            print("   ", c)

print()
print("=" * 110)
print("Total candidate files:", len(found))
