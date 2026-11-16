from pathlib import Path
import pandas as pd

ROOTS = [
    Path(
        "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
        "feature_reliability/gps_clinical/home_time"
    ),
    Path(
        "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
    ),
]

KEYWORDS = [
    "phq",
    "gad",
    "ema",
    "survey",
    "clinical",
    "questionnaire",
    "assessment",
]

IMPORTANT_COLS = [
    "participant",
    "subject",
    "id",
    "date",
    "time",
    "timestamp",
    "phq",
    "gad",
    "mood",
    "anxiety",
    "function",
    "sleep",
    "energy",
    "fatigue",
    "anhed",
    "home",
    "supported",
    "coverage",
]

seen = set()
candidates = []

for root in ROOTS:
    if not root.exists():
        continue

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {".csv", ".parquet"}:
            continue

        key = str(p.resolve())

        if key in seen:
            continue

        seen.add(key)

        name = p.name.lower()

        if any(k in name for k in KEYWORDS):
            candidates.append(p)


# Always include core home-time files if present
for p in [
    ROOTS[0] / "daily_home_time.csv",
    ROOTS[0] / "weekly_home_time.csv",
]:
    if p.exists() and str(p.resolve()) not in seen:
        candidates.insert(0, p)
        seen.add(str(p.resolve()))


print()
print("=" * 100)
print("HOME-TIME / CLINICAL INPUT AUDIT")
print("=" * 100)

for p in candidates[:60]:

    try:
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(
                p,
                nrows=5,
                low_memory=False,
            )
        else:
            df = pd.read_parquet(p).head(5)

    except Exception as e:
        print()
        print("FILE:", p)
        print("READ ERROR:", type(e).__name__)
        continue

    cols = list(df.columns)

    useful = [
        c for c in cols
        if any(
            token in c.lower()
            for token in IMPORTANT_COLS
        )
    ]

    if not useful:
        continue

    print()
    print("FILE:", p)
    print("ROWS SHOWN:", len(df))
    print("USEFUL COLUMNS:")
    for c in useful:
        print("  -", c)

print()
print("=" * 100)
