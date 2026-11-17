from pathlib import Path
import pandas as pd

ROOTS = [
    Path("/shared/Digital Clinic"),
    Path("/shared/Digital Clinic/cortex2_outputs/digital_clinic"),
]

EXCLUDE_PARTS = {
    "checkpoints",
    "participants",
    "pre_checkpoint_backups",
    "__pycache__",
}

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
    "ema",
]

seen_schema = set()
results = []

for root in ROOTS:

    if not root.exists():
        continue

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        if p.suffix.lower() not in {".csv", ".parquet"}:
            continue

        # Skip checkpoint / participant-level duplicates
        if any(part in EXCLUDE_PARTS for part in p.parts):
            continue

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
                    "social",
                    "lonely",
                    "phq",
                    "gad",
                    "score",
                    "response",
                    "ema",
                ]
            )
        ]

        if not clinical:
            continue

        # Deduplicate files with essentially identical schemas
        schema_key = tuple(sorted(map(str, df.columns)))

        if schema_key in seen_schema:
            continue

        seen_schema.add(schema_key)

        results.append(
            {
                "path": str(p),
                "clinical": clinical,
                "useful": useful,
                "n_columns": len(df.columns),
            }
        )

results.sort(
    key=lambda x: (
        -len(x["clinical"]),
        x["path"],
    )
)

print()
print("=" * 110)
print("UNIQUE TOP-LEVEL ACTIVE / CLINICAL DATA SCHEMAS")
print("=" * 110)

for r in results[:40]:

    print()
    print("FILE:", r["path"])

    print("CLINICAL COLUMNS:")
    for c in r["clinical"]:
        print("   ", c)

    print("ID / DATE COLUMNS:")
    for c in r["useful"]:
        if c not in r["clinical"]:
            print("   ", c)

print()
print("=" * 110)
print("Unique clinical schemas:", len(results))
