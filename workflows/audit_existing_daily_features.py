from pathlib import Path

import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex_longitudinal/data/"
    "participant_daily"
)


print("=" * 80)
print("DAILY FEATURE INVENTORY")
print("=" * 80)

if not ROOT.exists():
    raise RuntimeError(
        f"Directory not found: {ROOT}"
    )


files = sorted(
    list(ROOT.glob("*.csv"))
    + list(ROOT.glob("*.parquet"))
)

print(
    "Files:",
    len(files),
)


if not files:
    raise RuntimeError(
        "No daily feature files found."
    )


# ============================================================
# INSPECT FIRST FILES
# ============================================================

schemas = {}
examples = []


for path in files[:25]:

    try:

        if path.suffix == ".csv":

            df = pd.read_csv(
                path,
                nrows=20,
            )

        else:

            df = pd.read_parquet(
                path
            ).head(20)

    except Exception as e:

        print(
            "Could not read:",
            path.name,
            repr(e),
        )

        continue


    schemas[
        path.name
    ] = list(
        df.columns
    )


    if len(examples) < 3:

        examples.append(
            (
                path.name,
                df.head(3),
            )
        )


# ============================================================
# UNION / COMMON COLUMNS
# ============================================================

column_sets = [
    set(v)
    for v in schemas.values()
]


union = sorted(
    set().union(
        *column_sets
    )
)

common = sorted(
    set.intersection(
        *column_sets
    )
)


print()
print("=" * 80)
print("COMMON COLUMNS")
print("=" * 80)

for c in common:
    print(c)


print()
print("=" * 80)
print("ALL COLUMNS SEEN")
print("=" * 80)

for c in union:
    print(c)


# ============================================================
# EXAMPLES
# ============================================================

print()
print("=" * 80)
print("EXAMPLE FILES")
print("=" * 80)

for name, df in examples:

    print()
    print("-" * 80)
    print(name)

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================
# FILE-NAME SAMPLE
# ============================================================

print()
print("=" * 80)
print("FIRST 20 FILES")
print("=" * 80)

for path in files[:20]:
    print(path.name)


print()
print("=" * 80)
print("DONE")
print("=" * 80)
