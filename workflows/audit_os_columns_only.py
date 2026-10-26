from pathlib import Path
import re

import pandas as pd


ROOT = Path("/shared/Digital Clinic")

CANDIDATE = re.compile(
    r"^(?:"
    r"os|"
    r"os_version|"
    r"operating_system|"
    r"platform|"
    r"device_type|"
    r"device_model|"
    r"phone_type|"
    r"phone_model|"
    r"system_name|"
    r"system_version|"
    r"manufacturer|"
    r"brand|"
    r"model|"
    r"user_agent"
    r")$",
    flags=re.IGNORECASE,
)

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "cortex2_outputs",
}

results = []


def should_skip(path):
    return any(
        part in SKIP_DIRS
        for part in path.parts
    )


# ============================================================
# CSV / TSV HEADERS
# ============================================================

seen_header_signatures = set()

for path in ROOT.rglob("*"):

    if (
        not path.is_file()
        or should_skip(path)
    ):
        continue

    if path.suffix.lower() not in {
        ".csv",
        ".tsv",
    }:
        continue

    try:
        sep = (
            "\t"
            if path.suffix.lower() == ".tsv"
            else ","
        )

        df = pd.read_csv(
            path,
            sep=sep,
            nrows=5,
            low_memory=False,
        )

    except Exception:
        continue

    candidate_cols = [
        c
        for c in df.columns
        if CANDIDATE.match(
            str(c).strip()
        )
    ]

    if not candidate_cols:
        continue

    # Avoid printing thousands of identical passive-file schemas.
    signature = (
        str(path.parent),
        tuple(
            sorted(candidate_cols)
        ),
    )

    if signature in seen_header_signatures:
        continue

    seen_header_signatures.add(
        signature
    )

    for col in candidate_cols:

        values = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        results.append({
            "file":
                str(path),

            "column":
                str(col),

            "sample_values":
                ", ".join(
                    values[:10]
                ),
        })


# ============================================================
# PARQUET SCHEMAS
# ============================================================

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


if pq is not None:

    seen_parquet = set()

    for path in ROOT.rglob("*.parquet"):

        if should_skip(path):
            continue

        try:
            schema = (
                pq.ParquetFile(path)
                .schema_arrow
            )

        except Exception:
            continue

        candidate_cols = [
            c
            for c in schema.names
            if CANDIDATE.match(
                str(c).strip()
            )
        ]

        if not candidate_cols:
            continue

        signature = (
            str(path.parent),
            tuple(
                sorted(candidate_cols)
            ),
        )

        if signature in seen_parquet:
            continue

        seen_parquet.add(
            signature
        )

        try:
            df = pd.read_parquet(
                path,
                columns=candidate_cols,
            ).head(20)

        except Exception:
            df = None

        for col in candidate_cols:

            values = ""

            if (
                df is not None
                and col in df
            ):

                values = ", ".join(
                    df[col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()[:10]
                )

            results.append({
                "file":
                    str(path),

                "column":
                    str(col),

                "sample_values":
                    values,
            })


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("EXPLICIT OS / DEVICE METADATA COLUMNS")
print("=" * 80)

if not results:

    print(
        "No candidate OS/device metadata columns found."
    )

else:

    out = pd.DataFrame(
        results
    )

    print(
        out.to_string(
            index=False
        )
    )

    save = Path(
        "../cortex2_outputs/digital_clinic/"
        "inventory/os_column_candidates.csv"
    )

    save.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        save,
        index=False,
    )

    print()
    print(
        "Saved:",
        save,
    )
