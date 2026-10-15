from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path("/shared/Digital Clinic")

SKIP = {
    ".git",
    "__pycache__",
    "RAPHE",
    "raphe_outputs",
}

TERMS = [
    "lamp.analytics",
    "device_type",
    "user_agent",
]

hits = []


def skip(path):
    return any(
        part in SKIP
        for part in path.parts
    )


# ============================================================
# FILENAMES
# ============================================================

for path in ROOT.rglob("*"):

    if skip(path):
        continue

    name = path.name.lower()

    if any(
        term.replace(".", "_") in name
        or term in name
        for term in TERMS
    ):
        hits.append({
            "source": "filename",
            "file": str(path),
            "detail": path.name,
        })


# ============================================================
# CSV / TSV / JSON / TXT — small files only
# ============================================================

TEXT_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".txt",
}

for path in ROOT.rglob("*"):

    if (
        not path.is_file()
        or skip(path)
        or path.suffix.lower()
        not in TEXT_SUFFIXES
    ):
        continue

    try:
        if path.stat().st_size > 25_000_000:
            continue

        text = path.read_text(
            errors="ignore"
        )

    except Exception:
        continue

    lower = text.lower()

    found = [
        term
        for term in TERMS
        if term in lower
    ]

    if found:

        hits.append({
            "source": "text",
            "file": str(path),
            "detail": ", ".join(found),
        })


# ============================================================
# PARQUET SCHEMAS
# ============================================================

try:
    import pyarrow.parquet as pq

    for path in ROOT.rglob("*.parquet"):

        if skip(path):
            continue

        try:
            schema = (
                pq.ParquetFile(path)
                .schema_arrow
            )

            cols = [
                str(c)
                for c in schema.names
            ]

            matches = [
                c
                for c in cols
                if any(
                    term in c.lower()
                    for term in TERMS
                )
            ]

            if matches:

                hits.append({
                    "source": "parquet_schema",
                    "file": str(path),
                    "detail": ", ".join(matches),
                })

        except Exception:
            pass

except ImportError:
    pass


# ============================================================
# SQLITE / DB FILES
# ============================================================

for path in ROOT.rglob("*"):

    if (
        not path.is_file()
        or skip(path)
        or path.suffix.lower()
        not in {
            ".db",
            ".sqlite",
            ".sqlite3",
        }
    ):
        continue

    try:

        con = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
        )

        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """,
            con,
        )

        for table in tables["name"]:

            try:

                info = pd.read_sql_query(
                    f'PRAGMA table_info("{table}")',
                    con,
                )

                cols = (
                    info["name"]
                    .astype(str)
                    .tolist()
                )

                matches = [
                    c
                    for c in cols
                    if any(
                        term in c.lower()
                        for term in TERMS
                    )
                ]

                if matches:

                    hits.append({
                        "source": "sqlite_schema",
                        "file": str(path),
                        "detail":
                            f"{table}: "
                            + ", ".join(matches),
                    })

            except Exception:
                pass

        con.close()

    except Exception:
        pass


# ============================================================
# OUTPUT
# ============================================================

print("=" * 80)
print("LOCAL LAMP.ANALYTICS SEARCH")
print("=" * 80)

if not hits:

    print(
        "No local lamp.analytics / device_type / "
        "user_agent evidence found."
    )

else:

    out = (
        pd.DataFrame(hits)
        .drop_duplicates()
    )

    print(
        out.head(50).to_string(
            index=False
        )
    )

    print()
    print(
        "Total hits:",
        len(out),
    )

print("=" * 80)
