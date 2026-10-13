from pathlib import Path
import re
import pandas as pd


ROOT = Path("/shared/Digital Clinic")

OUT = Path(
    "../raphe_outputs/digital_clinic/"
    "inventory/os_metadata_search.csv"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# Exact OS/device-platform terms we actually care about.
PATTERN = re.compile(
    r"\b(?:ios|android|iphone|ipad)\b",
    flags=re.IGNORECASE,
)

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "raphe_outputs",
    "passive",
    "RAPHE",
}

MAX_SIZE = 20 * 1024 * 1024

matches = []


def skip(path):
    return any(
        part in SKIP_DIRS
        for part in path.parts
    )


for path in ROOT.rglob("*"):

    if (
        not path.is_file()
        or skip(path)
    ):
        continue

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # CSV / TSV
    # --------------------------------------------------------

    if suffix in {
        ".csv",
        ".tsv",
    }:

        try:
            if path.stat().st_size > MAX_SIZE:
                continue

            sep = (
                "\t"
                if suffix == ".tsv"
                else ","
            )

            df = pd.read_csv(
                path,
                sep=sep,
                dtype=str,
                low_memory=False,
            )

        except Exception:
            continue

        for col in df.columns:

            s = (
                df[col]
                .dropna()
                .astype(str)
            )

            if len(s) == 0:
                continue

            mask = s.str.contains(
                PATTERN,
                regex=True,
                na=False,
            )

            if not mask.any():
                continue

            values = (
                s[mask]
                .value_counts()
                .head(10)
            )

            for value, n in values.items():

                matches.append({
                    "file":
                        str(path),

                    "column":
                        str(col),

                    "value":
                        str(value)[:200],

                    "count":
                        int(n),
                })

    # --------------------------------------------------------
    # JSON / TXT
    # --------------------------------------------------------

    elif suffix in {
        ".json",
        ".jsonl",
        ".txt",
    }:

        try:
            if path.stat().st_size > MAX_SIZE:
                continue

            text = path.read_text(
                errors="ignore"
            )

        except Exception:
            continue

        found = sorted(
            set(
                m.group(0)
                for m in PATTERN.finditer(text)
            )
        )

        for value in found:

            matches.append({
                "file":
                    str(path),

                "column":
                    "<text>",

                "value":
                    value,

                "count":
                    None,
            })


# ============================================================
# SAVE + SHORT TERMINAL OUTPUT
# ============================================================

result = pd.DataFrame(
    matches
)

if len(result) == 0:

    print()
    print(
        "No explicit iOS / Android metadata found."
    )

else:

    result.to_csv(
        OUT,
        index=False,
    )

    print()
    print("=" * 72)
    print("OS METADATA SEARCH")
    print("=" * 72)

    print(
        "Matches:",
        f"{len(result):,}",
    )

    print(
        "Unique files:",
        result["file"].nunique(),
    )

    print()
    print(
        "Top candidate files:"
    )

    summary = (
        result
        .groupby(
            [
                "file",
                "column",
            ]
        )
        .agg(
            values=(
                "value",
                lambda x:
                    ", ".join(
                        sorted(
                            set(x)
                        )[:8]
                    )
            ),
            matches=(
                "value",
                "size",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "matches",
                "file",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(30)
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Full results saved:"
    )

    print(
        OUT
    )
