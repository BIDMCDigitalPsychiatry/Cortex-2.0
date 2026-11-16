from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata

ROOT = Path("/shared/Digital Clinic/demographics")

CROSSWALK = ROOT / "[deprecated] pt_start_dates.xlsx"

HOME = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/weekly_home_time.csv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def norm_text(x):
    if pd.isna(x):
        return None

    s = unicodedata.normalize(
        "NFKD",
        str(x)
    )

    s = s.encode(
        "ascii",
        "ignore"
    ).decode()

    s = s.lower().strip()

    s = re.sub(
        r"[^a-z0-9]+",
        " ",
        s
    )

    s = re.sub(
        r"\s+",
        " ",
        s
    ).strip()

    return s or None


def norm_col(x):
    return norm_text(x) or ""


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_age_col(columns):

    for c in columns:
        n = norm_col(c)

        if (
            "age" in n
            and "enrollment" in n
        ):
            return c

    return None


def find_name_col(columns):

    # Preferred specific forms first.
    preferred_terms = [
        "full name",
        "first and last",
        "first last",
    ]

    for term in preferred_terms:
        for c in columns:
            if term in norm_col(c):
                return c

    # Fallback to generic Name.
    for c in columns:
        if norm_col(c) == "name":
            return c

    return None


# ============================================================
# READ AGE SOURCES
# ============================================================

records = []

year_summary = []

for year in [
    2023,
    2024,
    2025,
    2026,
]:

    year_files = sorted(
        [
            p
            for p in ROOT.iterdir()
            if str(year) in p.name
            and p.suffix.lower()
            in {
                ".xlsx",
                ".xls",
                ".csv",
            }
        ]
    )

    for path in year_files:

        if path.name.startswith(
            "[deprecated]"
        ):
            continue

        if path.suffix.lower() in {
            ".xlsx",
            ".xls",
        }:

            xl = pd.ExcelFile(path)

            datasets = []

            for sheet in xl.sheet_names:
                try:
                    d = pd.read_excel(
                        path,
                        sheet_name=sheet,
                    )

                    datasets.append(
                        (
                            sheet,
                            d,
                        )
                    )

                except Exception:
                    pass

        else:

            datasets = []

            for encoding in [
                "utf-8",
                "utf-8-sig",
                "cp1252",
                "latin1",
            ]:
                try:
                    d = pd.read_csv(
                        path,
                        low_memory=False,
                        encoding=encoding,
                    )

                    datasets = [
                        (
                            f"CSV [{encoding}]",
                            d,
                        )
                    ]

                    break

                except UnicodeDecodeError:
                    continue

                except Exception:
                    continue

        for sheet, df in datasets:

            age_col = find_age_col(
                df.columns
            )

            name_col = find_name_col(
                df.columns
            )

            if (
                age_col is None
                or name_col is None
            ):
                continue

            tmp = pd.DataFrame(
                {
                    "year": year,
                    "name_norm":
                        df[name_col].apply(
                            norm_text
                        ),
                    "age":
                        pd.to_numeric(
                            df[age_col],
                            errors="coerce",
                        ),
                }
            )

            tmp = tmp[
                tmp["name_norm"].notna()
                & tmp["age"].between(
                    0,
                    120,
                )
            ].copy()

            records.append(
                tmp
            )

            year_summary.append(
                {
                    "year": year,
                    "file": path.name,
                    "sheet": sheet,
                    "valid_age_records":
                        len(tmp),
                    "unique_names":
                        tmp[
                            "name_norm"
                        ].nunique(),
                }
            )


if not records:
    raise RuntimeError(
        "No usable age + name sources found."
    )

age = pd.concat(
    records,
    ignore_index=True,
)


# ============================================================
# AGE CONSISTENCY
# ============================================================

age_by_name = (
    age.groupby(
        "name_norm"
    )["age"]
    .agg(
        [
            "count",
            "nunique",
            "min",
            "max",
        ]
    )
)

conflicting_age_names = (
    age_by_name[
        "nunique"
    ] > 1
).sum()

clean_names = set(
    age_by_name.index[
        age_by_name[
            "nunique"
        ].eq(1)
    ]
)

clean_age = (
    age[
        age["name_norm"].isin(
            clean_names
        )
    ]
    .groupby(
        "name_norm",
        as_index=False,
    )
    .agg(
        age=("age", "first"),
        first_source_year=(
            "year",
            "min",
        ),
        last_source_year=(
            "year",
            "max",
        ),
    )
)


# ============================================================
# CROSSWALK
# ============================================================

cw = pd.read_excel(
    CROSSWALK,
    sheet_name="6.10pt_start_dates",
)

required = {
    "Name",
    "Clinic ID",
}

missing = (
    required
    - set(cw.columns)
)

if missing:
    raise RuntimeError(
        f"Crosswalk missing columns: {missing}"
    )

cw = cw[
    [
        "Name",
        "Clinic ID",
    ]
].copy()

cw["name_norm"] = (
    cw["Name"].apply(
        norm_text
    )
)

cw["clinic_id"] = (
    cw["Clinic ID"]
    .astype(str)
    .str.strip()
)

cw = cw[
    cw["name_norm"].notna()
    & cw["clinic_id"].notna()
    & cw["clinic_id"].ne("")
    & cw["clinic_id"].ne("nan")
].copy()

clinic_counts = (
    cw.groupby(
        "name_norm"
    )["clinic_id"]
    .nunique()
)

ambiguous_crosswalk_names = (
    clinic_counts > 1
).sum()

unique_cw_names = set(
    clinic_counts.index[
        clinic_counts.eq(1)
    ]
)

cw_clean = (
    cw[
        cw["name_norm"].isin(
            unique_cw_names
        )
    ]
    .drop_duplicates(
        subset=[
            "name_norm"
        ]
    )
    [
        [
            "name_norm",
            "clinic_id",
        ]
    ]
)


# ============================================================
# HOME-TIME COHORT
# ============================================================

home = pd.read_csv(
    HOME,
    low_memory=False,
)

if "participant_id" not in home.columns:
    raise RuntimeError(
        "weekly_home_time.csv missing participant_id."
    )

# Keep only participants with an actual valid home-time measure.
preferred_home_cols = [
    "time_at_home_fraction",
    "time_at_home_pct",
    "mean_daily_home_fraction",
    "home_fraction",
    "home_time_fraction",
    "home_time_pct",
]

home_col = next(
    (
        c
        for c in preferred_home_cols
        if c in home.columns
    ),
    None,
)

if home_col is None:
    candidates = [
        c
        for c in home.columns
        if "home" in c.lower()
        and (
            "fraction" in c.lower()
            or "pct" in c.lower()
            or "percent" in c.lower()
        )
    ]

    if len(candidates) == 1:
        home_col = candidates[0]
    else:
        raise RuntimeError(
            "Could not identify valid home-time column. "
            f"Candidates: {candidates}"
        )

home[home_col] = pd.to_numeric(
    home[home_col],
    errors="coerce",
)

valid_home = home[
    home[home_col].notna()
].copy()

home_ids = set(
    valid_home[
        "participant_id"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

print(
    "\nHome-time validity column:",
    home_col,
)


# ============================================================
# LINK AGE -> CLINIC ID -> HOME TIME
# ============================================================

linked = clean_age.merge(
    cw_clean,
    on="name_norm",
    how="inner",
)

linked["in_home_cohort"] = (
    linked[
        "clinic_id"
    ].isin(
        home_ids
    )
)

home_age = linked[
    linked[
        "in_home_cohort"
    ]
].copy()

home_age = (
    home_age
    .drop_duplicates(
        subset=[
            "clinic_id"
        ]
    )
)


# ============================================================
# REPORT AGGREGATES ONLY
# ============================================================

print("=" * 110)
print("ALL-YEAR AGE COVERAGE AUDIT")
print("=" * 110)

print("\nAGE SOURCES")
print("-" * 110)

ys = pd.DataFrame(
    year_summary
)

if len(ys):
    print(
        ys[
            [
                "year",
                "sheet",
                "valid_age_records",
                "unique_names",
            ]
        ]
        .to_string(
            index=False
        )
    )

print()
print("COMBINED AGE DATA")
print("-" * 110)

print(
    "Total usable age records:",
    len(age),
)

print(
    "Unique people by normalized name:",
    age["name_norm"].nunique(),
)

print(
    "Names with conflicting ages across files:",
    int(conflicting_age_names),
)

print(
    "Names with one consistent age:",
    len(clean_age),
)

print()
print("CROSSWALK")
print("-" * 110)

print(
    "Unique crosswalk names:",
    cw["name_norm"].nunique(),
)

print(
    "Names linked to >1 Clinic ID:",
    int(
        ambiguous_crosswalk_names
    ),
)

print()
print("HOME-TIME COHORT")
print("-" * 110)

print(
    "Home-time participants:",
    len(home_ids),
)

print(
    "Home-time participants with clean age match:",
    len(home_age),
)

coverage = (
    100
    * len(home_age)
    / len(home_ids)
    if len(home_ids)
    else np.nan
)

print(
    "Age coverage:",
    f"{coverage:.1f}%"
)

if len(home_age):

    print()
    print("AGE DISTRIBUTION")
    print("-" * 110)

    print(
        "Minimum:",
        int(
            home_age["age"].min()
        )
    )

    print(
        "Q1:",
        round(
            home_age[
                "age"
            ].quantile(
                0.25
            ),
            1,
        )
    )

    print(
        "Median:",
        round(
            home_age[
                "age"
            ].median(),
            1,
        )
    )

    print(
        "Mean:",
        round(
            home_age[
                "age"
            ].mean(),
            1,
        )
    )

    print(
        "Q3:",
        round(
            home_age[
                "age"
            ].quantile(
                0.75
            ),
            1,
        )
    )

    print(
        "Maximum:",
        int(
            home_age[
                "age"
            ].max()
        )
    )

    print()
    print(
        "MATCHED HOME-TIME PARTICIPANTS BY EARLIEST AGE SOURCE YEAR"
    )
    print("-" * 110)

    print(
        home_age[
            "first_source_year"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

print()
print(
    "No names, emails, dates of birth, or patient-level records were printed."
)

