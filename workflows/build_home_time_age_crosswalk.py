from pathlib import Path
import pandas as pd
import re
import unicodedata

DEMOG_DIR = Path("/shared/Digital Clinic/demographics")

INTAKE_FILES = [
    DEMOG_DIR / "BIDMC Digital Psychiatry Clinic - Patient Intake Form 2025.xlsx",
]
INTAKE_FILES.extend(
    DEMOG_DIR.glob("*2026*Patient Demographics.xlsx")
)
INTAKE_FILES = list(dict.fromkeys(INTAKE_FILES))

MAP_FILE = DEMOG_DIR / "[deprecated] pt_start_dates.xlsx"

HOME_FILE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_clinical_weekly_features_with_home.parquet"
)

OUTFILE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "home_time_participant_age.csv"
)


def norm_name(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x or None


def norm_id(x):
    if pd.isna(x):
        return None
    return str(x).strip().upper()


def find_col(df, candidates):
    cmap = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in cmap:
            return cmap[candidate.lower()]

    return None


# ============================================================
# 1. INTAKE: NAME + AGE
# ============================================================

parts = []

for path in INTAKE_FILES:
    xls = pd.ExcelFile(path)

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)

        age_col = find_col(
            df,
            ["Age at enrollment", "Age"],
        )

        name_col = find_col(
            df,
            [
                "Name (First and Last)",
                "Name (First & Last)",
                "Patient Name",
                "Participant Name",
                "Full Name",
                "Name",
            ],
        )

        if age_col is None or name_col is None:
            continue

        sub = df[[name_col, age_col]].copy()
        sub.columns = ["name", "age"]

        parts.append(sub)

if not parts:
    raise RuntimeError("No usable intake name/age data found.")

intake = pd.concat(parts, ignore_index=True)

intake["name_norm"] = intake["name"].map(norm_name)
intake["age"] = pd.to_numeric(
    intake["age"],
    errors="coerce",
)

intake = intake[
    intake["name_norm"].notna()
    & intake["age"].between(18, 100)
].copy()

age_check = (
    intake.groupby("name_norm")["age"]
    .agg(["nunique", "min", "max"])
)

conflicting_age_names = set(
    age_check[
        age_check["nunique"] > 1
    ].index
)

intake_clean = intake[
    ~intake["name_norm"].isin(
        conflicting_age_names
    )
].copy()

intake_person = (
    intake_clean
    .groupby("name_norm", as_index=False)
    .agg(age=("age", "first"))
)

print("=" * 100)
print("INTAKE")
print("=" * 100)
print("Valid intake rows:", len(intake))
print("Unique intake names:", intake["name_norm"].nunique())
print("Names with conflicting ages:", len(conflicting_age_names))


# ============================================================
# 2. CROSSWALK: NAME -> CLINIC ID
# Use authoritative 6.10pt_start_dates sheet
# ============================================================

sheet = "6.10pt_start_dates"

mapping = pd.read_excel(
    MAP_FILE,
    sheet_name=sheet,
)

name_col = find_col(
    mapping,
    [
        "Name",
        "Patient Name",
        "Participant Name",
        "Full Name",
    ],
)

clinic_col = find_col(
    mapping,
    [
        "Clinic ID",
        "clinic_id",
        "clinicid",
    ],
)

if name_col is None or clinic_col is None:
    raise RuntimeError(
        "Could not find Name and Clinic ID in "
        "6.10pt_start_dates."
    )

mapping = mapping[
    [name_col, clinic_col]
].copy()

mapping.columns = [
    "name",
    "participant_id",
]

mapping["name_norm"] = mapping["name"].map(norm_name)
mapping["participant_id"] = mapping["participant_id"].map(norm_id)

mapping = mapping[
    mapping["name_norm"].notna()
    & mapping["participant_id"].notna()
].copy()

# valid Clinic ID pattern e.g. AMO327
clinic_pattern = re.compile(
    r"^[A-Z]{3}[0-9]{3}$"
)

mapping["valid_clinic_id"] = (
    mapping["participant_id"]
    .map(
        lambda x: bool(clinic_pattern.match(x))
    )
)

mapping = mapping[
    mapping["valid_clinic_id"]
].copy()

name_id_counts = (
    mapping.groupby("name_norm")["participant_id"]
    .nunique()
)

ambiguous_names = set(
    name_id_counts[
        name_id_counts > 1
    ].index
)

mapping_clean = mapping[
    ~mapping["name_norm"].isin(
        ambiguous_names
    )
].copy()

mapping_person = (
    mapping_clean[
        [
            "name_norm",
            "participant_id",
        ]
    ]
    .drop_duplicates()
)

print()
print("=" * 100)
print("CROSSWALK")
print("=" * 100)
print("Rows in 6.10pt_start_dates:", len(mapping))
print("Names linked to >1 Clinic ID:", len(ambiguous_names))


# ============================================================
# 3. AGE -> CLINIC ID
# ============================================================

age_map = intake_person.merge(
    mapping_person,
    on="name_norm",
    how="inner",
)

print()
print("=" * 100)
print("AGE + CLINIC ID")
print("=" * 100)
print("Validated matched people:", len(age_map))


# ============================================================
# 4. HOME-TIME COHORT
# ============================================================

home = pd.read_parquet(
    HOME_FILE,
    columns=[
        "participant_id",
        "time_at_home_pct",
    ],
)

home = home[
    home["time_at_home_pct"].notna()
].copy()

home_participants = (
    home[["participant_id"]]
    .drop_duplicates()
)

home_participants["participant_id"] = (
    home_participants["participant_id"]
    .map(norm_id)
)

print()
print("=" * 100)
print("HOME-TIME COHORT")
print("=" * 100)
print(
    "Home-time participants:",
    len(home_participants),
)


# ============================================================
# 5. FINAL MERGE
# ============================================================

final = home_participants.merge(
    age_map[
        [
            "participant_id",
            "age",
        ]
    ],
    on="participant_id",
    how="left",
)

matched = final[
    final["age"].notna()
].copy()

print()
print("=" * 100)
print("FINAL AGE COVERAGE")
print("=" * 100)

print(
    "Home-time participants with age:",
    len(matched),
)

print(
    "Coverage:",
    f"{100 * len(matched) / len(final):.1f}%"
)

if len(matched):
    print("Age min:", f"{matched['age'].min():.0f}")
    print("Age median:", f"{matched['age'].median():.1f}")
    print("Age mean:", f"{matched['age'].mean():.1f}")
    print("Age max:", f"{matched['age'].max():.0f}")


# ============================================================
# 6. SAVE DE-IDENTIFIED TABLE
# ============================================================

matched[
    [
        "participant_id",
        "age",
    ]
].drop_duplicates().to_csv(
    OUTFILE,
    index=False,
)

print()
print("Saved:")
print(OUTFILE)
