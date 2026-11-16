from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

SURVEY_PATH = Path(
    "/shared/Digital Clinic/active/survey_tabular_long.csv"
)

MAP_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "inventory/raphe_participant_map.csv"
)

if not MAP_PATH.exists():
    MAP_PATH = Path(
        "/shared/Digital Clinic/raphe_outputs/digital_clinic/"
        "inventory/raphe_participant_map.csv"
    )

HOME_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_home_time.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_functioning"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_OUT = OUT_DIR / "daily_functioning_audit_summary.csv"
DAILY_OUT = OUT_DIR / "daily_functioning_clean.csv"
OVERLAP_OUT = OUT_DIR / "daily_functioning_home_overlap.csv"

TARGET_ITEM = "I am able to manage my day-to-day life"

# ============================================================
# LOAD PARTICIPANT MAP
# ============================================================

mapping = pd.read_csv(
    MAP_PATH,
    low_memory=False,
)

required_map_cols = {
    "source_participant_id",
    "participant_id",
}

missing_map = (
    required_map_cols
    - set(mapping.columns)
)

if missing_map:
    raise RuntimeError(
        "Participant map missing expected columns: "
        + ", ".join(sorted(missing_map))
    )

mapping["source_participant_id"] = (
    mapping["source_participant_id"]
    .astype(str)
    .str.strip()
)

mapping["participant_id"] = (
    mapping["participant_id"]
    .astype(str)
    .str.strip()
)

# Preserve the same deterministic duplicate-source rule
# used in build_gps_clinical_weekly_windows.py.
duplicate_sources = (
    mapping
    .groupby("source_participant_id")["participant_id"]
    .nunique()
)

duplicate_sources = duplicate_sources[
    duplicate_sources > 1
]

if len(duplicate_sources):
    print("\nDuplicate source participant mappings detected:")
    for source_id in duplicate_sources.index:
        aliases = sorted(
            mapping.loc[
                mapping["source_participant_id"].eq(source_id),
                "participant_id",
            ]
            .astype(str)
            .unique()
        )
        print(
            f"  source with multiple aliases "
            f"-> retaining {aliases[0]}"
        )

mapping = (
    mapping
    .sort_values(
        [
            "source_participant_id",
            "participant_id",
        ]
    )
    .drop_duplicates(
        subset=["source_participant_id"],
        keep="first",
    )
    .copy()
)

print("\nParticipant map:")
print("  Source IDs:", mapping["source_participant_id"].nunique())
print("  Analytical IDs:", mapping["participant_id"].nunique())

# ============================================================
# LOAD SURVEY
# ============================================================

survey = pd.read_csv(SURVEY_PATH, low_memory=False)

print("=" * 110)
print("DAILY FUNCTIONING AUDIT")
print("=" * 110)

print("\nSurvey columns:")
print(list(survey.columns))

required = ["participant_id", "activity_name", "item"]
missing = [c for c in required if c not in survey.columns]
if missing:
    raise RuntimeError(f"Missing expected survey columns: {missing}")

# ============================================================
# IDENTIFY FUNCTIONING ITEMS
# ============================================================

daily = survey[
    survey["activity_name"].astype(str).str.strip().eq("Daily Survey")
].copy()

print("\nDaily Survey rows:", len(daily))
print("Daily Survey participants:", daily["participant_id"].nunique())

function_candidates = (
    daily.loc[
        daily["item"].astype(str).str.contains(
            "manage my day-to-day life",
            case=False,
            na=False,
        ),
        "item",
    ]
    .value_counts(dropna=False)
)

print("\nFunctioning item variants found:")
print(function_candidates.to_string())

func = daily[
    daily["item"].astype(str).str.contains(
        "manage my day-to-day life",
        case=False,
        na=False,
    )
].copy()

print("\nFunctioning rows:", len(func))
print(
    "Functioning source participants:",
    func["participant_id"].nunique(),
)

# Map MindLAMP source ID -> analytical participant ID.
func["source_participant_id"] = (
    func["participant_id"]
    .astype(str)
    .str.strip()
)

func = func.merge(
    mapping[
        [
            "source_participant_id",
            "participant_id",
        ]
    ],
    on="source_participant_id",
    how="inner",
    suffixes=(
        "_mindlamp",
        "_clinical",
    ),
)

func = func.rename(
    columns={
        "participant_id_clinical":
            "participant_id"
    }
)

print(
    "Mapped functioning rows:",
    len(func),
)

print(
    "Mapped functioning participants:",
    func["participant_id"].nunique(),
)

# ============================================================
# IDENTIFY VALUE COLUMN
# ============================================================

possible_value_cols = [
    "value",
    "response",
    "answer",
    "score",
    "data",
]

value_col = None
for c in possible_value_cols:
    if c in func.columns:
        nonnull = func[c].notna().sum()
        if nonnull > 0:
            value_col = c
            break

if value_col is None:
    # Find plausible columns by excluding identifiers / metadata
    print("\nCould not automatically identify response column.")
    print("\nNon-null counts in functioning rows:")
    print(func.notna().sum().sort_values(ascending=False).to_string())
    raise RuntimeError(
        "Please inspect the printed columns/non-null counts and identify the response column."
    )

print(f"\nUsing response column: {value_col}")

func["functioning_raw"] = func[value_col]
func["functioning_score"] = pd.to_numeric(
    func[value_col],
    errors="coerce",
)

print("\nRaw response distribution:")
print(
    func["functioning_raw"]
    .astype(str)
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nNumeric response distribution:")
print(
    func["functioning_score"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nNumeric summary:")
print(func["functioning_score"].describe().to_string())

# ============================================================
# TIMESTAMP
# ============================================================

timestamp_candidates = [
    "datetime",
    "timestamp",
    "time",
    "date",
    "start_time",
    "completed_at",
]

timestamp_col = None

for c in timestamp_candidates:
    if c not in func.columns:
        continue

    # Prefer human-readable datetime fields when available.
    if c == "datetime":
        parsed = pd.to_datetime(
            func[c],
            errors="coerce",
            utc=True,
        )
    else:
        vals = func[c]

        if pd.api.types.is_numeric_dtype(vals):
            numeric = pd.to_numeric(vals, errors="coerce")
            med = numeric.dropna().abs().median()

            # Infer Unix timestamp unit from magnitude.
            if pd.isna(med):
                continue
            elif med > 1e17:
                unit = "ns"
            elif med > 1e14:
                unit = "us"
            elif med > 1e11:
                unit = "ms"
            else:
                unit = "s"

            parsed = pd.to_datetime(
                numeric,
                unit=unit,
                errors="coerce",
                utc=True,
            )
        else:
            parsed = pd.to_datetime(
                vals,
                errors="coerce",
                utc=True,
            )

    if parsed.notna().sum() > 0:
        timestamp_col = c
        func["timestamp_utc"] = parsed
        break

if timestamp_col is None:
    print("\nCould not automatically identify a timestamp column.")
    print(func.notna().sum().sort_values(ascending=False).to_string())
    raise RuntimeError("Timestamp column not identified.")

print(f"\nUsing timestamp column: {timestamp_col}")

# Convert UTC timestamp to local Eastern time
func["timestamp_local"] = (
    func["timestamp_utc"]
    .dt.tz_convert("America/New_York")
)

func["assessment_date"] = (
    func["timestamp_local"]
    .dt.date
)

func["assessment_hour"] = (
    func["timestamp_local"]
    .dt.hour
    + func["timestamp_local"].dt.minute / 60
)

# ============================================================
# TIMING DISTRIBUTION
# ============================================================

print("\nAssessment hour distribution (local time):")
print(func["assessment_hour"].describe().to_string())

hour_bins = pd.cut(
    func["assessment_hour"],
    bins=[0, 6, 12, 18, 24],
    right=False,
    labels=[
        "00:00-05:59",
        "06:00-11:59",
        "12:00-17:59",
        "18:00-23:59",
    ],
)

print("\nAssessment time-of-day:")
print(hour_bins.value_counts().sort_index().to_string())

# ============================================================
# DUPLICATES PER PARTICIPANT-DAY
# ============================================================

per_day_counts = (
    func.groupby(
        ["participant_id", "assessment_date"]
    )
    .size()
    .rename("n_functioning_responses")
    .reset_index()
)

print("\nResponses per participant-day:")
print(
    per_day_counts["n_functioning_responses"]
    .describe()
    .to_string()
)

print(
    "\nParticipant-days with >1 functioning response:",
    int((per_day_counts["n_functioning_responses"] > 1).sum()),
)

print(
    "Percent participant-days with >1 response:",
    round(
        100
        * (per_day_counts["n_functioning_responses"] > 1).mean(),
        2,
    ),
)

# ============================================================
# COLLAPSE TO ONE VALUE PER PARTICIPANT-DAY
# For audit only: use LAST response of the day.
# We will revisit this choice before modeling if duplicates matter.
# ============================================================

func = func.sort_values(
    ["participant_id", "timestamp_local"]
)

daily_func = (
    func.dropna(subset=["functioning_score"])
    .groupby(
        ["participant_id", "assessment_date"],
        as_index=False,
    )
    .last()
)

keep_cols = [
    "participant_id",
    "assessment_date",
    "timestamp_local",
    "assessment_hour",
    "functioning_raw",
    "functioning_score",
]

daily_func = daily_func[keep_cols]

print("\nUnique participant-days with numeric functioning:")
print(len(daily_func))

print("Participants with numeric functioning:")
print(daily_func["participant_id"].nunique())

# ============================================================
# PARTICIPANT COVERAGE
# ============================================================

coverage = (
    daily_func.groupby("participant_id")
    .agg(
        n_functioning_days=("assessment_date", "nunique"),
        first_date=("assessment_date", "min"),
        last_date=("assessment_date", "max"),
    )
    .reset_index()
)

print("\nFunctioning days per participant:")
print(
    coverage["n_functioning_days"]
    .describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    .to_string()
)

# ============================================================
# LOAD HOME-TIME DATA
# ============================================================

home = pd.read_csv(HOME_PATH, low_memory=False)

print("\n" + "=" * 110)
print("HOME-TIME FILE")
print("=" * 110)

print("\nColumns:")
print(list(home.columns))

if "participant_id" not in home.columns:
    raise RuntimeError("participant_id not found in home-time file.")

# Find date column
home_date_candidates = [
    "date",
    "local_date",
    "day",
    "study_date",
    "assessment_date",
]

home_date_col = None
for c in home_date_candidates:
    if c in home.columns:
        home_date_col = c
        break

if home_date_col is None:
    raise RuntimeError(
        "Could not identify date column in daily_home_time.csv"
    )

home["assessment_date"] = pd.to_datetime(
    home[home_date_col],
    errors="coerce",
).dt.date

# Find home fraction / percent
home_value_candidates = [
    "time_at_home_pct",
    "time_at_home_fraction",
    "home_time_pct",
    "home_fraction",
    "home_time_fraction",
    "home_pct",
    "daily_home_fraction",
]

home_value_col = None
for c in home_value_candidates:
    if c in home.columns:
        home_value_col = c
        break

if home_value_col is None:
    print("\nPotential numeric columns in home file:")
    numeric_cols = home.select_dtypes(include=np.number).columns.tolist()
    print(numeric_cols)
    raise RuntimeError(
        "Could not identify home-time value column."
    )

print(f"\nUsing home date column: {home_date_col}")
print(f"Using home value column: {home_value_col}")

home["home_value"] = pd.to_numeric(
    home[home_value_col],
    errors="coerce",
)

# Convert fraction to percentage if necessary
non_null_home = home["home_value"].dropna()

if len(non_null_home) and non_null_home.quantile(0.99) <= 1.01:
    home["home_time_pct"] = home["home_value"] * 100
else:
    home["home_time_pct"] = home["home_value"]

# ============================================================
# SAME-DAY OVERLAP
# ============================================================

merged = daily_func.merge(
    home[
        [
            "participant_id",
            "assessment_date",
            "home_time_pct",
        ]
    ],
    on=["participant_id", "assessment_date"],
    how="left",
)

n_func = len(merged)
n_home = merged["home_time_pct"].notna().sum()

print("\n" + "=" * 110)
print("FUNCTIONING × SAME-DAY HOME-TIME OVERLAP")
print("=" * 110)

print(f"\nFunctioning participant-days: {n_func}")
print(f"With valid same-day home time: {n_home}")
print(
    f"Overlap: {100*n_home/n_func:.1f}%"
    if n_func
    else "Overlap: NA"
)

print(
    "Participants with >=1 same-day functioning/home pair:",
    merged.loc[
        merged["home_time_pct"].notna(),
        "participant_id",
    ].nunique(),
)

pairs_per_person = (
    merged.dropna(
        subset=["functioning_score", "home_time_pct"]
    )
    .groupby("participant_id")
    .size()
)

if len(pairs_per_person):
    print("\nMatched days per participant:")
    print(
        pairs_per_person.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9]
        ).to_string()
    )

# ============================================================
# SCORE × COVERAGE CROSS-TAB
# ============================================================

matched = merged.dropna(
    subset=["functioning_score", "home_time_pct"]
)

print("\nMatched functioning score distribution:")
print(
    matched["functioning_score"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# SAVE
# ============================================================

daily_func.to_csv(
    DAILY_OUT,
    index=False,
)

merged.to_csv(
    OVERLAP_OUT,
    index=False,
)

audit_rows = [
    {
        "metric": "raw_functioning_rows",
        "value": len(func),
    },
    {
        "metric": "functioning_participants",
        "value": func["participant_id"].nunique(),
    },
    {
        "metric": "unique_functioning_participant_days",
        "value": len(daily_func),
    },
    {
        "metric": "duplicate_participant_days",
        "value": int(
            (per_day_counts["n_functioning_responses"] > 1).sum()
        ),
    },
    {
        "metric": "same_day_home_pairs",
        "value": int(n_home),
    },
    {
        "metric": "participants_with_same_day_pairs",
        "value": int(
            merged.loc[
                merged["home_time_pct"].notna(),
                "participant_id",
            ].nunique()
        ),
    },
]

pd.DataFrame(audit_rows).to_csv(
    AUDIT_OUT,
    index=False,
)

print("\nSaved:")
print(" ", DAILY_OUT)
print(" ", OVERLAP_OUT)
print(" ", AUDIT_OUT)
