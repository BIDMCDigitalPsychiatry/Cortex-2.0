from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)

HOME_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_home_time.csv"
)

FUNC_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_functioning"
    / "daily_functioning_clean.csv"
)

EMA_FILE = (
    ROOT
    / "clinical/daily_ema"
    / "daily_ema_all_participants.csv"
)

MAP_FILE = (
    ROOT
    / "clinical/daily_ema"
    / "daily_ema_gps_participants.csv"
)

OUT = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_longitudinal"
)

OUT.mkdir(parents=True, exist_ok=True)

MERGED_OUT = OUT / "daily_home_ema_analysis_master.csv"
PARTICIPANT_OUT = OUT / "daily_home_ema_participant_counts.csv"


# ============================================================
# HELPERS
# ============================================================

def choose_col(df, candidates, label, required=True):
    for c in candidates:
        if c in df.columns:
            print(f"{label}: using '{c}'")
            return c

    if required:
        print()
        print(f"ERROR: could not identify {label}")
        print("Available columns:")
        print(df.columns.tolist())
        raise SystemExit(1)

    print(f"{label}: not found")
    return None


def normalize_date(s):
    return pd.to_datetime(
        s,
        errors="coerce"
    ).dt.normalize()


def print_summary(name, df, id_col="participant_id"):
    print()
    print("=" * 90)
    print(name)
    print("=" * 90)
    print("Rows:", len(df))

    if id_col in df.columns:
        print(
            "Participants:",
            df[id_col].nunique()
        )


# ============================================================
# LOAD
# ============================================================

home = pd.read_csv(
    HOME_FILE,
    low_memory=False
)

func = pd.read_csv(
    FUNC_FILE,
    low_memory=False
)

ema = pd.read_csv(
    EMA_FILE,
    low_memory=False
)

mapping = pd.read_csv(
    MAP_FILE,
    low_memory=False
)

print_summary("RAW DAILY HOME", home)
print("Columns:")
print(home.columns.tolist())

print_summary("RAW DAILY FUNCTIONING", func)
print("Columns:")
print(func.columns.tolist())

print_summary("RAW DAILY EMA", ema)
print("Columns:")
print(ema.columns.tolist())

print_summary(
    "RAW SOURCE ↔ CORTEX MAPPING",
    mapping,
    id_col="cortex_participant_id"
)
print("Columns:")
print(mapping.columns.tolist())


# ============================================================
# IDENTIFY HOME COLUMNS
# ============================================================

home_id = choose_col(
    home,
    [
        "participant_id",
        "cortex_participant_id",
    ],
    "home participant ID"
)

home_date = choose_col(
    home,
    [
        "date",
        "day",
        "calendar_date",
    ],
    "home date"
)

home_pct = choose_col(
    home,
    [
        "home_pct",
        "home_time_pct",
        "time_at_home_pct",
        "home_percentage",
        "percent_home",
        "pct_home",
        "home_fraction_pct",
        "home_fraction",
        "home_time_fraction",
    ],
    "home-time measure"
)

gps_hours = choose_col(
    home,
    [
        "supported_hours",
        "supported_duration_ms",
        "gps_supported_hours",
        "supported_gps_hours",
        "coverage_hours",
        "gps_hours",
        "observed_hours",
    ],
    "GPS support hours",
    required=False
)

home = home.rename(
    columns={
        home_id: "participant_id",
        home_date: "date",
        home_pct: "home_value_raw",
    }
)

home["participant_id"] = (
    home["participant_id"]
    .astype(str)
    .str.strip()
)

home["date"] = normalize_date(
    home["date"]
)

home["home_value_raw"] = pd.to_numeric(
    home["home_value_raw"],
    errors="coerce"
)

# Convert fraction to percentage if necessary.
q99 = home["home_value_raw"].quantile(0.99)

if pd.notna(q99) and q99 <= 1.05:
    print(
        "Home-time appears to be stored as fraction; "
        "converting to percentage."
    )
    home["home_pct"] = (
        home["home_value_raw"] * 100
    )
else:
    home["home_pct"] = (
        home["home_value_raw"]
    )

if gps_hours is not None:
    home["gps_hours"] = pd.to_numeric(
        home[gps_hours],
        errors="coerce"
    )

    if gps_hours == "supported_duration_ms":
        home["gps_hours"] = (
            home["gps_hours"] / 1000 / 60 / 60
        )
else:
    home["gps_hours"] = np.nan

home = (
    home[
        [
            "participant_id",
            "date",
            "home_pct",
            "gps_hours",
        ]
    ]
    .dropna(
        subset=[
            "participant_id",
            "date",
            "home_pct",
        ]
    )
)

# one home-time value per participant/day
home = (
    home.groupby(
        [
            "participant_id",
            "date",
        ],
        as_index=False
    )
    .agg(
        home_pct=("home_pct", "mean"),
        gps_hours=("gps_hours", "max"),
    )
)

print_summary(
    "CLEAN DAILY HOME",
    home
)


# ============================================================
# FUNCTIONING DATA
# ============================================================

func_id = choose_col(
    func,
    [
        "participant_id",
        "cortex_participant_id",
    ],
    "functioning participant ID"
)

func_date = choose_col(
    func,
    [
        "date",
        "day",
        "assessment_date",
    ],
    "functioning date"
)

func_value = choose_col(
    func,
    [
        "functioning_score",
        "daily_function",
        "functioning_raw",
        "functioning",
    ],
    "functioning score"
)

func = func.rename(
    columns={
        func_id: "participant_id",
        func_date: "date",
        func_value: "functioning",
    }
)

func["participant_id"] = (
    func["participant_id"]
    .astype(str)
    .str.strip()
)

func["date"] = normalize_date(
    func["date"]
)

func["functioning"] = pd.to_numeric(
    func["functioning"],
    errors="coerce"
)

func = (
    func[
        [
            "participant_id",
            "date",
            "functioning",
        ]
    ]
    .dropna(
        subset=[
            "participant_id",
            "date",
            "functioning",
        ]
    )
    .groupby(
        [
            "participant_id",
            "date",
        ],
        as_index=False
    )
    .agg(
        functioning=("functioning", "mean")
    )
)

print_summary(
    "CLEAN DAILY FUNCTIONING",
    func
)

print()
print("FUNCTIONING SCALE")
print(
    func["functioning"]
    .describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .95,
            .99,
        ]
    )
)

print()
print("Observed functioning values:")
vals = np.sort(
    func["functioning"]
    .dropna()
    .unique()
)

if len(vals) <= 30:
    print(vals)
else:
    print(
        "Too many unique values to print:",
        len(vals)
    )


# ============================================================
# CORRECT SOURCE -> CORTEX EMA MAPPING
# ============================================================

source_col = choose_col(
    mapping,
    [
        "source_participant_id",
        "source_participant_id_from_survey",
    ],
    "mapping source ID"
)

cortex_col = choose_col(
    mapping,
    [
        "cortex_participant_id",
        "participant_id",
    ],
    "mapping Cortex ID"
)

map_clean = (
    mapping[
        [
            source_col,
            cortex_col,
        ]
    ]
    .dropna()
    .rename(
        columns={
            source_col: "source_participant_id",
            cortex_col: "participant_id",
        }
    )
)

map_clean["source_participant_id"] = (
    map_clean["source_participant_id"]
    .astype(str)
    .str.strip()
)

map_clean["participant_id"] = (
    map_clean["participant_id"]
    .astype(str)
    .str.strip()
)

# Check ambiguity
source_ambiguity = (
    map_clean.groupby(
        "source_participant_id"
    )["participant_id"]
    .nunique()
)

cortex_ambiguity = (
    map_clean.groupby(
        "participant_id"
    )["source_participant_id"]
    .nunique()
)

print()
print("=" * 90)
print("MAPPING QC")
print("=" * 90)

print(
    "Unique source IDs:",
    map_clean["source_participant_id"].nunique()
)

print(
    "Unique Cortex IDs:",
    map_clean["participant_id"].nunique()
)

print(
    "Source IDs mapping to >1 Cortex ID:",
    int((source_ambiguity > 1).sum())
)

print(
    "Cortex IDs mapping to >1 source ID:",
    int((cortex_ambiguity > 1).sum())
)

if (source_ambiguity > 1).any():
    print(
        "\nERROR: ambiguous source IDs exist. "
        "Stopping rather than silently merging."
    )
    raise SystemExit(1)

map_clean = map_clean.drop_duplicates(
    subset=["source_participant_id"]
)


# ============================================================
# DAILY EMA
# ============================================================

ema_id = choose_col(
    ema,
    [
        "participant_id",
        "source_participant_id",
    ],
    "EMA source participant ID"
)

ema_date = choose_col(
    ema,
    [
        "date",
        "day",
    ],
    "EMA date"
)

anxiety_col = choose_col(
    ema,
    ["anxiety", "daily_anxiety"],
    "daily anxiety",
    required=False
)

mood_col = choose_col(
    ema,
    ["mood", "daily_mood"],
    "daily mood",
    required=False
)

ema_func_col = choose_col(
    ema,
    ["daily_function", "functioning_score"],
    "EMA functioning",
    required=False
)

ema["source_participant_id"] = (
    ema[ema_id]
    .astype(str)
    .str.strip()
)

ema["date"] = normalize_date(
    ema[ema_date]
)

ema = ema.merge(
    map_clean,
    on="source_participant_id",
    how="left",
    validate="many_to_one"
)

print()
print("=" * 90)
print("EMA MAPPING RESULT")
print("=" * 90)

print(
    "EMA source participants:",
    ema["source_participant_id"].nunique()
)

print(
    "EMA participants successfully mapped to Cortex:",
    ema.loc[
        ema["participant_id"].notna(),
        "participant_id"
    ].nunique()
)

print(
    "Unmapped EMA source participants:",
    ema.loc[
        ema["participant_id"].isna(),
        "source_participant_id"
    ].nunique()
)

for old, new in [
    (anxiety_col, "anxiety"),
    (mood_col, "mood"),
    (ema_func_col, "ema_functioning"),
]:
    if old is not None:
        ema[new] = pd.to_numeric(
            ema[old],
            errors="coerce"
        )

keep = [
    "participant_id",
    "date",
]

for c in [
    "anxiety",
    "mood",
    "ema_functioning",
]:
    if c in ema.columns:
        keep.append(c)

ema = (
    ema[
        ema["participant_id"].notna()
    ][keep]
    .groupby(
        [
            "participant_id",
            "date",
        ],
        as_index=False
    )
    .mean(
        numeric_only=True
    )
)

print_summary(
    "CLEAN MAPPED DAILY EMA",
    ema
)

for c in [
    "anxiety",
    "mood",
    "ema_functioning",
]:
    if c in ema.columns:
        print()
        print(c.upper())
        print(
            ema[c].describe()
        )
        unique = np.sort(
            ema[c]
            .dropna()
            .unique()
        )
        if len(unique) <= 30:
            print(
                "Observed values:",
                unique
            )


# ============================================================
# PARTICIPANT OVERLAPS
# ============================================================

home_ids = set(
    home["participant_id"]
)

func_ids = set(
    func["participant_id"]
)

ema_ids = set(
    ema["participant_id"]
)

print()
print("=" * 90)
print("PARTICIPANT OVERLAP")
print("=" * 90)

print(
    "Home participants:",
    len(home_ids)
)

print(
    "Functioning participants:",
    len(func_ids)
)

print(
    "Mapped EMA participants:",
    len(ema_ids)
)

print(
    "Home ∩ functioning:",
    len(home_ids & func_ids)
)

print(
    "Home ∩ EMA:",
    len(home_ids & ema_ids)
)

print(
    "Home ∩ functioning ∩ EMA:",
    len(home_ids & func_ids & ema_ids)
)


# ============================================================
# BUILD MASTER DAILY TABLE
# ============================================================

master = home.merge(
    func,
    on=[
        "participant_id",
        "date",
    ],
    how="outer"
)

master = master.merge(
    ema,
    on=[
        "participant_id",
        "date",
    ],
    how="outer"
)

master = master.sort_values(
    [
        "participant_id",
        "date",
    ]
)

# If the dedicated functioning file and EMA functioning overlap,
# compare them rather than assuming they are identical.
if "ema_functioning" in master.columns:
    overlap = master.dropna(
        subset=[
            "functioning",
            "ema_functioning",
        ]
    )

    print()
    print("=" * 90)
    print("DEDICATED FUNCTIONING vs EMA FUNCTIONING")
    print("=" * 90)

    print(
        "Overlapping participant-days:",
        len(overlap)
    )

    print(
        "Participants:",
        overlap["participant_id"].nunique()
    )

    if len(overlap) > 2:
        print(
            "Pearson correlation:",
            overlap[
                [
                    "functioning",
                    "ema_functioning",
                ]
            ].corr().iloc[0, 1]
        )

        diff = (
            overlap["functioning"]
            - overlap["ema_functioning"]
        )

        print(
            "Mean difference:",
            diff.mean()
        )

        print(
            "Max absolute difference:",
            diff.abs().max()
        )


# ============================================================
# GPS COVERAGE COUNTS
# ============================================================

print()
print("=" * 90)
print("GPS COVERAGE")
print("=" * 90)

if master["gps_hours"].notna().any():

    for threshold in [
        4,
        6,
        8,
    ]:
        z = master[
            master["gps_hours"] >= threshold
        ]

        print(
            f">= {threshold} h:",
            f"{len(z)} person-days |",
            f"{z['participant_id'].nunique()} participants"
        )

else:
    print(
        "No daily GPS-hours column was available "
        "in daily_home_time.csv."
    )


# ============================================================
# SAME-DAY PAIRED HOME + FUNCTIONING
# ============================================================

same_day = master.dropna(
    subset=[
        "home_pct",
        "functioning",
    ]
)

print()
print("=" * 90)
print("SAME-DAY HOME + FUNCTIONING")
print("=" * 90)

print(
    "Paired days:",
    len(same_day)
)

print(
    "Participants:",
    same_day["participant_id"].nunique()
)

if len(same_day):
    counts = (
        same_day.groupby(
            "participant_id"
        )
        .size()
    )

    print(
        "Paired days/person:"
    )
    print(
        counts.describe()
    )


# ============================================================
# BUILD t -> t+1 PAIRS
# ============================================================

# We deliberately match by exact calendar date + 1 day,
# rather than using shift(), so missing days do not create
# false "next day" observations.

next_func = (
    func.rename(
        columns={
            "date": "outcome_date",
            "functioning": "functioning_t1",
        }
    )
    .copy()
)

next_func["date"] = (
    next_func["outcome_date"]
    - pd.Timedelta(days=1)
)

lagged = home.merge(
    next_func[
        [
            "participant_id",
            "date",
            "outcome_date",
            "functioning_t1",
        ]
    ],
    on=[
        "participant_id",
        "date",
    ],
    how="inner"
)

# Add functioning on exposure day t
func_t = func.rename(
    columns={
        "functioning": "functioning_t"
    }
)

lagged = lagged.merge(
    func_t[
        [
            "participant_id",
            "date",
            "functioning_t",
        ]
    ],
    on=[
        "participant_id",
        "date",
    ],
    how="left"
)

# Add anxiety/mood on day t
lagged = lagged.merge(
    ema,
    on=[
        "participant_id",
        "date",
    ],
    how="left"
)

print()
print("=" * 90)
print("EXACT DAY t -> DAY t+1 PAIRS")
print("=" * 90)

print(
    "Home[t] + functioning[t+1] pairs:",
    len(lagged)
)

print(
    "Participants:",
    lagged["participant_id"].nunique()
)

if len(lagged):
    c = (
        lagged.groupby(
            "participant_id"
        )
        .size()
    )

    print()
    print("Pairs/person:")
    print(
        c.describe()
    )

    print(
        "\nParticipants with >= 5 pairs:",
        int((c >= 5).sum())
    )

    print(
        "Participants with >= 10 pairs:",
        int((c >= 10).sum())
    )

    print(
        "Participants with >= 20 pairs:",
        int((c >= 20).sum())
    )


# ============================================================
# PRIMARY-MODEL ELIGIBILITY
# ============================================================

eligible = lagged.copy()

if eligible["gps_hours"].notna().any():
    eligible = eligible[
        eligible["gps_hours"] >= 6
    ]

print()
print("=" * 90)
print("PRIMARY MODEL ELIGIBILITY: >=6 h GPS")
print("=" * 90)

print(
    "Home[t] + function[t+1] pairs:",
    len(eligible)
)

print(
    "Participants:",
    eligible["participant_id"].nunique()
)

with_prior_function = eligible.dropna(
    subset=["functioning_t"]
)

print(
    "Pairs also having function[t]:",
    len(with_prior_function)
)

print(
    "Participants with function[t] adjustment:",
    with_prior_function[
        "participant_id"
    ].nunique()
)

if "anxiety" in eligible.columns:
    anx = eligible.dropna(
        subset=["anxiety"]
    )
    print(
        "Pairs with anxiety[t]:",
        len(anx),
        "| participants:",
        anx["participant_id"].nunique()
    )

if "mood" in eligible.columns:
    mood = eligible.dropna(
        subset=["mood"]
    )
    print(
        "Pairs with mood[t]:",
        len(mood),
        "| participants:",
        mood["participant_id"].nunique()
    )


# ============================================================
# PARTICIPANT AUDIT TABLE
# ============================================================

all_ids = sorted(
    set(master["participant_id"].dropna())
)

participant_rows = []

for pid in all_ids:

    h = home[
        home["participant_id"] == pid
    ]

    f = func[
        func["participant_id"] == pid
    ]

    e = ema[
        ema["participant_id"] == pid
    ]

    l = lagged[
        lagged["participant_id"] == pid
    ]

    participant_rows.append(
        {
            "participant_id": pid,

            "n_home_days": len(h),

            "n_home_days_ge6h": (
                int(
                    (
                        h["gps_hours"] >= 6
                    ).sum()
                )
                if h["gps_hours"].notna().any()
                else np.nan
            ),

            "n_functioning_days": len(f),

            "n_ema_days": len(e),

            "n_lagged_home_function_pairs": len(l),

            "n_lagged_pairs_ge6h": (
                int(
                    (
                        l["gps_hours"] >= 6
                    ).sum()
                )
                if len(l)
                and l["gps_hours"].notna().any()
                else np.nan
            ),
        }
    )

participant_counts = pd.DataFrame(
    participant_rows
)

participant_counts.to_csv(
    PARTICIPANT_OUT,
    index=False
)

master.to_csv(
    MERGED_OUT,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(MERGED_OUT)
print(PARTICIPANT_OUT)

print()
print("=" * 90)
print("WHAT TO SEND BACK")
print("=" * 90)

print(
    """
Please paste the sections:

1. MAPPING QC
2. PARTICIPANT OVERLAP
3. FUNCTIONING SCALE
4. GPS COVERAGE
5. EXACT DAY t -> DAY t+1 PAIRS
6. PRIMARY MODEL ELIGIBILITY

That is enough to decide the exact primary model without guessing.
"""
)
