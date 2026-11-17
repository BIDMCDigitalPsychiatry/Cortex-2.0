from pathlib import Path
import pandas as pd
import numpy as np

SURVEY_PATH = Path(
    "/shared/Digital Clinic/active/survey_tabular_long.csv"
)

MAP_PATH = Path(
    "/shared/Digital Clinic/raphe_outputs/"
    "digital_clinic/inventory/raphe_participant_map.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/clinical/daily_ema"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

QUESTIONS = {
    "Overall, how would I rate my anxiety today?":
        "anxiety",

    "I am able to manage my day-to-day life":
        "daily_function",

    "Overall, how would you rate your mood today?":
        "mood",
}


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    SURVEY_PATH,
    low_memory=False,
)

m = pd.read_csv(
    MAP_PATH,
    low_memory=False,
)


print("=" * 100)
print("PARTICIPANT MAP")
print("=" * 100)

print(
    m.head().to_string(
        index=False
    )
)

print(
    "\nColumns:",
    list(m.columns),
)


# ============================================================
# DAILY SURVEY ONLY
# ============================================================

daily = (
    df[
        df[
            "activity_name"
        ].eq(
            "Daily Survey"
        )
        & df[
            "item"
        ].isin(
            QUESTIONS.keys()
        )
    ]
    .copy()
)

daily[
    "ema_variable"
] = daily[
    "item"
].map(
    QUESTIONS
)

daily[
    "value_numeric"
] = pd.to_numeric(
    daily[
        "value"
    ],
    errors="coerce",
)

daily[
    "datetime"
] = pd.to_datetime(
    daily[
        "datetime"
    ],
    utc=True,
    errors="coerce",
)

daily[
    "date"
] = daily[
    "datetime"
].dt.date


# ============================================================
# RAW SUMMARY
# ============================================================

print()
print("=" * 100)
print("DAILY EMA RAW SUMMARY")
print("=" * 100)

print(
    "Item rows:",
    len(daily)
)

print(
    "Unique source participants:",
    daily[
        "participant_id"
    ].nunique()
)

print(
    "Date range:",
    daily[
        "date"
    ].min(),
    "to",
    daily[
        "date"
    ].max(),
)

print()

for variable in [
    "anxiety",
    "mood",
    "daily_function",
]:

    g = daily[
        daily[
            "ema_variable"
        ].eq(
            variable
        )
    ]

    print(
        f"{variable:16s} "
        f"responses={len(g):6d} | "
        f"participants={g['participant_id'].nunique():4d} | "
        f"mean={g['value_numeric'].mean():.3f} | "
        f"sd={g['value_numeric'].std():.3f} | "
        f"min={g['value_numeric'].min():.3f} | "
        f"max={g['value_numeric'].max():.3f}"
    )


# ============================================================
# CHECK DUPLICATE COMPLETIONS PER PERSON-DAY
# ============================================================

counts = (
    daily.groupby(
        [
            "participant_id",
            "date",
            "ema_variable",
        ]
    )
    .size()
    .reset_index(
        name="n"
    )
)

print()
print("=" * 100)
print("MULTIPLE RESPONSES ON SAME DAY")
print("=" * 100)

print(
    counts[
        "n"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# KEEP LAST DAILY RESPONSE IF MULTIPLE
# ============================================================

daily = (
    daily.sort_values(
        "datetime"
    )
    .drop_duplicates(
        subset=[
            "participant_id",
            "date",
            "ema_variable",
        ],
        keep="last",
    )
)


# ============================================================
# WIDE DAILY TABLE
# ============================================================

wide = (
    daily.pivot_table(
        index=[
            "participant_id",
            "date",
        ],
        columns="ema_variable",
        values="value_numeric",
        aggfunc="last",
    )
    .reset_index()
)

wide.columns.name = None


# ============================================================
# PARTICIPANT-LEVEL COVERAGE
# ============================================================

coverage = (
    wide.groupby(
        "participant_id"
    )
    .agg(
        first_date=(
            "date",
            "min",
        ),
        last_date=(
            "date",
            "max",
        ),
        n_daily_survey_days=(
            "date",
            "nunique",
        ),
        n_anxiety=(
            "anxiety",
            "count",
        ),
        n_mood=(
            "mood",
            "count",
        ),
        n_daily_function=(
            "daily_function",
            "count",
        ),
        anxiety_sd=(
            "anxiety",
            "std",
        ),
        mood_sd=(
            "mood",
            "std",
        ),
        daily_function_sd=(
            "daily_function",
            "std",
        ),
    )
    .reset_index()
)


print()
print("=" * 100)
print("DAILY EMA DAYS PER PARTICIPANT — ALL PARTICIPANTS")
print("=" * 100)

print(
    coverage[
        "n_daily_survey_days"
    ]
    .describe(
        percentiles=[
            .25,
            .5,
            .75,
            .9,
        ]
    )
    .to_string()
)


# ============================================================
# MAP DAILY EMA SOURCE UID → CORTEX PARTICIPANT ID
# Auto-detect the two columns from the existing map.
# ============================================================

print()
print("=" * 100)
print("TRYING PARTICIPANT MAP")
print("=" * 100)


# Determine which map column contains survey source UIDs.
best_source_col = None
best_overlap = -1

survey_ids = set(
    wide[
        "participant_id"
    ].astype(str)
)

for c in m.columns:

    map_values = set(
        m[
            c
        ]
        .dropna()
        .astype(str)
    )

    overlap = len(
        survey_ids
        & map_values
    )

    print(
        f"{c}: overlap with survey IDs = {overlap}"
    )

    if overlap > best_overlap:
        best_overlap = overlap
        best_source_col = c


other_cols = [
    c
    for c in m.columns
    if c != best_source_col
]

if (
    best_source_col is None
    or best_overlap == 0
    or len(other_cols) != 1
):
    raise RuntimeError(
        "Could not safely identify participant-map columns."
    )

cortex_col = other_cols[0]

print()
print(
    "Source UID column:",
    best_source_col
)

print(
    "Cortex participant column:",
    cortex_col
)


mapped = wide.merge(
    m[
        [
            best_source_col,
            cortex_col,
        ]
    ],
    left_on="participant_id",
    right_on=best_source_col,
    how="inner",
    suffixes=("_survey", "_map"),
)

# After the merge:
# participant_id_survey = source survey UID
# participant_id_map    = Cortex participant ID
#
# Rename explicitly so downstream code is unambiguous.
mapped = mapped.rename(
    columns={
        "participant_id_map":
            "cortex_participant_id",
        "participant_id_survey":
            "source_participant_id_from_survey",
    }
)


# ============================================================
# GPS COHORT
#
# Existing clinical feature table tells us which participants
# currently have GPS clinical features.
# ============================================================

GPS_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical/"
    "gps_clinical_weekly_features.parquet"
)

gps = pd.read_parquet(
    GPS_PATH,
    columns=[
        "participant_id"
    ],
)

gps_ids = set(
    gps[
        "participant_id"
    ]
    .dropna()
    .astype(str)
    .unique()
)

mapped[
    "cortex_participant_id"
] = mapped[
    "cortex_participant_id"
].astype(str)

gps_ema = mapped[
    mapped[
        "cortex_participant_id"
    ].isin(
        gps_ids
    )
].copy()


# ============================================================
# GPS + EMA COVERAGE
# ============================================================

gps_cov = (
    gps_ema.groupby(
        "cortex_participant_id"
    )
    .agg(
        first_ema_date=(
            "date",
            "min",
        ),
        last_ema_date=(
            "date",
            "max",
        ),
        n_ema_days=(
            "date",
            "nunique",
        ),
        n_anxiety=(
            "anxiety",
            "count",
        ),
        n_mood=(
            "mood",
            "count",
        ),
        n_daily_function=(
            "daily_function",
            "count",
        ),
        anxiety_sd=(
            "anxiety",
            "std",
        ),
        mood_sd=(
            "mood",
            "std",
        ),
        daily_function_sd=(
            "daily_function",
            "std",
        ),
    )
    .reset_index()
    .sort_values(
        "n_ema_days",
        ascending=False,
    )
)


print()
print("=" * 100)
print("GPS PARTICIPANTS WITH DAILY EMA")
print("=" * 100)

print(
    "GPS participants total:",
    len(gps_ids)
)

print(
    "GPS participants with >=1 daily EMA:",
    gps_cov[
        "cortex_participant_id"
    ].nunique()
)

print(
    "Total GPS-participant EMA days:",
    gps_ema[
        [
            "cortex_participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print()

print(
    gps_cov.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("GPS COHORT EMA COVERAGE THRESHOLDS")
print("=" * 100)

for threshold in [
    5,
    10,
    20,
    30,
    50,
    100,
]:

    n = int(
        (
            gps_cov[
                "n_ema_days"
            ]
            >= threshold
        ).sum()
    )

    print(
        f">={threshold:3d} EMA days: {n} participants"
    )


# ============================================================
# SAVE
# ============================================================

WIDE_PATH = (
    OUT_DIR
    / "daily_ema_all_participants.csv"
)

GPS_EMA_PATH = (
    OUT_DIR
    / "daily_ema_gps_participants.csv"
)

COVERAGE_PATH = (
    OUT_DIR
    / "daily_ema_gps_participant_coverage.csv"
)

wide.to_csv(
    WIDE_PATH,
    index=False,
)

gps_ema.to_csv(
    GPS_EMA_PATH,
    index=False,
)

gps_cov.to_csv(
    COVERAGE_PATH,
    index=False,
)

print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(WIDE_PATH)
print(GPS_EMA_PATH)
print(COVERAGE_PATH)
