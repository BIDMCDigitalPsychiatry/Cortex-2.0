from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

DF_ROOT = (
    ROOT
    / "home_time/daily_functioning"
)

SLOPE_PATH = (
    DF_ROOT
    / "daily_functioning_individual_home_slopes.csv"
)

DAILY_PATH = (
    DF_ROOT
    / "daily_functioning_home_time_analysis_dataset.csv"
)

WEEKLY_PATH = (
    ROOT
    / "gps_clinical_weekly_features_with_home.parquet"
)

PAPER_PATH = (
    ROOT
    / "home_time/paper_replication/"
      "paper_replication_analysis_dataset.csv"
)

OUT = (
    DF_ROOT
    / "daily_functioning_slope_participant_phenotypes.csv"
)

# ============================================================
# LOAD SLOPES
# ============================================================

slopes = pd.read_csv(
    SLOPE_PATH,
    low_memory=False,
)

slopes = slopes[
    slopes["estimable"].eq(True)
    & slopes["within_slope_per_10pp"].notna()
].copy()

slopes["slope_group"] = np.select(
    [
        slopes["within_slope_per_10pp"] < 0,
        slopes["within_slope_per_10pp"] > 0,
    ],
    [
        "negative",
        "positive",
    ],
    default="zero",
)

base = slopes[
    [
        "participant_id",
        "within_slope_per_10pp",
        "slope_group",
        "n_days",
    ]
].copy()

# ============================================================
# DAILY FUNCTIONING + HOME-TIME PROFILE
# ============================================================

daily = pd.read_csv(
    DAILY_PATH,
    low_memory=False,
)

daily = daily[
    daily["analysis_definition"]
    .eq("daily_mean_primary")
].copy()

daily["assessment_date"] = pd.to_datetime(
    daily["assessment_date"],
    errors="coerce",
)

daily["functioning_score"] = pd.to_numeric(
    daily["functioning_score"],
    errors="coerce",
)

daily["home_time_pct"] = pd.to_numeric(
    daily["home_time_pct"],
    errors="coerce",
)

daily["is_weekend"] = (
    daily["assessment_date"]
    .dt.dayofweek
    .ge(5)
)

core = (
    daily.groupby("participant_id")
    .agg(
        functioning_mean=(
            "functioning_score",
            "mean",
        ),
        functioning_sd=(
            "functioning_score",
            "std",
        ),
        functioning_median=(
            "functioning_score",
            "median",
        ),
        home_mean_pct=(
            "home_time_pct",
            "mean",
        ),
        home_sd_pct=(
            "home_time_pct",
            "std",
        ),
        home_median_pct=(
            "home_time_pct",
            "median",
        ),
        matched_days=(
            "home_time_pct",
            "size",
        ),
    )
    .reset_index()
)

weekday = (
    daily.loc[
        ~daily["is_weekend"]
    ]
    .groupby("participant_id")[
        "home_time_pct"
    ]
    .mean()
    .rename("weekday_home_mean_pct")
)

weekend = (
    daily.loc[
        daily["is_weekend"]
    ]
    .groupby("participant_id")[
        "home_time_pct"
    ]
    .mean()
    .rename("weekend_home_mean_pct")
)

ww = pd.concat(
    [
        weekday,
        weekend,
    ],
    axis=1,
).reset_index()

ww["weekend_minus_weekday_home_pct"] = (
    ww["weekend_home_mean_pct"]
    - ww["weekday_home_mean_pct"]
)

core = core.merge(
    ww,
    on="participant_id",
    how="left",
)

base = base.merge(
    core,
    on="participant_id",
    how="left",
)

# ============================================================
# WEEKLY CLINICAL SCORES
# ============================================================

if WEEKLY_PATH.exists():

    weekly = pd.read_parquet(
        WEEKLY_PATH
    )

    print("\nWeekly clinical file columns:")
    print(
        list(weekly.columns)
    )

    weekly["participant_id"] = (
        weekly["participant_id"]
        .astype(str)
    )

    # Candidate score columns already produced by existing workflows.
    candidates = [
        "depression_score",
        "anxiety_score",
        "phq8_total",
        "phq9_total",
        "gad7_total",
        "phq8_score",
        "phq9_score",
        "gad7_score",
    ]

    score_cols = [
        c
        for c in candidates
        if c in weekly.columns
    ]

    print(
        "\nClinical score columns found:",
        score_cols,
    )

    clinical_parts = []

    for c in score_cols:

        weekly[c] = pd.to_numeric(
            weekly[c],
            errors="coerce",
        )

        tmp = (
            weekly.groupby(
                "participant_id"
            )[c]
            .agg(
                [
                    "mean",
                    "std",
                    "count",
                ]
            )
            .reset_index()
        )

        tmp = tmp.rename(
            columns={
                "mean": f"{c}_mean",
                "std": f"{c}_sd",
                "count": f"{c}_n",
            }
        )

        clinical_parts.append(
            tmp
        )

    for tmp in clinical_parts:
        base = base.merge(
            tmp,
            on="participant_id",
            how="left",
        )

else:
    print(
        "\nWARNING: weekly clinical file not found:"
    )
    print(WEEKLY_PATH)

# ============================================================
# AGE
# ============================================================

if PAPER_PATH.exists():

    paper = pd.read_csv(
        PAPER_PATH,
        low_memory=False,
    )

    if {
        "participant_id",
        "age",
    }.issubset(paper.columns):

        paper["age"] = pd.to_numeric(
            paper["age"],
            errors="coerce",
        )

        age = (
            paper.groupby(
                "participant_id"
            )["age"]
            .median()
            .rename("age")
            .reset_index()
        )

        base = base.merge(
            age,
            on="participant_id",
            how="left",
        )

# ============================================================
# COVERAGE
# ============================================================

print()
print("=" * 110)
print("PARTICIPANT PHENOTYPE TABLE")
print("=" * 110)

print(
    "\nParticipants:",
    len(base),
)

print("\nSlope groups:")
print(
    base["slope_group"]
    .value_counts()
    .to_string()
)

print()
print("=" * 110)
print("FEATURE COVERAGE")
print("=" * 110)

exclude = {
    "participant_id",
    "within_slope_per_10pp",
    "slope_group",
}

features = [
    c
    for c in base.columns
    if c not in exclude
]

coverage_rows = []

for c in features:

    valid = base[c].notna()

    row = {
        "feature": c,
        "overall_n":
            int(valid.sum()),
        "overall_pct":
            100 * valid.mean(),
    }

    for group in [
        "negative",
        "positive",
        "zero",
    ]:
        g = base[
            base["slope_group"].eq(group)
        ]

        row[
            f"{group}_n"
        ] = int(
            g[c].notna().sum()
        )

        row[
            f"{group}_pct"
        ] = (
            100
            * g[c].notna().mean()
            if len(g)
            else np.nan
        )

    coverage_rows.append(row)

coverage = pd.DataFrame(
    coverage_rows
)

print(
    coverage.round(1)
    .to_string(index=False)
)

# ============================================================
# NEGATIVE vs POSITIVE DESCRIPTIVE COMPARISON
# ============================================================

print()
print("=" * 110)
print("NEGATIVE vs POSITIVE SLOPE PARTICIPANTS")
print("=" * 110)

comparison_rows = []

for c in features:

    if not pd.api.types.is_numeric_dtype(
        base[c]
    ):
        continue

    neg = (
        base.loc[
            base["slope_group"].eq(
                "negative"
            ),
            c,
        ]
        .dropna()
    )

    pos = (
        base.loc[
            base["slope_group"].eq(
                "positive"
            ),
            c,
        ]
        .dropna()
    )

    if (
        len(neg) < 2
        or len(pos) < 2
    ):
        continue

    # Standardized mean difference:
    # descriptive effect size, not a significance claim.
    pooled_sd = np.sqrt(
        (
            neg.var(ddof=1)
            + pos.var(ddof=1)
        )
        / 2
    )

    smd = (
        (neg.mean() - pos.mean())
        / pooled_sd
        if pooled_sd > 0
        else np.nan
    )

    comparison_rows.append(
        {
            "feature": c,

            "negative_n":
                len(neg),
            "negative_mean":
                neg.mean(),
            "negative_median":
                neg.median(),
            "negative_q1":
                neg.quantile(.25),
            "negative_q3":
                neg.quantile(.75),

            "positive_n":
                len(pos),
            "positive_mean":
                pos.mean(),
            "positive_median":
                pos.median(),
            "positive_q1":
                pos.quantile(.25),
            "positive_q3":
                pos.quantile(.75),

            "standardized_mean_difference":
                smd,
        }
    )

comparison = pd.DataFrame(
    comparison_rows
)

if len(comparison):

    comparison[
        "abs_smd"
    ] = (
        comparison[
            "standardized_mean_difference"
        ]
        .abs()
    )

    comparison = (
        comparison
        .sort_values(
            "abs_smd",
            ascending=False,
        )
    )

    print(
        comparison[
            [
                "feature",
                "negative_n",
                "negative_mean",
                "negative_median",
                "positive_n",
                "positive_mean",
                "positive_median",
                "standardized_mean_difference",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

# ============================================================
# SAVE
# ============================================================

base.to_csv(
    OUT,
    index=False,
)

coverage.to_csv(
    DF_ROOT
    / "daily_functioning_slope_feature_coverage.csv",
    index=False,
)

comparison.to_csv(
    DF_ROOT
    / "daily_functioning_negative_positive_comparison.csv",
    index=False,
)

print("\nSaved:")
print(" ", OUT)
print(
    " ",
    DF_ROOT
    / "daily_functioning_slope_feature_coverage.csv",
)
print(
    " ",
    DF_ROOT
    / "daily_functioning_negative_positive_comparison.csv",
)
