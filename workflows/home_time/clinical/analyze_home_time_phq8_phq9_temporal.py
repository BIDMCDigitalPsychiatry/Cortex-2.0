from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian


# ============================================================
# PATHS
# ============================================================

DC = Path("/shared/Digital Clinic")

SURVEY_PATH = (
    DC / "active/survey_tabular_long.csv"
)

MAP_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/inventory/"
    "raphe_participant_map.csv"
)

if not MAP_PATH.exists():
    MAP_PATH = (
        DC
        / "raphe_outputs/digital_clinic/inventory/"
        "raphe_participant_map.csv"
    )

HOME_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/daily_home_time.csv"
)

OUT = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/phq_temporal_windows"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 20261110
N_WEEKDAY_DRAWS = 100


# ============================================================
# UTILITIES
# ============================================================

def identify_phq_item(text):
    """
    Map question text to PHQ item number.

    Uses content rather than row order so that the alternate
    PHQ-1 wording is handled correctly.
    """
    s = str(text).strip().lower()

    if "little interest or pleasure" in s:
        return "phq1"

    if "feeling down, depressed, or hopeless" in s:
        return "phq2"

    if "trouble falling or staying asleep" in s:
        return "phq3"

    if "feeling tired or having little energy" in s:
        return "phq4"

    if "poor appetite or overeating" in s:
        return "phq5"

    if "feeling bad about yourself" in s:
        return "phq6"

    if "trouble concentrating on things" in s:
        return "phq7"

    if "moving or speaking so slowly" in s:
        return "phq8"

    if (
        "better off dead" in s
        or "hurting yourself" in s
    ):
        return "phq9"

    if "how difficult have these problems" in s:
        return "difficulty"

    return "other"


def identify_home_column(df):
    """
    Locate daily home-time column without assuming the exact
    output name used by an earlier workflow version.
    """

    preferred = [
        "daily_home_fraction",
        "home_fraction",
        "time_at_home_fraction",
        "home_time_fraction",
        "daily_home_pct",
        "home_time_pct",
        "time_at_home_pct",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    candidates = []

    for c in df.columns:
        cl = c.lower()

        if (
            "home" in cl
            and (
                "fraction" in cl
                or "pct" in cl
                or "percent" in cl
            )
        ):
            candidates.append(c)

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify daily home-time column. "
        f"Candidates found: {candidates}"
    )


def median_or_nan(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.median(x))


def weekday_matched_summary(values, rng):
    """
    Paper-style weekday estimate.

    A complete 14-day window contains 10 weekdays and 4 weekend
    days. Draw 4 weekdays without replacement 100 times, compute
    the median home time for each draw, then take the median
    across the 100 draws.
    """

    values = np.asarray(values, dtype=float)

    if len(values) != 10:
        return np.nan

    if not np.isfinite(values).all():
        return np.nan

    estimates = []

    for _ in range(N_WEEKDAY_DRAWS):
        chosen = rng.choice(
            values,
            size=4,
            replace=False,
        )

        estimates.append(
            np.median(chosen)
        )

    return float(
        np.median(estimates)
    )


def gee_model(
    df,
    score_col,
    home_col,
    score_name,
    window_name,
):
    """
    Gaussian GEE:
        home-time percentage ~ symptom score

    Participant is the clustering unit.
    """

    d = df[
        [
            "participant_id",
            score_col,
            home_col,
        ]
    ].dropna().copy()

    result = {
        "score": score_name,
        "score_column": score_col,
        "window": window_name,
        "home_column": home_col,
        "n_assessments": len(d),
        "n_participants": (
            d["participant_id"].nunique()
        ),
        "score_mean": np.nan,
        "score_sd": np.nan,
        "home_mean_pct": np.nan,
        "home_median_pct": np.nan,
        "beta_pct_per_1_point": np.nan,
        "ci_low_pct_per_1_point": np.nan,
        "ci_high_pct_per_1_point": np.nan,
        "beta_pct_per_5_points": np.nan,
        "ci_low_pct_per_5_points": np.nan,
        "ci_high_pct_per_5_points": np.nan,
        "p_value": np.nan,
        "converged": False,
    }

    if len(d) == 0:
        return result

    result["score_mean"] = d[
        score_col
    ].mean()

    result["score_sd"] = d[
        score_col
    ].std()

    result["home_mean_pct"] = d[
        home_col
    ].mean()

    result["home_median_pct"] = d[
        home_col
    ].median()

    if (
        d["participant_id"].nunique() < 2
        or d[score_col].nunique() < 2
    ):
        return result

    X = sm.add_constant(
        d[[score_col]],
        has_constant="add",
    )

    model = sm.GEE(
        endog=d[home_col],
        exog=X,
        groups=d["participant_id"],
        family=Gaussian(),
        cov_struct=Exchangeable(),
    )

    fit = model.fit(
        maxiter=200
    )

    beta = float(
        fit.params[score_col]
    )

    ci = fit.conf_int().loc[
        score_col
    ]

    low = float(ci.iloc[0])
    high = float(ci.iloc[1])

    result.update(
        {
            "beta_pct_per_1_point":
                beta,

            "ci_low_pct_per_1_point":
                low,

            "ci_high_pct_per_1_point":
                high,

            "beta_pct_per_5_points":
                beta * 5,

            "ci_low_pct_per_5_points":
                low * 5,

            "ci_high_pct_per_5_points":
                high * 5,

            "p_value":
                float(
                    fit.pvalues[score_col]
                ),

            "converged":
                bool(
                    getattr(
                        fit,
                        "converged",
                        True,
                    )
                ),
        }
    )

    return result


# ============================================================
# LOAD PARTICIPANT MAP
# ============================================================

mapping = pd.read_csv(
    MAP_PATH
)

required_mapping = {
    "participant_id",
    "source_participant_id",
}

missing = (
    required_mapping
    - set(mapping.columns)
)

if missing:
    raise RuntimeError(
        f"Mapping missing columns: {missing}"
    )

mapping = mapping[
    [
        "participant_id",
        "source_participant_id",
    ]
].copy()

mapping[
    "source_participant_id"
] = (
    mapping[
        "source_participant_id"
    ]
    .astype(str)
)

# Same duplicate handling as the clinical workflow:
# retain one analytical alias per source participant.
mapping = (
    mapping
    .sort_values(
        [
            "source_participant_id",
            "participant_id",
        ]
    )
    .drop_duplicates(
        subset=[
            "source_participant_id"
        ],
        keep="first",
    )
)


# ============================================================
# BUILD PHQ-9 AND PHQ-8 FROM RAW SURVEY ITEMS
# ============================================================

survey = pd.read_csv(
    SURVEY_PATH,
    low_memory=False,
)

dep = survey[
    survey["activity_name"].eq(
        "Weekly Depression Survey"
    )
].copy()

dep["numeric_value"] = pd.to_numeric(
    dep["value"],
    errors="coerce",
)

dep["phq_item"] = dep[
    "item"
].apply(
    identify_phq_item
)

symptom_items = [
    f"phq{i}"
    for i in range(1, 10)
]

dep = dep[
    dep["phq_item"].isin(
        symptom_items
    )
    & dep["numeric_value"].between(
        0,
        3,
    )
].copy()

# Map MindLAMP source ID to analytical participant ID.
dep["participant_id"] = (
    dep["participant_id"]
    .astype(str)
)

dep = dep.merge(
    mapping,
    left_on="participant_id",
    right_on="source_participant_id",
    how="inner",
    suffixes=(
        "_mindlamp",
        "_clinical",
    ),
)

dep = dep.rename(
    columns={
        "participant_id_clinical":
            "participant_id"
    }
)

# Preserve exact assessment time.
dep["assessment_datetime"] = (
    pd.to_datetime(
        dep["datetime"],
        utc=True,
        errors="coerce",
    )
)

# Confirm no duplicate item within an assessment.
dupes = (
    dep.groupby(
        [
            "participant_id",
            "timestamp",
            "phq_item",
        ]
    )
    .size()
)

if (dupes > 1).any():
    raise RuntimeError(
        "Duplicate PHQ item responses detected "
        "within participant-assessment."
    )

wide = (
    dep.pivot_table(
        index=[
            "participant_id",
            "timestamp",
        ],
        columns="phq_item",
        values="numeric_value",
        aggfunc="first",
    )
    .reset_index()
)

assessment_times = (
    dep.groupby(
        [
            "participant_id",
            "timestamp",
        ],
        as_index=False,
    )
    .agg(
        assessment_datetime=(
            "assessment_datetime",
            "min",
        ),
    )
)

phq = wide.merge(
    assessment_times,
    on=[
        "participant_id",
        "timestamp",
    ],
    how="left",
)

for c in symptom_items:
    if c not in phq.columns:
        phq[c] = np.nan

phq["phq9_total"] = (
    phq[symptom_items]
    .sum(
        axis=1,
        min_count=9,
    )
)

phq8_items = [
    f"phq{i}"
    for i in range(1, 9)
]

phq["phq8_total"] = (
    phq[phq8_items]
    .sum(
        axis=1,
        min_count=8,
    )
)

# Use the UTC analysis day because the existing daily home-time
# table was generated using UTC day boundaries.
phq["assessment_date"] = (
    phq["assessment_datetime"]
    .dt.floor("D")
    .dt.tz_localize(None)
)

phq = phq[
    phq["phq9_total"].notna()
    & phq["phq8_total"].notna()
    & phq["assessment_date"].notna()
].copy()


# ============================================================
# LOAD DAILY HOME TIME
# ============================================================

home = pd.read_csv(
    HOME_PATH,
    low_memory=False,
)

if "participant_id" not in home.columns:
    raise RuntimeError(
        "daily_home_time.csv has no participant_id column."
    )

if "date" not in home.columns:
    raise RuntimeError(
        "daily_home_time.csv has no date column."
    )

home_col = identify_home_column(
    home
)

print(
    "Detected daily home-time column:",
    home_col,
)

home["date"] = pd.to_datetime(
    home["date"],
    errors="coerce",
).dt.floor("D")

home[home_col] = pd.to_numeric(
    home[home_col],
    errors="coerce",
)

# Convert to percentage if stored as fraction.
finite_home = (
    home[home_col]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .dropna()
)

if finite_home.empty:
    raise RuntimeError(
        "No numeric home-time values found."
    )

if finite_home.max() <= 1.05:
    home["home_pct"] = (
        home[home_col] * 100.0
    )
else:
    home["home_pct"] = (
        home[home_col]
    )

# Remove impossible values.
home.loc[
    ~home["home_pct"].between(
        0,
        100,
    ),
    "home_pct",
] = np.nan

# Ensure one participant-day.
duplicate_days = (
    home[
        [
            "participant_id",
            "date",
        ]
    ]
    .duplicated(
        keep=False
    )
)

if duplicate_days.any():

    check = (
        home.loc[
            duplicate_days,
            [
                "participant_id",
                "date",
                "home_pct",
            ],
        ]
        .groupby(
            [
                "participant_id",
                "date",
            ]
        )["home_pct"]
        .nunique(
            dropna=True
        )
    )

    if (check > 1).any():
        raise RuntimeError(
            "Conflicting duplicate daily home-time "
            "rows detected."
        )

    home = (
        home
        .sort_values(
            [
                "participant_id",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "participant_id",
                "date",
            ],
            keep="first",
        )
    )

home_lookup = {
    pid: g.set_index(
        "date"
    )["home_pct"]
    for pid, g in home.groupby(
        "participant_id"
    )
}


# ============================================================
# CONSTRUCT TEMPORAL HOME-TIME WINDOWS
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

rows = []

for _, r in phq.iterrows():

    pid = r["participant_id"]

    if pid not in home_lookup:
        continue

    assessment_date = pd.Timestamp(
        r["assessment_date"]
    )

    # Strictly PRECEDING complete analysis days:
    # assessment day itself is not used.
    dates14 = pd.date_range(
        end=(
            assessment_date
            - pd.Timedelta(days=1)
        ),
        periods=14,
        freq="D",
    )

    hs = home_lookup[pid].reindex(
        dates14
    )

    valid14 = hs.notna().sum()

    row = {
        "participant_id":
            pid,

        "timestamp":
            r["timestamp"],

        "assessment_datetime":
            r["assessment_datetime"],

        "assessment_date":
            assessment_date,

        "phq9_total":
            float(
                r["phq9_total"]
            ),

        "phq8_total":
            float(
                r["phq8_total"]
            ),

        "phq9_item":
            float(
                r["phq9"]
            ),

        "valid_days_14d":
            int(valid14),

        "complete_14d":
            bool(valid14 == 14),
    }

    # --------------------------------------------------------
    # 14-day entire segment
    # --------------------------------------------------------

    if valid14 == 14:

        row[
            "home_14d_all_pct"
        ] = median_or_nan(
            hs.values
        )

        weekdays = hs[
            hs.index.dayofweek < 5
        ]

        weekends = hs[
            hs.index.dayofweek >= 5
        ]

        # Every complete 14-day interval should contain
        # exactly 10 weekdays and 4 weekend days.
        row[
            "n_weekdays_14d"
        ] = len(weekdays)

        row[
            "n_weekends_14d"
        ] = len(weekends)

        row[
            "home_14d_weekday_matched_pct"
        ] = weekday_matched_summary(
            weekdays.values,
            rng,
        )

        row[
            "home_14d_weekend_pct"
        ] = (
            median_or_nan(
                weekends.values
            )
            if len(weekends) == 4
            else np.nan
        )

    else:

        row[
            "home_14d_all_pct"
        ] = np.nan

        row[
            "n_weekdays_14d"
        ] = np.nan

        row[
            "n_weekends_14d"
        ] = np.nan

        row[
            "home_14d_weekday_matched_pct"
        ] = np.nan

        row[
            "home_14d_weekend_pct"
        ] = np.nan

    # --------------------------------------------------------
    # Nested proximity windows
    #
    # Only computed when every requested day is valid.
    # --------------------------------------------------------

    for n in [
        7,
        3,
        2,
        1,
    ]:

        dates_n = dates14[-n:]

        values = hs.reindex(
            dates_n
        )

        row[
            f"valid_days_{n}d"
        ] = int(
            values.notna().sum()
        )

        if values.notna().sum() == n:

            row[
                f"home_{n}d_pct"
            ] = median_or_nan(
                values.values
            )

        else:

            row[
                f"home_{n}d_pct"
            ] = np.nan

    rows.append(row)


analysis = pd.DataFrame(
    rows
)

if analysis.empty:
    raise RuntimeError(
        "No PHQ assessments could be linked "
        "to daily home-time data."
    )


# ============================================================
# COMMON STRICT COHORT
#
# Primary temporal comparison:
# same assessments for every time window.
# ============================================================

strict = analysis[
    analysis["complete_14d"]
].copy()

WINDOWS = [
    (
        "14 days — all days",
        "home_14d_all_pct",
    ),
    (
        "14 days — weekdays",
        "home_14d_weekday_matched_pct",
    ),
    (
        "14 days — weekends",
        "home_14d_weekend_pct",
    ),
    (
        "Previous 7 days",
        "home_7d_pct",
    ),
    (
        "Previous 3 days",
        "home_3d_pct",
    ),
    (
        "Previous 2 days",
        "home_2d_pct",
    ),
    (
        "Previous 1 day",
        "home_1d_pct",
    ),
]


# ============================================================
# GEE MODELS
# ============================================================

stats = []

for score_name, score_col in [
    (
        "PHQ-9",
        "phq9_total",
    ),
    (
        "PHQ-8",
        "phq8_total",
    ),
]:

    for (
        window_name,
        window_col,
    ) in WINDOWS:

        stats.append(
            gee_model(
                strict,
                score_col,
                window_col,
                score_name,
                window_name,
            )
        )

stats = pd.DataFrame(
    stats
)


# ============================================================
# SAVE DATA
# ============================================================

analysis.to_csv(
    OUT
    / "phq8_phq9_home_time_all_assessments.csv",
    index=False,
)

strict.to_csv(
    OUT
    / "phq8_phq9_home_time_strict_14d_cohort.csv",
    index=False,
)

stats.to_csv(
    OUT
    / "phq8_phq9_temporal_gee_stats.csv",
    index=False,
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 100)
print("PHQ-8 / PHQ-9 × HOME-TIME TEMPORAL ANALYSIS")
print("=" * 100)

print(
    "PHQ assessments in mapped survey:",
    len(phq),
)

print(
    "Participants in mapped survey:",
    phq["participant_id"].nunique(),
)

print(
    "Assessments linked to home-time participants:",
    len(analysis),
)

print(
    "Participants linked to home time:",
    analysis["participant_id"].nunique(),
)

print()
print("STRICT 14-DAY COMMON COHORT")
print("-" * 100)

print(
    "Assessments with all 14 preceding valid home days:",
    len(strict),
)

print(
    "Participants:",
    strict["participant_id"].nunique(),
)

print(
    "Assessment date range:",
    strict["assessment_date"].min(),
    "→",
    strict["assessment_date"].max(),
)

print()
print("PHQ SCORE DISTRIBUTIONS")
print("-" * 100)

for c in [
    "phq9_total",
    "phq8_total",
]:
    print(
        c,
        "| mean:",
        round(
            strict[c].mean(),
            2,
        ),
        "| SD:",
        round(
            strict[c].std(),
            2,
        ),
        "| median:",
        round(
            strict[c].median(),
            2,
        ),
        "| range:",
        (
            strict[c].min(),
            strict[c].max(),
        ),
    )

print()
print(
    "Assessments with PHQ item 9 > 0:",
    int(
        (
            strict["phq9_item"] > 0
        ).sum()
    ),
)

print(
    "Percent with PHQ item 9 > 0:",
    round(
        100
        * (
            strict["phq9_item"] > 0
        ).mean(),
        1,
    ),
)

print()
print("=" * 100)
print(
    "PARTICIPANT-CLUSTERED GEE"
)
print(
    "Outcome = home time (%)"
)
print(
    "Effect = percentage-point change in home time per +5 PHQ points"
)
print("=" * 100)

display = stats[
    [
        "score",
        "window",
        "n_assessments",
        "n_participants",
        "beta_pct_per_5_points",
        "ci_low_pct_per_5_points",
        "ci_high_pct_per_5_points",
        "p_value",
    ]
].copy()

for c in [
    "beta_pct_per_5_points",
    "ci_low_pct_per_5_points",
    "ci_high_pct_per_5_points",
]:
    display[c] = display[c].round(
        3
    )

display[
    "p_value"
] = display[
    "p_value"
].map(
    lambda x:
        f"{x:.4g}"
        if pd.notna(x)
        else "NA"
)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# COEFFICIENT FIGURE
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

order = [
    x[0]
    for x in WINDOWS
]

fig, ax = plt.subplots(
    figsize=(
        8.6,
        5.8,
    )
)

y_base = np.arange(
    len(order)
)

offsets = {
    "PHQ-9": -0.12,
    "PHQ-8": 0.12,
}

markers = {
    "PHQ-9": "o",
    "PHQ-8": "s",
}

for score_name in [
    "PHQ-9",
    "PHQ-8",
]:

    d = (
        stats[
            stats["score"].eq(
                score_name
            )
        ]
        .set_index(
            "window"
        )
        .loc[
            order
        ]
        .reset_index()
    )

    y = (
        y_base
        + offsets[
            score_name
        ]
    )

    x = d[
        "beta_pct_per_5_points"
    ].to_numpy()

    low = d[
        "ci_low_pct_per_5_points"
    ].to_numpy()

    high = d[
        "ci_high_pct_per_5_points"
    ].to_numpy()

    xerr = np.vstack(
        [
            x - low,
            high - x,
        ]
    )

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt=markers[
            score_name
        ],
        capsize=3,
        linewidth=1.5,
        markersize=6,
        label=score_name,
    )

ax.axvline(
    0,
    linewidth=1,
    linestyle="--",
)

ax.set_yticks(
    y_base
)

ax.set_yticklabels(
    order
)

ax.invert_yaxis()

ax.set_xlabel(
    "Change in home time per +5 symptom points\n"
    "(percentage points; 95% CI)"
)

ax.set_ylabel(
    ""
)

ax.legend(
    frameon=False,
    loc="lower right",
)

ax.set_title(
    "Home time and depressive symptom severity across temporal windows",
    fontsize=13,
    pad=12,
)

fig.tight_layout()

fig.savefig(
    OUT
    / "phq8_phq9_temporal_effects.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "phq8_phq9_temporal_effects.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# WINDOW DESCRIPTIVES
# ============================================================

desc_rows = []

for window_name, window_col in WINDOWS:

    x = strict[
        window_col
    ].dropna()

    desc_rows.append(
        {
            "window":
                window_name,

            "n_assessments":
                len(x),

            "n_participants":
                strict.loc[
                    strict[
                        window_col
                    ].notna(),
                    "participant_id",
                ].nunique(),

            "mean_home_pct":
                x.mean(),

            "median_home_pct":
                x.median(),

            "q25_home_pct":
                x.quantile(
                    0.25
                ),

            "q75_home_pct":
                x.quantile(
                    0.75
                ),
        }
    )

desc = pd.DataFrame(
    desc_rows
)

desc.to_csv(
    OUT
    / "phq_home_time_temporal_descriptives.csv",
    index=False,
)

print()
print("=" * 100)
print("HOME-TIME DESCRIPTIVES")
print("=" * 100)

print(
    desc.round(2).to_string(
        index=False
    )
)

print()
print("Saved:")
for path in sorted(
    OUT.iterdir()
):
    print(" ", path)

