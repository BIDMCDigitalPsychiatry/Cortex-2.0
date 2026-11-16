from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

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
      "feature_reliability/gps_clinical/home_time/"
      "daily_home_time.csv"
)

OUT_DIR = (
    DC
    / "cortex2_outputs/digital_clinic/"
      "feature_reliability/gps_clinical/home_time/"
      "daily_functioning"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_OUT = (
    OUT_DIR
    / "daily_functioning_home_time_analysis_dataset.csv"
)

RESULTS_OUT = (
    OUT_DIR
    / "daily_functioning_home_time_gee_results.csv"
)

SUMMARY_OUT = (
    OUT_DIR
    / "daily_functioning_home_time_analysis_summary.csv"
)

# =============================================================================
# PARTICIPANT MAP
# =============================================================================

mapping = pd.read_csv(
    MAP_PATH,
    low_memory=False,
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

# Same duplicate-source rule as the existing GPS clinical workflow.
duplicate_sources = (
    mapping
    .groupby("source_participant_id")["participant_id"]
    .nunique()
)

duplicate_sources = duplicate_sources[
    duplicate_sources > 1
]

if len(duplicate_sources):
    print("Duplicate source mappings detected:")
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
            "  one source maps to multiple analytical IDs "
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

# =============================================================================
# FUNCTIONING RESPONSES
# =============================================================================

survey = pd.read_csv(
    SURVEY_PATH,
    low_memory=False,
)

func = survey[
    survey["activity_name"]
    .astype(str)
    .str.strip()
    .eq("Daily Survey")
    &
    survey["item"]
    .astype(str)
    .str.contains(
        "manage my day-to-day life",
        case=False,
        na=False,
    )
].copy()

func["functioning_score"] = pd.to_numeric(
    func["value"],
    errors="coerce",
)

func["assessment_datetime_utc"] = (
    pd.to_datetime(
        func["datetime"],
        errors="coerce",
        utc=True,
    )
)

func["assessment_datetime_local"] = (
    func["assessment_datetime_utc"]
    .dt.tz_convert("America/New_York")
)

func["assessment_date"] = (
    func["assessment_datetime_local"]
    .dt.date
)

func["assessment_hour"] = (
    func["assessment_datetime_local"].dt.hour
    + func["assessment_datetime_local"].dt.minute / 60
    + func["assessment_datetime_local"].dt.second / 3600
)

func = func.dropna(
    subset=[
        "participant_id",
        "functioning_score",
        "assessment_datetime_local",
        "assessment_date",
    ]
).copy()

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

func = func.sort_values(
    [
        "participant_id",
        "assessment_datetime_local",
    ]
)

# =============================================================================
# CREATE THREE DAILY FUNCTIONING DEFINITIONS
#
# Primary:
#   daily mean of all functioning responses.
#
# Sensitivity 1:
#   last functioning response of the day.
#
# Sensitivity 2:
#   participant-days with exactly one response only.
# =============================================================================

group_cols = [
    "participant_id",
    "assessment_date",
]

response_counts = (
    func.groupby(group_cols)
    .size()
    .rename("n_functioning_responses")
    .reset_index()
)

# -------------------------------------------------------------------------
# PRIMARY: daily mean
# -------------------------------------------------------------------------

daily_mean = (
    func.groupby(
        group_cols,
        as_index=False,
    )
    .agg(
        functioning_score=(
            "functioning_score",
            "mean",
        ),
        assessment_hour=(
            "assessment_hour",
            "mean",
        ),
        n_functioning_responses=(
            "functioning_score",
            "size",
        ),
    )
)

daily_mean["analysis_definition"] = (
    "daily_mean_primary"
)

# -------------------------------------------------------------------------
# SENSITIVITY: last response
# -------------------------------------------------------------------------

daily_last = (
    func.groupby(
        group_cols,
        as_index=False,
    )
    .tail(1)
    [
        [
            "participant_id",
            "assessment_date",
            "functioning_score",
            "assessment_hour",
        ]
    ]
    .copy()
)

daily_last = daily_last.merge(
    response_counts,
    on=group_cols,
    how="left",
)

daily_last["analysis_definition"] = (
    "last_response_sensitivity"
)

# -------------------------------------------------------------------------
# SENSITIVITY: single-response days only
# -------------------------------------------------------------------------

single_days = response_counts[
    response_counts[
        "n_functioning_responses"
    ].eq(1)
].copy()

daily_single = (
    func.merge(
        single_days,
        on=group_cols,
        how="inner",
    )
    [
        [
            "participant_id",
            "assessment_date",
            "functioning_score",
            "assessment_hour",
            "n_functioning_responses",
        ]
    ]
    .copy()
)

daily_single["analysis_definition"] = (
    "single_response_days_sensitivity"
)

# =============================================================================
# HOME TIME
# =============================================================================

home = pd.read_csv(
    HOME_PATH,
    low_memory=False,
)

home["participant_id"] = (
    home["participant_id"]
    .astype(str)
    .str.strip()
)

home["assessment_date"] = (
    pd.to_datetime(
        home["date"],
        errors="coerce",
    )
    .dt.date
)

home["home_time_pct"] = pd.to_numeric(
    home["time_at_home_pct"],
    errors="coerce",
)

home = home[
    [
        "participant_id",
        "assessment_date",
        "home_time_status",
        "supported_duration_ms",
        "home_time_pct",
    ]
].copy()

# One row per participant/date expected.
dup_home = (
    home.groupby(
        [
            "participant_id",
            "assessment_date",
        ]
    )
    .size()
)

if (dup_home > 1).any():
    raise RuntimeError(
        "daily_home_time.csv has duplicate participant/date rows."
    )

# =============================================================================
# MERGE + EXPOSURE DECOMPOSITION
# =============================================================================

all_sets = []

for d in [
    daily_mean,
    daily_last,
    daily_single,
]:

    x = d.merge(
        home,
        on=[
            "participant_id",
            "assessment_date",
        ],
        how="inner",
    )

    x = x.dropna(
        subset=[
            "functioning_score",
            "home_time_pct",
        ]
    ).copy()

    x["assessment_date_dt"] = (
        pd.to_datetime(
            x["assessment_date"]
        )
    )

    x["weekend"] = (
        x["assessment_date_dt"]
        .dt.dayofweek
        .ge(5)
        .astype(int)
    )

    # ---------------------------------------------------------
    # Within-between decomposition
    #
    # between = participant's mean home time
    # within  = today's deviation from participant mean
    #
    # Divide by 10 so coefficients correspond to a
    # 10-percentage-point difference in home time.
    # ---------------------------------------------------------

    x["participant_mean_home_pct"] = (
        x.groupby("participant_id")[
            "home_time_pct"
        ]
        .transform("mean")
    )

    x["home_within_deviation_pct"] = (
        x["home_time_pct"]
        - x["participant_mean_home_pct"]
    )

    x["home_between_10pp"] = (
        x["participant_mean_home_pct"]
        / 10.0
    )

    x["home_within_10pp"] = (
        x["home_within_deviation_pct"]
        / 10.0
    )

    all_sets.append(x)

analysis = pd.concat(
    all_sets,
    ignore_index=True,
)

analysis.to_csv(
    DATA_OUT,
    index=False,
)

# =============================================================================
# GEE
# =============================================================================

def fit_gee(data, definition):

    d = data[
        data["analysis_definition"].eq(
            definition
        )
    ].copy()

    # Model:
    # functioning =
    #   within-person home deviation
    # + between-person mean home time
    # + weekend indicator
    #
    # Gaussian identity is used for the primary interpretable
    # repeated-measures analysis of the 0-4 functioning score.
    # Robust participant-clustered GEE SEs are used.

    model = smf.gee(
        formula=(
            "functioning_score ~ "
            "home_within_10pp + "
            "home_between_10pp + "
            "weekend"
        ),
        groups="participant_id",
        data=d,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )

    fit = model.fit()

    ci = fit.conf_int()

    rows = []

    for term in fit.params.index:
        rows.append(
            {
                "analysis_definition":
                    definition,
                "term":
                    term,
                "beta":
                    fit.params[term],
                "ci_lower":
                    ci.loc[term, 0],
                "ci_upper":
                    ci.loc[term, 1],
                "p_value":
                    fit.pvalues[term],
                "n_observations":
                    len(d),
                "n_participants":
                    d["participant_id"].nunique(),
            }
        )

    return fit, pd.DataFrame(rows), d


definitions = [
    "daily_mean_primary",
    "last_response_sensitivity",
    "single_response_days_sensitivity",
]

result_frames = []
summary_rows = []

print()
print("=" * 110)
print("DAILY FUNCTIONING × HOME TIME")
print("WITHIN-/BETWEEN-PERSON GEE")
print("=" * 110)

for definition in definitions:

    fit, result_df, d = fit_gee(
        analysis,
        definition,
    )

    result_frames.append(result_df)

    print()
    print("-" * 110)
    print(definition)
    print("-" * 110)

    print(
        "Observations:",
        len(d),
    )

    print(
        "Participants:",
        d["participant_id"].nunique(),
    )

    print(
        "Median observations/person:",
        round(
            d.groupby("participant_id")
            .size()
            .median(),
            1,
        ),
    )

    print(
        "Mean functioning:",
        round(
            d["functioning_score"].mean(),
            3,
        ),
    )

    print(
        "Mean home time:",
        round(
            d["home_time_pct"].mean(),
            2,
        ),
        "%",
    )

    print()

    for term, label in [
        (
            "home_within_10pp",
            "WITHIN-PERSON HOME TIME"
        ),
        (
            "home_between_10pp",
            "BETWEEN-PERSON HOME TIME"
        ),
        (
            "weekend",
            "WEEKEND"
        ),
    ]:

        beta = fit.params[term]
        lo, hi = fit.conf_int().loc[term]
        p = fit.pvalues[term]

        print(label)
        print(
            f"  beta = {beta:.4f}"
        )
        print(
            f"  95% CI = [{lo:.4f}, {hi:.4f}]"
        )
        print(
            f"  p = {p:.4g}"
        )

    summary_rows.append(
        {
            "analysis_definition":
                definition,
            "n_observations":
                len(d),
            "n_participants":
                d["participant_id"].nunique(),
            "median_observations_per_participant":
                d.groupby(
                    "participant_id"
                ).size().median(),
            "mean_functioning":
                d["functioning_score"].mean(),
            "sd_functioning":
                d["functioning_score"].std(),
            "mean_home_time_pct":
                d["home_time_pct"].mean(),
            "sd_home_time_pct":
                d["home_time_pct"].std(),
        }
    )

results = pd.concat(
    result_frames,
    ignore_index=True,
)

summary = pd.DataFrame(
    summary_rows
)

results.to_csv(
    RESULTS_OUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)

print()
print("=" * 110)
print("INTERPRETATION GUIDE")
print("=" * 110)

print(
    """
home_within_10pp:
  Expected change in the 0-4 functioning score when a participant
  spends 10 percentage points MORE time at home than is typical
  for that same participant.

  Positive beta = better functioning on higher-home-time days.
  Negative beta = poorer functioning on higher-home-time days.

home_between_10pp:
  Expected difference in functioning between participants whose
  average home time differs by 10 percentage points.

  This is a between-person association and should NOT be interpreted
  as a within-person behavioral effect.

weekend:
  Average functioning difference on weekends versus weekdays,
  conditional on the home-time terms.
"""
)

print("Saved:")
print(" ", DATA_OUT)
print(" ", RESULTS_OUT)
print(" ", SUMMARY_OUT)
