from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_longitudinal"
)

# Use the already-audited daily data
DATA_FILE = ROOT / "home_t_function_t1_analysis_dataset.csv"

OUT_CSV = ROOT / "home_functioning_distributed_lag_results.csv"
OUT_FIG = ROOT / "home_functioning_distributed_lag.png"
OUT_DATA = ROOT / "home_functioning_distributed_lag_dataset.csv"

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# ============================================================
# RECONSTRUCT ONE DAILY RECORD PER PARTICIPANT
# ============================================================

daily = (
    df[
        [
            "participant_id",
            "date",
            "home_pct",
            "gps_hours",
            "functioning_t",
        ]
    ]
    .drop_duplicates(
        subset=[
            "participant_id",
            "date"
        ]
    )
    .dropna(
        subset=[
            "participant_id",
            "date",
            "home_pct",
        ]
    )
    .sort_values(
        [
            "participant_id",
            "date"
        ]
    )
)

# ============================================================
# PERSON-CENTER HOME TIME
# ============================================================

daily["home_mean"] = (
    daily.groupby(
        "participant_id"
    )["home_pct"]
    .transform("mean")
)

daily["home_dev"] = (
    daily["home_pct"]
    - daily["home_mean"]
)

# per +10 percentage points
daily["home_dev_10"] = (
    daily["home_dev"] / 10
)

# ============================================================
# BUILD EXACT CALENDAR-DAY LAGS
#
# lag0 = same-day home deviation
# lag1 = yesterday
# ...
# lag6 = six days earlier
#
# Exact date joins avoid treating missing calendar days
# as consecutive observations.
# ============================================================

analysis = daily[
    [
        "participant_id",
        "date",
        "functioning_t",
    ]
].copy()

analysis = analysis.rename(
    columns={
        "functioning_t": "functioning"
    }
)

for lag in range(0, 7):

    exposure = daily[
        [
            "participant_id",
            "date",
            "home_dev_10",
        ]
    ].copy()

    # A home observation at date d predicts outcome at d + lag
    exposure["date"] = (
        exposure["date"]
        + pd.Timedelta(days=lag)
    )

    exposure = exposure.rename(
        columns={
            "home_dev_10": f"home_lag{lag}"
        }
    )

    analysis = analysis.merge(
        exposure,
        on=[
            "participant_id",
            "date"
        ],
        how="left"
    )

# ============================================================
# BASIC TIME COVARIATES
# ============================================================

analysis["weekend"] = (
    analysis["date"].dt.dayofweek >= 5
).astype(int)

first_date = (
    analysis.groupby(
        "participant_id"
    )["date"]
    .transform("min")
)

analysis["study_day_10"] = (
    (
        analysis["date"]
        - first_date
    ).dt.days / 10
)

# ============================================================
# COMPLETE-CASE DISTRIBUTED-LAG COHORT
#
# Need functioning plus all seven home-time lags.
# This is required for the simultaneous lag model.
# ============================================================

lag_cols = [
    f"home_lag{i}"
    for i in range(7)
]

model_df = analysis.dropna(
    subset=[
        "functioning",
        "weekend",
        "study_day_10",
        *lag_cols,
    ]
).copy()

# Keep participants with at least 3 usable 7-day sequences
counts = (
    model_df.groupby(
        "participant_id"
    )
    .size()
)

keep = counts[
    counts >= 3
].index

model_df = model_df[
    model_df["participant_id"].isin(keep)
].copy()

model_df.to_csv(
    OUT_DATA,
    index=False
)

print()
print("=" * 100)
print("DISTRIBUTED-LAG HOME TIME -> FUNCTIONING")
print("=" * 100)

print(
    "Usable observations:",
    len(model_df)
)

print(
    "Participants:",
    model_df[
        "participant_id"
    ].nunique()
)

print()
print("Observations per participant:")
print(
    model_df.groupby(
        "participant_id"
    )
    .size()
    .describe()
)

# ============================================================
# SIMULTANEOUS 0-6 DAY MODEL
#
# Participant fixed effects:
# only within-person variation contributes to lag estimates.
# ============================================================

lag_formula = " + ".join(
    lag_cols
)

formula = (
    "functioning ~ "
    + lag_formula
    + " + weekend"
    + " + study_day_10"
    + " + C(participant_id)"
)

fit = smf.ols(
    formula,
    data=model_df
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups":
            model_df["participant_id"]
    }
)

# ============================================================
# EXTRACT LAG EFFECTS
# ============================================================

rows = []

for lag in range(7):

    term = f"home_lag{lag}"

    beta = float(
        fit.params[term]
    )

    se = float(
        fit.bse[term]
    )

    p = float(
        fit.pvalues[term]
    )

    rows.append(
        {
            "lag_days": lag,
            "beta_per_10pp_home": beta,
            "se": se,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "p_value": p,
        }
    )

res = pd.DataFrame(rows)

# ============================================================
# JOINT TEST:
# Do home-time deviations over the 0-6 day period matter
# collectively?
# ============================================================

restriction = ", ".join(
    f"home_lag{i} = 0"
    for i in range(7)
)

try:
    joint = fit.wald_test(
        restriction,
        scalar=True
    )

    joint_p = float(
        joint.pvalue
    )

except Exception:
    joint_p = np.nan

res["joint_0_6day_p"] = joint_p

res.to_csv(
    OUT_CSV,
    index=False
)

print()
print("=" * 100)
print("LAG-SPECIFIC RESULTS")
print("=" * 100)

print(
    res[
        [
            "lag_days",
            "beta_per_10pp_home",
            "ci_low",
            "ci_high",
            "p_value",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print(
    "Joint test for lags 0-6:",
    f"p = {joint_p:.4f}"
)

# ============================================================
# CUMULATIVE EFFECT
#
# Effect of +10pp home deviation maintained across all
# seven days = sum of lag coefficients.
# ============================================================

lag_names = [
    f"home_lag{i}"
    for i in range(7)
]

beta_vec = (
    fit.params[
        lag_names
    ].values
)

cov = (
    fit.cov_params()
    .loc[
        lag_names,
        lag_names
    ]
    .values
)

cumulative_beta = (
    beta_vec.sum()
)

cumulative_var = (
    np.ones(7)
    @ cov
    @ np.ones(7)
)

cumulative_se = np.sqrt(
    max(
        cumulative_var,
        0
    )
)

cum_low = (
    cumulative_beta
    - 1.96 * cumulative_se
)

cum_high = (
    cumulative_beta
    + 1.96 * cumulative_se
)

if cumulative_se > 0:

    from scipy.stats import norm

    z = (
        cumulative_beta
        / cumulative_se
    )

    cumulative_p = (
        2
        * (
            1
            - norm.cdf(
                abs(z)
            )
        )
    )

else:
    cumulative_p = np.nan

print()
print("=" * 100)
print("7-DAY CUMULATIVE EFFECT")
print("=" * 100)

print(
    "If home time is +10pp above personal usual "
    "across all 7 days:"
)

print(
    "cumulative beta =",
    f"{cumulative_beta:.4f}"
)

print(
    "95% CI =",
    f"{cum_low:.4f}",
    "to",
    f"{cum_high:.4f}"
)

print(
    "p =",
    f"{cumulative_p:.4f}"
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 10

fig, ax = plt.subplots(
    figsize=(8.5, 5.2)
)

ax.axhline(
    0,
    linewidth=1,
    linestyle="--",
    color="0.5"
)

ax.errorbar(
    res["lag_days"],
    res["beta_per_10pp_home"],
    yerr=[
        res["beta_per_10pp_home"]
        - res["ci_low"],

        res["ci_high"]
        - res["beta_per_10pp_home"],
    ],
    fmt="o-",
    linewidth=1.8,
    capsize=4,
)

ax.set_xticks(
    range(7)
)

ax.set_xticklabels(
    [
        "Same day",
        "1 day",
        "2 days",
        "3 days",
        "4 days",
        "5 days",
        "6 days",
    ]
)

ax.set_xlabel(
    "Home-time exposure before functioning assessment"
)

ax.set_ylabel(
    "Change in functioning per +10 pp\n"
    "home time above personal usual"
)

ax.set_title(
    "Within-person distributed-lag association\n"
    "between home time and daily functioning"
)

ax.text(
    0.02,
    0.04,
    (
        f"Joint lag test: p={joint_p:.3f}\n"
        f"7-day cumulative effect: "
        f"{cumulative_beta:.3f} "
        f"(95% CI {cum_low:.3f}, {cum_high:.3f})"
    ),
    transform=ax.transAxes,
    fontsize=9,
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

plt.tight_layout()

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("Saved:")
print(OUT_CSV)
print(OUT_DATA)
print(OUT_FIG)
